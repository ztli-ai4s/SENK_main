"""
EquiformerV2 self-trained four-weights spectra inference program
================================================================
SENK = SO(3) Equivariant Neural Kalman Networks — a response-state cascade
combining an equivariant transformer backbone, an Equivariant Neural Kalman
(ENK) bridge, and an NBO-informed Electron Prior (EP) pathway with Guided
Spectral Calibration (GSC).

Following the DetaNet harmonic-oscillator modeling route, all four weights are
produced by in-project self-training:
  - hii     : diagonal Hessian Hii  [N, 3, 3]      -> CleanEquiformerMultiTask
  - hij     : off-diagonal Hessian Hij [E, 3, 3]   -> CleanEquiformerMultiTask
  - dedipole: dipole derivative dmu/dR (Born effective charges) [N, 3, 3]
              -> CleanEquiformerMultiTask
  - depolar : polarizability derivative dalpha/dR (Raman activity tensor) [N, 3, 6]
              -> CleanEquiformerPolar(V1) / CleanEquiformerPolarExt(V2)

Each weight independently selects a training mode (--hii_mode, --hij_mode,
--dd_mode, --dp_mode):
  clean_equiformer          -> V1 backbone (GraphAttentionTransformer)
  equiformer_v2             -> V2 backbone (EquiformerV2) [recommended for hii/hij]
  equiformer_v2_enk         -> V2 + ENK  [recommended for dedipole/depolar]
  equiformer_v2_enk_ep      -> V2 + ENK + ElectronPrior

For each weight, lmax / num_layers / ENK are auto-detected from the checkpoint,
so manual specification is not required.

Supports SENK (ENK on/off) dual-curve comparison; the weight paths may be set
explicitly through DEFAULT_SENK_ENK_ON_* / DEFAULT_SENK_ENK_OFF_* at the top of
this module.

The post-processing pipeline reuses the physics functions in
detanet_nets.spectra_simulator
(hessfreq -> chain_rule_ir/raman -> Lorenz_broadening).

Usage examples:
  # Single-molecule XYZ (uses default ENK-on qme14s checkpoints)
  python v2_spectra_infer.py --xyz examples/ethanol.xyz --gpu 0

  # Single-molecule XYZ with explicit checkpoint paths
  python v2_spectra_infer.py --xyz examples/ethanol.xyz \\
      --hii_ckpt  "checkpoints/ENK on/hii/qme14s/best.pt"     --hii_mode equiformer_v2_enk \\
      --hij_ckpt  "checkpoints/ENK on/hij/qme14s/best.pt"     --hij_mode equiformer_v2_enk \\
      --dd_ckpt   "checkpoints/ENK on/dedipole/qme14s/best.pt" --dd_mode  equiformer_v2_enk \\
      --dp_ckpt   "checkpoints/ENK on/depolar/qme14s/best.pt"  --dp_mode  equiformer_v2_enk

  # EP cascade mode (ENK + Electron Prior + NBO-GSC, recommended)
  python v2_spectra_infer.py --xyz examples/ethanol.xyz --ep --gpu 0

  # NOTE: ENK on/off dual-curve comparison is ALWAYS produced in single-molecule
  # mode (no flag needed). The eight weights (4 ENK-on + 4 ENK-off) are loaded
  # automatically from DEFAULT_SENK_ENK_ON_* / DEFAULT_SENK_ENK_OFF_* and both
  # curves are overlaid in the output plot. Override individual paths with
  # --hii_ckpt/--hij_ckpt/--dd_ckpt/--dp_ckpt (ENK on) and
  # --hii_off_ckpt/--hij_off_ckpt/--dd_off_ckpt/--dp_off_ckpt (ENK off).

  # Single molecule + DetaNet comparison curves (overlaying four DetaNet weights)
  python v2_spectra_infer.py --xyz examples/ethanol.xyz \\
      --compare_detanet --out_png compare.png --gpu 0

  # Batch SMILES -> IR + Raman (all V2 weights)
  python v2_spectra_infer.py \\
      --batch_smi examples/mols.smi --batch_out_dir v2_batch_output --gpu 0
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from torch_geometric.nn import radius_graph

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detanet_nets.constant import atom_masses
from detanet_nets.spectra_simulator import (
    hessfreq,
    chain_rule_ir,
    chain_rule_raman,
    get_raman_act,
    Lorenz_broadening,
)
from nbo_spectral_calibration import NBOGuidedCalibrator, _extract_nbo_from_model
from ep_canonical_defaults import apply_canonical_ep_defaults
from ep_feature_runtime import make_calibrator as _make_inference_calibrator
import json

# Element table
_ELEMENTS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
]
SYM2Z = {s: i for i, s in enumerate(_ELEMENTS) if s}

_ALL_MODES = [
    "clean_equiformer",
    "equiformer_v2",
    "equiformer_v2_enk",
    "equiformer_v2_enk_ep",
]

# ---------------------------------------------------------------------------
# Explicit default checkpoint paths for SENK inference (edit as needed)
# ---------------------------------------------------------------------------
# SENK (ENK on) — replace with your own checkpoint paths after training
DEFAULT_SENK_ENK_ON_HII = "checkpoints/ENK on/hii/qme14s/best.pt"
DEFAULT_SENK_ENK_ON_HIJ = "checkpoints/ENK on/hij/qme14s/best.pt"
DEFAULT_SENK_ENK_ON_DD = "checkpoints/ENK on/dedipole/qme14s/best.pt"
DEFAULT_SENK_ENK_ON_DP = "checkpoints/ENK on/depolar/qme14s/best.pt"

# SENK (ENK off, pure V2 with ENK disabled)
DEFAULT_SENK_ENK_OFF_HII = "checkpoints/ENK off/hii/qme14s/best.pt"
DEFAULT_SENK_ENK_OFF_HIJ = "checkpoints/ENK off/hij/qme14s/best.pt"
DEFAULT_SENK_ENK_OFF_DD = "checkpoints/ENK off/dedipole/qme14s/best.pt"
DEFAULT_SENK_ENK_OFF_DP = "checkpoints/ENK off/depolar/qme14s/best.pt"

# Default DetaNet weight paths (trained on QM9)
DETANET_WEIGHTS = {
    "Hi":       ROOT / "detanet_nets" / "qm9spectra" / "Hi.pth",
    "Hij":      ROOT / "detanet_nets" / "qm9spectra" / "Hij.pth",
    "dedipole": ROOT / "detanet_nets" / "qm9spectra" / "dedipole.pth",
    "depolar":  ROOT / "detanet_nets" / "qm9spectra" / "depolar.pth",
}

# DetaNet compat (SENK-added keys, absent from the original QM9 weights)
_DETANET_COMPAT_IGNORE_KEYS = {"lek", "lev"}

# QM9S DetaNet checkpoints allocate rows up to F, but only CHONF are
# supervised. When inference expands the table for S/Cl/P/etc., avoid using
# random nuclear rows and random high-shell electronic columns.
DETANET_OOD_EMBEDDING_MODE = "mean_periodic"
_DETANET_QM9_SUPERVISED_Z = (1, 6, 7, 8, 9)
_DETANET_PERIODIC_SURROGATE_Z = {
    14: 6, 15: 7, 16: 8, 17: 9,      # Si/P/S/Cl -> C/N/O/F
    32: 6, 33: 7, 34: 8, 35: 9,      # Ge/As/Se/Br -> C/N/O/F
    53: 9,                            # I -> F fallback if the table is extended
}

# (lmax, sphere_channels) -> preset model_name
_LMAX_C_TO_MODEL = {
    (3, 128): "equiformer_v2_l3_m2",
    (4, 128): "equiformer_v2_l4_m2",
    (4,  64): "equiformer_v2_l4_m2_small",
    (6, 128): "equiformer_v2_l6_m2",
}


# --- XYZ parsing ---

def parse_xyz(path: str) -> Tuple[List[int], List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [l.strip() for l in f if l.strip()]
    if not raw:
        raise ValueError(f"Empty file: {path}")
    try:
        nat = int(raw[0].split()[0])
        body = raw[2: 2 + nat]
    except (ValueError, IndexError):
        body = raw
    atoms, pos = [], []
    for line in body:
        parts = line.split()
        sym = parts[0]
        x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])
        s = sym.capitalize() if sym.isalpha() else sym
        if sym.isdigit():
            znum = int(sym)
        else:
            if s not in SYM2Z:
                raise ValueError(f"Unknown element: {s}")
            znum = SYM2Z[s]
        parts = line.split()
        sym = parts[0]
        x, y, z_coord = float(parts[1]), float(parts[2]), float(parts[3])
        s = sym.capitalize() if sym.isalpha() else sym
        if sym.isdigit():
            znum = int(sym)
        else:
            if s not in SYM2Z:
                raise ValueError(f"Unknown element: {s}")
            znum = SYM2Z[s]
        atoms.append(znum)
        pos.append([x, y, z_coord])
    return atoms, pos


def parse_gaussian_log_geometry(path: str, conf_index: int = -1) -> Tuple[List[int], List[List[float]]]:
    """Extract atomic coordinates from a Gaussian .log/.out file.

    Parameters
    ----------
    path : str
        Path to the Gaussian log file.
    conf_index : int
        Index of the coordinate block to extract (-1 = last block, i.e. the
        final optimized geometry; 0 = input geometry). Defaults to -1 (final
        optimized coordinates).

    Returns
    -------
    (z_list, pos_list) in the same format as parse_xyz.
    """
    z_blocks = []
    pos_blocks = []
    
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Locate coordinate-block markers ("Standard orientation" or "Input orientation")
        if "orientation:" in line.lower():
            i += 1
            # Skip separator and header rows until a row with 6+ numeric fields is found
            while i < len(lines):
                parts = lines[i].split()
                # Check whether this is a separator/header row (no digits, or too few fields)
                if len(parts) >= 6:
                    try:
                        # Try parsing fields 2 and 4/5/6 as numbers (atom number, x, y, z)
                        int(parts[1])  # atom number
                        float(parts[3]), float(parts[4]), float(parts[5])  # coordinates
                        # Success: this is a data row, start reading coordinates
                        break
                    except (ValueError, IndexError):
                        pass
                i += 1
            
            # Read the atomic coordinate block
            z_tmp, pos_tmp = [], []
            while i < len(lines):
                parts = lines[i].split()
                # End-of-block condition: fewer than 6 fields or parse failure
                if len(parts) < 6:
                    break
                try:
                    # Gaussian coordinate format: center number, atom number, type, x, y, z
                    znum = int(parts[1])
                    x, y, z_coord = float(parts[3]), float(parts[4]), float(parts[5])
                    z_tmp.append(znum)
                    pos_tmp.append([x, y, z_coord])
                    i += 1
                except (ValueError, IndexError):
                    # Row format mismatch: end of coordinate block
                    break
            
            if z_tmp:
                z_blocks.append(z_tmp)
                pos_blocks.append(pos_tmp)
        else:
            i += 1
    
    if not z_blocks:
        raise ValueError(f"No coordinate block found in Gaussian file: {path}")
    
    # Select the coordinate block (-1 = last)
    idx = conf_index if conf_index >= 0 else len(z_blocks) + conf_index
    if idx < 0 or idx >= len(z_blocks):
        raise ValueError(f"Coordinate-block index out of range: {conf_index}, total {len(z_blocks)} blocks")
    
    z_list = z_blocks[idx]
    pos_list = pos_blocks[idx]
    if idx == len(z_blocks) - 1:
        geom_label = "(final optimized geometry)"
    elif idx == 0:
        geom_label = "(initial input geometry)"
    else:
        geom_label = f"(step {idx})"
    print(f"  [Gaussian] Extracted {len(z_list)} atoms from coordinate block {idx+1}/{len(z_blocks)} {geom_label}")
    return z_list, pos_list


def parse_sdf(path: str, conf_index: int = 0) -> Tuple[List[int], List[List[float]]]:
    """Read atomic coordinates of a specified conformer from an SDF/MOL file (auto-detects Gaussian .log format).

    Parameters
    ----------
    path : str
        Path to the SDF file (if the content is a Gaussian .log, parse_gaussian_log_geometry is invoked automatically).
    conf_index : int
        Conformer index. For SDF: 0-based index; for .log: -1 = last (final optimized). Defaults to 0.

    Returns
    -------
    (z_list, pos_list) in the same format as parse_xyz, directly usable by generate_spectrum().
    """
# Priority 0: auto-detect file content type via leading markers
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_lines = [f.readline() for _ in range(5)]

    first_block = "".join(first_lines).lower()
    if "gaussian" in first_block or "entering gaussian" in first_block:
        # Content is a Gaussian log file: redirect to the Gaussian parser
        print(f"  [SDF] Gaussian log format detected, redirecting to parse_gaussian_log_geometry...")
        # For .log files, -1 means last (final optimized); otherwise use conf_index
        log_conf_index = -1 if conf_index == 0 else conf_index
        return parse_gaussian_log_geometry(path, conf_index=log_conf_index)

# Priority 1: text-parse V2000 Molfile (more robust for the Gaussian SDF format)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Split multiple records by "$$$$" and select the conf_index-th
        records = content.split("$$$$")
        records = [r.strip() for r in records if r.strip()]
        if not records:
            raise ValueError(f"SDF file is empty or cannot be split")
        if conf_index >= len(records):
            raise ValueError(f"SDF file has only {len(records)} records, but conf_index={conf_index} was requested")

        lines = records[conf_index].splitlines()
        if len(lines) < 4:
            raise ValueError(f"SDF record has too few lines")

        # Locate the counts line (typically line 4, but Gaussian may shift it)
        # Gaussian V2000 format: counts line = "aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv"
        # The first 3 fields are 3 chars each: nat and nbond
        counts_line_idx = None
        nat = None

        # Try line 3 (0-indexed) first
        if len(lines) > 3:
            try:
                nat = int(lines[3][:3].strip())
                if nat > 0 and nat < 1000:  # sanity check
                    counts_line_idx = 3
            except (ValueError, IndexError):
                pass

        # Fallback: try other possible positions (handle potential comment lines)
        if counts_line_idx is None:
            for candidate_idx in [2, 4, 5]:  # try other rows
                if candidate_idx < len(lines):
                    try:
                        nat = int(lines[candidate_idx][:3].strip())
                        if nat > 0 and nat < 1000:
                            counts_line_idx = candidate_idx
                            break
                    except (ValueError, IndexError):
                        pass

        if counts_line_idx is None:
            raise ValueError(f"Could not find a valid counts line")

        # Read atomic coordinates starting after the counts line
        atom_block_start = counts_line_idx + 1
        z_list, pos_list = [], []

        for i in range(nat):
            lno = atom_block_start + i
            if lno >= len(lines):
                raise ValueError(f"SDF atom block has fewer than {nat} rows")

            line = lines[lno]
            # V2000 atom line format (40 chars min):
            # xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee
            # Coordinates occupy the first 30 chars (x, y, z, 10 chars each); symbol at positions 31-32

            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"SDF atom line has invalid format: {line!r}")

            try:
                x, y, z_coord = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                raise ValueError(f"SDF atom coordinate parsing failed: {line!r}")

            sym = parts[3].strip()
            s = sym.capitalize()
            if s not in SYM2Z:
                raise ValueError(f"Unknown element symbol: {s!r}")

            z_list.append(SYM2Z[s])
            pos_list.append([x, y, z_coord])

        print(f"  [SDF] Text parsing: {len(z_list)} atoms (conf_index={conf_index})")
        return z_list, pos_list

    except Exception as text_err:
        print(f"  [SDF] Text parsing failed: {text_err}, trying RDKit...")
        pass  # fall through to the RDKit fallback

# Priority 2: RDKit parsing (fallback)
    try:
        from rdkit import Chem
        # Try multiple parameter combinations
        for sanitize in [False, True]:
            for kekulize in [False, True]:
                try:
                    suppl = Chem.SDMolSupplier(str(path), removeHs=False, 
                                              sanitize=sanitize)
                    mols = [m for m in suppl if m is not None]
                    if mols:
                        if conf_index >= len(mols):
                            raise ValueError(f"RDKit read only {len(mols)} molecules, but conf_index={conf_index} was requested")
                        mol = mols[conf_index]
                        conf = mol.GetConformer()
                        z_list, pos_list = [], []
                        for i in range(mol.GetNumAtoms()):
                            atom = mol.GetAtomWithIdx(i)
                            znum = atom.GetAtomicNum()
                            p = conf.GetAtomPosition(i)
                            z_list.append(znum)
                            pos_list.append([p.x, p.y, p.z])
                        print(f"  [SDF] RDKit parsing: {len(z_list)} atoms (conf_index={conf_index})")
                        return z_list, pos_list
                except Exception:
                    continue
        
        raise ValueError("RDKit failed for all parameter combinations")

    except ImportError:
        raise ImportError("RDKit unavailable and text parsing already failed; cannot continue")
    except Exception as rdkit_err:
        raise ValueError(f"RDKit parsing failed: {rdkit_err}")


# --- SMILES -> 3D coordinates ---

def smiles_to_mol(smiles: str, seed: int = 42):
    """SMILES -> RDKit Mol (multi-conformer ETKDGv3 embedding + MMFF94s energy selection).

    Logically identical to generate_3d_conformer() in generate_raman_input.py:
    - EmbedMultipleConfs generates 20 candidate conformers (useRandomCoords=True)
    - MMFFOptimizeMoleculeConfs optimizes all of them (MMFF94s)
    - The MMFF-lowest-energy conformer that converged is selected
    - Returns a Chem.Mol object containing the single selected conformer
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit required: conda install -c conda-forge rdkit")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    num_confs = 20
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0
    params.useRandomCoords = True
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if not cids:
        raise RuntimeError(f"3D multi-conformer embedding failed: {smiles!r}")
    print(f"  [SMILES] Successfully embedded {len(cids)} initial conformers")

    conf_id = int(cids[0])
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s', numThreads=0)
        if results:
            valid = [
                (int(cids[i]), res[1])
                for i, res in enumerate(results)
                if res is not None and res[0] == 0 and res[1] is not None
            ]
            if valid:
                valid.sort(key=lambda x: x[1])
                conf_id = valid[0][0]
                print(f"  [SMILES] Selected lowest-MMFF94s-energy conformer (ID={conf_id}, E={valid[0][1]:.4f})")
            else:
                print(f"  [SMILES] No conformer converged under MMFF94s; using the first conformer")
        else:
            print(f"  [SMILES] MMFFOptimizeMoleculeConfs returned no result; using the first conformer")
    except Exception as mmff_e:
        print(f"  [SMILES] MMFF94s optimization error: {mmff_e}; using the first conformer")

    mol_out = Chem.Mol(mol)
    mol_out.RemoveAllConformers()
    try:
        mol_out.AddConformer(mol.GetConformer(conf_id), assignId=True)
    except ValueError:
        print(f"  [SMILES] Failed to retrieve conformer ID {conf_id}; falling back to the first")
        mol_out.AddConformer(mol.GetConformer(int(cids[0])), assignId=True)
    return mol_out


def _mol_to_xyz_file(mol, xyz_path: str, title: str = "") -> None:
    """Write the first conformer of an RDKit Mol to an XYZ file (same format as generate_raman_input.py)."""
    conf = mol.GetConformer()
    Path(xyz_path).parent.mkdir(parents=True, exist_ok=True)
    with open(xyz_path, "w", encoding="utf-8") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{title} - generated by SMILES via ETKDGv3+MMFF94s\n")
        for i in range(mol.GetNumAtoms()):
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            p = conf.GetAtomPosition(i)
            f.write(f"{sym} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")


def smiles_to_coords(smiles: str, seed: int = 42) -> Tuple[List[int], List[List[float]]]:
    """SMILES -> (z_list, pos_list); internally calls smiles_to_mol()."""
    mol = smiles_to_mol(smiles, seed=seed)
    conf = mol.GetConformer()
    z_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos_list = [
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ]
    return z_list, pos_list


def _smi_to_filename(smi: str, max_len: int = 60) -> str:
    """Convert a SMILES string to a filesystem-safe filename fragment."""
    safe = re.sub(r'[^\w\-]', '_', smi)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:max_len]


# --- Architecture auto-detection ---

def _detect_arch_from_ckpt(state: dict) -> dict:
    """Auto-detect architecture parameters from a checkpoint state_dict.

    Returns a dict containing some or all of the following keys:
      model_name  str           e.g. 'equiformer_v2_l3_m2'
      num_layers  int           actual layer count of the checkpoint
      enk_enabled bool          whether the checkpoint contains ENK parameters
      sphere_channels int       64 or 128
      lmax        int           3, 4, 6

    Detection logic:
      num_layers : backbone.blocks.N.*  max N+1
      lmax+C     : backbone.blocks.0.ga.so2_conv_1.fc_m0.weight  shape[1]
                   = (lmax+1)*2*C  (so2_conv_1 is called with 2*sphere_channels)
      enk_enabled: a key with the 'enk_' prefix exists in state_dict
    """
    result = {}

    # num_layers
    block_idx = {
        int(m.group(1))
        for k in state
        for m in [re.match(r'backbone\.blocks\.(\d+)\.', k)] if m
    }
    if block_idx:
        result["num_layers"] = max(block_idx) + 1

    # lmax + sphere_channels
    key_m0 = "backbone.blocks.0.ga.so2_conv_1.fc_m0.weight"
    if key_m0 in state:
        in_feat = state[key_m0].shape[1]
        for C in (128, 64):
            raw = in_feat / (2 * C)
            v = int(raw)
            if abs(v - raw) < 1e-6 and v >= 1:
                lmax = v - 1
                model_name = _LMAX_C_TO_MODEL.get((lmax, C))
                if model_name:
                    result["model_name"] = model_name
                    result["sphere_channels"] = C
                    result["lmax"] = lmax
                    break

    # ENK
    result["enk_enabled"] = any("enk_" in k for k in state)

    # ElectronPrior — detect presence and predictor mode (qcmol vs simg)
    _ep_pred_prefix = "electron_prior.predictor."
    _ep_pred_keys = {k for k in state if k.startswith(_ep_pred_prefix)}
    result["ep_enabled"] = len(_ep_pred_keys) > 0 or any(
        "nbo_prior" in k or "electron_prior" in k for k in state
    )
    if _ep_pred_keys:
        _is_qcmol = any(k.startswith(_ep_pred_prefix + "backbone.") for k in _ep_pred_keys)
        result["ep_predictor_mode"] = "qcmol" if _is_qcmol else "simg"

    # grid_resolution — from SO3_grid matrix shape [res, res, *]
    key_grid = "backbone.SO3_grid.0.0.to_grid_mat"
    if key_grid in state:
        result["grid_resolution"] = int(state[key_grid].shape[0])

    return result


def _load_raw_state(ckpt_path: str, device: torch.device) -> dict:
    """Load only the state dict (not into a model), for pre-detecting the architecture."""
    p = str(Path(ROOT / ckpt_path).resolve()) if not Path(ckpt_path).is_absolute() else str(ckpt_path)
    try:
        state = torch.load(p, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(p, map_location=device)
    # Unwrap training checkpoint format
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


# --- DetaNet loading (comparison curves) ---

def _load_state_dict_compat(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """Load a DetaNet state_dict (trainer-style forgiving merge + legacy activation format compatibility)."""

    def _has_named_token(key: str, token: str) -> bool:
        # Ensure token matching on path segments (e.g., ".lek.", ".lev.").
        return f".{token}." in f".{key}."

    def _is_compat_ignore_key(key: str) -> bool:
        return any(_has_named_token(key, ig) for ig in _DETANET_COMPAT_IGNORE_KEYS)

    def _legacy_activation_bases(key: str) -> List[str]:
        if not (key.endswith(".linear.weight") or key.endswith(".linear.bias")):
            return []
        base = key.rsplit(".linear.", 1)[0]
        bases = [base]
        # Some legacy checkpoints use an extra nested ".act" wrapper.
        if base.endswith(".act"):
            bases.append(base[:-4])
        return bases

    def _is_legacy_activation_linear_key(key: str, model_keys: set) -> bool:
        # Detect legacy activation parameters by checking whether a matching
        # alpha/beta target exists in current model keys.
        for base in _legacy_activation_bases(key):
            if f"{base}.alpha" in model_keys or f"{base}.beta" in model_keys:
                return True
        return False

    def _is_activation_alpha_beta_key(key: str, ckpt_keys: set) -> bool:
        # Detect alpha/beta missing keys that are explainable by legacy
        # *.linear.{weight,bias} activation parameters in checkpoint.
        if not (key.endswith(".alpha") or key.endswith(".beta")):
            return False
        base = key.rsplit(".", 1)[0]
        legacy_candidates = (
            f"{base}.linear.weight",
            f"{base}.linear.bias",
            f"{base}.act.linear.weight",
            f"{base}.act.linear.bias",
        )
        return any(c in ckpt_keys for c in legacy_candidates)

    def _short_list(keys: List[str], limit: int = 12) -> str:
        if not keys:
            return "[]"
        if len(keys) <= limit:
            return str(keys)
        return str(keys[:limit] + [f"... (+{len(keys) - limit} more)"])

    try:
        state = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]

    normalized_state = {}
    for k, v in state.items():
        nk = k
        while nk.startswith("module."):
            nk = nk[len("module.") :]
        while nk.startswith("model."):
            nk = nk[len("model.") :]
        normalized_state[nk] = v
    state = {
        k: v
        for k, v in normalized_state.items()
    }

    # Training-side forgiving merge: exact shape first, then dim-0 partial copy.
    current = model.state_dict()
    exact_keys: List[str] = []
    partial_keys: List[str] = []
    skipped_shape: List[str] = []
    unexpected_ckpt: List[str] = []
    for key, value in state.items():
        if key not in current:
            unexpected_ckpt.append(key)
            continue
        target = current[key]
        if tuple(value.shape) == tuple(target.shape):
            current[key] = value
            exact_keys.append(key)
            continue
        if (
            value.ndim == target.ndim
            and value.ndim >= 1
            and tuple(value.shape[1:]) == tuple(target.shape[1:])
            and value.shape[0] <= target.shape[0]
        ):
            patched = target.clone()
            patched[: value.shape[0]] = value
            current[key] = patched
            partial_keys.append(key)
            continue
        skipped_shape.append(key)

    model.load_state_dict(current, strict=False)

    loaded_set = set(exact_keys) | set(partial_keys)
    model_unloaded = [k for k in current.keys() if k not in loaded_set]
    model_key_set = set(current.keys())
    ckpt_key_set = set(state.keys())

    ignored_unexpected = [
        k for k in unexpected_ckpt
        if _is_legacy_activation_linear_key(k, model_key_set) or _is_compat_ignore_key(k)
    ]
    real_unexpected = [k for k in unexpected_ckpt if k not in ignored_unexpected]

    ignored_missing = [
        k for k in model_unloaded
        if _is_activation_alpha_beta_key(k, ckpt_key_set) or _is_compat_ignore_key(k)
    ]
    real_missing = [k for k in model_unloaded if k not in ignored_missing]

    ignored_shape = [k for k in skipped_shape if _is_legacy_activation_linear_key(k, model_key_set)]
    real_shape = [k for k in skipped_shape if k not in ignored_shape]

    if ignored_unexpected or ignored_missing or ignored_shape:
        print(
            "[INFO] DetaNet legacy-activation weight compatibility path enabled: "
            f"ignored_unexpected={len(ignored_unexpected)}, "
            f"ignored_missing={len(ignored_missing)}, "
            f"ignored_shape={len(ignored_shape)}"
        )

    if real_unexpected:
        print(f"[WARN] Unexpected keys (effective): {_short_list(real_unexpected)}")

    if real_missing or real_shape:
        raise RuntimeError(
            "DetaNet checkpoint has incompatible keys after compatibility merge: "
            f"missing={_short_list(real_missing)}, shape_mismatch={_short_list(real_shape)}"
        )

    if not exact_keys and not partial_keys:
        raise RuntimeError("DetaNet checkpoint load failed: no usable parameter key matched model state.")


def _infer_detanet_max_atomic_number(params_path: Path, device: torch.device) -> Optional[int]:
    """Infer DetaNet max_atomic_number from checkpoint embedding rows.

    Training-time DetaNet uses embedding shape [max_atomic_number + 1, hidden].
    """
    try:
        state = _load_raw_state(str(params_path), device)
    except Exception as e:
        print(f"[WARN] Could not infer max_atomic_number from the DetaNet checkpoint: {e}")
        return None

    emb = state.get("Embedding.nuclare_emb.weight", None)
    if isinstance(emb, torch.Tensor) and emb.ndim == 2 and int(emb.shape[0]) >= 2:
        return int(emb.shape[0]) - 1

    for key, val in state.items():
        if key.endswith("nuclare_emb.weight") and isinstance(val, torch.Tensor) and val.ndim == 2:
            if int(val.shape[0]) >= 2:
                return int(val.shape[0]) - 1
    return None


def _detanet_elec_norm_denom(max_atomic_number: int) -> float:
    """Return the raw max used by DetaNet's electronic feature normalization."""
    zmax = int(max_atomic_number)
    if zmax >= 29:
        return 10.0
    if zmax >= 10:
        return 6.0
    if zmax >= 9:
        return 4.0
    if zmax >= 7:
        return 3.0
    if zmax >= 2:
        return 2.0
    return 1.0


def _detanet_supervised_z(ckpt_max_atomic_number: int) -> List[int]:
    """Best-effort supervised rows for a DetaNet checkpoint."""
    ckpt_max = int(ckpt_max_atomic_number)
    if ckpt_max <= 9:
        return [z for z in _DETANET_QM9_SUPERVISED_Z if z <= ckpt_max]
    return list(range(1, ckpt_max + 1))


def _detanet_fill_expanded_embeddings(
    model: torch.nn.Module,
    *,
    kind: str,
    ckpt_max_atomic_number: int,
    effective_max_atomic_number: int,
    mode: str,
) -> None:
    """Stabilize DetaNet embedding expansion for checkpoint-OOD elements."""
    mode = str(mode or "mean_periodic").lower()
    if mode == "random":
        return

    ckpt_max = int(ckpt_max_atomic_number)
    effective_max = int(effective_max_atomic_number)
    if effective_max <= ckpt_max:
        return
    if mode == "error":
        raise ValueError(
            f"DetaNet[{kind}] checkpoint supports Z<= {ckpt_max}, "
            f"but inference requested Z<= {effective_max}."
        )
    if mode not in {"mean", "mean_periodic"}:
        raise ValueError(f"Unsupported --detanet_ood_embedding mode: {mode}")

    emb = getattr(model, "Embedding", None)
    if emb is None or not hasattr(emb, "nuclare_emb"):
        return

    supervised = [
        z for z in _detanet_supervised_z(ckpt_max)
        if 0 < z <= ckpt_max and z < emb.nuclare_emb.weight.shape[0]
    ]
    if not supervised:
        supervised = [z for z in range(1, min(ckpt_max + 1, emb.nuclare_emb.weight.shape[0]))]
    if not supervised:
        return

    with torch.no_grad():
        nuclear = emb.nuclare_emb.weight
        mean_vec = nuclear[supervised].mean(dim=0)
        filled = []
        start = min(ckpt_max + 1, nuclear.shape[0])
        stop = min(effective_max + 1, nuclear.shape[0])
        for z in range(start, stop):
            src = _DETANET_PERIODIC_SURROGATE_Z.get(z) if mode == "mean_periodic" else None
            if src in supervised and src < start:
                nuclear[z].copy_(nuclear[src])
                filled.append(f"{z}->{src}")
            else:
                nuclear[z].copy_(mean_vec)
                filled.append(f"{z}->mean")

        elec = getattr(emb, "elec", None)
        if isinstance(elec, torch.Tensor):
            eff_den = _detanet_elec_norm_denom(effective_max)
            ckpt_den = _detanet_elec_norm_denom(ckpt_max)
            if ckpt_den > 0 and eff_den != ckpt_den:
                elec.mul_(eff_den / ckpt_den)

        elec_emb = getattr(emb, "elec_emb", None)
        if elec_emb is not None and hasattr(elec_emb, "weight") and isinstance(elec, torch.Tensor):
            known_elec = elec[supervised]
            active_cols = set(torch.nonzero(known_elec.abs().sum(dim=0) > 0, as_tuple=False).flatten().tolist())
            inactive_cols = [c for c in range(elec_emb.weight.shape[1]) if c not in active_cols]
            if mode == "mean_periodic":
                col_map = {6: 2, 7: 3, 8: 4, 9: 5, 10: 2, 11: 3, 14: 4, 15: 5}
                for dst, src in col_map.items():
                    if (
                        dst in inactive_cols
                        and src in active_cols
                        and dst < elec_emb.weight.shape[1]
                        and src < elec_emb.weight.shape[1]
                    ):
                        elec_emb.weight[:, dst].copy_(elec_emb.weight[:, src])
                for dst in (12, 13):
                    if dst in inactive_cols and dst < elec_emb.weight.shape[1]:
                        elec_emb.weight[:, dst].zero_()
            else:
                for dst in inactive_cols:
                    elec_emb.weight[:, dst].zero_()

    preview = ", ".join(filled[:8])
    if len(filled) > 8:
        preview += f", ... (+{len(filled) - 8})"
    print(
        f"[INFO] DetaNet[{kind}] OOD embedding guard active: mode={mode}, "
        f"checkpoint_Z<= {ckpt_max}, model_Z<= {effective_max}; nuclear {preview}"
    )


def _build_detanet(kind: str, device: torch.device, params_path: Path, max_number: int = 9):
    """Build a single-task DetaNet and load its weights."""
    from detanet_nets.detanet import DetaNet
    params_path = params_path if params_path.is_absolute() else (ROOT / params_path).resolve()

    inferred_max = _infer_detanet_max_atomic_number(params_path, device)
    effective_max = int(max_number)
    if inferred_max is not None:
        effective_max = max(effective_max, int(inferred_max))
        if effective_max != int(max_number):
            print(
                f"[INFO] DetaNet[{kind}] auto-expanding max_atomic_number from checkpoint: "
                f"{int(max_number)} -> {effective_max}"
            )

    cfg_map = {
        "Hi":       dict(scalar_outsize=1, irreps_out=None,  summation=False,
                         norm=False, out_type="scalar",   grad_type="Hi"),
        "Hij":      dict(scalar_outsize=1, irreps_out=None,  summation=False,
                         norm=False, out_type="scalar",   grad_type="Hij"),
        "dedipole": dict(scalar_outsize=1, irreps_out="1o",  summation=False,
                         norm=False, out_type="dipole",   grad_type="dipole"),
        "depolar":  dict(scalar_outsize=2, irreps_out="2e",  summation=False,
                         norm=False, out_type="2_tensor", grad_type="polar"),
    }
    common = dict(
        num_features=128, act="swish", maxl=3, num_block=3,
        radial_type="trainable_bessel", num_radial=32, attention_head=8,
        rc=5.0, dropout=0.0, use_cutoff=False,
        max_atomic_number=effective_max, atom_ref=None, scale=1.0, device=device,
    )
    model = DetaNet(**common, **cfg_map[kind])
    _load_state_dict_compat(model, params_path, device)
    if inferred_max is not None:
        _detanet_fill_expanded_embeddings(
            model,
            kind=kind,
            ckpt_max_atomic_number=int(inferred_max),
            effective_max_atomic_number=int(effective_max),
            mode=DETANET_OOD_EMBEDDING_MODE,
        )
    model.to(device).eval()
    return model



def _load_state_dict_safe(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = False,
) -> None:
    ckpt_path = str(Path(ROOT / ckpt_path).resolve()) \
        if not Path(ckpt_path).is_absolute() else ckpt_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Weight file does not exist: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    # Unwrap training checkpoint format: {"model_state_dict": {...}, "optimizer_state_dict": ..., ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    result = model.load_state_dict(cleaned, strict=strict)
    if result.missing_keys:
        print(f"  [WARN] Missing keys ({len(result.missing_keys)}): "
              f"{result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        print(f"  [WARN] Unexpected keys ({len(result.unexpected_keys)}): "
              f"{result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")


# --- Electron Prior configuration ---

def _build_ep_config(args, ep_enabled_in_ckpt: bool = False,
                     ep_predictor_state=None,
                     ep_predictor_mode: str = "qcmol") -> Optional[dict]:
    """Build the electron_prior_config dict (or None) from CLI arguments.

    When EP parameters are detected in the checkpoint (ep_enabled_in_ckpt=True)
    and --electron_prior_ckpt is not supplied, the predictor sub-state_dict
    extracted from the main checkpoint is used to initialize NBOPriorBranch,
    enabling end-to-end inference without a separate NBO weight file.
    """
    # Auto-configuration path: checkpoint contains EP weights, user supplied no standalone NBO ckpt
    if ep_enabled_in_ckpt and ep_predictor_state and not args.electron_prior_ckpt:
        _mode = ep_predictor_mode
        if args.electron_prior_mode not in ("off", ""):
            _mode = args.electron_prior_mode  # CLI explicit override
        _stats = args.electron_prior_stats
        if _stats:
            _stats = str(Path(ROOT / _stats).resolve()) if not Path(_stats).is_absolute() else _stats
        print(
            f"  [auto-config] ElectronPrior: restoring embedded predictor weights from main checkpoint (mode={_mode})\n"
            f"  End-to-end inference: NBOFoundationModel predicts NBO features from atomic geometry, no external quantum-chemistry software required.",
            flush=True,
        )
        if not _stats:
            print(
                "  [WARN] Auto-restored EP only obtained predictor_state_dict; qcmol/simg statistics were not restored.\n"
                "  Falling back to feature_scale for output scaling; to match training-time de-normalization, pass"
                " --electron_prior_stats <norm_stats.pt/qcmol_stats.pt> explicitly.",
                flush=True,
            )
        return dict(
            mode=_mode,
            checkpoint_path="",           # empty path -> NBOPriorBranch uses predictor_state_dict
            predictor_state_dict=ep_predictor_state,
            stats_path=_stats or None,
            hidden_dim=args.hidden_nf,
            max_atomic_number=35,
            feature_scale=args.electron_prior_scale,
            use_auxiliary=args.electron_prior_use_aux,
            freeze_predictor=args.electron_prior_freeze,
            runtime_mode=args.electron_prior_runtime_mode,
        )

    if args.electron_prior_mode == "off":
        if ep_enabled_in_ckpt:
            print(
                f"  [INFO] Checkpoint contains ElectronPrior parameters, but --electron_prior_mode=off and"
                f" --electron_prior_ckpt is unspecified; EP is not activated.\n"
                f"  Pass --electron_prior_mode qcmol to enable end-to-end inference automatically.",
                flush=True,
            )
        return None

    if not args.electron_prior_ckpt:
        if ep_enabled_in_ckpt:
            print(
                f"  [WARN] Checkpoint contains ElectronPrior parameters, but --electron_prior_ckpt is unspecified; EP is not activated.",
                flush=True,
            )
        return None

    ckpt = args.electron_prior_ckpt
    ckpt = str(Path(ROOT / ckpt).resolve()) if not Path(ckpt).is_absolute() else ckpt
    stats = args.electron_prior_stats
    if stats:
        stats = str(Path(ROOT / stats).resolve()) if not Path(stats).is_absolute() else stats
    return dict(
        mode=args.electron_prior_mode,
        checkpoint_path=ckpt,
        stats_path=stats or None,
        hidden_dim=args.hidden_nf,
        max_atomic_number=35,
        feature_scale=args.electron_prior_scale,
        use_auxiliary=args.electron_prior_use_aux,
        freeze_predictor=args.electron_prior_freeze,
        runtime_mode=args.electron_prior_runtime_mode,
    )


# --- Multi-task model loading (hii / hij / dedipole) — supports V1/V2/ENK/EP + auto-detection ---

def _load_multitask(
    task: str,
    ckpt_path: str,
    mode: str,
    device: torch.device,
    args,
    force_disable_enk: bool = False,
) -> torch.nn.Module:
    """Load CleanEquiformerMultiTask, with auto-detection of lmax/num_layers/ENK from the checkpoint."""
    from nets.clean_equiformer_multitask import CleanEquiformerMultiTask

    _is_v2 = mode.startswith("equiformer_v2")
    _model_name = args.v2_model_name if _is_v2 else "graph_attention_transformer_nonlinear_l2"
    _num_layers = args.num_layers
    _enk = args.enk_enabled or mode in ("equiformer_v2_enk", "equiformer_v2_enk_ep")
    _grid_resolution = args.v2_grid_resolution
    if force_disable_enk:
        _enk = False

    _ep_has_ckpt = False
    _ep_predictor_state = None
    _ep_predictor_mode = "qcmol"
    if _is_v2 and ckpt_path:
        try:
            _state = _load_raw_state(ckpt_path, device)
            _arch = _detect_arch_from_ckpt(_state)
            if _arch.get("model_name"):
                _model_name = _arch["model_name"]
                print(f"  [auto-detect] {task}: model_name={_model_name}")
            if _arch.get("num_layers"):
                _num_layers = _arch["num_layers"]
                print(f"  [auto-detect] {task}: num_layers={_num_layers}")
            if _arch.get("grid_resolution"):
                _grid_resolution = _arch["grid_resolution"]
                print(f"  [auto-detect] {task}: grid_resolution={_grid_resolution}")
            if not force_disable_enk and "enk_enabled" in _arch and mode == "equiformer_v2":
                if _arch["enk_enabled"]:
                    _enk = True
                    print(f"  [auto-detect] {task}: checkpoint contains ENK parameters, enabling automatically")
            elif force_disable_enk and _arch.get("enk_enabled"):
                print(f"  [INFO] {task}: checkpoint contains ENK parameters, but ENK-off mode forces them off")
            if "ep_enabled" in _arch and _arch["ep_enabled"]:
                _ep_has_ckpt = True
                _ep_predictor_mode = _arch.get("ep_predictor_mode", "qcmol")
                print(f"  [auto-detect] {task}: checkpoint contains ElectronPrior parameters (mode={_ep_predictor_mode})")
                # Extract the predictor sub-state_dict so NBOPriorBranch can initialize without a separate weight file
                _ep_prefix = "electron_prior.predictor."
                _ep_sub = {k[len(_ep_prefix):]: v for k, v in _state.items() if k.startswith(_ep_prefix)}
                if _ep_sub:
                    _ep_predictor_state = _ep_sub
        except Exception as e:
            print(f"  [WARN] Architecture auto-detection failed ({task}): {e}; falling back to CLI parameters")

    if force_disable_enk:
        _enk = False

    _ep_config = None
    if _is_v2 and mode == "equiformer_v2_enk_ep":
        _branch_tag = _TASK_TO_NBO_BRANCH.get(task, task)
        if _branch_tag_in_ep_branches(args, _branch_tag):
            _ep_config = _build_ep_config(
                args,
                ep_enabled_in_ckpt=_ep_has_ckpt,
                ep_predictor_state=_ep_predictor_state,
                ep_predictor_mode=_ep_predictor_mode,
            )
        else:
            print(f"  [{task}] EP skipped (--nbo_ep_branches excludes this branch; using pure ENK)")
            _ep_has_ckpt = False

    model = CleanEquiformerMultiTask(
        task=task,
        hidden_nf=args.hidden_nf,
        model_name=_model_name,
        radius=args.radius,
        num_basis=args.num_basis,
        num_layers=_num_layers,
        drop_path=0.0,
        dropout=0.0,
        max_num_neighbors=args.v2_max_neighbors,
        num_gaussians=args.v2_num_gaussians,
        use_gradient_checkpointing=False,
        grid_resolution=_grid_resolution,
        use_gate_act=args.v2_use_gate_act,
        use_grid_mlp=args.v2_use_grid_mlp,
        enk_enabled=_enk,
        enk_init_r_bias=args.enk_init_r_bias,
        enk_init_q_bias=args.enk_init_q_bias,
        electron_prior_config=_ep_config,
        electron_prior_heads=args.electron_prior_heads,
    )
    _load_state_dict_safe(model, ckpt_path, device)
    model.to(device).eval()
    return model


# --- depolar model loading (V1/V2/ENK/EP + auto-detection) ---

def _load_depolar(
    ckpt_path: str,
    mode: str,
    device: torch.device,
    args,
    force_disable_enk: bool = False,
) -> torch.nn.Module:
    """Load the depolar model, with auto-detection of lmax/num_layers/ENK."""
    _is_v2 = mode.startswith("equiformer_v2")
    _model_name = args.v2_model_name if _is_v2 else "graph_attention_transformer_nonlinear_l2"
    _num_layers = args.num_layers
    _enk = args.enk_enabled or mode in ("equiformer_v2_enk", "equiformer_v2_enk_ep")
    _grid_resolution = args.v2_grid_resolution
    if force_disable_enk:
        _enk = False

    _ep_has_ckpt = False
    _ep_predictor_state = None
    _ep_predictor_mode = "qcmol"
    if _is_v2 and ckpt_path:
        try:
            _state = _load_raw_state(ckpt_path, device)
            _arch = _detect_arch_from_ckpt(_state)
            if _arch.get("model_name"):
                _model_name = _arch["model_name"]
                print(f"  [auto-detect] depolar: model_name={_model_name}")
            if _arch.get("num_layers"):
                _num_layers = _arch["num_layers"]
                print(f"  [auto-detect] depolar: num_layers={_num_layers}")
            if _arch.get("grid_resolution"):
                _grid_resolution = _arch["grid_resolution"]
                print(f"  [auto-detect] depolar: grid_resolution={_grid_resolution}")
            if not force_disable_enk and "enk_enabled" in _arch and mode == "equiformer_v2":
                if _arch["enk_enabled"]:
                    _enk = True
                    print(f"  [auto-detect] depolar: checkpoint contains ENK parameters, enabling automatically")
            elif force_disable_enk and _arch.get("enk_enabled"):
                print("  [INFO] depolar: checkpoint contains ENK parameters, but ENK-off mode forces them off")
            if "ep_enabled" in _arch and _arch["ep_enabled"]:
                _ep_has_ckpt = True
                _ep_predictor_mode = _arch.get("ep_predictor_mode", "qcmol")
                print(f"  [auto-detect] depolar: checkpoint contains ElectronPrior parameters (mode={_ep_predictor_mode})")
                _ep_prefix = "electron_prior.predictor."
                _ep_sub = {k[len(_ep_prefix):]: v for k, v in _state.items() if k.startswith(_ep_prefix)}
                if _ep_sub:
                    _ep_predictor_state = _ep_sub
        except Exception as e:
            print(f"  [WARN] Architecture auto-detection failed (depolar): {e}; falling back to CLI parameters")

    if force_disable_enk:
        _enk = False

    _ep_config = None
    if mode == "equiformer_v2_enk_ep":
        if _branch_tag_in_ep_branches(args, "dp"):
            _ep_config = _build_ep_config(
                args,
                ep_enabled_in_ckpt=_ep_has_ckpt,
                ep_predictor_state=_ep_predictor_state,
                ep_predictor_mode=_ep_predictor_mode,
            )
        else:
            print("  [depolar] EP skipped (--nbo_ep_branches excludes this branch; using pure ENK)")
            _ep_has_ckpt = False

    if mode == "clean_equiformer":
        from nets.clean_equiformer_polar import CleanEquiformerPolar
        model = CleanEquiformerPolar(
            hidden_nf=args.hidden_nf,
            model_name=_model_name,
            radius=args.radius,
            num_basis=args.num_basis,
            num_layers=_num_layers,
            drop_path=0.0,
            dropout=0.0,
            summation=False,
            grad_type="polar",
            enk_enabled=_enk,
            enk_init_r_bias=args.enk_init_r_bias,
            enk_init_q_bias=args.enk_init_q_bias,
        )
    else:
        from nets.clean_equiformer_polar_ext import CleanEquiformerPolarExt
        model = CleanEquiformerPolarExt(
            hidden_nf=args.hidden_nf,
            model_name=_model_name,
            radius=args.radius,
            num_basis=args.num_basis,
            num_layers=_num_layers,
            drop_path=0.0,
            dropout=0.0,
            summation=False,
            grad_type="polar",
            spectral_mask_enabled=False,
            spectral_mask_heads=4,
            tensor_sdm_enabled=False,
            tensor_sdm_heads=4,
            electron_prior_config=_ep_config,
            electron_prior_heads=args.electron_prior_heads,
            enk_enabled=_enk,
            enk_init_r_bias=args.enk_init_r_bias,
            enk_init_q_bias=args.enk_init_q_bias,
            max_num_neighbors=args.v2_max_neighbors,
            num_gaussians=args.v2_num_gaussians,
            use_gradient_checkpointing=False,
            grid_resolution=_grid_resolution,
            use_gate_act=args.v2_use_gate_act,
            use_grid_mlp=args.v2_use_grid_mlp,
        )

    _load_state_dict_safe(model, ckpt_path, device)
    model.to(device).eval()

# EP gate diagnostics: when EP is enabled, print the learned injection gate values
    if _ep_config is not None and hasattr(model, 'so3_ep_injector'):
        gate_vals = []
        for blk_k, blk_mod in model.so3_ep_injector.prior_blocks.items():
            g = torch.tanh(blk_mod.gate).item()
            gate_vals.append(f"block{blk_k}={g:.4f}")
        print(f"  [DIAG-depolar] EP global_gate(tanh) -> {', '.join(gate_vals)}")
        if all(abs(torch.tanh(blk_mod.gate).item()) < 0.05
               for blk_mod in model.so3_ep_injector.prior_blocks.values()):
            print("  [WARN-depolar] All EP global_gate < 0.05 -- EP injection is near zero!"
                  " Verify that the EP gate was sufficiently optimized during training.")

    return model


# --- Online skeleton generation (equivalent to tools/build_qm9s_skeletons.py on the training side) ---

# Covalent radii (Angstrom), kept consistent with detanet_nets/electron_prior.py
_COVALENT_RADII_ANG = {
    1:  0.31,  5:  0.82,  6:  0.76,  7:  0.71,  8:  0.66,  9:  0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 34: 1.20, 35: 1.20, 53: 1.39,
}
_COVALENT_RADII_DEFAULT = 0.80


def _infer_chem_bonds_for_skeleton(
    z: torch.Tensor, pos: torch.Tensor, tolerance: float = 0.40,
) -> torch.Tensor:
    """Infer covalent-bond topology from atom types and 3D coordinates; returns [2, E_bond] (row < col)."""
    n = int(z.size(0))
    device = pos.device
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    radii = [_COVALENT_RADII_ANG.get(int(zi), _COVALENT_RADII_DEFAULT) for zi in z.cpu().tolist()]
    r = pos.new_tensor(radii)
    thresh = r.unsqueeze(1) + r.unsqueeze(0) + tolerance
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = diff.norm(dim=-1)
    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
    mask = (dist[i_idx, j_idx] < thresh[i_idx, j_idx]) & (dist[i_idx, j_idx] > 0.1)
    if not mask.any():
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.stack([i_idx[mask], j_idx[mask]], dim=0).contiguous()


def _build_radius_graph_skeleton(pos: torch.Tensor, r: float = 5.0) -> torch.Tensor:
    """Build a radius graph on global nodes (atoms + bonds + LPs), identical to the training side."""
    if pos.size(0) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)
    dist = torch.cdist(pos, pos)
    mask = (dist < r) & (dist > 1e-6)
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


_LP_PREDICTOR_CACHE = {}


def _get_cached_lp_predictor(lp_ckpt: str, device: torch.device, predictor_cls):
    key = (str(Path(lp_ckpt).resolve()) if lp_ckpt else "", str(device))
    predictor = _LP_PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = predictor_cls(lp_ckpt, device=str(device))
        _LP_PREDICTOR_CACHE[key] = predictor
    return predictor


def _predict_lp_atoms(z: torch.Tensor, pos: torch.Tensor,
                      bond_pairs: list, lp_ckpt: str,
                      device: torch.device) -> list:
    """Invoke LPPredictor to identify atoms carrying a lone pair (LP); returns a list of atom indices."""
    try:
        from tools.lp_predictor import LPPredictor
    except ImportError:
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            from lp_predictor import LPPredictor
        except ImportError:
            print("  [WARN] LP predictor unavailable; skipping LP nodes.")
            return []

    if not os.path.exists(lp_ckpt):
        print(f"  [WARN] LP weight not found: {lp_ckpt}; skipping LP nodes.")
        return []

    try:
        from rdkit import Chem
    except ImportError:
        print("  [WARN] RDKit unavailable; skipping LP nodes.")
        return []

    # Build an RDKit Mol from z + bond_pairs (consistent with tools/build_qm9s_skeletons._build_mol_from_item)
    mol_edit = Chem.RWMol()
    for zi in z.cpu().tolist():
        mol_edit.AddAtom(Chem.Atom(int(zi)))
    for a, b in bond_pairs:
        try:
            mol_edit.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        except Exception:
            continue
    mol = mol_edit.GetMol()

    try:
        predictor = _get_cached_lp_predictor(lp_ckpt, device, LPPredictor)
        return predictor.predict_lp_atom_map(mol)
    except Exception as e:
        print(f"  [WARN] LP prediction failed: {e}; skipping LP nodes.")
        return []


def build_skeleton_online(
    z: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    lp_ckpt: Optional[str] = None,
    skeleton_radius: float = 5.0,
):
    """
    End-to-end real-time skeleton generation from atom types and coordinates,
    fully equivalent to tools/build_qm9s_skeletons on the training side.

    Generation logic:
      1. Infer chemical-bond topology from covalent radii -> bond_pairs
      2. (Optional) LP predictor infers the lone-pair count per atom -> lp_atoms
      3. Build the global-node graph:
         - Nodes: [atoms | bonds | LPs]  (node_type: 0=atom, 1=bond, 2=LP)
         - Coordinates: [atom_pos | bond_midpoints | lp_pos_on_parent]
         - atom_to_nbo_index: atom -> NBO-node connections
         - atom_bond_index: chemical-bond atom pairs
         - interaction_edge_index: global radius graph (r=5.0 Angstrom)

    Returns a SkelData-compatible Data object that can be passed directly as model(..., data=skel_data).
    """
    from torch_geometric.data import Data

    pos = pos.to(device).float()
    z = z.to(device)
    num_atoms = int(pos.size(0))

    # (1) Infer chemical bonds
    bond_idx = _infer_chem_bonds_for_skeleton(z, pos)
    if bond_idx.numel() > 0:
        bond_pairs = list(zip(bond_idx[0].cpu().tolist(), bond_idx[1].cpu().tolist()))
    else:
        bond_pairs = []
    num_bonds = len(bond_pairs)

    # (2) LP prediction
    lp_atoms = []
    if lp_ckpt:
        lp_atoms = _predict_lp_atoms(z, pos, bond_pairs, lp_ckpt, device)
    num_lps = len(lp_atoms)

    # (3) Global-node construction
    total_nodes = num_atoms + num_bonds + num_lps

    # node_type
    node_type = torch.zeros(total_nodes, dtype=torch.long, device=device)
    if num_bonds > 0:
        node_type[num_atoms: num_atoms + num_bonds] = 1
    if num_lps > 0:
        node_type[num_atoms + num_bonds:] = 2

    # bond midpoint coordinates
    if num_bonds > 0:
        atoms_a = torch.tensor([a for a, _ in bond_pairs], dtype=torch.long, device=device)
        atoms_b = torch.tensor([b for _, b in bond_pairs], dtype=torch.long, device=device)
        bond_pos = 0.5 * (pos[atoms_a] + pos[atoms_b])
        atom_bond_index = torch.stack([atoms_a, atoms_b], dim=0)
    else:
        bond_pos = torch.zeros((0, 3), dtype=torch.float32, device=device)
        atom_bond_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    # LP coordinates (placed at the parent atom position)
    if num_lps > 0:
        lp_parent = torch.tensor(lp_atoms, dtype=torch.long, device=device)
        lp_pos = pos[lp_parent]
    else:
        lp_pos = torch.zeros((0, 3), dtype=torch.float32, device=device)

    # atom_to_nbo_index: atom -> bond nodes + atom -> LP nodes
    atom_to_nbo_edges = []
    bond_offset = num_atoms
    lp_offset = num_atoms + num_bonds
    for bond_idx_i, (a, b) in enumerate(bond_pairs):
        gidx = bond_offset + bond_idx_i
        atom_to_nbo_edges.append((a, gidx))
        atom_to_nbo_edges.append((b, gidx))
    for lp_idx_i, atom_idx in enumerate(lp_atoms):
        atom_to_nbo_edges.append((atom_idx, lp_offset + lp_idx_i))

    if atom_to_nbo_edges:
        atom_to_nbo_index = torch.tensor(atom_to_nbo_edges, dtype=torch.long, device=device).t().contiguous()
    else:
        atom_to_nbo_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    # pos_global: [atom_pos, bond_midpoints, lp_pos]
    pos_global = torch.cat([pos, bond_pos, lp_pos], dim=0)

    # interaction_edge_index: global radius graph
    interaction_edge_index = _build_radius_graph_skeleton(pos_global, r=skeleton_radius)

    # Construct the Data object (fields compatible with SkelData)
    skel_data = Data(
        pos=pos,
        z=z,
        node_type=node_type,
        pos_global=pos_global,
        atom_to_nbo_index=atom_to_nbo_index,
        interaction_edge_index=interaction_edge_index,
        atom_bond_index=atom_bond_index,
        num_global_nodes=torch.tensor([total_nodes]),
    )

    print(f"  [skeleton] Online generation complete: {num_atoms} atoms, {num_bonds} bonds, {num_lps} LPs, "
          f"{total_nodes} global nodes, {interaction_edge_index.size(1)} interaction edges")

    return skel_data


# --- Inference helpers ---

def _collect_point_a_hook_states(model: torch.nn.Module) -> list:
    states = []
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None) if backbone is not None else None
    if blocks is None:
        return states
    for blk in blocks:
        if hasattr(blk, "_point_a_hook"):
            states.append((blk, getattr(blk, "_point_a_hook", None)))
    return states


def _disable_point_a_hooks(model: torch.nn.Module) -> list:
    states = _collect_point_a_hook_states(model)
    for blk, _ in states:
        setattr(blk, "_point_a_hook", None)
    return states


def _restore_point_a_hooks(states: list) -> None:
    for blk, hook in states:
        setattr(blk, "_point_a_hook", hook)


def _hard_clear_ep_injector_cache(model: torch.nn.Module) -> None:
    injector = getattr(model, "so3_ep_injector", None)
    if injector is None:
        return
    for key in ["_atom_prior", "_edge_prior", "_edge_index"]:
        if hasattr(injector, key):
            setattr(injector, key, None)


def _run_with_ep_cache_guard(model: torch.nn.Module, forward_fn):
    # Always clear residual injector state before each branch forward.
    _hard_clear_ep_injector_cache(model)
    hook_states = []
    try:
        # If runtime EP is disabled on an EP-capable model, force-disable point-A hooks
        # to avoid accidental consumption of stale cached priors.
        if hasattr(model, "so3_ep_injector") and not _is_ep_runtime_active(model):
            hook_states = _disable_point_a_hooks(model)
        return forward_fn()
    finally:
        _hard_clear_ep_injector_cache(model)
        if hook_states:
            _restore_point_a_hooks(hook_states)


# task -> CLI branch tag mapping (hii/hij/dd/dp)
_TASK_TO_NBO_BRANCH = {"hii": "hii", "hij": "hij", "dedipole": "dd", "depolar": "dp"}


def _branch_tag_in_ep_branches(args, branch_tag: str) -> bool:
    """Check if a NBO branch (hii/hij/dd/dp) should use Electron Prior."""
    raw = (getattr(args, "nbo_ep_branches", "") or "").strip()
    if not raw:
        return True  # unspecified -> all branches inherit --electron_prior_ckpt behavior (backward compatibility)
    allowed = set(b.strip() for b in raw.split(",") if b.strip())
    return branch_tag in allowed


def _is_ep_runtime_active(model: torch.nn.Module) -> bool:
    # Runtime-active means: model keeps EP enabled flag AND has an EP module.
    return bool(getattr(model, "electron_prior_enabled", False)) and hasattr(model, "electron_prior")

def _infer_hii(model, pos, z, device, data=None):
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    def _forward():
        with torch.enable_grad():
            return model(pos=pos.to(device), z=z.to(device), batch=batch, data=data).detach()
    return _run_with_ep_cache_guard(model, _forward)


def _infer_hij(model, pos, z, edge_index, device, data=None):
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    def _forward():
        with torch.enable_grad():
            return model(
                pos=pos.to(device), z=z.to(device), batch=batch,
                edge_index=edge_index.to(device), data=data,
            ).detach()
    return _run_with_ep_cache_guard(model, _forward)


def _infer_dedipole(model, pos, z, device, data=None):
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    def _forward():
        with torch.enable_grad():
            return model(pos=pos.to(device), z=z.to(device), batch=batch, data=data).detach()
    return _run_with_ep_cache_guard(model, _forward)


def _infer_depolar(model, pos, z, device, data=None):
    pos_in = pos.to(device).requires_grad_(True)
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    def _forward():
        with torch.enable_grad():
            return model(pos=pos_in, z=z.to(device), batch=batch, data=data).detach()
    return _run_with_ep_cache_guard(model, _forward)


def _print_ep_runtime_debug(tag: str, model: torch.nn.Module) -> None:
    prior = getattr(model, "electron_prior", None)
    if prior is None:
        return
    dbg = getattr(prior, "last_debug", None)
    if not isinstance(dbg, dict) or not dbg:
        return
    print(
        f"  [DIAG-{tag}] EP atom_gate={dbg.get('atom_gate', float('nan')):.4f} "
        f"edge_gate={dbg.get('edge_gate', float('nan')):.4f} "
        f"bond_edge_ratio={dbg.get('bond_edge_ratio', float('nan')):.4f} "
        f"interaction_enabled={int(dbg.get('interaction_prior_enabled', 0.0))} "
        f"interaction_mapped={int(dbg.get('interaction_edges_mapped', 0.0))}/"
        f"{int(dbg.get('interaction_edges_total', 0.0))}",
        flush=True,
    )


# --- Core spectrum generation ---

def _collect_enk_context(
    tagged_models,
    num_nodes: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Collect fresh atom-level ENK state from already-forwarded models."""
    branches: Dict[str, Dict[str, torch.Tensor]] = {}
    mean_ks: List[torch.Tensor] = []
    min_ks: List[torch.Tensor] = []
    low_k_scores: List[torch.Tensor] = []
    l_disagreements: List[torch.Tensor] = []

    for tag, model in tagged_models:
        bridge = getattr(model, "so3_enk_bridge", None)
        kg = getattr(bridge, "_last_kalman_gains", None) if bridge is not None else None
        if not isinstance(kg, dict) or not kg:
            continue
        vals = []
        for value in kg.values():
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                continue
            v = value.detach().to(device=device, dtype=torch.float32).view(-1)
            if v.numel() == int(num_nodes):
                vals.append(v.clamp(0.0, 1.0))
        if not vals:
            continue

        k_stack = torch.stack(vals, dim=0)
        mean_k = k_stack.mean(dim=0)
        min_k = k_stack.min(dim=0).values
        std_k = k_stack.std(dim=0, unbiased=False) if k_stack.size(0) > 1 else torch.zeros_like(mean_k)
        low_k = (1.0 - min_k).clamp(0.0, 1.0)
        branches[str(tag)] = {
            "mean_k": mean_k.detach(),
            "min_k": min_k.detach(),
            "std_k": std_k.detach(),
            "low_k_score": low_k.detach(),
        }
        mean_ks.append(mean_k)
        min_ks.append(min_k)
        low_k_scores.append(low_k)
        l_disagreements.append(std_k)

    if not branches:
        return {}

    mean_stack = torch.stack(mean_ks, dim=0)
    min_stack = torch.stack(min_ks, dim=0)
    low_stack = torch.stack(low_k_scores, dim=0)
    atom_ood = low_stack.max(dim=0).values
    l_disagreement = torch.stack(l_disagreements, dim=0).max(dim=0).values
    if mean_stack.size(0) > 1:
        branch_disagreement = mean_stack.std(dim=0, unbiased=False)
    else:
        branch_disagreement = torch.zeros_like(atom_ood)

    return {
        "atom_ood": atom_ood.detach(),
        "atom_k_mean": mean_stack.mean(dim=0).detach(),
        "atom_k_min": min_stack.min(dim=0).values.detach(),
        "l_disagreement": l_disagreement.detach(),
        "branch_disagreement": branch_disagreement.detach(),
        "branches": branches,
        "num_branches": len(branches),
    }


def generate_spectrum(
    pos: torch.Tensor,
    z: torch.Tensor,
    model_hii: torch.nn.Module,
    model_hij: torch.nn.Module,
    model_dd: torch.nn.Module,
    model_dp: torch.nn.Module,
    device: torch.device,
    linear: bool = False,
    scale: float = 0.965,
    sigma: float = 12.0,
    sigma_ir: Optional[float] = None,
    sigma_raman: Optional[float] = None,
    freq_scale_factor: float = 1.0,
    freq_range: Tuple[float, float] = (500, 4000),
    freq_points: int = 3501,
    radius: float = 5.0,
    lp_ckpt: Optional[str] = None,
    skeleton_radius: float = 5.0,
    nbo_calibrator: Optional[NBOGuidedCalibrator] = None,
    gsc_branches: Optional[set] = None,
    return_internal: bool = False,
) -> dict:
    """
    Self-trained four-weight spectrum generation (end-to-end skeleton + real-time NBO generation).

    Pipeline:
        1. Input (z, pos) -> online skeleton generation (bond inference + LP prediction -> global-node graph)
        2. Skeleton data fed to EP (Electron Prior) -> NBO feature prediction -> SO3 injection
        3. Hii + Hij -> Hessian -> mass weighting -> eigendecomposition -> freq + modes
        4. dd (dedipole) + modes -> chain_rule_ir -> IR intensity
        5. dp (depolar) + modes -> chain_rule_raman -> Raman tensor -> Raman activity
    """
    pos_dev = pos.to(device)
    z_dev = z.to(device)
    edge_index = radius_graph(x=pos_dev, r=radius, batch=None, max_num_neighbors=1000)

# End-to-end skeleton generation
    # Generate the skeleton when any model enables EP; otherwise data=None (no extra overhead)
    skel_data = None
    _any_ep = any(_is_ep_runtime_active(m) for m in [model_hii, model_hij, model_dd, model_dp])
    if _any_ep:
        print("  -> Online skeleton generation (end-to-end)...")
        skel_data = build_skeleton_online(
            z=z_dev, pos=pos_dev, device=device,
            lp_ckpt=lp_ckpt, skeleton_radius=skeleton_radius,
        )
    else:
        print("  -> EP inactive: skipping skeleton/LP pipeline.")

    print("  -> Inferring Hii ...")
    Hi = _infer_hii(model_hii, pos, z, device, data=skel_data)

# EP adaptive gating: ENK Kalman gain + NBO consensus
    _ep_gate_scale = 1.0
    _saved_gates = {}
    _enk_ood_score = None  # per-atom OOD score from ENK Kalman gain
    if _any_ep:
        # Step 1: only Hii has definitely been forward-passed here.  Avoid
        # reading stale Hij/dd/dp gains from a previous molecule; GSC collects
        # all branches again after their current forward passes.
        _enk_context_pre = _collect_enk_context(
            [("hii", model_hii)], int(z_dev.shape[0]), device,
        )
        _enk_ood_score = _enk_context_pre.get("atom_ood") if _enk_context_pre else None
        if _enk_ood_score is not None:
            _mean_ood = _enk_ood_score.mean().item()
        else:
            _mean_ood = 0.0

        # Step 2: compute NBO consensus as secondary OOD signal
        _nbo_consensus = 1.0
        nbo_first = None
        for m in (model_hii, model_hij, model_dd, model_dp):
            nbo_first = _extract_nbo_from_model(m)
            if nbo_first is not None:
                break
        if nbo_first is not None:
            bond_pred = nbo_first.get("bond_pred")
            if bond_pred is not None and bond_pred.numel() > 0:
                n_cols = int(bond_pred.size(1))
                ind_0 = bond_pred[:, 0:1].clamp(min=0.0)
                col_max_0 = ind_0.max().clamp(min=1e-6)
                indicators = [ind_0 / col_max_0]
                if n_cols >= 16:
                    for c in range(max(15, n_cols - 3), n_cols):
                        col = bond_pred[:, c:c + 1].clamp(min=0.0)
                        cmax = col.max().clamp(min=1e-6)
                        indicators.append(col / cmax)
                if len(indicators) >= 2:
                    stacked = torch.cat(indicators, dim=1)
                    col_mean = stacked.mean(dim=1).clamp(min=0.01)
                    col_std = stacked.std(dim=1)
                    cv = col_std / col_mean
                    _nbo_consensus = torch.exp(-cv * 3.0).mean().item()

        # Step 3: combine ENK OOD score and NBO consensus -> gate scale
        if _enk_ood_score is not None:
            # ENK-driven: OOD score scales EP injection with tanh soft-saturation
            # tanh(1.5 * ood) ∈ (0, 0.905) → gate_scale ∈ (1.0, 1.36)
            # This prevents extreme gate values that the injector never saw in training
            _ep_gate_scale = 1.0 + 0.4 * torch.tanh(torch.tensor(1.5 * _mean_ood)).item()
        else:
            # Fallback: NBO consensus only (same tanh soft-saturation)
            _ep_gate_scale = 1.0 + 0.4 * torch.tanh(torch.tensor(1.5 * (1.0 - _nbo_consensus))).item()

        _enk_ood_display = f"{_mean_ood:.3f}" if _enk_ood_score is not None else "N/A"
        print(f"  [EP-adapt] ENK_ood={_enk_ood_display}"
              f" NBO_consensus={_nbo_consensus:.3f}"
              f" -> gate_scale={_ep_gate_scale:.3f}"
              f" ({'OOD-boosted' if _ep_gate_scale > 1.1 else 'ID-default'})")

        # Apply gate scaling to SO3 injector on hij/dd/dp models
        for tag, m in [("hij", model_hij), ("dd", model_dd), ("dp", model_dp)]:
            injector = getattr(m, "so3_ep_injector", None)
            if injector is not None and hasattr(injector, "prior_blocks"):
                _saved_gates[tag] = {}
                for blk_k, blk_mod in injector.prior_blocks.items():
                    orig_gate = blk_mod.gate.data.clone()
                    _saved_gates[tag][blk_k] = orig_gate
                    blk_mod.gate.data = orig_gate * _ep_gate_scale

    try:
        print("  -> Inferring Hij ...")
        Hij = _infer_hij(model_hij, pos, z, edge_index, device, data=skel_data)

        print("  -> Inferring dedipole ...")
        dd = _infer_dedipole(model_dd, pos, z, device, data=skel_data)
        _print_ep_runtime_debug("dedipole", model_dd)

        print("  -> Inferring depolar ...")
        dp = _infer_depolar(model_dp, pos, z, device, data=skel_data)
        _print_ep_runtime_debug("depolar", model_dp)
    finally:
        # Restore original gates
        for tag, gates_dict in _saved_gates.items():
            m = {"hij": model_hij, "dd": model_dd, "dp": model_dp}[tag]
            injector = getattr(m, "so3_ep_injector", None)
            if injector is not None:
                for blk_k, orig_gate in gates_dict.items():
                    injector.prior_blocks[blk_k].gate.data.copy_(orig_gate)
    print(f"  [DIAG] dp shape={tuple(dp.shape)}  mean={dp.abs().mean().item():.4f}"
          f"  std={dp.std().item():.4f}  max={dp.abs().max().item():.4f}")
    masses = atom_masses[z_dev.cpu()].to(device)

    _gsc_enk_context = {}
    _gsc_enk_ood_score = _enk_ood_score
    _gsc_requested = set(gsc_branches) if gsc_branches else {"hij", "dd", "dp"}
    if _any_ep:
        _gsc_context_models = []
        if "hii" in _gsc_requested:
            _gsc_context_models.append(("hii", model_hii))
        if "hij" in _gsc_requested:
            _gsc_context_models.append(("hij", model_hij))
        if "dd" in _gsc_requested:
            _gsc_context_models.append(("dd", model_dd))
        if "dp" in _gsc_requested:
            _gsc_context_models.append(("dp", model_dp))
        _gsc_enk_context = _collect_enk_context(
            _gsc_context_models,
            int(z_dev.shape[0]),
            device,
        )
        if _gsc_enk_context:
            _gsc_enk_ood_score = _gsc_enk_context.get("atom_ood", _enk_ood_score)
            _branch_dis = _gsc_enk_context.get("branch_disagreement")
            _branch_dis_mean = float(_branch_dis.mean().item()) if isinstance(_branch_dis, torch.Tensor) else 0.0
            print(
                f"  [ENK-context] branches={int(_gsc_enk_context.get('num_branches', 0))}"
                f" atomOOD={float(_gsc_enk_ood_score.mean().item()):.3f}"
                f" branchDis={_branch_dis_mean:.3f}"
            )


# NBO-GSC per-branch calibration
    if nbo_calibrator is not None and _any_ep:
        nbo = None
        for m in (model_hii, model_hij, model_dd, model_dp):
            nbo = _extract_nbo_from_model(m)
            if nbo is not None:
                break
        if nbo is not None:
            _gsc = _gsc_requested
            num_nodes = int(z_dev.shape[0])
            _feature_ep = bool(getattr(nbo_calibrator, "feature_ep", False))
            if _feature_ep:
                from ep_feature_runtime import apply_feature
                pre_freq, pre_modes = hessfreq(Hi=Hi, Hij=Hij, masses=masses,
                    edge_index=edge_index, normal=False, linear=linear, scale=scale)
                Hi, Hij = apply_feature(nbo_calibrator, Hi, Hij, dd, dp, nbo,
                    edge_index, z_dev, pos_dev, masses, _gsc_enk_context,
                    pre_freq, pre_modes, skel_data, scale)
            elif "hij" in _gsc:
                pre_freq, pre_modes = None, None
                if bool(getattr(nbo_calibrator, "hij_mode_aware", False)):
                    pre_freq, pre_modes = hessfreq(
                        Hi=Hi, Hij=Hij, masses=masses,
                        edge_index=edge_index, normal=False,
                        linear=linear, scale=scale,
                    )
                    if freq_scale_factor != 1.0:
                        pre_freq = pre_freq * freq_scale_factor
                Hij = nbo_calibrator.calibrate_hij(
                    Hij, nbo, edge_index, num_nodes, pos=pos_dev,
                    enk_ood_score=_gsc_enk_ood_score,
                    enk_context=_gsc_enk_context,
                    z=z_dev,
                    mode_freq=pre_freq,
                    modes=pre_modes,
                    skeleton_data=skel_data,
                )
                Hi = nbo_calibrator.compensate_hii_from_hij_delta(
                    Hi, getattr(nbo_calibrator, "last_hij_delta", None), edge_index,
                )
            if "dd" in _gsc and not _feature_ep:
                dd = nbo_calibrator.calibrate_dedipole(
                    dd, nbo, pos=pos_dev,
                    enk_ood_score=_gsc_enk_ood_score,
                    enk_context=_gsc_enk_context,
                    z=z_dev,
                )
            if "dp" in _gsc and not _feature_ep:
                dp = nbo_calibrator.calibrate_depolar(
                    dp, nbo, edge_index, num_nodes,
                    enk_ood_score=_gsc_enk_ood_score,
                    enk_context=_gsc_enk_context,
                    z=z_dev,
                    pos=pos_dev,
                )
            _branches_str = ",".join(sorted(_gsc))
            _hij_dbg = getattr(nbo_calibrator, "last_hij_debug", {}) or {}
            _dd_dbg = getattr(nbo_calibrator, "last_dd_debug", {}) or {}
            _dp_dbg = getattr(nbo_calibrator, "last_dp_debug", {}) or {}
            _hij_dbg_str = ""
            if _hij_dbg:
                _hij_dbg_str = (
                    f" HijGSC active={int(_hij_dbg.get('active_edges', 0))}/"
                    f"{int(_hij_dbg.get('matched_edges', 0))}"
                    f" xh={int(_hij_dbg.get('xh_edges', 0))}"
                    f" amide={int(_hij_dbg.get('amide_edges', 0))}"
                    f" unkZ={int(_hij_dbg.get('unknown_atom_edges', 0))}"
                    f" unkClass={int(_hij_dbg.get('unknown_class_edges', 0))}"
                    f" unkEnv={int(_hij_dbg.get('unknown_env_signature_edges', 0))}"
                    f" hbond={int(_hij_dbg.get('hbond_edges', 0))}"
                    f" envOOD={float(_hij_dbg.get('mean_env_ood', 0.0)):.3f}"
                    f" envNeed={float(_hij_dbg.get('mean_env_need', 0.0)):.3f}"
                    f" enkNeed={float(_hij_dbg.get('mean_enk_need', 0.0)):.3f}"
                    f" physNeed={float(_hij_dbg.get('mean_physics_need', 0.0)):.3f}"
                    f" nboTrust={float(_hij_dbg.get('mean_nbo_reliability', 1.0)):.3f}"
                    f" gate_mean={float(_hij_dbg.get('mean_gate', 0.0)):.3f}"
                    f" trigger_mean={float(_hij_dbg.get('mean_trigger', 0.0)):.3f}"
                    f" edgeScale={float(_hij_dbg.get('mean_edge_global_scale', 0.0)):.3f}"
                    f" deltaScale={float(_hij_dbg.get('mean_delta_scale', 1.0)):.3f}"
                    f" physSoft={int(_hij_dbg.get('physical_soften_edges', 0))}"
                    f" physHard={int(_hij_dbg.get('physical_harden_edges', 0))}"
                    f" stark={float(_hij_dbg.get('mean_stark_score', 0.0)):.3f}"
                )
            _dd_dbg_str = ""
            if _dd_dbg:
                _dd_dbg_str = (
                    f" DdGSC atomNeed={float(_dd_dbg.get('mean_atom_need', 0.0)):.3f}"
                    f" traceNeed={float(_dd_dbg.get('mean_trace_need', 0.0)):.3f}"
                    f" traceGate={float(_dd_dbg.get('mean_trace_gate', 0.0)):.3f}"
                    f" bondGate={float(_dd_dbg.get('mean_bond_gate', 0.0)):.3f}"
                    f" activeBond={int(_dd_dbg.get('active_bonds', 0))}"
                )
            _dp_dbg_str = ""
            if _dp_dbg:
                _dp_dbg_str = (
                    f" DpGSC atomNeed={float(_dp_dbg.get('mean_atom_need', 0.0)):.3f}"
                    f" physNeed={float(_dp_dbg.get('mean_physics_need', 0.0)):.3f}"
                    f" gate={float(_dp_dbg.get('mean_gate', 0.0)):.3f}"
                    f" activeAtom={int(_dp_dbg.get('active_atoms', 0))}"
                )
            print(f"  -> NBO-GSC calibration [{_branches_str}]:"
                  f" Hij scale={Hij.norm(dim=(-2,-1)).mean().item():.4f}"
                  f"  dd trace={dd.diagonal(dim1=-2,dim2=-1).sum(-1).abs().mean().item():.4f}"
                  f"  dp norm={dp.norm(dim=-1).mean().item():.4f}"
                  f"{_hij_dbg_str}{_dd_dbg_str}{_dp_dbg_str}")
        else:
            if getattr(nbo_calibrator, "feature_ep", False):
                raise RuntimeError("Feature EP requires model NBO evidence")
            print("  -> NBO-GSC skipped (no EP model provides NBO features)")
    elif nbo_calibrator is not None and not _any_ep:
        print("  -> NBO-GSC skipped (EP not active in any model)")

# Vibrational analysis
    masses = atom_masses[z_dev.cpu()].to(device)
    print("  -> Hessian diagonalization -> normal-mode frequencies + normal coordinates ...")
    freq, modes = hessfreq(
        Hi=Hi, Hij=Hij, masses=masses,
        edge_index=edge_index, normal=False,
        linear=linear, scale=scale,
    )
    if freq_scale_factor != 1.0:
        freq = freq * freq_scale_factor

    ir_int = chain_rule_ir(dd=dd, modes=modes)
    raman_act = get_raman_act(chain_rule_raman(dp=dp, modes=modes))

# Lorenz broadening
    sigma_ir = sigma if sigma_ir is None else sigma_ir
    sigma_raman = sigma if sigma_raman is None else sigma_raman
    x_axis = torch.linspace(freq_range[0], freq_range[1], freq_points, device=device)
    yir = Lorenz_broadening(freq, ir_int, c=x_axis, sigma=sigma_ir)
    yram = Lorenz_broadening(freq, raman_act, c=x_axis, sigma=sigma_raman)

    out = {
        "freq": freq.detach().cpu(),
        "ir_int": ir_int.detach().cpu(),
        "raman_act": raman_act.detach().cpu(),
        "x_axis": x_axis.detach().cpu(),
        "yir": yir.detach().cpu(),
        "yram": yram.detach().cpu(),
    }
    if nbo_calibrator is not None and getattr(nbo_calibrator, "feature_ep", False):
        out["feature_ep"] = getattr(nbo_calibrator, "feature_ep_diagnostics", {})
    if return_internal:
        out.update({
            "modes": modes.detach().cpu(),
            "edge_index": edge_index.detach().cpu(),
            "Hi": Hi.detach().cpu(),
            "Hij": Hij.detach().cpu(),
            "dedipole": dd.detach().cpu(),
            "depolar": dp.detach().cpu(),
        })
        if nbo_calibrator is not None:
            out["gsc_debug"] = {
                "hij": dict(getattr(nbo_calibrator, "last_hij_debug", {}) or {}),
                "dedipole": dict(getattr(nbo_calibrator, "last_dd_debug", {}) or {}),
                "depolar": dict(getattr(nbo_calibrator, "last_dp_debug", {}) or {}),
            }
            hij_delta = getattr(nbo_calibrator, "last_hij_delta", None)
            if isinstance(hij_delta, torch.Tensor):
                out["hij_delta"] = hij_delta.detach().cpu()
    return out


def _spectrum_archive_value(value):
    """Keep tensor NPZ fields unchanged; encode feature diagnostics as JSON text."""
    if isinstance(value, dict):
        return np.asarray(json.dumps(value, ensure_ascii=False))
    return value.numpy()


# --- Reference spectrum parsing (Gaussian / experimental output formats) ---

def _to_numpy_1d(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(arr, dtype=np.float64)


def _lorentz_broadening_np(x0: np.ndarray, y0: np.ndarray,
                           c: np.ndarray, sigma: float) -> np.ndarray:
    if x0.size == 0 or y0.size == 0:
        return np.zeros_like(c, dtype=np.float64)
    dx = x0[:, None] - c[None, :]
    ly = (sigma / (2.0 * np.pi)) / (dx ** 2 + 0.25 * (sigma ** 2))
    return np.sum(y0[:, None] * ly, axis=0)


def _parse_gaussian_txt_sections(path: str) -> dict:
    sections = {
        "peak_information": {"x": [], "y": []},
        "spectra": {"x": [], "y": []},
    }
    current = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("# Peak information"):
                current = "peak_information"
                continue
            if s.startswith("# Spectra"):
                current = "spectra"
                continue
            if current is None or not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                sections[current]["x"].append(float(parts[0]))
                sections[current]["y"].append(float(parts[1]))
            except ValueError:
                continue
    out = {}
    for key, payload in sections.items():
        x_arr = np.array(payload["x"], dtype=np.float64)
        y_arr = np.array(payload["y"], dtype=np.float64)
        if x_arr.size:
            order = np.argsort(x_arr)
            x_arr = x_arr[order]
            y_arr = y_arr[order]
        out[key] = {"x": x_arr, "y": y_arr}
    return out


def _parse_gaussian_log_spectrum(path: str, kind: str) -> dict:
    freq_vals: List[float] = []
    inten_vals: List[float] = []
    current_freqs: Optional[List[float]] = None
    target = "IR Inten" if kind == "ir" else "Raman Activ"

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Frequencies --" in line:
                try:
                    current_freqs = [float(x) for x in line.split("--", 1)[1].split()]
                except Exception:
                    current_freqs = None
                continue
            if current_freqs is None or target not in line or "--" not in line:
                continue
            try:
                vals = [float(x) for x in line.split("--", 1)[1].split()]
            except Exception:
                current_freqs = None
                continue
            n = min(len(current_freqs), len(vals))
            freq_vals.extend(current_freqs[:n])
            inten_vals.extend(vals[:n])
            current_freqs = None

    if not freq_vals:
        raise ValueError(f"No Gaussian {kind} peak blocks found in: {path}")

    x_arr = np.array(freq_vals, dtype=np.float64)
    y_arr = np.array(inten_vals, dtype=np.float64)
    order = np.argsort(x_arr)
    return {
        "x": x_arr[order],
        "y": y_arr[order],
        "mode": "sticks",
        "source": f"gaussian_{kind}_log",
    }


def _parse_ref_spectrum(path: str, kind: str = "ir",
                        txt_mode: str = "auto") -> dict:
    """Parse Gaussian reference data.

    Preferred order:
      1. Gaussian .log/.out raw stick data (Frequencies/IR Inten/Raman Activ)
      2. Gaussian-exported .txt Peak information
      3. Gaussian-exported .txt Spectra curve

    Using stick data is the only apples-to-apples path with DetaNet/EnviroDetaNet,
    because both methods compare truth and prediction after the same Lorentz broadening.
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".log", ".out"}:
        return _parse_gaussian_log_spectrum(path, kind=kind)

    sections = _parse_gaussian_txt_sections(path)
    peak_info = sections["peak_information"]
    spectra = sections["spectra"]

    if txt_mode == "auto":
        if peak_info["x"].size:
            return {
                "x": peak_info["x"],
                "y": peak_info["y"],
                "mode": "sticks",
                "source": "gaussian_txt_peak_information",
            }
        if spectra["x"].size:
            return {
                "x": spectra["x"],
                "y": spectra["y"],
                "mode": "curve",
                "source": "gaussian_txt_spectra",
            }
    elif txt_mode == "peak_info":
        if peak_info["x"].size:
            return {
                "x": peak_info["x"],
                "y": peak_info["y"],
                "mode": "sticks",
                "source": "gaussian_txt_peak_information",
            }
        raise ValueError(f"No '# Peak information' section found in: {path}")
    elif txt_mode == "spectra":
        if spectra["x"].size:
            return {
                "x": spectra["x"],
                "y": spectra["y"],
                "mode": "curve",
                "source": "gaussian_txt_spectra",
            }
        raise ValueError(f"No '# Spectra' section found in: {path}")

    raise ValueError(f"No usable reference spectrum found in: {path}")


# --- Plotting utilities (journal style) ---

def _journal_rcparams() -> dict:
    """matplotlib rcParams for journal figures: Arial Black, bold borders."""
    return {
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial Black", "Arial", "Helvetica", "DejaVu Sans"],
        "font.weight":       "black",
        "axes.labelweight":  "black",
        "axes.titleweight":  "black",
        "axes.linewidth":    2.2,
        "xtick.major.width": 1.8,
        "xtick.minor.width": 1.2,
        "ytick.major.width": 1.8,
        "ytick.minor.width": 1.2,
        "xtick.major.size":  5.0,
        "xtick.minor.size":  3.0,
        "ytick.major.size":  5.0,
        "ytick.minor.size":  3.0,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         False,
        "ytick.right":       False,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "axes.labelsize":    13,
        "legend.fontsize":   11,
    }


def _apply_journal_style(ax, spine_lw: float = 2.5, grid: bool = True) -> None:
    """Bold 4-sided border + inward ticks on bottom/left only + dashed grid."""
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)
    ax.tick_params(which="both", direction="in", top=False, right=False,
                   width=1.8, length=5)
    ax.tick_params(which="minor", length=3)
    if grid:
        ax.grid(True, which="major", linestyle="--", linewidth=0.65,
                color="#bbbbbb", alpha=0.75, zorder=0)
        ax.set_axisbelow(True)


def _norm_spectrum(y: np.ndarray, x: np.ndarray,
                   xmin: float, xmax: float) -> np.ndarray:
    """Normalise y to [0, 1] within the frequency range [xmin, xmax]."""
    lo, hi = min(xmin, xmax), max(xmin, xmax)
    mask = (x >= lo) & (x <= hi)
    ymax = float(y[mask].max()) if mask.any() and y[mask].max() > 0 else 1.0
    return y / ymax


def _build_broadened_curve_from_results(results: dict, intensity_key: str,
                                        sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    x_axis = _to_numpy_1d(results["x_axis"])
    freq = _to_numpy_1d(results["freq"])
    intensity = _to_numpy_1d(results[intensity_key])
    valid = np.isfinite(freq) & np.isfinite(intensity) & (freq > 0) & (intensity > 0)
    if not valid.any():
        return x_axis, np.zeros_like(x_axis)
    return x_axis, _lorentz_broadening_np(freq[valid], intensity[valid], x_axis, sigma)


def _materialize_ref_curve(ref: Optional[dict], x_axis: np.ndarray,
                           sigma: float,
                           freq_scale: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert a parsed reference-spectrum dict into a broadened (x, y) tuple.

    Parameters
    ----------
    freq_scale : float
        Scale factor applied to raw stick frequencies BEFORE Lorentz broadening.
        Set to the same value as ``hessfreq``'s ``scale`` argument (default 0.965)
        so that the reference and prediction are in the same frequency space.
        Only applies when ``mode == 'sticks'``; already-broadened curves are
        plotted as-is.
    """
    if ref is None:
        return None
    if isinstance(ref, tuple):
        rx, ry = ref
        return _to_numpy_1d(rx), _to_numpy_1d(ry)

    rx = _to_numpy_1d(ref.get("x", []))
    ry = _to_numpy_1d(ref.get("y", []))
    mode = ref.get("mode", "curve")
    if rx.size == 0 or ry.size == 0:
        return None
    if mode == "sticks":
        valid = np.isfinite(rx) & np.isfinite(ry) & (rx > 0) & (ry > 0)
        if not valid.any():
            return None
        rx_scaled = rx[valid] * freq_scale
        return x_axis, _lorentz_broadening_np(rx_scaled, ry[valid], x_axis, sigma)
    order = np.argsort(rx)
    return rx[order], ry[order]


# Colour palette (consistent across all figure modes)
_SENK_COLOR   = "#D62728"   # crimson red   – SENK(ENK on)
_SENK_OFF_COLOR = "#FF7F0E"  # orange        – SENK(ENK off)
_DET_COLOR    = "#1F77B4"   # steel blue    – DetaNet baseline
_REF_COLOR    = "#C0C0C0"   # light silver  – DFT/Gaussian reference (filled)
_SENK_EP_COLOR = "#00695C"   # dark teal     – SENK-EP (Electron Prior)


# --- Plotting ---

def _plot_spectrum(
    results: dict,
    title_suffix: str = "",
    senk_off_results: Optional[dict] = None,
    ep_results: Optional[dict] = None,
    detanet_results: Optional[dict] = None,
    out_png: Optional[str] = None,
    ref_ir: Optional[dict] = None,
    ref_raman: Optional[dict] = None,
    model_label: str = "SENK(ENK on)",
    senk_off_label: str = "SENK(ENK off)",
    ep_label: str = "SENK-EP",
    freq_range: Tuple[float, float] = (500, 4000),
    sigma_ir: float = 12.0,
    sigma_raman: float = 12.0,
    ref_freq_scale: float = 0.965,
) -> None:
    """Journal-quality 2-panel IR (top) + Raman (bottom) comparison figure.

    Layer order (bottom → top):
    1. DFT/Gaussian reference – light gray filled area  (ref_ir / ref_raman)
    2. DetaNet baseline        – blue dashed line
    3. SENK(ENK off)           – orange solid line
    4. SENK-EP                  – dark teal solid line
    5. SENK(ENK on)            – red solid line

    Both PNG (300 dpi) and SVG vector files are saved automatically.

    Parameters
    ----------
    ref_ir / ref_raman : parsed reference dict from _parse_ref_spectrum().
    model_label : legend label for the SENK curve (default 'SENK').
    freq_range  : (fmin, fmax) in cm⁻¹ used for normalisation and axis limits.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    fmin, fmax = freq_range

    def _layer_ref(ax, ref, lbl="DFT Reference"):
        if ref is None:
            return
        rx, ry = ref
        mask = (rx >= min(fmin, fmax)) & (rx <= max(fmin, fmax))
        if not mask.any():
            return
        rx2 = rx[mask]
        ry2 = _norm_spectrum(ry[mask], rx[mask], fmin, fmax)
        ax.fill_between(rx2, 0, ry2, color=_REF_COLOR, alpha=0.90,
                        linewidth=0, label=lbl, zorder=1)

    def _layer_line(ax, x, y, color, ls, lw, lbl, zo):
        yn = _norm_spectrum(y, x, fmin, fmax)
        ax.plot(x, yn, color=color, linewidth=lw, linestyle=ls, label=lbl, zorder=zo)

    def _finish_ax(ax, ylabel, show_xlabel=False):
        ax.set_xlim(min(fmin, fmax), max(fmin, fmax))
        ax.set_ylim(0.0, 1.2)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_ylabel(ylabel, labelpad=6)
        if show_xlabel:
            ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", labelpad=6)
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
        # Show all x-axis tick labels with finer granularity
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(500))  # Major ticks every 500 cm⁻¹
        ax.xaxis.set_minor_locator(MultipleLocator(100))  # Minor ticks every 100 cm⁻¹
        leg = ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            bbox_transform=ax.transAxes,
            framealpha=0.85,
            edgecolor="#444444",
            frameon=True,
            fontsize=6,
            title_fontsize=6,
            borderaxespad=0.0,
        )
        for txt in leg.get_texts():
            txt.set_fontweight("black")
        _apply_journal_style(ax)

    with plt.rc_context(_journal_rcparams()):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                                 gridspec_kw={"hspace": 0.06})

        x_v2, y_v2_ir = _build_broadened_curve_from_results(results, "ir_int", sigma_ir)
        _, y_v2_ram = _build_broadened_curve_from_results(results, "raman_act", sigma_raman)
        ref_ir_curve = _materialize_ref_curve(ref_ir, x_v2, sigma_ir, freq_scale=ref_freq_scale)
        ref_raman_curve = _materialize_ref_curve(ref_raman, x_v2, sigma_raman, freq_scale=ref_freq_scale)

# IR panel
        ax = axes[0]
        _layer_ref(ax, ref_ir_curve)
        if detanet_results is not None and "yir" in detanet_results:
            x_det, y_det_ir = _build_broadened_curve_from_results(detanet_results, "ir_int", sigma_ir)
            _layer_line(ax, x_det, y_det_ir,
                        _DET_COLOR, "--", 1.6, "DetaNet", 2)
        if senk_off_results is not None and "yir" in senk_off_results:
            x_off, y_off_ir = _build_broadened_curve_from_results(senk_off_results, "ir_int", sigma_ir)
            _layer_line(ax, x_off, y_off_ir,
                        _SENK_OFF_COLOR, "-", 1.9, senk_off_label, 3)
        if ep_results is not None and "yir" in ep_results:
            x_ep, y_ep_ir = _build_broadened_curve_from_results(ep_results, "ir_int", sigma_ir)
            _layer_line(ax, x_ep, y_ep_ir,
                        _SENK_EP_COLOR, "-", 1.9, ep_label, 4)
        _layer_line(ax, x_v2, y_v2_ir,
                    _SENK_COLOR, "-", 1.9, model_label, 5)
        _finish_ax(ax, "IR Intensity (normalized)", show_xlabel=False)

# Raman panel
        ax = axes[1]
        _layer_ref(ax, ref_raman_curve)
        if detanet_results is not None and "yram" in detanet_results:
            x_det, y_det_ram = _build_broadened_curve_from_results(detanet_results, "raman_act", sigma_raman)
            _layer_line(ax, x_det, y_det_ram,
                        _DET_COLOR, "--", 1.6, "DetaNet", 2)
        if senk_off_results is not None and "yram" in senk_off_results:
            x_off, y_off_ram = _build_broadened_curve_from_results(senk_off_results, "raman_act", sigma_raman)
            _layer_line(ax, x_off, y_off_ram,
                        _SENK_OFF_COLOR, "-", 1.9, senk_off_label, 3)
        if ep_results is not None and "yram" in ep_results:
            x_ep, y_ep_ram = _build_broadened_curve_from_results(ep_results, "raman_act", sigma_raman)
            _layer_line(ax, x_ep, y_ep_ram,
                        _SENK_EP_COLOR, "-", 1.9, ep_label, 4)
        _layer_line(ax, x_v2, y_v2_ram,
                    _SENK_COLOR, "-", 1.9, model_label, 5)
        _finish_ax(ax, r"Raman Activity (normalized)", show_xlabel=True)

# save PNG + SVG
        out = out_png or "v2_spectrum.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"PNG saved: {out}")
        svg_out = str(Path(out).with_suffix(".svg"))
        plt.savefig(svg_out, format="svg", bbox_inches="tight")
        print(f"SVG saved: {svg_out}")
        plt.close()


def _plot_ir_only(
    results: dict,
    title: str = "",
    smiles: Optional[str] = None,
    out_png: Optional[str] = None,
    oor: bool = False,
    label: str = "SENK",
    ref_ir: Optional[dict] = None,
    freq_range: Tuple[float, float] = (500, 4000),
    sigma_ir: float = 12.0,
    ref_freq_scale: float = 0.965,
) -> None:
    """Journal-quality single-panel IR spectrum figure.

    Layers: DFT reference (gray fill) → red solid SENK line.
    Saves both PNG (300 dpi) and SVG.
    
    Parameters
    ----------
    sigma_ir : float
        IR Lorentzian half-width used for plotting.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    fmin, fmax = freq_range

    with plt.rc_context(_journal_rcparams()):
        fig, ax = plt.subplots(figsize=(10, 4))

        x_v2, y_v2_ir = _build_broadened_curve_from_results(results, "ir_int", sigma_ir)
        ref_ir_curve = _materialize_ref_curve(ref_ir, x_v2, sigma_ir, freq_scale=ref_freq_scale)

        if ref_ir_curve is not None:
            rx, ry = ref_ir_curve
            mask = (rx >= min(fmin, fmax)) & (rx <= max(fmin, fmax))
            if mask.any():
                ax.fill_between(rx[mask], 0,
                                _norm_spectrum(ry[mask], rx[mask], fmin, fmax),
                                color=_REF_COLOR, alpha=0.90,
                                linewidth=0, label="DFT Reference", zorder=1)

        y_ir_n = _norm_spectrum(y_v2_ir, x_v2, fmin, fmax)
        ax.plot(x_v2, y_ir_n, color=_SENK_COLOR, linewidth=1.9,
                label=label, zorder=3)

        ax.set_xlim(min(fmin, fmax), max(fmin, fmax))
        ax.set_ylim(0.0, 1.2)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", labelpad=6)
        ax.set_ylabel("IR Intensity (normalized)", labelpad=6)
        # Show all x-axis tick labels with finer granularity
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(500))  # Major ticks every 500 cm⁻¹
        ax.xaxis.set_minor_locator(MultipleLocator(100))  # Minor ticks every 100 cm⁻¹

        if oor:
            ax.text(0.02, 0.95,
                    "OOR: molecule contains out-of-range atomic numbers",
                    transform=ax.transAxes, fontsize=9,
                    color="darkred", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff4f4",
                              edgecolor="darkred", alpha=0.9))

        if smiles is not None:
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw
                mol2d = Chem.MolFromSmiles(smiles)
                if mol2d is not None:
                    img = Draw.MolToImage(mol2d, size=(160, 130))
                    axins = ax.inset_axes([0.01, 0.42, 0.20, 0.54])
                    axins.imshow(img)
                    axins.axis("off")
            except Exception:
                pass

        leg = ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            bbox_transform=ax.transAxes,
            framealpha=0.85,
            edgecolor="#444444",
            frameon=True,
            fontsize=6,
            title_fontsize=6,
            borderaxespad=0.0,
        )
        for txt in leg.get_texts():
            txt.set_fontweight("black")
        _apply_journal_style(ax)

        out = out_png or "ir_spectrum.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    PNG saved: {out}")
        svg_out = str(Path(out).with_suffix(".svg"))
        plt.savefig(svg_out, format="svg", bbox_inches="tight")
        print(f"    SVG saved: {svg_out}")
        plt.close()


# --- Full DetaNet pipeline (for comparison curves) ---

def generate_detanet_spectrum(
    pos: torch.Tensor,
    z: torch.Tensor,
    model_Hi,
    model_Hij,
    model_dd,
    model_dp,
    device: torch.device,
    linear: bool = False,
    scale: float = 0.965,
    sigma: float = 12.0,
    sigma_ir: Optional[float] = None,
    sigma_raman: Optional[float] = None,
    freq_scale_factor: float = 1.0,
    freq_range: Tuple[float, float] = (500, 4000),
    freq_points: int = 3501,
) -> dict:
    """Full IR + Raman pipeline with the four DetaNet models (for comparison against V2 results)."""
    pos_dev = pos.to(device)
    z_dev = z.to(device)
    edge_index = radius_graph(x=pos_dev, r=5.0, batch=None, max_num_neighbors=1000)

    # DetaNet forward: no batch argument
    pos_g = pos_dev.requires_grad_(True)
    with torch.enable_grad():
        Hi  = model_Hi( pos=pos_g, z=z_dev, batch=None).detach()
        Hij = model_Hij(pos=pos_g, z=z_dev, edge_index=edge_index, batch=None).detach()
        dd  = model_dd( pos=pos_g, z=z_dev, batch=None).detach()
        dp  = model_dp( pos=pos_g.detach().requires_grad_(True), z=z_dev, batch=None).detach()

    masses = atom_masses[z_dev.cpu()].to(device)
    freq, modes = hessfreq(Hi=Hi, Hij=Hij, masses=masses, edge_index=edge_index,
                           normal=False, linear=linear, scale=scale)
    real_mask = torch.isfinite(freq) & (freq > 0)
    freq  = freq[real_mask]
    modes = modes[real_mask]
    if freq_scale_factor != 1.0:
        freq = freq * freq_scale_factor

    ir_int    = chain_rule_ir(dd=dd, modes=modes)
    raman_act = get_raman_act(chain_rule_raman(dp=dp, modes=modes))

    sigma_ir = sigma if sigma_ir is None else sigma_ir
    sigma_raman = sigma if sigma_raman is None else sigma_raman
    x_axis = torch.linspace(freq_range[0], freq_range[1], freq_points, device=device)
    yir  = Lorenz_broadening(freq, ir_int,    c=x_axis, sigma=sigma_ir)
    yram = Lorenz_broadening(freq, raman_act, c=x_axis, sigma=sigma_raman)

    return {
        "freq":      freq.detach().cpu(),
        "ir_int":    ir_int.detach().cpu(),
        "raman_act": raman_act.detach().cpu(),
        "x_axis":    x_axis.detach().cpu(),
        "yir":       yir.detach().cpu(),
        "yram":      yram.detach().cpu(),
    }


# --- Batch SMI -> IR + Raman (all V2 models) ---

def _run_batch_full(args, device: torch.device) -> None:
    """
    Batch SMILES -> self-trained V2 four-weight IR + Raman spectrum generation.

    Pipeline:
      1. Read the .smi file -> RDKit ETKDGv3+MMFF generates 3D conformers
      2. Element-range check (max_atomic_number); out-of-range molecules are skipped
      3. Call generate_spectrum() (four V2 models) to infer spectra
      4. Optional: call generate_detanet_spectrum() (four DetaNet models) to generate comparison curves
      5. Save .npz (including a smiles key) and .png per molecule
    """
    smi_path = Path(args.batch_smi)
    if not smi_path.exists():
        print(f"[ERROR] .smi file does not exist: {smi_path}")
        sys.exit(1)

    raw_lines = smi_path.read_text(encoding="utf-8").splitlines()
    smiles_list = [
        ln.strip().split()[0] for ln in raw_lines
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if args.batch_max_mols > 0:
        smiles_list = smiles_list[: args.batch_max_mols]
    print(f"SMI file: {smi_path}, {len(smiles_list)} SMILES total")

    out_dir = Path(args.batch_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_ir = (args.sigma_ir if args.sigma_ir is not None else args.sigma) * args.sigma_scale_factor
    sigma_raman = (args.sigma_raman if args.sigma_raman is not None else args.sigma) * args.sigma_scale_factor
    ref_freq_scale = args.ref_freq_scale if args.ref_freq_scale is not None else args.scale

    trained_max_z = args.max_atomic_number

# Phase 1: 3D conformer generation
    print("Generating 3D conformers (RDKit ETKDGv3 + MMFF)...")
    valid_entries, skipped_oor, failed_embed = [], [], []
    for idx, smi in enumerate(smiles_list, 1):
        try:
            z_list, pos_list = smiles_to_coords(smi)
        except Exception as e:
            print(f"  [{idx:04d}] SKIP  3D embedding failed: {e}  ({smi[:50]})")
            failed_embed.append((idx, smi, str(e)))
            continue
        mz = max(z_list)
        if mz >= trained_max_z:
            print(f"  [{idx:04d}] [OOR]  natoms={len(z_list):3d}  maxZ={mz}  {smi[:55]}"
                  f"  ** SKIP: Z={mz} >= max_atomic_number={trained_max_z} **")
            skipped_oor.append((idx, smi, mz))
            continue
        print(f"  [{idx:04d}] [ OK]  natoms={len(z_list):3d}  maxZ={mz}  {smi[:55]}")
        valid_entries.append((idx, smi, z_list, pos_list))

    if not valid_entries:
        print("[ERROR] No processable molecules (all embedding failed or out of range); exiting.")
        return

# Phase 2: Load the four V2 models
    # Validate the four weights (ENK on)
    required = {"hii_ckpt": args.hii_ckpt, "hij_ckpt": args.hij_ckpt,
                "dd_ckpt": args.dd_ckpt, "dp_ckpt": args.dp_ckpt}
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"[ERROR] Batch mode requires the complete set of four weights (ENK on); missing: {missing}")
        sys.exit(1)

    # Validate the four weights (ENK off)
    required_off = {"hii_off_ckpt": args.hii_off_ckpt, "hij_off_ckpt": args.hij_off_ckpt,
                    "dd_off_ckpt": args.dd_off_ckpt, "dp_off_ckpt": args.dp_off_ckpt}
    missing_off = [k for k, v in required_off.items() if not v]
    if missing_off:
        print(f"[ERROR] Batch mode requires the complete set of four weights (ENK off); missing: {missing_off}")
        sys.exit(1)

    print(f"\nLoading four V2 models (ENK on) ...")
    model_hii = _load_multitask("hii",      args.hii_ckpt, args.hii_mode, device, args)
    model_hij = _load_multitask("hij",      args.hij_ckpt, args.hij_mode, device, args)
    model_dd  = _load_multitask("dedipole", args.dd_ckpt,  args.dd_mode,  device, args)
    model_dp  = _load_depolar(              args.dp_ckpt,  args.dp_mode,  device, args)

    print(f"Loading four V2 models (ENK off) ...")
    model_hii_off = _load_multitask("hii",      args.hii_off_ckpt, "equiformer_v2", device, args, force_disable_enk=True)
    model_hij_off = _load_multitask("hij",      args.hij_off_ckpt, "equiformer_v2", device, args, force_disable_enk=True)
    model_dd_off  = _load_multitask("dedipole", args.dd_off_ckpt,  "equiformer_v2", device, args, force_disable_enk=True)
    model_dp_off  = _load_depolar(              args.dp_off_ckpt,  "equiformer_v2", device, args, force_disable_enk=True)

# SENK-EP depolar (third comparison curve)
    model_dp_ep_batch = None
    if args.dp_ep_ckpt:
        print(f"\nLoading depolar model [SENK-EP] (mode={args.dp_ep_mode}): {args.dp_ep_ckpt}")
        model_dp_ep_batch = _load_depolar(args.dp_ep_ckpt, args.dp_ep_mode, device, args)
        print(f"  -> SENK-EP runtime active: {_is_ep_runtime_active(model_dp_ep_batch)}")

# SENK-EP dedipole
    model_dd_ep_batch = None
    if args.dd_ep_ckpt:
        print(f"\nLoading dedipole model [SENK-EP] (mode={args.dd_ep_mode}): {args.dd_ep_ckpt}")
        model_dd_ep_batch = _load_multitask("dedipole", args.dd_ep_ckpt, args.dd_ep_mode, device, args)
        print(f"  -> SENK-EP dd runtime active: {_is_ep_runtime_active(model_dd_ep_batch)}")

# SENK-EP hij
    model_hij_ep_batch = None
    if args.hij_ep_ckpt:
        print(f"\nLoading Hij model [SENK-EP] (mode={args.hij_ep_mode}): {args.hij_ep_ckpt}")
        model_hij_ep_batch = _load_multitask("hij", args.hij_ep_ckpt, args.hij_ep_mode, device, args)
        print(f"  -> SENK-EP hij runtime active: {_is_ep_runtime_active(model_hij_ep_batch)}")

# NBO-GSC calibrator (batch mode)
    nbo_calibrator = None
    _gsc_enabled = bool(getattr(args, "nbo_gsc_enabled", False))
    _gsc_branches = set(b.strip() for b in (getattr(args, "nbo_gsc_branches", "") or "hij,dd,dp").split(",") if b.strip()) if _gsc_enabled else set()
    _any_ep_batch = any(
        _is_ep_runtime_active(m) for m in [model_hii, model_hij, model_dd, model_dp]
    )
    _train_stats = None
    if _any_ep_batch and _gsc_branches and getattr(args, "nbo_train_stats", ""):
        _train_stats = torch.load(args.nbo_train_stats, weights_only=True, map_location="cpu")
        print(f"NBO-GSC training stats loaded: {args.nbo_train_stats}")
    if _any_ep_batch and _gsc_branches:
        nbo_calibrator = _make_inference_calibrator(args,
            alpha_hij=float(getattr(args, "nbo_gsc_alpha_hij", 0.25)),
            alpha_dd=float(getattr(args, "nbo_gsc_alpha_dd", 0.0)),
            alpha_dp=float(getattr(args, "nbo_gsc_alpha_dp", 0.20)),
            bond_order_exponent=float(getattr(args, "nbo_gsc_bond_exp", 0.6)),
            clamp_ratio=float(getattr(args, "nbo_gsc_clamp", 0.5)),
            bond_dipole_factor=float(getattr(args, "nbo_gsc_bond_dipole_factor", 0.20)),
            alpha_bond=float(getattr(args, "nbo_gsc_alpha_bond", 0.12)),
            dd_sum_rule_strength=float(getattr(args, "nbo_gsc_dd_sum_rule", 0.0)),
            hij_target_xh_only=bool(getattr(args, "nbo_gsc_hij_target_xh_only", False)),
            hij_non_target_scale=float(getattr(args, "nbo_gsc_hij_non_target_scale", 0.15)),
            hij_amide_scale=float(getattr(args, "nbo_gsc_hij_amide_scale", 0.15)),
            hij_consensus_id_threshold=float(getattr(args, "nbo_gsc_hij_consensus_id", 0.85)),
            hij_consensus_ood_threshold=float(getattr(args, "nbo_gsc_hij_consensus_ood", 0.45)),
            hij_enk_ood_weight=float(getattr(args, "nbo_gsc_hij_enk_weight", 0.5)),
            hij_mode_aware=bool(getattr(args, "nbo_gsc_hij_mode_aware", True)),
            hij_xh_freq_min=float(getattr(args, "nbo_gsc_hij_xh_freq_min", 2200.0)),
            hij_xh_freq_max=float(getattr(args, "nbo_gsc_hij_xh_freq_max", 4200.0)),
            hij_mode_participation_threshold=float(getattr(args, "nbo_gsc_hij_mode_participation", 0.015)),
            hij_hii_comp_strength=float(getattr(args, "nbo_gsc_hij_hii_comp", 1.0)),
            hij_gate_mode=str(getattr(args, "nbo_gsc_hij_gate_mode", "adaptive")),
            hij_train_elements=getattr(args, "nbo_gsc_hij_train_elements", "1,6,7,8,9"),
            hij_train_bond_classes=getattr(args, "nbo_gsc_hij_train_bond_classes", ""),
            hij_xh_scale=float(getattr(args, "nbo_gsc_hij_xh_scale", 1.0)),
            hij_unknown_scale=float(getattr(args, "nbo_gsc_hij_unknown_scale", 1.0)),
            hij_unknown_ood_boost=float(getattr(args, "nbo_gsc_hij_unknown_ood_boost", 0.35)),
            hij_strong_ood_threshold=float(getattr(args, "nbo_gsc_hij_strong_ood", 0.70)),
            hij_amide_freq_min=float(getattr(args, "nbo_gsc_hij_amide_freq_min", 1450.0)),
            hij_amide_freq_max=float(getattr(args, "nbo_gsc_hij_amide_freq_max", 1750.0)),
            hij_amide_mode_floor=float(getattr(args, "nbo_gsc_hij_amide_mode_floor", 0.20)),
            hij_global_scale=float(getattr(args, "nbo_gsc_hij_global_scale", 0.5)),
            hij_backbone_mode_floor=float(getattr(args, "nbo_gsc_hij_backbone_mode_floor", 0.08)),
            hij_ood_global_scale=float(getattr(args, "nbo_gsc_hij_ood_global_scale", 1.0)),
            hij_env_ood_hops=int(getattr(args, "nbo_gsc_hij_env_ood_hops", 3)),
            hij_env_ood_decay=float(getattr(args, "nbo_gsc_hij_env_ood_decay", 0.6)),
            hij_env_ood_boost=float(getattr(args, "nbo_gsc_hij_env_ood_boost", 0.40)),
            hij_unknown_mode_floor=float(getattr(args, "nbo_gsc_hij_unknown_mode_floor", 0.55)),
            hij_hbond_distance=float(getattr(args, "nbo_gsc_hij_hbond_distance", 2.45)),
            hij_hbond_acceptors=getattr(args, "nbo_gsc_hij_hbond_acceptors", "7,8,9,15,16,17"),
            hij_hbond_ood_boost=float(getattr(args, "nbo_gsc_hij_hbond_ood_boost", 0.45)),
            hij_hbond_scale=float(getattr(args, "nbo_gsc_hij_hbond_scale", 1.0)),
            hij_hbond_mode_floor=float(getattr(args, "nbo_gsc_hij_hbond_mode_floor", 0.55)),
            hij_ood_alpha_scale=float(getattr(args, "nbo_gsc_hij_ood_alpha_scale", 1.5)),
            hij_hbond_alpha_scale=float(getattr(args, "nbo_gsc_hij_hbond_alpha_scale", 1.6)),
            hij_policy=str(getattr(args, "nbo_gsc_hij_policy", "hybrid")),
            hij_xh_heuristic_scale=float(getattr(args, "nbo_gsc_hij_xh_heuristic_scale", 0.25)),
            hij_interaction_soften_scale=float(getattr(args, "nbo_gsc_hij_interaction_soften_scale", 1.0)),
            hij_interaction_alpha_scale=float(getattr(args, "nbo_gsc_hij_interaction_alpha_scale", 1.0)),
            hij_interaction_geometry_weight=float(getattr(args, "nbo_gsc_hij_interaction_geometry_weight", 0.35)),
            hij_interaction_min_score=float(getattr(args, "nbo_gsc_hij_interaction_min_score", 0.15)),
            hij_hbond_angle_min=float(getattr(args, "nbo_gsc_hij_hbond_angle_min", 115.0)),
            hij_interaction_e2_low_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_low_quantile", 0.75)),
            hij_interaction_e2_high_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_high_quantile", 0.95)),
            hij_interaction_acceptor_distance=float(getattr(args, "nbo_gsc_hij_interaction_acceptor_distance", 3.20)),
            hij_interaction_angle_min=float(getattr(args, "nbo_gsc_hij_interaction_angle_min", 85.0)),
            hij_interaction_field_weight=float(getattr(args, "nbo_gsc_hij_interaction_field_weight", 0.75)),
            hij_physical_rules=not bool(getattr(args, "nbo_gsc_hij_disable_physical_rules", False)),
            hij_physical_min_score=float(getattr(args, "nbo_gsc_hij_physical_min_score", 0.20)),
            hij_physical_soften_scale=float(getattr(args, "nbo_gsc_hij_physical_soften_scale", 0.65)),
            hij_physical_harden_scale=float(getattr(args, "nbo_gsc_hij_physical_harden_scale", 0.40)),
            hij_physical_alpha_scale=float(getattr(args, "nbo_gsc_hij_physical_alpha_scale", 1.35)),
            hij_stark_gate_scale=float(getattr(args, "nbo_gsc_hij_stark_gate_scale", 0.35)),
            dd_trace_mode=str(getattr(args, "nbo_gsc_dd_trace_mode", "off")),
            dd_bond_gate_mode=str(getattr(args, "nbo_gsc_dd_bond_gate_mode", "direct")),
            dd_interaction_gate_scale=float(getattr(args, "nbo_gsc_dd_interaction_gate_scale", 0.0)),
            dp_gate_mode=str(getattr(args, "nbo_gsc_dp_gate_mode", "legacy")),
            dp_interaction_gate_scale=float(getattr(args, "nbo_gsc_dp_interaction_gate_scale", 0.35)),
            dp_interaction_boost=float(getattr(args, "nbo_gsc_dp_interaction_boost", 0.20)),
            bond_occ_col_policy=str(getattr(args, "nbo_gsc_bond_occ_col_policy", "legacy")),
            train_stats=_train_stats,
        )
        print(f"NBO-GSC calibrator activated (batch mode): alpha_hij={nbo_calibrator.alpha_hij}"
              f" alpha_dd={nbo_calibrator.alpha_dd} alpha_dp={nbo_calibrator.alpha_dp}"
              f" branches={','.join(sorted(_gsc_branches))}")

# Standalone NBO-GSC calibrator for SENK-EP mode (batch)
    nbo_calibrator_ep_batch = None
    _ep_active_for_gsc_batch = (
        (model_dp_ep_batch is not None and _is_ep_runtime_active(model_dp_ep_batch))
        or (model_dd_ep_batch is not None and _is_ep_runtime_active(model_dd_ep_batch))
        or (model_hij_ep_batch is not None and _is_ep_runtime_active(model_hij_ep_batch))
    )
    if _ep_active_for_gsc_batch and _gsc_branches:
        _train_stats_ep_batch = None
        if getattr(args, "nbo_train_stats", ""):
            _train_stats_ep_batch = torch.load(args.nbo_train_stats, weights_only=True, map_location="cpu")
        nbo_calibrator_ep_batch = _make_inference_calibrator(args,
            alpha_hij=float(getattr(args, "nbo_gsc_alpha_hij", 0.25)),
            alpha_dd=float(getattr(args, "nbo_gsc_alpha_dd", 0.0)),
            alpha_dp=float(getattr(args, "nbo_gsc_alpha_dp", 0.20)),
            bond_order_exponent=float(getattr(args, "nbo_gsc_bond_exp", 0.6)),
            clamp_ratio=float(getattr(args, "nbo_gsc_clamp", 0.5)),
            bond_dipole_factor=float(getattr(args, "nbo_gsc_bond_dipole_factor", 0.20)),
            alpha_bond=float(getattr(args, "nbo_gsc_alpha_bond", 0.12)),
            dd_sum_rule_strength=float(getattr(args, "nbo_gsc_dd_sum_rule", 0.0)),
            hij_target_xh_only=bool(getattr(args, "nbo_gsc_hij_target_xh_only", False)),
            hij_non_target_scale=float(getattr(args, "nbo_gsc_hij_non_target_scale", 0.15)),
            hij_amide_scale=float(getattr(args, "nbo_gsc_hij_amide_scale", 0.15)),
            hij_consensus_id_threshold=float(getattr(args, "nbo_gsc_hij_consensus_id", 0.85)),
            hij_consensus_ood_threshold=float(getattr(args, "nbo_gsc_hij_consensus_ood", 0.45)),
            hij_enk_ood_weight=float(getattr(args, "nbo_gsc_hij_enk_weight", 0.5)),
            hij_mode_aware=bool(getattr(args, "nbo_gsc_hij_mode_aware", True)),
            hij_xh_freq_min=float(getattr(args, "nbo_gsc_hij_xh_freq_min", 2200.0)),
            hij_xh_freq_max=float(getattr(args, "nbo_gsc_hij_xh_freq_max", 4200.0)),
            hij_mode_participation_threshold=float(getattr(args, "nbo_gsc_hij_mode_participation", 0.015)),
            hij_hii_comp_strength=float(getattr(args, "nbo_gsc_hij_hii_comp", 1.0)),
            hij_gate_mode=str(getattr(args, "nbo_gsc_hij_gate_mode", "adaptive")),
            hij_train_elements=getattr(args, "nbo_gsc_hij_train_elements", "1,6,7,8,9"),
            hij_train_bond_classes=getattr(args, "nbo_gsc_hij_train_bond_classes", ""),
            hij_xh_scale=float(getattr(args, "nbo_gsc_hij_xh_scale", 1.0)),
            hij_unknown_scale=float(getattr(args, "nbo_gsc_hij_unknown_scale", 1.0)),
            hij_unknown_ood_boost=float(getattr(args, "nbo_gsc_hij_unknown_ood_boost", 0.35)),
            hij_strong_ood_threshold=float(getattr(args, "nbo_gsc_hij_strong_ood", 0.70)),
            hij_amide_freq_min=float(getattr(args, "nbo_gsc_hij_amide_freq_min", 1450.0)),
            hij_amide_freq_max=float(getattr(args, "nbo_gsc_hij_amide_freq_max", 1750.0)),
            hij_amide_mode_floor=float(getattr(args, "nbo_gsc_hij_amide_mode_floor", 0.20)),
            hij_global_scale=float(getattr(args, "nbo_gsc_hij_global_scale", 0.5)),
            hij_backbone_mode_floor=float(getattr(args, "nbo_gsc_hij_backbone_mode_floor", 0.08)),
            hij_ood_global_scale=float(getattr(args, "nbo_gsc_hij_ood_global_scale", 1.0)),
            hij_env_ood_hops=int(getattr(args, "nbo_gsc_hij_env_ood_hops", 3)),
            hij_env_ood_decay=float(getattr(args, "nbo_gsc_hij_env_ood_decay", 0.6)),
            hij_env_ood_boost=float(getattr(args, "nbo_gsc_hij_env_ood_boost", 0.40)),
            hij_unknown_mode_floor=float(getattr(args, "nbo_gsc_hij_unknown_mode_floor", 0.55)),
            hij_hbond_distance=float(getattr(args, "nbo_gsc_hij_hbond_distance", 2.45)),
            hij_hbond_acceptors=getattr(args, "nbo_gsc_hij_hbond_acceptors", "7,8,9,15,16,17"),
            hij_hbond_ood_boost=float(getattr(args, "nbo_gsc_hij_hbond_ood_boost", 0.45)),
            hij_hbond_scale=float(getattr(args, "nbo_gsc_hij_hbond_scale", 1.0)),
            hij_hbond_mode_floor=float(getattr(args, "nbo_gsc_hij_hbond_mode_floor", 0.55)),
            hij_ood_alpha_scale=float(getattr(args, "nbo_gsc_hij_ood_alpha_scale", 1.5)),
            hij_hbond_alpha_scale=float(getattr(args, "nbo_gsc_hij_hbond_alpha_scale", 1.6)),
            hij_policy=str(getattr(args, "nbo_gsc_hij_policy", "hybrid")),
            hij_xh_heuristic_scale=float(getattr(args, "nbo_gsc_hij_xh_heuristic_scale", 0.25)),
            hij_interaction_soften_scale=float(getattr(args, "nbo_gsc_hij_interaction_soften_scale", 1.0)),
            hij_interaction_alpha_scale=float(getattr(args, "nbo_gsc_hij_interaction_alpha_scale", 1.0)),
            hij_interaction_geometry_weight=float(getattr(args, "nbo_gsc_hij_interaction_geometry_weight", 0.35)),
            hij_interaction_min_score=float(getattr(args, "nbo_gsc_hij_interaction_min_score", 0.15)),
            hij_hbond_angle_min=float(getattr(args, "nbo_gsc_hij_hbond_angle_min", 115.0)),
            hij_interaction_e2_low_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_low_quantile", 0.75)),
            hij_interaction_e2_high_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_high_quantile", 0.95)),
            hij_interaction_acceptor_distance=float(getattr(args, "nbo_gsc_hij_interaction_acceptor_distance", 3.20)),
            hij_interaction_angle_min=float(getattr(args, "nbo_gsc_hij_interaction_angle_min", 85.0)),
            hij_interaction_field_weight=float(getattr(args, "nbo_gsc_hij_interaction_field_weight", 0.75)),
            hij_physical_rules=not bool(getattr(args, "nbo_gsc_hij_disable_physical_rules", False)),
            hij_physical_min_score=float(getattr(args, "nbo_gsc_hij_physical_min_score", 0.20)),
            hij_physical_soften_scale=float(getattr(args, "nbo_gsc_hij_physical_soften_scale", 0.65)),
            hij_physical_harden_scale=float(getattr(args, "nbo_gsc_hij_physical_harden_scale", 0.40)),
            hij_physical_alpha_scale=float(getattr(args, "nbo_gsc_hij_physical_alpha_scale", 1.35)),
            hij_stark_gate_scale=float(getattr(args, "nbo_gsc_hij_stark_gate_scale", 0.35)),
            dd_trace_mode=str(getattr(args, "nbo_gsc_dd_trace_mode", "off")),
            dd_bond_gate_mode=str(getattr(args, "nbo_gsc_dd_bond_gate_mode", "direct")),
            dd_interaction_gate_scale=float(getattr(args, "nbo_gsc_dd_interaction_gate_scale", 0.0)),
            dp_gate_mode=str(getattr(args, "nbo_gsc_dp_gate_mode", "legacy")),
            dp_interaction_gate_scale=float(getattr(args, "nbo_gsc_dp_interaction_gate_scale", 0.35)),
            dp_interaction_boost=float(getattr(args, "nbo_gsc_dp_interaction_boost", 0.20)),
            bond_occ_col_policy=str(getattr(args, "nbo_gsc_bond_occ_col_policy", "legacy")),
            train_stats=_train_stats_ep_batch,
        )
        print(f"NBO-GSC calibrator (SENK-EP batch) activated: alpha_hij={nbo_calibrator_ep_batch.alpha_hij}"
              f" alpha_dd={nbo_calibrator_ep_batch.alpha_dd} alpha_dp={nbo_calibrator_ep_batch.alpha_dp}"
              f" branches={','.join(sorted(_gsc_branches))}")

    _lp_ckpt = None if args.disable_lp else args.lp_ckpt

# Phase 2b: Optional DetaNet comparison models
    det_models = None
    if args.compare_detanet:
        print(f"Loading DetaNet comparison models (max_atomic_number={trained_max_z})...")
        det_models = {
            "Hi":       _build_detanet("Hi",       device, Path(args.detanet_Hi),       trained_max_z),
            "Hij":      _build_detanet("Hij",       device, Path(args.detanet_Hij),      trained_max_z),
            "dedipole": _build_detanet("dedipole",  device, Path(args.detanet_dedipole), trained_max_z),
            "depolar":  _build_detanet("depolar",   device, Path(args.detanet_depolar),  trained_max_z),
        }

# Phase 3: Batch inference
    failed_infer = []
    for seq, (idx, smi, z_list, pos_list) in enumerate(valid_entries, 1):
        fname_base = f"{idx:04d}_{_smi_to_filename(smi)}"
        print(f"[{seq}/{len(valid_entries)}] mol {idx:04d}  natoms={len(z_list)}")
        try:
            z   = torch.LongTensor(z_list)
            pos = torch.FloatTensor(pos_list)

            results = generate_spectrum(
                pos=pos, z=z,
                model_hii=model_hii, model_hij=model_hij,
                model_dd=model_dd,   model_dp=model_dp,
                device=device,
                linear=args.linear, scale=args.scale, sigma=args.sigma,
                sigma_ir=sigma_ir, sigma_raman=sigma_raman,
                freq_scale_factor=args.freq_scale_factor,
                freq_range=(args.freq_min, args.freq_max),
                freq_points=args.freq_points, radius=args.radius,
                lp_ckpt=_lp_ckpt,
                skeleton_radius=args.skeleton_radius,
                nbo_calibrator=nbo_calibrator,
                gsc_branches=_gsc_branches,
            )

            results_off = generate_spectrum(
                pos=pos, z=z,
                model_hii=model_hii_off, model_hij=model_hij_off,
                model_dd=model_dd_off,   model_dp=model_dp_off,
                device=device,
                linear=args.linear, scale=args.scale, sigma=args.sigma,
                sigma_ir=sigma_ir, sigma_raman=sigma_raman,
                freq_scale_factor=args.freq_scale_factor,
                freq_range=(args.freq_min, args.freq_max),
                freq_points=args.freq_points, radius=args.radius,
                lp_ckpt=_lp_ckpt,
                skeleton_radius=args.skeleton_radius,
                nbo_calibrator=nbo_calibrator,
                gsc_branches=_gsc_branches,
            )

            results_ep = None
            _has_any_ep_batch = (model_dp_ep_batch is not None or model_dd_ep_batch is not None
                               or model_hij_ep_batch is not None)
            if _has_any_ep_batch:
                _ep_hij_b = model_hij_ep_batch if model_hij_ep_batch is not None else model_hij
                _ep_dd = model_dd_ep_batch if model_dd_ep_batch is not None else model_dd
                _ep_dp = model_dp_ep_batch if model_dp_ep_batch is not None else model_dp
                results_ep = generate_spectrum(
                    pos=pos, z=z,
                    model_hii=model_hii, model_hij=_ep_hij_b,
                    model_dd=_ep_dd,   model_dp=_ep_dp,
                    device=device,
                    linear=args.linear, scale=args.scale, sigma=args.sigma,
                    sigma_ir=sigma_ir, sigma_raman=sigma_raman,
                    freq_scale_factor=args.freq_scale_factor,
                    freq_range=(args.freq_min, args.freq_max),
                    freq_points=args.freq_points, radius=args.radius,
                    lp_ckpt=_lp_ckpt,
                    skeleton_radius=args.skeleton_radius,
                    nbo_calibrator=nbo_calibrator_ep_batch,
                    gsc_branches=_gsc_branches,
                )

            # Optional DetaNet comparison
            det_results = None
            if det_models is not None:
                det_results = generate_detanet_spectrum(
                    pos=pos, z=z,
                    model_Hi=det_models["Hi"], model_Hij=det_models["Hij"],
                    model_dd=det_models["dedipole"], model_dp=det_models["depolar"],
                    device=device,
                    linear=args.linear, scale=args.scale, sigma=args.sigma,
                    sigma_ir=sigma_ir, sigma_raman=sigma_raman,
                    freq_scale_factor=args.freq_scale_factor,
                    freq_range=(args.freq_min, args.freq_max),
                    freq_points=args.freq_points,
                )

            # Save npz
            npz_path = out_dir / f"{fname_base}.npz"
            save_d = {k: _spectrum_archive_value(v) for k, v in results.items()}
            save_d["smiles"] = np.array([smi])
            if results_ep is not None:
                for k, v in results_ep.items():
                    save_d[f"ep_{k}"] = _spectrum_archive_value(v)
            if det_results is not None:
                for k, v in det_results.items():
                    save_d[f"det_{k}"] = v.numpy()
            np.savez(str(npz_path), **save_d)
            print(f"  data: {npz_path.name}")

            # Save PNG
            if args.batch_save_png:
                png_path = out_dir / f"{fname_base}.png"
                title_mol = f"mol {idx:04d}  {smi[:40]}"
                _plot_spectrum(results, title_suffix=title_mol,
                               senk_off_results=results_off,
                               ep_results=results_ep,
                               detanet_results=det_results,
                               out_png=str(png_path),
                               model_label="SENK(ENK on)",
                               senk_off_label="SENK(ENK off)",
                               ep_label="SENK-EP",
                               freq_range=(args.freq_min, args.freq_max),
                               sigma_ir=sigma_ir, sigma_raman=sigma_raman,
                               ref_freq_scale=ref_freq_scale)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [WARN] Inference failed: {e}")
            failed_infer.append((idx, smi, str(e)))

    total_ok = len(valid_entries) - len(failed_infer)
    print(f"\nBatch complete: {total_ok}/{len(smiles_list)} succeeded"
          f"  ({len(failed_embed)} embedding failures, {len(failed_infer)} inference failures,"
          f"  {len(skipped_oor)} OOR skipped)"
          f"\nOutput directory: {out_dir.resolve()}")


# --- CLI ---

def _str_to_bool(val):
    """Parse a string to boolean for argparse."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes", "t")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EquiformerV2 self-trained four-weight IR + Raman spectra inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output (--xyz / --sdf / --smiles / --batch_smi: choose one)
    p.add_argument("--xyz", default=None,
                   help="Input XYZ file (single-molecule mode; mutually exclusive with --sdf / --smiles / --batch_smi)")
    p.add_argument("--sdf", default=None,
                   help="Input SDF/MOL file (single-molecule mode; an SDF exported from an optimized Gaussian geometry is recommended"
                        " to keep the geometry consistent with the reference spectrum)")
    p.add_argument("--sdf_conf_index", type=int, default=0,
                   help="For multi-record SDF files, the 0-based record index to use (default 0)")
    p.add_argument("--smiles", default=None,
                   help="A single SMILES string (single-molecule mode; RDKit ETKDGv3+MMFF generates the 3D conformer;"
                        " mutually exclusive with --xyz / --sdf / --batch_smi)")
    p.add_argument("--out", default=None, help="Output .npz spectrum data file")
    p.add_argument("--out_png", default=None, help="Output .png image file")

# Batch SMILES mode
    p.add_argument("--batch_smi", type=str, default=None,
                   help="Path to a .smi file: batch-generate IR+Raman spectra for every SMILES in the file. Mutually exclusive with --xyz.")
    p.add_argument("--batch_out_dir", type=str, default="v2_batch_output",
                   help="Batch output directory (.npz and .png per molecule)")
    p.add_argument("--batch_max_mols", type=int, default=0,
                   help="Process at most the first N molecules; 0 = all")
    p.add_argument("--batch_save_png", type=_str_to_bool, default=True, choices=[True, False],
                   help="Whether to save a spectrum PNG for every molecule")
    p.add_argument("--max_atomic_number", type=int, default=9,
                   help="Maximum allowed atomic number (QM9 training set is 9, i.e. H-F); out-of-range molecules are skipped in batch mode")

# DetaNet comparison curves
    p.add_argument("--compare_detanet", action="store_true",
                   help="Also run the four DetaNet models and overlay their comparison curves (requires --detanet_* weight paths)")
    p.add_argument("--detanet_ood_embedding", type=str, default="mean_periodic",
                   choices=["mean_periodic", "mean", "random", "error"],
                   help="DetaNet checkpoint-OOD element embedding protection strategy. "
                        "mean_periodic replaces random table expansion with same-group CHONF row and periodic-column copies; "
                        "random preserves the legacy behavior; error raises directly when an element exceeds the checkpoint range.")
    p.add_argument("--detanet_Hi",       type=str, default=str(DETANET_WEIGHTS["Hi"]),
                   help="DetaNet Hii weight path")
    p.add_argument("--detanet_Hij",      type=str, default=str(DETANET_WEIGHTS["Hij"]),
                   help="DetaNet Hij weight path")
    p.add_argument("--detanet_dedipole", type=str, default=str(DETANET_WEIGHTS["dedipole"]),
                   help="DetaNet dedipole weight path")
    p.add_argument("--detanet_depolar",  type=str, default=str(DETANET_WEIGHTS["depolar"]),
                   help="DetaNet depolar weight path")
# DFT/Gaussian reference spectra (optional, for case-study comparison plots)
    p.add_argument("--ref_ir", type=str, default="",
                   help="DFT reference IR spectrum file (.txt/.log/.out). For Gaussian .txt, Peak information is read first.")
    p.add_argument("--ref_raman", type=str, default="",
                   help="DFT reference Raman spectrum file (.txt/.log/.out). For Gaussian .txt, Peak information is read first.")
    p.add_argument("--ref_txt_mode", type=str, default="auto",
                   choices=["auto", "peak_info", "spectra"],
                   help="How to read Gaussian .txt reference spectra. auto prefers Peak information over the already-broadened Spectra curve.")

    # Device
    p.add_argument("--ep_output_policy", choices=["feature_spring", "legacy"], default="feature_spring", help="Output EP policy; feature_spring is the validated default")
    p.add_argument("--ep_support_stats", default="nbo_train_stats_hij_support_v2.pt")
    p.add_argument("--ep_support_artifact", default="outputs/selective_ep_calibration/chemical_support_calibrator.json")
    p.add_argument("--ep", action="store_true",
                   help="Use the canonical SENK-EP defaults and auto-fill the EP checkpoints.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--gpu", type=int, default=None)

# Four weight paths and their respective training modes
    p.add_argument("--hii_ckpt", type=str, default=DEFAULT_SENK_ENK_ON_HII,
                   help="SENK(ENK on) Hii weight path")
    p.add_argument("--hii_mode", type=str, default="equiformer_v2",
                   choices=_ALL_MODES,
                   help="Training mode used for the Hii weight")

    p.add_argument("--hij_ckpt", type=str, default=DEFAULT_SENK_ENK_ON_HIJ,
                   help="SENK(ENK on) Hij weight path")
    p.add_argument("--hij_mode", type=str, default="equiformer_v2",
                   choices=_ALL_MODES,
                   help="Training mode used for the Hij weight")
    p.add_argument("--hij_ep_ckpt", type=str, default="",
                   help="SENK-EP Hij weight path (loads an additional EP model for a third comparison curve)")
    p.add_argument("--hij_ep_mode", type=str, default="equiformer_v2_enk_ep",
                   choices=_ALL_MODES,
                   help="Training mode used for the Hij EP weight")

    p.add_argument("--dd_ckpt", type=str, default=DEFAULT_SENK_ENK_ON_DD,
                   help="SENK(ENK on) dedipole weight path")
    p.add_argument("--dd_mode", type=str, default="equiformer_v2_enk",
                   choices=_ALL_MODES,
                   help="Training mode used for the dedipole weight")
    p.add_argument("--dd_ep_ckpt", type=str, default="",
                   help="SENK-EP dedipole weight path (independent of the ENK-on dedipole;"
                        " loads an additional EP model for a third comparison curve)")
    p.add_argument("--dd_ep_mode", type=str, default="equiformer_v2_enk_ep",
                   choices=_ALL_MODES,
                   help="Training mode used for the dedipole EP weight")

    p.add_argument("--dp_ckpt", type=str, default=DEFAULT_SENK_ENK_ON_DP,
                   help="SENK(ENK on) depolar weight path")
    p.add_argument("--dp_mode", type=str, default="equiformer_v2_enk",
                   choices=_ALL_MODES,
                   help="Training mode used for the depolar weight")
    p.add_argument("--dp_ep_ckpt", type=str, default="",
                   help="SENK-EP depolar weight path (independent of the ENK-on depolar;"
                        " loads an additional EP model for a third comparison curve)")
    p.add_argument("--dp_ep_mode", type=str, default="equiformer_v2_enk_ep",
                   choices=_ALL_MODES,
                   help="Training mode used for the depolar EP weight")

# ENK-off comparison weights (pure V2, ENK force-disabled)
    p.add_argument("--hii_off_ckpt", type=str, default=DEFAULT_SENK_ENK_OFF_HII,
                   help="SENK(ENK off) Hii weight path")
    p.add_argument("--hij_off_ckpt", type=str, default=DEFAULT_SENK_ENK_OFF_HIJ,
                   help="SENK(ENK off) Hij weight path")
    p.add_argument("--dd_off_ckpt", type=str, default=DEFAULT_SENK_ENK_OFF_DD,
                   help="SENK(ENK off) dedipole weight path")
    p.add_argument("--dp_off_ckpt", type=str, default=DEFAULT_SENK_ENK_OFF_DP,
                   help="SENK(ENK off) depolar weight path")

# Shared backbone parameters
    p.add_argument("--hidden_nf", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_basis", type=int, default=128)
    p.add_argument("--radius", type=float, default=5.0)

# V2 backbone parameters
    p.add_argument("--v2_model_name", type=str, default="equiformer_v2_l4_m2",
                   choices=["equiformer_v2_l3_m2", "equiformer_v2_l4_m2",
                            "equiformer_v2_l6_m2", "equiformer_v2_l4_m2_small"])
    p.add_argument("--v2_max_neighbors", type=int, default=64)
    p.add_argument("--v2_num_gaussians", type=int, default=64)
    p.add_argument("--v2_grid_resolution", type=int, default=14)
    p.add_argument("--v2_use_gate_act", type=_str_to_bool, default=True,
                   choices=[True, False])
    p.add_argument("--v2_use_grid_mlp", type=_str_to_bool, default=False,
                   choices=[True, False])

# ENK parameters
    p.add_argument("--enk_enabled", type=_str_to_bool, default=False, choices=[True, False],
                   help="Global ENK switch (auto-enabled in equiformer_v2_enk mode)")
    p.add_argument("--enk_init_r_bias", type=float, default=-1.0)
    p.add_argument("--enk_init_q_bias", type=float, default=1.0)

# Electron Prior parameters (used with equiformer_v2_enk_ep)
    p.add_argument("--electron_prior_mode", type=str, default="off",
                   choices=["off", "simg", "qcmol"])
    p.add_argument("--electron_prior_ckpt", type=str, default="",
                   help="NBOPriorBranch weight path")
    p.add_argument("--electron_prior_stats", type=str, default="")
    p.add_argument("--electron_prior_scale", type=float, default=1e-2)
    p.add_argument("--electron_prior_use_aux", type=_str_to_bool, default=True,
                   choices=[True, False])
    p.add_argument("--electron_prior_freeze", type=_str_to_bool, default=True,
                   choices=[True, False])
    p.add_argument("--electron_prior_runtime_mode", type=str, default="full",
                   choices=["full", "detached"])
    p.add_argument("--electron_prior_heads", type=int, default=4)

# NBO-GSC calibration parameters
    p.add_argument("--nbo_ep_branches", type=str, default="",
                   help="Comma-separated EP-activated branches: hii/hij/dd/dp. "
                        "For example, --nbo_ep_branches dp means only depolar uses EP. "
                        "Empty = all branches inherit the --electron_prior_ckpt behavior (backward compatibility)")
    p.add_argument("--nbo_gsc_branches", type=str, default="hij,dd",
                   help="Comma-separated GSC-calibrated branches: hii/hij/dd/dp. "
                        "For example, --nbo_gsc_branches hij,dd calibrates hij and dedipole. "
                        "Requires at least one EP-activated branch to supply NBO features. "
                        "Empty = calibrate all branches when EP is available")
    p.add_argument("--nbo_train_stats", type=str, default="",
                   help="Training-set norm_stats.pt for GSC z-score normalization (optional)")
    p.add_argument("--nbo_gsc_enabled", type=_str_to_bool, default=False, choices=[True, False],
                   help="Enable NBO-Guided Spectral Calibration (effective only when EP is active)")
    p.add_argument("--nbo_gsc_alpha_hij", type=float, default=0.15,
                   help="Hij calibration strength (0 = no calibration, 1 = full replacement)")
    p.add_argument("--nbo_gsc_alpha_dd", type=float, default=0.0,
                   help="dedipole calibration strength")
    p.add_argument("--nbo_gsc_alpha_dp", type=float, default=0.20,
                   help="depolar calibration strength")
    p.add_argument("--nbo_gsc_bond_exp", type=float, default=0.6,
                   help="Bond-order-|Hij| power-law exponent (Badger-rule variant)")
    p.add_argument("--nbo_gsc_clamp", type=float, default=0.5,
                   help="Calibration-ratio clamping range (1 +/- clamp)")
    p.add_argument("--nbo_gsc_hij_target_xh_only", type=_str_to_bool, default=False,
                   choices=[True, False],
                   help="Legacy Hij-GSC mode: restrict strong correction to O-H/N-H stretch-related bonds.")
    p.add_argument("--nbo_gsc_hij_non_target_scale", type=float, default=0.05,
                   help="Base chemistry gate for known non-X-H bonds before OOD evidence opens the gate.")
    p.add_argument("--nbo_gsc_hij_amide_scale", type=float, default=0.10,
                   help="Base chemistry gate for amide C=O/C-N bonds; strong OOD evidence can still open it.")
    p.add_argument("--nbo_gsc_hij_consensus_id", type=float, default=0.85,
                   help="NBO bond-order consensus above this is treated as ID for Hij-GSC.")
    p.add_argument("--nbo_gsc_hij_consensus_ood", type=float, default=0.45,
                   help="NBO bond-order consensus below this is treated as OOD for Hij-GSC.")
    p.add_argument("--nbo_gsc_hij_enk_weight", type=float, default=0.5,
                   help="Additional ENK OOD trigger weight for Hij-GSC.")
    p.add_argument("--nbo_gsc_hij_mode_aware", type=_str_to_bool, default=True,
                   choices=[True, False],
                   help="Use uncalibrated normal modes to gate Hij-GSC by high-frequency X-H stretch participation.")
    p.add_argument("--nbo_gsc_hij_xh_freq_min", type=float, default=2200.0,
                   help="Lower frequency bound for mode-aware X-H Hij-GSC (cm^-1).")
    p.add_argument("--nbo_gsc_hij_xh_freq_max", type=float, default=4200.0,
                   help="Upper frequency bound for mode-aware X-H Hij-GSC (cm^-1).")
    p.add_argument("--nbo_gsc_hij_mode_participation", type=float, default=0.015,
                   help="Bond-stretch participation threshold for mode-aware Hij-GSC.")
    p.add_argument("--nbo_gsc_hij_hii_comp", type=float, default=0.1,
                   help="Hii compensation strength after Hij-GSC; 0 disables acoustic consistency correction.")
    p.add_argument("--nbo_gsc_hij_gate_mode", type=str, default="adaptive",
                   choices=["adaptive", "legacy_xh", "xh_only", "static"],
                   help="Hij-GSC chemistry gate mode. adaptive uses bond class + OOD evidence.")
    p.add_argument("--nbo_gsc_hij_train_elements", type=str, default="1,6,7,8,9",
                   help="Comma-separated atomic numbers treated as training-set elements for Hij-GSC OOD gating.")
    p.add_argument("--nbo_gsc_hij_train_bond_classes", type=str, default="",
                   help="Optional comma-separated seen bond classes; if empty, use classes saved in nbo_train_stats.")
    p.add_argument("--nbo_gsc_hij_xh_scale", type=float, default=0.6,
                   help="Base chemistry gate for O-H/N-H stretch bonds.")
    p.add_argument("--nbo_gsc_hij_unknown_scale", type=float, default=0.8,
                   help="Chemistry gate for bonds with unseen atoms or unseen bond classes.")
    p.add_argument("--nbo_gsc_hij_unknown_ood_boost", type=float, default=0.35,
                   help="Additional Hij-GSC trigger for unseen atoms or unseen bond classes.")
    p.add_argument("--nbo_gsc_hij_strong_ood", type=float, default=0.70,
                   help="Trigger level where adaptive known-bond gates open toward unknown_scale.")
    p.add_argument("--nbo_gsc_hij_amide_freq_min", type=float, default=1450.0,
                   help="Lower mode window for amide C=O/C-N stretch participation.")
    p.add_argument("--nbo_gsc_hij_amide_freq_max", type=float, default=1750.0,
                   help="Upper mode window for amide C=O/C-N stretch participation.")
    p.add_argument("--nbo_gsc_hij_amide_mode_floor", type=float, default=0.20,
                   help="Minimum mode gate for amide bonds, avoiding a hard off switch.")
    p.add_argument("--nbo_gsc_hij_global_scale", type=float, default=0.5,
                   help="Global scale factor applied to the final Hij-GSC gate product. "
                        "Reduce to conservatively dial back all calibration (0=off, 1=full).")
    p.add_argument("--nbo_gsc_hij_backbone_mode_floor", type=float, default=0.08,
                   help="Upper clamp for mode-gate on non-XH/non-amide backbone bonds. "
                        "Prevents backbone C-C/C-N bonds from receiving full mode pass-through.")
    p.add_argument("--nbo_gsc_hij_ood_global_scale", type=float, default=1.0,
                   help="Final Hij-GSC gate scale for rare-atom / H-bond local environments.")
    p.add_argument("--nbo_gsc_hij_env_ood_hops", type=int, default=3,
                   help="Bond-graph hops over which unseen-atom OOD signal is propagated.")
    p.add_argument("--nbo_gsc_hij_env_ood_decay", type=float, default=0.6,
                   help="Decay per bond-graph hop for propagated unseen-atom OOD signal.")
    p.add_argument("--nbo_gsc_hij_env_ood_boost", type=float, default=0.40,
                   help="Additional Hij-GSC trigger from propagated local-environment OOD.")
    p.add_argument("--nbo_gsc_hij_unknown_mode_floor", type=float, default=0.55,
                   help="Mode-gate floor/cap used when a bond is in a rare local environment.")
    p.add_argument("--nbo_gsc_hij_hbond_distance", type=float, default=2.45,
                   help="Maximum H...acceptor distance for geometry-based hydrogen-bond OOD detection.")
    p.add_argument("--nbo_gsc_hij_hbond_acceptors", type=str, default="7,8,9,15,16,17",
                   help="Comma-separated acceptor atomic numbers for geometry-based H-bond detection.")
    p.add_argument("--nbo_gsc_hij_hbond_ood_boost", type=float, default=0.45,
                   help="Additional Hij-GSC trigger for detected X-H...acceptor hydrogen bonds.")
    p.add_argument("--nbo_gsc_hij_hbond_scale", type=float, default=1.0,
                   help="Chemistry gate target for detected hydrogen-bond donor X-H bonds.")
    p.add_argument("--nbo_gsc_hij_hbond_mode_floor", type=float, default=0.55,
                   help="Mode-gate floor for detected hydrogen-bond donor X-H bonds.")
    p.add_argument("--nbo_gsc_hij_ood_alpha_scale", type=float, default=1.5,
                   help="Local multiplier on Hij-GSC delta in rare-atom local environments.")
    p.add_argument("--nbo_gsc_hij_hbond_alpha_scale", type=float, default=1.6,
                   help="Local multiplier on Hij-GSC delta for detected hydrogen-bond donor X-H bonds.")
    p.add_argument("--nbo_gsc_hij_policy", type=str, default="hybrid",
                   choices=["heuristic", "hybrid", "interaction_rule"],
                   help="Hij-GSC X-H direction policy. heuristic keeps the original z_occ-z_hij rule; "
                        "hybrid uses interaction-driven X-H softening with a weak heuristic fallback.")
    p.add_argument("--nbo_gsc_hij_xh_heuristic_scale", type=float, default=0.25,
                   help="Residual strength of the original z_occ-z_hij rule on X-H bonds when no interaction evidence is present.")
    p.add_argument("--nbo_gsc_hij_interaction_soften_scale", type=float, default=1.0,
                   help="Strength of interaction-driven X-H softening when donor-acceptor evidence is detected.")
    p.add_argument("--nbo_gsc_hij_interaction_alpha_scale", type=float, default=1.0,
                   help="Post-clamp alpha for interaction-confirmed X-H softening only.")
    p.add_argument("--nbo_gsc_hij_interaction_geometry_weight", type=float, default=0.35,
                   help="Minimum geometry contribution in the interaction-driven X-H softening score.")
    p.add_argument("--nbo_gsc_hij_interaction_min_score", type=float, default=0.15,
                   help="Minimum X-H donor-acceptor interaction score required to override the heuristic direction.")
    p.add_argument("--nbo_gsc_hij_hbond_angle_min", type=float, default=115.0,
                   help="X-H...acceptor angle where geometry starts contributing to the softening score.")
    p.add_argument("--nbo_gsc_hij_interaction_e2_low_quantile", type=float, default=0.75,
                   help="Per-molecule lower quantile used to normalize predicted NBO interaction E2 scores.")
    p.add_argument("--nbo_gsc_hij_interaction_e2_high_quantile", type=float, default=0.95,
                   help="Per-molecule upper quantile used to normalize predicted NBO interaction E2 scores.")
    p.add_argument("--nbo_gsc_hij_interaction_acceptor_distance", type=float, default=3.20,
                   help="Maximum H...acceptor distance for X-H donor-acceptor interaction-field attribution.")
    p.add_argument("--nbo_gsc_hij_interaction_angle_min", type=float, default=85.0,
                   help="Minimum X-H...acceptor angle for interaction-field attribution.")
    p.add_argument("--nbo_gsc_hij_interaction_field_weight", type=float, default=0.75,
                   help="Weight assigned to bond-node interaction-field strength when targeted LP/acceptor mapping is missing.")
    p.add_argument("--nbo_gsc_hij_disable_physical_rules", action="store_true",
                   help="Disable non-XH physical evidence rules; keeps the current X-H interaction rule available.")
    p.add_argument("--nbo_gsc_hij_physical_min_score", type=float, default=0.20,
                   help="Minimum direct NBO/field evidence score for non-XH physical Hij softening/hardening.")
    p.add_argument("--nbo_gsc_hij_physical_soften_scale", type=float, default=0.65,
                   help="Strength for non-XH direct bond-weakening Hij softening.")
    p.add_argument("--nbo_gsc_hij_physical_harden_scale", type=float, default=0.40,
                   help="Strength for non-XH direct bond-hardening Hij correction.")
    p.add_argument("--nbo_gsc_hij_physical_alpha_scale", type=float, default=1.35,
                   help="Post-clamp local delta gain for non-XH physical evidence.")
    p.add_argument("--nbo_gsc_hij_stark_gate_scale", type=float, default=0.35,
                   help="Gate-only weight for Stark-like local electric-field evidence on polar probes.")
    p.add_argument("--nbo_gsc_dd_trace_mode", type=str, default="off",
                   choices=["legacy", "direct", "off"],
                   help="Dedipole trace correction gate mode.")
    p.add_argument("--nbo_gsc_dd_bond_gate_mode", type=str, default="direct",
                   choices=["legacy", "direct"],
                   help="Dedipole bond-directional anisotropy gate mode.")
    p.add_argument("--nbo_gsc_dd_interaction_gate_scale", type=float, default=0.0,
                   help="Atom-level NBO interaction exposure weight for dedipole GSC gates.")
    p.add_argument("--nbo_gsc_dp_gate_mode", type=str, default="legacy",
                   choices=["legacy", "direct"],
                   help="Depolar calibration gate mode.")
    p.add_argument("--nbo_gsc_dp_interaction_gate_scale", type=float, default=0.35,
                   help="Atom-level NBO interaction exposure weight for depolar GSC gates.")
    p.add_argument("--nbo_gsc_dp_interaction_boost", type=float, default=0.20,
                   help="Multiplicative delocalization-proxy boost from atom-level NBO interaction exposure.")
    p.add_argument("--nbo_gsc_bond_occ_col_policy", type=str, default="legacy",
                   choices=["legacy", "stats", "f0", "col0", "schema", "qcmol", "f14", "col14", "auto"],
                   help="Bond occupancy/proxy column policy. Legacy/f0 matches current stats; schema/f14 is exploratory only.")
    p.add_argument("--nbo_gsc_alpha_bond", type=float, default=0.12,
                   help="GSC dedipole bond-directional correction strength")
    p.add_argument("--nbo_gsc_bond_dipole_factor", type=float, default=0.20,
                   help="NBO charge-transfer factor for dedipole bond correction")
    p.add_argument("--nbo_gsc_dd_sum_rule", type=float, default=0.0,
                   help="Born effective charge sum-rule projection strength")

# End-to-end skeleton generation parameters
    p.add_argument("--lp_ckpt", type=str, default=str(ROOT / "tools" / "lp_pred_model.ckpt"),
                   help="LP predictor weight path (empty -> no LP nodes are generated)")
    p.add_argument("--disable_lp", action="store_true",
                   help="Force-disable LP prediction (generate only the atom + bond skeleton)")
    p.add_argument("--skeleton_radius", type=float, default=5.0,
                   help="Skeleton interaction-edge radius (Angstrom), must match the training side")

# Spectrum parameters
    p.add_argument("--scale", type=float, default=0.965,
                   help="Frequency scale factor (harmonic-oscillator correction), also passed as the hessfreq scale factor; "
                        "default 0.965 (B3LYP/TZVP harmonic-frequency correction)")
    p.add_argument("--ref_freq_scale", type=float, default=None,
                   help="Scale factor applied to the stick frequencies of the Gaussian .log/.txt reference spectrum. "
                        "Default None = automatically consistent with --scale (0.965), "
                        "placing the reference and predicted spectra in the same frequency space (both DFT x scale). "
                        "Set to 1.0 to display the raw uncorrected DFT frequencies.")
    p.add_argument("--freq_scale_factor", type=float, default=1.0,
                   help="Additional frequency scale factor for fine-tuning predicted frequencies to align with the reference spectrum (default 1.0; "
                        "affects predictions only, not the reference spectrum)")
    p.add_argument("--sigma", type=float, default=12.0,
                   help="Lorenz broadening half-width (cm^-1)")
    p.add_argument("--sigma_ir", type=float, default=None,
                   help="IR-specific broadening half-width (cm^-1). When unspecified, --sigma is used. DetaNet examples typically use 15.")
    p.add_argument("--sigma_raman", type=float, default=None,
                   help="Raman-specific broadening half-width (cm^-1). When unspecified, --sigma is used. DetaNet examples typically use 12.")
    p.add_argument("--sigma_scale_factor", type=float, default=1.0,
                   help="Additional broadening scale multiplier applied to the effective sigma of both IR and Raman.")
    p.add_argument("--linear", action="store_true",
                   help="Linear molecule (remove 5 translational/rotational modes instead of 6)")
    p.add_argument("--freq_min", type=float, default=500.0)
    p.add_argument("--freq_max", type=float, default=4000.0)
    p.add_argument("--freq_points", type=int, default=3501)

    return p


def main():
    global DETANET_OOD_EMBEDDING_MODE
    args = build_parser().parse_args()
    _ep_requested = bool(
        getattr(args, "ep", False)
        or getattr(args, "hij_ep_ckpt", "")
        or getattr(args, "dd_ep_ckpt", "")
        or getattr(args, "dp_ep_ckpt", "")
    )
    if _ep_requested:
        _ep_auto_filled = apply_canonical_ep_defaults(
            args,
            fill_ep_checkpoints=bool(getattr(args, "ep", False)),
        )
        if _ep_auto_filled:
            print(
                "[INFO] canonical SENK-EP defaults -> "
                + ", ".join(_ep_auto_filled)
            )
    DETANET_OOD_EMBEDDING_MODE = str(getattr(args, "detanet_ood_embedding", "mean_periodic"))
    sigma_ir = (args.sigma_ir if args.sigma_ir is not None else args.sigma) * args.sigma_scale_factor
    sigma_raman = (args.sigma_raman if args.sigma_raman is not None else args.sigma) * args.sigma_scale_factor
    # ref_freq_scale: None → auto-match --scale (0.965 by default)
    # This ensures Gaussian .log raw DFT sticks are shifted to the same "DFT × scale"
    # space as the prediction, matching DetaNet's paper comparison methodology.
    ref_freq_scale = args.ref_freq_scale if args.ref_freq_scale is not None else args.scale

# Device selection
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

# Batch SMILES mode
    if args.batch_smi:
        if args.xyz:
            print("[WARN] Both --batch_smi and --xyz specified; using batch mode and ignoring --xyz.")
        _run_batch_full(args, device)
        return

# Single-molecule mode
    if not args.xyz and not args.sdf and not args.smiles:
        print("[ERROR] Specify --xyz / --sdf / --smiles (single molecule) or --batch_smi (batch SMILES).")
        sys.exit(1)

    if args.smiles:
        # SMILES -> multi-conformer ETKDGv3+MMFF94s -> select lowest-energy conformer -> save XYZ -> read coordinates
        print(f"SMILES input: {args.smiles}")
        mol_name = _smi_to_filename(args.smiles) or "smiles_mol"
        # Determine the XYZ save path (same directory as --out_png, named by the SMILES-safe filename)
        _png_dir = str(Path(args.out_png).parent) if args.out_png else "."
        _xyz_save = str(Path(_png_dir) / f"{mol_name}.xyz")
        try:
            _rdmol = smiles_to_mol(args.smiles)
            _mol_to_xyz_file(_rdmol, _xyz_save, title=mol_name)
            print(f"  [SMILES] XYZ saved: {_xyz_save}")
            z_list, pos_list = parse_xyz(_xyz_save)
        except Exception as e:
            print(f"[ERROR] SMILES 3D conformer generation failed: {e}")
            sys.exit(1)
    elif args.sdf:
        # SDF / .log .out -> optimized geometry coordinates
        # Auto-detect file type: .log/.out -> Gaussian log; .sdf -> SDF/MOL file
        sdf_path = Path(args.sdf)
        suffix = sdf_path.suffix.lower()

        if suffix in {".log", ".out"}:
            # Gaussian log file - extract the final optimized coordinates (default: last converged step)
            print(f"Gaussian log input: {args.sdf}")
            try:
                z_list, pos_list = parse_gaussian_log_geometry(
                    args.sdf,
                    conf_index=args.sdf_conf_index if args.sdf_conf_index >= 0 else -1
                )
            except Exception as e:
                print(f"[ERROR] Gaussian coordinate extraction failed: {e}")
                sys.exit(1)
        else:
            # SDF / MOL file
            print(f"SDF input: {args.sdf}  (conf_index={args.sdf_conf_index})")
            try:
                z_list, pos_list = parse_sdf(args.sdf, conf_index=args.sdf_conf_index)
            except Exception as e:
                print(f"[ERROR] SDF parsing failed: {e}")
                sys.exit(1)

        mol_name = sdf_path.stem
        if args.sdf_conf_index > 0:
            mol_name = f"{mol_name}_c{args.sdf_conf_index}"
    else:
        z_list, pos_list = parse_xyz(args.xyz)
        mol_name = Path(args.xyz).stem

    z = torch.LongTensor(z_list)
    pos = torch.FloatTensor(pos_list)
    print(f"Molecule: {mol_name}, {len(z_list)} atoms, Z = {z_list}")

# Validate weight paths
    required = {"hii_ckpt": args.hii_ckpt, "hij_ckpt": args.hij_ckpt,
                "dd_ckpt": args.dd_ckpt, "dp_ckpt": args.dp_ckpt}
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"[ERROR] Weight paths not specified (ENK on): {missing}")
        print("  Specify the four weights via --hii_ckpt / --hij_ckpt / --dd_ckpt / --dp_ckpt.")
        sys.exit(1)

    required_off = {"hii_off_ckpt": args.hii_off_ckpt, "hij_off_ckpt": args.hij_off_ckpt,
                    "dd_off_ckpt": args.dd_off_ckpt, "dp_off_ckpt": args.dp_off_ckpt}
    missing_off = [k for k, v in required_off.items() if not v]
    if missing_off:
        print(f"[ERROR] Weight paths not specified (ENK off): {missing_off}")
        print("  Specify the four weights via --hii_off_ckpt / --hij_off_ckpt / --dd_off_ckpt / --dp_off_ckpt.")
        sys.exit(1)

# Load the four V2 models (ENK on)
    print(f"\nLoading Hii model [ENK on] (mode={args.hii_mode}): {args.hii_ckpt}")
    model_hii = _load_multitask("hii", args.hii_ckpt, args.hii_mode, device, args)

    print(f"Loading Hij model [ENK on] (mode={args.hij_mode}): {args.hij_ckpt}")
    model_hij = _load_multitask("hij", args.hij_ckpt, args.hij_mode, device, args)

    print(f"Loading dedipole model [ENK on] (mode={args.dd_mode}): {args.dd_ckpt}")
    model_dd = _load_multitask("dedipole", args.dd_ckpt, args.dd_mode, device, args)

    print(f"Loading depolar model [ENK on] (mode={args.dp_mode}): {args.dp_ckpt}")
    model_dp = _load_depolar(args.dp_ckpt, args.dp_mode, device, args)

# Load the four V2 models (ENK off, pure V2 with ENK force-disabled)
    print(f"Loading Hii model [ENK off] (mode=equiformer_v2): {args.hii_off_ckpt}")
    model_hii_off = _load_multitask("hii", args.hii_off_ckpt, "equiformer_v2", device, args, force_disable_enk=True)

    print(f"Loading Hij model [ENK off] (mode=equiformer_v2): {args.hij_off_ckpt}")
    model_hij_off = _load_multitask("hij", args.hij_off_ckpt, "equiformer_v2", device, args, force_disable_enk=True)

    print(f"Loading dedipole model [ENK off] (mode=equiformer_v2): {args.dd_off_ckpt}")
    model_dd_off = _load_multitask("dedipole", args.dd_off_ckpt, "equiformer_v2", device, args, force_disable_enk=True)

    print(f"Loading depolar model [ENK off] (mode=equiformer_v2): {args.dp_off_ckpt}")
    model_dp_off = _load_depolar(args.dp_off_ckpt, "equiformer_v2", device, args, force_disable_enk=True)

# SENK-EP depolar (third comparison curve, independent of ENK-on/off)
    model_dp_ep = None
    if args.dp_ep_ckpt:
        print(f"\nLoading depolar model [SENK-EP] (mode={args.dp_ep_mode}): {args.dp_ep_ckpt}")
        model_dp_ep = _load_depolar(args.dp_ep_ckpt, args.dp_ep_mode, device, args)
        _ep_active = _is_ep_runtime_active(model_dp_ep)
        print(f"  -> SENK-EP runtime active: {_ep_active}")
    else:
        print("\n  [SENK-EP] --dp_ep_ckpt not specified; skipping EP depolar.")

# SENK-EP dedipole
    model_dd_ep = None
    if args.dd_ep_ckpt:
        print(f"\nLoading dedipole model [SENK-EP] (mode={args.dd_ep_mode}): {args.dd_ep_ckpt}")
        model_dd_ep = _load_multitask("dedipole", args.dd_ep_ckpt, args.dd_ep_mode, device, args)
        _ep_active_dd = _is_ep_runtime_active(model_dd_ep)
        print(f"  -> SENK-EP dd runtime active: {_ep_active_dd}")
    else:
        print("\n  [SENK-EP] --dd_ep_ckpt not specified; dedipole uses the ENK-on weights.")

# SENK-EP hij
    model_hij_ep = None
    if args.hij_ep_ckpt:
        print(f"\nLoading Hij model [SENK-EP] (mode={args.hij_ep_mode}): {args.hij_ep_ckpt}")
        model_hij_ep = _load_multitask("hij", args.hij_ep_ckpt, args.hij_ep_mode, device, args)
        _ep_active_hij = _is_ep_runtime_active(model_hij_ep)
        print(f"  -> SENK-EP hij runtime active: {_ep_active_hij}")
    else:
        print("\n  [SENK-EP] --hij_ep_ckpt not specified; hij uses the ENK-on weights.")

# NBO-GSC calibrator
    nbo_calibrator = None
    _gsc_enabled_interactive = bool(getattr(args, "nbo_gsc_enabled", False))
    _gsc_branches_interactive = set(b.strip() for b in (getattr(args, "nbo_gsc_branches", "") or "hij,dd,dp").split(",") if b.strip()) if _gsc_enabled_interactive else set()
    _any_ep_main = any(
        _is_ep_runtime_active(m) for m in [model_hii, model_hij, model_dd, model_dp]
    )
    _train_stats_interactive = None
    if _any_ep_main and _gsc_branches_interactive and getattr(args, "nbo_train_stats", ""):
        _train_stats_interactive = torch.load(args.nbo_train_stats, weights_only=True, map_location="cpu")
        print(f"NBO-GSC training stats loaded: {args.nbo_train_stats}")
    if _any_ep_main and _gsc_branches_interactive:
        nbo_calibrator = _make_inference_calibrator(args,
            alpha_hij=float(getattr(args, "nbo_gsc_alpha_hij", 0.25)),
            alpha_dd=float(getattr(args, "nbo_gsc_alpha_dd", 0.0)),
            alpha_dp=float(getattr(args, "nbo_gsc_alpha_dp", 0.20)),
            bond_order_exponent=float(getattr(args, "nbo_gsc_bond_exp", 0.6)),
            clamp_ratio=float(getattr(args, "nbo_gsc_clamp", 0.5)),
            bond_dipole_factor=float(getattr(args, "nbo_gsc_bond_dipole_factor", 0.20)),
            alpha_bond=float(getattr(args, "nbo_gsc_alpha_bond", 0.12)),
            dd_sum_rule_strength=float(getattr(args, "nbo_gsc_dd_sum_rule", 0.0)),
            hij_target_xh_only=bool(getattr(args, "nbo_gsc_hij_target_xh_only", False)),
            hij_non_target_scale=float(getattr(args, "nbo_gsc_hij_non_target_scale", 0.15)),
            hij_amide_scale=float(getattr(args, "nbo_gsc_hij_amide_scale", 0.15)),
            hij_consensus_id_threshold=float(getattr(args, "nbo_gsc_hij_consensus_id", 0.85)),
            hij_consensus_ood_threshold=float(getattr(args, "nbo_gsc_hij_consensus_ood", 0.45)),
            hij_enk_ood_weight=float(getattr(args, "nbo_gsc_hij_enk_weight", 0.5)),
            hij_mode_aware=bool(getattr(args, "nbo_gsc_hij_mode_aware", True)),
            hij_xh_freq_min=float(getattr(args, "nbo_gsc_hij_xh_freq_min", 2200.0)),
            hij_xh_freq_max=float(getattr(args, "nbo_gsc_hij_xh_freq_max", 4200.0)),
            hij_mode_participation_threshold=float(getattr(args, "nbo_gsc_hij_mode_participation", 0.015)),
            hij_hii_comp_strength=float(getattr(args, "nbo_gsc_hij_hii_comp", 1.0)),
            hij_gate_mode=str(getattr(args, "nbo_gsc_hij_gate_mode", "adaptive")),
            hij_train_elements=getattr(args, "nbo_gsc_hij_train_elements", "1,6,7,8,9"),
            hij_train_bond_classes=getattr(args, "nbo_gsc_hij_train_bond_classes", ""),
            hij_xh_scale=float(getattr(args, "nbo_gsc_hij_xh_scale", 1.0)),
            hij_unknown_scale=float(getattr(args, "nbo_gsc_hij_unknown_scale", 1.0)),
            hij_unknown_ood_boost=float(getattr(args, "nbo_gsc_hij_unknown_ood_boost", 0.35)),
            hij_strong_ood_threshold=float(getattr(args, "nbo_gsc_hij_strong_ood", 0.70)),
            hij_amide_freq_min=float(getattr(args, "nbo_gsc_hij_amide_freq_min", 1450.0)),
            hij_amide_freq_max=float(getattr(args, "nbo_gsc_hij_amide_freq_max", 1750.0)),
            hij_amide_mode_floor=float(getattr(args, "nbo_gsc_hij_amide_mode_floor", 0.20)),
            hij_global_scale=float(getattr(args, "nbo_gsc_hij_global_scale", 0.5)),
            hij_backbone_mode_floor=float(getattr(args, "nbo_gsc_hij_backbone_mode_floor", 0.08)),
            hij_ood_global_scale=float(getattr(args, "nbo_gsc_hij_ood_global_scale", 1.0)),
            hij_env_ood_hops=int(getattr(args, "nbo_gsc_hij_env_ood_hops", 3)),
            hij_env_ood_decay=float(getattr(args, "nbo_gsc_hij_env_ood_decay", 0.6)),
            hij_env_ood_boost=float(getattr(args, "nbo_gsc_hij_env_ood_boost", 0.40)),
            hij_unknown_mode_floor=float(getattr(args, "nbo_gsc_hij_unknown_mode_floor", 0.55)),
            hij_hbond_distance=float(getattr(args, "nbo_gsc_hij_hbond_distance", 2.45)),
            hij_hbond_acceptors=getattr(args, "nbo_gsc_hij_hbond_acceptors", "7,8,9,15,16,17"),
            hij_hbond_ood_boost=float(getattr(args, "nbo_gsc_hij_hbond_ood_boost", 0.45)),
            hij_hbond_scale=float(getattr(args, "nbo_gsc_hij_hbond_scale", 1.0)),
            hij_hbond_mode_floor=float(getattr(args, "nbo_gsc_hij_hbond_mode_floor", 0.55)),
            hij_ood_alpha_scale=float(getattr(args, "nbo_gsc_hij_ood_alpha_scale", 1.5)),
            hij_hbond_alpha_scale=float(getattr(args, "nbo_gsc_hij_hbond_alpha_scale", 1.6)),
            hij_policy=str(getattr(args, "nbo_gsc_hij_policy", "hybrid")),
            hij_xh_heuristic_scale=float(getattr(args, "nbo_gsc_hij_xh_heuristic_scale", 0.25)),
            hij_interaction_soften_scale=float(getattr(args, "nbo_gsc_hij_interaction_soften_scale", 1.0)),
            hij_interaction_alpha_scale=float(getattr(args, "nbo_gsc_hij_interaction_alpha_scale", 1.0)),
            hij_interaction_geometry_weight=float(getattr(args, "nbo_gsc_hij_interaction_geometry_weight", 0.35)),
            hij_interaction_min_score=float(getattr(args, "nbo_gsc_hij_interaction_min_score", 0.15)),
            hij_hbond_angle_min=float(getattr(args, "nbo_gsc_hij_hbond_angle_min", 115.0)),
            hij_interaction_e2_low_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_low_quantile", 0.75)),
            hij_interaction_e2_high_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_high_quantile", 0.95)),
            hij_interaction_acceptor_distance=float(getattr(args, "nbo_gsc_hij_interaction_acceptor_distance", 3.20)),
            hij_interaction_angle_min=float(getattr(args, "nbo_gsc_hij_interaction_angle_min", 85.0)),
            hij_interaction_field_weight=float(getattr(args, "nbo_gsc_hij_interaction_field_weight", 0.75)),
            hij_physical_rules=not bool(getattr(args, "nbo_gsc_hij_disable_physical_rules", False)),
            hij_physical_min_score=float(getattr(args, "nbo_gsc_hij_physical_min_score", 0.20)),
            hij_physical_soften_scale=float(getattr(args, "nbo_gsc_hij_physical_soften_scale", 0.65)),
            hij_physical_harden_scale=float(getattr(args, "nbo_gsc_hij_physical_harden_scale", 0.40)),
            hij_physical_alpha_scale=float(getattr(args, "nbo_gsc_hij_physical_alpha_scale", 1.35)),
            hij_stark_gate_scale=float(getattr(args, "nbo_gsc_hij_stark_gate_scale", 0.35)),
            dd_trace_mode=str(getattr(args, "nbo_gsc_dd_trace_mode", "off")),
            dd_bond_gate_mode=str(getattr(args, "nbo_gsc_dd_bond_gate_mode", "direct")),
            dd_interaction_gate_scale=float(getattr(args, "nbo_gsc_dd_interaction_gate_scale", 0.0)),
            dp_gate_mode=str(getattr(args, "nbo_gsc_dp_gate_mode", "legacy")),
            dp_interaction_gate_scale=float(getattr(args, "nbo_gsc_dp_interaction_gate_scale", 0.35)),
            dp_interaction_boost=float(getattr(args, "nbo_gsc_dp_interaction_boost", 0.20)),
            bond_occ_col_policy=str(getattr(args, "nbo_gsc_bond_occ_col_policy", "legacy")),
            train_stats=_train_stats_interactive,
        )
        print(f"NBO-GSC calibrator activated: alpha_hij={nbo_calibrator.alpha_hij}"
              f" alpha_dd={nbo_calibrator.alpha_dd} alpha_dp={nbo_calibrator.alpha_dp}"
              f" branches={','.join(sorted(_gsc_branches_interactive))}")
    else:
        print("NBO-GSC calibrator not activated" + (" (EP not enabled)" if not _any_ep_main else ""))

# --- Independent NBO-GSC calibrator for SENK-EP mode ---
    nbo_calibrator_ep = None
    _ep_active_for_gsc = (
        (model_dp_ep is not None and _is_ep_runtime_active(model_dp_ep))
        or (model_dd_ep is not None and _is_ep_runtime_active(model_dd_ep))
        or (model_hij_ep is not None and _is_ep_runtime_active(model_hij_ep))
    )
    if _ep_active_for_gsc and _gsc_branches_interactive:
        _train_stats_ep = None
        if getattr(args, "nbo_train_stats", ""):
            _train_stats_ep = torch.load(args.nbo_train_stats, weights_only=True, map_location="cpu")
        nbo_calibrator_ep = _make_inference_calibrator(args,
            alpha_hij=float(getattr(args, "nbo_gsc_alpha_hij", 0.25)),
            alpha_dd=float(getattr(args, "nbo_gsc_alpha_dd", 0.0)),
            alpha_dp=float(getattr(args, "nbo_gsc_alpha_dp", 0.20)),
            bond_order_exponent=float(getattr(args, "nbo_gsc_bond_exp", 0.6)),
            clamp_ratio=float(getattr(args, "nbo_gsc_clamp", 0.5)),
            bond_dipole_factor=float(getattr(args, "nbo_gsc_bond_dipole_factor", 0.20)),
            alpha_bond=float(getattr(args, "nbo_gsc_alpha_bond", 0.12)),
            dd_sum_rule_strength=float(getattr(args, "nbo_gsc_dd_sum_rule", 0.0)),
            hij_target_xh_only=bool(getattr(args, "nbo_gsc_hij_target_xh_only", False)),
            hij_non_target_scale=float(getattr(args, "nbo_gsc_hij_non_target_scale", 0.15)),
            hij_amide_scale=float(getattr(args, "nbo_gsc_hij_amide_scale", 0.15)),
            hij_consensus_id_threshold=float(getattr(args, "nbo_gsc_hij_consensus_id", 0.85)),
            hij_consensus_ood_threshold=float(getattr(args, "nbo_gsc_hij_consensus_ood", 0.45)),
            hij_enk_ood_weight=float(getattr(args, "nbo_gsc_hij_enk_weight", 0.5)),
            hij_mode_aware=bool(getattr(args, "nbo_gsc_hij_mode_aware", True)),
            hij_xh_freq_min=float(getattr(args, "nbo_gsc_hij_xh_freq_min", 2200.0)),
            hij_xh_freq_max=float(getattr(args, "nbo_gsc_hij_xh_freq_max", 4200.0)),
            hij_mode_participation_threshold=float(getattr(args, "nbo_gsc_hij_mode_participation", 0.015)),
            hij_hii_comp_strength=float(getattr(args, "nbo_gsc_hij_hii_comp", 1.0)),
            hij_gate_mode=str(getattr(args, "nbo_gsc_hij_gate_mode", "adaptive")),
            hij_train_elements=getattr(args, "nbo_gsc_hij_train_elements", "1,6,7,8,9"),
            hij_train_bond_classes=getattr(args, "nbo_gsc_hij_train_bond_classes", ""),
            hij_xh_scale=float(getattr(args, "nbo_gsc_hij_xh_scale", 1.0)),
            hij_unknown_scale=float(getattr(args, "nbo_gsc_hij_unknown_scale", 1.0)),
            hij_unknown_ood_boost=float(getattr(args, "nbo_gsc_hij_unknown_ood_boost", 0.35)),
            hij_strong_ood_threshold=float(getattr(args, "nbo_gsc_hij_strong_ood", 0.70)),
            hij_amide_freq_min=float(getattr(args, "nbo_gsc_hij_amide_freq_min", 1450.0)),
            hij_amide_freq_max=float(getattr(args, "nbo_gsc_hij_amide_freq_max", 1750.0)),
            hij_amide_mode_floor=float(getattr(args, "nbo_gsc_hij_amide_mode_floor", 0.20)),
            hij_global_scale=float(getattr(args, "nbo_gsc_hij_global_scale", 0.5)),
            hij_backbone_mode_floor=float(getattr(args, "nbo_gsc_hij_backbone_mode_floor", 0.08)),
            hij_ood_global_scale=float(getattr(args, "nbo_gsc_hij_ood_global_scale", 1.0)),
            hij_env_ood_hops=int(getattr(args, "nbo_gsc_hij_env_ood_hops", 3)),
            hij_env_ood_decay=float(getattr(args, "nbo_gsc_hij_env_ood_decay", 0.6)),
            hij_env_ood_boost=float(getattr(args, "nbo_gsc_hij_env_ood_boost", 0.40)),
            hij_unknown_mode_floor=float(getattr(args, "nbo_gsc_hij_unknown_mode_floor", 0.55)),
            hij_hbond_distance=float(getattr(args, "nbo_gsc_hij_hbond_distance", 2.45)),
            hij_hbond_acceptors=getattr(args, "nbo_gsc_hij_hbond_acceptors", "7,8,9,15,16,17"),
            hij_hbond_ood_boost=float(getattr(args, "nbo_gsc_hij_hbond_ood_boost", 0.45)),
            hij_hbond_scale=float(getattr(args, "nbo_gsc_hij_hbond_scale", 1.0)),
            hij_hbond_mode_floor=float(getattr(args, "nbo_gsc_hij_hbond_mode_floor", 0.55)),
            hij_ood_alpha_scale=float(getattr(args, "nbo_gsc_hij_ood_alpha_scale", 1.5)),
            hij_hbond_alpha_scale=float(getattr(args, "nbo_gsc_hij_hbond_alpha_scale", 1.6)),
            hij_policy=str(getattr(args, "nbo_gsc_hij_policy", "hybrid")),
            hij_xh_heuristic_scale=float(getattr(args, "nbo_gsc_hij_xh_heuristic_scale", 0.25)),
            hij_interaction_soften_scale=float(getattr(args, "nbo_gsc_hij_interaction_soften_scale", 1.0)),
            hij_interaction_alpha_scale=float(getattr(args, "nbo_gsc_hij_interaction_alpha_scale", 1.0)),
            hij_interaction_geometry_weight=float(getattr(args, "nbo_gsc_hij_interaction_geometry_weight", 0.35)),
            hij_interaction_min_score=float(getattr(args, "nbo_gsc_hij_interaction_min_score", 0.15)),
            hij_hbond_angle_min=float(getattr(args, "nbo_gsc_hij_hbond_angle_min", 115.0)),
            hij_interaction_e2_low_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_low_quantile", 0.75)),
            hij_interaction_e2_high_quantile=float(getattr(args, "nbo_gsc_hij_interaction_e2_high_quantile", 0.95)),
            hij_interaction_acceptor_distance=float(getattr(args, "nbo_gsc_hij_interaction_acceptor_distance", 3.20)),
            hij_interaction_angle_min=float(getattr(args, "nbo_gsc_hij_interaction_angle_min", 85.0)),
            hij_interaction_field_weight=float(getattr(args, "nbo_gsc_hij_interaction_field_weight", 0.75)),
            hij_physical_rules=not bool(getattr(args, "nbo_gsc_hij_disable_physical_rules", False)),
            hij_physical_min_score=float(getattr(args, "nbo_gsc_hij_physical_min_score", 0.20)),
            hij_physical_soften_scale=float(getattr(args, "nbo_gsc_hij_physical_soften_scale", 0.65)),
            hij_physical_harden_scale=float(getattr(args, "nbo_gsc_hij_physical_harden_scale", 0.40)),
            hij_physical_alpha_scale=float(getattr(args, "nbo_gsc_hij_physical_alpha_scale", 1.35)),
            hij_stark_gate_scale=float(getattr(args, "nbo_gsc_hij_stark_gate_scale", 0.35)),
            dd_trace_mode=str(getattr(args, "nbo_gsc_dd_trace_mode", "off")),
            dd_bond_gate_mode=str(getattr(args, "nbo_gsc_dd_bond_gate_mode", "direct")),
            dd_interaction_gate_scale=float(getattr(args, "nbo_gsc_dd_interaction_gate_scale", 0.0)),
            dp_gate_mode=str(getattr(args, "nbo_gsc_dp_gate_mode", "legacy")),
            dp_interaction_gate_scale=float(getattr(args, "nbo_gsc_dp_interaction_gate_scale", 0.35)),
            dp_interaction_boost=float(getattr(args, "nbo_gsc_dp_interaction_boost", 0.20)),
            bond_occ_col_policy=str(getattr(args, "nbo_gsc_bond_occ_col_policy", "legacy")),
            train_stats=_train_stats_ep,
        )
        print(f"NBO-GSC calibrator (SENK-EP) activated: alpha_hij={nbo_calibrator_ep.alpha_hij}"
              f" alpha_dd={nbo_calibrator_ep.alpha_dd} alpha_dp={nbo_calibrator_ep.alpha_dp}"
              f" branches={','.join(sorted(_gsc_branches_interactive))}")

# --- Spectrum generation ---
    _lp_ckpt = None if args.disable_lp else args.lp_ckpt
    print("\nStarting V2 spectrum computation ...")
    results = generate_spectrum(
        pos=pos, z=z,
        model_hii=model_hii, model_hij=model_hij,
        model_dd=model_dd, model_dp=model_dp,
        device=device,
        linear=args.linear,
        scale=args.scale,
        sigma=args.sigma,
        sigma_ir=sigma_ir,
        sigma_raman=sigma_raman,
        freq_scale_factor=args.freq_scale_factor,
        freq_range=(args.freq_min, args.freq_max),
        freq_points=args.freq_points,
        radius=args.radius,
        lp_ckpt=_lp_ckpt,
        skeleton_radius=args.skeleton_radius,
        nbo_calibrator=nbo_calibrator,
        gsc_branches=_gsc_branches_interactive,
    )

    print("\nStarting ENK-off spectrum computation ...")
    results_off = generate_spectrum(
        pos=pos, z=z,
        model_hii=model_hii_off, model_hij=model_hij_off,
        model_dd=model_dd_off, model_dp=model_dp_off,
        device=device,
        linear=args.linear,
        scale=args.scale,
        sigma=args.sigma,
        sigma_ir=sigma_ir,
        sigma_raman=sigma_raman,
        freq_scale_factor=args.freq_scale_factor,
        freq_range=(args.freq_min, args.freq_max),
        freq_points=args.freq_points,
        radius=args.radius,
        lp_ckpt=_lp_ckpt,
        skeleton_radius=args.skeleton_radius,
        nbo_calibrator=nbo_calibrator,
        gsc_branches=_gsc_branches_interactive,
    )

# --- SENK-EP spectrum ---
    results_ep = None
    _has_any_ep = (model_dp_ep is not None or model_dd_ep is not None
                   or model_hij_ep is not None)
    if _has_any_ep:
        _ep_hij = model_hij_ep if model_hij_ep is not None else model_hij
        _ep_dd = model_dd_ep if model_dd_ep is not None else model_dd
        _ep_dp = model_dp_ep if model_dp_ep is not None else model_dp
        print("\nStarting SENK-EP spectrum computation ...")
        results_ep = generate_spectrum(
            pos=pos, z=z,
            model_hii=model_hii, model_hij=_ep_hij,
            model_dd=_ep_dd, model_dp=_ep_dp,
            device=device,
            linear=args.linear,
            scale=args.scale,
            sigma=args.sigma,
            sigma_ir=sigma_ir,
            sigma_raman=sigma_raman,
            freq_scale_factor=args.freq_scale_factor,
            freq_range=(args.freq_min, args.freq_max),
            freq_points=args.freq_points,
            radius=args.radius,
            lp_ckpt=_lp_ckpt,
            skeleton_radius=args.skeleton_radius,
            nbo_calibrator=nbo_calibrator_ep,
            gsc_branches=_gsc_branches_interactive,
        )

# --- Mode summary ---
    print(f"\nFour-weight mode configuration:")
    print(f"  hii     : {args.hii_mode}")
    print(f"  hij     : {args.hij_mode}")
    print(f"  dedipole: {args.dd_mode}")
    print(f"  depolar : {args.dp_mode}")
    print("  --- ENK off ---")
    print("  hii     : equiformer_v2 (ENK forced off)")
    print("  hij     : equiformer_v2 (ENK forced off)")
    print("  dedipole: equiformer_v2 (ENK forced off)")
    print("  depolar : equiformer_v2 (ENK forced off)")

# --- Print peak table ---
    print(f"\n{'Freq (cm^-1)':>14s}  {'IR intensity':>14s}  {'Raman activity':>14s}")
    print("-" * 48)
    for f, ir_v, ram_v in zip(results["freq"].tolist(),
                               results["ir_int"].tolist(),
                               results["raman_act"].tolist()):
        if float(f) > 0:
            print(f"{f:14.2f}  {ir_v:14.4e}  {ram_v:14.4e}")

# --- Optional: DetaNet comparison ---
    det_results = None
    if args.compare_detanet:
        max_z = max(z_list)
        det_max = max(args.max_atomic_number, max_z + 1)
        print(f"\nLoading DetaNet comparison model (max_atomic_number={det_max}) ...")
        det_hi  = _build_detanet("Hi",       device, Path(args.detanet_Hi),       det_max)
        det_hij = _build_detanet("Hij",       device, Path(args.detanet_Hij),      det_max)
        det_dd  = _build_detanet("dedipole",  device, Path(args.detanet_dedipole), det_max)
        det_dp  = _build_detanet("depolar",   device, Path(args.detanet_depolar),  det_max)
        print("DetaNet spectrum computation ...")
        det_results = generate_detanet_spectrum(
            pos=pos, z=z,
            model_Hi=det_hi, model_Hij=det_hij,
            model_dd=det_dd, model_dp=det_dp,
            device=device,
            linear=args.linear, scale=args.scale, sigma=args.sigma,
            sigma_ir=sigma_ir, sigma_raman=sigma_raman,
            freq_scale_factor=args.freq_scale_factor,
            freq_range=(args.freq_min, args.freq_max),
            freq_points=args.freq_points,
        )
        print(f"\n{'Freq (cm^-1)':>14s}  {'IR(V2)':>14s}  {'IR(DetaNet)':>14s}"
              f"  {'Raman(V2)':>14s}  {'Raman(DetaNet)':>14s}")
        print("-" * 74)
        for f, ir_v2, ir_det, ram_v2, ram_det in zip(
            results["freq"].tolist(), results["ir_int"].tolist(),
            det_results["ir_int"].tolist(),
            results["raman_act"].tolist(), det_results["raman_act"].tolist()
        ):
            if float(f) > 0:
                print(f"{f:14.2f}  {ir_v2:14.4e}  {ir_det:14.4e}"
                      f"  {ram_v2:14.4e}  {ram_det:14.4e}")

# --- Save npz ---
    if args.out:
        save_dict = {k: _spectrum_archive_value(v) for k, v in results.items()}
        if results_ep is not None:
            for k, v in results_ep.items():
                save_dict[f"ep_{k}"] = _spectrum_archive_value(v)
        if det_results is not None:
            for k, v in det_results.items():
                save_dict[f"det_{k}"] = v.numpy()
        np.savez(args.out, **save_dict)
        print(f"\nSpectrum data saved: {args.out}")

# --- Load DFT reference spectrum (optional) ---
    ref_ir_data, ref_raman_data = None, None
    if args.ref_ir:
        try:
            ref_ir_data = _parse_ref_spectrum(str(args.ref_ir), kind="ir", txt_mode=args.ref_txt_mode)
            print(
                f"Reference IR loaded : {args.ref_ir}"
                f"  source={ref_ir_data['source']}  mode={ref_ir_data['mode']}"
                f"  ({len(ref_ir_data['x'])} pts)"
            )
            if ref_ir_data["mode"] == "curve":
                print("[WARN] IR reference is an already-broadened curve. For strict apples-to-apples comparison, prefer Gaussian Peak information or .log/.out raw modes.")
        except Exception as _e:
            print(f"[WARN] Failed to load ref IR: {_e}")
    if args.ref_raman:
        try:
            ref_raman_data = _parse_ref_spectrum(str(args.ref_raman), kind="raman", txt_mode=args.ref_txt_mode)
            print(
                f"Reference Raman loaded: {args.ref_raman}"
                f"  source={ref_raman_data['source']}  mode={ref_raman_data['mode']}"
                f"  ({len(ref_raman_data['x'])} pts)"
            )
            if ref_raman_data["mode"] == "curve":
                print("[WARN] Raman reference is an already-broadened curve. For strict apples-to-apples comparison, prefer Gaussian Peak information or .log/.out raw modes.")
        except Exception as _e:
            print(f"[WARN] Failed to load ref Raman: {_e}")

# --- Frequency diagnostics printout (when reference stick data available) ---
    if ref_ir_data is not None and ref_ir_data.get("mode") == "sticks":
        pred_freq = results["freq"].numpy()
        pred_ir   = results["ir_int"].numpy()
        ref_x  = ref_ir_data["x"] * ref_freq_scale   # scaled reference
        ref_y  = ref_ir_data["y"]
        print(f"\n[DIAG] ref_freq_scale={ref_freq_scale:.4f}  "
              f"(raw DFT peaks × {ref_freq_scale:.4f} → 'DFT×scale' space)")
        print(f"  Reference (DFT×scale) top-5 IR peaks by intensity:")
        top5 = np.argsort(ref_y)[::-1][:5]
        for i in top5:
            print(f"    {ref_x[i]:8.2f} cm⁻¹  inten={ref_y[i]:.4e}")
        print(f"  Prediction top-5 IR peaks by intensity (scale={args.scale}):")
        valid_p = pred_freq > 0
        pf, pi_ = pred_freq[valid_p], pred_ir[valid_p]
        top5p = np.argsort(pi_)[::-1][:5]
        for i in top5p:
            print(f"    {pf[i]:8.2f} cm⁻¹  inten={pi_[i]:.4e}")

# --- Plotting ---
    _plot_spectrum(results, title_suffix=f"({mol_name})",
                  senk_off_results=results_off,
                  ep_results=results_ep,
                  detanet_results=det_results,
                  out_png=args.out_png,
                  ref_ir=ref_ir_data,
                  ref_raman=ref_raman_data,
                  model_label="SENK(ENK on)",
                  senk_off_label="SENK(ENK off)",
                  ep_label="SENK-EP",
                  freq_range=(args.freq_min, args.freq_max),
                  sigma_ir=sigma_ir,
                  sigma_raman=sigma_raman,
                  ref_freq_scale=ref_freq_scale)


if __name__ == "__main__":
    main()
