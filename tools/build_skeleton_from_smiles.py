#!/usr/bin/env python
"""Build a skeleton for a novel molecule from SMILES or SDF.

This script generates the skeleton .pt file required by the NBO electron
prior predictor for inference on novel molecules.
channel for a single molecule (e.g. a drug candidate not in QM9).

Usage
-----
    # From SMILES (e.g. Pomalidomide / Imnovid):
    python tools/build_skeleton_from_smiles.py \
        --smiles "O=C1CCC(N2C(=O)c3cccc(N)c3C2=O)C(=O)N1" \
        --output skeleton_imnovid.pt

    # From SDF file:
    python tools/build_skeleton_from_smiles.py \
        --sdf molecule.sdf \
        --output skeleton_molecule.pt

    # Without lone-pair prediction (atoms + bonds only):
    python tools/build_skeleton_from_smiles.py \
        --smiles "O=C1CCC(N2C(=O)c3cccc(N)c3C2=O)C(=O)N1" \
        --disable-lp \
        --output skeleton_imnovid_nolp.pt

Output
------
A .pt file (dict) containing:
    node_type            : (N_global,) long — 0=Atom, 1=Bond, 2=LP
    pos_global           : (N_global, 3) float — 3D positions
    atom_to_nbo_index    : (2, E) long — atom↔NBO bipartite edges
    interaction_edge_index : (2, E') long — radius graph on all nodes
    atom_bond_index      : (2, B) long — bond pairs (atom indices)
    num_global_nodes     : int
    num_atoms            : int
    num_lps              : int
    num_bonds            : int
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_LP_CKPT = os.path.join(SCRIPT_DIR, "lp_pred_model.ckpt")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Geometry Extraction

def _extract_geometry_from_mol(mol) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Extract atom positions and bond pairs from an RDKit mol object."""
    from rdkit import Chem

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Empty or invalid molecule.")
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no 3D conformer. Use --smiles with embedding or provide an SDF with 3D coords.")

    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    atom_pos = torch.tensor(positions, dtype=torch.float32)

    bond_pairs: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        bond_pairs.append((u, v))

    return atom_pos, bond_pairs


# Radius Graph

def _build_radius_graph(pos: torch.Tensor, r: float = 5.0) -> torch.Tensor:
    """All pairs within distance r, excluding self-loops."""
    if pos.size(0) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    dist = torch.cdist(pos, pos)
    mask = (dist < r) & (dist > 1e-6)
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


# Skeleton Assembly

def build_skeleton_entry(
    atom_pos: torch.Tensor,
    bond_pairs: List[Tuple[int, int]],
    lp_atoms: Optional[List[int]] = None,
    radius: float = 5.0,
) -> Dict:
    """Assemble a skeleton dict from atomic geometry and NBO topology.

    Args:
        atom_pos: (N, 3) atom positions in Angstrom.
        bond_pairs: list of (i, j) bond pairs (atom indices, i < j).
        lp_atoms: list of atom indices that host lone pairs (with repeats
                  for atoms with > 1 LP).
        radius: radius cutoff for interaction_edge_index.

    Returns:
        dict with all fields required by the NBO electron prior channel.
    """
    num_atoms = int(atom_pos.size(0))
    bond_pairs = sorted(set(bond_pairs))
    num_bonds = len(bond_pairs)
    lp_atoms = lp_atoms or []
    num_lps = len(lp_atoms)
    total_nodes = num_atoms + num_lps + num_bonds

    # node_type: 0=Atom, 2=LP, 1=Bond (in layout order)
    node_type = torch.zeros(total_nodes, dtype=torch.long)
    if num_lps:
        node_type[num_atoms: num_atoms + num_lps] = 2  # lone pairs
    if num_bonds:
        node_type[num_atoms + num_lps:] = 1  # bonds

    # LP positions: same as parent atom
    lp_pos = atom_pos[lp_atoms] if num_lps else torch.zeros((0, 3), dtype=torch.float32)

    # Bond positions: midpoint of two bonded atoms
    if num_bonds:
        atoms_a = torch.tensor([a for a, _ in bond_pairs], dtype=torch.long)
        atoms_b = torch.tensor([b for _, b in bond_pairs], dtype=torch.long)
        bond_pos = 0.5 * (atom_pos[atoms_a] + atom_pos[atoms_b])
    else:
        bond_pos = torch.zeros((0, 3), dtype=torch.float32)

    pos_global = torch.cat([atom_pos, lp_pos, bond_pos], dim=0)

    # atom_to_nbo_index: bipartite edges atom ↔ NBO nodes
    atom_to_nbo_edges: List[Tuple[int, int]] = []
    lp_offset = num_atoms
    bond_offset = num_atoms + num_lps
    for lp_idx, parent_atom in enumerate(lp_atoms):
        atom_to_nbo_edges.append((int(parent_atom), lp_offset + lp_idx))
    for bond_idx, (a, b) in enumerate(bond_pairs):
        global_idx = bond_offset + bond_idx
        atom_to_nbo_edges.append((a, global_idx))
        atom_to_nbo_edges.append((b, global_idx))
    if atom_to_nbo_edges:
        atom_to_nbo_index = torch.tensor(atom_to_nbo_edges, dtype=torch.long).t().contiguous()
    else:
        atom_to_nbo_index = torch.zeros((2, 0), dtype=torch.long)

    # atom_bond_index
    if bond_pairs:
        atom_bond_index = torch.tensor(bond_pairs, dtype=torch.long).t().contiguous()
    else:
        atom_bond_index = torch.zeros((2, 0), dtype=torch.long)

    # interaction_edge_index: radius graph over all global nodes
    interaction_edge_index = _build_radius_graph(pos_global, r=radius)

    entry = {
        "node_type": node_type,
        "atom_to_nbo_index": atom_to_nbo_index,
        "interaction_edge_index": interaction_edge_index,
        "atom_bond_index": atom_bond_index,
        "pos_global": pos_global,
        "num_global_nodes": total_nodes,
        "num_atoms": num_atoms,
        "num_lps": num_lps,
        "num_bonds": num_bonds,
    }
    return entry


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Build skeleton for a novel molecule from SMILES or SDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", type=str, help="SMILES string for the molecule")
    group.add_argument("--sdf", type=str, help="Path to an SDF file (first molecule is used)")

    parser.add_argument("--output", "-o", type=str, default="skeleton.pt",
                        help="Output .pt file path")
    parser.add_argument("--disable-lp", action="store_true",
                        help="Skip lone-pair prediction (atoms + bonds only)")
    parser.add_argument("--lp-ckpt", type=str, default=DEFAULT_LP_CKPT,
                        help="Path to SIMG lone-pair predictor checkpoint")
    parser.add_argument("--lp-device", type=str, default="cpu",
                        help="Device for LP predictor")
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Radius cutoff (Å) for interaction_edge_index")
    parser.add_argument("--embed-seed", type=int, default=42,
                        help="RDKit embedding seed (SMILES only)")
    parser.add_argument("--optimize", type=str, choices=["mmff", "uff", "none"], default="mmff",
                        help="Force-field optimization for initial geometry (SMILES only)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

# Load molecule
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    if args.smiles:
        mol = Chem.MolFromSmiles(args.smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {args.smiles}")
        mol = Chem.AddHs(mol)

        # 3D embedding
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = args.embed_seed
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            # Fallback: try without ETKDG
            status = AllChem.EmbedMolecule(mol, randomSeed=args.embed_seed)
        if status != 0:
            raise RuntimeError("Failed to embed molecule in 3D. Try providing an SDF instead.")

        # Optional geometry optimization
        if args.optimize == "mmff":
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                print("[warn] MMFF optimization failed; using unoptimized geometry")
        elif args.optimize == "uff":
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                print("[warn] UFF optimization failed; using unoptimized geometry")

        print(f"[mol] SMILES: {args.smiles}")
    else:
        suppl = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)
        mol = next(iter(suppl))
        if mol is None:
            raise ValueError(f"Failed to load molecule from SDF: {args.sdf}")
        # Ensure hydrogens
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
        print(f"[mol] SDF: {args.sdf}")

    n_atoms = mol.GetNumAtoms()
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    print(f"  formula: {formula}, atoms: {n_atoms}")

# Extract geometry
    atom_pos, bond_pairs = _extract_geometry_from_mol(mol)
    print(f"  bonds: {len(bond_pairs)}")

# Lone-pair prediction
    lp_atoms: List[int] = []
    if not args.disable_lp:
        if not os.path.exists(args.lp_ckpt):
            print(f"[warn] LP checkpoint not found: {args.lp_ckpt}")
            print("  → proceeding without LP prediction (use --disable-lp to suppress this warning)")
        else:
            from tools.lp_predictor import LPPredictor
            lp_pred = LPPredictor(args.lp_ckpt, device=args.lp_device)
            lp_atoms = lp_pred.predict_lp_atom_map(mol)
            print(f"  lone pairs: {len(lp_atoms)} LP nodes from {len(set(lp_atoms))} atoms")

# Build skeleton
    skeleton = build_skeleton_entry(atom_pos, bond_pairs, lp_atoms, radius=args.radius)
    total = skeleton["num_global_nodes"]
    print(f"  skeleton: {total} nodes "
          f"({skeleton['num_atoms']} atoms + {skeleton['num_lps']} LP + {skeleton['num_bonds']} bonds)")
    print(f"  atom_to_nbo edges: {skeleton['atom_to_nbo_index'].size(1)}")
    print(f"  interaction edges: {skeleton['interaction_edge_index'].size(1)}")

# Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    torch.save(skeleton, args.output)
    print(f"\n[saved] {os.path.abspath(args.output)}")

# Optional: also save a human-readable summary
    if args.verbose:
        summary = {
            "formula": formula,
            "smiles": args.smiles or "(from SDF)",
            "num_atoms": int(skeleton["num_atoms"]),
            "num_lps": int(skeleton["num_lps"]),
            "num_bonds": int(skeleton["num_bonds"]),
            "num_global_nodes": int(skeleton["num_global_nodes"]),
            "atom_to_nbo_edges": int(skeleton["atom_to_nbo_index"].size(1)),
            "interaction_edges": int(skeleton["interaction_edge_index"].size(1)),
            "radius": args.radius,
        }
        json_path = args.output.replace(".pt", "_info.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
