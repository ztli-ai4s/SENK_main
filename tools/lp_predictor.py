from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, MessagePassing
    from torch_geometric.nn.conv import PNAConv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "lp_predictor requires torch-geometric to be installed and importable."
    ) from exc

try:
    from rdkit import Chem
except Exception as exc:  # pragma: no cover
    raise ImportError("lp_predictor requires RDKit to be installed and importable.") from exc


# Keep aligned with SIMG's `simg/config.py`.
ATOMS: List[str] = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "As",
    "Br",
    "I",
    "Hg",
    "Bi",
]

LP_N_CLASSES = 5
N_ATOM_FEATURES_LP_MODEL = 16
N_BOND_FEATURES_LP_MODEL = 4


class GNN_LP_Model(nn.Module):
    """Minimal inference-only copy of SIMG's lone-pair predictor backbone.

    The original SIMG project wraps this in a LightningModule. Here we keep a
    plain `nn.Module` so we can load a Lightning checkpoint via `torch.load`
    without requiring pytorch_lightning as a dependency.
    """

    def __init__(self, hidden_size: Iterable[int], deg: torch.Tensor):
        super().__init__()

        layers: List[nn.Module] = []
        last_hs = N_ATOM_FEATURES_LP_MODEL

        aggregators = ["sum", "mean", "max", "min", "std"]
        scalers = ["identity"]

        for hs in hidden_size:
            layers.append(
                PNAConv(last_hs, int(hs), aggregators, scalers, deg=deg, edge_dim=N_BOND_FEATURES_LP_MODEL)
            )
            layers.append(nn.ReLU())
            last_hs = int(hs)

        layers.append(GCNConv(last_hs, N_ATOM_FEATURES_LP_MODEL))
        self.layers = nn.ModuleList(layers)

        self.fcn_head = nn.Sequential(
            nn.Linear(N_ATOM_FEATURES_LP_MODEL * 2, 10),
            nn.ReLU(),
            nn.Linear(10, LP_N_CLASSES * 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                if isinstance(layer, PNAConv):
                    out = layer(out, edge_index, edge_attr)
                else:
                    out = layer(out, edge_index)
            else:
                out = layer(out)

        out = torch.cat((out, x), dim=1)
        out = self.fcn_head(out)
        return out


@dataclass(frozen=True)
class LonePairCounts:
    num_lps: torch.LongTensor  # [num_atoms]
    num_conjugated_lps: torch.LongTensor  # [num_atoms]


def _one_hot_atom(symbol: str) -> List[float]:
    vec = [0.0] * len(ATOMS)
    try:
        vec[ATOMS.index(symbol)] = 1.0
    except ValueError:
        # Unknown atom type for this model. Keep all-zeros to avoid crashing.
        pass
    return vec


def _bond_type_one_hot(bond: Chem.Bond) -> List[float]:
    """Map RDKit bond types to the 4-class OpenBabel-style encoding used in SIMG.

    1: single, 2: double, 3: triple, 4: aromatic
    """

    bt = bond.GetBondType()
    if bond.GetIsAromatic() or bt == Chem.rdchem.BondType.AROMATIC:
        idx = 3
    elif bt == Chem.rdchem.BondType.SINGLE:
        idx = 0
    elif bt == Chem.rdchem.BondType.DOUBLE:
        idx = 1
    elif bt == Chem.rdchem.BondType.TRIPLE:
        idx = 2
    else:
        # Fallback: treat as single.
        idx = 0

    out = [0.0, 0.0, 0.0, 0.0]
    out[idx] = 1.0
    return out


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return dict(state_dict)
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _load_ckpt_state(ckpt_path: str) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning checkpoints store weights under `state_dict`.
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    hyper = {}
    if isinstance(ckpt, dict):
        hyper = ckpt.get("hyper_parameters", {}) or {}

    return hyper, state_dict


def _infer_hidden_size(hyper: Dict) -> List[int]:
    config = hyper.get("config") if isinstance(hyper.get("config"), dict) else None
    if config and isinstance(config.get("hidden_size"), (list, tuple)):
        return [int(x) for x in config["hidden_size"]]

    # Conservative fallback; only used if checkpoint lacks hyperparameters.
    return [64, 64, 64]


def _infer_deg(hyper: Dict) -> torch.Tensor:
    deg = hyper.get("deg")
    if isinstance(deg, torch.Tensor):
        return deg.to(torch.long)
    if isinstance(deg, (list, tuple)):
        return torch.tensor(list(deg), dtype=torch.long)

    # With `scalers=["identity"]`, deg has minimal effect, but PNAConv still
    # requires a non-empty histogram tensor.
    return torch.ones(8, dtype=torch.long)


class LPPredictor:
    """Loads SIMG LP checkpoint and predicts per-atom lone pair counts."""

    def __init__(self, ckpt_path: str, device: Optional[str] = None):
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"LP checkpoint not found: {ckpt_path}")

        self.device = torch.device(device or "cpu")

        hyper, state_dict = _load_ckpt_state(ckpt_path)
        hidden_size = _infer_hidden_size(hyper)
        deg = _infer_deg(hyper)

        self.model = GNN_LP_Model(hidden_size, deg)

        # The Lightning module has attribute `model`, so keys are usually `model.*`.
        cleaned = _strip_prefix(state_dict, "model.")
        if not cleaned:
            cleaned = state_dict

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if unexpected and not cleaned:
            raise RuntimeError(
                "Failed to load LP checkpoint: no compatible keys found in state_dict."
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_counts(self, mol: Chem.Mol) -> LonePairCounts:
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return LonePairCounts(
                num_lps=torch.zeros((0,), dtype=torch.long),
                num_conjugated_lps=torch.zeros((0,), dtype=torch.long),
            )

        x = torch.tensor([_one_hot_atom(a.GetSymbol()) for a in mol.GetAtoms()], dtype=torch.float32)

        edge_pairs: List[Tuple[int, int]] = []
        edge_attr: List[List[float]] = []
        for bond in mol.GetBonds():
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
            if a == b:
                continue
            feat = _bond_type_one_hot(bond)
            edge_pairs.append((a, b))
            edge_attr.append(feat)
            edge_pairs.append((b, a))
            edge_attr.append(feat)

        if not edge_pairs:
            # Degenerate case; model is edge-message-passing based.
            zeros = torch.zeros((num_atoms,), dtype=torch.long)
            return LonePairCounts(num_lps=zeros, num_conjugated_lps=zeros.clone())

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr_t = edge_attr_t.to(self.device)

        logits = self.model(x, edge_index, edge_attr_t)
        logits_1 = logits[:, :LP_N_CLASSES]
        logits_2 = logits[:, LP_N_CLASSES:]

        num_lps = logits_1.argmax(dim=1).to(torch.long).cpu()
        num_conj = logits_2.argmax(dim=1).to(torch.long).cpu()

        return LonePairCounts(num_lps=num_lps, num_conjugated_lps=num_conj)

    @torch.no_grad()
    def predict_lp_atom_map(self, mol: Chem.Mol) -> List[int]:
        counts = self.predict_counts(mol)
        lp_atoms: List[int] = []
        for atom_idx, n_lp in enumerate(counts.num_lps.tolist()):
            if n_lp <= 0:
                continue
            lp_atoms.extend([int(atom_idx)] * int(n_lp))
        return lp_atoms
