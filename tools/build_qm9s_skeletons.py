"""
Generate SENK-compatible skeleton files directly from qm9s.pt (with optional LP prediction).
Output: skeleton_all.pt and ratio-split skeleton_{train,valid,test}.pt.
Node type convention: 0=atom, 1=bond, 2=LP.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from rdkit import Chem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
TOOLS_DIR = SCRIPT_DIR
DEFAULT_LP_CKPT = os.path.join(TOOLS_DIR, "lp_pred_model.ckpt")

for p in [PROJECT_ROOT, TOOLS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _item_field(item: Any, name: str, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _pick_id(item: Any, fallback: int, mode: str) -> int:
    if mode == "index":
        return fallback
    num = _item_field(item, "number") if mode in {"auto", "number"} else None
    if num is not None:
        try:
            return int(num)
        except Exception:
            pass
    return fallback


def _extract_bonds(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    if edge_index is None:
        return []
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index expected shape [2, E], got {tuple(edge_index.shape)}")
    pairs = set()
    for u, v in edge_index.t():
        ua = int(u)
        va = int(v)
        if ua == va:
            continue
        a, b = (ua, va) if ua < va else (va, ua)
        pairs.add((a, b))
    return sorted(pairs)


def _build_radius_graph(pos: torch.Tensor, r: float = 5.0) -> torch.Tensor:
    """Builds a radius graph (all pairs within distance r, excluding self-loops)."""
    if pos.size(0) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    # Compute pairwise distances
    dist = torch.cdist(pos, pos)
    # Mask for distance < r and > 0 (exclude self-loops)
    mask = (dist < r) & (dist > 1e-6)
    row, col = torch.where(mask)
    return torch.stack([row, col], dim=0)


def _build_entry(atom_pos: torch.Tensor, bond_pairs: List[Tuple[int, int]], mol_id: int, lp_atoms: Optional[List[int]] = None) -> Dict[str, Any]:
    lp_atoms = lp_atoms or []
    num_atoms = int(atom_pos.size(0))
    num_bonds = len(bond_pairs)
    num_lps = len(lp_atoms)
    total_nodes = num_atoms + num_bonds + num_lps

    node_type = torch.zeros(total_nodes, dtype=torch.long)
    if num_bonds:
        node_type[num_atoms : num_atoms + num_bonds] = 1  # bonds
    if num_lps:
        node_type[num_atoms + num_bonds :] = 2  # LPs

    if num_bonds:
        atoms_a = torch.tensor([a for a, _ in bond_pairs], dtype=torch.long)
        atoms_b = torch.tensor([b for _, b in bond_pairs], dtype=torch.long)
        bond_pos = 0.5 * (atom_pos[atoms_a] + atom_pos[atoms_b])
        atom_bond_index = torch.stack([atoms_a, atoms_b], dim=0)
    else:
        bond_pos = torch.zeros((0, 3), dtype=torch.float32)
        atom_bond_index = torch.zeros((2, 0), dtype=torch.long)

    lp_pos = atom_pos[lp_atoms] if num_lps else torch.zeros((0, 3), dtype=torch.float32)

    atom_to_nbo_edges: List[Tuple[int, int]] = []
    bond_offset = num_atoms
    lp_offset = num_atoms + num_bonds
    for bond_idx, (a, b) in enumerate(bond_pairs):
        gidx = bond_offset + bond_idx
        atom_to_nbo_edges.append((int(a), gidx))
        atom_to_nbo_edges.append((int(b), gidx))
    for lp_idx, atom_idx in enumerate(lp_atoms):
        atom_to_nbo_edges.append((int(atom_idx), lp_offset + lp_idx))

    atom_to_nbo_index = (
        torch.tensor(atom_to_nbo_edges, dtype=torch.long).t().contiguous()
        if atom_to_nbo_edges
        else torch.zeros((2, 0), dtype=torch.long)
    )

    pos_global = torch.cat([atom_pos, bond_pos, lp_pos], dim=0)

    # Generate interaction edges (radius graph)
    interaction_edge_index = _build_radius_graph(pos_global, r=5.0)

    dense = {
        "qm9_id": int(mol_id),
        "node_type": node_type,
        "atom_to_nbo_index": atom_to_nbo_index,
        "interaction_edge_index": interaction_edge_index,
        "atom_bond_index": atom_bond_index,
        "pos_global": pos_global,
        "num_global_nodes": int(pos_global.size(0)),
        "num_atoms": num_atoms,
        "num_lps": num_lps,
        "num_bonds": num_bonds,
    }

    sparse = {
        "qm9_id": int(mol_id),
        "node_type": torch.zeros(num_atoms, dtype=torch.long),
        "atom_to_nbo_index": torch.zeros((2, 0), dtype=torch.long),
        "interaction_edge_index": torch.zeros((2, 0), dtype=torch.long),
        "atom_bond_index": atom_bond_index.clone(),
        "pos_global": atom_pos.clone(),
        "num_global_nodes": num_atoms,
        "num_atoms": num_atoms,
        "num_lps": 0,
        "num_bonds": num_bonds,
    }

    entry = dict(dense)
    entry["sparse_variant"] = sparse
    return entry


def _load_list(path: str) -> List[Any]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    raise TypeError(f"Expected list/tuple from {path}, got {type(data)}")


def _build_mol_from_item(item: Any, bond_pairs: List[Tuple[int, int]]) -> Optional[Chem.Mol]:
    smi = _item_field(item, "smile") or _item_field(item, "smiles")
    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol
    z = _item_field(item, "z")
    if z is None:
        return None
    try:
        z_list = [int(v) for v in z]
    except Exception:
        return None
    mol_edit = Chem.RWMol()
    for atomic_num in z_list:
        mol_edit.AddAtom(Chem.Atom(int(atomic_num)))
    for a, b in bond_pairs:
        try:
            mol_edit.AddBond(int(a), int(b), Chem.BondType.SINGLE)
        except Exception:
            continue
    return mol_edit.GetMol()


def _stats_report(count: int, atoms: int, bonds: int, lps: int, min_id: int, max_id: int) -> Dict[str, Any]:
    return {
        "total_molecules": count,
        "total_atoms": atoms,
        "total_bonds": bonds,
        "total_lp_nodes": lps,
        "avg_atoms_per_mol": atoms / max(count, 1),
        "avg_bonds_per_mol": bonds / max(count, 1),
        "avg_lp_nodes_per_mol": lps / max(count, 1),
        "id_min": min_id,
        "id_max": max_id,
    }


def _build_splits(keys: List[int], ratios: Tuple[float, float, float], seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.tensor(keys)[torch.randperm(len(keys), generator=g)].tolist()
    n_train = int(len(keys) * ratios[0])
    n_val = int(len(keys) * ratios[1])
    train_ids = perm[:n_train]
    val_ids = perm[n_train:n_train + n_val]
    test_ids = perm[n_train + n_val:]
    return {"train": train_ids, "valid": val_ids, "test": test_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SENK skeletons directly from qm9s.pt")
    ap.add_argument("--qm9s", default="datasets/qm9s.pt", help="Path to qm9s.pt list file")
    ap.add_argument("--output_dir", default="datasets/new_skeleton", help="Directory to store skeleton_all.pt")
    ap.add_argument("--id_mode", choices=["auto", "index", "number"], default="auto", help="Use Data.number when available or list index")
    ap.add_argument("--max_molecules", type=int, default=None, help="Optional limit for quick dry-runs")
    ap.add_argument("--disable_lp", action="store_true", help="Disable LP prediction and build atom+bond skeletons only")
    ap.add_argument("--lp_ckpt", type=str, default="tools/lp_pred_model.ckpt", help="LP checkpoint path")
    ap.add_argument("--lp_device", type=str, default="cuda", help="LP predictor device")
    ap.add_argument("--save_sparse_file", action="store_true", help="Additionally save skeleton_sparse_all.pt")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), help="train/val/test ratios")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_list = _load_list(args.qm9s)
    os.makedirs(args.output_dir, exist_ok=True)

    lp_predictor = None
    if not args.disable_lp:
        try:
            from tools.lp_predictor import LPPredictor  # type: ignore
        except ModuleNotFoundError:
            from lp_predictor import LPPredictor  # type: ignore
        lp_predictor = LPPredictor(args.lp_ckpt, device=args.lp_device)

    skeleton_all: Dict[int, Dict[str, Any]] = {}
    skeleton_sparse: Dict[int, Dict[str, Any]] = {}
    total_atoms = 0
    total_bonds = 0
    total_lps = 0
    min_id = 1 << 60
    max_id = -1
    mol_without_lp = 0
    mol_lp_fail = 0
    lp_from_smiles = 0
    lp_from_z = 0
    shape_issues = 0

    for idx, item in enumerate(data_list):
        if args.max_molecules is not None and idx >= args.max_molecules:
            break
        mol_id = _pick_id(item, idx, args.id_mode)
        atom_pos = _item_field(item, "pos")
        edge_index = _item_field(item, "edge_index")
        if atom_pos is None or edge_index is None:
            raise ValueError(f"Entry {idx} missing pos/edge_index")
        bonds = _extract_bonds(edge_index)

        lp_atoms: List[int] = []
        if lp_predictor is not None:
            mol = _build_mol_from_item(item, bonds)
            if mol is not None:
                try:
                    lp_atoms = lp_predictor.predict_lp_atom_map(mol)
                    if _item_field(item, "smile") or _item_field(item, "smiles"):
                        lp_from_smiles += 1
                    else:
                        lp_from_z += 1
                    if not lp_atoms:
                        mol_without_lp += 1
                except Exception:
                    mol_lp_fail += 1
                    lp_atoms = []
            else:
                mol_lp_fail += 1

        entry = _build_entry(atom_pos, bonds, mol_id, lp_atoms=lp_atoms)
        if entry["node_type"].numel() != entry["num_global_nodes"] or entry["pos_global"].size(0) != entry["num_global_nodes"]:
            shape_issues += 1
        skeleton_all[mol_id] = entry
        skeleton_sparse[mol_id] = entry["sparse_variant"]

        total_atoms += int(entry["num_atoms"])
        total_bonds += int(entry["num_bonds"])
        total_lps += int(entry["num_lps"])
        min_id = min(min_id, mol_id)
        max_id = max(max_id, mol_id)

    out_path = os.path.join(args.output_dir, "skeleton_all.pt")
    torch.save(skeleton_all, out_path)
    if args.save_sparse_file:
        torch.save(skeleton_sparse, os.path.join(args.output_dir, "skeleton_sparse_all.pt"))

    stats = _stats_report(len(skeleton_all), total_atoms, total_bonds, total_lps, min_id, max_id)
    stats["molecules_with_zero_lp"] = mol_without_lp
    stats["lp_predict_failures"] = mol_lp_fail
    stats["lp_from_smiles"] = lp_from_smiles
    stats["lp_from_z_fallback"] = lp_from_z
    stats["shape_issues"] = shape_issues
    with open(os.path.join(args.output_dir, "skeleton_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    splits = _build_splits(sorted(skeleton_all.keys()), tuple(args.ratios), args.seed)
    for split, ids in splits.items():
        split_map = {mid: skeleton_all[mid] for mid in ids}
        torch.save(split_map, os.path.join(args.output_dir, f"skeleton_{split}.pt"))

    print(f"Saved {len(skeleton_all)} skeletons to {out_path}")
    print(json.dumps(stats, indent=2))
    print("LP note: lp_from_smiles counts molecules where SMILES was available; lp_from_z_fallback counts molecules built from Z+bonds; lp_predict_failures counts prediction/build errors.")


if __name__ == "__main__":
    main()
