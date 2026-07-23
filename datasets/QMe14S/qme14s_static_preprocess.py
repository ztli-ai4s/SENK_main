import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


def _sym_3x3_to_vec6(mat: torch.Tensor) -> torch.Tensor:
    if mat.ndim == 3 and mat.size(0) == 1:
        mat = mat[0]
    if mat.ndim != 2 or mat.size(0) != 3 or mat.size(1) != 3:
        raise ValueError(f"Expected [3,3] or [1,3,3], got {tuple(mat.shape)}")
    xx = mat[0, 0]
    yy = mat[1, 1]
    zz = mat[2, 2]
    xy = mat[0, 1]
    xz = mat[0, 2]
    yz = mat[1, 2]
    return torch.stack([xx, yy, zz, xy, xz, yz], dim=-1)


def _vec6_to_sym(vec6: torch.Tensor) -> torch.Tensor:
    if vec6.ndim != 1 or vec6.numel() != 6:
        raise ValueError(f"Expected vec6 shape [6], got {tuple(vec6.shape)}")
    mat = vec6.new_zeros((3, 3))
    mat[0, 0] = vec6[0]
    mat[1, 1] = vec6[1]
    mat[2, 2] = vec6[2]
    mat[0, 1] = vec6[3]
    mat[1, 0] = vec6[3]
    mat[0, 2] = vec6[4]
    mat[2, 0] = vec6[4]
    mat[1, 2] = vec6[5]
    mat[2, 1] = vec6[5]
    return mat


def _get(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key, None)
    return getattr(item, key, None)


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.size(0) == 2:
        return edge_index.long().contiguous()
    if edge_index.size(1) == 2:
        return edge_index.t().long().contiguous()
    raise ValueError(f"edge_index must be [2,E] or [E,2], got {tuple(edge_index.shape)}")


def _normalize_item(item: Any, max_atoms: Optional[int], ensure_polar_vec6: bool) -> Optional[Dict[str, Any]]:
    pos = _get(item, "pos")
    z = _get(item, "z")
    edge_index = _get(item, "edge_index")
    dipole = _get(item, "dipole")
    polar = _get(item, "polar")
    polar_vec6 = _get(item, "polar_vec6")

    if pos is None or z is None or edge_index is None or dipole is None:
        return None

    pos = torch.as_tensor(pos, dtype=torch.float32)
    z = torch.as_tensor(z, dtype=torch.long)
    if max_atoms is not None and pos.size(0) > max_atoms:
        return None

    edge_index = _normalize_edge_index(torch.as_tensor(edge_index))
    dipole = torch.as_tensor(dipole, dtype=torch.float32).view(-1)
    if dipole.numel() != 3:
        return None

    out: Dict[str, Any] = {
        "pos": pos,
        "z": z,
        "edge_index": edge_index,
        "dipole": dipole,
        "smile": _get(item, "smile"),
        "number": _get(item, "number"),
    }

    if polar is not None:
        polar = torch.as_tensor(polar, dtype=torch.float32)
        out["polar"] = polar
    if polar_vec6 is not None:
        polar_vec6 = torch.as_tensor(polar_vec6, dtype=torch.float32).view(-1)
        if polar_vec6.numel() == 6:
            out["polar_vec6"] = polar_vec6

    if ensure_polar_vec6 and "polar_vec6" not in out and "polar" in out:
        try:
            out["polar_vec6"] = _sym_3x3_to_vec6(out["polar"])
        except Exception:
            pass

    if "polar" not in out and "polar_vec6" in out:
        try:
            out["polar"] = _vec6_to_sym(out["polar_vec6"])
        except Exception:
            pass

    return out


def preprocess_qme14s_static(input_path: str, output_path: str, max_atoms: Optional[int], ensure_polar_vec6: bool) -> None:
    raw = torch.load(input_path, map_location="cpu")
    if not isinstance(raw, list):
        raise TypeError(f"Expected list from {input_path}, got {type(raw)}")

    cleaned: List[Dict[str, Any]] = []
    for item in tqdm(raw, desc="Preprocess QMe14S static"):
        normed = _normalize_item(item, max_atoms=max_atoms, ensure_polar_vec6=ensure_polar_vec6)
        if normed is None:
            continue
        cleaned.append(normed)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(cleaned, output_path)
    print(f"Saved {len(cleaned)} items to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess QMe14S static list for polar/dipole training")
    ap.add_argument("--input", type=str, required=True, help="Input .pt list from extraction")
    ap.add_argument("--output", type=str, default="qme14s_static.pt", help="Output .pt list")
    ap.add_argument("--max_atoms", type=int, default=None, help="Optional filter on atom count")
    ap.add_argument("--no_polar_vec6", action="store_true", help="Skip enforcing polar_vec6")
    args = ap.parse_args()

    preprocess_qme14s_static(
        input_path=args.input,
        output_path=args.output,
        max_atoms=args.max_atoms,
        ensure_polar_vec6=(not args.no_polar_vec6),
    )


if __name__ == "__main__":
    main()
