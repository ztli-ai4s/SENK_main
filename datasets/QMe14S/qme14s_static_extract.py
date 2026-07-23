import argparse
import os
from typing import Any, Dict, List, Optional

import h5py
import torch
from tqdm import tqdm


def _vec6_from_sym(polar: torch.Tensor) -> torch.Tensor:
    if polar.ndim == 3 and polar.size(0) == 1:
        polar = polar[0]
    if polar.ndim != 2 or polar.size(0) != 3 or polar.size(1) != 3:
        raise ValueError(f"polar must be [3,3] (or [1,3,3]), got {tuple(polar.shape)}")
    xx = polar[0, 0]
    yy = polar[1, 1]
    zz = polar[2, 2]
    xy = polar[0, 1]
    xz = polar[0, 2]
    yz = polar[1, 2]
    return torch.stack([xx, yy, zz, xy, xz, yz], dim=-1)


def _ensure_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.size(0) == 2:
        return edge_index.long().contiguous()
    if edge_index.size(1) == 2:
        return edge_index.t().long().contiguous()
    raise ValueError(f"edge_index must be [2,E] or [E,2], got {tuple(edge_index.shape)}")


def _read_group(group, group_key: str, fallback_id: int, include_polar_vec6: bool) -> Optional[Dict[str, Any]]:
    try:
        pos = torch.tensor(group["pos"][:], dtype=torch.float32)
        z = torch.tensor(group["z"][:], dtype=torch.long)
        edge_index = _ensure_edge_index(torch.tensor(group["edge_index"][:]))
        dipole = torch.tensor(group["dipole"][:], dtype=torch.float32).view(-1)
        polar = torch.tensor(group["polar"][:], dtype=torch.float32)
    except Exception:
        return None

    if dipole.numel() != 3:
        return None

    item: Dict[str, Any] = {
        "pos": pos,
        "z": z,
        "edge_index": edge_index,
        "dipole": dipole,
        "polar": polar,
        "smile": group.attrs.get("smile", None),
    }

    try:
        item["number"] = int(group_key)
    except Exception:
        item["number"] = int(fallback_id)

    if include_polar_vec6:
        try:
            item["polar_vec6"] = _vec6_from_sym(polar)
        except Exception:
            pass

    return item


def extract_qme14s_static(h5_path: str, output: str, max_items: Optional[int], include_polar_vec6: bool) -> None:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    data_list: List[Dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5file:
        keys = list(h5file.keys())
        iterator = tqdm(keys, desc="Extract QMe14S static", unit="item")
        for idx, key in enumerate(iterator):
            if max_items is not None and idx >= max_items:
                break
            group = h5file[key]
            item = _read_group(group, key, idx, include_polar_vec6)
            if item is None:
                continue
            data_list.append(item)

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data_list, output)
    print(f"Saved {len(data_list)} items to {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract QMe14S static tensor dataset to a .pt list")
    ap.add_argument("--h5_path", type=str, default="OPT_186102.h5", help="Path to OPT_186102.h5")
    ap.add_argument("--output", type=str, default="qme14s_static_raw.pt", help="Output .pt list")
    ap.add_argument("--max_items", type=int, default=None, help="Optional limit for quick tests")
    ap.add_argument("--no_polar_vec6", action="store_true", help="Skip polar_vec6 generation")
    args = ap.parse_args()

    extract_qme14s_static(
        h5_path=args.h5_path,
        output=args.output,
        max_items=args.max_items,
        include_polar_vec6=(not args.no_polar_vec6),
    )


if __name__ == "__main__":
    main()
