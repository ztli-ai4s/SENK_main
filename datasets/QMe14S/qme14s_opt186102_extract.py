import argparse
import os
from typing import Any, Dict, List, Optional

import h5py
import torch
from tqdm import tqdm


def _ensure_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.size(0) == 2:
        return edge_index.long().contiguous()
    if edge_index.size(1) == 2:
        return edge_index.t().long().contiguous()
    raise ValueError(f"edge_index must be [2,E] or [E,2], got {tuple(edge_index.shape)}")


def _read_group(group, group_key: str, fallback_id: int) -> Optional[Dict[str, Any]]:
    try:
        pos = torch.tensor(group["pos"][:], dtype=torch.float32)
        z = torch.tensor(group["z"][:], dtype=torch.long)
        edge_index = _ensure_edge_index(torch.tensor(group["edge_index"][:]))
        hii = torch.tensor(group["Hii"][:], dtype=torch.float32)
        hij = torch.tensor(group["Hij"][:], dtype=torch.float32)
        dedipole = torch.tensor(group["dedipole"][:], dtype=torch.float32)
        depolar = torch.tensor(group["depolar"][:], dtype=torch.float32)
    except Exception:
        return None

    item: Dict[str, Any] = {
        "pos": pos,
        "z": z,
        "edge_index": edge_index,
        "Hii": hii,
        "Hij": hij,
        "dedipole": dedipole,
        "depolar": depolar,
        "smile": group.attrs.get("smile", None),
    }

    # optional scalar/tensor properties if present
    for key in ["dipole", "polar", "quadrupole", "octapole", "npacharge", "hyperpolar"]:
        if key in group:
            try:
                item[key] = torch.tensor(group[key][:])
            except Exception:
                pass

    try:
        item["number"] = int(group_key)
    except Exception:
        item["number"] = int(fallback_id)

    return item


def extract_opt186102(h5_path: str, output: str, max_items: Optional[int]) -> None:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    data_list: List[Dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5file:
        keys = list(h5file.keys())
        iterator = tqdm(keys, desc="Extract OPT_186102", unit="item")
        for idx, key in enumerate(iterator):
            if max_items is not None and idx >= max_items:
                break
            group = h5file[key]
            item = _read_group(group, key, idx)
            if item is None:
                continue
            data_list.append(item)

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data_list, output)
    print(f"Saved {len(data_list)} items to {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract OPT_186102.h5 to a raw .pt list (Hii/Hij/dedipole/depolar)")
    ap.add_argument("--h5_path", type=str, default="OPT_186102.h5", help="Path to OPT_186102.h5")
    ap.add_argument("--output", type=str, default="qme14s_opt186102_raw.pt", help="Output .pt list")
    ap.add_argument("--max_items", type=int, default=None, help="Optional limit for quick tests")
    args = ap.parse_args()

    extract_opt186102(args.h5_path, args.output, args.max_items)


if __name__ == "__main__":
    main()
