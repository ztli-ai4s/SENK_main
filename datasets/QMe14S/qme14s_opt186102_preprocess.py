import argparse
import os
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")
    if edge_index.size(0) == 2:
        return edge_index.long().contiguous()
    if edge_index.size(1) == 2:
        return edge_index.t().long().contiguous()
    raise ValueError(f"edge_index must be [2,E] or [E,2], got {tuple(edge_index.shape)}")


def _normalize_square33(x: torch.Tensor, name: str) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 3 and x.shape[1:] == (3, 3):
        return x
    if x.ndim == 2 and x.size(1) == 9:
        return x.view(-1, 3, 3)
    raise ValueError(f"{name} must be [N,3,3] or [N,9], got {tuple(x.shape)}")


def _normalize_hij(hij: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    hij = torch.as_tensor(hij, dtype=torch.float32)
    if hij.ndim == 3 and hij.shape[0] == 3 and hij.shape[1] == 3:
        hij = hij.permute(2, 0, 1).contiguous()
    elif hij.ndim == 2 and hij.size(1) == 9:
        hij = hij.view(-1, 3, 3)
    elif hij.ndim == 3 and hij.shape[1:] == (3, 3):
        hij = hij
    else:
        raise ValueError(f"Hij must be [E,3,3] or [E,9] or [3,3,E], got {tuple(hij.shape)}")
    if hij.size(0) != edge_index.size(1):
        raise ValueError(f"Hij size mismatch: Hij has {hij.size(0)} edges, edge_index has {edge_index.size(1)}")
    return hij


def _normalize_depolar(depolar: torch.Tensor) -> torch.Tensor:
    depolar = torch.as_tensor(depolar, dtype=torch.float32)
    if depolar.ndim == 3 and depolar.shape[1:] == (3, 6):
        return depolar
    if depolar.ndim == 3 and depolar.shape[1:] == (6, 3):
        return depolar.permute(0, 2, 1).contiguous()
    if depolar.ndim == 2 and depolar.size(1) == 18:
        return depolar.view(-1, 3, 6)
    raise ValueError(f"depolar must be [N,3,6] or [N,6,3] or [N,18], got {tuple(depolar.shape)}")


def _get(item: Any, key: str):
    if isinstance(item, dict):
        return item.get(key, None)
    return getattr(item, key, None)


def _normalize_item(item: Any, max_atoms: Optional[int]) -> Optional[Dict[str, Any]]:
    pos = _get(item, "pos")
    z = _get(item, "z")
    edge_index = _get(item, "edge_index")
    hii = _get(item, "Hii")
    if hii is None:
        hii = _get(item, "hii")
    hij = _get(item, "Hij")
    if hij is None:
        hij = _get(item, "hij")
    dedipole = _get(item, "dedipole")
    depolar = _get(item, "depolar")

    if pos is None or z is None or edge_index is None or hii is None or hij is None or dedipole is None or depolar is None:
        return None

    pos = torch.as_tensor(pos, dtype=torch.float32)
    z = torch.as_tensor(z, dtype=torch.long)
    if max_atoms is not None and pos.size(0) > max_atoms:
        return None

    edge_index = _normalize_edge_index(torch.as_tensor(edge_index))
    hii = _normalize_square33(hii, "Hii")
    hij = _normalize_hij(hij, edge_index)
    dedipole = _normalize_square33(dedipole, "dedipole")
    depolar = _normalize_depolar(depolar)

    out: Dict[str, Any] = {
        "pos": pos,
        "z": z,
        "edge_index": edge_index,
        "Hii": hii,
        "Hij": hij,
        "dedipole": dedipole,
        "depolar": depolar,
        "smile": _get(item, "smile"),
        "number": _get(item, "number"),
    }

    # keep optional properties if present
    for key in ["dipole", "polar", "quadrupole", "octapole", "npacharge", "hyperpolar"]:
        v = _get(item, key)
        if v is not None:
            out[key] = torch.as_tensor(v)

    return out


def preprocess_opt186102(input_path: str, output_path: str, max_atoms: Optional[int]) -> None:
    try:
        raw = torch.load(input_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(input_path, map_location="cpu")
    if not isinstance(raw, list):
        raise TypeError(f"Expected list from {input_path}, got {type(raw)}")

    cleaned: List[Dict[str, Any]] = []
    for item in tqdm(raw, desc="Preprocess OPT_186102"):
        normed = _normalize_item(item, max_atoms=max_atoms)
        if normed is None:
            continue
        cleaned.append(normed)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(cleaned, output_path)
    print(f"Saved {len(cleaned)} items to {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess OPT_186102 raw list for vib training")
    ap.add_argument("--input", type=str, default="qme14s_opt186102_raw.pt", help="Input .pt list from extraction")
    ap.add_argument("--output", type=str, default="qme14s_opt186102.pt", help="Output .pt list")
    ap.add_argument("--max_atoms", type=int, default=None, help="Optional filter on atom count")
    args = ap.parse_args()

    preprocess_opt186102(args.input, args.output, args.max_atoms)


if __name__ == "__main__":
    main()
