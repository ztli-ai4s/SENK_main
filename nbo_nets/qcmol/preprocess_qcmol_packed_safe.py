#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Packed qcMol preprocessing: stream HDF5 -> packed shard tensors with offsets.

Goals:
- single-process, single HDFStore open at a time (safe for FD)
- packed arrays + ptr offsets for fast training-time slicing
- shard-level files to avoid millions of small files
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise ImportError("pandas is required") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _log(msg: str) -> None:
    print(msg, flush=True)


def _should_log(last_ts: float, interval: float) -> bool:
    return interval > 0 and (time.time() - last_ts) >= interval


class _FatalPreprocessError(RuntimeError):
    pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from qcmol_loader import (
    QcMolKeys,
    _has_required_keys,
    _xyz_columns,
    _series_has_sequence,
    _xyz_table_to_pos,
    _row_to_array,
    _infer_z,
    _extract_bond_index,
    _extract_atom_targets,
    _extract_bond_targets_multi,
)


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_THIS_DIR, path))


_QCMOL_SUBSET_DESCRIPTIONS: Dict[str, str] = {
    "pubchem": (
        "PubChemQC ∩ ZINC drug-like subset; broad small-molecule quantum/NBO coverage."
    ),
    "pdbbind2020": (
        "PDBbind 2020 protein-ligand subset; biased toward bioactive bound ligands and recognition-relevant chemistry."
    ),
    "custom": "User-specified qcMol subset root/output paths.",
}


def _arg_provided(flag: str) -> bool:
    return flag in sys.argv


def _apply_subset_preset(args: argparse.Namespace) -> None:
    subset = str(getattr(args, "subset", "custom")).lower()
    if subset == "pdbbind2020":
        if not _arg_provided("--data_root"):
            args.data_root = "PDBbind2020"
        if not _arg_provided("--out_dir"):
            args.out_dir = "qcmol_packed_pdbbind2020"
    elif subset == "pubchem":
        if not _arg_provided("--out_dir"):
            args.out_dir = "qcmol_packed_1"

    _log(
        f"[Preset] subset={subset} desc={_QCMOL_SUBSET_DESCRIPTIONS.get(subset, 'n/a')} "
        f"data_root={args.data_root} out_dir={args.out_dir}"
    )


def _load_index_jsonl(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("path")
            if not p:
                continue
            out.append(str(p))
    return out


def _to_int(val: object, default: int = 0) -> int:
    try:
        if val is None:
            return int(default)
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            return int(val.strip())
        return int(default)
    except Exception:
        return int(default)


def _dataset_cache_signature(
    root: str,
    keys: QcMolKeys,
    files: List[str],
    max_files: Optional[int],
    max_mols_per_file: Optional[int],
    strict_keys: bool,
    cache_format: str,
    cache_shard_size: int,
) -> Tuple[str, str]:
    files_signature = hashlib.sha1("|".join(files).encode("utf-8")).hexdigest()
    mtimes: List[str] = []
    for path in files:
        try:
            mtimes.append(str(os.path.getmtime(path)))
        except Exception:
            mtimes.append("0")
    mtime_sig = hashlib.sha1("|".join(mtimes).encode("utf-8")).hexdigest()
    sig = "|".join(
        [
            "qcmol_packed_v1",
            root,
            str(keys),
            str(max_files),
            str(max_mols_per_file),
            str(strict_keys),
            str(cache_format),
            str(cache_shard_size),
            files_signature,
            mtime_sig,
        ]
    )
    cache_signature = hashlib.sha1(sig.encode("utf-8")).hexdigest()
    return files_signature, cache_signature


def _align_targets(
    targets: Optional[torch.Tensor],
    target_dim: Optional[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if targets is None or target_dim is None:
        return None, None
    if targets.ndim == 1:
        targets = targets.view(-1, 1)
    if targets.ndim != 2:
        return None, None

    cur_dim = int(targets.size(1))
    if cur_dim == target_dim:
        return targets, torch.ones((targets.size(0), 1), dtype=torch.float32)

    if cur_dim > target_dim:
        trimmed = targets[:, :target_dim]
        mask = torch.ones((trimmed.size(0), 1), dtype=torch.float32)
        return trimmed, mask

    pad = target_dim - cur_dim
    pad_vals = torch.zeros((targets.size(0), pad), dtype=targets.dtype)
    padded = torch.cat([targets, pad_vals], dim=1)
    mask = torch.cat(
        [
            torch.ones((targets.size(0), cur_dim), dtype=torch.float32),
            torch.zeros((targets.size(0), pad), dtype=torch.float32),
        ],
        dim=1,
    )
    del mask
    return padded, torch.ones((targets.size(0), 1), dtype=torch.float32)


def _detect_mode(df_xyz: pd.DataFrame) -> str:
    per_file = False
    if isinstance(df_xyz, pd.DataFrame):
        cols = _xyz_columns(df_xyz)
        if cols is not None and not _series_has_sequence(df_xyz.iloc[0]):
            per_file = True
    return "per_file" if per_file else "per_row"


def _store_get(store: pd.HDFStore, key: str) -> Optional[pd.DataFrame]:
    if key is None:
        return None
    norm = str(key).lstrip("/").lower()
    matched = None
    for k in store.keys():
        if k.lstrip("/").lower() == norm:
            matched = k
            break
    if matched is None:
        return None
    obj = store[matched]
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    return obj


def _safe_read_df_store(store: pd.HDFStore, key: str, strict_keys: bool) -> Optional[pd.DataFrame]:
    try:
        out = _store_get(store, key)
        if out is None:
            raise KeyError(key)
        return out
    except Exception as exc:
        if strict_keys:
            raise exc
        return None


def _probe_dims(
    files: List[str],
    keys: QcMolKeys,
    max_mols_per_file: Optional[int],
    strict_keys: bool,
    probe_limit: int,
    log_every_sec: float = 10.0,
) -> Dict[str, int]:
    dims = {"atom": 0, "bond": 0, "aux_atom": 0, "aux_bond": 0}
    seen = 0
    last_log = time.time()
    _log(f"[Stage] Probing dims: limit={probe_limit} files={len(files)}")
    iterator = enumerate(files, start=1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Probing dims", total=len(files))
    for file_idx, path in iterator:
        if tqdm is None and _should_log(last_log, log_every_sec):
            _log(f"[Probe] opening file {file_idx}/{len(files)}: {os.path.basename(path)}")
            last_log = time.time()
        try:
            with pd.HDFStore(path, mode="r") as store:
                df_xyz = _safe_read_df_store(store, keys.xyz, strict_keys)
                if df_xyz is None:
                    continue
                mode = _detect_mode(df_xyz)
                n_rows = 1 if mode == "per_file" else len(df_xyz)
                if max_mols_per_file is not None:
                    n_rows = min(n_rows, int(max_mols_per_file))

                df_bond_basic = _safe_read_df_store(store, keys.bond_basic, strict_keys)

                atom_tables = [_safe_read_df_store(store, k, strict_keys) for k in keys.atom_targets]
                bond_tables = [_safe_read_df_store(store, k, strict_keys) for k in keys.bond_targets]
                aux_atom_tables = [_safe_read_df_store(store, k, strict_keys) for k in keys.aux_atom_targets]
                aux_bond_tables = [_safe_read_df_store(store, k, strict_keys) for k in keys.aux_bond_targets]

                for row_idx in range(n_rows):
                    if mode == "per_file":
                        pos_arr = _xyz_table_to_pos(df_xyz)
                    else:
                        pos_arr = _row_to_array(df_xyz.iloc[row_idx])
                    if pos_arr is None or pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
                        continue
                    num_atoms = int(pos_arr.shape[0])

                    bond_basic_row = df_bond_basic if mode == "per_file" else (df_bond_basic.iloc[row_idx] if df_bond_basic is not None else None)
                    bond_index = _extract_bond_index(bond_basic_row, num_atoms)

                    atom_rows = [
                        (tbl if mode == "per_file" else (tbl.iloc[row_idx] if tbl is not None else None))
                        for tbl in atom_tables
                    ]
                    atom_targets = _extract_atom_targets(atom_rows, num_atoms)
                    if atom_targets is not None:
                        dims["atom"] = max(dims["atom"], int(atom_targets.shape[1]))

                    bond_rows = [
                        (tbl if mode == "per_file" else (tbl.iloc[row_idx] if tbl is not None else None))
                        for tbl in bond_tables
                    ]
                    bond_targets = _extract_bond_targets_multi(bond_rows, num_atoms, bond_index)
                    if bond_targets is not None:
                        dims["bond"] = max(dims["bond"], int(bond_targets.shape[1]))

                    aux_atom_rows = [
                        (tbl if mode == "per_file" else (tbl.iloc[row_idx] if tbl is not None else None))
                        for tbl in aux_atom_tables
                    ]
                    aux_atom_targets = _extract_atom_targets(aux_atom_rows, num_atoms)
                    if aux_atom_targets is not None:
                        dims["aux_atom"] = max(dims["aux_atom"], int(aux_atom_targets.shape[1]))

                    aux_bond_rows = [
                        (tbl if mode == "per_file" else (tbl.iloc[row_idx] if tbl is not None else None))
                        for tbl in aux_bond_tables
                    ]
                    aux_bond_targets = _extract_bond_targets_multi(aux_bond_rows, num_atoms, bond_index)
                    if aux_bond_targets is not None:
                        dims["aux_bond"] = max(dims["aux_bond"], int(aux_bond_targets.shape[1]))

                    seen += 1
                    if tqdm is None:
                        if _should_log(last_log, log_every_sec):
                            _log(f"[Probe] samples={seen} files_scanned={file_idx}/{len(files)}")
                            last_log = time.time()
                    if seen >= probe_limit:
                        return dims
        except Exception:
            continue
    return dims


def _atomic_save(payload: object, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _shard_path(out_dir: str, cache_signature: str, shard_id: int, worker_id: int, worker_count: int) -> str:
    if worker_count > 1:
        return os.path.join(out_dir, f"qcmol_packed_shard_{cache_signature}_w{worker_id:03d}_{shard_id:06d}.pt")
    return os.path.join(out_dir, f"qcmol_packed_shard_{cache_signature}_{shard_id:06d}.pt")


def _resume_state_path(out_dir: str, cache_signature: str, worker_id: int, worker_count: int) -> str:
    if worker_count > 1:
        name = f"qcmol_packed_resume_{cache_signature}_w{worker_id:03d}.json"
    else:
        name = f"qcmol_packed_resume_{cache_signature}.json"
    return os.path.join(out_dir, name)


def _read_shard_num_samples(path: str) -> int:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    n = int(payload.get("num_samples", 0) or 0)
    if n > 0:
        return n
    atom_ptr = payload.get("atom_ptr")
    if isinstance(atom_ptr, torch.Tensor) and atom_ptr.numel() >= 1:
        return int(atom_ptr.numel() - 1)
    return 0


def _scan_existing_shards(out_dir: str, cache_signature: str, worker_id: int, worker_count: int) -> Tuple[int, int]:
    shard_id = 0
    total_samples = 0
    while True:
        path = _shard_path(out_dir, cache_signature, shard_id, worker_id, worker_count)
        if not os.path.exists(path):
            break
        n = _read_shard_num_samples(path)
        if n <= 0:
            break
        total_samples += n
        shard_id += 1
    return shard_id, total_samples


def _load_resume_state(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _save_resume_state(path: str, state: Dict[str, object]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class ShardWriter:
    def __init__(
        self,
        shard_size: int,
        atom_dim: int,
        bond_dim: int,
        aux_atom_dim: int,
        aux_bond_dim: int,
    ) -> None:
        self.shard_size = shard_size
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.aux_atom_dim = aux_atom_dim
        self.aux_bond_dim = aux_bond_dim
        self.reset()

    def reset(self) -> None:
        self.pos_list: List[torch.Tensor] = []
        self.z_list: List[torch.Tensor] = []
        self.atom_targets_list: List[torch.Tensor] = []
        self.atom_mask_list: List[torch.Tensor] = []
        self.bond_index_list: List[torch.Tensor] = []
        self.bond_targets_list: List[torch.Tensor] = []
        self.bond_mask_list: List[torch.Tensor] = []
        self.aux_atom_targets_list: List[torch.Tensor] = []
        self.aux_atom_mask_list: List[torch.Tensor] = []
        self.aux_bond_targets_list: List[torch.Tensor] = []
        self.aux_bond_mask_list: List[torch.Tensor] = []
        self.atom_ptr: List[int] = [0]
        self.bond_ptr: List[int] = [0]
        self.num_samples = 0

    def add_sample(
        self,
        pos: torch.Tensor,
        z: torch.Tensor,
        atom_bond_index: torch.Tensor,
        atom_targets: Optional[torch.Tensor],
        atom_mask: Optional[torch.Tensor],
        bond_targets: Optional[torch.Tensor],
        bond_mask: Optional[torch.Tensor],
        aux_atom_targets: Optional[torch.Tensor],
        aux_atom_mask: Optional[torch.Tensor],
        aux_bond_targets: Optional[torch.Tensor],
        aux_bond_mask: Optional[torch.Tensor],
        sample_tag: Optional[str] = None,
    ) -> None:
        tag = f" [{sample_tag}]" if sample_tag else ""
        atom_offset = self.atom_ptr[-1]
        bond_offset = self.bond_ptr[-1]
        num_atoms = int(z.numel())
        num_bonds = int(atom_bond_index.size(1))

        if pos.ndim != 2 or int(pos.size(0)) != num_atoms or int(pos.size(1)) != 3:
            raise _FatalPreprocessError(
                f"Invalid pos shape{tag}: expected [{num_atoms}, 3], got {tuple(pos.shape)}"
            )
        if atom_bond_index.ndim != 2 or int(atom_bond_index.size(0)) != 2:
            raise _FatalPreprocessError(
                f"Invalid bond_index shape{tag}: expected [2, E], got {tuple(atom_bond_index.shape)}"
            )

        def _check_optional_2d(name: str, tensor: Optional[torch.Tensor], rows: int, cols: int) -> None:
            if tensor is None:
                return
            if tensor.ndim != 2 or int(tensor.size(0)) != int(rows) or int(tensor.size(1)) != int(cols):
                raise _FatalPreprocessError(
                    f"Invalid {name} shape{tag}: expected [{rows}, {cols}], got {tuple(tensor.shape)}"
                )

        if self.atom_dim > 0:
            _check_optional_2d("atom_targets", atom_targets, num_atoms, self.atom_dim)
            _check_optional_2d("atom_mask", atom_mask, num_atoms, 1)
        if self.bond_dim > 0:
            _check_optional_2d("bond_targets", bond_targets, num_bonds, self.bond_dim)
            _check_optional_2d("bond_mask", bond_mask, num_bonds, 1)
        if self.aux_atom_dim > 0:
            _check_optional_2d("aux_atom_targets", aux_atom_targets, num_atoms, self.aux_atom_dim)
            _check_optional_2d("aux_atom_mask", aux_atom_mask, num_atoms, 1)
        if self.aux_bond_dim > 0:
            _check_optional_2d("aux_bond_targets", aux_bond_targets, num_bonds, self.aux_bond_dim)
            _check_optional_2d("aux_bond_mask", aux_bond_mask, num_bonds, 1)

        self.pos_list.append(pos)
        self.z_list.append(z)

        if self.atom_dim > 0:
            self.atom_targets_list.append(atom_targets if atom_targets is not None else torch.zeros((num_atoms, self.atom_dim), dtype=torch.float32))
            self.atom_mask_list.append(atom_mask if atom_mask is not None else torch.zeros((num_atoms, 1), dtype=torch.float32))

        if self.bond_dim > 0:
            self.bond_targets_list.append(bond_targets if bond_targets is not None else torch.zeros((num_bonds, self.bond_dim), dtype=torch.float32))
            self.bond_mask_list.append(bond_mask if bond_mask is not None else torch.zeros((num_bonds, 1), dtype=torch.float32))

        if self.aux_atom_dim > 0:
            self.aux_atom_targets_list.append(aux_atom_targets if aux_atom_targets is not None else torch.zeros((num_atoms, self.aux_atom_dim), dtype=torch.float32))
            self.aux_atom_mask_list.append(aux_atom_mask if aux_atom_mask is not None else torch.zeros((num_atoms, 1), dtype=torch.float32))

        if self.aux_bond_dim > 0:
            self.aux_bond_targets_list.append(aux_bond_targets if aux_bond_targets is not None else torch.zeros((num_bonds, self.aux_bond_dim), dtype=torch.float32))
            self.aux_bond_mask_list.append(aux_bond_mask if aux_bond_mask is not None else torch.zeros((num_bonds, 1), dtype=torch.float32))

        if atom_bond_index.numel() > 0:
            self.bond_index_list.append(atom_bond_index + atom_offset)

        self.atom_ptr.append(atom_offset + num_atoms)
        self.bond_ptr.append(bond_offset + num_bonds)
        self.num_samples += 1

    def is_full(self) -> bool:
        return self.num_samples >= self.shard_size

    def finalize_payload(self) -> Dict[str, torch.Tensor]:
        pos = torch.cat(self.pos_list, dim=0) if self.pos_list else torch.empty((0, 3), dtype=torch.float32)
        z = torch.cat(self.z_list, dim=0) if self.z_list else torch.empty((0,), dtype=torch.long)
        bond_index = (
            torch.cat(self.bond_index_list, dim=1)
            if self.bond_index_list
            else torch.empty((2, 0), dtype=torch.long)
        )

        payload: Dict[str, torch.Tensor] = {
            "atom_ptr": torch.tensor(self.atom_ptr, dtype=torch.long),
            "bond_ptr": torch.tensor(self.bond_ptr, dtype=torch.long),
            "pos": pos,
            "z": z,
            "bond_index": bond_index,
        }

        if self.atom_dim > 0:
            payload["atom_targets"] = torch.cat(self.atom_targets_list, dim=0) if self.atom_targets_list else torch.empty((0, self.atom_dim), dtype=torch.float32)
            payload["atom_target_mask"] = torch.cat(self.atom_mask_list, dim=0) if self.atom_mask_list else torch.empty((0, 1), dtype=torch.float32)

        if self.bond_dim > 0:
            payload["bond_targets"] = torch.cat(self.bond_targets_list, dim=0) if self.bond_targets_list else torch.empty((0, self.bond_dim), dtype=torch.float32)
            payload["bond_target_mask"] = torch.cat(self.bond_mask_list, dim=0) if self.bond_mask_list else torch.empty((0, 1), dtype=torch.float32)

        if self.aux_atom_dim > 0:
            payload["aux_atom_targets"] = torch.cat(self.aux_atom_targets_list, dim=0) if self.aux_atom_targets_list else torch.empty((0, self.aux_atom_dim), dtype=torch.float32)
            payload["aux_atom_target_mask"] = torch.cat(self.aux_atom_mask_list, dim=0) if self.aux_atom_mask_list else torch.empty((0, 1), dtype=torch.float32)

        if self.aux_bond_dim > 0:
            payload["aux_bond_targets"] = torch.cat(self.aux_bond_targets_list, dim=0) if self.aux_bond_targets_list else torch.empty((0, self.aux_bond_dim), dtype=torch.float32)
            payload["aux_bond_target_mask"] = torch.cat(self.aux_bond_mask_list, dim=0) if self.aux_bond_mask_list else torch.empty((0, 1), dtype=torch.float32)

        return payload


def main():
    ap = argparse.ArgumentParser(description="Packed qcMol preprocessing to sharded pt files")
    ap.add_argument(
        "--subset",
        type=str,
        choices=["pubchem", "pdbbind2020", "custom"],
        default="pubchem",
        help="qcMol subset preset. pubchem preserves existing behavior; pdbbind2020 maps to qcmol/PDBbind2020 by default.",
    )
    ap.add_argument("--data_root", type=str, default="datasets/qcMol/PubChem")
    ap.add_argument("--index_jsonl", type=str, default=None, help="Optional JSONL index from build_qcmol_fs_index.py")
    ap.add_argument("--index_shard_id", type=int, default=0, help="Index shard id (for multi-process sharding)")
    ap.add_argument("--index_shard_count", type=int, default=1, help="Index shard count (for multi-process sharding)")
    ap.add_argument("--out_dir", type=str, default="qcmol_packed_1")
    ap.add_argument("--xyz_key", type=str, default="XYZ")
    ap.add_argument("--atom_basic_key", type=str, default="atom_basic")
    ap.add_argument("--bond_basic_key", type=str, default="bond_basic")
    ap.add_argument("--atom_keys", type=str, default="NAO,NPA")
    ap.add_argument("--bond_keys", type=str, default="NBO")
    ap.add_argument("--aux_atom_keys", type=str, default="ADCH,LI")
    ap.add_argument("--aux_bond_keys", type=str, default="DI,LBO,Mayer")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_mols_per_file", type=int, default=None)
    ap.add_argument("--strict_keys", type=int, default=1)
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--probe_limit", type=int, default=200)
    ap.add_argument("--full_scan_dims", action="store_true")
    ap.add_argument("--dims_meta", type=str, default=None, help="Optional meta file to reuse dims and skip probing")
    ap.add_argument("--save_index_cache", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True, help="Resume from existing packed shards when possible")
    ap.add_argument("--no_resume", action="store_false", dest="resume", help="Disable resume and rebuild from scratch")
    ap.add_argument("--log_every_sec", type=float, default=10.0, help="Periodic log interval when tqdm is unavailable")
    ap.add_argument("--no_tqdm", action="store_true", help="Disable tqdm and use periodic logs")
    args = ap.parse_args()

    _apply_subset_preset(args)

    data_root = _resolve_path(args.data_root)
    out_dir = _resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    worker_id = int(args.index_shard_id)
    worker_count = int(args.index_shard_count)
    if worker_count <= 0:
        raise ValueError(f"--index_shard_count must be positive, got {worker_count}")
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError(f"--index_shard_id must be in [0, {worker_count - 1}], got {worker_id}")

    keys = QcMolKeys(
        xyz=args.xyz_key,
        atom_basic=args.atom_basic_key,
        bond_basic=args.bond_basic_key,
        atom_targets=tuple(k.strip() for k in args.atom_keys.split(",") if k.strip()),
        bond_targets=tuple(k.strip() for k in args.bond_keys.split(",") if k.strip()),
        aux_atom_targets=tuple(k.strip() for k in args.aux_atom_keys.split(",") if k.strip()),
        aux_bond_targets=tuple(k.strip() for k in args.aux_bond_keys.split(",") if k.strip()),
    )

    if args.index_jsonl:
        index_path = _resolve_path(args.index_jsonl)
        files = _load_index_jsonl(index_path)
        files = [p for p in files if p.endswith(".local")]
        files = [p if os.path.isabs(p) else os.path.join(data_root, p) for p in files]
    else:
        files = sorted(glob.glob(os.path.join(data_root, "*.local")))

    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise FileNotFoundError(f"No .local files found in {data_root}")

    all_files = files
    if worker_count > 1:
        files = [p for i, p in enumerate(all_files) if (i % worker_count) == worker_id]
        if not files:
            raise ValueError(
                f"Index sharding produced 0 files for worker {worker_id}/{worker_count}. "
                "Check --index_shard_id/--index_shard_count."
            )

    if args.no_tqdm:
        _log("[Info] tqdm disabled; using periodic logs instead of progress bars.")
    elif tqdm is None:
        _log("[Info] tqdm not available; using periodic logs instead of progress bars.")
    if worker_count > 1:
        _log(
            f"[Stage] Files collected: total={len(all_files)} worker_files={len(files)} worker={worker_id}/{worker_count} | root={data_root}"
        )
    else:
        _log(f"[Stage] Files collected: {len(files)} | root={data_root}")

    files_signature, cache_signature = _dataset_cache_signature(
        root=data_root,
        keys=keys,
        files=all_files,
        max_files=args.max_files,
        max_mols_per_file=args.max_mols_per_file,
        strict_keys=bool(args.strict_keys),
        cache_format="packed",
        cache_shard_size=int(args.shard_size),
    )

    resume_state_file = _resume_state_path(out_dir, cache_signature, worker_id, worker_count)
    resume_next_shard_id = 0
    resume_processed_samples = 0
    resume_start_file_idx = 0
    resume_start_row_offset = 0
    resume_skip_success = 0

    if args.resume:
        state = _load_resume_state(resume_state_file)
        if state is not None:
            state_sig = str(state.get("cache_signature", ""))
            state_worker_id = _to_int(state.get("index_shard_id", worker_id), worker_id)
            state_worker_count = _to_int(state.get("index_shard_count", worker_count), worker_count)
            if state_sig == cache_signature and state_worker_id == worker_id and state_worker_count == worker_count:
                resume_next_shard_id = max(0, _to_int(state.get("next_shard_id", 0), 0))
                resume_processed_samples = max(0, _to_int(state.get("processed_samples", 0), 0))
                resume_start_file_idx = max(0, _to_int(state.get("next_file_idx", 0), 0))
                resume_start_row_offset = max(0, _to_int(state.get("next_row_offset", 0), 0))
                completed = bool(state.get("completed", False))
                if completed:
                    _log(f"[Resume] found completed state: shards={resume_next_shard_id}, samples={resume_processed_samples}")
                else:
                    _log(
                        "[Resume] fast resume from state: "
                        f"next_file={resume_start_file_idx + 1}/{len(files)} next_row={resume_start_row_offset} "
                        f"next_shard={resume_next_shard_id} samples={resume_processed_samples}"
                    )

        if resume_next_shard_id == 0 and resume_processed_samples == 0 and resume_start_file_idx == 0:
            existing_shards, existing_samples = _scan_existing_shards(out_dir, cache_signature, worker_id, worker_count)
            if existing_shards > 0 and existing_samples > 0:
                resume_next_shard_id = existing_shards
                resume_processed_samples = existing_samples
                resume_skip_success = existing_samples
                _log(
                    "[Resume] fallback resume from existing shards: "
                    f"shards={existing_shards}, samples={existing_samples}. "
                    "Will skip already-written successful samples."
                )

    dims = {"atom": 0, "bond": 0, "aux_atom": 0, "aux_bond": 0}
    if args.dims_meta:
        try:
            meta = torch.load(_resolve_path(args.dims_meta), map_location="cpu")
            dims["atom"] = int(meta.get("atom_dim", 0) or 0)
            dims["bond"] = int(meta.get("bond_dim", 0) or 0)
            dims["aux_atom"] = int(meta.get("aux_atom_dim", 0) or 0)
            dims["aux_bond"] = int(meta.get("aux_bond_dim", 0) or 0)
        except Exception:
            dims = {"atom": 0, "bond": 0, "aux_atom": 0, "aux_bond": 0}

    if all(v > 0 for v in dims.values()) is False:
        if args.full_scan_dims:
            probe_limit = 10**12
        else:
            probe_limit = max(1, int(args.probe_limit))
        dims = _probe_dims(
            files=all_files,
            keys=keys,
            max_mols_per_file=args.max_mols_per_file,
            strict_keys=bool(args.strict_keys),
            probe_limit=probe_limit,
            log_every_sec=float(args.log_every_sec),
        )
    _log(
        f"[Stage] Dims resolved: atom={dims['atom']} bond={dims['bond']} aux_atom={dims['aux_atom']} aux_bond={dims['aux_bond']}"
    )

    atom_dim = dims["atom"]
    bond_dim = dims["bond"]
    aux_atom_dim = dims["aux_atom"]
    aux_bond_dim = dims["aux_bond"]

    shard_size = int(args.shard_size)
    shard_id = int(resume_next_shard_id)
    writer = ShardWriter(
        shard_size=shard_size,
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        aux_atom_dim=aux_atom_dim,
        aux_bond_dim=aux_bond_dim,
    )
    total_samples = int(resume_processed_samples)
    skipped = 0
    index: List[Tuple[str, Optional[int], str]] = []
    resume_remaining = int(resume_skip_success)

    def _next_position(file_idx_local: int, mode_local: str, row_idx_local: int, n_rows_local: int) -> Tuple[int, int]:
        if mode_local == "per_row":
            next_file_idx_local = file_idx_local - 1
            next_row_offset_local = int(row_idx_local) + 1
            if next_row_offset_local >= int(n_rows_local):
                next_file_idx_local = file_idx_local
                next_row_offset_local = 0
            return int(next_file_idx_local), int(next_row_offset_local)
        return int(file_idx_local), 0

    def _flush_writer(reason: str, next_file_idx: int, next_row_offset: int) -> None:
        nonlocal shard_id
        if writer.num_samples <= 0:
            return
        shard_path = _shard_path(out_dir, cache_signature, shard_id, worker_id, worker_count)
        try:
            payload: Dict[str, object] = dict(writer.finalize_payload())
            payload.update({
                "version": 1,
                "shard_size": shard_size,
                "num_samples": writer.num_samples,
            })
            _atomic_save(payload, shard_path)
        except Exception as exc:
            raise _FatalPreprocessError(
                f"[Fatal] failed to save shard (reason={reason}) shard_id={shard_id} path={shard_path}: {exc}"
            ) from exc

        writer.reset()
        shard_id += 1

        if args.resume:
            _save_resume_state(
                resume_state_file,
                {
                    "cache_signature": cache_signature,
                    "index_shard_id": worker_id,
                    "index_shard_count": worker_count,
                    "next_shard_id": shard_id,
                    "processed_samples": total_samples,
                    "next_file_idx": int(next_file_idx),
                    "next_row_offset": int(next_row_offset),
                    "completed": False,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

    if resume_start_file_idx >= len(files):
        _log("[Resume] start position is at/after dataset end; no new files to process.")
        files_iter = []
    else:
        files_iter = files[resume_start_file_idx:]

    iterator = enumerate(files_iter, start=resume_start_file_idx + 1)
    use_tqdm = (tqdm is not None) and (not args.no_tqdm)
    if use_tqdm:
        assert tqdm is not None
        iterator = tqdm(iterator, desc="Preprocessing qcMol (packed)", total=len(files_iter))
    else:
        _log("[Stage] Start preprocessing (packed) ...")

    last_log = time.time()

    for file_idx, path in iterator:
        if not use_tqdm and _should_log(last_log, float(args.log_every_sec)):
            _log(f"[Process] opening file {file_idx}/{len(files)}: {os.path.basename(path)}")
            last_log = time.time()
        try:
            with pd.HDFStore(path, mode="r") as store:
                df_xyz = _safe_read_df_store(store, keys.xyz, bool(args.strict_keys))
                if df_xyz is None:
                    skipped += 1
                    continue

                mode = _detect_mode(df_xyz)

                df_atom_basic = _safe_read_df_store(store, keys.atom_basic, bool(args.strict_keys))
                df_bond_basic = _safe_read_df_store(store, keys.bond_basic, bool(args.strict_keys))

                atom_tables = [_safe_read_df_store(store, k, bool(args.strict_keys)) for k in keys.atom_targets]
                bond_tables = [_safe_read_df_store(store, k, bool(args.strict_keys)) for k in keys.bond_targets]
                aux_atom_tables = [_safe_read_df_store(store, k, bool(args.strict_keys)) for k in keys.aux_atom_targets]
                aux_bond_tables = [_safe_read_df_store(store, k, bool(args.strict_keys)) for k in keys.aux_bond_targets]

                if mode == "per_file":
                    n_rows = 1
                    if (file_idx - 1) == resume_start_file_idx and resume_start_row_offset > 0:
                        row_indices = []
                    else:
                        row_indices = [None]
                else:
                    n_rows = len(df_xyz)
                    if args.max_mols_per_file is not None:
                        n_rows = min(n_rows, int(args.max_mols_per_file))
                    start_row = 0
                    if (file_idx - 1) == resume_start_file_idx and resume_start_row_offset > 0:
                        start_row = min(int(resume_start_row_offset), int(n_rows))
                    row_indices = list(range(start_row, n_rows))

                for row_idx in row_indices:
                    try:
                        ridx = _to_int(row_idx, 0)
                        if mode == "per_file":
                            pos_arr = _xyz_table_to_pos(df_xyz)
                        else:
                            pos_arr = _row_to_array(df_xyz.iloc[ridx])

                        if pos_arr is None or pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
                            skipped += 1
                            continue

                        num_atoms = int(pos_arr.shape[0])
                        pos = torch.as_tensor(pos_arr, dtype=torch.float32)

                        atom_basic_row = df_atom_basic if mode == "per_file" else (df_atom_basic.iloc[ridx] if df_atom_basic is not None else None)
                        z = _infer_z(atom_basic_row) if atom_basic_row is not None else None
                        if z is None or (hasattr(z, "shape") and int(np.asarray(z).shape[0]) != num_atoms):
                            z = np.zeros((num_atoms,), dtype=int)
                        z_t = torch.as_tensor(z, dtype=torch.long)

                        bond_basic_row = df_bond_basic if mode == "per_file" else (df_bond_basic.iloc[ridx] if df_bond_basic is not None else None)
                        atom_bond_index = _extract_bond_index(bond_basic_row, int(z_t.numel()))

                        atom_rows = [
                            (tbl if mode == "per_file" else (tbl.iloc[ridx] if tbl is not None else None))
                            for tbl in atom_tables
                        ]
                        atom_targets = _extract_atom_targets(atom_rows, int(z_t.numel()))

                        bond_rows = [
                            (tbl if mode == "per_file" else (tbl.iloc[ridx] if tbl is not None else None))
                            for tbl in bond_tables
                        ]
                        bond_targets = _extract_bond_targets_multi(bond_rows, int(z_t.numel()), atom_bond_index)

                        aux_atom_rows = [
                            (tbl if mode == "per_file" else (tbl.iloc[ridx] if tbl is not None else None))
                            for tbl in aux_atom_tables
                        ]
                        aux_atom_targets = _extract_atom_targets(aux_atom_rows, int(z_t.numel()))

                        aux_bond_rows = [
                            (tbl if mode == "per_file" else (tbl.iloc[ridx] if tbl is not None else None))
                            for tbl in aux_bond_tables
                        ]
                        aux_bond_targets = _extract_bond_targets_multi(aux_bond_rows, int(z_t.numel()), atom_bond_index)

                        atom_targets, atom_mask = _align_targets(atom_targets, atom_dim or None)
                        bond_targets, bond_mask = _align_targets(bond_targets, bond_dim or None)
                        aux_atom_targets, aux_atom_mask = _align_targets(aux_atom_targets, aux_atom_dim or None)
                        aux_bond_targets, aux_bond_mask = _align_targets(aux_bond_targets, aux_bond_dim or None)

                        if resume_remaining > 0:
                            resume_remaining -= 1
                            continue

                        writer.add_sample(
                            pos=pos,
                            z=z_t,
                            atom_bond_index=atom_bond_index,
                            atom_targets=atom_targets,
                            atom_mask=atom_mask,
                            bond_targets=bond_targets,
                            bond_mask=bond_mask,
                            aux_atom_targets=aux_atom_targets,
                            aux_atom_mask=aux_atom_mask,
                            aux_bond_targets=aux_bond_targets,
                            aux_bond_mask=aux_bond_mask,
                            sample_tag=f"{os.path.basename(path)}#{'file' if mode == 'per_file' else ridx}",
                        )
                        if args.save_index_cache:
                            index.append((path, row_idx if mode == "per_row" else None, mode))
                        total_samples += 1

                        if writer.is_full():
                            next_file_idx, next_row_offset = _next_position(file_idx, mode, ridx, int(n_rows))
                            _flush_writer("full", next_file_idx, next_row_offset)

                        if not use_tqdm and _should_log(last_log, float(args.log_every_sec)):
                            _log(
                                f"[Process] files={file_idx}/{len(files)} samples={total_samples} skipped={skipped} shards={shard_id}"
                            )
                            last_log = time.time()
                    except _FatalPreprocessError:
                        raise
                    except Exception:
                        skipped += 1
                        continue
        except _FatalPreprocessError:
            raise
        except Exception:
            skipped += 1
            continue

    wrote_last_partial = False
    if writer.num_samples > 0:
        _flush_writer("final_partial", len(files), 0)
        wrote_last_partial = True

    final_shards = shard_id

    if args.resume:
        _save_resume_state(
            resume_state_file,
            {
                "cache_signature": cache_signature,
                "index_shard_id": worker_id,
                "index_shard_count": worker_count,
                "next_shard_id": int(final_shards),
                "processed_samples": int(total_samples),
                "next_file_idx": int(len(files)),
                "next_row_offset": 0,
                "completed": True,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    meta = {
        "root": data_root,
        "keys": str(keys),
        "files_signature": files_signature,
        "cache_signature": cache_signature,
        "shard_size": shard_size,
        "num_samples": total_samples,
        "skipped": skipped,
        "atom_dim": atom_dim,
        "bond_dim": bond_dim,
        "aux_atom_dim": aux_atom_dim,
        "aux_bond_dim": aux_bond_dim,
        "max_files": args.max_files,
        "max_mols_per_file": args.max_mols_per_file,
        "strict_keys": bool(args.strict_keys),
        "index_shard_id": worker_id,
        "index_shard_count": worker_count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_pt = os.path.join(out_dir, "qcmol_packed_meta.pt" if worker_count == 1 else f"qcmol_packed_meta_w{worker_id:03d}.pt")
    meta_json = os.path.join(out_dir, "qcmol_packed_meta.json" if worker_count == 1 else f"qcmol_packed_meta_w{worker_id:03d}.json")
    _atomic_save(meta, meta_pt)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.save_index_cache:
        index_cache_path = os.path.join(
            out_dir,
            f"qcmol_packed_index_{cache_signature}.pt" if worker_count == 1 else f"qcmol_packed_index_{cache_signature}_w{worker_id:03d}.pt",
        )
        _atomic_save(
            {
                "files_signature": files_signature,
                "keys": str(keys),
                "index": index,
                "atom_dim": atom_dim,
                "bond_dim": bond_dim,
                "aux_atom_dim": aux_atom_dim,
                "aux_bond_dim": aux_bond_dim,
                "index_shard_id": worker_id,
                "index_shard_count": worker_count,
            },
            index_cache_path,
        )

    print(
        f"Done. samples={total_samples}, skipped={skipped}, shards={final_shards}, out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
