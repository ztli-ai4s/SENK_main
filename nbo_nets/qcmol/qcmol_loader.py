#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qcMol .local (HDF5) dataset loader for NBO head enhancement."""

from __future__ import annotations

import glob
import json
import hashlib
import os
import re
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bisect

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise ImportError("pandas is required for qcMol .local loader") from exc

try:
    from torch_geometric.data import Data
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for qcMol .local loader") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _torch_load_compat(path: str, map_location=None):
    """Compatibility wrapper to silence FutureWarning by explicitly setting weights_only."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class QcMolKeys:
    xyz: str = "XYZ"
    atom_basic: str = "atom_basic"
    bond_basic: str = "bond_basic"
    atom_targets: Tuple[str, ...] = ("NAO", "LP", "NPA")
    bond_targets: Tuple[str, ...] = ("NBO",)
    aux_atom_targets: Tuple[str, ...] = ("ADCH", "LI", "ELF")
    aux_bond_targets: Tuple[str, ...] = ("DI", "LBO", "Mayer")


def _read_hdf(path: str, key: str):
    try:
        return pd.read_hdf(path, key)
    except (KeyError, ValueError) as exc:
        # fallback: try matching key with leading slash or different case
        try:
            with pd.HDFStore(path, mode="r") as store:
                norm = str(key).lstrip("/").lower()
                matched = None
                for k in store.keys():
                    if k.lstrip("/").lower() == norm:
                        matched = k
                        break
                if matched is not None:
                    return store[matched]
        except Exception:
            pass
        raise exc


def _has_required_keys(path: str, keys: QcMolKeys) -> bool:
    required = [keys.xyz, keys.atom_basic, keys.bond_basic]
    required = [str(k).lstrip("/").lower() for k in required if k]
    try:
        with pd.HDFStore(path, mode="r") as store:
            available = {k.lstrip("/").lower() for k in store.keys()}
        return all(k in available for k in required)
    except Exception:
        return False


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    drop_cols = [c for c in ["Atom", "atom", "type"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def _series_to_matrix(row: pd.Series) -> Optional[np.ndarray]:
    # Handle per-column array entries (e.g., XYZ columns storing arrays per molecule)
    arrays: List[Tuple[str, np.ndarray]] = []
    lengths: List[int] = []
    for name, val in row.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val)
            if arr.ndim == 0:
                continue
            arrays.append((str(name), arr))
            lengths.append(int(arr.shape[0]))

    if not arrays:
        return None
    if len(set(lengths)) != 1:
        return None

    n = lengths[0]
    cols: List[np.ndarray] = []
    for name, arr in arrays:
        if name in {"Atom", "atom", "type"}:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if arr.ndim == 1:
            cols.append(arr.reshape(n, 1))
        elif arr.ndim == 2 and arr.shape[0] == n:
            cols.append(arr)

    if not cols:
        return None
    return np.concatenate(cols, axis=1)


def _row_to_array(row: Any) -> np.ndarray:
    if isinstance(row, pd.Series):
        arr = _series_to_matrix(row)
        if arr is not None:
            return np.nan_to_num(arr)
        if row.size == 1:
            row = row.iloc[0]
        else:
            df = row.to_frame().T
            df = _to_numeric_df(df)
            arr = df.to_numpy()
            return np.nan_to_num(arr.squeeze(0))
    if isinstance(row, pd.DataFrame):
        df = _to_numeric_df(row)
        arr = df.to_numpy()
        return np.nan_to_num(arr)
    return np.nan_to_num(np.asarray(row))


def _series_has_sequence(row: pd.Series) -> bool:
    for val in row.values:
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.size > 1:
                return True
    return False


def _xyz_columns(df: pd.DataFrame) -> Optional[List[str]]:
    if not hasattr(df, "columns"):
        return None
    cols = list(df.columns)
    lookup = {str(c).lower(): c for c in cols}
    if all(k in lookup for k in ("x", "y", "z")):
        return [lookup["x"], lookup["y"], lookup["z"]]
    return None


def _xyz_table_to_pos(df_xyz: pd.DataFrame) -> Optional[np.ndarray]:
    cols = _xyz_columns(df_xyz)
    if cols is None:
        return None
    try:
        arr = df_xyz[cols].to_numpy(dtype=float)
    except Exception:
        arr = df_xyz[cols].to_numpy()
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    return None


def _infer_z(atom_basic_row: Any) -> Optional[np.ndarray]:
    if isinstance(atom_basic_row, pd.DataFrame):
        cols = list(atom_basic_row.columns)
        for key in ["AtomicNum", "atomic_number", "Z", "z"]:
            if key in cols:
                return atom_basic_row[key].to_numpy(dtype=int)
    if isinstance(atom_basic_row, pd.Series):
        for key in ["AtomicNum", "atomic_number", "Z", "z"]:
            if key in atom_basic_row.index:
                val = atom_basic_row[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.asarray(val)
                    if arr.ndim >= 1:
                        return arr.astype(int)
        # single-row series; may contain a nested array
        arr = _row_to_array(atom_basic_row)
        if arr.ndim == 2 and arr.shape[1] > 0:
            # Heuristic: first column is atomic number
            return arr[:, 0].astype(int)
    return None


def _extract_bond_index(bond_basic_row: Any, num_atoms: int) -> torch.Tensor:
    if bond_basic_row is None:
        return torch.empty((2, 0), dtype=torch.long)

    if isinstance(bond_basic_row, pd.DataFrame):
        cols = list(bond_basic_row.columns)
        if "BeginAtomIdx" in cols and "EndAtomIdx" in cols:
            idx = bond_basic_row[["BeginAtomIdx", "EndAtomIdx"]].to_numpy()
        elif "begin" in cols and "end" in cols:
            idx = bond_basic_row[["begin", "end"]].to_numpy()
        else:
            idx = _row_to_array(bond_basic_row)
    elif isinstance(bond_basic_row, pd.Series):
        cols = list(bond_basic_row.index)
        if "BeginAtomIdx" in cols and "EndAtomIdx" in cols:
            begin = np.asarray(bond_basic_row["BeginAtomIdx"])
            end = np.asarray(bond_basic_row["EndAtomIdx"])
            if begin.ndim == 1 and end.ndim == 1 and begin.shape[0] == end.shape[0]:
                idx = np.stack([begin, end], axis=1)
            else:
                idx = _row_to_array(bond_basic_row)
        elif "begin" in cols and "end" in cols:
            begin = np.asarray(bond_basic_row["begin"])
            end = np.asarray(bond_basic_row["end"])
            if begin.ndim == 1 and end.ndim == 1 and begin.shape[0] == end.shape[0]:
                idx = np.stack([begin, end], axis=1)
            else:
                idx = _row_to_array(bond_basic_row)
        else:
            idx = _row_to_array(bond_basic_row)
    else:
        idx = _row_to_array(bond_basic_row)

    if idx.ndim == 1:
        idx = idx.reshape(-1, 2) if idx.size % 2 == 0 else np.empty((0, 2))
    if idx.ndim >= 2 and idx.shape[1] > 2:
        idx = idx[:, :2]

    if idx.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Fix 1-based indices if needed
    idx = idx.astype(int)
    if idx.max() >= num_atoms and idx.min() >= 1:
        idx = idx - 1

    # Remove self-loops and invalid indices
    mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] >= 0) & (idx[:, 1] >= 0)
    mask &= (idx[:, 0] < num_atoms) & (idx[:, 1] < num_atoms)
    idx = idx[mask]

    if idx.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.as_tensor(idx, dtype=torch.long).t().contiguous()


def _parse_pair_label(label: str) -> Optional[Tuple[int, int]]:
    nums = re.findall(r"\d+", str(label))
    if len(nums) >= 2:
        return int(nums[0]) - 1, int(nums[1]) - 1
    return None


def _extract_bond_targets(
    nbo_row: Any,
    num_atoms: int,
    bond_index: torch.Tensor,
) -> Optional[torch.Tensor]:
    if nbo_row is None or bond_index.numel() == 0:
        return None

    if isinstance(nbo_row, pd.DataFrame) and "atom" in nbo_row.columns:
        # Try parsing labels like C1-C2
        pairs = [_parse_pair_label(v) for v in nbo_row["atom"].tolist()]
        valid = [(p, i) for i, p in enumerate(pairs) if p is not None]
        if valid:
            pair_to_idx = {p: i for p, i in valid}
            feats_df = nbo_row.drop(columns=["atom", "type"], errors="ignore")
            feats = _row_to_array(feats_df)
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            out = []
            for u, v in bond_index.t().tolist():
                key = (u, v)
                ridx = pair_to_idx.get(key)
                if ridx is None:
                    out.append(np.zeros((feats.shape[1],), dtype=feats.dtype))
                else:
                    out.append(feats[ridx])
            return torch.as_tensor(np.stack(out, axis=0), dtype=torch.float32)

    arr = _row_to_array(nbo_row)
    if arr.ndim == 3 and arr.shape[0] == num_atoms and arr.shape[1] == num_atoms:
        return torch.as_tensor(arr[bond_index[0], bond_index[1]], dtype=torch.float32)
    if arr.ndim == 2 and arr.shape[0] == num_atoms and arr.shape[1] == num_atoms:
        sel = arr[bond_index[0], bond_index[1]]
        if sel.ndim == 1:
            sel = sel.reshape(-1, 1)
        return torch.as_tensor(sel, dtype=torch.float32)
    if arr.ndim == 2 and arr.shape[0] == num_atoms * num_atoms:
        flat_idx = bond_index[0] * num_atoms + bond_index[1]
        return torch.as_tensor(arr[flat_idx], dtype=torch.float32)
    if arr.ndim == 2 and arr.shape[0] == bond_index.size(1):
        return torch.as_tensor(arr, dtype=torch.float32)
    if arr.ndim == 1 and arr.shape[0] == bond_index.size(1):
        return torch.as_tensor(arr.reshape(-1, 1), dtype=torch.float32)

    return None


def _extract_atom_targets(
    rows: List[Any],
    num_atoms: int,
) -> Optional[torch.Tensor]:
    feats: List[np.ndarray] = []
    for row in rows:
        if row is None:
            continue
        arr = _row_to_array(row)
        if arr.ndim == 1:
            if arr.shape[0] == num_atoms:
                arr = arr.reshape(num_atoms, 1)
            else:
                continue
        if arr.ndim == 2 and arr.shape[0] == num_atoms:
            feats.append(arr)
    if not feats:
        return None
    return torch.as_tensor(np.concatenate(feats, axis=-1), dtype=torch.float32)


def _read_optional_row(path: str, key: str, row_idx: int) -> Any:
    try:
        df = _read_hdf(path, key)
        return df.iloc[row_idx]
    except Exception:
        return None


def _read_optional_table(path: str, key: str) -> Any:
    try:
        return _read_hdf(path, key)
    except Exception:
        return None


def _extract_bond_targets_multi(
    rows: List[Any],
    num_atoms: int,
    bond_index: torch.Tensor,
) -> Optional[torch.Tensor]:
    feats: List[torch.Tensor] = []
    for row in rows:
        if row is None:
            continue
        target = _extract_bond_targets(row, num_atoms, bond_index)
        if target is None:
            continue
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if target.shape[0] != bond_index.size(1):
            continue
        feats.append(target)
    if not feats:
        return None
    return torch.cat(feats, dim=-1)


class QcMolLocalDataset(Dataset):
    def __init__(
        self,
        root: str,
        keys: Optional[QcMolKeys] = None,
        max_files: Optional[int] = None,
        max_mols_per_file: Optional[int] = None,
        cache_dir: Optional[str] = None,
        cache_mode: str = "none",
        cache_index: bool = True,
        cache_format: str = "sharded",
        cache_shard_size: int = 512,
        cache_shard_keep: int = 2,
        strict_keys: bool = True,
        verbose: bool = True,
    ):
        self.root = os.path.abspath(root)
        self.keys = keys or QcMolKeys()
        self.max_files = max_files
        self.max_mols_per_file = max_mols_per_file
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.cache_mode = str(cache_mode or "none").lower()
        self.cache_index = bool(cache_index)
        self.cache_format = str(cache_format or "sharded").lower()
        self.cache_shard_size = int(cache_shard_size or 512)
        self.cache_shard_keep = max(1, int(cache_shard_keep or 1))
        self.strict_keys = bool(strict_keys)
        self.verbose = bool(verbose)
        if self.cache_mode not in {"none", "read", "write", "readwrite"}:
            raise ValueError(f"cache_mode must be one of none/read/write/readwrite, got {self.cache_mode}")
        if self.cache_format not in {"sharded", "per_sample"}:
            raise ValueError(f"cache_format must be one of sharded/per_sample, got {self.cache_format}")
        if self.cache_shard_size <= 0:
            raise ValueError(f"cache_shard_size must be positive, got {self.cache_shard_size}")
        if self.cache_dir and self.cache_mode in {"write", "readwrite"}:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.files = sorted(glob.glob(os.path.join(self.root, "*.local")))
        if max_files is not None:
            self.files = self.files[: int(max_files)]
        files_signature = hashlib.sha1("|".join(self.files).encode("utf-8")).hexdigest()

        if not self.files:
            raise FileNotFoundError(f"No .local files found in {self.root}")

        self.files_signature = files_signature
        self.cache_signature = self._dataset_cache_signature(files_signature)
        self._shard_cache: "OrderedDict[int, List[Optional[Data]]]" = OrderedDict()

        self.index: List[Tuple[str, Optional[int], str]] = []
        scan_errors: List[str] = []
        cache_path = self._index_cache_path(files_signature)
        self.index_cache_path = cache_path
        self.index_cache_used = False
        if cache_path and os.path.exists(cache_path):
            try:
                payload = _torch_load_compat(cache_path, map_location="cpu")
                if payload.get("files_signature") == files_signature and payload.get("keys") == str(self.keys):
                    self.index = payload.get("index", [])
                    self.atom_dim = payload.get("atom_dim")
                    self.bond_dim = payload.get("bond_dim")
                    self.aux_atom_dim = payload.get("aux_atom_dim")
                    self.aux_bond_dim = payload.get("aux_bond_dim")
                    self.index_cache_used = True
                    if self.verbose:
                        print(f"qcMol index cache hit: {cache_path}")
            except Exception:
                self.index = []

        if not self.index:
            if self.verbose:
                if cache_path:
                    print(f"qcMol index cache miss: {cache_path}")
                else:
                    print("qcMol index cache disabled")
            iterator = self.files
            if self.verbose and tqdm is not None:
                iterator = tqdm(self.files, desc="Scanning qcMol files")
            for path in iterator:
                try:
                    if self.strict_keys and not _has_required_keys(path, self.keys):
                        scan_errors.append(f"{os.path.basename(path)}: missing required keys")
                        continue
                    df_xyz = _read_hdf(path, self.keys.xyz)
                    per_file = False
                    if isinstance(df_xyz, pd.DataFrame):
                        cols = _xyz_columns(df_xyz)
                        if cols is not None and not _series_has_sequence(df_xyz.iloc[0]):
                            per_file = True

                    if per_file:
                        self.index.append((path, None, "per_file"))
                    else:
                        n = len(df_xyz)
                        if max_mols_per_file is not None:
                            n = min(n, int(max_mols_per_file))
                        self.index.extend([(path, i, "per_row") for i in range(n)])
                except Exception as exc:
                    scan_errors.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
                    continue

        if not self.index:
            preview = "\n".join(scan_errors[:3]) if scan_errors else "(no detailed errors captured)"
            raise ValueError(
                "qcMol dataset is empty after scanning .local files. "
                "Please verify data_root, xyz key, and that the .local files contain molecules. "
                f"root={self.root}, xyz_key={self.keys.xyz}, files={len(self.files)}\n"
                f"first_errors:\n{preview}"
            )

        self.atom_dim: Optional[int] = None if not hasattr(self, "atom_dim") else self.atom_dim
        self.bond_dim: Optional[int] = None if not hasattr(self, "bond_dim") else self.bond_dim
        self.aux_atom_dim: Optional[int] = None if not hasattr(self, "aux_atom_dim") else self.aux_atom_dim
        self.aux_bond_dim: Optional[int] = None if not hasattr(self, "aux_bond_dim") else self.aux_bond_dim
        self._warned_dim: Dict[str, bool] = {
            "atom_targets": False,
            "bond_targets": False,
            "aux_atom_targets": False,
            "aux_bond_targets": False,
        }
        if self.atom_dim is None or self.bond_dim is None or self.aux_atom_dim is None or self.aux_bond_dim is None:
            probe_limit = min(len(self.index), 200)
            self._probe_dims(max_probe=probe_limit)
        if cache_path and self.cache_index:
            try:
                torch.save(
                    {
                        "files_signature": files_signature,
                        "keys": str(self.keys),
                        "index": self.index,
                        "atom_dim": self.atom_dim,
                        "bond_dim": self.bond_dim,
                        "aux_atom_dim": self.aux_atom_dim,
                        "aux_bond_dim": self.aux_bond_dim,
                    },
                    cache_path,
                )
                if self.verbose:
                    print(f"qcMol index cache saved: {cache_path}")
            except Exception:
                pass

    def _index_cache_path(self, files_signature: str) -> Optional[str]:
        if not self.cache_dir or not self.cache_index:
            return None
        sig = "|".join(
            [
                "qcmol_index_v2",
                self.root,
                str(self.keys),
                str(self.max_files),
                str(self.max_mols_per_file),
                str(self.strict_keys),
                files_signature,
            ]
        )
        fname = hashlib.sha1(sig.encode("utf-8")).hexdigest() + ".pt"
        return os.path.join(self.cache_dir, fname)

    def _dataset_cache_signature(self, files_signature: str) -> str:
        mtimes: List[str] = []
        for path in self.files:
            try:
                mtimes.append(str(os.path.getmtime(path)))
            except Exception:
                mtimes.append("0")
        mtime_sig = hashlib.sha1("|".join(mtimes).encode("utf-8")).hexdigest()
        sig = "|".join(
            [
                "qcmol_cache_v3",
                self.root,
                str(self.keys),
                str(self.max_files),
                str(self.max_mols_per_file),
                str(self.strict_keys),
                str(self.cache_format),
                str(self.cache_shard_size),
                files_signature,
                mtime_sig,
            ]
        )
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()

    def _cache_key(self, path: str, row_idx: Optional[int], mode: str) -> str:
        mtime = 0.0
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            pass
        key_str = "|".join(
            [
                os.path.abspath(path),
                str(row_idx),
                mode,
                str(mtime),
                str(self.keys),
            ]
        )
        return hashlib.sha1(key_str.encode("utf-8")).hexdigest()

    def _cache_path(self, path: str, row_idx: Optional[int], mode: str) -> Optional[str]:
        if not self.cache_dir or self.cache_mode == "none":
            return None
        fname = self._cache_key(path, row_idx, mode) + ".pt"
        return os.path.join(self.cache_dir, fname)

    def _shard_id(self, idx: int) -> Tuple[int, int]:
        shard_id = int(idx // self.cache_shard_size)
        offset = int(idx % self.cache_shard_size)
        return shard_id, offset

    def _shard_cache_path(self, shard_id: int) -> Optional[str]:
        if not self.cache_dir or self.cache_mode == "none":
            return None
        fname = f"qcmol_shard_{self.cache_signature}_{shard_id:06d}.pt"
        return os.path.join(self.cache_dir, fname)

    def _acquire_lock(self, lock_path: str, retries: int = 40, delay: float = 0.05) -> bool:
        for _ in range(retries):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                time.sleep(delay)
        return False

    def _release_lock(self, lock_path: str) -> None:
        try:
            os.remove(lock_path)
        except Exception:
            pass

    def _atomic_save(self, payload: object, path: str) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _load_shard_from_disk(self, shard_id: int) -> Optional[List[Optional[Data]]]:
        shard_path = self._shard_cache_path(shard_id)
        if not shard_path or not os.path.exists(shard_path):
            return None
        try:
            payload = _torch_load_compat(shard_path, map_location="cpu")
        except Exception:
            return None
        samples = None
        if isinstance(payload, dict) and "samples" in payload:
            samples = payload.get("samples")
        elif isinstance(payload, list):
            samples = payload
        if not isinstance(samples, list):
            return None
        if len(samples) < self.cache_shard_size:
            samples = samples + [None] * (self.cache_shard_size - len(samples))
        return samples

    def _get_cached_shard(self, shard_id: int) -> Optional[List[Optional[Data]]]:
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]
        samples = None
        if self.cache_mode in {"read", "readwrite"}:
            samples = self._load_shard_from_disk(shard_id)
        if samples is not None:
            self._shard_cache[shard_id] = samples
            self._shard_cache.move_to_end(shard_id)
            while len(self._shard_cache) > self.cache_shard_keep:
                self._shard_cache.popitem(last=False)
        return samples

    def _write_shard_sample(self, shard_id: int, offset: int, data: Data) -> None:
        shard_path = self._shard_cache_path(shard_id)
        if not shard_path:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        lock_path = shard_path + ".lock"
        if not self._acquire_lock(lock_path):
            return
        try:
            samples = self._shard_cache.get(shard_id)
            if samples is None:
                samples = self._load_shard_from_disk(shard_id)
            if samples is None:
                samples = [None] * self.cache_shard_size
            if offset >= len(samples):
                samples.extend([None] * (offset + 1 - len(samples)))
            samples[offset] = data
            payload = {
                "version": 1,
                "shard_size": self.cache_shard_size,
                "samples": samples,
            }
            self._atomic_save(payload, shard_path)
        finally:
            self._release_lock(lock_path)

        self._shard_cache[shard_id] = samples
        self._shard_cache.move_to_end(shard_id)
        while len(self._shard_cache) > self.cache_shard_keep:
            self._shard_cache.popitem(last=False)

    def _probe_dims(self, max_probe: int = 25) -> None:
        if not self.index:
            return
        for path, row_idx, mode in self.index[:max_probe]:
            try:
                df_xyz = _read_hdf(path, self.keys.xyz)
                if mode == "per_file":
                    pos_arr = _xyz_table_to_pos(df_xyz)
                else:
                    xyz_row = df_xyz.iloc[int(row_idx)]
                    pos_arr = _row_to_array(xyz_row)

                if pos_arr is None or pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
                    continue
                num_atoms = int(pos_arr.shape[0])

                df_bond_basic = _read_hdf(path, self.keys.bond_basic)
                bond_basic_row = df_bond_basic if mode == "per_file" else df_bond_basic.iloc[int(row_idx)]
                bond_index = _extract_bond_index(bond_basic_row, num_atoms)

                atom_rows = [
                    (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
                    for key in self.keys.atom_targets
                ]
                atom_targets = _extract_atom_targets(atom_rows, num_atoms)
                if atom_targets is not None:
                    dim = int(atom_targets.shape[1])
                    self.atom_dim = dim if self.atom_dim is None else max(self.atom_dim, dim)

                bond_rows = [
                    (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
                    for key in self.keys.bond_targets
                ]
                bond_targets = _extract_bond_targets_multi(bond_rows, num_atoms, bond_index)
                if bond_targets is not None:
                    dim = int(bond_targets.shape[1])
                    self.bond_dim = dim if self.bond_dim is None else max(self.bond_dim, dim)

                aux_atom_rows = [
                    (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
                    for key in self.keys.aux_atom_targets
                ]
                aux_atom_targets = _extract_atom_targets(aux_atom_rows, num_atoms)
                if aux_atom_targets is not None:
                    dim = int(aux_atom_targets.shape[1])
                    self.aux_atom_dim = dim if self.aux_atom_dim is None else max(self.aux_atom_dim, dim)

                aux_bond_rows = [
                    (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
                    for key in self.keys.aux_bond_targets
                ]
                aux_bond_targets = _extract_bond_targets_multi(aux_bond_rows, num_atoms, bond_index)
                if aux_bond_targets is not None:
                    dim = int(aux_bond_targets.shape[1])
                    self.aux_bond_dim = dim if self.aux_bond_dim is None else max(self.aux_bond_dim, dim)

                # keep scanning to find max dims across samples
            except Exception:
                continue

    def _align_targets(
        self,
        targets: Optional[torch.Tensor],
        target_dim: Optional[int],
        name: str,
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

        if not self._warned_dim.get(name, False):
            warnings.warn(
                f"qcMol target dim mismatch for {name}: got {cur_dim}, expected {target_dim}. "
                "Will pad/trim to expected dim.",
                RuntimeWarning,
            )
            self._warned_dim[name] = True

        if cur_dim > target_dim:
            trimmed = targets[:, :target_dim]
            mask = torch.ones((trimmed.size(0), target_dim), dtype=torch.float32)
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
        return padded, mask

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Data:
        path, row_idx, mode = self.index[idx]

        shard_id: Optional[int] = None
        shard_offset: Optional[int] = None
        cache_path = None
        if self.cache_format == "sharded":
            shard_id, shard_offset = self._shard_id(idx)
            if self.cache_mode in {"read", "readwrite"}:
                samples = self._get_cached_shard(shard_id)
                if samples is not None and shard_offset < len(samples):
                    cached = samples[shard_offset]
                    if cached is not None:
                        return cached
        else:
            cache_path = self._cache_path(path, row_idx, mode)
            if cache_path and self.cache_mode in {"read", "readwrite"} and os.path.exists(cache_path):
                try:
                    return _torch_load_compat(cache_path, map_location="cpu")
                except Exception:
                    pass

        df_xyz = _read_hdf(path, self.keys.xyz)
        if mode == "per_file":
            pos_arr = _xyz_table_to_pos(df_xyz)
        else:
            xyz_row = df_xyz.iloc[int(row_idx)]
            pos_arr = _row_to_array(xyz_row)

        if pos_arr is None:
            raise ValueError("XYZ data could not be parsed into positions")
        pos = torch.as_tensor(pos_arr, dtype=torch.float32)
        if pos.ndim != 2 or pos.size(1) != 3:
            raise ValueError(f"XYZ row has unexpected shape: {tuple(pos.shape)}")

        try:
            df_atom_basic = _read_hdf(path, self.keys.atom_basic)
        except Exception as exc:
            if self.strict_keys:
                raise exc
            df_atom_basic = None
        atom_basic_row = df_atom_basic if mode == "per_file" else (df_atom_basic.iloc[int(row_idx)] if df_atom_basic is not None else None)
        z = _infer_z(atom_basic_row) if atom_basic_row is not None else None
        if z is None:
            # fallback: zeros, but keep length
            z = np.zeros((pos.size(0),), dtype=int)
        z_t = torch.as_tensor(z, dtype=torch.long)

        try:
            df_bond_basic = _read_hdf(path, self.keys.bond_basic)
        except Exception as exc:
            if self.strict_keys:
                raise exc
            df_bond_basic = None
        bond_basic_row = df_bond_basic if mode == "per_file" else (df_bond_basic.iloc[int(row_idx)] if df_bond_basic is not None else None)
        atom_bond_index = _extract_bond_index(bond_basic_row, int(z_t.numel()))

        atom_rows = [
            (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
            for key in self.keys.atom_targets
        ]
        atom_targets = _extract_atom_targets(atom_rows, int(z_t.numel()))

        bond_rows = [
            (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
            for key in self.keys.bond_targets
        ]
        bond_targets = _extract_bond_targets_multi(bond_rows, int(z_t.numel()), atom_bond_index)

        aux_atom_rows = [
            (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
            for key in self.keys.aux_atom_targets
        ]
        aux_atom_targets = _extract_atom_targets(aux_atom_rows, int(z_t.numel()))

        aux_bond_rows = [
            (_read_optional_table(path, key) if mode == "per_file" else _read_optional_row(path, key, int(row_idx)))
            for key in self.keys.aux_bond_targets
        ]
        aux_bond_targets = _extract_bond_targets_multi(aux_bond_rows, int(z_t.numel()), atom_bond_index)

        node_type = torch.zeros((z_t.numel(),), dtype=torch.long)
        x = z_t.view(-1, 1).to(torch.float32)

        data = Data(
            pos=pos,
            z=z_t,
            x=x,
            node_type=node_type,
            atom_bond_index=atom_bond_index,
        )
        atom_targets, atom_mask = self._align_targets(atom_targets, self.atom_dim, "atom_targets")
        if atom_targets is not None:
            data.atom_targets = atom_targets
            data.atom_target_mask = atom_mask
        elif self.atom_dim is not None:
            data.atom_targets = torch.zeros((z_t.numel(), self.atom_dim), dtype=torch.float32)
            data.atom_target_mask = torch.zeros((z_t.numel(), 1), dtype=torch.float32)

        bond_targets, bond_mask = self._align_targets(bond_targets, self.bond_dim, "bond_targets")
        if bond_targets is not None:
            data.bond_targets = bond_targets
            data.bond_target_mask = bond_mask
        elif self.bond_dim is not None:
            data.bond_targets = torch.zeros((atom_bond_index.size(1), self.bond_dim), dtype=torch.float32)
            data.bond_target_mask = torch.zeros((atom_bond_index.size(1), 1), dtype=torch.float32)

        aux_atom_targets, aux_atom_mask = self._align_targets(aux_atom_targets, self.aux_atom_dim, "aux_atom_targets")
        if aux_atom_targets is not None:
            data.aux_atom_targets = aux_atom_targets
            data.aux_atom_target_mask = aux_atom_mask
        elif self.aux_atom_dim is not None:
            data.aux_atom_targets = torch.zeros((z_t.numel(), self.aux_atom_dim), dtype=torch.float32)
            data.aux_atom_target_mask = torch.zeros((z_t.numel(), 1), dtype=torch.float32)

        aux_bond_targets, aux_bond_mask = self._align_targets(aux_bond_targets, self.aux_bond_dim, "aux_bond_targets")
        if aux_bond_targets is not None:
            data.aux_bond_targets = aux_bond_targets
            data.aux_bond_target_mask = aux_bond_mask
        elif self.aux_bond_dim is not None:
            data.aux_bond_targets = torch.zeros((atom_bond_index.size(1), self.aux_bond_dim), dtype=torch.float32)
            data.aux_bond_target_mask = torch.zeros((atom_bond_index.size(1), 1), dtype=torch.float32)
        if self.cache_mode in {"write", "readwrite"}:
            if self.cache_format == "sharded":
                if shard_id is None or shard_offset is None:
                    shard_id, shard_offset = self._shard_id(idx)
                self._write_shard_sample(shard_id, shard_offset, data)
            elif cache_path:
                try:
                    torch.save(data, cache_path)
                except Exception:
                    pass

        return data


class QcMolPackedDataset(Dataset):
    """Dataset for packed qcMol shards produced by preprocess_qcmol_packed_safe.py."""

    def __init__(
        self,
        root: str,
        shard_glob: str = "qcmol_packed_shard_*.pt",
        meta_path: Optional[str] = None,
        cache_shards: int = 2,
        verbose: bool = True,
    ) -> None:
        self.root = os.path.abspath(root)
        self.shard_glob = shard_glob or "qcmol_packed_shard_*.pt"
        self.cache_shards = max(1, int(cache_shards))
        self.verbose = bool(verbose)
        self._shard_cache: "OrderedDict[int, Dict[str, torch.Tensor]]" = OrderedDict()

        pattern = self.shard_glob
        if not os.path.isabs(pattern):
            pattern = os.path.join(self.root, pattern)
        all_matches = sorted(glob.glob(pattern))
        self.shards = [
            path for path in all_matches
            if "shard" in os.path.basename(path).lower()
        ]
        self.files = list(self.shards)
        if not self.shards:
            raise FileNotFoundError(f"No packed shard files found under {self.root} with pattern {self.shard_glob}")

        meta = self._load_meta(meta_path)
        self.atom_dim = int(meta.get("atom_dim", 0) or 0) if meta else 0
        self.bond_dim = int(meta.get("bond_dim", 0) or 0) if meta else 0
        self.aux_atom_dim = int(meta.get("aux_atom_dim", 0) or 0) if meta else 0
        self.aux_bond_dim = int(meta.get("aux_bond_dim", 0) or 0) if meta else 0
        self.shard_size = int(meta.get("shard_size", 0) or 0) if meta else 0
        self.total_samples = int(meta.get("num_samples", 0) or 0) if meta else 0

        shard_sizes = self._infer_shard_sizes(meta)
        self._shard_offsets: List[int] = [0]
        for size in shard_sizes:
            self._shard_offsets.append(self._shard_offsets[-1] + int(size))

        if self._shard_offsets[-1] <= 0:
            raise ValueError("Packed qcMol dataset has 0 samples. Check shard files and meta.")

        if self.atom_dim == 0 and self.bond_dim == 0 and self.aux_atom_dim == 0 and self.aux_bond_dim == 0:
            # best-effort inference from the first shard
            payload = self._load_shard_payload(0)
            self.atom_dim = int(payload.get("atom_targets").shape[1]) if "atom_targets" in payload else 0
            self.bond_dim = int(payload.get("bond_targets").shape[1]) if "bond_targets" in payload else 0
            self.aux_atom_dim = int(payload.get("aux_atom_targets").shape[1]) if "aux_atom_targets" in payload else 0
            self.aux_bond_dim = int(payload.get("aux_bond_targets").shape[1]) if "aux_bond_targets" in payload else 0

    def _load_meta(self, meta_path: Optional[str]) -> Dict[str, Any]:
        if meta_path:
            path = meta_path
            if not os.path.isabs(path):
                path = os.path.join(self.root, path)
            return self._read_meta_file(path)

        default_pt = os.path.join(self.root, "qcmol_packed_meta.pt")
        default_json = os.path.join(self.root, "qcmol_packed_meta.json")
        if os.path.exists(default_pt):
            return self._read_meta_file(default_pt)
        if os.path.exists(default_json):
            return self._read_meta_file(default_json)
        return {}

    def _read_meta_file(self, path: str) -> Dict[str, Any]:
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return _torch_load_compat(path, map_location="cpu")
        except Exception:
            return {}

    def _infer_shard_sizes(self, meta: Dict[str, Any]) -> List[int]:
        shard_count = len(self.shards)
        shard_sizes: List[int] = []

        if meta and self.shard_size > 0 and self.total_samples > 0:
            full = max(0, shard_count - 1)
            last = self.total_samples - self.shard_size * full
            if last <= 0:
                last = self.shard_size
            shard_sizes = [self.shard_size] * full + [last]
            if sum(shard_sizes) == self.total_samples and len(shard_sizes) == shard_count:
                return shard_sizes
            shard_sizes = []

        # fallback: load each shard to read num_samples / ptr sizes
        for shard_id in range(shard_count):
            payload = self._load_shard_payload(shard_id)
            if "num_samples" in payload:
                shard_sizes.append(int(payload.get("num_samples", 0) or 0))
            elif isinstance(payload.get("samples"), list):
                shard_sizes.append(len(payload.get("samples") or []))
            else:
                atom_ptr = payload.get("atom_ptr")
                if atom_ptr is None:
                    shard_sizes.append(0)
                else:
                    shard_sizes.append(int(atom_ptr.numel() - 1))
        return shard_sizes

    def _load_shard_payload(self, shard_id: int) -> Dict[str, torch.Tensor]:
        shard_path = self.shards[shard_id]
        payload = _torch_load_compat(shard_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid shard payload: {shard_path}")
        return payload

    def _get_shard(self, shard_id: int) -> Dict[str, torch.Tensor]:
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]

        payload = self._load_shard_payload(shard_id)
        self._shard_cache[shard_id] = payload
        self._shard_cache.move_to_end(shard_id)
        while len(self._shard_cache) > self.cache_shards:
            self._shard_cache.popitem(last=False)
        return payload

    def __len__(self) -> int:
        return int(self._shard_offsets[-1])

    def __getitem__(self, idx: int) -> Data:
        if idx < 0:
            idx = len(self) + idx
        shard_id = bisect.bisect_right(self._shard_offsets, idx) - 1
        if shard_id < 0 or shard_id >= len(self.shards):
            raise IndexError(f"Index out of range: {idx}")
        local_idx = idx - self._shard_offsets[shard_id]

        payload = self._get_shard(shard_id)
        legacy_samples = payload.get("samples")
        if isinstance(legacy_samples, list):
            if local_idx < 0 or local_idx >= len(legacy_samples):
                raise IndexError(f"Packed shard local index out of range: shard={shard_id} local_idx={local_idx}")
            data = legacy_samples[local_idx]
            if not isinstance(data, Data):
                raise ValueError(f"Legacy packed shard sample is not a torch_geometric Data object: shard={shard_id} idx={local_idx}")
            if not hasattr(data, "node_type"):
                data.node_type = torch.zeros((int(data.z.numel()),), dtype=torch.long)
            if not hasattr(data, "x"):
                data.x = data.z.view(-1, 1).to(torch.float32)
            return data

        atom_ptr = payload.get("atom_ptr")
        bond_ptr = payload.get("bond_ptr")
        if atom_ptr is None or bond_ptr is None:
            raise ValueError("Packed shard missing atom_ptr/bond_ptr")

        atom_start = int(atom_ptr[local_idx].item())
        atom_end = int(atom_ptr[local_idx + 1].item())
        bond_start = int(bond_ptr[local_idx].item())
        bond_end = int(bond_ptr[local_idx + 1].item())

        pos = payload.get("pos")[atom_start:atom_end]
        z = payload.get("z")[atom_start:atom_end]
        bond_index = payload.get("bond_index")[:, bond_start:bond_end]
        if bond_index.numel() > 0:
            bond_index = bond_index - atom_start

        node_type = torch.zeros((z.numel(),), dtype=torch.long)
        x = z.view(-1, 1).to(torch.float32)

        data = Data(
            pos=pos,
            z=z,
            x=x,
            node_type=node_type,
            atom_bond_index=bond_index,
        )

        if "atom_targets" in payload:
            data.atom_targets = payload["atom_targets"][atom_start:atom_end]
            if "atom_target_mask" in payload:
                data.atom_target_mask = payload["atom_target_mask"][atom_start:atom_end]
            else:
                data.atom_target_mask = torch.ones((atom_end - atom_start, 1), dtype=torch.float32)
        elif self.atom_dim > 0:
            data.atom_targets = torch.zeros((atom_end - atom_start, self.atom_dim), dtype=torch.float32)
            data.atom_target_mask = torch.zeros((atom_end - atom_start, 1), dtype=torch.float32)

        if "bond_targets" in payload:
            data.bond_targets = payload["bond_targets"][bond_start:bond_end]
            if "bond_target_mask" in payload:
                data.bond_target_mask = payload["bond_target_mask"][bond_start:bond_end]
            else:
                data.bond_target_mask = torch.ones((bond_end - bond_start, 1), dtype=torch.float32)
        elif self.bond_dim > 0:
            data.bond_targets = torch.zeros((bond_end - bond_start, self.bond_dim), dtype=torch.float32)
            data.bond_target_mask = torch.zeros((bond_end - bond_start, 1), dtype=torch.float32)

        if "aux_atom_targets" in payload:
            data.aux_atom_targets = payload["aux_atom_targets"][atom_start:atom_end]
            if "aux_atom_target_mask" in payload:
                data.aux_atom_target_mask = payload["aux_atom_target_mask"][atom_start:atom_end]
            else:
                data.aux_atom_target_mask = torch.ones((atom_end - atom_start, 1), dtype=torch.float32)
        elif self.aux_atom_dim > 0:
            data.aux_atom_targets = torch.zeros((atom_end - atom_start, self.aux_atom_dim), dtype=torch.float32)
            data.aux_atom_target_mask = torch.zeros((atom_end - atom_start, 1), dtype=torch.float32)

        if "aux_bond_targets" in payload:
            data.aux_bond_targets = payload["aux_bond_targets"][bond_start:bond_end]
            if "aux_bond_target_mask" in payload:
                data.aux_bond_target_mask = payload["aux_bond_target_mask"][bond_start:bond_end]
            else:
                data.aux_bond_target_mask = torch.ones((bond_end - bond_start, 1), dtype=torch.float32)
        elif self.aux_bond_dim > 0:
            data.aux_bond_targets = torch.zeros((bond_end - bond_start, self.aux_bond_dim), dtype=torch.float32)
            data.aux_bond_target_mask = torch.zeros((bond_end - bond_start, 1), dtype=torch.float32)

        return data
