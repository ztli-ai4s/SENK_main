from __future__ import annotations

import bisect
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from dataloader_spectra import build_vib_loaders


DEFAULT_QM9S_PT = "datasets/qm9s.pt"
DEFAULT_QM9S_SKELETON = "datasets/new_skeleton/skeleton_all.pt"
DEFAULT_QME14S_OPT186102_PT = "datasets/QMe14S/qme14s_opt186102.pt"
DEFAULT_QME14S_SKELETON = "datasets/QMe14S/skeleton_all.pt"
QME14S_EDGE_RADIUS = 5.0
QME14S_EDGE_MAX_NEIGHBORS = 64

_PCC_PREFIXES = {"polar9", "depolar18", "dip", "dedipole9", "deriv_dedipole9", "deriv_depolar18"}
_R2_PREFIXES = {"dip", "polar9", "hii9", "hij9", "dedipole9", "depolar18", "deriv_dedipole9", "deriv_depolar18"}


def _resolve_local_cache_dir() -> str:
    return os.path.abspath("new_cache")


def _override_edge_params_for_qme14s(args) -> None:
    if args.dataset == "qme14s_opt186102":
        if args.radius != QME14S_EDGE_RADIUS or args.max_neighbors != QME14S_EDGE_MAX_NEIGHBORS:
            print(
                f"[edge] qme14s detected, using radius={QME14S_EDGE_RADIUS} "
                f"max_neighbors={QME14S_EDGE_MAX_NEIGHBORS}"
            )
        args.radius = QME14S_EDGE_RADIUS
        args.max_neighbors = QME14S_EDGE_MAX_NEIGHBORS


def _init_metric_sums() -> Dict[str, float]:
    prefixes = [
        "dip", "polar", "polar9", "hii9", "hij9", "dedipole9", "depolar18", "hess",
        "deriv_dedipole9", "deriv_depolar18",
    ]
    out: Dict[str, float] = {}
    for prefix in prefixes:
        out[f"{prefix}_abs_sum"] = 0.0
        out[f"{prefix}_sq_sum"] = 0.0
        out[f"{prefix}_count"] = 0.0
        if prefix in _PCC_PREFIXES:
            out[f"{prefix}_pred_sum"] = 0.0
            out[f"{prefix}_tgt_sum"] = 0.0
            out[f"{prefix}_pred_sq_sum"] = 0.0
            out[f"{prefix}_tgt_sq_sum"] = 0.0
            out[f"{prefix}_cross_sum"] = 0.0
        elif prefix in _R2_PREFIXES:
            out[f"{prefix}_tgt_sum"] = 0.0
            out[f"{prefix}_tgt_sq_sum"] = 0.0
    return out


def vec6_to_symmetric_3x3(vec6: torch.Tensor) -> torch.Tensor:
    if vec6.ndim != 2 or vec6.size(-1) != 6:
        raise ValueError(f"Expected vec6 with shape [B,6], got {tuple(vec6.shape)}")
    batch_size = vec6.size(0)
    mat = vec6.new_zeros((batch_size, 3, 3))
    mat[:, 0, 0] = vec6[:, 0]
    mat[:, 1, 1] = vec6[:, 1]
    mat[:, 2, 2] = vec6[:, 2]
    mat[:, 0, 1] = vec6[:, 3]
    mat[:, 1, 0] = vec6[:, 3]
    mat[:, 0, 2] = vec6[:, 4]
    mat[:, 2, 0] = vec6[:, 4]
    mat[:, 1, 2] = vec6[:, 5]
    mat[:, 2, 1] = vec6[:, 5]
    return mat


def sym_3x3_to_vec6(mat: torch.Tensor) -> torch.Tensor:
    if mat.ndim != 3 or mat.size(1) != 3 or mat.size(2) != 3:
        raise ValueError(f"Expected [N,3,3], got {tuple(mat.shape)}")
    xx = mat[:, 0, 0]
    yy = mat[:, 1, 1]
    zz = mat[:, 2, 2]
    xy = mat[:, 0, 1]
    xz = mat[:, 0, 2]
    yz = mat[:, 1, 2]
    return torch.stack([xx, yy, zz, xy, xz, yz], dim=-1)


def _filter_depolar_mask(data, max_abs: float):
    num_graphs = int(getattr(data, "num_graphs", 1))
    if max_abs is None or float(max_abs) <= 0:
        return None, 0, num_graphs
    if not hasattr(data, "y_depolar"):
        return None, 0, num_graphs
    y = data.y_depolar
    if not isinstance(y, torch.Tensor) or y.numel() == 0:
        return None, 0, num_graphs

    if not hasattr(data, "batch"):
        max_val = float(y.abs().max().item())
        if max_val > max_abs:
            return torch.zeros((y.size(0),), dtype=torch.bool, device=y.device), 1, num_graphs
        return None, 0, num_graphs

    batch = data.batch
    if not isinstance(batch, torch.Tensor) or batch.numel() != y.size(0):
        max_val = float(y.abs().max().item())
        if max_val > max_abs:
            return torch.zeros((y.size(0),), dtype=torch.bool, device=y.device), num_graphs, num_graphs
        return None, 0, num_graphs

    per_node_max = y.abs().view(y.size(0), -1).amax(dim=1)
    try:
        graph_max = torch.zeros(num_graphs, device=y.device)
        graph_max.scatter_reduce_(0, batch, per_node_max, reduce="amax")
    except Exception:
        graph_max = torch.full((num_graphs,), 0.0, device=y.device)
        for graph_index in range(num_graphs):
            mask_g = batch == graph_index
            if mask_g.any():
                graph_max[graph_index] = per_node_max[mask_g].max()

    bad_graphs = graph_max > float(max_abs)
    if not bad_graphs.any():
        return None, 0, num_graphs
    node_mask = ~bad_graphs[batch]
    return node_mask, int(bad_graphs.sum().item()), num_graphs


def _pearson_r_from_sums(
    n: float,
    pred_sum: float,
    tgt_sum: float,
    pred_sq: float,
    tgt_sq: float,
    cross: float,
) -> float:
    if n <= 1:
        return 0.0
    num = n * cross - pred_sum * tgt_sum
    den_p = n * pred_sq - pred_sum ** 2
    den_t = n * tgt_sq - tgt_sum ** 2
    if den_p <= 0 or den_t <= 0:
        return 0.0
    r = num / (den_p * den_t) ** 0.5
    return float(max(-1.0, min(1.0, r)))


def _r2_from_sums(n: float, tgt_sum: float, tgt_sq: float, sq_err: float) -> float:
    if n <= 1:
        return 0.0
    sst = tgt_sq - (tgt_sum ** 2) / n
    if sst <= 1.0e-12:
        return 0.0
    return float(1.0 - (sq_err / sst))


def _update_metric_sums(metric_sums: Dict[str, float], key_prefix: str, pred: torch.Tensor, target: torch.Tensor):
    diff = pred - target
    abs_sum = torch.abs(diff).sum().item()
    sq_sum = (diff ** 2).sum().item()
    count = float(diff.numel())
    metric_sums[f"{key_prefix}_abs_sum"] = metric_sums.get(f"{key_prefix}_abs_sum", 0.0) + abs_sum
    metric_sums[f"{key_prefix}_sq_sum"] = metric_sums.get(f"{key_prefix}_sq_sum", 0.0) + sq_sum
    metric_sums[f"{key_prefix}_count"] = metric_sums.get(f"{key_prefix}_count", 0.0) + count
    if key_prefix in _PCC_PREFIXES:
        pred_flat = pred.detach().float().reshape(-1)
        tgt_flat = target.detach().float().reshape(-1)
        metric_sums[f"{key_prefix}_pred_sum"] = metric_sums.get(f"{key_prefix}_pred_sum", 0.0) + float(pred_flat.sum().item())
        metric_sums[f"{key_prefix}_tgt_sum"] = metric_sums.get(f"{key_prefix}_tgt_sum", 0.0) + float(tgt_flat.sum().item())
        metric_sums[f"{key_prefix}_pred_sq_sum"] = metric_sums.get(f"{key_prefix}_pred_sq_sum", 0.0) + float((pred_flat * pred_flat).sum().item())
        metric_sums[f"{key_prefix}_tgt_sq_sum"] = metric_sums.get(f"{key_prefix}_tgt_sq_sum", 0.0) + float((tgt_flat * tgt_flat).sum().item())
        metric_sums[f"{key_prefix}_cross_sum"] = metric_sums.get(f"{key_prefix}_cross_sum", 0.0) + float((pred_flat * tgt_flat).sum().item())
    elif key_prefix in _R2_PREFIXES:
        tgt_flat = target.detach().float().reshape(-1)
        metric_sums[f"{key_prefix}_tgt_sum"] = metric_sums.get(f"{key_prefix}_tgt_sum", 0.0) + float(tgt_flat.sum().item())
        metric_sums[f"{key_prefix}_tgt_sq_sum"] = metric_sums.get(f"{key_prefix}_tgt_sq_sum", 0.0) + float((tgt_flat * tgt_flat).sum().item())


def _finalize_metrics(metric_sums: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for prefix, name in [
        ("dip", "dip"),
        ("polar9", "polar"),
        ("hii9", "hii"),
        ("hij9", "hij"),
        ("dedipole9", "dedipole"),
        ("depolar18", "depolar"),
        ("hess", "hessian_matrix"),
        ("deriv_dedipole9", "deriv_dedipole"),
        ("deriv_depolar18", "deriv_depolar"),
    ]:
        count = metric_sums.get(f"{prefix}_count", 0.0)
        if count <= 0:
            continue
        abs_sum = metric_sums.get(f"{prefix}_abs_sum", 0.0)
        sq_sum = metric_sums.get(f"{prefix}_sq_sum", 0.0)
        out[f"{name}_mae"] = abs_sum / count
        out[f"{name}_rmse"] = (sq_sum / count) ** 0.5
        if prefix in _R2_PREFIXES:
            out[f"{name}_r2"] = _r2_from_sums(
                n=count,
                tgt_sum=metric_sums.get(f"{prefix}_tgt_sum", 0.0),
                tgt_sq=metric_sums.get(f"{prefix}_tgt_sq_sum", 0.0),
                sq_err=sq_sum,
            )
        if prefix in _PCC_PREFIXES:
            out[f"{name}_pcc"] = _pearson_r_from_sums(
                n=count,
                pred_sum=metric_sums.get(f"{prefix}_pred_sum", 0.0),
                tgt_sum=metric_sums.get(f"{prefix}_tgt_sum", 0.0),
                pred_sq=metric_sums.get(f"{prefix}_pred_sq_sum", 0.0),
                tgt_sq=metric_sums.get(f"{prefix}_tgt_sq_sum", 0.0),
                cross=metric_sums.get(f"{prefix}_cross_sum", 0.0),
            )
    if "hii_mae" in out and "hij_mae" in out:
        out["hessian_mae"] = 0.5 * (out["hii_mae"] + out["hij_mae"])
    return out


__all__ = [
    "DEFAULT_QM9S_PT",
    "DEFAULT_QM9S_SKELETON",
    "DEFAULT_QME14S_SKELETON",
    "DEFAULT_QME14S_OPT186102_PT",
    "QME14S_EDGE_RADIUS",
    "QME14S_EDGE_MAX_NEIGHBORS",
    "build_vib_loaders",
    "_resolve_local_cache_dir",
    "_override_edge_params_for_qme14s",
    "_init_metric_sums",
    "vec6_to_symmetric_3x3",
    "sym_3x3_to_vec6",
    "_filter_depolar_mask",
    "_update_metric_sums",
    "_finalize_metrics",
    "ElectronPriorShardDataset",
    "ElectronPriorCachedDataset",
]


#  Electron prior cache dataset classes


class ElectronPriorShardDataset(Dataset):
    """Sharded on-disk cache for electron prior predictions.

    Loads shards lazily with LRU eviction (``cache_shards`` in-memory).
    """

    def __init__(self, shard_paths: List[str], shard_sizes: List[int], cache_shards: int = 2):
        self.shard_paths = list(shard_paths)
        self.shard_sizes = list(shard_sizes)
        self.cache_shards = max(1, int(cache_shards))
        self._prefix = [0]
        total = 0
        for size in self.shard_sizes:
            total += int(size)
            self._prefix.append(total)
        self._cache: Dict[int, List[Dict[str, torch.Tensor]]] = {}
        self._cache_order: List[int] = []

    def __len__(self) -> int:
        return self._prefix[-1]

    def _load_shard(self, shard_idx: int) -> List[Dict[str, torch.Tensor]]:
        if shard_idx in self._cache:
            try:
                self._cache_order.remove(shard_idx)
            except ValueError:
                pass
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]
        try:
            data = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(self.shard_paths[shard_idx], map_location="cpu")
        self._cache[shard_idx] = data
        self._cache_order.append(shard_idx)
        if len(self._cache_order) > self.cache_shards:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx = bisect.bisect_right(self._prefix, idx) - 1
        if shard_idx < 0:
            shard_idx = 0
        offset = idx - self._prefix[shard_idx]
        return self._load_shard(shard_idx)[offset]


class ElectronPriorCachedDataset(Dataset):
    """Wraps a base dataset and attaches cached electron prior predictions."""

    def __init__(self, base_dataset: Dataset, cache_dataset: Dataset):
        if len(base_dataset) != len(cache_dataset):
            raise ValueError(
                f"Electron prior cache length mismatch: "
                f"dataset={len(base_dataset)} cache={len(cache_dataset)}"
            )
        self.base_dataset = base_dataset
        self.cache_dataset = cache_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        data = self.base_dataset[idx]
        cache_entry = self.cache_dataset[idx]
        if hasattr(data, "clone"):
            data = data.clone()
        data.ep_atom_pred = cache_entry["ep_atom_pred"]
        data.ep_bond_pred = cache_entry["ep_bond_pred"]
        data.ep_interaction_pred = cache_entry["ep_interaction_pred"]

        atom_len = int(getattr(data, "z").size(0)) if hasattr(data, "z") else -1
        if int(data.ep_atom_pred.size(0)) != atom_len:
            raise ValueError(
                f"Cached ep_atom_pred length mismatch at idx={idx}: "
                f"cache={tuple(data.ep_atom_pred.shape)} atoms={atom_len}"
            )
        return data