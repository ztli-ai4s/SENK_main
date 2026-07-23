"""Spectra dataloaders for SENK training and inference.

This file contains the implementation for all spectra data loading.
Public entrypoints are ``build_polar_dipole_loaders`` and ``build_vib_loaders``.
"""

import bisect
import gc
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from tqdm import tqdm

logger = logging.getLogger(__name__)


DEFAULT_RADIUS = 4.5
DEFAULT_MAX_NEIGHBORS = 32
QME14S_EDGE_RADIUS = 5.0
QME14S_EDGE_MAX_NEIGHBORS = 64


def _resolve_edge_params(pt_path: Optional[str], radius: float, max_neighbors: int) -> Tuple[float, int]:
    if pt_path and "qme14s" in str(pt_path).lower():
        return QME14S_EDGE_RADIUS, QME14S_EDGE_MAX_NEIGHBORS
    return float(radius), int(max_neighbors)


def _infer_dataset_kind(pt_path: Optional[str]) -> str:
    if pt_path is None:
        return "unknown"
    name = str(pt_path).lower()
    if "qme14s" in name:
        return "qme14s"
    if "qm9s" in name or "qm9" in name:
        return "qm9s"
    return "unknown"


def _get_item_field(item: Any, keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        if isinstance(item, dict):
            if key in item and item[key] is not None:
                return item[key]
        else:
            if hasattr(item, key):
                value = getattr(item, key)
                if value is not None:
                    return value
    return None


def _resolve_hii_field(item: Any, dataset_kind: str) -> Any:
    if dataset_kind == "qm9s":
        keys = ("Hi", "Hii", "hi", "hii")
    elif dataset_kind == "qme14s":
        keys = ("Hii", "Hi", "hii", "hi")
    else:
        keys = ("Hii", "hii", "Hi", "hi")
    value = _get_item_field(item, keys)
    if value is None:
        raise ValueError(f"Missing Hii/Hi field for dataset_kind={dataset_kind}; tried {keys}")
    return value


def _resolve_hij_field(item: Any, dataset_kind: str) -> Any:
    keys = ("Hij", "hij")
    value = _get_item_field(item, keys)
    if value is None:
        raise ValueError(f"Missing Hij field for dataset_kind={dataset_kind}; tried {keys}")
    return value


def _build_radius_edge_index(pos: torch.Tensor, radius: float, max_neighbors: int) -> torch.Tensor:
    return radius_graph(
        x=pos,
        r=radius,
        batch=torch.zeros(pos.size(0), dtype=torch.long, device=pos.device),
        max_num_neighbors=max_neighbors,
    )


class SkelData(Data):
    """Preserve cat/inc semantics for skeleton data batching."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in [
            "interaction_edge_index",
            "atom_bond_index",
            "atom_to_nbo_index",
            "link_prediction_candidates",
            "edge_d_index",
            "edge_index",
        ]:
            return 1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["interaction_edge_index", "atom_bond_index", "atom_to_nbo_index", "link_prediction_candidates"]:
            num_global = int(self.node_type.size(0)) if hasattr(self, "node_type") and isinstance(self.node_type, torch.Tensor) else 0
            return num_global
        if key in ["edge_d_index", "edge_index"]:
            num_atoms = int(self.pos.size(0)) if hasattr(self, "pos") and isinstance(self.pos, torch.Tensor) else 0
            return num_atoms
        return 0


def _edge_index_to_atom_bond(edge_index: Optional[torch.Tensor]) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    row, col = edge_index
    mask = row < col
    if not mask.any():
        return torch.empty((2, 0), dtype=torch.long)
    pairs = torch.stack([row[mask], col[mask]], dim=0)
    return pairs.contiguous()


def _edge_index_from_atom_bond(atom_bond_index: torch.Tensor) -> torch.Tensor:
    if atom_bond_index is None or atom_bond_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    atom_a, atom_b = atom_bond_index
    edges = torch.stack([torch.cat([atom_a, atom_b], dim=0), torch.cat([atom_b, atom_a], dim=0)], dim=0)
    return edges.contiguous()


def _sym_3x3_to_vec6(mat: torch.Tensor) -> torch.Tensor:
    xx = mat[:, 0, 0]
    yy = mat[:, 1, 1]
    zz = mat[:, 2, 2]
    xy = mat[:, 0, 1]
    xz = mat[:, 0, 2]
    yz = mat[:, 1, 2]
    return torch.stack([xx, yy, zz, xy, xz, yz], dim=-1)


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


def _maybe_center(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    centers = torch.zeros(batch.max().item() + 1, 3, device=pos.device, dtype=pos.dtype)
    centers = centers.index_add(0, batch, pos)
    counts = torch.bincount(batch).clamp(min=1).view(-1, 1)
    centers = centers / counts
    return pos - centers[batch]


def _pick_id(item: Any, idx: int) -> int:
    num = getattr(item, "number", None)
    if isinstance(item, dict):
        num = item.get("number", num)
    if num is not None:
        try:
            return int(num)
        except Exception:
            pass
    return idx


def _build_id_map(raw_list: List[Any]) -> Dict[int, Any]:
    id_map: Dict[int, Any] = {}
    for item_index, item in enumerate(raw_list):
        mol_id = _pick_id(item, item_index)
        id_map[mol_id] = item
    return id_map


def _build_splits(num_items: int, ratios: Tuple[float, float, float], seed: int, ids: Optional[List[int]] = None) -> Dict[str, List[int]]:
    generator = torch.Generator().manual_seed(seed)
    if ids is None:
        perm = torch.randperm(num_items, generator=generator).tolist()
    else:
        perm = torch.tensor(ids)[torch.randperm(len(ids), generator=generator)].tolist()
    n_train = int(num_items * ratios[0])
    n_val = int(num_items * ratios[1])
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


class CachedDataset(Dataset):
    """Lightweight wrapper to restore a Dataset from a saved Data list."""

    def __init__(self, data_list: List[Data]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class LazyCachedDataset(Dataset):
    """Lazy cache that only stores per-item paths and loads on demand."""

    def __init__(self, item_paths: List[str]):
        self.item_paths = list(item_paths)

    def __len__(self):
        return len(self.item_paths)

    def __getitem__(self, idx):
        path = self.item_paths[idx]
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


class ShardedCachedDataset(Dataset):
    """Sharded cache to reduce small-file IO pressure."""

    def __init__(self, shard_paths: List[str], shard_sizes: List[int], cache_shards: int = 2):
        self.shard_paths = list(shard_paths)
        self.shard_sizes = list(shard_sizes)
        self.cache_shards = max(1, int(cache_shards))
        self._prefix = [0]
        total = 0
        for shard_size in self.shard_sizes:
            total += int(shard_size)
            self._prefix.append(total)
        self._cache = {}
        self._cache_order: List[int] = []

    def __len__(self):
        return self._prefix[-1]

    def _load_shard(self, shard_idx: int):
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

    def __getitem__(self, idx):
        shard_idx = bisect.bisect_right(self._prefix, idx) - 1
        if shard_idx < 0:
            shard_idx = 0
        offset = idx - self._prefix[shard_idx]
        data_list = self._load_shard(shard_idx)
        return data_list[offset]


def _extract_data_list(dataset: Dataset) -> List[Data]:
    if hasattr(dataset, "data_list"):
        return dataset.data_list
    return [dataset[i] for i in range(len(dataset))]


def _write_lazy_cache(datasets: Dict[str, Dataset], stats: Dict[str, torch.Tensor], cache_root: Path) -> Dict[str, List[str]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, List[str]] = {}
    for split in ["train", "val", "test"]:
        split_dir = cache_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        iterable, total = _iter_dataset(datasets[split])
        paths: List[str] = []
        for item_index, data in enumerate(tqdm(iterable, desc=f"Cache {split}", total=total, leave=False)):
            item_path = split_dir / f"{item_index}.pt"
            torch.save(data, item_path)
            paths.append(str(item_path))
        split_paths[split] = paths
    meta = {"paths": split_paths, "stats": stats}
    torch.save(meta, cache_root / "meta.pt")
    return split_paths


def _write_sharded_cache(
    datasets: Dict[str, Dataset],
    stats: Dict[str, torch.Tensor],
    cache_root: Path,
    shard_size: int,
) -> Dict[str, Dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    split_meta: Dict[str, Dict[str, Any]] = {}
    shard_size = max(1, int(shard_size))
    for split in ["train", "val", "test"]:
        split_dir = cache_root / "shards" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        sizes: List[int] = []
        buf: List[Data] = []
        iterable, total = _iter_dataset(datasets[split])
        for data in tqdm(iterable, desc=f"Shard {split}", total=total, leave=False):
            buf.append(data)
            if len(buf) >= shard_size:
                shard_path = split_dir / f"shard_{len(paths):06d}.pt"
                torch.save(buf, shard_path)
                paths.append(str(shard_path))
                sizes.append(len(buf))
                buf = []
        if buf:
            shard_path = split_dir / f"shard_{len(paths):06d}.pt"
            torch.save(buf, shard_path)
            paths.append(str(shard_path))
            sizes.append(len(buf))
        split_meta[split] = {"paths": paths, "sizes": sizes}
    meta = {"format": "sharded", "stats": stats, "shard_size": shard_size, "splits": split_meta}
    torch.save(meta, cache_root / "meta.pt")
    return split_meta


def _build_lazy_cache_only(
    pt_path: str,
    skeleton_path: Optional[str],
    radius: float,
    max_neighbors: int,
    split_ratios: Tuple[float, float, float],
    seed: int,
    center_positions: bool,
    cache_root: Path,
    cache_format: str = "sharded",
    cache_shard_size: int = 256,
):
    edge_radius, edge_max_neighbors = _resolve_edge_params(pt_path, radius, max_neighbors)
    try:
        raw = torch.load(pt_path, weights_only=False)
    except TypeError:
        raw = torch.load(pt_path)

    if skeleton_path is None:
        splits = _build_splits(len(raw), split_ratios, seed)
        datasets: Dict[str, Dataset] = {}
        for split in ["train", "val", "test"]:
            idxs = splits[split]
            datasets[split] = PolarDipoleDataset(
                pt_path=pt_path,
                indices=idxs,
                radius=edge_radius,
                max_neighbors=edge_max_neighbors,
                center_positions=center_positions,
            )
    else:
        try:
            skeleton_raw = torch.load(skeleton_path, weights_only=False)
        except TypeError:
            skeleton_raw = torch.load(skeleton_path)
        skeleton_variants: Dict[int, List[Dict[str, Any]]] = {}
        for key, value in tqdm(skeleton_raw.items(), desc="Index skeleton", leave=False):
            try:
                mol_id = int(key)
            except Exception:
                continue
            if isinstance(value, list):
                variants = [item for item in value if isinstance(item, dict)]
            elif isinstance(value, dict):
                variants = [value]
            else:
                variants = []
            if variants:
                skeleton_variants[mol_id] = variants
        mol_ids = sorted(skeleton_variants.keys())
        splits = _build_splits(len(mol_ids), split_ratios, seed, ids=mol_ids)
        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = PolarDipoleSkeletonDataset(
                raw_list=raw,
                skeleton_variants=skeleton_variants,
                split_ids=splits[split],
                center_positions=center_positions,
            )

    stats = _compute_stats(datasets["train"])
    if str(cache_format).lower() == "lazy":
        _write_lazy_cache(datasets, stats, cache_root)
    else:
        _write_sharded_cache(datasets, stats, cache_root, shard_size=cache_shard_size)


def _iter_dataset(dataset: Any):
    if isinstance(dataset, list):
        return dataset, len(dataset)
    if hasattr(dataset, "data_list"):
        return dataset.data_list, len(dataset.data_list)
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        total = len(dataset)
        return (dataset[i] for i in range(total)), total
    return dataset, None


def _compute_stats(dataset: Any) -> Dict[str, torch.Tensor]:
    dipoles = []
    polars = []
    iterable, total = _iter_dataset(dataset)
    for data in tqdm(iterable, desc="Compute stats", total=total, leave=False):
        if hasattr(data, "y_dipole") and isinstance(data.y_dipole, torch.Tensor) and data.y_dipole.numel() > 0:
            dipoles.append(data.y_dipole)
        if hasattr(data, "y_polar_vec6") and isinstance(data.y_polar_vec6, torch.Tensor) and data.y_polar_vec6.numel() > 0:
            polars.append(data.y_polar_vec6)
    if not dipoles or not polars:
        return {}
    dipoles = torch.cat(dipoles, dim=0)
    polars = torch.cat(polars, dim=0)

    def _mean_mad(x: torch.Tensor):
        mean = x.mean(dim=0)
        mad = (x - mean).abs().mean(dim=0).clamp(min=1e-6)
        return mean, mad

    d_mean, d_mad = _mean_mad(dipoles)
    p_mean, p_mad = _mean_mad(polars)
    return {"dipole_mean": d_mean, "dipole_mad": d_mad, "polar_mean": p_mean, "polar_mad": p_mad}


class PolarDipoleData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "y_dipole"):
            self.y_dipole = torch.empty((0, 3))
        if not hasattr(self, "y_polar_vec6"):
            self.y_polar_vec6 = torch.empty((0, 6))


class VibTensorData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "y_hii"):
            self.y_hii = torch.empty((0, 3, 3))
        if not hasattr(self, "y_hij"):
            self.y_hij = torch.empty((0, 3, 3))
        if not hasattr(self, "y_dedipole"):
            self.y_dedipole = torch.empty((0, 3, 3))
        if not hasattr(self, "y_depolar"):
            self.y_depolar = torch.empty((0, 3, 6))
        if not hasattr(self, "y_dipole"):
            self.y_dipole = torch.empty((0, 3))
        if not hasattr(self, "y_polar_vec6"):
            self.y_polar_vec6 = torch.empty((0, 6))


class VibTensorDataset(Dataset):
    """OPT_186102 vib dataset: Hii/Hij/dedipole/depolar labels."""

    def __init__(
        self,
        pt_path: str,
        indices: List[int],
        radius: float,
        max_neighbors: int,
        center_positions: bool,
        need_hij: bool = True,
        dataset_kind: Optional[str] = None,
    ):
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.center_positions = center_positions
        self.need_hij = bool(need_hij)
        self.dataset_kind = dataset_kind or _infer_dataset_kind(pt_path)
        try:
            raw = torch.load(pt_path, weights_only=False)
        except TypeError:
            raw = torch.load(pt_path)
        self.data_list: List[VibTensorData] = []
        skipped = 0
        for idx in tqdm(indices, desc="Build vib tensor dataset", leave=False):
            item = raw[idx]
            try:
                self.data_list.append(self._to_data(item))
            except ValueError as exc:
                if self.need_hij and "Hij size mismatch" in str(exc):
                    skipped += 1
                    continue
                raise
        if self.need_hij and skipped > 0:
            logger.warning("[hij] skipped %d molecules due to Hij/radius edge mismatch", skipped)

    def _to_data(self, item: Dict) -> VibTensorData:
        pos = item["pos"] if "pos" in item else item["positions"]
        z = item["z"].long() if "z" in item else item["atomic_numbers"].long()
        edge_index = _build_radius_edge_index(pos, self.radius, self.max_neighbors)

        hii_raw = _resolve_hii_field(item, self.dataset_kind)
        hii = _normalize_square33(hii_raw, "Hii")
        if self.need_hij:
            hij_raw = _resolve_hij_field(item, self.dataset_kind)
            hij = _normalize_hij(hij_raw, edge_index)
        else:
            hij = torch.empty((0, 3, 3))
        dedipole = _normalize_square33(_get_item_field(item, ("dedipole",)), "dedipole")
        depolar = _normalize_depolar(_get_item_field(item, ("depolar",)))

        dipole = torch.zeros(1, 3)
        polar_vec6 = torch.zeros(1, 6)
        try:
            if "dipole" in item:
                dipole = item["dipole"].float().view(1, 3)
            if "polar_vec6" in item:
                polar_vec6 = item["polar_vec6"].float().view(1, 6)
            elif "polar" in item:
                polar_vec6 = _sym_3x3_to_vec6(item["polar"].float().view(-1, 3, 3))[0].view(1, 6)
        except Exception:
            pass

        if self.center_positions:
            batch_dummy = torch.zeros(pos.size(0), dtype=torch.long)
            pos = _maybe_center(pos, batch_dummy)

        data = VibTensorData(
            pos=pos.float(),
            z=z,
            y_hii=hii.float(),
            y_hij=hij.float(),
            y_dedipole=dedipole.float(),
            y_depolar=depolar.float(),
            y_dipole=dipole.float(),
            y_polar_vec6=polar_vec6.float(),
            node_type=torch.zeros(pos.size(0), dtype=torch.long),
            pos_global=pos.float(),
            atom_to_nbo_index=torch.zeros((2, 0), dtype=torch.long),
            interaction_edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.edge_index = edge_index
        data.atom_bond_index = _edge_index_to_atom_bond(edge_index)
        data.num_global_nodes = torch.tensor([pos.size(0)], dtype=torch.long)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class VibTensorSkeletonDataset(Dataset):
    """Skeleton mode for vib tasks with skeleton-side electronic channels."""

    def __init__(
        self,
        raw_list: List[Any],
        skeleton_variants: Dict[int, List[Dict[str, Any]]],
        split_ids: List[int],
        radius: float,
        max_neighbors: int,
        center_positions: bool = True,
        need_hij: bool = True,
        dataset_kind: Optional[str] = None,
    ):
        self.center_positions = center_positions
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.need_hij = bool(need_hij)
        self.dataset_kind = dataset_kind or "unknown"
        id_map = _build_id_map(raw_list)
        data_list: List[SkelData] = []
        skipped = 0
        for mol_id in tqdm(split_ids, desc="Build vib skeleton dataset", leave=False):
            if mol_id not in skeleton_variants:
                continue
            raw_item = id_map.get(mol_id)
            if raw_item is None:
                continue
            variants = skeleton_variants[mol_id]
            entries: List[Dict[str, Any]] = []
            if len(variants) == 1 and isinstance(variants[0], dict) and "sparse_variant" in variants[0]:
                dense_entry = variants[0]
                entries.append(dense_entry)
                sparse_entry = variants[0].get("sparse_variant", None)
                if sparse_entry is not None and isinstance(sparse_entry, dict):
                    entries.append(sparse_entry)
            else:
                entries = [entry for entry in variants if isinstance(entry, dict)]

            temp_list: List[SkelData] = []
            skip_mol = False
            for entry in entries:
                try:
                    is_sparse = bool(entry.get("is_sparse", False))
                    temp_list.append(self._make_data(entry, raw_item, mol_id, is_sparse=is_sparse))
                except ValueError as exc:
                    if self.need_hij and "Hij size mismatch" in str(exc):
                        skip_mol = True
                        break
                    raise
            if skip_mol:
                skipped += 1
                continue
            data_list.extend(temp_list)
        self.data_list = data_list
        if self.need_hij and skipped > 0:
            logger.warning("[hij] skipped %d molecules due to Hij/radius edge mismatch", skipped)

    def _make_data(self, entry: Dict, raw_item: Any, mol_id: int, is_sparse: bool = False) -> SkelData:
        if isinstance(raw_item, dict):
            pos_atoms = raw_item["pos"] if "pos" in raw_item else raw_item["positions"]
            z = raw_item["z"].long() if "z" in raw_item else raw_item["atomic_numbers"].long()
            dedipole = _normalize_square33(_get_item_field(raw_item, ("dedipole",)), "dedipole")
            depolar = _normalize_depolar(_get_item_field(raw_item, ("depolar",)))
            dipole = torch.zeros(1, 3)
            polar_vec6 = torch.zeros(1, 6)
            try:
                if "dipole" in raw_item:
                    dipole = raw_item["dipole"].float().view(1, 3)
                if "polar_vec6" in raw_item:
                    polar_vec6 = raw_item["polar_vec6"].float().view(1, 6)
                elif "polar" in raw_item:
                    polar_vec6 = _sym_3x3_to_vec6(raw_item["polar"].float().view(-1, 3, 3))[0].view(1, 6)
            except Exception:
                pass
        else:
            pos_atoms = raw_item.pos
            z = raw_item.z.long()
            dedipole = _normalize_square33(_get_item_field(raw_item, ("dedipole",)), "dedipole")
            depolar = _normalize_depolar(_get_item_field(raw_item, ("depolar",)))
            dipole = torch.zeros(1, 3)
            polar_vec6 = torch.zeros(1, 6)
            try:
                if hasattr(raw_item, "dipole") and raw_item.dipole is not None:
                    dipole = raw_item.dipole.float().view(1, 3)
                if hasattr(raw_item, "polar_vec6") and raw_item.polar_vec6 is not None:
                    polar_vec6 = raw_item.polar_vec6.float().view(1, 6)
                elif hasattr(raw_item, "polar") and raw_item.polar is not None:
                    polar_vec6 = _sym_3x3_to_vec6(raw_item.polar.float().view(-1, 3, 3))[0].view(1, 6)
            except Exception:
                pass

        edge_index = _build_radius_edge_index(pos_atoms, self.radius, self.max_neighbors)
        hii_raw = _resolve_hii_field(raw_item, self.dataset_kind)
        hii = _normalize_square33(hii_raw, "Hii")
        if self.need_hij:
            hij_raw = _resolve_hij_field(raw_item, self.dataset_kind)
            hij = _normalize_hij(hij_raw, edge_index)
        else:
            hij = torch.empty((0, 3, 3))

        pos_global = entry["pos_global"].clone()
        node_type = entry["node_type"].clone()
        atom_to_nbo_index = entry.get("atom_to_nbo_index", torch.zeros((2, 0), dtype=torch.long)).clone()
        atom_bond_index = entry.get("atom_bond_index", torch.zeros((2, 0), dtype=torch.long)).clone()
        interaction_edge_index = entry.get("interaction_edge_index", torch.zeros((2, 0), dtype=torch.long)).clone()
        num_global_nodes = entry.get("num_global_nodes", int(node_type.numel()))

        node_type_clean = node_type.clone()
        if (node_type_clean == 3).any():
            node_type_clean[node_type_clean == 3] = 2
        if (node_type_clean == 2).any() and (node_type_clean == 1).sum() == 0 and atom_to_nbo_index.numel() > 0:
            tgt = atom_to_nbo_index[1]
            counts = torch.bincount(tgt, minlength=node_type_clean.shape[0])
            combined_mask = node_type_clean == 2
            bond_mask = combined_mask & (counts >= 2)
            lp_mask = combined_mask & (counts < 2)
            node_type_clean[bond_mask] = 1
            node_type_clean[lp_mask] = 2

        if self.center_positions:
            center = pos_atoms.mean(dim=0, keepdim=True)
            pos_atoms = pos_atoms - center
            pos_global = pos_global - center

        data = SkelData(
            pos=pos_atoms.float(),
            z=z,
            y_hii=hii.float(),
            y_hij=hij.float(),
            y_dedipole=dedipole.float(),
            y_depolar=depolar.float(),
            y_dipole=dipole.float(),
            y_polar_vec6=polar_vec6.float(),
            node_type=node_type_clean,
            pos_global=pos_global.float(),
            atom_to_nbo_index=atom_to_nbo_index,
            interaction_edge_index=interaction_edge_index,
            atom_bond_index=atom_bond_index,
            edge_index=edge_index,
            mol_id=int(mol_id),
            is_sparse=bool(is_sparse),
        )
        data.num_global_nodes = torch.tensor([num_global_nodes], dtype=torch.long)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class PolarDipoleDataset(Dataset):
    """Atom-only polar/dipole dataset built directly from pt files."""

    def __init__(self, pt_path: str, indices: List[int], radius: float, max_neighbors: int, center_positions: bool):
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.center_positions = center_positions
        try:
            raw = torch.load(pt_path, weights_only=False)
        except TypeError:
            raw = torch.load(pt_path)
        self.data_list: List[PolarDipoleData] = []
        for idx in tqdm(indices, desc="Build polar/dipole dataset", leave=False):
            item = raw[idx]
            self.data_list.append(self._to_data(item))

    def _to_data(self, item: Dict) -> PolarDipoleData:
        pos = item["pos"] if "pos" in item else item["positions"]
        z = item["z"].long() if "z" in item else item["atomic_numbers"].long()
        dipole = item["dipole"].view(1, 3)
        if "polar_vec6" in item:
            polar_vec6 = item["polar_vec6"].view(1, 6)
        else:
            polar_vec6 = _sym_3x3_to_vec6(item["polar"].view(-1, 3, 3))[0].view(1, 6)

        if self.center_positions:
            batch_dummy = torch.zeros(pos.size(0), dtype=torch.long)
            pos = _maybe_center(pos, batch_dummy)

        data = PolarDipoleData(
            pos=pos.float(),
            z=z,
            y_dipole=dipole.float(),
            y_polar_vec6=polar_vec6.float(),
            node_type=torch.zeros(pos.size(0), dtype=torch.long),
            pos_global=pos.float(),
            atom_to_nbo_index=torch.zeros((2, 0), dtype=torch.long),
            interaction_edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.edge_index = radius_graph(
            x=data.pos,
            r=self.radius,
            batch=torch.zeros(data.pos.size(0), dtype=torch.long),
            max_num_neighbors=self.max_neighbors,
        )
        data.atom_bond_index = _edge_index_to_atom_bond(data.edge_index)
        data.num_global_nodes = torch.tensor([pos.size(0)], dtype=torch.long)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class PolarDipoleSkeletonDataset(Dataset):
    """Skeleton mode that keeps all dense/sparse variants per molecule."""

    def __init__(self, raw_list: List[Any], skeleton_variants: Dict[int, List[Dict[str, Any]]], split_ids: List[int], center_positions: bool = True):
        self.center_positions = center_positions
        id_map = _build_id_map(raw_list)
        data_list: List[SkelData] = []
        for mol_id in tqdm(split_ids, desc="Build skeleton dataset", leave=False):
            if mol_id not in skeleton_variants:
                continue
            raw_item = id_map.get(mol_id)
            if raw_item is None:
                continue
            variants = skeleton_variants[mol_id]
            if len(variants) == 1 and isinstance(variants[0], dict) and "sparse_variant" in variants[0]:
                dense_entry = variants[0]
                sparse_entry = variants[0].get("sparse_variant", None)
                data_list.append(self._make_data(dense_entry, raw_item, mol_id, is_sparse=False))
                if sparse_entry is not None:
                    data_list.append(self._make_data(sparse_entry, raw_item, mol_id, is_sparse=True))
            else:
                for entry in variants:
                    is_sparse = bool(entry.get("is_sparse", False)) if isinstance(entry, dict) else False
                    data_list.append(self._make_data(entry, raw_item, mol_id, is_sparse=is_sparse))
        self.data_list = data_list

    def _make_data(self, entry: Dict, raw_item: Any, mol_id: int, is_sparse: bool = False) -> SkelData:
        if isinstance(raw_item, dict):
            pos_atoms = raw_item["pos"] if "pos" in raw_item else raw_item["positions"]
            z = raw_item["z"].long() if "z" in raw_item else raw_item["atomic_numbers"].long()
            dipole = raw_item["dipole"].view(1, 3)
            if "polar_vec6" in raw_item:
                polar_vec6 = raw_item["polar_vec6"].view(1, 6)
            else:
                polar_vec6 = _sym_3x3_to_vec6(raw_item["polar"].view(-1, 3, 3))[0].view(1, 6)
        else:
            pos_atoms = raw_item.pos
            z = raw_item.z.long()
            dipole = raw_item.dipole.view(1, 3)
            if hasattr(raw_item, "polar_vec6"):
                polar_vec6 = raw_item.polar_vec6.view(1, 6)
            else:
                polar_vec6 = _sym_3x3_to_vec6(raw_item.polar.view(-1, 3, 3))[0].view(1, 6)

        pos_global = entry["pos_global"].clone()
        node_type = entry["node_type"].clone()
        atom_to_nbo_index = entry.get("atom_to_nbo_index", torch.zeros((2, 0), dtype=torch.long)).clone()
        atom_bond_index = entry.get("atom_bond_index", torch.zeros((2, 0), dtype=torch.long)).clone()
        interaction_edge_index = entry.get("interaction_edge_index", torch.zeros((2, 0), dtype=torch.long)).clone()
        num_global_nodes = entry.get("num_global_nodes", int(node_type.numel()))

        node_type_clean = node_type.clone()
        if (node_type_clean == 3).any():
            node_type_clean[node_type_clean == 3] = 2
        if (node_type_clean == 2).any() and (node_type_clean == 1).sum() == 0 and atom_to_nbo_index.numel() > 0:
            tgt = atom_to_nbo_index[1]
            counts = torch.bincount(tgt, minlength=node_type_clean.shape[0])
            combined_mask = node_type_clean == 2
            bond_mask = combined_mask & (counts >= 2)
            lp_mask = combined_mask & (counts < 2)
            node_type_clean[bond_mask] = 1
            node_type_clean[lp_mask] = 2

        edge_index = _edge_index_from_atom_bond(atom_bond_index)

        if self.center_positions:
            center = pos_atoms.mean(dim=0, keepdim=True)
            pos_atoms = pos_atoms - center
            pos_global = pos_global - center

        data = SkelData(
            pos=pos_atoms.float(),
            z=z,
            y_dipole=dipole.float(),
            y_polar_vec6=polar_vec6.float(),
            node_type=node_type_clean,
            pos_global=pos_global.float(),
            atom_to_nbo_index=atom_to_nbo_index,
            interaction_edge_index=interaction_edge_index,
            atom_bond_index=atom_bond_index,
            edge_index=edge_index,
            mol_id=int(mol_id),
            is_sparse=bool(is_sparse),
        )
        data.num_global_nodes = torch.tensor([num_global_nodes], dtype=torch.long)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def build_polar_dipole_loaders(
    pt_path: str,
    skeleton_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    mp_context: Optional[str] = "spawn",
    cache_format: str = "sharded",
    cache_shard_size: int = 256,
    cache_shard_keep: int = 2,
    radius: float = DEFAULT_RADIUS,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    center_positions: bool = True,
    cache_dir: Optional[str] = "datasets/cache",
    use_cache: bool = True,
    auto_build_cache: bool = True,
    require_cache: bool = False,
):
    """Build train/val/test dataloaders and normalization stats."""

    edge_radius, edge_max_neighbors = _resolve_edge_params(pt_path, radius, max_neighbors)
    dataset_kind = _infer_dataset_kind(pt_path)
    if (edge_radius, edge_max_neighbors) != (radius, max_neighbors):
        logger.info("[edge] qme14s detected, using radius=%s max_neighbors=%s", edge_radius, edge_max_neighbors)

    cache_root = None
    if use_cache and cache_dir is not None:
        try:
            key_src = f"{pt_path}|{skeleton_path}|{edge_radius}|{edge_max_neighbors}|{split_ratios}|{seed}|{center_positions}"
            key = hashlib.md5(key_src.encode()).hexdigest()[:16]
            cache_root = Path(cache_dir) / f"polar_dipole_cache_{key}"
            meta_path = cache_root / "meta.pt"
            fallback_cache_path = Path(cache_dir) / f"polar_dipole_cache_{key}.pt"
            cache_root.mkdir(parents=True, exist_ok=True)

            if meta_path.exists():
                try:
                    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                except TypeError:
                    meta = torch.load(meta_path, map_location="cpu")
                meta_format = str(meta.get("format", "lazy")).lower()
                if meta_format == "sharded":
                    datasets = {
                        split: ShardedCachedDataset(
                            meta["splits"][split]["paths"],
                            meta["splits"][split]["sizes"],
                            cache_shards=cache_shard_keep,
                        )
                        for split in ["train", "val", "test"]
                    }
                else:
                    datasets = {split: LazyCachedDataset(meta["paths"][split]) for split in ["train", "val", "test"]}
                loaders = {}
                nw = int(num_workers)
                pm = bool(pin_memory)
                pw_train = bool(persistent_workers) if nw > 0 else False
                pf_train = int(prefetch_factor) if nw > 0 else None
                pf_eval = 1 if nw > 0 else None
                for split in ["train", "val", "test"]:
                    loaders[split] = DataLoader(
                        datasets[split],
                        batch_size=batch_size,
                        shuffle=(split == "train"),
                        num_workers=num_workers,
                        pin_memory=pm,
                        persistent_workers=(pw_train if split == "train" else False),
                        prefetch_factor=(pf_train if split == "train" else pf_eval),
                        multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
                    )
                stats = meta.get("stats", None)
                if stats is None:
                    stats = _compute_stats(datasets["train"])
                return loaders, stats

            if fallback_cache_path.exists():
                try:
                    cached = torch.load(fallback_cache_path, map_location="cpu", weights_only=False)
                except TypeError:
                    cached = torch.load(fallback_cache_path, map_location="cpu")
                datasets = {split: CachedDataset(cached["datasets"][split]) for split in ["train", "val", "test"]}
                stats = cached.get("stats", None)
                if stats is None:
                    stats = _compute_stats(datasets["train"].data_list)
                if str(cache_format).lower() == "lazy":
                    split_paths = _write_lazy_cache(datasets, stats, cache_root)
                    datasets = {split: LazyCachedDataset(split_paths[split]) for split in ["train", "val", "test"]}
                else:
                    split_meta = _write_sharded_cache(datasets, stats, cache_root, shard_size=cache_shard_size)
                    datasets = {
                        split: ShardedCachedDataset(
                            split_meta[split]["paths"],
                            split_meta[split]["sizes"],
                            cache_shards=cache_shard_keep,
                        )
                        for split in ["train", "val", "test"]
                    }
                loaders = {}
                nw = int(num_workers)
                pm = bool(pin_memory)
                pw_train = bool(persistent_workers) if nw > 0 else False
                pf_train = int(prefetch_factor) if nw > 0 else None
                pf_eval = 1 if nw > 0 else None
                for split in ["train", "val", "test"]:
                    loaders[split] = DataLoader(
                        datasets[split],
                        batch_size=batch_size,
                        shuffle=(split == "train"),
                        num_workers=num_workers,
                        pin_memory=pm,
                        persistent_workers=(pw_train if split == "train" else False),
                        prefetch_factor=(pf_train if split == "train" else pf_eval),
                        multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
                    )
                return loaders, stats

            if auto_build_cache:
                try:
                    import torch.multiprocessing as mp

                    ctx = mp.get_context("spawn")
                    process = ctx.Process(
                        target=_build_lazy_cache_only,
                        args=(
                            pt_path,
                            skeleton_path,
                            edge_radius,
                            edge_max_neighbors,
                            split_ratios,
                            seed,
                            center_positions,
                            cache_root,
                            cache_format,
                            cache_shard_size,
                        ),
                    )
                    process.start()
                    process.join()
                except Exception:
                    pass

                if meta_path.exists():
                    try:
                        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        meta = torch.load(meta_path, map_location="cpu")
                    meta_format = str(meta.get("format", "lazy")).lower()
                    if meta_format == "sharded":
                        datasets = {
                            split: ShardedCachedDataset(
                                meta["splits"][split]["paths"],
                                meta["splits"][split]["sizes"],
                                cache_shards=cache_shard_keep,
                            )
                            for split in ["train", "val", "test"]
                        }
                    else:
                        datasets = {split: LazyCachedDataset(meta["paths"][split]) for split in ["train", "val", "test"]}
                    loaders = {}
                    nw = int(num_workers)
                    pm = bool(pin_memory)
                    pw_train = bool(persistent_workers) if nw > 0 else False
                    pf_train = int(prefetch_factor) if nw > 0 else None
                    pf_eval = 1 if nw > 0 else None
                    mp_ctx = None if mp_context in (None, "none") else mp_context
                    for split in ["train", "val", "test"]:
                        loaders[split] = DataLoader(
                            datasets[split],
                            batch_size=batch_size,
                            shuffle=(split == "train"),
                            num_workers=num_workers,
                            pin_memory=pm,
                            persistent_workers=(pw_train if split == "train" else False),
                            prefetch_factor=(pf_train if split == "train" else pf_eval),
                            multiprocessing_context=mp_ctx if num_workers > 0 else None,
                        )
                    stats = meta.get("stats", None)
                    if stats is None:
                        stats = _compute_stats(datasets["train"])
                    return loaders, stats
        except Exception:
            cache_root = None

    if require_cache:
        raise RuntimeError(
            "Lazy cache not found. Please build cache once (e.g., --build_cache_only) and rerun training."
        )

    try:
        raw = torch.load(pt_path, weights_only=False)
    except TypeError:
        raw = torch.load(pt_path)

    if skeleton_path is None:
        splits = _build_splits(len(raw), split_ratios, seed)
        datasets: Dict[str, Dataset] = {}
        for split in ["train", "val", "test"]:
            idxs = splits[split]
            datasets[split] = PolarDipoleDataset(
                pt_path=pt_path,
                indices=idxs,
                radius=edge_radius,
                max_neighbors=edge_max_neighbors,
                center_positions=center_positions,
            )
    else:
        try:
            skeleton_raw = torch.load(skeleton_path, weights_only=False)
        except TypeError:
            skeleton_raw = torch.load(skeleton_path)
        skeleton_variants: Dict[int, List[Dict[str, Any]]] = {}
        for key, value in tqdm(skeleton_raw.items(), desc="Index skeleton", leave=False):
            try:
                mol_id = int(key)
            except Exception:
                continue
            if isinstance(value, list):
                variants = [item for item in value if isinstance(item, dict)]
            elif isinstance(value, dict):
                variants = [value]
            else:
                variants = []
            if variants:
                skeleton_variants[mol_id] = variants
        mol_ids = sorted(skeleton_variants.keys())
        splits = _build_splits(len(mol_ids), split_ratios, seed, ids=mol_ids)
        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = PolarDipoleSkeletonDataset(
                raw_list=raw,
                skeleton_variants=skeleton_variants,
                split_ids=splits[split],
                center_positions=center_positions,
            )

    stats = _compute_stats(datasets["train"])

    if use_cache and cache_root is not None:
        try:
            if str(cache_format).lower() == "lazy":
                split_paths = _write_lazy_cache(datasets, stats, cache_root)
                datasets = {split: LazyCachedDataset(split_paths[split]) for split in ["train", "val", "test"]}
            else:
                split_meta = _write_sharded_cache(datasets, stats, cache_root, shard_size=cache_shard_size)
                datasets = {
                    split: ShardedCachedDataset(
                        split_meta[split]["paths"],
                        split_meta[split]["sizes"],
                        cache_shards=cache_shard_keep,
                    )
                    for split in ["train", "val", "test"]
                }
            try:
                del raw
            except Exception:
                pass
            try:
                del skeleton_raw
            except Exception:
                pass
            try:
                del skeleton_variants
            except Exception:
                pass
            gc.collect()
            loaders = {}
            nw = int(num_workers)
            pm = bool(pin_memory)
            pw_train = bool(persistent_workers) if nw > 0 else False
            pf_train = int(prefetch_factor) if nw > 0 else None
            pf_eval = 1 if nw > 0 else None
            for split in ["train", "val", "test"]:
                loaders[split] = DataLoader(
                    datasets[split],
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    pin_memory=pm,
                    persistent_workers=(pw_train if split == "train" else False),
                    prefetch_factor=(pf_train if split == "train" else pf_eval),
                    multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
                )
        except Exception:
            pass

    if "loaders" not in locals():
        loaders = {}
        nw = int(num_workers)
        pm = bool(pin_memory)
        pw_train = bool(persistent_workers) if nw > 0 else False
        pf_train = int(prefetch_factor) if nw > 0 else None
        pf_eval = 1 if nw > 0 else None
        for split in ["train", "val", "test"]:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pm,
                persistent_workers=(pw_train if split == "train" else False),
                prefetch_factor=(pf_train if split == "train" else pf_eval),
                multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
            )

    return loaders, stats


def _build_vib_cache_only(
    pt_path: str,
    skeleton_path: Optional[str],
    radius: float,
    max_neighbors: int,
    split_ratios: Tuple[float, float, float],
    seed: int,
    center_positions: bool,
    cache_root: Path,
    need_hij: bool = True,
    cache_format: str = "sharded",
    cache_shard_size: int = 256,
):
    edge_radius, edge_max_neighbors = _resolve_edge_params(pt_path, radius, max_neighbors)
    dataset_kind = _infer_dataset_kind(pt_path)
    try:
        raw = torch.load(pt_path, weights_only=False)
    except TypeError:
        raw = torch.load(pt_path)

    datasets: Dict[str, Dataset] = {}
    if skeleton_path is not None:
        if not os.path.exists(skeleton_path):
            raise FileNotFoundError(f"skeleton_path not found: {skeleton_path}")
        try:
            skeleton_raw = torch.load(skeleton_path, weights_only=False)
        except TypeError:
            skeleton_raw = torch.load(skeleton_path)
        skeleton_variants: Dict[int, List[Dict[str, Any]]] = {}
        for key, value in tqdm(skeleton_raw.items(), desc="Index skeleton", leave=False):
            try:
                mol_id = int(key)
            except Exception:
                continue
            if isinstance(value, list):
                variants = [item for item in value if isinstance(item, dict)]
            elif isinstance(value, dict):
                variants = [value]
            else:
                variants = []
            if variants:
                skeleton_variants[mol_id] = variants
        mol_ids = sorted(skeleton_variants.keys())
        splits = _build_splits(len(mol_ids), split_ratios, seed, ids=mol_ids)
        for split in ["train", "val", "test"]:
            datasets[split] = VibTensorSkeletonDataset(
                raw_list=raw,
                skeleton_variants=skeleton_variants,
                split_ids=splits[split],
                radius=edge_radius,
                max_neighbors=edge_max_neighbors,
                center_positions=center_positions,
                need_hij=need_hij,
                dataset_kind=dataset_kind,
            )
    else:
        splits = _build_splits(len(raw), split_ratios, seed)
        for split in ["train", "val", "test"]:
            idxs = splits[split]
            datasets[split] = VibTensorDataset(
                pt_path=pt_path,
                indices=idxs,
                radius=edge_radius,
                max_neighbors=edge_max_neighbors,
                center_positions=center_positions,
                need_hij=need_hij,
                dataset_kind=dataset_kind,
            )

    stats = _compute_stats(datasets["train"])
    if str(cache_format).lower() == "lazy":
        _write_lazy_cache(datasets, stats, cache_root)
    else:
        _write_sharded_cache(datasets, stats, cache_root, shard_size=cache_shard_size)


def build_vib_loaders(
    pt_path: str,
    skeleton_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    mp_context: Optional[str] = "spawn",
    cache_format: str = "sharded",
    cache_shard_size: int = 256,
    cache_shard_keep: int = 2,
    radius: float = DEFAULT_RADIUS,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    center_positions: bool = True,
    need_hij: bool = True,
    cache_dir: Optional[str] = "datasets/cache",
    use_cache: bool = True,
    auto_build_cache: bool = True,
    require_cache: bool = False,
):
    """Build vib tensor dataloaders (Hii/Hij/dedipole/depolar)."""

    edge_radius, edge_max_neighbors = _resolve_edge_params(pt_path, radius, max_neighbors)
    dataset_kind = _infer_dataset_kind(pt_path)
    if (edge_radius, edge_max_neighbors) != (radius, max_neighbors):
        logger.info("[edge] qme14s detected, using radius=%s max_neighbors=%s", edge_radius, edge_max_neighbors)

    cache_root = None
    if use_cache and cache_dir is not None:
        try:
            _SCHEMA_VERSION = "v2_polar_vec6"
            key_src = (
                f"{pt_path}|{skeleton_path}|{edge_radius}|{edge_max_neighbors}|"
                f"{split_ratios}|{seed}|{center_positions}|need_hij={bool(need_hij)}|dataset={dataset_kind}"
                f"|schema={_SCHEMA_VERSION}"
            )
            key = hashlib.md5(key_src.encode()).hexdigest()[:16]
            cache_root = Path(cache_dir) / f"vib_tensor_cache_{key}"
            meta_path = cache_root / "meta.pt"
            cache_root.mkdir(parents=True, exist_ok=True)
            logger.info("[cache] vib cache_root=%s", cache_root)

            if meta_path.exists():
                logger.info("[cache] hit: loading vib cache metadata from %s", meta_path)
                try:
                    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                except TypeError:
                    meta = torch.load(meta_path, map_location="cpu")
                meta_format = str(meta.get("format", "lazy")).lower()
                if meta_format == "sharded":
                    datasets = {
                        split: ShardedCachedDataset(
                            meta["splits"][split]["paths"],
                            meta["splits"][split]["sizes"],
                            cache_shards=cache_shard_keep,
                        )
                        for split in ["train", "val", "test"]
                    }
                else:
                    datasets = {split: LazyCachedDataset(meta["paths"][split]) for split in ["train", "val", "test"]}
                loaders = {}
                nw = int(num_workers)
                pm = bool(pin_memory)
                pw_train = bool(persistent_workers) if nw > 0 else False
                pf_train = int(prefetch_factor) if nw > 0 else None
                pf_eval = 1 if nw > 0 else None
                for split in ["train", "val", "test"]:
                    loaders[split] = DataLoader(
                        datasets[split],
                        batch_size=batch_size,
                        shuffle=(split == "train"),
                        num_workers=num_workers,
                        pin_memory=pm,
                        persistent_workers=(pw_train if split == "train" else False),
                        prefetch_factor=(pf_train if split == "train" else pf_eval),
                        multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
                    )
                stats = meta.get("stats", None)
                if stats is None:
                    stats = _compute_stats(datasets["train"])
                return loaders, stats

            if auto_build_cache:
                logger.info("[cache] miss: building vib cache at %s", cache_root)
                try:
                    import torch.multiprocessing as mp

                    ctx = mp.get_context("spawn")
                    process = ctx.Process(
                        target=_build_vib_cache_only,
                        args=(
                            pt_path,
                            skeleton_path,
                            edge_radius,
                            edge_max_neighbors,
                            split_ratios,
                            seed,
                            center_positions,
                            cache_root,
                            need_hij,
                            cache_format,
                            cache_shard_size,
                        ),
                    )
                    process.start()
                    process.join()
                except Exception:
                    pass

                if meta_path.exists():
                    logger.info("[cache] build complete: loading vib cache metadata from %s", meta_path)
                    try:
                        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        meta = torch.load(meta_path, map_location="cpu")
                    meta_format = str(meta.get("format", "lazy")).lower()
                    if meta_format == "sharded":
                        datasets = {
                            split: ShardedCachedDataset(
                                meta["splits"][split]["paths"],
                                meta["splits"][split]["sizes"],
                                cache_shards=cache_shard_keep,
                            )
                            for split in ["train", "val", "test"]
                        }
                    else:
                        datasets = {split: LazyCachedDataset(meta["paths"][split]) for split in ["train", "val", "test"]}
                    loaders = {}
                    nw = int(num_workers)
                    pm = bool(pin_memory)
                    pw_train = bool(persistent_workers) if nw > 0 else False
                    pf_train = int(prefetch_factor) if nw > 0 else None
                    pf_eval = 1 if nw > 0 else None
                    mp_ctx = None if mp_context in (None, "none") else mp_context
                    for split in ["train", "val", "test"]:
                        loaders[split] = DataLoader(
                            datasets[split],
                            batch_size=batch_size,
                            shuffle=(split == "train"),
                            num_workers=num_workers,
                            pin_memory=pm,
                            persistent_workers=(pw_train if split == "train" else False),
                            prefetch_factor=(pf_train if split == "train" else pf_eval),
                            multiprocessing_context=mp_ctx if num_workers > 0 else None,
                        )
                    stats = meta.get("stats", None)
                    if stats is None:
                        stats = _compute_stats(datasets["train"])
                    return loaders, stats
        except Exception:
            logger.warning("[cache] vib cache load/build failed, falling back to direct dataset construction")
            cache_root = None

    if require_cache:
        raise RuntimeError(
            "Lazy cache not found. Please build cache once (e.g., --build_cache_only) and rerun training."
        )

    try:
        raw = torch.load(pt_path, weights_only=False)
    except TypeError:
        raw = torch.load(pt_path)

    datasets: Dict[str, Dataset] = {}
    if skeleton_path is not None:
        logger.info("[skeleton] building vib dataset from precomputed skeleton file: %s", skeleton_path)
        if not os.path.exists(skeleton_path):
            raise FileNotFoundError(f"skeleton_path not found: {skeleton_path}")
        try:
            skeleton_raw = torch.load(skeleton_path, weights_only=False)
        except TypeError:
            skeleton_raw = torch.load(skeleton_path)
        skeleton_variants: Dict[int, List[Dict[str, Any]]] = {}
        for key, value in tqdm(skeleton_raw.items(), desc="Index skeleton", leave=False):
            try:
                mol_id = int(key)
            except Exception:
                continue
            if isinstance(value, list):
                variants = [item for item in value if isinstance(item, dict)]
            elif isinstance(value, dict):
                variants = [value]
            else:
                variants = []
            if variants:
                skeleton_variants[mol_id] = variants
        mol_ids = sorted(skeleton_variants.keys())
        splits = _build_splits(len(mol_ids), split_ratios, seed, ids=mol_ids)
        for split in ["train", "val", "test"]:
            datasets[split] = VibTensorSkeletonDataset(
                raw_list=raw,
                skeleton_variants=skeleton_variants,
                split_ids=splits[split],
                radius=edge_radius,
                max_neighbors=edge_max_neighbors,
                center_positions=center_positions,
                need_hij=need_hij,
                dataset_kind=dataset_kind,
            )
    else:
        logger.info("[skeleton] no skeleton provided; vib dataset will use atom-only fallback")
        splits = _build_splits(len(raw), split_ratios, seed)
        for split in ["train", "val", "test"]:
            idxs = splits[split]
            datasets[split] = VibTensorDataset(
                pt_path=pt_path,
                indices=idxs,
                radius=edge_radius,
                max_neighbors=edge_max_neighbors,
                center_positions=center_positions,
                need_hij=need_hij,
                dataset_kind=dataset_kind,
            )

    stats = _compute_stats(datasets["train"])

    if use_cache and cache_root is not None:
        try:
            if str(cache_format).lower() == "lazy":
                split_paths = _write_lazy_cache(datasets, stats, cache_root)
                datasets = {split: LazyCachedDataset(split_paths[split]) for split in ["train", "val", "test"]}
            else:
                split_meta = _write_sharded_cache(datasets, stats, cache_root, shard_size=cache_shard_size)
                datasets = {
                    split: ShardedCachedDataset(
                        split_meta[split]["paths"],
                        split_meta[split]["sizes"],
                        cache_shards=cache_shard_keep,
                    )
                    for split in ["train", "val", "test"]
                }
            try:
                del raw
            except Exception:
                pass
            gc.collect()
            loaders = {}
            nw = int(num_workers)
            pm = bool(pin_memory)
            pw_train = bool(persistent_workers) if nw > 0 else False
            pf_train = int(prefetch_factor) if nw > 0 else None
            pf_eval = 1 if nw > 0 else None
            for split in ["train", "val", "test"]:
                loaders[split] = DataLoader(
                    datasets[split],
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    pin_memory=pm,
                    persistent_workers=(pw_train if split == "train" else False),
                    prefetch_factor=(pf_train if split == "train" else pf_eval),
                    multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
                )
        except Exception:
            pass

    if "loaders" not in locals():
        loaders = {}
        nw = int(num_workers)
        pm = bool(pin_memory)
        pw_train = bool(persistent_workers) if nw > 0 else False
        pf_train = int(prefetch_factor) if nw > 0 else None
        pf_eval = 1 if nw > 0 else None
        for split in ["train", "val", "test"]:
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pm,
                persistent_workers=(pw_train if split == "train" else False),
                prefetch_factor=(pf_train if split == "train" else pf_eval),
                multiprocessing_context=(None if mp_context in (None, "none") else mp_context) if num_workers > 0 else None,
            )

    return loaders, stats


__all__ = [
    "DEFAULT_RADIUS",
    "DEFAULT_MAX_NEIGHBORS",
    "QME14S_EDGE_RADIUS",
    "QME14S_EDGE_MAX_NEIGHBORS",
    "build_polar_dipole_loaders",
    "build_vib_loaders",
]