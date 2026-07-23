"""prep_prior_cache.py — Offline precompute of NBO electron-prior cache (SENK standalone).

Uses the SENK data loading pipeline (dataloader_spectra.build_vib_loaders) and the
NBOPriorBranch model directly, without depending on train_detanet.py.

Usage examples
--------------
python prep_prior_cache.py \\
    --dataset qme14s_opt186102 \\
    --electron-prior-mode simg \\
    --electron-prior-ckpt nbo_nets/checkpoints/.../nbo_foundation_v2_best.pt \\
    --gpu 0

python prep_prior_cache.py \\
    --dataset qm9s \\
    --electron-prior-mode qcmol \\
    --electron-prior-ckpt nbo_nets/checkpoints/.../nbo_foundation_training_best.pt \\
    --electron-prior-stats nbo_nets/checkpoints/.../nbo_foundation_training_norm_stats.pt \\
    --gpu 0

# Force rebuild
python prep_prior_cache.py ... --refresh
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from dataloader_spectra import build_vib_loaders
from detanet_nets.electron_prior import NBOPriorBranch

logger = logging.getLogger(__name__)


# --- Default paths
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_PT_PATHS = {
    "qme14s_opt186102": str(PROJECT_DIR / "datasets" / "QMe14S" / "qme14s_opt186102.pt"),
    "qm9s":             str(PROJECT_DIR / "datasets" / "qm9s.pt"),
}

DEFAULT_SKELETON_PATHS = {
    "qme14s_opt186102": str(PROJECT_DIR / "datasets" / "QMe14S" / "skeleton_all.pt"),
    "qm9s":             str(PROJECT_DIR / "datasets" / "new_skeleton" / "skeleton_all.pt"),
}


# --- Helper functions


def _task_family(task: str) -> str:
    return "hij" if task in ("hij", "spectra4", "both") else task


def _need_hij(task: str) -> bool:
    return task in ("hij", "spectra4")


def _electron_prior_cache_root(
    task: str,
    args: argparse.Namespace,
    pt_path: Path,
    skeleton_path: Optional[Path],
) -> Path:
    """Determine the cache root directory; consistent with train_senk.py._electron_prior_cache_root."""
    ckpt = Path(args.electron_prior_ckpt).resolve() if args.electron_prior_ckpt else Path("")
    stats = Path(args.electron_prior_stats).resolve() if args.electron_prior_stats else Path("")
    key_src = "|".join([
        _task_family(task),
        f"pt={pt_path}",
        f"skeleton={skeleton_path}",
        f"radius={float(args.radius)}",
        f"max_neighbors={int(args.max_neighbors)}",
        f"split={tuple(args.split_ratios)}",
        f"seed={int(args.seed)}",
        f"center={bool(args.center_positions)}",
        f"need_hij={_need_hij(task)}",
        f"mode={args.electron_prior_mode}",
        f"ckpt={ckpt}",
        f"stats={stats}",
        f"use_aux={bool(args.electron_prior_use_aux)}",
        f"scale={float(args.electron_prior_scale)}",
    ])
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:16]
    root = Path(args.electron_prior_cache_dir) / f"electron_prior_cache_{key}"
    return root.resolve()


def _resolve_max_atomic_number(pt_path: Path) -> int:
    """Infer the maximum atomic number from the dataset."""
    try:
        raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(pt_path, map_location="cpu")
    if isinstance(raw, dict):
        raw = list(raw.values())[0] if raw else []
    max_z = 0
    for data in raw:
        z = getattr(data, "z", None)
        if isinstance(z, torch.Tensor) and z.numel() > 0:
            max_z = max(max_z, int(z.max().item()))
    return max(max_z, 10)  # at least cover H–Ne


def _resolve_dataset_paths(args: argparse.Namespace) -> Tuple[Path, Optional[Path], int]:
    if args.pt_path:
        pt_path = Path(args.pt_path).resolve()
    elif args.dataset in DEFAULT_PT_PATHS:
        pt_path = Path(DEFAULT_PT_PATHS[args.dataset])
    else:
        raise ValueError(f"Unknown dataset {args.dataset!r}, provide --pt-path")

    skeleton_path = None
    if args.skeleton_path:
        skeleton_path = Path(args.skeleton_path).resolve()
    elif args.dataset in DEFAULT_SKELETON_PATHS:
        sp = Path(DEFAULT_SKELETON_PATHS[args.dataset])
        if sp.exists():
            skeleton_path = sp

    max_atomic_number = _resolve_max_atomic_number(pt_path)
    return pt_path, skeleton_path, max_atomic_number


def _build_cache(
    args: argparse.Namespace,
    device: torch.device,
    pt_path: Path,
    skeleton_path: Optional[Path],
    max_atomic_number: int,
) -> Path:
    """Build the electron-prior cache: load data -> NBO inference -> sharded storage."""
    cache_root = _electron_prior_cache_root(args.task, args, pt_path, skeleton_path)
    meta_path = cache_root / "meta.pt"
    if meta_path.exists() and not args.refresh:
        logger.info("[cache] cache exists: %s", cache_root)
        return cache_root

    # 1. Build the data loaders
    task = str(args.task).lower()
    need_hij_flag = _need_hij(task)
    loaders, _ = build_vib_loaders(
        pt_path=str(pt_path),
        skeleton_path=str(skeleton_path) if skeleton_path else None,
        batch_size=1,                               # batch_size=1 for cache construction (sequential, memory-efficient)
        radius=args.radius,
        max_neighbors=args.max_neighbors,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
        center_positions=args.center_positions,
        need_hij=need_hij_flag,
        num_workers=args.num_workers,
        use_cache=False,
        auto_build_cache=False,
        require_cache=False,
    )
    logger.info("[cache] dataset loaded, train=%s val=%s test=%s",
                len(loaders.get('train', []).dataset),
                len(loaders.get('val', []).dataset),
                len(loaders.get('test', []).dataset))

    # 2. Build the NBO prior model
    ckpt_path = Path(args.electron_prior_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (PROJECT_DIR / ckpt_path).resolve()
    stats_path = None
    if args.electron_prior_stats:
        stats_path = Path(args.electron_prior_stats)
        if not stats_path.is_absolute():
            stats_path = (PROJECT_DIR / stats_path).resolve()

    prior = NBOPriorBranch(
        num_features=128,
        num_radial=32,
        mode=args.electron_prior_mode,
        checkpoint_path=str(ckpt_path),
        stats_path=(str(stats_path) if stats_path else None),
        hidden_dim=128,
        max_atomic_number=max_atomic_number,
        feature_scale=args.electron_prior_scale,
        use_auxiliary=args.electron_prior_use_aux,
        freeze_predictor=True,
        runtime_mode="detached",
    ).to(device)
    prior.eval()
    prior._forward_profile_pending = False
    logger.info("[cache] NBO prior model loaded (%s)", args.electron_prior_mode)

    # 3. Iterate over the data and generate the cache
    shard_size = max(1, int(args.shard_size))
    split_meta: Dict[str, Dict[str, List]] = {}

    for split in ["train", "val", "test"]:
        if split not in loaders:
            logger.info("[cache] skip %s: no loader", split)
            continue

        split_dir = cache_root / "shards" / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = loaders[split].dataset
        cache_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        paths: List[str] = []
        sizes: List[int] = []
        buf: List[Dict[str, torch.Tensor]] = []

        for data in tqdm(cache_loader, desc=f"[cache] {split}", leave=True, dynamic_ncols=True):
            data = data.to(device, non_blocking=(device.type == "cuda"))
            entry = prior.export_cache_tensors(
                z=data.z, pos=data.pos, edge_index=data.edge_index, data=data
            )
            buf.append(entry)
            if len(buf) >= shard_size:
                shard_path = split_dir / f"shard_{len(paths):06d}.pt"
                torch.save(buf, shard_path)
                logger.info("[cache] %s: %s (%s entries)", split, shard_path.name, len(buf))
                paths.append(str(shard_path))
                sizes.append(len(buf))
                buf = []

        if buf:
            shard_path = split_dir / f"shard_{len(paths):06d}.pt"
            torch.save(buf, shard_path)
            logger.info("[cache] %s: %s (%s entries)", split, shard_path.name, len(buf))
            paths.append(str(shard_path))
            sizes.append(len(buf))

        split_meta[split] = {"paths": paths, "sizes": sizes}
        total_entries = sum(sizes)
        logger.info("[cache] %s done: %s entries, %s shards", split, total_entries, len(paths))

    meta = {
        "format": "sharded",
        "cache_shards": args.keep_shards,
        "splits": split_meta,
        "task": args.task,
        "mode": args.electron_prior_mode,
        "checkpoint": str(ckpt_path),
        "stats": (str(stats_path) if stats_path else None),
        "use_auxiliary": bool(args.electron_prior_use_aux),
        "feature_scale": float(args.electron_prior_scale),
        "pt_path": str(pt_path),
        "skeleton_path": (str(skeleton_path) if skeleton_path else None),
        "need_hij": need_hij_flag,
    }
    torch.save(meta, meta_path)
    logger.info("[cache] metadata written: %s", meta_path)
    return cache_root


# --- Command-line interface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline electron prior cache builder (SENK standalone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--dataset", choices=["qm9s", "qme14s_opt186102", "custom"],
                   default="qme14s_opt186102")
    p.add_argument("--task", choices=["hii", "hij", "dedipole", "depolar", "spectra4", "polar", "both"],
                   default="spectra4",
                   help="Task type; spectra4 = hii+hij+dedipole+depolar")
    p.add_argument("--pt-path", type=str, default=None, help="Custom .pt data path (required when --dataset custom)")
    p.add_argument("--skeleton-path", type=str, default=None, help="Path to skeleton_all.pt (auto-inferred if omitted)")
    p.add_argument("--radius", type=float, default=5.0)
    p.add_argument("--max-neighbors", type=int, default=64)
    p.add_argument("--split-ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--center-positions", action="store_true", default=True)
    p.add_argument("--no-center-positions", action="store_false", dest="center_positions")

    # Electron prior
    p.add_argument("--electron-prior-mode", choices=["simg", "qcmol"], required=True,
                   help="NBO predictor type")
    p.add_argument("--electron-prior-ckpt", required=True,
                   help="Checkpoint path for NBOFoundationModel")
    p.add_argument("--electron-prior-stats", default=None,
                   help="Normalization stats path (optional; not needed if embedded in the checkpoint)")
    p.add_argument("--electron-prior-scale", type=float, default=1e-2)
    p.add_argument("--electron-prior-use-aux", action="store_true", default=True)
    p.add_argument("--no-electron-prior-use-aux", action="store_false", dest="electron_prior_use_aux")

    # Cache
    p.add_argument("--electron-prior-cache-dir", type=str,
                   default="datasets/cache/electron_prior",
                   help="Cache root directory (a subdirectory is auto-created per run via parameter hash)")
    p.add_argument("--shard-size", type=int, default=256,
                   help="Number of samples stored per shard .pt file")
    p.add_argument("--keep-shards", type=int, default=2,
                   help="Number of shards held in memory simultaneously (affects training-time memory)")
    p.add_argument("--refresh", action="store_true", default=False,
                   help="Force cache rebuild even if it already exists")

    # Runtime environment
    p.add_argument("--gpu", type=int, default=None, help="GPU index (None = auto-select)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; auto-reduced to 0 during cache construction")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    # Device selection
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("[prep_prior_cache] start electron prior cache precompute (device=%s)...", device)

    # Path resolution
    pt_path, skeleton_path, max_atomic_number = _resolve_dataset_paths(args)
    logger.info("[prep_prior_cache] pt_path=%s skeleton=%s max_z=%s", pt_path, skeleton_path, max_atomic_number)

    # Build cache
    cache_root = _build_cache(args, device, pt_path, skeleton_path, max_atomic_number)

    elapsed = time.perf_counter() - t0
    logger.info("[prep_prior_cache] done, cache directory: %s", cache_root)
    logger.info("[prep_prior_cache] elapsed %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()