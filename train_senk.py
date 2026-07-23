"""
Full-scale training script for Equiformer Backbone models.

Modes:
  1. clean_equiformer          — V1 Equiformer backbone + polar head (legacy baseline)
  2. equiformer_electron_prior — V1 + ElectronPriorAttention
  3. equiformer_v2             — EquiformerV2 (SO(2)-conv + S² activation)
  4. equiformer_v2_enk         — V2 + ENK (Equivariant Neural Kalman)
  5. equiformer_v2_enk_ep      — V2 + ENK + Electron Prior (recommended)

All modes follow DetaNet training paradigm:
  - Separate polar + depolar models (not shared weights)
  - polar: scatter_sum → molecular α [B,3,3], L1 loss
  - depolar: autograd ∂α/∂R → [N,3,6], L2/L1 loss
  - AdamW + AMSGrad + cosine LR decay + gradient clipping
  - Best model saved on validation metric

Usage:
  # V2 + ENK + Electron Prior (recommended)
  python train_senk.py --mode equiformer_v2_enk_ep \\
      --electron_prior_mode simg \\
      --electron_prior_ckpt nbo_nets/checkpoints/.../nbo_foundation_v2_best.pt \\
      --dataset qme14s_opt186102

  # V2 baseline
  python train_senk.py --mode equiformer_v2 --dataset qme14s_opt186102

  # V1 legacy baseline
  python train_senk.py --mode clean_equiformer --dataset qme14s_opt186102
"""

import argparse
import atexit
import copy
from datetime import datetime
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import senk_train_shared as train_mod

ROOT = Path(__file__).resolve().parent
DEFAULT_ELECTRON_PRIOR_CACHE_DIR = ROOT / "datasets" / "cache" / "electron_prior"


# ============================================================================
#  Utilities
# ============================================================================

def _resolve_local_path(path_str: str) -> str:
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def _default_log_path() -> str:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return str((log_dir / f"train_{ts}.log").resolve())


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        for stream in self._streams:
            try:
                if stream.isatty():
                    return True
            except Exception:
                continue
        return False


def _setup_log_stream(log_file: Optional[str]) -> Optional[str]:
    if log_file is None:
        log_path = _default_log_path()
    else:
        lower = str(log_file).strip().lower()
        if lower in {"none", "null", "false", "0"}:
            return None
        log_path = _default_log_path() if lower == "" else log_file
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_fh = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, log_fh)
    sys.stderr = _TeeStream(sys.stderr, log_fh)
    atexit.register(log_fh.close)
    return log_path


def _should_require_skeleton(args: argparse.Namespace) -> bool:
    return str(getattr(args, "electron_prior_mode", "off")).lower() != "off"


def _infer_default_skeleton_path(dataset: str, pt_path: str) -> Optional[str]:
    dataset_name = str(dataset).lower()
    pt_name = str(pt_path).lower()
    if dataset_name == "qm9s" or "qm9s" in pt_name or "qm9" in pt_name:
        return str(train_mod.DEFAULT_QM9S_SKELETON)
    if dataset_name in {"qme14s_opt186102"} or "qme14s" in pt_name:
        return str(train_mod.DEFAULT_QME14S_SKELETON)
    return None


def _clone_batch(batch):
    return batch.clone() if hasattr(batch, "clone") else batch


def _electron_prior_task_family(task: str) -> str:
    return "vib" if str(task).lower() in {"depolar", "dedipole", "hii", "hij", "sobolev_polar"} else "polar"


def _electron_prior_need_hij(task: str) -> bool:
    return str(task).lower() in {"hij", "spectra4"}


def _should_use_electron_prior_cache(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "mode", "")).lower() == "equiformer_electron_prior"
        and str(getattr(args, "electron_prior_runtime_mode", "full")).lower() == "cached"
    )


def _electron_prior_cache_root(
    task: str,
    args: argparse.Namespace,
    pt_path: Path,
    skeleton_path: Optional[Path],
) -> Path:
    family = _electron_prior_task_family(task)
    need_hij = _electron_prior_need_hij(task)
    ckpt = Path(args.electron_prior_ckpt).resolve() if args.electron_prior_ckpt else Path("")
    stats = Path(args.electron_prior_stats).resolve() if args.electron_prior_stats else Path("")
    key_src = "|".join([
        family,
        f"pt={pt_path}",
        f"skeleton={skeleton_path}",
        f"radius={float(args.radius)}",
        f"max_neighbors={int(args.max_neighbors)}",
        f"split={tuple(args.split_ratios)}",
        f"seed={int(args.seed)}",
        f"center={bool(args.center)}",
        f"need_hij={need_hij}",
        f"mode={args.electron_prior_mode}",
        f"ckpt={ckpt}",
        f"stats={stats}",
        f"use_aux={bool(args.electron_prior_use_aux)}",
        f"scale={float(args.electron_prior_scale)}",
    ])
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:16]
    return Path(args.electron_prior_cache_dir) / f"electron_prior_cache_{key}"


def _rebuild_loader_from_dataset(split: str, dataset, args: argparse.Namespace):
    num_workers = int(args.workers)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": (split == "train"),
        "num_workers": num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers) if split == "train" else False
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
        mp_context = str(args.mp_context)
        if mp_context != "none":
            kwargs["multiprocessing_context"] = mp_context
    return DataLoader(**kwargs)


def _attach_electron_prior_cache_to_loaders(
    task: str,
    loaders,
    args: argparse.Namespace,
):
    from senk_train_shared import ElectronPriorCachedDataset, ElectronPriorShardDataset

    pt_path = Path(args.pt_path).resolve()
    skeleton_path = Path(args.skeleton_path).resolve() if args.skeleton_path else None
    cache_root = _electron_prior_cache_root(task, args, pt_path, skeleton_path)
    meta_path = cache_root / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Electron prior cache not found: {meta_path}. "
            "Please precompute cache first with prep_prior_cache.py or switch to detached/full mode."
        )
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    except TypeError:
        meta = torch.load(meta_path, map_location="cpu")

    out = {}
    cache_shards = int(meta.get("cache_shards", args.electron_prior_cache_keep_shards))
    for split in ["train", "val", "test"]:
        split_meta = meta["splits"][split]
        cache_dataset = ElectronPriorShardDataset(
            split_meta["paths"],
            split_meta["sizes"],
            cache_shards=cache_shards,
        )
        wrapped = ElectronPriorCachedDataset(loaders[split].dataset, cache_dataset)
        out[split] = _rebuild_loader_from_dataset(split, wrapped, args)
    return out, cache_root


def _merge_metric_sums(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value)


def _metric_subset(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    names = [f"{prefix}_mae", f"{prefix}_rmse", f"{prefix}_r2", f"{prefix}_pcc"]
    return {name: float(metrics[name]) for name in names if name in metrics}


def _mat33_to_vec6(polar_33: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] → [B, 6] convention: xx, yy, zz, xy, xz, yz."""
    return torch.stack([
        polar_33[:, 0, 0],
        polar_33[:, 1, 1],
        polar_33[:, 2, 2],
        polar_33[:, 0, 1],
        polar_33[:, 0, 2],
        polar_33[:, 1, 2],
    ], dim=-1)


def _save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _has_nonfinite_grad(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


def _exceeds_pred_limit(pred: torch.Tensor, limit: float) -> Tuple[bool, float]:
    if float(limit) <= 0:
        return False, 0.0
    pred_abs_max = float(pred.detach().abs().max().item()) if pred.numel() > 0 else 0.0
    return pred_abs_max > float(limit), pred_abs_max


def _filter_label_mask(data, attr: str, max_abs: float):
    """Generic graph-level label outlier filter (mirrors train_mod._filter_depolar_mask).

    Returns (atom_mask, n_filtered, n_graphs):
      - atom_mask: bool [N] tensor to keep (True=keep), or None if no filtering needed
      - n_filtered: number of graphs excluded
      - n_graphs: total graphs in batch
    """
    num_graphs = int(getattr(data, 'num_graphs', 1))
    if float(max_abs) <= 0:
        return None, 0, num_graphs
    y = getattr(data, attr, None)
    if y is None or not isinstance(y, torch.Tensor) or y.numel() == 0:
        return None, 0, num_graphs

    batch = getattr(data, 'batch', None)
    if batch is None or not isinstance(batch, torch.Tensor) or batch.numel() != y.size(0):
        if float(y.abs().max().item()) > max_abs:
            return torch.zeros(y.size(0), dtype=torch.bool, device=y.device), num_graphs, num_graphs
        return None, 0, num_graphs

    per_node_max = y.abs().view(y.size(0), -1).amax(dim=1)
    graph_max = torch.zeros(num_graphs, device=y.device)
    try:
        graph_max.scatter_reduce_(0, batch, per_node_max, reduce='amax')
    except Exception:
        for g in range(num_graphs):
            mask_g = batch == g
            if mask_g.any():
                graph_max[g] = per_node_max[mask_g].max()

    bad_graphs = graph_max > float(max_abs)
    n_filtered = int(bad_graphs.sum().item())
    if n_filtered == 0:
        return None, 0, num_graphs

    # Build per-atom keep mask
    keep_graphs = ~bad_graphs
    atom_mask = keep_graphs[batch]
    return atom_mask, n_filtered, num_graphs


# ============================================================================
#  Checkpoint architecture auto-detection
# ============================================================================

# (lmax, sphere_channels) → v2_model_name preset
_LMAX_C_TO_MODEL = {
    (3, 128): "equiformer_v2_l3_m2",
    (4, 128): "equiformer_v2_l4_m2",
    (4,  64): "equiformer_v2_l4_m2_small",
    (6, 128): "equiformer_v2_l6_m2",
}


def _detect_arch_from_ckpt(ckpt_path: str, device: str = "cpu") -> dict:
    """Inspect a V2 checkpoint and return detected architecture params.

    Returns dict with a subset of:
      model_name      str   e.g. 'equiformer_v2_l3_m2'
      num_layers      int   actual block count
      grid_resolution int   SO3_Grid resolution (from SO3_grid matrix shape)
      enk_enabled     bool  checkpoint contains ENK params

    Detection logic:
      num_layers     : max block index + 1  (backbone.blocks.N.*)
      lmax + C       : backbone.blocks.0.ga.so2_conv_1.fc_m0.weight shape[1]
                       = (lmax+1)*2*C
      grid_resolution: backbone.SO3_grid.0.0.to_grid_mat shape[0]
      enk_enabled    : any key contains 'enk_'
    """
    import re as _re
    result: dict = {}
    if not ckpt_path or not os.path.exists(ckpt_path):
        return result
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  [WARN] _detect_arch_from_ckpt: could not load {ckpt_path}: {e}")
        return result

    # strip DDP "module." prefix
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}

    # num_layers
    block_idx = {
        int(m.group(1))
        for k in state
        for m in [_re.match(r'backbone\.blocks\.(\d+)\.', k)] if m
    }
    if block_idx:
        result["num_layers"] = max(block_idx) + 1

    # lmax + sphere_channels
    key_m0 = "backbone.blocks.0.ga.so2_conv_1.fc_m0.weight"
    if key_m0 in state:
        in_feat = state[key_m0].shape[1]
        for C in (128, 64):
            raw = in_feat / (2 * C)
            v = int(raw)
            if abs(v - raw) < 1e-6 and v >= 1:
                lmax = v - 1
                model_name = _LMAX_C_TO_MODEL.get((lmax, C))
                if model_name:
                    result["model_name"] = model_name
                    result["sphere_channels"] = C
                    result["lmax"] = lmax
                    break

    # grid_resolution
    key_grid = "backbone.SO3_grid.0.0.to_grid_mat"
    if key_grid in state:
        result["grid_resolution"] = int(state[key_grid].shape[0])

    # ENK
    result["enk_enabled"] = any("enk_" in k for k in state)
    return result


# ============================================================================
#  Sobolev joint training utilities
# ============================================================================

_SOBOLEV_TRIL_MASK: Optional[torch.Tensor] = None


def _get_tril_mask(device: torch.device) -> torch.Tensor:
    """Lower-triangular mask for extracting 6 unique elements from 3x3 symmetric matrix."""
    global _SOBOLEV_TRIL_MASK
    if _SOBOLEV_TRIL_MASK is None:
        _SOBOLEV_TRIL_MASK = torch.tril(torch.ones(3, 3)).flatten()
    return _SOBOLEV_TRIL_MASK.to(device=device)


def _compute_depolar_from_polar(
    mol_polar: torch.Tensor,
    pos: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Compute per-atom depolarizability ∂α/∂R via autograd.

    Same mathematics as CleanEquiformerPolar._grad_polarizability and
    DetaNet.grad_polarzability, but applied externally so a single polar
    model can jointly train static α and derivative ∂α/∂R (Sobolev mode).

    Args:
        mol_polar: [B, 3, 3] molecular polarizability (graph-level, from scatter_sum)
        pos: [N, 3] atom positions (must have requires_grad=True)
        training: create_graph=True for backprop through autograd

    Returns:
        [N, 3, 6] per-atom depolarizability -∂α/∂R
    """
    mask = _get_tril_mask(mol_polar.device)
    polars_flat = mol_polar.flatten(start_dim=1)[:, mask == 1]  # [B, 6]
    depolar = torch.zeros(pos.shape[0], 3, 6, device=pos.device, dtype=pos.dtype)
    for i in range(6):
        grad_i = torch.autograd.grad(
            polars_flat[:, i].sum(), pos,
            create_graph=training, retain_graph=True,
        )[0]
        depolar[:, :, i] = -grad_i
    return depolar


def _jacobian_floor_loss(
    depolar: torch.Tensor,
    batch: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Penalize when per-molecule Jacobian Frobenius norm falls below a threshold.

    Prevents derivative collapse → constant α(t) → single spike at ω=0.
    Loss = mean(max(0, floor − ||J_mol||_F)²) over molecules.
    """
    if floor <= 0:
        return depolar.new_tensor(0.0)
    atom_norm_sq = (depolar ** 2).sum(dim=(1, 2))  # [N]
    num_graphs = int(batch.max().item()) + 1
    mol_norm_sq = torch.zeros(num_graphs, device=depolar.device, dtype=depolar.dtype)
    mol_norm_sq.index_add_(0, batch, atom_norm_sq)
    mol_norm = torch.sqrt(mol_norm_sq + 1e-12)
    deficit = torch.clamp(floor - mol_norm, min=0.0)
    return (deficit ** 2).mean()


def _compute_dedipole_from_dipole(
    mol_dipole: torch.Tensor,
    pos: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Compute per-atom dedipole ∂μ/∂R via autograd.

    Same mathematics as CleanEquiformerMultiTask._grad_dipole, but applied
    externally so a single dipole model can jointly train static μ and
    derivative ∂μ/∂R (Sobolev dipole mode).

    Args:
        mol_dipole: [B, 3] molecular dipole (graph-level, from scatter_sum)
        pos: [N, 3] atom positions (must have requires_grad=True)
        training: create_graph=True for backprop through autograd

    Returns:
        [N, 3, 3] per-atom dedipole -∂μ/∂R
    """
    dedipole = torch.zeros(pos.shape[0], 3, 3, device=pos.device, dtype=pos.dtype)
    for i in range(3):
        grad_i = torch.autograd.grad(
            mol_dipole[:, i].sum(), pos,
            create_graph=training, retain_graph=True,
        )[0]
        dedipole[:, :, i] = -grad_i
    return dedipole


def _sobolev_metric_subset(metrics: Dict[str, float]) -> Dict[str, float]:
    """Extract both polar and depolar metrics for sobolev_polar branch."""
    combined = {}
    combined.update(_metric_subset(metrics, "polar"))
    combined.update(_metric_subset(metrics, "depolar"))
    return combined


def _sobolev_dipole_metric_subset(metrics: Dict[str, float]) -> Dict[str, float]:
    """Extract both dipole and dedipole metrics for sobolev_dipole branch."""
    combined = {}
    combined.update(_metric_subset(metrics, "dip"))
    combined.update(_metric_subset(metrics, "dedipole"))
    return combined


# ============================================================================
#  NBO-CR warmup helpers
# ============================================================================

def _nbo_cr_effective_weight(args: argparse.Namespace, epoch: int) -> float:
    warmup = int(getattr(args, "nbo_cr_warmup_epochs", 3))
    target = float(getattr(args, "nbo_cr_weight", 0.0))
    if warmup <= 0 or epoch >= warmup:
        return target
    return target * (epoch + 1) / warmup


def _nbo_cr_warmup_gates(model: torch.nn.Module, args: argparse.Namespace, epoch: int):
    warmup = int(getattr(args, "nbo_cr_warmup_epochs", 3))
    if warmup <= 0 or epoch >= warmup:
        return
    progress = (epoch + 1) / warmup
    target_gate = float(getattr(args, "nbo_cr_gate_init", 0.5))
    cur_gate = target_gate * progress

    prior = getattr(model, "electron_prior", None)
    if prior is not None and hasattr(prior, "atom_gate"):
        prior.atom_gate.data.fill_(cur_gate)
        prior.edge_gate.data.fill_(cur_gate)
        if hasattr(prior, "interaction_atom_gate"):
            prior.interaction_atom_gate.data.fill_(cur_gate)
        if hasattr(prior, "interaction_edge_gate"):
            prior.interaction_edge_gate.data.fill_(cur_gate)

    injector = getattr(model, "so3_ep_injector", None)
    if injector is not None and hasattr(injector, "prior_blocks"):
        inj_gate = progress
        for blk_k, blk_mod in injector.prior_blocks.items():
            if hasattr(blk_mod, "gate"):
                blk_mod.gate.data.fill_(inj_gate)


# ============================================================================
#  Argparse
# ============================================================================

def _str_to_bool(val):
    """Parse a string to boolean for argparse."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes", "t")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-scale Equiformer backbone training"
    )

    # --- Mode ---
    parser.add_argument("--mode", type=str, default="equiformer_v2_enk_ep",
                        choices=["clean_equiformer", "equiformer_electron_prior",
                                 "equiformer_v2", "equiformer_v2_enk",
                                 "equiformer_v2_enk_ep", "equiformer_v2_electron_prior"],
                        help="Which Equiformer variant to train. "
                             "equiformer_v2_* uses EquiformerV2 (SO(2)-conv + S² activation). "
                             "equiformer_v2_enk_ep: V2 native path with ENK+ElectronPrior (recommended).")

    # --- Dataset ---
    parser.add_argument("--dataset", type=str,
                        choices=["qm9s", "qme14s_opt186102"],
                        default="qme14s_opt186102")
    parser.add_argument("--pt_path", type=str, default=train_mod.DEFAULT_QM9S_PT)
    parser.add_argument("--skeleton_path", type=str, default=None,
                        help="Optional skeleton_all.pt path. Only needed for electron-prior modes.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--radius", type=float, default=5.0,
                        help="Edge construction radius (also used for electron prior graph).")
    parser.add_argument("--max_neighbors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=_str_to_bool, default=False, choices=[True, False])
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", type=_str_to_bool, default=False, choices=[True, False])
    parser.add_argument("--mp_context", type=str, default="none",
                        choices=["spawn", "fork", "forkserver", "none"])
    parser.add_argument("--auto_build_cache", type=_str_to_bool, default=False, choices=[True, False])
    parser.add_argument("--require_cache", type=_str_to_bool, default=False, choices=[True, False])
    parser.add_argument("--cache_format", type=str, default="sharded",
                        choices=["sharded", "lazy"])
    parser.add_argument("--cache_shard_size", type=int, default=128)
    parser.add_argument("--cache_shard_keep", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=train_mod._resolve_local_cache_dir())
    parser.add_argument("--no_cache", action="store_true")

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--task", type=str, default="depolar",
                        choices=["polar", "depolar",
                                 "sobolev_polar", "sobolev_dipole",
                                 "dipole", "dedipole", "hii", "hij", "spectra4"],
                        help="Task to train. 'sobolev_polar'=joint static α+∂α/∂R via autograd. "
                             "'sobolev_dipole'=joint static μ+∂μ/∂R via autograd (for MD IR weights). "
                             "'spectra4'=hii+hij+dedipole+depolar (sequential).")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=3.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step"])
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--lr_step_size", type=int, default=10,
                        help="StepLR step size when --lr_scheduler=step.")
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="StepLR gamma when --lr_scheduler=step.")
    parser.add_argument("--static_loss_type", type=str, default="l1",
                        choices=["auto", "l1", "l2", "huber"])
    parser.add_argument("--deriv_loss_type", type=str, default="l1",
                        choices=["auto", "l1", "l2", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--depolar_max_abs", type=float, default=500.0,
                        help="Graph-level depolar filter: exclude graphs with max|depolar|>this from "
                             "derivative loss. Applied to both train and val. "
                             "Default 500 covers QMe14S outliers; auto-lowered to 100 for qm9s "
                             "(P99 per-graph max ~113). Set 0 to disable.")
    parser.add_argument("--dedipole_max_abs", type=float, default=0.0,
                        help="Graph-level dedipole label filter: exclude graphs with max|dedipole|>this. "
                             "0 disables. Auto-set to 50.0 for qme14s (dedipole P99 ~40 a.u.). "
                             "QM9 dedipole values are small (<5 a.u.), leave at 0.")
    parser.add_argument("--hii_max_abs", type=float, default=0.0,
                        help="Graph-level Hii label filter: exclude graphs with max|Hii|>this. "
                             "0 disables. Auto-set to 200.0 for qme14s (diagonal Hessian P99 ~170). "
                             "QM9 Hii values are moderate, leave at 0.")
    parser.add_argument("--hij_max_abs", type=float, default=0.0,
                        help="Graph-level Hij label filter: exclude graphs with max|Hij|>this. "
                             "0 disables. Auto-set to 100.0 for qme14s (off-diagonal Hessian P99 ~85). "
                             "Most effective filter: off-diagonal Hessian has the most outliers in QMe14S.")
    parser.add_argument("--skip_nonfinite_batches", type=_str_to_bool, default=True, choices=[True, False],
                        help="Skip train/eval batches when prediction or loss is non-finite.")
    parser.add_argument("--pred_abs_max", type=float, default=500.0,
                        help="Skip batches when max|prediction| exceeds this threshold; <=0 disables.")
    parser.add_argument("--skip_grad_norm", type=float, default=50.0,
                        help="Skip optimizer step when unclipped grad norm exceeds this threshold; <=0 disables.")

    # --- Sobolev joint training (task=sobolev_polar) ---
    parser.add_argument("--sobolev_lambda_static", type=float, default=1.0,
                        help="Weight for static polar loss in sobolev_polar joint training.")
    parser.add_argument("--sobolev_lambda_deriv", type=float, default=1.0,
                        help="Weight for derivative (depolar) loss in sobolev_polar joint training.")
    parser.add_argument("--sobolev_deriv_loss_type", type=str, default="l1",
                        choices=["l1", "l2", "huber"],
                        help="Loss type for derivative in sobolev_polar training. "
                             "L1 bounds second-order gradient magnitude, robust to outliers.")
    parser.add_argument("--sobolev_warmup_epochs", type=int, default=3,
                        help="Pure static warmup: skip autograd + derivative loss for N epochs. "
                             "Lets polar model converge before adding derivative supervision.")
    parser.add_argument("--sobolev_depolar_clip", type=float, default=500.0,
                        help="Clamp autograd depolar predictions to +/-this before loss; <=0 disables.")
    parser.add_argument("--sobolev_jac_floor", type=float, default=0.0,
                        help="Minimum per-molecule Jacobian Frobenius norm. "
                             "Penalizes derivative collapse; <=0 disables.")
    parser.add_argument("--sobolev_best_metric", type=str, default="depolar_mae",
                        choices=["depolar_mae", "polar_mae", "combined"],
                        help="Best model selection metric for sobolev_polar.")

    # --- Sobolev joint dipole training (task=sobolev_dipole) ---
    parser.add_argument("--sobolev_dedipole_clip", type=float, default=50.0,
                        help="Clamp autograd dedipole predictions to +/-this before loss; <=0 disables. "
                             "Dedipole magnitudes are much smaller than depolar (~5 a.u. for QM9).")
    parser.add_argument("--sobolev_dipole_best_metric", type=str, default="dedipole_mae",
                        choices=["dedipole_mae", "dip_mae", "combined"],
                        help="Best model selection metric for sobolev_dipole.")

    # --- Backbone ---
    parser.add_argument("--hidden_nf", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_basis", type=int, default=128)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.05)

    # --- EquiformerV2 backbone options ---
    parser.add_argument("--v2_model_name", type=str, default="equiformer_v2_l4_m2",
                        choices=["equiformer_v2_l4_m2", "equiformer_v2_l6_m2",
                                 "equiformer_v2_l3_m2",
                                 "equiformer_v2_l4_m2_small"],
                        help="V2 factory preset. equiformer_v2_l4_m2: lmax=4,mmax=2,C=128; "
                             "equiformer_v2_l3_m2: lmax=3,mmax=2,C=128 (~30%% faster, QM9 recommended); "
                             "equiformer_v2_l6_m2: lmax=6; equiformer_v2_l4_m2_small: 4L,C=64 (toy).")

    # --- Checkpoint ---
    parser.add_argument("--init_ckpt", type=str, default="",
                        help="Initialize model from checkpoint.")
    parser.add_argument("--polar_init_ckpt", type=str, default="",
                        help="Initialize depolar/sobolev_polar branch from polar checkpoint (weight transfer).")
    parser.add_argument("--dipole_init_ckpt", type=str, default="",
                        help="Initialize sobolev_dipole branch from dipole checkpoint (weight transfer).")

    # --- Electron Prior (mode=equiformer_electron_prior / equiformer_v2_enk_ep) ---
    parser.add_argument("--electron_prior_mode", type=str, default="off",
                        choices=["off", "simg", "qcmol"])
    parser.add_argument("--electron_prior_ckpt", type=str, default="")
    parser.add_argument("--electron_prior_stats", type=str, default="")
    parser.add_argument("--electron_prior_scale", type=float, default=1e-2)
    parser.add_argument("--electron_prior_use_aux", type=_str_to_bool, default=True, choices=[True, False])
    parser.add_argument("--electron_prior_freeze", type=_str_to_bool, default=True, choices=[True, False])
    parser.add_argument("--electron_prior_runtime_mode", type=str, default="full",
                        choices=["full", "detached", "cached"])
    parser.add_argument("--electron_prior_heads", type=int, default=4)
    parser.add_argument("--electron_prior_cache_dir", type=str,
                        default=str(DEFAULT_ELECTRON_PRIOR_CACHE_DIR),
                        help="Directory of precomputed electron-prior cache shards for runtime_mode=cached.")
    parser.add_argument("--electron_prior_cache_keep_shards", type=int, default=2,
                        help="Number of electron-prior cache shards kept in RAM when runtime_mode=cached.")

    # --- NBO Consistency Regularization (NBO-CR) ---
    parser.add_argument("--nbo_cr_weight", type=float, default=0.0,
                        help="NBO-CR consistency loss weight. 0=disabled, >0 enables NBO-CR training. "
                             "Recommended: 0.1-0.5 for EP enhancement stages.")
    parser.add_argument("--nbo_cr_gate_init", type=float, default=0.5,
                        help="Initial value for EP injection gates when NBO-CR is active. "
                             "Old default was 0.01 (near-zero); NBO-CR uses 0.5 to force EP channel open.")
    parser.add_argument("--nbo_cr_alpha_init", type=float, default=0.6,
                        help="Initial Badger-rule exponent for |Hij| ∝ occ^alpha.")
    parser.add_argument("--nbo_cr_lr_scale", type=float, default=0.1,
                        help="LR scale for NBO-CR learnable params relative to base LR.")
    parser.add_argument("--nbo_cr_warmup_epochs", type=int, default=3,
                        help="Epochs over which NBO-CR weight ramps from 0 to full. "
                             "Prevents cold-start shock: EP gates and CR loss start at 0 "
                             "and linearly increase over this many epochs.")

    # --- Equivariant Neural Kalman (ENK) ---
    parser.add_argument("--enk_enabled", type=_str_to_bool, default=False, choices=[True, False],
                        help="Enable ENK module between backbone and polar head.")
    parser.add_argument("--enk_init_r_bias", type=float, default=-1.0,
                        help="ENK initial bias for observation noise logit (negative = trust obs more).")
    parser.add_argument("--enk_init_q_bias", type=float, default=1.0,
                        help="ENK initial bias for process noise logit (positive = trust obs more).")

    # --- V2 backbone memory/speed options ---
    parser.add_argument("--v2_max_neighbors", type=int, default=64,
                        help="V2 backbone max_num_neighbors for radius_graph. QM9 rarely needs >50; "
                             "500 (OCP default) wastes memory. Default 64 covers QM9 comfortably.")
    parser.add_argument("--v2_num_gaussians", type=int, default=64,
                        help="V2 backbone Gaussian RBF basis size. OCP default=600 is overkill for QM9. "
                             "64 is sufficient for 5Å range (~12.8 bins/Å). DetaNet uses 32.")
    parser.add_argument("--v2_grid_resolution", type=int, default=14,
                        help="V2 S² activation grid resolution (SO3_Grid). OCP default=18; "
                             "each TransBlock has grid intermediate [E, res*(2res-1), C]. "
                             "14→(14/18)²=0.61×, 12→0.44× S²-grid memory and compute. "
                             "lmax=4 Nyquist requires ≥9; 14 is safe with margin.")
    parser.add_argument("--v2_gradient_checkpointing", type=_str_to_bool, default=False,
                        choices=[True, False],
                        help="Enable gradient checkpointing on V2 backbone blocks. "
                             "Trades ~2x compute for ~7/8 activation memory. "
                             "Use if --v2_grid_resolution=12 still OOMs.")
    parser.add_argument("--v2_use_gate_act", type=_str_to_bool, default=True,
                        choices=[True, False],
                        help="V2 activation type: True=GateActivation (fast; skip SO3_Grid einsum in "
                             "every block, ~33x cheaper per activation). False=SeparableS2Activation "
                             "(original V2, slightly higher accuracy ceiling). "
                             "Default True for QM9/depolar. GateActivation still benefits from lmax=4.")
    parser.add_argument("--v2_use_grid_mlp", type=_str_to_bool, default=False,
                        choices=[True, False],
                        help="V2 FFN grid MLP: True=project node feats to S²-grid + 3-layer MLP "
                             "(original V2 FFN, expensive). False=use gate/sep-S2-act only. "
                             "Set True only to reproduce OCP paper numbers.")

    # --- System ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="checkpoints/equiformer_backbone")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--log_file", type=str, default="",
                        help="Write training log to a file. If empty, defaults to logs/train_<timestamp>.log; pass none/false to disable.")

    return parser


# ============================================================================
#  Argument normalization
# ============================================================================

def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode in ("equiformer_v2_enk", "equiformer_v2_enk_ep"):
        args.enk_enabled = True
    if args.mode == "equiformer_v2_enk_ep" and getattr(args, "electron_prior_mode", "off") == "off":
        raise ValueError(
            "mode=equiformer_v2_enk_ep requires --electron_prior_mode {simg,qcmol} "
            "and matching electron-prior checkpoint/stat paths."
        )

    # Dataset path resolution
    if args.dataset == "qme14s_opt186102":
        if args.pt_path == train_mod.DEFAULT_QM9S_PT:
            args.pt_path = train_mod.DEFAULT_QME14S_OPT186102_PT

    for attr in ("pt_path", "skeleton_path", "cache_dir", "output_dir",
                 "init_ckpt", "polar_init_ckpt", "electron_prior_ckpt",
                 "electron_prior_stats", "electron_prior_cache_dir", "log_file"):
        val = getattr(args, attr, "")
        if val:
            setattr(args, attr, _resolve_local_path(val))
    if args.output_json:
        args.output_json = _resolve_local_path(args.output_json)

    if _should_require_skeleton(args):
        if not args.skeleton_path:
            args.skeleton_path = _infer_default_skeleton_path(args.dataset, args.pt_path)
        if not args.skeleton_path:
            raise FileNotFoundError(
                "Electron prior requires skeleton data, but no skeleton_path was provided and no default could be inferred."
            )
        if not os.path.exists(args.skeleton_path):
            raise FileNotFoundError(f"skeleton_path not found: {args.skeleton_path}")
    else:
        args.skeleton_path = None

    train_mod._override_edge_params_for_qme14s(args)

    # Fixed flags for SENK models
    args.train_epochs = int(args.epochs)
    args.ssp = False
    args.use_psd = False
    args.geom_only = True
    args.equivariant_backbone = True
    args.polar_head_v2 = True
    args.electron_mode = "off"
    args.electron_aux_weight = 0.0

    if getattr(args, "nbo_cr_weight", 0.0) > 0:
        if args.electron_prior_mode == "off":
            args.electron_prior_mode = "qcmol"
        if args.electron_prior_freeze:
            args.electron_prior_runtime_mode = "full"
        print(f"[NBO-CR] auto-config: mode={args.electron_prior_mode}"
              f" freeze={args.electron_prior_freeze}"
              f" runtime={args.electron_prior_runtime_mode}"
              f" gate_init={args.nbo_cr_gate_init}"
              f" weight={args.nbo_cr_weight}")

    args.strict_detanet_loss = False
    args.enk_radius = float(args.radius)

    if args.dataset == "qme14s_opt186102" and "SENK_MAX_ATOM_TYPE" not in os.environ:
        os.environ["SENK_MAX_ATOM_TYPE"] = "36"

    # QM9S: default depolar_max_abs=500 exceeds the QM9S maximum (~440), so no filtering is applied.
    # Lower to 100 to filter the most extreme ~1.7% of graphs (per-graph max > 100).
    # A user-supplied value < 500 is respected as-is.
    if args.dataset == "qm9s" and args.depolar_max_abs >= 500.0:
        args.depolar_max_abs = 100.0

    # QME14S: enable label-magnitude filtering for dedipole/hii/hij (outlier protection on the label side; only when the user has not set a value manually).
    # QM9 defaults to 0 (disabled) and is not modified here.
    if args.dataset == "qme14s_opt186102":
        if args.dedipole_max_abs == 0.0:
            args.dedipole_max_abs = 50.0   # dedipole P99 ~40 a.u.
        if args.hii_max_abs == 0.0:
            args.hii_max_abs = 200.0       # diagonal Hessian P99 ~170
        if args.hij_max_abs == 0.0:
            args.hij_max_abs = 100.0       # off-diagonal Hessian P99 ~85 (heaviest outlier tail)

    if torch.cuda.is_available():
        if args.gpu is not None:
            args.device = f"cuda:{args.gpu}"
            torch.cuda.set_device(args.gpu)
        elif not str(args.device).startswith("cuda"):
            args.device = "cuda"

    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    return args


# ============================================================================
#  Data loading (reuse existing infrastructure)
# ============================================================================

def _build_loaders(args: argparse.Namespace):
    need_hij = (str(args.task).lower() in ("hij", "spectra4"))
    loaders, stats = train_mod.build_vib_loaders(
        pt_path=args.pt_path,
        skeleton_path=args.skeleton_path,
        batch_size=args.batch_size,
        radius=args.radius,
        max_neighbors=args.max_neighbors,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
        center_positions=args.center,
        need_hij=need_hij,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        mp_context=args.mp_context,
        cache_format=args.cache_format,
        cache_shard_size=args.cache_shard_size,
        cache_shard_keep=args.cache_shard_keep,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        auto_build_cache=args.auto_build_cache,
        require_cache=args.require_cache,
    )
    if _should_use_electron_prior_cache(args):
        loaders, cache_root = _attach_electron_prior_cache_to_loaders(str(args.task).lower(), loaders, args)
        print(f"[cache] electron prior attached from {cache_root}")
    return loaders, stats


# ============================================================================
#  Model construction
# ============================================================================

def _resolve_electron_prior_config(args: argparse.Namespace) -> Optional[Dict]:
    """Build electron prior config dict from args, or None if disabled."""
    mode = str(args.electron_prior_mode).lower()
    if mode == "off":
        return None

    ckpt_path = str(args.electron_prior_ckpt or "")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Electron prior checkpoint not found: {ckpt_path!r}. "
            "Provide --electron_prior_ckpt."
        )

    stats_path = str(args.electron_prior_stats or "")
    if stats_path and not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Electron prior stats not found: {stats_path!r}."
        )

    max_z = int(os.environ.get("SENK_MAX_ATOM_TYPE", "36"))
    return {
        "mode": mode,
        "checkpoint_path": ckpt_path,
        "stats_path": (stats_path if stats_path else None),
        "feature_scale": float(args.electron_prior_scale),
        "use_auxiliary": bool(args.electron_prior_use_aux),
        "freeze_predictor": bool(args.electron_prior_freeze),
        "runtime_mode": str(args.electron_prior_runtime_mode).lower(),
        "hidden_dim": int(args.hidden_nf),
        "max_atomic_number": max_z,
    }


# Multi-task tasks that use CleanEquiformerMultiTask
_MULTITASK_BRANCHES = {"dipole", "dedipole", "hii", "hij"}
# Derivative tasks (need enable_grad, not no_grad)
_DERIVATIVE_BRANCHES = {"depolar", "dedipole", "hii", "hij", "sobolev_polar", "sobolev_dipole"}
# Tasks with spectra4 bundle
_SPECTRA4_TASKS = ["hii", "hij", "dedipole", "depolar"]
# Metric name mapping: branch → prefix used in _finalize_metrics output
_EVAL_METRIC_NAMES = {
    "polar": "polar", "depolar": "depolar", "dipole": "dip",
    "dedipole": "dedipole", "hii": "hii", "hij": "hij",
    "sobolev_polar": None,  # special: uses both polar + depolar
    "sobolev_dipole": None,  # special: uses both dip + dedipole
}


def _build_model(
    kind: str,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Build the appropriate Equiformer model for a given branch.

    Args:
        kind: "polar", "depolar", "dipole", "dedipole", "hii", "hij",
              or "sobolev_polar" (builds a polar model; derivatives computed externally)
        args: parsed arguments with mode and architecture config
    """
    mode = args.mode

    # sobolev_polar: same architecture as polar (summation=True, grad_type=None)
    # sobolev_dipole: same architecture as dipole (summation=True, grad_type=None)
    # Derivatives are computed externally via autograd in the training loop.
    effective_kind = kind
    if kind == "sobolev_polar":
        effective_kind = "polar"
    elif kind == "sobolev_dipole":
        effective_kind = "dipole"

    # --- New multi-task branches (dipole, dedipole, hii, hij) ---
    if effective_kind in _MULTITASK_BRANCHES:
        from nets.clean_equiformer_multitask import CleanEquiformerMultiTask

        # Resolve backbone: V2 when mode requests it, V1 otherwise
        _is_v2_mt = mode.startswith('equiformer_v2')
        if _is_v2_mt:
            _mt_model_name = str(getattr(args, 'v2_model_name', 'equiformer_v2_l4_m2'))
        else:
            _mt_model_name = 'graph_attention_transformer_nonlinear_l2'

        # ENK: enable for V2 modes that request it
        _mt_enk = bool(args.enk_enabled) or (mode in ('equiformer_v2_enk', 'equiformer_v2_enk_ep'))

        # Electron prior: only for V2 modes that request it
        _mt_ep_config = None
        if _is_v2_mt and mode in ('equiformer_v2_enk_ep', 'equiformer_v2_electron_prior'):
            _mt_ep_config = _resolve_electron_prior_config(args)

        model = CleanEquiformerMultiTask(
            task=effective_kind,
            hidden_nf=int(args.hidden_nf),
            model_name=_mt_model_name,
            radius=float(args.radius),
            num_basis=int(args.num_basis),
            num_layers=int(args.num_layers),
            drop_path=float(args.drop_path),
            dropout=float(args.dropout),
            # V2 options
            max_num_neighbors=int(getattr(args, 'v2_max_neighbors', 64)),
            num_gaussians=int(getattr(args, 'v2_num_gaussians', 64)),
            use_gradient_checkpointing=bool(getattr(args, 'v2_gradient_checkpointing', False)),
            grid_resolution=int(getattr(args, 'v2_grid_resolution', 14)),
            use_gate_act=bool(getattr(args, 'v2_use_gate_act', True)),
            use_grid_mlp=bool(getattr(args, 'v2_use_grid_mlp', False)),
            # ENK
            enk_enabled=_mt_enk,
            enk_init_r_bias=float(args.enk_init_r_bias),
            enk_init_q_bias=float(args.enk_init_q_bias),
            # Electron prior
            electron_prior_config=_mt_ep_config,
            electron_prior_heads=int(args.electron_prior_heads),
        )
        model.to(args.device)
        return model

    # --- Original polar/depolar branches ---
    # Resolve model_name: V1 vs V2 backbone
    _is_v2 = mode.startswith('equiformer_v2')
    if _is_v2:
        _model_name = str(getattr(args, 'v2_model_name', 'equiformer_v2_l4_m2'))
    else:
        _model_name = 'graph_attention_transformer_nonlinear_l2'

    # Resolve effective mode (strip v2 prefix for logic below)
    # equiformer_v2 → clean_equiformer
    # equiformer_v2_enk → equiformer_v2_enk (handled separately; uses CleanEquiformerPolarExt)
    # equiformer_v2_enk_ep → equiformer_v2_enk_ep (ditto, with EP enabled)
    # equiformer_v2_XX → equiformer_XX
    _eff_mode = mode.replace('equiformer_v2_', 'equiformer_') \
                    .replace('equiformer_v2', 'clean_equiformer')
    # equiformer_v2_enk / equiformer_v2_enk_ep: route to CleanEquiformerPolarExt
    if mode in ('equiformer_v2_enk', 'equiformer_v2_enk_ep'):
        _eff_mode = 'equiformer_v2_enk'

    if _eff_mode == "clean_equiformer":
        if _is_v2:
            # Plain V2 should also use the native SO3 readout path instead of
            # falling back to the old flat irreps decomposition.
            from nets.clean_equiformer_polar_ext import CleanEquiformerPolarExt
            model = CleanEquiformerPolarExt(
                hidden_nf=int(args.hidden_nf),
                model_name=_model_name,
                radius=float(args.radius),
                num_basis=int(args.num_basis),
                num_layers=int(args.num_layers),
                drop_path=float(args.drop_path),
                dropout=float(args.dropout),
                summation=(effective_kind == "polar"),
                grad_type=("polar" if effective_kind == "depolar" else None),
                electron_prior_config=None,
                electron_prior_heads=int(args.electron_prior_heads),
                enk_enabled=bool(args.enk_enabled),
                enk_init_r_bias=float(args.enk_init_r_bias),
                enk_init_q_bias=float(args.enk_init_q_bias),
                max_num_neighbors=int(getattr(args, 'v2_max_neighbors', 64)),
                num_gaussians=int(getattr(args, 'v2_num_gaussians', 64)),
                use_gradient_checkpointing=bool(getattr(args, 'v2_gradient_checkpointing', False)),
                grid_resolution=int(getattr(args, 'v2_grid_resolution', 14)),
                use_gate_act=bool(getattr(args, 'v2_use_gate_act', True)),
                use_grid_mlp=bool(getattr(args, 'v2_use_grid_mlp', False)),
            )
        else:
            from nets.clean_equiformer_polar import CleanEquiformerPolar
            model = CleanEquiformerPolar(
                hidden_nf=int(args.hidden_nf),
                model_name=_model_name,
                radius=float(args.radius),
                num_basis=int(args.num_basis),
                num_layers=int(args.num_layers),
                drop_path=float(args.drop_path),
                dropout=float(args.dropout),
                summation=(effective_kind == "polar"),
                grad_type=("polar" if effective_kind == "depolar" else None),
                enk_enabled=bool(args.enk_enabled),
                enk_init_r_bias=float(args.enk_init_r_bias),
                enk_init_q_bias=float(args.enk_init_q_bias),
            )
    elif _eff_mode == "equiformer_v2_enk":
        # V2 + ENK (+ optionally ElectronPrior): uses CleanEquiformerPolarExt
        # _use_v2_native_path is auto-detected inside the model; ENK is optional
        from nets.clean_equiformer_polar_ext import CleanEquiformerPolarExt
        _ep_config = None
        if mode == 'equiformer_v2_enk_ep':
            _ep_config = _resolve_electron_prior_config(args)
        model = CleanEquiformerPolarExt(
            hidden_nf=int(args.hidden_nf),
            model_name=_model_name,
            radius=float(args.radius),
            num_basis=int(args.num_basis),
            num_layers=int(args.num_layers),
            drop_path=float(args.drop_path),
            dropout=float(args.dropout),
            summation=(effective_kind == "polar"),
            grad_type=("polar" if effective_kind == "depolar" else None),
            electron_prior_config=_ep_config,
            electron_prior_heads=int(args.electron_prior_heads),
            enk_enabled=bool(args.enk_enabled),
            enk_init_r_bias=float(args.enk_init_r_bias),
            enk_init_q_bias=float(args.enk_init_q_bias),
            max_num_neighbors=int(getattr(args, 'v2_max_neighbors', 64)),
            num_gaussians=int(getattr(args, 'v2_num_gaussians', 64)),
            use_gradient_checkpointing=bool(getattr(args, 'v2_gradient_checkpointing', False)),
            grid_resolution=int(getattr(args, 'v2_grid_resolution', 14)),
            use_gate_act=bool(getattr(args, 'v2_use_gate_act', True)),
            use_grid_mlp=bool(getattr(args, 'v2_use_grid_mlp', False)),
        )
    elif _eff_mode == "equiformer_electron_prior":
        from nets.clean_equiformer_polar_ext import CleanEquiformerPolarExt

        electron_prior_config = _resolve_electron_prior_config(args)

        model = CleanEquiformerPolarExt(
            hidden_nf=int(args.hidden_nf),
            model_name=_model_name,
            radius=float(args.radius),
            num_basis=int(args.num_basis),
            num_layers=int(args.num_layers),
            drop_path=float(args.drop_path),
            dropout=float(args.dropout),
            summation=(effective_kind == "polar"),
            grad_type=("polar" if effective_kind == "depolar" else None),
            electron_prior_config=electron_prior_config,
            electron_prior_heads=int(args.electron_prior_heads),
            enk_enabled=bool(args.enk_enabled),
            enk_init_r_bias=float(args.enk_init_r_bias),
            enk_init_q_bias=float(args.enk_init_q_bias),
            max_num_neighbors=int(getattr(args, 'v2_max_neighbors', 64)),
            num_gaussians=int(getattr(args, 'v2_num_gaussians', 64)),
            use_gradient_checkpointing=bool(getattr(args, 'v2_gradient_checkpointing', False)),
            grid_resolution=int(getattr(args, 'v2_grid_resolution', 14)),
            use_gate_act=bool(getattr(args, 'v2_use_gate_act', True)),
            use_grid_mlp=bool(getattr(args, 'v2_use_grid_mlp', False)),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.to(args.device)
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str) -> Dict:
    """Load checkpoint into model, return info dict.

    Uses strict=False and additionally skips keys whose shape does not match
    the current model to prevent RuntimeError on architecture mismatches.
    Shape-mismatched keys are reported in info["shape_mismatch_keys"].
    """
    info = {"loaded": False, "ckpt_path": ckpt_path,
            "missing_keys": [], "unexpected_keys": [], "shape_mismatch_keys": []}
    if not ckpt_path or not os.path.exists(ckpt_path):
        return info
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cleaned = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "") if key.startswith("module.") else key] = value

    # Filter out shape-mismatched keys so load_state_dict(strict=False) doesn't raise
    model_state = model.state_dict()
    compatible, mismatched = {}, []
    for k, v in cleaned.items():
        if k in model_state and model_state[k].shape != v.shape:
            mismatched.append(k)
        else:
            compatible[k] = v
    if mismatched:
        print(f"  [WARN] _load_checkpoint: {len(mismatched)} shape-mismatched keys skipped "
              f"(ckpt arch differs from model arch): {mismatched[:3]}"
              f"{'...' if len(mismatched) > 3 else ''}")

    msg = model.load_state_dict(compatible, strict=False)
    info["missing_keys"] = list(msg.missing_keys)
    info["unexpected_keys"] = list(msg.unexpected_keys)
    info["shape_mismatch_keys"] = mismatched
    info["loaded"] = True
    return info


# ============================================================================
#  Forward helpers
# ============================================================================

def _forward_model(model: torch.nn.Module, data, mode: str, branch: str = "depolar") -> torch.Tensor:
    """Unified forward call for all modes and tasks.

    For multi-task (dipole/dedipole/hii/hij): calls with task-specific args
    For clean_equiformer: calls model(pos=, z=, batch=)
    For ext modes: additionally passes data= for mask/prior access
    """
    if branch in _MULTITASK_BRANCHES or branch == "sobolev_dipole":
        kwargs = dict(pos=data.pos, z=data.z, batch=data.batch)
        if branch == "hij":
            kwargs["edge_index"] = data.edge_index
        kwargs["data"] = data  # EP needs full data for edge construction
        return model(**kwargs)
    if mode == "clean_equiformer":
        return model(pos=data.pos, z=data.z, batch=data.batch)
    else:
        return model(pos=data.pos, z=data.z, batch=data.batch, data=data)


# ============================================================================
#  Loss computation
# ============================================================================

def _resolve_loss_type(loss_type: str, task_name: str) -> str:
    lt = str(loss_type).lower()
    if lt in ("l1", "l2", "huber"):
        return lt
    # auto
    if task_name in ("depolar", "sobolev_depolar"):
        return "l2"
    return "l1"


def _regression_loss(x: torch.Tensor, y: torch.Tensor,
                     loss_type: str, task_name: str,
                     huber_delta: float = 1.0) -> torch.Tensor:
    lt = _resolve_loss_type(loss_type, task_name)
    if lt == "l1":
        return F.l1_loss(x, y)
    if lt == "l2":
        return F.mse_loss(x, y)
    return F.smooth_l1_loss(x, y, beta=float(huber_delta))


# ============================================================================
#  Training one epoch
# ============================================================================

def _train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    stats: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    epoch: int,
    branch: str,  # "polar", "depolar", "sobolev_polar", "sobolev_dipole", "dipole", "dedipole", "hii", "hij"
) -> Dict[str, object]:
    """Train one epoch for a single branch."""
    model.train()
    _nbo_cr_warmup_gates(model, args, epoch)
    metric_sums = train_mod._init_metric_sums()
    loss_sum = 0.0
    steps = 0
    skipped_nonfinite = 0
    skipped_pred_limit = 0
    skipped_grad_limit = 0
    nbo_cr_accum = 0.0
    nbo_cr_steps = 0

    # Branch-specific config
    if branch == "polar":
        task_str = "polar"
        loss_type = args.static_loss_type
    elif branch == "depolar":
        task_str = "sobolev_depolar"
        loss_type = args.deriv_loss_type
    elif branch == "sobolev_polar":
        task_str = "sobolev_polar"
        loss_type = args.static_loss_type  # static branch type; deriv uses sobolev_deriv_loss_type
    elif branch == "sobolev_dipole":
        task_str = "sobolev_dipole"
        loss_type = args.static_loss_type  # static branch type; deriv uses sobolev_deriv_loss_type
    elif branch == "dipole":
        task_str = "dipole"
        loss_type = args.static_loss_type
    elif branch in ("dedipole", "hii", "hij"):
        task_str = branch
        loss_type = args.deriv_loss_type
    else:
        raise ValueError(f"Unknown branch: {branch}")

    try:
        from tqdm import tqdm as _tqdm_cls
        _iter = _tqdm_cls(loader, desc=f"[{branch}] ep{epoch}", leave=False,
                          mininterval=10.0, unit="batch", ncols=120)
    except ImportError:
        _iter = loader

    for step, batch_cpu in enumerate(_iter):
        data = _clone_batch(batch_cpu).to(args.device, non_blocking=False)

        # Depolar filtering
        depolar_mask = None
        if branch == "depolar" and float(args.depolar_max_abs) > 0:
            depolar_mask, _, _ = train_mod._filter_depolar_mask(data, args.depolar_max_abs)
            if depolar_mask is not None and depolar_mask.sum().item() == 0:
                continue

        optimizer.zero_grad(set_to_none=True)

        # Branch-specific consistency tensors for filtering / thresholds / metrics.
        pred_for_limit = None
        hij_pred_metric = None
        hij_tgt_metric = None

        with torch.enable_grad():
            # --- Sobolev joint polar: enable grad on pos for autograd ---
            # Warmup phase: skip autograd entirely (pure static, saves compute + avoids
            # second-order gradient instability from random model weights).
            _sobolev_pos_grad = None
            _sobolev_depolar_pred = None
            _sobolev_dedipole_pred = None
            if branch in ("sobolev_polar", "sobolev_dipole") and epoch > int(args.sobolev_warmup_epochs):
                _sobolev_pos_grad = data.pos.detach().requires_grad_(True)
                data.pos = _sobolev_pos_grad

            pred = _forward_model(model, data, args.mode, branch=branch)
            pred_for_limit = pred

            if branch == "polar":
                # pred: [B, 3, 3], target: [B, 3, 3]
                target = train_mod.vec6_to_symmetric_3x3(
                    data.y_polar_vec6.to(pred.device))
                loss = _regression_loss(pred, target, loss_type, "polar", args.huber_delta)

            elif branch == "depolar":
                # pred: [N, 3, 6] depolar
                target_dp = data.y_depolar.to(pred.device)
                pred_dp = pred.view(-1, 3, 6)

                # --- Protection: nan_to_num on pred (matches DetaNet) ---
                pred_dp = torch.nan_to_num(pred_dp, nan=0.0, posinf=0.0, neginf=0.0)

                # --- Protection: filter non-finite target atoms ---
                finite_mask = torch.isfinite(target_dp.view(target_dp.size(0), -1)).all(dim=1)
                if not finite_mask.all():
                    pred_dp = pred_dp[finite_mask]
                    target_dp = target_dp[finite_mask]
                    if depolar_mask is not None:
                        depolar_mask = depolar_mask[finite_mask]
                    if target_dp.numel() == 0:
                        continue

                if depolar_mask is not None and depolar_mask.any():
                    pred_dp = pred_dp[depolar_mask]
                    target_dp = target_dp[depolar_mask]
                loss = _regression_loss(pred_dp, target_dp, loss_type, "depolar", args.huber_delta)

            elif branch == "dipole":
                # pred: [B, 3]
                target = data.y_dipole.to(pred.device)
                pred_clean = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                loss = _regression_loss(pred_clean, target, loss_type, "dipole", args.huber_delta)

            elif branch == "dedipole":
                # pred: [N, 3, 3]
                target = data.y_dedipole.to(pred.device)
                pred_clean = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
                finite_mask = torch.isfinite(target.view(target.size(0), -1)).all(dim=1)
                if not finite_mask.all():
                    pred_clean = pred_clean[finite_mask]
                    target = target[finite_mask]
                    if target.numel() == 0:
                        continue
                # Graph-level label outlier filter
                if float(args.dedipole_max_abs) > 0:
                    amp_mask = target.abs().view(target.size(0), -1).amax(dim=1) <= float(args.dedipole_max_abs)
                    pred_clean = pred_clean[amp_mask]
                    target = target[amp_mask]
                    if target.numel() == 0:
                        continue
                loss = _regression_loss(pred_clean, target, loss_type, "dedipole", args.huber_delta)

            elif branch == "hii":
                # pred: [N, 3, 3]
                target = data.y_hii.to(pred.device)
                pred_clean = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
                finite_mask = torch.isfinite(target.view(target.size(0), -1)).all(dim=1)
                if not finite_mask.all():
                    pred_clean = pred_clean[finite_mask]
                    target = target[finite_mask]
                    if target.numel() == 0:
                        continue
                # Graph-level label outlier filter
                if float(args.hii_max_abs) > 0:
                    amp_mask = target.abs().view(target.size(0), -1).amax(dim=1) <= float(args.hii_max_abs)
                    pred_clean = pred_clean[amp_mask]
                    target = target[amp_mask]
                    if target.numel() == 0:
                        continue
                loss = _regression_loss(pred_clean, target, loss_type, "hii", args.huber_delta)

            elif branch == "hij":
                # pred: [E, 3, 3]
                target = data.y_hij.to(pred.device)
                pred_full = pred.view(-1, 3, 3)
                finite_pred = torch.isfinite(pred_full.view(pred_full.size(0), -1)).all(dim=1)
                finite_tgt = torch.isfinite(target.view(target.size(0), -1)).all(dim=1)
                keep_mask = finite_pred & finite_tgt
                if not keep_mask.any():
                    continue
                pred_clean = pred_full[keep_mask]
                target = target[keep_mask]
                # Graph-level label outlier filter
                if float(args.hij_max_abs) > 0:
                    amp_mask = target.abs().view(target.size(0), -1).amax(dim=1) <= float(args.hij_max_abs)
                    pred_clean = pred_clean[amp_mask]
                    target = target[amp_mask]
                    if target.numel() == 0:
                        continue
                loss = _regression_loss(pred_clean, target, loss_type, "hij", args.huber_delta)
                hij_pred_metric = pred_clean
                hij_tgt_metric = target

            elif branch == "sobolev_polar":
                # Sobolev joint: polar model + external autograd → depolar
                mol_polar = pred  # [B, 3, 3]

                # Static polar loss (always active)
                target_polar = train_mod.vec6_to_symmetric_3x3(
                    data.y_polar_vec6.to(mol_polar.device))
                loss_static = _regression_loss(
                    mol_polar, target_polar, args.static_loss_type, "polar", args.huber_delta)

                if _sobolev_pos_grad is not None:
                    # --- Post-warmup: compute depolar via autograd ---
                    depolar_pred = _compute_depolar_from_polar(
                        mol_polar, _sobolev_pos_grad, training=True)
                    depolar_pred = torch.nan_to_num(depolar_pred, nan=0.0, posinf=0.0, neginf=0.0)

                    # Clamp predictions to prevent extreme loss spikes
                    _clip = float(args.sobolev_depolar_clip)
                    if _clip > 0:
                        depolar_pred = depolar_pred.clamp(-_clip, _clip)

                    _sobolev_depolar_pred = depolar_pred

                    # Filter targets: non-finite atoms + graph-level depolar_max_abs
                    target_dp = data.y_depolar.to(depolar_pred.device)
                    dp_pred = depolar_pred
                    dp_tgt = target_dp
                    finite_mask = torch.isfinite(dp_tgt.view(dp_tgt.size(0), -1)).all(dim=1)
                    if not finite_mask.all():
                        dp_pred = dp_pred[finite_mask]
                        dp_tgt = dp_tgt[finite_mask]
                    if float(args.depolar_max_abs) > 0:
                        dm, _, _ = train_mod._filter_depolar_mask(data, args.depolar_max_abs)
                        if dm is not None:
                            dm_f = dm[finite_mask] if not finite_mask.all() else dm
                            dp_pred = dp_pred[dm_f]
                            dp_tgt = dp_tgt[dm_f]

                    if dp_pred.numel() > 0:
                        loss_deriv = _regression_loss(
                            dp_pred, dp_tgt, args.sobolev_deriv_loss_type,
                            "depolar", args.huber_delta)
                        loss_jac = _jacobian_floor_loss(
                            depolar_pred, data.batch, args.sobolev_jac_floor)
                        loss = (float(args.sobolev_lambda_static) * loss_static
                                + float(args.sobolev_lambda_deriv) * loss_deriv
                                + loss_jac)
                    else:
                        # All derivative atoms filtered; use static only for this batch
                        loss = float(args.sobolev_lambda_static) * loss_static
                else:
                    # Warmup phase: pure static polar loss, no autograd
                    loss = float(args.sobolev_lambda_static) * loss_static

            elif branch == "sobolev_dipole":
                # Sobolev joint: dipole model + external autograd → dedipole
                mol_dipole = pred  # [B, 3]

                # Static dipole loss (always active)
                target_dipole = data.y_dipole.to(mol_dipole.device)
                pred_clean = torch.nan_to_num(mol_dipole, nan=0.0, posinf=0.0, neginf=0.0)
                loss_static = _regression_loss(
                    pred_clean, target_dipole, args.static_loss_type, "dipole", args.huber_delta)

                if _sobolev_pos_grad is not None:
                    # --- Post-warmup: compute dedipole via autograd ---
                    dedipole_pred = _compute_dedipole_from_dipole(
                        mol_dipole, _sobolev_pos_grad, training=True)
                    dedipole_pred = torch.nan_to_num(dedipole_pred, nan=0.0, posinf=0.0, neginf=0.0)

                    # Clamp predictions to prevent extreme loss spikes
                    _clip = float(args.sobolev_dedipole_clip)
                    if _clip > 0:
                        dedipole_pred = dedipole_pred.clamp(-_clip, _clip)

                    _sobolev_dedipole_pred = dedipole_pred

                    # Filter targets: non-finite atoms + dedipole_max_abs
                    target_dd = data.y_dedipole.to(dedipole_pred.device)
                    dd_pred = dedipole_pred
                    dd_tgt = target_dd
                    finite_mask = torch.isfinite(dd_tgt.view(dd_tgt.size(0), -1)).all(dim=1)
                    if not finite_mask.all():
                        dd_pred = dd_pred[finite_mask]
                        dd_tgt = dd_tgt[finite_mask]
                    if float(args.dedipole_max_abs) > 0:
                        amp_mask = dd_tgt.abs().view(dd_tgt.size(0), -1).amax(dim=1) <= float(args.dedipole_max_abs)
                        dd_pred = dd_pred[amp_mask]
                        dd_tgt = dd_tgt[amp_mask]

                    if dd_pred.numel() > 0:
                        loss_deriv = _regression_loss(
                            dd_pred, dd_tgt, args.sobolev_deriv_loss_type,
                            "dedipole", args.huber_delta)
                        loss_jac = _jacobian_floor_loss(
                            dedipole_pred, data.batch, args.sobolev_jac_floor)
                        loss = (float(args.sobolev_lambda_static) * loss_static
                                + float(args.sobolev_lambda_deriv) * loss_deriv
                                + loss_jac)
                    else:
                        # All derivative atoms filtered; use static only for this batch
                        loss = float(args.sobolev_lambda_static) * loss_static
                else:
                    # Warmup phase: pure static dipole loss, no autograd
                    loss = float(args.sobolev_lambda_static) * loss_static

        if bool(args.skip_nonfinite_batches):
            pred_check = pred_for_limit if pred_for_limit is not None else pred
            pred_ok = _is_finite_tensor(pred_check)
            loss_ok = _is_finite_tensor(loss.detach())
            if not pred_ok or not loss_ok:
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

        pred_limit_tensor = pred_for_limit if pred_for_limit is not None else pred
        over_pred_limit, pred_abs_max = _exceeds_pred_limit(pred_limit_tensor, float(args.pred_abs_max))
        if over_pred_limit:
            skipped_pred_limit += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        # --- NBO-CR consistency loss ---
        if getattr(args, "nbo_cr_weight", 0.0) > 0:
            try:
                from nbo_consistency_loss import NBOConsistencyLoss
                if not hasattr(model, "_nbo_cr_module"):
                    model._nbo_cr_module = NBOConsistencyLoss().to(pred.device)
                cr_loss = model._nbo_cr_module(model, data, pred, branch)
                if cr_loss.requires_grad and cr_loss.item() > 0:
                    cr_weight = _nbo_cr_effective_weight(args, epoch)
                    # ENK OOD boost: stronger CR for OOD atoms
                    bridge = getattr(model, "so3_enk_bridge", None)
                    if bridge is not None and hasattr(bridge, "_last_kalman_gains"):
                        kg = bridge._last_kalman_gains
                        if kg:
                            k_stack = torch.stack([v for v in kg.values() if v.numel() > 0], dim=0)
                            mean_k = k_stack.mean().item()
                            ood_score = 1.0 - mean_k
                            cr_weight = cr_weight * (1.0 + 0.5 * ood_score)
                    loss = loss + cr_weight * cr_loss
                    nbo_cr_accum += float(cr_loss.detach().item())
                    nbo_cr_steps += 1
            except Exception:
                pass

        loss.backward()

        if branch == "hij" and _has_nonfinite_grad(model):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        total_grad_norm = 0.0
        if float(args.max_grad_norm) > 0:
            total_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm)))
        elif float(args.skip_grad_norm) > 0:
            total_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))

        # All derivative branches (depolar, dedipole, hii, hij, sobolev_polar) involve
        # second-order autograd (backward through create_graph=True forward pass), so
        # per-batch gradient norms are naturally large (50–500+).  max_grad_norm clipping
        # already handles numerical stability; skip_grad_norm would discard virtually every
        # batch and stall training.  Disable skip_grad_norm for all derivative branches.
        _eff_skip_gnorm = 0.0 if branch in _DERIVATIVE_BRANCHES else float(args.skip_grad_norm)
        if _eff_skip_gnorm > 0 and total_grad_norm > _eff_skip_gnorm:
            skipped_grad_limit += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        loss_sum += float(loss.detach().item())
        steps += 1

        # Metrics
        if branch == "polar":
            tgt_vec6 = data.y_polar_vec6.to(pred.device)
            train_mod._update_metric_sums(
                metric_sums, "polar9",
                pred.view(-1, 3, 3),
                train_mod.vec6_to_symmetric_3x3(tgt_vec6),
            )
        elif branch == "depolar":
            pred_full = torch.nan_to_num(pred.view(-1, 3, 6), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_full = data.y_depolar.to(pred.device)
            finite_m = torch.isfinite(tgt_full.view(tgt_full.size(0), -1)).all(dim=1)
            if not finite_m.all():
                pred_full = pred_full[finite_m]
                tgt_full = tgt_full[finite_m]
            # NOTE: depolar_mask may have been sliced by finite_mask in the loss block above,
            # so it already matches the current pred_full/tgt_full shape exactly. Do NOT
            # re-index with finite_m again.
            if depolar_mask is not None and depolar_mask.any():
                pred_full = pred_full[depolar_mask]
                tgt_full = tgt_full[depolar_mask]
            if pred_full.numel() > 0:
                train_mod._update_metric_sums(metric_sums, "depolar18", pred_full, tgt_full)
        elif branch == "dipole":
            pred_m = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            train_mod._update_metric_sums(metric_sums, "dip", pred_m, data.y_dipole.to(pred.device))
        elif branch == "dedipole":
            pred_m = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_m = data.y_dedipole.to(pred.device)
            fm = torch.isfinite(tgt_m.view(tgt_m.size(0), -1)).all(dim=1)
            if fm.any():
                train_mod._update_metric_sums(metric_sums, "dedipole9", pred_m[fm], tgt_m[fm])
        elif branch == "hii":
            pred_m = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_m = data.y_hii.to(pred.device)
            fm = torch.isfinite(tgt_m.view(tgt_m.size(0), -1)).all(dim=1)
            if fm.any():
                train_mod._update_metric_sums(metric_sums, "hii9", pred_m[fm], tgt_m[fm])
        elif branch == "hij":
            if hij_pred_metric is not None and hij_tgt_metric is not None and hij_tgt_metric.numel() > 0:
                train_mod._update_metric_sums(metric_sums, "hij9", hij_pred_metric, hij_tgt_metric)
        elif branch == "sobolev_polar":
            # Track both polar and depolar metrics
            tgt_vec6 = data.y_polar_vec6.to(pred.device)
            train_mod._update_metric_sums(
                metric_sums, "polar9",
                pred.view(-1, 3, 3),
                train_mod.vec6_to_symmetric_3x3(tgt_vec6),
            )
            if _sobolev_depolar_pred is not None:
                dp_full = _sobolev_depolar_pred.detach()
                dp_tgt_full = data.y_depolar.to(dp_full.device)
                fm = torch.isfinite(dp_tgt_full.view(dp_tgt_full.size(0), -1)).all(dim=1)
                if fm.any():
                    train_mod._update_metric_sums(
                        metric_sums, "depolar18",
                        dp_full[fm] if not fm.all() else dp_full,
                        dp_tgt_full[fm] if not fm.all() else dp_tgt_full)
        elif branch == "sobolev_dipole":
            # Track both dipole and dedipole metrics
            pred_clean = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            train_mod._update_metric_sums(
                metric_sums, "dip",
                pred_clean, data.y_dipole.to(pred.device),
            )
            if _sobolev_dedipole_pred is not None:
                dd_full = _sobolev_dedipole_pred.detach()
                dd_tgt_full = data.y_dedipole.to(dd_full.device)
                fm = torch.isfinite(dd_tgt_full.view(dd_tgt_full.size(0), -1)).all(dim=1)
                if fm.any():
                    train_mod._update_metric_sums(
                        metric_sums, "dedipole9",
                        dd_full[fm] if not fm.all() else dd_full,
                        dd_tgt_full[fm] if not fm.all() else dd_tgt_full)

        _BRANCH_METRIC_PREFIXES = {
            "polar": ("polar_",), "depolar": ("depolar_",),
            "dipole": ("dip_",), "dedipole": ("dedipole_",),
            "hii": ("hii_",), "hij": ("hij_",),
            "sobolev_polar": ("polar_", "depolar_"),
            "sobolev_dipole": ("dip_", "dedipole_"),
        }
        if step % max(1, int(args.print_freq)) == 0:
            avg_metrics = train_mod._finalize_metrics(metric_sums)
            prefixes = _BRANCH_METRIC_PREFIXES.get(branch, ())
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(avg_metrics.items())
                if any(k.startswith(p) for p in prefixes)
            )
            extra = []
            if float(args.pred_abs_max) > 0:
                extra.append(f"pred_abs_max={pred_abs_max:.3e}")
            if float(args.skip_grad_norm) > 0 or branch == "sobolev_polar":
                extra.append(f"grad_norm={total_grad_norm:.3e}")
            extra_str = (" " + " ".join(extra)) if extra else ""
            print(f"  [{branch}] Epoch {epoch} Step [{step}/{len(loader)}] "
                  f"loss={loss_sum / max(1, steps):.4f} {metric_str}{extra_str}")

    avg_loss = loss_sum / max(1, steps)
    train_metrics = train_mod._finalize_metrics(metric_sums)

    if skipped_nonfinite > 0 or skipped_pred_limit > 0 or skipped_grad_limit > 0:
        print(f"  [{branch}] skipped: nonfinite={skipped_nonfinite} "
              f"pred_limit={skipped_pred_limit} grad_limit={skipped_grad_limit}")

    if nbo_cr_steps > 0:
        cr_w = _nbo_cr_effective_weight(args, epoch)
        print(f"  [{branch}] NBO-CR avg={nbo_cr_accum / max(1, nbo_cr_steps):.5f} "
              f"active={nbo_cr_steps}/{steps} weight={cr_w:.4f}")

    return {
        "epoch": epoch,
        "branch": branch,
        "avg_loss": float(avg_loss),
        "steps": steps,
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_pred_limit": skipped_pred_limit,
        "skipped_grad_limit": skipped_grad_limit,
        "train_metrics": train_metrics,
    }


# ============================================================================
#  Evaluation
# ============================================================================

def _evaluate(
    model: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    branch: str,
) -> Dict[str, object]:
    """Evaluate a single branch model on a data loader."""
    agg_sums = train_mod._init_metric_sums()
    batch_outputs = []
    skipped_nonfinite = 0
    skipped_pred_limit = 0

    model.eval()
    for batch_idx, batch_cpu in enumerate(loader):
        data = _clone_batch(batch_cpu).to(args.device, non_blocking=False)

        depolar_mask = None
        filtered_graphs = 0
        if branch == "depolar" and float(args.depolar_max_abs) > 0:
            depolar_mask, filtered_graphs, _ = train_mod._filter_depolar_mask(data, args.depolar_max_abs)
            if depolar_mask is not None and depolar_mask.sum().item() == 0:
                batch_outputs.append({
                    "batch_index": batch_idx,
                    "num_graphs": int(getattr(data, "num_graphs", 0)),
                    "filtered_graphs": int(filtered_graphs),
                    "metrics": {},
                    "skipped": True,
                })
                continue

        batch_sums = train_mod._init_metric_sums()

        # Forward: use enable_grad for derivative branches, no_grad otherwise
        _sobolev_pos_grad_eval = None
        if branch in _DERIVATIVE_BRANCHES:
            with torch.enable_grad():
                if branch in ("sobolev_polar", "sobolev_dipole"):
                    _sobolev_pos_grad_eval = data.pos.detach().requires_grad_(True)
                    data.pos = _sobolev_pos_grad_eval
                pred = _forward_model(model, data, args.mode, branch=branch)
        else:
            with torch.no_grad():
                pred = _forward_model(model, data, args.mode, branch=branch)

        if bool(args.skip_nonfinite_batches) and not _is_finite_tensor(pred):
            skipped_nonfinite += 1
            batch_outputs.append({
                "batch_index": batch_idx,
                "num_graphs": int(getattr(data, "num_graphs", 0)),
                "filtered_graphs": int(filtered_graphs),
                "metrics": {},
                "skipped": True,
            })
            continue

        over_pred_limit, pred_abs_max = _exceeds_pred_limit(pred, float(args.pred_abs_max))
        if over_pred_limit:
            skipped_pred_limit += 1
            batch_outputs.append({
                "batch_index": batch_idx,
                "num_graphs": int(getattr(data, "num_graphs", 0)),
                "filtered_graphs": int(filtered_graphs),
                "metrics": {"pred_abs_max": pred_abs_max},
                "skipped": True,
            })
            continue

        if branch == "polar":
            target = train_mod.vec6_to_symmetric_3x3(
                data.y_polar_vec6.to(pred.device))
            train_mod._update_metric_sums(
                batch_sums, "polar9", pred.view(-1, 3, 3), target)
        elif branch == "depolar":
            pred_dp = torch.nan_to_num(pred.view(-1, 3, 6), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_dp = data.y_depolar.to(pred.device)
            finite_m = torch.isfinite(tgt_dp.view(tgt_dp.size(0), -1)).all(dim=1)
            if not finite_m.all():
                pred_dp = pred_dp[finite_m]
                tgt_dp = tgt_dp[finite_m]
            if depolar_mask is not None and depolar_mask.any():
                dm = depolar_mask[finite_m] if not finite_m.all() else depolar_mask
                pred_dp = pred_dp[dm]
                tgt_dp = tgt_dp[dm]
            if pred_dp.numel() > 0:
                train_mod._update_metric_sums(batch_sums, "depolar18", pred_dp, tgt_dp)
        elif branch == "dipole":
            pred_m = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            train_mod._update_metric_sums(batch_sums, "dip", pred_m, data.y_dipole.to(pred.device))
        elif branch == "dedipole":
            pred_m = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_m = data.y_dedipole.to(pred.device)
            fm = torch.isfinite(tgt_m.view(tgt_m.size(0), -1)).all(dim=1)
            if fm.any():
                pm, tm = pred_m[fm], tgt_m[fm]
                if float(args.dedipole_max_abs) > 0:
                    amp = tm.abs().view(tm.size(0), -1).amax(dim=1) <= float(args.dedipole_max_abs)
                    pm, tm = pm[amp], tm[amp]
                if tm.numel() > 0:
                    train_mod._update_metric_sums(batch_sums, "dedipole9", pm, tm)
        elif branch == "hii":
            pred_m = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_m = data.y_hii.to(pred.device)
            fm = torch.isfinite(tgt_m.view(tgt_m.size(0), -1)).all(dim=1)
            if fm.any():
                pm, tm = pred_m[fm], tgt_m[fm]
                if float(args.hii_max_abs) > 0:
                    amp = tm.abs().view(tm.size(0), -1).amax(dim=1) <= float(args.hii_max_abs)
                    pm, tm = pm[amp], tm[amp]
                if tm.numel() > 0:
                    train_mod._update_metric_sums(batch_sums, "hii9", pm, tm)
        elif branch == "hij":
            pred_m = torch.nan_to_num(pred.view(-1, 3, 3), nan=0.0, posinf=0.0, neginf=0.0)
            tgt_m = data.y_hij.to(pred.device)
            fm = torch.isfinite(tgt_m.view(tgt_m.size(0), -1)).all(dim=1)
            if fm.any():
                pm, tm = pred_m[fm], tgt_m[fm]
                if float(args.hij_max_abs) > 0:
                    amp = tm.abs().view(tm.size(0), -1).amax(dim=1) <= float(args.hij_max_abs)
                    pm, tm = pm[amp], tm[amp]
                if tm.numel() > 0:
                    train_mod._update_metric_sums(batch_sums, "hij9", pm, tm)
        elif branch == "sobolev_polar":
            # Polar metrics
            target_polar = train_mod.vec6_to_symmetric_3x3(
                data.y_polar_vec6.to(pred.device))
            train_mod._update_metric_sums(
                batch_sums, "polar9", pred.view(-1, 3, 3), target_polar)
            # Depolar metrics via autograd
            if _sobolev_pos_grad_eval is not None:
                with torch.enable_grad():
                    depolar_eval = _compute_depolar_from_polar(
                        pred, _sobolev_pos_grad_eval, training=False)
                depolar_eval = torch.nan_to_num(depolar_eval, nan=0.0, posinf=0.0, neginf=0.0)
                _clip = float(getattr(args, 'sobolev_depolar_clip', 0))
                if _clip > 0:
                    depolar_eval = depolar_eval.clamp(-_clip, _clip)
                tgt_dp = data.y_depolar.to(depolar_eval.device)
                dp_e = depolar_eval
                dp_t = tgt_dp
                fm = torch.isfinite(dp_t.view(dp_t.size(0), -1)).all(dim=1)
                if not fm.all():
                    dp_e = dp_e[fm]
                    dp_t = dp_t[fm]
                if float(args.depolar_max_abs) > 0:
                    dm, _, _ = train_mod._filter_depolar_mask(data, args.depolar_max_abs)
                    if dm is not None:
                        dm_f = dm[fm] if not fm.all() else dm
                        dp_e = dp_e[dm_f]
                        dp_t = dp_t[dm_f]
                if dp_e.numel() > 0:
                    train_mod._update_metric_sums(
                        batch_sums, "depolar18", dp_e, dp_t)
        elif branch == "sobolev_dipole":
            # Dipole metrics
            pred_m = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            train_mod._update_metric_sums(
                batch_sums, "dip", pred_m, data.y_dipole.to(pred.device))
            # Dedipole metrics via autograd
            if _sobolev_pos_grad_eval is not None:
                with torch.enable_grad():
                    dedipole_eval = _compute_dedipole_from_dipole(
                        pred, _sobolev_pos_grad_eval, training=False)
                dedipole_eval = torch.nan_to_num(dedipole_eval, nan=0.0, posinf=0.0, neginf=0.0)
                _clip = float(getattr(args, 'sobolev_dedipole_clip', 0))
                if _clip > 0:
                    dedipole_eval = dedipole_eval.clamp(-_clip, _clip)
                tgt_dd = data.y_dedipole.to(dedipole_eval.device)
                dd_e = dedipole_eval
                dd_t = tgt_dd
                fm = torch.isfinite(dd_t.view(dd_t.size(0), -1)).all(dim=1)
                if not fm.all():
                    dd_e = dd_e[fm]
                    dd_t = dd_t[fm]
                if float(args.dedipole_max_abs) > 0:
                    amp = dd_t.abs().view(dd_t.size(0), -1).amax(dim=1) <= float(args.dedipole_max_abs)
                    dd_e = dd_e[amp]
                    dd_t = dd_t[amp]
                if dd_e.numel() > 0:
                    train_mod._update_metric_sums(
                        batch_sums, "dedipole9", dd_e, dd_t)

        # Branch-specific metric name for result
        metric_name = _EVAL_METRIC_NAMES.get(branch, branch)

        batch_metrics = train_mod._finalize_metrics(batch_sums)
        _merge_metric_sums(agg_sums, batch_sums)
        batch_outputs.append({
            "batch_index": batch_idx,
            "num_graphs": int(getattr(data, "num_graphs", 0)),
            "filtered_graphs": int(filtered_graphs),
            "metrics": _sobolev_metric_subset(batch_metrics) if branch == "sobolev_polar"
                       else _sobolev_dipole_metric_subset(batch_metrics) if branch == "sobolev_dipole"
                       else _metric_subset(batch_metrics, metric_name),
            "skipped": False,
        })

    aggregate = train_mod._finalize_metrics(agg_sums)
    if branch == "sobolev_polar":
        agg_subset = _sobolev_metric_subset(aggregate)
    elif branch == "sobolev_dipole":
        agg_subset = _sobolev_dipole_metric_subset(aggregate)
    else:
        metric_name = _EVAL_METRIC_NAMES.get(branch, branch)
        agg_subset = _metric_subset(aggregate, metric_name)
    return {
        "aggregate": agg_subset,
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_pred_limit": skipped_pred_limit,
        "batches": batch_outputs,
    }


# ============================================================================
#  Branch training orchestration
# ============================================================================

def _train_branch(
    branch: str,
    args: argparse.Namespace,
    loaders: Dict,
    stats: Dict[str, torch.Tensor],
    output_dir: Path,
) -> Dict[str, object]:
    """Train one branch from scratch.

    Args:
        branch: "polar", "depolar", "sobolev_polar", "dipole", "dedipole", "hii", or "hij"
        args: parsed arguments
        loaders: dict with "train", "val", "test" data loaders
        stats: normalization stats
        output_dir: base output directory

    Returns:
        Dict with training history and final results.
    """
    # Determine which checkpoint will be loaded for this branch
    ckpt_path = args.init_ckpt
    if branch == "depolar" and args.polar_init_ckpt:
        ckpt_path = args.polar_init_ckpt
    if branch == "sobolev_polar" and args.polar_init_ckpt:
        ckpt_path = args.polar_init_ckpt
    if branch == "sobolev_dipole" and args.dipole_init_ckpt:
        ckpt_path = args.dipole_init_ckpt

    # Pre-detect architecture from checkpoint BEFORE building model
    # (avoids SO3_grid / SO3_rotation shape mismatch at load time)
    _is_v2 = str(getattr(args, 'mode', '')).startswith('equiformer_v2')
    if _is_v2 and ckpt_path and os.path.exists(ckpt_path):
        _arch = _detect_arch_from_ckpt(ckpt_path, device=str(args.device))
        if _arch.get('model_name') and _arch['model_name'] != getattr(args, 'v2_model_name', ''):
            print(f"  [{branch}] auto-detect: model_name {getattr(args, 'v2_model_name', '?')} "
                  f"→ {_arch['model_name']} (from checkpoint)")
            args.v2_model_name = _arch['model_name']
        if _arch.get('num_layers') and _arch['num_layers'] != int(getattr(args, 'num_layers', 0)):
            print(f"  [{branch}] auto-detect: num_layers {getattr(args, 'num_layers', '?')} "
                  f"→ {_arch['num_layers']} (from checkpoint)")
            args.num_layers = _arch['num_layers']
        if _arch.get('grid_resolution') and _arch['grid_resolution'] != int(getattr(args, 'v2_grid_resolution', 0)):
            print(f"  [{branch}] auto-detect: v2_grid_resolution {getattr(args, 'v2_grid_resolution', '?')} "
                  f"→ {_arch['grid_resolution']} (from checkpoint)")
            args.v2_grid_resolution = _arch['grid_resolution']

    # Build model (now with correct arch from checkpoint)
    model = _build_model(branch, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{branch}] params={n_params:,}")
    if bool(args.enk_enabled):
        enk_params = sum(p.numel() for n, p in model.named_parameters() if 'enk' in n)
        print(f"  [{branch}] ENK enabled: {enk_params:,} extra params")

    # Load checkpoint
    load_info = _load_checkpoint(model, ckpt_path, args.device)
    if load_info["loaded"]:
        print(f"  [{branch}] loaded checkpoint: {ckpt_path} "
              f"missing={len(load_info['missing_keys'])} "
              f"unexpected={len(load_info['unexpected_keys'])}")
    else:
        print(f"  [{branch}] training from scratch")

    # --- EP setup: reset gates for ALL electron-prior models ---
    if hasattr(model, "electron_prior"):
        _ep_warmup_epochs = int(getattr(args, "nbo_cr_warmup_epochs", 2))
        gate_init = float(getattr(args, "nbo_cr_gate_init", 0.5))
        if _ep_warmup_epochs > 0:
            gate_init = 0.0
        prior = model.electron_prior
        if hasattr(prior, "atom_gate"):
            prior.atom_gate.data.fill_(gate_init)
            prior.edge_gate.data.fill_(gate_init)
            if hasattr(prior, "interaction_atom_gate"):
                prior.interaction_atom_gate.data.fill_(gate_init)
            if hasattr(prior, "interaction_edge_gate"):
                prior.interaction_edge_gate.data.fill_(gate_init)
            print(f"  [{branch}] EP gates init={gate_init:.3f}"
                  f" (target={float(getattr(args, 'nbo_cr_gate_init', 0.5)):.3f} warmup={_ep_warmup_epochs}ep)")
        injector = getattr(model, "so3_ep_injector", None)
        if injector is not None and hasattr(injector, "prior_blocks"):
            inj_gate = 0.0 if _ep_warmup_epochs > 0 else 1.0
            for blk_k, blk_mod in injector.prior_blocks.items():
                if hasattr(blk_mod, "gate"):
                    blk_mod.gate.data.fill_(inj_gate)
            print(f"  [{branch}] SO3 injector gates init={inj_gate:.3f}"
                  f" (target=1.0 warmup={_ep_warmup_epochs}ep)")

        # --- Freeze backbone + EP predictor; train only SO3 injector + EP gates ---
        _n_frozen = 0
        _n_injector = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (n.startswith("so3_ep_injector.") or
                ("electron_prior." in n and "gate" in n.lower()) or
                n.startswith("nbo_prior_branch.")):
                _n_injector += 1
            else:
                p.requires_grad = False
                _n_frozen += 1
        print(f"  [{branch}] frozen={_n_frozen} backbone+enk+pred params, "
              f"trainable={_n_injector} injector+gate params")

    # --- NBO-CR module (only when CR weight > 0) ---
    if getattr(args, "nbo_cr_weight", 0.0) > 0:
        from nbo_consistency_loss import NBOConsistencyLoss
        model._nbo_cr_module = NBOConsistencyLoss(
            alpha_init=float(getattr(args, "nbo_cr_alpha_init", 0.6)),
        ).to(args.device)
        print(f"  [{branch}] NBO-CR module attached: "
              f"{sum(1 for _ in model._nbo_cr_module.parameters())} learnable params"
              f" (alpha={model._nbo_cr_module.alpha.item():.3f})")

    # Optimizer
    if hasattr(model, "_nbo_cr_module"):
        cr_lr_scale = float(getattr(args, "nbo_cr_lr_scale", 0.1))
        base_lr = float(args.lr)
        cr_params = list(model._nbo_cr_module.parameters())
        other_params = [p for n, p in model.named_parameters()
                        if "_nbo_cr_module" not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": base_lr},
            {"params": cr_params, "lr": base_lr * cr_lr_scale},
        ], weight_decay=float(args.weight_decay), amsgrad=True)
        print(f"  [{branch}] NBO-CR optimizer: injector_lr={base_lr:.2e} cr_lr={base_lr * cr_lr_scale:.2e}")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            amsgrad=True,
        )

    # LR scheduler
    sched_type = str(args.lr_scheduler).lower()
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.train_epochs), eta_min=float(args.lr_min))
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_step_size),
            gamma=float(args.lr_gamma),
        )
    else:
        scheduler = None

    # Best model tracking
    _BEST_KEY_MAP = {
        "polar": "polar_mae", "depolar": "depolar_mae",
        "dipole": "dip_mae", "dedipole": "dedipole_mae",
        "hii": "hii_mae", "hij": "hij_mae",
    }
    if branch == "sobolev_polar":
        bm = str(getattr(args, 'sobolev_best_metric', 'depolar_mae')).lower()
        if bm == "combined":
            best_key = "_sobolev_combined"  # special: polar_mae + depolar_mae
        elif bm == "polar_mae":
            best_key = "polar_mae"
        else:
            best_key = "depolar_mae"
    elif branch == "sobolev_dipole":
        bm = str(getattr(args, 'sobolev_dipole_best_metric', 'dedipole_mae')).lower()
        if bm == "combined":
            best_key = "_sobolev_dipole_combined"  # special: dip_mae + dedipole_mae
        elif bm == "dip_mae":
            best_key = "dip_mae"
        else:
            best_key = "dedipole_mae"
    else:
        best_key = _BEST_KEY_MAP.get(branch, f"{branch}_mae")
    best_val = float("inf")
    _run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_dir = output_dir / branch / _run_ts
    branch_dir.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()

    for epoch in range(1, int(args.train_epochs) + 1):
        epoch_t0 = time.time()

        # Train
        train_entry = _train_one_epoch(
            model, loaders["train"], optimizer, stats, args, epoch, branch)

        # Evaluate
        val_result = _evaluate(model, loaders["val"], args, branch)
        test_result = _evaluate(model, loaders["test"], args, branch)

        epoch_dt = time.time() - epoch_t0

        record = {
            "epoch": epoch,
            "train": train_entry,
            "val": val_result["aggregate"],
            "test": test_result["aggregate"],
            "epoch_seconds": epoch_dt,
        }
        history.append(record)

        # Print summary
        train_metrics = train_entry.get("train_metrics", {})
        train_str = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(train_metrics.items())
        )
        val_str = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(val_result["aggregate"].items())
        )
        test_str = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(test_result["aggregate"].items())
        )
        lr_current = optimizer.param_groups[0]["lr"]
        print(f"  [{branch}] Epoch {epoch}/{args.train_epochs} ({epoch_dt:.1f}s) lr={lr_current:.2e}")
        print(f"    train: loss={train_entry['avg_loss']:.4f} {train_str}")
        print(f"    val:   {val_str}")
        print(f"    test:  {test_str}")

        # LR step
        if scheduler is not None:
            scheduler.step()

        # Checkpoint best
        if best_key == "_sobolev_combined":
            p_mae = float(val_result["aggregate"].get("polar_mae", float("inf")))
            d_mae = float(val_result["aggregate"].get("depolar_mae", float("inf")))
            score = p_mae + d_mae
        elif best_key == "_sobolev_dipole_combined":
            dip_mae = float(val_result["aggregate"].get("dip_mae", float("inf")))
            dd_mae = float(val_result["aggregate"].get("dedipole_mae", float("inf")))
            score = dip_mae + dd_mae
        else:
            score = float(val_result["aggregate"].get(best_key, float("inf")))
        if score < best_val:
            best_val = score
            _save_checkpoint(model, branch_dir / "best.pt")
            print(f"    ** saved best {branch} model, {best_key}={best_val:.4f}")

    # Save last
    _save_checkpoint(model, branch_dir / "last.pt")

    total_time = time.time() - t0
    print(f"  [{branch}] training done. total={total_time:.1f}s best_{best_key}={best_val:.4f}")

    return {
        "params": n_params,
        "load_info": load_info,
        "history": history,
        "best_val": {best_key: best_val},
    }


# ============================================================================
#  Main
# ============================================================================

def main():
    args = _normalize_args(build_parser().parse_args())
    mode = args.mode
    log_path = _setup_log_stream(args.log_file)
    if log_path:
        print(f"[log] writing to {log_path}")

    print("=" * 70)
    print(f" Equiformer Backbone Training: mode={mode}")
    print("=" * 70)
    print(f"  dataset={args.dataset}")
    print(f"  task={args.task}")
    print(f"  epochs={args.epochs} lr={args.lr} batch_size={args.batch_size}")
    print(f"  backbone: layers={args.num_layers} hidden={args.hidden_nf} "
          f"basis={args.num_basis} radius={args.radius}")
    print(f"  optimizer: AdamW amsgrad=True wd={args.weight_decay} "
          f"grad_clip={args.max_grad_norm} scheduler={args.lr_scheduler}")
    print(f"  device={args.device}")

    if mode == "clean_equiformer":
        print("  model: CleanEquiformerPolar (V1 backbone, no extensions)")
    elif mode == "equiformer_electron_prior":
        print(f"  model: CleanEquiformerPolarExt + ElectronPriorAttention")
        print(f"    prior: mode={args.electron_prior_mode} "
              f"freeze={args.electron_prior_freeze} "
              f"runtime={args.electron_prior_runtime_mode}")
        print(f"    ckpt: {args.electron_prior_ckpt}")
    elif mode.startswith("equiformer_v2"):
        print(f"  model: EquiformerV2 ({args.v2_model_name})")
        if args.enk_enabled or mode in ("equiformer_v2_enk", "equiformer_v2_enk_ep"):
            print(f"    ENK: enabled (r_bias={args.enk_init_r_bias}, q_bias={args.enk_init_q_bias})")
        if mode in ("equiformer_v2_enk_ep", "equiformer_v2_electron_prior"):
            print(f"    EP: mode={args.electron_prior_mode} "
                  f"freeze={args.electron_prior_freeze} "
                  f"runtime={args.electron_prior_runtime_mode}")
            print(f"    ckpt: {args.electron_prior_ckpt}")

    if args.init_ckpt:
        print(f"  init_ckpt: {args.init_ckpt}")
    if args.polar_init_ckpt:
        print(f"  polar_init_ckpt: {args.polar_init_ckpt} (for depolar weight transfer)")
    if getattr(args, 'dipole_init_ckpt', ''):
        print(f"  dipole_init_ckpt: {args.dipole_init_ckpt} (for sobolev_dipole weight transfer)")

    # Load data
    print("-" * 70)
    print("Loading data...")
    loaders, stats = _build_loaders(args)
    for split, loader in loaders.items():
        n_batches = len(loader)
        print(f"  {split}: {n_batches} batches")
    print("-" * 70)

    output_dir = Path(args.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "mode": mode,
        "dataset": args.dataset,
        "device": args.device,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lr_scheduler": args.lr_scheduler,
            "lr_min": args.lr_min,
            "hidden_nf": args.hidden_nf,
            "num_layers": args.num_layers,
            "num_basis": args.num_basis,
            "radius": args.radius,
            "static_loss_type": args.static_loss_type,
            "deriv_loss_type": args.deriv_loss_type,
        },
    }

    if mode == "equiformer_electron_prior":
        results["config"]["electron_prior"] = {
            "mode": args.electron_prior_mode,
            "ckpt": args.electron_prior_ckpt,
            "freeze": args.electron_prior_freeze,
            "runtime_mode": args.electron_prior_runtime_mode,
            "scale": args.electron_prior_scale,
            "heads": args.electron_prior_heads,
        }

    # --- Train branches ---
    if args.task == "polar":
        print("\n" + "=" * 70)
        print(f" Training POLAR branch ({mode})")
        print("=" * 70)
        polar_args = copy.copy(args)
        polar_args.strict_detanet_loss = False
        polar_args.sobolev_lambda_static = 1.0
        polar_args.sobolev_lambda_deriv = 0.0
        results["polar"] = _train_branch("polar", polar_args, loaders, stats, output_dir)

    if args.task == "depolar":
        print("\n" + "=" * 70)
        print(f" Training DEPOLAR branch ({mode})")
        print(f"  depolar_max_abs={args.depolar_max_abs} (graph-level outlier filter)")
        print("=" * 70)
        depolar_args = copy.copy(args)
        depolar_args.strict_detanet_loss = True
        depolar_args.sobolev_lambda_static = 0.0
        depolar_args.sobolev_lambda_deriv = 1.0
        results["depolar"] = _train_branch("depolar", depolar_args, loaders, stats, output_dir)

    if args.task == "sobolev_polar":
        print("\n" + "=" * 70)
        print(f" Training SOBOLEV POLAR branch ({mode})")
        print(f" Joint training: static \u03b1 + derivative \u2202\u03b1/\u2202R via autograd")
        print("=" * 70)
        print(f"  \u03bb_static={args.sobolev_lambda_static} \u03bb_deriv={args.sobolev_lambda_deriv}")
        print(f"  deriv_loss={args.sobolev_deriv_loss_type} warmup={args.sobolev_warmup_epochs}")
        print(f"  depolar_clip={args.sobolev_depolar_clip} depolar_max_abs={args.depolar_max_abs}")
        print(f"  jac_floor={args.sobolev_jac_floor} best_metric={args.sobolev_best_metric}")
        sp_args = copy.copy(args)
        results["sobolev_polar"] = _train_branch("sobolev_polar", sp_args, loaders, stats, output_dir)

    if args.task == "sobolev_dipole":
        print("\n" + "=" * 70)
        print(f" Training SOBOLEV DIPOLE branch ({mode})")
        print(f" Joint training: static \u03bc + derivative \u2202\u03bc/\u2202R via autograd")
        print("=" * 70)
        print(f"  \u03bb_static={args.sobolev_lambda_static} \u03bb_deriv={args.sobolev_lambda_deriv}")
        print(f"  deriv_loss={args.sobolev_deriv_loss_type} warmup={args.sobolev_warmup_epochs}")
        print(f"  dedipole_clip={args.sobolev_dedipole_clip} dedipole_max_abs={args.dedipole_max_abs}")
        print(f"  jac_floor={args.sobolev_jac_floor} best_metric={args.sobolev_dipole_best_metric}")
        sd_args = copy.copy(args)
        results["sobolev_dipole"] = _train_branch("sobolev_dipole", sd_args, loaders, stats, output_dir)

    if args.task == "dipole":
        print("\n" + "=" * 70)
        print(f" Training DIPOLE branch ({mode})")
        print("=" * 70)
        dip_args = copy.copy(args)
        dip_args.strict_detanet_loss = False
        results["dipole"] = _train_branch("dipole", dip_args, loaders, stats, output_dir)

    if args.task == "dedipole":
        print("\n" + "=" * 70)
        print(f" Training DEDIPOLE branch ({mode})")
        print("=" * 70)
        dd_args = copy.copy(args)
        dd_args.strict_detanet_loss = True
        results["dedipole"] = _train_branch("dedipole", dd_args, loaders, stats, output_dir)

    if args.task == "hii":
        print("\n" + "=" * 70)
        print(f" Training HII branch ({mode})")
        print("=" * 70)
        hii_args = copy.copy(args)
        hii_args.strict_detanet_loss = True
        results["hii"] = _train_branch("hii", hii_args, loaders, stats, output_dir)

    if args.task == "hij":
        print("\n" + "=" * 70)
        print(f" Training HIJ branch ({mode})")
        print("=" * 70)
        hij_args = copy.copy(args)
        hij_args.strict_detanet_loss = True
        results["hij"] = _train_branch("hij", hij_args, loaders, stats, output_dir)

    if args.task == "spectra4":
        # Train all 4 spectral tasks sequentially: hii → hij → dedipole → depolar
        for sp_branch in _SPECTRA4_TASKS:
            print("\n" + "=" * 70)
            print(f" Training {sp_branch.upper()} branch [spectra4] ({mode})")
            print("=" * 70)
            sp_args = copy.copy(args)
            sp_args.strict_detanet_loss = (sp_branch != "polar")
            results[sp_branch] = _train_branch(sp_branch, sp_args, loaders, stats, output_dir)

    # --- Save results ---
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    # --- Print summary ---
    _ALL_BRANCHES = ["polar", "depolar", "sobolev_polar", "sobolev_dipole", "dipole", "dedipole", "hii", "hij"]
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    for branch_name in _ALL_BRANCHES:
        if branch_name not in results:
            continue
        br = results[branch_name]
        best = br.get("best_val", {})
        params = br.get("params", 0)
        # Get last epoch val/test metrics
        if br["history"]:
            last = br["history"][-1]
            val_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(last.get("val", {}).items()))
            test_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(last.get("test", {}).items()))
        else:
            val_str = "N/A"
            test_str = "N/A"
        best_str = ", ".join(f"{k}={v:.4f}" for k, v in best.items())
        print(f"  {branch_name}: params={params:,} best=[{best_str}]")
        print(f"    last val:  {val_str}")
        print(f"    last test: {test_str}")

    print(f"\nCheckpoints saved in: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
