#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training Pipeline 2: Joint simg + qcMol training for NBO Foundation Model v2.

Three training modes:
  --mode simg_pretrain : Stage 1 — pure simg warmup (identical to train_nbo_v2.py)
  --mode joint         : Stage 2 — simg + qcMol mixed-batch training
  --mode qcmol_heavy   : Stage 3 — qcMol-heavy with simg replay

Usage:
    # Full 3-stage curriculum:
    python train_nbo_joint.py --mode simg_pretrain --epochs 50 ...
    python train_nbo_joint.py --mode joint --resume_from checkpoints/.../best.pt --epochs 100 ...
    python train_nbo_joint.py --mode qcmol_heavy --resume_from checkpoints/.../best.pt --epochs 60 ...

    # Or load Pipeline 1 weights directly:
    python train_nbo_joint.py --mode joint --simg_backbone checkpoints/.../nbo_foundation_v2_best.pt ...
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import random
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from torch_geometric.data import Batch
except ImportError as e:
    raise ImportError("torch_geometric is required") from e

# Local imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model_nbo_v2 import NBOFoundationModel, NBOFoundationModelQcMol
from nbo_loader import get_nbo_dataloaders, collate_for_nbo_lp

# qcMol imports
sys.path.insert(0, os.path.join(_THIS_DIR, 'qcmol'))
from qcmol_loader import QcMolKeys, QcMolLocalDataset, QcMolPackedDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_autocast_ctx(use_amp: bool, amp_dtype: str):
    if use_amp and torch.cuda.is_available():
        dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        return torch.amp.autocast('cuda', dtype=dtype)
    return contextlib.nullcontext()


_QCMOL_SUBSET_DESCRIPTIONS = {
    'pubchem': 'PubChemQC ∩ ZINC drug-like subset; broad small-molecule quantum/NBO coverage.',
    'pdbbind2020': 'PDBbind 2020 protein-ligand subset; biased toward bioactive bound ligands and recognition-relevant chemistry.',
    'custom': 'User-specified qcMol subset root/output paths.',
}


def _arg_provided(flag: str) -> bool:
    return flag in sys.argv


def _resolve_local_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_THIS_DIR, path))


def _apply_qcmol_subset_preset(args) -> None:
    subset = str(getattr(args, 'qcmol_subset', 'custom')).lower()
    if subset == 'pdbbind2020':
        if not _arg_provided('--qcmol_data_root'):
            args.qcmol_data_root = 'qcmol/PDBbind2020'
        if not _arg_provided('--qcmol_packed_dir'):
            args.qcmol_packed_dir = 'qcmol/qcmol_packed_pdbbind2020'
    elif subset == 'pubchem':
        if not _arg_provided('--qcmol_packed_dir'):
            args.qcmol_packed_dir = 'qcmol/qcmol_packed_1'

    if args.qcmol_data_root:
        args.qcmol_data_root = _resolve_local_path(args.qcmol_data_root)
    if args.qcmol_packed_dir:
        args.qcmol_packed_dir = _resolve_local_path(args.qcmol_packed_dir)

    print(
        f"qcMol subset: {subset} | desc={_QCMOL_SUBSET_DESCRIPTIONS.get(subset, 'n/a')} | "
        f"format={args.qcmol_format} | data_root={args.qcmol_data_root} | packed_dir={args.qcmol_packed_dir}"
    )


def _get_unique_parameters(*modules) -> list:
    params = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(param)
    return params


def _resolve_resume_bundle(state) -> Optional[dict]:
    if not isinstance(state, dict):
        return None
    if 'checkpoint_version' in state and 'simg_model' in state:
        return state
    return None


def _build_exact_mixed_schedule(total_batches: int, simg_ratio: float, seed: int) -> list:
    simg_ratio = float(min(max(simg_ratio, 0.0), 1.0))
    simg_batches = int(round(total_batches * simg_ratio))
    simg_batches = min(max(simg_batches, 0), total_batches)
    qcmol_batches = total_batches - simg_batches
    schedule = [True] * simg_batches + [False] * qcmol_batches
    if total_batches > 1:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        perm = torch.randperm(total_batches, generator=generator).tolist()
        schedule = [schedule[i] for i in perm]
    return schedule


def _save_checkpoint_bundle(
    path: str,
    epoch: int,
    best_score: float,
    mode: str,
    simg_model,
    qcmol_model,
    optimizer,
    scheduler,
    simg_norm_stats: dict,
    qcmol_stats: dict,
    args,
    qcmol_split: Optional[dict] = None,
):
    bundle = {
        'checkpoint_version': 1,
        'epoch': int(epoch),
        'best_score': float(best_score),
        'mode': mode,
        'simg_model': simg_model.state_dict(),
        'qcmol_model': qcmol_model.state_dict() if qcmol_model is not None else None,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'simg_norm_stats': simg_norm_stats,
        'qcmol_stats': qcmol_stats,
        'args': vars(args),
        'qcmol_split': qcmol_split,
    }
    torch.save(bundle, path)


# ===================================================================
# Loss utilities
# ===================================================================

def _masked_l1(pred, target, mask, mean=None, std=None):
    if pred is None or target is None or pred.numel() == 0 or target.numel() == 0:
        return None
    if pred.shape != target.shape:
        return None
    if mean is not None and std is not None:
        mean = mean.to(pred.device)
        std = std.to(pred.device)
        pred = (pred - mean) / std
        target = (target - mean) / std
    if mask is None:
        return nn.functional.l1_loss(pred, target)
    if mask.dim() == 1:
        mask = mask.view(-1, 1)
    if mask.shape[0] != pred.shape[0]:
        return None
    if mask.shape[1] == 1 and pred.dim() == 2:
        mask = mask.expand_as(pred)
    denom = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def _masked_mae_sums(pred, target, mask):
    if pred is None or target is None or pred.numel() == 0 or target.numel() == 0:
        return None
    if pred.shape != target.shape:
        return None
    if pred.dim() == 1:
        pred = pred.view(-1, 1)
    if target.dim() == 1:
        target = target.view(-1, 1)
    if mask is None:
        mask = torch.ones_like(target)
    else:
        if mask.dim() == 1:
            mask = mask.view(-1, 1)
        if mask.shape[1] == 1 and target.dim() == 2:
            mask = mask.expand_as(target)
        if mask.shape != target.shape:
            return None
    diff = torch.abs(pred - target) * mask
    return diff.sum(dim=0), mask.sum(dim=0)


def _format_mae(vec) -> str:
    if isinstance(vec, torch.Tensor):
        vec = vec.tolist()
    return "[" + ", ".join(f"{v:.4f}" for v in vec) + "]"


# ===================================================================
# Composite validation score
# ===================================================================

def composite_val_score(maes: dict, link_metrics: dict) -> float:
    charge = maes.get('charge', 0.0)
    bond = maes.get('bond_props', 0.0)
    inter = maes.get('interaction_E2', 0.0)
    link_err = 1.0 - link_metrics.get('acc', 0.0)
    return 0.2 * charge + 0.2 * bond + 0.4 * inter + 0.2 * link_err


# ===================================================================
# simg training epoch (for simg_pretrain mode or simg replay)
# ===================================================================

def simg_train_epoch(
    model, loader, optimizer, norm_stats, device, loss_weights,
    max_grad_norm=1.0, use_amp=False, amp_dtype='bf16', scaler=None,
):
    """Standard simg training epoch on NBOFoundationModel."""
    model.train()
    total_loss = 0.0
    w_atom, w_bond, w_inter, w_link = loss_weights
    crit_prop = nn.L1Loss()
    crit_link = nn.BCEWithLogitsLoss()
    atom_mean = norm_stats['atom_mean'].to(device)
    atom_std = norm_stats['atom_std'].to(device)
    bond_mean = norm_stats['bond_mean'].to(device)
    bond_std = norm_stats['bond_std'].to(device)

    autocast_ctx = _make_autocast_ctx(use_amp, amp_dtype)

    for data in tqdm(loader, desc="simg Train"):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        data.y_atomic_props = (data.y_atomic_props - atom_mean) / atom_std
        if data.y_bond_props.numel() > 0:
            data.y_bond_props = (data.y_bond_props - bond_mean) / bond_std

        with autocast_ctx:
            out = model(data, link_prediction_candidates=data.link_prediction_candidates)
            pred_a = torch.nan_to_num(out['pred_atom'])
            pred_b = torch.nan_to_num(out['pred_bond'])
            pred_i = torch.nan_to_num(out['pred_interaction'])
            pred_l = torch.nan_to_num(out['pred_link_logits'])

            loss_a = crit_prop(pred_a, data.y_atomic_props)
            loss_b = torch.tensor(0.0, device=device)
            if data.atom_bond_index.numel() > 0 and pred_b.numel() > 0:
                mask = getattr(data, 'bond_supervised_mask', None)
                if mask is not None and mask.numel() == pred_b.shape[0] and mask.any():
                    loss_b = crit_prop(pred_b[mask], data.y_bond_props[mask])
                elif data.y_bond_props.numel() > 0 and data.y_bond_props.shape[0] == pred_b.shape[0]:
                    loss_b = crit_prop(pred_b, data.y_bond_props)
            loss_i = torch.tensor(0.0, device=device)
            if data.interaction_edge_index.numel() > 0 and pred_i.numel() > 0:
                loss_i = crit_prop(pred_i, data.y_interaction_props)
            loss_link = torch.tensor(0.0, device=device)
            if pred_l.numel() > 0:
                loss_link = crit_link(pred_l, data.link_prediction_labels)

            loss = w_atom * loss_a + w_bond * loss_b + w_inter * loss_i + w_link * loss_link
        if not torch.isfinite(loss):
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / max(1, len(loader.dataset))


# ===================================================================
# qcMol training epoch
# ===================================================================

def qcmol_train_epoch(model, loader, optimizer, stats, device,
                      aux_atom_weight=0.5, aux_bond_weight=0.5, max_grad_norm=1.0,
                      use_amp=False, amp_dtype='bf16', scaler=None):
    """qcMol training epoch on NBOFoundationModelQcMol."""
    model.train()
    total_loss = 0.0

    atom_mean = stats.get('atom_mean')
    atom_std = stats.get('atom_std')
    bond_mean = stats.get('bond_mean')
    bond_std = stats.get('bond_std')
    aux_atom_mean = stats.get('aux_atom_mean')
    aux_atom_std = stats.get('aux_atom_std')
    aux_bond_mean = stats.get('aux_bond_mean')
    aux_bond_std = stats.get('aux_bond_std')

    autocast_ctx = _make_autocast_ctx(use_amp, amp_dtype)

    for data in tqdm(loader, desc="qcMol Train"):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            out = model(data)
            pred_atom = out['pred_atom']
            pred_bond = out['pred_bond']
            pred_aux_atom = out.get('pred_aux_atom', None)
            pred_aux_bond = out.get('pred_aux_bond', None)

            losses = []

            if hasattr(data, 'atom_targets') and data.atom_targets.numel() > 0:
                la = _masked_l1(pred_atom, data.atom_targets,
                                getattr(data, 'atom_target_mask', None), atom_mean, atom_std)
                if la is not None:
                    losses.append(la)

            if hasattr(data, 'bond_targets') and data.bond_targets.numel() > 0 and pred_bond.numel() > 0:
                lb = _masked_l1(pred_bond, data.bond_targets,
                                getattr(data, 'bond_target_mask', None), bond_mean, bond_std)
                if lb is not None:
                    losses.append(lb)

            if aux_atom_weight > 0 and pred_aux_atom is not None and hasattr(data, 'aux_atom_targets') and data.aux_atom_targets.numel() > 0:
                la2 = _masked_l1(pred_aux_atom, data.aux_atom_targets,
                                 getattr(data, 'aux_atom_target_mask', None), aux_atom_mean, aux_atom_std)
                if la2 is not None:
                    losses.append(la2 * aux_atom_weight)

            if aux_bond_weight > 0 and pred_aux_bond is not None and hasattr(data, 'aux_bond_targets') and data.aux_bond_targets.numel() > 0:
                lb2 = _masked_l1(pred_aux_bond, data.aux_bond_targets,
                                 getattr(data, 'aux_bond_target_mask', None), aux_bond_mean, aux_bond_std)
                if lb2 is not None:
                    losses.append(lb2 * aux_bond_weight)

        if losses:
            loss = torch.stack(losses).sum()
            if torch.isfinite(loss):
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                total_loss += loss.item() * data.num_graphs

    return total_loss / max(1, len(loader.dataset))


# ===================================================================
# qcMol eval epoch
# ===================================================================

def qcmol_eval_epoch(model, loader, stats, device, aux_atom_weight=0.5, aux_bond_weight=0.5):
    model.eval()
    total_loss = 0.0

    atom_mean = stats.get('atom_mean')
    atom_std = stats.get('atom_std')
    bond_mean = stats.get('bond_mean')
    bond_std = stats.get('bond_std')
    aux_atom_mean = stats.get('aux_atom_mean')
    aux_atom_std = stats.get('aux_atom_std')
    aux_bond_mean = stats.get('aux_bond_mean')
    aux_bond_std = stats.get('aux_bond_std')

    mae_sums: Dict[str, torch.Tensor] = {}
    mae_counts: Dict[str, torch.Tensor] = {}

    def _update(name, pred, target, mask):
        r = _masked_mae_sums(pred, target, mask)
        if r is None:
            return
        s, c = r
        s, c = s.float().cpu(), c.float().cpu()
        if name not in mae_sums:
            mae_sums[name] = s.clone()
            mae_counts[name] = c.clone()
        else:
            mae_sums[name] += s
            mae_counts[name] += c

    with torch.no_grad():
        for data in tqdm(loader, desc="qcMol Eval"):
            data = data.to(device)
            out = model(data)
            pred_atom = out['pred_atom']
            pred_bond = out['pred_bond']
            pred_aux_atom = out.get('pred_aux_atom', None)
            pred_aux_bond = out.get('pred_aux_bond', None)

            losses = []

            if hasattr(data, 'atom_targets') and data.atom_targets.numel() > 0:
                la = _masked_l1(pred_atom, data.atom_targets,
                                getattr(data, 'atom_target_mask', None), atom_mean, atom_std)
                if la is not None:
                    losses.append(la)
                _update('atom', pred_atom, data.atom_targets, getattr(data, 'atom_target_mask', None))

            if hasattr(data, 'bond_targets') and data.bond_targets.numel() > 0 and pred_bond.numel() > 0:
                lb = _masked_l1(pred_bond, data.bond_targets,
                                getattr(data, 'bond_target_mask', None), bond_mean, bond_std)
                if lb is not None:
                    losses.append(lb)
                _update('bond', pred_bond, data.bond_targets, getattr(data, 'bond_target_mask', None))

            if aux_atom_weight > 0 and pred_aux_atom is not None and hasattr(data, 'aux_atom_targets') and data.aux_atom_targets.numel() > 0:
                la2 = _masked_l1(pred_aux_atom, data.aux_atom_targets,
                                 getattr(data, 'aux_atom_target_mask', None), aux_atom_mean, aux_atom_std)
                if la2 is not None:
                    losses.append(la2 * aux_atom_weight)
                _update('aux_atom', pred_aux_atom, data.aux_atom_targets, getattr(data, 'aux_atom_target_mask', None))

            if aux_bond_weight > 0 and pred_aux_bond is not None and hasattr(data, 'aux_bond_targets') and data.aux_bond_targets.numel() > 0:
                lb2 = _masked_l1(pred_aux_bond, data.aux_bond_targets,
                                 getattr(data, 'aux_bond_target_mask', None), aux_bond_mean, aux_bond_std)
                if lb2 is not None:
                    losses.append(lb2 * aux_bond_weight)
                _update('aux_bond', pred_aux_bond, data.aux_bond_targets, getattr(data, 'aux_bond_target_mask', None))

            if losses:
                loss = torch.stack(losses).sum()
                if torch.isfinite(loss):
                    total_loss += loss.item() * data.num_graphs

    mae = {}
    for name, s in mae_sums.items():
        c = mae_counts[name]
        mae_vec = torch.full_like(c, float('nan'))
        valid = c > 0
        mae_vec[valid] = s[valid] / c[valid]
        mae[name] = mae_vec

    return total_loss / max(1, len(loader.dataset)), mae


# ===================================================================
# simg validation (reuse from train_nbo_v2 logic)
# ===================================================================

def simg_validate_epoch(model, loader, norm_stats, device):
    """Validate on simg data. model must be NBOFoundationModel (not QcMol variant)."""
    from train_nbo_v2 import validate_epoch
    return validate_epoch(model, loader, norm_stats, device)


# ===================================================================
# Compute qcMol stats
# ===================================================================

def compute_qcmol_stats(loader) -> Dict[str, torch.Tensor]:
    """Compute mean/std for qcMol targets."""
    sums, sumsqs, counts = {}, {}, {}

    def _expand(mask, target):
        if mask.dim() == 1:
            mask = mask.view(-1, 1)
        if mask.dim() == 2 and mask.shape[1] == 1 and target.dim() == 2:
            mask = mask.expand_as(target)
        if mask.shape != target.shape:
            return None
        return mask

    def _acc(name, target, mask):
        if target is None or target.numel() == 0:
            return
        if target.dim() == 1:
            target = target.view(-1, 1)
        if mask is None:
            mask = torch.ones_like(target)
        else:
            mask = _expand(mask, target)
        if mask is None:
            return
        if name not in sums:
            sums[name] = torch.zeros(target.size(1), dtype=torch.float32)
            sumsqs[name] = torch.zeros(target.size(1), dtype=torch.float32)
            counts[name] = torch.zeros(target.size(1), dtype=torch.float32)
        masked = target * mask
        sums[name] += masked.sum(dim=0).float()
        sumsqs[name] += (masked * target).sum(dim=0).float()
        counts[name] += mask.sum(dim=0).float()

    for data in tqdm(loader, desc="qcMol stats"):
        if hasattr(data, 'atom_targets') and data.atom_targets.numel() > 0:
            _acc('atom', data.atom_targets, getattr(data, 'atom_target_mask', None))
        if hasattr(data, 'bond_targets') and data.bond_targets.numel() > 0:
            _acc('bond', data.bond_targets, getattr(data, 'bond_target_mask', None))
        if hasattr(data, 'aux_atom_targets') and data.aux_atom_targets.numel() > 0:
            _acc('aux_atom', data.aux_atom_targets, getattr(data, 'aux_atom_target_mask', None))
        if hasattr(data, 'aux_bond_targets') and data.aux_bond_targets.numel() > 0:
            _acc('aux_bond', data.aux_bond_targets, getattr(data, 'aux_bond_target_mask', None))

    stats = {}
    for name in ['atom', 'bond', 'aux_atom', 'aux_bond']:
        if name not in sums:
            continue
        count = counts[name].clamp(min=1.0)
        mean = sums[name] / count
        var = sumsqs[name] / count - mean.pow(2)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        std[std < 1e-8] = 1.0
        stats[f'{name}_mean'] = mean
        stats[f'{name}_std'] = std
    return stats


# ===================================================================
# Joint training epoch (alternating simg + qcMol)
# ===================================================================

def joint_train_epoch(
    simg_model, qcmol_model,
    simg_loader, qcmol_loader,
    optimizer, simg_norm_stats, qcmol_stats, device,
    simg_weights, qcmol_aux_weights, simg_ratio=0.5, max_grad_norm=1.0,
    use_amp=False, amp_dtype='bf16', scaler=None, epoch_seed=0,
):
    """Alternating mini-batch: sample from simg with probability simg_ratio,
    else from qcMol. Both models share the backbone, so gradients flow through."""
    simg_model.train()
    qcmol_model.train()
    total = 0.0
    n_batches = 0

    crit_prop = nn.L1Loss()
    crit_link = nn.BCEWithLogitsLoss()
    w_atom, w_bond, w_inter, w_link = simg_weights
    aux_aw, aux_bw = qcmol_aux_weights

    am = simg_norm_stats['atom_mean'].to(device)
    astd = simg_norm_stats['atom_std'].to(device)
    bm = simg_norm_stats['bond_mean'].to(device)
    bstd = simg_norm_stats['bond_std'].to(device)
    clip_params = _get_unique_parameters(simg_model, qcmol_model)

    simg_iter = iter(simg_loader)
    qcmol_iter = iter(qcmol_loader)

    total_batches = len(simg_loader) + len(qcmol_loader)
    batch_schedule = _build_exact_mixed_schedule(total_batches, simg_ratio, epoch_seed)
    pbar = tqdm(total=total_batches, desc="Joint Train")
    autocast_ctx = _make_autocast_ctx(use_amp, amp_dtype)

    for use_simg in batch_schedule:

        if use_simg:
            try:
                data = next(simg_iter)
            except StopIteration:
                simg_iter = iter(simg_loader)
                data = next(simg_iter)
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            data.y_atomic_props = (data.y_atomic_props - am) / astd
            if data.y_bond_props.numel() > 0:
                data.y_bond_props = (data.y_bond_props - bm) / bstd

            with autocast_ctx:
                out = simg_model(data, link_prediction_candidates=data.link_prediction_candidates)
                pa = torch.nan_to_num(out['pred_atom'])
                pb = torch.nan_to_num(out['pred_bond'])
                pi_ = torch.nan_to_num(out['pred_interaction'])
                pl = torch.nan_to_num(out['pred_link_logits'])

                la = crit_prop(pa, data.y_atomic_props)
                lb_ = torch.tensor(0.0, device=device)
                if data.atom_bond_index.numel() > 0 and pb.numel() > 0:
                    mask = getattr(data, 'bond_supervised_mask', None)
                    if mask is not None and mask.numel() == pb.shape[0] and mask.any():
                        lb_ = crit_prop(pb[mask], data.y_bond_props[mask])
                    elif data.y_bond_props.numel() > 0 and data.y_bond_props.shape[0] == pb.shape[0]:
                        lb_ = crit_prop(pb, data.y_bond_props)
                li = torch.tensor(0.0, device=device)
                if data.interaction_edge_index.numel() > 0 and pi_.numel() > 0:
                    li = crit_prop(pi_, data.y_interaction_props)
                ll = torch.tensor(0.0, device=device)
                if pl.numel() > 0:
                    ll = crit_link(pl, data.link_prediction_labels)
                loss = w_atom * la + w_bond * lb_ + w_inter * li + w_link * ll
        else:
            try:
                data = next(qcmol_iter)
            except StopIteration:
                qcmol_iter = iter(qcmol_loader)
                data = next(qcmol_iter)
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx:
                out = qcmol_model(data)
                losses = []
                if hasattr(data, 'atom_targets') and data.atom_targets.numel() > 0:
                    l = _masked_l1(out['pred_atom'], data.atom_targets,
                                   getattr(data, 'atom_target_mask', None),
                                   qcmol_stats.get('atom_mean'), qcmol_stats.get('atom_std'))
                    if l is not None:
                        losses.append(l)
                if hasattr(data, 'bond_targets') and data.bond_targets.numel() > 0 and out['pred_bond'].numel() > 0:
                    l = _masked_l1(out['pred_bond'], data.bond_targets,
                                   getattr(data, 'bond_target_mask', None),
                                   qcmol_stats.get('bond_mean'), qcmol_stats.get('bond_std'))
                    if l is not None:
                        losses.append(l)
                if aux_aw > 0 and hasattr(data, 'aux_atom_targets') and data.aux_atom_targets.numel() > 0:
                    l = _masked_l1(out.get('pred_aux_atom'), data.aux_atom_targets,
                                   getattr(data, 'aux_atom_target_mask', None),
                                   qcmol_stats.get('aux_atom_mean'), qcmol_stats.get('aux_atom_std'))
                    if l is not None:
                        losses.append(l * aux_aw)
                if aux_bw > 0 and hasattr(data, 'aux_bond_targets') and data.aux_bond_targets.numel() > 0:
                    l = _masked_l1(out.get('pred_aux_bond'), data.aux_bond_targets,
                                   getattr(data, 'aux_bond_target_mask', None),
                                   qcmol_stats.get('aux_bond_mean'), qcmol_stats.get('aux_bond_std'))
                    if l is not None:
                        losses.append(l * aux_bw)
                loss = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=device)

        if torch.isfinite(loss) and loss.requires_grad:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
                optimizer.step()
            total += loss.item()
            n_batches += 1

        pbar.update(1)

    pbar.close()
    return total / max(1, n_batches)


# ===================================================================
# Main
# ===================================================================

def _collate_qcmol(batch):
    return Batch.from_data_list(batch)


def _build_qcmol_split_indices(n: int, seed: int, train_ratio: float = 0.9, restore: Optional[dict] = None):
    if restore is not None:
        train_idx = restore.get('train_indices', [])
        val_idx = restore.get('val_indices', [])
        if len(train_idx) + len(val_idx) == n:
            return train_idx, val_idx
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * train_ratio)
    return indices[:n_train], indices[n_train:]


def main(args):
    set_seed(args.seed)
    _apply_qcmol_subset_preset(args)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu is not None else 'cpu')
    print(f"Device: {device}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(args.out_dir, f"{timestamp}_{args.mode}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and args.amp_dtype == 'fp16')

    resume_bundle = None
    resume_qcmol_split = None
    start_epoch = 1

    if args.resume_from:
        print(f"Loading resume state from {args.resume_from}")
        raw_resume_state = torch.load(args.resume_from, map_location=device, weights_only=False)
        resume_bundle = _resolve_resume_bundle(raw_resume_state)
        if resume_bundle is not None:
            start_epoch = int(resume_bundle.get('epoch', 0)) + 1
            resume_qcmol_split = resume_bundle.get('qcmol_split')

    # ---- simg data ----
    simg_train_loader, simg_val_loader, _, simg_norm_stats = get_nbo_dataloaders(
        dataset_root=args.simg_dataset_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ---- qcMol data (for joint/qcmol_heavy modes) ----
    qcmol_train_loader = None
    qcmol_val_loader = None
    qcmol_stats = {}
    qcmol_dataset = None

    if args.mode in ('joint', 'qcmol_heavy'):
        keys = QcMolKeys(
            atom_targets=tuple(k.strip() for k in args.qcmol_atom_keys.split(',') if k.strip()),
            bond_targets=tuple(k.strip() for k in args.qcmol_bond_keys.split(',') if k.strip()),
            aux_atom_targets=tuple(k.strip() for k in args.qcmol_aux_atom_keys.split(',') if k.strip()),
            aux_bond_targets=tuple(k.strip() for k in args.qcmol_aux_bond_keys.split(',') if k.strip()),
        )

        if args.qcmol_format == 'packed':
            qcmol_dataset = QcMolPackedDataset(
                root=args.qcmol_packed_dir,
                shard_glob=args.qcmol_shard_glob,
                verbose=True,
            )
        else:
            qcmol_dataset = QcMolLocalDataset(
                root=args.qcmol_data_root,
                keys=keys,
                max_files=args.qcmol_max_files,
                max_mols_per_file=args.qcmol_max_mols,
                verbose=True,
            )

        n = len(qcmol_dataset)
        train_indices, val_indices = _build_qcmol_split_indices(
            n=n, seed=args.seed, train_ratio=0.9, restore=resume_qcmol_split,
        )
        qcmol_train_sub = Subset(qcmol_dataset, train_indices)
        qcmol_val_sub = Subset(qcmol_dataset, val_indices)
        qcmol_train_loader = DataLoader(qcmol_train_sub, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, collate_fn=_collate_qcmol)
        qcmol_val_loader = DataLoader(qcmol_val_sub, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, collate_fn=_collate_qcmol)

        print("Computing qcMol normalization stats...")
        qcmol_stats = compute_qcmol_stats(qcmol_train_loader)
        torch.save(qcmol_stats, os.path.join(run_dir, 'qcmol_stats.pt'))

    # ---- Model ----
    simg_model = NBOFoundationModel(
        hidden_dim=args.hidden_dim,
        n_token_mp_layers=args.n_token_mp_layers,
        n_token_mp_blocks=args.n_token_mp_blocks,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        atom_out_dim=4,
        bond_out_dim=2,
        interaction_out_dim=3,
        use_pair_guide=args.use_pair_guide,
        use_base_delta=args.use_base_delta,
    ).to(device)

    qcmol_model = None
    if args.mode in ('joint', 'qcmol_heavy'):
        # Infer dims from dataset
        qcmol_atom_dim = getattr(qcmol_dataset, 'atom_dim', None) or 12
        qcmol_bond_dim = getattr(qcmol_dataset, 'bond_dim', None) or 15
        aux_atom_dim = getattr(qcmol_dataset, 'aux_atom_dim', None) or 0
        aux_bond_dim = getattr(qcmol_dataset, 'aux_bond_dim', None) or 0

        qcmol_model = NBOFoundationModelQcMol(
            hidden_dim=args.hidden_dim,
            n_token_mp_layers=args.n_token_mp_layers,
            n_token_mp_blocks=args.n_token_mp_blocks,
            num_rbf=args.num_rbf,
            cutoff=args.cutoff,
            atom_out_dim=4,
            bond_out_dim=2,
            interaction_out_dim=3,
            qcmol_atom_dim=qcmol_atom_dim,
            qcmol_bond_dim=qcmol_bond_dim,
            aux_atom_dim=aux_atom_dim,
            aux_bond_dim=aux_bond_dim,
            use_pair_guide=args.use_pair_guide,
            use_base_delta=args.use_base_delta,
        ).to(device)

        # Share backbone weights (point to same object)
        qcmol_model.backbone = simg_model

    # ---- Load pretrained weights ----
    if args.simg_backbone:
        print(f"Loading simg backbone from {args.simg_backbone}")
        _raw = torch.load(args.simg_backbone, map_location=device, weights_only=False)
        # Support both plain state_dict and joint-bundle format
        # (joint bundle: dict with keys checkpoint_version / simg_model / qcmol_model / ...)
        if isinstance(_raw, dict) and 'simg_model' in _raw:
            print("  [auto] detected joint-bundle format — extracting simg_model weights")
            state = _raw['simg_model']
            # Optionally also seed the qcmol heads when they are present
            if qcmol_model is not None and _raw.get('qcmol_model') is not None:
                print("  [auto] also loading qcmol_model weights from bundle")
                qcmol_model.load_state_dict(_raw['qcmol_model'], strict=False)
        else:
            state = _raw
        simg_model.load_state_dict(state, strict=True)

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        state = resume_bundle if resume_bundle is not None else raw_resume_state
        if resume_bundle is not None:
            simg_model.load_state_dict(resume_bundle['simg_model'], strict=True)
            if qcmol_model is not None and resume_bundle.get('qcmol_model') is not None:
                qcmol_model.load_state_dict(resume_bundle['qcmol_model'], strict=False)
        else:
            # Backward-compatible raw state_dict loading
            try:
                simg_model.load_state_dict(state, strict=True)
            except RuntimeError:
                if qcmol_model is not None:
                    qcmol_model.load_state_dict(state, strict=False)

    # ---- Optimizer ----
    if qcmol_model is not None:
        # Collect unique parameters from both models (backbone is shared)
        all_params = _get_unique_parameters(simg_model, qcmol_model)
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(simg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience,
    )

    simg_weights = (args.w_atom, args.w_bond, args.w_interaction, args.w_link)
    qcmol_aux_weights = (args.aux_atom_weight, args.aux_bond_weight)

    best_score = float('inf')

    if resume_bundle is not None:
        if resume_bundle.get('optimizer') is not None:
            optimizer.load_state_dict(resume_bundle['optimizer'])
        if resume_bundle.get('scheduler') is not None:
            scheduler.load_state_dict(resume_bundle['scheduler'])
        best_score = float(resume_bundle.get('best_score', best_score))
        if resume_bundle.get('qcmol_stats'):
            qcmol_stats = resume_bundle['qcmol_stats']
        print(f"Resume bundle loaded: start_epoch={start_epoch}, best_score={best_score:.4f}")

    print(f"\n--- Mode: {args.mode} | Epochs: {args.epochs} ---")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.mode == 'simg_pretrain':
            train_loss = simg_train_epoch(
                simg_model, simg_train_loader, optimizer,
                simg_norm_stats, device, simg_weights, args.max_grad_norm,
                use_amp=use_amp, amp_dtype=args.amp_dtype, scaler=scaler,
            )
            val_maes, link_met = simg_validate_epoch(simg_model, simg_val_loader, simg_norm_stats, device)
            score = composite_val_score(val_maes, link_met)
            scheduler.step(score)

            lr_now = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:03d} [lr={lr_now:.2e}] Train={train_loss:.4f} | "
                  f"Charge={val_maes['charge']:.4f} Bond={val_maes['bond_props']:.4f} "
                  f"Int_E2={val_maes['interaction_E2']:.4f} Comp={score:.4f}")

        elif args.mode == 'joint':
            simg_ratio = args.simg_ratio
            train_loss = joint_train_epoch(
                simg_model, qcmol_model,
                simg_train_loader, qcmol_train_loader,
                optimizer, simg_norm_stats, qcmol_stats, device,
                simg_weights, qcmol_aux_weights,
                simg_ratio=simg_ratio, max_grad_norm=args.max_grad_norm,
                use_amp=use_amp, amp_dtype=args.amp_dtype, scaler=scaler,
                epoch_seed=args.seed + epoch,
            )
            # Validate on simg
            val_maes, link_met = simg_validate_epoch(simg_model, simg_val_loader, simg_norm_stats, device)
            simg_score = composite_val_score(val_maes, link_met)
            # Validate on qcMol
            qcmol_val_loss, qcmol_mae = qcmol_eval_epoch(
                qcmol_model, qcmol_val_loader, qcmol_stats, device,
                args.aux_atom_weight, args.aux_bond_weight,
            )
            score = 0.5 * simg_score + 0.5 * qcmol_val_loss
            scheduler.step(score)

            lr_now = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:03d} [lr={lr_now:.2e}] Train={train_loss:.4f} | "
                  f"simg_comp={simg_score:.4f} qcmol_val={qcmol_val_loss:.4f} | "
                  f"combined={score:.4f}")
            for k, v in qcmol_mae.items():
                print(f"  qcmol_{k}_mae={_format_mae(v)}")

        elif args.mode == 'qcmol_heavy':
            # qcMol-heavy with simg replay
            simg_ratio = args.simg_replay_ratio
            train_loss = joint_train_epoch(
                simg_model, qcmol_model,
                simg_train_loader, qcmol_train_loader,
                optimizer, simg_norm_stats, qcmol_stats, device,
                simg_weights, qcmol_aux_weights,
                simg_ratio=simg_ratio, max_grad_norm=args.max_grad_norm,
                use_amp=use_amp, amp_dtype=args.amp_dtype, scaler=scaler,
                epoch_seed=args.seed + epoch,
            )
            qcmol_val_loss, qcmol_mae = qcmol_eval_epoch(
                qcmol_model, qcmol_val_loader, qcmol_stats, device,
                args.aux_atom_weight, args.aux_bond_weight,
            )
            # Also validate simg to catch forgetting
            val_maes, link_met = simg_validate_epoch(simg_model, simg_val_loader, simg_norm_stats, device)
            simg_score = composite_val_score(val_maes, link_met)
            score = 0.3 * simg_score + 0.7 * qcmol_val_loss
            scheduler.step(score)

            lr_now = optimizer.param_groups[0]['lr']
            print(f"Ep {epoch:03d} [lr={lr_now:.2e}] Train={train_loss:.4f} | "
                  f"simg_comp={simg_score:.4f} qcmol_val={qcmol_val_loss:.4f} | "
                  f"combined={score:.4f}")
            for k, v in qcmol_mae.items():
                print(f"  qcmol_{k}_mae={_format_mae(v)}")

        if score < best_score:
            best_score = score
            # Save simg backbone
            torch.save(simg_model.state_dict(), os.path.join(run_dir, 'nbo_foundation_backbone_best.pt'))
            # Save full qcmol model if applicable
            if qcmol_model is not None:
                torch.save(qcmol_model.state_dict(), os.path.join(run_dir, 'nbo_foundation_qcmol_best.pt'))
            _save_checkpoint_bundle(
                path=os.path.join(run_dir, 'nbo_foundation_training_best.pt'),
                epoch=epoch,
                best_score=best_score,
                mode=args.mode,
                simg_model=simg_model,
                qcmol_model=qcmol_model,
                optimizer=optimizer,
                scheduler=scheduler,
                simg_norm_stats=simg_norm_stats,
                qcmol_stats=qcmol_stats,
                args=args,
                qcmol_split={
                    'train_indices': train_indices if qcmol_dataset is not None else [],
                    'val_indices': val_indices if qcmol_dataset is not None else [],
                },
            )
            # Save stats
            all_stats = dict(simg_norm_stats)
            all_stats.update(qcmol_stats)
            all_stats['best_score'] = best_score
            all_stats['best_epoch'] = epoch
            all_stats['mode'] = args.mode
            torch.save(all_stats, os.path.join(run_dir, 'norm_stats.pt'))
            print(f"  -> New best score={score:.4f}. Saved.")

        _save_checkpoint_bundle(
            path=os.path.join(run_dir, 'nbo_foundation_training_last.pt'),
            epoch=epoch,
            best_score=best_score,
            mode=args.mode,
            simg_model=simg_model,
            qcmol_model=qcmol_model,
            optimizer=optimizer,
            scheduler=scheduler,
            simg_norm_stats=simg_norm_stats,
            qcmol_stats=qcmol_stats,
            args=args,
            qcmol_split={
                'train_indices': train_indices if qcmol_dataset is not None else [],
                'val_indices': val_indices if qcmol_dataset is not None else [],
            },
        )

    print(f"\nDone. Best score: {best_score:.4f}")
    # Save args as JSON
    with open(os.path.join(run_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='NBO Foundation v2 — Joint Training')

    # Mode
    ap.add_argument('--mode', type=str, choices=['simg_pretrain', 'joint', 'qcmol_heavy'], default='joint')
    ap.add_argument('--run_name', type=str, default='nbo_joint')

    # Paths
    ap.add_argument('--simg_dataset_root', type=str,
                    default='datasets')
    ap.add_argument('--out_dir', type=str, default='checkpoints')
    ap.add_argument('--simg_backbone', type=str, default='', help='Path to Pipeline 1 simg backbone .pt')
    ap.add_argument('--resume_from', type=str, default='', help='Resume full model from .pt')

    # qcMol paths
    ap.add_argument('--qcmol_subset', type=str, choices=['pubchem', 'pdbbind2020', 'custom'], default='pubchem')
    ap.add_argument('--qcmol_data_root', type=str,
                    default='datasets/qcMol/PubChem')
    ap.add_argument('--qcmol_format', type=str, choices=['local', 'packed'], default='packed')
    ap.add_argument('--qcmol_packed_dir', type=str, default='qcmol/qcmol_packed_1')
    ap.add_argument('--qcmol_shard_glob', type=str, default='qcmol_packed_shard_*.pt')
    ap.add_argument('--qcmol_max_files', type=int, default=None)
    ap.add_argument('--qcmol_max_mols', type=int, default=None)
    ap.add_argument('--qcmol_atom_keys', type=str, default='NAO,LP,NPA')
    ap.add_argument('--qcmol_bond_keys', type=str, default='NBO')
    ap.add_argument('--qcmol_aux_atom_keys', type=str, default='ADCH,LI,ELF')
    ap.add_argument('--qcmol_aux_bond_keys', type=str, default='DI,LBO,Mayer')

    # Training
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--max_grad_norm', type=float, default=1.0)

    # Architecture
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--n_token_mp_layers', type=int, default=2)
    ap.add_argument('--n_token_mp_blocks', type=int, default=3)
    ap.add_argument('--num_rbf', type=int, default=64)
    ap.add_argument('--cutoff', type=float, default=5.0)
    ap.add_argument('--use_pair_guide', action='store_true', default=True)
    ap.add_argument('--no_pair_guide', dest='use_pair_guide', action='store_false')
    ap.add_argument('--use_base_delta', action='store_true', default=True)
    ap.add_argument('--no_base_delta', dest='use_base_delta', action='store_false')

    # Loss weights (simg)
    ap.add_argument('--w_atom', type=float, default=0.2)
    ap.add_argument('--w_bond', type=float, default=0.2)
    ap.add_argument('--w_interaction', type=float, default=0.6)
    ap.add_argument('--w_link', type=float, default=1.0)

    # qcMol weights
    ap.add_argument('--aux_atom_weight', type=float, default=0.5)
    ap.add_argument('--aux_bond_weight', type=float, default=0.5)

    # Joint training ratio
    ap.add_argument('--simg_ratio', type=float, default=0.5, help='simg batch probability in joint mode')
    ap.add_argument('--simg_replay_ratio', type=float, default=0.2, help='simg replay ratio in qcmol_heavy mode')

    # Hardware
    ap.add_argument('--gpu', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--amp_dtype', type=str, choices=['bf16', 'fp16'], default='bf16')

    args = ap.parse_args()
    main(args)
