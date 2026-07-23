"""
compute_nbo_train_stats.py
==========================
Generate NBO-GSC v2 train_stats.pt using the SAME training data pipeline
as train_senk.py.  No external file lists required.

Iterates over QMe14s training batches with EP-enabled models, collects
per-bond and per-atom NBO features + model predictions, and saves
{mean, std} for z-score-based GSC calibration.

Output keys:
  log_hij_long_mean / log_hij_long_std   -- model's along-bond log|Hij|
  log_occ_mean / log_occ_std             -- NBO log(bond occupancy)
  dd_trace_mean / dd_trace_std           -- model's dd trace
  npa_charge_mean / npa_charge_std       -- NBO NPA natural charge
  dp_norm_mean / dp_norm_std             -- model's dp per-atom norm
  delocal_mean / delocal_std             -- NBO delocalisation proxy

Usage (minimal -- reuses training defaults):
    python compute_nbo_train_stats.py \\
        --task hij \\
        --hij_ckpt  checkpoints/.../hij/best.pt  --hij_mode  equiformer_v2_enk_ep \\
        --dd_ckpt   checkpoints/.../dd/best.pt   --dd_mode   equiformer_v2_enk_ep \\
        --dp_ckpt   checkpoints/.../dp/best.pt   --dp_mode   equiformer_v2_enk_ep \\
        --electron_prior_ckpt  nbo_nets/.../nbo_foundation_training_best.pt \\
        --electron_prior_stats nbo_nets/.../norm_stats.pt \\
        --num_batches 100 --output nbo_train_stats.pt --gpu 0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.nn import radius_graph

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import senk_train_shared as train_mod
from v2_spectra_infer import (
    _load_multitask,
    _load_depolar,
    _is_ep_runtime_active,
    _extract_nbo_from_model,
    _collect_enk_context,
    _run_with_ep_cache_guard,
    _ALL_MODES,
    build_skeleton_online,
)
from nbo_consistency_loss import _match_undirected_edges_simple
from nbo_spectral_calibration import NBOGuidedCalibrator
from nbo_spectral_calibration import _bond_occupancy_col

# Re-use training-data loader builder from train_senk
from train_senk import _build_loaders


# --- Stats accumulation (Welford-style running statistics) ---

def _running_stats(accum: dict, key: str, values: torch.Tensor) -> None:
    v = values.detach().cpu().float()
    finite = torch.isfinite(v)
    v = v[finite]
    if v.numel() == 0:
        return
    if key not in accum:
        accum[key] = {"n": 0, "sum": 0.0, "sum_sq": 0.0}
    accum[key]["n"] += int(v.numel())
    accum[key]["sum"] += float(v.sum().item())
    accum[key]["sum_sq"] += float((v * v).sum().item())


def _running_hij_class_stats(accum: dict, class_name: str, key: str, values: torch.Tensor) -> None:
    class_accum = accum.setdefault("__hij_bond_class_stats__", {})
    cls_accum = class_accum.setdefault(str(class_name), {})
    _running_stats(cls_accum, key, values)


def _finalize_one_accum(accum: dict) -> dict:
    out = {}
    for key, a in accum.items():
        if not isinstance(a, dict) or not {"n", "sum", "sum_sq"}.issubset(a.keys()):
            continue
        n = a["n"]
        if n < 2:
            continue
        mean = a["sum"] / n
        variance = max(a["sum_sq"] / n - mean * mean, 1e-12)
        out[f"{key}_mean"] = float(mean)
        out[f"{key}_std"] = float(math.sqrt(variance))
        out[f"{key}_n"] = int(n)
    return out


def _finalize_stats(accum: dict) -> dict:
    out = _finalize_one_accum(accum)
    seen_elements = accum.get("__hij_seen_elements__", set())
    if seen_elements:
        out["hij_seen_elements"] = sorted(int(x) for x in seen_elements)
    class_accum = accum.get("__hij_bond_class_stats__", {})
    class_stats = {}
    if isinstance(class_accum, dict):
        for cls, cls_accum in class_accum.items():
            finalized = _finalize_one_accum(cls_accum)
            if finalized:
                class_stats[str(cls)] = finalized
    if class_stats:
        out["hij_bond_class_stats"] = class_stats
        out["hij_seen_bond_classes"] = sorted(class_stats.keys())
    env_signatures = accum.get("__hij_seen_env_signatures__", set())
    if env_signatures:
        out["hij_seen_env_signatures"] = sorted(str(x) for x in env_signatures)
    return out


# --- Per-batch stats collection ---

def _drop_cross_batch_bonds(
    atom_bond_local: Optional[torch.Tensor],
    bond_pred: Optional[torch.Tensor],
    batch: torch.Tensor,
    num_nodes: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if atom_bond_local is None or atom_bond_local.numel() == 0:
        return atom_bond_local, bond_pred
    bonds = atom_bond_local.long()
    src, dst = bonds[0], bonds[1]
    in_range = (
        (src >= 0)
        & (src < int(num_nodes))
        & (dst >= 0)
        & (dst < int(num_nodes))
    )
    src_safe = src.clamp(0, max(int(num_nodes) - 1, 0))
    dst_safe = dst.clamp(0, max(int(num_nodes) - 1, 0))
    valid = in_range & (batch[src_safe] == batch[dst_safe])
    if bool(valid.all().item()):
        return atom_bond_local, bond_pred
    filtered_bonds = atom_bond_local[:, valid]
    filtered_pred = bond_pred
    if bond_pred is not None and bond_pred.size(0) == valid.numel():
        filtered_pred = bond_pred[valid]
    return filtered_bonds, filtered_pred


def _slice_local_runtime_data(
    data: Any,
    mol_id: int,
    device: torch.device,
    lp_ckpt: Optional[str],
    skeleton_radius: float,
    quiet_skeleton: bool = True,
) -> Optional[Any]:
    if not hasattr(data, "batch") or data.batch is None:
        return None
    batch_cpu = data.batch.detach().cpu()
    mask = batch_cpu == int(mol_id)
    if int(mask.sum().item()) < 2:
        return None
    idx = torch.where(mask)[0]
    local_map = torch.full((int(data.z.size(0)),), -1, dtype=torch.long)
    local_map[idx] = torch.arange(idx.numel(), dtype=torch.long)
    src = data.edge_index[0].detach().cpu()
    dst = data.edge_index[1].detach().cpu()
    edge_mask = mask[src] & mask[dst]
    if int(edge_mask.sum().item()) == 0:
        return None
    local_edge_index = torch.stack(
        [local_map[src[edge_mask]], local_map[dst[edge_mask]]],
        dim=0,
    ).to(device)
    z_local = data.z[idx].to(device)
    pos_local = data.pos[idx].to(device)
    if quiet_skeleton:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                skel = build_skeleton_online(
                    z=z_local,
                    pos=pos_local,
                    device=device,
                    lp_ckpt=lp_ckpt,
                    skeleton_radius=float(skeleton_radius),
                )
    else:
        skel = build_skeleton_online(
            z=z_local,
            pos=pos_local,
            device=device,
            lp_ckpt=lp_ckpt,
            skeleton_radius=float(skeleton_radius),
        )
    skel.z = z_local
    skel.pos = pos_local
    skel.edge_index = local_edge_index
    skel.batch = torch.zeros(int(z_local.numel()), dtype=torch.long, device=device)
    return skel


def _iter_runtime_local_data(
    batch_data: Any,
    device: torch.device,
    lp_ckpt: Optional[str],
    skeleton_radius: float,
    quiet_skeleton: bool = True,
):
    if not hasattr(batch_data, "batch") or batch_data.batch is None:
        return
    mol_ids = batch_data.batch.detach().cpu().unique().tolist()
    for mol_id in mol_ids:
        local = _slice_local_runtime_data(
            batch_data,
            int(mol_id),
            device=device,
            lp_ckpt=lp_ckpt,
            skeleton_radius=skeleton_radius,
            quiet_skeleton=quiet_skeleton,
        )
        if local is not None:
            yield local


def _get_ep_prior(model: Any):
    if model is None:
        return None
    prior = getattr(model, "electron_prior", None)
    if prior is not None:
        return prior
    return getattr(model, "external_electron_prior", None)


def _set_ep_runtime_mode(model: Any, mode: str) -> Optional[str]:
    prior = _get_ep_prior(model)
    if prior is None or not hasattr(prior, "runtime_mode"):
        return None
    old = str(getattr(prior, "runtime_mode"))
    setattr(prior, "runtime_mode", str(mode))
    return old


def _restore_ep_runtime_modes(saved: Dict[Any, Optional[str]]) -> None:
    for model, old in saved.items():
        if old is not None:
            _set_ep_runtime_mode(model, old)


def _install_ep_cache_on_data(
    data: Any,
    nbo: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Tuple[bool, Any]]:
    prev: Dict[str, Tuple[bool, Any]] = {}
    if data is None or not isinstance(nbo, dict):
        return prev
    mapping = {
        "ep_atom_pred": nbo.get("atom_pred"),
        "ep_bond_pred": nbo.get("bond_pred"),
        "ep_interaction_pred": nbo.get("interaction_pred"),
    }
    for key, value in mapping.items():
        prev[key] = (hasattr(data, key), getattr(data, key, None))
        if isinstance(value, torch.Tensor):
            setattr(data, key, value.detach().to(device=device, non_blocking=True))
        elif key == "ep_interaction_pred":
            setattr(data, key, None)
    return prev


def _restore_data_attrs(data: Any, prev: Dict[str, Tuple[bool, Any]]) -> None:
    if data is None:
        return
    for key, (existed, value) in prev.items():
        if existed:
            setattr(data, key, value)
        elif hasattr(data, key):
            delattr(data, key)


def _forward_stats_branch(
    tag: str,
    model: Any,
    *,
    pos: torch.Tensor,
    z: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    data: Any,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if model is None:
        return None
    if tag == "hij":
        def _fwd_hij():
            with torch.enable_grad():
                return model(
                    pos=pos.to(device), z=z.to(device), batch=batch,
                    edge_index=edge_index.to(device), data=data,
                ).detach()
        return _run_with_ep_cache_guard(model, _fwd_hij)
    if tag == "dd":
        def _fwd_dd():
            with torch.enable_grad():
                return model(
                    pos=pos.to(device), z=z.to(device), batch=batch, data=data,
                ).detach()
        return _run_with_ep_cache_guard(model, _fwd_dd)
    if tag == "dp":
        pos_in = pos.to(device).requires_grad_(True)
        def _fwd_dp():
            with torch.enable_grad():
                return model(
                    pos=pos_in, z=z.to(device), batch=batch, data=data,
                ).detach()
        return _run_with_ep_cache_guard(model, _fwd_dp)
    raise ValueError(f"unknown stats branch: {tag}")


def _collect_batch_stats(
    pos: torch.Tensor,
    z: torch.Tensor,
    data: Any,
    model_hij, model_dd, model_dp,
    device: torch.device,
    accum: dict,
    reuse_ep_cache: bool = True,
) -> None:
    """Run one training batch through EP models and accumulate stats."""
    batch = data.batch.to(device) if hasattr(data, "batch") and data.batch is not None else torch.zeros(len(z), dtype=torch.long, device=device)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        edge_index = radius_graph(x=pos.to(device), r=5.0, batch=batch, max_num_neighbors=1000)

    # --- Run models ---
    outputs = {"hij": None, "dd": None, "dp": None}
    models = {"hij": model_hij, "dd": model_dd, "dp": model_dp}
    source_tag = next((tag for tag in ("hij", "dd", "dp") if models[tag] is not None), None)
    nbo = None
    saved_modes: Dict[Any, Optional[str]] = {}
    saved_attrs: Dict[str, Tuple[bool, Any]] = {}

    try:
        if source_tag is not None:
            outputs[source_tag] = _forward_stats_branch(
                source_tag,
                models[source_tag],
                pos=pos,
                z=z,
                batch=batch,
                edge_index=edge_index,
                data=data,
                device=device,
            )
            nbo = _extract_nbo_from_model(models[source_tag])
            if bool(reuse_ep_cache) and nbo is not None:
                saved_attrs = _install_ep_cache_on_data(data, nbo, device)
                for tag, model in models.items():
                    if model is not None and tag != source_tag:
                        saved_modes[model] = _set_ep_runtime_mode(model, "cached")

        for tag in ("hij", "dd", "dp"):
            if tag == source_tag or models[tag] is None:
                continue
            outputs[tag] = _forward_stats_branch(
                tag,
                models[tag],
                pos=pos,
                z=z,
                batch=batch,
                edge_index=edge_index,
                data=data,
                device=device,
            )
    finally:
        _restore_ep_runtime_modes(saved_modes)
        _restore_data_attrs(data, saved_attrs)

    hij = outputs["hij"]
    dd = outputs["dd"]
    dp = outputs["dp"]

    # --- Extract NBO features from any EP-active model ---
    if nbo is None:
        for m in (model_hij, model_dd, model_dp):
            if m is not None:
                nbo = _extract_nbo_from_model(m)
                if nbo is not None:
                    break
    if nbo is None:
        return

    atom_pred = nbo.get("atom_pred")
    bond_pred = nbo.get("bond_pred")
    atom_bond_local = nbo.get("atom_bond_local")
    pos_dev = pos.to(device)
    num_nodes = int(z.shape[0])

    # Move all NBO tensors explicitly to device (training loader data is CPU)
    ei_dev = edge_index.to(device)
    bond_pred_dev = bond_pred.to(device) if bond_pred is not None else None
    atom_bond_local_dev = atom_bond_local.to(device) if atom_bond_local is not None else None
    atom_pred_dev = atom_pred.to(device) if atom_pred is not None else None
    atom_bond_local_dev, bond_pred_dev = _drop_cross_batch_bonds(
        atom_bond_local_dev, bond_pred_dev, batch, num_nodes,
    )
    accum.setdefault("__hij_seen_elements__", set()).update(
        int(x) for x in z.detach().cpu().view(-1).tolist()
    )
    enk_context = _collect_enk_context(
        [("hij", model_hij), ("dd", model_dd), ("dp", model_dp)],
        num_nodes,
        device,
    )
    if enk_context:
        atom_ood = enk_context.get("atom_ood")
        atom_k_mean = enk_context.get("atom_k_mean")
        atom_k_min = enk_context.get("atom_k_min")
        branch_dis = enk_context.get("branch_disagreement")
        if isinstance(atom_ood, torch.Tensor):
            _running_stats(accum, "enk_atom_ood", atom_ood)
        if isinstance(atom_k_mean, torch.Tensor):
            _running_stats(accum, "enk_atom_k_mean", atom_k_mean)
        if isinstance(atom_k_min, torch.Tensor):
            _running_stats(accum, "enk_atom_k_min", atom_k_min)
        if isinstance(branch_dis, torch.Tensor):
            _running_stats(accum, "enk_branch_disagreement", branch_dis)

    # --- hij stats: log|Hij_along| + log(occ) ---
    if bond_pred_dev is not None and atom_bond_local_dev is not None:
        if bond_pred_dev.numel() > 0 and atom_bond_local_dev.numel() > 0:
            classifier = NBOGuidedCalibrator(alpha_hij=0.0, train_stats={})
            bsrc = atom_bond_local_dev[0].long()
            bdst = atom_bond_local_dev[1].long()
            class_names, *_ = classifier._hij_bond_context(
                z.to(device), atom_bond_local_dev, bsrc, bdst, pos=pos_dev,
            )
            env_signatures = classifier._hij_environment_signatures(
                z.to(device), atom_bond_local_dev, bsrc, bdst, class_names,
            )
            accum.setdefault("__hij_seen_env_signatures__", set()).update(env_signatures)

    if hij is not None and bond_pred_dev is not None and atom_bond_local_dev is not None:
        if bond_pred_dev.numel() > 0 and atom_bond_local_dev.numel() > 0:
            bond_ids, bond_mask = _match_undirected_edges_simple(
                ei_dev, atom_bond_local_dev, num_nodes,
            )
            if bond_mask.any():
                hij_mat = hij.view(-1, 3, 3)
                edge_src = ei_dev[0, bond_mask].long()
                edge_dst = ei_dev[1, bond_mask].long()
                r_vec = pos_dev[edge_dst] - pos_dev[edge_src]
                r_norm = r_vec.norm(dim=-1, keepdim=True).clamp(min=0.01)
                r_hat = r_vec / r_norm
                hij_bond_block = hij_mat[bond_mask]
                hij_long = torch.einsum('bi,bij,bj->b', r_hat, hij_bond_block, r_hat)
                hij_strength = hij_long.abs().clamp(min=1e-10)
                _running_stats(accum, "log_hij_long", torch.log(hij_strength))

                occ_col = _bond_occupancy_col(int(bond_pred_dev.size(1)))
                if occ_col is not None:
                    bd_occ = bond_pred_dev[:, occ_col].clamp(min=0.01)
                    matched_occ = bd_occ[bond_ids[bond_mask]].clamp(min=0.01)
                    _running_stats(accum, "log_occ", torch.log(matched_occ))

                    # Bond-class baselines let inference avoid mixing X-H,
                    # carbonyl/amide, and generic backbone statistics.
                    classifier = NBOGuidedCalibrator(alpha_hij=0.0, train_stats={})
                    class_names, *_ = classifier._hij_bond_context(
                        z.to(device), atom_bond_local_dev, edge_src, edge_dst, pos=pos_dev,
                    )
                    env_signatures = classifier._hij_environment_signatures(
                        z.to(device), atom_bond_local_dev, edge_src, edge_dst, class_names,
                    )
                    accum.setdefault("__hij_seen_env_signatures__", set()).update(env_signatures)
                    log_hij_class = torch.log(hij_strength)
                    log_occ_class = torch.log(matched_occ)
                    for cls in sorted(set(class_names)):
                        cls_mask = torch.tensor(
                            [name == cls for name in class_names],
                            dtype=torch.bool,
                            device=device,
                        )
                        _running_hij_class_stats(accum, cls, "log_hij_long", log_hij_class[cls_mask])
                        _running_hij_class_stats(accum, cls, "log_occ", log_occ_class[cls_mask])

    # --- dd stats: dd_trace + npa_charge ---
    if dd is not None and atom_pred_dev is not None:
        dd_trace = dd.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _running_stats(accum, "dd_trace", dd_trace)
        npa_charge = atom_pred_dev[:, 7]
        _running_stats(accum, "npa_charge", npa_charge)

    # --- dp stats: dp_norm + delocal ---
    if dp is not None and atom_pred_dev is not None:
        dp_norm_val = dp.norm(dim=-1).mean(dim=-1)
        _running_stats(accum, "dp_norm", dp_norm_val)

        delocal_proxy = atom_pred_dev[:, 1].clamp(min=0.01)
        if bond_pred_dev is not None and atom_bond_local_dev is not None:
            if bond_pred_dev.numel() > 0 and atom_bond_local_dev.numel() > 0:
                occ_col = _bond_occupancy_col(int(bond_pred_dev.size(1)))
                if occ_col is not None:
                    bd_occ = bond_pred_dev[:, occ_col].clamp(min=0.0)
                    src_b = atom_bond_local_dev[0].long()
                    dst_b = atom_bond_local_dev[1].long()
                    bond_order_sum = torch.zeros(atom_pred_dev.size(0), device=device)
                    bond_order_sum.scatter_add_(0, src_b, bd_occ)
                    bond_order_sum.scatter_add_(0, dst_b, bd_occ)
                    delocal_proxy = (delocal_proxy + bond_order_sum).clamp(min=0.01)
        _running_stats(accum, "delocal", delocal_proxy)


# --- Main ---

def _signature_prefix(sig: str) -> str:
    return str(sig).split("|", 1)[0]


def _top_counter(counter: Counter, k: int = 20) -> List[Dict[str, Any]]:
    total = max(int(sum(counter.values())), 1)
    return [
        {"key": str(key), "n": int(value), "fraction": float(value) / total}
        for key, value in counter.most_common(int(k))
    ]


def _summarize_signatures(signatures: List[str], seen_env: set, top_k: int = 20) -> Dict[str, Any]:
    total = int(len(signatures))
    counts = Counter(str(x) for x in signatures)
    class_counts = Counter(_signature_prefix(x) for x in signatures)
    unknown = [str(x) for x in signatures if seen_env and str(x) not in seen_env]
    unknown_counts = Counter(unknown)
    return {
        "n": total,
        "unique": int(len(counts)),
        "known": int(total - len(unknown)) if seen_env else 0,
        "unknown": int(len(unknown)) if seen_env else 0,
        "unknown_pct": float(len(unknown)) / max(total, 1) * 100.0 if seen_env else None,
        "top_classes": _top_counter(class_counts, top_k),
        "top_signatures": _top_counter(counts, top_k),
        "top_unknown_signatures": _top_counter(unknown_counts, top_k),
    }


def _tensor_brief(x: Any) -> Dict[str, Any]:
    if not isinstance(x, torch.Tensor):
        return {"type": type(x).__name__}
    out: Dict[str, Any] = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "numel": int(x.numel()),
    }
    if x.numel() > 0 and torch.is_floating_point(x):
        v = x.detach().float().view(-1)
        finite = torch.isfinite(v)
        if bool(finite.any().item()):
            vf = v[finite]
            out.update({"min": float(vf.min().item()), "max": float(vf.max().item()), "mean": float(vf.mean().item())})
    elif x.numel() > 0:
        v = x.detach().view(-1)
        out.update({
            "min": int(v.min().item()),
            "max": int(v.max().item()),
            "sample": [int(a) for a in v[:12].detach().cpu().tolist()],
        })
    return out


def _batch_data_brief(data: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": type(data).__name__}
    try:
        out["keys"] = list(data.keys())
    except Exception:
        out["keys"] = []
    for key in ("z", "pos", "batch", "edge_index", "atom_bond_index", "atom_bond_local", "y_hij"):
        if hasattr(data, key):
            out[key] = _tensor_brief(getattr(data, key))
    return out


def _run_ep_once_for_nbo(data: Any, model_hij, model_dd, model_dp, device: torch.device):
    z = data.z.to(device)
    pos = data.pos.to(device)
    batch = data.batch.to(device) if hasattr(data, "batch") and data.batch is not None else torch.zeros(len(z), dtype=torch.long, device=device)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        edge_index = radius_graph(x=pos, r=5.0, batch=batch, max_num_neighbors=1000)
    edge_index = edge_index.to(device)

    if model_hij is not None:
        def _fwd_hij():
            with torch.enable_grad():
                return model_hij(pos=pos, z=z, batch=batch, edge_index=edge_index, data=data).detach()
        _ = _run_with_ep_cache_guard(model_hij, _fwd_hij)
        nbo = _extract_nbo_from_model(model_hij)
        if nbo is not None:
            return nbo, edge_index

    if model_dd is not None:
        def _fwd_dd():
            with torch.enable_grad():
                return model_dd(pos=pos, z=z, batch=batch, data=data).detach()
        _ = _run_with_ep_cache_guard(model_dd, _fwd_dd)
        nbo = _extract_nbo_from_model(model_dd)
        if nbo is not None:
            return nbo, edge_index

    if model_dp is not None:
        pos_in = pos.detach().clone().requires_grad_(True)
        def _fwd_dp():
            with torch.enable_grad():
                return model_dp(pos=pos_in, z=z, batch=batch, data=data).detach()
        _ = _run_with_ep_cache_guard(model_dp, _fwd_dp)
        nbo = _extract_nbo_from_model(model_dp)
        if nbo is not None:
            return nbo, edge_index

    return None, edge_index


def _collect_signature_views_from_batch(
    data: Any,
    model_hij,
    model_dd,
    model_dp,
    device: torch.device,
    train_stats: Dict[str, Any],
    top_k: int = 20,
) -> Dict[str, Any]:
    z = data.z.to(device)
    pos = data.pos.to(device)
    batch = data.batch.to(device) if hasattr(data, "batch") and data.batch is not None else torch.zeros(len(z), dtype=torch.long, device=device)
    num_nodes = int(z.numel())
    nbo, edge_index = _run_ep_once_for_nbo(data, model_hij, model_dd, model_dp, device)
    out: Dict[str, Any] = {"data": _batch_data_brief(data), "nbo_available": nbo is not None}
    if nbo is None:
        return out

    atom_bond_local = nbo.get("atom_bond_local")
    bond_pred = nbo.get("bond_pred")
    out["nbo"] = {"atom_bond_local": _tensor_brief(atom_bond_local), "bond_pred": _tensor_brief(bond_pred)}
    if atom_bond_local is None or atom_bond_local.numel() == 0:
        return out

    atom_bond_local = atom_bond_local.to(device)
    bond_pred = bond_pred.to(device) if bond_pred is not None else None
    atom_bond_local, bond_pred = _drop_cross_batch_bonds(atom_bond_local, bond_pred, batch, num_nodes)
    cal = NBOGuidedCalibrator(alpha_hij=0.0, train_stats=train_stats)
    seen_env = cal._effective_train_env_signatures()

    views: Dict[str, Any] = {}
    raw_views: Dict[str, List[str]] = {}
    if atom_bond_local is not None and atom_bond_local.numel() > 0:
        src = atom_bond_local[0].long()
        dst = atom_bond_local[1].long()
        class_names, *_ = cal._hij_bond_context(z, atom_bond_local, src, dst, pos=pos)
        sigs = cal._hij_environment_signatures(z, atom_bond_local, src, dst, class_names)
        views["nbo_atom_bond_local"] = _summarize_signatures(sigs, seen_env, top_k)
        raw_views["nbo_atom_bond_local"] = sigs

    bond_ids, bond_mask = _match_undirected_edges_simple(edge_index, atom_bond_local, num_nodes)
    views["edge_match"] = {
        "edge_count": int(edge_index.size(1)),
        "matched_edge_count": int(bond_mask.sum().item()),
        "atom_bond_count": int(atom_bond_local.size(1)),
    }
    if bool(bond_mask.any().item()):
        src = edge_index[0, bond_mask].long()
        dst = edge_index[1, bond_mask].long()
        class_names, *_ = cal._hij_bond_context(z, atom_bond_local, src, dst, pos=pos)
        sigs = cal._hij_environment_signatures(z, atom_bond_local, src, dst, class_names)
        views["matched_hij_edges"] = _summarize_signatures(sigs, seen_env, top_k)
        raw_views["matched_hij_edges"] = sigs
        if bond_ids.numel() == edge_index.size(1):
            views["edge_match"]["unique_matched_bond_ids"] = int(torch.unique(bond_ids[bond_mask]).numel())
    out["signature_views"] = views
    out["_raw_signature_views"] = raw_views
    return out


def _run_signature_self_check(
    loader,
    model_hij,
    model_dd,
    model_dp,
    device: torch.device,
    train_stats: Dict[str, Any],
    max_batches: int,
    top_k: int = 20,
) -> Dict[str, Any]:
    cal = NBOGuidedCalibrator(alpha_hij=0.0, train_stats=train_stats)
    seen_env = cal._effective_train_env_signatures()
    result: Dict[str, Any] = {
        "seen_env_signature_count": int(len(seen_env)),
        "seen_bond_class_count": int(len(cal._effective_train_bond_classes())),
        "batches": [],
        "aggregate": {},
    }
    aggregate = {"nbo_atom_bond_local": Counter(), "matched_hij_edges": Counter()}
    max_batches = min(max(int(max_batches), 0), len(loader))
    for batch_idx, data in enumerate(loader):
        if batch_idx >= max_batches:
            break
        try:
            diag = _collect_signature_views_from_batch(data, model_hij, model_dd, model_dp, device, train_stats, top_k=top_k)
            raw_views = diag.pop("_raw_signature_views", {})
            diag["batch_idx"] = int(batch_idx)
            result["batches"].append(diag)
            for view_name in aggregate:
                aggregate[view_name].update(str(sig) for sig in raw_views.get(view_name, []))
        except Exception as exc:
            result["batches"].append({"batch_idx": int(batch_idx), "error": repr(exc)})

    for view_name, counts in aggregate.items():
        sigs: List[str] = []
        for sig, n in counts.items():
            sigs.extend([sig] * int(n))
        result["aggregate"][view_name] = _summarize_signatures(sigs, seen_env, top_k)
    return result


def _run_runtime_signature_self_check(
    loader,
    model_hij,
    model_dd,
    model_dp,
    device: torch.device,
    train_stats: Dict[str, Any],
    max_batches: int,
    top_k: int = 20,
    lp_ckpt: Optional[str] = None,
    skeleton_radius: float = 5.0,
    max_mols: int = 0,
    quiet_skeleton: bool = True,
) -> Dict[str, Any]:
    cal = NBOGuidedCalibrator(alpha_hij=0.0, train_stats=train_stats)
    seen_env = cal._effective_train_env_signatures()
    result: Dict[str, Any] = {
        "runtime_aligned": True,
        "seen_env_signature_count": int(len(seen_env)),
        "seen_bond_class_count": int(len(cal._effective_train_bond_classes())),
        "batches": [],
        "aggregate": {},
    }
    aggregate = {"nbo_atom_bond_local": Counter(), "matched_hij_edges": Counter()}
    n_mols = 0
    max_batches = min(max(int(max_batches), 0), len(loader))
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= max_batches:
            break
        for local in _iter_runtime_local_data(batch_data, device, lp_ckpt, skeleton_radius, quiet_skeleton=quiet_skeleton):
            if max_mols > 0 and n_mols >= int(max_mols):
                break
            try:
                diag = _collect_signature_views_from_batch(local, model_hij, model_dd, model_dp, device, train_stats, top_k=top_k)
                raw_views = diag.pop("_raw_signature_views", {})
                diag["batch_idx"] = int(batch_idx)
                diag["mol_idx"] = int(n_mols)
                result["batches"].append(diag)
                for view_name in aggregate:
                    aggregate[view_name].update(str(sig) for sig in raw_views.get(view_name, []))
                n_mols += 1
            except Exception as exc:
                result["batches"].append({"batch_idx": int(batch_idx), "mol_idx": int(n_mols), "error": repr(exc)})
        if max_mols > 0 and n_mols >= int(max_mols):
            break

    result["n_molecules_checked"] = int(n_mols)
    for view_name, counts in aggregate.items():
        sigs: List[str] = []
        for sig, n in counts.items():
            sigs.extend([sig] * int(n))
        result["aggregate"][view_name] = _summarize_signatures(sigs, seen_env, top_k)
    return result


def _str_to_bool(val):
    """Parse a string to boolean for argparse."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes", "t")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NBO-GSC v2 train_stats from training data pipeline",
    )

    # --- Training data config (same as train_senk.py) ---
    parser.add_argument("--dataset", type=str, default="qme14s_opt186102")
    parser.add_argument("--pt_path", type=str, default=train_mod.DEFAULT_QM9S_PT)
    parser.add_argument("--skeleton_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--max_neighbors", type=int, default=1000)
    parser.add_argument("--split_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center", type=_str_to_bool, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_cache", type=_str_to_bool, default=False)
    parser.add_argument("--auto_build_cache", type=_str_to_bool, default=True)
    parser.add_argument("--require_cache", type=_str_to_bool, default=True)
    parser.add_argument("--cache_format", type=str, default="pickle")
    parser.add_argument("--cache_shard_size", type=int, default=5000)
    parser.add_argument("--cache_shard_keep", type=int, default=2)
    parser.add_argument("--cache_dir", type=str,
                        default=str(Path.home() / ".cache" / "senk_vib"))
    parser.add_argument("--pin_memory", type=_str_to_bool, default=True)
    parser.add_argument("--persistent_workers", type=_str_to_bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mp_context", type=str, default="spawn")

    # --- EP config ---
    parser.add_argument("--electron_prior_ckpt", type=str, required=True)
    parser.add_argument("--electron_prior_stats", type=str, required=True)
    parser.add_argument("--electron_prior_mode", type=str, default="qcmol")
    parser.add_argument("--electron_prior_freeze", type=_str_to_bool, default=True)
    parser.add_argument("--electron_prior_scale", type=float, default=5e-3)
    parser.add_argument("--electron_prior_runtime_mode", type=str, default="full")
    parser.add_argument("--electron_prior_use_aux", type=_str_to_bool, default=True)
    parser.add_argument("--electron_prior_heads", type=int, default=4)

    # --- Model paths ---
    # Task determines which branches to load (default: all three)
    parser.add_argument("--task", type=str, default="hij",
                        help="Which branch loading determines EP setup (hij/dd/dp)")
    parser.add_argument("--hij_ckpt", type=str, default="")
    parser.add_argument("--hij_mode", type=str, default="equiformer_v2_enk_ep",
                        choices=_ALL_MODES)
    parser.add_argument("--dd_ckpt", type=str, default="")
    parser.add_argument("--dd_mode", type=str, default="equiformer_v2_enk_ep",
                        choices=_ALL_MODES)
    parser.add_argument("--dp_ckpt", type=str, default="")
    parser.add_argument("--dp_mode", type=str, default="equiformer_v2_enk_ep",
                        choices=_ALL_MODES)

    # --- Model architecture (from training) ---
    parser.add_argument("--hidden_nf", type=int, default=128)
    parser.add_argument("--num_basis", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--v2_model_name", type=str, default="equiformer_v2_l4_m2")
    parser.add_argument("--v2_grid_resolution", type=int, default=14)
    parser.add_argument("--v2_max_neighbors", type=int, default=1000)
    parser.add_argument("--v2_num_gaussians", type=int, default=50)
    parser.add_argument("--v2_use_gate_act", type=_str_to_bool, default=False)
    parser.add_argument("--v2_use_grid_mlp", type=_str_to_bool, default=False)
    parser.add_argument("--enk_enabled", type=_str_to_bool, default=False)
    parser.add_argument("--enk_init_r_bias", type=float, default=-1.0)
    parser.add_argument("--enk_init_q_bias", type=float, default=1.0)

    # --- Run control ---
    parser.add_argument("--num_batches", type=int, default=100,
                        help="Number of training batches to process")
    parser.add_argument("--max_mols", type=int, default=0,
                        help="Optional molecule cap, mainly for --stats_runtime_aligned short tests")
    parser.add_argument("--output", type=str, default="nbo_train_stats.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stats_runtime_aligned", action="store_true",
                        help="Generate stats with the same per-molecule online skeleton used by inference/GSC")
    parser.add_argument("--reuse_ep_cache", type=_str_to_bool, default=True, choices=[True, False],
                        help="Reuse NBO predictor outputs across Hij/DD/DP forwards on the same molecule or batch")
    parser.add_argument("--lp_ckpt", type=str, default=str(ROOT / "tools" / "lp_pred_model.ckpt"),
                        help="Lone-pair model used by online skeleton generation in --stats_runtime_aligned")
    parser.add_argument("--disable_lp", action="store_true",
                        help="Disable LP model in --stats_runtime_aligned online skeleton generation")
    parser.add_argument("--runtime_skeleton_radius", type=float, default=5.0,
                        help="Online skeleton radius used by --stats_runtime_aligned")
    parser.add_argument("--quiet_runtime_skeleton", type=_str_to_bool, default=True, choices=[True, False],
                        help="Suppress per-molecule online skeleton logs during runtime-aligned stats")
    parser.add_argument("--signature_self_check", action="store_true",
                        help="After saving stats, re-run a few batches and verify exact Hij env signature self-coverage")
    parser.add_argument("--signature_check_batches", type=int, default=2,
                        help="Number of train-loader batches used by --signature_self_check")
    parser.add_argument("--signature_check_top_k", type=int, default=20,
                        help="Top signatures/classes written in the self-check JSON")
    parser.add_argument("--signature_check_output", type=str, default="",
                        help="Optional output JSON path for --signature_self_check")
    args = parser.parse_args()

    # --- Auto-set paths (mimics train_senk.py) ---
    if args.dataset == "qme14s_opt186102":
        if args.pt_path == train_mod.DEFAULT_QM9S_PT:
            args.pt_path = train_mod.DEFAULT_QME14S_OPT186102_PT
        if not args.skeleton_path:
            if "qme14s" in str(args.pt_path).lower():
                args.skeleton_path = str(train_mod.DEFAULT_QME14S_SKELETON)
        train_mod._override_edge_params_for_qme14s(args)

    # Expand relative paths vs ROOT
    for attr in ("pt_path", "skeleton_path", "cache_dir", "lp_ckpt"):
        val = getattr(args, attr, "")
        if isinstance(val, str) and val and not Path(val).is_absolute():
            setattr(args, attr, str(ROOT / val))

    print(f"Dataset: {args.dataset}")
    print(f"PT path: {args.pt_path}")
    print(f"Skeleton: {args.skeleton_path}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Build training data loader ---
    print("\nBuilding data loaders...")
    loaders, _data_stats = _build_loaders(args)
    train_loader = loaders["train"]
    num_batches_to_process = len(train_loader) if args.num_batches <= 0 else min(args.num_batches, len(train_loader))
    print(f"Train batches: {len(train_loader)} (using {num_batches_to_process})")

    # --- Load EP models ---
    print("\nLoading models...")
    # Auto-detect architecture params from checkpoint that _detect_arch_from_ckpt
    # doesn't cover (num_gaussians, use_gate_act, use_grid_mlp).
    _ckpt_paths = [p for p in (args.hij_ckpt, args.dd_ckpt, args.dp_ckpt) if p]
    if _ckpt_paths:
        from v2_spectra_infer import _load_raw_state
        _first_ckpt = str(Path(ROOT / _ckpt_paths[0]).resolve()) \
            if not Path(_ckpt_paths[0]).is_absolute() else _ckpt_paths[0]
        try:
            _peek = _load_raw_state(_first_ckpt, torch.device("cpu"))
        except Exception:
            _peek = {}
        _offset_key = "backbone.distance_expansion.offset"
        if _offset_key in _peek:
            _ng = int(_peek[_offset_key].shape[0])
            args.v2_num_gaussians = _ng
            logger.info("  [auto] num_gaussians=%s (from checkpoint)", _ng)
        _gate_key = "backbone.blocks.0.ffn.gating_linear.weight"
        if _gate_key in _peek:
            _gw = _peek[_gate_key]
            _has_gate = (_gw.ndim == 2 and _gw.shape[0] != _gw.shape[1])
            args.v2_use_gate_act = _has_gate
            if _has_gate:
                logger.info("  [auto] use_gate_act=True (from checkpoint gating shape %s)", tuple(_gw.shape))
        _grid_key = "backbone.blocks.0.ffn.SO3_grid.0.0.to_grid_mat"
        if _grid_key in _peek:
            args.v2_use_grid_mlp = True
            logger.info("  [auto] use_grid_mlp=True (from checkpoint)")

    model_hij = _load_multitask("hij", args.hij_ckpt, args.hij_mode, device, args) if args.hij_ckpt else None
    model_dd = _load_multitask("dedipole", args.dd_ckpt, args.dd_mode, device, args) if args.dd_ckpt else None
    model_dp = _load_depolar(args.dp_ckpt, args.dp_mode, device, args) if args.dp_ckpt else None

    for tag, m in [("hij", model_hij), ("dd", model_dd), ("dp", model_dp)]:
        if m is not None:
            print(f"  {tag}: EP active={_is_ep_runtime_active(m)}")
        else:
            print(f"  {tag}: skipped (no checkpoint)")

    if model_hij is None and model_dd is None and model_dp is None:
        logger.error("at least one of --hij_ckpt / --dd_ckpt / --dp_ckpt required")
        sys.exit(1)

    # --- Collect stats from training batches ---
    print(f"\nProcessing {num_batches_to_process} batches...")
    accum = {}
    n_runtime_mols = 0
    runtime_lp_ckpt = None if args.disable_lp else args.lp_ckpt
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= num_batches_to_process:
            break
        if args.max_mols > 0 and n_runtime_mols >= int(args.max_mols):
            break
        if batch_idx % 10 == 0:
            n_samples = sum(
                a["n"] for a in accum.values()
                if isinstance(a, dict) and "n" in a
            )
            print(f"  [{batch_idx:4d}/{num_batches_to_process}] samples: {n_samples:>10d}")

        try:
            if args.stats_runtime_aligned:
                for local_data in _iter_runtime_local_data(
                    batch_data,
                    device=device,
                    lp_ckpt=runtime_lp_ckpt,
                    skeleton_radius=float(args.runtime_skeleton_radius),
                    quiet_skeleton=bool(args.quiet_runtime_skeleton),
                ):
                    if args.max_mols > 0 and n_runtime_mols >= int(args.max_mols):
                        break
                    _collect_batch_stats(
                        pos=local_data.pos, z=local_data.z, data=local_data,
                        model_hij=model_hij, model_dd=model_dd, model_dp=model_dp,
                        device=device, accum=accum,
                        reuse_ep_cache=bool(args.reuse_ep_cache),
                    )
                    n_runtime_mols += 1
            else:
                pos = getattr(batch_data, "pos", None)
                z = getattr(batch_data, "z", None)
                if pos is None or z is None:
                    continue
                _collect_batch_stats(
                    pos=pos, z=z, data=batch_data,
                    model_hij=model_hij, model_dd=model_dd, model_dp=model_dp,
                    device=device, accum=accum,
                    reuse_ep_cache=bool(args.reuse_ep_cache),
                )
        except Exception as e:
            logger.error("batch %s: %s", batch_idx, e)
            continue

    # --- Finalize and save ---
    stats = _finalize_stats(accum)
    print(f"\nComputed {len(stats)} statistics:")
    for key in sorted(stats.keys()):
        value = stats[key]
        if isinstance(value, (int, float)):
            print(f"  {key:30s} = {value:.6f}")
        elif key == "hij_bond_class_stats" and isinstance(value, dict):
            print(f"  {key:30s} = {len(value)} classes")
        elif key == "hij_seen_env_signatures" and isinstance(value, (list, tuple, set)):
            print(f"  {key:30s} = {len(value)} signatures")
        else:
            print(f"  {key:30s} = {value}")

    out_path = Path(args.output)
    torch.save(stats, out_path)
    print(f"\nSaved to: {out_path.resolve()}")
    print(f"\nInference usage:  --nbo_train_stats {out_path.resolve()}")

    if args.signature_self_check:
        print("\nRunning signature self-check on the train loader...")
        # The train loader is shuffled. Resetting the RNG makes a short
        # self-check inspect the same leading batches used to build short stats.
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        if args.stats_runtime_aligned:
            self_check = _run_runtime_signature_self_check(
                train_loader,
                model_hij=model_hij,
                model_dd=model_dd,
                model_dp=model_dp,
                device=device,
                train_stats=stats,
                max_batches=int(args.signature_check_batches),
                top_k=int(args.signature_check_top_k),
                lp_ckpt=runtime_lp_ckpt,
                skeleton_radius=float(args.runtime_skeleton_radius),
                max_mols=int(args.max_mols),
                quiet_skeleton=bool(args.quiet_runtime_skeleton),
            )
        else:
            self_check = _run_signature_self_check(
                train_loader,
                model_hij=model_hij,
                model_dd=model_dd,
                model_dp=model_dp,
                device=device,
                train_stats=stats,
                max_batches=int(args.signature_check_batches),
                top_k=int(args.signature_check_top_k),
            )
        check_path = Path(args.signature_check_output) if args.signature_check_output else out_path.with_suffix(out_path.suffix + ".signature_self_check.json")
        check_path.write_text(
            json.dumps(self_check, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Signature self-check saved to: {check_path.resolve()}")
        for view_name, summary in self_check.get("aggregate", {}).items():
            print(
                f"  {view_name:22s} n={summary.get('n', 0)} "
                f"unique={summary.get('unique', 0)} "
                f"unknown={summary.get('unknown_pct', None)}%"
            )


if __name__ == "__main__":
    main()
