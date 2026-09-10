"""Chemically independent four-level training-support estimation.

This module intentionally does *not* predict local response error.  It answers
one narrower question:

    how strongly is this local chemical environment represented by independent
    training molecules?

The raw score is monotone in every support count and is diagnostic until a
chemical-group-holdout probability artifact is supplied.  Response anomalies,
ENK state, DFT error and EP benefit are prohibited from the support artifact.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch


SUPPORT_SCHEMA_VERSION = 2
SUPPORT_COUNT_UNIT = "unique_molecule"

SUPPORT_FEATURES = (
    "p_support_raw",
    "support_exact_seen",
    "log_exact_molecule_count",
    "log_motif_molecule_count",
    "log_class_molecule_count",
    "log_element_pair_molecule_count",
    "support_backoff_depth",
)

_COUNT_KEYS = {
    "exact": "chemical_support_exact_molecule_counts",
    "motif": "chemical_support_motif_molecule_counts",
    "class": "chemical_support_class_molecule_counts",
    "pair": "chemical_support_element_pair_molecule_counts",
}

# Coarser levels are useful fallbacks, but cannot substitute for an exact local
# match with full confidence.  These values shape only the raw diagnostic; the
# final probability is calibrated on chemical-group holdouts.
DEFAULT_MIDPOINTS = {"exact": 4.0, "motif": 12.0, "class": 32.0, "pair": 64.0}
DEFAULT_FALLBACK_DISCOUNTS = {"motif": 0.65, "class": 0.35, "pair": 0.15}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def element_pair_key(za: int, zb: int) -> str:
    lo, hi = sorted((int(za), int(zb)))
    return f"z{lo}-z{hi}"


def _count_map(stats: Mapping[str, Any], key: str) -> Dict[str, float]:
    value = stats.get(key)
    if not isinstance(value, Mapping):
        return {}
    return {str(k): max(_finite_float(v), 0.0) for k, v in value.items()}


def _tensor_from_keys(
    keys: Sequence[str],
    counts: Mapping[str, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        [max(_finite_float(counts.get(str(key), 0.0)), 0.0) for key in keys],
        device=device,
        dtype=dtype,
    )


def _side_motif(z_cpu: torch.Tensor, neighbours: Sequence[Sequence[int]], center: int, other: int) -> str:
    values = [int(z_cpu[idx].item()) for idx in neighbours[int(center)] if int(idx) != int(other)]
    degree = len(neighbours[int(center)])
    h_count = sum(value == 1 for value in values)
    hetero_count = sum(value not in {1, 6} for value in values)
    heavy_count = sum(value != 1 for value in values)
    return f"z{int(z_cpu[int(center)].item())}:d{degree}:h{h_count}:x{hetero_count}:v{heavy_count}"


def environment_motif_keys(
    z: torch.Tensor,
    atom_bond_local: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    class_names: Sequence[str],
) -> List[str]:
    """Return a level distinct from both exact signatures and bond classes.

    The motif retains endpoint coordination and donor/acceptor-like composition
    counts, while discarding the exact neighbour-element multiset.  Endpoints
    are canonicalised so both directions of one bond share the same key.
    """
    z_cpu = z.detach().cpu().long()
    bonds = atom_bond_local.detach().cpu().long()
    neighbours: List[List[int]] = [[] for _ in range(int(z_cpu.numel()))]
    if bonds.numel():
        for left, right in bonds.t().tolist():
            if 0 <= int(left) < len(neighbours) and 0 <= int(right) < len(neighbours):
                neighbours[int(left)].append(int(right))
                neighbours[int(right)].append(int(left))

    motifs: List[str] = []
    for cls, src, dst in zip(
        class_names,
        edge_src.detach().cpu().long().tolist(),
        edge_dst.detach().cpu().long().tolist(),
    ):
        left = _side_motif(z_cpu, neighbours, int(src), int(dst))
        right = _side_motif(z_cpu, neighbours, int(dst), int(src))
        if right < left:
            left, right = right, left
        motifs.append(f"{cls}|motif:{left}|{right}")
    return motifs


def _support_confidence(count: torch.Tensor, midpoint: float) -> torch.Tensor:
    """Monotone support confidence in [0, 1]."""
    midpoint = max(float(midpoint), 1e-6)
    count = count.clamp(min=0.0)
    return count / (count + midpoint)


def support_risk_from_counts(
    exact_count: torch.Tensor,
    motif_count: torch.Tensor,
    class_count: torch.Tensor,
    pair_count: torch.Tensor,
    *,
    midpoints: Optional[Mapping[str, float]] = None,
    discounts: Optional[Mapping[str, float]] = None,
) -> torch.Tensor:
    """Compute a strictly count-monotone hierarchical support-risk score.

    Exact evidence is trusted directly.  Each coarser fallback is discounted so
    an abundant bond class cannot make an unseen exact environment look fully
    supported.  Increasing any one count can never increase risk.
    """
    mid = dict(DEFAULT_MIDPOINTS)
    if midpoints:
        mid.update({str(k): float(v) for k, v in midpoints.items()})
    rho = dict(DEFAULT_FALLBACK_DISCOUNTS)
    if discounts:
        rho.update({str(k): float(v) for k, v in discounts.items()})
    rho = {key: min(max(float(value), 0.0), 1.0) for key, value in rho.items()}

    s_exact = _support_confidence(exact_count, mid["exact"])
    s_motif = _support_confidence(motif_count, mid["motif"])
    s_class = _support_confidence(class_count, mid["class"])
    s_pair = _support_confidence(pair_count, mid["pair"])

    pair_path = rho["pair"] * s_pair
    class_path = rho["class"] * (s_class + (1.0 - s_class) * pair_path)
    motif_path = rho["motif"] * (s_motif + (1.0 - s_motif) * class_path)
    supported = s_exact + (1.0 - s_exact) * motif_path
    return (1.0 - supported).clamp(0.0, 1.0)


def compute_four_level_support(
    calibrator: Any,
    z: torch.Tensor,
    atom_bond_local: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    class_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Build independent-count support features for matched Hij edges."""
    device = edge_src.device
    dtype = torch.float32
    stats_obj = getattr(calibrator, "train_stats", None)
    stats: Mapping[str, Any] = stats_obj if isinstance(stats_obj, Mapping) else {}
    z_dev = z.to(device).long()
    zi, zj = z_dev[edge_src.long()], z_dev[edge_dst.long()]
    pair_keys = [
        element_pair_key(a, b)
        for a, b in zip(zi.detach().cpu().tolist(), zj.detach().cpu().tolist())
    ]
    motif_keys = environment_motif_keys(z_dev, atom_bond_local, edge_src, edge_dst, class_names)
    exact_keys = calibrator._hij_environment_signatures(
        z_dev, atom_bond_local.to(device), edge_src, edge_dst, class_names,
    )

    maps = {level: _count_map(stats, key) for level, key in _COUNT_KEYS.items()}
    exact_count = _tensor_from_keys(exact_keys, maps["exact"], device=device, dtype=dtype)
    motif_count = _tensor_from_keys(motif_keys, maps["motif"], device=device, dtype=dtype)
    class_count = _tensor_from_keys(class_names, maps["class"], device=device, dtype=dtype)
    pair_count = _tensor_from_keys(pair_keys, maps["pair"], device=device, dtype=dtype)

    schema_ok = int(stats.get("chemical_support_schema_version", 0)) == SUPPORT_SCHEMA_VERSION
    unit_ok = str(stats.get("chemical_support_count_unit", "")) == SUPPORT_COUNT_UNIT
    maps_ok = all(bool(value) for value in maps.values())
    valid_stats = bool(schema_ok and unit_ok and maps_ok)

    risk = support_risk_from_counts(exact_count, motif_count, class_count, pair_count)
    backoff_depth = torch.full_like(exact_count, 4.0)
    backoff_depth = torch.where(pair_count > 0.0, torch.full_like(backoff_depth, 3.0), backoff_depth)
    backoff_depth = torch.where(class_count > 0.0, torch.full_like(backoff_depth, 2.0), backoff_depth)
    backoff_depth = torch.where(motif_count > 0.0, torch.ones_like(backoff_depth), backoff_depth)
    backoff_depth = torch.where(exact_count > 0.0, torch.zeros_like(backoff_depth), backoff_depth)

    return {
        "p_support_raw": risk,
        "support_exact_seen": (exact_count > 0.0).to(dtype=dtype),
        "log_exact_molecule_count": torch.log1p(exact_count),
        "log_motif_molecule_count": torch.log1p(motif_count),
        "log_class_molecule_count": torch.log1p(class_count),
        "log_element_pair_molecule_count": torch.log1p(pair_count),
        "support_backoff_depth": backoff_depth,
        "support_stats_available": torch.full_like(risk, 1.0 if valid_stats else 0.0),
    }


def _validate_support_artifact(payload: Mapping[str, Any], path: Path) -> None:
    if int(payload.get("schema_version", 0)) != SUPPORT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported chemical-support artifact schema: {path}")
    if str(payload.get("kind", "")) != "chemical_support_probability":
        raise ValueError(f"Artifact is not a chemical-support probability: {path}")
    if str(payload.get("target", "")) != "insufficient_training_support":
        raise ValueError(f"Chemical-support artifact has the wrong target: {path}")
    if str(payload.get("count_unit", "")) != SUPPORT_COUNT_UNIT:
        raise ValueError(f"Chemical-support artifact is not based on independent molecules: {path}")
    fit_split = str(payload.get("fit_split", ""))
    if fit_split not in {"chemical_group_holdout", "scaffold_group_holdout"}:
        raise ValueError(f"Chemical-support artifact was not fit on a chemical holdout: {path}")
    names = payload.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Chemical-support artifact has no features: {path}")
    forbidden = ("response", "error", "enk", "benefit", "ep_")
    if any(any(token in str(name).lower() for token in forbidden) for name in names):
        raise ValueError(f"Chemical-support artifact contains forbidden response/error features: {path}")
    for key in ("feature_mean", "feature_scale", "coefficients"):
        value = payload.get(key)
        if not isinstance(value, list) or len(value) != len(names):
            raise ValueError(f"Invalid {key} in chemical-support artifact: {path}")


@lru_cache(maxsize=8)
def load_support_artifact(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_support_artifact(payload, path)
    return dict(payload)


def apply_support_artifact(
    features: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
) -> torch.Tensor:
    names = [str(name) for name in artifact["feature_names"]]
    missing = [name for name in names if name not in features]
    if missing:
        raise KeyError(f"Chemical-support runtime features missing: {missing}")
    reference = features[names[0]]
    x = torch.stack([features[name].to(reference) for name in names], dim=-1)
    mean = torch.tensor(artifact["feature_mean"], device=x.device, dtype=x.dtype)
    scale = torch.tensor(artifact["feature_scale"], device=x.device, dtype=x.dtype).clamp(min=1e-8)
    coef = torch.tensor(artifact["coefficients"], device=x.device, dtype=x.dtype)
    intercept = _finite_float(artifact.get("intercept", 0.0))
    return torch.sigmoid(((x - mean) / scale) @ coef + intercept).clamp(0.0, 1.0)
