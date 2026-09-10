"""Joint local-error probability and selective EP action routing helpers."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


RISK_SCHEMA_VERSION = 2


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _validate_joint_artifact(payload: Mapping[str, Any], path: Path) -> None:
    if int(payload.get("schema_version", 0)) != RISK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported joint-risk artifact schema: {path}")
    if str(payload.get("kind", "")) != "joint_local_error_probability":
        raise ValueError(f"Artifact is not a joint local-error probability: {path}")
    if str(payload.get("target", "")) != "pre_gsc_ep_aware_hij_local_error_ge_threshold":
        raise ValueError(f"Joint-risk artifact has the wrong target: {path}")
    if str(payload.get("fit_split", "")) != "val_group_crossfit":
        raise ValueError(f"Joint-risk artifact was not group-cross-fitted on validation: {path}")
    names = payload.get("feature_names")
    if not isinstance(names, list) or "p_support" not in names:
        raise ValueError(f"Joint-risk artifact must consume p_support: {path}")
    if not any(str(name).startswith("enk_") for name in names):
        raise ValueError(f"Joint-risk artifact must consume rich ENK state: {path}")
    for key in ("feature_mean", "feature_scale", "coefficients"):
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != len(names):
            raise ValueError(f"Invalid {key} in joint-risk artifact: {path}")


@lru_cache(maxsize=8)
def load_joint_risk_artifact(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_joint_artifact(payload, path)
    return dict(payload)


def apply_joint_risk_artifact(
    features: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
) -> torch.Tensor:
    names = [str(name) for name in artifact["feature_names"]]
    missing = [name for name in names if name not in features]
    if missing:
        raise KeyError(f"Joint-risk runtime features missing: {missing}")
    reference = features[names[0]]
    x = torch.stack([features[name].to(reference) for name in names], dim=-1)
    mean = torch.tensor(artifact["feature_mean"], device=x.device, dtype=x.dtype)
    scale = torch.tensor(artifact["feature_scale"], device=x.device, dtype=x.dtype).clamp(min=1e-8)
    coef = torch.tensor(artifact["coefficients"], device=x.device, dtype=x.dtype)
    intercept = _finite_float(artifact.get("intercept", 0.0))
    return torch.sigmoid(((x - mean) / scale) @ coef + intercept).clamp(0.0, 1.0)


def smooth_selective_gate(probability: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """C1 smooth threshold with an exact zero below ``low``."""
    low_f = min(max(float(low), 0.0), 1.0)
    high_f = min(max(float(high), low_f + 1e-6), 1.0)
    x = ((probability.clamp(0.0, 1.0) - low_f) / (high_f - low_f)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def selective_ep_trigger(
    p_error: torch.Tensor,
    ep_evidence: torch.Tensor,
    physics_need: torch.Tensor,
    reliability: torch.Tensor,
    *,
    error_low: float = 0.20,
    error_high: float = 0.60,
    ep_low: float = 0.35,
    ep_high: float = 0.75,
    strong_ep_floor: float = 0.35,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build one non-duplicated action trigger.

    Error probability selects risky edges.  EP evidence supplies physical action
    authority.  Strong physical evidence receives a bounded floor so a low ENK
    score cannot veto a clearly identified special interaction.  The returned
    trigger is exactly zero for low-risk, low-evidence edges.
    """
    p_error = p_error.clamp(0.0, 1.0)
    ep_evidence = ep_evidence.to(p_error).clamp(0.0, 1.0)
    physics_need = physics_need.to(p_error).clamp(0.0, 1.0)
    reliability = reliability.to(p_error).clamp(0.0, 1.0)

    risk_gate = smooth_selective_gate(p_error, error_low, error_high)
    ep_gate = smooth_selective_gate(ep_evidence, ep_low, ep_high)
    floor = min(max(float(strong_ep_floor), 0.0), 1.0) * ep_gate
    action_need = floor + (1.0 - floor) * risk_gate
    physical_authority = torch.maximum(physics_need, ep_evidence)
    trigger = (physical_authority * reliability * action_need).clamp(0.0, 1.0)
    return trigger, {
        "risk_gate": risk_gate,
        "ep_gate": ep_gate,
        "action_need": action_need,
        "physical_authority": physical_authority,
    }

