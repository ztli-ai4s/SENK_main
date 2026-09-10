from __future__ import annotations

from typing import Any, List


DEFAULT_SENK_ENK_EP_HIJ = "checkpoints/EP/hij/qme14s/best.pt"
DEFAULT_SENK_ENK_EP_DD = "checkpoints/EP/dedipole/qme14s/best.pt"
DEFAULT_SENK_ENK_EP_DP = "checkpoints/EP/depolar/qme14s/best.pt"
# Joint bundle (SIMG+qcMol); pairs with electron_prior_mode=qcmol.
DEFAULT_ELECTRON_PRIOR_CKPT = "nbo_nets/checkpoints/20260325-143400_joint_pdbbind_from_pubchem_joint/nbo_foundation_training_best.pt"
DEFAULT_ELECTRON_PRIOR_STATS = "nbo_nets/checkpoints/20260325-143400_joint_pdbbind_from_pubchem_joint/norm_stats.pt"
DEFAULT_NBO_TRAIN_STATS = "nbo_train_stats.pt"

# SIMG v2 prior (train_nbo_v2.py Stage-1); pairs with electron_prior_mode=simg.
DEFAULT_ELECTRON_PRIOR_CKPT_SIMG = "nbo_nets/checkpoints/2026-03-10_00-01-48_nbo_foundation_v2/nbo_foundation_v2_best.pt"
DEFAULT_ELECTRON_PRIOR_STATS_SIMG = "nbo_nets/checkpoints/2026-03-10_00-01-48_nbo_foundation_v2/nbo_foundation_v2_norm_stats.pt"


def apply_simg_prior_preset(args: Any) -> List[str]:
    """Switch inference args to the SIMG v2 foundation prior."""
    changes: List[str] = []
    _set(args, "electron_prior_mode", "simg", changes)
    _set(args, "electron_prior_ckpt", DEFAULT_ELECTRON_PRIOR_CKPT_SIMG, changes)
    _set(args, "electron_prior_stats", DEFAULT_ELECTRON_PRIOR_STATS_SIMG, changes)
    return changes


def _set(args: Any, name: str, value: Any, changes: List[str]) -> None:
    if getattr(args, name, None) != value:
        setattr(args, name, value)
        changes.append(f"{name}={value}")


def _set_if_empty(args: Any, name: str, value: Any, changes: List[str]) -> None:
    if not getattr(args, name, None):
        _set(args, name, value, changes)


def _set_if_outdated(args: Any, name: str, legacy_values: tuple[Any, ...], value: Any, changes: List[str]) -> None:
    current = getattr(args, name, None)
    if current in legacy_values:
        _set(args, name, value, changes)


def apply_canonical_ep_defaults(args: Any, *, fill_ep_checkpoints: bool = False) -> List[str]:
    """Apply EP defaults, selecting feature_spring unless legacy is requested.

    The feature path retains internal electron priors and uses a complete local
    Hessian spring correction; output DD/DP corrections are disabled. Historical
    GSC settings below remain available to the explicit legacy output policy.
    """
    changes: List[str] = []

    if fill_ep_checkpoints:
        _set_if_empty(args, "hij_ep_ckpt", DEFAULT_SENK_ENK_EP_HIJ, changes)
        _set_if_empty(args, "dd_ep_ckpt", DEFAULT_SENK_ENK_EP_DD, changes)
        _set_if_empty(args, "dp_ep_ckpt", DEFAULT_SENK_ENK_EP_DP, changes)

    if not bool(getattr(args, "nbo_gsc_enabled", False)):
        _set(args, "nbo_gsc_enabled", True, changes)

    branches = str(getattr(args, "nbo_gsc_branches", "") or "").strip()
    if not branches or branches == "hij,dd":
        _set(args, "nbo_gsc_branches", "hij", changes)

    _set_if_empty(args, "nbo_train_stats", DEFAULT_NBO_TRAIN_STATS, changes)
    if str(getattr(args, "nbo_train_stats", "")).replace("\\", "/").endswith("nbo_train_stats_6.3.pt"):
        _set(args, "nbo_train_stats", DEFAULT_NBO_TRAIN_STATS, changes)
    _set_if_empty(args, "electron_prior_ckpt", DEFAULT_ELECTRON_PRIOR_CKPT, changes)
    _set_if_empty(args, "electron_prior_stats", DEFAULT_ELECTRON_PRIOR_STATS, changes)
    if getattr(args, "electron_prior_mode", "off") == "off":
        _set(args, "electron_prior_mode", "qcmol", changes)
    _set_if_outdated(args, "electron_prior_scale", (5e-3,), 1e-2, changes)

    # Match the cascade-tested GSC policy. These lines convert older single-molecule
    # defaults into the current EP/cascade defaults without clobbering custom values.
    _set_if_outdated(args, "nbo_gsc_alpha_hij", (0.25,), 0.15, changes)
    _set_if_outdated(args, "nbo_gsc_alpha_dd", (0.0,), 0.20, changes)
    _set_if_outdated(args, "nbo_gsc_alpha_dp", (0.20,), 0.0, changes)
    _set_if_outdated(args, "nbo_gsc_bond_dipole_factor", (0.20,), 0.30, changes)
    _set_if_outdated(args, "nbo_gsc_alpha_bond", (0.12,), 0.15, changes)
    _set_if_outdated(args, "nbo_gsc_dd_sum_rule", (0.0,), 1.0, changes)
    _set_if_outdated(args, "nbo_gsc_dd_interaction_gate_scale", (0.0,), 0.35, changes)
    _set_if_outdated(args, "nbo_gsc_dp_interaction_gate_scale", (0.35,), 0.0, changes)
    _set_if_outdated(args, "nbo_gsc_dp_interaction_boost", (0.20,), 0.0, changes)
    _set_if_outdated(args, "nbo_gsc_hij_policy", ("interaction_rule",), "hybrid", changes)
    _set_if_outdated(args, "nbo_gsc_hij_hbond_ood_boost", (0.0,), 0.45, changes)
    _set_if_outdated(args, "nbo_gsc_hij_hbond_alpha_scale", (1.0,), 1.6, changes)
    _set_if_outdated(args, "nbo_gsc_hij_interaction_soften_scale", (1.3,), 1.0, changes)
    _set_if_outdated(args, "nbo_gsc_hij_interaction_alpha_scale", (3.0,), 1.0, changes)
    # Final feature EP: preserve model/weight layout; replace output correction.
    if getattr(args, "ep_output_policy", "feature_spring") == "feature_spring":
        _set(args, "ep_output_policy", "feature_spring", changes)
        _set(args, "electron_prior_scale", 0.005, changes)
        _set(args, "nbo_gsc_branches", "hij", changes)
        _set(args, "nbo_gsc_hij_policy", "interaction_rule", changes)
        if getattr(args, "nbo_train_stats", "") in ("", "nbo_train_stats.pt"):
            _set(args, "nbo_train_stats", "nbo_train_stats_runtime_hij_full.pt", changes)
    return changes
