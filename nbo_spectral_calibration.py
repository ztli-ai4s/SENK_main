"""
NBO-Guided Spectral Calibration (NBO-GSC)
==========================================

Post-hoc calibration of spectral predictions using NBO (Natural Bond Orbital)
information from the electron prior predictor.  Activated **only** when EP is
runtime-active; pure v2_enk inference is completely unaffected.

v2: Training-set statistical normalization
-------------------------------------------
The v1 calibration used per-molecule averages (avg_occ, avg_hij) as the
normalization baseline.  This caused systematic distortion on large molecules
where the average is dominated by numerous backbone bonds (C-C, C-N) while
the chemically interesting bonds (O-H, C=O) are a tiny minority.

The v2 calibration uses **training-set statistics** as the baseline, so the
calibration measures absolute deviation from the training distribution rather
than relative deviation within the molecule.  This ensures:
  - A C-C bond with occ=1.98 gets the same calibration regardless of whether
    the molecule has 5 or 500 other bonds
  - An O-H bond with occ=1.85 is correctly identified as "weaker than
    training average" even if the molecule average is pulled down by many
    weak interactions

Training-set statistics are optional.  When not provided, the calibration
falls back to per-molecule normalization (v1 behavior) with a warning.
"""

from __future__ import annotations

import re
import torch
from typing import Optional, Dict, Any, List, Sequence, Set, Tuple


def _match_undirected_edges_simple(
    edge_index: torch.Tensor,
    atom_bond_index: torch.Tensor,
    num_nodes: int,
):
    if atom_bond_index.numel() == 0:
        return (torch.zeros(0, dtype=torch.long, device=edge_index.device),
                torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device))
    src, dst = edge_index[0], edge_index[1]
    bsrc, bdst = atom_bond_index[0], atom_bond_index[1]
    n_edges = edge_index.size(1)
    n_bonds = atom_bond_index.size(1)
    edge_hash = src.long() * num_nodes + dst.long()
    bond_hash_fwd = bsrc.long() * num_nodes + bdst.long()
    bond_hash_rev = bdst.long() * num_nodes + bsrc.long()
    bond_mask = torch.zeros(n_edges, dtype=torch.bool, device=edge_index.device)
    bond_ids = torch.zeros(n_edges, dtype=torch.long, device=edge_index.device)
    if n_bonds > 0 and n_edges > 0:
        eh = edge_hash.unsqueeze(1)
        bh_fwd = bond_hash_fwd.unsqueeze(0)
        bh_rev = bond_hash_rev.unsqueeze(0)
        match_any = (eh == bh_fwd) | (eh == bh_rev)
        bond_mask = match_any.any(dim=1)
        bond_ids = match_any.float().argmax(dim=1)
    return bond_ids, bond_mask


def _extract_nbo_from_model(model: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
    prior = getattr(model, "electron_prior", None)
    if prior is None:
        return None
    nbo = getattr(prior, "_last_nbo_predictions", None)
    if not isinstance(nbo, dict) or not nbo:
        return None
    return nbo


def _bond_occupancy_col(n_cols: int, policy: str = "legacy") -> Optional[int]:
    """Return the bond occupancy/proxy column for GSC normalization.

    Existing NBO-GSC stats were computed with the legacy f0 proxy.  The qcMol
    diagnostic schema labels f14 as occupancy, but switching to it requires
    separately matched stats; do not change the default silently.
    """
    p = str(policy or "legacy").lower()
    if p in {"legacy", "stats", "f0", "col0"}:
        return 0 if n_cols > 0 else None
    if p in {"schema", "qcmol", "f14", "col14"}:
        return 14 if n_cols >= 15 else (0 if n_cols > 0 else None)
    if n_cols >= 15:
        return 14
    if n_cols > 0:
        return 0
    return None


class NBOGuidedCalibrator:
    """Post-hoc NBO-guided spectral calibration with training-set normalization.

    Parameters
    ----------
    alpha_hij, alpha_dd, alpha_dp : float
        Calibration strengths per property.
    bond_order_exponent : float
        Power-law exponent for |Hij| ∝ occ^exp (Badger rule).
    clamp_ratio : float
        Maximum absolute calibration ratio deviation from 1.0.
    train_stats : dict or None
        Training-set statistics for absolute normalization.  Expected keys:
          - "log_hij_bond_mean", "log_hij_bond_std"
          - "log_occ_mean", "log_occ_std"
          - "dd_trace_mean", "dd_trace_std"
          - "npa_charge_mean", "npa_charge_std"
          - "dp_norm_mean", "dp_norm_std"
          - "delocal_mean", "delocal_std"
        When None, falls back to per-molecule normalization (v1).
    """

    def __init__(
        self,
        alpha_hij: float = 0.15,
        alpha_dd: float = 0.0,
        alpha_dp: float = 0.0,   # DP GSC disabled — delocalization proxy has no predictive signal
        bond_order_exponent: float = 0.6,
        clamp_ratio: float = 0.5,
        train_stats: Optional[Dict[str, Any]] = None,
        bond_dipole_factor: float = 0.20,
        alpha_bond: float = 0.12,
        dd_sum_rule_strength: float = 0.0,
        hij_target_xh_only: bool = False,
        hij_non_target_scale: float = 0.05,
        hij_amide_scale: float = 0.10,
        hij_consensus_id_threshold: float = 0.85,
        hij_consensus_ood_threshold: float = 0.45,
        hij_enk_ood_weight: float = 0.5,
        hij_mode_aware: bool = True,
        hij_xh_freq_min: float = 2200.0,
        hij_xh_freq_max: float = 4200.0,
        hij_mode_participation_threshold: float = 0.015,
        hij_hii_comp_strength: float = 0.1,
        hij_gate_mode: str = "adaptive",
        hij_train_elements: Any = "1,6,7,8,9",
        hij_train_bond_classes: Any = "",
        hij_xh_scale: float = 0.6,
        hij_unknown_scale: float = 0.8,
        hij_unknown_ood_boost: float = 0.35,
        hij_strong_ood_threshold: float = 0.70,
        hij_amide_freq_min: float = 1450.0,
        hij_amide_freq_max: float = 1750.0,
        hij_amide_mode_floor: float = 0.20,
        hij_global_scale: float = 0.5,
        hij_backbone_mode_floor: float = 0.08,
        hij_ood_global_scale: float = 1.0,
        hij_env_ood_hops: int = 3,
        hij_env_ood_decay: float = 0.6,
        hij_env_ood_boost: float = 0.40,
        hij_unknown_mode_floor: float = 0.55,
        hij_hbond_distance: float = 2.45,
        hij_hbond_acceptors: Any = "7,8,9,15,16,17",
        hij_hbond_ood_boost: float = 0.45,
        hij_hbond_scale: float = 1.0,
        hij_hbond_mode_floor: float = 0.55,
        hij_ood_alpha_scale: float = 1.5,
        hij_hbond_alpha_scale: float = 1.6,
        hij_policy: str = "hybrid",
        hij_xh_heuristic_scale: float = 0.25,
        hij_interaction_soften_scale: float = 1.0,
        hij_interaction_alpha_scale: float = 1.0,
        hij_interaction_geometry_weight: float = 0.35,
        hij_interaction_min_score: float = 0.15,
        hij_hbond_angle_min: float = 115.0,
        hij_interaction_e2_low_quantile: float = 0.75,
        hij_interaction_e2_high_quantile: float = 0.95,
        hij_interaction_acceptor_distance: float = 3.20,
        hij_interaction_angle_min: float = 85.0,
        hij_interaction_field_weight: float = 0.75,
        hij_physical_rules: bool = True,
        hij_physical_min_score: float = 0.20,
        hij_physical_soften_scale: float = 0.65,
        hij_physical_harden_scale: float = 0.40,
        hij_physical_alpha_scale: float = 1.35,
        hij_stark_gate_scale: float = 0.35,
        dd_trace_mode: str = "off",          # trace correction disabled (harms DD at any alpha>0)
        dd_bond_gate_mode: str = "direct",   # AND-logic gate — most conservative, prevents over-correction
        dd_interaction_gate_scale: float = 0.0,  # interaction gating off (inflates atom_need, catastrophic)
        dp_gate_mode: str = "direct",        # direct gate (alpha_dp=0 makes this a no-op; kept for safety)
        dp_interaction_gate_scale: float = 0.0,  # DP interaction gating off
        dp_interaction_boost: float = 0.0,   # DP interaction boost off
        bond_occ_col_policy: str = "legacy",
    ):
        self.alpha_hij = alpha_hij
        self.alpha_dd = alpha_dd
        self.alpha_dp = alpha_dp
        self.bond_order_exponent = bond_order_exponent
        self.clamp_ratio = clamp_ratio
        self.train_stats = train_stats
        self.bond_dipole_factor = bond_dipole_factor
        self.alpha_bond = alpha_bond
        self.dd_sum_rule_strength = dd_sum_rule_strength
        self.hij_target_xh_only = bool(hij_target_xh_only)
        self.hij_non_target_scale = float(hij_non_target_scale)
        self.hij_amide_scale = float(hij_amide_scale)
        self.hij_consensus_id_threshold = float(hij_consensus_id_threshold)
        self.hij_consensus_ood_threshold = float(hij_consensus_ood_threshold)
        self.hij_enk_ood_weight = float(hij_enk_ood_weight)
        self.hij_mode_aware = bool(hij_mode_aware)
        self.hij_xh_freq_min = float(hij_xh_freq_min)
        self.hij_xh_freq_max = float(hij_xh_freq_max)
        self.hij_mode_participation_threshold = float(hij_mode_participation_threshold)
        self.hij_hii_comp_strength = float(hij_hii_comp_strength)
        self.hij_gate_mode = str(hij_gate_mode or "adaptive").lower()
        self.hij_train_elements = self._parse_int_set(hij_train_elements)
        self.hij_train_bond_classes = self._parse_str_set(hij_train_bond_classes)
        self.hij_xh_scale = float(hij_xh_scale)
        self.hij_unknown_scale = float(hij_unknown_scale)
        self.hij_unknown_ood_boost = float(hij_unknown_ood_boost)
        self.hij_strong_ood_threshold = float(hij_strong_ood_threshold)
        self.hij_amide_freq_min = float(hij_amide_freq_min)
        self.hij_amide_freq_max = float(hij_amide_freq_max)
        self.hij_amide_mode_floor = float(hij_amide_mode_floor)
        self.hij_global_scale = float(hij_global_scale)
        self.hij_backbone_mode_floor = float(hij_backbone_mode_floor)
        self.hij_ood_global_scale = float(hij_ood_global_scale)
        self.hij_env_ood_hops = int(hij_env_ood_hops)
        self.hij_env_ood_decay = float(hij_env_ood_decay)
        self.hij_env_ood_boost = float(hij_env_ood_boost)
        self.hij_unknown_mode_floor = float(hij_unknown_mode_floor)
        self.hij_hbond_distance = float(hij_hbond_distance)
        self.hij_hbond_acceptors = self._parse_int_set(hij_hbond_acceptors)
        self.hij_hbond_ood_boost = float(hij_hbond_ood_boost)
        self.hij_hbond_scale = float(hij_hbond_scale)
        self.hij_hbond_mode_floor = float(hij_hbond_mode_floor)
        self.hij_ood_alpha_scale = float(hij_ood_alpha_scale)
        self.hij_hbond_alpha_scale = float(hij_hbond_alpha_scale)
        self.hij_policy = str(hij_policy or "hybrid").lower()
        self.hij_xh_heuristic_scale = float(hij_xh_heuristic_scale)
        self.hij_interaction_soften_scale = float(hij_interaction_soften_scale)
        self.hij_interaction_alpha_scale = float(hij_interaction_alpha_scale)
        self.hij_interaction_geometry_weight = float(hij_interaction_geometry_weight)
        self.hij_interaction_min_score = float(hij_interaction_min_score)
        self.hij_hbond_angle_min = float(hij_hbond_angle_min)
        self.hij_interaction_e2_low_quantile = float(hij_interaction_e2_low_quantile)
        self.hij_interaction_e2_high_quantile = float(hij_interaction_e2_high_quantile)
        self.hij_interaction_acceptor_distance = float(hij_interaction_acceptor_distance)
        self.hij_interaction_angle_min = float(hij_interaction_angle_min)
        self.hij_interaction_field_weight = float(hij_interaction_field_weight)
        self.hij_physical_rules = bool(hij_physical_rules)
        self.hij_physical_min_score = float(hij_physical_min_score)
        self.hij_physical_soften_scale = float(hij_physical_soften_scale)
        self.hij_physical_harden_scale = float(hij_physical_harden_scale)
        self.hij_physical_alpha_scale = float(hij_physical_alpha_scale)
        self.hij_stark_gate_scale = float(hij_stark_gate_scale)
        self.dd_trace_mode = str(dd_trace_mode or "off").lower()
        self.dd_bond_gate_mode = str(dd_bond_gate_mode or "direct").lower()
        self.dd_interaction_gate_scale = float(dd_interaction_gate_scale)
        self.dp_gate_mode = str(dp_gate_mode or "legacy").lower()
        self.dp_interaction_gate_scale = float(dp_interaction_gate_scale)
        self.dp_interaction_boost = float(dp_interaction_boost)
        self.bond_occ_col_policy = str(bond_occ_col_policy or "legacy").lower()
        self.last_hij_delta: Optional[torch.Tensor] = None
        self.last_hij_debug: Dict[str, float] = {}
        self.last_dd_debug: Dict[str, float] = {}
        self.last_dp_debug: Dict[str, float] = {}
        if train_stats is None:
            import warnings
            warnings.warn(
                "NBOGuidedCalibrator: train_stats not provided. "
                "Falling back to per-molecule normalization (v1). "
                "For best results on large molecules, provide training-set statistics.",
                stacklevel=2,
            )

    @staticmethod
    def _parse_int_set(value: Any) -> Set[int]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {int(x) for x in re.split(r"[\s,;]+", value.strip()) if x}
        if isinstance(value, torch.Tensor):
            return {int(x) for x in value.detach().cpu().view(-1).tolist()}
        if isinstance(value, (list, tuple, set)):
            return {int(x) for x in value}
        return {int(value)}

    @staticmethod
    def _parse_str_set(value: Any) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {x.strip() for x in re.split(r"[\s,;]+", value.strip()) if x.strip()}
        if isinstance(value, torch.Tensor):
            return {str(x) for x in value.detach().cpu().view(-1).tolist()}
        if isinstance(value, (list, tuple, set)):
            return {str(x).strip() for x in value if str(x).strip()}
        return {str(value).strip()}

    def _train_bond_class_stats(self) -> Dict[str, Any]:
        if not isinstance(self.train_stats, dict):
            return {}
        for key in ("hij_bond_class_stats", "bond_class_stats", "hij_class_stats"):
            value = self.train_stats.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _effective_train_bond_classes(self) -> Set[str]:
        classes = set(self.hij_train_bond_classes)
        if not isinstance(self.train_stats, dict):
            return classes
        for key in ("hij_seen_bond_classes", "seen_bond_classes", "bond_classes"):
            if key in self.train_stats:
                classes.update(self._parse_str_set(self.train_stats.get(key)))
        classes.update(str(k) for k in self._train_bond_class_stats().keys())
        return classes

    def _effective_train_env_signatures(self) -> Set[str]:
        if not isinstance(self.train_stats, dict):
            return set()
        signatures: Set[str] = set()
        for key in ("hij_seen_env_signatures", "seen_env_signatures", "env_signatures"):
            if key in self.train_stats:
                signatures.update(self._parse_str_set(self.train_stats.get(key)))
        return signatures

    def _effective_train_elements(self) -> Set[int]:
        elements = set(self.hij_train_elements)
        if not isinstance(self.train_stats, dict):
            return elements
        for key in ("hij_seen_elements", "seen_elements", "train_elements"):
            if key in self.train_stats:
                elements.update(self._parse_int_set(self.train_stats.get(key)))
        return elements

    def _class_stat_tensor(
        self,
        class_names: Sequence[str],
        stat_key: str,
        fallback_key: str,
        default: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(self.train_stats, dict):
            return torch.full((len(class_names),), float(default), device=device, dtype=dtype)

        global_value = self.train_stats.get(stat_key, self.train_stats.get(fallback_key, default))
        class_stats = self._train_bond_class_stats()
        values: List[float] = []
        for cls in class_names:
            value = None
            nested = class_stats.get(cls)
            if isinstance(nested, dict):
                value = nested.get(stat_key, nested.get(fallback_key))
            if value is None:
                value = self.train_stats.get(f"{stat_key}__{cls}")
            if value is None and fallback_key:
                value = self.train_stats.get(f"{fallback_key}__{cls}")
            if value is None:
                value = global_value
            values.append(float(value))
        return torch.tensor(values, device=device, dtype=dtype)

    def _class_count_tensor(
        self,
        class_names: Sequence[str],
        key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        class_stats = self._train_bond_class_stats()
        values: List[float] = []
        for cls in class_names:
            nested = class_stats.get(cls)
            value = 0.0
            if isinstance(nested, dict):
                value = float(nested.get(key, nested.get("n", 0.0)))
            values.append(value)
        return torch.tensor(values, device=device, dtype=dtype)

    def _score_above_train(
        self,
        values: torch.Tensor,
        mean_key: str,
        std_key: str,
        start_sigma: float = 1.0,
        full_sigma: float = 3.0,
    ) -> torch.Tensor:
        if not isinstance(self.train_stats, dict):
            return values.clamp(0.0, 1.0)
        if mean_key not in self.train_stats or std_key not in self.train_stats:
            return values.clamp(0.0, 1.0)
        mu = float(self.train_stats.get(mean_key, 0.0))
        sig = max(float(self.train_stats.get(std_key, 1.0)), 1e-6)
        z = (values - mu) / sig
        return ((z - float(start_sigma)) / max(float(full_sigma - start_sigma), 1e-6)).clamp(0.0, 1.0)

    def _atom_training_ood_score(
        self,
        z: torch.Tensor,
        atom_bond_local: torch.Tensor,
    ) -> torch.Tensor:
        device = z.device
        train_elements = self._effective_train_elements()
        if not train_elements or z.numel() == 0:
            return torch.zeros(z.numel(), device=device, dtype=torch.float32)

        known = torch.zeros_like(z, dtype=torch.bool)
        for elem in train_elements:
            known = known | (z == int(elem))
        score = (~known).to(dtype=torch.float32)

        if atom_bond_local is None or atom_bond_local.numel() == 0:
            return score
        hops = max(int(self.hij_env_ood_hops), 0)
        if hops <= 0:
            return score

        bonds = atom_bond_local.to(device).long()
        b0, b1 = bonds[0], bonds[1]
        env_score = score.clone()
        frontier = score.clone()
        decay = min(max(float(self.hij_env_ood_decay), 0.0), 1.0)
        for _ in range(hops):
            propagated = torch.zeros_like(frontier)
            propagated.index_add_(0, b0, frontier[b1])
            propagated.index_add_(0, b1, frontier[b0])
            propagated = propagated.clamp(0.0, 1.0) * decay
            env_score = torch.maximum(env_score, propagated)
            frontier = propagated
        return env_score.clamp(0.0, 1.0)

    def _hij_environment_signatures(
        self,
        z: Optional[torch.Tensor],
        atom_bond_local: Optional[torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        class_names: Sequence[str],
    ) -> List[str]:
        if z is None or atom_bond_local is None or atom_bond_local.numel() == 0:
            return [f"{cls}|env:unknown" for cls in class_names]
        device = edge_src.device
        z_cpu = z.detach().cpu().long()
        bonds = atom_bond_local.detach().cpu().long()
        n_nodes = int(z_cpu.numel())
        neigh: List[List[int]] = [[] for _ in range(n_nodes)]
        for a, b in bonds.t().tolist():
            if 0 <= int(a) < n_nodes and 0 <= int(b) < n_nodes:
                neigh[int(a)].append(int(b))
                neigh[int(b)].append(int(a))

        src_l = edge_src.detach().cpu().long().tolist()
        dst_l = edge_dst.detach().cpu().long().tolist()
        signatures: List[str] = []
        for cls, s, d in zip(class_names, src_l, dst_l):
            zs = int(z_cpu[int(s)].item())
            zd = int(z_cpu[int(d)].item())

            def _side(center: int, other: int) -> str:
                vals = sorted(int(z_cpu[nbr].item()) for nbr in neigh[int(center)] if int(nbr) != int(other))
                if not vals:
                    return "none"
                compact: List[str] = []
                last = None
                count = 0
                for val in vals + [None]:
                    if last is None:
                        last = val
                        count = 1
                    elif val == last:
                        count += 1
                    else:
                        compact.append(f"{last}x{count}")
                        last = val
                        count = 1
                return ".".join(compact)

            left = f"z{zs}[{_side(int(s), int(d))}]"
            right = f"z{zd}[{_side(int(d), int(s))}]"
            if (zd, right) < (zs, left):
                left, right = right, left
            signatures.append(f"{cls}|{left}|{right}")
        return signatures

    def _atom_context_need(
        self,
        z: Optional[torch.Tensor],
        atom_bond_local: Optional[torch.Tensor],
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        pos: Optional[torch.Tensor] = None,
        enk_ood_score: Optional[torch.Tensor] = None,
        enk_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_need = torch.zeros(int(num_nodes), device=device, dtype=dtype)
        edge_need = torch.zeros(0, device=device, dtype=dtype)

        if z is not None:
            z_dev = z.to(device).long()
            if atom_bond_local is not None and atom_bond_local.numel() > 0:
                bonds = atom_bond_local.to(device).long()
                src, dst = bonds[0], bonds[1]
                atom_ood = self._atom_training_ood_score(z_dev, bonds).to(device=device, dtype=dtype)
                class_names, _, _, unknown_atom, unknown_class, env_ood, hbond_mask = self._hij_bond_context(
                    z_dev, bonds, src, dst, pos=pos,
                )
                signatures = self._hij_environment_signatures(z_dev, bonds, src, dst, class_names)
                seen_signatures = self._effective_train_env_signatures()
                if seen_signatures:
                    unknown_sig = torch.tensor(
                        [sig not in seen_signatures for sig in signatures],
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    unknown_sig = torch.zeros(src.numel(), dtype=torch.bool, device=device)

                edge_need = torch.maximum(
                    env_ood.to(device=device, dtype=dtype),
                    (unknown_atom | unknown_class | unknown_sig).to(device=device, dtype=dtype),
                )
                if hbond_mask is not None and hbond_mask.numel() == edge_need.numel():
                    edge_need = torch.maximum(edge_need, hbond_mask.to(device=device, dtype=dtype))
                edge_to_atom = torch.zeros_like(atom_need)
                edge_to_atom.index_add_(0, src.long(), edge_need)
                edge_to_atom.index_add_(0, dst.long(), edge_need)
                atom_need = torch.maximum(atom_need, atom_ood)
                atom_need = torch.maximum(atom_need, edge_to_atom.clamp(0.0, 1.0))
            else:
                atom_need = torch.maximum(
                    atom_need,
                    self._atom_training_ood_score(z_dev, atom_bond_local).to(device=device, dtype=dtype),
                )

        atom_enk = None
        atom_dis = None
        if isinstance(enk_context, dict) and enk_context:
            maybe_ood = enk_context.get("atom_ood")
            if isinstance(maybe_ood, torch.Tensor) and maybe_ood.numel() == int(num_nodes):
                atom_enk = maybe_ood.to(device=device, dtype=dtype)
            maybe_dis = enk_context.get("branch_disagreement")
            if isinstance(maybe_dis, torch.Tensor) and maybe_dis.numel() == int(num_nodes):
                atom_dis = maybe_dis.to(device=device, dtype=dtype)
        if atom_enk is None and enk_ood_score is not None and enk_ood_score.numel() == int(num_nodes):
            atom_enk = enk_ood_score.to(device=device, dtype=dtype)
        if atom_enk is not None:
            atom_need = torch.maximum(
                atom_need,
                self._score_above_train(atom_enk, "enk_atom_ood_mean", "enk_atom_ood_std"),
            )
        if atom_dis is not None:
            atom_need = torch.maximum(
                atom_need,
                self._score_above_train(atom_dis, "enk_branch_disagreement_mean", "enk_branch_disagreement_std"),
            )
        return atom_need.clamp(0.0, 1.0), edge_need.clamp(0.0, 1.0)

    def _hbond_details(
        self,
        z: torch.Tensor,
        pos: Optional[torch.Tensor],
        atom_bond_local: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        is_xh: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = edge_src.device
        n_edges = int(edge_src.numel())
        edge_mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
        atom_score = torch.zeros(z.numel(), dtype=torch.float32, device=device)
        empty_atom = torch.full((n_edges,), -1, dtype=torch.long, device=device)
        dist_out = torch.full((n_edges,), float("inf"), dtype=torch.float32, device=device)
        angle_out = torch.zeros(n_edges, dtype=torch.float32, device=device)
        geom_score = torch.zeros(n_edges, dtype=torch.float32, device=device)
        details = {
            "edge_mask": edge_mask,
            "atom_score": atom_score,
            "h_atom": empty_atom.clone(),
            "donor_atom": empty_atom.clone(),
            "acceptor_atom": empty_atom.clone(),
            "h_acceptor_distance": dist_out,
            "xh_acceptor_angle": angle_out,
            "geometry_score": geom_score,
        }
        if pos is None or n_edges == 0 or not is_xh.any() or not self.hij_hbond_acceptors:
            return details

        pos = pos.to(device)
        zi = z[edge_src]
        zj = z[edge_dst]
        h_atom = torch.where(zi == 1, edge_src, edge_dst).long()
        donor_atom = torch.where(zi == 1, edge_dst, edge_src).long()
        xh_idx = torch.nonzero(is_xh, as_tuple=False).view(-1)
        if xh_idx.numel() == 0:
            return details

        acceptor_mask = torch.zeros(z.numel(), dtype=torch.bool, device=device)
        for elem in self.hij_hbond_acceptors:
            acceptor_mask = acceptor_mask | (z == int(elem))
        if not acceptor_mask.any():
            return details

        h_sel = h_atom[xh_idx]
        donor_sel = donor_atom[xh_idx]
        dist = torch.cdist(pos[h_sel], pos)
        valid = acceptor_mask.unsqueeze(0).expand_as(dist).clone()
        valid.scatter_(1, h_sel[:, None], False)
        valid.scatter_(1, donor_sel[:, None], False)
        dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
        min_dist, acceptor = dist.min(dim=1)
        v_hd = pos[donor_sel] - pos[h_sel]
        v_ha = pos[acceptor] - pos[h_sel]
        cosang = (v_hd * v_ha).sum(dim=-1) / (
            v_hd.norm(dim=-1).clamp(min=1e-6) * v_ha.norm(dim=-1).clamp(min=1e-6)
        )
        angle = torch.acos(cosang.clamp(-1.0, 1.0)) * (180.0 / torch.pi)
        details["h_atom"][xh_idx] = h_sel
        details["donor_atom"][xh_idx] = donor_sel
        details["acceptor_atom"][xh_idx] = acceptor.long()
        details["h_acceptor_distance"][xh_idx] = min_dist.float()
        details["xh_acceptor_angle"][xh_idx] = angle.float()
        dist_score = (
            (float(self.hij_hbond_distance) - min_dist)
            / max(float(self.hij_hbond_distance) - 1.55, 1e-6)
        ).clamp(0.0, 1.0)
        angle_score = (
            (angle - float(self.hij_hbond_angle_min))
            / max(180.0 - float(self.hij_hbond_angle_min), 1e-6)
        ).clamp(0.0, 1.0)
        geom_score[xh_idx] = torch.sqrt((dist_score * angle_score).clamp(0.0, 1.0)).float()
        is_hbond = (
            (min_dist >= 1.15)
            & (min_dist <= float(self.hij_hbond_distance))
            & (angle >= float(self.hij_hbond_angle_min))
        )
        if not is_hbond.any():
            details["geometry_score"] = geom_score.clamp(0.0, 1.0)
            return details

        hb_edges = xh_idx[is_hbond]
        hb_h = h_sel[is_hbond]
        hb_donor = donor_sel[is_hbond]
        hb_acceptor = acceptor[is_hbond]
        edge_mask[hb_edges] = True
        atom_score[hb_h] = 1.0
        atom_score[hb_donor] = 1.0
        atom_score[hb_acceptor] = 1.0

        if atom_bond_local is not None and atom_bond_local.numel() > 0:
            bonds = atom_bond_local.to(device).long()
            b0, b1 = bonds[0], bonds[1]
            propagated = torch.zeros_like(atom_score)
            propagated.index_add_(0, b0, atom_score[b1])
            propagated.index_add_(0, b1, atom_score[b0])
            atom_score = torch.maximum(atom_score, 0.5 * propagated.clamp(0.0, 1.0))
        details["edge_mask"] = edge_mask
        details["atom_score"] = atom_score.clamp(0.0, 1.0)
        details["geometry_score"] = geom_score.clamp(0.0, 1.0)
        return details

    def _hbond_environment(
        self,
        z: torch.Tensor,
        pos: Optional[torch.Tensor],
        atom_bond_local: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        is_xh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        details = self._hbond_details(z, pos, atom_bond_local, edge_src, edge_dst, is_xh)
        return details["edge_mask"], details["atom_score"]

    @staticmethod
    def _get_skeleton_tensor(
        nbo: Dict[str, torch.Tensor],
        skeleton_data: Optional[Any],
        name: str,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        value = getattr(skeleton_data, name, None) if skeleton_data is not None else None
        if value is None and isinstance(nbo, dict):
            value = nbo.get(name)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return None

    def _interaction_e2_payload(
        self,
        nbo: Dict[str, torch.Tensor],
        skeleton_data: Optional[Any],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        interaction_pred = nbo.get("interaction_pred") if isinstance(nbo, dict) else None
        interaction_edge_index = self._get_skeleton_tensor(nbo, skeleton_data, "interaction_edge_index", device)
        qlo = torch.tensor(0.0, device=device, dtype=dtype)
        qhi = torch.tensor(1.0, device=device, dtype=dtype)
        denom = torch.tensor(1.0, device=device, dtype=dtype)
        if not isinstance(interaction_pred, torch.Tensor) or interaction_edge_index is None or interaction_edge_index.numel() == 0:
            return None, None, qlo, qhi, denom
        e2 = interaction_pred.to(device=device, dtype=dtype)
        if e2.numel() == 0:
            return None, None, qlo, qhi, denom
        if e2.dim() > 1:
            e2 = e2[:, 0]
        e2 = torch.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        positive = e2[torch.isfinite(e2) & (e2 > 0)]
        if positive.numel() >= 4:
            qlo = torch.quantile(
                positive,
                min(max(float(self.hij_interaction_e2_low_quantile), 0.0), 0.99),
            )
            qhi = torch.quantile(
                positive,
                min(max(float(self.hij_interaction_e2_high_quantile), 0.01), 1.0),
            )
        elif positive.numel() > 0:
            qlo = positive.min()
            qhi = positive.max()
        denom = (qhi - qlo).abs().clamp(min=1e-6)
        return e2, interaction_edge_index.to(device=device, dtype=torch.long), qlo, qhi, denom

    @staticmethod
    def _score_from_e2(
        raw_e2: torch.Tensor,
        qlo: torch.Tensor,
        qhi: torch.Tensor,
        denom: torch.Tensor,
    ) -> torch.Tensor:
        quantile_score = ((raw_e2 - qlo) / denom).clamp(0.0, 1.0)
        relative_score = (raw_e2 / qhi.abs().clamp(min=1e-6)).clamp(0.0, 1.0)
        return torch.maximum(quantile_score, 0.75 * relative_score).clamp(0.0, 1.0)

    def _interaction_atom_exposure(
        self,
        nbo: Dict[str, torch.Tensor],
        num_atoms: int,
        device: torch.device,
        dtype: torch.dtype,
        skeleton_data: Optional[Any] = None,
    ) -> torch.Tensor:
        score = torch.zeros(int(num_atoms), dtype=dtype, device=device)
        e2, ie, qlo, qhi, denom = self._interaction_e2_payload(nbo, skeleton_data, device, torch.float32)
        if e2 is None or ie is None or ie.numel() == 0:
            return score
        e2_score = self._score_from_e2(e2, qlo, qhi, denom)
        max_node = int(ie.max().detach().cpu().item()) + 1 if ie.numel() > 0 else int(num_atoms)
        node_score = torch.zeros(max_node, dtype=torch.float32, device=device)
        src_l = ie[0].detach().cpu().tolist()
        dst_l = ie[1].detach().cpu().tolist()
        val_l = e2_score.detach().cpu().tolist()
        for src_i, dst_i, val_i in zip(src_l, dst_l, val_l):
            val_f = float(val_i)
            if val_f <= 0.0:
                continue
            if 0 <= int(src_i) < max_node:
                node_score[int(src_i)] = torch.maximum(node_score[int(src_i)], node_score.new_tensor(val_f))
            if 0 <= int(dst_i) < max_node:
                node_score[int(dst_i)] = torch.maximum(node_score[int(dst_i)], node_score.new_tensor(val_f))

        n = min(int(num_atoms), int(node_score.numel()))
        if n > 0:
            score[:n] = torch.maximum(score[:n], node_score[:n].to(dtype=dtype))
        atom_to_nbo = self._get_skeleton_tensor(nbo, skeleton_data, "atom_to_nbo_index", device)
        if isinstance(atom_to_nbo, torch.Tensor) and atom_to_nbo.numel() > 0 and node_score.numel() > 0:
            a2n = atom_to_nbo.to(device=device, dtype=torch.long)
            atom_idx = a2n[0]
            node_idx = a2n[1]
            valid = (
                (atom_idx >= 0)
                & (atom_idx < int(num_atoms))
                & (node_idx >= 0)
                & (node_idx < int(node_score.numel()))
            )
            if valid.any():
                mapped = torch.zeros_like(score)
                mapped.index_reduce_(0, atom_idx[valid].long(), node_score[node_idx[valid].long()].to(dtype=dtype), reduce="amax")
                score = torch.maximum(score, mapped)
        return score.clamp(0.0, 1.0)

    def _hij_bond_interaction_exposure(
        self,
        nbo: Dict[str, torch.Tensor],
        skeleton_data: Optional[Any],
        matched_bond_ids: torch.Tensor,
        num_atoms: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = matched_bond_ids.device
        n_edges = int(matched_bond_ids.numel())
        raw_e2 = torch.zeros(n_edges, dtype=torch.float32, device=device)
        e2, ie, qlo, qhi, denom = self._interaction_e2_payload(nbo, skeleton_data, device, torch.float32)
        if e2 is None or ie is None or ie.numel() == 0 or n_edges == 0:
            return raw_e2, raw_e2, raw_e2
        bond_to_local = {
            int(bid): i
            for i, bid in enumerate(matched_bond_ids.detach().cpu().tolist())
        }
        src_l = ie[0].detach().cpu().tolist()
        dst_l = ie[1].detach().cpu().tolist()
        e2_l = e2.detach().cpu().tolist()
        for src_i, dst_i, val_i in zip(src_l, dst_l, e2_l):
            val_f = float(val_i)
            if val_f <= 0.0:
                continue
            for node_i in (int(src_i), int(dst_i)):
                bond_id = node_i - int(num_atoms)
                local_i = bond_to_local.get(bond_id)
                if local_i is not None and val_f > float(raw_e2[local_i].item()):
                    raw_e2[local_i] = val_f
        e2_score = self._score_from_e2(raw_e2, qlo, qhi, denom)
        return e2_score.clamp(0.0, 1.0), raw_e2, e2_score

    def _hij_bond_field_score(
        self,
        atom_pred: Optional[torch.Tensor],
        pos: Optional[torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        r_hat: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_src.device
        n_edges = int(edge_src.numel())
        score = torch.zeros(n_edges, dtype=torch.float32, device=device)
        field_proj = torch.zeros(n_edges, dtype=torch.float32, device=device)
        if (
            not isinstance(atom_pred, torch.Tensor)
            or atom_pred.numel() == 0
            or atom_pred.dim() < 2
            or atom_pred.size(-1) <= 7
            or pos is None
            or r_hat is None
            or n_edges == 0
        ):
            return score, field_proj
        pos_dev = pos.to(device=device, dtype=torch.float32)
        charges = atom_pred.to(device=device, dtype=torch.float32)[: pos_dev.size(0), 7]
        mid = 0.5 * (pos_dev[edge_src.long()] + pos_dev[edge_dst.long()])
        diff = mid[:, None, :] - pos_dev[None, :, :]
        dist = diff.norm(dim=-1).clamp(min=1e-3)
        valid = torch.ones_like(dist, dtype=torch.bool)
        valid.scatter_(1, edge_src.long()[:, None], False)
        valid.scatter_(1, edge_dst.long()[:, None], False)
        valid = valid & (dist > 0.7)
        contrib = charges[None, :, None] * diff / dist.clamp(min=1e-3).pow(3).unsqueeze(-1)
        contrib = torch.where(valid.unsqueeze(-1), contrib, torch.zeros_like(contrib))
        field = contrib.sum(dim=1)
        field_proj = (field * r_hat.to(device=device, dtype=torch.float32)).sum(dim=-1)
        raw = field_proj.abs()
        positive = raw[torch.isfinite(raw) & (raw > 0)]
        if positive.numel() > 0:
            qhi = torch.quantile(positive, 0.90) if positive.numel() >= 4 else positive.max()
            score = (raw / qhi.clamp(min=1e-6)).clamp(0.0, 1.0)
        return score.clamp(0.0, 1.0), field_proj

    def _hij_physical_evidence(
        self,
        nbo: Dict[str, torch.Tensor],
        skeleton_data: Optional[Any],
        z: Optional[torch.Tensor],
        atom_pred: Optional[torch.Tensor],
        pos: Optional[torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        r_hat: Optional[torch.Tensor],
        matched_bond_ids: torch.Tensor,
        class_names: Sequence[str],
        is_xh: torch.Tensor,
        amide_mask: torch.Tensor,
        occ_deviation: torch.Tensor,
        raw_delta_ratio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = edge_src.device
        n_edges = int(edge_src.numel())
        zeros = torch.zeros(n_edges, dtype=torch.float32, device=device)
        out = {
            "soften_score": zeros.clone(),
            "harden_score": zeros.clone(),
            "stark_score": zeros.clone(),
            "interaction_score": zeros.clone(),
            "interaction_e2": zeros.clone(),
            "field_score": zeros.clone(),
            "field_proj": zeros.clone(),
            "soften_mask": torch.zeros(n_edges, dtype=torch.bool, device=device),
            "harden_mask": torch.zeros(n_edges, dtype=torch.bool, device=device),
        }
        if not bool(getattr(self, "hij_physical_rules", True)) or n_edges == 0 or z is None:
            return out

        interaction_score, raw_e2, _ = self._hij_bond_interaction_exposure(
            nbo, skeleton_data, matched_bond_ids, int(z.numel()),
        )
        field_score, field_proj = self._hij_bond_field_score(atom_pred, pos, edge_src, edge_dst, r_hat)
        out["interaction_score"] = interaction_score
        out["interaction_e2"] = raw_e2
        out["field_score"] = field_score
        out["field_proj"] = field_proj

        z_dev = z.to(device).long()
        zi = z_dev[edge_src.long()]
        zj = z_dev[edge_dst.long()]
        one_h = (zi == 1) ^ (zj == 1)
        heavy_z = torch.where(zi == 1, zj, zi)
        min_z = torch.minimum(zi, zj)
        max_z = torch.maximum(zi, zj)
        halogen = torch.tensor([9, 17, 35, 53], dtype=torch.long, device=device)
        is_c_halogen = ((min_z == 6) & torch.isin(max_z, halogen))
        is_ch_like = one_h & ((heavy_z == 6) | (heavy_z == 14))
        is_cn = ((min_z == 6) & (max_z == 7))
        is_nitrile = torch.zeros(n_edges, dtype=torch.bool, device=device)
        if pos is not None and is_cn.any():
            pos_dev = pos.to(device)
            dist = (pos_dev[edge_dst.long()] - pos_dev[edge_src.long()]).norm(dim=-1)
            is_nitrile = is_cn & (dist <= 1.25) & (~amide_mask.to(device).bool())
        is_carbonyl = torch.tensor(
            [cls in {"carbonyl_co", "amide_co"} for cls in class_names],
            dtype=torch.bool,
            device=device,
        )
        polar_probe = is_carbonyl | is_nitrile | is_c_halogen
        hard_probe = is_ch_like | is_c_halogen

        occ_dev = occ_deviation.to(device=device, dtype=torch.float32)
        occ_drop_score = (-occ_dev / 1.5).clamp(0.0, 1.0)
        occ_gain_score = (occ_dev / 1.5).clamp(0.0, 1.0)
        ratio_negative = (raw_delta_ratio.to(device=device, dtype=torch.float32) < 0.0)
        ratio_positive = (raw_delta_ratio.to(device=device, dtype=torch.float32) > 0.0)
        min_score = max(float(self.hij_physical_min_score), 0.0)

        # Direct NBO interaction is strongest evidence.  A large projected
        # local field is only used as secondary Stark-like evidence, and only
        # on known probe bonds.
        direct_score = torch.maximum(
            interaction_score,
            polar_probe.to(torch.float32) * field_score * float(self.hij_stark_gate_scale),
        ).clamp(0.0, 1.0)
        class_soft_weight = torch.zeros(n_edges, dtype=torch.float32, device=device)
        class_soft_weight = torch.where(is_carbonyl, torch.full_like(class_soft_weight, 0.80), class_soft_weight)
        class_soft_weight = torch.where(is_nitrile, torch.full_like(class_soft_weight, 0.70), class_soft_weight)
        class_soft_weight = torch.where(is_c_halogen, torch.full_like(class_soft_weight, 0.60), class_soft_weight)
        weaken_evidence = direct_score * class_soft_weight * (0.25 + 0.75 * occ_drop_score)
        weaken_ok = (
            polar_probe
            & (~is_xh.to(device).bool())
            & (direct_score >= min_score)
            & ((occ_drop_score >= 0.15) | ratio_negative)
        )
        soften_score = torch.where(weaken_ok, weaken_evidence, zeros).clamp(0.0, 1.0)

        class_hard_weight = torch.zeros(n_edges, dtype=torch.float32, device=device)
        class_hard_weight = torch.where(is_ch_like, torch.full_like(class_hard_weight, 0.55), class_hard_weight)
        class_hard_weight = torch.where(is_c_halogen, torch.maximum(class_hard_weight, torch.full_like(class_hard_weight, 0.45)), class_hard_weight)
        hard_evidence = direct_score * class_hard_weight * occ_gain_score
        hard_ok = (
            hard_probe
            & (~is_xh.to(device).bool())
            & (direct_score >= min_score)
            & ((occ_gain_score >= 0.25) | ratio_positive)
        )
        harden_score = torch.where(hard_ok, hard_evidence, zeros).clamp(0.0, 1.0)

        out["soften_score"] = soften_score
        out["harden_score"] = harden_score
        out["stark_score"] = (polar_probe.to(torch.float32) * field_score).clamp(0.0, 1.0)
        out["soften_mask"] = soften_score > 0.0
        out["harden_mask"] = harden_score > 0.0
        return out

    def _hij_xh_interaction_softening_signal(
        self,
        nbo: Dict[str, torch.Tensor],
        skeleton_data: Optional[Any],
        z: torch.Tensor,
        pos: Optional[torch.Tensor],
        atom_bond_local: Optional[torch.Tensor],
        matched_bond_ids: torch.Tensor,
        is_xh: torch.Tensor,
        hbond_details: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = matched_bond_ids.device
        dtype = torch.float32
        n_edges = int(matched_bond_ids.numel())
        score = torch.zeros(n_edges, dtype=dtype, device=device)
        raw_e2 = torch.zeros(n_edges, dtype=dtype, device=device)
        e2_score = torch.zeros(n_edges, dtype=dtype, device=device)
        geom_score = torch.zeros(n_edges, dtype=dtype, device=device)
        pair_e2 = torch.zeros(n_edges, dtype=dtype, device=device)
        field_e2 = torch.zeros(n_edges, dtype=dtype, device=device)
        acceptor_atom_out = torch.full((n_edges,), -1, dtype=torch.long, device=device)
        if n_edges == 0 or z is None or not is_xh.any():
            return score, raw_e2, e2_score

        interaction_pred = nbo.get("interaction_pred") if isinstance(nbo, dict) else None
        interaction_edge_index = self._get_skeleton_tensor(nbo, skeleton_data, "interaction_edge_index", device)
        node_type = self._get_skeleton_tensor(nbo, skeleton_data, "node_type", device)
        atom_to_nbo = self._get_skeleton_tensor(nbo, skeleton_data, "atom_to_nbo_index", device)

        e2 = None
        qlo = torch.tensor(0.0, device=device, dtype=dtype)
        qhi = torch.tensor(1.0, device=device, dtype=dtype)
        pair_to_e2: Dict[Tuple[int, int], float] = {}
        node_field: Optional[torch.Tensor] = None
        if isinstance(interaction_pred, torch.Tensor) and interaction_edge_index is not None and interaction_edge_index.numel() > 0:
            e2 = interaction_pred.to(device=device, dtype=dtype)
            if e2.dim() > 1:
                e2 = e2[:, 0]
            e2 = torch.nan_to_num(e2, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            positive = e2[torch.isfinite(e2) & (e2 > 0)]
            if positive.numel() >= 4:
                qlo = torch.quantile(
                    positive,
                    min(max(float(self.hij_interaction_e2_low_quantile), 0.0), 0.99),
                )
                qhi = torch.quantile(
                    positive,
                    min(max(float(self.hij_interaction_e2_high_quantile), 0.01), 1.0),
                )
            elif positive.numel() > 0:
                qlo = positive.min()
                qhi = positive.max()
            else:
                qlo = torch.tensor(0.0, device=device, dtype=dtype)
                qhi = torch.tensor(1.0, device=device, dtype=dtype)

            ie = interaction_edge_index.to(device=device, dtype=torch.long)
            max_node = int(ie.max().detach().cpu().item()) + 1 if ie.numel() > 0 else int(z.numel())
            node_field = torch.zeros(max_node, dtype=dtype, device=device)
            src_l = ie[0].detach().cpu().tolist()
            dst_l = ie[1].detach().cpu().tolist()
            e2_l = e2.detach().cpu().tolist()
            for src_i, dst_i, val_i in zip(src_l, dst_l, e2_l):
                src_i = int(src_i)
                dst_i = int(dst_i)
                val_f = float(val_i)
                if val_f <= 0.0:
                    continue
                key = (src_i, dst_i)
                if val_f > pair_to_e2.get(key, 0.0):
                    pair_to_e2[key] = val_f
                if 0 <= src_i < max_node and val_f > float(node_field[src_i].item()):
                    node_field[src_i] = val_f
                if 0 <= dst_i < max_node and val_f > float(node_field[dst_i].item()):
                    node_field[dst_i] = val_f
        denom = (qhi - qlo).abs().clamp(min=1e-6)

        num_atoms = int(z.numel())
        h_atoms = hbond_details.get("h_atom", torch.full_like(matched_bond_ids, -1)).to(device).long()
        donor_atoms = hbond_details.get("donor_atom", torch.full_like(matched_bond_ids, -1)).to(device).long()
        acceptors = hbond_details.get("acceptor_atom", torch.full_like(matched_bond_ids, -1)).to(device).long()
        if isinstance(node_type, torch.Tensor):
            node_type_l = node_type.to(device).long()
        else:
            node_type_l = None
        if isinstance(atom_to_nbo, torch.Tensor):
            atom_to_nbo_l = atom_to_nbo.to(device).long()
        else:
            atom_to_nbo_l = None

        def _acceptor_nodes(atom_idx: int) -> List[int]:
            nodes: List[int] = [int(atom_idx)]
            if atom_to_nbo_l is not None and atom_to_nbo_l.numel() > 0:
                mask = atom_to_nbo_l[0] == int(atom_idx)
                for node in atom_to_nbo_l[1, mask].detach().cpu().tolist():
                    node_i = int(node)
                    if node_type_l is None or (0 <= node_i < int(node_type_l.numel()) and int(node_type_l[node_i].item()) == 2):
                        nodes.append(node_i)
            return sorted(set(nodes))

        z_dev = z.to(device).long()
        pos_dev = pos.to(device).float() if isinstance(pos, torch.Tensor) else None
        acceptor_mask = torch.zeros(num_atoms, dtype=torch.bool, device=device)
        for elem in self.hij_hbond_acceptors:
            acceptor_mask = acceptor_mask | (z_dev == int(elem))
        dmax = max(float(self.hij_interaction_acceptor_distance), float(self.hij_hbond_distance))
        angle_min = min(max(float(self.hij_interaction_angle_min), 0.0), 179.0)
        field_weight = min(max(float(self.hij_interaction_field_weight), 0.0), 1.0)

        def _acceptor_weight(atom_idx: int) -> float:
            elem = int(z_dev[atom_idx].detach().cpu().item())
            if elem in {7, 8, 9}:
                return 1.0
            if elem in {15, 16, 17, 35, 53}:
                return 0.75
            return 0.60

        def _geometry_candidate(i: int) -> Tuple[float, int]:
            h_i = int(h_atoms[i].detach().cpu().item())
            donor_i = int(donor_atoms[i].detach().cpu().item())
            if pos_dev is None or h_i < 0 or donor_i < 0:
                geom0 = hbond_details.get("geometry_score")
                acc0 = acceptors[i] if isinstance(acceptors, torch.Tensor) else torch.tensor(-1, device=device)
                g = float(geom0[i].detach().cpu().item()) if isinstance(geom0, torch.Tensor) and geom0.numel() > i else 0.0
                return g, int(acc0.detach().cpu().item())
            best_g = 0.0
            best_a = -1
            h_pos = pos_dev[h_i]
            d_pos = pos_dev[donor_i]
            for a_i in torch.nonzero(acceptor_mask, as_tuple=False).view(-1).detach().cpu().tolist():
                a_i = int(a_i)
                if a_i == h_i or a_i == donor_i:
                    continue
                a_pos = pos_dev[a_i]
                ha = float((a_pos - h_pos).norm().detach().cpu().item())
                if ha < 1.15 or ha > dmax:
                    continue
                v_hd = d_pos - h_pos
                v_ha = a_pos - h_pos
                cosang = (v_hd * v_ha).sum() / (v_hd.norm().clamp(min=1e-6) * v_ha.norm().clamp(min=1e-6))
                ang = float((torch.acos(cosang.clamp(-1.0, 1.0)) * (180.0 / torch.pi)).detach().cpu().item())
                if ang < angle_min:
                    continue
                dist_score = ((dmax - ha) / max(dmax - 1.55, 1e-6))
                angle_score = ((ang - angle_min) / max(180.0 - angle_min, 1e-6))
                g = max(0.0, min(1.0, (dist_score * angle_score) ** 0.5))
                g *= _acceptor_weight(a_i)
                if g > best_g:
                    best_g = g
                    best_a = a_i
            return best_g, best_a

        xh_idx = torch.nonzero(is_xh.to(device).bool(), as_tuple=False).view(-1)
        for idx_t in xh_idx.detach().cpu().tolist():
            i = int(idx_t)
            bond_node = num_atoms + int(matched_bond_ids[i].detach().cpu().item())
            h_node = int(h_atoms[i].detach().cpu().item())
            geom_i, geom_acc = _geometry_candidate(i)
            acc_atom = geom_acc if geom_acc >= 0 else int(acceptors[i].detach().cpu().item())
            if acc_atom < 0:
                continue
            geom_score[i] = float(geom_i)
            acceptor_atom_out[i] = int(acc_atom)
            acceptor_nodes = _acceptor_nodes(acc_atom)
            candidates: List[Tuple[int, int]] = []
            for acc_node in acceptor_nodes:
                candidates.extend([
                    (acc_node, bond_node),
                    (bond_node, acc_node),
                    (acc_node, h_node),
                    (h_node, acc_node),
                ])
            max_e2 = 0.0
            for key in candidates:
                max_e2 = max(max_e2, pair_to_e2.get(key, 0.0))
            pair_e2[i] = float(max_e2)
            if node_field is not None and 0 <= bond_node < int(node_field.numel()):
                field_e2[i] = node_field[bond_node] * field_weight
            raw_e2[i] = torch.maximum(pair_e2[i], field_e2[i])

        quantile_score = ((raw_e2 - qlo) / denom).clamp(0.0, 1.0)
        relative_score = (raw_e2 / qhi.abs().clamp(min=1e-6)).clamp(0.0, 1.0)
        e2_score = torch.maximum(quantile_score, 0.75 * relative_score).clamp(0.0, 1.0)
        geom_weight = min(max(float(self.hij_interaction_geometry_weight), 0.0), 1.0)
        mixed = geom_score * (geom_weight + (1.0 - geom_weight) * e2_score)
        score = torch.where(is_xh.to(device).bool(), mixed, score)
        score = torch.where(score >= float(self.hij_interaction_min_score), score, torch.zeros_like(score))
        hbond_details["interaction_geometry_score"] = geom_score.detach()
        hbond_details["interaction_pair_e2"] = pair_e2.detach()
        hbond_details["interaction_field_e2"] = field_e2.detach()
        hbond_details["interaction_acceptor_atom"] = acceptor_atom_out.detach()
        return score.clamp(0.0, 1.0), raw_e2, e2_score

    def _hij_bond_context(
        self,
        z: Optional[torch.Tensor],
        atom_bond_local: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = edge_src.device
        n_edges = int(edge_src.numel())
        if z is None:
            zeros = torch.zeros(n_edges, dtype=torch.bool, device=device)
            return ["unknown"] * n_edges, zeros, zeros, zeros, zeros, zeros.float(), zeros

        z = z.to(device).long()
        zi = z[edge_src]
        zj = z[edge_dst]
        one_h = (zi == 1) ^ (zj == 1)
        heavy_z = torch.where(zi == 1, zj, zi)
        is_xh = one_h & ((heavy_z == 7) | (heavy_z == 8))
        is_oh = one_h & (heavy_z == 8)
        is_nh = one_h & (heavy_z == 7)
        amide_mask = self._amide_bond_mask(z, atom_bond_local, edge_src, edge_dst, pos=pos)

        is_co = ((zi == 6) & (zj == 8)) | ((zi == 8) & (zj == 6))
        is_cn = ((zi == 6) & (zj == 7)) | ((zi == 7) & (zj == 6))
        carbonyl_like = is_co.clone()
        if pos is not None and n_edges > 0:
            pos_dev = pos.to(device)
            dist = (pos_dev[edge_dst] - pos_dev[edge_src]).norm(dim=-1)
            carbonyl_like = carbonyl_like & (dist <= 1.45)

        min_z = torch.minimum(zi, zj).detach().cpu().tolist()
        max_z = torch.maximum(zi, zj).detach().cpu().tolist()
        is_oh_l = is_oh.detach().cpu().tolist()
        is_nh_l = is_nh.detach().cpu().tolist()
        one_h_l = one_h.detach().cpu().tolist()
        amide_l = amide_mask.detach().cpu().tolist()
        is_co_l = is_co.detach().cpu().tolist()
        is_cn_l = is_cn.detach().cpu().tolist()
        carbonyl_l = carbonyl_like.detach().cpu().tolist()

        class_names: List[str] = []
        for i in range(n_edges):
            if is_oh_l[i]:
                cls = "xh_oh"
            elif is_nh_l[i]:
                cls = "xh_nh"
            elif one_h_l[i]:
                cls = f"xh_z{int(max_z[i])}"
            elif amide_l[i] and is_co_l[i]:
                cls = "amide_co"
            elif amide_l[i] and is_cn_l[i]:
                cls = "amide_cn"
            elif carbonyl_l[i]:
                cls = "carbonyl_co"
            else:
                cls = f"pair_{int(min_z[i])}_{int(max_z[i])}"
            class_names.append(cls)

        atom_ood = self._atom_training_ood_score(z, atom_bond_local)
        edge_env_ood = torch.maximum(atom_ood[edge_src], atom_ood[edge_dst])
        unknown_atom = edge_env_ood >= 0.999

        hbond_mask, hbond_atom_score = self._hbond_environment(
            z, pos, atom_bond_local, edge_src, edge_dst, is_xh,
        )
        if hbond_atom_score.numel() == z.numel():
            hbond_edge_score = torch.maximum(hbond_atom_score[edge_src], hbond_atom_score[edge_dst])
            edge_env_ood = torch.maximum(edge_env_ood, hbond_edge_score)

        known_classes = self._effective_train_bond_classes()
        if known_classes:
            unknown_class = torch.tensor(
                [cls not in known_classes for cls in class_names],
                dtype=torch.bool,
                device=device,
            )
        else:
            unknown_class = torch.zeros(n_edges, dtype=torch.bool, device=device)

        return class_names, is_xh, amide_mask, unknown_atom, unknown_class, edge_env_ood.clamp(0.0, 1.0), hbond_mask

    def _hij_chemistry_gate(
        self,
        class_names: Sequence[str],
        is_xh: torch.Tensor,
        amide_mask: torch.Tensor,
        unknown_atom: torch.Tensor,
        unknown_class: torch.Tensor,
        env_ood_score: Optional[torch.Tensor] = None,
        hbond_mask: Optional[torch.Tensor] = None,
        trigger: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = is_xh.device
        gate = torch.full(
            (len(class_names),),
            float(self.hij_non_target_scale),
            device=device,
            dtype=torch.float32,
        )
        gate[is_xh] = float(self.hij_xh_scale)
        gate[amide_mask] = float(self.hij_amide_scale)

        if str(self.hij_gate_mode).lower() in {"legacy_xh", "xh_only"} or self.hij_target_xh_only:
            legacy = torch.full_like(gate, float(self.hij_non_target_scale))
            legacy[is_xh] = float(self.hij_xh_scale)
            gate = legacy
            gate[amide_mask] = float(self.hij_amide_scale)

        unknown_env = unknown_atom | unknown_class
        if unknown_env.any():
            gate[unknown_env] = torch.maximum(
                gate[unknown_env],
                torch.full_like(gate[unknown_env], float(self.hij_unknown_scale)),
            )
        if env_ood_score is not None:
            env = env_ood_score.to(device=device, dtype=gate.dtype).clamp(0.0, 1.0)
            env_target = torch.full_like(gate, float(self.hij_unknown_scale))
            gate = gate + (env_target - gate) * env
        if hbond_mask is not None and hbond_mask.any():
            hb = hbond_mask.to(device).bool()
            gate[hb] = torch.maximum(
                gate[hb],
                torch.full_like(gate[hb], float(self.hij_hbond_scale)),
            )

        if trigger is not None and str(self.hij_gate_mode).lower() == "adaptive":
            strong_thr = min(max(float(self.hij_strong_ood_threshold), 0.0), 0.999)
            strong = ((trigger.to(device=device, dtype=gate.dtype) - strong_thr) / (1.0 - strong_thr)).clamp(0.0, 1.0)
            target = torch.maximum(gate, torch.full_like(gate, float(self.hij_unknown_scale)))
            gate = gate + (target - gate) * strong

        return gate.clamp(0.0, max(1.0, float(self.hij_unknown_scale)))

    def _amide_bond_mask(
        self,
        z: torch.Tensor,
        atom_bond_local: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if atom_bond_local is None or atom_bond_local.numel() == 0 or z.numel() == 0:
            return torch.zeros(edge_src.numel(), dtype=torch.bool, device=edge_src.device)

        device = edge_src.device
        bonds = atom_bond_local.to(device).long()
        b0, b1 = bonds[0], bonds[1]
        z0, z1 = z[b0], z[b1]
        has_pos = pos is not None
        if has_pos:
            pos = pos.to(device)
            bdist = (pos[b1] - pos[b0]).norm(dim=-1)
            carbonyl_len = bdist <= 1.45
            amide_cn_len = bdist <= 1.65
        else:
            carbonyl_len = torch.ones_like(b0, dtype=torch.bool)
            amide_cn_len = torch.ones_like(b0, dtype=torch.bool)

        co_bond = (((z0 == 6) & (z1 == 8)) | ((z0 == 8) & (z1 == 6))) & carbonyl_len
        cn_bond = (((z0 == 6) & (z1 == 7)) | ((z0 == 7) & (z1 == 6))) & amide_cn_len
        c_in_co = torch.where(z0 == 6, b0, b1)[co_bond]
        c_in_cn = torch.where(z0 == 6, b0, b1)[cn_bond]
        if c_in_co.numel() == 0 or c_in_cn.numel() == 0:
            return torch.zeros(edge_src.numel(), dtype=torch.bool, device=device)

        n_nodes = int(z.numel())
        carbonyl_c = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        cn_c = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        carbonyl_c[c_in_co] = True
        cn_c[c_in_cn] = True
        amide_c = carbonyl_c & cn_c
        if not amide_c.any():
            return torch.zeros(edge_src.numel(), dtype=torch.bool, device=device)

        ei, ej = edge_src.long(), edge_dst.long()
        ezi, ezj = z[ei], z[ej]
        c_atom = torch.where(ezi == 6, ei, ej)
        is_co_edge = ((ezi == 6) & (ezj == 8)) | ((ezi == 8) & (ezj == 6))
        is_cn_edge = ((ezi == 6) & (ezj == 7)) | ((ezi == 7) & (ezj == 6))
        return (is_co_edge | is_cn_edge) & amide_c[c_atom]

    def _hij_mode_gate(
        self,
        mode_freq: Optional[torch.Tensor],
        modes: Optional[torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        r_hat: Optional[torch.Tensor],
        is_xh: Optional[torch.Tensor] = None,
        amide_mask: Optional[torch.Tensor] = None,
        env_ood_score: Optional[torch.Tensor] = None,
        hbond_mask: Optional[torch.Tensor] = None,
        trigger: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = torch.ones(edge_src.numel(), device=edge_src.device)
        if not self.hij_mode_aware or mode_freq is None or modes is None or r_hat is None:
            return gate
        if modes.numel() == 0 or mode_freq.numel() == 0:
            return gate
        if is_xh is None:
            is_xh = torch.ones(edge_src.numel(), dtype=torch.bool, device=edge_src.device)
        else:
            is_xh = is_xh.to(edge_src.device).bool()
        if amide_mask is None:
            amide_mask = torch.zeros(edge_src.numel(), dtype=torch.bool, device=edge_src.device)
        else:
            amide_mask = amide_mask.to(edge_src.device).bool()
        if env_ood_score is None:
            env_ood_score = torch.zeros(edge_src.numel(), device=edge_src.device)
        else:
            env_ood_score = env_ood_score.to(edge_src.device).float().clamp(0.0, 1.0)
        if hbond_mask is None:
            hbond_mask = torch.zeros(edge_src.numel(), dtype=torch.bool, device=edge_src.device)
        else:
            hbond_mask = hbond_mask.to(edge_src.device).bool()
        if trigger is None:
            trigger_for_floor = torch.zeros(edge_src.numel(), device=edge_src.device)
        else:
            trigger_for_floor = trigger.to(edge_src.device).float().clamp(0.0, 1.0)

        mode_freq = mode_freq.to(edge_src.device)
        modes = modes.to(edge_src.device)
        thr = max(float(self.hij_mode_participation_threshold), 1e-12)

        def _window_gate(freq_min: float, freq_max: float) -> torch.Tensor:
            window = (mode_freq >= float(freq_min)) & (mode_freq <= float(freq_max))
            if not window.any():
                return torch.zeros_like(gate)
            mode_sel = modes[window]  # [M, N, 3]
            rel = mode_sel[:, edge_dst.long(), :] - mode_sel[:, edge_src.long(), :]
            stretch = (rel * r_hat.unsqueeze(0)).sum(dim=-1).pow(2)
            denom = mode_sel.pow(2).sum(dim=(1, 2)).clamp(min=1e-12).unsqueeze(-1)
            participation = stretch / denom
            max_part = participation.max(dim=0).values
            return (max_part / thr).clamp(0.0, 1.0)

        if is_xh.any():
            xh_gate = _window_gate(self.hij_xh_freq_min, self.hij_xh_freq_max)
            gate[is_xh] = xh_gate[is_xh]
            if hbond_mask.any():
                hb_floor = min(max(float(self.hij_hbond_mode_floor), 0.0), 1.0)
                hb_xh = hbond_mask & is_xh
                if hb_xh.any():
                    gate[hb_xh] = torch.maximum(
                        gate[hb_xh],
                        torch.full_like(gate[hb_xh], hb_floor),
                    )

        if amide_mask.any():
            amide_gate = _window_gate(self.hij_amide_freq_min, self.hij_amide_freq_max)
            floor = min(max(float(self.hij_amide_mode_floor), 0.0), 1.0)
            adaptive = torch.maximum(env_ood_score, trigger_for_floor)
            unknown_floor = min(max(float(self.hij_unknown_mode_floor), 0.0), 1.0)
            amide_floor = floor + (unknown_floor - floor) * adaptive
            gate[amide_mask] = torch.maximum(amide_gate[amide_mask], amide_floor[amide_mask])

        # Backbone bonds (non-XH, non-amide): clamp to backbone_mode_floor.
        # Known ID backbone bonds are usually well-predicted.  Rare-atom or
        # hydrogen-bond environments relax this cap instead of being hard off.
        backbone = ~(is_xh | amide_mask)
        if backbone.any():
            bb_floor = min(max(float(self.hij_backbone_mode_floor), 0.0), 1.0)
            unknown_floor = min(max(float(self.hij_unknown_mode_floor), 0.0), 1.0)
            adaptive = torch.maximum(env_ood_score, trigger_for_floor)
            backbone_cap = bb_floor + (unknown_floor - bb_floor) * adaptive
            gate[backbone] = torch.minimum(gate[backbone], backbone_cap[backbone])

        return gate

    def compensate_hii_from_hij_delta(
        self,
        hii_pred: torch.Tensor,
        hij_delta: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        strength = float(getattr(self, "hij_hii_comp_strength", 0.0))
        if strength <= 0.0 or hij_delta is None or hij_delta.numel() == 0:
            return hii_pred
        device = hii_pred.device
        delta = hij_delta.to(device=device, dtype=hii_pred.dtype)
        if not torch.isfinite(delta).all():
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        dst = edge_index.to(device)[1].long()
        correction = torch.zeros_like(hii_pred)
        correction.index_add_(0, dst, -delta)
        return hii_pred + strength * correction

    def calibrate_hij(
        self,
        hij_pred: torch.Tensor,
        nbo: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        num_nodes: int,
        pos: Optional[torch.Tensor] = None,
        enk_ood_score: Optional[torch.Tensor] = None,
        enk_context: Optional[Dict[str, Any]] = None,
        z: Optional[torch.Tensor] = None,
        mode_freq: Optional[torch.Tensor] = None,
        modes: Optional[torch.Tensor] = None,
        skeleton_data: Optional[Any] = None,
    ) -> torch.Tensor:
        self.last_hij_delta = None
        self.last_hij_debug = {}
        atom_pred = nbo.get("atom_pred")
        bond_pred = nbo.get("bond_pred")
        atom_bond_local = nbo.get("atom_bond_local")

        if bond_pred is None or atom_bond_local is None:
            return hij_pred
        if bond_pred.numel() == 0 or atom_bond_local.numel() == 0:
            return hij_pred

        device = hij_pred.device
        bond_pred = bond_pred.to(device)
        atom_bond_local = atom_bond_local.to(device)
        edge_index = edge_index.to(device)

        bond_ids, bond_mask = _match_undirected_edges_simple(
            edge_index, atom_bond_local, num_nodes,
        )
        if not bond_mask.any():
            return hij_pred

        matched_bond_ids = bond_ids[bond_mask]
        edge_src_bond = edge_index[0, bond_mask].long()
        edge_dst_bond = edge_index[1, bond_mask].long()
        occ_col = _bond_occupancy_col(int(bond_pred.size(1)), self.bond_occ_col_policy)
        if occ_col is None:
            return hij_pred
        bd_occ = bond_pred[:, occ_col].clamp(min=0.01)
        matched_occ = bd_occ[matched_bond_ids].clamp(min=0.01)
        r_hat = None
        hij_bond_block = hij_pred[bond_mask]

        if pos is not None:
            pos = pos.to(device)
            r_vec = pos[edge_dst_bond] - pos[edge_src_bond]
            r_hat = r_vec / r_vec.norm(dim=-1, keepdim=True).clamp(min=0.01)
            hij_long = torch.einsum('bi,bij,bj->b', r_hat, hij_bond_block, r_hat)
            hij_strength = hij_long.abs().clamp(min=1e-10)
            stat_mean_key = "log_hij_long_mean"
            stat_std_key = "log_hij_long_std"
            fallback_mean_key = "log_hij_bond_mean"
            fallback_std_key = "log_hij_bond_std"
        else:
            hij_norm = hij_pred.norm(dim=(-2, -1)).clamp(min=1e-10)
            hij_strength = hij_norm[bond_mask]
            stat_mean_key = fallback_mean_key = "log_hij_bond_mean"
            stat_std_key = fallback_std_key = "log_hij_bond_std"

        (
            class_names,
            is_xh,
            amide_mask,
            unknown_atom,
            unknown_class,
            env_ood_score,
            hbond_mask,
        ) = self._hij_bond_context(
            z, atom_bond_local, edge_src_bond, edge_dst_bond, pos=pos,
        )
        if z is not None:
            hbond_details = self._hbond_details(
                z.to(device).long(), pos, atom_bond_local,
                edge_src_bond, edge_dst_bond, is_xh,
            )
            if isinstance(hbond_details.get("edge_mask"), torch.Tensor):
                hbond_mask = hbond_mask | hbond_details["edge_mask"].to(device).bool()
        else:
            hbond_details = {
                "edge_mask": torch.zeros_like(is_xh, dtype=torch.bool),
                "geometry_score": torch.zeros_like(is_xh, dtype=torch.float32),
            }
        env_signatures = self._hij_environment_signatures(
            z, atom_bond_local, edge_src_bond, edge_dst_bond, class_names,
        )
        seen_env_signatures = self._effective_train_env_signatures()
        if seen_env_signatures:
            unknown_env_signature = torch.tensor(
                [sig not in seen_env_signatures for sig in env_signatures],
                dtype=torch.bool,
                device=device,
            )
        else:
            unknown_env_signature = torch.zeros_like(unknown_class)
        if unknown_env_signature.any():
            env_ood_score = torch.maximum(
                env_ood_score,
                unknown_env_signature.to(device=device, dtype=env_ood_score.dtype),
            )

        if self.train_stats is not None:
            log_hij = torch.log(hij_strength)
            log_occ = torch.log(matched_occ)
            mu_h = self._class_stat_tensor(
                class_names, stat_mean_key, fallback_mean_key, 0.0, device, log_hij.dtype,
            )
            sig_h = self._class_stat_tensor(
                class_names, stat_std_key, fallback_std_key, 1.0, device, log_hij.dtype,
            ).clamp(min=1e-6)
            mu_o = self._class_stat_tensor(
                class_names, "log_occ_mean", "log_occ_mean", 0.0, device, log_occ.dtype,
            )
            sig_o = self._class_stat_tensor(
                class_names, "log_occ_std", "log_occ_std", 1.0, device, log_occ.dtype,
            ).clamp(min=1e-6)
            z_hij = (log_hij - mu_h) / sig_h
            z_occ = (log_occ - mu_o) / sig_o
            occ_deviation = z_occ
            nbo_implied_z = self.bond_order_exponent * z_occ
            delta_z = nbo_implied_z - z_hij
            ratio_bond = (1.0 + delta_z).clamp(
                1.0 - self.clamp_ratio, 1.0 + self.clamp_ratio,
            )
        else:
            avg_hij_bond = hij_strength.mean().clamp(min=1e-10)
            avg_occ = bd_occ.mean().clamp(min=0.01)
            nbo_implied_scale = (matched_occ / avg_occ) ** self.bond_order_exponent
            pred_relative_bond = (hij_strength / avg_hij_bond).clamp(min=1e-10)
            ratio_bond = (nbo_implied_scale / pred_relative_bond).clamp(
                1.0 - self.clamp_ratio, 1.0 + self.clamp_ratio,
            )
            delta_z = (ratio_bond - 1.0) / max(float(self.clamp_ratio), 1e-6)
            occ_deviation = delta_z

        raw_delta_ratio = float(self.alpha_hij) * (ratio_bond - 1.0)
        policy_delta_ratio = raw_delta_ratio
        max_delta = abs(float(self.alpha_hij) * float(self.clamp_ratio))
        xh_interaction_score = torch.zeros_like(raw_delta_ratio)
        xh_interaction_e2 = torch.zeros_like(raw_delta_ratio)
        xh_interaction_e2_score = torch.zeros_like(raw_delta_ratio)
        policy = str(getattr(self, "hij_policy", "hybrid") or "hybrid").lower()
        if policy in {"hybrid", "interaction_rule"} and z is not None and is_xh.any():
            (
                xh_interaction_score,
                xh_interaction_e2,
                xh_interaction_e2_score,
            ) = self._hij_xh_interaction_softening_signal(
                nbo=nbo,
                skeleton_data=skeleton_data,
                z=z.to(device).long(),
                pos=pos,
                atom_bond_local=atom_bond_local,
                matched_bond_ids=matched_bond_ids,
                is_xh=is_xh,
                hbond_details=hbond_details,
            )
            weak_xh_delta = raw_delta_ratio * float(self.hij_xh_heuristic_scale)
            soften_delta = (
                -max_delta
                * float(self.hij_interaction_soften_scale)
                * xh_interaction_score.to(device=device, dtype=raw_delta_ratio.dtype)
            )
            if policy == "interaction_rule":
                xh_delta = torch.where(
                    xh_interaction_score > 0.0,
                    soften_delta,
                    weak_xh_delta,
                )
            else:
                mix = xh_interaction_score.to(device=device, dtype=raw_delta_ratio.dtype).clamp(0.0, 1.0)
                xh_delta = weak_xh_delta * (1.0 - mix) + soften_delta
            policy_delta_ratio = torch.where(is_xh, xh_delta, raw_delta_ratio)
            policy_delta_ratio = policy_delta_ratio.clamp(-max_delta, max_delta)
        interaction_xh_mask = (
            (xh_interaction_score.to(device) > 0.0) & is_xh.to(device).bool()
            if xh_interaction_score.numel() == raw_delta_ratio.numel()
            else torch.zeros_like(is_xh.to(device).bool())
        )
        physical = self._hij_physical_evidence(
            nbo=nbo,
            skeleton_data=skeleton_data,
            z=z.to(device).long() if z is not None else None,
            atom_pred=atom_pred,
            pos=pos,
            edge_src=edge_src_bond,
            edge_dst=edge_dst_bond,
            r_hat=r_hat,
            matched_bond_ids=matched_bond_ids,
            class_names=class_names,
            is_xh=is_xh,
            amide_mask=amide_mask,
            occ_deviation=occ_deviation,
            raw_delta_ratio=raw_delta_ratio,
        )
        physical_soften_score = physical["soften_score"].to(device=device, dtype=raw_delta_ratio.dtype)
        physical_harden_score = physical["harden_score"].to(device=device, dtype=raw_delta_ratio.dtype)
        if max_delta > 0.0 and physical_soften_score.numel() == policy_delta_ratio.numel():
            soften_mask = physical["soften_mask"].to(device).bool()
            harden_mask = physical["harden_mask"].to(device).bool()
            physical_soften_delta = (
                -max_delta
                * float(self.hij_physical_soften_scale)
                * physical_soften_score
            )
            physical_harden_delta = (
                max_delta
                * float(self.hij_physical_harden_scale)
                * physical_harden_score
            )
            policy_delta_ratio = torch.where(
                soften_mask,
                torch.minimum(policy_delta_ratio, physical_soften_delta),
                policy_delta_ratio,
            )
            policy_delta_ratio = torch.where(
                harden_mask,
                torch.maximum(policy_delta_ratio, physical_harden_delta),
                policy_delta_ratio,
            )
            policy_delta_ratio = policy_delta_ratio.clamp(-max_delta, max_delta)
        physical_special_score = torch.maximum(physical_soften_score, physical_harden_score)
        if physical.get("stark_score") is not None and physical["stark_score"].numel() == physical_special_score.numel():
            physical_special_score = torch.maximum(
                physical_special_score,
                0.5 * physical["stark_score"].to(device=device, dtype=physical_special_score.dtype),
            ).clamp(0.0, 1.0)
        physical_interaction_mask = physical_special_score > 0.0

        # --- NBO consensus weighting ---
        # When multiple independent bond-order indicators (occ, DI, LBO,
        # Mayer) AGREE, NBO is confident → the bond is ID (in-distribution)
        # → WEAK calibration (the model already handles ID bonds well).
        # When they DISAGREE, NBO is uncertain → the bond is OOD →
        # STRONGER calibration (the model needs NBO guidance here).
        matched_bp = bond_pred[matched_bond_ids].detach()
        n_cols = int(matched_bp.size(1))
        ind_0 = matched_bp[:, 0:1].clamp(min=0.0)
        col_max_0 = ind_0.max().clamp(min=1e-6)
        indicators = [ind_0 / col_max_0]
        if n_cols >= 16:
            for c in range(max(15, n_cols - 3), n_cols):
                col = matched_bp[:, c:c + 1].clamp(min=0.0)
                cmax = col.max().clamp(min=1e-6)
                indicators.append(col / cmax)
        if len(indicators) >= 2:
            stacked = torch.cat(indicators, dim=1)
            col_mean = stacked.mean(dim=1).clamp(min=0.01)
            col_std = stacked.std(dim=1)
            cv = col_std / col_mean
            consensus = torch.exp(-cv * 3.0)
        else:
            consensus = torch.ones_like(raw_delta_ratio)
        # Invert: low consensus (OOD) → strong calibration, high consensus (ID) → weak
        id_thr = float(self.hij_consensus_id_threshold)
        ood_thr = float(self.hij_consensus_ood_threshold)
        consensus_ood = ((id_thr - consensus) / max(id_thr - ood_thr, 1e-6)).clamp(0.0, 1.0)
        ratio_ood = (
            policy_delta_ratio.abs()
            / max(float(self.alpha_hij) * float(self.clamp_ratio), 1e-6)
        ).clamp(0.0, 1.0)
        physics_need = torch.maximum(
            ratio_ood,
            (delta_z.abs() / 2.0).clamp(0.0, 1.0),
        )
        if xh_interaction_score.numel() == physics_need.numel():
            physics_need = torch.maximum(
                physics_need,
                xh_interaction_score.to(device=device, dtype=physics_need.dtype),
            )
        if physical_special_score.numel() == physics_need.numel():
            physics_need = torch.maximum(
                physics_need,
                physical_special_score.to(device=device, dtype=physics_need.dtype),
            )

        atom_enk_ood = None
        atom_branch_dis = None
        if isinstance(enk_context, dict) and enk_context:
            maybe_ood = enk_context.get("atom_ood")
            if isinstance(maybe_ood, torch.Tensor) and maybe_ood.numel() == num_nodes:
                atom_enk_ood = maybe_ood.to(device=device, dtype=raw_delta_ratio.dtype)
            maybe_dis = enk_context.get("branch_disagreement")
            if isinstance(maybe_dis, torch.Tensor) and maybe_dis.numel() == num_nodes:
                atom_branch_dis = maybe_dis.to(device=device, dtype=raw_delta_ratio.dtype)
        if atom_enk_ood is None and enk_ood_score is not None and enk_ood_score.numel() == num_nodes:
            atom_enk_ood = enk_ood_score.to(device=device, dtype=raw_delta_ratio.dtype)
        if atom_enk_ood is None:
            edge_enk_need = torch.zeros_like(raw_delta_ratio)
        else:
            raw_edge_enk = 0.5 * (atom_enk_ood[edge_src_bond] + atom_enk_ood[edge_dst_bond])
            edge_enk_need = self._score_above_train(
                raw_edge_enk, "enk_atom_ood_mean", "enk_atom_ood_std",
            )
        if atom_branch_dis is not None:
            raw_edge_dis = 0.5 * (atom_branch_dis[edge_src_bond] + atom_branch_dis[edge_dst_bond])
            edge_enk_need = torch.maximum(
                edge_enk_need,
                self._score_above_train(
                    raw_edge_dis, "enk_branch_disagreement_mean", "enk_branch_disagreement_std",
                ),
            )

        unknown_env = unknown_atom | unknown_class | unknown_env_signature
        unknown_env_f = unknown_env.to(device=device, dtype=raw_delta_ratio.dtype)
        env_need = torch.maximum(
            env_ood_score.to(device=device, dtype=raw_delta_ratio.dtype),
            unknown_env_f,
        )
        if hbond_mask is not None:
            env_need = torch.maximum(env_need, hbond_mask.to(device=device, dtype=raw_delta_ratio.dtype))
        if xh_interaction_score.numel() == env_need.numel():
            env_need = torch.maximum(
                env_need,
                xh_interaction_score.to(device=device, dtype=raw_delta_ratio.dtype),
            )
        if physical_special_score.numel() == env_need.numel():
            env_need = torch.maximum(
                env_need,
                physical_special_score.to(device=device, dtype=raw_delta_ratio.dtype),
            )
        env_need = torch.maximum(env_need, edge_enk_need).clamp(0.0, 1.0)

        nbo_reliability = (0.35 + 0.65 * consensus.clamp(0.0, 1.0)).to(device=device, dtype=raw_delta_ratio.dtype)
        class_count = self._class_count_tensor(
            class_names, "log_hij_long_n", device, raw_delta_ratio.dtype,
        )
        if class_count.numel() == nbo_reliability.numel() and class_count.max().item() > 0:
            class_support = (class_count / 64.0).sqrt().clamp(0.35, 1.0)
            class_support = torch.where(unknown_class, torch.full_like(class_support, 0.55), class_support)
            nbo_reliability = nbo_reliability * class_support

        # The final trigger is a policy score, not just an anomaly score:
        # ENK/environment decides where the model needs help; NBO residual
        # decides whether there is a physically meaningful direction.
        trigger = torch.clamp(
            physics_need * nbo_reliability
            + env_need * (0.25 + 0.75 * physics_need) * nbo_reliability
            + 0.20 * consensus_ood,
            0.0,
            1.0,
        )

        gate_hbond_mask = hbond_mask
        if hbond_mask is not None and interaction_xh_mask.numel() == hbond_mask.numel():
            gate_hbond_mask = hbond_mask.to(device).bool() | interaction_xh_mask
        physical_direct_mask = (physical_soften_score > 0.0) | (physical_harden_score > 0.0)
        if physical_direct_mask.numel() == is_xh.numel():
            if gate_hbond_mask is None:
                gate_hbond_mask = physical_direct_mask.to(device).bool()
            else:
                gate_hbond_mask = gate_hbond_mask.to(device).bool() | physical_direct_mask.to(device).bool()
        chem_gate = self._hij_chemistry_gate(
            class_names,
            is_xh,
            amide_mask,
            unknown_atom,
            unknown_class,
            env_ood_score=env_need,
            hbond_mask=gate_hbond_mask,
            trigger=trigger,
        )
        mode_gate = self._hij_mode_gate(
            mode_freq, modes, edge_src_bond, edge_dst_bond, r_hat,
            is_xh=is_xh,
            amide_mask=amide_mask,
            env_ood_score=env_need,
            hbond_mask=gate_hbond_mask,
            trigger=trigger,
        )

        special_score = env_need.to(device=device, dtype=trigger.dtype).clamp(0.0, 1.0)
        gate_special_score = special_score
        if interaction_xh_mask.numel() == special_score.numel():
            # Interaction evidence already passed geometry and NBO-field checks.
            # Use it to open the local correction gate; keep the graded score for
            # softening magnitude below, so medium-strength interactions are not
            # discounted twice by both gate and alpha.
            gate_special_score = torch.where(
                interaction_xh_mask,
                torch.ones_like(gate_special_score),
                gate_special_score,
            )
        if physical_direct_mask.numel() == special_score.numel():
            gate_special_score = torch.where(
                physical_direct_mask.to(device).bool(),
                torch.ones_like(gate_special_score),
                gate_special_score,
            )
        base_global = float(self.hij_global_scale)
        ood_global = float(self.hij_ood_global_scale)
        edge_global_scale = base_global + (ood_global - base_global) * gate_special_score
        local_delta_scale = 1.0 + (float(self.hij_ood_alpha_scale) - 1.0) * special_score
        if hbond_mask is not None and hbond_mask.any():
            hb_scale = torch.full_like(local_delta_scale, float(self.hij_hbond_alpha_scale))
            local_delta_scale = torch.where(hbond_mask.to(device).bool(), hb_scale, local_delta_scale)
        if xh_interaction_score.numel() == local_delta_scale.numel():
            # Post-clamp gain for interaction-confirmed X-H softening. This is
            # deliberately separate from the broad H-bond alpha so strong O-H/N-H
            # red-shift corrections do not also amplify unrelated H-bond side
            # effects.
            interaction_alpha = 1.0 + (
                float(self.hij_interaction_alpha_scale) - 1.0
            ) * xh_interaction_score.to(device=device, dtype=local_delta_scale.dtype).clamp(0.0, 1.0)
            interaction_active = (xh_interaction_score.to(device) > 0.0) & is_xh.to(device).bool()
            local_delta_scale = torch.where(
                interaction_active,
                torch.maximum(local_delta_scale, interaction_alpha),
                local_delta_scale,
            )
        if physical_special_score.numel() == local_delta_scale.numel():
            physical_alpha = 1.0 + (
                float(self.hij_physical_alpha_scale) - 1.0
            ) * physical_special_score.to(device=device, dtype=local_delta_scale.dtype).clamp(0.0, 1.0)
            local_delta_scale = torch.where(
                physical_direct_mask.to(device).bool(),
                torch.maximum(local_delta_scale, physical_alpha),
                local_delta_scale,
            )
        hij_gate = (chem_gate.to(device) * mode_gate.to(device) * trigger * edge_global_scale).clamp(0.0, 1.0)
        calibration_bond = 1.0 + policy_delta_ratio * local_delta_scale * hij_gate
        if pos is not None:
            delta_long = (calibration_bond - 1.0) * hij_long
            projector = torch.einsum('bi,bj->bij', r_hat, r_hat)
            hij_cal = hij_pred.clone()
            hij_cal[bond_mask] = hij_bond_block + delta_long[:, None, None] * projector
        else:
            calibration = torch.ones(hij_pred.size(0), device=device, dtype=hij_pred.dtype)
            calibration[bond_mask] = calibration_bond
            hij_cal = hij_pred * calibration.unsqueeze(-1).unsqueeze(-1)
        self.last_hij_delta = (hij_cal - hij_pred).detach()
        active = hij_gate > 1e-4
        self.last_hij_debug = {
            "matched_edges": float(bond_mask.sum().detach().cpu().item()),
            "active_edges": float(active.sum().detach().cpu().item()),
            "xh_edges": float(is_xh.sum().detach().cpu().item()),
            "amide_edges": float(amide_mask.sum().detach().cpu().item()),
            "unknown_atom_edges": float(unknown_atom.sum().detach().cpu().item()),
            "unknown_class_edges": float(unknown_class.sum().detach().cpu().item()),
            "unknown_env_signature_edges": float(unknown_env_signature.sum().detach().cpu().item()),
            "hbond_edges": float(hbond_mask.sum().detach().cpu().item()),
            "interaction_gate_edges": float(interaction_xh_mask.sum().detach().cpu().item()) if interaction_xh_mask.numel() else 0.0,
            "mean_env_ood": float(env_ood_score.mean().detach().cpu().item()) if env_ood_score.numel() else 0.0,
            "mean_env_need": float(env_need.mean().detach().cpu().item()) if env_need.numel() else 0.0,
            "mean_enk_need": float(edge_enk_need.mean().detach().cpu().item()) if edge_enk_need.numel() else 0.0,
            "mean_physics_need": float(physics_need.mean().detach().cpu().item()) if physics_need.numel() else 0.0,
            "mean_nbo_reliability": float(nbo_reliability.mean().detach().cpu().item()) if nbo_reliability.numel() else 1.0,
            "mean_edge_global_scale": float(edge_global_scale.mean().detach().cpu().item()) if edge_global_scale.numel() else base_global,
            "mean_delta_scale": float(local_delta_scale.mean().detach().cpu().item()) if local_delta_scale.numel() else 1.0,
            "mean_gate": float(hij_gate.mean().detach().cpu().item()) if hij_gate.numel() else 0.0,
            "max_gate": float(hij_gate.max().detach().cpu().item()) if hij_gate.numel() else 0.0,
            "mean_consensus": float(consensus.mean().detach().cpu().item()) if consensus.numel() else 1.0,
            "mean_mode_gate": float(mode_gate.mean().detach().cpu().item()) if mode_gate.numel() else 1.0,
            "mean_trigger": float(trigger.mean().detach().cpu().item()) if trigger.numel() else 0.0,
            "hij_policy": policy,
            "mean_raw_delta_ratio": float(raw_delta_ratio.mean().detach().cpu().item()) if raw_delta_ratio.numel() else 0.0,
            "mean_policy_delta_ratio": float(policy_delta_ratio.mean().detach().cpu().item()) if policy_delta_ratio.numel() else 0.0,
            "xh_interaction_edges": float((xh_interaction_score > 0.0).sum().detach().cpu().item()) if xh_interaction_score.numel() else 0.0,
            "mean_xh_interaction_score": float(xh_interaction_score.mean().detach().cpu().item()) if xh_interaction_score.numel() else 0.0,
            "mean_xh_interaction_e2": float(xh_interaction_e2.mean().detach().cpu().item()) if xh_interaction_e2.numel() else 0.0,
            "mean_xh_interaction_e2_score": float(xh_interaction_e2_score.mean().detach().cpu().item()) if xh_interaction_e2_score.numel() else 0.0,
            "mean_xh_interaction_alpha": float(interaction_alpha.mean().detach().cpu().item()) if "interaction_alpha" in locals() and interaction_alpha.numel() else 1.0,
            "mean_xh_interaction_geom": float(hbond_details["interaction_geometry_score"].mean().detach().cpu().item()) if isinstance(hbond_details.get("interaction_geometry_score"), torch.Tensor) else 0.0,
            "mean_xh_interaction_pair_e2": float(hbond_details["interaction_pair_e2"].mean().detach().cpu().item()) if isinstance(hbond_details.get("interaction_pair_e2"), torch.Tensor) else 0.0,
            "mean_xh_interaction_field_e2": float(hbond_details["interaction_field_e2"].mean().detach().cpu().item()) if isinstance(hbond_details.get("interaction_field_e2"), torch.Tensor) else 0.0,
            "physical_soften_edges": float((physical_soften_score > 0.0).sum().detach().cpu().item()) if physical_soften_score.numel() else 0.0,
            "physical_harden_edges": float((physical_harden_score > 0.0).sum().detach().cpu().item()) if physical_harden_score.numel() else 0.0,
            "mean_physical_soften_score": float(physical_soften_score.mean().detach().cpu().item()) if physical_soften_score.numel() else 0.0,
            "mean_physical_harden_score": float(physical_harden_score.mean().detach().cpu().item()) if physical_harden_score.numel() else 0.0,
            "mean_physical_interaction_score": float(physical["interaction_score"].mean().detach().cpu().item()) if isinstance(physical.get("interaction_score"), torch.Tensor) and physical["interaction_score"].numel() else 0.0,
            "mean_stark_score": float(physical["stark_score"].mean().detach().cpu().item()) if isinstance(physical.get("stark_score"), torch.Tensor) and physical["stark_score"].numel() else 0.0,
            "mean_field_score": float(physical["field_score"].mean().detach().cpu().item()) if isinstance(physical.get("field_score"), torch.Tensor) and physical["field_score"].numel() else 0.0,
            "mean_physical_alpha": float(physical_alpha.mean().detach().cpu().item()) if "physical_alpha" in locals() and physical_alpha.numel() else 1.0,
        }
        return hij_cal

    def _apply_born_sum_rule(
        self,
        dd_cal: torch.Tensor,
        npa_charge: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        strength = float(getattr(self, "dd_sum_rule_strength", 1.0))
        if strength <= 0.0 or dd_cal.numel() == 0:
            return dd_cal

        device = dd_cal.device
        eye3 = torch.eye(3, device=device, dtype=dd_cal.dtype)
        if batch is None:
            target_sum = npa_charge.sum().to(dtype=dd_cal.dtype) * eye3
            correction = (target_sum - dd_cal.sum(dim=0)) / max(int(dd_cal.size(0)), 1)
            return dd_cal + strength * correction.unsqueeze(0)

        batch = batch.to(device).long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        sum_dd = torch.zeros(num_graphs, 3, 3, device=device, dtype=dd_cal.dtype)
        sum_dd.index_add_(0, batch, dd_cal)
        q_sum = torch.zeros(num_graphs, device=device, dtype=dd_cal.dtype)
        q_sum.index_add_(0, batch, npa_charge.to(device=device, dtype=dd_cal.dtype))
        counts = torch.bincount(batch, minlength=num_graphs).to(device=device, dtype=dd_cal.dtype).clamp(min=1.0)
        target_sum = q_sum[:, None, None] * eye3.unsqueeze(0)
        correction = (target_sum - sum_dd) / counts[:, None, None]
        return dd_cal + strength * correction[batch]

    def calibrate_dedipole(
        self,
        dd_pred: torch.Tensor,
        nbo: Dict[str, torch.Tensor],
        pos: Optional[torch.Tensor] = None,
        enk_ood_score: Optional[torch.Tensor] = None,
        enk_context: Optional[Dict[str, Any]] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calibrate dipole derivative using NPA charge + NBO bond features.

        Physical principle — TWO contributions to Born effective charge Z*:

          1. ISOTROPIC (ionic):  tr(Z*) ∝ q_NPA
             Constrained by the atom's own NPA charge.  Handled by the
             per-atom trace correction (existing logic).

          2. ANISOTROPIC (covalent charge-transfer):
             Z*_a^∥ = q_a  +  η · (q_a − q_b) · BO_ab
             where Z*_a^∥ = R̂^T · Z*_a · R̂ is the Born charge of atom a
             projected along the a→b bond direction, q_a/q_b are NPA
             charges, and BO_ab is the NBO bond occupancy.

             Physical mechanism: when a polar bond stretches, electrons
             flow towards the more electronegative atom, creating a dipole
             response BEYOND the static charge prediction.  This excess
             only affects Z* components along the bond axis — perpendicular
             components are charge-dominated.

        The isotropic trace correction alone systematically UNDER-corrects
        Z* along polar bonds (e.g. C=O, C=N), leading to wrong IR
        intensities in the 1500–1750 cm⁻¹ double-bond stretching region.

        The new bond-directional term adds the missing covalent contribution
        while preserving the trace (charge constraint) using a trace-free
        projector (R̂⊗R̂ − I/3).

        NBO atom_pred layout:
          [7]   NPA natural charge  ← q_NPA
        NBO bond_pred layout:
          [0]   bond occupancy      ← BO_ab

        Parameters
        ----------
        dd_pred : (N, 3, 3) predicted dipole derivative tensor
        nbo : dict with 'atom_pred', 'bond_pred', 'atom_bond_local'
        pos : (N, 3) atom positions for bond direction computation
        """
        self.last_dd_debug = {}
        atom_pred = nbo.get("atom_pred")
        if atom_pred is None:
            return dd_pred

        device = dd_pred.device
        atom_pred = atom_pred.to(device)
        npa_charge = atom_pred[:, 7]
        dd_trace = dd_pred.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        bond_pred = nbo.get("bond_pred")
        atom_bond_local = nbo.get("atom_bond_local")
        atom_need, _ = self._atom_context_need(
            z, atom_bond_local, int(dd_pred.size(0)), device, dd_pred.dtype,
            pos=pos, enk_ood_score=enk_ood_score, enk_context=enk_context,
        )
        dd_interaction_need = self._interaction_atom_exposure(
            nbo, int(dd_pred.size(0)), device, dd_pred.dtype,
        ) * float(self.dd_interaction_gate_scale)
        atom_need = torch.maximum(atom_need, dd_interaction_need.clamp(0.0, 1.0))

        # Step 1: isotropic trace correction
        if self.train_stats is not None:
            mu_t = float(self.train_stats.get("dd_trace_mean", 0.0))
            sig_t = float(self.train_stats.get("dd_trace_std", 1.0))
            mu_q = float(self.train_stats.get("npa_charge_mean", 0.0))
            sig_q = float(self.train_stats.get("npa_charge_std", 1.0))
            if self.dd_trace_mode == "off":
                trace_delta = torch.zeros_like(dd_trace)
                trace_need = torch.zeros_like(dd_trace)
            else:
                z_trace = (dd_trace - mu_t) / max(sig_t, 1e-6)
                z_charge = (npa_charge - mu_q) / max(sig_q, 1e-6)
                delta_z = z_charge - z_trace
                trace_scale = max(sig_t, 1e-6)
                trace_delta = (delta_z * trace_scale).clamp(
                    -self.clamp_ratio * trace_scale,
                    self.clamp_ratio * trace_scale,
                )
                trace_need = (delta_z.abs() / 2.0).clamp(0.0, 1.0)
        else:
            avg_charge = npa_charge.abs().mean().clamp(min=0.01)
            avg_trace = dd_trace.abs().mean().clamp(min=1e-10)
            if self.dd_trace_mode == "off":
                trace_delta = torch.zeros_like(dd_trace)
                trace_need = torch.zeros_like(dd_trace)
            else:
                target_trace = npa_charge * (avg_trace / avg_charge)
                trace_delta = (target_trace - dd_trace).clamp(
                    -self.clamp_ratio * avg_trace,
                    self.clamp_ratio * avg_trace,
                )
                trace_need = (trace_delta.abs() / (float(self.clamp_ratio) * avg_trace).clamp(min=1e-6)).clamp(0.0, 1.0)

        eye3 = torch.eye(3, device=device, dtype=dd_pred.dtype)
        if self.dd_trace_mode == "off":
            trace_gate = torch.zeros_like(trace_need)
        elif self.dd_trace_mode == "direct":
            trace_gate = (atom_need * trace_need).clamp(0.0, 1.0)
        else:
            trace_gate = (0.35 + 0.65 * torch.maximum(atom_need, trace_need)).clamp(0.0, 1.0)
        dd_cal = dd_pred + (float(self.alpha_dd) * trace_delta * trace_gate / 3.0)[:, None, None] * eye3

        # Step 2: bond-directional anisotropic correction
        if pos is None or bond_pred is None or atom_bond_local is None:
            self.last_dd_debug = {
                "mean_atom_need": float(atom_need.mean().detach().cpu().item()) if atom_need.numel() else 0.0,
                "mean_interaction_need": float(dd_interaction_need.mean().detach().cpu().item()) if dd_interaction_need.numel() else 0.0,
                "mean_trace_need": float(trace_need.mean().detach().cpu().item()) if trace_need.numel() else 0.0,
                "mean_trace_gate": float(trace_gate.mean().detach().cpu().item()) if trace_gate.numel() else 0.0,
                "mean_bond_gate": 0.0,
                "active_bonds": 0.0,
            }
            return self._apply_born_sum_rule(dd_cal, npa_charge)
        if bond_pred.numel() == 0 or atom_bond_local.numel() == 0:
            self.last_dd_debug = {
                "mean_atom_need": float(atom_need.mean().detach().cpu().item()) if atom_need.numel() else 0.0,
                "mean_interaction_need": float(dd_interaction_need.mean().detach().cpu().item()) if dd_interaction_need.numel() else 0.0,
                "mean_trace_need": float(trace_need.mean().detach().cpu().item()) if trace_need.numel() else 0.0,
                "mean_trace_gate": float(trace_gate.mean().detach().cpu().item()) if trace_gate.numel() else 0.0,
                "mean_bond_gate": 0.0,
                "active_bonds": 0.0,
            }
            return self._apply_born_sum_rule(dd_cal, npa_charge)

        bond_pred = bond_pred.to(device)
        atom_bond_local = atom_bond_local.to(device)
        pos = pos.to(device)

        occ_col = _bond_occupancy_col(int(bond_pred.size(1)), self.bond_occ_col_policy)
        if occ_col is None:
            return dd_pred
        bd_occ = bond_pred[:, occ_col].clamp(min=0.01)         # bond occupancy
        src, dst = atom_bond_local[0], atom_bond_local[1]

        # Bond direction vectors
        R_ab = pos[dst] - pos[src]                        # [n_bonds, 3]
        R_len = R_ab.norm(dim=-1, keepdim=True).clamp(min=0.01)
        R_hat = R_ab / R_len                              # unit vectors

        # Charge difference across bond (signed: + when src more positive)
        dq = npa_charge[src] - npa_charge[dst]            # [n_bonds]

        # NBO-implied excess Born charge along bond:
        #   Z*_a^∥ − q_a  =  η * dq * BO
        # η (bond_dipole_factor) ≈ 0.3 from typical Born charge anomalies:
        #   C=O: Δq≈1.0, BO≈1.8, Z* anomaly≈0.5  → η≈0.28
        _eta = float(getattr(self, 'bond_dipole_factor', 0.3))
        nbo_excess_src =  _eta * dq * bd_occ             # excess for src atom
        nbo_excess_dst = -_eta * dq * bd_occ             # excess for dst (opposite sign)

        # Predicted Born charge along bond direction for each atom
        # dd_cal[a] is (3,3); we want  R̂^T · dd_cal[a] · R̂  (scalar)
        dd_src_along = torch.einsum('bi,bij,bj->b',
                                     R_hat, dd_cal[src], R_hat)   # [n_bonds]
        dd_dst_along = torch.einsum('bi,bij,bj->b',
                                     R_hat, dd_cal[dst], R_hat)

        # NBO-implied along-bond Born charge
        nbo_src_along = npa_charge[src] + nbo_excess_src
        nbo_dst_along = npa_charge[dst] + nbo_excess_dst

        # Mismatch: NBO-implied minus predicted (signed)
        delta_src = nbo_src_along - dd_src_along
        delta_dst = nbo_dst_along - dd_dst_along

        # Trace-free projector: (R̂⊗R̂ − I/3)
        # Adding  δ × (R̂⊗R̂ − I/3)  preserves Tr(Z*) while moving Born
        # charge magnitude toward/away from the bond direction.
        _alpha = float(getattr(self, 'alpha_bond', 0.15))
        _clamp = self.clamp_ratio
        bond_env_need = torch.maximum(atom_need[src.long()], atom_need[dst.long()]).to(dtype=dd_cal.dtype)
        bond_physics_need = (
            (delta_src.abs() + delta_dst.abs()) / max(2.0 * float(_clamp), 1e-6)
        ).clamp(0.0, 1.0)
        bond_policy = torch.maximum(bond_env_need, bond_physics_need).clamp(0.0, 1.0)
        if self.dd_bond_gate_mode == "direct":
            bond_gate = (bond_env_need * bond_physics_need).clamp(0.0, 1.0)
        else:
            bond_gate = (0.35 + 0.65 * bond_policy) * (1.0 + 0.5 * bond_env_need)
            bond_gate = bond_gate.clamp(0.0, 1.5)
        delta_src = delta_src.clamp(-_clamp, _clamp) * _alpha * bond_gate
        delta_dst = delta_dst.clamp(-_clamp, _clamp) * _alpha * bond_gate

        eye3 = torch.eye(3, device=device, dtype=dd_cal.dtype)
        # projector[a,b] = R̂_a ⊗ R̂_b − δ_{ab}/3
        proj_src = torch.einsum('bi,bj->bij', R_hat, R_hat) - eye3 / 3.0  # [B,3,3]
        proj_dst = proj_src  # same projector (bond direction is same for both)

        # Accumulate corrections per atom
        # For atom a with multiple bonds, corrections from each bond sum up
        correction = torch.zeros_like(dd_cal)
        correction.index_add_(0, src, proj_src * delta_src.unsqueeze(-1).unsqueeze(-1))
        correction.index_add_(0, dst, proj_dst * delta_dst.unsqueeze(-1).unsqueeze(-1))
        dd_cal = dd_cal + correction

        self.last_dd_debug = {
            "mean_atom_need": float(atom_need.mean().detach().cpu().item()) if atom_need.numel() else 0.0,
            "mean_interaction_need": float(dd_interaction_need.mean().detach().cpu().item()) if dd_interaction_need.numel() else 0.0,
            "mean_trace_need": float(trace_need.mean().detach().cpu().item()) if trace_need.numel() else 0.0,
            "mean_trace_gate": float(trace_gate.mean().detach().cpu().item()) if trace_gate.numel() else 0.0,
            "mean_bond_env_need": float(bond_env_need.mean().detach().cpu().item()) if bond_env_need.numel() else 0.0,
            "mean_bond_physics_need": float(bond_physics_need.mean().detach().cpu().item()) if bond_physics_need.numel() else 0.0,
            "mean_bond_gate": float(bond_gate.mean().detach().cpu().item()) if bond_gate.numel() else 0.0,
            "active_bonds": float((bond_policy > 0.05).sum().detach().cpu().item()) if bond_policy.numel() else 0.0,
        }
        return self._apply_born_sum_rule(dd_cal, npa_charge)

    def calibrate_depolar(
        self,
        dp_pred: torch.Tensor,
        nbo: Dict[str, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        enk_ood_score: Optional[torch.Tensor] = None,
        enk_context: Optional[Dict[str, Any]] = None,
        z: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calibrate polarizability derivative using delocalization proxy.

        Physical principle:
          |∂α/∂R| ∝ delocalization_proxy

        The delocalization proxy is:
          atom_pred[:, 1] (NAO p-orbital occupancy, moderate +correlation)
          + bond_order_sum (strong delocalization signal from the bond occupancy field)

        NAO p-orbital occupancy is not the optimal delocalization metric, but
        combined with bond order sum it provides a usable gradient signal for
        polarizability derivative calibration.  For a more principled alternative,
        LI (atom_pred[:, 12], local ionization energy) is inversely correlated
        with delocalization and would need an inverted power-law relationship.
        """
        atom_pred = nbo.get("atom_pred")
        bond_pred = nbo.get("bond_pred")
        atom_bond_local = nbo.get("atom_bond_local")

        self.last_dp_debug = {}
        if atom_pred is None:
            return dp_pred

        device = dp_pred.device
        atom_pred = atom_pred.to(device)
        n_atoms = int(dp_pred.size(0))
        atom_need, _ = self._atom_context_need(
            z, atom_bond_local, n_atoms, device, dp_pred.dtype,
            pos=pos, enk_ood_score=enk_ood_score, enk_context=enk_context,
        )
        dp_interaction_exposure = self._interaction_atom_exposure(
            nbo, n_atoms, device, dp_pred.dtype,
        )
        dp_interaction_need = (dp_interaction_exposure * float(self.dp_interaction_gate_scale)).clamp(0.0, 1.0)
        if self.dp_gate_mode != "direct":
            atom_need = torch.maximum(atom_need, dp_interaction_need)

        delocal_proxy = atom_pred[:, 1].clamp(min=0.01)

        if bond_pred is not None and atom_bond_local is not None and bond_pred.numel() > 0 and atom_bond_local.numel() > 0:
            bond_pred = bond_pred.to(device)
            atom_bond_local = atom_bond_local.to(device)
            occ_col = _bond_occupancy_col(int(bond_pred.size(1)), self.bond_occ_col_policy)
            if occ_col is None:
                return dp_pred
            bd_occ = bond_pred[:, occ_col].clamp(min=0.0)
            src_b, dst_b = atom_bond_local[0], atom_bond_local[1]
            bond_order_sum = torch.zeros(atom_pred.size(0), device=device)
            bond_order_sum.scatter_add_(0, src_b.long(), bd_occ)
            bond_order_sum.scatter_add_(0, dst_b.long(), bd_occ)
            delocal_proxy = (delocal_proxy + bond_order_sum).clamp(min=0.01)
        if self.dp_gate_mode != "direct" and dp_interaction_exposure.numel() == delocal_proxy.numel():
            delocal_proxy = (
                delocal_proxy
                * (1.0 + float(self.dp_interaction_boost) * dp_interaction_exposure.clamp(0.0, 1.0))
            ).clamp(min=0.01)

        dp_norm = dp_pred.norm(dim=-1).mean(dim=-1).clamp(min=1e-10)

        if self.train_stats is not None:
            mu_d = float(self.train_stats.get("dp_norm_mean", 0.0))
            sig_d = float(self.train_stats.get("dp_norm_std", 1.0))
            mu_l = float(self.train_stats.get("delocal_mean", 0.0))
            sig_l = float(self.train_stats.get("delocal_std", 1.0))
            z_dp = (dp_norm - mu_d) / max(sig_d, 1e-6)
            z_delocal = (delocal_proxy - mu_l) / max(sig_l, 1e-6)
            delta_z = z_delocal - z_dp
            nbo_ratio = (1.0 + delta_z).clamp(
                1.0 - self.clamp_ratio, 1.0 + self.clamp_ratio,
            )
            physics_need = (delta_z.abs() / 2.0).clamp(0.0, 1.0)
        else:
            avg_delocal = delocal_proxy.mean().clamp(min=0.01)
            avg_dp_norm = dp_norm.mean().clamp(min=1e-10)
            delocal_relative = (delocal_proxy / avg_delocal).clamp(
                1.0 - self.clamp_ratio, 1.0 + self.clamp_ratio,
            )
            dp_relative = (dp_norm / avg_dp_norm).clamp(min=1e-10)
            nbo_ratio = (delocal_relative / dp_relative).clamp(
                1.0 - self.clamp_ratio, 1.0 + self.clamp_ratio,
            )
            physics_need = (
                (nbo_ratio - 1.0).abs() / max(float(self.clamp_ratio), 1e-6)
            ).clamp(0.0, 1.0)
        if self.dp_gate_mode != "direct" and dp_interaction_need.numel() == physics_need.numel():
            physics_need = torch.maximum(physics_need, dp_interaction_need)

        policy = torch.maximum(atom_need, physics_need).clamp(0.0, 1.0)
        if self.dp_gate_mode == "direct":
            policy = (atom_need * physics_need).clamp(0.0, 1.0)
            dp_gate = policy
        else:
            dp_gate = (0.35 + 0.65 * policy) * (1.0 + 0.5 * atom_need)
            dp_gate = dp_gate.clamp(0.0, 1.5)
        calibration = 1.0 + self.alpha_dp * (nbo_ratio - 1.0) * dp_gate
        dp_cal = dp_pred * calibration.unsqueeze(-1).unsqueeze(-1)
        self.last_dp_debug = {
            "mean_atom_need": float(atom_need.mean().detach().cpu().item()) if atom_need.numel() else 0.0,
            "mean_interaction_need": float(dp_interaction_need.mean().detach().cpu().item()) if dp_interaction_need.numel() else 0.0,
            "mean_interaction_exposure": float(dp_interaction_exposure.mean().detach().cpu().item()) if dp_interaction_exposure.numel() else 0.0,
            "mean_physics_need": float(physics_need.mean().detach().cpu().item()) if physics_need.numel() else 0.0,
            "mean_gate": float(dp_gate.mean().detach().cpu().item()) if dp_gate.numel() else 0.0,
            "active_atoms": float((policy > 0.05).sum().detach().cpu().item()) if policy.numel() else 0.0,
        }
        return dp_cal

    def calibrate_all(
        self,
        Hi: torch.Tensor,
        Hij: torch.Tensor,
        dd: torch.Tensor,
        dp: torch.Tensor,
        models: Dict[str, torch.nn.Module],
        edge_index: torch.Tensor,
        num_nodes: int,
        pos: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        nbo = None
        for tag in ("hij", "dd", "dp", "hii"):
            m = models.get(tag)
            if m is not None:
                nbo = _extract_nbo_from_model(m)
                if nbo is not None:
                    break
        if nbo is None:
            return {"Hi": Hi, "Hij": Hij, "dd": dd, "dp": dp}

        Hij_cal = self.calibrate_hij(Hij, nbo, edge_index, num_nodes, pos=pos)
        dd_cal = self.calibrate_dedipole(dd, nbo, pos=pos)
        dp_cal = self.calibrate_depolar(dp, nbo, edge_index, num_nodes)
        return {"Hi": Hi, "Hij": Hij_cal, "dd": dd_cal, "dp": dp_cal}
