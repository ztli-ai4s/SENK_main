"""
NBO Consistency Regularization (NBO-CR) Loss
=============================================

Forces the spectral prediction model to maintain physical consistency with
NBO (Natural Bond Orbital) predictions from the electron prior predictor.

Physical relationships enforced (TRAINING-TIME only):
  1. |Hij_along| ∝ k * occ^α  (Badger rule, NBO-consensus-weighted)
  2. Σ Z*_i = Q_total × I      (acoustic sum rule — EXACT physical law)
  3. |∂α/∂R| ∝ delocal         (polarizability derivative ↔ electron delocalization)

Born charge magnitude relationships (||Z*||_F ∝ |q|^b_q * delocal^b_d) and
directional corrections (Z*_a^∥ ≈ q_a + η·Δq·BO) are handled at INFERENCE
time by NBOGuidedCalibrator (GSC), NOT here in CR.  These are statistical
trends, not physical laws — enforcing them during training destroys DFT
accuracy because the same NPA charge can correspond to Born charges differing
by 10× depending on element, hybridization, and coordination.

These relationships are **universal** — they hold for any molecule, not just
training-set molecules.  By enforcing them during EP training, the model
learns to respect NBO-implied physics, which naturally transfers to unseen
molecules where geometric features alone are insufficient.

Why NOT Hii?
------------
Hii (diagonal Hessian) is NOT separately regularized because:
  - Hii is the row-sum of the full Hessian: Hii_αα ≈ -Σ_j Hij_αβ
  - When Hij is physically consistent with NBO, Hii is implicitly consistent
  - Hii is dominated by local bond-stretching force constants that are
    well-captured by distance-based features; NBO adds little new information
  - A direct NBO→Hii proxy is physically weaker than NBO→Hij because Hii
    sums contributions from ALL neighbors, diluting any single NBO signal

Usage in training:
    nbo_cr = NBOConsistencyLoss(alpha_init=0.6, ...)
    # After forward pass, before optimizer step:
    cr_loss = nbo_cr(model, data, pred, branch)
    total_loss = spectral_loss + nbo_cr_weight * cr_loss

CR loss gradient direction:
    NBO features are detached so the electron-prior predictor is not trained
    by this loss. The spectral prediction and the small CR calibration
    parameters both receive gradients, so the proxy must stay low-dimensional
    and physically interpretable.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any

from nbo_spectral_calibration import _bond_occupancy_col


def _extract_nbo_from_model(model: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
    prior = getattr(model, "electron_prior", None)
    if prior is None:
        return None
    nbo = getattr(prior, "_last_nbo_predictions", None)
    if not isinstance(nbo, dict) or not nbo:
        return None
    return nbo


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


class NBOConsistencyLoss(torch.nn.Module):
    """NBO consistency regularization for spectral prediction models.

    Learnable parameters:
      - log_k_hij, alpha: Badger rule |Hij_along| ∝ k * occ^alpha,
        per-bond weighted by NBO consensus across independent bond-order
        indicators (occupancy, DI, LBO, Mayer).  When indicators agree
        the CR signal is strong; when they disagree (OOD bonds) it mutes.
      - log_c_dp, log_d_dp: |∂α/∂R| ∝ c * delocal^d (power law)

    Dedipole (Born charge) only enforces the acoustic sum rule
    Σ Z*_i = Q_total × I — an EXACT physical law that never conflicts
    with DFT accuracy.  Magnitude and directional constraints are
    handled by NBOGuidedCalibrator (GSC) at inference time.
    """

    def __init__(
        self,
        alpha_init: float = 0.6,
        a_bec_init: float = 1.0,
        b_bec_q_init: float = 0.8,
        b_bec_d_init: float = 0.3,
        c_dp_init: float = 1.0,
        d_dp_init: float = 0.5,
        eta_init: float = 0.3,
        hij_scale_init: float = 8.0,
        dd_sum_rule_weight: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(max(alpha_init, 0.01))))
        self.log_k_hij = torch.nn.Parameter(torch.tensor(math.log(max(hij_scale_init, 0.01))))
        self.log_a_bec = torch.nn.Parameter(torch.tensor(math.log(max(a_bec_init, 0.01))))
        self.b_bec_q = torch.nn.Parameter(torch.tensor(b_bec_q_init))
        self.b_bec_d = torch.nn.Parameter(torch.tensor(b_bec_d_init))
        self.log_c_dp = torch.nn.Parameter(torch.tensor(math.log(max(c_dp_init, 0.01))))
        self.log_d_dp = torch.nn.Parameter(torch.tensor(math.log(max(d_dp_init, 0.01))))
        self.log_eta = torch.nn.Parameter(torch.tensor(math.log(max(eta_init, 0.01))))
        self.dd_sum_rule_weight = dd_sum_rule_weight
        self.eps = eps

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def a_bec(self):
        return self.log_a_bec.exp()

    @property
    def c_dp(self):
        return self.log_c_dp.exp()

    @property
    def d_dp(self):
        return self.log_d_dp.exp()

    @property
    def eta(self):
        return self.log_eta.exp()

    def forward(
        self,
        model: torch.nn.Module,
        data: Any,
        pred: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        """Compute NBO consistency loss for the given branch.

        Parameters
        ----------
        model : the spectral prediction model (must have electron_prior)
        data : the batch data (must have edge_index, pos, z, batch)
        pred : the model's spectral prediction tensor
        branch : one of 'hij', 'dedipole', 'depolar', 'hii', etc.

        Returns
        -------
        Scalar loss tensor (0 if NBO unavailable or branch not applicable)
        """
        nbo = _extract_nbo_from_model(model)
        if nbo is None:
            return torch.tensor(0.0, device=pred.device)

        if branch == "hij":
            return self._hij_consistency(pred, nbo, data)
        elif branch in ("dedipole", "sobolev_dipole"):
            return self._dedipole_consistency(pred, nbo, data)
        elif branch in ("depolar", "sobolev_polar"):
            return self._depolar_consistency(pred, nbo, data)
        else:
            return torch.tensor(0.0, device=pred.device)

    def _hij_consistency(
        self,
        hij_pred: torch.Tensor,
        nbo: Dict[str, torch.Tensor],
        data: Any,
    ) -> torch.Tensor:
        """Consensus-weighted Badger rule for off-diagonal Hessian.

        Uses NBO bond occupancy to predict |Hij_along| via Badger's rule:
          |Hij_along| ∝ k * occ^alpha

        The key innovation is NBO CONSENSUS WEIGHTING: the qcmol bond_pred
        (18-dim) contains multiple independent bond-order indicators —
        NBO occupancy, Wiberg index, Mayer bond order, DI, LBO — from
        different quantum-chemistry methodologies.  When these indicators
        AGREE, NBO is confident about the bond type and the Badger signal
        is at full strength.  When they DISAGREE, the bond is unfamiliar
        (OOD relative to NBO's training data) and the CR signal is
        automatically muted.

        This lets the model learn from NBO's broad chemical knowledge
        (PubChem + PDBbind) where NBO is confident, without being forced
        toward a crude formula for exotic bonds where NBO itself is
        uncertain.  The consensus weight is fully detached — it only
        scales the per-bond loss, no gradient flows to NBO features.
        """
        bond_pred = nbo.get("bond_pred")
        atom_bond_local = nbo.get("atom_bond_local")

        if bond_pred is None or atom_bond_local is None:
            return torch.tensor(0.0, device=hij_pred.device)
        if bond_pred.numel() == 0 or atom_bond_local.numel() == 0:
            return torch.tensor(0.0, device=hij_pred.device)

        device = hij_pred.device
        bond_pred = bond_pred.to(device)
        atom_bond_local = atom_bond_local.to(device)

        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            return torch.tensor(0.0, device=device)
        edge_index = edge_index.to(device)
        num_nodes = int(data.z.shape[0]) if hasattr(data, "z") else int(edge_index.max().item() + 1)

        bond_ids, bond_mask = _match_undirected_edges_simple(
            edge_index, atom_bond_local, num_nodes,
        )
        if not bond_mask.any():
            return torch.tensor(0.0, device=device)

        hij_mat = hij_pred.view(-1, 3, 3)
        pos = getattr(data, "pos", None)
        if pos is not None:
            pos = pos.to(device)
            edge_src = edge_index[0, bond_mask].long()
            edge_dst = edge_index[1, bond_mask].long()
            r_vec = pos[edge_dst] - pos[edge_src]
            r_hat = r_vec / r_vec.norm(dim=-1, keepdim=True).clamp(min=0.01)
            hij_bond_block = hij_mat[bond_mask]
            hij_long = torch.einsum('bi,bij,bj->b', r_hat, hij_bond_block, r_hat)
            hij_bond = hij_long.abs().clamp(min=self.eps)
        else:
            hij_norm = hij_mat.norm(dim=(-2, -1))
            hij_bond = hij_norm[bond_mask].clamp(min=self.eps)

        log_hij = torch.log(hij_bond)

        # Badger rule: NBO occupancy → predicted log|Hij_along|
        occ_col = _bond_occupancy_col(int(bond_pred.size(1)))
        if occ_col is None:
            return torch.tensor(0.0, device=device)
        bd_occ = bond_pred[:, occ_col].clamp(min=self.eps)
        matched_occ = bd_occ[bond_ids[bond_mask]].detach().clamp(min=self.eps)
        log_occ = torch.log(matched_occ)
        nbo_implied = self.log_k_hij + self.alpha * log_occ
        per_bond_se = (log_hij - nbo_implied).pow(2)

        # NBO consensus: cross-check independent bond-order indicators
        matched_bp = bond_pred[bond_ids[bond_mask]].detach()
        n_cols = int(matched_bp.size(1))

        # Collect independent bond-strength indicators.
        # Column 0 = NBO occupancy (always present).
        ind_0 = matched_bp[:, 0:1].clamp(min=0.0)
        col_max_0 = ind_0.max().clamp(min=self.eps)
        indicators = [ind_0 / col_max_0]

        # Auxiliary columns (last 3 when present): DI, LBO, Mayer bond order.
        # These are only present in qcmol mode with aux features enabled.
        if n_cols >= 16:
            for c in range(max(15, n_cols - 3), n_cols):
                col = matched_bp[:, c:c + 1].clamp(min=0.0)
                cmax = col.max().clamp(min=self.eps)
                indicators.append(col / cmax)

        if len(indicators) >= 2:
            stacked = torch.cat(indicators, dim=1)
            col_mean = stacked.mean(dim=1).clamp(min=0.01)
            col_std = stacked.std(dim=1)
            cv = col_std / col_mean
            consensus = torch.exp(-cv * 3.0)
        else:
            consensus = torch.ones_like(log_hij)

        return (consensus * per_bond_se).mean()

    def _born_sum_rule_loss(
        self,
        dd_pred: torch.Tensor,
        npa_charge: torch.Tensor,
        data: Any = None,
    ) -> torch.Tensor:
        """Soft acoustic charge sum rule: sum_i Z*_i = Q_total I."""
        device = dd_pred.device
        eye3 = torch.eye(3, device=device, dtype=dd_pred.dtype)
        batch = getattr(data, "batch", None) if data is not None else None

        if batch is None:
            target = npa_charge.sum().to(dtype=dd_pred.dtype) * eye3
            denom = math.sqrt(max(int(dd_pred.size(0)), 1))
            residual = (dd_pred.sum(dim=0) - target) / denom
            return residual.pow(2).mean()

        batch = batch.to(device).long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        sum_dd = torch.zeros(num_graphs, 3, 3, device=device, dtype=dd_pred.dtype)
        sum_dd.index_add_(0, batch, dd_pred)
        q_sum = torch.zeros(num_graphs, device=device, dtype=dd_pred.dtype)
        q_sum.index_add_(0, batch, npa_charge.to(device=device, dtype=dd_pred.dtype))
        counts = torch.bincount(batch, minlength=num_graphs).to(device=device, dtype=dd_pred.dtype).clamp(min=1.0)
        target = q_sum[:, None, None] * eye3.unsqueeze(0)
        residual = (sum_dd - target) / counts.sqrt()[:, None, None]
        return residual.pow(2).mean()

    def _dedipole_consistency(
        self,
        dd_pred: torch.Tensor,
        nbo: Dict[str, torch.Tensor],
        data: Any = None,
    ) -> torch.Tensor:
        """Acoustic sum rule for Born effective charges.

        Physical law (EXACT):
            sum_i Z*_i = Q_total × I

        The sum of all atomic Born charge tensors must equal the total
        molecular charge times the identity matrix.  This is the acoustic
        sum rule — it follows from translational invariance of the dipole
        and holds for ANY molecule regardless of composition or bonding.

        Why ONLY the sum rule (no Frobenius norm power law)?
          - ||Z*||_F ∝ |q|^b_q × delocal^b_d is a 3-parameter statistical
            trend, not a physical law.  The same NPA charge produces Born
            charges differing by 10× depending on element, hybridization,
            and coordination (sp³ C vs sp² C=O vs sp² C=C).  Forcing the
            model to match a crude power law destroys DFT accuracy.
          - The sum rule is EXACT — it never conflicts with DFT, and it
            enforces global charge conservation that helps OOD molecules.
          - Bond-directional corrections are handled at INFERENCE time by
            NBOGuidedCalibrator (GSC), where they cannot damage training.

        Parameters
        ----------
        dd_pred : (N, 3, 3) predicted dipole derivative tensor
        nbo : dict with 'atom_pred' (needs column 7 = NPA charge)
        data : batch data with 'batch' tensor for batched molecules
        """
        atom_pred = nbo.get("atom_pred")
        if atom_pred is None:
            return torch.tensor(0.0, device=dd_pred.device)

        device = dd_pred.device
        atom_pred = atom_pred.to(device)
        npa_charge = atom_pred[:, 7].clone()

        dd_mat = dd_pred.view(-1, 3, 3)
        loss = self._born_sum_rule_loss(dd_mat, npa_charge.detach(), data)
        return loss

    def _depolar_consistency(
        self,
        dp_pred: torch.Tensor,
        nbo: Dict[str, torch.Tensor],
        data: Any,
    ) -> torch.Tensor:
        """Enforce |∂α/∂R| ∝ delocalization_proxy^d."""
        atom_pred = nbo.get("atom_pred")
        bond_pred = nbo.get("bond_pred")
        atom_bond_local = nbo.get("atom_bond_local")

        if atom_pred is None:
            return torch.tensor(0.0, device=dp_pred.device)

        device = dp_pred.device
        atom_pred = atom_pred.to(device)

        delocal_proxy = atom_pred[:, 1].clamp(min=self.eps)

        if bond_pred is not None and atom_bond_local is not None and bond_pred.numel() > 0 and atom_bond_local.numel() > 0:
            bond_pred = bond_pred.to(device)
            atom_bond_local = atom_bond_local.to(device)
            occ_col = _bond_occupancy_col(int(bond_pred.size(1)))
            if occ_col is None:
                return torch.tensor(0.0, device=device)
            bd_occ = bond_pred[:, occ_col].clamp(min=0.0)
            src_b, dst_b = atom_bond_local[0], atom_bond_local[1]
            bond_order_sum = torch.zeros(atom_pred.size(0), device=device)
            bond_order_sum.scatter_add_(0, src_b.long(), bd_occ)
            bond_order_sum.scatter_add_(0, dst_b.long(), bd_occ)
            delocal_proxy = (delocal_proxy + bond_order_sum).clamp(min=self.eps)

        dp_norm = dp_pred.norm(dim=-1).mean(dim=-1).clamp(min=self.eps)

        log_dp = torch.log(dp_norm)
        log_delocal = torch.log(delocal_proxy)

        nbo_target = self.log_c_dp + self.log_d_dp.exp() * log_delocal.detach()
        loss = F.mse_loss(log_dp, nbo_target)

        return loss
