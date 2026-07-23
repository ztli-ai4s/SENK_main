"""
Equivariant Neural Kalman (ENK) Module
=======================================

A physics-informed neural module that embeds Kalman filter principles into the
Equiformer backbone for molecular vibration property prediction.

Design Principles:
  1. **Learnable State Transition**: Instead of a fixed F matrix, a lightweight
     equivariant MLP learns the state transition dynamics from the backbone features,
     incorporating molecular geometry priors (bond lengths, angles via the Equiformer
     backbone's spherical harmonic edge features).
  2. **Dual Uncertainty Estimation**: Neural networks predict both:
     - Process noise Q (how much the molecular state changes between perturbations)
     - Observation noise R (how uncertain the current per-atom prediction is)
     These are predicted per-atom as scalar gates from the backbone features.
  3. **Equivariant Kalman Update**: The Kalman gain K is computed in a way that
     respects SO(3) equivariance — separate scalar gains for the isotropic (0e)
     channel and tensor gains broadcast over the 1e/2e channels.
  4. **Prior Conditioner Compatible**: The ENK module operates on the projected
     features (h_atom, atom_t) AFTER backbone extraction and BEFORE the
     PolarizabilityTensorHead, leaving the electron prior injection point intact.

Physical Motivation:
  In molecular vibration, a molecule perturbed from equilibrium evolves its
  polarizability tensor α(t) smoothly. The ENK module models this as:
    - "State": the per-atom contribution to the molecular polarizability tensor
    - "Transition": how the local chemical environment predicts the response to
      geometric perturbation (learned from data, not fixed physics)
    - "Observation": the raw backbone output, treated as a noisy measurement
    - "Update": optimally fuse the predicted state with the noisy observation,
      weighted by learned confidence (Kalman gain)

  For training (static batches): ENK acts as a single-step state estimator that
  learns to decompose backbone features into "reliable signal" vs "noise", providing
  implicit regularization that smooths the energy landscape for derivatives.

  For MD inference (trajectory): ENK naturally extends to multi-frame temporal
  filtering via the companion TensorRTSSmoother class.

Integration:
  Inserted between backbone feature extraction and the PolarizabilityTensorHead.
  Compatible with both CleanEquiformerPolar and CleanEquiformerPolarExt architectures.

Architecture:
  h_atom [N, H]  +-> StatePredictor  -> x_pred [N, H]
                 |
                 +-> QPredictor      -> q [N, 1] (process confidence)
                 |
                 +-> RPredictor      -> r [N, 1] (observation confidence)
                 |
                 +-> KalmanUpdate(x_pred, h_atom, q, r) -> h_filtered [N, H]

  atom_t [N, H, 3, 3] -> TensorKalmanGate -> atom_t_filtered [N, H, 3, 3]
"""
import math
import torch
from torch import nn, Tensor
from typing import Optional, Tuple


class EquivariantNeuralKalman(nn.Module):
    """
    Equivariant Neural Kalman filter module for molecular property prediction.

    Operates on per-atom features from the Equiformer backbone:
      - Scalar channel h_atom [N, H]: full Kalman update with learned Q, R, gain
      - Tensor channel atom_t [N, H, 3, 3]: gated refinement preserving equivariance

    The module decomposes backbone output into "predicted state" (what the local
    chemical environment expects) and "innovation" (the residual surprise), then
    uses learned confidence weights to optimally combine them.
    """

    def __init__(
        self,
        hidden_nf: int = 128,
        dropout: float = 0.1,
        init_r_bias: float = -1.0,
        init_q_bias: float = 1.0,
        tensor_gate: bool = True,
    ):
        """
        Args:
            hidden_nf: Hidden feature dimension (must match backbone proj_s output).
            dropout: Dropout rate in predictor MLPs.
            init_r_bias: Initial bias for observation noise logit.
                Negative → small R → trust observation more at init.
            init_q_bias: Initial bias for process noise logit.
                Positive → large Q → less trust in prediction at init.
            tensor_gate: Whether to apply gated refinement to atom_t tensor channel.
        """
        super().__init__()
        self.hidden_nf = hidden_nf
        self.tensor_gate = tensor_gate

        # State predictor: predicts the "expected" features from the input
        # This is analogous to x_pred = F @ x_prev in classical Kalman
        # Here we learn F as a residual MLP: x_pred = x + MLP(x)
        self.state_predictor = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf),
        )
        # Initialize near-identity: output ≈ 0 → x_pred ≈ x at start
        nn.init.zeros_(self.state_predictor[-1].weight)
        nn.init.zeros_(self.state_predictor[-1].bias)

        # Process noise predictor Q: per-atom scalar → how much state prediction
        # deviates from truth (high Q = low confidence in prediction)
        self.q_predictor = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf // 4),
            nn.SiLU(),
            nn.Linear(hidden_nf // 4, 1),
        )
        self.q_predictor[-1].bias.data.fill_(init_q_bias)

        # Observation noise predictor R: per-atom scalar → how noisy the
        # backbone observation is (high R = low confidence in observation)
        self.r_predictor = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf // 4),
            nn.SiLU(),
            nn.Linear(hidden_nf // 4, 1),
        )
        self.r_predictor[-1].bias.data.fill_(init_r_bias)

        # Innovation gate: modulates how much of the innovation (residual)
        # actually enters the update. Learns to suppress spurious high-frequency
        # fluctuations that hurt derivative quality.
        self.innovation_gate = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            nn.Sigmoid(),
        )

        # Tensor channel gate (for atom_t equivariant features)
        if tensor_gate:
            self.tensor_confidence = nn.Sequential(
                nn.LayerNorm(hidden_nf),
                nn.Linear(hidden_nf, 1),
            )
            # Initialize so tensor gate ≈ 1 (pass-through) at start
            self.tensor_confidence[-1].bias.data.fill_(2.0)

    def forward(
        self,
        h_atom: Tensor,
        atom_t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], dict]:
        """
        Apply Equivariant Neural Kalman filtering to backbone features.

        Args:
            h_atom: [N, H] scalar features from backbone projection
            atom_t: [N, H, 3, 3] l=2 tensor features (optional)

        Returns:
            h_filtered: [N, H] filtered scalar features
            atom_t_filtered: [N, H, 3, 3] filtered tensor features (or None)
            info: dict with diagnostic quantities for logging
        """
        # === Prediction step: x_pred = x + residual_MLP(x) ===
        x_pred = h_atom + self.state_predictor(h_atom)  # [N, H]

        # === Uncertainty estimation ===
        log_q = self.q_predictor(h_atom)    # [N, 1] process noise logit
        log_r = self.r_predictor(h_atom)    # [N, 1] observation noise logit

        # Convert to positive variances via softplus for numerical stability
        q = torch.nn.functional.softplus(log_q)  # [N, 1]
        r = torch.nn.functional.softplus(log_r)  # [N, 1]

        # === Kalman gain computation ===
        # K = P_pred / (P_pred + R) where P_pred ∝ Q
        # Simplification: K = Q / (Q + R) ∈ (0, 1)
        # High Q, low R → K → 1 → trust observation
        # Low Q, high R → K → 0 → trust prediction
        kalman_gain = q / (q + r + 1e-8)  # [N, 1]

        # === Innovation: difference between observation and prediction ===
        innovation = h_atom - x_pred  # [N, H]

        # === Gated innovation: suppress spurious fluctuations ===
        gate = self.innovation_gate(h_atom)  # [N, H] element-wise gate
        gated_innovation = innovation * gate  # [N, H]

        # === Kalman update: x_filtered = x_pred + K * gated_innovation ===
        h_filtered = x_pred + kalman_gain * gated_innovation  # [N, H]

        # === Tensor channel: equivariant gated refinement ===
        atom_t_filtered = atom_t
        if atom_t is not None and self.tensor_gate:
            # Scalar confidence gate broadcast over [3, 3] tensor indices
            # This preserves SO(3) equivariance: scalar × tensor = tensor
            t_conf = torch.sigmoid(self.tensor_confidence(h_atom))  # [N, 1]
            t_conf = t_conf.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1, 1]
            atom_t_filtered = atom_t * t_conf  # [N, H, 3, 3]

        info = {
            "enk_kalman_gain_mean": kalman_gain.mean().detach(),
            "enk_q_mean": q.mean().detach(),
            "enk_r_mean": r.mean().detach(),
            "enk_innovation_norm": innovation.norm(dim=-1).mean().detach(),
        }

        return h_filtered, atom_t_filtered, info


class TemporalENK(nn.Module):
    """
    Temporal extension of ENK for multi-frame MD trajectory inference.

    During MD trajectory prediction, this module maintains a running state
    estimate and covariance, updating them frame-by-frame using the single-frame
    ENK as the observation model.

    This is the "Deep Kalman" variant: the state transition and noise models
    are learned (from the EquivariantNeuralKalman module), but the temporal
    filtering logic follows the classical Kalman recursion.

    Usage:
        tenk = TemporalENK(enk_module)
        tenk.reset()
        for frame_features in trajectory:
            filtered = tenk.step(frame_features)
    """

    def __init__(self, enk: EquivariantNeuralKalman, momentum: float = 0.9):
        """
        Args:
            enk: Trained EquivariantNeuralKalman module (frozen or fine-tuned).
            momentum: Exponential moving average factor for state transition.
                Higher = more smoothing (trust history more).
        """
        super().__init__()
        self.enk = enk
        self.momentum = momentum
        self._state: Optional[Tensor] = None
        self._P: Optional[Tensor] = None

    def reset(self):
        """Reset temporal state for a new trajectory."""
        self._state = None
        self._P = None

    @torch.no_grad()
    def step(
        self,
        h_atom: Tensor,
        atom_t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Process one frame of a trajectory.

        Args:
            h_atom: [N, H] backbone features for current frame
            atom_t: [N, H, 3, 3] tensor features for current frame

        Returns:
            h_filtered: [N, H] temporally filtered features
            atom_t_filtered: [N, H, 3, 3] or None
        """
        # Get single-frame ENK output
        h_filt, t_filt, info = self.enk(h_atom, atom_t)

        if self._state is None:
            # First frame: initialize state
            self._state = h_filt
            self._P = info["enk_q_mean"].expand_as(h_filt)
        else:
            # Temporal prediction: x_pred = momentum * x_prev + (1-momentum) * x_new
            mu = self.momentum
            # Predicted state from temporal model
            x_pred_temporal = mu * self._state + (1.0 - mu) * h_filt

            # Get observation confidence from ENK
            q = info["enk_q_mean"]
            r = info["enk_r_mean"]
            k_temporal = q / (q + r + 1e-8)

            # Fuse temporal prediction with current observation
            innovation = h_filt - x_pred_temporal
            self._state = x_pred_temporal + k_temporal * innovation
            h_filt = self._state

        return h_filt, t_filt


# ---------------------------------------------------------------------------
#  SO3ENKBridge — ENK operating directly on SO3_Embedding channels
# ---------------------------------------------------------------------------

class SO3ENKBridge(nn.Module):
    """
    ENK bridge that operates at the SO3_Embedding level for V2 backbones.

    **Filtered channels: l=0 (scalar) and l=2 (primary tensor) ONLY.**

    Why not all L-channels
    ----------------------
    V2SO3PolarReadout uses two distinct paths:

      Linear paths (ENK-safe):
        l=0  →  proj_s (Linear)  →  scalar features
        l=2  →  proj_t (Linear)  →  traceless tensor

      Quadratic CG paths (ENK-unsafe):
        l=1  →  CG(x₁ ⊗ x₁)|_{l=2}   [_L1TensorContrib]
        l≥3  →  CG(xₗ ⊗ xₗ)|_{l=2}   [_HighLEquivariantContrib]

    When SO3ENKBridge multiplies x_l by a scalar scale = (1 + delta),
    the CG output scales as scale² (quadratic), not scale (linear).
    This causes distorted gradients and unpredictable amplification of the
    CG contributions, breaking ENK's ability to regularize the backbone.

    By restricting to {l=0, l=2} only, ENK acts on the same features as
    the V1 EquivariantNeuralKalman (projected scalars + primary tensor),
    while the CG contribution paths receive unmodified backbone features.

    EP coordination
    ---------------
    When atom_prior is passed, the R predictor is conditioned via a learned
    FiLM scale/shift.  Gate init=0 → zero conditioning at start.

    Equivariance safety
    -------------------
    Kalman gain K is scalar per atom per channel, broadcast over m.
    Multiplicative modulation x_l * (1 + delta) → equivariant because
    the same scale is applied to all m components.

    delta is clamped via tanh to (-1, 1), ensuring the multiplicative scale
    stays in (0, 2) and preventing gradient sign-flip in the backbone.
    """

    def __init__(
        self,
        sphere_channels: int,     # C
        lmax: int,
        hidden_nf: int = 128,     # H (must match readout hidden_nf)
        prior_dim: int = 128,     # EP atom_prior feature dim (0 = no EP conditioning)
        dropout: float = 0.1,
        init_r_bias: float = -1.0,
        init_q_bias: float = 1.0,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.hidden_nf = hidden_nf
        self.prior_dim = prior_dim
        C = sphere_channels

        # Only filter the two LINEAR readout paths.
        # l=1 and l≥3 use quadratic CG self-interaction → skip them.
        self._filter_l = frozenset(l for l in (0, 2) if l <= lmax)
        _num_filter = len(self._filter_l)  # 1 if lmax<2; 2 if lmax>=2

        # Q, R, and state predictors — one set per filtered L-channel only.
        # Indexed by str(l) to avoid confusion with ModuleList indexing.
        self.q_preds = nn.ModuleDict()
        self.r_preds = nn.ModuleDict()
        self.state_preds = nn.ModuleDict()
        for l in sorted(self._filter_l):
            q_pred = nn.Sequential(
                nn.LayerNorm(C),
                nn.Linear(C, max(C // 4, 4)),
                nn.SiLU(),
                nn.Linear(max(C // 4, 4), 1),
            )
            q_pred[-1].bias.data.fill_(init_q_bias)
            self.q_preds[str(l)] = q_pred

            r_pred = nn.Sequential(
                nn.LayerNorm(C),
                nn.Linear(C, max(C // 4, 4)),
                nn.SiLU(),
                nn.Linear(max(C // 4, 4), 1),
            )
            r_pred[-1].bias.data.fill_(init_r_bias)
            self.r_preds[str(l)] = r_pred

            # State predictor: outputs per-C delta, zero-init → identity at start.
            # delta is passed through tanh in forward() to keep scale in (0, 2).
            sp = nn.Sequential(
                nn.LayerNorm(C),
                nn.Linear(C, C),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(C, C),
            )
            nn.init.zeros_(sp[-1].weight)
            nn.init.zeros_(sp[-1].bias)
            self.state_preds[str(l)] = sp

        # EP FiLM: one output per filtered L (len=2 when lmax>=2)
        if prior_dim > 0:
            self.ep_film_scale = nn.Linear(prior_dim, _num_filter, bias=True)
            self.ep_film_shift = nn.Linear(prior_dim, _num_filter, bias=True)
            nn.init.zeros_(self.ep_film_scale.weight)
            nn.init.zeros_(self.ep_film_scale.bias)
            nn.init.zeros_(self.ep_film_shift.weight)
            nn.init.zeros_(self.ep_film_shift.bias)
        else:
            self.ep_film_scale = None
            self.ep_film_shift = None

    def forward(
        self,
        so3_emb: Tensor,              # [N, (lmax+1)², C]
        atom_prior: Optional[Tensor] = None,  # [N, prior_dim]  EP atom features
    ) -> Tensor:
        """
        Args:
            so3_emb:    SO3_Embedding.embedding  [N, (lmax+1)², C]
            atom_prior: Optional EP atom prior   [N, prior_dim]

        Returns:
            so3_emb_filtered: [N, (lmax+1)², C]
                l=0 and l=2 Kalman-filtered; all other L channels unchanged.
        """
        x0 = so3_emb[:, 0, :]   # [N, C]  l=0 invariant input to all predictors

        # EP FiLM: per-filtered-L adjustment on R logit
        ep_r_adjust = None
        if atom_prior is not None and self.ep_film_scale is not None:
            ep_scale = torch.tanh(self.ep_film_scale(atom_prior))   # [N, _num_filter]
            ep_shift = torch.tanh(self.ep_film_shift(atom_prior))   # [N, _num_filter]
            ep_r_adjust = (ep_scale, ep_shift)

        # Avoid full clone: only the filtered L-channels are overwritten,
        # so we use a single clone only for those slices and build the result
        # by index-scatter assignment.  This saves one full [N,25,C] buffer
        # while keeping the autograd graph intact for the unmodified channels.
        result = so3_emb.clone()
        offset = 0
        filter_idx = 0  # sequential index into EP FiLM outputs
        for l in range(self.lmax + 1):
            nm = 2 * l + 1
            if l not in self._filter_l:
                # l=1 and l≥3: CG quadratic paths — pass through unchanged.
                offset += nm
                continue

            x_l = so3_emb[:, offset:offset + nm, :]   # [N, nm, C]

            # --- State prediction ---
            # delta bounded to (-1, 1) via tanh → multiplicative scale in (0, 2).
            # This prevents: sign-flip of x_l, unbounded amplification, and
            # negative gradient factors (1 + (1-K)*delta) reaching zero or below.
            delta = torch.tanh(self.state_preds[str(l)](x0))   # [N, C]
            x_l_pred = x_l + x_l * delta.unsqueeze(1)          # [N, nm, C]

            # --- Noise estimation ---
            q_val = torch.nn.functional.softplus(self.q_preds[str(l)](x0))  # [N, 1]
            r_logit = self.r_preds[str(l)](x0)                              # [N, 1]

            # Apply EP FiLM: lower R for high-EP atoms → trust observation more
            if ep_r_adjust is not None:
                scale_l, shift_l = ep_r_adjust
                r_logit = (r_logit * (1.0 + scale_l[:, filter_idx:filter_idx + 1])
                           + shift_l[:, filter_idx:filter_idx + 1])

            r_val = torch.nn.functional.softplus(r_logit)   # [N, 1]

            # --- Kalman gain K = Q / (Q + R)  ∈ (0, 1) ---
            K = q_val / (q_val + r_val + 1e-8)              # [N, 1]

            if not hasattr(self, '_last_kalman_gains'):
                self._last_kalman_gains = {}
            self._last_kalman_gains[l] = K.detach().squeeze(-1)  # [N]

            # --- Update: x_filt = x_pred + K * (x_l - x_pred) ---
            # Expanding: x_filt = x_l * (1 + (1-K)*delta)
            # With tanh(delta) ∈ (-1,1) and K ∈ (0,1): scale stays in (max(K,0), 2-K).
            innov  = x_l - x_l_pred                              # [N, nm, C]
            x_l_filt = x_l_pred + K.unsqueeze(1) * innov         # [N, nm, C]

            result[:, offset:offset + nm, :] = x_l_filt
            offset += nm
            filter_idx += 1

        return result

