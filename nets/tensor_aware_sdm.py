"""
Tensor-Aware Spectral Dynamic Masking (TA-SDM).

Unlike the original ``SpectralMaskCrossAttention`` which only modulates
scalar (0e) features, this module jointly modulates **both** the scalar
features (h_atom) **and** the equivariant tensor features (atom_t, 2e).

Physical motivation
-------------------
The depolar (∂α/∂R) signal flows through two channels in the polar head:
  1. sa·Q(r) — scalar coefficient × geometric tensor  (0e × geometry)
  2. outt   — l=2 equivariant tensor readout            (2e direct)

The original SDM context only enters path 1 (via h_atom → sa), so the 2e
direct output is invisible to spectral conditioning.  For derivative tasks,
this means the model can still shortcut through the unmodulated tensor path.

TA-SDM injects spectral context into BOTH paths:
  - Scalar: multi-head cross-attention on h_atom (same as original SDM)
  - Tensor: FiLM conditioning on atom_t via scalar gates from spectral context
    atom_t_new = atom_t * (1 + γ·tanh(scale)) + β·shift
    This is equivariance-safe because scale/shift are scalar per-atom per-channel,
    and scalar × equivariant tensor = equivariant tensor.

SDM Auxiliary Losses (derivative-first)
---------------------------------------
Also provides standalone functions for the two SDM derivative-first losses
that were designed but not connected in ``train_equiformer_backbone.py``:
  - ``sdm_dyn_loss``: Cross-graph derivative PCC ordering for masked components
  - ``sdm_deriv_emphasis_loss``: Per-atom MSE on masked-component derivatives (MCDE)
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter_sum, scatter_max


#  Utility: scatter softmax

def _scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    max_val = scatter_max(src, index, dim=0, dim_size=dim_size)[0]
    out = torch.exp(src - max_val[index])
    denom = scatter_sum(out, index, dim=0, dim_size=dim_size)
    return out / denom[index].clamp(min=1e-8)


#  TensorAwareSpectralMaskAttention

class TensorAwareSpectralMaskAttention(nn.Module):
    """Joint scalar + tensor modulation from spectral mask context.

    Parameters
    ----------
    hidden_nf : int
        Hidden dimension (matches h_atom and atom_t channel count).
    num_heads : int
        Number of attention heads for scalar cross-attention.
    context_input_dim : int
        Input dim of context = 6 (observed values) + 6 (mask bits) = 12.
    tensor_film_enabled : bool
        Whether to apply FiLM conditioning on atom_t (2e tensor path).
        When False, falls back to scalar-only SDM (identical to original).
    """

    def __init__(
        self,
        hidden_nf: int,
        num_heads: int = 4,
        context_input_dim: int = 12,
        tensor_film_enabled: bool = True,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.num_heads = num_heads
        self.d_head = hidden_nf // num_heads
        self.tensor_film_enabled = tensor_film_enabled

        # ---- Shared context encoder ----
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf * 2),   # → K and V for scalar attn
        )

        # ---- Scalar cross-attention (same as SpectralMaskCrossAttention) ----
        self.q_proj = nn.Linear(hidden_nf, hidden_nf)
        self.out_proj = nn.Linear(hidden_nf, hidden_nf)
        self.norm_s = nn.LayerNorm(hidden_nf)
        self.gate_s = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # ---- Tensor FiLM conditioning (new) ----
        if tensor_film_enabled:
            # Context → per-atom (scale, shift) for each of hidden_nf tensor channels
            self.film_proj = nn.Sequential(
                nn.Linear(context_input_dim, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, hidden_nf * 2),   # → (scale, shift)
            )
            self.gate_t = nn.Parameter(torch.zeros(1))
            self.norm_t = nn.LayerNorm(hidden_nf)
            # Zero-init the final linear so FiLM starts as identity
            nn.init.zeros_(self.film_proj[-1].weight)
            nn.init.zeros_(self.film_proj[-1].bias)

    def forward(
        self,
        h_atom: torch.Tensor,           # [N, hidden_nf]  scalar features
        atom_t: Optional[torch.Tensor],  # [N, hidden_nf, 3, 3]  tensor features (or None)
        context_input: torch.Tensor,     # [B, 6] observed (unmasked) polar values
        context_mask: torch.Tensor,      # [B, 6] bool mask
        batch: torch.Tensor,             # [N] graph indices
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        h_atom_new : [N, hidden_nf]
        atom_t_new : [N, hidden_nf, 3, 3] or None
        """
        # ---- Build context ----
        ctx_raw = torch.cat([
            context_input.float(),
            context_mask.float(),
        ], dim=-1)  # [B, 12]

        # ---- Scalar cross-attention (identical to original) ----
        ctx_kv = self.context_encoder(ctx_raw)       # [B, 2H]
        ctx_k, ctx_v = ctx_kv.chunk(2, dim=-1)       # each [B, hidden_nf]

        ctx_k = ctx_k[batch]   # [N, hidden_nf]
        ctx_v = ctx_v[batch]   # [N, hidden_nf]

        N, H, D = h_atom.size(0), self.num_heads, self.d_head
        Q = self.q_proj(h_atom).view(N, H, D)
        K = ctx_k.view(N, H, D)
        V = ctx_v.view(N, H, D)

        attn = torch.sigmoid((Q * K).sum(dim=-1) / math.sqrt(D))  # [N, H]
        out = (attn.unsqueeze(-1) * V).reshape(N, -1)
        out = self.out_proj(out)
        h_atom_new = self.norm_s(h_atom + torch.tanh(self.gate_s) * out)

        # ---- Tensor FiLM conditioning ----
        atom_t_new = atom_t
        if self.tensor_film_enabled and atom_t is not None:
            film_params = self.film_proj(ctx_raw)            # [B, 2H]
            film_scale, film_shift = film_params.chunk(2, dim=-1)  # each [B, hidden_nf]

            # Broadcast to atom level
            film_scale = film_scale[batch]  # [N, hidden_nf]
            film_shift = film_shift[batch]  # [N, hidden_nf]

            # FiLM: atom_t_new = atom_t * (1 + gate * tanh(scale)) + gate * shift
            # scale/shift are [N, H], atom_t is [N, H, 3, 3]
            gate_val = torch.tanh(self.gate_t)
            scale = gate_val * torch.tanh(film_scale)     # [N, H]
            shift = gate_val * film_shift                  # [N, H]

            atom_t_modulated = atom_t * (1.0 + scale[:, :, None, None]) + \
                               shift[:, :, None, None]

            # Enforce tracelessness after FiLM (shift can break it)
            tr = atom_t_modulated.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / 3.0
            atom_t_modulated = atom_t_modulated - tr * torch.eye(
                3, device=atom_t.device, dtype=atom_t.dtype)

            # Apply layernorm on channel dim (reshape for LN compatibility)
            N_t, H_t = atom_t_modulated.shape[:2]
            # LN on hidden_nf: flatten spatial dims, apply, reshape back
            flat = atom_t_modulated.reshape(N_t, H_t, 9)  # [N, H, 9]
            flat = flat.permute(0, 2, 1)                    # [N, 9, H]
            flat = self.norm_t(flat)                         # LN on last dim (H)
            flat = flat.permute(0, 2, 1)                    # [N, H, 9]
            atom_t_new = flat.reshape(N_t, H_t, 3, 3)

        return h_atom_new, atom_t_new


#  SDM Derivative-First Losses

def sdm_dyn_loss(
    pred_depolar: torch.Tensor,      # [N, 3, 6] predicted derivatives
    target_depolar: torch.Tensor,    # [N, 3, 6] target derivatives
    mask_graph: torch.Tensor,        # [B, 6] bool mask (which components are masked per graph)
    batch: torch.Tensor,             # [N] graph indices
    num_graphs: int,
    min_masked_graphs: int = 3,
) -> Optional[torch.Tensor]:
    """Cross-graph derivative PCC ordering for masked polar components.

    For each masked component k, compute per-graph mean |depolar|, then
    Pearson correlation between predicted and target across graphs.
    Loss = mean(1 - PCC_k) over masked components.
    """
    if num_graphs < min_masked_graphs:
        return None

    # Per-graph mean |depolar| per polar component: [B, 6]
    pred_mag = pred_depolar.abs().mean(dim=1)    # [N, 6]
    tgt_mag = target_depolar.abs().mean(dim=1)   # [N, 6]

    pred_graph = scatter_sum(pred_mag, batch, dim=0, dim_size=num_graphs)   # [B, 6]
    tgt_graph = scatter_sum(tgt_mag, batch, dim=0, dim_size=num_graphs)     # [B, 6]

    counts = torch.bincount(batch, minlength=num_graphs).to(pred_mag.dtype).clamp(min=1.0)
    pred_graph = pred_graph / counts.unsqueeze(-1)
    tgt_graph = tgt_graph / counts.unsqueeze(-1)

    loss_terms = []
    for k in range(6):
        mask_k = mask_graph[:, k]
        n_masked = int(mask_k.sum().item())
        if n_masked < min_masked_graphs:
            continue
        pk = pred_graph[mask_k, k]
        tk = tgt_graph[mask_k, k]
        p_mean = pk.mean()
        t_mean = tk.mean()
        p_c = pk - p_mean
        t_c = tk - t_mean
        num = (p_c * t_c).sum()
        den_sq = (p_c ** 2).sum() * (t_c ** 2).sum()
        if den_sq.item() < 1e-16:
            continue
        pcc = num / den_sq.sqrt().clamp(min=1e-8)
        loss_terms.append(1.0 - pcc)

    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()


def sdm_deriv_emphasis_loss(
    pred_depolar: torch.Tensor,      # [N, 3, 6]
    target_depolar: torch.Tensor,    # [N, 3, 6]
    mask_graph: torch.Tensor,        # [B, 6] bool mask
    batch: torch.Tensor,             # [N]
    valid_mask: Optional[torch.Tensor] = None,  # [N] optional atom-level valid mask
) -> Optional[torch.Tensor]:
    """Per-atom MSE on masked-component derivatives (MCDE).

    For each masked polar component k, select atoms belonging to graphs where
    component k is masked, compute MSE on the 3-dim derivative of that component.
    """
    if mask_graph.ndim != 2 or mask_graph.size(-1) != 6:
        return None

    atom_valid = None
    if isinstance(valid_mask, torch.Tensor) and valid_mask.ndim == 1 and valid_mask.size(0) == batch.size(0):
        atom_valid = valid_mask.bool()

    loss_terms = []
    for k in range(6):
        atom_mask = mask_graph[batch, k]
        if atom_valid is not None:
            atom_mask = atom_mask & atom_valid
        if int(atom_mask.sum().item()) <= 0:
            continue
        pk = pred_depolar[atom_mask, :, k]
        tk = target_depolar[atom_mask, :, k]
        if pk.numel() == 0:
            continue
        loss_terms.append(torch.nn.functional.mse_loss(pk, tk))

    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()


def compute_sdm_aux_losses(
    pred_depolar: torch.Tensor,      # [N, 3, 6]
    target_depolar: torch.Tensor,    # [N, 3, 6]
    data,
    mask_cfg: Dict,
    depolar_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute all SDM derivative-first auxiliary losses.

    Returns a dict of named loss tensors. The caller is responsible for
    weighting and adding them to the main loss.

    Keys:
      - loss_sdm_dyn: cross-graph PCC ordering (weighted by dyn_weight)
      - loss_sdm_deriv_emph: MCDE masked-component derivative MSE (weighted by deriv_emphasis)
    """
    losses = {}
    dyn_weight = float(mask_cfg.get("dyn_weight", 0.0))
    deriv_emphasis = float(mask_cfg.get("deriv_emphasis", 0.0))

    if dyn_weight <= 0 and deriv_emphasis <= 0:
        return losses

    mask_bool = getattr(data, 'spectral_mask_polar_vec6_mask', None)
    if mask_bool is None or not isinstance(mask_bool, torch.Tensor):
        return losses
    mask_bool = mask_bool.bool()
    if mask_bool.ndim != 2 or mask_bool.size(-1) != 6:
        return losses

    if not hasattr(data, 'batch') or not isinstance(data.batch, torch.Tensor):
        return losses

    batch_idx = data.batch
    num_graphs = int(batch_idx.max().item()) + 1

    pred_dp = pred_depolar.view(-1, 3, 6)
    tgt_dp = target_depolar.view(-1, 3, 6)

    # Apply depolar_mask if provided
    valid_mask = None
    if depolar_mask is not None and depolar_mask.any():
        valid_mask = depolar_mask

    if dyn_weight > 0:
        dyn = sdm_dyn_loss(pred_dp, tgt_dp, mask_bool, batch_idx, num_graphs)
        if dyn is not None:
            losses["loss_sdm_dyn"] = dyn_weight * dyn

    if deriv_emphasis > 0:
        emph = sdm_deriv_emphasis_loss(pred_dp, tgt_dp, mask_bool, batch_idx, valid_mask)
        if emph is not None:
            losses["loss_sdm_deriv_emph"] = deriv_emphasis * emph

    return losses
