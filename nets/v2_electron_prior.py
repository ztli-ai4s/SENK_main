"""
V2-Native Electron Prior for **true point-A injection** into EquiformerV2 blocks.

Injection point A = after attention Residual (1) BEFORE FFN norm_2.
This means the FFN of each selected block fully processes the electron prior signal,
giving maximum coupling between electronic structure and geometry representations.

Architecture:
  NBOPriorBranch -> (atom_prior [N,F], edge_prior [E,F])

  InjectableTransBlockV2
    x_res = x.embedding
    x -> norm_1 -> SO2Attn -> drop_path -> += x_res   (Residual (1))
    -> INJECTION POINT A: electron prior modulates SO3_Embedding here ->
    x -> norm_2 -> FFN -> drop_path -> += x_res        (Residual (2))

  Per selected block:
    _SO3PriorBlock(x.embedding, atom_prior, edge_prior, edge_index)
      1. Encode atom_prior -> C-dim hidden via MLP
      2. Neighbor attention with edge_prior bias -> context
      3. Per-L gating: for each l=0..lmax, scale all m components
      4. l=0 additive invariant bias
      5. Global gate (init=0 -> identity start)

  upgrade_backbone_to_injectable(backbone, injector):
    In-place class swap of selected TransBlockV2 -> InjectableTransBlockV2.
    All parameters preserved (zero copy). Called once at model __init__.
"""

import math
from typing import Callable, List, Optional, Set

import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_max


def _scatter_softmax_stable(scores: torch.Tensor, dst: torch.Tensor, N: int) -> torch.Tensor:
    """Numerically stable scatter softmax via per-destination max subtraction.

    Args:
        scores: [E, H] raw attention logits
        dst:    [E]   destination node indices
        N:      int   number of destination nodes
    Returns:
        alpha:  [E, H] softmax weights (sum to 1 per destination)
    """
    max_val = scatter_max(scores, dst, dim=0, dim_size=N)[0]   # [N, H]
    exp_s = torch.exp(scores - max_val[dst])                   # [E, H]
    denom = scatter_sum(exp_s, dst, dim=0, dim_size=N)         # [N, H]
    return exp_s / denom[dst].clamp(min=1e-8)                  # [E, H]


# ---------------------------------------------------------------------------
#  _SO3PriorBlock: one injection layer
# ---------------------------------------------------------------------------

class _SO3PriorBlock(nn.Module):
    """Project scalar electron prior into full SO3_Embedding modulation.

    Per-L factored gating (rotationally consistent):
      - All m components of the same l-channel share the same gate vector
      - This preserves equivariance: rotating the molecule doesn't change
        which *channels* are gated, only how the irrep components are mixed
    """

    def __init__(
        self,
        sphere_channels: int,   # C - backbone sphere_channels
        prior_atom_dim: int,    # F - NBOPriorBranch atom output dim
        prior_edge_dim: int,    # F - NBOPriorBranch edge output dim
        lmax: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.num_heads = num_heads
        C = sphere_channels
        assert C % num_heads == 0
        self.d_head = C // num_heads

        # ---- atom prior encoder ----
        self.atom_enc = nn.Sequential(
            nn.LayerNorm(prior_atom_dim),
            nn.Linear(prior_atom_dim, C),
            nn.SiLU(),
        )

        # ---- per-L scale projections ----
        # gate_l: hidden C -> C channel-wise gate for degree l
        # All m components of an l-block are multiplied by the same gate vector
        self.l_gate_projs = nn.ModuleList([
            nn.Linear(C, C, bias=True) for _ in range(lmax + 1)
        ])
        # l=0 additive invariant bias
        self.l0_bias_proj = nn.Linear(C, C)

        # ---- edge-prior neighbor attention ----
        self.q_proj = nn.Linear(C, C)
        self.k_proj = nn.Linear(prior_atom_dim, C)
        self.v_proj = nn.Linear(prior_atom_dim, C)
        self.edge_bias_proj = nn.Linear(prior_edge_dim, num_heads)
        self.neighbor_out = nn.Linear(C, C)

        # ---- global gate: non-zero init ensures gradients reach delta projections from step 1 ----
        # tanh(1)≈0.76 so identity is preserved by zero-init l_gate_projs, not by gate=0.
        # With gate=0 AND injection=0 (from bias=1 multiplicative gate), BOTH parameters
        # receive exactly zero gradient — a structural deadlock that prevents Point-A from ever
        # learning. Using gate=1 + additive-delta formula breaks this deadlock.
        self.gate = nn.Parameter(torch.ones(1))

        # Zero-init all output projections → identity (zero injection) at start
        nn.init.zeros_(self.neighbor_out.weight)
        nn.init.zeros_(self.neighbor_out.bias)
        nn.init.zeros_(self.l0_bias_proj.weight)
        nn.init.zeros_(self.l0_bias_proj.bias)
        for proj in self.l_gate_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)   # additive delta=0 at init; identity via zero projection

    def forward(
        self,
        x_embedding: torch.Tensor,   # [N, (lmax+1)^2, C]
        atom_prior: torch.Tensor,     # [N, F]
        edge_prior: torch.Tensor,     # [E, F]
        edge_index: torch.Tensor,     # [2, E]
    ) -> torch.Tensor:
        N, num_coeffs, C = x_embedding.shape
        global_gate = torch.tanh(self.gate)

        # ---- encode atom prior ----
        h = self.atom_enc(atom_prior)       # [N, C]

        # ---- neighbor attention for edge context ----
        src, dst = edge_index[0], edge_index[1]
        H, D = self.num_heads, self.d_head
        Q = self.q_proj(h).view(N, H, D)
        K = self.k_proj(atom_prior).view(N, H, D)
        V = self.v_proj(atom_prior).view(N, H, D)

        scores = ((Q[dst] * K[src]).sum(dim=-1) / math.sqrt(D)   # [E, H]
                  + self.edge_bias_proj(edge_prior))               # [E, H]

        # Numerically stable scatter softmax (scatter_max for per-dst max)
        alpha = _scatter_softmax_stable(scores, dst, N)            # [E, H]

        agg = scatter_sum(
            alpha.unsqueeze(-1) * V[src],   # [E, H, D]
            dst, dim=0, dim_size=N,          # [N, H, D]
        )
        h_combined = h + self.neighbor_out(agg.reshape(N, -1))   # [N, C]

        # ---- per-L additive delta injection (equivariant: shared delta across m) ----
        # Formula: injection_l = delta_l(h)  [additive, zero-init → 0 at start]
        # Previously used multiplicative gating (blk * gate_l - blk) which created a structural
        # gradient deadlock: injection=0 when gate_l=1 → global_gate gets zero gradient;
        # global_gate=0 → gate_l parameters get zero gradient. Both stuck simultaneously.
        # Additive delta avoids this: delta_l weights receive gradient = global_gate * h_combined
        # (non-zero once global_gate > 0, which is immediate with gate=1 init).
        injection = torch.zeros_like(x_embedding)
        offset = 0
        for l in range(self.lmax + 1):
            nm = 2 * l + 1
            delta_l = self.l_gate_projs[l](h_combined)           # [N, C], zero-init → 0 at start
            # Additive injection: broadcast delta across all 2l+1 m-components (equivariant)
            injection[:, offset: offset + nm, :] = delta_l.unsqueeze(1).expand(-1, nm, -1)
            if l == 0:
                injection[:, offset, :] += self.l0_bias_proj(h_combined)
            offset += nm

        return x_embedding + global_gate * injection


# ---------------------------------------------------------------------------
#  InjectableTransBlockV2: point-A hook inside TransBlockV2
# ---------------------------------------------------------------------------

class InjectableTransBlockV2:
    """Mixin that adds point-A injection to a TransBlockV2 instance.

    NOT a standalone class - used only via ``upgrade_backbone_to_injectable``,
    which in-place swaps the ``__class__`` of existing TransBlockV2 objects.
    All module parameters are preserved untouched.

    Overrides forward() to fire ``self._point_a_hook(so3_embedding)`` after
    Residual (1) and before FFN (injection point A).
    """

    def set_point_a_hook(self, hook: Callable):
        """Install point-A hook. hook(SO3_Embedding) -> SO3_Embedding."""
        self._point_a_hook = hook

    def forward(self, x, atomic_numbers, edge_distance, edge_index, batch):
        # Identical to TransBlockV2.forward() except for the hook call at point A.
        output_embedding = x

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(output_embedding, atomic_numbers,
                                   edge_distance, edge_index)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch)

        output_embedding.embedding = output_embedding.embedding + x_res  # Residual (1)

        # ======= INJECTION POINT A =======
        if getattr(self, '_point_a_hook', None) is not None:
            output_embedding = self._point_a_hook(output_embedding)
        # ==================================

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_2(output_embedding.embedding)
        output_embedding = self.ffn(output_embedding)

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch)
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch)

        if self.ffn_shortcut is not None:
            from .eqv2_core.so3 import SO3_Embedding
            sc = SO3_Embedding(
                0, output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device, dtype=output_embedding.dtype,
            )
            sc.set_embedding(x_res)
            sc.set_lmax_mmax(output_embedding.lmax_list.copy(),
                             output_embedding.lmax_list.copy())
            sc = self.ffn_shortcut(sc)
            x_res = sc.embedding

        output_embedding.embedding = output_embedding.embedding + x_res  # Residual (2)
        return output_embedding


# ---------------------------------------------------------------------------
#  SO3ElectronPriorInjector
# ---------------------------------------------------------------------------

class SO3ElectronPriorInjector(nn.Module):
    """
    V2-native electron prior: deep injection at TransBlockV2 point A.

    Usage (in CleanEquiformerPolarExt.__init__)::

        self.so3_ep_injector = SO3ElectronPriorInjector(...)
        upgrade_backbone_to_injectable(self.backbone, self.so3_ep_injector)

    Usage (in CleanEquiformerPolarExt.forward)::

        self.so3_ep_injector.prepare(atom_prior, edge_prior, ep_edge_index)
        h_eq = self.backbone.forward_features_equivariant(pos, batch, z)
        # -> hooks inside blocks fire automatically during backbone forward
        self.so3_ep_injector.clear_cache()
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        num_backbone_layers: int,
        prior_atom_dim: int,
        prior_edge_dim: int,
        num_heads: int = 4,
        inject_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.num_backbone_layers = num_backbone_layers

        if inject_layers is None:
            # Default: last half of blocks
            inject_layers = list(range(num_backbone_layers // 2,
                                       num_backbone_layers))
        self.inject_layer_set: Set[int] = set(inject_layers)

        self.prior_blocks = nn.ModuleDict({
            str(idx): _SO3PriorBlock(
                sphere_channels=sphere_channels,
                prior_atom_dim=prior_atom_dim,
                prior_edge_dim=prior_edge_dim,
                lmax=lmax,
                num_heads=num_heads,
            )
            for idx in sorted(self.inject_layer_set)
        })

        # Cached priors (set per forward pass)
        self._atom_prior: Optional[torch.Tensor] = None
        self._edge_prior: Optional[torch.Tensor] = None
        self._edge_index: Optional[torch.Tensor] = None

    def prepare(
        self,
        atom_prior: torch.Tensor,
        edge_prior: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """Cache electron priors for the current forward pass."""
        self._atom_prior = atom_prior
        self._edge_prior = edge_prior
        self._edge_index = edge_index

    def clear_cache(self):
        """Deferred release: do not immediately set to None; let the next prepare() call naturally overwrite.

        Reason: when using gradient_checkpointing, torch.autograd.grad(create_graph=True)
        and loss.backward() trigger block recomputation after forward returns.
        Recomputation re-executes _point_a_hook, which still needs to read self._atom_prior and other caches.
        If set to None immediately here, the hook would receive NoneType, causing LayerNorm to crash.
        Actual release happens when the next batch's prepare() overwrites the old reference, freeing GPU memory.
        """
        pass  # deferred: released on next prepare()

    def _make_point_a_hook(self, block_idx: int) -> Callable:
        """Return a closure that fires _SO3PriorBlock at point A for block_idx."""
        def hook(so3_emb):
            so3_emb.embedding = self.prior_blocks[str(block_idx)](
                so3_emb.embedding,
                self._atom_prior,
                self._edge_prior,
                self._edge_index,
            )
            return so3_emb
        return hook


# ---------------------------------------------------------------------------
#  upgrade_backbone_to_injectable  (called once at model __init__)
# ---------------------------------------------------------------------------

def upgrade_backbone_to_injectable(
    backbone: nn.Module,
    injector: SO3ElectronPriorInjector,
) -> None:
    """
    In-place upgrade: swap selected TransBlockV2 -> InjectableTransBlockV2.

    Uses Python class-swap trick: changes ``blk.__class__`` to
    ``InjectableTransBlockV2`` (a mixin class) and installs the point-A hook.
    All module parameters, buffers, and sub-modules are PRESERVED - no copies.

    Must be called ONCE during model __init__, after both backbone and injector
    are constructed.
    """
    from .eqv2_core.transformer_block import TransBlockV2

    for idx in sorted(injector.inject_layer_set):
        blk = backbone.blocks[idx]
        if not isinstance(blk, TransBlockV2):
            raise TypeError(
                f"backbone.blocks[{idx}] is {type(blk)}, expected TransBlockV2"
            )
        # Build a new class that inherits BOTH: original TransBlockV2 (for __init__,
        # parameters, all sub-modules) and InjectableTransBlockV2 (for forward).
        # MRO: InjectableTransBlockV2 forward() shadows TransBlockV2 forward().
        InjectableCls = type(
            'InjectableTransBlockV2',
            (InjectableTransBlockV2, type(blk)),
            {},
        )
        blk.__class__ = InjectableCls
        blk._point_a_hook = None
        blk.set_point_a_hook(injector._make_point_a_hook(idx))
