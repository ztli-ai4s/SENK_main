"""
Extended Clean Equiformer with **Spectral Mask Attention**, **Tensor SDM**,
**V2-Native Electron Prior**, and **V2-native SO3 ENK + Readout**.

When all extensions are disabled and backbone is V1, the forward path is
bit-identical to ``CleanEquiformerPolar``.

**V2 Full-Performance Path** (new, used when backbone=EquiformerV2 and the
    mode does not require legacy post-decomposition modulation):
  1. EP pre-computed before backbone.
  2. EP injected at TransBlock point-A (SO3ElectronPriorInjector).
  3. backbone.forward_features_so3() → SO3_Embedding [N, (lmax+1)², C]
    4. Optional SO3ENKBridge filters each L-channel independently with
         EP-FiLM conditioning. ENK Q/R predicted from l=0 invariants; EP adjusts
         R (higher EP trust).
    5. V2SO3PolarReadout reads all channels:
       l=0  → scalar head
       l=1  → vector→traceless-rank2 (gated)
       l=2  → primary equivariant tensor (same as V1)
       l≥3  → high-L equivariant contributions (gated)
  6. Result: per-atom [N, 3, 3] → sum → [B, 3, 3].

**Legacy decomposition path** (for V1 backbone, or V2 modes that still need
    scalar/tensor post-processing such as spectral-mask / tensor-SDM):
    Falls back to the original flat-tensor decomposition + V1 readout.
    Electron prior is V2-native if backbone is V2, V1-fallback otherwise.

All modules are **residually gated** with gate initialised to 0.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter_sum, scatter_max

from e3nn import o3

from . import model_entrypoint
from .graph_attention_transformer import safe_radius_graph
from .tensor_aware_sdm import TensorAwareSpectralMaskAttention


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _compute_2e_to_cart33() -> torch.Tensor:
    """Change-of-basis [9, 5]: l=2 real SH → traceless symmetric 3×3."""
    test_vecs = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
        [1., 1., 1.], [-1., 1., 0.], [1., -1., 1.],
    ], dtype=torch.float64)
    test_vecs = test_vecs / test_vecs.norm(dim=-1, keepdim=True)
    Y2 = o3.spherical_harmonics(2, test_vecs, normalize=False,
                                normalization='component').double()
    T_list = []
    for v in test_vecs:
        rr = v.unsqueeze(-1) * v.unsqueeze(-2)
        rr_tl = rr - rr.trace() / 3.0 * torch.eye(3, dtype=torch.float64)
        T_list.append(rr_tl.flatten())
    T = torch.stack(T_list)
    U_T = torch.linalg.lstsq(Y2, T).solution
    return U_T.T.float()


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions for distance encoding."""
    def __init__(self, start: float = 0.0, end: float = 5.0, num_basis: int = 20):
        super().__init__()
        self.register_buffer('centers', torch.linspace(start, end, num_basis))
        self.gamma = 1.0 / ((end - start) / max(num_basis, 1)) ** 2

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


def _scatter_softmax(
    src: torch.Tensor,   # [E, H]
    index: torch.Tensor, # [E]
    dim_size: int,
) -> torch.Tensor:
    """Numerically-stable scatter softmax over ``index``."""
    max_val = scatter_max(src, index, dim=0, dim_size=dim_size)[0]   # [N, H]
    out = torch.exp(src - max_val[index])
    denom = scatter_sum(out, index, dim=0, dim_size=dim_size)        # [N, H]
    return out / denom[index].clamp(min=1e-8)


# ---------------------------------------------------------------------------
#  SpectralMaskCrossAttention
# ---------------------------------------------------------------------------

class SpectralMaskCrossAttention(nn.Module):
    """Spectral mask → graph-level context → atom feature modulation.

    Physical motivation
    -------------------
    When some polar-tensor components are masked (hidden), the remaining
    *observed* values serve as partial spectral knowledge.  Injecting this
    knowledge into the model forces it to learn cross-component correlations
    through geometry rather than relying on shortcut interpolation.

    Architecture
    ------------
    1. Context encoder: ``(observed_vec6 ‖ mask_bool) → K, V``  per graph
    2. Multi-head cross-attention:
       * Q from per-atom hidden features
       * K, V from graph context broadcast to atom level
       * single-key attention uses sigmoid gating
    3. Gated residual (gate=0 → identity init → model starts as baseline)
    """

    def __init__(self, hidden_nf: int, num_heads: int = 4, context_input_dim: int = 12):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.num_heads = num_heads
        self.d_head = hidden_nf // num_heads

        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf * 2),   # → K and V
        )

        self.q_proj = nn.Linear(hidden_nf, hidden_nf)
        self.out_proj = nn.Linear(hidden_nf, hidden_nf)
        self.norm = nn.LayerNorm(hidden_nf)

        # Identity init
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,              # [N, hidden_nf]
        context_input: torch.Tensor,   # [B, 6] observed (masked) values
        context_mask: torch.Tensor,    # [B, 6] bool mask
        batch: torch.Tensor,           # [N]
    ) -> torch.Tensor:
        ctx_raw = torch.cat([context_input.float(), context_mask.float()], dim=-1)  # [B, 12]
        ctx_kv = self.context_encoder(ctx_raw)       # [B, 2H]
        ctx_k, ctx_v = ctx_kv.chunk(2, dim=-1)       # each [B, hidden_nf]

        ctx_k = ctx_k[batch]   # [N, hidden_nf]
        ctx_v = ctx_v[batch]   # [N, hidden_nf]

        N, H, D = h.size(0), self.num_heads, self.d_head
        Q = self.q_proj(h).view(N, H, D)
        K = ctx_k.view(N, H, D)
        V = ctx_v.view(N, H, D)

        attn = torch.sigmoid((Q * K).sum(dim=-1) / math.sqrt(D))  # [N, H]
        out = (attn.unsqueeze(-1) * V).reshape(N, -1)
        out = self.out_proj(out)

        return self.norm(h + torch.tanh(self.gate) * out)


# ---------------------------------------------------------------------------
#  ElectronPriorAttention
# ---------------------------------------------------------------------------

class ElectronPriorAttention(nn.Module):
    """Electron prior → neighbor-aware attention → feature enhancement.

    Physical motivation
    -------------------
    NBO electronic structure (atom populations, bond orders, interaction
    energies) encodes the electron density landscape.  This attention layer
    lets backbone features query the electronic structure to produce
    *electron-informed* representations that respect the true electronic
    environment, not just geometry.

    Architecture
    ------------
    1. Atom prior → K, V; backbone features → Q
    2. Edge-level attention: ``score_ij = Q_i · K_j / √d + bias_ij``
       where ``bias_ij`` comes from edge prior projection
    3. Scatter softmax over neighbours → weighted sum of V
    4. Gated residual (gate=0 → identity init)
    """

    def __init__(
        self,
        hidden_nf: int,
        prior_atom_dim: int,
        prior_edge_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.num_heads = num_heads
        self.d_head = hidden_nf // num_heads

        self.q_proj = nn.Linear(hidden_nf, hidden_nf)
        self.k_proj = nn.Linear(prior_atom_dim, hidden_nf)
        self.v_proj = nn.Linear(prior_atom_dim, hidden_nf)
        self.edge_bias_proj = nn.Linear(prior_edge_dim, num_heads)
        self.out_proj = nn.Linear(hidden_nf, hidden_nf)
        self.norm = nn.LayerNorm(hidden_nf)

        # Identity init
        self.gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,            # [N, hidden_nf]
        atom_prior: torch.Tensor,    # [N, prior_atom_dim]
        edge_prior: torch.Tensor,    # [E, prior_edge_dim]
        edge_index: torch.Tensor,    # [2, E]
    ) -> torch.Tensor:
        N, H, D = h.size(0), self.num_heads, self.d_head
        src, dst = edge_index          # messages: src → dst

        Q = self.q_proj(h).view(N, H, D)
        K = self.k_proj(atom_prior).view(N, H, D)
        V = self.v_proj(atom_prior).view(N, H, D)

        # Edge-level attention scores
        Q_dst = Q[dst]                # [E, H, D]
        K_src = K[src]                # [E, H, D]
        scores = (Q_dst * K_src).sum(dim=-1) / math.sqrt(D)  # [E, H]
        scores = scores + self.edge_bias_proj(edge_prior)     # [E, H]

        attn = _scatter_softmax(scores, dst, dim_size=N)      # [E, H]

        V_src = V[src]                                        # [E, H, D]
        msg = attn.unsqueeze(-1) * V_src                      # [E, H, D]
        agg = scatter_sum(msg, dst, dim=0, dim_size=N)        # [N, H, D]

        out = self.out_proj(agg.reshape(N, -1))
        return self.norm(h + torch.tanh(self.gate) * out)


# ---------------------------------------------------------------------------
#  CleanEquiformerPolarExt
# ---------------------------------------------------------------------------

class CleanEquiformerPolarExt(nn.Module):
    """Extended Clean Equiformer with optional spectral mask / electron prior.

    When *both* ``spectral_mask_enabled=False`` and
    ``electron_prior_config=None`` the forward is identical to
    ``CleanEquiformerPolar`` (same backbone, same head, same param count).

    **V2-native electron prior**: when the backbone is EquiformerV2
    (detected via ``hasattr(backbone, 'forward_features_so3')``), the
    electron prior uses ``SO3ElectronPriorInjector`` for deep in-block
    injection on the full SO3_Embedding.  Falls back to scalar-only
    ``ElectronPriorAttention`` for V1 backbones.
    """

    def __init__(
        self,
        hidden_nf: int = 128,
        model_name: str = 'graph_attention_transformer_nonlinear_l2',
        radius: float = 5.0,
        num_basis: int = 128,
        num_layers: int = 4,
        drop_path: float = 0.1,
        dropout: float = 0.1,
        summation: bool = True,
        grad_type: Optional[str] = None,
        # --- Extension options ---
        spectral_mask_enabled: bool = False,
        spectral_mask_heads: int = 4,
        tensor_sdm_enabled: bool = False,
        tensor_sdm_heads: int = 4,
        electron_prior_config: Optional[Dict] = None,
        electron_prior_heads: int = 4,
        # --- ENK options ---
        enk_enabled: bool = False,
        enk_init_r_bias: float = -1.0,
        enk_init_q_bias: float = 1.0,
        # --- V2 backbone memory-efficiency options ---
        max_num_neighbors: int = 64,
        num_gaussians: int = 64,
        use_gradient_checkpointing: bool = False,
        grid_resolution: int = 14,
        use_gate_act: bool = True,    # GateActivation: ~33x cheaper than SeparableS2Activation (no SO3_Grid einsum)
        use_grid_mlp: bool = False,   # disable S²-grid MLP in FFN for QM9 speed
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.summation = summation
        self.grad_type = grad_type
        self.radius = radius
        self.spectral_mask_enabled = spectral_mask_enabled
        self.tensor_sdm_enabled = tensor_sdm_enabled
        self.electron_prior_enabled = electron_prior_config is not None
        self.enk_enabled = enk_enabled

        # ---- Backbone ----
        create_model = model_entrypoint(model_name)
        self.backbone = create_model(
            irreps_in='5x0e',
            radius=radius,
            num_basis=num_basis,
            task_mean=0.0,
            task_std=1.0,
            atomref=None,
            drop_path=drop_path,
            ssp=False,
            num_layers=num_layers,
            # V2-specific memory/speed options (silently ignored by V1 factory)
            max_num_neighbors=max_num_neighbors,
            num_gaussians=num_gaussians,
            use_gradient_checkpointing=use_gradient_checkpointing,
            grid_resolution=grid_resolution,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
        )

        # Detect V2 backbone (has forward_features_so3)
        self._is_v2 = hasattr(self.backbone, 'forward_features_so3')

        # V2 full-performance path: raw SO3_Embedding + optional SO3ENKBridge
        # + V2SO3PolarReadout. Modes that depend on the legacy scalar/tensor
        # post-processing path (spectral-mask / tensor-SDM) stay on the old branch.
        self._use_v2_native_path = self._is_v2 and not (
            spectral_mask_enabled or tensor_sdm_enabled
        )

        # ---- Parse irreps — supports all L levels ----
        _irreps = self.backbone.irreps_node_embedding
        self._l_info = {}
        _off = 0
        for mul, ir in _irreps:
            l = ir.l
            if l not in self._l_info:
                self._l_info[l] = (0, _off)
            old_mul, old_off = self._l_info[l]
            self._l_info[l] = (old_mul + mul, old_off if old_mul == 0 else old_off)
            _off += mul * ir.dim

        self._n0e = self._l_info.get(0, (0, 0))[0]
        self._n1e = self._l_info.get(1, (0, 0))[0]
        self._n2e = self._l_info.get(2, (0, 0))[0]
        self._off_0e = self._l_info.get(0, (0, 0))[1]
        self._off_1e = self._l_info.get(1, (0, 0))[1]
        self._off_2e = self._l_info.get(2, (0, 0))[1]
        self._backbone_lmax = max(self._l_info.keys()) if self._l_info else 2

        # ================================================================
        #  V2 NATIVE PATH: raw SO3 path + optional SO3ENKBridge + V2 readout
        # ================================================================
        if self._use_v2_native_path:
            # sphere_channels is the V2 backbone C
            _C = self.backbone.sphere_channels

            self.so3_enk_bridge = None
            if enk_enabled:
                # EP prior dim for ENK FiLM conditioning (set to hidden_nf if EP enabled)
                _ep_prior_dim = hidden_nf if electron_prior_config is not None else 0

                # SO3ENKBridge: per-L-channel Kalman filter on SO3_Embedding
                from .equivariant_neural_kalman import SO3ENKBridge
                self.so3_enk_bridge = SO3ENKBridge(
                    sphere_channels=_C,
                    lmax=self._backbone_lmax,
                    hidden_nf=hidden_nf,
                    prior_dim=_ep_prior_dim,
                    dropout=dropout,
                    init_r_bias=enk_init_r_bias,
                    init_q_bias=enk_init_q_bias,
                )

            # V2SO3PolarReadout: full l=0..lmax equivariant readout
            from .v2_so3_polar_readout import V2SO3PolarReadout
            self.v2_readout = V2SO3PolarReadout(
                sphere_channels=_C,
                lmax=self._backbone_lmax,
                hidden_nf=hidden_nf,
                dropout=dropout,
            )

            # Shared tril mask for depolar
            tril_mask = torch.tril(torch.ones(3, 3)).flatten()
            self.register_buffer('_tril_mask', tril_mask)

        else:
            # ---- V1 / V2-baseline path: high-L pooling + V1 readout ----

            # High-L pooling (l≥3 → invariant norm → scalar enrichment)
            _highL_total = 0
            for l in range(3, self._backbone_lmax + 1):
                if l in self._l_info:
                    _highL_total += self._l_info[l][0] * (2 * l + 1)
            self._highL_total = _highL_total

            if _highL_total > 0:
                _highL_muls = sum(
                    self._l_info[l][0]
                    for l in range(3, self._backbone_lmax + 1)
                    if l in self._l_info
                )
                self._highL_muls = _highL_muls
                self.highL_pool = nn.Sequential(
                    nn.LayerNorm(_highL_muls),
                    nn.Linear(_highL_muls, hidden_nf),
                    nn.SiLU(),
                    nn.Linear(hidden_nf, hidden_nf),
                )
            else:
                self._highL_muls = 0
                self.highL_pool = None

            # Projections
            _n0e = self._n0e
            _n2e = self._n2e
            self.proj_s = nn.Sequential(
                nn.LayerNorm(_n0e),
                nn.Linear(_n0e, hidden_nf),
            )
            if _n2e > 0:
                self.proj_t = nn.Linear(_n2e, hidden_nf, bias=False)
                cob = _compute_2e_to_cart33()
                self.register_buffer('_cob_2e_to_cart', cob)
            else:
                self.proj_t = None
                self.register_buffer('_cob_2e_to_cart', torch.zeros(9, 5))

            # Polar head
            self.sout = nn.Sequential(
                nn.LayerNorm(hidden_nf),
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_nf, 2),
            )
            if _n2e > 0:
                self.t_readout = nn.Linear(hidden_nf, 1, bias=False)
            else:
                self.t_readout = None

            # Mass table
            mass_table = torch.ones(100, dtype=torch.float32)
            mass_table[1] = 1.0079; mass_table[6] = 12.011; mass_table[7] = 14.0067
            mass_table[8] = 15.999; mass_table[9] = 18.998
            self.register_buffer('_mass_table', mass_table)

            tril_mask = torch.tril(torch.ones(3, 3)).flatten()
            self.register_buffer('_tril_mask', tril_mask)

            # Optional: old-style ENK on projected features (V1/V2-baseline)
            self.enk = None
            if enk_enabled:
                from .equivariant_neural_kalman import EquivariantNeuralKalman
                self.enk = EquivariantNeuralKalman(
                    hidden_nf=hidden_nf,
                    dropout=dropout,
                    init_r_bias=enk_init_r_bias,
                    init_q_bias=enk_init_q_bias,
                    tensor_gate=True,
                )

        # ================================================================
        #  Extension 1a: Spectral Mask Cross-Attention (scalar-only)
        # ================================================================
        if spectral_mask_enabled and not tensor_sdm_enabled:
            self.spectral_mask_attn = SpectralMaskCrossAttention(
                hidden_nf=hidden_nf,
                num_heads=spectral_mask_heads,
                context_input_dim=12,
            )

        # ================================================================
        #  Extension 1b: Tensor-Aware SDM (scalar + tensor joint)
        # ================================================================
        if tensor_sdm_enabled:
            self.tensor_sdm_attn = TensorAwareSpectralMaskAttention(
                hidden_nf=hidden_nf,
                num_heads=tensor_sdm_heads,
                context_input_dim=12,
                tensor_film_enabled=True,
            )

        # ================================================================
        #  Extension 2: Electron Prior
        #  V2 backbone → SO3ElectronPriorInjector (deep in-block injection)
        #  V1 backbone → ElectronPriorAttention (scalar post-fusion, legacy)
        # ================================================================
        if self.electron_prior_enabled:
            from detanet_nets.electron_prior import NBOPriorBranch

            ep_cfg = dict(electron_prior_config)
            ep_num_features = hidden_nf
            ep_num_radial = 20
            self.electron_prior = NBOPriorBranch(
                num_features=ep_num_features,
                num_radial=ep_num_radial,
                mode=ep_cfg.pop('mode', 'simg'),
                checkpoint_path=ep_cfg.pop('checkpoint_path', ''),
                stats_path=ep_cfg.pop('stats_path', None),
                hidden_dim=ep_cfg.pop('hidden_dim', 128),
                max_atomic_number=ep_cfg.pop('max_atomic_number', 35),
                feature_scale=ep_cfg.pop('feature_scale', 1e-2),
                use_auxiliary=ep_cfg.pop('use_auxiliary', True),
                freeze_predictor=ep_cfg.pop('freeze_predictor', True),
                runtime_mode=ep_cfg.pop('runtime_mode', 'full'),
                predictor_state_dict=ep_cfg.pop('predictor_state_dict', None),
            )
            self.electron_rbf = GaussianRBF(0.0, radius, ep_num_radial)

            if self._is_v2:
                # V2-native: point-A deep injection into TransBlock loop
                from .v2_electron_prior import (
                    SO3ElectronPriorInjector, upgrade_backbone_to_injectable,
                )
                self.so3_ep_injector = SO3ElectronPriorInjector(
                    sphere_channels=self.backbone.sphere_channels,
                    lmax=self._backbone_lmax,
                    num_backbone_layers=self.backbone.num_layers,
                    prior_atom_dim=ep_num_features,
                    prior_edge_dim=ep_num_features,
                    num_heads=electron_prior_heads,
                )
                # In-place class swap: selected blocks gain point-A hook
                upgrade_backbone_to_injectable(self.backbone, self.so3_ep_injector)

                if not self._use_v2_native_path:
                    # V2-baseline: post-decomposition tensor path modulation
                    # (only needed when NOT using V2SO3PolarReadout)
                    self.ep_tensor_gate = nn.Parameter(torch.zeros(1))
                    self.ep_tensor_proj = nn.Sequential(
                        nn.Linear(ep_num_features, hidden_nf),
                        nn.SiLU(),
                        nn.Linear(hidden_nf, hidden_nf),
                    )
                    nn.init.zeros_(self.ep_tensor_proj[-1].weight)
                    nn.init.zeros_(self.ep_tensor_proj[-1].bias)
            else:
                # V1 fallback: scalar-only post-fusion attention
                self.electron_prior_attn = ElectronPriorAttention(
                    hidden_nf=hidden_nf,
                    prior_atom_dim=ep_num_features,
                    prior_edge_dim=ep_num_features,
                    num_heads=electron_prior_heads,
                )

    # ----- helper: center of mass (V1/V2-baseline path) -----
    def _center_of_mass(self, pos, z, batch):
        num_graphs = int(batch.max().item()) + 1
        z_clamped = z.clamp(min=0, max=self._mass_table.numel() - 1)
        masses = self._mass_table[z_clamped].to(dtype=pos.dtype, device=pos.device).unsqueeze(-1)
        mass_sum = scatter_sum(masses, batch, dim=0, dim_size=num_graphs).clamp(min=1e-8)
        pos_mass_sum = scatter_sum(pos * masses, batch, dim=0, dim_size=num_graphs)
        return pos_mass_sum / mass_sum

    # ----- helper: per-atom polar tensor (V1/V2-baseline path) -----
    def _polar_tensor(self, h_atom, atom_t, pos, z, batch):
        eye3 = torch.eye(3, device=pos.device, dtype=pos.dtype)
        s = self.sout(h_atom)
        sa, sb = s[:, :1], s[:, 1:]

        if atom_t is not None and self.t_readout is not None:
            t_comp = atom_t.permute(0, 2, 3, 1)
            outt = self.t_readout(t_comp).squeeze(-1)
            outt = 0.5 * (outt + outt.transpose(-1, -2))
            tr_out = (outt[:, 0, 0] + outt[:, 1, 1] + outt[:, 2, 2]) / 3.0
            outt = outt - tr_out[:, None, None] * eye3
        else:
            outt = pos.new_zeros(pos.shape[0], 3, 3)

        com = self._center_of_mass(pos, z, batch)
        ra = pos - com[batch]
        Q = ra.unsqueeze(-1) * ra.unsqueeze(-2)
        r_sq = (ra * ra).sum(dim=-1, keepdim=True).unsqueeze(-1)
        Q = Q - (r_sq / 3.0) * eye3

        return sb.unsqueeze(-1) * (eye3 / 3.0) + outt + sa.unsqueeze(-1) * Q

    # ----- helper: autograd depolar -----
    def _grad_polarizability(self, polars, pos):
        polars_flat = polars.flatten(start_dim=1)[:, self._tril_mask == 1]
        depolar = torch.zeros(pos.shape[0], 3, 6, device=pos.device, dtype=pos.dtype)
        for i in range(6):
            depolar[:, :, i] = -torch.autograd.grad(
                polars_flat[:, i].sum(), pos,
                create_graph=self.training, retain_graph=True,
            )[0]
        return depolar

    # ================================================================
    #  Forward
    # ================================================================
    def forward(self, *, pos, z, batch, data=None, **kwargs):
        """
        Returns
        -------
        If grad_type is None  (polar model):  ``[B, 3, 3]`` molecular polarizability
        If grad_type == polar (depolar model): ``[N, 3, 6]`` ∂α/∂R

                **V2 full-performance path** (when _use_v2_native_path=True):
                    EP → SO3 point-A injection → optional SO3ENKBridge (EP-conditioned)
                    → V2SO3PolarReadout

                **Legacy decomposition path** (otherwise):
          Flat feature decomposition → V1 readout (same as before)
        """
        if self.grad_type == 'polar':
            pos = pos.detach().requires_grad_(True)

        # ---- Electron prior: pre-compute priors (needed before backbone for V2) ----
        atom_prior = edge_prior = ep_edge_index = None
        if self.electron_prior_enabled and hasattr(self, 'electron_prior'):
            edge_src, edge_dst = safe_radius_graph(
                pos, batch, r=self.radius, max_num_neighbors=1000,
            )
            edge_vec = pos[edge_src] - pos[edge_dst]
            edge_dist = edge_vec.norm(dim=-1)
            rbf_feat = self.electron_rbf(edge_dist)
            ep_edge_index = torch.stack([edge_src, edge_dst], dim=0)

            atom_prior, edge_prior = self.electron_prior(
                z, pos, ep_edge_index, rbf_feat, data=data,
            )

        # ---- Prepare V2 EP injector cache (hooks fire inside blocks) ----
        if self._is_v2 and self.electron_prior_enabled and hasattr(self, 'so3_ep_injector'):
            self.so3_ep_injector.prepare(atom_prior, edge_prior, ep_edge_index)

        # ==============================================================
        #  BRANCH A: V2 full-performance path
        #  SO3_Embedding → SO3ENKBridge → V2SO3PolarReadout
        # ==============================================================
        if self._use_v2_native_path:
            # Get raw SO3_Embedding (EP hooks fire inside backbone blocks)
            so3_obj = self.backbone.forward_features_so3(
                pos=pos, batch=batch, node_atom=z,
            )

            # Clear EP injector cache
            if self.electron_prior_enabled and hasattr(self, 'so3_ep_injector'):
                self.so3_ep_injector.clear_cache()

            if self.so3_enk_bridge is not None:
                # SO3ENKBridge: per-L-channel Kalman filter
                # EP atom_prior conditions the R predictor (EP-ENK coordination)
                so3_filtered = self.so3_enk_bridge(
                    so3_obj.embedding,                    # [N, (lmax+1)², C]
                    atom_prior=atom_prior,                 # None if EP disabled
                )
            else:
                so3_filtered = so3_obj.embedding

            # V2SO3PolarReadout: l=0..lmax full equivariant readout
            per_atom_polar = self.v2_readout(
                so3_filtered, pos, z, batch,
            )

        # ==============================================================
        #  BRANCH B: V1 / V2-baseline path (flat feature decomposition)
        # ==============================================================
        else:
            h_eq = self.backbone.forward_features_equivariant(
                pos=pos, batch=batch, node_atom=z,
            )

            # Clear EP injector cache
            if self._is_v2 and self.electron_prior_enabled and hasattr(self, 'so3_ep_injector'):
                self.so3_ep_injector.clear_cache()

            # ---- Decompose irreps ----
            h_s_raw = h_eq[:, self._off_0e: self._off_0e + self._n0e]
            h_atom = self.proj_s(h_s_raw)

            # High-L pooling: l≥3 invariant norm → scalar enrichment
            if self.highL_pool is not None and self._backbone_lmax >= 3:
                norms = []
                for l in range(3, self._backbone_lmax + 1):
                    if l not in self._l_info:
                        continue
                    mul_l, off_l = self._l_info[l]
                    dim_l = mul_l * (2 * l + 1)
                    h_l = h_eq[:, off_l: off_l + dim_l].view(-1, mul_l, 2 * l + 1)
                    norm_l = h_l.norm(dim=-1)
                    norms.append(norm_l)
                if norms:
                    highL_inv = torch.cat(norms, dim=-1)
                    h_atom = h_atom + self.highL_pool(highL_inv)

            atom_t = None
            if self._n2e > 0 and self.proj_t is not None:
                d2 = self._n2e * 5
                h_t_raw = h_eq[:, self._off_2e: self._off_2e + d2].view(-1, self._n2e, 5)
                h_t_proj = self.proj_t(h_t_raw.transpose(1, 2)).transpose(1, 2)
                h_t_cart = torch.einsum('nhm, cm -> nhc', h_t_proj, self._cob_2e_to_cart)
                atom_t = h_t_cart.view(h_t_cart.size(0), self.hidden_nf, 3, 3)
                tr = atom_t.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / 3.0
                atom_t = atom_t - tr * torch.eye(3, device=atom_t.device, dtype=atom_t.dtype)

            # Old-style ENK (V1/V2-baseline only)
            if self.enk is not None:
                h_atom, atom_t, _enk_info = self.enk(h_atom, atom_t)

            # ---- Spectral mask / tensor SDM (V1/V2-baseline only) ----
            if self.tensor_sdm_enabled and data is not None:
                ctx_input = getattr(data, 'spectral_mask_polar_vec6_input', None)
                ctx_mask = getattr(data, 'spectral_mask_polar_vec6_mask', None)
                if ctx_input is not None and ctx_mask is not None:
                    h_atom, atom_t = self.tensor_sdm_attn(
                        h_atom,
                        atom_t,
                        ctx_input.to(h_atom.device),
                        ctx_mask.to(h_atom.device),
                        batch,
                    )
            elif self.spectral_mask_enabled and data is not None:
                ctx_input = getattr(data, 'spectral_mask_polar_vec6_input', None)
                ctx_mask = getattr(data, 'spectral_mask_polar_vec6_mask', None)
                if ctx_input is not None and ctx_mask is not None:
                    h_atom = self.spectral_mask_attn(
                        h_atom,
                        ctx_input.to(h_atom.device),
                        ctx_mask.to(h_atom.device),
                        batch,
                    )

            # ---- Electron prior post-decomposition (V2-baseline or V1) ----
            if self.electron_prior_enabled and atom_prior is not None:
                if self._is_v2 and hasattr(self, 'so3_ep_injector'):
                    # V2-baseline: deep injection already happened in blocks.
                    # Post-decomposition tensor path modulation (if enabled).
                    if atom_t is not None and hasattr(self, 'ep_tensor_proj'):
                        t_mod = self.ep_tensor_proj(atom_prior)
                        t_gate = torch.tanh(self.ep_tensor_gate)
                        atom_t = atom_t + t_gate * t_mod.unsqueeze(-1).unsqueeze(-1) * atom_t
                else:
                    # V1 fallback
                    h_atom = self.electron_prior_attn(
                        h_atom, atom_prior, edge_prior, ep_edge_index,
                    )

            per_atom_polar = self._polar_tensor(h_atom, atom_t, pos, z, batch)

        # ---- Sum over atoms → molecular polar ----
        if self.summation:
            num_graphs = int(batch.max().item()) + 1
            mol_polar = scatter_sum(per_atom_polar, batch, dim=0, dim_size=num_graphs)
        else:
            mol_polar = per_atom_polar

        if self.grad_type == 'polar':
            return self._grad_polarizability(mol_polar, pos)
        return mol_polar
