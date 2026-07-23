"""
Standalone EquiformerV2 backbone for molecular polarizability prediction.

Uses EquiformerV2's SO(2)-convolution + S² activation architecture without
the OCP framework dependency.

**API surface (maximum-performance V2)**:
  - .irreps_node_embedding    (e3nn Irreps, includes ALL l=0..lmax, C each)
  - .forward_features_equivariant(pos, batch, node_atom)  → [N, flat_dim]
      flat_dim = C × Σ(2l+1) for l=0..lmax  (e.g. lmax=4 → 25C)
  - .forward_features_so3(pos, batch, node_atom)  → SO3_Embedding
      raw SO3_Embedding [N, (lmax+1)², C] for deep injection modules
  - TransBlock loop supports optional ``block_hook`` callback:
      hook(block_idx, x: SO3_Embedding, ...) → SO3_Embedding
      allows electron prior injection INSIDE the V2 block loop

Key V2 advantages for polarizability
  ● SO(2) convolution: supports L_max ≤ 10 at O(L²C²), vs e3nn TP O(L³C²)
  ● All L share C channels → C×0e + C×1e + ... + C×(lmax)e
  ● Separable S² activation, attention re-normalisation, RMS-norm on SH
  ● Full high-L output (l=3,4,5,6) enriches tensor readout for polarizability

Reference
  EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree
  Representations (Liao et al., ICLR 2024)
"""
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from .registry import register_model

# ---------------------------------------------------------------------------
#  EquiformerV2 core modules — vendored in nets/eqv2_core/ (no external deps)
# ---------------------------------------------------------------------------
from .eqv2_core.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from .eqv2_core.transformer_block import (
    TransBlockV2,
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
)
from .eqv2_core.input_block import EdgeDegreeEmbedding
from .eqv2_core.edge_rot_mat import init_edge_rot_mat
from .eqv2_core.module_list import ModuleListInfo
from .eqv2_core.layer_norm import get_normalization_layer
from .eqv2_core.radial_function import RadialFunction


# ---------------------------------------------------------------------------
#  GaussianSmearing  (replaces OCP's ocpmodels.models.scn.smearing)
# ---------------------------------------------------------------------------
class GaussianSmearing(nn.Module):
    """Fixed-width Gaussian radial basis (same API as OCP GaussianSmearing)."""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=128,
                 basis_width_scalar=2.0):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2 / basis_width_scalar
        self.register_buffer('offset', offset)
        self.num_output = num_gaussians

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


# ---------------------------------------------------------------------------
#  EquiformerV2Backbone — standalone molecular backbone
# ---------------------------------------------------------------------------
class EquiformerV2Backbone(nn.Module):
    """
    EquiformerV2 backbone adapted for small-molecule property prediction.

    No OCP BaseModel or generate_graph — builds radius graph internally.
    Exposes ``forward_features_equivariant(pos, batch, node_atom)`` returning
    a flat tensor [N, 9C] whose layout matches ``Cx0e + Cx1e + Cx2e``.

    Args
        max_radius         Cutoff radius for neighbours (Å).
        max_num_elements   Size of atomic-number embedding table.
        num_layers         Number of TransBlockV2 layers.
        sphere_channels    Channels per spherical-harmonic degree (C).
        lmax_list          Maximum L for irrep basis  ([4] or [6]).
        mmax_list          Maximum m for SO(2) truncation ([2] or [4]).
        grid_resolution    Resolution of the S²-grid  (None → default 18).
        avg_num_nodes      Average atoms per graph (for energy rescaling).
        avg_degree         Average neighbours per atom (for edge-degree rescale).
        *Other args match Liao et al. (2024) Table 9.*
    """

    def __init__(
        self,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        num_layers: int = 8,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 64,
        num_heads: int = 8,
        attn_alpha_channels: int = 64,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 128,
        norm_type: str = 'layer_norm_sh',
        lmax_list=None,
        mmax_list=None,
        grid_resolution=18,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        num_distance_basis: int = 512,
        attn_activation: str = 'silu',
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = 'silu',
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = 'uniform',
        avg_num_nodes: float = 18.0,
        avg_degree: float = 12.0,
        max_num_neighbors: int = 64,
        num_gaussians: int = 128,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if lmax_list is None:
            lmax_list = [4]
        if mmax_list is None:
            mmax_list = [2]

        self.max_radius = max_radius
        self.max_num_elements = max_num_elements
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * sphere_channels
        self.weight_init = weight_init
        self.avg_num_nodes = avg_num_nodes
        self.max_num_neighbors = max_num_neighbors
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # e3nn-compatible irreps for downstream parsers — full lmax
        C = sphere_channels
        lmax = max(lmax_list)
        self.irreps_node_embedding = o3.Irreps(
            '+'.join(f'{C}x{l}e' for l in range(lmax + 1))
        )
        self._lmax = lmax
        self._flat_dim = C * sum(2 * l + 1 for l in range(lmax + 1))  # C*(lmax+1)²

        # ---- distance expansion ----
        self.distance_expansion = GaussianSmearing(0.0, max_radius, num_gaussians, 2.0)
        self.edge_channels_list = [self.distance_expansion.num_output] + \
                                  [edge_channels] * 2

        # ---- atom-edge embeddings ----
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if share_atom_edge_embedding and use_atom_edge_embedding:
            self.block_use_atom_edge_embedding = False
            self.source_embedding = nn.Embedding(max_num_elements,
                                                 self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(max_num_elements,
                                                 self.edge_channels_list[-1])
            self.edge_channels_list[0] += 2 * self.edge_channels_list[-1]
        else:
            self.block_use_atom_edge_embedding = use_atom_edge_embedding
            self.source_embedding = self.target_embedding = None

        # ---- atom embedding ----
        self.sphere_embedding = nn.Embedding(max_num_elements,
                                             self.sphere_channels_all)

        # ---- SO(3) infrastructure ----
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(lmax_list[i]))

        self.mappingReduced = CoefficientMappingModule(lmax_list, mmax_list)

        self.SO3_grid = ModuleListInfo(
            '({}, {})'.format(max(lmax_list), max(lmax_list))
        )
        for l in range(max(lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m, resolution=grid_resolution,
                             normalization='component')
                )
            self.SO3_grid.append(SO3_m_grid)

        # ---- edge-degree embedding ----
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels, lmax_list, mmax_list,
            self.SO3_rotation, self.mappingReduced,
            max_num_elements, self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=avg_degree,
        )

        # ---- transformer blocks ----
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = TransBlockV2(
                sphere_channels,
                attn_hidden_channels, num_heads,
                attn_alpha_channels, attn_value_channels,
                ffn_hidden_channels, sphere_channels,
                lmax_list, mmax_list,
                self.SO3_rotation, self.mappingReduced, self.SO3_grid,
                max_num_elements, self.edge_channels_list,
                self.block_use_atom_edge_embedding, use_m_share_rad,
                attn_activation, use_s2_act_attn, use_attn_renorm,
                ffn_activation, use_gate_act, use_grid_mlp, use_sep_s2_act,
                norm_type, alpha_drop, drop_path_rate, proj_drop,
            )
            self.blocks.append(block)

        # ---- output normalisation ----
        self.norm = get_normalization_layer(
            norm_type, lmax=max(lmax_list), num_channels=sphere_channels
        )

        # ---- weight initialisation ----
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    # ------------------------------------------------------------------
    #  Weight init  (follows original V2 code)
    # ------------------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            std = 1.0 / math.sqrt(m.in_features)
            if self.weight_init == 'normal':
                nn.init.normal_(m.weight, 0, std)
            else:
                nn.init.uniform_(m.weight, -std, std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            for child in m.modules():
                if isinstance(child, nn.Linear):
                    if child.bias is not None:
                        nn.init.constant_(child.bias, 0)
                    std = 1.0 / math.sqrt(child.in_features)
                    nn.init.uniform_(child.weight, -std, std)

    # ------------------------------------------------------------------
    #  Graph construction  (molecular, no PBC)
    # ------------------------------------------------------------------
    def _build_graph(self, pos, batch, max_radius):
        edge_index = radius_graph(pos, r=max_radius, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_dist = torch.sqrt(torch.sum(edge_vec ** 2, dim=1).clamp(min=1e-12))
        return edge_index, edge_vec, edge_dist

    # ------------------------------------------------------------------
    #  Core forward — returns SO3_Embedding (internal API)
    # ------------------------------------------------------------------
    def _forward_core(
        self,
        edge_vec: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_atom: torch.Tensor,
        use_checkpoint: bool = True,
        block_hook: Optional[Callable] = None,
        detach_wigner: bool = True,
    ) -> 'SO3_Embedding':
        """Shared transformer pipeline: (edge_vec, edge_index) → SO3_Embedding.

        All public forward variants build their own edge_vec/edge_index and
        then delegate to this method for the actual transformer computation.

        Args:
            detach_wigner: When False, the Wigner-D rotation matrices retain
                their autograd connection to *edge_vec* (and ultimately to
                *pos*). This is required for 2nd-order gradient tasks (Hij)
                so angular coupling gradients can flow through the SO(3)
                rotation path.
        """
        N = node_atom.shape[0]
        device = edge_vec.device
        dtype = edge_vec.dtype
        atomic_numbers = node_atom.long()

        edge_dist = torch.sqrt(torch.sum(edge_vec ** 2, dim=1).clamp(min=1e-12))

        # ---- edge rotation matrices → Wigner-D ----
        edge_rot_mat = init_edge_rot_mat(
            edge_vec,
            detach=detach_wigner,
            stable=(not detach_wigner),
        )
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(
                edge_rot_mat,
                detach=detach_wigner,
                stable=(not detach_wigner),
            )

        # ---- node embedding ----
        x = SO3_Embedding(N, self.lmax_list, self.sphere_channels,
                          device, dtype)
        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                )[:, offset: offset + self.sphere_channels]
            offset += self.sphere_channels
            offset_res += int((self.lmax_list[i] + 1) ** 2)

        # ---- edge distance encoding ----
        edge_distance = self.distance_expansion(edge_dist)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]
            target_element = atomic_numbers[edge_index[1]]
            source_emb = self.source_embedding(source_element)
            target_emb = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_emb, target_emb), dim=1)

        # ---- edge-degree embedding ----
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        # ---- transformer blocks with optional hook ----
        for blk_idx, blk in enumerate(self.blocks):
            if (use_checkpoint and self.use_gradient_checkpointing
                    and self.training and block_hook is None):
                def _blk_fn(emb, blk=blk):
                    x_tmp = SO3_Embedding(0, x.lmax_list.copy(), x.num_channels,
                                         device=emb.device, dtype=emb.dtype)
                    x_tmp.set_embedding(emb)
                    x_tmp.set_lmax_mmax(x.lmax_list.copy(), x.mmax_list.copy())
                    out = blk(x_tmp, atomic_numbers, edge_distance, edge_index,
                              batch=batch)
                    return out.embedding
                new_emb = gradient_checkpoint(_blk_fn, x.embedding, use_reentrant=False)
                x.embedding = new_emb
            else:
                x = blk(x, atomic_numbers, edge_distance, edge_index,
                         batch=batch)
                if block_hook is not None:
                    x = block_hook(blk_idx, x, atomic_numbers, edge_distance,
                                   edge_index, batch)

        # ---- final normalisation ----
        x.embedding = self.norm(x.embedding)
        return x

    # ------------------------------------------------------------------
    #  _forward_so3 — standard single-pos forward (delegates to _forward_core)
    # ------------------------------------------------------------------
    def _forward_so3(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        node_atom: torch.Tensor,
        block_hook: Optional[Callable] = None,
    ) -> 'SO3_Embedding':
        """
        Run full V2 forward and return :class:`SO3_Embedding`.

        Parameters
        ----------
        block_hook : callable, optional
            ``hook(block_idx: int, x: SO3_Embedding, atomic_numbers, edge_distance,
            edge_index, batch) → SO3_Embedding``
            Called after each TransBlockV2.  Enables mid-network injection
            (e.g., electron prior) without modifying TransBlockV2 source.
        """
        edge_index, edge_vec, _ = self._build_graph(
            pos, batch, self.max_radius,
        )
        return self._forward_core(
            edge_vec, edge_index, batch, node_atom,
            use_checkpoint=True, block_hook=block_hook,
        )

    # ------------------------------------------------------------------
    #  Forward — raw SO3_Embedding (for V2-native deep injection)
    # ------------------------------------------------------------------
    def forward_features_so3(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        node_atom: torch.Tensor,
        block_hook: Optional[Callable] = None,
    ) -> 'SO3_Embedding':
        """
        Return normalised :class:`SO3_Embedding` ``[N, (lmax+1)², C]``.

        Used by V2-native modules that need the full spherical representation
        (e.g., ``SO3ElectronPriorInjector``).
        """
        return self._forward_so3(pos, batch, node_atom, block_hook=block_hook)

    # ------------------------------------------------------------------
    #  Two-clone forward — used for Hessian diagonal/off-diagonal blocks
    # ------------------------------------------------------------------
    def forward_features_so3_twoclone(
        self,
        posa: torch.Tensor,
        posb: torch.Tensor,
        batch: torch.Tensor,
        node_atom: torch.Tensor,
        edge_index_ext: Optional[torch.Tensor] = None,
        block_hook: Optional[Callable] = None,
    ) -> 'SO3_Embedding':
        """Two-clone forward for Hii (diagonal Hessian blocks).

        ``edge_vec = posa[edge_index[0]] - posb[edge_index[1]]`` so *both*
        leaf tensors participate in the autograd graph, enabling the
        cross-derivative ``∂²E / ∂posa ∂posb = Hii``.

        Gradient checkpointing is intentionally disabled to preserve the
        full autograd path through edge_vec → posa/posb.
        """
        if edge_index_ext is not None:
            edge_index = edge_index_ext
        else:
            with torch.no_grad():
                pos_ref = ((posa.detach() + posb.detach()) * 0.5).contiguous()
            edge_index = radius_graph(pos_ref, r=self.max_radius, batch=batch,
                                      max_num_neighbors=self.max_num_neighbors)

        edge_vec = posa[edge_index[0]] - posb[edge_index[1]]
        return self._forward_core(
            edge_vec, edge_index, batch, node_atom,
            use_checkpoint=False, block_hook=block_hook,
        )

    # ------------------------------------------------------------------
    #  Hij forward — single pos, returns edge-level (posj, posi) in graph
    # ------------------------------------------------------------------
    def forward_features_so3_hij(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_atom: torch.Tensor,
        block_hook: Optional[Callable] = None,
    ):
        """Forward for Hij (off-diagonal Hessian blocks) — single-pos variant.

        Mirrors DetaNet's ``grad_hess_ij`` indexing pattern:
            posi = pos[edge_index[0]]  [E, 3]
            posj = pos[edge_index[1]]  [E, 3]
            edge_vec = posj - posi  (DetaNet convention)

        Returns ``(SO3_Embedding, posj, posi)`` where
            posj and posi are both in the autograd graph through the same
            position leaf tensor.

        Gradient checkpointing is disabled to preserve the full autograd path.
        """
        posi = pos[edge_index[0]]   # [E, 3], i/src atoms
        posj = pos[edge_index[1]]   # [E, 3], j/dst atoms
        edge_vec = posj - posi      # align with DetaNet Hij mother-function convention

        x = self._forward_core(
            edge_vec, edge_index, batch, node_atom,
            use_checkpoint=False, block_hook=block_hook,
            detach_wigner=False,
        )
        return x, posj, posi

    # ------------------------------------------------------------------
    #  Forward — equivariant per-atom features (flat tensor)
    # ------------------------------------------------------------------
    def forward_features_equivariant(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        node_atom: torch.Tensor,
        block_hook: Optional[Callable] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Return per-atom equivariant features as flat tensor.

        Output shape:  ``[N, (lmax+1)² × C]``

        For **lmax=4** (default):
            flat_dim = 25C;  layout ``Cx0e + Cx1e + Cx2e + Cx3e + Cx4e``
            [:, 0   : C   ]  →  l=0 scalars        (1  × C)
            [:, C   : 4C  ]  →  l=1 vectors         (3  × C)
            [:, 4C  : 9C  ]  →  l=2 tensors         (5  × C)
            [:, 9C  : 16C ]  →  l=3 octupoles        (7  × C)
            [:, 16C : 25C ]  →  l=4 hexadecapoles    (9  × C)

        Compatible with V1 parsers: when lmax=2, output is identical
        to the original ``[N, 9C]`` layout.
        """
        x = self._forward_so3(pos, batch, node_atom, block_hook=block_hook)

        N = pos.shape[0]
        C = self.sphere_channels
        lmax = self._lmax

        # Extract all L levels and concatenate
        parts = []
        offset = 0
        for l in range(lmax + 1):
            nm = 2 * l + 1  # number of m components for this l
            parts.append(x.embedding[:, offset:offset + nm, :].reshape(N, nm * C))
            offset += nm

        return torch.cat(parts, dim=-1)  # [N, (lmax+1)² × C]

    # ------------------------------------------------------------------
    #  Convenience
    # ------------------------------------------------------------------
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def lmax(self) -> int:
        """Maximum angular momentum degree for the backbone."""
        return self._lmax

    def extra_repr(self):
        return (
            f"sphere_channels={self.sphere_channels}, "
            f"lmax={self.lmax_list}, mmax={self.mmax_list}, "
            f"layers={self.num_layers}, "
            f"flat_dim={self._flat_dim}, "
            f"params={self.num_params:,}"
        )


#  Factory functions  (same signature as V1 for CleanEquiformerPolar compat)
def _v2_factory(
    irreps_in, radius, num_basis=128,
    atomref=None, task_mean=None, task_std=None, ssp=None,
    *,
    lmax_list, mmax_list, num_layers, sphere_channels,
    attn_hidden_channels, ffn_hidden_channels,
    alpha_drop, drop_path_rate,
    **kwargs,
):
    """Shared factory body — kwargs override any default."""
    return EquiformerV2Backbone(
        max_radius=radius,
        num_layers=num_layers,
        sphere_channels=sphere_channels,
        lmax_list=lmax_list,
        mmax_list=mmax_list,
        attn_hidden_channels=attn_hidden_channels,
        ffn_hidden_channels=ffn_hidden_channels,
        alpha_drop=alpha_drop,
        drop_path_rate=drop_path_rate if drop_path_rate is not None else float(kwargs.get('drop_path', 0.05)),
        num_heads=int(kwargs.get('num_heads', 8)),
        attn_alpha_channels=int(kwargs.get('attn_alpha_channels', 64)),
        attn_value_channels=int(kwargs.get('attn_value_channels', 16)),
        edge_channels=int(kwargs.get('edge_channels', 128)),
        norm_type=str(kwargs.get('norm_type', 'layer_norm_sh')),
        grid_resolution=kwargs.get('grid_resolution', 18),
        use_atom_edge_embedding=bool(kwargs.get('use_atom_edge_embedding', True)),
        use_gate_act=bool(kwargs.get('use_gate_act', False)),
        use_grid_mlp=bool(kwargs.get('use_grid_mlp', True)),
        use_sep_s2_act=bool(kwargs.get('use_sep_s2_act', True)),
        weight_init=str(kwargs.get('weight_init', 'uniform')),
        avg_num_nodes=float(kwargs.get('avg_num_nodes', 18.0)),
        avg_degree=float(kwargs.get('avg_degree', 12.0)),   # OCP default; used as EdgeDegreeEmbedding rescale_factor (runtime division)
        max_num_neighbors=int(kwargs.get('max_num_neighbors', 64)),
        num_gaussians=int(kwargs.get('num_gaussians', 128)),
        use_gradient_checkpointing=bool(kwargs.get('use_gradient_checkpointing', False)),
    )


@register_model
def equiformer_v2_l4_m2(irreps_in, radius, num_basis=128,
                         atomref=None, task_mean=None, task_std=None,
                         ssp=None, **kwargs):
    """EquiformerV2  lmax=4 mmax=2 · 8 layers · C=128.
    Good balance of expressiveness and speed for molecules ≤ 50 atoms.
    """
    return _v2_factory(
        irreps_in, radius, num_basis,
        atomref=atomref, task_mean=task_mean, task_std=task_std, ssp=ssp,
        lmax_list=[4], mmax_list=[2],
        num_layers=int(kwargs.pop('num_layers', 8)),
        sphere_channels=128,
        attn_hidden_channels=64,
        ffn_hidden_channels=128,
        alpha_drop=0.1,
        drop_path_rate=float(kwargs.pop('drop_path', 0.05)),
        **kwargs,
    )


@register_model
def equiformer_v2_l6_m2(irreps_in, radius, num_basis=128,
                         atomref=None, task_mean=None, task_std=None,
                         ssp=None, **kwargs):
    """EquiformerV2  lmax=6 mmax=2 · 8 layers · C=128.
    Highest angular resolution — more costly but stronger l=2 features.
    """
    return _v2_factory(
        irreps_in, radius, num_basis,
        atomref=atomref, task_mean=task_mean, task_std=task_std, ssp=ssp,
        lmax_list=[6], mmax_list=[2],
        num_layers=int(kwargs.pop('num_layers', 8)),
        sphere_channels=128,
        attn_hidden_channels=64,
        ffn_hidden_channels=128,
        alpha_drop=0.1,
        drop_path_rate=float(kwargs.pop('drop_path', 0.05)),
        **kwargs,
    )


@register_model
def equiformer_v2_l3_m2(irreps_in, radius, num_basis=128,
                        atomref=None, task_mean=None, task_std=None,
                        ssp=None, **kwargs):
    """EquiformerV2  lmax=3 mmax=2 · default 4 layers · C=128.
    ~30% faster than l4_m2 (SO2-conv coefficients 16C vs 25C per block).
    Suitable for QM9 static/derivative tasks where l=4 marginal gain is small.
    """
    return _v2_factory(
        irreps_in, radius, num_basis,
        atomref=atomref, task_mean=task_mean, task_std=task_std, ssp=ssp,
        lmax_list=[3], mmax_list=[2],
        num_layers=int(kwargs.pop('num_layers', 4)),
        sphere_channels=128,
        attn_hidden_channels=64,
        ffn_hidden_channels=128,
        alpha_drop=0.1,
        drop_path_rate=float(kwargs.pop('drop_path', 0.05)),
        **kwargs,
    )


@register_model
def equiformer_v2_l4_m2_small(irreps_in, radius, num_basis=128,
                               atomref=None, task_mean=None, task_std=None,
                               ssp=None, **kwargs):
    """EquiformerV2  lmax=4 mmax=2 · 4 layers · C=64.
    Lightweight config for quick ablation and toy tests.
    """
    return _v2_factory(
        irreps_in, radius, num_basis,
        atomref=atomref, task_mean=task_mean, task_std=task_std, ssp=ssp,
        lmax_list=[4], mmax_list=[2],
        num_layers=int(kwargs.pop('num_layers', 4)),
        sphere_channels=64,
        attn_hidden_channels=32,
        ffn_hidden_channels=64,
        alpha_drop=0.1,
        drop_path_rate=float(kwargs.pop('drop_path', 0.05)),
        **kwargs,
    )
