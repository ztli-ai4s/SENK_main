"""
Multi-task Equiformer for all spectral property predictions.

Supports V1 (graph_attention_transformer) and V2 (EquiformerV2) backbones.
When model_name starts with ``equiformer_v2_``, the V2 SO3-native path is
used: backbone.forward_features_so3() → SO3_Embedding → l=0 scalar extraction.

Optional modules (V2 only):
  - SO3ENKBridge: per-L Kalman filter on backbone features before the head.
    For hii/hij only l=0 is filtered (noise reduction before double autograd).
  - SO3ElectronPriorInjector: deep injection at TransBlock point-A.

Supports the same six tasks as DetaNet:
  - polar   : static polarizability α  [B, 3, 3]
  - depolar : ∂α/∂R  [N, 3, 6]  (Raman activity tensor)
  - dipole  : dipole moment μ  [B, 3]
  - dedipole: ∂μ/∂R  [N, 3, 3]  (Born effective charge / IR intensity)
  - hii     : diagonal Hessian block Hii  [N, 3, 3]
  - hij     : off-diagonal Hessian block Hij  [E, 3, 3]

Architecture per task:
  Equiformer backbone → irreps → task-specific head

  polar/depolar: 0e scalar(×2) + 2e tensor → cal_p_tensor → scatter / autograd
  dipole/dedipole: 0e scalar(×1) + 1e vector → cal_dipole → scatter / autograd
    hii: 0e scalar(×1) → per-atom energy → double autograd ∂²E/∂R² (two-clone leaves)
  hij: 0e scalar(×1) → per-atom energy → double autograd (posj, posi)
"""
import torch
from torch import nn
from torch.autograd import grad
from torch_scatter import scatter_sum
from e3nn import o3
from typing import Dict, Optional

from . import model_entrypoint


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


class CleanEquiformerMultiTask(nn.Module):
    """
    Multi-task Equiformer backbone with task-specific heads.

    Supports V1 and V2 backbones.  When a V2 model_name is used, the
    SO3-native path extracts l=0 scalars directly from SO3_Embedding
    instead of going through the flat irreps decomposition.

    task configs:
        polar:    summation=True,  grad_type=None
        depolar:  summation=False, grad_type='polar'
        dipole:   summation=True,  grad_type=None
        dedipole: summation=False, grad_type='dipole'
        hii:      summation=False, grad_type='Hi'
        hij:      summation=False, grad_type='Hij'
    """

    VALID_TASKS = {'polar', 'depolar', 'dipole', 'dedipole', 'hii', 'hij'}

    def __init__(
        self,
        task: str,
        hidden_nf: int = 128,
        model_name: str = 'graph_attention_transformer_nonlinear_l2',
        radius: float = 5.0,
        num_basis: int = 128,
        num_layers: int = 4,
        drop_path: float = 0.1,
        dropout: float = 0.1,
        # --- V2 backbone options ---
        max_num_neighbors: int = 64,
        num_gaussians: int = 64,
        use_gradient_checkpointing: bool = False,
        grid_resolution: int = 14,
        use_gate_act: bool = True,
        use_grid_mlp: bool = False,
        # --- ENK options (V2 only) ---
        enk_enabled: bool = False,
        enk_init_r_bias: float = -1.0,
        enk_init_q_bias: float = 1.0,
        # --- Electron Prior options (V2 only) ---
        electron_prior_config: Optional[Dict] = None,
        electron_prior_heads: int = 4,
    ):
        super().__init__()
        assert task in self.VALID_TASKS, f"Unknown task: {task}"
        self.task = task
        self.hidden_nf = hidden_nf
        self.radius = radius

        # Task properties
        self.summation = task in ('polar', 'dipole')
        self.grad_type = {
            'polar': None, 'depolar': 'polar',
            'dipole': None, 'dedipole': 'dipole',
            'hii': 'Hi', 'hij': 'Hij',
        }[task]

        # Build backbone
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
            # V2-specific (silently ignored by V1 factory)
            max_num_neighbors=max_num_neighbors,
            num_gaussians=num_gaussians,
            use_gradient_checkpointing=use_gradient_checkpointing,
            grid_resolution=grid_resolution,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
        )

        # Detect V2 backbone
        self._is_v2 = hasattr(self.backbone, 'forward_features_so3')
        self.electron_prior_enabled = electron_prior_config is not None

        # Parse irreps (works for both V1 and V2)
        _backbone_irreps = self.backbone.irreps_node_embedding
        self._l_info = {}
        _off = 0
        for mul, ir in _backbone_irreps:
            l = ir.l
            if l not in self._l_info:
                self._l_info[l] = (0, _off)
            old_mul, old_off = self._l_info[l]
            self._l_info[l] = (old_mul + mul, old_off if old_mul == 0 else old_off)
            _off += mul * ir.dim

        _n0e = self._l_info.get(0, (0, 0))[0]
        _n1e = self._l_info.get(1, (0, 0))[0]
        _n2e = self._l_info.get(2, (0, 0))[0]
        _off_0e = self._l_info.get(0, (0, 0))[1]
        _off_1e = self._l_info.get(1, (0, 0))[1]
        _off_2e = self._l_info.get(2, (0, 0))[1]
        self._n0e, self._n1e, self._n2e = _n0e, _n1e, _n2e
        self._off_0e, self._off_1e, self._off_2e = _off_0e, _off_1e, _off_2e
        self._backbone_lmax = max(self._l_info.keys()) if self._l_info else 2

        # V2 SO3-native path: use sphere_channels directly
        if self._is_v2:
            _C = self.backbone.sphere_channels  # = _n0e for V2
        else:
            _C = _n0e

        # Scalar projection (used by ALL tasks)
        self.proj_s = nn.Sequential(
            nn.LayerNorm(_C),
            nn.Linear(_C, hidden_nf),
        )

        # ---- V2 optional: SO3ENKBridge ----
        self.so3_enk_bridge = None
        if self._is_v2 and enk_enabled:
            from .equivariant_neural_kalman import SO3ENKBridge
            _ep_prior_dim = hidden_nf if electron_prior_config is not None else 0
            self.so3_enk_bridge = SO3ENKBridge(
                sphere_channels=self.backbone.sphere_channels,
                lmax=self._backbone_lmax,
                hidden_nf=hidden_nf,
                prior_dim=_ep_prior_dim,
                dropout=dropout,
                init_r_bias=enk_init_r_bias,
                init_q_bias=enk_init_q_bias,
            )

        # ---- V2 optional: Electron Prior ----
        if self.electron_prior_enabled and self._is_v2:
            from .clean_equiformer_polar_ext import GaussianRBF
            from detanet_nets.electron_prior import NBOPriorBranch
            from .v2_electron_prior import (
                SO3ElectronPriorInjector, upgrade_backbone_to_injectable,
            )

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
            self.so3_ep_injector = SO3ElectronPriorInjector(
                sphere_channels=self.backbone.sphere_channels,
                lmax=self._backbone_lmax,
                num_backbone_layers=self.backbone.num_layers,
                prior_atom_dim=ep_num_features,
                prior_edge_dim=ep_num_features,
                num_heads=electron_prior_heads,
            )
            upgrade_backbone_to_injectable(self.backbone, self.so3_ep_injector)

        # ---- Task-specific heads ----
        if task in ('polar', 'depolar'):
            # 2e tensor projection
            if _n2e > 0:
                self.proj_t = nn.Linear(_n2e, hidden_nf, bias=False)
                cob = _compute_2e_to_cart33()
                self.register_buffer('_cob_2e_to_cart', cob)
            else:
                self.proj_t = None
                self.register_buffer('_cob_2e_to_cart', torch.zeros(9, 5))

            # Polar head: scalar → (sa, sb); tensor → outt
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

            mask = torch.tril(torch.ones(3, 3)).flatten()
            self.register_buffer('_tril_mask', mask)

        elif task in ('dipole', 'dedipole'):
            # 1e vector projection
            if _n1e > 0:
                self.proj_v = nn.Linear(_n1e, hidden_nf, bias=False)
            else:
                self.proj_v = None

            # Dipole head: scalar → outs(×1); vector → outt(3D)
            self.sout = nn.Sequential(
                nn.LayerNorm(hidden_nf),
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_nf, 1),
            )
            if _n1e > 0:
                self.v_readout = nn.Linear(hidden_nf, 1, bias=False)
            else:
                self.v_readout = None

        elif task in ('hii', 'hij'):
            # Scalar energy head → per-atom scalar
            self.sout = nn.Sequential(
                nn.LayerNorm(hidden_nf),
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_nf, 1),
            )

        # Mass table for COM
        mass_table = torch.ones(100, dtype=torch.float32)
        mass_table[1] = 1.0079
        mass_table[6] = 12.0110
        mass_table[7] = 14.0067
        mass_table[8] = 15.9990
        mass_table[9] = 18.9980
        self.register_buffer("_mass_table", mass_table)

    # ---- Helpers ----
    def _center_of_mass(self, pos, z, batch):
        num_graphs = int(batch.max().item()) + 1
        z_clamped = z.clamp(min=0, max=self._mass_table.numel() - 1)
        masses = self._mass_table[z_clamped].to(dtype=pos.dtype, device=pos.device).unsqueeze(-1)
        mass_sum = scatter_sum(masses, batch, dim=0, dim_size=num_graphs).clamp(min=1e-8)
        pos_mass_sum = scatter_sum(pos * masses, batch, dim=0, dim_size=num_graphs)
        return pos_mass_sum / mass_sum

    def _centroid_coordinate(self, pos, z, batch):
        """Center-of-mass-relative coordinates: ra = pos - COM[batch]"""
        com = self._center_of_mass(pos, z, batch)
        return pos - com[batch]

    # ---- Decompose backbone output ----
    def _decompose_irreps(self, h_eq):
        """Extract scalar/vector/tensor features from equivariant backbone output."""
        h_s_raw = h_eq[:, self._off_0e: self._off_0e + self._n0e]
        h_atom = self.proj_s(h_s_raw)  # [N, hidden_nf]
        return h_atom, h_eq

    # ---- Task heads ----
    def _polar_tensor(self, h_atom, h_eq, pos, z, batch):
        """Per-atom polarizability: α_i = sb·(I/3) + outt + sa·Q(r)"""
        eye3 = torch.eye(3, device=pos.device, dtype=pos.dtype)
        s = self.sout(h_atom)
        sa, sb = s[:, :1], s[:, 1:]

        # l=2 tensor path
        if self._n2e > 0 and self.proj_t is not None and self.t_readout is not None:
            d2 = self._n2e * 5
            h_t_raw = h_eq[:, self._off_2e: self._off_2e + d2].view(-1, self._n2e, 5)
            h_t_proj = self.proj_t(h_t_raw.transpose(1, 2)).transpose(1, 2)
            h_t_cart = torch.einsum('nhm, cm -> nhc', h_t_proj, self._cob_2e_to_cart)
            atom_t = h_t_cart.view(h_t_cart.size(0), self.hidden_nf, 3, 3)
            # Enforce tracelessness
            tr = atom_t.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / 3.0
            atom_t = atom_t - tr * eye3

            t_comp = atom_t.permute(0, 2, 3, 1)
            outt = self.t_readout(t_comp).squeeze(-1)
            outt = 0.5 * (outt + outt.transpose(-1, -2))
            tr_out = (outt[:, 0, 0] + outt[:, 1, 1] + outt[:, 2, 2]) / 3.0
            outt = outt - tr_out[:, None, None] * eye3
        else:
            outt = pos.new_zeros(pos.shape[0], 3, 3)

        # Geometric tensor Q(r)
        ra = self._centroid_coordinate(pos, z, batch)
        Q = ra.unsqueeze(-1) * ra.unsqueeze(-2)
        r_sq = (ra * ra).sum(dim=-1, keepdim=True).unsqueeze(-1)
        Q = Q - (r_sq / 3.0) * eye3

        return sb.unsqueeze(-1) * (eye3 / 3.0) + outt + sa.unsqueeze(-1) * Q

    def _dipole_vector(self, h_atom, h_eq, pos, z, batch):
        """Per-atom dipole: μ_i = outs · ra + outt_vec"""
        outs = self.sout(h_atom)  # [N, 1]
        ra = self._centroid_coordinate(pos, z, batch)  # [N, 3]

        # l=1 vector path
        if self._n1e > 0 and self.proj_v is not None and self.v_readout is not None:
            d1 = self._n1e * 3
            h_v_raw = h_eq[:, self._off_1e: self._off_1e + d1].view(-1, self._n1e, 3)
            h_v_proj = self.proj_v(h_v_raw.transpose(1, 2)).transpose(1, 2)  # [N, H, 3]
            outt_vec = self.v_readout(h_v_proj.permute(0, 2, 1)).squeeze(-1)  # [N, 3]
        else:
            outt_vec = pos.new_zeros(pos.shape[0], 3)

        return outs * ra + outt_vec  # [N, 3]

    def _v2_dipole_vector(self, h_atom, v1_features, pos, z, batch):
        """Per-atom dipole using V2 SO3 l=1 features.

        v1_features: [N, 3, C] — l=1 spherical harmonic coefficients from SO3_Embedding.
        """
        outs = self.sout(h_atom)  # [N, 1]
        ra = self._centroid_coordinate(pos, z, batch)

        if v1_features is not None and self.proj_v is not None and self.v_readout is not None:
            # v1_features: [N, 3, C] where C = sphere_channels
            h_v_proj = self.proj_v(v1_features)  # [N, 3, hidden_nf]
            outt_vec = self.v_readout(h_v_proj).squeeze(-1)  # [N, 3]
        else:
            outt_vec = pos.new_zeros(pos.shape[0], 3)

        return outs * ra + outt_vec

    def _scalar_energy(self, h_atom):
        """Per-atom scalar energy for Hessian computation."""
        return self.sout(h_atom)  # [N, 1]

    # ---- Autograd methods (identical to DetaNet) ----
    def _grad_polarizability(self, polars, pos):
        polars_flat = polars.flatten(start_dim=1)[:, self._tril_mask == 1]
        depolar = torch.zeros(pos.shape[0], 3, 6, device=pos.device, dtype=pos.dtype)
        for i in range(6):
            depolar[:, :, i] = -torch.autograd.grad(
                polars_flat[:, i].sum(), pos,
                create_graph=self.training, retain_graph=True,
            )[0]
        return depolar

    def _grad_dipole(self, dipole, pos):
        """∂μ/∂R: [N_atoms, 3, 3]"""
        dedipole = torch.zeros(pos.shape[0], 3, 3, device=pos.device, dtype=pos.dtype)
        for i in range(3):
            dedipole[:, :, i] = -torch.autograd.grad(
                dipole[:, i].sum(), pos,
                create_graph=self.training, retain_graph=True,
            )[0]
        return dedipole

    def _grad_hess_ii(self, energy, posa, posb):
        """Diagonal Hessian ∂²E/∂R²: [N, 3, 3]

                V2 two-clone cross-derivative:
                    edge_vec = posa[src] - posb[dst]
                    posa supplies the source-side coordinates and posb supplies the
                    target-side coordinates, exactly matching the standard V2 forward.
                    At posa == posb == pos, the mixed derivative reproduces the
                    diagonal Hessian block without changing backbone semantics.

        Both posa and posb MUST be in the computation graph through edge_vec;
        this is guaranteed by forward_features_so3_twoclone.
        """
        f = -torch.autograd.grad(energy.sum(), posa, create_graph=True)[0]
        Hii = torch.zeros(f.shape[0], 3, 3, device=f.device, dtype=f.dtype)
        for i in range(3):
            Hii[:, i] = -torch.autograd.grad(
                f[:, i].sum(), posb,
                create_graph=self.training, retain_graph=True,
            )[0]
        return Hii

    def _grad_hess_ij(self, energy, posj, posi):
        """Off-diagonal Hessian ∂²E/∂Rⱼ∂Rᵢ: [E, 3, 3]"""
        fj = -torch.autograd.grad(energy.sum(), posj, create_graph=True)[0]
        Hji = torch.zeros(fj.shape[0], 3, 3, device=fj.device, dtype=fj.dtype)
        for i in range(3):
            Hji[:, i] = -torch.autograd.grad(
                fj[:, i].sum(), posi,
                create_graph=self.training, retain_graph=True,
            )[0]
        return Hji

    # ---- V2 two-clone backbone helper for Hessian tasks ----
    def _v2_backbone_hessian(self, posa, posb, z, batch, data=None,
                              edge_index_ext=None):
        """Run V2 backbone with two-clone edge vectors.

        edge_vec = posa[edge_index[0]] - posb[edge_index[1]]
        This preserves the native V2 edge convention ``pos[src] - pos[dst]``
        while splitting source and target coordinates onto two autograd leaves.
        The resulting mixed derivative is used for diagonal Hessian blocks.

        Args:
            posa:           [N, 3], requires_grad=True  (src role in edge_vec)
            posb:           [N, 3], requires_grad=True  (dst role in edge_vec)
            edge_index_ext: [2, E] — if given, use these edges; None = auto
                            from radius_graph.
        Returns:
            h_atom: [N, hidden_nf]
        """
        pos_ref = posa.detach()  # EP priors use real coordinates (no grad needed)

        atom_prior = None
        if self.electron_prior_enabled and hasattr(self, 'electron_prior'):
            atom_prior, edge_prior, ep_edge_index = self._compute_electron_priors(
                pos_ref, z, batch, data=data,
            )
            self.so3_ep_injector.prepare(atom_prior, edge_prior, ep_edge_index)

        so3_obj = self.backbone.forward_features_so3_twoclone(
            posa=posa, posb=posb, batch=batch, node_atom=z,
            edge_index_ext=edge_index_ext,
        )

        if self.electron_prior_enabled and hasattr(self, 'so3_ep_injector'):
            self.so3_ep_injector.clear_cache()

        so3_emb = so3_obj.embedding  # [N, (lmax+1)², C]

        if self.so3_enk_bridge is not None:
            so3_emb = self.so3_enk_bridge(so3_emb, atom_prior=atom_prior)

        x0 = so3_emb[:, 0, :]
        h_atom = self.proj_s(x0)
        return h_atom

    # ---- V2 off-diagonal Hessian backbone helper ----
    def _v2_backbone_hij(self, pos, z, batch, edge_index, data=None):
        """Run V2 backbone for Hij with a single position leaf (DetaNet style).

        posi = pos[edge_index[0]]  [E, 3]  (src)
        posj = pos[edge_index[1]]  [E, 3]  (dst)
        edge_vec = posj - posi

        Returns:
            h_atom: [N, hidden_nf]
            posj:   [E, 3]
            posi:   [E, 3]
        """
        pos_ref = pos.detach()

        atom_prior = None
        if self.electron_prior_enabled and hasattr(self, 'electron_prior'):
            atom_prior, edge_prior, ep_edge_index = self._compute_electron_priors(
                pos_ref, z, batch, data=data,
            )
            self.so3_ep_injector.prepare(atom_prior, edge_prior, ep_edge_index)

        so3_obj, posj, posi = self.backbone.forward_features_so3_hij(
            pos, edge_index, batch, z,
        )

        if self.electron_prior_enabled and hasattr(self, 'so3_ep_injector'):
            self.so3_ep_injector.clear_cache()

        so3_emb = so3_obj.embedding

        if self.so3_enk_bridge is not None:
            so3_emb = self.so3_enk_bridge(so3_emb, atom_prior=atom_prior)

        x0 = so3_emb[:, 0, :]
        h_atom = self.proj_s(x0)
        return h_atom, posj, posi

    # ---- V2 Electron Prior helpers ----
    def _compute_electron_priors(self, pos, z, batch, data=None):
        """Pre-compute EP atom/edge priors for V2 injection."""
        from .graph_attention_transformer import safe_radius_graph
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
        return atom_prior, edge_prior, ep_edge_index

    # ---- V2 backbone forward: SO3 path ----
    def _v2_backbone_forward(self, pos_backbone, z, batch, data=None):
        """Run V2 backbone with optional EP injection + ENK, return h_atom."""
        atom_prior = None
        if self.electron_prior_enabled:
            atom_prior, edge_prior, ep_edge_index = self._compute_electron_priors(
                pos_backbone, z, batch, data=data,
            )
            self.so3_ep_injector.prepare(atom_prior, edge_prior, ep_edge_index)

        so3_obj = self.backbone.forward_features_so3(
            pos=pos_backbone, batch=batch, node_atom=z,
        )

        if self.electron_prior_enabled:
            self.so3_ep_injector.clear_cache()

        so3_emb = so3_obj.embedding  # [N, (lmax+1)², C]

        if self.so3_enk_bridge is not None:
            so3_emb = self.so3_enk_bridge(so3_emb, atom_prior=atom_prior)

        # Extract l=0 scalar: so3_emb[:, 0, :] → [N, C]
        x0 = so3_emb[:, 0, :]
        h_atom = self.proj_s(x0)

        # For dipole/dedipole we also need l=1 vector features
        if self.task in ('dipole', 'dedipole') and self._backbone_lmax >= 1:
            # l=1 coefficients at indices 1:4, shape [N, 3, C]
            x1 = so3_emb[:, 1:4, :]  # [N, 3, C]
            return h_atom, x1, so3_emb

        return h_atom, None, so3_emb

    # ---- Forward ----
    def forward(self, *, pos, z, batch, edge_index=None, data=None, **kwargs):
        """
        Returns depend on task:
          polar:    [B, 3, 3]
          depolar:  [N, 3, 6]
          dipole:   [B, 3]
          dedipole: [N, 3, 3]
          hii:      [N, 3, 3]
          hij:      [E, 3, 3]
        """
        # Prepare pos for autograd
        if self.task == 'hii':
            # Two-clone trick: posa/posb are independent leaves so cross-derivative ≠ 0.
            # single-pos would give ∂²E/∂pos² = 0 by translational invariance.
            posa = pos.detach().clone().requires_grad_(True)
            posb = pos.detach().clone().requires_grad_(True)
            pos_backbone = posb  # nominal ref; V1 path raises NotImplementedError anyway
        elif self.grad_type is not None:
            pos = pos.detach().requires_grad_(True)
            pos_backbone = pos
        else:
            pos_backbone = pos

        # ---- V2 SO3-native path ----
        # polar/depolar should use CleanEquiformerPolarExt in V2 mode, not this class.
        # If polar/depolar lands here with V2 backbone, fall through to V1 flat path.
        if self._is_v2 and self.task not in ('polar', 'depolar'):
            # hii/hij both use dedicated Hessian paths and skip _v2_backbone_forward.
            if self.task == 'hii':
                h_atom = self._v2_backbone_hessian(
                    posa, posb, z, batch, data=data,
                    edge_index_ext=edge_index,
                )
                energy = self._scalar_energy(h_atom)
                return self._grad_hess_ii(energy, posa, posb)

            elif self.task == 'hij':
                if edge_index is None:
                    raise ValueError("hij task requires edge_index")
                # DetaNet-style single-pos edge slices for off-diagonal Hessian blocks.
                h_atom, posj, posi = self._v2_backbone_hij(
                    pos_backbone, z, batch, edge_index, data=data,
                )
                energy = self._scalar_energy(h_atom)
                return self._grad_hess_ij(energy, posj, posi)

            # dipole/dedipole: standard single-pos V2 forward
            h_atom, v1_features, so3_emb = self._v2_backbone_forward(
                pos_backbone, z, batch, data=data,
            )

            if self.task in ('dipole', 'dedipole'):
                per_atom = self._v2_dipole_vector(h_atom, v1_features, pos_backbone, z, batch)
                if self.summation:
                    num_g = int(batch.max().item()) + 1
                    out = scatter_sum(per_atom, batch, dim=0, dim_size=num_g)
                else:
                    out = per_atom
                if self.grad_type == 'dipole':
                    return self._grad_dipole(out.reshape(-1, 3), pos_backbone)
                return out

        # ---- V1 flat decomposition path ----
        h_eq = self.backbone.forward_features_equivariant(
            pos=pos_backbone, batch=batch, node_atom=z,
        )
        h_atom, h_eq_full = self._decompose_irreps(h_eq)

        if self.task in ('polar', 'depolar'):
            per_atom = self._polar_tensor(h_atom, h_eq_full, pos_backbone, z, batch)
            if self.summation:
                num_g = int(batch.max().item()) + 1
                out = scatter_sum(per_atom, batch, dim=0, dim_size=num_g)
            else:
                out = per_atom
            if self.grad_type == 'polar':
                return self._grad_polarizability(out, pos_backbone)
            return out

        elif self.task in ('dipole', 'dedipole'):
            per_atom = self._dipole_vector(h_atom, h_eq_full, pos_backbone, z, batch)
            if self.summation:
                num_g = int(batch.max().item()) + 1
                out = scatter_sum(per_atom, batch, dim=0, dim_size=num_g)
            else:
                out = per_atom
            if self.grad_type == 'dipole':
                return self._grad_dipole(out.reshape(-1, 3), pos_backbone)
            return out

        elif self.task == 'hii':
            # V1 backbone does not support the two-clone trick without architectural changes.
            raise NotImplementedError(
                "hii task is only supported with V2 backbone (--mode equiformer_v2*). "
                "V1 equiformer computes edge_vec internally from a single pos tensor, "
                "making the two-clone trick infeasible without backbone modification."
            )

        elif self.task == 'hij':
            raise NotImplementedError(
                "hij task is only supported with V2 backbone (--mode equiformer_v2*). "
                "Same reason as hii: V1 cannot support the required two-clone approach."
            )
