"""
Clean Equiformer for polarizability / depolarizability prediction.

No DualPath wrapper, no electron channel, no spectral mask, no clamp.
Pure: Equiformer backbone → irreps decomposition → PolarizabilityTensorHead.
Derivative ∂α/∂R computed via autograd (identical to DetaNet paradigm).

Optional: EquivariantNeuralKalman (ENK) module for Kalman-filter-inspired
feature refinement between backbone and polar head (--enk_enabled).
"""
import torch
from torch import nn
from torch_scatter import scatter_sum
from e3nn import o3
from typing import Optional, Dict, Tuple

from . import model_entrypoint


def _compute_2e_to_cart33() -> torch.Tensor:
    """Change-of-basis matrix [9, 5]: l=2 real SH → traceless symmetric 3x3 Cartesian."""
    test_vecs = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
        [1., 1., 1.], [-1., 1., 0.], [1., -1., 1.],
    ], dtype=torch.float64)
    test_vecs = test_vecs / test_vecs.norm(dim=-1, keepdim=True)
    Y2 = o3.spherical_harmonics(2, test_vecs, normalize=False, normalization='component').double()
    T_list = []
    for v in test_vecs:
        rr = v.unsqueeze(-1) * v.unsqueeze(-2)
        rr_tl = rr - rr.trace() / 3.0 * torch.eye(3, dtype=torch.float64)
        T_list.append(rr_tl.flatten())
    T = torch.stack(T_list)
    U_T = torch.linalg.lstsq(Y2, T).solution
    return U_T.T.float()


class CleanEquiformerPolar(nn.Module):
    """
    Clean Equiformer backbone + DetaNet-style polarizability head.

    Architecture:
        pos → Equiformer TransBlock × N → full irreps (0e + 1e + 2e)
            → proj_s → scalar features
            → proj_t → l=2 → CG → traceless 3×3 tensor
            → PolarizabilityTensorHead: α = sb·(I/3) + outt + sa·Q(r)
            → scatter_sum → molecular α [B, 3, 3]
            → (if depolar) autograd ∂α/∂R → [N, 3, 6]

    No clamp, no nan_to_num, no electron channel, no spectral mask.
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
        enk_enabled: bool = False,
        enk_init_r_bias: float = -1.0,
        enk_init_q_bias: float = 1.0,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.summation = summation
        self.grad_type = grad_type
        self.enk_enabled = enk_enabled

        # Optional: Equivariant Neural Kalman module
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
        )

        # Parse irreps from backbone — supports all L levels
        _irreps = self.backbone.irreps_node_embedding
        # Collect per-l multiplicities and offsets
        self._l_info = {}   # l → (mul, offset_in_flat)
        _off = 0
        for mul, ir in _irreps:
            l = ir.l
            if l not in self._l_info:
                self._l_info[l] = (0, _off)
            old_mul, old_off = self._l_info[l]
            self._l_info[l] = (old_mul + mul, old_off if old_mul == 0 else old_off)
            _off += mul * ir.dim

        # Backward-compat aliases
        self._n0e = self._l_info.get(0, (0, 0))[0]
        self._n1e = self._l_info.get(1, (0, 0))[0]
        self._n2e = self._l_info.get(2, (0, 0))[0]
        self._off_0e = self._l_info.get(0, (0, 0))[1]
        self._off_1e = self._l_info.get(1, (0, 0))[1]
        self._off_2e = self._l_info.get(2, (0, 0))[1]

        # Detect backbone lmax
        self._backbone_lmax = max(self._l_info.keys()) if self._l_info else 2

        # High-L pooling: compress l≥3 features into scalar enrichment
        _highL_total = 0
        for l in range(3, self._backbone_lmax + 1):
            if l in self._l_info:
                mul_l = self._l_info[l][0]
                _highL_total += mul_l * (2 * l + 1)
        self._highL_total = _highL_total

        if _highL_total > 0:
            # Pool high-L features: invariant norm per l channel → scalar
            # For each l≥3 block [N, mul, 2l+1]: compute ||·|| → [N, mul]
            # Sum all muls → [N, total_highL_muls]
            _highL_muls = sum(self._l_info[l][0] for l in range(3, self._backbone_lmax + 1) if l in self._l_info)
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

        # Projection layers (0e → scalar, 2e → tensor)
        _n0e = self._n0e
        _n2e = self._n2e
        # If high-L features exist, proj_s takes concatenated [l0; highL_pool]
        # But highL_pool output has its own layer — we add them after projection
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

        # Polarizability head (same math as DetaNet's cal_p_tensor)
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

        # Mass table for COM
        mass_table = torch.ones(100, dtype=torch.float32)
        mass_table[1] = 1.0079
        mass_table[6] = 12.0110
        mass_table[7] = 14.0067
        mass_table[8] = 15.9990
        mass_table[9] = 18.9980
        self.register_buffer("_mass_table", mass_table)

        # Lower-triangular mask for extracting 6 unique symmetric tensor components
        mask = torch.tril(torch.ones(3, 3)).flatten()
        self.register_buffer('_tril_mask', mask)

    def _center_of_mass(self, pos, z, batch):
        num_graphs = int(batch.max().item()) + 1
        z_clamped = z.clamp(min=0, max=self._mass_table.numel() - 1)
        masses = self._mass_table[z_clamped].to(dtype=pos.dtype, device=pos.device).unsqueeze(-1)
        mass_sum = scatter_sum(masses, batch, dim=0, dim_size=num_graphs).clamp(min=1e-8)
        pos_mass_sum = scatter_sum(pos * masses, batch, dim=0, dim_size=num_graphs)
        return pos_mass_sum / mass_sum

    def _polar_tensor(self, h_atom, atom_t, pos, z, batch):
        """Per-atom polarizability tensor: α_i = sb·(I/3) + outt + sa·Q(r)"""
        eye3 = torch.eye(3, device=pos.device, dtype=pos.dtype)

        s = self.sout(h_atom)
        sa = s[:, :1]
        sb = s[:, 1:]

        # l=2 equivariant direct output
        if atom_t is not None and self.t_readout is not None:
            t_comp = atom_t.permute(0, 2, 3, 1)  # [N, 3, 3, H]
            outt = self.t_readout(t_comp).squeeze(-1)  # [N, 3, 3]
            outt = 0.5 * (outt + outt.transpose(-1, -2))
            tr_out = (outt[:, 0, 0] + outt[:, 1, 1] + outt[:, 2, 2]) / 3.0
            outt = outt - tr_out[:, None, None] * eye3
        else:
            outt = pos.new_zeros(pos.shape[0], 3, 3)

        # Geometric tensor Q(r) = r⊗r − (r²/3)·I
        com = self._center_of_mass(pos, z, batch)
        ra = pos - com[batch]
        Q = ra.unsqueeze(-1) * ra.unsqueeze(-2)
        r_sq = (ra * ra).sum(dim=-1, keepdim=True).unsqueeze(-1)
        Q = Q - (r_sq / 3.0) * eye3

        per_atom = sb.unsqueeze(-1) * (eye3 / 3.0) + outt + sa.unsqueeze(-1) * Q
        return per_atom  # [N, 3, 3]

    def _grad_polarizability(self, polars, pos):
        """Compute ∂α/∂R via autograd (same as DetaNet.grad_polarzability)."""
        polars_flat = polars.flatten(start_dim=1)[:, self._tril_mask == 1]  # [N, 6]
        depolar = torch.zeros(pos.shape[0], 3, 6, device=pos.device, dtype=pos.dtype)
        for i in range(6):
            depolar[:, :, i] = -torch.autograd.grad(
                polars_flat[:, i].sum(), pos, create_graph=self.training, retain_graph=True
            )[0]
        return depolar

    def forward(self, *, pos, z, batch, **kwargs):
        """
        Args:
            pos: [N, 3] atomic positions
            z: [N] atomic numbers
            batch: [N] graph indices
        Returns:
            If grad_type is None (polar model):  [B, 3, 3] molecular polarizability
            If grad_type == 'polar' (depolar model): [N, 3, 6] ∂α/∂R
        """
        if self.grad_type == 'polar':
            pos = pos.detach().requires_grad_(True)

        # Backbone: full equivariant features
        h_eq = self.backbone.forward_features_equivariant(
            pos=pos, batch=batch, node_atom=z
        )

        # Decompose irreps
        d0, d2 = self._n0e, self._n2e * 5
        h_s_raw = h_eq[:, self._off_0e: self._off_0e + self._n0e]
        h_atom = self.proj_s(h_s_raw)  # [N, hidden_nf]

        # High-L pooling: extract l≥3, compute invariant norm, pool into scalar
        if self.highL_pool is not None and self._backbone_lmax >= 3:
            norms = []
            for l in range(3, self._backbone_lmax + 1):
                if l not in self._l_info:
                    continue
                mul_l, off_l = self._l_info[l]
                dim_l = mul_l * (2 * l + 1)
                h_l = h_eq[:, off_l: off_l + dim_l].view(-1, mul_l, 2 * l + 1)
                norm_l = h_l.norm(dim=-1)  # [N, mul_l] — invariant
                norms.append(norm_l)
            if norms:
                highL_inv = torch.cat(norms, dim=-1)  # [N, total_highL_muls]
                h_atom = h_atom + self.highL_pool(highL_inv)  # residual enrichment

        atom_t = None
        if self._n2e > 0 and self.proj_t is not None:
            h_t_raw = h_eq[:, self._off_2e: self._off_2e + d2].view(-1, self._n2e, 5)
            h_t_proj = self.proj_t(h_t_raw.transpose(1, 2)).transpose(1, 2)
            h_t_cart = torch.einsum('nhm, cm -> nhc', h_t_proj, self._cob_2e_to_cart)
            atom_t = h_t_cart.view(h_t_cart.size(0), self.hidden_nf, 3, 3)
            # Enforce tracelessness
            tr = atom_t.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / 3.0
            atom_t = atom_t - tr * torch.eye(3, device=atom_t.device, dtype=atom_t.dtype)

        # Optional: ENK filtering between backbone features and polar head
        enk_info = {}
        if self.enk is not None:
            h_atom, atom_t, enk_info = self.enk(h_atom, atom_t)

        # Polarizability tensor
        per_atom_polar = self._polar_tensor(h_atom, atom_t, pos, z, batch)  # [N, 3, 3]

        if self.summation:
            num_graphs = int(batch.max().item()) + 1
            mol_polar = scatter_sum(per_atom_polar, batch, dim=0, dim_size=num_graphs)  # [B, 3, 3]
        else:
            mol_polar = per_atom_polar

        if self.grad_type == 'polar':
            return self._grad_polarizability(mol_polar, pos)  # [N, 3, 6]
        return mol_polar  # [B, 3, 3]
