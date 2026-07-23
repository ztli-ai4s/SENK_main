"""
V2-Native SO3 Polar Readout Head
=================================

Purpose
-------
Replaces the V1-style readout (proj_s + proj_t → l=0 scalar + l=2 tensor)
with a fully V2-native head that exploits ALL angular-momentum channels
directly from the SO3_Embedding ``[N, (lmax+1)², C]``.

Upgrade over legacy CleanEquiformerPolar head
----------------------------------------------
V1 legacy:
  l=0  → proj_s → [N, H] → scalar head → sb·(I/3)
  l=2  → proj_t → H heads → CG → [N, H, 3, 3] → t_readout → [N, 3, 3]
  l≥3  → invariant norm only → residual scalar enrichment (loses equivariance)

V2 native (this module):
  l=0  → LN → Linear → [N, H]  scalar features  (unchanged, fully scalar)
  l=1  → [N, 3, C] → CG self-interaction 1⊗1→2 → [N, 5, C]
         → channel proj C→H → CG-to-cart → [N, H, 3, 3]  (gated, zero init)
  l=2  → [N, 5, C] → proj_t C→H → CG-to-cart → [N, H, 3, 3]  primary path
  l≥3  → [N, 2l+1, C] → CG self-interaction l⊗l→2 → [N, 5, C]
         → channel proj C→H → CG-to-cart → [N, H, 3, 3]  (gated, zero init)
  All higher-L paths add to the same [N, H, 3, 3] tensor output.

ENK-SO3 fusion (optional)
--------------------------
When enk is passed to V2SO3PolarReadout.forward(), ENK operates on the
per-atom SCALAR channel (l=0, after projection) — this is equivariance-safe
because ENK only gates scalars and broadcasts a scalar confidence to tensor
channels. The tensor channels are then gated by ENK tensor_confidence.

This means ENK is effectively inside the readout, not after it, giving it
direct influence on which equivariant channels are trusted.

Multi-resolution potential
--------------------------
The architecture naturally supports multiple lmax_list resolutions because
each l-channel is processed independently and additively combined. Future
extensions can inject different lmax sub-networks at checkpoints during
inference.

Training safety
---------------
- All high-L paths are zero-initialized (gate=0 → contribution=0 at start)
- l=0+l=2 path = exact V1 behavior at initialization
- No new numerical singularities: all contractions are CG tensor products
    and linear projections (no division, no softmax over L dimension)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from e3nn import o3


# ---------------------------------------------------------------------------
#  Change-of-basis: real SH l=2  →  traceless symmetric 3×3 (flat 9-dim)
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
    return U_T.T.float()   # [9, 5]


# ---------------------------------------------------------------------------
#  _HighLEquivariantContrib:  l→2 equivariant contribution
# ---------------------------------------------------------------------------

class _HighLEquivariantContrib(nn.Module):
    """Route high-L tensor features into l=2 via CG self-interaction.

    Uses the Clebsch-Gordan coupling  l ⊗ l → 2  to produce l=2 SH
    coefficients from the l-th order features.  This is *strictly*
    equivariant by the Wigner-Eckart theorem:

        x_l → D^l(R) x_l  ⟹  CG(x_l ⊗ x_l)|_{l=2} → D^2(R) · result

    After the CG contraction, a channel projection C → H maps
    [N, 5, C] → [N, 5, H].  The projection is equivariance-safe because
    the same weight is applied to every m-component (scalar per channel).

    At init: proj weight=0  +  gate=0  → zero contribution.
    """

    def __init__(self, sphere_channels: int, l: int, hidden_nf: int):
        super().__init__()
        self.l = l
        self.nm = 2 * l + 1
        # Pre-compute CG coefficients: l ⊗ l → 2   [2l+1, 2l+1, 5]
        cg = o3.wigner_3j(l, l, 2)
        self.register_buffer('_cg', cg.float())
        # Channel projection C → H (equivariance-safe: shared across m)
        self.proj = nn.Linear(sphere_channels, hidden_nf, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_l: [N, nm, C]  the l-th order SO3_Embedding slice
        Returns:
            contrib: [N, 5, hidden_nf]  l=2 SH contribution
        """
        # CG self-interaction: l ⊗ l → 2
        # y[n,k,c] = Σ_{m1,m2} cg[m1,m2,k] · x[n,m1,c] · x[n,m2,c]
        y = torch.einsum('ijk, nic, njc -> nkc', self._cg, x_l, x_l)
        out = self.proj(y)                          # [N, 5, H]
        return torch.tanh(self.gate) * out


# ---------------------------------------------------------------------------
#  _L1TensorContrib:  l=1 vector path → rank-2 tensor contribution
# ---------------------------------------------------------------------------

class _L1TensorContrib(nn.Module):
    """Contribution from l=1 features to rank-2 tensor via CG self-interaction.

    Uses the Clebsch-Gordan coupling  1 ⊗ 1 → 2  to produce l=2 SH
    coefficients from l=1 features.  The irreducible decomposition
      1 ⊗ 1 → 0 ⊕ 1 ⊕ 2
    guarantees that the CG contraction to l=2 extracts exactly the
    traceless symmetric (rank-2) part, matching the outer-product approach
    but staying in the SH basis — no Cartesian basis ordering needed.

    Equivariance:  v → D¹(R)v  ⟹  CG(v⊗v)|_{l=2} → D²(R) · result
    Gate init=0 → zero contribution at start.
    """

    def __init__(self, sphere_channels: int, hidden_nf: int):
        super().__init__()
        # CG coefficients: 1 ⊗ 1 → 2   [3, 3, 5]
        cg = o3.wigner_3j(1, 1, 2)
        self.register_buffer('_cg', cg.float())
        self.proj = nn.Linear(sphere_channels, hidden_nf, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_1: [N, 3, C]  l=1 SO3_Embedding (m=-1,0,+1 × C channels)
        Returns:
            contrib: [N, 5, hidden_nf]  l=2 SH contribution
        """
        # CG self-interaction: 1 ⊗ 1 → 2
        y = torch.einsum('ijk, nic, njc -> nkc', self._cg, x_1, x_1)
        out = self.proj(y)                                 # [N, 5, H]
        return torch.tanh(self.gate) * out


# ---------------------------------------------------------------------------
#  V2SO3PolarReadout — the main class
# ---------------------------------------------------------------------------

class V2SO3PolarReadout(nn.Module):
    """
    V2-native readout head: SO3_Embedding → per-atom polarizability tensor.

    Replaces the V1 readout in CleanEquiformerPolar/CleanEquiformerPolarExt
    when the backbone is EquiformerV2.

    Input:  SO3_Embedding.embedding  [N, (lmax+1)², C]
    Output: per-atom polarizability tensor [N, 3, 3]

    Architecture
    ------------
    l=0 path (scalar):
        x[:,0,:] → LN → Linear → [N, H]  →  sout → (sa, sb)  [N,1],[N,1]

    l=1 path (vector):
        x[:,1:4,:] → _L1TensorContrib → [N, H, 3, 3]  (gated, zero init)

    l=2 path (tensor, primary):
        x[:,4:9,:] → proj_t → [N, H, 5] → CG (2e→cart33) → [N, H, 3, 3]
                   → t_readout → [N, 3, 3]  (same as V1)

    l≥3 paths (high-L equivariant):
        each x[:,off:off+nm,:] → _HighLEquivariantContrib → [N, 5, H]
                                → CG → [N, H, 3, 3]  (gated, zero init)

    Final add-reduce:
        all [N, H, 3, 3] tensors added → sum_t  [N, H, 3, 3]
        t_readout: [N, H, 3, 3] → [N, 3, 3]  (project hidden dimension out)
        enforce tracelessness on sum

    Polarizability formula (same as V1):
        α_i = sb · (I/3)  +  outt  +  sa · Q(r_i)
    """

    def __init__(
        self,
        sphere_channels: int,    # C from backbone
        lmax: int,               # lmax from backbone
        hidden_nf: int = 128,    # H, readout hidden dimension
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.hidden_nf = hidden_nf
        C = sphere_channels

        # --- l=0: scalar path ---
        self.proj_s = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, hidden_nf),
        )
        self.sout = nn.Sequential(
            nn.LayerNorm(hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, 2),       # → (sa, sb)
        )

        # --- l=2: primary tensor path (same as V1, always active) ---
        self.proj_t = nn.Linear(C, hidden_nf, bias=False)
        cob = _compute_2e_to_cart33()                      # [9, 5]
        self.register_buffer('_cob_2e_to_cart', cob)
        self.t_readout = nn.Linear(hidden_nf, 1, bias=False)

        # --- l=1: vector→tensor path (gated, zero init) ---
        self.l1_contrib = _L1TensorContrib(C, hidden_nf)

        # --- l≥3: high-L equivariant contributions (gated, zero init) ---
        self.highL_contribs = nn.ModuleDict()
        for l in range(3, lmax + 1):
            self.highL_contribs[str(l)] = _HighLEquivariantContrib(C, l, hidden_nf)

        # --- mass table for center-of-mass ---
        mass_table = torch.ones(100, dtype=torch.float32)
        mass_table[1] = 1.0079; mass_table[6] = 12.011
        mass_table[7] = 14.0067; mass_table[8] = 15.999; mass_table[9] = 18.998
        self.register_buffer('_mass_table', mass_table)

        tril_mask = torch.tril(torch.ones(3, 3)).flatten()
        self.register_buffer('_tril_mask', tril_mask)

    # ------------------------------------------------------------------
    def _center_of_mass(self, pos, z, batch):
        num_graphs = int(batch.max().item()) + 1
        z_c = z.clamp(0, self._mass_table.numel() - 1)
        masses = self._mass_table[z_c].to(dtype=pos.dtype, device=pos.device).unsqueeze(-1)
        mass_sum = scatter_sum(masses, batch, dim=0, dim_size=num_graphs).clamp(1e-8)
        pos_m = scatter_sum(pos * masses, batch, dim=0, dim_size=num_graphs)
        return pos_m / mass_sum

    # ------------------------------------------------------------------
    def forward(
        self,
        so3_emb: torch.Tensor,   # [N, (lmax+1)², C]
        pos: torch.Tensor,        # [N, 3]
        z: torch.Tensor,          # [N]
        batch: torch.Tensor,      # [N]
        h_enk: Optional[torch.Tensor] = None,   # [N, H] ENK-filtered scalars (optional)
        t_enk: Optional[torch.Tensor] = None,   # [N, H, 3, 3] ENK-filtered tensors (optional)
    ) -> torch.Tensor:
        """
        Args:
            so3_emb:  Raw SO3_Embedding.embedding [N, (lmax+1)², C]
            pos, z, batch:  Standard molecular graph inputs
            h_enk:  If not None, replaces l=0 scalar path output
            t_enk:  If not None, *adds* to the tensor sum (ENK-filtered tensor channel)
        Returns:
            per_atom_polar: [N, 3, 3]
        """
        N = so3_emb.shape[0]
        C = self.sphere_channels
        lmax = self.lmax
        eye3 = torch.eye(3, device=pos.device, dtype=pos.dtype)

        # ---- l=0: scalar path ----
        x0 = so3_emb[:, 0, :]                             # [N, C]
        if h_enk is not None:
            h_atom = h_enk                                 # [N, H] from ENK
        else:
            h_atom = self.proj_s(x0)                       # [N, H]

        s = self.sout(h_atom)
        sa = s[:, :1]   # [N, 1]
        sb = s[:, 1:]   # [N, 1]

        # ---- l=2: primary tensor path ----
        # SO3_Embedding layout: offset 0 = l=0 (1 coeff), 1:4 = l=1 (3), 4:9 = l=2 (5)
        x2 = so3_emb[:, 4:9, :]                           # [N, 5, C]
        # Project C → H channels: [N, 5, H]
        x2_h = self.proj_t(x2)                            # [N, 5, H]
        # SH → cart33: [N, H, 3, 3]
        x2_cart = torch.einsum('nsh, cs -> nhc', x2_h, self._cob_2e_to_cart)  # [N, H, 9]
        x2_cart = x2_cart.reshape(N, self.hidden_nf, 3, 3)                    # [N, H, 3, 3]

        # Start accumulating tensor contributions
        # t_enk overrides l=2 if provided (ENK already filtered this channel)
        if t_enk is not None:
            tensor_acc = t_enk                             # [N, H, 3, 3]
        else:
            tensor_acc = x2_cart                           # [N, H, 3, 3]

        # ---- l=1: vector path → l=2 CG self-interaction ----
        x1 = so3_emb[:, 1:4, :]                           # [N, 3, C]
        l1_5H = self.l1_contrib(x1)                        # [N, 5, H]  (gated, CG)
        l1_cart = torch.einsum(
            'nsh, cs -> nhc', l1_5H, self._cob_2e_to_cart
        ).reshape(N, self.hidden_nf, 3, 3)                 # [N, H, 3, 3]
        tensor_acc = tensor_acc + l1_cart

        # ---- l≥3: high-L equivariant contributions ----
        offset = 0
        for l in range(lmax + 1):
            nm = 2 * l + 1
            if l >= 3 and str(l) in self.highL_contribs:
                x_l = so3_emb[:, offset:offset + nm, :]   # [N, nm, C]
                # Contrib: [N, 5, H] → via CG → [N, H, 3, 3]
                contrib_5H = self.highL_contribs[str(l)](x_l)   # [N, 5, H]
                contrib_cart = torch.einsum(
                    'nsh, cs -> nhc', contrib_5H, self._cob_2e_to_cart
                ).reshape(N, self.hidden_nf, 3, 3)              # [N, H, 3, 3]
                tensor_acc = tensor_acc + contrib_cart
            offset += nm

        # ---- Enforce tracelessness on accumulated tensor ----
        tr = tensor_acc.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) / 3.0
        tensor_acc = tensor_acc - tr * eye3

        # ---- t_readout: [N, H, 3, 3] → [N, 3, 3] ----
        # permute to [N, 3, 3, H], apply readout, squeeze
        outt_raw = self.t_readout(tensor_acc.permute(0, 2, 3, 1)).squeeze(-1)  # [N, 3, 3]
        # symmetrize + re-enforce tracelessness
        outt = 0.5 * (outt_raw + outt_raw.transpose(-1, -2))
        tr_out = (outt[:, 0, 0] + outt[:, 1, 1] + outt[:, 2, 2]) / 3.0
        outt = outt - tr_out[:, None, None] * eye3                             # [N, 3, 3]

        # ---- Geometric quadrupole Q(r) ----
        com = self._center_of_mass(pos, z, batch)
        ra = pos - com[batch]
        Qt = ra.unsqueeze(-1) * ra.unsqueeze(-2)
        r_sq = (ra * ra).sum(-1, keepdim=True).unsqueeze(-1)
        Qt = Qt - (r_sq / 3.0) * eye3

        # ---- Per-atom polarizability: α_i = sb·(I/3) + outt + sa·Q(r) ----
        per_atom = sb.unsqueeze(-1) * (eye3 / 3.0) + outt + sa.unsqueeze(-1) * Qt
        return per_atom   # [N, 3, 3]

    def depolar_flat(self, per_atom_polar: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Autograd depolarizability: ∂α/∂R  [N, 3, 6]."""
        batch_size = per_atom_polar.shape[0]
        # Lower triangle (6 unique components of 3×3 symmetric tensor)
        polar_flat = per_atom_polar.flatten(1)[:, self._tril_mask == 1]  # [N, 6] or [B, 6]
        depolar = torch.zeros(pos.shape[0], 3, 6, device=pos.device, dtype=pos.dtype)
        for i in range(6):
            depolar[:, :, i] = -torch.autograd.grad(
                polar_flat[:, i].sum(), pos,
                create_graph=self.training, retain_graph=True,
            )[0]
        return depolar
