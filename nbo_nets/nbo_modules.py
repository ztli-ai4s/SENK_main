#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained modules for NBO Foundation Model.

Extracted from:
  - nets/electron_channel_local.py  (EquivariantTokenMP, DistanceEncoder)
  - nets/dualpath_nbo.py            (ElectronBaseDeltaAdapter, ElectronPairGuide, helpers)
  - nets/gaussian_rbf.py            (GaussianRadialBasisLayer)

Isolated here to avoid circular import chains (dualpath_nbo -> nbo_nets.model_nbo).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter_add


def _scatter_add_safe(src: torch.Tensor, index: torch.Tensor, out: torch.Tensor, dim: int = 0):
    if src.dtype != out.dtype:
        src = src.to(out.dtype)
    return scatter_add(src, index, out=out, dim=dim)


# ---------------------------------------------------------------------------
# GaussianRadialBasisLayer (from nets/gaussian_rbf.py)
# ---------------------------------------------------------------------------

@torch.jit.script
def _gaussian_fn(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianRadialBasisLayer(nn.Module):
    """Learnable Gaussian RBF (from Graphormer)."""

    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = nn.Parameter(torch.zeros(1, num_basis))
        self.std = nn.Parameter(torch.zeros(1, num_basis))
        self.weight = nn.Parameter(torch.ones(1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        nn.init.uniform_(self.mean, 0.0, 1.0)
        nn.init.uniform_(self.std, 1.0 / num_basis, 1.0)
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)

    def forward(self, dist: torch.Tensor, **kwargs) -> torch.Tensor:
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        return _gaussian_fn(x, mean, std)


# ---------------------------------------------------------------------------
# DistanceEncoder (fixed RBF, from electron_channel_local.py)
# ---------------------------------------------------------------------------

class DistanceEncoder(nn.Module):
    def __init__(self, num_rbf: int = 32, cutoff: float = 5.0, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('centers', torch.linspace(0.0, cutoff, num_rbf))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.numel() == 0:
            return dist.new_zeros((0, self.centers.numel()))
        diff = dist.view(-1, 1) - self.centers.view(1, -1)
        return torch.exp(-self.gamma * diff.pow(2))


# ---------------------------------------------------------------------------
# EquivariantTokenMP (from electron_channel_local.py)
# ---------------------------------------------------------------------------

class EquivariantTokenMP(nn.Module):
    """Lightweight 0e/1o/2e equivariant message passing for typed tokens.

    All gates/weights are scalar (0e);
    1o updates along edge direction r_hat;
    2e updates use traceless second-order basis Q(r_hat).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 2,
        edge_feat_dim: int = 3,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.eps = float(eps)

        self.dist_enc = DistanceEncoder(num_rbf=num_rbf, cutoff=cutoff, gamma=1.0)
        self.type_emb = nn.Embedding(4, hidden_dim)

        in_dim = hidden_dim * 2 + num_rbf + edge_feat_dim + hidden_dim * 2
        self.msg_mlps = nn.ModuleList()
        self.norm_s = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_v = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])
        self.norm_t = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.n_layers)])

        for _ in range(self.n_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim * 4),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim * 6),
                )
            )

    @staticmethod
    def _traceless_Q(r_hat: torch.Tensor) -> torch.Tensor:
        rr = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)
        eye = torch.eye(3, device=r_hat.device, dtype=r_hat.dtype).unsqueeze(0)
        return rr - (1.0 / 3.0) * eye

    def forward(
        self,
        s0: torch.Tensor,
        pos_global: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (s, v, t): scalar [N,H], vector [N,H,3], tensor [N,H,3,3]."""
        n = int(s0.size(0))
        h = int(s0.size(-1))
        device = s0.device
        dtype = s0.dtype

        s = s0
        v = s0.new_zeros((n, h, 3))
        t = s0.new_zeros((n, h, 3, 3))

        if edge_index is None or edge_index.numel() == 0:
            return s, v, t

        src, dst = edge_index
        r = pos_global[dst].float() - pos_global[src].float()
        _eps_sq = max(self.eps * self.eps, 1e-12)
        dist_raw = torch.sqrt(torch.clamp((r * r).sum(dim=-1), min=_eps_sq))
        valid_dir = dist_raw > (10.0 * self.eps)
        dist = dist_raw
        r_hat = r / dist.unsqueeze(-1)
        if (~valid_dir).any():
            r_hat = r_hat.clone()
            r_hat[~valid_dir] = 0.0
        rbf = self.dist_enc(dist).to(dtype)

        if edge_feat is None or edge_feat.numel() == 0:
            edge_feat = s0.new_zeros((edge_index.size(1), 3))
        edge_feat = edge_feat.to(dtype)

        type_src = self.type_emb(node_type[src].clamp(min=0, max=3)).to(dtype)
        type_dst = self.type_emb(node_type[dst].clamp(min=0, max=3)).to(dtype)

        Q = self._traceless_Q(r_hat).to(dtype)
        if (~valid_dir).any():
            Q = Q.clone()
            Q[~valid_dir] = 0.0

        for li in range(self.n_layers):
            feat = torch.cat([s[src], s[dst], rbf, edge_feat, type_src, type_dst], dim=-1)
            out = self.msg_mlps[li](feat)

            ms = out[:, 0:h]
            mv = out[:, h:2*h]
            mt = out[:, 2*h:3*h]
            gs = torch.sigmoid(out[:, 3*h:4*h])
            gv = torch.sigmoid(out[:, 4*h:5*h])
            gt = torch.sigmoid(out[:, 5*h:6*h])

            ms = ms * gs
            mv = mv * gv
            mt = mt * gt

            ds = s.new_zeros((n, h))
            _scatter_add_safe(ms, dst, out=ds, dim=0)

            dv_msg = mv.unsqueeze(-1) * r_hat.unsqueeze(1)
            dv = v.new_zeros((n, h, 3))
            _scatter_add_safe(dv_msg, dst, out=dv, dim=0)

            dt_msg = mt.view(-1, h, 1, 1) * Q.view(-1, 1, 3, 3)
            dt = t.new_zeros((n, h, 3, 3))
            _scatter_add_safe(dt_msg, dst, out=dt, dim=0)

            s = self.norm_s[li](s + ds)
            v = v + dv
            v = self.norm_v[li](v.transpose(1, 2)).transpose(1, 2)
            t_flat = t.view(n, h, 9)
            dt_flat = dt.view(n, h, 9)
            t_flat = self.norm_t[li]((t_flat + dt_flat).transpose(1, 2)).transpose(1, 2)
            t = t_flat.view(n, h, 3, 3)

            trace = (t[:, :, 0, 0] + t[:, :, 1, 1] + t[:, :, 2, 2]) / 3.0
            t = t - trace.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=t.device, dtype=t.dtype)

        return s, v, t


# ---------------------------------------------------------------------------
# Helper functions (from dualpath_nbo.py)
# ---------------------------------------------------------------------------

def _gaussian_rbf_fixed(dist: torch.Tensor, num_rbf: int = 16, cutoff: float = 5.0) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, num_rbf, device=dist.device, dtype=dist.dtype)
    diff = dist.view(-1, 1) - centers.view(1, -1)
    return torch.exp(-diff.pow(2))


def _cosine_envelope(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    ratio = (dist / max(float(cutoff), 1e-8)).clamp(min=0.0, max=1.0)
    env = 0.5 * (torch.cos(math.pi * ratio) + 1.0)
    return torch.where(dist <= cutoff, env, torch.zeros_like(env))


def _build_node_rbf_context(
    pos: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    num_rbf: int = 16,
    cutoff: float = 5.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-node local radial statistics: (rbf_ctx [N, num_rbf], aux [N, 2])."""
    if edge_index is None or edge_index.numel() == 0:
        return pos.new_zeros((num_nodes, num_rbf)), pos.new_zeros((num_nodes, 2))

    row, col = edge_index
    rij = pos[col] - pos[row]
    dist = torch.norm(rij, dim=-1).clamp(min=eps)

    env = _cosine_envelope(dist, cutoff=cutoff)
    rbf = _gaussian_rbf_fixed(dist, num_rbf=num_rbf, cutoff=cutoff)
    rbf_msg = rbf * env.unsqueeze(-1)

    rbf_sum = pos.new_zeros((num_nodes, num_rbf))
    _scatter_add_safe(rbf_msg, row, out=rbf_sum, dim=0)

    w_sum = pos.new_zeros((num_nodes, 1))
    _scatter_add_safe(env.unsqueeze(-1), row, out=w_sum, dim=0)

    rbf_ctx = rbf_sum / w_sum.clamp(min=1e-8)

    dist_sum = pos.new_zeros((num_nodes, 1))
    _scatter_add_safe((dist * env).unsqueeze(-1), row, out=dist_sum, dim=0)
    dist_mean = dist_sum / w_sum.clamp(min=1e-8)
    dist_mean = dist_mean / max(float(cutoff), 1e-8)

    deg = pos.new_zeros((num_nodes, 1))
    _scatter_add_safe(torch.ones_like(dist).unsqueeze(-1), row, out=deg, dim=0)
    deg_feat = torch.log1p(deg)

    aux = torch.cat([dist_mean, deg_feat], dim=-1)
    return torch.nan_to_num(rbf_ctx), torch.nan_to_num(aux)


# ---------------------------------------------------------------------------
# ElectronBaseDeltaAdapter (from dualpath_nbo.py)
# ---------------------------------------------------------------------------

class ElectronBaseDeltaAdapter(nn.Module):
    """Split electron pathway into base/delta and project as scalar correction."""

    def __init__(self, hidden_dim: int, atomic_feat_dim: int = 4, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.atomic_feat_dim = int(atomic_feat_dim)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)

        base_in_dim = self.atomic_feat_dim + self.num_rbf + 2
        self.base_mlp = nn.Sequential(
            nn.LayerNorm(base_in_dim),
            nn.Linear(base_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        delta_in_dim = hidden_dim * 3 + self.atomic_feat_dim + self.num_rbf + 2
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(delta_in_dim),
            nn.Linear(delta_in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.base_scale = nn.Parameter(torch.tensor(0.20))
        self.delta_scale = nn.Parameter(torch.tensor(0.10))

    def forward(
        self,
        pred_atomic: Optional[torch.Tensor],
        geom_atom: torch.Tensor,
        elec_atom: torch.Tensor,
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = geom_atom.dtype
        num_nodes = int(geom_atom.size(0))
        if isinstance(pred_atomic, torch.Tensor) and pred_atomic.numel() > 0 and pred_atomic.size(0) == num_nodes:
            atomic_feat = pred_atomic.to(dtype=dtype)
            if atomic_feat.size(-1) != self.atomic_feat_dim:
                if atomic_feat.size(-1) > self.atomic_feat_dim:
                    atomic_feat = atomic_feat[:, :self.atomic_feat_dim]
                else:
                    pad = atomic_feat.new_zeros((num_nodes, self.atomic_feat_dim - atomic_feat.size(-1)))
                    atomic_feat = torch.cat([atomic_feat, pad], dim=-1)
        else:
            atomic_feat = geom_atom.new_zeros((num_nodes, self.atomic_feat_dim))

        rbf_ctx, aux_ctx = _build_node_rbf_context(
            pos=pos.to(dtype=dtype), edge_index=edge_index,
            num_nodes=num_nodes, num_rbf=self.num_rbf, cutoff=self.cutoff,
        )
        rbf_ctx = rbf_ctx.to(dtype=dtype)
        aux_ctx = aux_ctx.to(dtype=dtype)

        base_feat = torch.cat([atomic_feat, rbf_ctx, aux_ctx], dim=-1)
        elec_base = self.base_mlp(base_feat)

        delta_feat = torch.cat([
            elec_atom, geom_atom, elec_atom - elec_base,
            atomic_feat, rbf_ctx, aux_ctx,
        ], dim=-1)
        elec_delta = self.delta_mlp(delta_feat)
        guided = self.out_norm(
            geom_atom
            + torch.tanh(self.base_scale) * elec_base
            + torch.tanh(self.delta_scale) * elec_delta
        )
        return torch.nan_to_num(elec_base), torch.nan_to_num(elec_delta), torch.nan_to_num(guided)


# ---------------------------------------------------------------------------
# ElectronPairGuide (from dualpath_nbo.py)
# ---------------------------------------------------------------------------

class ElectronPairGuide(nn.Module):
    """Pair-aware electron context: lifts node info to edge-sensitive features."""

    def __init__(self, hidden_dim: int, basis_dim: int = 6, eps: float = 1e-8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.basis_dim = int(basis_dim)
        self.eps = float(eps)
        pair_in_dim = hidden_dim * 3 + self.basis_dim
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_in_dim),
            nn.Linear(pair_in_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.node_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        elec_base: torch.Tensor,
        elec_delta: torch.Tensor,
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        num_nodes = int(elec_base.size(0))
        if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
            return elec_base.new_zeros((num_nodes, self.hidden_dim))

        row, col = edge_index
        rij = pos[col].float() - pos[row].float()
        dist = torch.sqrt(torch.clamp((rij * rij).sum(dim=-1), min=max(self.eps ** 2, 1e-12)))

        deg = elec_base.new_zeros((num_nodes, 1))
        ones = deg.new_ones((dist.size(0), 1))
        _scatter_add_safe(ones, row, out=deg, dim=0)
        _scatter_add_safe(ones, col, out=deg, dim=0)

        dist_sum = elec_base.new_zeros((num_nodes, 1))
        _scatter_add_safe(dist.unsqueeze(-1), row, out=dist_sum, dim=0)
        _scatter_add_safe(dist.unsqueeze(-1), col, out=dist_sum, dim=0)
        d_eq = dist_sum / deg.clamp(min=1.0)
        d_eq_ij = 0.5 * (d_eq[row] + d_eq[col]).clamp(min=1e-2)

        ratio = (dist.unsqueeze(-1).to(dtype=elec_base.dtype) / d_eq_ij).clamp(min=0.0, max=8.0)
        basis = torch.cat([
            ratio,
            1.0 / (1.0 + ratio),
            torch.exp(-ratio),
            torch.exp(-(ratio - 1.0).pow(2)),
            (dist.unsqueeze(-1).to(dtype=elec_base.dtype) / (d_eq_ij + 1.0)),
            torch.log1p(ratio),
        ], dim=-1)

        pair_feat = torch.cat([
            elec_base[row] * elec_base[col],
            torch.abs(elec_delta[row] - elec_delta[col]),
            0.5 * (elec_delta[row] + elec_delta[col]),
            basis,
        ], dim=-1)
        msg = self.pair_mlp(pair_feat)

        node_ctx = elec_base.new_zeros((num_nodes, self.hidden_dim))
        _scatter_add_safe(msg, row, out=node_ctx, dim=0)
        _scatter_add_safe(msg, col, out=node_ctx, dim=0)
        node_ctx = node_ctx / deg.clamp(min=1.0)
        node_ctx = torch.tanh(self.scale) * self.node_proj(node_ctx)
        return torch.nan_to_num(node_ctx)
