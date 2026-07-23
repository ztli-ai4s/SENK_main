#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NBO Foundation Model v2.

Architecture:
  Stage A — Atom geometry stem (embedding + learnable Gaussian RBF)
  Stage B — Heterogeneous electron graph via EquivariantTokenMP
  Stage C — ElectronBaseDeltaAdapter + ElectronPairGuide refinement
  Stage D — Decoupled multi-head prediction

Key improvements over v1:
  - Explicit atom/bond/LP tokens with independent message passing
  - Learnable GaussianRadialBasisLayer replaces fixed EGNN distance filters
  - Equivariant 0e/1o/2e features via EquivariantTokenMP
  - Base/delta residual adapter and pair-aware refinement
  - Separate, decoupled prediction heads per task family
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter_add, scatter

from nbo_shared_modules import (
    GaussianRadialBasisLayer,
    DistanceEncoder,
    EquivariantTokenMP,
    ElectronBaseDeltaAdapter,
    ElectronPairGuide,
)


def _scatter_add_safe(src: torch.Tensor, index: torch.Tensor, out: torch.Tensor, dim: int = 0):
    if src.dtype != out.dtype:
        src = src.to(out.dtype)
    return scatter_add(src, index, out=out, dim=dim)


# Stage A: Atom geometry stem

class AtomGeometryStem(nn.Module):
    """Shared low-level atomic encoder with learnable Gaussian RBF."""

    def __init__(self, hidden_dim: int, max_z: int = 100, num_rbf: int = 64, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_embed = nn.Embedding(max_z + 1, hidden_dim)
        self.rbf = GaussianRadialBasisLayer(num_rbf, cutoff)
        self.cutoff = cutoff

        # Atom self-update from neighbourhood
        self.env_proj = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        z: torch.Tensor,         # [N_atom] atomic numbers
        pos: torch.Tensor,       # [N_atom, 3]
        edge_index: torch.Tensor, # [2, E] atom–atom bond edges
    ) -> torch.Tensor:
        """Returns [N_atom, hidden_dim] atom embeddings."""
        h = self.z_embed(z)

        if edge_index.numel() > 0:
            row, col = edge_index
            diff = pos[col] - pos[row]
            dist = torch.sqrt(torch.clamp((diff * diff).sum(-1), min=1e-12))
            rbf_feat = self.rbf(dist)  # [E, num_rbf]
            env_msg = self.env_proj(rbf_feat)  # [E, H]

            agg = torch.zeros_like(h)
            _scatter_add_safe(env_msg, row, out=agg, dim=0)
            h = self.norm(h + agg)
        return h


# Stage B: Token expansion + heterogeneous message passing

class TokenExpander(nn.Module):
    """Expand atom embeddings into explicit atom/bond/LP tokens."""

    def __init__(self, hidden_dim: int, max_lp_slots: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Node type embedding: 0=atom, 1=bond, 3=LP (matching EquivariantTokenMP convention)
        self.type_embed = nn.Embedding(4, hidden_dim)
        # Bond token: pair of atom features → token
        self.bond_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # LP token
        self.lp_slot_embed = nn.Embedding(max_lp_slots, hidden_dim)
        self.lp_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_atom: torch.Tensor,        # [N_atom, H]
        node_type: torch.Tensor,      # [N_global] 0=atom,1=bond,2or3=LP
        atom_bond_index: torch.Tensor, # [2, E_bond] atom-atom bond edges
        atom_to_nbo_index: torch.Tensor, # [2, E_a2nbo]
        is_atom_mask: torch.Tensor,   # [N_global] bool
    ) -> torch.Tensor:
        """Returns [N_global, H] tokens for all nodes."""
        device = h_atom.device
        dtype = h_atom.dtype
        N_global = node_type.size(0)
        N_atom = h_atom.size(0)

        h_global = self.type_embed(node_type.clamp(min=0, max=3)).to(dtype)
        h_global[is_atom_mask] = h_atom

        # --- Bond tokens ---
        bond_mask = (node_type == 1)
        bond_nodes = torch.where(bond_mask)[0]
        if bond_nodes.numel() > 0 and atom_to_nbo_index.numel() > 0:
            # Build remapping: global atom idx → local atom idx
            remap = torch.full((N_global,), -1, dtype=torch.long, device=device)
            remap[is_atom_mask] = torch.arange(N_atom, device=device)

            a_src, n_tgt = atom_to_nbo_index
            # For each bond node, find its parent atoms and create features
            for bn in bond_nodes.tolist():
                parents_mask = (n_tgt == bn) & (node_type[a_src] == 0)
                parent_atoms = a_src[parents_mask]
                if parent_atoms.numel() >= 2:
                    # Use first two connected atoms
                    la, lb = remap[parent_atoms[0]].item(), remap[parent_atoms[1]].item()
                    if la >= 0 and lb >= 0:
                        pair_feat = torch.cat([h_atom[la], h_atom[lb]], dim=-1)
                        h_global[bn] = self.bond_token_mlp(pair_feat.unsqueeze(0)).squeeze(0).to(h_global.dtype)
                elif parent_atoms.numel() == 1:
                    la = remap[parent_atoms[0]].item()
                    if la >= 0:
                        h_global[bn] = h_atom[la].to(h_global.dtype)

        # --- LP tokens ---
        lp_mask = (node_type == 3)
        lp_nodes = torch.where(lp_mask)[0]
        if lp_nodes.numel() > 0 and atom_to_nbo_index.numel() > 0:
            remap = torch.full((N_global,), -1, dtype=torch.long, device=device)
            remap[is_atom_mask] = torch.arange(N_atom, device=device)

            a_src, n_tgt = atom_to_nbo_index
            parent_local = torch.full((N_global,), -1, dtype=torch.long, device=device)

            for s, t in zip(a_src.tolist(), n_tgt.tolist()):
                if node_type[t] == 3 and parent_local[t] == -1:
                    pa = remap[s].item()
                    if pa >= 0:
                        parent_local[t] = pa

            # Assign slot ids per parent atom
            slot_id = torch.zeros(N_global, dtype=torch.long, device=device)
            per_atom_counter = torch.zeros(N_atom, dtype=torch.long, device=device)
            for t in lp_nodes.tolist():
                pa = parent_local[t]
                if pa >= 0:
                    slot_id[t] = per_atom_counter[pa]
                    per_atom_counter[pa] += 1

            valid_lp = parent_local[lp_nodes] >= 0
            if valid_lp.any():
                lp_valid = lp_nodes[valid_lp]
                pa = parent_local[lp_valid]
                lp_base = h_atom[pa]
                slot_emb = self.lp_slot_embed(slot_id[lp_valid].clamp(max=self.lp_slot_embed.num_embeddings - 1))
                lp_token = self.lp_mlp(torch.cat([lp_base, slot_emb.to(dtype)], dim=-1)).to(h_global.dtype)
                h_global[lp_valid] = lp_token

        return h_global


# Stage D: Prediction heads

class AtomPropertyHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(h.float())).to(h.dtype)


class BondPropertyHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, h_bond_tokens: torch.Tensor) -> torch.Tensor:
        """h_bond_tokens: [N_bond, H] from token embeddings."""
        return self.mlp(self.norm(h_bond_tokens.float())).to(h_bond_tokens.dtype)


class BondPairHead(nn.Module):
    """Fallback bond head when bond tokens are absent (uses atom pair concat)."""
    def __init__(self, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h_atom: torch.Tensor, bond_index: torch.Tensor) -> torch.Tensor:
        h_pair = torch.cat([h_atom[bond_index[0]], h_atom[bond_index[1]]], dim=-1)
        return self.mlp(self.norm(h_pair.float())).to(h_atom.dtype)


class InteractionHead(nn.Module):
    """Predicts E2 and other interaction descriptors on candidate edges."""
    def __init__(self, hidden_dim: int, out_dim: int = 3):
        super().__init__()
        self.embed_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_norm = nn.LayerNorm(hidden_dim)
        self.dist_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dir_net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, h_global: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        diff = pos[row] - pos[col]
        _eps = 1e-8
        dist_sq = (diff * diff).sum(dim=-1)
        dist = torch.sqrt(torch.clamp(dist_sq, min=_eps * _eps))
        distances = (dist / 3.0).clamp(0, 2).unsqueeze(-1)
        directions = diff / dist.unsqueeze(-1)
        near_zero = dist_sq.detach() < (_eps * _eps * 100)
        if near_zero.any():
            directions = directions.clone()
            directions[near_zero] = 0.0

        with torch.amp.autocast('cuda', enabled=False):
            h_i = self.embed_norm(self.embed_mlp(h_global[row].float()))
            h_j = self.embed_norm(self.embed_mlp(h_global[col].float()))
            df = self.dist_net(distances.float())
            de = self.dir_net(directions.float())
            emb = h_i * de * df * h_j
            emb = torch.nan_to_num(emb)
            return self.out_mlp(emb).to(h_global.dtype)


class LinkPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_global: torch.Tensor, candidate_edges: torch.Tensor) -> torch.Tensor:
        row, col = candidate_edges
        h_pairs = torch.cat([h_global[row], h_global[col]], dim=-1)
        with torch.amp.autocast('cuda', enabled=False):
            logits = self.mlp(h_pairs.float()).squeeze(-1)
        logits = torch.nan_to_num(logits).clamp(-30, 30)
        return logits.to(h_global.dtype)


# Full Model

class NBOFoundationModel(nn.Module):
    """NBO Foundation Model v2.

    Args:
        hidden_dim:       Hidden dimension for all submodules.
        n_token_mp_layers: Number of EquivariantTokenMP message passing layers.
        n_token_mp_blocks: Number of stacked EquivariantTokenMP blocks.
        num_rbf:          Number of Gaussian radial basis functions.
        cutoff:           Distance cutoff in Angstroms.
        max_z:            Maximum atomic number.
        atom_out_dim:     Output dim of atom property head.
        bond_out_dim:     Output dim of bond property head.
        interaction_out_dim: Output dim of interaction head.
        use_pair_guide:   Whether to apply ElectronPairGuide.
        use_base_delta:   Whether to apply ElectronBaseDeltaAdapter.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_token_mp_layers: int = 2,
        n_token_mp_blocks: int = 3,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        max_z: int = 100,
        atom_out_dim: int = 4,
        bond_out_dim: int = 2,
        interaction_out_dim: int = 3,
        use_pair_guide: bool = True,
        use_base_delta: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Stage A: Atom geometry stem
        self.stem = AtomGeometryStem(hidden_dim, max_z=max_z, num_rbf=num_rbf, cutoff=cutoff)

        # Stage B: Token expansion + equivariant MP
        self.token_expander = TokenExpander(hidden_dim)
        self.token_mp_blocks = nn.ModuleList([
            EquivariantTokenMP(
                hidden_dim=hidden_dim,
                n_layers=n_token_mp_layers,
                edge_feat_dim=3,
                num_rbf=32,
                cutoff=cutoff,
            )
            for _ in range(n_token_mp_blocks)
        ])

        # Stage C: Refinement adapters
        self.use_base_delta = use_base_delta
        self.use_pair_guide = use_pair_guide
        if use_base_delta:
            self.base_delta_adapter = ElectronBaseDeltaAdapter(
                hidden_dim=hidden_dim,
                atomic_feat_dim=atom_out_dim,
                num_rbf=16,
                cutoff=cutoff,
            )
        if use_pair_guide:
            self.pair_guide = ElectronPairGuide(hidden_dim=hidden_dim)

        # Stage D: Prediction heads
        self.atom_head = AtomPropertyHead(hidden_dim, atom_out_dim)
        self.bond_head = BondPropertyHead(hidden_dim, bond_out_dim)
        self.bond_pair_head = BondPairHead(hidden_dim, bond_out_dim)
        self.interaction_head = InteractionHead(hidden_dim, interaction_out_dim)
        self.link_head = LinkPredictionHead(hidden_dim)

    def _build_token_edge_index(
        self,
        atom_bond_index: torch.Tensor,
        interaction_edge_index: torch.Tensor,
        atom_to_nbo_index: torch.Tensor,
        is_atom_mask: torch.Tensor,
        N_global: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge all edge types into a single edge index for EquivariantTokenMP.

        Returns: (edge_index [2, E_total], edge_feat [E_total, 3])
        """
        edges = []
        # Atom-atom bonds (bidirectional)
        if atom_bond_index.numel() > 0:
            # These are already in global space for simg data;
            # for qcMol atom-only data, they should also be global atom indices
            fwd = atom_bond_index
            rev = torch.stack([atom_bond_index[1], atom_bond_index[0]], dim=0)
            edges.append(fwd)
            edges.append(rev)

        # atom ↔ NBO (bond/LP) structure edges (bidirectional)
        if atom_to_nbo_index.numel() > 0:
            edges.append(atom_to_nbo_index)
            edges.append(torch.stack([atom_to_nbo_index[1], atom_to_nbo_index[0]], dim=0))

        # interaction edges (bidirectional)
        if interaction_edge_index.numel() > 0:
            edges.append(interaction_edge_index)
            edges.append(torch.stack([interaction_edge_index[1], interaction_edge_index[0]], dim=0))

        if not edges:
            ei = torch.empty((2, 0), dtype=torch.long, device=device)
            ef = torch.zeros((0, 3), dtype=torch.float, device=device)
            return ei, ef

        ei = torch.cat(edges, dim=1)
        # Deduplicate
        key = ei[0] * N_global + ei[1]
        uniq_key, inv = torch.unique(key, return_inverse=True)
        row = torch.div(uniq_key, N_global, rounding_mode='floor')
        col = uniq_key.remainder(N_global)
        ei = torch.stack([row, col], dim=0)

        # Placeholder edge features (3-dim zeros; EquivariantTokenMP also uses distance enc)
        ef = torch.zeros((ei.size(1), 3), dtype=torch.float, device=device)
        return ei, ef

    def forward(
        self,
        data,
        link_prediction_candidates: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG Data/Batch with fields:
                - x: [N, F] node features (col 0 = Z for atoms)
                - pos: [N, 3] positions
                - node_type: [N] (0=atom, 1=bond, 2/3=LP)
                - atom_bond_index: [2, E_bond] global atom–atom edges
                - interaction_edge_index: [2, E_int] NBO interaction edges
                - atom_to_nbo_index: [2, E_a2n] atom → NBO structure edges
            link_prediction_candidates: [2, E_cand] optional

        Returns:
            dict with keys: pred_atom, pred_bond, pred_interaction,
                            pred_link_logits, [h_global if return_hidden]
        """
        x_raw = data.x
        pos_global = data.pos
        node_type = data.node_type.clone()

        # Normalize node_type: 2 → 3 (LP) for EquivariantTokenMP convention
        node_type[node_type == 2] = 3

        atom_bond_index = getattr(data, 'atom_bond_index', torch.empty((2, 0), dtype=torch.long, device=x_raw.device))
        interaction_edge_index = getattr(data, 'interaction_edge_index', torch.empty((2, 0), dtype=torch.long, device=x_raw.device))
        atom_to_nbo_index = getattr(data, 'atom_to_nbo_index', torch.empty((2, 0), dtype=torch.long, device=x_raw.device))

        device = x_raw.device
        N_global = node_type.size(0)
        is_atom_mask = (node_type == 0)
        N_atom = int(is_atom_mask.sum().item())

        # Remapping: global atom index → local atom index
        remap_to_local = torch.full((N_global,), -1, dtype=torch.long, device=device)
        remap_to_local[is_atom_mask] = torch.arange(N_atom, device=device)

        # Remap atom_bond_index to local atom space for stem
        atom_bond_local = torch.empty((2, 0), dtype=torch.long, device=device)
        if atom_bond_index.numel() > 0:
            atom_bond_local = remap_to_local[atom_bond_index]
            # Filter invalid (bond to non-atom)
            valid = (atom_bond_local[0] >= 0) & (atom_bond_local[1] >= 0)
            atom_bond_local = atom_bond_local[:, valid]

        # ====== Stage A ======
        z = x_raw[is_atom_mask, 0].long()
        pos_atom = pos_global[is_atom_mask]
        h_atom = self.stem(z, pos_atom, atom_bond_local)

        # ====== Stage B ======
        h_global = self.token_expander(h_atom, node_type, atom_bond_index, atom_to_nbo_index, is_atom_mask)

        # Build merged edge index for EquivariantTokenMP
        token_ei, token_ef = self._build_token_edge_index(
            atom_bond_index, interaction_edge_index, atom_to_nbo_index,
            is_atom_mask, N_global, device,
        )

        # Multi-block EquivariantTokenMP
        s = h_global
        for block in self.token_mp_blocks:
            s, v, t = block(s, pos_global, node_type, token_ei, token_ef)
            # Use only scalar channel; v and t reserved for future use

        h_global_final = s

        # ====== Stage C: Refinement ======
        h_atom_final = h_global_final[is_atom_mask]

        if self.use_base_delta or self.use_pair_guide:
            # Get atom-level predictions for base/delta input
            pred_atom_prelim = self.atom_head(h_atom_final)

            # Use atom-level bond edges for adapter
            bd_edge = atom_bond_local if atom_bond_local.numel() > 0 else None

            if self.use_base_delta:
                elec_base, elec_delta, guided_atom = self.base_delta_adapter(
                    pred_atom_prelim, h_atom_final, h_atom_final, pos_atom, bd_edge,
                )
                h_atom_final = guided_atom

                if self.use_pair_guide:
                    pair_ctx = self.pair_guide(elec_base, elec_delta, pos_atom, bd_edge)
                    h_atom_final = h_atom_final + pair_ctx
            else:
                elec_base = h_atom_final
                elec_delta = h_atom_final
                if self.use_pair_guide:
                    pair_ctx = self.pair_guide(elec_base, elec_delta, pos_atom, bd_edge)
                    h_atom_final = h_atom_final + pair_ctx

        # Write refined atom features back into global
        h_global_final = h_global_final.clone()
        h_global_final[is_atom_mask] = h_atom_final

        # ====== Stage D: Prediction ======
        results: Dict[str, torch.Tensor] = {}

        # D1: Atom properties
        results['pred_atom'] = torch.nan_to_num(self.atom_head(h_atom_final))

        # D2: Bond properties — always use atom-pair head to align with
        #     simg y_bond_props which is indexed by atom_bond_index edges.
        #     (bond_head is kept for potential per-token use in future.)
        if atom_bond_local.numel() > 0:
            results['pred_bond'] = torch.nan_to_num(self.bond_pair_head(h_atom_final, atom_bond_local))
        else:
            out_dim = self.bond_pair_head.mlp[-1].out_features
            results['pred_bond'] = torch.zeros(0, out_dim, device=device, dtype=h_atom_final.dtype)

        # D3: Interaction properties
        if interaction_edge_index.numel() > 0:
            results['pred_interaction'] = torch.nan_to_num(
                self.interaction_head(h_global_final, interaction_edge_index, pos_global)
            )
        else:
            out_dim = self.interaction_head.out_mlp[-1].out_features
            results['pred_interaction'] = torch.zeros(0, out_dim, device=device, dtype=h_atom_final.dtype)

        # D4: Link prediction
        lpc = link_prediction_candidates
        if lpc is not None and lpc.numel() > 0:
            results['pred_link_logits'] = torch.nan_to_num(self.link_head(h_global_final, lpc))
        else:
            results['pred_link_logits'] = torch.tensor([], device=device, dtype=h_atom_final.dtype)

        if return_hidden:
            results['h_global'] = h_global_final
            results['h_atom'] = h_atom_final

        return results


# QcMol variant: configurable output dimensions, optional aux heads

class NBOFoundationModelQcMol(nn.Module):
    """QcMol-compatible wrapper that adds configurable aux heads.

    Reuses NBOFoundationModel as backbone but replaces output heads
    with dataset-specific dimensions and adds auxiliary heads.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_token_mp_layers: int = 2,
        n_token_mp_blocks: int = 3,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        max_z: int = 100,
        # simg-compat heads
        atom_out_dim: int = 4,
        bond_out_dim: int = 2,
        interaction_out_dim: int = 3,
        # qcMol-specific dims
        qcmol_atom_dim: int = 12,
        qcmol_bond_dim: int = 15,
        aux_atom_dim: int = 0,
        aux_bond_dim: int = 0,
        use_pair_guide: bool = True,
        use_base_delta: bool = True,
    ):
        super().__init__()
        # Core backbone (shared with simg pipeline)
        self.backbone = NBOFoundationModel(
            hidden_dim=hidden_dim,
            n_token_mp_layers=n_token_mp_layers,
            n_token_mp_blocks=n_token_mp_blocks,
            num_rbf=num_rbf,
            cutoff=cutoff,
            max_z=max_z,
            atom_out_dim=atom_out_dim,
            bond_out_dim=bond_out_dim,
            interaction_out_dim=interaction_out_dim,
            use_pair_guide=use_pair_guide,
            use_base_delta=use_base_delta,
        )

        # qcMol-specific heads
        self.qcmol_atom_head = AtomPropertyHead(hidden_dim, qcmol_atom_dim)
        self.qcmol_bond_head = BondPairHead(hidden_dim, qcmol_bond_dim)
        self.aux_atom_head = AtomPropertyHead(hidden_dim, aux_atom_dim) if aux_atom_dim > 0 else None
        self.aux_bond_head = BondPairHead(hidden_dim, aux_bond_dim) if aux_bond_dim > 0 else None

    def forward(self, data) -> Dict[str, torch.Tensor]:
        # Forward through backbone to get hidden representations
        out = self.backbone(data, return_hidden=True)

        h_atom = out['h_atom']
        device = h_atom.device
        dtype = h_atom.dtype

        results: Dict[str, torch.Tensor] = {}
        results['pred_atom'] = torch.nan_to_num(self.qcmol_atom_head(h_atom))
        results['pred_interaction'] = torch.nan_to_num(out['pred_interaction'])

        # Bond: use atom pair for qcMol (atom-only node_type)
        atom_bond_index = getattr(data, 'atom_bond_index', torch.empty((2, 0), dtype=torch.long, device=device))
        node_type = data.node_type
        is_atom_mask = (node_type == 0) | (node_type == 2) | (node_type == 3)
        # For qcMol, all nodes are atoms, remap to local
        if (node_type == 0).sum() == node_type.size(0) or (node_type.max() == 0):
            # Pure atom graph
            if atom_bond_index.numel() > 0:
                results['pred_bond'] = torch.nan_to_num(self.qcmol_bond_head(h_atom, atom_bond_index))
            else:
                results['pred_bond'] = torch.zeros(0, self.qcmol_bond_head.mlp[-1].out_features, device=device, dtype=dtype)
        else:
            # Heterogeneous: use backbone's remap
            N_global = node_type.size(0)
            is_atom = (node_type == 0)
            N_a = int(is_atom.sum())
            remap = torch.full((N_global,), -1, dtype=torch.long, device=device)
            remap[is_atom] = torch.arange(N_a, device=device)
            if atom_bond_index.numel() > 0:
                local_bi = remap[atom_bond_index]
                valid = (local_bi[0] >= 0) & (local_bi[1] >= 0)
                local_bi = local_bi[:, valid]
                if local_bi.numel() > 0:
                    results['pred_bond'] = torch.nan_to_num(self.qcmol_bond_head(h_atom, local_bi))
                else:
                    results['pred_bond'] = torch.zeros(0, self.qcmol_bond_head.mlp[-1].out_features, device=device, dtype=dtype)
            else:
                results['pred_bond'] = torch.zeros(0, self.qcmol_bond_head.mlp[-1].out_features, device=device, dtype=dtype)

        if self.aux_atom_head is not None:
            results['pred_aux_atom'] = torch.nan_to_num(self.aux_atom_head(h_atom))
        else:
            results['pred_aux_atom'] = torch.zeros(0, 0, device=device, dtype=dtype)

        if self.aux_bond_head is not None and atom_bond_index.numel() > 0:
            if (node_type.max() == 0):
                results['pred_aux_bond'] = torch.nan_to_num(self.aux_bond_head(h_atom, atom_bond_index))
            else:
                N_global = node_type.size(0)
                is_atom = (node_type == 0)
                N_a = int(is_atom.sum())
                remap = torch.full((N_global,), -1, dtype=torch.long, device=device)
                remap[is_atom] = torch.arange(N_a, device=device)
                local_bi = remap[atom_bond_index]
                valid = (local_bi[0] >= 0) & (local_bi[1] >= 0)
                local_bi = local_bi[:, valid]
                if local_bi.numel() > 0:
                    results['pred_aux_bond'] = torch.nan_to_num(self.aux_bond_head(h_atom, local_bi))
                else:
                    results['pred_aux_bond'] = torch.zeros(0, self.aux_bond_head.mlp[-1].out_features, device=device, dtype=dtype)
        else:
            abd = self.aux_bond_head.mlp[-1].out_features if self.aux_bond_head is not None else 0
            results['pred_aux_bond'] = torch.zeros(0, abd, device=device, dtype=dtype)

        return results

    def load_simg_backbone(self, state_dict: dict, strict: bool = False):
        """Load weights from simg-trained NBOFoundationModel into backbone."""
        self.backbone.load_state_dict(state_dict, strict=strict)
