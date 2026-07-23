from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter as _scatter_sum

_logger = logging.getLogger(__name__)


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
_NBO_DIR = _PROJECT_DIR / "nbo_nets"
if str(_NBO_DIR) not in sys.path:
    sys.path.insert(0, str(_NBO_DIR))

from model_nbo_v2 import NBOFoundationModel, NBOFoundationModelQcMol


_RUNTIME_MODES = {"full", "detached", "cached"}

# Covalent radii (Å), Alvarez 2008 values for common organic/bio elements.
# Used to infer chemical bond topology from geometry when no data object is provided.
_COVALENT_RADII_ANG: Dict[int, float] = {
    1:  0.31,  # H
    5:  0.82,  # B
    6:  0.76,  # C
    7:  0.71,  # N
    8:  0.66,  # O
    9:  0.57,  # F
    14: 1.11,  # Si
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    34: 1.20,  # Se
    35: 1.20,  # Br
    53: 1.39,  # I
}
_COVALENT_RADII_DEFAULT = 0.80  # Å — fallback for elements not in the table


def _infer_chem_bonds(
    z: torch.Tensor,
    pos: torch.Tensor,
    tolerance: float = 0.40,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Infer covalent bond topology from atomic numbers and 3-D positions.

    Two atoms i, j are considered bonded when:
        dist(i, j)  <  r_cov[Z_i] + r_cov[Z_j] + tolerance

    When a batched atom tensor is provided, candidate bonds are restricted to
    atoms from the same molecule. This is essential for centered molecular
    batches, where unrelated atoms from different molecules can otherwise be
    geometrically close.

    Returns undirected bond index [2, E_bond] with row[k] < col[k] guaranteed.
    This replaces the radius-graph fallback for mode=qcmol so that the bond
    distribution seen at inference matches the covalent bond graphs used during
    qcMol NBO training.
    """
    device = pos.device
    n = int(z.size(0))
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    z_cpu = z.cpu().tolist()
    radii = [_COVALENT_RADII_ANG.get(int(zi), _COVALENT_RADII_DEFAULT) for zi in z_cpu]
    r = pos.new_tensor(radii)  # [N]

    # Pairwise threshold matrix  [N, N]
    thresh = r.unsqueeze(1) + r.unsqueeze(0) + tolerance

    # Pairwise distances  [N, N]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = diff.norm(dim=-1)

    # Bond: dist < thresh AND strictly positive (no self-loops)
    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)
    mask = (dist[i_idx, j_idx] < thresh[i_idx, j_idx]) & (dist[i_idx, j_idx] > 0.1)
    if isinstance(batch, torch.Tensor) and int(batch.numel()) == n:
        batch = batch.to(device=device).view(-1)
        mask = mask & (batch[i_idx] == batch[j_idx])
    if not mask.any():
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.stack([i_idx[mask], j_idx[mask]], dim=0).contiguous()


class ScalarRBFLayer(nn.Module):
    def __init__(self, start: float = -2.0, end: float = 5.0, num_bins: int = 16, gamma: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("centers", torch.linspace(start, end, num_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, int(self.centers.numel())))
        diff = x.reshape(-1, 1) - self.centers.view(1, -1).to(device=x.device, dtype=x.dtype)
        return torch.exp(-self.gamma * diff.square())


def _torch_load_compat(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sync_cuda_if_needed(device: torch.device) -> None:
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _edge_index_to_atom_bond(edge_index: Optional[torch.Tensor]) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    row, col = edge_index
    mask = row < col
    if not mask.any():
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.stack([row[mask], col[mask]], dim=0).contiguous()


def _match_undirected_edges(
    edge_index: torch.Tensor,
    atom_bond_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0 or atom_bond_index.numel() == 0:
        empty = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        return empty, mask

    row, col = edge_index
    edge_lo = torch.minimum(row, col)
    edge_hi = torch.maximum(row, col)
    edge_keys = edge_lo * num_nodes + edge_hi

    bond_row, bond_col = atom_bond_index
    bond_lo = torch.minimum(bond_row, bond_col)
    bond_hi = torch.maximum(bond_row, bond_col)
    bond_keys = bond_lo * num_nodes + bond_hi

    sorted_keys, perm = torch.sort(bond_keys)
    if sorted_keys.numel() == 0:
        empty = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        return empty, mask

    pos = torch.searchsorted(sorted_keys, edge_keys)
    mask = pos < sorted_keys.numel()
    safe_pos = pos.clamp(max=max(int(sorted_keys.numel()) - 1, 0))
    mask = mask & (sorted_keys[safe_pos] == edge_keys)

    bond_ids = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    if mask.any():
        bond_ids[mask] = perm[safe_pos[mask]]
    return bond_ids, mask


def _infer_stats_path(checkpoint_path: Path) -> Optional[Path]:
    stem = checkpoint_path.stem  # e.g. "nbo_v2_best" or "nbo_foundation_training_best"
    candidates = [
        checkpoint_path.with_name("norm_stats.pt"),
        checkpoint_path.with_name(f"{stem}_norm_stats.pt"),
    ]
    # Handle train_nbo_v2.py pattern: {model_name}_best.pt → {model_name}_norm_stats.pt
    # Strip common trailing tokens (_best, _last, _epochNNN) to recover base model name
    import re
    base = re.sub(r"(_best|_last|_epoch\d*)$", "", stem)
    if base and base != stem:
        candidates.append(checkpoint_path.with_name(f"{base}_norm_stats.pt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _unwrap_simg_state(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in raw_state:
        return raw_state["model_state_dict"]
    if "simg_model" in raw_state:
        return raw_state["simg_model"]
    return raw_state


def _unwrap_qcmol_state(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in raw_state:
        return raw_state["model_state_dict"]
    if "qcmol_model" in raw_state and raw_state["qcmol_model"] is not None:
        return raw_state["qcmol_model"]
    return raw_state


def _infer_qcmol_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    atom_dim = int(state_dict["qcmol_atom_head.mlp.2.weight"].shape[0])
    bond_dim = int(state_dict["qcmol_bond_head.mlp.2.weight"].shape[0])
    aux_atom_key = "aux_atom_head.mlp.2.weight"
    aux_bond_key = "aux_bond_head.mlp.2.weight"
    aux_atom_dim = int(state_dict[aux_atom_key].shape[0]) if aux_atom_key in state_dict else 0
    aux_bond_dim = int(state_dict[aux_bond_key].shape[0]) if aux_bond_key in state_dict else 0
    return atom_dim, bond_dim, aux_atom_dim, aux_bond_dim


def _infer_simg_backbone_dims(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    z_embed_key = "stem.z_embed.weight"
    atom_key = "atom_head.mlp.2.weight"
    bond_key = "bond_pair_head.mlp.2.weight"
    interaction_key = "interaction_head.out_mlp.2.weight"

    if z_embed_key not in state_dict:
        raise KeyError(f"Missing required key in NBO checkpoint: {z_embed_key}")
    if atom_key not in state_dict or bond_key not in state_dict or interaction_key not in state_dict:
        raise KeyError("Missing required head weights in NBO checkpoint")

    hidden_dim = int(state_dict[z_embed_key].shape[1])
    max_atomic_number = int(state_dict[z_embed_key].shape[0]) - 1
    atom_out_dim = int(state_dict[atom_key].shape[0])
    bond_out_dim = int(state_dict[bond_key].shape[0])
    interaction_out_dim = int(state_dict[interaction_key].shape[0])
    return {
        "hidden_dim": hidden_dim,
        "max_atomic_number": max_atomic_number,
        "atom_out_dim": atom_out_dim,
        "bond_out_dim": bond_out_dim,
        "interaction_out_dim": interaction_out_dim,
    }


def _infer_qcmol_backbone_dims(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise KeyError("Missing backbone.* keys in qcmol checkpoint")
    return _infer_simg_backbone_dims(backbone_state)


def _has_loaded_interaction_predictor(mode: str, predictor: nn.Module, missing_keys) -> bool:
    mode = str(mode).lower()
    if mode == "simg":
        if not hasattr(predictor, "interaction_head"):
            return False
        missing_prefix = "interaction_head."
    elif mode == "qcmol":
        backbone = getattr(predictor, "backbone", None)
        if backbone is None or not hasattr(backbone, "interaction_head"):
            return False
        missing_prefix = "backbone.interaction_head."
    else:
        return False
    return not any(str(key).startswith(missing_prefix) for key in missing_keys)


class NBOPriorBranch(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_radial: int,
        mode: str,
        checkpoint_path: str = "",
        stats_path: Optional[str] = None,
        hidden_dim: int = 128,
        max_atomic_number: int = 35,
        feature_scale: float = 1e-2,
        use_auxiliary: bool = True,
        freeze_predictor: bool = True,
        runtime_mode: str = "full",
        predictor_state_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.feature_scale = float(feature_scale)
        self.use_auxiliary = bool(use_auxiliary)
        self.freeze_predictor = bool(freeze_predictor)
        self.runtime_mode = str(runtime_mode).lower()
        if self.runtime_mode not in _RUNTIME_MODES:
            raise ValueError(f"Unsupported electron prior runtime_mode: {runtime_mode}")
        self.last_debug: Dict[str, float] = {}
        self._forward_profile_pending = True
        self._last_predictor_grad_enabled: Optional[bool] = None
        self._last_track_geometry_grad = False

        if predictor_state_dict is not None and not checkpoint_path:
            # Weights provided directly — extracted from main training checkpoint.
            # NBOFoundationModel architecture is inferred from these tensors directly,
            # final weights are loaded again when the outer model loads its checkpoint.
            raw_state = dict(predictor_state_dict)
        elif checkpoint_path:
            _ckpt = Path(checkpoint_path)
            if not _ckpt.exists():
                raise FileNotFoundError(f"Electron prior checkpoint not found: {_ckpt}")
            raw_state = _torch_load_compat(_ckpt, map_location="cpu")
        else:
            raise ValueError(
                "NBOPriorBranch: either checkpoint_path or predictor_state_dict must be provided"
            )

        if self.mode == "simg":
            model_state = _unwrap_simg_state(raw_state)
            predictor_dims = _infer_simg_backbone_dims(model_state)
            self.predictor = NBOFoundationModel(
                hidden_dim=predictor_dims["hidden_dim"],
                max_z=predictor_dims["max_atomic_number"],
                atom_out_dim=predictor_dims["atom_out_dim"],
                bond_out_dim=predictor_dims["bond_out_dim"],
                interaction_out_dim=predictor_dims["interaction_out_dim"],
            )
            load_msg = self.predictor.load_state_dict(model_state, strict=False)
            self.atom_dim = predictor_dims["atom_out_dim"]
            self.bond_dim = predictor_dims["bond_out_dim"]
            self.aux_atom_dim = 0
            self.aux_bond_dim = 0
        elif self.mode == "qcmol":
            model_state = _unwrap_qcmol_state(raw_state)
            predictor_dims = _infer_qcmol_backbone_dims(model_state)
            atom_dim, bond_dim, aux_atom_dim, aux_bond_dim = _infer_qcmol_dims(model_state)
            self.predictor = NBOFoundationModelQcMol(
                hidden_dim=predictor_dims["hidden_dim"],
                max_z=predictor_dims["max_atomic_number"],
                atom_out_dim=predictor_dims["atom_out_dim"],
                bond_out_dim=predictor_dims["bond_out_dim"],
                interaction_out_dim=predictor_dims["interaction_out_dim"],
                qcmol_atom_dim=atom_dim,
                qcmol_bond_dim=bond_dim,
                aux_atom_dim=aux_atom_dim,
                aux_bond_dim=aux_bond_dim,
            )
            load_msg = self.predictor.load_state_dict(model_state, strict=False)
            self.atom_dim = atom_dim
            self.bond_dim = bond_dim
            self.aux_atom_dim = aux_atom_dim
            self.aux_bond_dim = aux_bond_dim
        else:
            raise ValueError(f"Unsupported electron prior mode: {mode}")

        self.predictor_load_info = {
            "missing_keys": list(load_msg.missing_keys),
            "unexpected_keys": list(load_msg.unexpected_keys),
            "predictor_hidden_dim": int(predictor_dims["hidden_dim"]),
            "predictor_max_atomic_number": int(predictor_dims["max_atomic_number"]),
            "predictor_atom_dim": int(predictor_dims["atom_out_dim"]),
            "predictor_bond_dim": int(predictor_dims["bond_out_dim"]),
            "predictor_interaction_dim": int(predictor_dims["interaction_out_dim"]),
        }

        atom_input_dim = self.atom_dim + (self.aux_atom_dim if self.mode == "qcmol" and self.use_auxiliary else 0)
        bond_input_dim = self.bond_dim + (self.aux_bond_dim if self.mode == "qcmol" and self.use_auxiliary else 0)
        self.atom_input_dim = atom_input_dim
        self.bond_input_dim = bond_input_dim
        self.use_interaction_prior = _has_loaded_interaction_predictor(self.mode, self.predictor, load_msg.missing_keys)
        self.predictor_load_info["interaction_prior_enabled"] = bool(self.use_interaction_prior)

        self.atom_adapter = nn.Sequential(
            nn.LayerNorm(atom_input_dim),
            nn.Linear(atom_input_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        self.bond_adapter = nn.Sequential(
            nn.LayerNorm(bond_input_dim + num_radial),
            nn.Linear(bond_input_dim + num_radial, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        self.nonbond_adapter = nn.Sequential(
            nn.LayerNorm(atom_input_dim * 2 + num_radial),
            nn.Linear(atom_input_dim * 2 + num_radial, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        interaction_role_dim = 8
        interaction_energy_dim = 16
        self.interaction_role_embed = nn.Embedding(4, interaction_role_dim)
        self.interaction_energy_rbf = ScalarRBFLayer(start=-2.0, end=5.0, num_bins=interaction_energy_dim, gamma=1.0)
        interaction_base_dim = interaction_energy_dim + 2 + interaction_role_dim * 2
        self.interaction_atom_donor_adapter = nn.Sequential(
            nn.LayerNorm(atom_input_dim * 2 + interaction_base_dim),
            nn.Linear(atom_input_dim * 2 + interaction_base_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        self.interaction_atom_acceptor_adapter = nn.Sequential(
            nn.LayerNorm(atom_input_dim * 2 + interaction_base_dim),
            nn.Linear(atom_input_dim * 2 + interaction_base_dim, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        self.interaction_edge_adapter = nn.Sequential(
            nn.LayerNorm(atom_input_dim * 2 + interaction_base_dim + num_radial),
            nn.Linear(atom_input_dim * 2 + interaction_base_dim + num_radial, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
        self.atom_gate = nn.Parameter(torch.tensor(0.01))
        self.edge_gate = nn.Parameter(torch.tensor(0.01))
        self.interaction_atom_gate = nn.Parameter(torch.tensor(0.01))
        self.interaction_edge_gate = nn.Parameter(torch.tensor(0.01))

        self.register_buffer("atom_mean", torch.zeros(self.atom_dim))
        self.register_buffer("atom_std", torch.ones(self.atom_dim))
        self.register_buffer("bond_mean", torch.zeros(self.bond_dim))
        self.register_buffer("bond_std", torch.ones(self.bond_dim))

        if self.aux_atom_dim > 0:
            self.register_buffer("aux_atom_mean", torch.zeros(self.aux_atom_dim))
            self.register_buffer("aux_atom_std", torch.ones(self.aux_atom_dim))
        else:
            self.aux_atom_mean = None
            self.aux_atom_std = None
        if self.aux_bond_dim > 0:
            self.register_buffer("aux_bond_mean", torch.zeros(self.aux_bond_dim))
            self.register_buffer("aux_bond_std", torch.ones(self.aux_bond_dim))
        else:
            self.aux_bond_mean = None
            self.aux_bond_std = None

        self.stats_are_set = False
        # Priority: (1) explicit stats_path arg, (2) inferred file path, (3) stats embedded in bundle
        _ckpt_for_stats = Path(checkpoint_path) if checkpoint_path else None
        stats_file = Path(stats_path) if stats_path else (
            _infer_stats_path(_ckpt_for_stats) if _ckpt_for_stats else None
        )
        if stats_file is not None and stats_file.exists():
            self._load_stats(stats_file)
        if not self.stats_are_set and isinstance(raw_state, dict):
            # Joint training bundle stores stats under different keys depending on predictor mode.
            stats_keys = ["norm_stats"]
            if self.mode == "qcmol":
                stats_keys = ["qcmol_stats", "norm_stats"]
            elif self.mode == "simg":
                stats_keys = ["simg_norm_stats", "norm_stats"]
            for key in stats_keys:
                embedded = raw_state.get(key)
                if embedded and isinstance(embedded, dict) and "atom_mean" in embedded:
                    self._load_stats_from_dict(embedded)
                    break

        if self.freeze_predictor:
            for parameter in self.predictor.parameters():
                parameter.requires_grad_(False)

    def set_runtime_mode(self, runtime_mode: str) -> None:
        runtime_mode = str(runtime_mode).lower()
        if runtime_mode not in _RUNTIME_MODES:
            raise ValueError(f"Unsupported electron prior runtime_mode: {runtime_mode}")
        self.runtime_mode = runtime_mode

    def _has_nbo_topology(self, data) -> bool:
        """Return True only when data contains real NBO topology (virtual orbital nodes).

        Real NBO data must have:
          - data.node_type: tensor with some non-zero entries (BD/LP orbital nodes)
          - data.atom_to_nbo_index: non-empty atom→NBO mapping

        When False, the predictor would fall back to a geometry-only atom graph,
        which is out-of-distribution relative to training and produces misleading features.
        """
        if data is None:
            return False
        node_type = getattr(data, "node_type", None)
        if not isinstance(node_type, torch.Tensor):
            return False
        atom_to_nbo = getattr(data, "atom_to_nbo_index", None)
        return isinstance(atom_to_nbo, torch.Tensor) and atom_to_nbo.numel() > 0

    def _prepare_predictor_features(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        atom_pred = self._denorm(outputs["pred_atom"], self.atom_mean, self.atom_std)
        bond_pred = self._denorm(outputs["pred_bond"], self.bond_mean, self.bond_std)
        interaction_pred = outputs.get("pred_interaction")

        if self.mode == "qcmol" and self.use_auxiliary:
            aux_atom = outputs.get("pred_aux_atom")
            aux_bond = outputs.get("pred_aux_bond")
            if isinstance(aux_atom, torch.Tensor) and aux_atom.numel() > 0 and isinstance(self.aux_atom_mean, torch.Tensor):
                aux_atom = self._denorm(aux_atom, self.aux_atom_mean, self.aux_atom_std)
                atom_pred = torch.cat([atom_pred, aux_atom], dim=-1)
            if isinstance(aux_bond, torch.Tensor) and aux_bond.numel() > 0 and isinstance(self.aux_bond_mean, torch.Tensor):
                aux_bond = self._denorm(aux_bond, self.aux_bond_mean, self.aux_bond_std)
                bond_pred = torch.cat([bond_pred, aux_bond], dim=-1)
        return atom_pred, bond_pred, interaction_pred

    def export_cache_tensors(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        data=None,
    ) -> Dict[str, torch.Tensor]:
        self.predictor.eval()
        nbo_input, _ = self._prepare_nbo_inputs(z, pos, edge_index, data=data)
        with torch.no_grad():
            outputs = self.predictor(nbo_input)
            atom_pred, bond_pred, interaction_pred = self._prepare_predictor_features(outputs)
        interaction_tensor = interaction_pred if isinstance(interaction_pred, torch.Tensor) else bond_pred.new_zeros((0, 0))
        return {
            "ep_atom_pred": atom_pred.detach().cpu(),
            "ep_bond_pred": bond_pred.detach().cpu(),
            "ep_interaction_pred": interaction_tensor.detach().cpu(),
        }

    def _load_cached_predictor_features(self, data, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        if data is None:
            return None
        atom_pred = getattr(data, "ep_atom_pred", None)
        bond_pred = getattr(data, "ep_bond_pred", None)
        interaction_pred = getattr(data, "ep_interaction_pred", None)
        if not isinstance(atom_pred, torch.Tensor) or not isinstance(bond_pred, torch.Tensor):
            return None
        atom_pred = atom_pred.to(device=device)
        bond_pred = bond_pred.to(device=device)
        if isinstance(interaction_pred, torch.Tensor):
            interaction_pred = interaction_pred.to(device=device)
        else:
            interaction_pred = None
        return atom_pred, bond_pred, interaction_pred

    def _load_stats(self, stats_path: Path) -> None:
        stats = _torch_load_compat(stats_path, map_location="cpu")
        self._load_stats_from_dict(stats)

    def _load_stats_from_dict(self, stats: dict) -> None:
        # Validate shape before copying: stats files may not match the predictor mode
        # (for example, qcmol stats next to a simg prior checkpoint, or vice versa).
        # If sizes don't match, skip this stats source so the caller can fall through
        # to the bundle-embedded stats for the active mode.
        am = stats.get("atom_mean")
        if isinstance(am, torch.Tensor) and am.numel() != self.atom_mean.numel():
            import warnings
            warnings.warn(
                f"NBOPriorBranch: stats 'atom_mean' has {am.numel()} element(s) "
                f"but predictor expects {self.atom_mean.numel()} (mode={self.mode}). "
                "Skipping this stats source and falling back to mode-matched embedded stats.",
                RuntimeWarning, stacklevel=2,
            )
            return  # do NOT set stats_are_set → fallback to embedded bundle stats

        if "atom_mean" in stats and "atom_std" in stats:
            self.atom_mean.copy_(stats["atom_mean"].float().view_as(self.atom_mean))
            self.atom_std.copy_(stats["atom_std"].float().view_as(self.atom_std))
        if "bond_mean" in stats and "bond_std" in stats:
            self.bond_mean.copy_(stats["bond_mean"].float().view_as(self.bond_mean))
            self.bond_std.copy_(stats["bond_std"].float().view_as(self.bond_std))
        if self.aux_atom_dim > 0 and isinstance(self.aux_atom_mean, torch.Tensor):
            if "aux_atom_mean" in stats and "aux_atom_std" in stats:
                self.aux_atom_mean.copy_(stats["aux_atom_mean"].float().view_as(self.aux_atom_mean))
                self.aux_atom_std.copy_(stats["aux_atom_std"].float().view_as(self.aux_atom_std))
        if self.aux_bond_dim > 0 and isinstance(self.aux_bond_mean, torch.Tensor):
            if "aux_bond_mean" in stats and "aux_bond_std" in stats:
                self.aux_bond_mean.copy_(stats["aux_bond_mean"].float().view_as(self.aux_bond_mean))
                self.aux_bond_std.copy_(stats["aux_bond_std"].float().view_as(self.aux_bond_std))
        self.stats_are_set = True

    def _denorm(self, pred: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
        if pred.numel() == 0:
            return pred
        pred = torch.nan_to_num(pred)
        if self.stats_are_set and isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            pred = pred * std.to(pred.device) + mean.to(pred.device)
        else:
            pred = pred * self.feature_scale
        return torch.nan_to_num(pred).clamp(-10, 10)

    def _prepare_nbo_inputs(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        data=None,
    ) -> Tuple[SimpleNamespace, torch.Tensor]:
        device = pos.device
        num_atoms = int(pos.size(0))

        if data is not None:
            node_type = getattr(data, "node_type", None)
            pos_global = getattr(data, "pos_global", None)
            atom_to_nbo_index = getattr(data, "atom_to_nbo_index", None)
            atom_bond_global = getattr(data, "atom_bond_index", None)
            interaction_edge_index = getattr(data, "interaction_edge_index", None)
        else:
            node_type = None
            pos_global = None
            atom_to_nbo_index = None
            atom_bond_global = None
            interaction_edge_index = None

        if not isinstance(node_type, torch.Tensor) or not isinstance(pos_global, torch.Tensor) or int((node_type == 0).sum().item()) != num_atoms:
            node_type = torch.zeros(num_atoms, dtype=torch.long, device=device)
            pos_global = pos
            atom_to_nbo_index = torch.empty((2, 0), dtype=torch.long, device=device)
            interaction_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            # qcmol was trained on covalent bond graphs (RDKit/QM BeginAtomIdx/EndAtomIdx).
            # Using the radius graph (~10x more edges) would cause severe distribution shift.
            # Infer chemical bonds via covalent radii thresholds instead.
            if self.mode == "qcmol":
                atom_batch = getattr(data, "batch", None) if data is not None else None
                atom_bond_global = _infer_chem_bonds(z, pos, batch=atom_batch).to(device)
            else:
                atom_bond_global = _edge_index_to_atom_bond(edge_index).to(device)
            atom_bond_local = atom_bond_global
            global_atom_idx = torch.arange(num_atoms, device=device)
        else:
            node_type = node_type.to(device=device, dtype=torch.long)
            pos_global = pos_global.to(device=device, dtype=pos.dtype)
            atom_to_nbo_index = atom_to_nbo_index if isinstance(atom_to_nbo_index, torch.Tensor) else torch.empty((2, 0), dtype=torch.long, device=device)
            atom_to_nbo_index = atom_to_nbo_index.to(device=device, dtype=torch.long)
            interaction_edge_index = interaction_edge_index if isinstance(interaction_edge_index, torch.Tensor) else torch.empty((2, 0), dtype=torch.long, device=device)
            interaction_edge_index = interaction_edge_index.to(device=device, dtype=torch.long)
            atom_bond_global = atom_bond_global if isinstance(atom_bond_global, torch.Tensor) else _edge_index_to_atom_bond(edge_index).to(device)
            atom_bond_global = atom_bond_global.to(device=device, dtype=torch.long)

            global_atom_idx = torch.nonzero(node_type == 0, as_tuple=False).view(-1)
            remap_global_to_local = torch.full((node_type.size(0),), -1, dtype=torch.long, device=device)
            remap_global_to_local[global_atom_idx] = torch.arange(num_atoms, device=device)
            if atom_bond_global.numel() > 0:
                atom_bond_local = remap_global_to_local[atom_bond_global]
                valid = (atom_bond_local[0] >= 0) & (atom_bond_local[1] >= 0)
                atom_bond_local = atom_bond_local[:, valid]
            else:
                atom_bond_local = torch.empty((2, 0), dtype=torch.long, device=device)

        x_global = torch.zeros((node_type.size(0), 1), dtype=pos.dtype, device=device)
        x_global[global_atom_idx, 0] = z.to(pos.dtype)
        nbo_input = SimpleNamespace(
            x=x_global,
            pos=pos_global,
            node_type=node_type,
            atom_bond_index=atom_bond_global,
            interaction_edge_index=interaction_edge_index,
            atom_to_nbo_index=atom_to_nbo_index,
        )
        return nbo_input, atom_bond_local

    def _run_predictor(self, nbo_input: SimpleNamespace, track_geometry_grad: bool) -> Dict[str, torch.Tensor]:
        self.predictor.eval()
        self._last_track_geometry_grad = bool(track_geometry_grad)
        if self.runtime_mode == "detached":
            self._last_predictor_grad_enabled = False
            with torch.no_grad():
                return self.predictor(nbo_input)
        if self.freeze_predictor and not track_geometry_grad:
            self._last_predictor_grad_enabled = False
            with torch.no_grad():
                return self.predictor(nbo_input)
        self._last_predictor_grad_enabled = True
        return self.predictor(nbo_input)

    def _apply_interaction_prior(
        self,
        atom_prior: torch.Tensor,
        edge_prior: torch.Tensor,
        atom_pred: torch.Tensor,
        interaction_pred: Optional[torch.Tensor],
        nbo_input: SimpleNamespace,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if not self.use_interaction_prior:
            return atom_prior, edge_prior, {
                "interaction_edges_total": 0.0,
                "interaction_edges_mapped": 0.0,
                "interaction_pairs_matched": 0.0,
            }
        if interaction_pred is None or interaction_pred.numel() == 0:
            return atom_prior, edge_prior, {
                "interaction_edges_total": 0.0,
                "interaction_edges_mapped": 0.0,
                "interaction_pairs_matched": 0.0,
            }

        interaction_edge_index = nbo_input.interaction_edge_index
        atom_to_nbo_index = nbo_input.atom_to_nbo_index
        node_type = nbo_input.node_type
        if interaction_edge_index.numel() == 0 or atom_pred.numel() == 0:
            return atom_prior, edge_prior, {
                "interaction_edges_total": float(interaction_edge_index.size(1)),
                "interaction_edges_mapped": 0.0,
                "interaction_pairs_matched": 0.0,
            }

        device = atom_prior.device
        num_atoms = int(atom_pred.size(0))
        num_edges = int(edge_index.size(1))
        num_nodes = int(node_type.size(0))
        E_inter = int(interaction_edge_index.size(1))
        MAX_P = 4  # safe upper bound for parents per NBO node

        # --- Build parent mapping as padded tensors [num_nodes, MAX_P] ---
        parent_atoms = torch.full((num_nodes, MAX_P), -1, dtype=torch.long, device=device)
        parent_count = torch.zeros(num_nodes, dtype=torch.long, device=device)
        atom_mask_bool = (node_type == 0)
        atom_global_idx = torch.nonzero(atom_mask_bool, as_tuple=False).view(-1)
        remap = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        remap[atom_global_idx] = torch.arange(num_atoms, device=device)

        # Atom nodes: parent = self
        parent_atoms[atom_global_idx, 0] = remap[atom_global_idx]
        parent_count[atom_global_idx] = 1

        # NBO nodes: parents from atom_to_nbo_index (small loop, ~1000 edges, <1ms)
        if atom_to_nbo_index.numel() > 0:
            a2n_src, a2n_tgt = atom_to_nbo_index
            valid_a2n = (
                (a2n_src >= 0) & (a2n_src < num_nodes)
                & (a2n_tgt >= 0) & (a2n_tgt < num_nodes)
                & atom_mask_bool[a2n_src]
            )
            a2n_local = remap[a2n_src]
            valid_a2n = valid_a2n & (a2n_local >= 0)
            if valid_a2n.any():
                v_local = a2n_local[valid_a2n].tolist()
                v_tgt = a2n_tgt[valid_a2n].tolist()
                pa_cpu = parent_atoms.cpu()
                pc_cpu = parent_count.cpu()
                for s, t in zip(v_local, v_tgt):
                    c = pc_cpu[t].item()
                    if c >= MAX_P:
                        continue
                    if (pa_cpu[t, :c] == s).any():
                        continue
                    pa_cpu[t, c] = s
                    pc_cpu[t] = c + 1
                parent_atoms = pa_cpu.to(device)
                parent_count = pc_cpu.to(device)

        # Weights = 1 / parent_count (uniform, matching original NBO semantics)
        parent_weights = torch.zeros((num_nodes, MAX_P), dtype=torch.float32, device=device)
        valid_parent_mask = parent_atoms >= 0
        nz_count = parent_count.clamp(min=1).float().unsqueeze(1)
        parent_weights = valid_parent_mask.float() / nz_count

        # --- Feature preparation (on GPU, gradients tracked through rbf/embed) ---
        interaction_pred_f = torch.nan_to_num(interaction_pred.float()).clamp(-10, 10)
        if interaction_pred_f.size(-1) < 3:
            interaction_pred_f = torch.cat([
                interaction_pred_f,
                interaction_pred_f.new_zeros(E_inter, 3 - interaction_pred_f.size(-1)),
            ], dim=-1)

        energy_feat = self.interaction_energy_rbf(interaction_pred_f[:, 0])   # [E, rbf_bins]
        other_feat = interaction_pred_f[:, 1:3]                               # [E, 2]
        role_feat = self.interaction_role_embed(node_type.clamp(min=0, max=3)).float()  # [N, embed_dim]

        # --- Cross-product expansion: interaction edges -> atom-atom pairs ---
        src_nodes_ie = interaction_edge_index[0]  # [E]
        dst_nodes_ie = interaction_edge_index[1]  # [E]

        # Filter to edges whose *both* endpoints have ≥1 parent atom
        src_has = parent_count[src_nodes_ie] > 0
        dst_has = parent_count[dst_nodes_ie] > 0
        mapped_mask = src_has & dst_has
        interaction_edges_mapped = int(mapped_mask.sum().item())

        if interaction_edges_mapped == 0:
            return atom_prior, edge_prior, {
                "interaction_edges_total": float(E_inter),
                "interaction_edges_mapped": 0.0,
                "interaction_pairs_matched": 0.0,
            }

        m_idx = torch.nonzero(mapped_mask, as_tuple=False).view(-1)
        m_src = src_nodes_ie[m_idx]
        m_dst = dst_nodes_ie[m_idx]
        M = m_idx.size(0)

        # Gather parents [M, MAX_P]
        sp = parent_atoms[m_src];  sw = parent_weights[m_src]
        dp = parent_atoms[m_dst];  dw = parent_weights[m_dst]

        # Cross-product [M, MAX_P, MAX_P] → flatten [M * MAX_P^2]
        sp_exp = sp.unsqueeze(2).expand(-1, -1, MAX_P).reshape(-1)
        sw_exp = sw.unsqueeze(2).expand(-1, -1, MAX_P).reshape(-1)
        dp_exp = dp.unsqueeze(1).expand(-1, MAX_P, -1).reshape(-1)
        dw_exp = dw.unsqueeze(1).expand(-1, MAX_P, -1).reshape(-1)
        pair_w = sw_exp * dw_exp
        pair_valid = (sp_exp >= 0) & (dp_exp >= 0) & (pair_w > 0)

        if not pair_valid.any():
            return atom_prior, edge_prior, {
                "interaction_edges_total": float(E_inter),
                "interaction_edges_mapped": float(interaction_edges_mapped),
                "interaction_pairs_matched": 0.0,
            }

        # Expand corresponding IDs the same way
        eid_exp = m_idx.unsqueeze(1).unsqueeze(2).expand(-1, MAX_P, MAX_P).reshape(-1)
        sn_exp = m_src.unsqueeze(1).unsqueeze(2).expand(-1, MAX_P, MAX_P).reshape(-1)
        dn_exp = m_dst.unsqueeze(1).unsqueeze(2).expand(-1, MAX_P, MAX_P).reshape(-1)

        # Select valid pairs only
        v_sa  = sp_exp[pair_valid]   # src atom (local)
        v_da  = dp_exp[pair_valid]   # dst atom (local)
        v_pw  = pair_w[pair_valid]   # pair weight
        v_eid = eid_exp[pair_valid]  # interaction edge id (into energy_feat / other_feat)
        v_sn  = sn_exp[pair_valid]   # src global node (into role_feat)
        v_dn  = dn_exp[pair_valid]   # dst global node (into role_feat)

        # --- Build features via index gather (vectorized, on GPU, with grad) ---
        base_feat = torch.cat([
            energy_feat[v_eid],
            other_feat[v_eid],
            role_feat[v_sn],
            role_feat[v_dn],
        ], dim=-1)

        atom_pred_f = atom_pred.float()
        donor_feat = torch.cat([atom_pred_f[v_sa], atom_pred_f[v_da], base_feat], dim=-1)
        acceptor_feat = torch.cat([atom_pred_f[v_da], atom_pred_f[v_sa], base_feat], dim=-1)

        # --- Run adapters (single batched MLP forward each) ---
        donor_delta = self.interaction_atom_donor_adapter(donor_feat)
        acceptor_delta = self.interaction_atom_acceptor_adapter(acceptor_feat)

        # --- Differentiable weighted scatter (torch_scatter guarantees grad flow) ---
        w_col = v_pw.unsqueeze(-1)
        feat_dim = atom_prior.size(-1)

        all_atom_delta = torch.cat([donor_delta * w_col, acceptor_delta * w_col], dim=0)
        all_atom_idx = torch.cat([v_sa, v_da], dim=0)
        all_atom_w = torch.cat([w_col, w_col], dim=0)
        atom_update = _scatter_sum(all_atom_delta, all_atom_idx, dim=0, dim_size=num_atoms)
        atom_count = _scatter_sum(all_atom_w, all_atom_idx, dim=0, dim_size=num_atoms)

        # --- Edge matching (vectorized searchsorted) ---
        directed_keys = edge_index[0] * num_atoms + edge_index[1]
        sorted_keys, key_perm = torch.sort(directed_keys)
        pair_keys = v_sa * num_atoms + v_da

        search_pos = torch.searchsorted(sorted_keys, pair_keys)
        sp_clamped = search_pos.clamp(max=max(int(sorted_keys.size(0)) - 1, 0))
        edge_matched = (search_pos < sorted_keys.size(0)) & (sorted_keys[sp_clamped] == pair_keys)
        interaction_pairs_matched = int(edge_matched.sum().item())

        if edge_matched.any():
            matched_pair_idx = torch.nonzero(edge_matched, as_tuple=False).view(-1)
            matched_edge_idx = key_perm[sp_clamped[matched_pair_idx]]

            edge_feat = torch.cat([
                atom_pred_f[v_sa[matched_pair_idx]],
                atom_pred_f[v_da[matched_pair_idx]],
                base_feat[matched_pair_idx],
                rbf[matched_edge_idx].float(),
            ], dim=-1)

            edge_delta = self.interaction_edge_adapter(edge_feat)
            matched_w = w_col[matched_pair_idx]

            edge_update = _scatter_sum(edge_delta * matched_w, matched_edge_idx, dim=0, dim_size=num_edges)
            edge_count = _scatter_sum(matched_w, matched_edge_idx, dim=0, dim_size=num_edges)
        else:
            edge_update = torch.zeros(num_edges, feat_dim, device=device, dtype=torch.float32)
            edge_count = torch.zeros(num_edges, 1, device=device, dtype=torch.float32)

        # --- Normalize and gate ---
        a_mask = atom_count.squeeze(-1) > 0
        if a_mask.any():
            atom_update[a_mask] = atom_update[a_mask] / atom_count[a_mask].clamp(min=1e-6)
            atom_prior = atom_prior + atom_update.to(atom_prior.dtype) * torch.tanh(self.interaction_atom_gate)

        e_mask = edge_count.squeeze(-1) > 0
        if e_mask.any():
            edge_update[e_mask] = edge_update[e_mask] / edge_count[e_mask].clamp(min=1e-6)
            edge_prior = edge_prior + edge_update.to(edge_prior.dtype) * torch.tanh(self.interaction_edge_gate)

        return atom_prior, edge_prior, {
            "interaction_edges_total": float(E_inter),
            "interaction_edges_mapped": float(interaction_edges_mapped),
            "interaction_pairs_matched": float(interaction_pairs_matched),
        }

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        data=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        profile_this_forward = bool(self._forward_profile_pending)
        if profile_this_forward:
            start_time = time.perf_counter()
            _logger.info(
                "[electron_prior] first forward start mode=%s atoms=%d radius_edges=%d",
                self.mode, int(z.numel()), int(edge_index.size(1)),
            )

        # --- pure-geometry inference notice ---
        # When data=None, _prepare_nbo_inputs falls back to the atom-bond graph (no NBO virtual node).
        # NBOFoundationModel can still predict NBO features end-to-end from geometry. This is intended behavior.
        # A one-time diagnostic is printed on the first forward pass; subsequent calls are silent.
        if profile_this_forward and not self._has_nbo_topology(data):
            _logger.info(
                "[electron_prior] inference mode: data=None, no NBO virtual-node graph. "
                "Predictor runs end-to-end on the atomic geometry graph (atom types + bond lengths -> NBO features).",
            )
        # ---
        prep_t0 = time.perf_counter() if profile_this_forward else 0.0
        nbo_input, atom_bond_local = self._prepare_nbo_inputs(z, pos, edge_index, data=data)
        if profile_this_forward:
            prep_t1 = time.perf_counter()
            _logger.info(
                "[electron_prior] prepare inputs done global_nodes=%d atom_bonds=%d "
                "interaction_edges=%d dt=%.2fs",
                int(nbo_input.node_type.size(0)),
                int(nbo_input.atom_bond_index.size(1)),
                int(nbo_input.interaction_edge_index.size(1)),
                prep_t1 - prep_t0,
            )

        pred_t0 = time.perf_counter() if profile_this_forward else 0.0
        cached_features = self._load_cached_predictor_features(data, pos.device) if self.runtime_mode == "cached" else None
        if self.runtime_mode == "cached":
            if cached_features is None:
                raise RuntimeError(
                    "electron prior runtime_mode=cached but batch has no ep_* cached tensors. "
                    "Please precompute cache first or switch to detached/full mode."
                )
            atom_pred, bond_pred, interaction_pred = cached_features
            if profile_this_forward:
                _sync_cuda_if_needed(pos.device)
                pred_t1 = time.perf_counter()
                inter_shape = tuple(interaction_pred.shape) if isinstance(interaction_pred, torch.Tensor) else (0,)
                _logger.info(
                    "[electron_prior] predictor cache hit atom=%s bond=%s "
                    "pred_interaction=%s dt=%.2fs",
                    tuple(atom_pred.shape), tuple(bond_pred.shape), inter_shape,
                    pred_t1 - pred_t0,
                )
        else:
            outputs = self._run_predictor(nbo_input, track_geometry_grad=(self.runtime_mode == "full" and bool(pos.requires_grad)))
            atom_pred, bond_pred, interaction_pred = self._prepare_predictor_features(outputs)
            if self.runtime_mode == "detached":
                atom_pred = atom_pred.detach()
                bond_pred = bond_pred.detach()
                if isinstance(interaction_pred, torch.Tensor):
                    interaction_pred = interaction_pred.detach()
            if profile_this_forward:
                _sync_cuda_if_needed(pos.device)
                pred_t1 = time.perf_counter()
                _logger.info(
                    "[electron_prior] predictor done pred_atom=%s pred_bond=%s "
                    "pred_interaction=%s predictor_grad=%s geom_grad=%s "
                    "atom_req_grad=%s dt=%.2fs mode=%s",
                    tuple(atom_pred.shape), tuple(bond_pred.shape),
                    tuple(interaction_pred.shape) if isinstance(interaction_pred, torch.Tensor) else (0,),
                    self._last_predictor_grad_enabled,
                    self._last_track_geometry_grad,
                    bool(atom_pred.requires_grad),
                    pred_t1 - pred_t0, self.runtime_mode,
                )

        adapt_t0 = time.perf_counter() if profile_this_forward else 0.0

        atom_prior = self.atom_adapter(atom_pred.float()) * torch.tanh(self.atom_gate)

        num_edges = int(edge_index.size(1))
        edge_prior = atom_prior.new_zeros((num_edges, atom_prior.size(-1)))
        bond_ids, bond_mask = _match_undirected_edges(edge_index, atom_bond_local, num_nodes=pos.size(0))

        if bond_mask.any() and bond_pred.numel() > 0:
            bond_inputs = torch.cat([bond_pred[bond_ids[bond_mask]], rbf[bond_mask].float()], dim=-1)
            edge_prior[bond_mask] = self.bond_adapter(bond_inputs)

        nonbond_mask = ~bond_mask
        if nonbond_mask.any():
            row, col = edge_index
            nonbond_inputs = torch.cat(
                [
                    atom_pred[row[nonbond_mask]].float(),
                    atom_pred[col[nonbond_mask]].float(),
                    rbf[nonbond_mask].float(),
                ],
                dim=-1,
            )
            edge_prior[nonbond_mask] = self.nonbond_adapter(nonbond_inputs)

        edge_prior = edge_prior * torch.tanh(self.edge_gate)

        interaction_t0 = time.perf_counter() if profile_this_forward else 0.0
        atom_prior, edge_prior, interaction_debug = self._apply_interaction_prior(
            atom_prior=atom_prior,
            edge_prior=edge_prior,
            atom_pred=atom_pred,
            interaction_pred=interaction_pred,
            nbo_input=nbo_input,
            edge_index=edge_index,
            rbf=rbf,
        )
        if profile_this_forward:
            _sync_cuda_if_needed(pos.device)
            end_time = time.perf_counter()
            interaction_dt = end_time - interaction_t0
            adapt_dt = interaction_t0 - adapt_t0
            _logger.info(
                "[electron_prior] adapt done dt=%.2fs interaction_dt=%.2fs "
                "mapped=%d/%d matched=%d total=%.2fs",
                adapt_dt, interaction_dt,
                int(interaction_debug['interaction_edges_mapped']),
                int(interaction_debug['interaction_edges_total']),
                int(interaction_debug['interaction_pairs_matched']),
                end_time - start_time,
            )
            self._forward_profile_pending = False

        self.last_debug = {
            "atom_gate": float(torch.tanh(self.atom_gate).detach().cpu().item()),
            "edge_gate": float(torch.tanh(self.edge_gate).detach().cpu().item()),
            "interaction_prior_enabled": 1.0 if self.use_interaction_prior else 0.0,
            "interaction_atom_gate": float(torch.tanh(self.interaction_atom_gate).detach().cpu().item()),
            "interaction_edge_gate": float(torch.tanh(self.interaction_edge_gate).detach().cpu().item()),
            "bond_edge_ratio": float(bond_mask.float().mean().detach().cpu().item()) if num_edges > 0 else 0.0,
            "interaction_edges_total": float(interaction_debug["interaction_edges_total"]),
            "interaction_edges_mapped": float(interaction_debug["interaction_edges_mapped"]),
            "interaction_pairs_matched": float(interaction_debug["interaction_pairs_matched"]),
        }
        self._last_nbo_predictions = {
            "atom_pred": atom_pred.detach().cpu(),
            "bond_pred": bond_pred.detach().cpu() if isinstance(bond_pred, torch.Tensor) else None,
            "interaction_pred": interaction_pred.detach().cpu() if isinstance(interaction_pred, torch.Tensor) else None,
            "edge_index": edge_index.detach().cpu(),
            "atom_bond_local": atom_bond_local.detach().cpu() if isinstance(atom_bond_local, torch.Tensor) else None,
            "bond_ids": bond_ids.detach().cpu() if isinstance(bond_ids, torch.Tensor) else None,
            "bond_mask": bond_mask.detach().cpu() if isinstance(bond_mask, torch.Tensor) else None,
            "interaction_edge_index": nbo_input.interaction_edge_index.detach().cpu() if hasattr(nbo_input, "interaction_edge_index") else None,
            "node_type": nbo_input.node_type.detach().cpu() if hasattr(nbo_input, "node_type") else None,
            "atom_to_nbo_index": nbo_input.atom_to_nbo_index.detach().cpu() if hasattr(nbo_input, "atom_to_nbo_index") else None,
            "pos_global": nbo_input.pos.detach().cpu() if hasattr(nbo_input, "pos") else None,
        }
        return atom_prior.to(pos.dtype), edge_prior.to(pos.dtype)