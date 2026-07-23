#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training Pipeline 1: simg-only NBO pretraining with NBOFoundationModel v2.

Usage:
    python train_nbo_v2.py --dataset_root /path/to/datasets --gpu 0

Changes from train_nbo.py:
  - Uses NBOFoundationModel (EquivariantTokenMP backbone)
  - Composite validation score (not single metric)
  - Per-task metric logging
  - Compatible checkpoint format for Pipeline 2 loading
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
except ImportError:
    accuracy_score = precision_score = recall_score = None

# Local imports
from model_nbo_v2 import NBOFoundationModel
from nbo_loader import get_nbo_dataloaders


# ===================================================================
# Composite validation score
# ===================================================================

def composite_val_score(maes: dict, link_metrics: dict) -> float:
    """Weighted composite: lower is better.

    Components:
      - charge MAE (atom)  × 0.2
      - bond MAE          × 0.2
      - interaction E2 MAE × 0.4
      - (1 - link_acc)    × 0.2
    """
    charge = maes.get('charge', 0.0)
    bond = maes.get('bond_props', 0.0)
    inter = maes.get('interaction_E2', 0.0)
    link_err = 1.0 - link_metrics.get('acc', 0.0)
    return 0.2 * charge + 0.2 * bond + 0.4 * inter + 0.2 * link_err


# ===================================================================
# Training loop
# ===================================================================

def train_epoch(
    model, loader, optimizer, criterions, loss_weights, norm_stats, device,
    use_amp=False, amp_dtype='bf16', scaler=None, max_grad_norm=1.0,
):
    model.train()
    total_loss = 0.0
    total_prop_loss = 0.0
    total_link_loss = 0.0

    crit_atom, crit_bond, crit_interaction, crit_link = criterions
    w_atom, w_bond, w_interaction, w_link = loss_weights

    atom_mean = norm_stats['atom_mean'].to(device)
    atom_std = norm_stats['atom_std'].to(device)
    bond_mean = norm_stats['bond_mean'].to(device)
    bond_std = norm_stats['bond_std'].to(device)

    autocast_ctx = contextlib.nullcontext()
    if use_amp and torch.cuda.is_available():
        dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
        autocast_ctx = torch.amp.autocast('cuda', dtype=dtype)

    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Normalize targets
        data.y_atomic_props = (data.y_atomic_props - atom_mean) / atom_std
        if data.y_bond_props.numel() > 0:
            data.y_bond_props = (data.y_bond_props - bond_mean) / bond_std

        with autocast_ctx:
            out = model(data, link_prediction_candidates=data.link_prediction_candidates)

        pred_atomic = torch.nan_to_num(out['pred_atom'])
        pred_bond = torch.nan_to_num(out['pred_bond'])
        pred_interaction = torch.nan_to_num(out['pred_interaction'])
        pred_logits = torch.nan_to_num(out['pred_link_logits'])

        loss_atom = crit_atom(pred_atomic, data.y_atomic_props)

        loss_bond = torch.tensor(0.0, device=device)
        if data.atom_bond_index.numel() > 0 and pred_bond.numel() > 0:
            mask = getattr(data, 'bond_supervised_mask', None)
            if mask is not None and mask.numel() == pred_bond.shape[0] and mask.any():
                loss_bond = crit_bond(pred_bond[mask], data.y_bond_props[mask])
            elif data.y_bond_props.numel() > 0 and data.y_bond_props.shape[0] == pred_bond.shape[0]:
                loss_bond = crit_bond(pred_bond, data.y_bond_props)

        loss_interaction = torch.tensor(0.0, device=device)
        if data.interaction_edge_index.numel() > 0 and pred_interaction.numel() > 0:
            loss_interaction = crit_interaction(pred_interaction, data.y_interaction_props)

        prop_loss = w_atom * loss_atom + w_bond * loss_bond + w_interaction * loss_interaction

        link_loss = torch.tensor(0.0, device=device)
        if pred_logits.numel() > 0:
            link_loss = crit_link(pred_logits, data.link_prediction_labels)

        loss = prop_loss + w_link * link_loss

        if not torch.isfinite(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_prop_loss += prop_loss.item() * data.num_graphs
        total_link_loss += link_loss.item() * data.num_graphs

    N = max(1, len(loader.dataset))
    return total_loss / N, {'prop': total_prop_loss / N, 'link': total_link_loss / N}


# ===================================================================
# Validation loop
# ===================================================================

def validate_epoch(model, loader, norm_stats, device):
    model.eval()
    prop_errors = {'charge': [], 'bond_props': [], 'interaction_props': []}
    link_preds, link_trues = [], []
    all_probs = []

    atom_mean = norm_stats['atom_mean'].to(device)
    atom_std = norm_stats['atom_std'].to(device)
    bond_mean = norm_stats['bond_mean'].to(device)
    bond_std = norm_stats['bond_std'].to(device)

    with torch.no_grad():
        for data in tqdm(loader, desc="Validating"):
            data = data.to(device)
            data.y_atomic_props = (data.y_atomic_props - atom_mean) / atom_std
            if data.y_bond_props.numel() > 0:
                data.y_bond_props = (data.y_bond_props - bond_mean) / bond_std

            out = model(data, link_prediction_candidates=data.link_prediction_candidates)

            pred_atomic = out['pred_atom']
            pred_bond = out['pred_bond']
            pred_interaction = out['pred_interaction']
            pred_logits = out['pred_link_logits']

            # De-normalize for MAE
            pred_atom_dn = pred_atomic * atom_std + atom_mean
            true_atom_dn = data.y_atomic_props * atom_std + atom_mean
            prop_errors['charge'].append(torch.abs(pred_atom_dn[:, 0] - true_atom_dn[:, 0]))

            if data.atom_bond_index.numel() > 0 and pred_bond.numel() > 0:
                mask = getattr(data, 'bond_supervised_mask', None)
                if mask is not None and mask.numel() == pred_bond.shape[0] and mask.any():
                    pbd = pred_bond[mask] * bond_std + bond_mean
                    tbd = data.y_bond_props[mask] * bond_std + bond_mean
                    prop_errors['bond_props'].append(torch.abs(pbd - tbd))
                elif data.y_bond_props.numel() > 0 and data.y_bond_props.shape[0] == pred_bond.shape[0]:
                    pbd = pred_bond * bond_std + bond_mean
                    tbd = data.y_bond_props * bond_std + bond_mean
                    prop_errors['bond_props'].append(torch.abs(pbd - tbd))

            if data.interaction_edge_index.numel() > 0 and pred_interaction.numel() > 0:
                prop_errors['interaction_props'].append(
                    torch.abs(pred_interaction - data.y_interaction_props)
                )

            if pred_logits.numel() > 0:
                probs = torch.sigmoid(pred_logits)
                preds = (probs > 0.5).cpu().numpy()
                trues = data.link_prediction_labels.cpu().numpy()
                link_preds.append(preds)
                link_trues.append(trues)
                all_probs.append(probs.cpu().numpy())

    maes = {}
    maes['charge'] = float(torch.cat(prop_errors['charge']).mean().item()) if prop_errors['charge'] else 0.0
    maes['bond_props'] = float(torch.cat(prop_errors['bond_props']).mean().item()) if prop_errors['bond_props'] else 0.0
    if prop_errors['interaction_props']:
        int_maes = torch.cat(prop_errors['interaction_props']).mean(dim=0)
        maes['interaction_E2'] = float(int_maes[0].item())
    else:
        maes['interaction_E2'] = 0.0

    link_metrics = {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'pred_pos_rate': 0.0,
                    'true_pos_rate': 0.0, 'prob_mean': 0.0, 'prob_std': 0.0}
    if link_preds and accuracy_score is not None:
        lp = np.concatenate(link_preds)
        lt = np.concatenate(link_trues)
        link_metrics['acc'] = float(accuracy_score(lt, lp))
        link_metrics['prec'] = float(precision_score(lt, lp, zero_division=0))
        link_metrics['rec'] = float(recall_score(lt, lp, zero_division=0))
        link_metrics['pred_pos_rate'] = float(lp.mean())
        link_metrics['true_pos_rate'] = float(lt.mean())
        if all_probs:
            pa = np.concatenate(all_probs)
            link_metrics['prob_mean'] = float(pa.mean())
            link_metrics['prob_std'] = float(pa.std())

    return maes, link_metrics


# ===================================================================
# Main
# ===================================================================

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu is not None else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(args.checkpoints_dir, f"{timestamp}_{args.model_name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoints dir: {run_dir}")

    # Data
    train_loader, val_loader, _, norm_stats = get_nbo_dataloaders(
        dataset_root=args.dataset_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Model
    model = NBOFoundationModel(
        hidden_dim=args.hidden_dim,
        n_token_mp_layers=args.n_token_mp_layers,
        n_token_mp_blocks=args.n_token_mp_blocks,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        atom_out_dim=4,
        bond_out_dim=2,
        interaction_out_dim=3,
        use_pair_guide=args.use_pair_guide,
        use_base_delta=args.use_base_delta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience,
    )

    criterions = (nn.L1Loss(), nn.L1Loss(), nn.L1Loss(), nn.BCEWithLogitsLoss())
    loss_weights = (args.w_atom, args.w_bond, args.w_interaction, args.w_link)

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and args.amp_dtype == 'fp16')

    best_composite = float('inf')

    print("\n--- NBO Foundation Model v2: simg pretraining ---")
    for epoch in range(1, args.epochs + 1):
        train_loss, avg_losses = train_epoch(
            model, train_loader, optimizer, criterions, loss_weights, norm_stats, device,
            use_amp=use_amp, amp_dtype=args.amp_dtype, scaler=scaler, max_grad_norm=args.max_grad_norm,
        )
        val_maes, link_metrics = validate_epoch(model, val_loader, norm_stats, device)

        comp_score = composite_val_score(val_maes, link_metrics)
        scheduler.step(comp_score)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} [lr={lr_now:.2e}]: "
              f"Train Loss={train_loss:.4f} (Prop={avg_losses['prop']:.4f}, Link={avg_losses['link']:.4f})")
        print(f"  Val MAE -> Charge={val_maes['charge']:.4f} | Bond={val_maes['bond_props']:.4f} "
              f"| Int_E2={val_maes['interaction_E2']:.4f} | Composite={comp_score:.4f}")
        print(f"  Link -> Acc={link_metrics['acc']:.3f} | Prec={link_metrics['prec']:.3f} "
              f"| Rec={link_metrics['rec']:.3f}")

        if comp_score < best_composite:
            best_composite = comp_score
            best_path = os.path.join(run_dir, f"{args.model_name}_best.pt")
            stats_path = os.path.join(run_dir, f"{args.model_name}_norm_stats.pt")
            print(f"  -> New best composite={comp_score:.4f}. Saving.")
            torch.save(model.state_dict(), best_path)
            export_stats = dict(norm_stats)
            export_stats.update({
                'best_composite': comp_score,
                'best_epoch': epoch,
                'val_maes': val_maes,
                'link_metrics': link_metrics,
            })
            torch.save(export_stats, stats_path)

    print(f"\nTraining complete. Best composite score: {best_composite:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBO Foundation Model v2 — simg pretraining')

    parser.add_argument('--dataset_root', type=str,
                        default='datasets')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='nbo_foundation_v2')

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_token_mp_layers', type=int, default=2)
    parser.add_argument('--n_token_mp_blocks', type=int, default=3)
    parser.add_argument('--num_rbf', type=int, default=64)
    parser.add_argument('--cutoff', type=float, default=5.0)
    parser.add_argument('--use_pair_guide', action='store_true', default=True)
    parser.add_argument('--no_pair_guide', dest='use_pair_guide', action='store_false')
    parser.add_argument('--use_base_delta', action='store_true', default=True)
    parser.add_argument('--no_base_delta', dest='use_base_delta', action='store_false')

    # Loss weights
    parser.add_argument('--w_atom', type=float, default=0.2)
    parser.add_argument('--w_bond', type=float, default=0.2)
    parser.add_argument('--w_interaction', type=float, default=0.6)
    parser.add_argument('--w_link', type=float, default=1.0)

    # Hardware
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--amp_dtype', type=str, choices=['bf16', 'fp16'], default='bf16')

    args = parser.parse_args()
    main(args)
