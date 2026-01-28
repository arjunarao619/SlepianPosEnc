"""
Centralized training utilities for ACE experiments.

This module consolidates training loop code that was previously duplicated
across train_ste.py, train_dpss.py, train_baselines.py, and train_dpss_sweep.py.
"""

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping handler with patience."""

    def __init__(self, patience=15, min_delta=0.0):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score):
        """
        Check if training should stop.

        Args:
            val_score: Validation metric (lower is better)

        Returns:
            True if this is the best score so far, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            return True

        if val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def offdiag_penalty(feats, device, normality_flag=False, exponent=1):
    """
    Compute off-diagonal Frobenius penalty for orthogonality regularization.

    Args:
        feats: (B, D) feature matrix
        device: torch device
        normality_flag: If True, penalize full correlation matrix - I
        exponent: Power to raise Frobenius norm

    Returns:
        Scalar penalty value
    """
    if feats is None:
        return torch.zeros((), device=device)
    B = feats.shape[0]
    if B == 0:
        return torch.zeros((), device=device)

    C = (feats.T @ feats) / B
    if normality_flag:
        I = torch.eye(C.shape[0], device=device, dtype=C.dtype)
        P = C - I
    else:
        P = C - torch.diag(torch.diag(C))

    exp = max(1, int(exponent))
    return torch.linalg.norm(P, ord='fro') ** exp


def train_epoch(model, dataloader, optimizer, device, ortho_weights=None, time_grad_weight=0.0):
    """
    Train model for one epoch.

    Args:
        model: STEEncoder model
        dataloader: Training DataLoader
        optimizer: Optimizer
        device: torch device
        ortho_weights: Dict with 'final', 'space', 'time' weights (or None to use model attrs)
        time_grad_weight: Weight for time gradient penalty (or None to use model attr)

    Returns:
        avg_mse: Average MSE loss
        avg_reg: Average regularization loss
    """
    model.train()
    mse_fn = nn.MSELoss()

    total_mse = 0.0
    total_reg = 0.0
    total_tg = 0.0
    n_batches = 0

    # Get regularization weights
    if ortho_weights is not None:
        w_final = float(ortho_weights.get('final', 0.0))
        w_space = float(ortho_weights.get('space', 0.0))
        w_time = float(ortho_weights.get('time', 0.0))
    else:
        w_final = float(getattr(model, 'ortho_weight', 0.0))
        w_space = float(getattr(model, 'ortho_weight_space', 0.0))
        w_time = float(getattr(model, 'ortho_weight_time', 0.0))

    if time_grad_weight is None:
        w_tg = float(getattr(model, 'time_grad_penalty_weight', 0.0))
    else:
        w_tg = float(time_grad_weight)

    normality_flag = getattr(model, 'normality_flag', False)
    ortho_exponent = getattr(model, 'ortho_exponent', 1)

    for coords, targets in dataloader:
        coords = coords.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Enable gradient tracking for time coordinate if needed
        if w_tg > 0.0:
            coords.requires_grad_(True)

        # Forward pass
        preds = model(coords)
        mse_loss = mse_fn(preds, targets)

        # Orthogonality regularization from cached activations
        reg_loss = torch.zeros((), device=device)
        if (w_final + w_space + w_time) > 0.0:
            sf = getattr(model, '_cache_space_feats', None)
            tf = getattr(model, '_cache_time_feats', None)
            ff = getattr(model, '_cache_final_feats', None)

            # Fallback to legacy method if caches missing
            if (sf is None or tf is None or ff is None) and hasattr(model, 'get_regularization_loss'):
                reg_loss = model.get_regularization_loss(coords)
            else:
                reg_final = offdiag_penalty(ff, device, normality_flag, ortho_exponent)
                reg_space = offdiag_penalty(sf, device, normality_flag, ortho_exponent)
                reg_time = offdiag_penalty(tf, device, normality_flag, ortho_exponent)
                reg_loss = w_final * reg_final + w_space * reg_space + w_time * reg_time

        # Time gradient penalty
        tg_loss = torch.zeros((), device=device)
        if w_tg > 0.0:
            s = preds.sum()
            grads = torch.autograd.grad(s, coords, create_graph=True, retain_graph=True)[0]
            t_grad = grads[:, 2]  # time column
            tg_loss = w_tg * t_grad.pow(2).mean()

        loss = mse_loss + reg_loss + tg_loss
        loss.backward()
        optimizer.step()

        total_mse += float(mse_loss.detach())
        total_reg += float(reg_loss.detach()) if isinstance(reg_loss, torch.Tensor) else float(reg_loss)
        total_tg += float(tg_loss.detach()) if isinstance(tg_loss, torch.Tensor) else float(tg_loss)
        n_batches += 1

    avg_mse = total_mse / max(1, n_batches)
    avg_reg = total_reg / max(1, n_batches)

    return avg_mse, avg_reg


@torch.no_grad()
def evaluate(model, dataloader, device, normalizer):
    """
    Evaluate model on a dataset.

    Args:
        model: STEEncoder model
        dataloader: DataLoader (yields coords, targets_normalized)
        device: torch device
        normalizer: DataNormalizer for inverse transform

    Returns:
        rmse: Mean RMSE across 8 temperature variables (in original units)
        mae: Mean MAE across 8 temperature variables (in original units)
    """
    model.eval()
    preds_all, targs_all = [], []

    for batch in dataloader:
        coords = batch[0].to(device)
        targets_norm = batch[1].to(device)

        preds_norm = model(coords)

        # Denormalize to original scale
        preds = normalizer.inverse_transform(preds_norm.cpu())
        targs = normalizer.inverse_transform(targets_norm.cpu())

        preds_all.append(preds)
        targs_all.append(targs)

    P = torch.cat(preds_all)
    T = torch.cat(targs_all)

    # Per-variable metrics
    rmse_per_var = []
    mae_per_var = []
    for i in range(8):
        diff = P[:, i] - T[:, i]
        rmse_per_var.append(torch.sqrt(torch.mean(diff ** 2)).item())
        mae_per_var.append(torch.mean(torch.abs(diff)).item())

    return float(np.mean(rmse_per_var)), float(np.mean(mae_per_var))


@torch.no_grad()
def evaluate_detailed(model, dataloader, device, normalizer):
    """
    Evaluate model with detailed per-variable metrics.

    Args:
        model: STEEncoder model
        dataloader: DataLoader
        device: torch device
        normalizer: DataNormalizer

    Returns:
        dict with 'rmse_mean', 'mae_mean', 'rmse_per_var', 'mae_per_var'
    """
    model.eval()
    preds_all, targs_all = [], []

    for batch in dataloader:
        coords = batch[0].to(device)
        targets_norm = batch[1].to(device)

        preds_norm = model(coords)
        preds = normalizer.inverse_transform(preds_norm.cpu())
        targs = normalizer.inverse_transform(targets_norm.cpu())

        preds_all.append(preds)
        targs_all.append(targs)

    P = torch.cat(preds_all)
    T = torch.cat(targs_all)

    rmse_per_var = []
    mae_per_var = []
    for i in range(8):
        diff = P[:, i] - T[:, i]
        rmse_per_var.append(torch.sqrt(torch.mean(diff ** 2)).item())
        mae_per_var.append(torch.mean(torch.abs(diff)).item())

    return {
        'rmse_mean': float(np.mean(rmse_per_var)),
        'mae_mean': float(np.mean(mae_per_var)),
        'rmse_per_var': rmse_per_var,
        'mae_per_var': mae_per_var
    }


def create_optimizer(model, lr=1e-3, weight_decay=1e-5):
    """
    Create Adam optimizer with weight decay only on head parameters.

    Args:
        model: STEEncoder model
        lr: Learning rate
        weight_decay: Weight decay for head parameters

    Returns:
        torch.optim.Adam optimizer
    """
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    return torch.optim.Adam([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=lr)
