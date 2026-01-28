#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
California Housing with cached Slepian-cap features (PySHTOOLS).
Clear separation between feature computation and model training.
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Add parent directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Performance settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

# Check for PySHTOOLS
try:
    import pyshtools as pysh
    HAVE_PYSH = True
    print(f"PySHTOOLS version: {pysh.__version__}")
except ImportError:
    HAVE_PYSH = False
    print("WARNING: PySHTOOLS not found. Install with: pip install pyshtools")

from spherical_harmonics_ylm import SH as SH_analytic
from nn import build_indexed_location_model


# =============================================================================
# Feature Computation
# =============================================================================

def compute_ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real spherical harmonics Y_lm(theta, phi) using analytic implementation."""
    # Convert to torch tensors
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.float64)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)

    # Tiny clamp to avoid issues near poles
    eps = 1e-12
    theta = torch.clamp(theta, eps, math.pi - eps)

    # Compute using analytic SH
    y = SH_analytic(m, l, phi, theta)

    # Handle scalar returns
    if not torch.is_tensor(y):
        y = torch.full_like(theta, float(y))
    elif y.ndim == 0:
        y = torch.full_like(theta, y.item())

    # Convert back to numpy
    return y.detach().cpu().numpy()


def compute_global_sh_features(coords: np.ndarray, L: int) -> np.ndarray:
    """
    Compute global spherical harmonic features.
    
    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        L: Maximum degree
    
    Returns:
        [N, L^2] array of SH features
    """
    if L == 0:
        return np.zeros((len(coords), 0), dtype=np.float32)
    
    lon_rad = coords[:, 0] * np.pi / 180.0
    lat_rad = coords[:, 1] * np.pi / 180.0
    phi = lon_rad + np.pi           # [0, 2π]
    theta = np.pi/2.0 - lat_rad     # [0, π]
    
    features = []
    for l in range(L):
        for m in range(-l, l+1):
            ylm = compute_ylm(l, m, theta, phi)
            features.append(ylm)
    
    return np.column_stack(features).astype(np.float32)


def compute_coverage(
    coords: np.ndarray,
    cap_center: Tuple[float, float],
    cap_radius_deg: float
) -> float:
    """
    Compute the percentage of data points within the spherical cap.

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        cap_center: (lon, lat) center of cap in degrees
        cap_radius_deg: Cap radius in degrees

    Returns:
        Coverage percentage (0-100)
    """
    # Convert to radians
    lon1 = np.radians(coords[:, 0])
    lat1 = np.radians(coords[:, 1])
    lon2 = np.radians(cap_center[0])
    lat2 = np.radians(cap_center[1])

    # Haversine formula for angular distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    angular_dist_deg = np.degrees(2 * np.arcsin(np.sqrt(a)))

    # Count points within cap
    inside_cap = angular_dist_deg <= cap_radius_deg
    coverage_pct = 100.0 * inside_cap.sum() / len(coords)

    return coverage_pct


def compute_angular_distances(
    coords: np.ndarray,
    cap_center: Tuple[float, float]
) -> np.ndarray:
    """
    Compute angular distance from cap center for all points.

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        cap_center: (lon, lat) center of cap in degrees

    Returns:
        [N] array of angular distances in degrees
    """
    lon1 = np.radians(coords[:, 0])
    lat1 = np.radians(coords[:, 1])
    lon2 = np.radians(cap_center[0])
    lat2 = np.radians(cap_center[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    angular_dist_deg = np.degrees(2 * np.arcsin(np.sqrt(a)))

    return angular_dist_deg


def partition_by_cap(
    coords: np.ndarray,
    cap_center: Tuple[float, float],
    cap_radius_deg: float
) -> np.ndarray:
    """
    Partition points into inside-cap and outside-cap.

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        cap_center: (lon, lat) center of cap in degrees
        cap_radius_deg: Cap radius in degrees

    Returns:
        Boolean mask (True = inside cap)
    """
    angular_dist_deg = compute_angular_distances(coords, cap_center)
    return angular_dist_deg <= cap_radius_deg


def find_coverage_radius(
    coords: np.ndarray,
    cap_center: Tuple[float, float],
    target_coverage: float = 0.5,
) -> float:
    """
    Binary search for the SMALLEST radius that covers at least target_coverage fraction.

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        cap_center: (lon, lat) center of cap in degrees
        target_coverage: Target fraction of points inside cap (0-1)

    Returns:
        Smallest cap radius (in degrees) that covers at least target_coverage
    """
    low, high = 0.1, 10.0
    while high - low > 0.1:
        mid = (low + high) / 2
        cov = compute_coverage(coords, cap_center, mid) / 100.0
        if cov < target_coverage:
            low = mid
        else:
            high = mid
    # Return high to ensure we have at least target coverage
    return high


def create_concentration_splits(
    coords: np.ndarray,
    targets: np.ndarray,
    cap_center: Tuple[float, float],
    cap_radius_deg: float,
    test_size_per_region: int = 1000,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Create train/val/test splits for concentration control experiment.
    DEPRECATED: Use create_coverage_curve_splits instead.

    - Train: ONLY inside-cap points
    - Val: ONLY inside-cap points (subset for early stopping)
    - Test_inside: N points from inside cap
    - Test_outside: N points from outside cap (SAME N for equal comparison)

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        targets: [N] array of target values
        cap_center: (lon, lat) center of cap in degrees
        cap_radius_deg: Cap radius in degrees
        test_size_per_region: Number of test points per region
        val_ratio: Fraction of inside-cap training data for validation
        seed: Random seed

    Returns:
        Dictionary with indices and metadata
    """
    inside_mask = partition_by_cap(coords, cap_center, cap_radius_deg)

    inside_idx = np.where(inside_mask)[0]
    outside_idx = np.where(~inside_mask)[0]

    # Determine test size (equal for both regions)
    N_test = min(test_size_per_region, len(inside_idx) // 5, len(outside_idx) // 2)

    rng = np.random.default_rng(seed)

    # Sample test sets (equal size from inside and outside)
    inside_test_idx = rng.choice(inside_idx, N_test, replace=False)
    outside_test_idx = rng.choice(outside_idx, N_test, replace=False)

    # Remaining inside points for train/val
    inside_trainval_idx = np.setdiff1d(inside_idx, inside_test_idx)

    # Split train/val from inside-only pool
    n_val = int(len(inside_trainval_idx) * val_ratio)
    rng.shuffle(inside_trainval_idx)
    val_idx = inside_trainval_idx[:n_val]
    train_idx = inside_trainval_idx[n_val:]

    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_inside_idx': inside_test_idx,
        'test_outside_idx': outside_test_idx,
        'n_test_per_region': N_test,
        'n_inside': len(inside_idx),
        'n_outside': len(outside_idx),
        'coverage': len(inside_idx) / len(coords) * 100
    }


def create_standard_splits(
    coords: np.ndarray,
    targets: np.ndarray,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Create train/val/test splits for cap sweep experiment.

    - Train: ALL data
    - Val: ALL data
    - Test: ALL data

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        targets: [N] array of target values
        test_ratio: Fraction of data for test set
        val_ratio: Fraction of remaining data for validation
        seed: Random seed

    Returns:
        Dictionary with indices and metadata
    """
    n_total = len(coords)
    rng = np.random.default_rng(seed)

    # Shuffle all indices
    all_idx = np.arange(n_total)
    rng.shuffle(all_idx)

    # Split into train/val/test
    n_test = int(n_total * test_ratio)
    n_val = int((n_total - n_test) * val_ratio)

    test_idx = all_idx[:n_test]
    val_idx = all_idx[n_test:n_test + n_val]
    train_idx = all_idx[n_test + n_val:]

    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
    }


def compute_slepian_features(
    coords: np.ndarray,
    L_slepian: int,
    cap_radius_deg: float,
    num_modes: int,
    cap_center: Tuple[float, float] = (-119.5, 37.0),
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute Slepian cap features using PySHTOOLS.

    Args:
        coords: [N, 2] array of (lon, lat) in degrees
        L_slepian: Maximum degree for Slepian functions
        cap_radius_deg: Cap radius in degrees
        num_modes: Number of Slepian modes to compute
        cap_center: (lon, lat) center of cap
        verbose: Print progress info

    Returns:
        features: [N, num_modes] array of Slepian features
        eigenvalues: [num_modes] array of eigenvalues
        basis_time: time for basis construction (from_cap + rotate)
        eval_time: time for evaluating at all coordinates
    """
    if not HAVE_PYSH:
        raise ImportError("PySHTOOLS required for Slepian features")

    # Cap num_modes at maximum possible: (L+1)^2
    max_modes = (L_slepian + 1) ** 2
    if num_modes > max_modes:
        if verbose:
            print(f"Warning: num_modes={num_modes} exceeds max={(L_slepian+1)**2} for L={L_slepian}, capping.")
        num_modes = max_modes

    if verbose:
        print(f"Computing Slepian features:")
        print(f"  Cap center: {cap_center}, Radius: {cap_radius_deg}°")
        print(f"  L_slepian={L_slepian}, num_modes={num_modes}")

    # === PHASE 1: Basis construction ===
    t_basis_start = time.time()

    # Create Slepian basis on cap
    slepian = pysh.Slepian.from_cap(
        theta=cap_radius_deg,
        lmax=L_slepian,
        nmax=num_modes
    )

    # Rotate to cap center
    slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=num_modes)

    # Get eigenvalues
    eigenvalues = slepian.eigenvalues[:num_modes].astype(np.float32)

    basis_time = time.time() - t_basis_start

    if verbose:
        print(f"  Eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
        print(f"  #λ>0.5: {(eigenvalues > 0.5).sum()}")
        print(f"  Basis construction time: {basis_time:.2f}s")

    # === PHASE 2: Evaluate at coordinates ===
    t_eval_start = time.time()

    lon = coords[:, 0]
    lat = coords[:, 1]
    lon_360 = np.where(lon < 0.0, lon + 360.0, lon)

    features = []
    for i in range(num_modes):
        coeffs = slepian.to_shcoeffs(i)
        vals = coeffs.expand(lon=lon_360, lat=lat, degrees=True)
        features.append(vals)

    features = np.column_stack(features).astype(np.float32)

    eval_time = time.time() - t_eval_start

    if verbose:
        print(f"  Evaluation time: {eval_time:.2f}s")

    return features, eigenvalues, basis_time, eval_time


def compute_and_cache_features(
    coords: np.ndarray,
    L_global: int,
    L_slepian: int,
    cap_radius_deg: float,
    num_modes: int,
    lambda_thresh: float,
    cap_center: Tuple[float, float] = (-119.5, 37.0),
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute all features (global SH + Slepian), trim by eigenvalue, and optionally cache.
    
    Returns dict with:
        - features: [N, D] combined features
        - metadata: dict with configuration and dimensions
    """
    t0 = time.time()
    
    # Compute global SH features
    if verbose:
        print(f"Computing global SH features (L={L_global})...")
    global_features = compute_global_sh_features(coords, L_global)
    global_dim = global_features.shape[1]
    
    # Compute Slepian features
    if verbose:
        print(f"Computing Slepian features...")
    slepian_features, eigenvalues, slepian_basis_time, slepian_eval_time = compute_slepian_features(
        coords, L_slepian, cap_radius_deg, num_modes,
        cap_center=cap_center,
        verbose=verbose
    )

    # Trim Slepian features by eigenvalue threshold
    keep_mask = eigenvalues > lambda_thresh
    kept_modes = keep_mask.sum()
    slepian_trimmed = slepian_features[:, keep_mask]

    if verbose:
        print(f"Eigenvalue trimming: λ>{lambda_thresh} → kept {kept_modes}/{num_modes} modes")

    # Combine features
    features = np.hstack([global_features, slepian_trimmed])

    # Create metadata
    metadata = {
        'L_global': L_global,
        'L_slepian': L_slepian,
        'cap_radius_deg': cap_radius_deg,
        'cap_center_lon': cap_center[0],
        'cap_center_lat': cap_center[1],
        'num_modes_initial': num_modes,
        'num_modes_kept': int(kept_modes),
        'lambda_thresh': lambda_thresh,
        'global_dim': int(global_dim),
        'slepian_dim': int(kept_modes),
        'total_dim': int(features.shape[1]),
        'n_samples': int(features.shape[0]),
        'eigenvalues_kept': eigenvalues[keep_mask].tolist(),
        'slepian_basis_time_sec': slepian_basis_time,
        'slepian_eval_time_sec': slepian_eval_time,
    }
    
    # Cache if requested
    if cache_path:
        cache_data = {
            'features': torch.tensor(features, dtype=torch.float32),
            'metadata': metadata
        }
        torch.save(cache_data, cache_path)
        if verbose:
            print(f"Cached features to {cache_path}")
    
    dt = time.time() - t0
    if verbose:
        print(f"Feature computation time: {dt:.2f}s")
        print(f"Final feature dimension: {features.shape[1]}")
    
    return {
        'features': torch.tensor(features, dtype=torch.float32),
        'metadata': metadata
    }


def load_cached_features(cache_path: str, verbose: bool = True) -> Dict:
    """
    Load precomputed features from cache.
    
    Returns dict with:
        - features: [N, D] tensor
        - metadata: dict with configuration
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    cache_data = torch.load(cache_path, map_location='cpu')
    
    # Handle old format (just tensor) vs new format (dict)
    if isinstance(cache_data, torch.Tensor):
        features = cache_data
        metadata = {'total_dim': features.shape[1], 'n_samples': features.shape[0]}
    else:
        features = cache_data['features']
        metadata = cache_data['metadata']
    
    if verbose:
        print(f"Loaded cached features from {cache_path}")
        print(f"  Shape: {features.shape}")
        if 'L_slepian' in metadata:
            print(f"  Config: L_global={metadata.get('L_global', '?')}, "
                  f"L_slepian={metadata['L_slepian']}, "
                  f"kept_modes={metadata.get('num_modes_kept', '?')}")
    
    return {'features': features, 'metadata': metadata}


# =============================================================================
# Dataset and Model
# =============================================================================

class IndexedDataset(Dataset):
    """Dataset that provides global indices for cached feature lookup."""
    
    def __init__(self, coords: torch.Tensor, targets: torch.Tensor, global_indices: torch.Tensor):
        self.coords = coords
        self.targets = targets
        self.global_indices = global_indices
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.global_indices[idx], self.coords[idx], self.targets[idx]


class CachedFeatureEncoder(nn.Module):
    """
    Encoder that uses precomputed features via lookup.
    Features are stored in float16 for memory efficiency.
    """
    
    def __init__(self, features: torch.Tensor):
        super().__init__()
        # Store as float16 to save memory
        features_fp16 = features.to(torch.float16)
        if torch.cuda.is_available():
            features_fp16 = features_fp16.pin_memory()
        self.register_buffer("features", features_fp16, persistent=False)
        self.n_features = features.shape[1]
    
    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # coords not used; we index by global row id
        device = coords.device
        idx_cpu = indices.detach().cpu()
        features = self.features[idx_cpu].to(device=device, dtype=torch.float32, non_blocking=True)
        return features


class LocationRegressor(nn.Module):
    """MLP regressor on top of location encoder."""
    
    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(encoder.n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        features = self.encoder(coords, indices)
        return self.mlp(features).squeeze(-1)


# =============================================================================
# Training and Evaluation
# =============================================================================

def create_data_subsets(
    train_ds: Dataset,
    fraction: float,
    batch_size: int,
    seed: int,
    num_workers: int = 8
) -> DataLoader:
    """Create a random subset of the training data."""
    n_total = len(train_ds)
    n_subset = max(1, int(fraction * n_total))
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_subset, replace=False)
    
    subset = Subset(train_ds, indices.tolist())
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 20
) -> Tuple[List[float], List[float]]:
    """Train model with early stopping."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for idx, coords, targets in train_loader:
            coords = coords.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            predictions = model(coords, idx)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(targets)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, coords, targets in val_loader:
                coords = coords.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)
                
                predictions = model(coords, idx)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * len(targets)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_state)
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return train_losses, val_losses


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    y_min: float,
    y_max: float
) -> Dict:
    """Evaluate model and compute metrics."""
    
    model.eval()
    predictions = []
    targets = []
    
    for idx, coords, y in test_loader:
        coords = coords.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        
        pred = model(coords, idx)
        predictions.append(pred.cpu())
        targets.append(y)
    
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    
    # Metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1.0 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
    
    # Convert back to dollars
    predictions_dollars = predictions * (y_max - y_min) + y_min
    targets_dollars = targets * (y_max - y_min) + y_min
    mae_dollars = np.mean(np.abs(predictions_dollars - targets_dollars))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'mae_dollars': float(mae_dollars),
        'predictions': predictions,
        'targets': targets
    }


@torch.no_grad()
def evaluate_concentration(
    model: nn.Module,
    test_inside_loader: DataLoader,
    test_outside_loader: DataLoader,
    device: torch.device,
    y_min: float,
    y_max: float
) -> Dict:
    """
    Evaluate model separately on inside and outside test sets.
    DEPRECATED: Use evaluate_coverage_curve instead.

    Args:
        model: Trained model
        test_inside_loader: DataLoader for inside-cap test points
        test_outside_loader: DataLoader for outside-cap test points
        device: Device to evaluate on
        y_min: Minimum target value (for denormalization)
        y_max: Maximum target value (for denormalization)

    Returns:
        Dictionary with metrics for both regions and delta (concentration measure)
    """
    metrics_inside = evaluate_model(model, test_inside_loader, device, y_min, y_max)
    metrics_outside = evaluate_model(model, test_outside_loader, device, y_min, y_max)

    delta_r2 = metrics_inside['r2'] - metrics_outside['r2']

    return {
        'r2_inside': metrics_inside['r2'],
        'r2_outside': metrics_outside['r2'],
        'delta_r2': delta_r2,  # PRIMARY METRIC: concentration measure
        'mse_inside': metrics_inside['mse'],
        'mse_outside': metrics_outside['mse'],
        'mae_inside': metrics_inside['mae'],
        'mae_outside': metrics_outside['mae'],
        'mae_dollars_inside': metrics_inside['mae_dollars'],
        'mae_dollars_outside': metrics_outside['mae_dollars'],
    }




# =============================================================================
# Main Experiment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="California Housing with Slepian Features")
    
    # Feature configuration
    parser.add_argument("--L-global", type=int, default=10,
                       help="Max degree for global SH")
    parser.add_argument("--L-slepian", type=int, default=120,
                       help="Max degree for Slepian functions")
    parser.add_argument("--cap-radius", type=float, default=5.0,
                       help="Cap radius in degrees")
    parser.add_argument("--num-modes", type=int, default=800,
                       help="Initial number of Slepian modes")
    parser.add_argument("--lambda-thresh", type=float, default=0.05,
                       help="Eigenvalue threshold for mode selection")
    parser.add_argument("--cap-center-lon", type=float, default=-119.5,
                       help="Cap center longitude (default: -119.5 for California)")
    parser.add_argument("--cap-center-lat", type=float, default=37.0,
                       help="Cap center latitude (default: 37.0 for California)")
    parser.add_argument("--cache-path", type=str, default=None,
                       help="Path to cache/load features")
    parser.add_argument("--force-recompute", action="store_true",
                       help="Force recomputation even if cache exists")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    
    # Experiment configuration
    parser.add_argument("--label-fracs", type=str, default="0.01,0.10,0.25,0.50,0.75,1.00",
                       help="Comma-separated training fractions")
    parser.add_argument("--n-runs", type=int, default=1,
                       help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output configuration
    parser.add_argument("--csv-path", type=str, default=None,
                       help="Path to save CSV results")
    parser.add_argument("--fig-dir", type=str, default="figs",
                       help="Directory for figures")
    parser.add_argument("--results-json", type=str, default=None,
                       help="Path to save JSON results")

    # Architecture selection
    parser.add_argument("--arch", type=str, default="mlp",
                       choices=["linear", "mlp", "resmlp", "siren", "glu"],
                       help="Neural network architecture (default: mlp)")

    # Cap sweep experiment: vary cap radius to test concentration hypothesis
    parser.add_argument("--cap-sweep", action="store_true",
                       help="Run cap radius sweep experiment: build NEW Slepian basis for each coverage, train on ALL data, evaluate on ALL test data")
    parser.add_argument("--target-coverages", type=str, default="0.1,0.5,1.0",
                       help="Comma-separated target coverage fractions (cap radius found via binary search)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Fraction of remaining data for validation (default: 0.1)")

    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load California housing data
    print("Loading California housing data...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    coords = df[["Longitude", "Latitude"]].values.astype(np.float32)
    targets = df["MedHouseVal"].values.astype(np.float32)
    
    # Normalize targets to [0, 1]
    y_min, y_max = targets.min(), targets.max()
    targets_norm = (targets - y_min) / (y_max - y_min + 1e-12)

    # =========================================================================
    # CAP SWEEP EXPERIMENT: Vary cap radius to test concentration hypothesis
    # =========================================================================
    if args.cap_sweep:
        print("\n" + "=" * 70)
        print("CAP SWEEP EXPERIMENT")
        print("Build NEW Slepian basis for each coverage fraction")
        print("Train on ALL data, evaluate on ALL test data")
        print("=" * 70)

        cap_center = (args.cap_center_lon, args.cap_center_lat)
        target_coverages = [float(x) for x in args.target_coverages.split(',')]

        if not HAVE_PYSH:
            raise RuntimeError("PySHTOOLS required for cap sweep experiment")

        # Create standard train/val/test splits (SAME for all configs)
        splits = create_standard_splits(
            coords, targets_norm,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

        print(f"\nData splits (SAME for all cap radii):")
        print(f"  Train: {splits['n_train']:,}")
        print(f"  Val: {splits['n_val']:,}")
        print(f"  Test: {splits['n_test']:,}")

        # Create coordinate tensors
        train_coords_np = coords[splits['train_idx']]
        val_coords_np = coords[splits['val_idx']]
        test_coords_np = coords[splits['test_idx']]

        train_targets_t = torch.tensor(targets_norm[splits['train_idx']], dtype=torch.float32)
        val_targets_t = torch.tensor(targets_norm[splits['val_idx']], dtype=torch.float32)
        test_targets_t = torch.tensor(targets_norm[splits['test_idx']], dtype=torch.float32)

        train_indices_t = torch.tensor(splits['train_idx'], dtype=torch.long)
        val_indices_t = torch.tensor(splits['val_idx'], dtype=torch.long)
        test_indices_t = torch.tensor(splits['test_idx'], dtype=torch.long)

        csv_results = []

        # ---------------------------------------------------------------------
        # SWEEP OVER TARGET COVERAGES
        # ---------------------------------------------------------------------
        for target_cov in target_coverages:
            print(f"\n{'=' * 70}")
            print(f"TARGET COVERAGE: {target_cov*100:.0f}%")
            print(f"{'=' * 70}")

            # Find cap radius that achieves target coverage
            cap_radius = find_coverage_radius(coords, cap_center, target_cov)
            actual_cov = compute_coverage(coords, cap_center, cap_radius) / 100.0

            print(f"  Cap center: {cap_center}")
            print(f"  Cap radius: {cap_radius:.2f}° → actual coverage: {actual_cov*100:.1f}%")

            # Build cache path for this cap radius
            cache_dir = "cache/cali_cap_sweep"
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(
                cache_dir,
                f"cali_L{args.L_slepian}_rad{cap_radius:.2f}_lon{cap_center[0]}_lat{cap_center[1]}_lam{args.lambda_thresh}.pt"
            )

            # Compute or load features for this cap radius
            if os.path.exists(cache_path) and not args.force_recompute:
                print(f"  Loading cached features from {cache_path}...")
                feature_data = load_cached_features(cache_path, verbose=False)
            else:
                print(f"  Computing NEW Slepian basis for cap_radius={cap_radius:.2f}°...")
                feature_data = compute_and_cache_features(
                    coords,
                    L_global=args.L_global,
                    L_slepian=args.L_slepian,
                    cap_radius_deg=cap_radius,
                    num_modes=args.num_modes,
                    lambda_thresh=args.lambda_thresh,
                    cap_center=cap_center,
                    cache_path=cache_path,
                    verbose=True
                )

            features = feature_data['features']
            metadata = feature_data['metadata']

            feature_dim = features.shape[1]
            slepian_modes = metadata.get('num_modes_kept', 0)
            global_dim = metadata.get('global_dim', args.L_global ** 2)

            print(f"  Feature dim: {feature_dim} (global={global_dim}, slepian={slepian_modes})")

            # Create encoder with features for this cap
            encoder = CachedFeatureEncoder(features)

            # Create datasets (using same splits)
            train_dataset = IndexedDataset(
                torch.tensor(train_coords_np, dtype=torch.float32),
                train_targets_t, train_indices_t
            )
            val_dataset = IndexedDataset(
                torch.tensor(val_coords_np, dtype=torch.float32),
                val_targets_t, val_indices_t
            )
            test_dataset = IndexedDataset(
                torch.tensor(test_coords_np, dtype=torch.float32),
                test_targets_t, test_indices_t
            )

            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )

            # -----------------------------------------------------------------
            # MULTIPLE RUNS
            # -----------------------------------------------------------------
            for run_idx in range(args.n_runs):
                run_seed = args.seed + run_idx
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)

                print(f"\n  Run {run_idx + 1}/{args.n_runs} (seed={run_seed})")

                # Create fresh model
                model = build_indexed_location_model(
                    encoder, task="regression", arch=args.arch,
                    hidden_dim=args.hidden_dim, dropout=args.dropout
                ).to(device)

                # Train on ALL data
                t_start = time.time()
                train_losses, val_losses = train_model(
                    model, train_loader, val_loader, device,
                    epochs=args.epochs, lr=args.lr, patience=args.patience
                )
                train_time = time.time() - t_start

                # Evaluate on ALL test data
                metrics = evaluate_model(model, test_loader, device, y_min, y_max)

                print(f"    R²={metrics['r2']:.4f}, MAE=${metrics['mae_dollars']:,.0f}, time={train_time:.1f}s")

                # Record results
                csv_results.append({
                    'method': 'slepian_cap',
                    'dataset': 'california',
                    'target_coverage': target_cov,
                    'cap_radius_deg': cap_radius,
                    'actual_coverage_pct': actual_cov * 100,
                    'L_global': args.L_global,
                    'L_slepian': args.L_slepian,
                    'lambda_thresh': args.lambda_thresh,
                    'feature_dim': feature_dim,
                    'slepian_modes_kept': slepian_modes,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'r2': metrics['r2'],
                    'rmse_raw': np.sqrt(metrics['mse']) * (y_max - y_min),
                    'mae_raw': metrics['mae_dollars'],
                    'n_train': splits['n_train'],
                    'n_val': splits['n_val'],
                    'n_test': splits['n_test'],
                    'train_time_sec': train_time,
                    'arch': args.arch,
                })

        # Save CSV results
        if args.csv_path:
            df_results = pd.DataFrame(csv_results)
            os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
            df_results.to_csv(args.csv_path, index=False)
            print(f"\n{'=' * 70}")
            print(f"Saved cap sweep results to {args.csv_path}")

            # Print summary by target coverage
            print("\nSummary (mean ± std R² by target coverage):")
            summary = df_results.groupby('target_coverage')['r2'].agg(['mean', 'std'])
            print(summary.round(4).to_string())

            # Also show feature dimensions
            print("\nFeature dimensions by coverage:")
            dim_summary = df_results.groupby('target_coverage')[['feature_dim', 'slepian_modes_kept', 'cap_radius_deg']].first()
            print(dim_summary.to_string())

        # Save JSON metadata
        if args.results_json:
            json_data = {
                'experiment': 'cap_sweep',
                'hypothesis': 'Slepian concentrates capacity inside cap - R² should increase with coverage',
                'configuration': vars(args),
                'target_coverages': target_coverages,
                'splits': {
                    'n_train': int(splits['n_train']),
                    'n_val': int(splits['n_val']),
                    'n_test': int(splits['n_test']),
                },
                'csv_path': args.csv_path
            }
            os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
            with open(args.results_json, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Saved metadata to {args.results_json}")

        return  # Exit after cap sweep experiment

    # =========================================================================
    # STANDARD EXPERIMENT (original code path)
    # =========================================================================
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        coords, targets_norm, test_size=0.4, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )

    print(f"Dataset splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Combine all coordinates for feature computation
    all_coords = np.vstack([X_train, X_val, X_test])
    n_train = len(X_train)
    n_val = len(X_val)

    # Compute coverage
    cap_center = (args.cap_center_lon, args.cap_center_lat)
    coverage_pct = compute_coverage(all_coords, cap_center, args.cap_radius)
    print(f"Cap coverage: {coverage_pct:.1f}% of data points within cap")

    # Compute or load features
    if args.cache_path and os.path.exists(args.cache_path) and not args.force_recompute:
        print(f"Loading cached features from {args.cache_path}")
        feature_data = load_cached_features(args.cache_path)
        # Note: When loading cached features, the original configuration is already baked in
        # The num_modes parameter here is irrelevant - we use whatever was cached
    else:
        if not HAVE_PYSH:
            raise RuntimeError("PySHTOOLS required to compute Slepian features")
        
        print("Computing features from scratch...")
        cap_center = (args.cap_center_lon, args.cap_center_lat)
        feature_data = compute_and_cache_features(
            all_coords,
            L_global=args.L_global,
            L_slepian=args.L_slepian,
            cap_radius_deg=args.cap_radius,
            num_modes=args.num_modes,
            lambda_thresh=args.lambda_thresh,
            cap_center=cap_center,
            cache_path=args.cache_path
        )
    
    features = feature_data['features']
    metadata = feature_data['metadata']
    
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Feature metadata: {metadata}")
    
    # Create encoder with cached features
    encoder = CachedFeatureEncoder(features)
    
    # Create datasets with global indices
    global_indices_train = torch.arange(0, n_train)
    global_indices_val = torch.arange(n_train, n_train + n_val)
    global_indices_test = torch.arange(n_train + n_val, n_train + n_val + len(X_test))
    
    train_dataset = IndexedDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        global_indices_train
    )
    
    val_dataset = IndexedDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        global_indices_val
    )
    
    test_dataset = IndexedDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        global_indices_test
    )
    
    # Fixed validation and test loaders
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Run experiments
    label_fracs = [float(x) for x in args.label_fracs.split(',')]
    csv_results = []
    
    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
        print(f"{'='*60}")
        
        for frac in label_fracs:
            pct = int(frac * 100)
            print(f"\nTraining with {pct}% of data (run {run_idx+1})...")
            
            # Create subset loader
            train_subset_loader = create_data_subsets(
                train_dataset, frac, args.batch_size, run_seed, args.num_workers
            )
            
            # Create fresh model
            model = build_indexed_location_model(
                encoder, task="regression", arch=args.arch,
                hidden_dim=args.hidden_dim, dropout=args.dropout
            ).to(device)
            
            # Train
            t_start = time.time()
            train_losses, val_losses = train_model(
                model, train_subset_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, patience=args.patience
            )
            train_time = time.time() - t_start
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device, y_min, y_max)
            
            print(f"[{pct}%] R²={metrics['r2']:.4f}, MAE=${metrics['mae_dollars']:,.0f}")
            
            # Record results
            csv_results.append({
                'method': 'slepian_cap',
                'arch': args.arch,
                'L_global': metadata.get('L_global', args.L_global),
                'L_slepian': metadata.get('L_slepian', args.L_slepian),
                'cap_radius': args.cap_radius,
                'cap_center_lon': args.cap_center_lon,
                'cap_center_lat': args.cap_center_lat,
                'coverage_pct': coverage_pct,
                'num_modes_kept': metadata.get('num_modes_kept', '?'),
                'feature_dim': features.shape[1],
                'run': run_idx + 1,
                'seed': run_seed,
                'train_fraction': frac,
                'train_percent': pct,
                'train_samples': len(train_subset_loader.dataset),
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mae_dollars': metrics['mae_dollars'],
                'train_loss': train_losses[-1] if train_losses else 0,
                'val_loss': val_losses[-1] if val_losses else 0,
                'slepian_basis_time_sec': metadata.get('slepian_basis_time_sec', 0),
                'slepian_eval_time_sec': metadata.get('slepian_eval_time_sec', 0),
                'train_time_sec': train_time,
            })
    
    # Save CSV results
    if args.csv_path:
        df_results = pd.DataFrame(csv_results)
        df_results.to_csv(args.csv_path, index=False)
        print(f"\nSaved results to {args.csv_path}")
        
        # Print summary
        print("\nSummary (mean ± std across runs):")
        summary = df_results.groupby('train_percent')[['r2', 'mae_dollars']].agg(['mean', 'std'])
        print(summary)
    
    # Save JSON metadata
    if args.results_json:
        json_data = {
            'configuration': vars(args),
            'feature_metadata': metadata,
            'csv_path': args.csv_path
        }
        with open(args.results_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved metadata to {args.results_json}")


if __name__ == "__main__":
    main()