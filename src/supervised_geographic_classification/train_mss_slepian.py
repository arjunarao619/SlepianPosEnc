#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arctic Mean Sea Surface (MSS) with Slepian Cap Features.

Uses global SH (L=10) + Slepian cap centered at North Pole for
high-resolution Arctic representation.

Usage:
    python train_mss_slepian.py --L-slepian 40 --epochs 100
    python train_mss_slepian.py --L-slepian 80 --epochs 100
    python train_mss_slepian.py --L-slepian 120 --epochs 100
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

# Import nn module for architecture selection
from nn import build_indexed_location_model

# Import shared utilities
from utils.geo import compute_coverage


# =============================================================================
# Feature Computation
# =============================================================================

def compute_ylm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real spherical harmonics Y_lm(theta, phi) using analytic implementation."""
    if not isinstance(phi, torch.Tensor):
        phi = torch.tensor(phi, dtype=torch.float64)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float64)

    eps = 1e-12
    theta = torch.clamp(theta, eps, math.pi - eps)

    y = SH_analytic(m, l, phi, theta)

    if not torch.is_tensor(y):
        y = torch.full_like(theta, float(y))
    elif y.ndim == 0:
        y = torch.full_like(theta, y.item())

    return y.detach().cpu().numpy()


def compute_global_sh_features(coords: np.ndarray, L: int, verbose: bool = True) -> np.ndarray:
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

    if verbose:
        print(f"Computing global SH features (L={L}, dim={L**2})...")

    lon_rad = coords[:, 0] * np.pi / 180.0
    lat_rad = coords[:, 1] * np.pi / 180.0
    phi = lon_rad + np.pi
    theta = np.pi / 2.0 - lat_rad

    features = []
    for l in range(L):
        for m in range(-l, l + 1):
            ylm = compute_ylm(l, m, theta, phi)
            features.append(ylm)

    return np.column_stack(features).astype(np.float32)


def compute_slepian_features(
    coords: np.ndarray,
    L_slepian: int,
    cap_radius_deg: float,
    num_modes: int,
    cap_center: Tuple[float, float] = (0.0, 90.0),  # North Pole
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
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

    # Create Slepian basis on cap
    slepian = pysh.Slepian.from_cap(
        theta=cap_radius_deg,
        lmax=L_slepian,
        nmax=num_modes
    )

    # Rotate to cap center (North Pole doesn't need rotation, but we do it for consistency)
    slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=num_modes)

    # Get eigenvalues
    eigenvalues = slepian.eigenvalues[:num_modes].astype(np.float32)

    if verbose:
        print(f"  Eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
        print(f"  #λ>0.5: {(eigenvalues > 0.5).sum()}, #λ>0.1: {(eigenvalues > 0.1).sum()}")
        print(f"  Evaluating {num_modes} modes at {len(coords):,} points (this may take a few minutes)...")

    # Evaluate Slepian functions at coordinates
    lon = coords[:, 0]
    lat = coords[:, 1]
    lon_360 = np.where(lon < 0.0, lon + 360.0, lon)

    features = []
    iterator = tqdm(range(num_modes), desc="  Slepian modes", disable=not verbose)
    for i in iterator:
        coeffs = slepian.to_shcoeffs(i)
        vals = coeffs.expand(lon=lon_360, lat=lat, degrees=True)
        features.append(vals)

    features = np.column_stack(features).astype(np.float32)

    return features, eigenvalues


def compute_and_cache_features(
    coords: np.ndarray,
    L_global: int,
    L_slepian: int,
    cap_radius_deg: float,
    num_modes: int,
    lambda_thresh: float,
    cap_center: Tuple[float, float] = (0.0, 90.0),
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute all features (global SH + Slepian), trim by eigenvalue, and optionally cache.

    Returns dict with:
        - features: [N, D] combined features tensor
        - metadata: dict with configuration and dimensions
    """
    t0 = time.time()

    # Compute global SH features
    global_features = compute_global_sh_features(coords, L_global, verbose=verbose)
    global_dim = global_features.shape[1]

    # Compute Slepian features
    slepian_features, eigenvalues = compute_slepian_features(
        coords, L_slepian, cap_radius_deg, num_modes,
        cap_center=cap_center, verbose=verbose
    )

    # Trim Slepian features by eigenvalue threshold
    keep_mask = eigenvalues > lambda_thresh
    kept_modes = keep_mask.sum()
    slepian_trimmed = slepian_features[:, keep_mask]
    eigenvalues_kept = eigenvalues[keep_mask]

    if verbose:
        print(f"Eigenvalue trimming: λ>{lambda_thresh} → kept {kept_modes}/{num_modes} modes")

    # Combine features
    features = np.hstack([global_features, slepian_trimmed])

    # Create metadata
    metadata = {
        'method': 'slepian_cap',
        'L_global': L_global,
        'L_slepian': L_slepian,
        'cap_radius_deg': cap_radius_deg,
        'cap_center': list(cap_center),
        'cap_center_lon': cap_center[0],
        'cap_center_lat': cap_center[1],
        'num_modes_initial': num_modes,
        'num_modes_kept': int(kept_modes),
        'lambda_thresh': lambda_thresh,
        'global_dim': int(global_dim),
        'slepian_dim': int(kept_modes),
        'total_dim': int(features.shape[1]),
        'n_samples': int(features.shape[0]),
        'eigenvalues_kept': eigenvalues_kept.tolist()
    }

    # Cache if requested
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
        print(f"Final feature dimension: {features.shape[1]} (global={global_dim}, slepian={kept_modes})")

    return {
        'features': torch.tensor(features, dtype=torch.float32),
        'metadata': metadata
    }


def load_cached_features(cache_path: str, verbose: bool = True) -> Dict:
    """Load precomputed features from cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)

    if isinstance(cache_data, torch.Tensor):
        features = cache_data
        metadata = {'total_dim': features.shape[1], 'n_samples': features.shape[0]}
    else:
        features = cache_data['features']
        metadata = cache_data['metadata']

    if verbose:
        print(f"Loaded cached features from {cache_path}")
        print(f"  Shape: {features.shape}")
        print(f"  L_global={metadata.get('L_global', '?')}, "
              f"L_slepian={metadata.get('L_slepian', '?')}, "
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
    """Encoder that uses precomputed features via lookup."""

    def __init__(self, features: torch.Tensor):
        super().__init__()
        # Store as float16 to save memory
        features_fp16 = features.to(torch.float16)
        if torch.cuda.is_available():
            features_fp16 = features_fp16.pin_memory()
        self.register_buffer("features", features_fp16, persistent=False)
        self.n_features = features.shape[1]

    def forward(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        device = coords.device
        idx_cpu = indices.detach().cpu()
        features = self.features[idx_cpu].to(device=device, dtype=torch.float32, non_blocking=True)
        return features


# =============================================================================
# Data Loading
# =============================================================================

def load_mss_data(data_path: str, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Load MSS data from DRF Experiment 1."""
    exp1_dir = os.path.join(data_path, "exp1")

    hdf5_path = os.path.join(exp1_dir, "along_track_sample_from_mss_ground_ABC.h5")
    obs_data = pd.read_hdf(hdf5_path, "data")

    if verbose:
        print(f"Loaded {len(obs_data):,} synthetic observations")

    coords = obs_data[["lon", "lat"]].values.astype(np.float32)
    targets = obs_data["obs"].values.astype(np.float32)

    if verbose:
        print(f"  Longitude range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
        print(f"  Latitude range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
        print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")

    return {'coords': coords, 'targets': targets}


# =============================================================================
# Training Functions
# =============================================================================

def create_data_subset(
    dataset: Dataset,
    fraction: float,
    batch_size: int,
    seed: int,
    num_workers: int = 8
) -> DataLoader:
    """Create a random subset of the dataset."""
    n_total = len(dataset)
    n_subset = max(1, int(fraction * n_total))

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_subset, replace=False)

    subset = Subset(dataset, indices.tolist())
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
    patience: int = 30,
    verbose: bool = True
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
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float
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

    # Metrics on normalized values
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    ss_tot = np.sum((targets - targets.mean()) ** 2)
    ss_res = np.sum((targets - predictions) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Denormalize for physical units
    predictions_raw = predictions * y_std + y_mean
    targets_raw = targets * y_std + y_mean
    rmse_raw = np.sqrt(np.mean((predictions_raw - targets_raw) ** 2))
    mae_raw = np.mean(np.abs(predictions_raw - targets_raw))

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'rmse_raw': float(rmse_raw),
        'mae_raw': float(mae_raw),
        'predictions': predictions,
        'targets': targets,
    }


# =============================================================================
# Visualization
# =============================================================================

def save_orthographic_comparison(
    lon: np.ndarray,
    lat: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    L_slepian: int,
    r2: float,
    rmse: float,
    save_path: str,
    train_pct: str
):
    """Create camera-ready orthographic comparison plot."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping polar plot")
        return

    # Subsample
    n_points = len(lon)
    max_points = 40000
    if n_points > max_points:
        step = n_points // max_points
        lon = lon[::step]
        lat = lat[::step]
        ground_truth = ground_truth[::step]
        predictions = predictions[::step]

    # Compute colorbar limits from GROUND TRUTH only (not predictions)
    vmin, vmax = np.percentile(ground_truth, [2, 98])

    fig = plt.figure(figsize=(6, 6))

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })

    # Single panel: predictions only, zoomed into north pole
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
        central_longitude=0,
        central_latitude=75
    ))

    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='#606060', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#606060')
    ax.gridlines(draw_labels=False, linewidth=0.15, color='#a0a0a0', alpha=0.4, linestyle='-')

    sc = ax.scatter(lon, lat, c=predictions, s=1, alpha=0.95,
                    cmap='RdBu_r', vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree(), rasterized=True)

    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Arctic MSS with Slepian Features")

    # Data configuration
    parser.add_argument("--data-path", type=str,
                        default="/scratch/local/arra4944_images/drf/Experiment_data",
                        help="Path to DRF experiment data")

    # Feature configuration
    parser.add_argument("--L-global", type=int, default=10,
                        help="Max degree for global SH (fixed at 10)")
    parser.add_argument("--L-slepian", type=int, default=40,
                        help="Max degree for Slepian functions (40, 80, or 120)")
    parser.add_argument("--cap-radius", type=float, default=25.0,
                        help="Cap radius in degrees (25° covers 65°N-90°N)")
    parser.add_argument("--num-modes", type=int, default=None,
                        help="Number of Slepian modes (default: auto based on L)")
    parser.add_argument("--lambda-thresh", type=float, default=0.05,
                        help="Eigenvalue threshold for mode selection")
    parser.add_argument("--cap-center-lon", type=float, default=0.0,
                        help="Cap center longitude (default: 0.0 for North Pole)")
    parser.add_argument("--cap-center-lat", type=float, default=90.0,
                        help="Cap center latitude (default: 90.0 for North Pole)")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Directory for feature cache")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recomputation even if cache exists")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=16)

    # Experiment configuration
    parser.add_argument("--label-fracs", type=str, default="0.02,0.05,0.10,1.00",
                        help="Comma-separated training fractions")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)

    # Quick test mode (for sanity checks - reduces data size)
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test mode: use 1%% val/test, subsample train for feature computation")

    # Output configuration
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to save CSV results (default: auto)")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help="Directory for figures (default: auto)")
    parser.add_argument("--results-json", type=str, default=None,
                        help="Path to save JSON results (default: auto)")

    # Architecture selection
    parser.add_argument("--arch", type=str, default="mlp",
                       choices=["linear", "mlp", "resmlp", "siren", "glu"],
                       help="Neural network architecture (default: mlp)")

    args = parser.parse_args()

    # Auto-set num_modes based on L_slepian if not specified
    # Shannon number for a cap ≈ (L+1)² * (1 - cos(theta)) / 2
    if args.num_modes is None:
        theta_rad = args.cap_radius * np.pi / 180.0
        shannon_approx = int((args.L_slepian + 1) ** 2 * (1 - np.cos(theta_rad)) / 2)
        # Use Shannon number directly - eigenvalue filtering handles mode selection
        args.num_modes = min(shannon_approx, (args.L_slepian + 1) ** 2)
        print(f"Auto-set num_modes={args.num_modes} (Shannon≈{shannon_approx})")

    # Auto-set output paths
    if args.csv_path is None:
        args.csv_path = f"results/mss/slepian_L{args.L_slepian}_results.csv"
    if args.fig_dir is None:
        args.fig_dir = f"results/mss/figs_slepian_L{args.L_slepian}"
    if args.results_json is None:
        args.results_json = f"results/mss/slepian_L{args.L_slepian}_results.json"

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MSS data
    print("\n" + "=" * 70)
    print("Loading Arctic Mean Sea Surface (MSS) Data")
    print("=" * 70)
    data = load_mss_data(args.data_path)

    coords = data['coords']
    targets = data['targets']

    # Normalize targets
    y_mean = targets.mean()
    y_std = targets.std()
    targets_norm = (targets - y_mean) / y_std

    print(f"\nTarget normalization: mean={y_mean:.6f}, std={y_std:.4f}")

    # Quick test mode: override splits
    if args.quick_test:
        print("\n*** QUICK TEST MODE: Using reduced data sizes ***")
        args.test_split = 0.01
        args.val_split = 0.01

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        coords, targets_norm, test_size=args.test_split, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_split, random_state=args.seed
    )

    # Quick test mode: subsample training data for feature computation
    if args.quick_test:
        label_fracs = [float(x) for x in args.label_fracs.split(',')]
        max_frac = max(label_fracs)
        n_subsample = max(1, int(max_frac * len(X_train)))

        rng = np.random.default_rng(args.seed)
        subsample_idx = rng.choice(len(X_train), size=n_subsample, replace=False)

        X_train = X_train[subsample_idx]
        y_train = y_train[subsample_idx]

        print(f"  Subsampled training to {len(X_train):,} samples (max label_frac={max_frac})")

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")

    # Combine all coordinates for feature computation
    all_coords = np.vstack([X_train, X_val, X_test])
    n_train = len(X_train)
    n_val = len(X_val)

    # Compute coverage
    cap_center = (args.cap_center_lon, args.cap_center_lat)
    coverage_pct = compute_coverage(all_coords, cap_center, args.cap_radius)
    print(f"Cap coverage: {coverage_pct:.1f}% of data points within cap")

    # Cache path (include quicktest suffix if in quick test mode)
    cache_suffix = "_quicktest" if args.quick_test else ""
    cache_path = os.path.join(
        args.cache_dir,
        f"mss_slepian_Lg{args.L_global}_Ls{args.L_slepian}_r{args.cap_radius}{cache_suffix}_features.pt"
    )

    # Compute or load features
    if os.path.exists(cache_path) and not args.force_recompute:
        print(f"\nLoading cached features...")
        feature_data = load_cached_features(cache_path)
    else:
        if not HAVE_PYSH:
            raise RuntimeError("PySHTOOLS required to compute Slepian features")

        print(f"\nComputing features from scratch...")
        feature_data = compute_and_cache_features(
            all_coords,
            L_global=args.L_global,
            L_slepian=args.L_slepian,
            cap_radius_deg=args.cap_radius,
            num_modes=args.num_modes,
            lambda_thresh=args.lambda_thresh,
            cap_center=cap_center,
            cache_path=cache_path
        )

    features = feature_data['features']
    feat_metadata = feature_data['metadata']

    print(f"\nFeature dimension: {features.shape[1]}")
    print(f"  Global SH: {feat_metadata.get('global_dim', '?')}")
    print(f"  Slepian: {feat_metadata.get('slepian_dim', '?')}")

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

    # Fixed loaders
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # Create output directories
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    # Run experiments
    label_fracs = [float(x) for x in args.label_fracs.split(',')]
    csv_results = []

    print(f"\n{'=' * 70}")
    print(f"Running experiments with L_global={args.L_global}, L_slepian={args.L_slepian}")
    print(f"Training fractions: {label_fracs}")
    print(f"Runs per configuration: {args.n_runs}")
    print(f"{'=' * 70}")

    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        print(f"\n{'=' * 60}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
        print(f"{'=' * 60}")

        for frac in label_fracs:
            # Format percentage string for small fractions (e.g., 0.1% -> "0_1pct")
            pct_val = frac * 100
            if pct_val >= 1:
                pct_str = f"{int(pct_val)}"
                pct_file = f"{int(pct_val)}pct"
            else:
                pct_str = f"{pct_val:.2f}".rstrip('0').rstrip('.')
                pct_file = f"{pct_val:.4f}".rstrip('0').rstrip('.').replace('.', '_') + "pct"
            print(f"\n[Slepian L={args.L_slepian}] Training with {pct_str}% of data...")

            # Create subset loader
            train_subset_loader = create_data_subset(
                train_dataset, frac, args.batch_size, run_seed, args.num_workers
            )

            n_train_samples = len(train_subset_loader.dataset)
            print(f"  Training samples: {n_train_samples:,}")

            # Create fresh model
            model = build_indexed_location_model(
                encoder, task="regression", arch=args.arch,
                hidden_dim=args.hidden_dim, dropout=args.dropout
            ).to(device)

            # Train
            t_start = time.time()
            train_losses, val_losses = train_model(
                model, train_subset_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                verbose=(run_idx == 0 and frac == label_fracs[-1])
            )
            train_time = time.time() - t_start

            # Evaluate
            metrics = evaluate_model(model, test_loader, device, y_mean, y_std)

            print(f"  R²={metrics['r2']:.4f}, MSE={metrics['mse']:.6f}, "
                  f"RMSE={metrics['rmse_raw']:.4f}, Time={train_time:.1f}s")

            # Save orthographic plot (first run only)
            if run_idx == 0:
                preds_raw = metrics['predictions'] * y_std + y_mean
                targets_raw = metrics['targets'] * y_std + y_mean

                save_path = os.path.join(args.fig_dir, f"polar_slepian_L{args.L_slepian}_{pct_file}.png")
                save_orthographic_comparison(
                    X_test[:, 0], X_test[:, 1],
                    targets_raw, preds_raw,
                    args.L_slepian, metrics['r2'], metrics['rmse_raw'],
                    save_path, pct_str
                )

            # Record results
            csv_results.append({
                'method': 'slepian_cap',
                'arch': args.arch,
                'L_global': args.L_global,
                'L_slepian': args.L_slepian,
                'cap_radius': args.cap_radius,
                'cap_center_lon': args.cap_center_lon,
                'cap_center_lat': args.cap_center_lat,
                'coverage_pct': coverage_pct,
                'feature_dim': features.shape[1],
                'global_dim': feat_metadata.get('global_dim', 0),
                'slepian_dim': feat_metadata.get('slepian_dim', 0),
                'run': run_idx + 1,
                'seed': run_seed,
                'train_frac': frac,
                'train_fraction': frac,
                'train_percent': pct_str,
                'train_samples': n_train_samples,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'rmse_raw': metrics['rmse_raw'],
                'mae_raw': metrics['mae_raw'],
                'train_loss': train_losses[-1] if train_losses else 0,
                'val_loss': val_losses[-1] if val_losses else 0,
                'train_time_sec': train_time
            })

    # Save CSV results
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv(args.csv_path, index=False)
    print(f"\nSaved results to {args.csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Mean R² by Training Fraction")
    print("=" * 70)
    summary = df_results.groupby('train_percent')[['r2', 'rmse_raw', 'mae_raw']].agg(['mean', 'std'])
    print(summary.round(4).to_string())

    # Save JSON metadata
    if args.results_json:
        json_data = {
            'method': 'slepian_cap',
            'dataset': 'arctic_mss',
            'configuration': {
                'L_global': args.L_global,
                'L_slepian': args.L_slepian,
                'cap_radius': args.cap_radius,
                'cap_center': [args.cap_center_lon, args.cap_center_lat],
                'coverage_pct': coverage_pct,
                'num_modes_initial': args.num_modes,
                'lambda_thresh': args.lambda_thresh,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'patience': args.patience,
                'lr': args.lr,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'n_runs': args.n_runs,
                'test_split': args.test_split,
                'val_split': args.val_split,
            },
            'feature_metadata': feat_metadata,
            'data_stats': {
                'total_samples': len(coords),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'y_mean': float(y_mean),
                'y_std': float(y_std),
            },
            'label_fractions': label_fracs,
            'csv_path': args.csv_path,
            'device': str(device)
        }

        os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
        with open(args.results_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved metadata to {args.results_json}")

    print(f"\n{'=' * 70}")
    print("Experiment Complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
