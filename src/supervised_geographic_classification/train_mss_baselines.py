#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arctic Mean Sea Surface (MSS) with Baseline Positional Encoders.

Systematic comparison of positional encoding methods on synthetic MSS data
from DRF paper (Exp1). Generates camera-ready orthographic visualizations
for each encoder.

Usage:
    python train_mss_baselines.py --label-fracs "0.10,1.00" --epochs 100
"""

import os
import sys
import time
import json
import copy
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

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
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import baseline encoders
from pe_baselines import (
    Direct, Cartesian3D, Wrap, Grid,
    SphereC, SphereCPlus, SphereM, SphereMPlus, Theory,
    Wavelets
)

# Import nn module for architecture selection
from nn import build_location_model

# Performance settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

# =============================================================================
# Baseline Configurations
# =============================================================================
BASELINE_CONFIGS = {
    'direct': {
        'class': Direct,
        'params': {}
    },
    'cartesian3d': {
        'class': Cartesian3D,
        'params': {}
    },
    'wrap': {
        'class': Wrap,
        'params': {}
    },
    'grid': {
        'class': Grid,
        'params': {
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        }
    },
    'spherec': {
        'class': SphereC,
        'params': {
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        }
    },
    'spherecplus': {
        'class': SphereCPlus,
        'params': {
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        }
    },
    'spherem': {
        'class': SphereM,
        'params': {
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        }
    },
    'spheremplus': {
        'class': SphereMPlus,
        'params': {
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        }
    },
    'theory': {
        'class': Theory,
        'params': {
            'frequency_num': 16,
            'max_radius': 10000,
            'min_radius': 1000,
            'freq_init': 'geometric'
        }
    },
    'wavelets': {
        'class': Wavelets,
        'params': {
            'max_scale': 3,
            'max_rotations': 75,
            'k_val': 6,
            'scale_factor': 1.0,
            'scale_shift': 1,
            'dilation_step': 6,
            'wavelet_type': 'butterfly'
        }
    }
}


# =============================================================================
# Model
# =============================================================================
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

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.encoder(coords)
        # Ensure features are on the same device as the MLP
        mlp_device = next(self.mlp.parameters()).device
        features = features.to(device=mlp_device, dtype=coords.dtype)
        return self.mlp(features).squeeze(-1)


# =============================================================================
# Data Loading
# =============================================================================
def load_mss_data(data_path: str, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Load MSS data from DRF Experiment 1."""
    exp1_dir = os.path.join(data_path, "exp1")

    # Load observation data
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
        print(f"  Target mean: {targets.mean():.6f}, std: {targets.std():.4f}")

    # Load test grid for visualization
    test_path = os.path.join(exp1_dir, "test_locs.csv")
    test_locs = pd.read_csv(test_path)
    test_coords = test_locs[["lon", "lat"]].values.astype(np.float32)

    if verbose:
        print(f"Test grid: {len(test_coords):,} locations")

    return {
        'coords': coords,
        'targets': targets,
        'test_coords': test_coords,
        'test_locs_df': test_locs,
    }


# =============================================================================
# Training Functions
# =============================================================================
def create_data_subset(dataset, fraction, batch_size, seed, num_workers=8):
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


def train_model(model, train_loader, val_loader, device,
                epochs=100, lr=1e-3, patience=20, verbose=True):
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
        for coords, targets in train_loader:
            coords = coords.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(coords)
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
            for coords, targets in val_loader:
                coords = coords.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                predictions = model(coords)
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
def evaluate_model(model, test_loader, device, y_mean, y_std):
    """Evaluate model and compute metrics."""
    model.eval()
    predictions = []
    targets = []

    for coords, y in test_loader:
        coords = coords.to(device, non_blocking=True)
        pred = model(coords)
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


@torch.no_grad()
def predict_on_coords(model, coords: np.ndarray, device, batch_size=4096):
    """Generate predictions on arbitrary coordinates."""
    model.eval()
    predictions = []
    test_tensor = torch.tensor(coords, dtype=torch.float32)
    n_samples = len(test_tensor)

    for i in range(0, n_samples, batch_size):
        batch = test_tensor[i:i+batch_size].to(device)
        pred = model(batch)
        predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions)


# =============================================================================
# Visualization (Camera-Ready)
# =============================================================================
def save_orthographic_comparison(
    lon: np.ndarray,
    lat: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    encoder_name: str,
    r2: float,
    rmse: float,
    save_path: str,
    train_pct: str
):
    """
    Create camera-ready orthographic plot: predictions only, zoomed into north pole.
    Color scale based on ground truth.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping polar plot")
        return

    # Subsample for reasonable file size
    n_points = len(lon)
    max_points = 40000
    if n_points > max_points:
        step = n_points // max_points
        lon = lon[::step]
        lat = lat[::step]
        ground_truth = ground_truth[::step]
        predictions = predictions[::step]

    # Color scale from ground truth only
    vmin, vmax = np.percentile(ground_truth, [2, 98])

    fig = plt.figure(figsize=(6, 6))

    # Single panel: predictions only, zoomed into north pole
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
        central_longitude=0,
        central_latitude=75
    ))

    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='#606060', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#606060')
    ax.gridlines(draw_labels=False, linewidth=0.15, color='#a0a0a0', alpha=0.4, linestyle='-')

    ax.scatter(lon, lat, c=predictions, s=1, alpha=0.95,
               cmap='RdBu_r', vmin=vmin, vmax=vmax,
               transform=ccrs.PlateCarree(), rasterized=True)

    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_ground_truth_plot(
    lon: np.ndarray,
    lat: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str
):
    """Save ground truth as a separate zoomed-in globe image."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping ground truth plot")
        return

    # Subsample
    n_points = len(lon)
    max_points = 40000
    if n_points > max_points:
        step = n_points // max_points
        lon = lon[::step]
        lat = lat[::step]
        ground_truth = ground_truth[::step]

    vmin, vmax = np.percentile(ground_truth, [2, 98])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(
        central_longitude=0,
        central_latitude=75
    ))

    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='#606060', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='#606060')
    ax.gridlines(draw_labels=False, linewidth=0.15, color='#a0a0a0', alpha=0.4, linestyle='-')

    ax.scatter(lon, lat, c=ground_truth, s=1, alpha=0.95,
               cmap='RdBu_r', vmin=vmin, vmax=vmax,
               transform=ccrs.PlateCarree(), rasterized=True)

    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"  Saved ground truth: {save_path}")


def save_colorbar(vmin: float, vmax: float, save_path: str):
    """Save a standalone vertical colorbar as a separate image."""
    fig, ax = plt.subplots(figsize=(1.0, 4.0))

    # Create colorbar using a ScalarMappable
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_label('Normalized MSS', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f"  Saved colorbar: {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Arctic MSS with Baseline Encoders")

    # Data configuration
    parser.add_argument("--data-path", type=str,
                        default="/scratch/local/arra4944_images/drf/Experiment_data",
                        help="Path to DRF experiment data")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=16)

    # Experiment configuration
    parser.add_argument("--label-fracs", type=str, default="0.02,0.05,0.1,1.0",
                        help="Comma-separated training fractions")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)

    # Encoder selection
    parser.add_argument("--encoders", type=str, default="all",
                        help="Comma-separated list or 'all'")

    # Output configuration
    parser.add_argument("--csv-path", type=str, default='results/mss/baselines_results.csv')
    parser.add_argument("--fig-dir", type=str, default="results/mss/figs_baselines")
    parser.add_argument("--results-json", type=str, default='results/mss/baselines_results.json')

    # Architecture selection
    parser.add_argument("--arch", type=str, default="mlp",
                       choices=["linear", "mlp", "resmlp", "siren", "glu"],
                       help="Neural network architecture (default: mlp)")

    # Resolution sweep parameters
    parser.add_argument("--max-radius", type=float, default=None,
                        help="Max radius for frequency encoders. "
                             "If set, overrides BASELINE_CONFIGS for grid/sphere* encoders.")
    parser.add_argument("--min-radius", type=float, default=None,
                        help="Min radius for frequency encoders. "
                             "If set, overrides BASELINE_CONFIGS for grid/sphere* encoders.")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create modifiable copy of baseline configs
    baseline_configs = copy.deepcopy(BASELINE_CONFIGS)

    # Override resolution if specified (for frequency-based encoders)
    if args.max_radius is not None or args.min_radius is not None:
        # Use provided values or defaults
        max_r = args.max_radius if args.max_radius is not None else 360.0
        min_r = args.min_radius if args.min_radius is not None else 1.0

        # Compute frequency_num from radius ratio (log scale)
        import math
        freq_num = max(4, int(math.log2(max_r / min_r) * 4))

        print(f"Resolution override: max_radius={max_r}, min_radius={min_r}, frequency_num={freq_num}")

        # Update frequency-based encoders
        freq_encoders = ['grid', 'spherec', 'spherecplus', 'spherem', 'spheremplus', 'theory']
        for enc_name in freq_encoders:
            if enc_name in baseline_configs:
                baseline_configs[enc_name]['params']['max_radius'] = max_r
                baseline_configs[enc_name]['params']['min_radius'] = min_r
                baseline_configs[enc_name]['params']['frequency_num'] = freq_num

    # Select encoders
    if args.encoders == "all":
        encoder_names = list(baseline_configs.keys())
    else:
        encoder_names = [e.strip() for e in args.encoders.split(',')]

    print(f"Testing encoders: {encoder_names}")

    # Load MSS data
    print("\n" + "="*70)
    print("Loading Arctic Mean Sea Surface (MSS) Data")
    print("="*70)
    data = load_mss_data(args.data_path)

    coords = data['coords']
    targets = data['targets']

    # Normalize targets
    y_mean = targets.mean()
    y_std = targets.std()
    targets_norm = (targets - y_mean) / y_std

    print(f"\nTarget normalization: mean={y_mean:.6f}, std={y_std:.4f}")

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        coords, targets_norm, test_size=args.test_split, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_split, random_state=args.seed
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
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

    # Save ground truth and colorbar (once, before running experiments)
    gt_denorm = y_test * y_std + y_mean
    vmin, vmax = np.percentile(gt_denorm, [2, 98])
    save_ground_truth_plot(X_test[:, 0], X_test[:, 1], gt_denorm,
                           os.path.join(args.fig_dir, "ground_truth.png"))
    save_colorbar(vmin, vmax, os.path.join(args.fig_dir, "colorbar.png"))

    # Run experiments
    label_fracs = [float(x) for x in args.label_fracs.split(',')]
    csv_results = []

    print(f"\n{'='*70}")
    print(f"Running experiments")
    print(f"Training fractions: {label_fracs}")
    print(f"Encoders: {len(encoder_names)}")
    print(f"{'='*70}")

    for encoder_name in encoder_names:
        config = baseline_configs[encoder_name]

        print(f"\n{'='*60}")
        print(f"Encoder: {encoder_name.upper()}")
        print(f"{'='*60}")

        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx

            for frac in label_fracs:
                # Format percentage string for small fractions (e.g., 0.1% -> "0_1pct")
                pct_val = frac * 100
                if pct_val >= 1:
                    pct_str = f"{int(pct_val)}"
                    pct_file = f"{int(pct_val)}pct"
                else:
                    pct_str = f"{pct_val:.2f}".rstrip('0').rstrip('.')
                    pct_file = f"{pct_val:.4f}".rstrip('0').rstrip('.').replace('.', '_') + "pct"
                print(f"\n[{encoder_name}] Run {run_idx+1}, {pct_str}% training data...")

                # Create subset loader
                train_subset_loader = create_data_subset(
                    train_dataset, frac, args.batch_size, run_seed, args.num_workers
                )

                n_train_samples = len(train_subset_loader.dataset)

                # Create encoder and model
                encoder = config['class'](**config['params'])
                model = build_location_model(
                    encoder, task="regression", arch=args.arch,
                    hidden_dim=args.hidden_dim, dropout=args.dropout
                ).to(device)

                print(f"  Encoder dim: {encoder.n_features}, Train samples: {n_train_samples:,}")

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
                    # Predict on test holdout for visualization
                    preds_test = predict_on_coords(model, X_test, device)

                    # Denormalize
                    preds_raw = preds_test * y_std + y_mean
                    targets_raw = y_test * y_std + y_mean

                    save_path = os.path.join(
                        args.fig_dir,
                        f"polar_{encoder_name}_{pct_file}.png"
                    )
                    save_orthographic_comparison(
                        X_test[:, 0], X_test[:, 1],
                        targets_raw, preds_raw,
                        encoder_name, metrics['r2'], metrics['rmse_raw'],
                        save_path, pct_str
                    )

                # Record results
                result_row = {
                    'method': encoder_name,  # For compatibility with combined CSV
                    'encoder': encoder_name,
                    'arch': args.arch,
                    'encoder_dim': encoder.n_features,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'train_frac': frac,  # For compatibility with combined CSV
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
                }

                # Add resolution info if applicable
                if 'max_radius' in config['params']:
                    result_row['max_radius'] = config['params']['max_radius']
                    result_row['min_radius'] = config['params']['min_radius']

                csv_results.append(result_row)

    # Save CSV results
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv(args.csv_path, index=False)
    print(f"\nSaved results to {args.csv_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Mean R² by Encoder and Training Fraction")
    print("="*70)
    summary = df_results.pivot_table(
        index='encoder', columns='train_percent', values='r2', aggfunc='mean'
    )
    print(summary.round(4).to_string())

    # Best encoder per fraction
    print("\nBest encoder by training fraction:")
    for pct in df_results['train_percent'].unique():
        subset = df_results[df_results['train_percent'] == pct]
        best = subset.loc[subset['r2'].idxmax()]
        print(f"  {int(pct):3d}%: {best['encoder']:12s} R²={best['r2']:.4f}")

    # Save JSON metadata
    if args.results_json:
        json_data = {
            'method': 'baseline_encoders',
            'dataset': 'arctic_mss',
            'encoders_tested': encoder_names,
            'configuration': {
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

    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
