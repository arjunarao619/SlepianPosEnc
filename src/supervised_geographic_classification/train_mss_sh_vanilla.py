#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arctic Mean Sea Surface (MSS) with vanilla Spherical Harmonics.

Clean implementation for benchmarking geographic location encoders.
Uses synthetic MSS data from DRF paper (Exp1) - ideal for testing
spatial interpolation because:
- Known ground truth field available
- Controlled noise (sigma=0.01)
- High signal-to-noise ratio (~22:1)
- Strong spatial structure (lat correlation r=-0.40)

Usage:
    python train_mss_sh_vanilla.py --L 10 --data-path /path/to/data
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
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Performance settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

from spherical_harmonics_ylm import SH as SH_analytic

# Import nn module for architecture selection
from nn import build_location_model


# =============================================================================
# Spherical Harmonics Encoder
# =============================================================================

class VanillaSphericalHarmonics(nn.Module):
    """
    Standard spherical harmonics encoder up to degree L.
    Computes features on-the-fly (not cached).
    """

    def __init__(self, L: int = 10, verbose: bool = True):
        super().__init__()
        self.L = int(L)
        self.n_features = self.L * self.L

        if verbose:
            print(f"Vanilla SH Encoder:")
            print(f"  L={self.L}")
            print(f"  Feature dimension: {self.n_features}")

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        """
        Compute SH features for input coordinates.

        Args:
            lonlat: [batch, 2] tensor of (longitude, latitude) in degrees

        Returns:
            [batch, L^2] tensor of SH features
        """
        batch_shape = lonlat.shape[:-1]
        coords = lonlat.reshape(-1, 2)

        # Convert to spherical coordinates
        lon = coords[:, 0]
        lat = coords[:, 1]
        phi = torch.deg2rad(lon + 180.0)
        theta = torch.deg2rad(90.0 - lat)  # Colatitude

        # Clamp to avoid issues near poles
        eps = 1e-12
        theta = torch.clamp(theta, eps, math.pi - eps)

        # Compute SH features
        features = []
        for l in range(self.L):
            for m in range(-l, l+1):
                y = SH_analytic(m, l, phi, theta)
                # Handle scalar returns
                if not torch.is_tensor(y):
                    y = torch.full_like(theta, float(y))
                elif y.ndim == 0:
                    y = torch.full_like(theta, y.item())
                features.append(y)

        features = torch.stack(features, dim=-1)  # [N, L^2]
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.reshape(*batch_shape, -1)


# =============================================================================
# Model
# =============================================================================

class LocationRegressor(nn.Module):
    """
    MLP regressor on top of location encoder.
    """

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
        return self.mlp(features).squeeze(-1)


# =============================================================================
# Data Loading
# =============================================================================

def load_mss_data(
    data_path: str,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load MSS (Mean Sea Surface) data from DRF Experiment 1.

    This is synthetic data:
    - Ground truth MSS field from 12 years of altimetry
    - Artificial observations along satellite tracks
    - Controlled i.i.d. Gaussian noise (sigma=0.01)

    Args:
        data_path: Path to DRF experiment data directory
        verbose: Print loading statistics

    Returns:
        Dictionary with coordinates, targets, and test grid data
    """
    exp1_dir = os.path.join(data_path, "exp1")

    # Load observation data (synthetic noisy measurements)
    hdf5_path = os.path.join(exp1_dir, "along_track_sample_from_mss_ground_ABC.h5")
    obs_data = pd.read_hdf(hdf5_path, "data")

    if verbose:
        print(f"Loaded {len(obs_data):,} synthetic observations")
        print(f"Columns: {obs_data.columns.tolist()}")

    # Extract coordinates and target
    # Use 'obs' column - this is what DRF code uses
    coords = obs_data[["lon", "lat"]].values.astype(np.float32)
    targets = obs_data["obs"].values.astype(np.float32)

    if verbose:
        print(f"\nObservation data statistics:")
        print(f"  Longitude range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
        print(f"  Latitude range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
        print(f"  Target (obs) range: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"  Target mean: {targets.mean():.6f}, std: {targets.std():.4f}")

    # Load test grid locations
    test_path = os.path.join(exp1_dir, "test_locs.csv")
    test_locs = pd.read_csv(test_path)
    test_coords = test_locs[["lon", "lat"]].values.astype(np.float32)

    if verbose:
        print(f"\nTest grid: {len(test_coords):,} locations")
        print(f"  Lat range: [{test_locs['lat'].min():.2f}, {test_locs['lat'].max():.2f}]")

    # Load ground truth field for evaluation
    gt_path = os.path.join(exp1_dir, "CryosatMSS-arco-2yr-140821_with_geoid_h.csv")
    gt_df = pd.read_csv(gt_path)

    if verbose:
        print(f"\nGround truth field: {len(gt_df):,} grid points")
        print(f"  MSS range: [{gt_df['mss'].min():.2f}, {gt_df['mss'].max():.2f}]")

    return {
        'coords': coords,
        'targets': targets,
        'test_coords': test_coords,
        'test_locs_df': test_locs,
        'gt_df': gt_df,
    }


# =============================================================================
# Training and Evaluation
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
    patience: int = 20,
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
                    print(f"Early stopping at epoch {epoch}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # Load best model
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

    for coords, y in test_loader:
        coords = coords.to(device, non_blocking=True)
        pred = model(coords)
        predictions.append(pred.cpu())
        targets.append(y)

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    # Compute metrics on normalized values
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # R²
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
def predict_on_grid(
    model: nn.Module,
    test_coords: np.ndarray,
    device: torch.device,
    batch_size: int = 4096
) -> np.ndarray:
    """Generate predictions on test grid."""
    model.eval()
    predictions = []

    test_tensor = torch.tensor(test_coords, dtype=torch.float32)
    n_samples = len(test_tensor)

    for i in range(0, n_samples, batch_size):
        batch = test_tensor[i:i+batch_size].to(device)
        pred = model(batch)
        predictions.append(pred.cpu().numpy())

    return np.concatenate(predictions)


# =============================================================================
# Visualization
# =============================================================================

def save_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str
):
    """Save training curves plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Curves - MSS Prediction', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


def save_polar_comparison_plot(
    test_locs_df: pd.DataFrame,
    predictions: np.ndarray,
    y_mean: float,
    y_std: float,
    save_path: str,
    title_suffix: str = ""
):
    """
    Create orthographic projection comparing ground truth (if available)
    and predicted MSS values.

    Args:
        test_locs_df: DataFrame with test locations (lon, lat)
        predictions: Predicted values (normalized)
        y_mean: Mean for denormalization
        y_std: Std for denormalization
        save_path: Path to save figure
        title_suffix: Additional title text
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping polar plot")
        return

    # Denormalize predictions
    pred_raw = predictions * y_std + y_mean

    lon = test_locs_df['lon'].values
    lat = test_locs_df['lat'].values

    # Subsample for faster plotting
    n_points = len(lon)
    if n_points > 50000:
        step = n_points // 50000
        lon = lon[::step]
        lat = lat[::step]
        pred_raw = pred_raw[::step]

    # Create figure with orthographic projection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))

    # Set extent and add features
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')

    # Percentile clipping for better visualization
    vmin, vmax = np.percentile(pred_raw, [2, 98])

    # Scatter plot
    sc = ax.scatter(lon, lat, c=pred_raw, s=2, alpha=0.8,
                    cmap='RdBu_r', vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree())

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05,
                        shrink=0.6, aspect=30)
    cbar.set_label('Predicted MSS (normalized)', fontsize=12)

    # Title
    ax.set_title(f'Predicted Mean Sea Surface{title_suffix}\n'
                 f'({len(lon):,} grid points)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved polar plot to {save_path}")


def save_polar_gt_vs_pred_plot(
    test_locs_df: pd.DataFrame,
    predictions: np.ndarray,
    targets: np.ndarray,
    y_mean: float,
    y_std: float,
    save_path: str,
    L: int,
    r2: float,
    rmse: float,
    train_pct: str
):
    """
    Create camera-ready orthographic comparison plot matching slepian script style.

    Args:
        test_locs_df: DataFrame with test locations (lon, lat)
        predictions: Predicted values (normalized)
        targets: Target values (normalized) - from holdout
        y_mean: Mean for denormalization
        y_std: Std for denormalization
        save_path: Path to save figure
        L: Spherical harmonic degree
        r2: R² score
        rmse: RMSE value
        train_pct: Training percentage string
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping polar plot")
        return

    lon = test_locs_df['lon'].values
    lat = test_locs_df['lat'].values

    # Use NORMALIZED values (not denormalized) to match slepian script
    # This ensures ground truth visualization is not corrupted by bad predictions
    ground_truth = targets.copy()
    preds = predictions.copy()

    # Subsample for faster plotting
    n_points = len(lon)
    max_points = 40000
    if n_points > max_points:
        step = n_points // max_points
        lon = lon[::step]
        lat = lat[::step]
        ground_truth = ground_truth[::step]
        preds = preds[::step]

    # Compute colorbar limits from GROUND TRUTH only (not predictions)
    # This prevents catastrophic predictions from corrupting the visualization
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

    sc = ax.scatter(lon, lat, c=preds, s=0.3, alpha=0.95,
                    cmap='RdBu_r', vmin=vmin, vmax=vmax,
                    transform=ccrs.PlateCarree(), rasterized=True)

    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print(f"Saved polar plot to {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Arctic MSS with Vanilla SH")

    # Data configuration
    parser.add_argument("--data-path", type=str,
                        default="/scratch/local/arra4944_images/drf/Experiment_data",
                        help="Path to DRF experiment data")

    # Model configuration
    parser.add_argument("--L", type=int, default=10,
                        help="Maximum degree for spherical harmonics")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size (larger for 1M+ samples)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (increased for stability)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=16)

    # Experiment configuration
    parser.add_argument("--label-fracs", type=str, default="0.01,0.10,0.25,0.50,0.75,1.00",
                        help="Comma-separated training fractions")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of runs per configuration for variability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Fraction of data for test set")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of remaining data for validation")

    # Output configuration
    parser.add_argument("--csv-path", type=str, default='results/mss/results.csv',
                        help="Path to save CSV results")
    parser.add_argument("--fig-dir", type=str, default="results/mss/figs_vanilla_sh",
                        help="Directory for figures")
    parser.add_argument("--results-json", type=str, default='results/mss/results.json',
                        help="Path to save JSON results")

    # Architecture selection
    parser.add_argument("--arch", type=str, default="mlp",
                       choices=["mlp", "resmlp", "siren", "glu"],
                       help="Neural network architecture (default: mlp)")

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MSS data
    print("\n" + "="*70)
    print("Loading Arctic Mean Sea Surface (MSS) Data")
    print("="*70)
    data = load_mss_data(args.data_path)

    coords = data['coords']
    targets = data['targets']
    test_grid_coords = data['test_coords']
    test_locs_df = data['test_locs_df']

    # Normalize targets (z-score normalization)
    y_mean = targets.mean()
    y_std = targets.std()
    targets_norm = (targets - y_mean) / y_std

    print(f"\nTarget normalization (z-score):")
    print(f"  Mean: {y_mean:.6f}, Std: {y_std:.4f}")

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        coords, targets_norm, test_size=args.test_split, random_state=args.seed
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_split, random_state=args.seed
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train):,} ({100*(1-args.test_split)*(1-args.val_split):.0f}%)")
    print(f"  Val: {len(X_val):,} ({100*(1-args.test_split)*args.val_split:.0f}%)")
    print(f"  Test: {len(X_test):,} ({100*args.test_split:.0f}%)")
    print(f"  Test grid: {len(test_grid_coords):,} (for visualization)")

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

    # Fixed validation and test loaders
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

    # Create output directory
    os.makedirs(args.fig_dir, exist_ok=True)

    # Run experiments
    label_fracs = [float(x) for x in args.label_fracs.split(',')]
    csv_results = []

    print(f"\n{'='*70}")
    print(f"Running experiments with L={args.L} (dim={args.L**2})")
    print(f"Training fractions: {label_fracs}")
    print(f"Runs per configuration: {args.n_runs}")
    print(f"{'='*70}")

    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
        print(f"{'='*60}")

        for frac in label_fracs:
            # Format percentage string for small fractions (e.g., 0.1% -> "0_1pct")
            pct_val = frac * 100
            if pct_val >= 1:
                pct_str = f"{int(pct_val)}"
                pct_file = f"{int(pct_val)}pct"
            else:
                pct_str = f"{pct_val:.2f}".rstrip('0').rstrip('.')
                pct_file = f"{pct_val:.4f}".rstrip('0').rstrip('.').replace('.', '_') + "pct"
            print(f"\nTraining with {pct_str}% of data...")

            # Create subset loader with run-specific seed
            train_subset_loader = create_data_subset(
                train_dataset, frac, args.batch_size, run_seed, args.num_workers
            )

            n_train_samples = len(train_subset_loader.dataset)
            print(f"  Training samples: {n_train_samples:,}")

            # Create fresh model
            encoder = VanillaSphericalHarmonics(
                L=args.L,
                verbose=(run_idx == 0 and frac == label_fracs[0])
            )
            model = build_location_model(
                encoder, task="regression", arch=args.arch,
                hidden_dim=args.hidden_dim, dropout=args.dropout
            ).to(device)

            # Train
            t_start = time.time()
            train_losses, val_losses = train_model(
                model, train_subset_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                verbose=(run_idx == 0)
            )
            train_time = time.time() - t_start

            # Evaluate on test set
            metrics = evaluate_model(model, test_loader, device, y_mean, y_std)

            print(f"[{pct_str:>6}%] R²={metrics['r2']:.4f}, "
                  f"MSE={metrics['mse']:.6f}, "
                  f"MAE={metrics['mae_raw']:.4f}, "
                  f"RMSE={metrics['rmse_raw']:.4f}, "
                  f"Time={train_time:.1f}s")

            # Save plots (only on first run)
            if run_idx == 0:
                # Training curves
                save_path = os.path.join(args.fig_dir, f"train_curves_{pct_file}.png")
                save_training_curves(train_losses, val_losses, save_path)

                # Polar plot - predictions on test holdout
                # Create a temporary dataframe for test holdout locations
                test_holdout_df = pd.DataFrame({
                    'lon': X_test[:, 0],
                    'lat': X_test[:, 1]
                })
                save_path = os.path.join(args.fig_dir, f"polar_vanilla_sh_L{args.L}_{pct_file}.png")
                save_polar_gt_vs_pred_plot(
                    test_holdout_df,
                    metrics['predictions'],
                    metrics['targets'],
                    y_mean, y_std,
                    save_path,
                    L=args.L,
                    r2=metrics['r2'],
                    rmse=metrics['rmse_raw'],
                    train_pct=pct_str
                )

                # Also save predictions on the full test grid (for visualization)
                if frac == 1.0:  # Only for 100% training
                    grid_preds = predict_on_grid(model, test_grid_coords, device)
                    save_path = os.path.join(args.fig_dir, f"polar_grid_prediction.png")
                    save_polar_comparison_plot(
                        test_locs_df, grid_preds, y_mean, y_std,
                        save_path, title_suffix=f" (L={args.L}, Full Grid)"
                    )

            # Record results
            csv_results.append({
                'method': 'vanilla_sh',
                'arch': args.arch,
                'L': args.L,
                'feature_dim': (args.L + 1) ** 2,
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
    if args.csv_path:
        os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
        df_results = pd.DataFrame(csv_results)
        df_results.to_csv(args.csv_path, index=False)
        print(f"\nSaved results to {args.csv_path}")

        # Print summary statistics
        print("\nSummary Statistics (mean ± std across runs):")
        summary = df_results.groupby('train_percent')[['r2', 'mae_raw', 'rmse_raw']].agg(['mean', 'std'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        print(summary.round(4))

        # Print best performance
        best_idx = df_results.groupby('train_percent')['r2'].mean().idxmax()
        best_r2 = df_results[df_results['train_percent'] == best_idx]['r2'].mean()
        print(f"\nBest average R²: {best_r2:.4f} at {best_idx}% training data")

    # Save JSON metadata
    if args.results_json:
        json_data = {
            'method': 'vanilla_sh',
            'dataset': 'arctic_mss',
            'configuration': {
                'L': args.L,
                'feature_dim': args.L ** 2,
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
                'test_grid_samples': len(test_grid_coords),
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