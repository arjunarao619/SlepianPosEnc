#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
California Housing with cached Slepian-cap features (PySHTOOLS).
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

# Import shared utilities
from utils.geo import compute_coverage


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
