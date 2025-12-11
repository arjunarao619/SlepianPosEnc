#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
California Housing with vanilla Spherical Harmonics.
Clean implementation for comparison with Slepian approach.
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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Performance settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

from spherical_harmonics_ylm import SH as SH_analytic

# Import nn module for architecture selection
from nn import build_location_model

# =============================================================================
# Spherical Harmonics
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
        theta = torch.deg2rad(lat + 90.0)
        
        # Tiny clamp to avoid issues near poles
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
    Same architecture as Slepian version for fair comparison.
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
# Training and Evaluation
# =============================================================================

def create_data_subset(
    dataset: Dataset,
    fraction: float,
    batch_size: int,
    seed: int,
    num_workers: int = 8
) -> DataLoader:
    """
    Create a random subset of the dataset.
    
    Args:
        dataset: Full dataset
        fraction: Fraction to keep (0-1)
        batch_size: Batch size for loader
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader for the subset
    """
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
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        verbose: Print progress
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
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
    y_min: float,
    y_max: float
) -> Dict:
    """
    Evaluate model and compute metrics.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        y_min: Minimum target value (for denormalization)
        y_max: Maximum target value (for denormalization)
    
    Returns:
        Dictionary with metrics
    """
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
    
    # Compute metrics
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
    ax.set_title('Training Curves', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


def save_predictions_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str
):
    """Save predictions vs targets plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Predictions vs True Values', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved predictions plot to {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="California Housing with Vanilla SH")
    
    # Model configuration
    parser.add_argument("--L", type=int, default=10,
                       help="Maximum degree for spherical harmonics")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
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
    
    # Output configuration
    parser.add_argument("--csv-path", type=str, default='results/cali/results.csv',
                       help="Path to save CSV results")
    parser.add_argument("--fig-dir", type=str, default="results/cali/figs_vanilla_sh",
                       help="Directory for figures")
    parser.add_argument("--results-json", type=str, default='results/cali/results.json',
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
    
    # Load California housing data
    print("Loading California housing data...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    coords = df[["Longitude", "Latitude"]].values.astype(np.float32)
    targets = df["MedHouseVal"].values.astype(np.float32)
    
    # Normalize targets to [0, 1]
    y_min, y_max = targets.min(), targets.max()
    targets_norm = (targets - y_min) / (y_max - y_min + 1e-12)
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(coords)}")
    print(f"  Longitude range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  Latitude range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  Target range: ${y_min:.0f}k - ${y_max:.0f}k")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        coords, targets_norm, test_size=0.4, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )
    
    print(f"Dataset splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Create datasets
    from torch.utils.data import TensorDataset
    
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
    
    print(f"\nRunning experiments with L={args.L} (dim={args.L**2})")
    print(f"Training fractions: {label_fracs}")
    print(f"Runs per configuration: {args.n_runs}")
    
    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
        print(f"{'='*60}")
        
        for frac in label_fracs:
            pct = int(frac * 100)
            print(f"\nTraining with {pct}% of data...")
            
            # Create subset loader with run-specific seed
            train_subset_loader = create_data_subset(
                train_dataset, frac, args.batch_size, run_seed, args.num_workers
            )
            
            # Create fresh model
            encoder = VanillaSphericalHarmonics(L=args.L, verbose=(run_idx == 0 and frac == label_fracs[0]))
            model = build_location_model(
                encoder, task="regression", arch=args.arch,
                hidden_dim=args.hidden_dim, dropout=args.dropout
            ).to(device)
            
            # Train
            t_start = time.time()
            train_losses, val_losses = train_model(
                model, train_subset_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, patience=args.patience,
                verbose=(run_idx == 0)  # Only verbose on first run
            )
            train_time = time.time() - t_start
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device, y_min, y_max)
            
            print(f"[{pct:3d}%] R²={metrics['r2']:.4f}, "
                  f"MSE={metrics['mse']:.6f}, "
                  f"MAE=${metrics['mae_dollars']:,.0f}, "
                  f"Time={train_time:.1f}s")
            
            # Save plots (only on first run to avoid overwriting)
            if run_idx == 0:
                save_path = os.path.join(args.fig_dir, f"train_curves_{pct}pct.png")
                save_training_curves(train_losses, val_losses, save_path)
                
                save_path = os.path.join(args.fig_dir, f"predictions_{pct}pct.png")
                save_predictions_plot(metrics['predictions'], metrics['targets'], save_path)
            
            # Record results
            csv_results.append({
                'method': 'vanilla_sh',
                'arch': args.arch,
                'L': args.L,
                'feature_dim': args.L ** 2,
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
        summary = df_results.groupby('train_percent')[['r2', 'mae', 'mae_dollars']].agg(['mean', 'std'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.round(4)
        print(summary)
        
        # Print best performance
        best_idx = df_results.groupby('train_percent')['r2'].mean().idxmax()
        best_r2 = df_results[df_results['train_percent'] == best_idx]['r2'].mean()
        print(f"\nBest average R²: {best_r2:.4f} at {best_idx}% training data")
    
    # Save JSON metadata
    if args.results_json:
        json_data = {
            'method': 'vanilla_sh',
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
            },
            'label_fractions': label_fracs,
            'csv_path': args.csv_path,
            'device': str(device)
        }
        
        os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
        with open(args.results_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved metadata to {args.results_json}")


if __name__ == "__main__":
    main()