#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
California Housing with baseline positional encoders.
Systematic comparison with Slepian and Spherical Harmonics approaches.
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
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Import baseline encoders
from pe_baselines import (
    Direct, Cartesian3D, Wrap, Grid,
    SphereC, SphereCPlus, SphereM, SphereMPlus, Theory
)

# Import nn module for architecture selection
from nn import build_location_model

# Performance settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

# ============================================================================
# Baseline Configurations (from paper hyperparameters)
# ============================================================================
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
    }
}

# ============================================================================
# Model
# ============================================================================
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
        return self.mlp(features).squeeze(-1)

# ============================================================================
# Training Functions (reuse from original)
# ============================================================================
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
                    print(f"Early stopping at epoch {epoch}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return train_losses, val_losses

@torch.no_grad()
def evaluate_model(model, test_loader, device, y_min, y_max):
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

# ============================================================================
# Main Experiment
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="California Housing with Baseline Encoders")
    
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
                       help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output configuration
    parser.add_argument("--csv-path", type=str, default='results/baselines/california_results.csv',
                       help="Path to save CSV results")
    parser.add_argument("--fig-dir", type=str, default="results/baselines/figs",
                       help="Directory for figures")
    
    # Encoder selection
    parser.add_argument("--encoders", type=str, default="all",
                       help="Comma-separated list of encoders or 'all'")

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
    
    # Select encoders
    if args.encoders == "all":
        encoder_names = list(BASELINE_CONFIGS.keys())
    else:
        encoder_names = args.encoders.split(',')
    
    print(f"Testing encoders: {encoder_names}")
    
    # Load California housing data
    print("Loading California housing data...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    coords = df[["Longitude", "Latitude"]].values.astype(np.float32)
    targets = df["MedHouseVal"].values.astype(np.float32)
    
    # Normalize targets
    y_min, y_max = targets.min(), targets.max()
    targets_norm = (targets - y_min) / (y_max - y_min + 1e-12)
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(coords)}")
    print(f"  Target range: ${y_min:.0f}k - ${y_max:.0f}k")
    
    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        coords, targets_norm, test_size=0.4, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )
    
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
    
    # Create output directory
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    
    # Run experiments
    label_fracs = [float(x) for x in args.label_fracs.split(',')]
    csv_results = []
    
    for encoder_name in encoder_names:
        print(f"\n{'='*60}")
        print(f"Testing encoder: {encoder_name.upper()}")
        print(f"{'='*60}")
        
        config = BASELINE_CONFIGS[encoder_name]
        
        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx
            print(f"\nRun {run_idx + 1}/{args.n_runs} (seed={run_seed})")
            
            for frac in label_fracs:
                pct = int(frac * 100)
                print(f"\nTraining with {pct}% of data...")
                
                # Create subset loader
                train_subset_loader = create_data_subset(
                    train_dataset, frac, args.batch_size, run_seed, args.num_workers
                )
                
                # Create encoder and model
                encoder = config['class'](**config['params'])
                model = build_location_model(
                    encoder, task="regression", arch=args.arch,
                    hidden_dim=args.hidden_dim, dropout=args.dropout
                ).to(device)
                
                print(f"Encoder dimension: {encoder.n_features}")
                
                # Train
                t_start = time.time()
                train_losses, val_losses = train_model(
                    model, train_subset_loader, val_loader, device,
                    epochs=args.epochs, lr=args.lr, patience=args.patience,
                    verbose=(run_idx == 0 and frac == label_fracs[0])
                )
                train_time = time.time() - t_start
                
                # Evaluate
                metrics = evaluate_model(model, test_loader, device, y_min, y_max)
                
                print(f"[{encoder_name}@{pct:3d}%] R²={metrics['r2']:.4f}, "
                      f"MSE={metrics['mse']:.6f}, "
                      f"MAE=${metrics['mae_dollars']:,.0f}, "
                      f"Time={train_time:.1f}s")
                
                # Record results
                csv_results.append({
                    'encoder': encoder_name,
                    'arch': args.arch,
                    'encoder_dim': encoder.n_features,
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
    
    # Save results
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv(args.csv_path, index=False)
    print(f"\nSaved results to {args.csv_path}")
    
    # Print summary
    print("\nSummary by Encoder (mean ± std across all runs and fractions):")
    summary = df_results.groupby('encoder')[['r2', 'mae_dollars']].agg(['mean', 'std'])
    print(summary)
    
    # Print best performance per training fraction
    print("\nBest R² by training fraction:")
    best_by_frac = df_results.groupby('train_percent').apply(
        lambda x: x.nlargest(1, 'r2')[['encoder', 'r2']].iloc[0]
    )
    print(best_by_frac)

if __name__ == "__main__":
    main()