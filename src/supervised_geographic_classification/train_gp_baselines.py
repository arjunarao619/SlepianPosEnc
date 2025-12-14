#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified training script for GP baselines.

Supports:
- exact_gp: Exact GP regression/classification (small datasets)
- svgp: Sparse Variational GP (scales to large datasets)
- planar_rff: Random Fourier Features with Bayesian linear regression
- spherical_svgp: SVGP with spherical Matern kernel
- spherical_rff: Spherical Random Fourier Features

Matches the interface of existing Slepian training scripts for easy comparison.

Example usage:
    # California Housing with SVGP
    python train_gp_baselines.py \
        --method svgp \
        --dataset california_housing \
        --kernel matern52 \
        --num-inducing 500 \
        --label-fracs 0.01,0.10,0.25,0.50,0.75,1.00 \
        --n-runs 3 \
        --csv-path results/gp_california_svgp.csv

    # Japan Prefecture with RFF
    python train_gp_baselines.py \
        --method planar_rff \
        --dataset japan_prefecture \
        --num-features 2000 \
        --n-runs 5 \
        --csv-path results/gp_japan_rff.csv
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Add parent directory to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Import GP baselines
from pe_baselines.gp import (
    # Utilities
    normalize_lonlat_planar,
    CoordinateTransformer,
    # Exact GP
    train_exact_gp_regression,
    train_exact_gp_classification,
    predict_regression,
    predict_classification,
    evaluate_regression,
    evaluate_classification,
    # SVGP
    train_svgp_regression,
    train_svgp_classification,
    predict_svgp_regression,
    predict_svgp_classification,
    evaluate_svgp_regression,
    evaluate_svgp_classification,
    # Planar RFF
    train_rff_regression,
    train_rff_classification,
    evaluate_rff_regression,
    evaluate_rff_classification,
    # Spherical SVGP
    train_spherical_svgp_regression,
    predict_spherical_svgp,
    evaluate_spherical_svgp,
    # Spherical RFF
    train_spherical_rff_regression,
    train_spherical_rff_classification,
    evaluate_spherical_rff_regression,
    evaluate_spherical_rff_classification,
)


def load_california_housing(seed: int = 42):
    """Load California Housing dataset."""
    print("Loading California Housing data...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    coords = df[["Longitude", "Latitude"]].values.astype(np.float32)
    targets = df["MedHouseVal"].values.astype(np.float32)

    # Normalize targets
    y_min, y_max = targets.min(), targets.max()
    targets_norm = (targets - y_min) / (y_max - y_min + 1e-12)

    # Train/val/test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        coords, targets_norm, test_size=0.4, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {
        'train_coords': torch.tensor(X_train, dtype=torch.float32),
        'train_targets': torch.tensor(y_train, dtype=torch.float32),
        'val_coords': torch.tensor(X_val, dtype=torch.float32),
        'val_targets': torch.tensor(y_val, dtype=torch.float32),
        'test_coords': torch.tensor(X_test, dtype=torch.float32),
        'test_targets': torch.tensor(y_test, dtype=torch.float32),
        'y_min': y_min,
        'y_max': y_max,
        'task': 'regression'
    }


def load_japan_prefecture(dataset_dir: str, seed: int = 42):
    """Load Japan Prefecture dataset."""
    print(f"Loading Japan Prefecture data from {dataset_dir}...")

    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(dataset_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

    with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_classes = metadata['n_prefectures']

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Classes: {n_classes}")

    return {
        'train_coords': torch.tensor(train_df[['longitude', 'latitude']].values, dtype=torch.float32),
        'train_targets': torch.tensor(train_df['prefecture_id'].values, dtype=torch.long),
        'val_coords': torch.tensor(val_df[['longitude', 'latitude']].values, dtype=torch.float32),
        'val_targets': torch.tensor(val_df['prefecture_id'].values, dtype=torch.long),
        'test_coords': torch.tensor(test_df[['longitude', 'latitude']].values, dtype=torch.float32),
        'test_targets': torch.tensor(test_df['prefecture_id'].values, dtype=torch.long),
        'num_classes': n_classes,
        'task': 'classification'
    }


def load_mss(data_path: str, seed: int = 42):
    """Load Arctic Mean Sea Surface (MSS) dataset."""
    print(f"Loading MSS data from {data_path}...")

    exp1_dir = os.path.join(data_path, "exp1")
    hdf5_path = os.path.join(exp1_dir, "along_track_sample_from_mss_ground_ABC.h5")

    obs_data = pd.read_hdf(hdf5_path, "data")

    coords = obs_data[["lon", "lat"]].values.astype(np.float32)
    targets = obs_data["obs"].values.astype(np.float32)

    print(f"  Total samples: {len(coords):,}")
    print(f"  Longitude range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  Latitude range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

    # Standardize targets (mean=0, std=1)
    y_mean = targets.mean()
    y_std = targets.std()
    targets_norm = (targets - y_mean) / (y_std + 1e-8)

    # Train/val/test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        coords, targets_norm, test_size=0.4, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    return {
        'train_coords': torch.tensor(X_train, dtype=torch.float32),
        'train_targets': torch.tensor(y_train, dtype=torch.float32),
        'val_coords': torch.tensor(X_val, dtype=torch.float32),
        'val_targets': torch.tensor(y_val, dtype=torch.float32),
        'test_coords': torch.tensor(X_test, dtype=torch.float32),
        'test_targets': torch.tensor(y_test, dtype=torch.float32),
        # Use y_min/y_max format for consistent evaluation
        # original = norm * (y_max - y_min) + y_min = norm * y_std + y_mean
        'y_min': y_mean,
        'y_max': y_mean + y_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'task': 'regression',
        'spherical': True  # Flag for spherical methods
    }


def create_subset(coords, targets, fraction, seed):
    """Create a random subset of the data."""
    n = len(coords)
    n_subset = max(1, int(fraction * n))
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=n_subset, replace=False)
    return coords[indices], targets[indices]


def create_balanced_subset(coords, targets, samples_per_class, num_classes, seed):
    """
    Create a balanced subset with exactly N samples per class.
    Used for Japan Prefecture to match existing experiments.
    """
    rng = np.random.default_rng(seed)

    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx in range(len(targets)):
        class_id = int(targets[idx].item())
        class_indices[class_id].append(idx)

    # Sample from each class
    selected_indices = []
    for class_id, indices in class_indices.items():
        if len(indices) >= samples_per_class:
            sampled = rng.choice(indices, size=samples_per_class, replace=False)
        else:
            # If not enough samples, use all with replacement
            sampled = rng.choice(indices, size=samples_per_class, replace=True)
        selected_indices.extend(sampled.tolist())

    selected_indices = np.array(selected_indices)
    return coords[selected_indices], targets[selected_indices]


def run_experiment(
    method: str,
    data: Dict,
    args,
    fraction: float,
    run_seed: int,
    device: torch.device,
    samples_per_class: int = None
) -> Dict:
    """
    Run a single experiment with the specified method and data fraction.

    Args:
        samples_per_class: If provided, use balanced sampling (for classification)

    Returns metrics dictionary.
    """
    task = data['task']

    # Create training subset
    if samples_per_class is not None and task == 'classification':
        # Balanced sampling for classification (Japan Prefecture)
        train_coords, train_targets = create_balanced_subset(
            data['train_coords'], data['train_targets'],
            samples_per_class, data['num_classes'], run_seed
        )
    else:
        # Fraction-based sampling for regression
        train_coords, train_targets = create_subset(
            data['train_coords'], data['train_targets'],
            fraction, run_seed
        )

    # Prepare coordinates (normalize for planar methods)
    if method in ['exact_gp', 'svgp', 'planar_rff']:
        transformer = CoordinateTransformer(coord_type='planar')
        train_x = transformer.fit_transform(train_coords)
        test_x = transformer.transform(data['test_coords'])
        val_x = transformer.transform(data['val_coords'])
    else:
        # Spherical methods - keep original coords, they handle conversion internally
        train_x = train_coords
        test_x = data['test_coords']
        val_x = data['val_coords']

    train_y = train_targets
    test_y = data['test_targets']

    t_start = time.time()
    metrics = {}

    if task == 'regression':
        y_min = data.get('y_min')
        y_max = data.get('y_max')

        if method == 'exact_gp':
            model, likelihood, _ = train_exact_gp_regression(
                train_x.to(device), train_y.to(device),
                kernel_type=args.kernel,
                num_iterations=args.num_iterations,
                lr=args.lr,
                device=device,
                verbose=args.verbose
            )
            preds = predict_regression(model, likelihood, test_x.to(device))
            metrics = evaluate_regression(preds, test_y, y_min, y_max)

        elif method == 'svgp':
            model, likelihood, _ = train_svgp_regression(
                train_x.to(device), train_y.to(device),
                num_inducing=args.num_inducing,
                kernel_type=args.kernel,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            preds = predict_svgp_regression(model, likelihood, test_x.to(device))
            metrics = evaluate_svgp_regression(preds, test_y, y_min, y_max)

        elif method == 'planar_rff':
            model, _ = train_rff_regression(
                train_x.to(device), train_y.to(device),
                num_features=args.num_features,
                kernel_type=args.kernel,
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            metrics = evaluate_rff_regression(model, test_x.to(device), test_y.to(device), y_min, y_max)

        elif method == 'spherical_svgp':
            try:
                model, likelihood, _ = train_spherical_svgp_regression(
                    train_coords.to(device), train_y.to(device),
                    num_inducing=args.num_inducing,
                    nu=_nu_from_kernel(args.kernel),
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    seed=run_seed,
                    verbose=args.verbose
                )
                preds = predict_spherical_svgp(model, likelihood, data['test_coords'].to(device))
                metrics = evaluate_spherical_svgp(preds, test_y, y_min, y_max)
            except Exception as e:
                print(f"  WARNING: Spherical SVGP failed ({type(e).__name__}), falling back to Spherical RFF")
                # Fall back to spherical RFF
                model, _ = train_spherical_rff_regression(
                    train_coords.to(device), train_y.to(device),
                    num_features=args.num_features if args.num_features else 2000,
                    nu=_nu_from_kernel(args.kernel),
                    device=device,
                    seed=run_seed,
                    verbose=args.verbose
                )
                metrics = evaluate_spherical_rff_regression(model, data['test_coords'].to(device), test_y.to(device), y_min, y_max)
                metrics['fallback'] = 'spherical_rff'

        elif method == 'spherical_rff':
            model, _ = train_spherical_rff_regression(
                train_coords.to(device), train_y.to(device),
                num_features=args.num_features,
                nu=_nu_from_kernel(args.kernel),
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            metrics = evaluate_spherical_rff_regression(model, data['test_coords'].to(device), test_y.to(device), y_min, y_max)

    else:  # classification
        num_classes = data['num_classes']

        if method == 'exact_gp':
            # Exact GP classification doesn't scale well to many classes
            # Use RFF features + logistic regression as a proxy
            print("  Note: Using RFF+logistic for exact_gp classification (Dirichlet GP unstable for many classes)")
            model, _ = train_rff_classification(
                train_x.to(device), train_y.to(device),
                num_classes=num_classes,
                num_features=500,  # Fewer features for "exact-like" behavior
                kernel_type=args.kernel,
                num_epochs=100,
                batch_size=256,
                lr=0.01,
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            metrics = evaluate_rff_classification(model, test_x.to(device), test_y.to(device))

        elif method == 'svgp':
            model, likelihood, _ = train_svgp_classification(
                train_x.to(device), train_y.to(device),
                num_classes=num_classes,
                num_inducing=args.num_inducing,
                kernel_type=args.kernel,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            preds = predict_svgp_classification(model, likelihood, test_x.to(device))
            metrics = evaluate_svgp_classification(preds, test_y)

        elif method == 'planar_rff':
            model, _ = train_rff_classification(
                train_x.to(device), train_y.to(device),
                num_classes=num_classes,
                num_features=args.num_features,
                kernel_type=args.kernel,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            metrics = evaluate_rff_classification(model, test_x.to(device), test_y.to(device))

        elif method == 'spherical_rff':
            model, _ = train_spherical_rff_classification(
                train_coords.to(device), train_y.to(device),
                num_classes=num_classes,
                num_features=args.num_features,
                nu=_nu_from_kernel(args.kernel),
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=run_seed,
                verbose=args.verbose
            )
            metrics = evaluate_spherical_rff_classification(model, data['test_coords'].to(device), test_y.to(device))

    train_time = time.time() - t_start
    metrics['train_time_sec'] = train_time

    return metrics


def _nu_from_kernel(kernel: str) -> float:
    """Convert kernel name to Matern nu parameter."""
    if kernel == 'matern32':
        return 1.5
    elif kernel == 'matern52':
        return 2.5
    elif kernel == 'rbf':
        return float('inf')  # RBF is infinitely smooth
    else:
        return 2.5


def main():
    parser = argparse.ArgumentParser(description="GP Baselines Training Script")

    # Method selection
    parser.add_argument("--method", type=str, required=True,
                       choices=["exact_gp", "svgp", "planar_rff", "spherical_svgp", "spherical_rff"],
                       help="GP method to use")

    # Dataset selection
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["california_housing", "japan_prefecture", "mss"],
                       help="Dataset to use")
    parser.add_argument("--dataset-dir", type=str, default="datasets/japan_prefectures",
                       help="Directory for Japan Prefecture dataset")
    parser.add_argument("--mss-data-path", type=str,
                       default="/scratch/local/arra4944_images/drf/Experiment_data",
                       help="Path to MSS data directory")

    # Kernel configuration
    parser.add_argument("--kernel", type=str, default="matern52",
                       choices=["rbf", "matern32", "matern52"],
                       help="Kernel type")

    # SVGP / inducing points
    parser.add_argument("--num-inducing", type=int, default=500,
                       help="Number of inducing points for SVGP")

    # RFF configuration
    parser.add_argument("--num-features", type=int, default=1000,
                       help="Number of random features for RFF")

    # Training configuration
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of iterations for exact GP")
    parser.add_argument("--num-epochs", type=int, default=50,
                       help="Number of epochs for SVGP/RFF")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for SVGP/RFF")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")

    # Experiment configuration
    parser.add_argument("--label-fracs", type=str, default="0.01,0.10,0.25,0.50,0.75,1.00",
                       help="Comma-separated training fractions (for regression datasets)")
    parser.add_argument("--samples-per-class", type=str, default="1,2,5,10,20,30,50",
                       help="Comma-separated samples per class (for Japan Prefecture)")
    parser.add_argument("--n-runs", type=int, default=3,
                       help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")

    # Output
    parser.add_argument("--csv-path", type=str, default=None,
                       help="Path to save CSV results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if args.dataset == "california_housing":
        data = load_california_housing(args.seed)
    elif args.dataset == "japan_prefecture":
        data = load_japan_prefecture(args.dataset_dir, args.seed)
    elif args.dataset == "mss":
        data = load_mss(args.mss_data_path, args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Run experiments
    results = []

    # Different experiment loops for classification vs regression
    if args.dataset == "japan_prefecture":
        # Japan Prefecture: use samples per class (balanced sampling)
        samples_per_class_list = [int(x) for x in args.samples_per_class.split(',')]

        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
            print(f"{'='*60}")

            for n_samples in samples_per_class_list:
                total_samples = n_samples * data['num_classes']
                print(f"\nTraining with {n_samples} samples/class ({total_samples} total)...")

                try:
                    metrics = run_experiment(
                        method=args.method,
                        data=data,
                        args=args,
                        fraction=1.0,  # Not used for balanced sampling
                        run_seed=run_seed,
                        device=device,
                        samples_per_class=n_samples
                    )

                    # Log results
                    result = {
                        'method': args.method,
                        'kernel': args.kernel,
                        'dataset': args.dataset,
                        'run': run_idx + 1,
                        'seed': run_seed,
                        'samples_per_class': n_samples,
                        'train_samples': total_samples,
                        **metrics
                    }

                    # Add method-specific params
                    if args.method in ['svgp', 'spherical_svgp']:
                        result['num_inducing'] = args.num_inducing
                    if args.method in ['planar_rff', 'spherical_rff']:
                        result['num_features'] = args.num_features

                    results.append(result)
                    print(f"  [{n_samples}/class] Accuracy={metrics['accuracy']:.4f}")

                except Exception as e:
                    print(f"  ERROR [{n_samples}/class]: {type(e).__name__}: {e}")
                    # Log failed result with NaN metrics
                    result = {
                        'method': args.method,
                        'kernel': args.kernel,
                        'dataset': args.dataset,
                        'run': run_idx + 1,
                        'seed': run_seed,
                        'samples_per_class': n_samples,
                        'train_samples': total_samples,
                        'accuracy': float('nan'),
                        'error': str(e)
                    }
                    if args.method in ['svgp', 'spherical_svgp']:
                        result['num_inducing'] = args.num_inducing
                    if args.method in ['planar_rff', 'spherical_rff']:
                        result['num_features'] = args.num_features
                    results.append(result)

    else:
        # California Housing / MSS: use label fractions
        label_fracs = [float(x) for x in args.label_fracs.split(',')]

        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
            print(f"{'='*60}")

            for frac in label_fracs:
                pct = int(frac * 100) if frac >= 0.01 else frac * 100
                print(f"\nTraining with {pct}% of data...")

                try:
                    metrics = run_experiment(
                        method=args.method,
                        data=data,
                        args=args,
                        fraction=frac,
                        run_seed=run_seed,
                        device=device
                    )

                    # Log results
                    result = {
                        'method': args.method,
                        'kernel': args.kernel,
                        'dataset': args.dataset,
                        'run': run_idx + 1,
                        'seed': run_seed,
                        'train_fraction': frac,
                        'train_percent': pct,
                        'train_samples': int(frac * len(data['train_coords'])),
                        **metrics
                    }

                    # Add method-specific params
                    if args.method in ['svgp', 'spherical_svgp']:
                        result['num_inducing'] = args.num_inducing
                    if args.method in ['planar_rff', 'spherical_rff']:
                        result['num_features'] = args.num_features

                    results.append(result)
                    print(f"  [{pct}%] R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

                except Exception as e:
                    print(f"  ERROR [{pct}%]: {type(e).__name__}: {e}")
                    # Log failed result with NaN metrics
                    result = {
                        'method': args.method,
                        'kernel': args.kernel,
                        'dataset': args.dataset,
                        'run': run_idx + 1,
                        'seed': run_seed,
                        'train_fraction': frac,
                        'train_percent': pct,
                        'train_samples': int(frac * len(data['train_coords'])),
                        'r2': float('nan'),
                        'mae': float('nan'),
                        'mse': float('nan'),
                        'error': str(e)
                    }
                    if args.method in ['svgp', 'spherical_svgp']:
                        result['num_inducing'] = args.num_inducing
                    if args.method in ['planar_rff', 'spherical_rff']:
                        result['num_features'] = args.num_features
                    results.append(result)

    # Save results
    if args.csv_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(args.csv_path) if os.path.dirname(args.csv_path) else '.', exist_ok=True)

        df_results = pd.DataFrame(results)
        df_results.to_csv(args.csv_path, index=False)
        print(f"\nSaved results to {args.csv_path}")

        # Print summary
        print("\nSummary (mean ± std across runs):")
        if args.dataset == "japan_prefecture":
            summary = df_results.groupby('samples_per_class')['accuracy'].agg(['mean', 'std'])
        elif data['task'] == 'regression':
            summary = df_results.groupby('train_percent')['r2'].agg(['mean', 'std'])
        else:
            summary = df_results.groupby('train_percent')['accuracy'].agg(['mean', 'std'])
        print(summary.round(4))


if __name__ == "__main__":
    main()
