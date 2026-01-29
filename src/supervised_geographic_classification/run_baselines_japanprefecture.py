#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japan Prefecture Classification with Baseline Positional Encoders.
Tests all baseline encoders with varying samples per prefecture.
"""

import os
import sys

# Add parent directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import pyproj
pyproj.datadir.set_data_dir(
    "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
)

import argparse
import json
import time
import math
import copy
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore', category=FutureWarning)

# Performance settings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

# Import baseline encoders
from pe_baselines import (
    Direct, Cartesian3D, Wrap, Grid,
    SphereC, SphereCPlus, SphereM, SphereMPlus, Theory,
    Wavelets
)

# Import nn module for architecture selection
from nn import build_location_model

# =============================================================================
# Dataset Generation (reusing your existing code)
# =============================================================================




def generate_japan_prefecture_dataset(
    shapefile_path: str,
    output_dir: str,
    samples_per_prefecture: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Generate dataset for Japan prefecture classification."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Loading shapefile: {shapefile_path}")
        print(f"Generating {samples_per_prefecture} samples per prefecture")
    
    gdf = gpd.read_file(shapefile_path)
    
    if gdf.crs and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    name_cols = [col for col in gdf.columns if 'NAME' in col.upper()]
    name_col = name_cols[0] if name_cols else None
    
    all_samples = []
    prefecture_info = {}
    
    if verbose:
        iterator = tqdm(gdf.iterrows(), total=len(gdf), desc="Sampling prefectures")
    else:
        iterator = gdf.iterrows()
    
    for idx, row in iterator:
        geometry = row.geometry
        prefecture_id = idx
        
        if isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
            areas = [p.area for p in polygons]
            total_area = sum(areas)
            
            samples_list = []
            for poly, area in zip(polygons, areas):
                n_samples = int(samples_per_prefecture * (area / total_area))
                if n_samples > 0:
                    points = sample_points_in_polygon(poly, n_samples, seed=seed+idx)
                    samples_list.append(points)
            
            samples = np.vstack(samples_list) if samples_list else np.empty((0, 2))
        elif isinstance(geometry, Polygon):
            samples = sample_points_in_polygon(geometry, samples_per_prefecture, seed=seed+idx)
        else:
            continue
        
        # Ensure we have exactly samples_per_prefecture points
        if len(samples) > samples_per_prefecture:
            samples = samples[:samples_per_prefecture]
        elif len(samples) < samples_per_prefecture:
            # Resample if we have too few
            if len(samples) > 0:
                indices = np.random.choice(len(samples), samples_per_prefecture, replace=True)
                samples = samples[indices]
        
        prefecture_samples = np.column_stack([
            samples, np.full(len(samples), prefecture_id)
        ])
        all_samples.append(prefecture_samples)
        
        prefecture_info[int(prefecture_id)] = {
            'name': row[name_col] if name_col else f'Prefecture_{idx}',
            'area_sq_deg': float(geometry.area)
        }
    
    all_samples = np.vstack(all_samples)
    df = pd.DataFrame(all_samples, columns=['longitude', 'latitude', 'prefecture_id'])
    df['prefecture_id'] = df['prefecture_id'].astype(int)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    X = df[['longitude', 'latitude']].values
    y = df['prefecture_id'].values
    
    # Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=seed, stratify=y_temp
    )
    
    # Save splits
    train_df = pd.DataFrame(X_train, columns=['longitude', 'latitude'])
    train_df['prefecture_id'] = y_train
    
    val_df = pd.DataFrame(X_val, columns=['longitude', 'latitude'])
    val_df['prefecture_id'] = y_val
    
    test_df = pd.DataFrame(X_test, columns=['longitude', 'latitude'])
    test_df['prefecture_id'] = y_test
    
    train_df.to_csv(os.path.join(output_dir, f'train_spp{samples_per_prefecture}.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, f'val_spp{samples_per_prefecture}.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, f'test_spp{samples_per_prefecture}.csv'), index=False)
    
    metadata = {
        'n_prefectures': len(gdf),
        'samples_per_prefecture': samples_per_prefecture,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'prefecture_info': prefecture_info,
        'japan_center': [138.0, 36.0],
        'suggested_cap_radius': 10.0
    }
    
    metadata_path = os.path.join(output_dir, f'metadata_spp{samples_per_prefecture}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"Dataset generated: {len(df)} total samples")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return metadata


# =============================================================================
# Baseline Configurations
# =============================================================================

def get_baseline_encoder(name: str, params: Dict = None, resolution_override: Dict = None):
    """Create a baseline encoder with specified parameters.

    Args:
        name: Encoder name
        params: Full parameter dict (takes precedence if provided)
        resolution_override: Resolution params to override defaults for freq-based encoders
    """

    # Default parameters for Japan (centered around 138°E, 36°N)
    default_params = {
        'grid': {
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        },
        'theory': {
            'frequency_num': 16,
            'max_radius': 10000,  # Larger for Theory
            'min_radius': 1000,
            'freq_init': 'geometric'
        },
        'spherical': {  # For SphereC, SphereCPlus, SphereM, SphereMPlus
            'frequency_num': 16,
            'max_radius': 360,
            'min_radius': 1,
            'freq_init': 'geometric'
        }
    }

    # Apply resolution override to defaults if specified
    if resolution_override is not None:
        for key in ['grid', 'theory', 'spherical']:
            default_params[key] = resolution_override.copy()

    if name == 'direct':
        return Direct()
    elif name == 'cartesian3d':
        return Cartesian3D()
    elif name == 'wrap':
        return Wrap()
    elif name == 'grid':
        p = params or default_params['grid']
        return Grid(**p)
    elif name == 'spherec':
        p = params or default_params['spherical']
        return SphereC(**p)
    elif name == 'spherecplus':
        p = params or default_params['spherical']
        return SphereCPlus(**p)
    elif name == 'spherem':
        p = params or default_params['spherical']
        return SphereM(**p)
    elif name == 'spheremplus':
        p = params or default_params['spherical']
        return SphereMPlus(**p)
    elif name == 'theory':
        p = params or default_params['theory']
        return Theory(**p)
    elif name == 'wavelets':
        wavelets_params = params or {
            'max_scale': 3,
            'max_rotations': 75,
            'k_val': 6,
            'scale_factor': 1.0,
            'scale_shift': 1,
            'dilation_step': 6,
            'wavelet_type': 'butterfly'
        }
        return Wavelets(**wavelets_params)
    else:
        raise ValueError(f"Unknown encoder: {name}")


# =============================================================================
# Dataset and Model
# =============================================================================

class PrefectureDataset(Dataset):
    """Dataset for prefecture classification."""
    
    def __init__(self, coords: np.ndarray, labels: np.ndarray):
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.labels[idx]


def create_prefecture_subset(
    train_dataset: Dataset,
    samples_per_prefecture: int,
    n_prefectures: int,
    batch_size: int,
    seed: int,
    num_workers: int = 8
) -> DataLoader:
    """
    Create a subset with exactly `samples_per_prefecture` samples from each prefecture.
    """
    # Group indices by prefecture
    prefecture_indices = {i: [] for i in range(n_prefectures)}
    
    for idx in range(len(train_dataset)):
        coords, label = train_dataset[idx]
        prefecture_indices[int(label.item())].append(idx)
    
    # Sample exactly `samples_per_prefecture` from each prefecture
    rng = np.random.default_rng(seed)
    selected_indices = []
    
    for pref_id, indices in prefecture_indices.items():
        if len(indices) >= samples_per_prefecture:
            sampled = rng.choice(indices, size=samples_per_prefecture, replace=False)
        else:
            # If not enough samples, use all available with replacement
            sampled = rng.choice(indices, size=samples_per_prefecture, replace=True)
        selected_indices.extend(sampled.tolist())
    
    from torch.utils.data import Subset
    subset = Subset(train_dataset, selected_indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader


# =============================================================================
# Training and Evaluation
# =============================================================================

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
    """Train classifier with early stopping."""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        
        for coords, labels in train_loader:
            coords = coords.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(coords)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += len(labels)
        
        train_acc = correct / total
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for coords, labels in val_loader:
                coords = coords.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(coords)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += len(labels)
        
        val_acc = correct / total
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return train_accs, val_accs


@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model and compute metrics."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for coords, labels in test_loader:
        coords = coords.to(device, non_blocking=True)
        outputs = model(coords)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'accuracy': float(accuracy),
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Japan Prefecture Classification with Baselines")
    
    # Dataset
    parser.add_argument("--dataset-dir", type=str, 
                       default="experiments_japan_label_efficiency/dataset",
                       help="Directory containing the dataset")
    
    # Experiment configuration
    parser.add_argument("--train-samples-per-prefecture", type=str, 
                       default="1,2,5,10,20,50,100",
                       help="Comma-separated samples per prefecture for training")
    parser.add_argument("--encoders", type=str, default="all",
                       help="Comma-separated list of encoders or 'all'")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    
    # Other
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", type=str, default="results/japan_baselines.csv")
    parser.add_argument("--verbose", action="store_true")

    # Architecture selection
    parser.add_argument("--arch", type=str, default="mlp",
                       choices=["linear", "mlp", "resmlp", "siren", "glu"],
                       help="Neural network architecture (default: mlp)")

    # Resolution parameters for frequency-based encoders
    parser.add_argument("--max-radius", type=float, default=None,
                       help="Max radius (coarsest wavelength) for frequency-based encoders.")
    parser.add_argument("--min-radius", type=float, default=None,
                       help="Min radius (finest wavelength) for frequency-based encoders.")
    parser.add_argument("--frequency-num", type=int, default=None,
                       help="Number of frequencies. If not set, auto-calculated as ~2 per octave.")

    args = parser.parse_args()

    # Build resolution override params if specified
    resolution_override = None
    if args.max_radius is not None or args.min_radius is not None:
        max_r = args.max_radius if args.max_radius is not None else 360.0
        min_r = args.min_radius if args.min_radius is not None else 1.0

        if args.frequency_num is not None:
            freq_num = args.frequency_num
        else:
            num_octaves = math.log2(max_r / min_r)
            freq_num = max(4, int(round(2 * num_octaves)))

        resolution_override = {
            'max_radius': max_r,
            'min_radius': min_r,
            'frequency_num': freq_num,
            'freq_init': 'geometric'
        }
        print(f"Resolution override: max_radius={max_r}, min_radius={min_r}, frequency_num={freq_num}")
    
    # Parse samples per prefecture
    # Parse samples per prefecture
    samples_per_pref_list = [int(x) for x in args.train_samples_per_prefecture.split(',')]
    
    # Select encoders
    all_encoders = ['direct', 'cartesian3d', 'wrap', 'grid', 'spherec',
                    'spherecplus', 'spherem', 'spheremplus', 'theory', 'wavelets']
    if args.encoders == 'all':
        encoder_names = all_encoders
    else:
        encoder_names = args.encoders.split(',')
    
    print(f"Testing encoders: {encoder_names}")
    print(f"Training samples per prefecture: {samples_per_pref_list}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset (once, like slepian script)
    train_path = os.path.join(args.dataset_dir, 'train.csv')
    val_path = os.path.join(args.dataset_dir, 'val.csv')
    test_path = os.path.join(args.dataset_dir, 'test.csv')
    metadata_path = os.path.join(args.dataset_dir, 'metadata.json')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path, metadata_path]):
        print(f"Missing dataset files in {args.dataset_dir}")
        return
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    n_classes = metadata['n_prefectures']
    
    print(f"Dataset loaded: {n_classes} classes")
    print(f"Total samples - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create full datasets
    train_dataset = PrefectureDataset(
        train_df[['longitude', 'latitude']].values,
        train_df['prefecture_id'].values
    )
    val_dataset = PrefectureDataset(
        val_df[['longitude', 'latitude']].values,
        val_df['prefecture_id'].values
    )
    test_dataset = PrefectureDataset(
        test_df[['longitude', 'latitude']].values,
        test_df['prefecture_id'].values
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
    
    # Results storage
    csv_results = []
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    
    # Test each encoder
    for encoder_name in encoder_names:
        print(f"\n{'='*60}")
        print(f"ENCODER: {encoder_name.upper()}")
        print(f"{'='*60}")
        
        # Create encoder once to check dimensions
        encoder = get_baseline_encoder(encoder_name, resolution_override=resolution_override)
        print(f"Encoder dimension: {encoder.n_features}")
        
        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx
            print(f"\nRun {run_idx + 1}/{args.n_runs} (seed={run_seed})")
            
            for n_samples in samples_per_pref_list:
                if args.verbose:
                    print(f"  Training with {n_samples} samples per prefecture")
                
                # Set seeds
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                
                # Create subset loader with exact samples per prefecture
                train_subset_loader = create_prefecture_subset(
                    train_dataset, n_samples, n_classes,
                    args.batch_size, run_seed, args.num_workers
                )
                
                # Create fresh model
                encoder = get_baseline_encoder(encoder_name, resolution_override=resolution_override)
                model = build_location_model(
                    encoder, task="classification", arch=args.arch,
                    num_classes=n_classes, hidden_dim=args.hidden_dim, dropout=args.dropout
                ).to(device)
                
                # Train
                t_start = time.time()
                train_accs, val_accs = train_model(
                    model, train_subset_loader, val_loader, device,
                    epochs=args.epochs, lr=args.lr, patience=args.patience,
                    verbose=False  # Too many combinations for verbose
                )
                train_time = time.time() - t_start
                
                # Evaluate
                metrics = evaluate_model(model, test_loader, device)
                
                print(f"    [{n_samples:3d} samples] Accuracy: {metrics['accuracy']:.4f}")
                
                # Record results
                result_row = {
                    'encoder': encoder_name,
                    'arch': args.arch,
                    'encoder_dim': encoder.n_features,
                    'samples_per_prefecture': n_samples,
                    'total_train_samples': n_samples * n_classes,
                    'run': run_idx + 1,
                    'seed': run_seed,
                    'accuracy': metrics['accuracy'],
                    'train_acc': train_accs[-1] if train_accs else 0,
                    'val_acc': val_accs[-1] if val_accs else 0,
                    'train_time_sec': train_time,
                    'n_classes': n_classes
                }
                # Add resolution params if override was specified
                if resolution_override is not None:
                    result_row['max_radius'] = resolution_override['max_radius']
                    result_row['min_radius'] = resolution_override['min_radius']
                    result_row['frequency_num'] = resolution_override['frequency_num']
                csv_results.append(result_row)
    
    # Save results
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv(args.csv_path, index=False)
    print(f"\nResults saved to {args.csv_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY (mean accuracy by encoder and samples/prefecture):")
    print("="*70)
    
    summary = df_results.groupby(
        ['encoder', 'samples_per_prefecture']
    )['accuracy'].agg(['mean', 'std']).reset_index()
    
    # Best encoder at each sample count
    print("\nBest encoder at each sample count:")
    for n_samples in samples_per_pref_list:
        sample_data = summary[summary['samples_per_prefecture'] == n_samples]
        if not sample_data.empty:
            best = sample_data.loc[sample_data['mean'].idxmax()]
            print(f"{n_samples:3d} samples/pref: {best['encoder']:15s} "
                  f"(acc={best['mean']:.4f}±{best['std']:.4f})")


if __name__ == "__main__":
    main()