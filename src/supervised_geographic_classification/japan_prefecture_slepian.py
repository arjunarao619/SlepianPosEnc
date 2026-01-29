#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Japan Prefecture Classification with cached Slepian features.
"""

import os
import sys

# Add parent directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Set pyproj data directory if available (needed for geopandas CRS transforms)
try:
    import pyproj
    pyproj.datadir.set_data_dir(
        "/projects/arra4944/arm64/software/miniforge/envs/bg2/share/proj"
    )
except ImportError:
    pass  # pyproj not available, geopandas may still work with default settings

import argparse
import json
import time
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from spherical_harmonics_ylm import SH as SH_analytic

warnings.filterwarnings('ignore', category=FutureWarning)

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

# Import nn module for architecture selection
from nn import build_indexed_location_model

# Import shared utilities
from utils.geo import compute_coverage


# =============================================================================
# Dataset Generation (keeping your existing generation code)
# =============================================================================

def sample_points_in_polygon(polygon: Polygon, n_points: int, seed: Optional[int] = None) -> np.ndarray:
    """Sample n_points uniformly within a polygon using rejection sampling."""
    if seed is not None:
        np.random.seed(seed)

    minx, miny, maxx, maxy = polygon.bounds
    points = []
    attempts = 0
    max_attempts = n_points * 100

    while len(points) < n_points and attempts < max_attempts:
        n_try = min(n_points * 10, 10000)
        random_points = np.column_stack([
            np.random.uniform(minx, maxx, n_try),
            np.random.uniform(miny, maxy, n_try)
        ])

        for lon, lat in random_points:
            if polygon.contains(Point(lon, lat)):
                points.append([lon, lat])
                if len(points) >= n_points:
                    break
        attempts += n_try

    return np.array(points[:n_points])


def sample_points_near_boundary(polygon: Polygon, n_points: int, buffer_dist: float = 0.02,
                                 seed: Optional[int] = None) -> np.ndarray:
    """Sample points in a thin ring near the polygon boundary."""
    if seed is not None:
        np.random.seed(seed)

    eroded = polygon.buffer(-buffer_dist)
    if eroded.is_empty or not eroded.is_valid:
        eroded = polygon.buffer(-buffer_dist / 4)
    if eroded.is_empty or not eroded.is_valid:
        return sample_points_in_polygon(polygon, n_points, seed)

    ring = polygon.difference(eroded)
    if ring.is_empty or not ring.is_valid:
        return sample_points_in_polygon(polygon, n_points, seed)

    minx, miny, maxx, maxy = ring.bounds
    points = []
    attempts = 0
    max_attempts = n_points * 200

    while len(points) < n_points and attempts < max_attempts:
        n_try = min(n_points * 20, 20000)
        random_points = np.column_stack([
            np.random.uniform(minx, maxx, n_try),
            np.random.uniform(miny, maxy, n_try)
        ])
        for lon, lat in random_points:
            if ring.contains(Point(lon, lat)):
                points.append([lon, lat])
                if len(points) >= n_points:
                    break
        attempts += n_try

    if len(points) < n_points:
        extra = sample_points_in_polygon(polygon, n_points - len(points), seed)
        if len(extra) > 0:
            points.extend(extra.tolist())

    return np.array(points[:n_points])


def _sample_from_geometry(geometry, n_points: int, near_boundary: bool,
                          buffer_dist: float, seed: int) -> np.ndarray:
    """Sample points from a geometry (Polygon or MultiPolygon)."""
    sample_fn = sample_points_near_boundary if near_boundary else sample_points_in_polygon

    if isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
        if near_boundary:
            perimeters = [p.length for p in polygons]
            total = sum(perimeters)
            weights = [l / total for l in perimeters]
        else:
            areas = [p.area for p in polygons]
            total = sum(areas)
            weights = [a / total for a in areas]

        samples_list = []
        for poly, w in zip(polygons, weights):
            n = max(1, int(n_points * w))
            if near_boundary:
                pts = sample_fn(poly, n, buffer_dist=buffer_dist, seed=seed)
            else:
                pts = sample_fn(poly, n, seed=seed)
            if len(pts) > 0:
                samples_list.append(pts)
        return np.vstack(samples_list) if samples_list else np.empty((0, 2))

    elif isinstance(geometry, Polygon):
        if near_boundary:
            return sample_fn(geometry, n_points, buffer_dist=buffer_dist, seed=seed)
        return sample_fn(geometry, n_points, seed=seed)

    return np.empty((0, 2))


def generate_japan_prefecture_dataset(
    shapefile_path: str,
    output_dir: str,
    samples_per_prefecture: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    boundary_buffer: float = 0.02,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Generate dataset with train points random inside, test/val near boundaries."""

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Loading shapefile: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)
    if gdf.crs and gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    name_cols = [col for col in gdf.columns if 'NAME' in col.upper()]
    name_col = name_cols[0] if name_cols else None

    n_train = int(samples_per_prefecture * train_ratio)
    n_val = int(samples_per_prefecture * val_ratio)
    n_test = samples_per_prefecture - n_train - n_val

    train_samples, val_samples, test_samples = [], [], []
    prefecture_info = {}

    iterator = tqdm(gdf.iterrows(), total=len(gdf), desc="Sampling") if verbose else gdf.iterrows()

    for idx, row in iterator:
        geometry = row.geometry
        pref_id = idx

        train_pts = _sample_from_geometry(geometry, n_train, near_boundary=False,
                                          buffer_dist=boundary_buffer, seed=seed+idx)
        val_pts = _sample_from_geometry(geometry, n_val, near_boundary=True,
                                        buffer_dist=boundary_buffer, seed=seed+idx+1000)
        test_pts = _sample_from_geometry(geometry, n_test, near_boundary=True,
                                         buffer_dist=boundary_buffer, seed=seed+idx+2000)

        for pts, collection in [(train_pts, train_samples),
                                (val_pts, val_samples),
                                (test_pts, test_samples)]:
            if len(pts) > 0:
                collection.append(np.column_stack([pts, np.full(len(pts), pref_id)]))

        prefecture_info[int(pref_id)] = {
            'name': row[name_col] if name_col else f'Prefecture_{idx}',
            'area_sq_deg': float(geometry.area)
        }

    def make_df(samples_list):
        arr = np.vstack(samples_list)
        df = pd.DataFrame(arr, columns=['longitude', 'latitude', 'prefecture_id'])
        df['prefecture_id'] = df['prefecture_id'].astype(int)
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = make_df(train_samples)
    val_df = make_df(val_samples)
    test_df = make_df(test_samples)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    metadata = {
        'n_prefectures': len(gdf),
        'samples_per_prefecture': samples_per_prefecture,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'boundary_buffer_deg': boundary_buffer,
        'prefecture_info': prefecture_info,
        'japan_center': [138.0, 36.0],
        'suggested_cap_radius': 10.0
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Dataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        print(f"Test/val points sampled within {boundary_buffer}° of boundaries")

    return {'stats': metadata}


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
    """Compute global spherical harmonic features."""
    if L == 0:
        return np.zeros((len(coords), 0), dtype=np.float32)

    lon_rad = coords[:, 0] * np.pi / 180.0
    lat_rad = coords[:, 1] * np.pi / 180.0
    phi = lon_rad + np.pi
    theta = np.pi/2.0 - lat_rad

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
    cap_center: Tuple[float, float] = (138.0, 36.0),  # Japan center
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Slepian cap features using PySHTOOLS."""

    if not HAVE_PYSH:
        raise ImportError("PySHTOOLS required for Slepian features")

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

    # Rotate to cap center
    slepian.rotate(clat=cap_center[1], clon=cap_center[0], nrot=num_modes)

    # Get eigenvalues
    eigenvalues = slepian.eigenvalues[:num_modes].astype(np.float32)

    if verbose:
        print(f"  Eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
        print(f"  #λ>0.5: {(eigenvalues > 0.5).sum()}")

    # Evaluate Slepian functions at coordinates
    lon = coords[:, 0]
    lat = coords[:, 1]
    lon_360 = np.where(lon < 0.0, lon + 360.0, lon)

    features = []
    for i in range(num_modes):
        coeffs = slepian.to_shcoeffs(i)
        vals = coeffs.expand(lon=lon_360, lat=lat, degrees=True)
        features.append(vals)

    features = np.column_stack(features).astype(np.float32)

    return features, eigenvalues


def compute_and_cache_features(
    coords: np.ndarray,
    method: str,
    L_global: int,
    L_slepian: int,
    cap_radius_deg: float,
    num_modes: int,
    lambda_thresh: float,
    cap_center: Tuple[float, float] = (138.0, 36.0),
    cache_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Compute all features, trim by eigenvalue if Slepian, and optionally cache.
    Matches California housing approach.
    """
    t0 = time.time()

    if method == "slepian":
        # Compute global SH features
        if verbose:
            print(f"Computing global SH features (L={L_global})...")
        global_features = compute_global_sh_features(coords, L_global)
        global_dim = global_features.shape[1]

        # Compute Slepian features
        if verbose:
            print(f"Computing Slepian features...")
        slepian_features, eigenvalues = compute_slepian_features(
            coords, L_slepian, cap_radius_deg, num_modes,
            cap_center=cap_center, verbose=verbose
        )

        # Trim Slepian features by eigenvalue threshold
        keep_mask = eigenvalues > lambda_thresh
        kept_modes = keep_mask.sum()
        slepian_trimmed = slepian_features[:, keep_mask]

        if verbose:
            print(f"Eigenvalue trimming: λ>{lambda_thresh} → kept {kept_modes}/{num_modes} modes")

        # Combine features
        features = np.hstack([global_features, slepian_trimmed])

        metadata = {
            'method': 'slepian',
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
            'n_samples': int(features.shape[0])
        }

    else:  # vanilla_sh
        if verbose:
            print(f"Computing vanilla SH features (L={L_slepian})...")
        features = compute_global_sh_features(coords, L_slepian)

        metadata = {
            'method': 'vanilla_sh',
            'L': L_slepian,
            'total_dim': int(features.shape[1]),
            'n_samples': int(features.shape[0])
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
    """Load precomputed features from cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    cache_data = torch.load(cache_path, map_location='cpu')

    if isinstance(cache_data, torch.Tensor):
        features = cache_data
        metadata = {'total_dim': features.shape[1], 'n_samples': features.shape[0]}
    else:
        features = cache_data['features']
        metadata = cache_data['metadata']

    if verbose:
        print(f"Loaded cached features from {cache_path}")
        print(f"  Shape: {features.shape}")
        print(f"  Method: {metadata.get('method', 'unknown')}")

    return {'features': features, 'metadata': metadata}


# =============================================================================
# Dataset and Model (matching California housing)
# =============================================================================

class IndexedDataset(Dataset):
    """Dataset that provides global indices for cached feature lookup."""

    def __init__(self, coords: torch.Tensor, labels: torch.Tensor, global_indices: torch.Tensor):
        self.coords = coords
        self.labels = labels
        self.global_indices = global_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.global_indices[idx], self.coords[idx], self.labels[idx]


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
# Training and Evaluation
# =============================================================================

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
        _, _, label = train_dataset[idx]
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

    subset = Subset(train_dataset, selected_indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader

def create_data_subset(
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
    n_classes: int,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 20
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train classifier with early stopping."""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for idx, coords, labels in train_loader:
            coords = coords.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(coords, idx)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += len(labels)

        train_loss /= total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for idx, coords, labels in val_loader:
                coords = coords.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                idx = idx.to(device, non_blocking=True)

                outputs = model(coords, idx)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * len(labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += len(labels)

        val_loss /= total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, train_accs, val_losses, val_accs


@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model and compute metrics."""

    model.eval()
    all_preds = []
    all_labels = []

    for idx, coords, labels in test_loader:
        coords = coords.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        outputs = model(coords, idx)
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Japan Prefecture Classification")

    # Dataset generation
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--shapefile", type=str,
                       default="/projects/arra4944/SlepianEncoder/shapefiles/japan_adm1/JPN_adm1.shp")
    parser.add_argument("--dataset-dir", type=str, default="datasets/japan_prefectures")
    parser.add_argument("--samples-per-prefecture", type=int, default=100)
    parser.add_argument("--boundary-buffer", type=float, default=0.02,
                       help="Buffer distance (degrees) for boundary sampling of test/val points")

    # Feature configuration
    parser.add_argument("--method", type=str, default="slepian", choices=["slepian", "vanilla_sh"])
    parser.add_argument("--L-global", type=int, default=10)
    parser.add_argument("--L-slepian", type=int, default=120,
                       help="L for Slepian or vanilla SH")
    parser.add_argument("--cap-radius", type=float, default=10.0)
    parser.add_argument("--num-modes", type=int, default=800)
    parser.add_argument("--lambda-thresh", type=float, default=0.05)
    parser.add_argument("--cap-center-lon", type=float, default=138.0,
                       help="Cap center longitude (default: 138.0 for Japan)")
    parser.add_argument("--cap-center-lat", type=float, default=36.0,
                       help="Cap center latitude (default: 36.0 for Japan)")
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--force-recompute", action="store_true")

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)

    # Experiment configuration
    parser.add_argument("--train-samples-per-prefecture", type=str,
                    default="1,2,5,10,20,30,50",
                    help="Comma-separated samples per prefecture for training (label efficiency)")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--csv-path", type=str, default=None)

    # Architecture selection
    parser.add_argument("--arch", type=str, default="mlp",
                       choices=["linear", "mlp", "resmlp", "siren", "glu"],
                       help="Neural network architecture (default: mlp)")

    args = parser.parse_args()

    # Generate dataset if requested
    if args.generate:
        generate_japan_prefecture_dataset(
            shapefile_path=args.shapefile,
            output_dir=args.dataset_dir,
            samples_per_prefecture=args.samples_per_prefecture,
            boundary_buffer=args.boundary_buffer,
            seed=args.seed,
            verbose=True
        )
        return

    # Load data
    train_df = pd.read_csv(os.path.join(args.dataset_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.dataset_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.dataset_dir, 'test.csv'))

    with open(os.path.join(args.dataset_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    n_classes = metadata['n_prefectures']

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Method: {args.method}, Classes: {n_classes}")

    # Combine all coordinates for feature computation
    all_coords = np.vstack([
        train_df[['longitude', 'latitude']].values,
        val_df[['longitude', 'latitude']].values,
        test_df[['longitude', 'latitude']].values
    ])
    n_train = len(train_df)
    n_val = len(val_df)

    # Compute coverage
    cap_center = (args.cap_center_lon, args.cap_center_lat)
    coverage_pct = compute_coverage(all_coords, cap_center, args.cap_radius)
    print(f"Cap coverage: {coverage_pct:.1f}% of data points within cap")

    # Compute or load features (matching California housing approach)
    if args.cache_path and os.path.exists(args.cache_path) and not args.force_recompute:
        print(f"Loading cached features from {args.cache_path}")
        feature_data = load_cached_features(args.cache_path)
    else:
        if args.method == "slepian" and not HAVE_PYSH:
            raise RuntimeError("PySHTOOLS required for Slepian features")

        print("Computing features from scratch...")
        feature_data = compute_and_cache_features(
            all_coords,
            method=args.method,
            L_global=args.L_global,
            L_slepian=args.L_slepian,
            cap_radius_deg=args.cap_radius,
            num_modes=args.num_modes,
            lambda_thresh=args.lambda_thresh,
            cap_center=cap_center,
            cache_path=args.cache_path
        )

    features = feature_data['features']
    feat_metadata = feature_data['metadata']

    print(f"Feature dimension: {features.shape[1]}")

    # Create encoder with cached features
    encoder = CachedFeatureEncoder(features)

    # Create datasets with global indices (matching California housing)
    global_indices_train = torch.arange(0, n_train)
    global_indices_val = torch.arange(n_train, n_train + n_val)
    global_indices_test = torch.arange(n_train + n_val, n_train + n_val + len(test_df))

    train_dataset = IndexedDataset(
        torch.tensor(train_df[['longitude', 'latitude']].values, dtype=torch.float32),
        torch.tensor(train_df['prefecture_id'].values, dtype=torch.long),
        global_indices_train
    )

    val_dataset = IndexedDataset(
        torch.tensor(val_df[['longitude', 'latitude']].values, dtype=torch.float32),
        torch.tensor(val_df['prefecture_id'].values, dtype=torch.long),
        global_indices_val
    )

    test_dataset = IndexedDataset(
        torch.tensor(test_df[['longitude', 'latitude']].values, dtype=torch.float32),
        torch.tensor(test_df['prefecture_id'].values, dtype=torch.long),
        global_indices_test
    )

    # Fixed loaders
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Parse samples per prefecture
    samples_per_pref_list = [int(x) for x in args.train_samples_per_prefecture.split(',')]
    csv_results = []

    for run_idx in range(args.n_runs):
        run_seed = args.seed + run_idx
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
        print(f"{'='*60}")

        for n_samples in samples_per_pref_list:
            print(f"\nTraining with {n_samples} samples per prefecture...")

            # Create subset loader with exact samples per prefecture
            train_subset_loader = create_prefecture_subset(
                train_dataset, n_samples, n_classes,
                args.batch_size, run_seed, args.num_workers
            )

            # Create fresh model
            model = build_indexed_location_model(
                encoder, task="classification", arch=args.arch,
                num_classes=n_classes, hidden_dim=args.hidden_dim, dropout=args.dropout
            ).to(device)

            # Train
            t_start = time.time()
            train_losses, train_accs, val_losses, val_accs = train_model(
                model, train_subset_loader, val_loader, device, n_classes,
                epochs=args.epochs, lr=args.lr, patience=args.patience
            )
            train_time = time.time() - t_start

            # Evaluate
            metrics = evaluate_model(model, test_loader, device)

            print(f"[{n_samples} samples/pref] Accuracy: {metrics['accuracy']:.4f}")

            # Record results
            csv_results.append({
                'method': args.method,
                'arch': args.arch,
                'L_slepian': args.L_slepian,
                'L_global': args.L_global if args.method == "slepian" else 0,
                'cap_radius': args.cap_radius,
                'cap_center_lon': args.cap_center_lon,
                'cap_center_lat': args.cap_center_lat,
                'coverage_pct': coverage_pct,
                'feature_dim': features.shape[1],
                'run': run_idx + 1,
                'seed': run_seed,
                'samples_per_prefecture': n_samples,
                'total_train_samples': n_samples * n_classes,
                'accuracy': metrics['accuracy'],
                'train_loss': train_losses[-1] if train_losses else 0,
                'val_loss': val_losses[-1] if val_losses else 0,
                'train_acc': train_accs[-1] if train_accs else 0,
                'val_acc': val_accs[-1] if val_accs else 0,
                'train_time_sec': train_time
            })

    # Save results
    if args.csv_path:
        df_results = pd.DataFrame(csv_results)
        df_results.to_csv(args.csv_path, index=False)
        print(f"\nSaved results to {args.csv_path}")

        # Print summary
        print("\nSummary (mean ± std across runs):")
        summary = df_results.groupby('samples_per_prefecture')['accuracy'].agg(['mean', 'std'])
        print(summary.round(4))


if __name__ == "__main__":
    main()
