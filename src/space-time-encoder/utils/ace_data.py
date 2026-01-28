"""
Centralized ACE dataset loading utilities.

This module consolidates data loading code that was previously duplicated
across train_ste.py, train_dpss.py, train_baselines.py, train_dpss_sweep.py,
and test_ace.py.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
import netCDF4

from .data_utils import DataNormalizer


def load_ace_month(filepath, verbose=True):
    """
    Load a single month of ACE data from NetCDF file.

    Args:
        filepath: Path to NetCDF file (e.g., '202101010.nc')
        verbose: Print loading progress

    Returns:
        coords: (N, 3) array with [lon, lat, time] columns
        targets: (N, 8) array with 8 temperature variables
    """
    if verbose:
        print(f"  Loading {filepath}")

    with netCDF4.Dataset(filepath, 'r') as nc:
        lons = nc.variables['grid_xt'][:] - 180  # Shift to [-180, 180]
        lats = nc.variables['grid_yt'][:]
        times = nc.variables['time'][:]

        # Load 8 temperature variables
        temp_vars = []
        for i in range(8):
            name = f'air_temperature_{i}'
            if name in nc.variables:
                temp_vars.append(nc.variables[name][:])
            else:
                raise ValueError(f"Variable {name} not found in {filepath}")

        temps = np.stack(temp_vars, axis=-1)  # (time, lat, lon, 8)

        # Create coordinate grids
        lon_grid, lat_grid, time_grid = np.meshgrid(lons, lats, times, indexing='ij')

        # Flatten to (N, 3) and (N, 8)
        coords = np.stack([
            lon_grid.flatten(),
            lat_grid.flatten(),
            time_grid.flatten()
        ], axis=1).astype(np.float32)

        targets = temps.transpose(2, 1, 0, 3).reshape(-1, 8).astype(np.float32)

    return coords, targets


def load_ace_year(data_path, months=range(1, 13), year=2021, verbose=True):
    """
    Load multiple months of ACE data.

    Args:
        data_path: Directory containing NetCDF files
        months: Iterable of month numbers (1-12)
        year: Year to load
        verbose: Print loading progress

    Returns:
        coords: (N, 3) tensor with [lon, lat, time]
        targets: (N, 8) tensor with temperature values
    """
    if verbose:
        print("Loading ACE data...")

    coords_all, targets_all = [], []

    for month in months:
        filepath = os.path.join(data_path, f"{year}{month:02d}0100.nc")
        if os.path.exists(filepath):
            c, y = load_ace_month(filepath, verbose=verbose)
            coords_all.append(c)
            targets_all.append(y)
        elif verbose:
            print(f"  Warning: {filepath} not found, skipping")

    if not coords_all:
        raise FileNotFoundError(f"No ACE data files found in {data_path}")

    coords = torch.from_numpy(np.concatenate(coords_all, axis=0))
    targets = torch.from_numpy(np.concatenate(targets_all, axis=0))

    return coords, targets


def normalize_time(coords):
    """
    Normalize time coordinate to [-1, 1] range.

    Args:
        coords: (N, 3) tensor with time in column 2

    Returns:
        coords with normalized time (modified in place)
    """
    tmin, tmax = coords[:, 2].min(), coords[:, 2].max()
    coords[:, 2] = 2 * (coords[:, 2] - tmin) / (tmax - tmin) - 1
    return coords


def prepare_ace_data(data_path, train_frac=0.01, val_frac=0.01, test_frac=0.01,
                     seed=None, include_raw_targets=False, verbose=True):
    """
    Load ACE data and prepare train/val/test splits.

    Args:
        data_path: Directory containing ACE NetCDF files
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for testing
        seed: Random seed for reproducibility (None = non-deterministic)
        include_raw_targets: If True, test dataset includes unnormalized targets
        verbose: Print progress

    Returns:
        train_ds: TensorDataset with (coords, targets_normalized)
        val_ds: TensorDataset with (coords, targets_normalized)
        test_ds: TensorDataset with (coords, targets_normalized) or
                 (coords, targets_normalized, targets_raw) if include_raw_targets
        normalizer: DataNormalizer fitted on training data
    """
    # Load all data
    coords, targets = load_ace_year(data_path, verbose=verbose)

    # Normalize time to [-1, 1]
    coords = normalize_time(coords)

    # Compute split sizes
    N = coords.shape[0]
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    n_test = int(N * test_frac)

    # Random permutation
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=generator)
    else:
        perm = torch.randperm(N)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]

    # Fit normalizer on training data
    normalizer = DataNormalizer()
    normalizer.fit(targets[train_idx])
    targets_norm = normalizer.transform(targets)

    # Create datasets
    train_ds = TensorDataset(coords[train_idx], targets_norm[train_idx])
    val_ds = TensorDataset(coords[val_idx], targets_norm[val_idx])

    if include_raw_targets:
        test_ds = TensorDataset(
            coords[test_idx],
            targets_norm[test_idx],
            targets[test_idx]
        )
    else:
        test_ds = TensorDataset(coords[test_idx], targets_norm[test_idx])

    if verbose:
        print(f"Split sizes: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")

    return train_ds, val_ds, test_ds, normalizer
