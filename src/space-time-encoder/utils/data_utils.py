"""
Data utilities for ACE dataset processing
Extracted and cleaned from STE codebase
"""

import torch
import numpy as np


class DataNormalizer:
    """Normalizes data using training statistics"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        """Compute normalization statistics from training data"""
        self.mean = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        # Avoid division by zero
        self.std = torch.where(self.std > 0, self.std, torch.ones_like(self.std))
        
    def transform(self, data):
        """Normalize data using computed statistics"""
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before transform()")
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        """Denormalize data back to original scale"""
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        return data * self.std + self.mean
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)


def normalize_coordinates(coords):
    """
    Normalize coordinates for neural network input
    
    Args:
        coords: (n_samples, 3) tensor with [lon, lat, time]
        
    Returns:
        Normalized coordinates with:
        - lon: [-180, 180] -> [-1, 1] (or keep in degrees)
        - lat: [-90, 90] -> [-1, 1] (or keep in degrees)  
        - time: [min, max] -> [-1, 1]
    """
    coords_norm = coords.clone()
    
    # Time normalization (always normalize to [-1, 1])
    time_min = coords[:, 2].min()
    time_max = coords[:, 2].max()
    coords_norm[:, 2] = 2 * (coords[:, 2] - time_min) / (time_max - time_min) - 1
    
    # Spatial coordinates - keep in degrees for spherical harmonics
    # Only normalize if using non-spherical encoders
    
    return coords_norm


def split_data_randomly(coords, targets, train_frac=0.01, val_frac=0.01, test_frac=0.01):
    """
    Randomly split data into train/val/test sets
    
    Args:
        coords: (n_samples, 3) coordinates
        targets: (n_samples, n_vars) target values
        train_frac: Fraction for training
        val_frac: Fraction for validation  
        test_frac: Fraction for testing
        
    Returns:
        Tuple of (train_data, val_data, test_data) where each is
        a dict with 'coords' and 'targets'
    """
    n_total = coords.shape[0]
    indices = torch.randperm(n_total)
    
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:n_train+n_val+n_test]
    
    return {
        'train': {'coords': coords[train_idx], 'targets': targets[train_idx]},
        'val': {'coords': coords[val_idx], 'targets': targets[val_idx]},
        'test': {'coords': coords[test_idx], 'targets': targets[test_idx]}
    }