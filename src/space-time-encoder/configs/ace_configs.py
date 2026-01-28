"""
Configuration for ACE dataset experiments.

This module provides centralized configuration for training and evaluation.
"""

import os
from pathlib import Path


def get_data_path():
    """Get ACE data path from environment or default."""
    return os.environ.get(
        'ACE_DATA_PATH',
        '/scratch/local/arra4944_images/ace/'
    )


def get_checkpoint_dir():
    """Get checkpoint directory from environment or default."""
    default = Path(__file__).resolve().parent.parent / 'checkpoints'
    return Path(os.environ.get('ACE_CHECKPOINT_DIR', str(default)))


# Default training hyperparameters
TRAINING_DEFAULTS = {
    'batch_size': 40000,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 15,
}

# Default data split fractions
DATA_SPLIT_DEFAULTS = {
    'train_frac': 0.01,
    'val_frac': 0.01,
    'test_frac': 0.01,
}

# Default regularization settings
REGULARIZATION_DEFAULTS = {
    'ortho_weight_final': 1e-3,
    'ortho_weight_space': 1e-3,
    'ortho_weight_time': 1e-3,
    'ortho_exponent': 1,
    'normality_flag': False,
    'time_grad_penalty_weight': 1e-6,
}

# No regularization settings
REGULARIZATION_NOREG = {
    'ortho_weight_final': 0.0,
    'ortho_weight_space': 0.0,
    'ortho_weight_time': 0.0,
    'ortho_exponent': 1,
    'normality_flag': False,
    'time_grad_penalty_weight': 0.0,
}

# Default model architecture
MODEL_DEFAULTS = {
    'spatial_L': 20,  # L^2 = 400 spatial dimensions
    'temporal_dim': 40,
    'combination': 'concatenate',
    'hidden_dim': 1024,
    'num_layers': 4,
    'output_dim': 8,  # 8 temperature variables
}

# DPSS-specific defaults
DPSS_DEFAULTS = {
    'dpss_N': 365 * 4,  # 1460 timesteps (4 per day for 365 days)
    'dpss_NW': 20,  # Time-bandwidth product
    'dpss_concentration_threshold': 0.85,
}

# Variable names for ACE temperature levels
VAR_NAMES = [
    "T0 (~26hPa)",
    "T1 (~99hPa)",
    "T2 (~203hPa)",
    "T3 (~337hPa)",
    "T4 (~504hPa)",
    "T5 (~690hPa)",
    "T6 (~850hPa)",
    "T7 (~964hPa)",
]


def get_ace_file_paths(data_root=None, n_months=12, year=2021):
    """
    Generate ACE NetCDF file paths.

    Args:
        data_root: Root directory for ACE data (None = use default)
        n_months: Number of months to include
        year: Year to load

    Returns:
        List of file paths
    """
    if data_root is None:
        data_root = get_data_path()

    file_paths = []
    for month in range(1, n_months + 1):
        filename = f"{year}{month:02d}0100.nc"
        file_paths.append(os.path.join(data_root, filename))

    return file_paths


def get_experiment_name(temporal_type, use_reg=True, nw=None, seed=42):
    """
    Generate descriptive experiment name.

    Args:
        temporal_type: Type of temporal encoding
        use_reg: Whether regularization is enabled
        nw: NW parameter for DPSS (optional)
        seed: Random seed

    Returns:
        Experiment name string
    """
    reg_str = "reg" if use_reg else "noreg"

    if temporal_type == 'dpss' and nw is not None:
        return f"dpss_NW{nw}_{reg_str}_seed{seed}"
    else:
        return f"{temporal_type}_{reg_str}_seed{seed}"
