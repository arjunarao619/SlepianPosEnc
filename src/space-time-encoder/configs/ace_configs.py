"""
Configuration for ACE dataset experiments
Extracted and cleaned from STE codebase
"""

import os
from pathlib import Path


# ACE Dataset Configuration
ACE_CONFIG = {
    # Data paths
    'data_root': "/scratch/local/arra4944_images/ace/",
    'variables': list(range(15, 23)),  # 8 temperature variables
    
    # Data splits
    'train_fraction': 0.01,
    'val_fraction': 0.01,
    'test_fraction': 0.01,
    
    # Model architecture
    'spatial_encoder': {
        'type': 'spherical_harmonics',
        'L': 20,  # Legendre degree (L^2 = 400 basis functions)
    },
    
    'temporal_encoder': {
        'type': 'dpss',  # or 'fourier', 'legendre'
        'dim': 40,
        'N': 365 * 4,  # 365 days * 4 timesteps per day
        'NW': 10,  # Time-bandwidth product for DPSS
    },
    
    'combination': 'concatenate',  # or 'outer_product', 'hadamard'
    
    'head': {
        'type': 'fcnet',
        'hidden_dim': 1024,
        'num_layers': 4,
    },
    
    # Training
    'batch_size': 40000,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 5,
    
    # Regularization
    'ortho_weight': 1.0,
    'ortho_normality': False,
    'ortho_exponent': 1,
}


def get_ace_file_paths(data_root, n_months=12, start_year=2021, start_month=1):
    """
    Generate ACE NetCDF file paths
    
    Args:
        data_root: Root directory for ACE data
        n_months: Number of months to include
        start_year: Starting year
        start_month: Starting month
        
    Returns:
        List of file paths
    """
    file_paths = []
    
    for i in range(n_months):
        month = (start_month - 1 + i) % 12 + 1
        year = start_year + (start_month - 1 + i) // 12
        
        # ACE file naming: YYYYMM0100.nc
        filename = f"{year}{month:02d}0100.nc"
        file_paths.append(os.path.join(data_root, filename))
    
    return file_paths


def get_experiment_name(config):
    """Generate descriptive experiment name from config"""
    temporal_type = config['temporal_encoder']['type']
    spatial_type = config['spatial_encoder']['type']
    combination = config['combination']
    ortho = 'ortho' if config['ortho_weight'] > 0 else 'no_ortho'
    
    return f"{spatial_type}_{temporal_type}_{combination}_{ortho}"