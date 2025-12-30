# dpss_experiments/run_comparison.py

import torch
from models.space_time_encoder import SpaceTimeEncoder
from models.dpss_time import DPSSTimeEncoder
from models.baseline_time import FourierEncoder, LegendreEncoder
from models.spatial_slepian import SpatialSlepianEncoder  # Your implementation

def compare_encoders():
    # Spatial encoder (your Slepians)
    spatial_enc = SpatialSlepianEncoder(region='global', L=20)
    
    # Time encoders to compare
    encoders = {
        'DPSS': DPSSTimeEncoder(N=365*4, NW=20),  # 4 timesteps/day * 365 days
        'Fourier': FourierEncoder(dim=40),
        'Legendre': LegendreEncoder(dim=40)
    }
    
    results = {}
    
    for name, time_enc in encoders.items():
        model = SpaceTimeEncoder(
            spatial_encoder=spatial_enc,
            temporal_encoder=time_enc,
            combination_type='concatenate',
            ortho_weight=1.0 if name == 'DPSS' else 0.0
        )
        
        # Train and evaluate
        results[name] = train_and_evaluate(model)
    
    return results