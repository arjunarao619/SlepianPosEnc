import torch
import numpy as np
import math
from .base_encoder import BaseLocationEncoder
from .common import _cal_freq_list

class SphericalVariant(BaseLocationEncoder):
    """Base class for all spherical harmonic variants."""
    
    def __init__(self, variant_name, frequency_num=16, max_radius=360, min_radius=1, freq_init="geometric"):
        super().__init__()
        self.variant_name = variant_name
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        
        # Calculate frequencies
        self.freq_list = _cal_freq_list(freq_init, frequency_num, max_radius, min_radius)
        self.freq_mat = np.repeat(np.expand_dims(self.freq_list, axis=1), 2, axis=1)
        
        # Set embedding dimension based on variant
        if variant_name == "spherec":
            self.embedding_dim = 3 * frequency_num
        elif variant_name == "spherecplus":
            self.embedding_dim = 3 * frequency_num + 4 * frequency_num  # spherec + grid
        elif variant_name == "spherem":
            self.embedding_dim = 5 * frequency_num
        elif variant_name == "spheremplus":
            self.embedding_dim = 5 * frequency_num + 4 * frequency_num  # spherem + grid
    
    def forward(self, coords):
        device = coords.device
        dtype = coords.dtype
        batch_size = coords.size(0)
        
        # Convert to radians
        coords_rad = torch.deg2rad(coords)
        coords_np = coords_rad.cpu().numpy()
        
        # Expand for frequency processing
        coords_mat = np.expand_dims(coords_np, axis=2)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=2)
        
        # Scale by frequencies
        lon_scaled = coords_mat[:, 0, :] * self.freq_list
        lat_scaled = coords_mat[:, 1, :] * self.freq_list
        
        # Original unscaled coordinates for spherem
        lon_orig = coords_np[:, 0:1]
        lat_orig = coords_np[:, 1:2]
        
        # Compute trigonometric functions
        sin_lon = np.sin(lon_scaled)
        cos_lon = np.cos(lon_scaled)
        sin_lat = np.sin(lat_scaled)
        cos_lat = np.cos(lat_scaled)
        
        sin_lon_orig = np.sin(lon_orig)
        cos_lon_orig = np.cos(lon_orig)
        sin_lat_orig = np.sin(lat_orig)
        cos_lat_orig = np.cos(lat_orig)
        
        if self.variant_name == "spherec":
            # [sin(φ), cos(φ)cos(λ), cos(φ)sin(λ)]
            embeds = np.concatenate([
                sin_lat,
                cos_lat * cos_lon,
                cos_lat * sin_lon
            ], axis=-1)
            
        elif self.variant_name == "spherecplus":
            # spherec + grid
            spherec_embeds = np.concatenate([
                sin_lat,
                cos_lat * cos_lon,
                cos_lat * sin_lon
            ], axis=-1)
            
            grid_embeds = np.concatenate([
                sin_lon,
                cos_lon,
                sin_lat,
                cos_lat
            ], axis=-1)
            
            embeds = np.concatenate([spherec_embeds, grid_embeds], axis=-1)
            
        elif self.variant_name == "spherem":
            # More complex mixing of scaled and unscaled
            embeds = np.concatenate([
                sin_lat,
                cos_lat * np.repeat(cos_lon_orig, self.frequency_num, axis=1),
                np.repeat(cos_lat_orig, self.frequency_num, axis=1) * cos_lon,
                cos_lat * np.repeat(sin_lon_orig, self.frequency_num, axis=1),
                np.repeat(cos_lat_orig, self.frequency_num, axis=1) * sin_lon
            ], axis=-1)
            
        elif self.variant_name == "spheremplus":
            # spherem + grid
            spherem_embeds = np.concatenate([
                sin_lat,
                cos_lat * np.repeat(cos_lon_orig, self.frequency_num, axis=1),
                np.repeat(cos_lat_orig, self.frequency_num, axis=1) * cos_lon,
                cos_lat * np.repeat(sin_lon_orig, self.frequency_num, axis=1),
                np.repeat(cos_lat_orig, self.frequency_num, axis=1) * sin_lon
            ], axis=-1)
            
            grid_embeds = np.concatenate([
                sin_lon,
                cos_lon,
                sin_lat,
                cos_lat
            ], axis=-1)
            
            embeds = np.concatenate([spherem_embeds, grid_embeds], axis=-1)
        
        return torch.from_numpy(embeds).to(dtype).to(device)

# Create specific classes for each variant
class SphereC(SphericalVariant):
    def __init__(self, frequency_num=16, max_radius=360, min_radius=1, freq_init="geometric"):
        super().__init__("spherec", frequency_num, max_radius, min_radius, freq_init)

class SphereCPlus(SphericalVariant):
    def __init__(self, frequency_num=16, max_radius=360, min_radius=1, freq_init="geometric"):
        super().__init__("spherecplus", frequency_num, max_radius, min_radius, freq_init)

class SphereM(SphericalVariant):
    def __init__(self, frequency_num=16, max_radius=360, min_radius=1, freq_init="geometric"):
        super().__init__("spherem", frequency_num, max_radius, min_radius, freq_init)

class SphereMPlus(SphericalVariant):
    def __init__(self, frequency_num=16, max_radius=360, min_radius=1, freq_init="geometric"):
        super().__init__("spheremplus", frequency_num, max_radius, min_radius, freq_init)