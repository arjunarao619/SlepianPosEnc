import torch
from .base_encoder import BaseLocationEncoder

class Cartesian3D(BaseLocationEncoder):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 3
    
    def forward(self, coords):
        # Input is lon/lat in degrees
        coords_rad = torch.deg2rad(coords)
        lon = coords_rad[:, 0]
        lat = coords_rad[:, 1]
        
        cos_lon = torch.cos(lon)
        sin_lon = torch.sin(lon)
        cos_lat = torch.cos(lat)
        sin_lat = torch.sin(lat)
        
        # Standard spherical to Cartesian conversion
        x = cos_lat * cos_lon
        y = cos_lat * sin_lon
        z = sin_lat
        
        return torch.stack([x, y, z], dim=1)