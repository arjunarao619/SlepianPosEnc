import torch
from .base_encoder import BaseLocationEncoder

class Wrap(BaseLocationEncoder):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 4
    
    def forward(self, coords):
        # coords: [batch, 2] with (longitude, latitude) in degrees
        coords_rad = torch.deg2rad(coords)
        lon = coords_rad[:, 0]
        lat = coords_rad[:, 1]
        
        cos_lon = torch.cos(lon).unsqueeze(-1)
        sin_lon = torch.sin(lon).unsqueeze(-1)
        cos_lat = torch.cos(lat).unsqueeze(-1)
        sin_lat = torch.sin(lat).unsqueeze(-1)
        
        return torch.cat([cos_lon, sin_lon, cos_lat, sin_lat], dim=1)