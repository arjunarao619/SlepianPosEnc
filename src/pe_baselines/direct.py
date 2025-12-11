import torch
from .base_encoder import BaseLocationEncoder

class Direct(BaseLocationEncoder):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 2
    
    def forward(self, coords):
        # Convert to radians and shift to [-π, π] range
        coords = torch.deg2rad(coords) - torch.pi
        return coords
