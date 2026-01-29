import torch
from .base_encoder import BaseLocationEncoder

class Direct(BaseLocationEncoder):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 2
    
    def forward(self, coords):
        # Map degrees to radians and center around zero
        coords = torch.deg2rad(coords) - torch.pi
        return coords
