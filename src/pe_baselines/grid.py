import torch
import numpy as np
from .base_encoder import BaseLocationEncoder
from .common import _cal_freq_list

class Grid(BaseLocationEncoder):
    def __init__(self, frequency_num=16, max_radius=360, min_radius=1, freq_init="geometric"):
        super().__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        
        # Calculate frequencies
        self.freq_list = _cal_freq_list(freq_init, frequency_num, max_radius, min_radius)
        self.freq_mat = np.repeat(np.expand_dims(self.freq_list, axis=1), 2, axis=1)
        self.embedding_dim = 4 * frequency_num
    
    def forward(self, coords):
        device = coords.device
        dtype = coords.dtype
        batch_size = coords.size(0)
        
        # Convert degrees to radians
        coords_rad = torch.deg2rad(coords)
        coords_np = coords_rad.cpu().numpy()
        
        # Expand dimensions for broadcasting
        coords_mat = np.expand_dims(coords_np, axis=2)  # [batch, 2, 1]
        coords_mat = np.expand_dims(coords_mat, axis=3)  # [batch, 2, 1, 1]
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=2)  # [batch, 2, freq_num, 1]
        coords_mat = np.repeat(coords_mat, 2, axis=3)  # [batch, 2, freq_num, 2]
        
        # Apply frequencies
        spr_embeds = coords_mat * self.freq_mat
        
        # Apply sin and cos
        spr_embeds[:, :, :, 0::2] = np.sin(spr_embeds[:, :, :, 0::2])  # sin
        spr_embeds[:, :, :, 1::2] = np.cos(spr_embeds[:, :, :, 1::2])  # cos
        
        # Reshape to [batch, 4*frequency_num]
        spr_embeds = spr_embeds.transpose(0, 2, 1, 3)  # [batch, freq_num, 2, 2]
        spr_embeds = spr_embeds.reshape(batch_size, -1)
        
        return torch.from_numpy(spr_embeds).to(dtype).to(device)