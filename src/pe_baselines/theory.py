import torch
import numpy as np
import math
from .base_encoder import BaseLocationEncoder
from .common import _cal_freq_list

class Theory(BaseLocationEncoder):
    def __init__(self, frequency_num=16, max_radius=10000, min_radius=1000, freq_init="geometric"):
        super().__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init
        
        # Calculate frequencies
        self.freq_list = _cal_freq_list(freq_init, frequency_num, max_radius, min_radius)
        self.freq_mat = np.repeat(np.expand_dims(self.freq_list, axis=1), 6, axis=1)
        
        # Three unit vectors 120 degrees apart
        self.unit_vec1 = np.array([1.0, 0.0])
        self.unit_vec2 = np.array([-0.5, math.sqrt(3) / 2.0])
        self.unit_vec3 = np.array([-0.5, -math.sqrt(3) / 2.0])
        
        self.embedding_dim = 6 * frequency_num
    
    def forward(self, coords):
        device = coords.device
        dtype = coords.dtype
        batch_size = coords.size(0)
        
        # Convert to numpy for processing
        coords_np = coords.cpu().numpy()
        
        # For Theory, we need to convert lon/lat to a planar projection
        # Using simple equirectangular projection
        coords_scaled = coords_np.copy()
        coords_scaled[:, 0] *= math.pi / 180.0  # lon to radians
        coords_scaled[:, 1] *= math.pi / 180.0  # lat to radians
        
        # Project onto three unit vectors
        angle_mat1 = np.expand_dims(np.dot(coords_scaled, self.unit_vec1), axis=-1)
        angle_mat2 = np.expand_dims(np.dot(coords_scaled, self.unit_vec2), axis=-1)
        angle_mat3 = np.expand_dims(np.dot(coords_scaled, self.unit_vec3), axis=-1)
        
        # Concatenate [vec1, vec1, vec2, vec2, vec3, vec3] for sin/cos pairs
        angle_mat = np.concatenate([
            angle_mat1, angle_mat1, 
            angle_mat2, angle_mat2, 
            angle_mat3, angle_mat3
        ], axis=-1)
        
        # Expand for frequencies
        angle_mat = np.expand_dims(angle_mat, axis=-2)
        angle_mat = np.repeat(angle_mat, self.frequency_num, axis=-2)
        
        # Apply frequency scaling
        angle_mat = angle_mat * self.freq_mat
        
        # Reshape
        spr_embeds = angle_mat.reshape(batch_size, -1)
        
        # Interleave sin and cos
        spr_embeds[:, 0::2] = np.sin(spr_embeds[:, 0::2])
        spr_embeds[:, 1::2] = np.cos(spr_embeds[:, 1::2])
        
        return torch.from_numpy(spr_embeds).to(dtype).to(device)