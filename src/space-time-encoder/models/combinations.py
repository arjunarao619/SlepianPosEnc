"""
Methods for combining spatial and temporal embeddings
Extracted and cleaned from STE codebase
"""

import torch
import torch.nn as nn


class EmbeddingCombiner(nn.Module):
    """Combines spatial and temporal embeddings using various strategies"""
    
    def __init__(self, combination_type='concatenate'):
        super().__init__()
        self.combination_type = combination_type
        
    def forward(self, spatial_emb, temporal_emb):
        """
        Combine spatial and temporal embeddings
        
        Args:
            spatial_emb: (batch, spatial_dim)
            temporal_emb: (batch, temporal_dim)
            
        Returns:
            Combined embeddings with appropriate dimensions
        """
        if self.combination_type == 'concatenate':
            return torch.cat([spatial_emb, temporal_emb], dim=1)
            
        elif self.combination_type == 'outer_product':
            # Tensor product: (batch, spatial, temporal) -> (batch, spatial*temporal)
            outer = torch.einsum('bi,bj->bij', spatial_emb, temporal_emb)
            return outer.flatten(start_dim=1)
            
        elif self.combination_type == 'hadamard':
            # Element-wise product (requires same dimensions)
            if spatial_emb.shape[1] != temporal_emb.shape[1]:
                raise ValueError(
                    f"Hadamard product requires same dimensions: "
                    f"got {spatial_emb.shape[1]} and {temporal_emb.shape[1]}"
                )
            return spatial_emb * temporal_emb
            
        elif self.combination_type == 'addition':
            # Simple addition (requires same dimensions)
            if spatial_emb.shape[1] != temporal_emb.shape[1]:
                raise ValueError(
                    f"Addition requires same dimensions: "
                    f"got {spatial_emb.shape[1]} and {temporal_emb.shape[1]}"
                )
            return spatial_emb + temporal_emb
            
        else:
            raise ValueError(f"Unknown combination type: {self.combination_type}")
    
    def get_output_dim(self, spatial_dim, temporal_dim):
        """Calculate output dimension based on combination type"""
        if self.combination_type == 'concatenate':
            return spatial_dim + temporal_dim
        elif self.combination_type == 'outer_product':
            return spatial_dim * temporal_dim
        elif self.combination_type in ['hadamard', 'addition']:
            if spatial_dim != temporal_dim:
                raise ValueError(f"{self.combination_type} requires equal dimensions")
            return spatial_dim
        else:
            raise ValueError(f"Unknown combination type: {self.combination_type}")