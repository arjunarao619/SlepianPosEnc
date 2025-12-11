import torch
import torch.nn as nn

class BaseLocationEncoder(nn.Module):
    """Base class to standardize encoder interface."""
    def __init__(self):
        super().__init__()
        self.embedding_dim = None
    
    @property
    def n_features(self):
        """Alias for compatibility with existing code."""
        return self.embedding_dim