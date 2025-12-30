# models/dpss_time.py

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import windows
from scipy.interpolate import interp1d

class DPSSTimeEncoder(nn.Module):
    """
    DPSS (Discrete Prolate Spheroidal Sequences) temporal encoding
    Optimal time-frequency concentration for finite-duration signals
    """
    
    def __init__(self, 
                 N=1460,      # 365 days × 4 timesteps/day for ACE
                 NW=20,       # Time-bandwidth product
                 K=40,        # Number of sequences to use
                 concentration_threshold=0.9):
        super().__init__()
        
        self.N = N
        self.NW = NW
        self.K = min(K, int(2*NW) - 1)  # Can't exceed 2*NW - 1
        
        # Generate DPSS sequences
        dpss_seqs, eigenvalues = windows.dpss(N, NW, self.K, return_ratios=True)
        order = np.argsort(eigenvalues)[::-1]
        dpss_seqs = dpss_seqs[order]
        eigenvalues = eigenvalues[order]
        # hard-select top-K
        dpss_seqs = dpss_seqs[:self.K]
        eigenvalues = eigenvalues[:self.K]
        
        # Keep sequences based on concentration threshold
        good_idx = eigenvalues > concentration_threshold
        n_good = good_idx.sum()
        
        if n_good < self.K:
            print(f"Warning: Only {n_good}/{self.K} DPSS sequences have concentration > {concentration_threshold}")
            # Use top K sequences regardless of threshold
            sorted_idx = np.argsort(eigenvalues)[::-1][:self.K]
            dpss_seqs = dpss_seqs[sorted_idx]
            eigenvalues = eigenvalues[sorted_idx]
        
        # Normalize sequences
        for i in range(len(dpss_seqs)):
            dpss_seqs[i] = dpss_seqs[i] / np.linalg.norm(dpss_seqs[i])
        
        # Store as buffers
        self.register_buffer('dpss_seqs', torch.from_numpy(dpss_seqs).float())
        self.register_buffer('eigenvalues', torch.from_numpy(eigenvalues).float())
        
        # Set output dimension
        self.output_dim = len(dpss_seqs)
        
        # Create interpolators for smooth evaluation
        self.interpolators = []
        t_discrete = np.linspace(-1, 1, self.N)
        
        for i in range(self.output_dim):
            seq = self.dpss_seqs[i].cpu().numpy()
            interp = interp1d(t_discrete, seq, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
            self.interpolators.append(interp)
        
        print(f"DPSS Encoder initialized:")
        print(f"  N={N}, NW={NW}, K={self.output_dim}")
        print(f"  Mean eigenvalue: {eigenvalues.mean():.4f}")
        
    def forward(self, time_coords):
        """
        Args:
            time_coords: [batch, 1] or [batch] tensor, normalized to [-1, 1]
        """
        if time_coords.dim() == 2:
            time_coords = time_coords.squeeze(1)
        
        batch_size = time_coords.shape[0]
        device = time_coords.device
        
        # Move to CPU for interpolation
        t_cpu = time_coords.detach().cpu().numpy()
        
        # Initialize output
        features = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Evaluate each DPSS sequence
        for i in range(self.output_dim):
            values = self.interpolators[i](t_cpu)
            features[:, i] = torch.from_numpy(values).float().to(device)
        
        return features


class OptimizedDPSSTimeEncoder(nn.Module):
    """
    Optimized DPSS encoder with learnable mixing weights
    """
    
    def __init__(self,
                 N=1460,
                 NW=20,
                 K_dpss=60,    # Generate more DPSS sequences
                 K_output=40,  # Project to this dimension
                 concentration_threshold=0.8):
        super().__init__()
        
        # Create base DPSS encoder
        self.dpss_base = DPSSTimeEncoder(N, NW, K_dpss, concentration_threshold)
        
        # Learnable projection - adjust dimensions
        actual_dpss_dim = self.dpss_base.output_dim
        self.projection = nn.Linear(actual_dpss_dim, K_output, bias=False)
        
        # Initialize projection
        nn.init.orthogonal_(self.projection.weight)
        
        self.output_dim = K_output
        
        print(f"Optimized DPSS: {actual_dpss_dim} DPSS → {K_output} output dims")
        
    def forward(self, time_coords):
        # Get DPSS features
        dpss_features = self.dpss_base(time_coords)
        
        # Apply learnable projection
        output = self.projection(dpss_features)
        
        return output