"""
Orthogonality regularizers for Space-Time Encoders
Extracted and cleaned from STE codebase
"""

import torch
import torch.nn as nn


def compute_correlation_matrix(features):
    """Compute normalized correlation matrix of features"""
    batch_size = features.shape[0]
    correlation = torch.mm(features.T, features) / batch_size
    return correlation


def orthogonality_regularizer(features, normality=False, exponent=1):
    """
    Compute orthogonality regularization loss
    
    Args:
        features: Tensor of shape (batch_size, feature_dim)
        normality: If True, enforce identity matrix (orthonormal)
                  If False, only enforce diagonal (orthogonal)
        exponent: Power to raise the regularizer to
    
    Returns:
        Regularization loss (scalar)
    """
    correlation = compute_correlation_matrix(features)
    device = correlation.device
    
    if normality:
        # Enforce correlation = identity matrix
        identity = torch.eye(correlation.shape[0], device=device)
        regularizer = torch.norm(correlation - identity, p='fro')
    else:
        # Only enforce diagonal structure
        diagonal = torch.diag(torch.diag(correlation))
        regularizer = torch.norm(correlation - diagonal, p='fro')
    
    return regularizer ** exponent


class OrthogonalityRegularizer(nn.Module):
    """Module wrapper for orthogonality regularization"""
    
    def __init__(self, weight=1.0, normality=False, exponent=1):
        super().__init__()
        self.weight = weight
        self.normality = normality
        self.exponent = exponent
    
    def forward(self, features):
        if self.weight > 0:
            return self.weight * orthogonality_regularizer(
                features, self.normality, self.exponent
            )
        return 0.0