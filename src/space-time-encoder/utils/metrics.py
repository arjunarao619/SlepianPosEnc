"""
Evaluation metrics for ACE climate regression tasks
Extracted and cleaned from STE codebase
"""

import numpy as np
import torch


def compute_mae(predictions, targets, reduction='mean'):
    """
    Compute Mean Absolute Error
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        MAE value(s)
    """
    mae = torch.abs(predictions - targets)
    
    if reduction == 'mean':
        return torch.mean(mae)
    elif reduction == 'sum':
        return torch.sum(mae)
    else:
        return mae


def compute_rmse(predictions, targets, reduction='mean'):
    """
    Compute Root Mean Square Error
    
    Args:
        predictions: Model predictions  
        targets: Ground truth values
        reduction: 'mean' or 'none'
        
    Returns:
        RMSE value(s)
    """
    mse = (predictions - targets) ** 2
    
    if reduction == 'mean':
        return torch.sqrt(torch.mean(mse))
    else:
        return torch.sqrt(mse)


def compute_multivariable_metrics(predictions, targets):
    """
    Compute metrics for multi-variable regression (e.g., 8 temperature variables)
    
    Args:
        predictions: (batch, n_vars) predictions
        targets: (batch, n_vars) targets
        
    Returns:
        Dictionary with per-variable and averaged metrics
    """
    n_vars = predictions.shape[1]
    
    # Per-variable metrics
    mae_per_var = []
    rmse_per_var = []
    
    for i in range(n_vars):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        # Handle NaN values
        mask = ~torch.isnan(target_i)
        if mask.sum() > 0:
            mae_i = compute_mae(pred_i[mask], target_i[mask])
            rmse_i = compute_rmse(pred_i[mask], target_i[mask])
        else:
            mae_i = torch.tensor(float('nan'))
            rmse_i = torch.tensor(float('nan'))
            
        mae_per_var.append(mae_i)
        rmse_per_var.append(rmse_i)
    
    # Average across variables
    mae_per_var = torch.stack(mae_per_var)
    rmse_per_var = torch.stack(rmse_per_var)
    
    return {
        'mae_per_var': mae_per_var.cpu().numpy(),
        'rmse_per_var': rmse_per_var.cpu().numpy(),
        'mae_mean': torch.nanmean(mae_per_var).item(),
        'rmse_mean': torch.nanmean(rmse_per_var).item()
    }


def compute_spatial_temporal_metrics(predictions, targets, coords):
    """
    Compute spatial and temporal aggregate metrics
    
    Args:
        predictions: (n_samples, n_vars) predictions
        targets: (n_samples, n_vars) targets  
        coords: (n_samples, 3) [lon, lat, time] coordinates
        
    Returns:
        Dictionary with spatial and temporal averaged metrics
    """
    # Extract coordinate components
    lons = coords[:, 0]
    lats = coords[:, 1]
    times = coords[:, 2]
    
    # Compute spatial averages per timestep
    unique_times = torch.unique(times)
    spatial_avg_pred = []
    spatial_avg_target = []
    
    for t in unique_times:
        mask = times == t
        if mask.sum() > 0:
            spatial_avg_pred.append(predictions[mask].mean(dim=0))
            spatial_avg_target.append(targets[mask].mean(dim=0))
    
    if len(spatial_avg_pred) > 0:
        spatial_avg_pred = torch.stack(spatial_avg_pred)
        spatial_avg_target = torch.stack(spatial_avg_target)
        
        # Compute error of spatial averages
        spatial_mae = compute_mae(spatial_avg_pred, spatial_avg_target)
        spatial_rmse = compute_rmse(spatial_avg_pred, spatial_avg_target)
    else:
        spatial_mae = torch.tensor(float('nan'))
        spatial_rmse = torch.tensor(float('nan'))
    
    return {
        'spatial_mae': spatial_mae.item(),
        'spatial_rmse': spatial_rmse.item(),
    }