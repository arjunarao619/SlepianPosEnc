"""
Training script for STE on ACE dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import netCDF4
import numpy as np
from pathlib import Path

# Import our modules
from models.ste_encoder import STEEncoder
from utils.metrics import compute_rmse, compute_mae
from utils.data_utils import DataNormalizer

# Configuration

USE_REG = True #Set to true if we want to regularize according to the STE paper

config = {
    'data_path': '/scratch/local/arra4944_images/ace/',
    'train_frac': 0.01,
    'val_frac': 0.01,
    'test_frac': 0.01,
    'batch_size': 40000,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 15,  
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # new: regularization hyperparams
    'ortho_weight_final': 1e-3,
    'ortho_weight_space': 1e-3,
    'ortho_weight_time': 1e-3,
    'ortho_exponent': 1,
    'normality_flag': False,
    'time_grad_penalty_weight': 1e-6,
}

if not USE_REG:
    config.update({
        'ortho_weight_final': 0.0,
        'ortho_weight_space': 0.0,
        'ortho_weight_time': 0.0,
        'time_grad_penalty_weight': 0.0,
    })
    CKPT_NAME = 'best_ste_model_noreg.pt'
else:
    CKPT_NAME = 'best_ste_model_reg.pt'


def load_ace_month(filepath):
    """Load one month of ACE data"""
    print(f"Loading {filepath}")
    
    with netCDF4.Dataset(filepath, 'r') as nc:
        # Get dimensions
        time_steps = nc.dimensions['time'].size
        lat_size = nc.dimensions['grid_yt'].size  
        lon_size = nc.dimensions['grid_xt'].size
        
        # Get coordinates
        lons = nc.variables['grid_xt'][:] - 180  # Shift to [-180, 180]
        lats = nc.variables['grid_yt'][:]
        times = nc.variables['time'][:]
        
        # Get the 8 temperature variables (air_temperature_0 through air_temperature_7)
        temp_vars = []
        for i in range(8):
            var_name = f'air_temperature_{i}'
            if var_name in nc.variables:
                data = nc.variables[var_name][:]
                temp_vars.append(data)
                print(f"  Loaded {var_name}: shape {data.shape}")
            else:
                print(f"  Warning: {var_name} not found!")
        
        # Stack temperature variables
        temps = np.stack(temp_vars, axis=-1)  # (time, lat, lon, 8)
        
        # Create coordinate grids
        lon_grid, lat_grid, time_grid = np.meshgrid(lons, lats, times, indexing='ij')
        
        # Flatten everything
        coords = np.stack([
            lon_grid.flatten(),
            lat_grid.flatten(), 
            time_grid.flatten()
        ], axis=1)  # (n_samples, 3)
        
        targets = temps.transpose(2, 1, 0, 3).reshape(-1, 8)  # (n_samples, 8)
        
        print(f"  Loaded month: coords {coords.shape}, targets {targets.shape}")
        
        return coords, targets

def prepare_ace_data():
    """Load and prepare ACE dataset"""
    print("Loading ACE data...")
    
    # Load all months
    all_coords = []
    all_targets = []
    
    for month in range(1, 13):
        filepath = os.path.join(
            config['data_path'],
            f"2021{month:02d}0100.nc"
        )
        
        if os.path.exists(filepath):
            coords, targets = load_ace_month(filepath)
            all_coords.append(coords)
            all_targets.append(targets)
    
    # Concatenate all data
    coords = np.concatenate(all_coords, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Convert to torch tensors
    coords = torch.FloatTensor(coords)
    targets = torch.FloatTensor(targets)
    
    # Normalize time to [-1, 1]
    time_min = coords[:, 2].min()
    time_max = coords[:, 2].max()
    coords[:, 2] = 2 * (coords[:, 2] - time_min) / (time_max - time_min) - 1
    
    print(f"Loaded data shape: coords {coords.shape}, targets {targets.shape}")
    
    # Split data
    n_total = coords.shape[0]
    n_train = int(n_total * config['train_frac'])
    n_val = int(n_total * config['val_frac'])
    n_test = int(n_total * config['test_frac'])
    
    # Random permutation
    indices = torch.randperm(n_total)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:n_train+n_val+n_test]
    
    # Normalize targets using training statistics
    normalizer = DataNormalizer()
    normalizer.fit(targets[train_idx])
    
    targets_norm = normalizer.transform(targets)
    
    # Create datasets
    train_ds = TensorDataset(coords[train_idx], targets_norm[train_idx])
    val_ds = TensorDataset(coords[val_idx], targets_norm[val_idx])
    test_ds = TensorDataset(coords[test_idx], targets_norm[test_idx])
    
    print(f"Split sizes: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")
    
    return train_ds, val_ds, test_ds, normalizer

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch with same-pass orthoreg and optional time-grad penalty."""
    model.train()
    mse_fn = nn.MSELoss()

    total_mse = 0.0
    total_reg = 0.0
    total_tg  = 0.0  # tracked but not returned (keeps API stable)
    n_batches = 0

    # small helper: off-diagonal Frobenius penalty on correlation matrix
    def offdiag_penalty(feats):
        if feats is None:
            return torch.zeros((), device=device)
        B = feats.shape[0]
        if B == 0:
            return torch.zeros((), device=device)
        C = (feats.T @ feats) / B
        if getattr(model, "normality_flag", False):
            I = torch.eye(C.shape[0], device=device, dtype=C.dtype)
            P = C - I
        else:
            P = C - torch.diag(torch.diag(C))
        exp = max(1, int(getattr(model, "ortho_exponent", 1)))
        return torch.linalg.norm(P, ord='fro') ** exp

    w_final = float(getattr(model, "ortho_weight", 0.0))
    w_space = float(getattr(model, "ortho_weight_space", 0.0))
    w_time  = float(getattr(model, "ortho_weight_time", 0.0))
    w_tg    = float(getattr(model, "time_grad_penalty_weight", 0.0))

    for coords, targets in dataloader:
        coords = coords.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # enable time derivative only if needed
        if w_tg > 0.0:
            coords.requires_grad_(True)

        # Forward: should cache per-stream and final features in model._cache_*
        preds = model(coords)

        # Primary loss (on normalized scale)
        mse_loss = mse_fn(preds, targets)

        # Orthogonality regularization from cached activations (same pass)
        reg_loss = torch.zeros((), device=device)
        if (w_final + w_space + w_time) > 0.0:
            space_feats = getattr(model, "_cache_space_feats", None)
            time_feats  = getattr(model, "_cache_time_feats",  None)
            final_feats = getattr(model, "_cache_final_feats", None)

            # Fallback to legacy method if caches are missing
            if (space_feats is None or time_feats is None or final_feats is None) and hasattr(model, "get_regularization_loss"):
                reg_loss = model.get_regularization_loss(coords)
            else:
                reg_final = offdiag_penalty(final_feats)
                reg_space = offdiag_penalty(space_feats)
                reg_time  = offdiag_penalty(time_feats)
                reg_loss  = w_final * reg_final + w_space * reg_space + w_time * reg_time

        # Time-gradient penalty ‖∂f/∂t‖^2
        tg_loss = torch.zeros((), device=device)
        if w_tg > 0.0:
            # derivative of sum of outputs w.r.t. the input time coordinate
            s = preds.sum()
            grads = torch.autograd.grad(s, coords, create_graph=True, retain_graph=True, allow_unused=False)[0]
            t_grad = grads[:, 2]  # time column
            tg_loss = w_tg * t_grad.pow(2).mean()

        loss = mse_loss + reg_loss + tg_loss
        loss.backward()
        optimizer.step()

        total_mse += float(mse_loss.detach())
        total_reg += float(reg_loss.detach()) if isinstance(reg_loss, torch.Tensor) else float(reg_loss)
        total_tg  += float(tg_loss.detach()) if isinstance(tg_loss, torch.Tensor) else float(tg_loss)
        n_batches += 1

    avg_mse = total_mse / max(1, n_batches)
    avg_reg = total_reg / max(1, n_batches)
    # If you want to log the time-grad term, use `avg_tg = total_tg / max(1, n_batches)`
    return avg_mse, avg_reg


def evaluate(model, dataloader, device, normalizer):
    """Evaluate model - matching STE's approach"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for coords, targets_norm in dataloader:
            coords = coords.to(device)
            targets_norm = targets_norm.to(device)
            
            # Get predictions (normalized scale)
            predictions_norm = model(coords)
            
            # CRITICAL: Denormalize to original scale (Kelvin)
            predictions = normalizer.inverse_transform(predictions_norm.cpu())
            targets = normalizer.inverse_transform(targets_norm.cpu())
            
            all_preds.append(predictions)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Compute metrics on ORIGINAL SCALE (matching STE)
    rmse_per_var = []
    mae_per_var = []
    
    for i in range(8):
        mse = torch.mean((all_preds[:, i] - all_targets[:, i])**2)
        rmse_per_var.append(torch.sqrt(mse).item())
        mae_per_var.append(torch.mean(torch.abs(all_preds[:, i] - all_targets[:, i])).item())
    
    # Average across 8 temperature variables (as in Table 2)
    rmse_avg = np.mean(rmse_per_var)
    mae_avg = np.mean(mae_per_var)
    
    return rmse_avg, mae_avg

def main():
    """Main training loop - matching STE setup"""
    print("="*50)
    print("STE Training on ACE Dataset")
    print("="*50)
    
    # Load data
    train_ds, val_ds, test_ds, normalizer = prepare_ace_data()
    
    # Create dataloaders (match their batch size)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])
    
    # Create model (matching their architecture)
    model = STEEncoder(
        spatial_L=20,
        temporal_type='',
        temporal_dim=40,
        combination='concatenate',
        hidden_dim=1024,
        num_layers=4,
        output_dim=8,
        # map config -> model args
        ortho_weight=config['ortho_weight_final'],
        ortho_weight_space=config['ortho_weight_space'],
        ortho_weight_time=config['ortho_weight_time'],
        ortho_exponent=config['ortho_exponent'],
        normality_flag=config['normality_flag'],
        time_grad_penalty_weight=config['time_grad_penalty_weight'],
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'head' in n:
            decay.append(p)
        else:
            nodecay.append(p)

    optimizer = optim.Adam(
        [
            {'params': decay,   'weight_decay': config['weight_decay']},
            {'params': nodecay, 'weight_decay': 0.0},
        ],
        lr=config['learning_rate'],
    )
    
    # Training with early stopping
    BEST_CKPT = CKPT_NAME

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch+1}/{config['max_epochs']}")
        mse_loss, reg_loss = train_epoch(model, train_loader, optimizer, config['device'])

        val_rmse, val_mae = evaluate(model, val_loader, config['device'], normalizer)
        print(f"Train MSE: {mse_loss:.4f}, Reg Loss: {reg_loss:.4f}")
        print(f"Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), BEST_CKPT)
            print("  -> Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("\n" + "="*50)
    print("Final Test Evaluation")
    model.load_state_dict(torch.load(BEST_CKPT, map_location=config['device']))
    test_rmse, test_mae = evaluate(model, test_loader, config['device'], normalizer)
    print(f"Test RMSE: {test_rmse:.4f} K")
    print(f"Test MAE: {test_mae:.4f} K")
    
if __name__ == "__main__":
    main()