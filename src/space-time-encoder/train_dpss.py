# train_dpss.py
"""
Training script for STE+DPSS on ACE, with sequential hyperparam runs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import netCDF4
import numpy as np

from models.ste_encoder import STEEncoder
from utils.data_utils import DataNormalizer

# ----------------------------
# Top-level config
# ----------------------------
SEED = 42  # Default seed (can be overridden by bash runner)

BASE = {
    'data_path': '/scratch/local/arra4944_images/ace/',
    'train_frac': 0.01,
    'val_frac':   0.01,
    'test_frac':  0.01,
    'batch_size': 40000,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 500,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # regularization knobs (match your "with reg" best setting)
    'ortho_weight_final': 1e-3,
    'ortho_weight_space': 1e-3,
    'ortho_weight_time':  1e-3,
    'ortho_exponent': 1,
    'normality_flag': False,
    'time_grad_penalty_weight': 1e-6,

    # DPSS defaults
    'dpss_N': 365*4,            # 1460
    'dpss_concentration_threshold': 0.85,
}

# Small grid of DPSS runs (feel free to expand)
RUNS = [
    # (NW,  K,  optimized)
    (10, 32, False),
    (12, 40, False),
    (16, 40, False),
    (20, 40, False),
    (12, 40, True),
    (16, 40, True),
    (20, 40, True),
]

# ----------------------------
# Data loading (same as your STE script)
# ----------------------------
def load_ace_month(filepath):
    print(f"Loading {filepath}")
    with netCDF4.Dataset(filepath, 'r') as nc:
        lons  = nc.variables['grid_xt'][:] - 180
        lats  = nc.variables['grid_yt'][:]
        times = nc.variables['time'][:]

        temp_vars = []
        for i in range(8):
            name = f'air_temperature_{i}'
            if name in nc.variables:
                temp_vars.append(nc.variables[name][:])
            else:
                print(f"  Warning: {name} not found!")

        temps = np.stack(temp_vars, axis=-1)  # (time, lat, lon, 8)

        lon_grid, lat_grid, time_grid = np.meshgrid(lons, lats, times, indexing='ij')

        coords = np.stack([lon_grid.flatten(), lat_grid.flatten(), time_grid.flatten()], axis=1)   # (N,3)
        targets = temps.transpose(2, 1, 0, 3).reshape(-1, 8)                                       # (N,8)

        print(f"  Loaded month: coords {coords.shape}, targets {targets.shape}")
        return coords, targets

def prepare_ace_data(cfg):
    print("Loading ACE data...")
    coords_all, targets_all = [], []
    for month in range(1, 13):
        fp = os.path.join(cfg['data_path'], f"2021{month:02d}0100.nc")
        if os.path.exists(fp):
            c, y = load_ace_month(fp)
            coords_all.append(c); targets_all.append(y)

    coords = torch.tensor(np.concatenate(coords_all, axis=0), dtype=torch.float32)
    targets = torch.tensor(np.concatenate(targets_all, axis=0), dtype=torch.float32)

    # normalize time to [-1,1]
    tmin, tmax = coords[:,2].min(), coords[:,2].max()
    coords[:,2] = 2 * (coords[:,2] - tmin) / (tmax - tmin) - 1

    N  = coords.shape[0]
    ntr = int(N * cfg['train_frac'])
    nva = int(N * cfg['val_frac'])
    nte = int(N * cfg['test_frac'])

    idx = torch.randperm(N)
    tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:ntr+nva+nte]

    # normalize targets by train stats
    norm = DataNormalizer()
    norm.fit(targets[tr])
    targets_norm = norm.transform(targets)

    train_ds = TensorDataset(coords[tr], targets_norm[tr])
    val_ds   = TensorDataset(coords[va], targets_norm[va])
    test_ds  = TensorDataset(coords[te], targets_norm[te])

    print(f"Split sizes: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")
    return train_ds, val_ds, test_ds, norm

# ----------------------------
# Train / Eval (same behavior as your STE)
# ----------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    mse_fn = nn.MSELoss()
    total_mse = total_reg = total_tg = 0.0
    n_batches = 0

    def offdiag_penalty(feats):
        if feats is None: return torch.zeros((), device=device)
        B = feats.shape[0]
        if B == 0: return torch.zeros((), device=device)
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
        coords = coords.to(device); targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if w_tg > 0.0:
            coords.requires_grad_(True)

        preds = model(coords)
        mse_loss = mse_fn(preds, targets)

        reg_loss = torch.zeros((), device=device)
        if (w_final + w_space + w_time) > 0.0:
            sf = getattr(model, "_cache_space_feats", None)
            tf = getattr(model, "_cache_time_feats",  None)
            ff = getattr(model, "_cache_final_feats", None)
            if (sf is None or tf is None or ff is None) and hasattr(model, "get_regularization_loss"):
                reg_loss = model.get_regularization_loss(coords)
            else:
                reg_final = offdiag_penalty(ff)
                reg_space = offdiag_penalty(sf)
                reg_time  = offdiag_penalty(tf)
                reg_loss  = w_final * reg_final + w_space * reg_space + w_time * reg_time

        tg_loss = torch.zeros((), device=device)
        if w_tg > 0.0:
            s = preds.sum()
            grads = torch.autograd.grad(s, coords, create_graph=True, retain_graph=True)[0]
            t_grad = grads[:, 2]
            tg_loss = w_tg * t_grad.pow(2).mean()

        loss = mse_loss + reg_loss + tg_loss
        loss.backward(); optimizer.step()

        total_mse += float(mse_loss.detach())
        total_reg += float(reg_loss.detach()) if isinstance(reg_loss, torch.Tensor) else float(reg_loss)
        total_tg  += float(tg_loss.detach()) if isinstance(tg_loss, torch.Tensor) else float(tg_loss)
        n_batches += 1

    return total_mse/max(1,n_batches), total_reg/max(1,n_batches)

@torch.no_grad()
def evaluate(model, dataloader, device, normalizer):
    model.eval()
    preds_all, targs_all = [], []
    for coords, tnorm in dataloader:
        coords = coords.to(device); tnorm = tnorm.to(device)
        pnorm = model(coords)
        preds = normalizer.inverse_transform(pnorm.cpu())
        targs = normalizer.inverse_transform(tnorm.cpu())
        preds_all.append(preds); targs_all.append(targs)
    P = torch.cat(preds_all); T = torch.cat(targs_all)
    rmse = []; mae = []
    for i in range(8):
        diff = P[:,i]-T[:,i]
        rmse.append(torch.sqrt(torch.mean(diff**2)).item())
        mae.append(torch.mean(torch.abs(diff)).item())
    return float(np.mean(rmse)), float(np.mean(mae))

# ----------------------------
# Main: iterate DPSS configs
# ----------------------------
def main():
    print("="*50)
    print("STE + DPSS Training on ACE")
    print("="*50)

    train_ds, val_ds, test_ds, norm = prepare_ace_data(BASE)
    train_loader = DataLoader(train_ds, batch_size=BASE['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BASE['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=BASE['batch_size'])

    for (NW, K, OPT) in RUNS:
        print("\n" + "-"*50)
        tag = f"dpss_NW{NW}_K{K}_{'opt' if OPT else 'base'}"
        print(f"Run: {tag}")

        model = STEEncoder(
            spatial_L=20,
            temporal_type='dpss',
            temporal_dim=K,                         # K sequences (capped at 2*NW-1 inside)
            combination='concatenate',
            hidden_dim=1024,
            num_layers=4,
            output_dim=8,
            # regularization — keep same as your winning reg setup for fairness
            ortho_weight=BASE['ortho_weight_final'],
            ortho_weight_space=BASE['ortho_weight_space'],
            ortho_weight_time=BASE['ortho_weight_time'],
            ortho_exponent=BASE['ortho_exponent'],
            normality_flag=BASE['normality_flag'],
            time_grad_penalty_weight=BASE['time_grad_penalty_weight'],
            # DPSS
            dpss_N=BASE['dpss_N'],
            dpss_NW=NW,
            dpss_optimized=OPT,
            dpss_concentration_threshold=BASE['dpss_concentration_threshold'],
        ).to(BASE['device'])

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # weight decay only on head (as in your latest STE script)
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'head' in n:
                decay.append(p)
            else:
                nodecay.append(p)

        optimizer = optim.Adam(
            [
                {'params': decay,   'weight_decay': BASE['weight_decay']},
                {'params': nodecay, 'weight_decay': 0.0},
            ],
            lr=BASE['learning_rate'],
        )

        # Include seed in checkpoint filename
        BEST_CKPT = f"best_ste_{tag}_seed{SEED}.pt"
        best_val = float('inf'); patience = 0

        for epoch in range(BASE['max_epochs']):
            print(f"\nEpoch {epoch+1}/{BASE['max_epochs']}  [{tag}]")
            tr_mse, tr_reg = train_epoch(model, train_loader, optimizer, BASE['device'])
            va_rmse, va_mae = evaluate(model, val_loader, BASE['device'], norm)
            print(f"Train MSE: {tr_mse:.4f}, Reg: {tr_reg:.4f}")
            print(f"Val   RMSE: {va_rmse:.4f}, MAE: {va_mae:.4f}")

            if va_rmse < best_val:
                best_val = va_rmse
                torch.save(model.state_dict(), BEST_CKPT)
                print("  -> Saved best model")
                patience = 0
            else:
                patience += 1
                if patience >= BASE['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Final test
        print("\nTesting best checkpoint...")
        model.load_state_dict(torch.load(BEST_CKPT, map_location=BASE['device']))
        te_rmse, te_mae = evaluate(model, test_loader, BASE['device'], norm)
        print(f"[{tag}_seed{SEED}] Test RMSE: {te_rmse:.4f} K   Test MAE: {te_mae:.4f} K")

if __name__ == "__main__":
    main()