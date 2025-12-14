#!/usr/bin/env python3
# train_baselines.py
import os, argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import netCDF4

from models.ste_encoder import STEEncoder
from utils.data_utils import DataNormalizer

def load_ace_month(fp):
    with netCDF4.Dataset(fp, 'r') as nc:
        lons = nc.variables['grid_xt'][:] - 180
        lats = nc.variables['grid_yt'][:]
        times = nc.variables['time'][:]
        temps = np.stack([nc.variables[f'air_temperature_{i}'][:] for i in range(8)], axis=-1)
        lon_g, lat_g, time_g = np.meshgrid(lons, lats, times, indexing='ij')
        coords = np.stack([lon_g.flatten(), lat_g.flatten(), time_g.flatten()], axis=1).astype(np.float32)
        targets = temps.transpose(2,1,0,3).reshape(-1, 8).astype(np.float32)
        return coords, targets

def prepare_ace(data_path, train_frac=0.01, val_frac=0.01, test_frac=0.01):
    Cs, Ys = [], []
    for m in range(1, 13):
        fp = os.path.join(data_path, f"2021{m:02d}0100.nc")
        if os.path.exists(fp):
            c, y = load_ace_month(fp); Cs.append(c); Ys.append(y)
    coords = torch.from_numpy(np.concatenate(Cs, 0))
    targets = torch.from_numpy(np.concatenate(Ys, 0))
    # normalize time to [-1, 1]
    tmin, tmax = coords[:,2].min(), coords[:,2].max()
    coords[:,2] = 2 * (coords[:,2] - tmin) / (tmax - tmin) - 1

    N = coords.shape[0]
    n_tr, n_va, n_te = int(N*train_frac), int(N*val_frac), int(N*test_frac)
    perm = torch.randperm(N)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:n_tr+n_va+n_te]

    normalizer = DataNormalizer()
    normalizer.fit(targets[tr])
    targets_norm = normalizer.transform(targets)

    train_ds = TensorDataset(coords[tr], targets_norm[tr])
    val_ds   = TensorDataset(coords[va], targets_norm[va])
    test_ds  = TensorDataset(coords[te], targets_norm[te])
    return train_ds, val_ds, test_ds, normalizer

@torch.no_grad()
def evaluate(model, dl, device, normalizer):
    model.eval(); P, T = [], []
    for x, y_n in dl:
        x = x.to(device, non_blocking=True)
        y_n = y_n.to(device, non_blocking=True)
        p_n = model(x)
        P.append(normalizer.inverse_transform(p_n.cpu()))
        T.append(normalizer.inverse_transform(y_n.cpu()))
    P, T = torch.cat(P), torch.cat(T)
    rmse, mae = [], []
    for i in range(8):
        mse = torch.mean((P[:,i]-T[:,i])**2).item()
        rmse.append(math.sqrt(mse))
        mae.append(torch.mean(torch.abs(P[:,i]-T[:,i])).item())
    return float(np.mean(rmse)), float(np.mean(mae))

def train_epoch(model, dl, opt, device):
    model.train()
    mse_fn = nn.MSELoss()
    total_mse, total_reg, total_tg, n = 0.0, 0.0, 0.0, 0
    
    def offdiag(feats):
        if feats is None: return torch.zeros((), device=device)
        B = feats.shape[0]
        if B == 0: return torch.zeros((), device=device)
        C = (feats.T @ feats)/B
        P = C - torch.diag(torch.diag(C))
        return torch.linalg.norm(P, 'fro') ** max(1, int(getattr(model, "ortho_exponent", 1)))
    
    wF = float(getattr(model, "ortho_weight", 0.0))
    wS = float(getattr(model, "ortho_weight_space", 0.0))
    wT = float(getattr(model, "ortho_weight_time", 0.0))
    w_tg = float(getattr(model, "time_grad_penalty_weight", 0.0))

    for coords, targets in dl:
        coords = coords.to(device)
        targets = targets.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # Enable gradient tracking for time coordinate if needed
        if w_tg > 0.0:
            coords.requires_grad_(True)
        
        preds = model(coords)
        mse = mse_fn(preds, targets)
        
        # Orthogonality regularization
        reg = torch.zeros((), device=device)
        if (wF + wS + wT) > 0.0:
            sf = getattr(model, "_cache_space_feats", None)
            tf = getattr(model, "_cache_time_feats",  None)
            ff = getattr(model, "_cache_final_feats", None)
            if (sf is None or ff is None) and hasattr(model, "get_regularization_loss"):
                reg = model.get_regularization_loss(coords)
            else:
                reg = wF*offdiag(ff) + wS*offdiag(sf) + wT*offdiag(tf)
        
        # Time gradient penalty
        tg_loss = torch.zeros((), device=device)
        if w_tg > 0.0:
            s = preds.sum()
            grads = torch.autograd.grad(s, coords, create_graph=True, retain_graph=True, allow_unused=False)[0]
            t_grad = grads[:, 2]  # time column
            tg_loss = w_tg * t_grad.pow(2).mean()
        
        loss = mse + reg + tg_loss
        loss.backward()
        opt.step()
        
        total_mse += float(mse.detach())
        total_reg += float(reg.detach()) if isinstance(reg, torch.Tensor) else float(reg)
        total_tg += float(tg_loss.detach()) if isinstance(tg_loss, torch.Tensor) else float(tg_loss)
        n += 1
    
    return total_mse/max(1,n), total_reg/max(1,n)

def main():
    p = argparse.ArgumentParser("Train STE baselines for ACE")
    p.add_argument("--data_path", type=str, default="/scratch/local/arra4944_images/ace/")
    p.add_argument("--temporal_type", choices=["no_time","time_copy","triangle","monomial","legendre","fourier"], default="legendre")
    p.add_argument("--use_reg", action="store_true", help="Enable regularization (matching train_ste.py)")
    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--train_frac", type=float, default=0.01)
    p.add_argument("--val_frac",   type=float, default=0.01)
    p.add_argument("--test_frac",  type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=40000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=15)  # FIX 1: Changed from 5 to 15
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

    # FIX 2: Set regularization weights to match train_ste.py
    if args.use_reg:
        ortho_weight_final = 1e-3
        ortho_weight_space = 1e-3
        ortho_weight_time = 1e-3
        time_grad_penalty = 1e-6
        reg_suffix = "reg"
    else:
        ortho_weight_final = 0.0
        ortho_weight_space = 0.0
        ortho_weight_time = 0.0
        time_grad_penalty = 0.0
        reg_suffix = "noreg"

    train_ds, val_ds, test_ds, normalizer = prepare_ace(args.data_path, args.train_frac, args.val_frac, args.test_frac)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=16)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, num_workers=16)

    # FIX 3: Apply regularization to ALL components (final, space, time)
    model = STEEncoder(
        spatial_L=20,
        temporal_type=args.temporal_type,
        temporal_dim=40,
        combination='concatenate',
        hidden_dim=1024,
        num_layers=4,
        output_dim=8,
        ortho_weight=ortho_weight_final,           # Final layer
        ortho_weight_space=ortho_weight_space,     # Space encoder
        ortho_weight_time=ortho_weight_time,       # Time encoder
        normality_flag=False,
        ortho_exponent=1,
        time_grad_penalty_weight=time_grad_penalty,  # Time gradient smoothness
    ).to(args.device)

    # Weight decay only on head (matching train_ste.py)
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (decay if 'head' in n else nodecay).append(p)

    opt = optim.Adam(
        [{'params': decay, 'weight_decay': args.weight_decay},
         {'params': nodecay, 'weight_decay': 0.0}],
        lr=args.lr
    )

    best_val = float('inf')
    best_path = Path(f"best_{args.temporal_type}_{reg_suffix}_seed{args.seed}.pt")
    patience_counter = 0
    
    print(f"Training {args.temporal_type} ({'WITH' if args.use_reg else 'WITHOUT'} regularization)")
    print(f"Checkpoint: {best_path}")
    
    for epoch in range(args.max_epochs):
        mse, reg = train_epoch(model, train_dl, opt, args.device)
        val_rmse, val_mae = evaluate(model, val_dl, args.device, normalizer)
        print(f"Epoch {epoch+1}: TrainMSE={mse:.4f} Reg={reg:.4f} | Val RMSE={val_rmse:.4f} MAE={val_mae:.4f}")
        
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"Early stopping @ epoch {epoch+1}")
            break

    print("\n" + "="*50)
    print("Final Test Evaluation")
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_rmse, test_mae = evaluate(model, test_dl, args.device, normalizer)
    print(f"Test RMSE: {test_rmse:.4f} K")
    print(f"Test MAE:  {test_mae:.4f} K")

if __name__ == "__main__":
    main()