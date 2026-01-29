#!/usr/bin/env python3
# train_slepian_vs_sh_multiscale_v3.py
# Optimized version with precomputed design matrix support for parallel experiments.

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Import trunk classes from nn module
from nn.models import ResMLPTrunk, SirenTrunk, GLUTrunk

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter

# Only import if not using precomputed
try:
    from spherical_harmonics_ylm import SH as SH_analytic
    HAVE_SH = True
except ImportError:
    HAVE_SH = False

try:
    import pyshtools as pysh
    HAVE_PYSH = True
except ImportError:
    HAVE_PYSH = False

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_float32_matmul_precision("high")

EMBEDDING_CONFIGS = {
    "alphaearth": {"column": "embedding", "dim": 63},
    "galileo": {"column": "galileo_embedding", "dim": 128},
}

# ==================== UTILITIES ====================

def to_raster(values, gy0, gx0, ny, nx, fill=np.nan, dtype=np.float32):
    A = np.full((ny, nx), fill, dtype=dtype)
    A[gy0, gx0] = values
    return A

def gaussian_smooth_nan(A, sigma_y_pix, sigma_x_pix):
    M = np.isfinite(A).astype(np.float32)
    V = np.where(M > 0, A, 0.0).astype(np.float32)
    num = gaussian_filter(V, sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    den = gaussian_filter(M, sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    return np.where(den > 0, num / np.maximum(den, 1e-8), np.nan)

def r2_masked(y, yhat, m):
    y, yhat = y[m], yhat[m]
    if y.size < 2: return np.nan
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot

def morans_I_rook(A):
    M = np.isfinite(A)
    if M.sum() < 4: return np.nan
    Z = np.where(M, A - np.nanmean(A), 0.0)
    W, num = 0.0, 0.0
    ny, nx = A.shape
    R = M[:, 1:] & M[:, :-1]
    W += R.sum(); num += np.sum(Z[:, 1:][R] * Z[:, :-1][R])
    C = M[1:, :] & M[:-1, :]
    W += C.sum(); num += np.sum(Z[1:, :][C] * Z[:-1, :][C])
    if W == 0: return np.nan
    return float((M.sum() / W) * (num / np.sum(Z[M] ** 2)))

def make_bands(y_raster, dx_km, dy_km):
    sig40 = (40.0 / dx_km, 40.0 / dy_km)
    sig10 = (10.0 / dx_km, 10.0 / dy_km)
    Y_low = gaussian_smooth_nan(y_raster, sig40[1], sig40[0])
    Y_low10 = gaussian_smooth_nan(y_raster, sig10[1], sig10[0])
    return Y_low, Y_low10 - Y_low, y_raster - Y_low10

# ==================== DESIGN MATRICES ====================

def load_precomputed_sh(precompute_dir: Path, region: str, L: int) -> np.ndarray:
    path = precompute_dir / region / f"SH_L{L}.npy"
    if path.exists():
        return np.load(path)
    return None

def load_precomputed_slepian(precompute_dir: Path, region: str, L: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    # Find file with K_max >= K
    region_dir = precompute_dir / region
    for f in sorted(region_dir.glob(f"Slepian_L{L}_Kmax*.npy")):
        X = np.load(f)
        lam = np.load(str(f).replace(".npy", "_lambda.npy"))
        if X.shape[1] >= K:
            return X[:, :K], lam[:K]
    return None, None

def sh_design_matrix_compute(coords_deg: np.ndarray, L: int) -> np.ndarray:
    if not HAVE_SH or L <= 0:
        return np.zeros((len(coords_deg), 0), dtype=np.float32)
    lon = torch.as_tensor(coords_deg[:, 0], dtype=torch.float64)
    lat = torch.as_tensor(coords_deg[:, 1], dtype=torch.float64)
    phi = torch.deg2rad(lon + 180.0)
    theta = torch.clamp(torch.deg2rad(lat + 90.0), 1e-12, math.pi - 1e-12)
    cols = []
    for l in range(L):
        for m in range(-l, l + 1):
            y = SH_analytic(m, l, phi, theta)
            if not torch.is_tensor(y):
                y = torch.full_like(theta, float(y))
            elif y.ndim == 0:
                y = torch.full_like(theta, y.item())
            cols.append(y.to(torch.float64))
    X = torch.nan_to_num(torch.stack(cols, dim=1), nan=0.0)
    return X.to(torch.float32).numpy()

def slepian_design_matrix_compute(coords_deg, cap_center, cap_radius, L_s, K) -> Tuple[np.ndarray, np.ndarray]:
    if not HAVE_PYSH:
        raise RuntimeError("PySHTOOLS required")
    sl = pysh.Slepian.from_cap(theta=cap_radius, lmax=L_s)
    sl.rotate(clon=float(cap_center[0]), clat=float(cap_center[1]))
    lam = sl.eigenvalues[:K].astype(np.float32)
    lon_360 = np.where(coords_deg[:, 0] < 0, coords_deg[:, 0] + 360, coords_deg[:, 0])
    cols = [sl.to_shcoeffs(i).expand(lon=lon_360, lat=coords_deg[:, 1], degrees=True).astype(np.float32) for i in range(K)]
    return np.column_stack(cols) if cols else np.zeros((len(coords_deg), 0), np.float32), lam

def shannon_K(L_s: int, cap_radius_deg: float) -> int:
    A = 2 * math.pi * (1.0 - math.cos(math.radians(cap_radius_deg)))
    return max(1, int(round(((L_s + 1)**2) * (A / (4 * math.pi)))))

# ==================== DATASET & MODEL ====================

class BuildingsDataset(Dataset):
    def __init__(self, df, target_col, emb_col, emb_dim):
        self.coords = df[["lon", "lat"]].to_numpy(np.float32)
        self.emb = np.stack([np.frombuffer(b, dtype=np.float32)[:emb_dim] for b in df[emb_col].values])
        self.y = df[target_col].to_numpy(np.float32)
        self.idx = df.index.to_numpy(np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.idx[i], torch.from_numpy(self.emb[i]), torch.from_numpy(self.coords[i]), torch.tensor(self.y[i])

class LocBottleneck(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out_dim = hidden
    def forward(self, x): return self.net(x)


class LinearProjection(nn.Module):
    """Simple linear projection with out_dim attribute."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim
    def forward(self, x): return self.net(x)


def build_loc_projection(in_dim: int, out_dim: int, arch: str, hidden_dim: int = 256) -> nn.Module:
    """Build location projection using nn module trunks.

    Args:
        in_dim: Input feature dimension (from positional encoding)
        out_dim: Output dimension for location features
        arch: Architecture type ('linear', 'mlp', 'resmlp', 'siren', 'glu')
        hidden_dim: Hidden layer dimension (for non-linear archs)

    Returns:
        Module with out_dim attribute for downstream use.
    """
    if arch == "linear":
        return LinearProjection(in_dim, out_dim)
    elif arch == "mlp":
        return LocBottleneck(in_dim, out_dim)
    elif arch == "resmlp":
        trunk = ResMLPTrunk(in_dim, hidden_dim=hidden_dim, out_dim=out_dim, depth=2, dropout=0.1)
        return trunk
    elif arch == "siren":
        trunk = SirenTrunk(in_dim, hidden_dim=hidden_dim, out_dim=out_dim, depth=2,
                           omega_0=1.0, omega_0_initial=30.0)
        return trunk
    elif arch == "glu":
        trunk = GLUTrunk(in_dim, hidden_dim=hidden_dim, out_dim=out_dim, depth=2, dropout=0.1)
        return trunk
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class EmbOnlyRegressor(nn.Module):
    """Embedding-only model (no location features)."""
    def __init__(self, emb_dim, hidden=128, p_drop=0.1):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, 1))

    def forward(self, idx, emb):
        return self.head(emb).squeeze(-1)


class EmbPlusLocRegressor(nn.Module):
    def __init__(self, loc_raw, loc_proj, emb_dim, hidden=128, p_drop=0.1):
        super().__init__()
        self.register_buffer("loc_raw_cache", loc_raw, persistent=False)
        self.loc_proj = loc_proj
        self.head = nn.Sequential(nn.Linear(emb_dim + loc_proj.out_dim, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, 1))

    def forward(self, idx, emb):
        loc_raw = self.loc_raw_cache[idx.cpu()].to(emb.device)
        return self.head(torch.cat([emb, self.loc_proj(loc_raw)], dim=-1)).squeeze(-1)

def train_one(model, loader, device, epochs, lr, wd, patience, val_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best = (1e9, None)
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for idx, emb, _, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(idx.to(device), emb.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)

        val_loss = tot / max(n, 1)
        if val_loader:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for idx, emb, _, y in val_loader:
                    loss = loss_fn(model(idx.to(device), emb.to(device)), y.to(device))
                    tot += loss.item() * y.size(0); n += y.size(0)
            val_loss = tot / max(n, 1)

        if val_loss < best[0] - 1e-6:
            best = (val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    if best[1]: model.load_state_dict(best[1])
    return model

# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--region", required=True)
    ap.add_argument("--embedding-type", choices=["alphaearth", "galileo"], default="alphaearth")
    ap.add_argument("--target-base", default="log_density")
    ap.add_argument("--scales-km", default="0,1,2,3,4,5,10,20,40")
    ap.add_argument("--use-parquet-suffix", default="_ms")
    ap.add_argument("--L-sh", type=int, default=40)
    ap.add_argument("--L-slepian", type=int, default=96)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--cap-radius-deg", type=float, default=None)
    ap.add_argument("--cap-center", nargs=2, type=float, default=None)
    ap.add_argument("--precompute-dir", type=Path, default=None, help="Dir with precomputed design matrices")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--arch", type=str, default="mlp",
                    choices=["linear", "mlp", "resmlp", "siren", "glu"],
                    help="Neural network architecture for location projection")
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--image-only", action="store_true",
                    help="Only run image embedding baseline (skip SH and Slepian)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    emb_cfg = EMBEDDING_CONFIGS[args.embedding_type]
    emb_col, emb_dim = emb_cfg["column"], emb_cfg["dim"]

    # Load data
    ddir = Path(args.data_dir)
    df = pd.read_parquet(ddir / f"{args.region}_grid{args.use_parquet_suffix}.parquet")
    meta = json.load(open(ddir / f"{args.region}_metadata.json"))
    aux = json.load(open(ddir / f"{args.region}_grid_aux.json"))

    coords_all = df[["lon", "lat"]].to_numpy(np.float32)
    ny, nx = aux["ny"], aux["nx"]
    dx_km, dy_km = aux["dx_km"], aux["dy_km"]

    # Cap parameters
    b = meta["bounds"]
    cap_radius = args.cap_radius_deg or max(b[3] - b[1], b[2] - b[0]) * 0.65
    cap_center = tuple(args.cap_center) if args.cap_center else (meta.get("center") or [df["lon"].mean(), df["lat"].mean()])
    cap_center = (float(cap_center[0]), float(cap_center[1]))
    K = args.K or shannon_K(args.L_slepian, cap_radius)

    if args.image_only:
        print(f"[{args.region}] {args.embedding_type} | arch={args.arch} (IMAGE-ONLY)")
    else:
        print(f"[{args.region}] {args.embedding_type} | arch={args.arch} L_sh={args.L_sh} L_slep={args.L_slepian} K={K}")

    # Load or compute SH (skip if image-only)
    Xsh = None
    if not args.image_only:
        if args.precompute_dir:
            Xsh = load_precomputed_sh(args.precompute_dir, args.region, args.L_sh)
            if Xsh is not None:
                print(f"  Loaded precomputed SH: {Xsh.shape}")
        if Xsh is None:
            print(f"  Computing SH L={args.L_sh}...")
            Xsh = sh_design_matrix_compute(coords_all, args.L_sh)
        Xsh = torch.from_numpy(Xsh)

    # Load or compute Slepian (skip if image-only)
    Xsl, lam = None, None
    if not args.image_only:
        if args.precompute_dir:
            Xsl, lam = load_precomputed_slepian(args.precompute_dir, args.region, args.L_slepian, K)
            if Xsl is not None:
                print(f"  Loaded precomputed Slepian: {Xsl.shape}")
        if Xsl is None:
            print(f"  Computing Slepian L={args.L_slepian} K={K}...")
            Xsl, lam = slepian_design_matrix_compute(coords_all, cap_center, cap_radius, args.L_slepian, K)
        Xsl = torch.from_numpy(Xsl)

    # Targets
    scales = [int(float(s)) for s in args.scales_km.split(",")]
    target_cols = [f"{args.target_base}_s{km}km" for km in scales]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for km, tcol in zip(scales, target_cols):
        df_tr, df_va, df_te = df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]
        ds_tr = BuildingsDataset(df_tr, tcol, emb_col, emb_dim)
        ds_va = BuildingsDataset(df_va, tcol, emb_col, emb_dim)
        ds_te = BuildingsDataset(df_te, tcol, emb_col, emb_dim)

        dl = lambda ds, shuf: DataLoader(ds, batch_size=args.batch_size, shuffle=shuf, num_workers=args.num_workers, pin_memory=True)
        Ltr, Lva, Lte = dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False)

        def run_model(name, loc_raw):
            if loc_raw is None:
                # Use embedding-only model (no location features)
                model = EmbOnlyRegressor(emb_dim).to(device)
            else:
                loc_proj = build_loc_projection(loc_raw.shape[1], 256, args.arch)
                model = EmbPlusLocRegressor(loc_raw, loc_proj, emb_dim).to(device)
            model = train_one(model, Ltr, device, args.epochs, args.lr, args.weight_decay, args.patience, Lva)

            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for idx, emb, _, y in Lte:
                    y_pred.append(model(idx.to(device), emb.to(device)).cpu().numpy())
                    y_true.append(y.numpy())
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

            r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-12)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

            Yt = to_raster(y_true, df_te["grid_lat_idx"].to_numpy() - aux["gy_min"],
                           df_te["grid_lon_idx"].to_numpy() - aux["gx_min"], ny, nx)
            Yp = to_raster(y_pred, df_te["grid_lat_idx"].to_numpy() - aux["gy_min"],
                           df_te["grid_lon_idx"].to_numpy() - aux["gx_min"], ny, nx)
            mI = morans_I_rook(Yt - Yp)
            low_t, mid_t, high_t = make_bands(Yt, dx_km, dy_km)
            low_p, mid_p, high_p = make_bands(Yp, dx_km, dy_km)
            mask = np.isfinite(Yt) & np.isfinite(Yp)

            return dict(R2=float(r2), RMSE=float(rmse), MoranI=float(mI) if mI == mI else np.nan,
                        R2_low=r2_masked(low_t, low_p, mask & np.isfinite(low_t) & np.isfinite(low_p)),
                        R2_mid=r2_masked(mid_t, mid_p, mask & np.isfinite(mid_t) & np.isfinite(mid_p)),
                        R2_high=r2_masked(high_t, high_p, mask & np.isfinite(high_t) & np.isfinite(high_p)))

        emb_name = args.embedding_type.upper()
        res_emb = run_model(f"{emb_name}_only", None)

        if args.image_only:
            # Image-only mode: only run embedding model
            res = {
                "region": args.region, "arch": args.arch, "sigma_km": km, "embedding_type": args.embedding_type,
                f"R2_{emb_name}": res_emb["R2"], f"RMSE_{emb_name}": res_emb["RMSE"],
            }
            print(f"  σ={km:2d}km: R2_emb={res_emb['R2']:.4f}")
        else:
            # Full comparison mode: run all three models
            res_sh = run_model(f"{emb_name}+SH", Xsh)
            res_sl = run_model(f"{emb_name}+Slepian", Xsl)

            res = {
                "region": args.region, "arch": args.arch, "sigma_km": km, "embedding_type": args.embedding_type,
                f"R2_{emb_name}": res_emb["R2"], "R2_SH": res_sh["R2"], "R2_Slepian": res_sl["R2"],
                "ΔR2_SH": res_sh["R2"] - res_emb["R2"], "ΔR2_Slepian": res_sl["R2"] - res_emb["R2"],
                f"RMSE_{emb_name}": res_emb["RMSE"], "RMSE_SH": res_sh["RMSE"], "RMSE_Slepian": res_sl["RMSE"],
                "L_sh": args.L_sh, "L_slepian": args.L_slepian, "K": K, "cap_radius_deg": cap_radius,
            }
            print(f"  σ={km:2d}km: R2_emb={res_emb['R2']:.4f} R2_SH={res_sh['R2']:.4f} R2_Slep={res_sl['R2']:.4f}")
        results.append(res)

    if args.csv_out:
        pd.DataFrame(results).to_csv(args.csv_out, index=False)
        print(f"Saved: {args.csv_out}")


if __name__ == "__main__":
    main()
