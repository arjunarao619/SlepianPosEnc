#!/usr/bin/env python3
# train_slepian_vs_sh_multiscale.py
# Multi-scale test of AE-only vs AE+SH vs AE+Slepian on OpenBuildings+AlphaEarth

import os
import sys
import re
import math
import json
import argparse
import time
from pathlib import Path
from typing import Tuple, Optional, Dict

# Add parent directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.special import lpmv
from scipy.ndimage import gaussian_filter
from spherical_harmonics_ylm import SH as SH_analytic
# Optional: PySHTOOLS for Slepians
try:
    import pyshtools as pysh
    HAVE_PYSH = True
except Exception:
    HAVE_PYSH = False

# tame BLAS thread storms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

# ---------------- utils: grid & smoothing ---------------- #

def to_raster(values, gy0, gx0, ny, nx, fill=np.nan, dtype=np.float32):
    A = np.full((ny, nx), fill, dtype=dtype)
    A[gy0, gx0] = values
    return A

def from_raster(A, gy0, gx0):
    return A[gy0, gx0]

def gaussian_smooth_nan(A, sigma_y_pix, sigma_x_pix):
    M = np.isfinite(A).astype(np.float32)
    V = np.where(M > 0, A, 0.0).astype(np.float32)
    num = gaussian_filter(V, sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    den = gaussian_filter(M, sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    return np.where(den > 0, num / np.maximum(den, 1e-8), np.nan)

def pixel_size_km(meta, mean_lat_deg, default_deg=0.01):
    deg = meta.get("grid_size_deg", default_deg)
    dy_km = 111.0 * deg
    dx_km = 111.0 * math.cos(math.radians(mean_lat_deg)) * deg
    return dx_km, dy_km

# ---------------- metrics ---------------- #

def r2_masked(y, yhat, m):
    y = y[m]; yhat = yhat[m]
    if y.size < 2: return np.nan
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot

def rmse_masked(y, yhat, m):
    y = y[m]; yhat = yhat[m]
    if y.size < 1: return np.nan
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def morans_I_rook(residual_raster):
    """Simple Moran's I using 4-neighbor rook on valid mask."""
    A = residual_raster
    M = np.isfinite(A)
    if M.sum() < 4: return np.nan
    Z = np.where(M, A - np.nanmean(A), 0.0)
    # neighbor pairs
    W = 0.0; num = 0.0
    ny, nx = A.shape
    # horizontal
    R = M[:, 1:] & M[:, :-1]
    W += R.sum()
    num += np.sum(Z[:, 1:][R] * Z[:, :-1][R])
    # vertical
    C = M[1:, :] & M[:-1, :]
    W += C.sum()
    num += np.sum(Z[1:, :][C] * Z[:-1, :][C])
    if W == 0: return np.nan
    z2 = np.sum(Z[M] ** 2)
    n = M.sum()
    return float((n / W) * (num / z2))

# ---------------- SH & Slepian features ---------------- #

def sh_design_matrix(coords_deg: np.ndarray, L: int, use_double: bool = True) -> np.ndarray:
    """
    Analytic real SH identical to the SphericalHarmonics(nn.Module) class:
      - l = 0..L-1  → L^2 columns
      - phi = deg2rad(lon + 180), theta = deg2rad(lat + 90)
      - returns float32 NumPy [N, L^2]
    """
    if L <= 0 or coords_deg.shape[0] == 0:
        return np.zeros((len(coords_deg), 0), dtype=np.float32)

    dtype = torch.float64 if use_double else torch.float32
    lon = torch.as_tensor(coords_deg[:, 0], dtype=dtype)
    lat = torch.as_tensor(coords_deg[:, 1], dtype=dtype)

    phi   = torch.deg2rad(lon + 180.0)
    theta = torch.deg2rad(lat + 90.0)

    # tiny clamp to avoid rare sqrt(neg_tiny) in generated code near poles
    eps = 1e-12
    theta = torch.clamp(theta, eps, math.pi - eps)

    cols = []
    for l in range(L):                 # NOTE: up to L-1 → L^2 dims
        for m in range(-l, l + 1):
            y = SH_analytic(m, l, phi, theta)  # may return scalar for some (l,m)
            if not torch.is_tensor(y):
                y = torch.full_like(theta, float(y))
            elif y.ndim == 0:
                y = torch.full_like(theta, y.item())
            else:
                y = y.to(dtype)
            if y.shape != theta.shape:
                y = (theta * 0) + y
            cols.append(y)

    X = torch.stack(cols, dim=1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.detach().cpu().to(torch.float32).numpy()

def slepian_design_matrix(coords_deg: np.ndarray,
                          cap_center_deg: Tuple[float,float],
                          cap_radius_deg: float,
                          L_s: int,
                          K_keep: Optional[int] = None,
                          lam_thresh: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Top-K Slepian cap functions evaluated at coords."""
    assert HAVE_PYSH, "PySHTOOLS required for Slepians."
    sl = pysh.Slepian.from_cap(theta=cap_radius_deg, lmax=L_s)
    # rotate to center (clon, clat) in degrees
    clon = float(cap_center_deg[0]); clat = float(cap_center_deg[1])
    sl.rotate(clon=clon, clat=clat)

    lam = sl.eigenvalues
    idx = np.arange(len(lam))
    if lam_thresh is not None:
        idx = idx[lam >= lam_thresh]
    if K_keep is not None:
        idx = idx[:K_keep]
    lam_kept = lam[idx].astype(np.float32)

    lon = coords_deg[:, 0]
    lat = coords_deg[:, 1]
    lon_360 = np.where(lon < 0.0, lon + 360.0, lon)
    cols = []
    for i in idx:
        vals = sl.to_shcoeffs(i).expand(lon=lon_360, lat=lat, degrees=True)
        cols.append(vals.astype(np.float32))
    X = np.column_stack(cols) if cols else np.zeros((len(coords_deg), 0), np.float32)
    return X, lam_kept

def shannon_K(L_s: int, cap_radius_deg: float) -> int:
    """Shannon number ≈ expected # well-concentrated modes in cap."""
    A = 2 * math.pi * (1.0 - math.cos(math.radians(cap_radius_deg)))
    K = ((L_s + 1)**2) * (A / (4 * math.pi))
    return max(1, int(round(K)))

# ---------------- dataset ---------------- #

class BuildingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.coords = df[["lon","lat"]].to_numpy(np.float32)
        # embeddings are stored as bytes -> float32 (63,)
        self.emb = np.stack([np.frombuffer(b, dtype=np.float32) for b in df["embedding"].values])
        self.y = df[target_col].to_numpy(np.float32)
        self.idx = df.index.to_numpy(np.int64)  # global index within this slice

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (self.idx[i],
                torch.from_numpy(self.emb[i]),
                torch.from_numpy(self.coords[i]),
                torch.tensor(self.y[i], dtype=torch.float32))

# ---------------- model ---------------- #

class LocBottleneck(nn.Module):
    """Capacity-matched 256-dim bottleneck for ANY raw location features."""
    def __init__(self, in_dim: int, hidden: int = 256, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden
    def forward(self, x): return self.net(x)

class AEPlusLocRegressor(nn.Module):
    def __init__(self, loc_raw: torch.Tensor, loc_proj: LocBottleneck, ae_dim=63, hidden=128, p_drop=0.1):
        """
        loc_raw: [N, D_raw] cached tensor aligned with the full (train+val+test) df order.
        """
        super().__init__()
        self.register_buffer("loc_raw_cache", loc_raw, persistent=False)
        self.loc_proj = loc_proj
        self.head = nn.Sequential(
            nn.Linear(ae_dim + loc_proj.out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, idx: torch.Tensor, emb: torch.Tensor):
        # idx are positions into loc_raw_cache (already aligned with concat df order)
        loc_raw = self.loc_raw_cache[idx.cpu()].to(emb.device, non_blocking=True)
        z_loc = self.loc_proj(loc_raw)
        z = torch.cat([emb, z_loc], dim=-1)
        return self.head(z).squeeze(-1)

# ---------------- bands for analysis ---------------- #

def make_bands(y_raster, dx_km, dy_km):
    """Return (low>=40km, mid 10-40, high <=10) band rasters, NaN-safe."""
    # choose sigma so that ≈ wavelength/√(8 ln 2) if treating sigma ~ FWHM/2.355; good enough:
    sig40 = (40.0 / dx_km, 40.0 / dy_km)
    sig10 = (10.0 / dx_km, 10.0 / dy_km)
    Y = y_raster
    Y_low  = gaussian_smooth_nan(Y, sig40[1], sig40[0])
    Y_low10 = gaussian_smooth_nan(Y, sig10[1], sig10[0])
    Y_high = Y - Y_low10
    Y_mid  = Y_low10 - Y_low
    return Y_low, Y_mid, Y_high

# ---------------- training loop ---------------- #

def train_one(model, loader, device, epochs=100, lr=1e-3, wd=1e-4, patience=15, val_loader=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best = (1e9, None)
    bad = 0
    for ep in range(1, epochs+1):
        model.train()
        tot, n = 0.0, 0
        for idx, emb, coords, y in loader:
            idx = idx.to(device); y = y.to(device)
            emb = emb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yhat = model(idx, emb)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = tot / max(n,1)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for idx, emb, coords, y in val_loader:
                    idx = idx.to(device); y = y.to(device)
                    emb = emb.to(device, non_blocking=True)
                    yhat = model(idx, emb)
                    loss = loss_fn(yhat, y)
                    tot += loss.item() * y.size(0); n += y.size(0)
            val_loss = tot / max(n,1)

        if val_loss < best[0] - 1e-6:
            best = (val_loss, {k: v.cpu().clone() for k,v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0:
            print(f"  epoch {ep:03d}  train {train_loss:.4f}  val {val_loss:.4f}")
    # restore best
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model

# ---------------- main ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--region", required=True)
    ap.add_argument("--target-base", default="log_density", help="base column name (e.g., log_density)")
    ap.add_argument("--scales-km", default="0,1,2,3,4,5,10,20,40")
    ap.add_argument("--use-parquet-suffix", default="_ms", help="parquet suffix written by prep script")
    # location encoders (capacity matched @ 256 after bottleneck)
    ap.add_argument("--L-sh", type=int, default=64)  # SH degree (inclusive)
    ap.add_argument("--L-slepian", type=int, default=96)
    ap.add_argument("--cap-radius-deg", type=float, default=None,
                    help="If None, derive from region bounds (loose).")
    ap.add_argument("--K", type=int, default=None, help="Slepian K (top modes). If None, use Shannon K.")
    ap.add_argument("--lambda-thresh", type=float, default=None, help="Optional λ threshold before K.")
    ap.add_argument(
        "--cap-center", nargs=2, type=float, metavar=("LON","LAT"), default=None,
        help="Slepian cap center in degrees as: --cap-center <lon> <lat>. "
            "If omitted, uses metadata['center'] or mean(lon,lat)."
    )    
    # training
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    # output
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--experiment-id", type=str, default=None,
               help="Experiment identifier for output filename")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ----- load data & meta ----- #
    ddir = Path(args.data_dir)
    pq = ddir / f"{args.region}_grid{args.use_parquet_suffix}.parquet"
    meta = json.load(open(ddir / f"{args.region}_metadata.json"))
    aux  = json.load(open(ddir / f"{args.region}_grid_aux.json"))
    df = pd.read_parquet(pq)
    assert "split" in df.columns and "spatial_bin" in df.columns, "Need split & spatial_bin columns."

    mean_lat = float(df["lat"].mean())
    dx_km, dy_km = aux["dx_km"], aux["dy_km"]
    ny, nx = aux["ny"], aux["nx"]
    gx0 = (df["grid_lon_idx"].to_numpy() - aux["gx_min"]).astype(int)
    gy0 = (df["grid_lat_idx"].to_numpy() - aux["gy_min"]).astype(int)

    # ----- choose cap center & radius ----- #
    if args.cap_radius_deg is None:
        # simple heuristic: half of max(bbox height/width) in deg → generous cap
        b = meta["bounds"]  # [min_lon, min_lat, max_lon, max_lat]
        h = (b[3] - b[1]); w = (b[2] - b[0])
        cap_radius_deg = max(h, w) * 0.65
    else:
        cap_radius_deg = float(args.cap_radius_deg)

    # cap center precedence: CLI > metadata center > mean(lon,lat)
    if args.cap_center is not None:
        cap_center = (float(args.cap_center[0]), float(args.cap_center[1]))
    elif "center" in meta and isinstance(meta["center"], (list, tuple)) and len(meta["center"]) == 2:
        cap_center = (float(meta["center"][0]), float(meta["center"][1]))
    else:
        cap_center = (float(df["lon"].mean()), float(df["lat"].mean()))

    print(f"Using Slepian cap center={cap_center}, radius={cap_radius_deg:.2f}°")

    # ----- multi-scale targets ----- #
    scales = [int(float(s)) for s in args.scales_km.split(",")]
    target_base = args.target_base 
    target_cols = [f"{target_base}_s{km}km" for km in scales]
    for c in target_cols:
        if c not in df.columns:
            raise ValueError(f"Missing target column {c}. Run prep_multiscale_targets.py first.")

    # ----- build location raw features for whole df (SH, Slepian) ----- #
    coords_all = df[["lon","lat"]].to_numpy(np.float32)

    print(f"Building SH design matrix L={args.L_sh} (raw dims={(args.L_sh+1)**2}) ...")
    Xsh = sh_design_matrix(coords_all, args.L_sh)  # [N, Dsh]
    Xsh = torch.from_numpy(Xsh)  # cached CPU tensor

    print(f"Building Slepian design matrix Ls={args.L_slepian}, cap={cap_center}, r={cap_radius_deg}° ...")
    if not HAVE_PYSH:
        raise RuntimeError("PySHTOOLS not available.")
    K = args.K if args.K is not None else shannon_K(args.L_slepian, cap_radius_deg)
    Xsl_np, lam = slepian_design_matrix(coords_all, cap_center, cap_radius_deg,
                                        L_s=args.L_slepian, K_keep=K, lam_thresh=args.lambda_thresh)
    print(f"Slepian kept K={Xsl_np.shape[1]} (Shannon~{K}); λ_min={lam.min():.3f}, λ_med={np.median(lam):.3f}")
    Xsl = torch.from_numpy(Xsl_np)

    # ----- datasets ----- #
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for km, tcol in zip(scales, target_cols):
        print(f"\n=== Scale σ={km} km  (target: {tcol}) ===")
        df_tr = df[df["split"]=="train"]
        df_va = df[df["split"]=="val"]
        df_te = df[df["split"]=="test"]

        ds_tr = BuildingsDataset(df_tr, tcol)
        ds_va = BuildingsDataset(df_va, tcol)
        ds_te = BuildingsDataset(df_te, tcol)

        # DataLoaders (keep idx to fetch cached loc features)
        def dl(dataset, bs, shuffle):
            return DataLoader(dataset, batch_size=bs, shuffle=shuffle,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0))

        Ltr = dl(ds_tr, args.batch_size, True)
        Lva = dl(ds_va, args.batch_size, False)
        Lte = dl(ds_te, args.batch_size, False)

        # ---------- three models: AE-only, AE+SH, AE+Slepian ---------- #
        def run_model(name, loc_raw_tensor):
            if loc_raw_tensor is None:
                # dummy zero loc feat -> project 0 to 256 (still trainable biases)
                loc_raw_tensor = torch.zeros((len(df), 1), dtype=torch.float32)
            loc_proj = LocBottleneck(in_dim=loc_raw_tensor.shape[1], hidden=256, p_drop=0.0)
            model = AEPlusLocRegressor(loc_raw=loc_raw_tensor, loc_proj=loc_proj,
                                       ae_dim=63, hidden=128, p_drop=0.1).to(device)
            model = train_one(model, Ltr, device, epochs=args.epochs, lr=args.lr,
                              wd=args.weight_decay, patience=args.patience, val_loader=Lva)

            # collect predictions on test
            y_true, y_pred = [], []
            model.eval()
            with torch.no_grad():
                for idx, emb, coords, y in Lte:
                    p = model(idx.to(device), emb.to(device, non_blocking=True)).cpu().numpy()
                    y_pred.append(p); y_true.append(y.numpy())
            y_pred = np.concatenate(y_pred); y_true = np.concatenate(y_true)

            # metrics
            r2 = float(1.0 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12))
            rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))

            # per-tile R2 (spatial_bin)
            te_bins = df_te["spatial_bin"].to_numpy()
            bins = np.unique(te_bins)
            r2_bins = []
            for b in bins:
                m = te_bins == b
                if m.sum() > 3:
                    r2_bins.append(r2_masked(y_true, y_pred, m))
            r2_bins = np.array(r2_bins, float)
            r2_bins_mean = float(np.nanmean(r2_bins))

            # residual Moran's I on raster
            # place residuals on full test raster
            Yt = to_raster(y_true, df_te["grid_lat_idx"].to_numpy()-aux["gy_min"],
                           df_te["grid_lon_idx"].to_numpy()-aux["gx_min"], ny, nx)
            Yp = to_raster(y_pred, df_te["grid_lat_idx"].to_numpy()-aux["gy_min"],
                           df_te["grid_lon_idx"].to_numpy()-aux["gx_min"], ny, nx)
            res = Yt - Yp
            mI = morans_I_rook(res)

            # band-wise R2
            low_t, mid_t, high_t = make_bands(Yt, dx_km, dy_km)
            low_p, mid_p, high_p = make_bands(Yp, dx_km, dy_km)
            mask = np.isfinite(Yt) & np.isfinite(Yp)
            r2_low  = r2_masked(low_t,  low_p,  mask & np.isfinite(low_t)  & np.isfinite(low_p))
            r2_mid  = r2_masked(mid_t,  mid_p,  mask & np.isfinite(mid_t)  & np.isfinite(mid_p))
            r2_high = r2_masked(high_t, high_p, mask & np.isfinite(high_t) & np.isfinite(high_p))

            return dict(model=name, sigma_km=km, R2=r2, RMSE=rmse,
                        R2_bins_mean=r2_bins_mean, MoranI=residual_safe(mI),
                        R2_low=r2_low, R2_mid=r2_mid, R2_high=r2_high)

        def residual_safe(x):  # guard NaNs in json/csv
            return float(x) if (x==x) else np.nan

        res_ae  = run_model("AE_only",            None)
        res_sh  = run_model("AE_plus_SH",         Xsh)
        res_sl  = run_model("AE_plus_Slepian",    Xsl)

        # improvements
        res = {
            "region": args.region, "sigma_km": km,
            "R2_AE": res_ae["R2"], "R2_SH": res_sh["R2"], "R2_Slepian": res_sl["R2"],
            "ΔR2_SH": res_sh["R2"] - res_ae["R2"],
            "ΔR2_Slepian": res_sl["R2"] - res_ae["R2"],
            "Advantage_Slepian_minus_SH": (res_sl["R2"] - res_ae["R2"]) - (res_sh["R2"] - res_ae["R2"]),
            "RMSE_AE": res_ae["RMSE"], "RMSE_SH": res_sh["RMSE"], "RMSE_Slepian": res_sl["RMSE"],
            "R2bins_AE": res_ae["R2_bins_mean"], "R2bins_SH": res_sh["R2_bins_mean"], "R2bins_Slepian": res_sl["R2_bins_mean"],
            "MoranI_AE": res_ae["MoranI"], "MoranI_SH": res_sh["MoranI"], "MoranI_Slepian": res_sl["MoranI"],
            "R2_low_AE": res_ae["R2_low"], "R2_mid_AE": res_ae["R2_mid"], "R2_high_AE": res_ae["R2_high"],
            "R2_low_SH": res_sh["R2_low"], "R2_mid_SH": res_sh["R2_mid"], "R2_high_SH": res_sh["R2_high"],
            "R2_low_Slepian": res_sl["R2_low"], "R2_mid_Slepian": res_sl["R2_mid"], "R2_high_Slepian": res_sl["R2_high"],
            "L_sh": args.L_sh, "L_slepian": args.L_slepian, "cap_radius_deg": cap_radius_deg,
            "K_kept": int(Xsl.shape[1])
        }
        print({k: v for k,v in res.items() if k.startswith("R2") or k.startswith("Δ") or k=="sigma_km"})
        results.append(res)

    if args.csv_out:
        csv_path = Path(args.csv_out)
        
        # If it's a directory or ends without .csv, auto-generate filename
        if csv_path.is_dir() or (csv_path.suffix != '.csv'):
            csv_path.mkdir(parents=True, exist_ok=True)
            K_actual = int(Xsl.shape[1])
            filename = f"{args.region}_SH{args.L_sh}_Slep{args.L_slepian}_K{K_actual}_seed{args.seed}.csv"
            csv_path = csv_path / filename
        
        out = pd.DataFrame(results)
        out.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()
