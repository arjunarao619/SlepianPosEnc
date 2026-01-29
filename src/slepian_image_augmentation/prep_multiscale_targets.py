#!/usr/bin/env python3
# prep_multiscale_targets.py
# Create multi-scale target columns for OpenBuildings+AlphaEarth grids.

import os, math, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# tame BLAS storms
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def to_raster(values, gy0, gx0, ny, nx, fill=np.nan, dtype=np.float32):
    A = np.full((ny, nx), fill, dtype=dtype)
    A[gy0, gx0] = values
    return A

def from_raster(A, gy0, gx0):
    return A[gy0, gx0]

def pixel_size_km(meta, mean_lat_deg, default_deg=0.01):
    deg = meta.get("grid_size_deg", default_deg)
    dy_km = 111.0 * deg
    dx_km = 111.0 * math.cos(math.radians(mean_lat_deg)) * deg
    return dx_km, dy_km

def gaussian_smooth_nan(A, sigma_y_pix, sigma_x_pix):
    """NaN-safe Gaussian smoothing via normalized convolution."""
    M = np.isfinite(A).astype(np.float32)
    A0 = np.where(M > 0, A, 0.0).astype(np.float32)
    num = gaussian_filter(A0, sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    den = gaussian_filter(M,  sigma=(sigma_y_pix, sigma_x_pix), mode="nearest")
    out = np.where(den > 0, num / np.maximum(den, 1e-8), np.nan)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--region", required=True)
    ap.add_argument("--target", default="log_density")  # column to multi-scale
    ap.add_argument("--sigmas-km", default="0,1,2,3,4,5,10,20,40",
                    help="comma list of Gaussian σ in km (0 keeps original)")
    ap.add_argument("--out-suffix", default="_ms")      # suffix for new parquet
    args = ap.parse_args()

    ddir = Path(args.data_dir)
    pq = ddir / f"{args.region}_grid.parquet"
    meta_path = ddir / f"{args.region}_metadata.json"
    df = pd.read_parquet(pq)
    with open(meta_path) as f:
        meta = json.load(f)

    # grid geometry
    gx = df["grid_lon_idx"].to_numpy()
    gy = df["grid_lat_idx"].to_numpy()
    nx = int(gx.max() - gx.min() + 1)
    ny = int(gy.max() - gy.min() + 1)
    gx0 = (gx - gx.min()).astype(int)
    gy0 = (gy - gy.min()).astype(int)
    mean_lat = float(df["lat"].mean())
    dx_km, dy_km = pixel_size_km(meta, mean_lat)

    # base field
    y_vec = df[args.target].to_numpy().astype(np.float32)
    Y = to_raster(y_vec, gy0, gx0, ny, nx)

    sigmas = [float(s) for s in args.sigmas_km.split(",")]
    print(f"{args.region}: grid {ny}×{nx}, pixel ~({dx_km:.2f},{dy_km:.2f}) km; scales={sigmas} km")

    for s_km in sigmas:
        if s_km <= 0:
            col = f"{args.target}_s{int(s_km)}km"
            df[col] = y_vec
            continue
        sx = max(1e-6, s_km / dx_km)
        sy = max(1e-6, s_km / dy_km)
        Ys = gaussian_smooth_nan(Y, sy, sx)
        df[f"{args.target}_s{int(s_km)}km"] = from_raster(Ys, gy0, gx0).astype(np.float32)

    out_pq = ddir / f"{args.region}_grid{args.out_suffix}.parquet"
    df.to_parquet(out_pq, index=False)
    # small meta drop with grid + pixel info for reuse
    aux = {"ny": ny, "nx": nx, "dx_km": dx_km, "dy_km": dy_km,
           "gx_min": int(gx.min()), "gy_min": int(gy.min())}
    with open(ddir / f"{args.region}_grid_aux.json", "w") as f:
        json.dump(aux, f, indent=2)
    print(f"Wrote {out_pq} and {args.region}_grid_aux.json")

if __name__ == "__main__":
    main()
