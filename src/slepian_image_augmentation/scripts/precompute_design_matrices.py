#!/usr/bin/env python3
"""
Precompute SH and Slepian design matrices for all regions.
This saves significant time when running many experiments with different K values.

Usage:
    python scripts/precompute_design_matrices.py \
        --data-dir $DATA_DIR \
        --output-dir $DATA_DIR/design_matrices
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

import numpy as np
import pandas as pd
import torch

# Add parent to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from spherical_harmonics_ylm import SH as SH_analytic

try:
    import pyshtools as pysh
    HAVE_PYSH = True
except ImportError:
    HAVE_PYSH = False

# Suppress BLAS threading
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


def sh_design_matrix(coords_deg: np.ndarray, L: int) -> np.ndarray:
    """Compute SH design matrix."""
    if L <= 0:
        return np.zeros((len(coords_deg), 0), dtype=np.float32)

    lon = torch.as_tensor(coords_deg[:, 0], dtype=torch.float64)
    lat = torch.as_tensor(coords_deg[:, 1], dtype=torch.float64)
    phi = torch.deg2rad(lon + 180.0)
    theta = torch.deg2rad(lat + 90.0)
    theta = torch.clamp(theta, 1e-12, math.pi - 1e-12)

    cols = []
    for l in range(L):
        for m in range(-l, l + 1):
            y = SH_analytic(m, l, phi, theta)
            if not torch.is_tensor(y):
                y = torch.full_like(theta, float(y))
            elif y.ndim == 0:
                y = torch.full_like(theta, y.item())
            cols.append(y.to(torch.float64))

    X = torch.stack(cols, dim=1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.to(torch.float32).numpy()


def slepian_design_matrix(coords_deg: np.ndarray, cap_center: tuple,
                          cap_radius: float, L_s: int, K_max: int) -> tuple:
    """Compute Slepian design matrix with top K_max modes."""
    assert HAVE_PYSH, "PySHTOOLS required"

    sl = pysh.Slepian.from_cap(theta=cap_radius, lmax=L_s)
    sl.rotate(clon=float(cap_center[0]), clat=float(cap_center[1]))

    lam = sl.eigenvalues
    K_actual = min(K_max, len(lam))

    lon = coords_deg[:, 0]
    lat = coords_deg[:, 1]
    lon_360 = np.where(lon < 0.0, lon + 360.0, lon)

    cols = []
    for i in range(K_actual):
        vals = sl.to_shcoeffs(i).expand(lon=lon_360, lat=lat, degrees=True)
        cols.append(vals.astype(np.float32))

    X = np.column_stack(cols) if cols else np.zeros((len(coords_deg), 0), np.float32)
    return X, lam[:K_actual].astype(np.float32)


def compute_for_region(args):
    """Compute all design matrices for one region."""
    region, data_dir, output_dir, L_sh_values, L_slep_values, K_max = args

    print(f"[{region}] Starting...")

    # Load data
    pq_path = data_dir / f"{region}_grid.parquet"
    meta_path = data_dir / f"{region}_metadata.json"

    if not pq_path.exists():
        print(f"[{region}] SKIP - parquet not found")
        return region, False

    df = pd.read_parquet(pq_path)
    with open(meta_path) as f:
        meta = json.load(f)

    coords = df[["lon", "lat"]].to_numpy(np.float32)
    n_points = len(coords)

    # Cap parameters
    if "center" in meta:
        cap_center = (float(meta["center"][0]), float(meta["center"][1]))
    else:
        cap_center = (float(df["lon"].mean()), float(df["lat"].mean()))

    bounds = meta["bounds"]
    h = bounds[3] - bounds[1]
    w = bounds[2] - bounds[0]
    cap_radius = max(h, w) * 0.65

    region_dir = output_dir / region
    region_dir.mkdir(parents=True, exist_ok=True)

    # Save coordinates for verification
    np.save(region_dir / "coords.npy", coords)

    # Compute SH matrices
    for L in L_sh_values:
        out_path = region_dir / f"SH_L{L}.npy"
        if out_path.exists():
            print(f"[{region}] SH L={L} exists, skipping")
            continue

        print(f"[{region}] Computing SH L={L} ({(L+1)**2} dims)...")
        X = sh_design_matrix(coords, L)
        np.save(out_path, X)
        print(f"[{region}] SH L={L} done: {X.shape}")

    # Compute Slepian matrices
    for L_s in L_slep_values:
        out_path = region_dir / f"Slepian_L{L_s}_Kmax{K_max}.npy"
        lam_path = region_dir / f"Slepian_L{L_s}_Kmax{K_max}_lambda.npy"

        if out_path.exists():
            print(f"[{region}] Slepian L={L_s} exists, skipping")
            continue

        print(f"[{region}] Computing Slepian L={L_s}, K_max={K_max}...")
        X, lam = slepian_design_matrix(coords, cap_center, cap_radius, L_s, K_max)
        np.save(out_path, X)
        np.save(lam_path, lam)
        print(f"[{region}] Slepian L={L_s} done: {X.shape}, λ_min={lam.min():.4f}")

    # Save metadata
    meta_out = {
        "region": region,
        "n_points": n_points,
        "cap_center": cap_center,
        "cap_radius_deg": cap_radius,
        "L_sh_values": L_sh_values,
        "L_slep_values": L_slep_values,
        "K_max": K_max,
    }
    with open(region_dir / "precompute_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[{region}] Complete!")
    return region, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--L-sh", nargs="+", type=int, default=[10, 40])
    parser.add_argument("--L-slepian", nargs="+", type=int, default=[40, 64, 96])
    parser.add_argument("--K-max", type=int, default=128,
                        help="Max K to precompute (can select subset at train time)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find regions
    if args.regions:
        regions = args.regions
    else:
        pq_files = list(args.data_dir.glob("*_grid.parquet"))
        regions = [f.stem.replace("_grid", "") for f in pq_files
                   if not f.stem.endswith("_galileo") and not f.stem.endswith("_ms")]

    print("=" * 60)
    print("PRECOMPUTE DESIGN MATRICES")
    print("=" * 60)
    print(f"Regions: {regions}")
    print(f"L_SH values: {args.L_sh}")
    print(f"L_Slepian values: {args.L_slepian}")
    print(f"K_max: {args.K_max}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Prepare tasks
    tasks = [
        (r, args.data_dir, args.output_dir, args.L_sh, args.L_slepian, args.K_max)
        for r in regions
    ]

    # Run in parallel
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(compute_for_region, t): t[0] for t in tasks}
            for future in as_completed(futures):
                region = futures[future]
                try:
                    _, success = future.result()
                except Exception as e:
                    print(f"[{region}] ERROR: {e}")
    else:
        for task in tasks:
            compute_for_region(task)

    print("\n" + "=" * 60)
    print("PRECOMPUTE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
