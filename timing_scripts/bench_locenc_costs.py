#!/usr/bin/env python3
# bench_locenc_costs_fixed_regions.py
# Timing / throughput comparison of SH (analytic) vs Slepians (PySHTOOLS)
# on random lon/lat sampled from fixed world regions.
#
# Usage for unified table (same L values for SH and Slepians):
#   python bench_locenc_costs.py --region brazil --unified-table \
#       --L-list 10,20,30,40,50 --cap-radius-list 6,10,20

import os, math, time, argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Torch only for the Russwurm analytic SH (and fallback if allowed)
import torch

# --- performance hygiene ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_float32_matmul_precision("high")

# -------- Regions (approximate bbox) --------
# (min_lon, min_lat, max_lon, max_lat), and a reasonable cap center
REGIONS = {
    "brazil":      {"bbox": (-74.0, -34.0, -34.0,   6.0), "center": (-54.0, -14.0)},
    "sri_lanka":   {"bbox": ( 79.5,   5.5,  82.0,  10.0), "center": ( 80.8,   7.6)},
    "mozambique":  {"bbox": ( 30.0, -27.0,  41.0, -10.0), "center": ( 35.5, -18.5)},
    "china":       {"bbox": ( 73.5,  18.0, 135.0,  53.5), "center": (104.0,  35.0)},
    "antarctica":  {"bbox": (-180., -90.0, 180.0, -60.0), "center": (  0.0, -75.0)},
    "india":       {"bbox": ( 68.0,   6.0,  97.5,  36.0), "center": ( 82.5,  22.0)},
    "greenland":   {"bbox": ( -73.,  59.0, -12.0,  83.0), "center": ( -40.,  72.0)},
}

# -------- Analytic SH (Russwurm) --------
ANALYTIC_OK = False
try:
    from spherical_harmonics_ylm import SH as SH_analytic  # your generated file
    ANALYTIC_OK = True
except Exception:
    ANALYTIC_OK = False

# -------- PySHTOOLS availability --------
HAVE_PYSH = False
HAVE_PYSH_LEG = False
try:
    import pyshtools as pysh
    HAVE_PYSH = True
    # try to import a Legendre backend
    try:
        # Orthonormalized associated Legendre functions up to lmax at x
        from pyshtools.legendre import PlmON as pysh_Plms
        HAVE_PYSH_LEG = True
    except Exception:
        HAVE_PYSH_LEG = False
except Exception:
    HAVE_PYSH = False
    HAVE_PYSH_LEG = False

# -------- Sampling random lon/lat inside bbox --------
def sample_random_coords(bbox: Tuple[float,float,float,float], n: int, seed: int=123) -> np.ndarray:
    mnlon, mnlat, mxlon, mxlat = bbox
    rng = np.random.default_rng(seed)
    lons = rng.uniform(mnlon, mxlon, n).astype(np.float32)
    lats = rng.uniform(mnlat, mxlat, n).astype(np.float32)
    return np.stack([lons, lats], axis=1)

# -------- Utilities --------
def timeit(fn, *args, warmup: int=1, repeat: int=1, **kwargs) -> float:
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) / repeat, out

def mb(x: int) -> float:
    return x / (1024.0 * 1024.0)

# -------- SH: Analytic (Russwurm) --------
def build_sh_analytic(coords_deg: np.ndarray, L: int) -> np.ndarray:
    if not ANALYTIC_OK:
        raise ImportError("Analytic SH module not found (spherical_harmonics_ylm.py).")
    lon = torch.from_numpy(coords_deg[:,0]).to(torch.float32)
    lat = torch.from_numpy(coords_deg[:,1]).to(torch.float32)
    phi   = torch.deg2rad(lon + 180.0)
    theta = torch.deg2rad(lat +  90.0)
    cols = []
    for l in range(L+1):
        for m in range(-l, l+1):
            y = SH_analytic(m, l, phi, theta)
            if isinstance(y, float):
                y = y * torch.ones_like(phi)
            cols.append(y)
    X = torch.stack(cols, dim=1)  # [N, (L+1)^2]
    return X.detach().cpu().numpy()

# -------- SH: PySHTOOLS Legendre backend --------
def _pysh_legendre_matrix_for_point(L: int, x: float) -> np.ndarray:
    """
    Returns P_lm (orthonormalized) as a (L+1, L+1) array with entries for 0<=m<=l,
    zeros where m>l. Uses PySHTOOLS' PlmON.
    """
    Plm = pysh_Plms(L, x)  # Some versions return triangular 1D; handle both.
    if isinstance(Plm, np.ndarray) and Plm.ndim == 2:
        # expected shape (L+1, L+1) with P[l, m] valid for m<=l
        return Plm
    else:
        # fallback: flatten triangular -> (L+1,L+1)
        P = np.zeros((L+1, L+1), dtype=float)
        idx = 0
        for l in range(L+1):
            for m in range(l+1):
                P[l, m] = Plm[idx]
                idx += 1
        return P

def build_sh_pysh(coords_deg: np.ndarray, L: int) -> np.ndarray:
    """
    Build real SH design matrix using PySHTOOLS' Legendre functions (no SciPy).
    Normalization will be orthonormal (PySH’s PlmON). For timing, exact scale
    constants don’t matter; we only care about compute cost.
    """
    if not (HAVE_PYSH and HAVE_PYSH_LEG):
        raise ImportError("PySHTOOLS Legendre backend not available.")
    lon = coords_deg[:,0].astype(np.float64)
    lat = coords_deg[:,1].astype(np.float64)
    phi   = np.deg2rad(lon + 180.0)
    theta = np.deg2rad(lat +  90.0)
    x = np.cos(theta)
    N = len(coords_deg)
    D = (L+1)*(L+1)
    X = np.empty((N, D), dtype=np.float32)

    # column ordering matches analytic: iterate l then m=-l..l
    col = 0
    # Precompute cos(m phi), sin(m phi) up to L for each sample
    cos_mphi = np.empty((N, L+1), dtype=np.float64)
    sin_mphi = np.empty((N, L+1), dtype=np.float64)
    cos_mphi[:,0] = 1.0
    sin_mphi[:,0] = 0.0
    for m in range(1, L+1):
        cos_mphi[:,m] = np.cos(m*phi)
        sin_mphi[:,m] = np.sin(m*phi)

    for l in range(L+1):
        # compute P_lm for this l at all points (by point; simple & clear)
        # (We could batch, but this is a benchmark.)
        # Fill columns for m=-l..l
        for m in range(-l, l+1):
            # fill one column
            vals = np.empty(N, dtype=np.float64)
            am = abs(m)
            for i in range(N):
                P = _pysh_legendre_matrix_for_point(L, x[i])  # (L+1,L+1)
                Plm = P[l, am]
                if m > 0:
                    vals[i] = Plm * cos_mphi[i, am]
                elif m < 0:
                    vals[i] = Plm * sin_mphi[i, am]
                else:
                    vals[i] = Plm
            X[:, col] = vals.astype(np.float32)
            col += 1
    return X

# -------- Slepians via PySHTOOLS --------
def slepian_build(center: Tuple[float,float], R: float, Ls: int):
    sl = pysh.Slepian.from_cap(theta=R, lmax=Ls)
    sl.rotate(clon=center[0], clat=center[1])
    return sl

def slepian_eval(sl, coords_deg: np.ndarray, K_keep: Optional[int]=None, lam_thresh: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    lam = sl.eigenvalues
    order = np.arange(len(lam))
    if lam_thresh is not None:
        order = order[lam >= lam_thresh]
    if K_keep is not None:
        order = order[:K_keep]
    lam_kept = lam[order].astype(np.float32)
    lon = coords_deg[:,0]
    lat = coords_deg[:,1]
    lon360 = np.where(lon < 0.0, lon + 360.0, lon)
    cols = []
    for i in order:
        vals = sl.to_shcoeffs(i).expand(lon=lon360, lat=lat, degrees=True).astype(np.float32)
        cols.append(vals)
    X = np.column_stack(cols) if cols else np.zeros((len(coords_deg), 0), np.float32)
    return X, lam_kept

def shannon_K(Ls: int, cap_radius_deg: float) -> int:
    A = 2 * math.pi * (1.0 - math.cos(math.radians(cap_radius_deg)))  # cap area
    return max(1, int(round(((Ls + 1)**2) * (A / (4 * math.pi)))))

# -------- Main benchmark --------
def run(args):
    region = args.region.lower()
    if region not in REGIONS:
        raise ValueError(f"--region must be one of: {list(REGIONS.keys())}")
    info = REGIONS[region]
    bbox   = info["bbox"]
    center = info["center"]

    # coords
    coords = sample_random_coords(bbox, n=args.n_samp, seed=args.seed)
    print(f"Region={region}  coords={coords.shape}  bbox={bbox}  center={center}")

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    sh_backends = [b.strip().lower() for b in args.sh_backends.split(",") if b.strip()]
    # --- SH sweeps (both analytic & pysh if selected) ---
    for L in [int(x) for x in args.L_list.split(",") if x.strip()]:
        dim = (L + 1) ** 2

        if "analytic" in sh_backends:
            def _eval_analytic():
                return build_sh_analytic(coords, L)
            t, X = timeit(_eval_analytic, warmup=args.warmup, repeat=args.repeat)
            feats = coords.shape[0] * dim
            thr = (feats / t) / 1e6  # million feature-evals/sec
            mem_mb = mb(coords.shape[0] * dim * 4)
            results.append(dict(
                family="SH_analytic", region=region, N=coords.shape[0], L=L, dim=dim,
                cap_radius=None, K=None, time_sec=t, throughput_Mfeats_per_s=thr, est_memory_MB=mem_mb
            ))
            print(f"[SH_analytic] L={L:>3} dim={dim:>5}  time={t:.3f}s  thr={thr:.1f} Mfeats/s  ~{mem_mb:.1f}MB")

        if "pysh" in sh_backends:
            if not (HAVE_PYSH and HAVE_PYSH_LEG):
                print("  (skip SH_pysh: PySHTOOLS Legendre backend unavailable)")
            else:
                def _eval_pysh():
                    return build_sh_pysh(coords, L)
                t, X = timeit(_eval_pysh, warmup=args.warmup, repeat=args.repeat)
                feats = coords.shape[0] * dim
                thr = (feats / t) / 1e6
                mem_mb = mb(coords.shape[0] * dim * 4)
                results.append(dict(
                    family="SH_pysh", region=region, N=coords.shape[0], L=L, dim=dim,
                    cap_radius=None, K=None, time_sec=t, throughput_Mfeats_per_s=thr, est_memory_MB=mem_mb
                ))
                print(f"[SH_pysh    ] L={L:>3} dim={dim:>5}  time={t:.3f}s  thr={thr:.1f} Mfeats/s  ~{mem_mb:.1f}MB")

    # --- Slepian sweep (with build vs eval timing) ---
    if not HAVE_PYSH:
        print("PySHTOOLS not available; skipping Slepians.")
    else:
        Ls_list = [int(x) for x in args.Ls_list.split(",") if x.strip()]
        R_list  = [float(x) for x in args.cap_radius_list.split(",") if x.strip()]

        for Ls in Ls_list:
            for R in R_list:
                K_shan = shannon_K(Ls, R)
                K_keep = args.K_keep if args.K_keep is not None else K_shan

                # Build once
                t_build, sl = timeit(slepian_build, center, R, Ls, warmup=0, repeat=1)
                # Evaluate all kept modes
                def _eval():
                    Xs, lam = slepian_eval(sl, coords, K_keep=K_keep, lam_thresh=args.lam_thresh)
                    return Xs
                t_eval, Xs = timeit(_eval, warmup=args.warmup, repeat=args.repeat)

                dim = Xs.shape[1]
                feats = coords.shape[0] * max(dim, 1)
                thr = (feats / t_eval) / 1e6
                mem_mb = mb(coords.shape[0] * max(dim,1) * 4)
                results.append(dict(
                    family="Slepian", region=region, N=coords.shape[0],
                    L=Ls, dim=int(dim), cap_radius=R, K=int(dim),
                    time_sec=t_build + t_eval,
                    time_build_sec=t_build, time_eval_sec=t_eval,
                    throughput_Mfeats_per_s=thr, est_memory_MB=mem_mb
                ))
                print(f"[Slepian    ] Ls={Ls:>3} R={R:>4.1f}°  K={dim:>4}  build={t_build:.3f}s  eval={t_eval:.3f}s  thr={thr:.1f} Mfeats/s  ~{mem_mb:.1f}MB")

    # --- Save CSV ---
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = outdir / f"locenc_bench_{region}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # --- Plots ---
    # 1) time vs dimension
    fig, ax = plt.subplots(figsize=(7,4))
    if not df.empty:
        for fam, g in df.groupby("family"):
            ax.plot(g["dim"], g["time_sec"], "o-", label=fam)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{region} – time vs dim (N={coords.shape[0]})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(outdir / f"time_vs_dim_{region}.png", dpi=160, bbox_inches="tight")

    # 2) throughput vs dimension
    fig, ax = plt.subplots(figsize=(7,4))
    if not df.empty:
        for fam, g in df.groupby("family"):
            ax.plot(g["dim"], g["throughput_Mfeats_per_s"], "o-", label=fam)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("throughput (million feature-evals/s)")
    ax.set_title(f"{region} – throughput vs dim (N={coords.shape[0]})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(outdir / f"throughput_vs_dim_{region}.png", dpi=160, bbox_inches="tight")

    # 3) Slepian build vs eval stacked bars (if present)
    if "time_build_sec" in df.columns:
        g = df[df["family"]=="Slepian"].copy()
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(g["dim"], g["time_build_sec"], label="build", alpha=0.7)
            ax.bar(g["dim"], g["time_eval_sec"], bottom=g["time_build_sec"], label="eval", alpha=0.7)
            ax.set_xlabel("Slepian dimension (K kept)")
            ax.set_ylabel("time (s)")
            ax.set_title(f"{region} – Slepian build vs eval (N={coords.shape[0]})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(outdir / f"slepian_build_vs_eval_{region}.png", dpi=160, bbox_inches="tight")

    print(f"Saved plots in {outdir}")


def run_unified_table(args):
    """
    Run benchmark with same L values for both SH (analytic) and Slepians.
    Generates a unified comparison table.
    """
    region = args.region.lower()
    if region not in REGIONS:
        raise ValueError(f"--region must be one of: {list(REGIONS.keys())}")
    info = REGIONS[region]
    bbox   = info["bbox"]
    center = info["center"]

    # coords
    coords = sample_random_coords(bbox, n=args.n_samp, seed=args.seed)
    print(f"Region={region}  coords={coords.shape}  bbox={bbox}  center={center}")

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse L values (same for both SH and Slepian)
    L_values = [int(x) for x in args.L_list.split(",") if x.strip()]
    cap_radii = [float(x) for x in args.cap_radius_list.split(",") if x.strip()]

    print(f"\nUnified benchmark with L values: {L_values}")
    print(f"Cap radii for Slepians: {cap_radii}")
    print("=" * 80)

    results: List[Dict] = []

    # --- Run benchmarks for each L ---
    for L in L_values:
        dim_sh = (L + 1) ** 2

        # 1. Global SH (analytic only)
        if not ANALYTIC_OK:
            print(f"[ERROR] Analytic SH module not available!")
            continue

        def _eval_analytic():
            return build_sh_analytic(coords, L)
        t_sh, X_sh = timeit(_eval_analytic, warmup=args.warmup, repeat=args.repeat)
        mem_sh = mb(coords.shape[0] * dim_sh * 4)

        results.append(dict(
            method="Global SH",
            L=L,
            cap_radius=None,
            K=dim_sh,
            time_total_sec=t_sh,
            time_build_sec=0.0,
            time_eval_sec=t_sh,
            mem_MB=mem_sh,
        ))
        print(f"[Global SH ] L={L:>3}  dim={dim_sh:>5}  time={t_sh:.3f}s  mem={mem_sh:.1f}MB")

        # 2. Slepians at same L with different cap sizes
        if HAVE_PYSH:
            for R in cap_radii:
                K_shan = shannon_K(L, R)
                K_keep = args.K_keep if args.K_keep is not None else K_shan

                # Build
                t_build, sl = timeit(slepian_build, center, R, L, warmup=0, repeat=1)

                # Evaluate
                def _eval():
                    Xs, lam = slepian_eval(sl, coords, K_keep=K_keep, lam_thresh=args.lam_thresh)
                    return Xs
                t_eval, Xs = timeit(_eval, warmup=args.warmup, repeat=args.repeat)

                dim_slep = Xs.shape[1]
                mem_slep = mb(coords.shape[0] * max(dim_slep, 1) * 4)
                t_total = t_build + t_eval

                results.append(dict(
                    method="Slepian",
                    L=L,
                    cap_radius=R,
                    K=int(dim_slep),
                    time_total_sec=t_total,
                    time_build_sec=t_build,
                    time_eval_sec=t_eval,
                    mem_MB=mem_slep,
                ))
                print(f"[Slepian   ] L={L:>3}  R={R:>4.1f}°  K={dim_slep:>4}  "
                      f"build={t_build:.2f}s  eval={t_eval:.2f}s  total={t_total:.2f}s  mem={mem_slep:.1f}MB")
        else:
            print("  (PySHTOOLS not available for Slepians)")

    # --- Save CSV ---
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = outdir / f"unified_bench_{region}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # --- Generate LaTeX table ---
    latex_path = outdir / f"unified_table_{region}.tex"
    generate_latex_table(df, latex_path, region, coords.shape[0])
    print(f"Saved LaTeX: {latex_path}")

    # --- Summary plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time comparison
    ax = axes[0]
    sh_df = df[df["method"] == "Global SH"]
    ax.plot(sh_df["L"], sh_df["time_total_sec"], "o-", label="Global SH", linewidth=2, markersize=8)

    for R in cap_radii:
        slep_df = df[(df["method"] == "Slepian") & (df["cap_radius"] == R)]
        if not slep_df.empty:
            ax.plot(slep_df["L"], slep_df["time_total_sec"], "s--", label=f"Slepian (R={R}°)", linewidth=2, markersize=6)

    ax.set_xlabel("Band-limit L", fontsize=12)
    ax.set_ylabel("Total Time (s)", fontsize=12)
    ax.set_title(f"Computation Time Comparison ({region})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Memory comparison
    ax = axes[1]
    ax.plot(sh_df["L"], sh_df["mem_MB"], "o-", label="Global SH", linewidth=2, markersize=8)

    for R in cap_radii:
        slep_df = df[(df["method"] == "Slepian") & (df["cap_radius"] == R)]
        if not slep_df.empty:
            ax.plot(slep_df["L"], slep_df["mem_MB"], "s--", label=f"Slepian (R={R}°)", linewidth=2, markersize=6)

    ax.set_xlabel("Band-limit L", fontsize=12)
    ax.set_ylabel("Memory (MB)", fontsize=12)
    ax.set_title(f"Memory Usage Comparison ({region})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / f"unified_comparison_{region}.png", dpi=160, bbox_inches="tight")
    print(f"Saved plot: {outdir / f'unified_comparison_{region}.png'}")


def generate_latex_table(df, output_path: Path, region: str, n_samples: int):
    """
    Generate a unified LaTeX table comparing Global SH and Slepian timings.
    """
    import pandas as pd

    sh_df = df[df["method"] == "Global SH"].sort_values("L")
    slep_df = df[df["method"] == "Slepian"]

    # Get unique L values and cap radii
    L_values = sorted(df["L"].unique())
    cap_radii = sorted(slep_df["cap_radius"].dropna().unique())

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{\textbf{Computation time and memory comparison: Global SH vs Regional Slepian.} ")
    lines.append(f"Evaluated on {n_samples:,} random coordinates in the {region} region. ")
    lines.append(r"Global SH uses analytic computation. Slepian times include basis construction and evaluation.}")
    lines.append(r"\label{tab:unified_timing}")
    lines.append(r"\vspace{1ex}")
    lines.append("")

    # Build header
    n_caps = len(cap_radii)
    header_cols = "r" + "rr" + "rr" * n_caps  # L | SH time, SH mem | (Slep time, Slep mem) per cap
    lines.append(r"\begin{tabular}{@{}" + header_cols + r"@{}}")
    lines.append(r"\toprule")

    # Two-level header
    header1 = r"\multirow{2}{*}{\textbf{$L$}} & \multicolumn{2}{c}{\textbf{Global SH}}"
    for R in cap_radii:
        header1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{Slepian ({R:.0f}$^\\circ$)}}}}"
    header1 += r" \\"
    lines.append(header1)

    header2 = " & \\textbf{Time [s]} & \\textbf{Mem [MB]}"
    for R in cap_radii:
        header2 += r" & \textbf{Time [s]} & \textbf{Mem [MB]}"
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){2-3}" + "".join([f"\\cmidrule(lr){{{2+2*i+2}-{2+2*i+3}}}" for i in range(n_caps)]))
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    for L in L_values:
        sh_row = sh_df[sh_df["L"] == L]
        if sh_row.empty:
            continue

        sh_time = sh_row["time_total_sec"].values[0]
        sh_mem = sh_row["mem_MB"].values[0]

        row = f"{L} & {sh_time:.2f} & {sh_mem:.1f}"

        for R in cap_radii:
            slep_row = slep_df[(slep_df["L"] == L) & (slep_df["cap_radius"] == R)]
            if slep_row.empty:
                row += " & -- & --"
            else:
                slep_time = slep_row["time_total_sec"].values[0]
                slep_mem = slep_row["mem_MB"].values[0]
                slep_K = slep_row["K"].values[0]
                # Add K in parentheses after time
                row += f" & {slep_time:.2f} ($K$={slep_K}) & {slep_mem:.1f}"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Benchmark SH (analytic) vs Slepian (PySHTOOLS) on fixed regions.")
    ap.add_argument("--region", required=True,
                    choices=list(REGIONS.keys()),
                    help="brazil, sri_lanka, mozambique, china, antarctica, india, greenland")
    ap.add_argument("--n-samp", type=int, default=30000, help="Number of random lon/lat samples")
    ap.add_argument("--seed", type=int, default=123)

    # Mode selection
    ap.add_argument("--unified-table", action="store_true",
                    help="Generate unified table with same L values for SH and Slepians (analytic SH only)")

    # L values (used for both SH and Slepian in unified mode)
    ap.add_argument("--L-list", type=str, default="10,20,30,40,50",
                    help="L values for SH (and Slepian in unified mode)")
    ap.add_argument("--sh-backends", type=str, default="analytic,pysh",
                    help="Comma-separated: analytic, pysh (ignored in unified mode)")

    # Slepian sweep (Ls-list only used in legacy mode)
    ap.add_argument("--Ls-list", type=str, default="10,20,30,40,50,64,96,120",
                    help="L values for Slepian (legacy mode only)")
    ap.add_argument("--cap-radius-list", type=str, default="6,10,20",
                    help="Cap radii in degrees for Slepian")
    ap.add_argument("--K-keep", type=int, default=None,
                    help="Keep top-K Slepian modes; default uses Shannon K")
    ap.add_argument("--lam-thresh", type=float, default=None,
                    help="Optional eigenvalue threshold before top-K")

    # Timing
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)

    # Output
    ap.add_argument("--out-dir", type=Path, default=Path("./locenc_bench_outputs"))
    args = ap.parse_args()

    if args.unified_table:
        run_unified_table(args)
    else:
        run(args)

if __name__ == "__main__":
    main()
