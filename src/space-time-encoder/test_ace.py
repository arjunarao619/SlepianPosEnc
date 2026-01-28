#!/usr/bin/env python3
"""
Batch evaluation of all baseline (noreg) and DPSS checkpoints on ACE data.
Reports per-variable RMSE and MAE for all 8 air temperature levels.

Usage:
    cd /projects/arra4944/SlepianPosEnc/src/space-time-encoder
    python test_ace.py [--test_frac 0.01] [--seed 0]
"""

import os, sys, math, re, argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import netCDF4

from models.ste_encoder import STEEncoder
from utils.data_utils import DataNormalizer

# ── constants ──────────────────────────────────────────────────────────
DATA_PATH = "/scratch/local/arra4944_images/ace/"
BASELINE_DIR = Path(__file__).resolve().parent / "checkpoints_baselines"
DPSS_DIR     = Path(__file__).resolve().parent / "checkpoints"
OUTPUT_CSV   = Path(__file__).resolve().parent / "test_results.csv"

VAR_NAMES = [
    "T0 (~26hPa)",
    "T1 (~99hPa)",
    "T2 (~203hPa)",
    "T3 (~337hPa)",
    "T4 (~504hPa)",
    "T5 (~690hPa)",
    "T6 (~850hPa)",
    "T7 (~964hPa)",
]

# ── data loading ───────────────────────────────────────────────────────
def load_ace_month(fp):
    with netCDF4.Dataset(fp, 'r') as nc:
        lons  = nc.variables['grid_xt'][:] - 180
        lats  = nc.variables['grid_yt'][:]
        times = nc.variables['time'][:]
        temps = np.stack([nc.variables[f'air_temperature_{i}'][:] for i in range(8)], axis=-1)
        lon_g, lat_g, time_g = np.meshgrid(lons, lats, times, indexing='ij')
        coords  = np.stack([lon_g.flatten(), lat_g.flatten(), time_g.flatten()], axis=1).astype(np.float32)
        targets = temps.transpose(2, 1, 0, 3).reshape(-1, 8).astype(np.float32)
    return coords, targets


def prepare_ace(data_path, train_frac, test_frac, split_seed=0):
    Cs, Ys = [], []
    for m in range(1, 13):
        fp = os.path.join(data_path, f"2021{m:02d}0100.nc")
        if os.path.exists(fp):
            c, y = load_ace_month(fp)
            Cs.append(c); Ys.append(y)
    coords  = torch.from_numpy(np.concatenate(Cs, 0))
    targets = torch.from_numpy(np.concatenate(Ys, 0))

    # normalize time to [-1, 1]
    tmin, tmax = coords[:, 2].min(), coords[:, 2].max()
    coords[:, 2] = 2 * (coords[:, 2] - tmin) / (tmax - tmin) - 1

    N = coords.shape[0]
    n_tr = int(N * train_frac)
    n_te = int(N * test_frac)

    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(N, generator=gen)
    tr = perm[:n_tr]
    te = perm[n_tr:n_tr + n_te]

    normalizer = DataNormalizer()
    normalizer.fit(targets[tr])
    targets_norm = normalizer.transform(targets)

    test_ds = TensorDataset(coords[te], targets_norm[te], targets[te])
    return test_ds, normalizer


# ── architecture inference ─────────────────────────────────────────────
def parse_basis_from_name(name):
    n = name.lower()
    if "dpss"      in n: return "dpss"
    if "no_time"   in n: return "no_time"
    if "time_copy" in n: return "time_copy"
    if "fourier"   in n: return "fourier"
    if "legendre"  in n: return "legendre"
    if "monomial"  in n: return "monomial"
    if "triangle"  in n: return "triangle"
    return None


def infer_arch_from_state_dict(sd):
    info = {}
    is_dpss = "temporal_encoder.dpss_base.dpss_seqs" in sd
    info["model"] = "dpss" if is_dpss else "ste"

    W_space0    = sd["space_mlp.feats.0.weight"]
    proj_dim    = int(W_space0.shape[0])
    spatial_dim = int(W_space0.shape[1])
    L_guess     = int(round(math.sqrt(spatial_dim)))
    spatial_L   = L_guess if L_guess * L_guess == spatial_dim else 20
    info.update(dict(projection_dim=proj_dim, spatial_dim=spatial_dim, spatial_L=spatial_L))

    W_head0 = sd["head.feats.0.weight"]
    hidden_dim, combined_dim = int(W_head0.shape[0]), int(W_head0.shape[1])
    output_dim = int(sd["head.class_emb.weight"].shape[0])

    has_time_mlp = "time_mlp.feats.0.weight" in sd

    if is_dpss:
        dpss_base = sd["temporal_encoder.dpss_base.dpss_seqs"]
        K_dpss, N = int(dpss_base.shape[0]), int(dpss_base.shape[1])
        has_proj  = "temporal_encoder.projection.weight" in sd
        info["dpss_optimized"] = bool(has_proj)
        if has_proj:
            K_out = int(sd["temporal_encoder.projection.weight"].shape[0])
        else:
            K_out = int(sd["time_mlp.feats.0.weight"].shape[1])
        NW_min = int(math.ceil((max(K_dpss, K_out) + 1) / 2.0))
        info.update(dict(has_time=True, temporal_dim=K_out,
                         dpss_N=N, dpss_K_dpss=K_dpss, dpss_K_out=K_out, dpss_NW_min=NW_min))
    else:
        if has_time_mlp:
            temporal_dim = int(sd["time_mlp.feats.0.weight"].shape[1])
            info.update(dict(has_time=True, temporal_dim=temporal_dim, dpss_optimized=False))
        else:
            info.update(dict(has_time=False, temporal_dim=0, dpss_optimized=False))

    # combination
    if combined_dim == 2 * proj_dim and info.get("has_time", False):
        combo = "concatenate"
    elif combined_dim == proj_dim and not info.get("has_time", True):
        combo = "concatenate"
    elif combined_dim == proj_dim and info.get("has_time", False):
        combo = "hadamard"
    else:
        combo = "concatenate"  # safe default
    info["combination"] = combo

    # count residual blocks
    res_indices = set()
    for k in sd:
        if k.startswith("head.feats.") and k.endswith(".w1.weight"):
            parts = k.split(".")
            if len(parts) >= 4:
                try: res_indices.add(int(parts[2]))
                except ValueError: pass
    info.update(dict(hidden_dim=hidden_dim, combined_dim=combined_dim,
                     output_dim=output_dim, num_layers=len(res_indices)))
    return info


# ── model building ─────────────────────────────────────────────────────
def build_model(basis, info, device):
    spatial_L = info.get("spatial_L", 20)
    if basis == "dpss":
        model = STEEncoder(
            spatial_L=spatial_L, temporal_type="dpss",
            temporal_dim=int(info["dpss_K_out"]),
            combination=info["combination"],
            hidden_dim=info["hidden_dim"], num_layers=info["num_layers"],
            output_dim=info["output_dim"], projection_dim=info["projection_dim"],
            dpss_N=int(info["dpss_N"]), dpss_NW=int(info["dpss_NW_min"]),
            dpss_optimized=bool(info["dpss_optimized"]),
            dpss_K_dpss=int(info["dpss_K_dpss"]),
        )
    else:
        model = STEEncoder(
            spatial_L=spatial_L, temporal_type=basis,
            temporal_dim=int(info["temporal_dim"]),
            combination=info["combination"],
            hidden_dim=info["hidden_dim"], num_layers=info["num_layers"],
            output_dim=info["output_dim"], projection_dim=info["projection_dim"],
        )
    return model.to(device)


# ── evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_per_var(model, dl, device, normalizer):
    """Returns (rmse_list, mae_list) each of length 8."""
    model.eval()
    P, T = [], []
    for x, y_norm, y_raw in dl:
        x = x.to(device, non_blocking=True)
        p_norm = model(x)
        P.append(normalizer.inverse_transform(p_norm.cpu()))
        T.append(y_raw)
    P, T = torch.cat(P), torch.cat(T)
    rmse, mae = [], []
    for i in range(8):
        mse_i = torch.mean((P[:, i] - T[:, i]) ** 2).item()
        rmse.append(math.sqrt(mse_i))
        mae.append(torch.mean(torch.abs(P[:, i] - T[:, i])).item())
    return rmse, mae


# ── checkpoint discovery ───────────────────────────────────────────────
def discover_checkpoints():
    """Return list of (label, basis, seed_str, path)."""
    ckpts = []

    # baselines: noreg only
    if BASELINE_DIR.exists():
        for p in sorted(BASELINE_DIR.glob("best_*_noreg_seed*.pt")):
            basis = parse_basis_from_name(p.stem)
            m = re.search(r"seed(\d+)", p.stem)
            seed_str = m.group(1) if m else "?"
            label = f"{basis}_noreg"
            ckpts.append((label, basis, seed_str, p))

    # DPSS
    if DPSS_DIR.exists():
        for p in sorted(DPSS_DIR.glob("best_dpss_NW*_K*_opt_seed*.pt")):
            m_nw = re.search(r"NW(\d+)", p.stem)
            m_k  = re.search(r"K(\d+)", p.stem)
            m_sd = re.search(r"seed(\d+)", p.stem)
            nw   = m_nw.group(1) if m_nw else "?"
            k    = m_k.group(1) if m_k else "?"
            seed_str = m_sd.group(1) if m_sd else "?"
            label = f"dpss_NW{nw}_K{k}"
            ckpts.append((label, "dpss", seed_str, p))

    return ckpts


# ── main ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Batch test all noreg baselines + DPSS on ACE")
    p.add_argument("--data_path",  type=str, default=DATA_PATH)
    p.add_argument("--train_frac", type=float, default=0.01)
    p.add_argument("--test_frac",  type=float, default=0.01)
    p.add_argument("--split_seed", type=int, default=0, help="RNG seed for train/test split")
    p.add_argument("--batch_size", type=int, default=40000)
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_csv", type=str, default=str(OUTPUT_CSV))
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    torch.set_float32_matmul_precision("high")

    # ── Load data once ──
    print("Loading ACE data...")
    test_ds, normalizer = prepare_ace(args.data_path, args.train_frac, args.test_frac, args.split_seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_ds)} samples\n")

    # ── Discover checkpoints ──
    ckpts = discover_checkpoints()
    if not ckpts:
        print("ERROR: No checkpoints found.")
        sys.exit(1)
    print(f"Found {len(ckpts)} checkpoints\n")

    # ── Header ──
    var_hdr = "  ".join(f"{v:>12s}" for v in VAR_NAMES)
    sep = "-" * 160
    print(f"{'Model':<25s} {'Seed':>4s}  {var_hdr}  {'Mean RMSE':>10s}")
    print(sep)

    rows = []  # for CSV

    for label, basis, seed_str, ckpt_path in ckpts:
        try:
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "state_dict" in sd:
                sd = {k.replace("module.", ""): v for k, v in sd["state_dict"].items()}
            info = infer_arch_from_state_dict(sd)
            model = build_model(basis, info, device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  WARNING [{label} seed{seed_str}]: {len(missing)} missing keys")

            rmse, mae = evaluate_per_var(model, test_loader, device, normalizer)
            mean_rmse = float(np.mean(rmse))
            mean_mae  = float(np.mean(mae))

            # print RMSE row
            vals = "  ".join(f"{r:12.4f}" for r in rmse)
            print(f"{label:<25s} {seed_str:>4s}  {vals}  {mean_rmse:10.4f}")

            rows.append(dict(
                model=label, basis=basis, seed=seed_str,
                **{f"rmse_{VAR_NAMES[i]}": rmse[i] for i in range(8)},
                mean_rmse=mean_rmse,
                **{f"mae_{VAR_NAMES[i]}": mae[i] for i in range(8)},
                mean_mae=mean_mae,
            ))

            del model
            torch.cuda.empty_cache() if device == "cuda" else None

        except Exception as e:
            print(f"{label:<25s} {seed_str:>4s}  ERROR: {e}")

    # ── Summary: mean ± std across seeds per baseline type ──
    print(f"\n{'='*160}")
    print("Summary (mean ± std RMSE across seeds)")
    print(f"{'='*160}")

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r["model"]].append(r)

    print(f"{'Model':<25s}  {var_hdr}  {'Mean RMSE':>10s}")
    print(sep)
    summary_rows = []
    for model_name in dict.fromkeys(r["model"] for r in rows):  # preserve order
        group = groups[model_name]
        if len(group) == 1:
            r = group[0]
            vals = "  ".join(f"{r[f'rmse_{VAR_NAMES[i]}']:12.4f}" for i in range(8))
            print(f"{model_name:<25s}  {vals}  {r['mean_rmse']:10.4f}")
            summary_rows.append(dict(
                model=model_name, n_seeds=1,
                **{f"rmse_{VAR_NAMES[i]}": r[f"rmse_{VAR_NAMES[i]}"] for i in range(8)},
                mean_rmse=r["mean_rmse"],
            ))
        else:
            means = [np.mean([g[f"rmse_{VAR_NAMES[i]}"] for g in group]) for i in range(8)]
            stds  = [np.std([g[f"rmse_{VAR_NAMES[i]}"]  for g in group]) for i in range(8)]
            overall_mean = np.mean([g["mean_rmse"] for g in group])
            overall_std  = np.std([g["mean_rmse"] for g in group])
            vals = "  ".join(f"{m:6.3f}±{s:4.3f}" for m, s in zip(means, stds))
            print(f"{model_name:<25s}  {vals}  {overall_mean:5.3f}±{overall_std:4.3f}")
            summary_rows.append(dict(
                model=model_name, n_seeds=len(group),
                **{f"rmse_{VAR_NAMES[i]}": means[i] for i in range(8)},
                **{f"rmse_std_{VAR_NAMES[i]}": stds[i] for i in range(8)},
                mean_rmse=overall_mean, mean_rmse_std=overall_std,
            ))

    # ── Save CSV (per-seed rows) ──
    csv_path = Path(args.output_csv)
    fieldnames = ["model", "basis", "seed"]
    for v in VAR_NAMES:
        fieldnames.append(f"rmse_{v}")
    fieldnames.append("mean_rmse")
    for v in VAR_NAMES:
        fieldnames.append(f"mae_{v}")
    fieldnames.append("mean_mae")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nResults saved to {csv_path}")
    print(f"Total: {len(rows)} checkpoints evaluated")


if __name__ == "__main__":
    main()
