#!/usr/bin/env python3
# test_ace.py
import os, argparse, math, re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import netCDF4

from models.ste_encoder import STEEncoder
from utils.data_utils import DataNormalizer

# ---------------- data ----------------
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

# ---------------- eval ----------------
@torch.no_grad()
def evaluate(model, dl, device, normalizer, per_var=False):
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
    if per_var:
        print("Per-var RMSE:", [f"{x:.4f}" for x in rmse])
        print("Per-var  MAE:", [f"{x:.4f}" for x in mae])
    return float(np.mean(rmse)), float(np.mean(mae))

# ---------------- inference helpers ----------------
def parse_basis_from_name(name: str):
    n = name.lower()
    # DPSS: handle first
    if "dpss" in n:
        return "dpss"
    # analytic baselines
    if "no_time"   in n: return "no_time"
    if "time_copy" in n: return "time_copy"
    if "fourier"   in n: return "fourier"
    if "legendre"  in n: return "legendre"
    if "monomial"  in n: return "monomial"
    if "triangle"  in n: return "triangle"
    return None

def infer_arch_from_state_dict(sd):
    """
    Infer architecture from a checkpoint state_dict with rigorous checks.
    Returns a dict with keys:
      model ('dpss'|'ste'), spatial_L, projection_dim, temporal_dim,
      dpss_optimized (bool), dpss_N, dpss_K_dpss, dpss_K_out, dpss_NW_min,
      hidden_dim, combined_dim, combination, output_dim, num_layers,
      has_time (bool)
    """
    info = {}
    is_dpss = "temporal_encoder.dpss_base.dpss_seqs" in sd
    info["model"] = "dpss" if is_dpss else "ste"

    # spatial stream
    try:
        W_space0 = sd["space_mlp.feats.0.weight"]  # [proj_dim, spatial_dim]
    except KeyError as e:
        raise KeyError("Missing key 'space_mlp.feats.0.weight' in checkpoint") from e
    proj_dim   = int(W_space0.shape[0])
    spatial_dim= int(W_space0.shape[1])
    L_guess = int(round(math.sqrt(spatial_dim)))
    spatial_L = L_guess if (L_guess * L_guess == spatial_dim) else None
    info.update(dict(projection_dim=proj_dim, spatial_dim=spatial_dim, spatial_L=spatial_L))

    # head & combination
    try:
        W_head0 = sd["head.feats.0.weight"]  # [hidden_dim, combined_dim]
        hidden_dim, combined_dim = int(W_head0.shape[0]), int(W_head0.shape[1])
    except KeyError as e:
        raise KeyError("Missing key 'head.feats.0.weight' in checkpoint") from e
    try:
        output_dim = int(sd["head.class_emb.weight"].shape[0])
    except KeyError as e:
        raise KeyError("Missing key 'head.class_emb.weight' in checkpoint") from e

    # time side
    has_time_mlp = "time_mlp.feats.0.weight" in sd

    if is_dpss:
        dpss_base = sd["temporal_encoder.dpss_base.dpss_seqs"]  # [K_dpss, N]
        K_dpss = int(dpss_base.shape[0])
        N      = int(dpss_base.shape[1])
        has_proj = "temporal_encoder.projection.weight" in sd
        info["dpss_optimized"] = bool(has_proj)
        if has_proj:
            W_proj = sd["temporal_encoder.projection.weight"]  # [K_out, K_dpss]
            K_out  = int(W_proj.shape[0])
            K_dpss_from_proj = int(W_proj.shape[1])
            if K_dpss_from_proj != K_dpss:
                raise ValueError(f"Mismatch: dpss_base K={K_dpss} but projection expects {K_dpss_from_proj}")
        else:
            if not has_time_mlp:
                raise KeyError("Non-optimized DPSS is missing 'time_mlp' weights")
            K_out = int(sd["time_mlp.feats.0.weight"].shape[1])
            if K_out != K_dpss:
                raise ValueError(f"Non-optimized DPSS expects K_out==K_dpss, got {K_out} vs {K_dpss}")
        # NW minimal to avoid capping K (K ≤ 2*NW − 1)
        K_needed = max(K_dpss, K_out)
        NW_min = int(math.ceil((K_needed + 1) / 2.0))
        info.update(dict(
            has_time=True,
            temporal_dim=K_out,
            dpss_N=N, dpss_K_dpss=K_dpss, dpss_K_out=K_out, dpss_NW_min=NW_min
        ))
    else:
        if has_time_mlp:
            temporal_dim = int(sd["time_mlp.feats.0.weight"].shape[1])
            info.update(dict(has_time=True, temporal_dim=temporal_dim, dpss_optimized=False))
        else:
            # true "no_time" checkpoint: ensure combined_dim == proj_dim
            if combined_dim != proj_dim:
                raise ValueError("no_time checkpoint inferred but combined_dim != proj_dim")
            info.update(dict(has_time=False, temporal_dim=0, dpss_optimized=False))

    # deduce combination exactly
    if combined_dim == 2 * proj_dim and info.get("has_time", False):
        combo = "concatenate"
    elif combined_dim == proj_dim and not info.get("has_time", True):
        combo = "concatenate"  # spatial only
    elif combined_dim == proj_dim and info.get("has_time", False):
        combo = "hadamard"
    elif combined_dim == proj_dim * proj_dim and info.get("has_time", False):
        combo = "outer_product"
    else:
        raise ValueError(f"Cannot infer combination from combined_dim={combined_dim}, proj_dim={proj_dim}, has_time={info.get('has_time')}")

    # count res blocks
    res_indices = set()
    for k in sd.keys():
        if k.startswith("head.feats.") and k.endswith(".w1.weight"):
            parts = k.split(".")
            if len(parts) >= 4:
                try: res_indices.add(int(parts[2]))
                except ValueError: pass
    num_layers = len(res_indices)

    info.update(dict(
        hidden_dim   = hidden_dim,
        combined_dim = combined_dim,
        combination  = combo,
        output_dim   = output_dim,
        num_layers   = num_layers,
    ))
    return info

# ---------------- args & main ----------------
def parse_args():
    p = argparse.ArgumentParser("Auto-infer test for STE/DPSS from checkpoint")
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--data_path", type=str, default="/scratch/local/arra4944_images/ace/")
    p.add_argument("--batch_size", type=int, default=40000)
    p.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--train_frac", type=float, default=0.01)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--test_frac", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per_var", action="store_true")
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--basis_override", type=str, default=None,
                   choices=[None,"no_time","time_copy","triangle","monomial","legendre","fourier","dpss"],
                   help="Force temporal basis (overrides filename inference).")
    p.add_argument("--no_infer_from_name", action="store_true",
                   help="Do not infer basis from checkpoint filename.")
    return p.parse_args()

def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

    # Data (we only need test split/stats)
    _, _, test_ds, normalizer = prepare_ace(args.data_path, args.train_frac, args.val_frac, args.test_frac)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    ckpt_path = Path(args.ckpt)
    sd = torch.load(ckpt_path, map_location=args.device)
    if "state_dict" in sd:  # lightning-style
        sd = {k.replace("module.",""): v for k, v in sd["state_dict"].items()}

    info = infer_arch_from_state_dict(sd)
    print("Inferred:", info)

    # Decide temporal_type
    basis = args.basis_override
    if basis is None and not args.no_infer_from_name:
        basis = parse_basis_from_name(ckpt_path.name)

    if basis is None:
        # fallbacks
        basis = "dpss" if info["model"] == "dpss" else ("no_time" if not info["has_time"] else None)
    if basis is None:
        raise ValueError("Could not safely infer analytic basis from filename. Provide --basis_override.")

    spatial_L = info["spatial_L"] if info["spatial_L"] is not None else 20

    # Build model with EXACT shapes
    if basis == "dpss":
        NW_needed = int(info["dpss_NW_min"])
        model = STEEncoder(
            spatial_L      = spatial_L,
            temporal_type  = "dpss",
            temporal_dim   = int(info["dpss_K_out"]),   # K_out
            combination    = info["combination"],
            hidden_dim     = info["hidden_dim"],
            num_layers     = info["num_layers"],
            output_dim     = info["output_dim"],
            projection_dim = info["projection_dim"],
            dpss_N         = int(info["dpss_N"]),
            dpss_NW        = NW_needed,
            dpss_optimized = bool(info["dpss_optimized"]),
            dpss_K_dpss    = int(info["dpss_K_dpss"]),  # <--- critical
        )
    else:
        model = STEEncoder(
            spatial_L      = spatial_L,
            temporal_type  = basis,                      # exact analytic family
            temporal_dim   = int(info["temporal_dim"]),
            combination    = info["combination"],
            hidden_dim     = info["hidden_dim"],
            num_layers     = info["num_layers"],
            output_dim     = info["output_dim"],
            projection_dim = info["projection_dim"],
        )

    model = model.to(args.device)

    # Load weights
    missing, unexpected = model.load_state_dict(sd, strict=args.strict_load)
    if not args.strict_load:
        print(f"Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:   print("  Missing (first 10):", list(missing)[:10])
        if unexpected:print("  Unexpected (first 10):", list(unexpected)[:10])

    # Evaluate
    rmse, mae = evaluate(model, test_loader, args.device, normalizer, per_var=args.per_var)
    print("\nFinal Test Evaluation")
    print(f"Test RMSE: {rmse:.4f} K")
    print(f"Test MAE:  {mae:.4f} K")

if __name__ == "__main__":
    main()
