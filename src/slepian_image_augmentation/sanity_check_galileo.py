#!/usr/bin/env python3
"""
Galileo Model Sanity Check
==========================
Simple script to verify correct usage of pre-trained Galileo model on Sentinel-2 imagery.

Usage:
    python sanity_check_galileo.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Add galileo to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "galileo"))
from single_file_galileo import Encoder

# ===================== CONSTANTS =====================
# Normalization stats from Galileo pretraining (indices 2-11 are S2 bands)
SPACE_TIME_MEAN = np.array([
    -11.73, -18.86,  # S1
    1395.34, 1338.40, 1343.10, 1543.86, 2186.20, 2525.09, 2410.34, 2750.29, 2234.91, 1474.53,  # S2
    0.289,  # NDVI
], dtype=np.float32)

SPACE_TIME_STD = np.array([
    4.89, 5.73,  # S1
    917.70, 913.30, 1092.68, 1047.22, 1048.01, 1143.69, 1098.98, 1204.47, 1145.98, 980.24,  # S2
    0.272,  # NDVI
], dtype=np.float32)

# Tensor dimensions
N_SPACE_TIME_BANDS, N_SPACE_TIME_GROUPS = 13, 7
N_SPACE_BANDS, N_SPACE_GROUPS = 16, 3
N_TIME_BANDS, N_TIME_GROUPS = 6, 3
N_STATIC_BANDS, N_STATIC_GROUPS = 18, 4
S2_START_IDX, S2_END_IDX = 2, 12  # S2 bands in space_time tensor
S2_MASK_GROUPS = [1, 2, 3, 4, 5]  # S2 groups to unmask


def construct_s2_input(s2_patch: np.ndarray, device: str, month: int = 6) -> dict:
    """Construct Galileo input tensors from a single S2 patch [H, W, 10]."""
    h, w, c = s2_patch.shape
    assert c == 10, f"Expected 10 S2 bands, got {c}"
    t = 1  # single timestep

    # Space-time: [H, W, T, 13] - place S2 at indices 2-11
    s_t_x = np.zeros((h, w, t, N_SPACE_TIME_BANDS), dtype=np.float32)
    s_t_x[:, :, 0, S2_START_IDX:S2_END_IDX] = s2_patch
    s_t_x = (s_t_x - SPACE_TIME_MEAN) / SPACE_TIME_STD  # normalize

    # Other modalities: zeros (masked out)
    sp_x = np.zeros((h, w, N_SPACE_BANDS), dtype=np.float32)
    t_x = np.zeros((t, N_TIME_BANDS), dtype=np.float32)
    st_x = np.zeros((N_STATIC_BANDS,), dtype=np.float32)

    # Masks: 0=visible, 1=masked
    s_t_m = np.ones((h, w, t, N_SPACE_TIME_GROUPS), dtype=np.float32)
    for g in S2_MASK_GROUPS:
        s_t_m[:, :, :, g] = 0  # unmask S2 groups
    sp_m = np.ones((h, w, N_SPACE_GROUPS), dtype=np.float32)
    t_m = np.ones((t, N_TIME_GROUPS), dtype=np.float32)
    st_m = np.ones((N_STATIC_GROUPS,), dtype=np.float32)

    months = np.array([month], dtype=np.int64)

    return {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in {
        's_t_x': s_t_x, 'sp_x': sp_x, 't_x': t_x, 'st_x': st_x,
        's_t_m': s_t_m, 'sp_m': sp_m, 't_m': t_m, 'st_m': st_m, 'months': months
    }.items()}


def crop_to_multiple(patch: np.ndarray, multiple: int = 8) -> np.ndarray:
    """Center-crop so dimensions are divisible by multiple."""
    h, w, c = patch.shape
    new_h, new_w = (h // multiple) * multiple, (w // multiple) * multiple
    start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
    return patch[start_h:start_h+new_h, start_w:start_w+new_w, :]


@torch.no_grad()
def extract_embedding(model, patch: np.ndarray, device: str, patch_size: int = 8) -> np.ndarray:
    """Extract embedding from S2 patch."""
    patch = crop_to_multiple(patch, patch_size)
    inputs = construct_s2_input(patch, device)

    out = model(
        s_t_x=inputs['s_t_x'], sp_x=inputs['sp_x'], t_x=inputs['t_x'], st_x=inputs['st_x'],
        s_t_m=inputs['s_t_m'], sp_m=inputs['sp_m'], t_m=inputs['t_m'], st_m=inputs['st_m'],
        months=inputs['months'], patch_size=patch_size
    )
    s_t_out, sp_out, t_out, st_out, s_t_m_out, sp_m_out, t_m_out, st_m_out, _ = out

    embedding = model.average_tokens(
        s_t_out, sp_out, t_out, st_out, s_t_m_out, sp_m_out, t_m_out, st_m_out
    )
    return embedding.cpu().numpy().squeeze()


def main():
    print("=" * 60)
    print("GALILEO MODEL SANITY CHECK")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1] Device: {device}")

    # Load model
    model_path = SCRIPT_DIR / "galileo" / "data" / "models" / "nano"
    print(f"\n[2] Loading model from: {model_path}")
    model = Encoder.load_from_folder(model_path, device=torch.device(device))
    model = model.to(device).eval()
    print(f"    Model loaded! Embedding dim: 128, Depth: 4 blocks")

    # Load sample patches
    h5_path = "/scratch/local/arra4944_images/openbuildings/southflorida_s2_patches.h5"
    print(f"\n[3] Loading patches from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())[:5]  # first 5 patches
        patches = [np.array(f[k]).astype(np.float32) for k in keys]

    print(f"    Loaded {len(patches)} patches")
    print(f"    Sample shape: {patches[0].shape}, dtype: {patches[0].dtype}")

    # Extract embeddings
    print(f"\n[4] Extracting embeddings...")
    embeddings = []
    for i, (key, patch) in enumerate(zip(keys, patches)):
        emb = extract_embedding(model, patch, device)
        embeddings.append(emb)
        print(f"    Patch {i+1} ({key}): shape={emb.shape}, "
              f"mean={emb.mean():.4f}, std={emb.std():.4f}, "
              f"min={emb.min():.4f}, max={emb.max():.4f}")

    # Check for NaN/Inf
    embeddings = np.stack(embeddings)
    print(f"\n[5] Embedding sanity checks:")
    print(f"    All finite: {np.isfinite(embeddings).all()}")
    print(f"    Any NaN: {np.isnan(embeddings).any()}")

    # Pairwise cosine similarities
    print(f"\n[6] Pairwise cosine similarities:")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            print(f"    Patch {i+1} vs {j+1}: {sim_matrix[i,j]:.4f}")

    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
