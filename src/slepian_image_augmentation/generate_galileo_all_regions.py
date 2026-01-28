#!/usr/bin/env python3
"""
Generate Galileo Embeddings for All OpenBuildings Regions
==========================================================
Processes all S2 patches and adds galileo_embedding column to parquet files.

Usage:
    python generate_galileo_all_regions.py --data-dir $DATA_DIR
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add galileo to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "galileo"))
from single_file_galileo import Encoder

# ===================== CONSTANTS =====================
SPACE_TIME_MEAN = np.array([
    -11.73, -18.86,
    1395.34, 1338.40, 1343.10, 1543.86, 2186.20, 2525.09, 2410.34, 2750.29, 2234.91, 1474.53,
    0.289,
], dtype=np.float32)

SPACE_TIME_STD = np.array([
    4.89, 5.73,
    917.70, 913.30, 1092.68, 1047.22, 1048.01, 1143.69, 1098.98, 1204.47, 1145.98, 980.24,
    0.272,
], dtype=np.float32)

N_SPACE_TIME_BANDS, N_SPACE_TIME_GROUPS = 13, 7
N_SPACE_BANDS, N_SPACE_GROUPS = 16, 3
N_TIME_BANDS, N_TIME_GROUPS = 6, 3
N_STATIC_BANDS, N_STATIC_GROUPS = 18, 4
S2_START_IDX, S2_END_IDX = 2, 12
S2_MASK_GROUPS = [1, 2, 3, 4, 5]


def construct_s2_input(s2_patch: np.ndarray, device: str, month: int = 6) -> dict:
    """Construct Galileo input tensors from S2 patch [H, W, 10]."""
    h, w, c = s2_patch.shape
    t = 1

    s_t_x = np.zeros((h, w, t, N_SPACE_TIME_BANDS), dtype=np.float32)
    s_t_x[:, :, 0, S2_START_IDX:S2_END_IDX] = s2_patch
    s_t_x = (s_t_x - SPACE_TIME_MEAN) / SPACE_TIME_STD

    sp_x = np.zeros((h, w, N_SPACE_BANDS), dtype=np.float32)
    t_x = np.zeros((t, N_TIME_BANDS), dtype=np.float32)
    st_x = np.zeros((N_STATIC_BANDS,), dtype=np.float32)

    s_t_m = np.ones((h, w, t, N_SPACE_TIME_GROUPS), dtype=np.float32)
    for g in S2_MASK_GROUPS:
        s_t_m[:, :, :, g] = 0
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
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
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


def make_coord_key(lon: float, lat: float) -> str:
    """Create key matching H5 format: 'lon_lat'."""
    return f"{lon:.4f}_{lat:.4f}"


def process_region(data_dir: Path, region: str, model, device: str):
    """Process a single region: load parquet, extract Galileo embeddings, save updated parquet."""
    print(f"\n{'='*60}")
    print(f"Processing: {region}")
    print(f"{'='*60}")

    pq_path = data_dir / f"{region}_grid.parquet"
    h5_path = data_dir / f"{region}_s2_patches.h5"
    out_path = data_dir / f"{region}_grid_galileo.parquet"

    if not pq_path.exists():
        print(f"  [SKIP] Parquet not found: {pq_path}")
        return
    if not h5_path.exists():
        print(f"  [SKIP] H5 not found: {h5_path}")
        return

    # Load parquet
    df = pd.read_parquet(pq_path)
    print(f"  Loaded {len(df)} rows from parquet")

    # Build lookup from H5 keys
    embeddings_dict = {}
    failed = 0

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"  Found {len(keys)} S2 patches in H5")

        for key in tqdm(keys, desc=f"  Extracting {region}"):
            try:
                patch = np.array(f[key]).astype(np.float32)
                emb = extract_embedding(model, patch, device)
                if emb is not None:
                    embeddings_dict[key] = emb
                else:
                    failed += 1
            except Exception as e:
                failed += 1

    print(f"  Extracted {len(embeddings_dict)} embeddings, {failed} failed")

    # Match embeddings to parquet rows by lon/lat
    galileo_embeddings = []
    matched = 0
    unmatched = 0

    for idx, row in df.iterrows():
        lon, lat = row['lon'], row['lat']
        key = make_coord_key(lon, lat)

        if key in embeddings_dict:
            emb = embeddings_dict[key]
            galileo_embeddings.append(emb.tobytes())
            matched += 1
        else:
            # Try nearby keys (floating point precision issues)
            found = False
            for k in embeddings_dict.keys():
                try:
                    k_lon, k_lat = float(k.split('_')[0]), float(k.split('_')[1])
                    if abs(k_lon - lon) < 0.0001 and abs(k_lat - lat) < 0.0001:
                        galileo_embeddings.append(embeddings_dict[k].tobytes())
                        matched += 1
                        found = True
                        break
                except:
                    continue
            if not found:
                # Use zero embedding as placeholder
                galileo_embeddings.append(np.zeros(128, dtype=np.float32).tobytes())
                unmatched += 1

    df['galileo_embedding'] = galileo_embeddings
    df['galileo_embedding_dim'] = 128

    # Save
    df.to_parquet(out_path, index=False)
    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory with openbuildings data')
    parser.add_argument('--regions', nargs='+', default=None,
                        help='Specific regions to process (default: all)')
    parser.add_argument('--model-size', default='nano', choices=['nano', 'tiny', 'base'])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Galileo model
    model_path = SCRIPT_DIR / "galileo" / "data" / "models" / args.model_size
    print(f"Loading Galileo-{args.model_size} from {model_path}")
    model = Encoder.load_from_folder(model_path, device=torch.device(device))
    model = model.to(device).eval()

    # Find regions
    if args.regions:
        regions = args.regions
    else:
        # Auto-detect from *_grid.parquet files
        pq_files = list(args.data_dir.glob("*_grid.parquet"))
        regions = [f.stem.replace("_grid", "") for f in pq_files]
        # Exclude files that already have galileo
        regions = [r for r in regions if not r.endswith("_galileo")]

    print(f"\nRegions to process: {regions}")

    for region in regions:
        process_region(args.data_dir, region, model, device)

    print("\n" + "="*60)
    print("DONE! All regions processed.")
    print("="*60)


if __name__ == "__main__":
    main()
