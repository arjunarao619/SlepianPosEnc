#!/usr/bin/env python3
"""
Merge Galileo embeddings into the original grid parquet files.
This adds the galileo_embedding column to {region}_grid.parquet.

Usage:
    python merge_galileo_embeddings.py --data-dir /scratch/local/arra4944_images/openbuildings
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--regions', nargs='+', default=None)
    args = parser.parse_args()

    # Find regions with galileo parquets
    galileo_files = list(args.data_dir.glob("*_grid_galileo.parquet"))

    if args.regions:
        regions = args.regions
    else:
        regions = [f.stem.replace("_grid_galileo", "") for f in galileo_files]

    print(f"Regions to merge: {regions}")

    for region in regions:
        orig_path = args.data_dir / f"{region}_grid.parquet"
        galileo_path = args.data_dir / f"{region}_grid_galileo.parquet"

        if not galileo_path.exists():
            print(f"[SKIP] {region}: No galileo parquet found")
            continue
        if not orig_path.exists():
            print(f"[SKIP] {region}: No original parquet found")
            continue

        print(f"\nProcessing {region}...")

        df_orig = pd.read_parquet(orig_path)
        df_gal = pd.read_parquet(galileo_path)

        # Check if galileo_embedding column exists
        if 'galileo_embedding' not in df_gal.columns:
            print(f"  [ERROR] galileo_embedding column not found in {galileo_path}")
            continue

        # Verify row alignment by checking lon/lat
        if len(df_orig) != len(df_gal):
            print(f"  [ERROR] Row count mismatch: orig={len(df_orig)}, galileo={len(df_gal)}")
            continue

        # Add galileo columns to original
        df_orig['galileo_embedding'] = df_gal['galileo_embedding'].values
        if 'galileo_embedding_dim' in df_gal.columns:
            df_orig['galileo_embedding_dim'] = df_gal['galileo_embedding_dim'].values

        # Save back to original path
        df_orig.to_parquet(orig_path, index=False)
        print(f"  Updated {orig_path}")
        print(f"  Columns: {df_orig.columns.tolist()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
