#!/usr/bin/env python3
"""
Aggregate experiment results and generate TikZ coordinate strings.

Usage:
    python scripts/aggregate_results_to_tikz.py --results-dir /scratch/local/arra4944_images/openbuildings/results
"""

import argparse
from pathlib import Path
import pandas as pd
import re


def parse_filename(filename: str) -> dict:
    """Parse experiment parameters from filename."""
    # Format 1: region_embedding_SH{L}_Slep{L}_seed{S}.csv (Shannon K)
    pattern1 = r"(.+)_(alphaearth|galileo)_SH(\d+)_Slep(\d+)_seed(\d+)\.csv"
    match = re.match(pattern1, filename)
    if match:
        return {
            "region": match.group(1),
            "embedding": match.group(2),
            "L_sh": int(match.group(3)),
            "L_slep": int(match.group(4)),
            "K": None,  # Shannon number, read from CSV
            "seed": int(match.group(5)),
        }

    # Format 2: region_embedding_SH{L}_Slep{L}_K{K}_seed{S}.csv (explicit K)
    pattern2 = r"(.+)_(alphaearth|galileo)_SH(\d+)_Slep(\d+)_K(\d+)_seed(\d+)\.csv"
    match = re.match(pattern2, filename)
    if match:
        return {
            "region": match.group(1),
            "embedding": match.group(2),
            "L_sh": int(match.group(3)),
            "L_slep": int(match.group(4)),
            "K": int(match.group(5)),
            "seed": int(match.group(6)),
        }
    return None


def format_tikz_coords(df: pd.DataFrame, col: str, precision: int = 4) -> str:
    """Format dataframe column as TikZ coordinates."""
    coords = []
    for _, row in df.iterrows():
        sigma = int(row["sigma_km"])
        val = row[col]
        coords.append(f"({sigma},{val:.{precision}f})")

    # Split into lines of 5 for readability
    lines = []
    for i in range(0, len(coords), 5):
        lines.append(" ".join(coords[i:i+5]))
    return "\n        ".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for TikZ code (default: print to stdout)")
    args = parser.parse_args()

    # Find all CSV files
    csv_files = list(args.results_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.results_dir}")
        return

    # Parse and load all results
    all_results = []
    for f in csv_files:
        params = parse_filename(f.name)
        if params is None:
            print(f"[SKIP] Could not parse: {f.name}")
            continue

        df = pd.read_csv(f)
        df["filename"] = f.name
        for k, v in params.items():
            df[k] = v
        all_results.append(df)

    if not all_results:
        print("No valid results found")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Group by region and embedding
    regions = sorted(combined["region"].unique())
    embeddings = sorted(combined["embedding"].unique())
    k_values = sorted(combined["K"].unique())

    output_lines = []
    output_lines.append("% Auto-generated TikZ coordinates from experiment results")
    output_lines.append(f"% Regions: {regions}")
    output_lines.append(f"% Embeddings: {embeddings}")
    output_lines.append(f"% K values: {k_values}")
    output_lines.append("")

    for region in regions:
        output_lines.append(f"% {'='*60}")
        output_lines.append(f"% {region.upper()}")
        output_lines.append(f"% {'='*60}")
        output_lines.append("")

        for emb in embeddings:
            output_lines.append(f"    % --- {emb.upper()} ---")

            # Filter data
            mask = (combined["region"] == region) & (combined["embedding"] == emb)
            subset = combined[mask]

            if subset.empty:
                output_lines.append(f"    % No data for {region}/{emb}")
                output_lines.append("")
                continue

            # Get column names based on embedding type
            emb_upper = emb.upper()
            ae_col = f"R2_{emb_upper}"

            # Find the baseline (first K value or any - they should all have same baseline)
            baseline_df = subset[subset["K"] == k_values[0]][["sigma_km", ae_col]].drop_duplicates()
            baseline_df = baseline_df.sort_values("sigma_km")

            # AE-only baseline (beige dashed)
            output_lines.append(f"    % {emb_upper}-only baseline (beige dashed)")
            output_lines.append(f"    \\addplot[beige, dashed, line width=1.5pt] coordinates {{")
            output_lines.append(f"        {format_tikz_coords(baseline_df, ae_col)}")
            output_lines.append(f"    }};")
            output_lines.append("")

            # SH line (same for all K, take first)
            sh_df = subset[subset["K"] == k_values[0]][["sigma_km", "R2_SH"]].drop_duplicates()
            sh_df = sh_df.sort_values("sigma_km")
            output_lines.append(f"    % {emb_upper}+SH L=64 (SHred)")
            output_lines.append(f"    \\addplot[SHred, mark=square, line width=1.2pt, mark size=1.5pt] coordinates {{")
            output_lines.append(f"        {format_tikz_coords(sh_df, 'R2_SH')}")
            output_lines.append(f"    }};")
            output_lines.append("")

            # Slepian lines for each K
            purple_styles = [
                ("slepianpurple1", "*", "1.5pt"),
                ("slepianpurple2", "triangle", "2.5pt"),
                ("slepianpurple3", "diamond", "2.5pt"),
                ("slepianpurple4", "pentagon", "2.5pt"),
            ]

            for i, K in enumerate(k_values):
                if i >= len(purple_styles):
                    break

                style, mark, size = purple_styles[i]
                slep_df = subset[subset["K"] == K][["sigma_km", "R2_Slepian"]].drop_duplicates()
                slep_df = slep_df.sort_values("sigma_km")

                if slep_df.empty:
                    continue

                output_lines.append(f"    % {emb_upper}+Slepian K={K} ({style})")
                output_lines.append(f"    \\addplot[{style}, mark={mark}, line width=1.2pt, mark size={size}] coordinates {{")
                output_lines.append(f"        {format_tikz_coords(slep_df, 'R2_Slepian')}")
                output_lines.append(f"    }};")
                output_lines.append("")

        output_lines.append("")

    # Output
    output_text = "\n".join(output_lines)

    if args.output:
        args.output.write_text(output_text)
        print(f"Saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
