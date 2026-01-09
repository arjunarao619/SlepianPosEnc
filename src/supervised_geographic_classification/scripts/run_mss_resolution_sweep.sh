#!/bin/bash
# Arctic MSS Resolution Sweep Experiment
# Sweeps through resolutions and neural network architectures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Data path (must be set by user)
DATA_PATH="${MSS_DATA_PATH:-/scratch/local/arra4944_images/drf/Experiment_data/Experiment_data}"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: MSS data not found at $DATA_PATH"
    echo "Set MSS_DATA_PATH environment variable to the correct path."
    exit 1
fi

# Output directories
RESULTS_DIR="$ROOT_DIR/results/mss_resolution_sweep"
mkdir -p "$RESULTS_DIR"

# Fixed parameters
N_RUNS=5
EPOCHS=200
BATCH_SIZE=2048
PATIENCE=30
NUM_WORKERS=16
LABEL_FRAC="1.0"
MAX_RADIUS=360

# Encoders to test (frequency-based only)
ENCODERS="grid,spherec,spherecplus,spherem,spheremplus,theory"

# Neural network architectures to test
ARCHS=("linear" "mlp" "resmlp" "siren" "glu")

# Resolution scales (min_radius values, from coarse to fine)
MIN_RADII=(5 1 0.5 0.1 0.05)

echo "=============================================="
echo "Arctic MSS Resolution Sweep Experiment"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Encoders: $ENCODERS"
echo "Architectures: ${ARCHS[*]}"
echo "Max radius: $MAX_RADIUS (fixed)"
echo "Min radii: ${MIN_RADII[*]}"
echo "Label fraction: $LABEL_FRAC"
echo "Runs per config: $N_RUNS"
echo "=============================================="
echo ""

for ARCH in "${ARCHS[@]}"; do
    echo ""
    echo "######################################################"
    echo "# Architecture: $ARCH"
    echo "######################################################"

    for MIN_R in "${MIN_RADII[@]}"; do
        echo ""
        echo "======================================================"
        echo "Arch: $ARCH | Resolution: min_radius=$MIN_R"
        echo "======================================================"

        CSV_PATH="$RESULTS_DIR/results_${ARCH}_minr${MIN_R}.csv"

        if ! python "$PARENT_DIR/train_mss_baselines.py" \
            --data-path "$DATA_PATH" \
            --encoders "$ENCODERS" \
            --arch "$ARCH" \
            --max-radius "$MAX_RADIUS" \
            --min-radius "$MIN_R" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --lr 1e-3 \
            --hidden-dim 128 \
            --dropout 0.1 \
            --num-workers "$NUM_WORKERS" \
            --label-fracs "$LABEL_FRAC" \
            --n-runs "$N_RUNS" \
            --csv-path "$CSV_PATH"; then
            echo "ERROR: Training failed for arch=$ARCH, min_radius=$MIN_R"
            echo "Re-running with full output:"
            python "$PARENT_DIR/train_mss_baselines.py" \
                --data-path "$DATA_PATH" \
                --encoders "$ENCODERS" \
                --arch "$ARCH" \
                --max-radius "$MAX_RADIUS" \
                --min-radius "$MIN_R" \
                --batch-size "$BATCH_SIZE" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                --lr 1e-3 \
                --hidden-dim 128 \
                --dropout 0.1 \
                --num-workers "$NUM_WORKERS" \
                --label-fracs "$LABEL_FRAC" \
                --n-runs "$N_RUNS" \
                --csv-path "$CSV_PATH"
            exit 1
        fi

        echo "Results saved to: $CSV_PATH"
    done
done

# Aggregate all results into a single file
echo ""
echo "=============================================="
echo "Aggregating results..."
echo "=============================================="

python -c "
import pandas as pd
from glob import glob
import os

results_dir = '$RESULTS_DIR'
csv_files = sorted(glob(os.path.join(results_dir, 'results_*.csv')))

if csv_files:
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)

    print('\nResults Summary (mean R² across runs):')
    print('=' * 90)

    # Summary by arch, encoder, min_radius
    summary = combined.groupby(['arch', 'encoder', 'min_radius'])['r2'].agg(['mean', 'std'])
    summary = summary.reset_index()
    summary['r2_str'] = summary.apply(lambda x: f\"{x['mean']:.4f}±{x['std']:.4f}\", axis=1)

    for arch in combined['arch'].unique():
        print(f'\n--- Architecture: {arch} ---')
        arch_data = summary[summary['arch'] == arch]
        pivot = arch_data.pivot(index='encoder', columns='min_radius', values='r2_str')
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        print(pivot.to_string())

    print(f'\nFull results saved to: {results_dir}/all_results.csv')
"

echo ""
echo "Done!"
