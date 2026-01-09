#!/bin/bash
# Japan Prefecture Classification: Resolution Sweep Experiment
# Sweeps through resolutions and neural network architectures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Output directories
RESULTS_DIR="$ROOT_DIR/results/japan_resolution_sweep"
DATASET_DIR="$ROOT_DIR/data/japan"
mkdir -p "$RESULTS_DIR"

# Fixed parameters
N_RUNS=5
EPOCHS=200
BATCH_SIZE=256
PATIENCE=50
NUM_WORKERS=8
MAX_RADIUS=360

# Use a fixed number of samples per prefecture for resolution comparison
SAMPLES_PER_PREF="50"

# Encoders to test (frequency-based only)
ENCODERS="grid,spherec,spherecplus,spherem,spheremplus,theory"

# Neural network architectures to test
ARCHS=("linear" "mlp" "resmlp" "siren" "glu")

# Resolution scales (min_radius values, from coarse to fine)
MIN_RADII=(5 1 0.5 0.1 0.05)

echo "=============================================="
echo "Japan Prefecture Resolution Sweep Experiment"
echo "=============================================="
echo "Encoders: $ENCODERS"
echo "Architectures: ${ARCHS[*]}"
echo "Max radius: $MAX_RADIUS (fixed)"
echo "Min radii: ${MIN_RADII[*]}"
echo "Samples per prefecture: $SAMPLES_PER_PREF"
echo "Runs per config: $N_RUNS"
echo "=============================================="

# Check dataset exists
if [ ! -f "$DATASET_DIR/metadata.json" ]; then
    echo "Error: Dataset not found at $DATASET_DIR"
    echo "Run run_japan_baselines.sh first to generate the dataset."
    exit 1
fi

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

        if ! python "$PARENT_DIR/run_baselines_japanprefecture.py" \
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
            --train-samples-per-prefecture "$SAMPLES_PER_PREF" \
            --n-runs "$N_RUNS" \
            --dataset-dir "$DATASET_DIR" \
            --csv-path "$CSV_PATH"; then
            echo "ERROR: Training failed for arch=$ARCH, min_radius=$MIN_R"
            echo "Re-running with full output:"
            python "$PARENT_DIR/run_baselines_japanprefecture.py" \
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
                --train-samples-per-prefecture "$SAMPLES_PER_PREF" \
                --n-runs "$N_RUNS" \
                --dataset-dir "$DATASET_DIR" \
                --csv-path "$CSV_PATH"
            exit 1
        fi

        echo "Results saved to: $CSV_PATH"
    done
done

# Aggregate all results
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

    print('\\nAccuracy Summary (mean ± std across runs):')
    print('=' * 90)

    # Summary by arch, encoder, min_radius
    summary = combined.groupby(['arch', 'encoder', 'min_radius'])['accuracy'].agg(['mean', 'std'])
    summary = summary.reset_index()
    summary['acc_str'] = summary.apply(lambda x: f\"{x['mean']*100:.1f}%±{x['std']*100:.1f}%\", axis=1)

    for arch in combined['arch'].unique():
        print(f'\\n--- Architecture: {arch} ---')
        arch_data = summary[summary['arch'] == arch]
        pivot = arch_data.pivot(index='encoder', columns='min_radius', values='acc_str')
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        print(pivot.to_string())

    print(f'\\nFull results: {results_dir}/all_results.csv')
"

echo ""
echo "Done!"
