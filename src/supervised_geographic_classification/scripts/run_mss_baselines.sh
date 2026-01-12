#!/bin/bash
# Arctic MSS Reconstruction: Baseline encoder experiments
#
# This script completes the missing baseline experiments:
# 1. Direct, Cartesian3D, Wrap - for MLP, ResMLP, SIREN, GLU (Linear shows "--")
# 2. Grid, SphereC, SphereC+, SphereM, SphereM+, Theory - for SIREN and GLU only
#    (Linear, MLP, ResMLP already completed in resolution sweep)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Data path
DATA_PATH="${MSS_DATA_PATH:-/scratch/local/arra4944_images/drf/Experiment_data/Experiment_data}"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: MSS data not found at $DATA_PATH"
    echo "Set MSS_DATA_PATH environment variable to the correct path."
    exit 1
fi

# Output directories
RESULTS_DIR="$ROOT_DIR/results/mss"
FIGURES_DIR="$RESULTS_DIR/figures"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

# Training parameters
N_RUNS=5
EPOCHS=200
BATCH_SIZE=2048
PATIENCE=30
NUM_WORKERS=16
LABEL_FRACS="1.0"

echo "Arctic MSS Baseline Experiments"
echo "================================"
echo ""

# -----------------------------------------------------------------------------
# Part 1: Direct, Cartesian3D, Wrap encoders
# Run on all architectures including Linear
# -----------------------------------------------------------------------------
SIMPLE_ENCODERS="direct,cartesian3d,wrap"
SIMPLE_ARCHS="linear mlp resmlp siren glu"

echo "Part 1: Simple encoders (Direct, Cartesian3D, Wrap)"
echo "----------------------------------------------------"

for ARCH in $SIMPLE_ARCHS; do
    echo "-> Running $SIMPLE_ENCODERS with arch=$ARCH"
    python "$PARENT_DIR/train_mss_baselines.py" \
        --data-path "$DATA_PATH" \
        --encoders $SIMPLE_ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
        --num-workers $NUM_WORKERS \
        --label-fracs $LABEL_FRACS --n-runs $N_RUNS --seed 42 \
        --csv-path "$RESULTS_DIR/baselines_simple_${ARCH}.csv" \
        --fig-dir "$FIGURES_DIR/baselines_simple_${ARCH}"
done

echo ""

# -----------------------------------------------------------------------------
# Part 2: Frequency-based encoders (Grid, SphereC, etc.) for SIREN and GLU
# Linear, MLP, ResMLP already completed in resolution sweep experiments
# -----------------------------------------------------------------------------
FREQ_ENCODERS="grid,spherec,spherecplus,spherem,spheremplus,theory"
FREQ_ARCHS="siren glu"

echo "Part 2: Frequency encoders for SIREN and GLU"
echo "---------------------------------------------"

for ARCH in $FREQ_ARCHS; do
    echo "-> Running $FREQ_ENCODERS with arch=$ARCH"
    python "$PARENT_DIR/train_mss_baselines.py" \
        --data-path "$DATA_PATH" \
        --encoders $FREQ_ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
        --num-workers $NUM_WORKERS \
        --label-fracs $LABEL_FRACS --n-runs $N_RUNS --seed 42 \
        --csv-path "$RESULTS_DIR/baselines_freq_${ARCH}.csv" \
        --fig-dir "$FIGURES_DIR/baselines_freq_${ARCH}"
done

echo ""
echo "Aggregating results..."
echo "----------------------"

# Aggregate results
python -c "
import pandas as pd
from glob import glob
import os

results_dir = '$RESULTS_DIR'

# Find all baseline result files (both simple and freq)
csv_files = glob(os.path.join(results_dir, 'baselines_simple_*.csv'))
csv_files += glob(os.path.join(results_dir, 'baselines_freq_*.csv'))

if csv_files:
    print(f'Found {len(csv_files)} result files to aggregate')
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    agg_path = os.path.join(results_dir, 'aggregated_results.csv')
    if os.path.exists(agg_path):
        existing = pd.read_csv(agg_path)
        # Avoid duplicates by checking method+arch+seed combinations
        existing_keys = set(zip(existing['method'], existing['arch'], existing['seed']))
        new_rows = []
        for _, row in df.iterrows():
            key = (row['method'], row['arch'], row['seed'])
            if key not in existing_keys:
                new_rows.append(row)
        if new_rows:
            df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            print(f'Added {len(new_rows)} new results')
        else:
            df = existing
            print('No new results to add (all already exist)')
    else:
        print(f'Creating new aggregated_results.csv with {len(df)} results')

    df.to_csv(agg_path, index=False)
    print(f'Saved to {agg_path}')

    # Clean up individual files
    for f in csv_files:
        os.remove(f)
        print(f'Removed {os.path.basename(f)}')
else:
    print('No result files found to aggregate')
"

echo ""
echo "============================================"
echo "Done! Results saved to: $RESULTS_DIR/aggregated_results.csv"
echo ""
echo "Experiments completed:"
echo "  - Direct, Cartesian3D, Wrap: Linear, MLP, ResMLP, SIREN, GLU"
echo "  - Grid, SphereC, SphereC+, SphereM, SphereM+, Theory: SIREN, GLU"
echo "============================================"
