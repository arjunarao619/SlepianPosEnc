#!/bin/bash
# Arctic MSS Reconstruction: Baseline encoder experiments

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
ARCHS="linear mlp resmlp siren glu"
N_RUNS=5
EPOCHS=200
BATCH_SIZE=2048
PATIENCE=30
NUM_WORKERS=16
LABEL_FRACS="1.0"

# Baseline encoders
ENCODERS="direct,cartesian3d,wrap"

echo "Arctic MSS Baseline Experiments"
echo "================================"

for ARCH in $ARCHS; do
    echo "-> Baselines with arch=$ARCH"
    python "$PARENT_DIR/train_mss_baselines.py" \
        --data-path "$DATA_PATH" \
        --encoders $ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
        --num-workers $NUM_WORKERS \
        --label-fracs $LABEL_FRACS --n-runs $N_RUNS --seed 42 \
        --csv-path "$RESULTS_DIR/baselines_${ARCH}.csv" \
        --fig-dir "$FIGURES_DIR/baselines_${ARCH}"
done

# Aggregate results
python -c "
import pandas as pd
from glob import glob
import os

results_dir = '$RESULTS_DIR'
csv_files = glob(os.path.join(results_dir, 'baselines_*.csv'))

if csv_files:
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    agg_path = os.path.join(results_dir, 'aggregated_results.csv')
    if os.path.exists(agg_path):
        existing = pd.read_csv(agg_path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(agg_path, index=False)
    print(f'Updated aggregated_results.csv with baselines')

    for f in csv_files:
        os.remove(f)
"

echo "Done. Results: $RESULTS_DIR/aggregated_results.csv"
