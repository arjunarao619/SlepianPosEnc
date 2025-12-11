#!/bin/bash
# Japan Prefecture Classification: Baseline encoder experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Output directories
RESULTS_DIR="$ROOT_DIR/results/japan"
FIGURES_DIR="$RESULTS_DIR/figures"
DATASET_DIR="$ROOT_DIR/data/japan"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

# Training parameters
ARCHS="mlp resmlp siren glu"
N_RUNS=3
EPOCHS=200
BATCH_SIZE=256
PATIENCE=50
NUM_WORKERS=8
SAMPLES_PER_PREF="1 2 5 10 50 100"

# Baseline encoders
ENCODERS="direct,cartesian3d,wrap,grid,spherec,spherecplus,spherem,spheremplus,theory"

echo "Japan Prefecture Baseline Experiments"
echo "======================================"

# Check dataset exists
if [ ! -f "$DATASET_DIR/metadata.json" ]; then
    echo "Error: Dataset not found. Run run_japan_experiments.sh first."
    exit 1
fi

for ARCH in $ARCHS; do
    echo "-> Baselines with arch=$ARCH"
    python "$PARENT_DIR/run_baselines_japanprefecture.py" \
        --encoders $ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
        --num-workers $NUM_WORKERS \
        --train-samples-per-prefecture $SAMPLES_PER_PREF \
        --n-runs $N_RUNS --seed 42 \
        --dataset-dir "$DATASET_DIR" \
        --csv-path "$RESULTS_DIR/baselines_${ARCH}.csv"
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
