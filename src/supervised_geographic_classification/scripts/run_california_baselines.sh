#!/bin/bash
# California Housing Regression: Baseline encoder experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Output directories
RESULTS_DIR="$ROOT_DIR/results/california"
FIGURES_DIR="$RESULTS_DIR/figures"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

# Training parameters
ARCHS="mlp resmlp siren glu"
N_RUNS=5
EPOCHS=500
BATCH_SIZE=512
PATIENCE=50
NUM_WORKERS=8
LABEL_FRACS="0.01 0.05 0.10 0.25 0.50 1.0"

# Baseline encoders
ENCODERS="direct cartesian3d wrap grid spherec spherecplus spherem spheremplus theory"

echo "California Housing Baseline Experiments"
echo "======================================="

for ARCH in $ARCHS; do
    echo "-> Baselines with arch=$ARCH"
    python "$PARENT_DIR/run_baselines_californiahousing.py" \
        --encoders $ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --num-workers $NUM_WORKERS \
        --label-fracs $LABEL_FRACS --n-runs $N_RUNS \
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

    # Append to existing aggregated file or create new
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
