#!/bin/bash
# Japan Prefecture Classification: Slepian and Vanilla SH experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Output directories
RESULTS_DIR="$ROOT_DIR/results/japan"
CACHE_DIR="$ROOT_DIR/cache/japan"
FIGURES_DIR="$RESULTS_DIR/figures"
DATASET_DIR="$ROOT_DIR/data/japan"
mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$FIGURES_DIR" "$DATASET_DIR"

# Training parameters
ARCHS="linear mlp resmlp siren glu"
N_RUNS=5
EPOCHS=200
BATCH_SIZE=256
PATIENCE=50
NUM_WORKERS=8
SAMPLES_PER_PREF="50"

# Slepian parameters
L_GLOBAL=10
CAP_RADIUS=10.0
LAMBDA_THRESH=0.05

echo "Japan Prefecture Classification Experiments"
echo "============================================"

# Generate dataset if not exists
if [ ! -f "$DATASET_DIR/metadata.json" ]; then
    echo "-> Generating dataset..."
    python "$PARENT_DIR/japan_prefecture_slepian.py" \
        --generate --dataset-dir "$DATASET_DIR" \
        --samples-per-prefecture 100 --seed 42
fi

# Slepian experiments
for ARCH in $ARCHS; do
    for L in 40 80 120; do
        NUM_MODES=$(python -c "
import numpy as np
alpha = np.radians($CAP_RADIUS)
shannon = (1.0 - np.cos(alpha)) / 2.0 * ($L + 1)**2
print(min(int(2 * shannon), ($L + 1)**2))
")
        echo "-> Slepian L=$L, arch=$ARCH (modes=$NUM_MODES)"
        python "$PARENT_DIR/japan_prefecture_slepian.py" \
            --method slepian --arch $ARCH \
            --L-global $L_GLOBAL --L-slepian $L \
            --cap-radius $CAP_RADIUS --num-modes $NUM_MODES \
            --lambda-thresh $LAMBDA_THRESH \
            --batch-size $BATCH_SIZE --epochs $EPOCHS \
            --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
            --num-workers $NUM_WORKERS \
            --train-samples-per-prefecture $SAMPLES_PER_PREF \
            --n-runs $N_RUNS --seed 42 \
            --dataset-dir "$DATASET_DIR" \
            --cache-path "$CACHE_DIR/slepian_L${L}.pt" \
            --csv-path "$RESULTS_DIR/slepian_L${L}_${ARCH}.csv"
    done
done

# Vanilla SH experiments
for ARCH in $ARCHS; do
    for L in 10 40; do
        echo "-> Vanilla SH L=$L, arch=$ARCH"
        python "$PARENT_DIR/japan_prefecture_slepian.py" \
            --method vanilla_sh --arch $ARCH --L-slepian $L \
            --batch-size $BATCH_SIZE --epochs $EPOCHS \
            --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
            --num-workers $NUM_WORKERS \
            --train-samples-per-prefecture $SAMPLES_PER_PREF \
            --n-runs $N_RUNS --seed 42 \
            --dataset-dir "$DATASET_DIR" \
            --cache-path "$CACHE_DIR/vanilla_L${L}.pt" \
            --csv-path "$RESULTS_DIR/vanilla_sh_L${L}_${ARCH}.csv"
    done
done

# Aggregate results
python -c "
import pandas as pd
from glob import glob
import os

results_dir = '$RESULTS_DIR'
csv_files = glob(os.path.join(results_dir, '*.csv'))
csv_files = [f for f in csv_files if 'aggregated' not in f]

if csv_files:
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df.to_csv(os.path.join(results_dir, 'aggregated_results.csv'), index=False)
    print(f'Aggregated {len(csv_files)} files -> aggregated_results.csv')

    for f in csv_files:
        os.remove(f)
"

echo "Done. Results: $RESULTS_DIR/aggregated_results.csv"
