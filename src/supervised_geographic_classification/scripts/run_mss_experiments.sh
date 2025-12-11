#!/bin/bash
# Arctic MSS Reconstruction: Slepian and Vanilla SH experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Data path (must be set by user)
DATA_PATH="${MSS_DATA_PATH:-/path/to/mss/data}"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: MSS data not found at $DATA_PATH"
    echo "Set MSS_DATA_PATH environment variable to the correct path."
    exit 1
fi

# Output directories
RESULTS_DIR="$ROOT_DIR/results/mss"
CACHE_DIR="$ROOT_DIR/cache/mss"
FIGURES_DIR="$RESULTS_DIR/figures"
mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$FIGURES_DIR"

# Training parameters
ARCHS="mlp resmlp siren glu"
N_RUNS=5
EPOCHS=200
BATCH_SIZE=2048
PATIENCE=30
NUM_WORKERS=16
LABEL_FRACS="0.01 0.05 0.10 0.25 0.50 1.0"

# Slepian parameters
L_GLOBAL=10
CAP_RADIUS=25.0
LAMBDA_THRESH=0.05

echo "Arctic MSS Reconstruction Experiments"
echo "======================================"

# Slepian experiments
for ARCH in $ARCHS; do
    for L in 40 64 80 120; do
        NUM_MODES=$(python -c "
import numpy as np
theta_rad = $CAP_RADIUS * np.pi / 180.0
shannon = int(($L + 1)**2 * (1 - np.cos(theta_rad)) / 2)
print(min(shannon, ($L + 1)**2))
")
        echo "-> Slepian L=$L, arch=$ARCH (modes=$NUM_MODES)"
        python "$PARENT_DIR/train_mss_slepian.py" \
            --data-path "$DATA_PATH" \
            --L-global $L_GLOBAL --L-slepian $L \
            --cap-radius $CAP_RADIUS --num-modes $NUM_MODES \
            --lambda-thresh $LAMBDA_THRESH \
            --arch $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS \
            --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
            --num-workers $NUM_WORKERS \
            --label-fracs $LABEL_FRACS --n-runs $N_RUNS --seed 42 \
            --cache-dir "$CACHE_DIR" \
            --csv-path "$RESULTS_DIR/slepian_L${L}_${ARCH}.csv" \
            --fig-dir "$FIGURES_DIR/slepian_L${L}_${ARCH}"
    done
done

# Vanilla SH experiments
for ARCH in $ARCHS; do
    for L in 10 20 30 40; do
        echo "-> Vanilla SH L=$L, arch=$ARCH"
        python "$PARENT_DIR/train_mss_sh_vanilla.py" \
            --data-path "$DATA_PATH" \
            --L $L --arch $ARCH \
            --batch-size $BATCH_SIZE --epochs $EPOCHS \
            --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
            --num-workers $NUM_WORKERS \
            --label-fracs $LABEL_FRACS --n-runs $N_RUNS --seed 42 \
            --csv-path "$RESULTS_DIR/vanilla_sh_L${L}_${ARCH}.csv" \
            --fig-dir "$FIGURES_DIR/vanilla_sh_L${L}_${ARCH}"
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
