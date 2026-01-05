#!/bin/bash
# California Housing Regression: Slepian and Vanilla SH experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Output directories (unified at project root)
RESULTS_DIR="$ROOT_DIR/results/california"
CACHE_DIR="$ROOT_DIR/cache/california"
FIGURES_DIR="$RESULTS_DIR/figures"
mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$FIGURES_DIR"

# Training parameters
# ARCHS="mlp resmlp siren glu"
ARCHS="mlp"
N_RUNS=1
EPOCHS=500
BATCH_SIZE=512
PATIENCE=50
NUM_WORKERS=8
LABEL_FRACS="1.0"

# Slepian parameters
L_GLOBAL=10
CAP_RADIUS=7.5
LAMBDA_THRESH=0.05

echo "California Housing Experiments"
echo "=============================="

for ARCH in $ARCHS; do
    for L in 10 20 30 40 50 60 70 80 90 100 110 120; do
    # for L in 10; do

        echo "-> Slepian L=$L, arch=$ARCH"
        python "$PARENT_DIR/cached_slepian_demo_cali.py" \
            --L-global $L_GLOBAL --L-slepian $L \
            --cap-radius $CAP_RADIUS --num-modes 800 --lambda-thresh $LAMBDA_THRESH \
            --arch $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS \
            --patience $PATIENCE --num-workers $NUM_WORKERS \
            --label-fracs $LABEL_FRACS --n-runs $N_RUNS \
            --cache-path "$CACHE_DIR/slepian_L${L}.pt" \
            --csv-path "$RESULTS_DIR/slepian_L${L}_${ARCH}.csv" \
            --fig-dir "$FIGURES_DIR/slepian_L${L}_${ARCH}"
    done
done

# # Vanilla SH experiments
# for ARCH in $ARCHS; do
#     for L in 10 15 20 25 30 35 40 45 50 55 60; do
#         echo "-> Vanilla SH L=$L, arch=$ARCH"
#         python "$PARENT_DIR/train_california_sh_vanilla.py" \
#             --L $L --arch $ARCH --batch-size $BATCH_SIZE --epochs $EPOCHS \
#             --patience $PATIENCE --num-workers $NUM_WORKERS \
#             --label-fracs $LABEL_FRACS --n-runs $N_RUNS \
#             --csv-path "$RESULTS_DIR/vanilla_sh_L${L}_${ARCH}.csv" \
#             --fig-dir "$FIGURES_DIR/vanilla_sh_L${L}_${ARCH}"
#     done
# done

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

    # Clean up individual files
    for f in csv_files:
        os.remove(f)
    print('Removed individual CSV files')
"

echo "Done. Results: $RESULTS_DIR/aggregated_results.csv"
