#!/bin/bash
# OpenBuildings + AlphaEarth: Slepian vs SH experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$(dirname "$PARENT_DIR")"
ROOT_DIR="$(dirname "$SRC_DIR")"

# Data path (must be set by user)
DATA_DIR="${BUILDINGS_DATA_PATH:-/path/to/buildings/data}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Buildings data not found at $DATA_DIR"
    echo "Set BUILDINGS_DATA_PATH environment variable to the correct path."
    exit 1
fi

# Output directories
RESULTS_DIR="$ROOT_DIR/results/buildings"
FIGURES_DIR="$RESULTS_DIR/figures"
mkdir -p "$RESULTS_DIR" "$FIGURES_DIR"

# Regions and their cap radii
declare -A REGIONS
REGIONS["maharashtra"]="7.5"
REGIONS["dhaka"]="7.5"
REGIONS["mexicocity"]="7.5"

# Seeds
SEEDS=(123 456 789 1011 1213)

# Configurations to test
SH_CONFIGS=(10 40)
SLEPIAN_CONFIGS=(40 64 96 128)

# Training settings
EPOCHS=120
PATIENCE=20
BATCH_SIZE=512
LAMBDA_THRESH=0.02

# Total count
TOTAL=$(( ${#REGIONS[@]} * ${#SH_CONFIGS[@]} * ${#SLEPIAN_CONFIGS[@]} * ${#SEEDS[@]} ))
CURRENT=0

echo "OpenBuildings + AlphaEarth Experiments"
echo "======================================="
echo "Regions: ${!REGIONS[@]}"
echo "Total experiments: $TOTAL"
echo ""

for region in "${!REGIONS[@]}"; do
    cap_radius="${REGIONS[$region]}"

    for L_sh in "${SH_CONFIGS[@]}"; do
        for L_slep in "${SLEPIAN_CONFIGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                CURRENT=$((CURRENT + 1))

                echo "[${CURRENT}/${TOTAL}] ${region} | SH=${L_sh}, Slep=${L_slep} | seed=${seed}"

                python "$PARENT_DIR/train_slepian_vs_sh_multiscale.py" \
                    --data-dir "$DATA_DIR" \
                    --region "${region}" \
                    --L-sh ${L_sh} \
                    --L-slepian ${L_slep} \
                    --lambda-thresh ${LAMBDA_THRESH} \
                    --cap-radius-deg ${cap_radius} \
                    --epochs ${EPOCHS} \
                    --patience ${PATIENCE} \
                    --batch-size ${BATCH_SIZE} \
                    --csv-out "$RESULTS_DIR/${region}_sh${L_sh}_slep${L_slep}_seed${seed}.csv" \
                    --seed ${seed}

                echo "  Done"
                echo ""
            done
        done
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
