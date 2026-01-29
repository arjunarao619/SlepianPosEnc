#!/bin/bash
# Japan Prefecture Classification: Baseline encoder experiments
#
# Runs Direct, Cartesian3D, Wrap encoders for MLP, ResMLP, SIREN, GLU
# (Linear is excluded - too few features for meaningful linear model)
# Note: Frequency-based encoders (Grid, SphereC, etc.) are in resolution_sweep

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
N_RUNS=5
EPOCHS=200
BATCH_SIZE=256
PATIENCE=50
NUM_WORKERS=8
SAMPLES_PER_PREF="50"  # Use 100 samples per prefecture for table results

# Simple encoders (Direct=2d, Cartesian3D=3d, Wrap=4d)
# These have too few features for Linear model (shows "--" in table)
ENCODERS="direct,cartesian3d,wrap"
ARCHS="linear"

echo "Japan Prefecture Baseline Experiments"
echo "======================================"
echo "Running: $ENCODERS"
echo "Architectures: $ARCHS"
echo ""

# Check dataset exists
if [ ! -f "$DATASET_DIR/metadata.json" ]; then
    echo "Error: Dataset not found. Run run_japan_experiments.sh first."
    exit 1
fi

for ARCH in $ARCHS; do
    echo "-> Running $ENCODERS with arch=$ARCH"
    python "$PARENT_DIR/run_baselines_japanprefecture.py" \
        --encoders $ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --hidden-dim 128 --dropout 0.1 \
        --num-workers $NUM_WORKERS \
        --train-samples-per-prefecture $SAMPLES_PER_PREF \
        --n-runs $N_RUNS --seed 42 \
        --dataset-dir "$DATASET_DIR" \
        --csv-path "$RESULTS_DIR/baselines_simple_${ARCH}.csv"
done

echo ""
echo "Aggregating results..."

# Aggregate results
python -c "
import pandas as pd
from glob import glob
import os

results_dir = '$RESULTS_DIR'
csv_files = glob(os.path.join(results_dir, 'baselines_simple_*.csv'))

if csv_files:
    print(f'Found {len(csv_files)} result files')
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    agg_path = os.path.join(results_dir, 'aggregated_results.csv')
    if os.path.exists(agg_path):
        existing = pd.read_csv(agg_path)
        # Avoid duplicates
        if 'method' in existing.columns and 'method' in df.columns:
            existing_keys = set(zip(existing['method'], existing['arch'], existing.get('seed', existing.get('run', range(len(existing))))))
            new_rows = []
            for _, row in df.iterrows():
                key = (row['method'], row['arch'], row.get('seed', row.get('run', 0)))
                if key not in existing_keys:
                    new_rows.append(row)
            if new_rows:
                df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
                print(f'Added {len(new_rows)} new results')
            else:
                df = existing
                print('No new results (all already exist)')
        else:
            df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(agg_path, index=False)
    print(f'Saved to {agg_path}')

    for f in csv_files:
        os.remove(f)
else:
    print('No result files found')
"

echo ""
echo "Done. Results: $RESULTS_DIR/aggregated_results.csv"
