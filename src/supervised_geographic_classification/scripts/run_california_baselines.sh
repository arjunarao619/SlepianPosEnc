#!/bin/bash
# California Housing Regression: Baseline encoder experiments
#
# Runs Direct, Cartesian3D, Wrap encoders for MLP, ResMLP, SIREN, GLU
# (Linear is excluded - too few features for meaningful linear model)

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
N_RUNS=5
EPOCHS=500
BATCH_SIZE=512
PATIENCE=50
NUM_WORKERS=8
LABEL_FRACS="1.0"

# Simple encoders (Direct=2d, Cartesian3D=3d, Wrap=4d)
ENCODERS="direct,cartesian3d,wrap"
ARCHS="linear"

echo "California Housing Baseline Experiments"
echo "======================================="
echo "Running: $ENCODERS"
echo "Architectures: $ARCHS"
echo ""

for ARCH in $ARCHS; do
    echo "-> Running $ENCODERS with arch=$ARCH"
    python "$PARENT_DIR/run_baselines_californiahousing.py" \
        --encoders $ENCODERS --arch $ARCH \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --patience $PATIENCE --lr 1e-3 --num-workers $NUM_WORKERS \
        --label-fracs $LABEL_FRACS --n-runs $N_RUNS \
        --csv-path "$RESULTS_DIR/baselines_simple_${ARCH}.csv" \
        --fig-dir "$FIGURES_DIR/baselines_simple_${ARCH}"
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
