#!/bin/bash
# Comprehensive baseline sweep for Space-Time Encoder
# Runs 4 experiments in parallel, waits for completion, then next batch
#
# Usage: ./run_baselines.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs checkpoints

DATA_PATH="/scratch/local/arra4944_images/ace/"
MAX_JOBS=4

echo "=============================================="
echo "Baseline Sweep - 4 jobs in parallel"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Configuration
TEMPORAL_TYPES=(no_time time_copy triangle monomial legendre fourier)
ORTHO_WEIGHTS=(0 0.001 0.1 1.0)  # Paper uses α=1
SEEDS=(42 123 456)

# Collect all experiment configs
declare -a EXPERIMENTS

for TYPE in "${TEMPORAL_TYPES[@]}"; do
    for ALPHA in "${ORTHO_WEIGHTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Create tag based on alpha
            if [ "$ALPHA" == "0" ]; then
                TAG="${TYPE}_noreg_seed${SEED}"
                ALPHA_ARG=""
            else
                TAG="${TYPE}_alpha${ALPHA}_seed${SEED}"
                ALPHA_ARG="--ortho_weight $ALPHA"
            fi

            # Skip if already done
            if [ -f "checkpoints/best_${TAG}.pt" ] || [ -f "best_${TAG}.pt" ]; then
                echo "Skipping $TAG (checkpoint exists)"
                continue
            fi

            EXPERIMENTS+=("$TAG|$TYPE|$ALPHA_ARG|$SEED")
        done
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Total experiments to run: $TOTAL"
echo ""

# Function to run single experiment
run_one() {
    local TAG="$1"
    local TYPE="$2"
    local ALPHA_ARG="$3"
    local SEED="$4"
    local LOGFILE="logs/${TAG}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $TAG"

    # Run with explicit arguments
    python train_baselines.py \
        --temporal_type "$TYPE" \
        --seed "$SEED" \
        --data_path "$DATA_PATH" \
        $ALPHA_ARG \
        > "$LOGFILE" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        # Extract final RMSE from log
        RMSE=$(grep "Test RMSE" "$LOGFILE" | tail -1 | awk '{print $3}')
        echo "[$(date '+%H:%M:%S')] Done: $TAG -> RMSE=$RMSE"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $TAG (see $LOGFILE)"
    fi
    return $EXIT_CODE
}

# Run in batches of MAX_JOBS
BATCH_NUM=0
for ((i=0; i<TOTAL; i+=MAX_JOBS)); do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_END=$((i + MAX_JOBS))
    if [ $BATCH_END -gt $TOTAL ]; then
        BATCH_END=$TOTAL
    fi

    echo ""
    echo "=== Batch $BATCH_NUM: experiments $((i+1))-$BATCH_END of $TOTAL ==="

    PIDS=()
    for ((j=i; j<BATCH_END; j++)); do
        # Parse experiment config
        IFS='|' read -r TAG TYPE ALPHA_ARG SEED <<< "${EXPERIMENTS[$j]}"

        run_one "$TAG" "$TYPE" "$ALPHA_ARG" "$SEED" &
        PIDS+=($!)
    done

    # Wait for batch to complete
    echo "Waiting for batch $BATCH_NUM to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "Batch $BATCH_NUM complete."
done

echo ""
echo "=============================================="
echo "Sweep Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""

# Aggregate results
echo "Aggregating results..."
python -c "
import os
import re
import glob

results = []
for logfile in sorted(glob.glob('logs/*.log')):
    name = os.path.basename(logfile).replace('.log', '')

    # Parse name: {type}_{reg_info}_seed{seed}
    # Examples: legendre_noreg_seed42, fourier_alpha1.0_seed123

    with open(logfile) as f:
        content = f.read()

    # Extract test RMSE
    match = re.search(r'Test RMSE:\s*([\d.]+)', content)
    if match:
        rmse = float(match.group(1))

        # Parse components from filename
        parts = name.rsplit('_seed', 1)
        if len(parts) == 2:
            type_reg = parts[0]
            seed = parts[1]

            if '_noreg' in type_reg:
                ttype = type_reg.replace('_noreg', '')
                alpha = 0.0
            elif '_alpha' in type_reg:
                ttype, alpha_str = type_reg.rsplit('_alpha', 1)
                alpha = float(alpha_str)
            elif '_reg' in type_reg:
                ttype = type_reg.replace('_reg', '')
                alpha = 1.0  # default reg
            else:
                ttype = type_reg
                alpha = 0.0

            results.append({
                'type': ttype,
                'alpha': alpha,
                'seed': seed,
                'rmse': rmse,
                'name': name
            })

if not results:
    print('No results found!')
else:
    # Sort by RMSE
    results.sort(key=lambda x: x['rmse'])

    print()
    print('='*70)
    print('RESULTS (sorted by RMSE)')
    print('='*70)
    print(f'{\"Type\":<12} {\"Alpha\":<8} {\"Seed\":<6} {\"RMSE\":<10}')
    print('-'*40)
    for r in results:
        print(f'{r[\"type\"]:<12} {r[\"alpha\"]:<8} {r[\"seed\"]:<6} {r[\"rmse\"]:<10.4f}')

    # Summary by type and alpha
    print()
    print('='*70)
    print('SUMMARY BY TYPE AND ALPHA (mean RMSE)')
    print('='*70)
    from collections import defaultdict
    summary = defaultdict(list)
    for r in results:
        key = (r['type'], r['alpha'])
        summary[key].append(r['rmse'])

    print(f'{\"Type\":<12} {\"Alpha\":<8} {\"Mean RMSE\":<10} {\"Std\":<8} {\"N\":<4}')
    print('-'*50)
    for (ttype, alpha), rmses in sorted(summary.items()):
        import statistics
        mean = statistics.mean(rmses)
        std = statistics.stdev(rmses) if len(rmses) > 1 else 0
        print(f'{ttype:<12} {alpha:<8} {mean:<10.4f} {std:<8.4f} {len(rmses):<4}')

    # Best overall
    best = results[0]
    print()
    print(f'Best: {best[\"name\"]} with RMSE={best[\"rmse\"]:.4f}')
    print()
    print('Paper targets:')
    print('  Legendre (α=0): 1.63')
    print('  Legendre (α=1): 1.41  <-- TARGET')
    print('  Fourier  (α=1): 1.48')
"

echo ""
echo "Done! Check logs/ for individual run details."
