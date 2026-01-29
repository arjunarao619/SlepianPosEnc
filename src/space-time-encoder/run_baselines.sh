#!/bin/bash
# Baseline temporal encoding sweep (no regularization)
#
# Usage: ./run_baselines.sh

set -e
cd "$(dirname "$0")"

mkdir -p logs checkpoints_baselines

DATA_PATH="/scratch/local/arra4944_images/ace/"
MAX_JOBS=4

echo "=============================================="
echo "Baseline Sweep (noreg only) - $MAX_JOBS parallel jobs"
echo "=============================================="
echo "Start: $(date)"
echo ""

# Configuration
TEMPORAL_TYPES=(no_time time_copy triangle monomial legendre fourier)
SEEDS=(42 123 456)

# Collect experiments
declare -a EXPERIMENTS

for TYPE in "${TEMPORAL_TYPES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="${TYPE}_noreg_seed${SEED}"

        # Skip if checkpoint exists
        if [ -f "checkpoints_baselines/best_${TAG}.pt" ]; then
            echo "Skipping $TAG (exists)"
            continue
        fi

        EXPERIMENTS+=("$TAG|$TYPE|$SEED")
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Experiments to run: $TOTAL"
echo ""

run_one() {
    local TAG="$1" TYPE="$2" SEED="$3"
    local LOG="logs/${TAG}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $TAG"

    python train_baselines.py \
        --temporal_type "$TYPE" \
        --seed "$SEED" \
        --data_path "$DATA_PATH" \
        > "$LOG" 2>&1

    if [ $? -eq 0 ]; then
        RMSE=$(grep "Test RMSE" "$LOG" | tail -1 | awk '{print $3}')
        echo "[$(date '+%H:%M:%S')] Done: $TAG -> RMSE=$RMSE"
        # Move checkpoint to baselines directory
        mv -f "best_${TAG}.pt" "checkpoints_baselines/" 2>/dev/null || true
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $TAG (see $LOG)"
    fi
}

# Run in batches
for ((i=0; i<TOTAL; i+=MAX_JOBS)); do
    BATCH_END=$((i + MAX_JOBS))
    [ $BATCH_END -gt $TOTAL ] && BATCH_END=$TOTAL

    echo "=== Batch: $((i+1))-$BATCH_END of $TOTAL ==="

    PIDS=()
    for ((j=i; j<BATCH_END; j++)); do
        IFS='|' read -r TAG TYPE SEED <<< "${EXPERIMENTS[$j]}"
        run_one "$TAG" "$TYPE" "$SEED" &
        PIDS+=($!)
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
done

echo ""
echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
echo ""
echo "Checkpoints: checkpoints_baselines/"
echo "Logs: logs/"
echo ""
echo "Run 'python test_ace.py' to evaluate all checkpoints"
