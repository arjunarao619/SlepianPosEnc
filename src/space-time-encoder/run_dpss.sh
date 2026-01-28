#!/bin/bash
# DPSS temporal encoding sweep (optimized, no regularization)
#
# Usage: ./run_dpss.sh

set -e
cd "$(dirname "$0")"

mkdir -p logs checkpoints

DATA_PATH="/scratch/local/arra4944_images/ace/"

echo "=============================================="
echo "DPSS Sweep (optimized, noreg)"
echo "=============================================="
echo "Start: $(date)"
echo ""

# Configuration
NW_VALUES=(10 15 20 25 30 35)
SEEDS=(42 123 456)

# Count experiments
declare -a EXPERIMENTS

for NW in "${NW_VALUES[@]}"; do
    K=$((2 * NW - 1))  # Shannon number
    for SEED in "${SEEDS[@]}"; do
        TAG="dpss_NW${NW}_K${K}_opt_seed${SEED}"

        # Skip if checkpoint exists
        if [ -f "checkpoints/best_${TAG}.pt" ]; then
            echo "Skipping $TAG (exists)"
            continue
        fi

        EXPERIMENTS+=("$TAG|$NW|$K|$SEED")
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "Experiments to run: $TOTAL"
echo ""

# Run sequentially (DPSS training is memory-intensive)
for ((i=0; i<TOTAL; i++)); do
    IFS='|' read -r TAG NW K SEED <<< "${EXPERIMENTS[$i]}"
    LOG="logs/${TAG}.log"

    echo "[$((i+1))/$TOTAL] Running: $TAG"

    python train_dpss_sweep.py \
        --nw "$NW" \
        --k "$K" \
        --seed "$SEED" \
        --optimized \
        --data-path "$DATA_PATH" \
        --output-dir checkpoints \
        --ortho-weight 0.0 \
        --ortho-weight-space 0.0 \
        --ortho-weight-time 0.0 \
        --time-grad-weight 0.0 \
        2>&1 | tee "$LOG"

    if [ $? -eq 0 ]; then
        RMSE=$(grep "Test RMSE" "$LOG" | tail -1 | awk '{print $3}')
        echo "Done: $TAG -> RMSE=$RMSE"
    else
        echo "FAILED: $TAG"
    fi
    echo ""
done

echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
echo ""
echo "Checkpoints: checkpoints/"
echo "Logs: logs/"
echo ""
echo "Run 'python test_ace.py' to evaluate all checkpoints"
