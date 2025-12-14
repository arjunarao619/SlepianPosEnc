#!/usr/bin/env bash
set -euo pipefail

DATA=/scratch/local/arra4944_images/ace/
ENCS=(time_copy monomial)
SEEDS=(55)  # Multiple seeds for robustness

echo "Running STE baseline experiments..."

for E in "${ENCS[@]}"; do
  for S in "${SEEDS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Temporal: $E, Seed: $S"
    echo "=================================================="
    
    # WITH regularization
    echo "Training WITH regularization..."
    python train_baselines.py \
      --data_path "$DATA" \
      --temporal_type "$E" \
      --use_reg \
      --seed "$S"

    # # WITHOUT regularization
    # echo "Training WITHOUT regularization..."
    # python train_baselines.py \
    #   --data_path "$DATA" \
    #   --temporal_type "$E" \
    #   --seed "$S"
    

  done
done

echo ""
echo "All experiments completed!"