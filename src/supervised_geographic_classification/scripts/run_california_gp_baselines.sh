#!/bin/bash
# Run all GP baselines on California Housing dataset

# Don't exit on error - continue to next method
set +e

# Setup environment
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parameters
FRACS="0.01,0.05,0.10,0.25,0.50,1.00"
RESULTS_DIR="$SCRIPT_DIR/results/gp_baselines"
N_RUNS=3
SEED=42

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "California Housing GP Baselines"
echo "Label fractions: $FRACS"
echo "Results: $RESULTS_DIR"
echo "========================================"

# 1. SVGP
echo ""
echo "[1/3] SVGP Matern-5/2"
python train_gp_baselines.py \
    --method svgp \
    --dataset california_housing \
    --kernel matern52 \
    --num-inducing 500 \
    --num-epochs 50 \
    --batch-size 1024 \
    --lr 0.01 \
    --label-fracs "$FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_svgp_matern52.csv"

# 2. Planar RFF
echo ""
echo "[2/3] Planar RFF Matern-5/2"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset california_housing \
    --kernel matern52 \
    --num-features 2000 \
    --label-fracs "$FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_planar_rff_matern52.csv"

# 3. Exact GP (only small fractions - O(n^3) complexity)
echo ""
echo "[3/3] Exact GP Matern-5/2 (small fractions only)"
python train_gp_baselines.py \
    --method exact_gp \
    --dataset california_housing \
    --kernel matern52 \
    --num-iterations 100 \
    --lr 0.1 \
    --label-fracs "0.01,0.05,0.10" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_exact_gp_matern52.csv"

echo ""
echo "========================================"
echo "All California GP Baselines Complete!"
echo "========================================"
echo "Results:"
ls -la "$RESULTS_DIR"/california_*.csv
