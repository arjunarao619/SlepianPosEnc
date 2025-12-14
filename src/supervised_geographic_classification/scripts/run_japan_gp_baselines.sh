#!/bin/bash
# Run all GP baselines on Japan Prefecture dataset

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
SAMPLES_PER_CLASS="1,2,5,10,20,30,50"
JAPAN_DIR="$PROJECT_DIR/datasets/japan_prefectures"
RESULTS_DIR="$SCRIPT_DIR/results/gp_baselines"
N_RUNS=3
SEED=42

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Japan Prefecture GP Baselines"
echo "Samples per class: $SAMPLES_PER_CLASS"
echo "Results: $RESULTS_DIR"
echo "========================================"

# 1. SVGP
echo ""
echo "[1/2] SVGP Matern-5/2"
python train_gp_baselines.py \
    --method svgp \
    --dataset japan_prefecture \
    --dataset-dir "$JAPAN_DIR" \
    --kernel matern52 \
    --num-inducing 300 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.01 \
    --samples-per-class "$SAMPLES_PER_CLASS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_svgp_matern52.csv"

# 2. Planar RFF
echo ""
echo "[2/2] Planar RFF Matern-5/2"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset japan_prefecture \
    --dataset-dir "$JAPAN_DIR" \
    --kernel matern52 \
    --num-features 2000 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.01 \
    --samples-per-class "$SAMPLES_PER_CLASS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_planar_rff_matern52.csv"

echo ""
echo "========================================"
echo "All Japan GP Baselines Complete!"
echo "========================================"
echo "Results:"
ls -la "$RESULTS_DIR"/japan_*.csv
