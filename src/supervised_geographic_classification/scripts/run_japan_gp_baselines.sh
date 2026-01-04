#!/bin/bash
# Run GP baselines on Japan Prefecture dataset
# Aligned with Deep Random Features paper (arxiv 2412.11350)

# Don't exit on error - continue to next method
set +e

# Setup environment
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parameters
SAMPLES_PER_CLASS="1,2,5,10,20,30,50"
JAPAN_DIR="/projects/arra4944/SlepianPosEnc/data/japan"
RESULTS_DIR="$SCRIPT_DIR/results/gp_baselines"
N_RUNS=3
SEED=42

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Japan Prefecture GP Baselines (Paper-aligned)"
echo "Samples per class: $SAMPLES_PER_CLASS"
echo "Results: $RESULTS_DIR"
echo "========================================"

# 1. Deep RFF (paper-aligned) - Recommended
echo ""
echo "[1/2] Deep RFF Matern-5/2 (Paper-aligned, 3 layers)"
python train_gp_baselines.py \
    --method deep_rff \
    --dataset japan_prefecture \
    --dataset-dir "$JAPAN_DIR" \
    --kernel matern52 \
    --hidden-dim 1000 \
    --bottleneck-dim 64 \
    --num-layers 3 \
    --amplitude 1.0 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.001 \
    --samples-per-class "$SAMPLES_PER_CLASS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_deep_rff_matern52.csv"

# 2. Planar RFF (single layer baseline)
echo ""
echo "[2/2] Planar RFF Matern-5/2 (Single layer baseline)"
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
echo "Note: SVGP removed (causes OOM, not in paper)"
echo "========================================"
echo "Results:"
ls -la "$RESULTS_DIR"/japan_*.csv
