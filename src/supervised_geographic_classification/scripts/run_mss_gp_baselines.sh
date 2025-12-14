#!/bin/bash
# Run all GP baselines on MSS Arctic dataset

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
MSS_PATH="/scratch/local/arra4944_images/drf/Experiment_data"
FRACS="0.001,0.01,0.05,1.00"
RESULTS_DIR="$SCRIPT_DIR/results/gp_baselines"
N_RUNS=3
SEED=42

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "MSS Arctic GP Baselines"
echo "Label fractions: $FRACS"
echo "Results: $RESULTS_DIR"
echo "========================================"

# 1. SVGP (planar)
echo ""
echo "[1/4] SVGP Matern-5/2 (planar)"
python train_gp_baselines.py \
    --method svgp \
    --dataset mss \
    --mss-data-path "$MSS_PATH" \
    --kernel matern52 \
    --num-inducing 2000 \
    --num-epochs 30 \
    --batch-size 4096 \
    --lr 0.01 \
    --label-fracs "$FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/mss_svgp_matern52.csv"

# 2. Planar RFF
echo ""
echo "[2/4] Planar RFF Matern-5/2"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset mss \
    --mss-data-path "$MSS_PATH" \
    --kernel matern52 \
    --num-features 3000 \
    --label-fracs "$FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/mss_planar_rff_matern52.csv"

# 3. Spherical SVGP
echo ""
echo "[3/4] Spherical SVGP Matern-5/2"
python train_gp_baselines.py \
    --method spherical_svgp \
    --dataset mss \
    --mss-data-path "$MSS_PATH" \
    --kernel matern52 \
    --num-inducing 2000 \
    --num-epochs 30 \
    --batch-size 4096 \
    --lr 0.01 \
    --label-fracs "$FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/mss_spherical_svgp_matern52.csv"

# 4. Spherical RFF
echo ""
echo "[4/4] Spherical RFF Matern-5/2"
python train_gp_baselines.py \
    --method spherical_rff \
    --dataset mss \
    --mss-data-path "$MSS_PATH" \
    --kernel matern52 \
    --num-features 3000 \
    --label-fracs "$FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/mss_spherical_rff_matern52.csv"

echo ""
echo "========================================"
echo "All MSS GP Baselines Complete!"
echo "========================================"
echo "Results:"
ls -la "$RESULTS_DIR"/mss_*.csv
