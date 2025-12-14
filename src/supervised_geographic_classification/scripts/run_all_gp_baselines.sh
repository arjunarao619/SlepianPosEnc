#!/bin/bash
#SBATCH --job-name=gp_baselines
#SBATCH --output=logs/gp_baselines_%j.out
#SBATCH --error=logs/gp_baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# =============================================================================
# Comprehensive GP Baselines Experiments
# =============================================================================
# Runs GP baselines on all datasets:
#   - California Housing (regression, ~30K points)
#   - Japan Prefecture (classification, ~7K points)
#   - MSS Arctic (regression, >1M points)
#
# Methods:
#   - exact_gp: Only for small datasets (California small fractions, Japan)
#   - svgp: For all datasets
#   - planar_rff: For California and Japan
#   - spherical_svgp: For MSS (proper spherical geometry)
#   - spherical_rff: For MSS
# =============================================================================

set -e  # Exit on error

# Setup environment
echo "Setting up environment..."
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Results directory
RESULTS_DIR="$SCRIPT_DIR/results/gp_baselines"
mkdir -p "$RESULTS_DIR"
mkdir -p "$SCRIPT_DIR/../logs"

# Common parameters
N_RUNS=3
SEED=42

echo "========================================"
echo "GP Baselines Experiments"
echo "Results directory: $RESULTS_DIR"
echo "========================================"
echo ""

# =============================================================================
# California Housing
# =============================================================================
CALI_FRACS="0.01,0.10,0.25,0.50,0.75,1.00"

echo "========================================"
echo "California Housing Dataset"
echo "========================================"

# SVGP (main baseline)
echo ""
echo "[California] SVGP Matern-5/2"
python train_gp_baselines.py \
    --method svgp \
    --dataset california_housing \
    --kernel matern52 \
    --num-inducing 500 \
    --num-epochs 50 \
    --batch-size 1024 \
    --lr 0.01 \
    --label-fracs "$CALI_FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_svgp_matern52.csv"

# Planar RFF
echo ""
echo "[California] Planar RFF Matern-5/2"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset california_housing \
    --kernel matern52 \
    --num-features 2000 \
    --label-fracs "$CALI_FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_planar_rff_matern52.csv"

# Exact GP (only small fractions - too slow for large data)
echo ""
echo "[California] Exact GP Matern-5/2 (small fractions only)"
python train_gp_baselines.py \
    --method exact_gp \
    --dataset california_housing \
    --kernel matern52 \
    --num-iterations 100 \
    --lr 0.1 \
    --label-fracs "0.01,0.10,0.25" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_exact_gp_matern52.csv"

# =============================================================================
# Japan Prefecture (samples per class for label efficiency)
# =============================================================================
JAPAN_SAMPLES="1,2,5,10,20,30,50"
JAPAN_DIR="$PROJECT_DIR/datasets/japan_prefectures"

echo ""
echo "========================================"
echo "Japan Prefecture Dataset"
echo "========================================"

# SVGP
echo ""
echo "[Japan] SVGP Matern-5/2"
python train_gp_baselines.py \
    --method svgp \
    --dataset japan_prefecture \
    --dataset-dir "$JAPAN_DIR" \
    --kernel matern52 \
    --num-inducing 300 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.01 \
    --samples-per-class "$JAPAN_SAMPLES" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_svgp_matern52.csv"

# Planar RFF
echo ""
echo "[Japan] Planar RFF Matern-5/2"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset japan_prefecture \
    --dataset-dir "$JAPAN_DIR" \
    --kernel matern52 \
    --num-features 2000 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.01 \
    --samples-per-class "$JAPAN_SAMPLES" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_planar_rff_matern52.csv"

# Note: Exact GP skipped for classification (Dirichlet likelihood unstable for 47 classes)
# SVGP and RFF are the recommended methods for multi-class classification

# =============================================================================
# MSS Arctic (Large-scale, spherical geometry)
# =============================================================================
MSS_FRACS="0.02,0.05,0.10,1.00"
MSS_PATH="/scratch/local/arra4944_images/drf/Experiment_data"

echo ""
echo "========================================"
echo "MSS Arctic Dataset"
echo "========================================"

# Check if MSS data exists
if [ -d "$MSS_PATH/exp1" ]; then
    # SVGP (planar approximation)
    echo ""
    echo "[MSS] SVGP Matern-5/2 (planar)"
    python train_gp_baselines.py \
        --method svgp \
        --dataset mss \
        --mss-data-path "$MSS_PATH" \
        --kernel matern52 \
        --num-inducing 2000 \
        --num-epochs 30 \
        --batch-size 4096 \
        --lr 0.01 \
        --label-fracs "$MSS_FRACS" \
        --n-runs $N_RUNS \
        --seed $SEED \
        --csv-path "$RESULTS_DIR/mss_svgp_matern52.csv"

    # Planar RFF
    echo ""
    echo "[MSS] Planar RFF Matern-5/2"
    python train_gp_baselines.py \
        --method planar_rff \
        --dataset mss \
        --mss-data-path "$MSS_PATH" \
        --kernel matern52 \
        --num-features 3000 \
        --label-fracs "$MSS_FRACS" \
        --n-runs $N_RUNS \
        --seed $SEED \
        --csv-path "$RESULTS_DIR/mss_planar_rff_matern52.csv"

    # Spherical SVGP (proper geometry)
    echo ""
    echo "[MSS] Spherical SVGP Matern-5/2"
    python train_gp_baselines.py \
        --method spherical_svgp \
        --dataset mss \
        --mss-data-path "$MSS_PATH" \
        --kernel matern52 \
        --num-inducing 2000 \
        --num-epochs 30 \
        --batch-size 4096 \
        --lr 0.01 \
        --label-fracs "$MSS_FRACS" \
        --n-runs $N_RUNS \
        --seed $SEED \
        --csv-path "$RESULTS_DIR/mss_spherical_svgp_matern52.csv"

    # Spherical RFF
    echo ""
    echo "[MSS] Spherical RFF Matern-5/2"
    python train_gp_baselines.py \
        --method spherical_rff \
        --dataset mss \
        --mss-data-path "$MSS_PATH" \
        --kernel matern52 \
        --num-features 3000 \
        --label-fracs "$MSS_FRACS" \
        --n-runs $N_RUNS \
        --seed $SEED \
        --csv-path "$RESULTS_DIR/mss_spherical_rff_matern52.csv"
else
    echo "WARNING: MSS data not found at $MSS_PATH"
    echo "Skipping MSS experiments"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "All GP Baselines Complete!"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files created:"
ls -la "$RESULTS_DIR"/*.csv 2>/dev/null || echo "No CSV files found"
echo ""
echo "To aggregate results, run:"
echo "  python aggregate_all_results.py --results-dir $RESULTS_DIR"
