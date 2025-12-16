#!/bin/bash
# Run GP baseline experiments

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Results directory
RESULTS_DIR="$SCRIPT_DIR/results/gp_baselines"
mkdir -p "$RESULTS_DIR"

# Common parameters
LABEL_FRACS="0.01,0.10,0.25,0.50,0.75,1.00"
N_RUNS=3
SEED=42

echo "========================================"
echo "Running GP Baselines"
echo "========================================"

# California Housing - SVGP
echo ""
echo "California Housing - SVGP (Matern 5/2)"
python train_gp_baselines.py \
    --method svgp \
    --dataset california_housing \
    --kernel matern52 \
    --num-inducing 500 \
    --num-epochs 50 \
    --batch-size 1024 \
    --lr 0.01 \
    --label-fracs "$LABEL_FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_svgp_matern52.csv"

# California Housing - Planar RFF
echo ""
echo "California Housing - Planar RFF (Matern 5/2)"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset california_housing \
    --kernel matern52 \
    --num-features 2000 \
    --label-fracs "$LABEL_FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/california_planar_rff_matern52.csv"

# California Housing - Exact GP (only for small subsets)
echo ""
echo "California Housing - Exact GP (Matern 5/2) - small subsets only"
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

# Japan Prefecture - SVGP
echo ""
echo "Japan Prefecture - SVGP (Matern 5/2)"
python train_gp_baselines.py \
    --method svgp \
    --dataset japan_prefecture \
    --dataset-dir "$PROJECT_DIR/datasets/japan_prefectures" \
    --kernel matern52 \
    --num-inducing 300 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.01 \
    --label-fracs "$LABEL_FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_svgp_matern52.csv"

# Japan Prefecture - Planar RFF
echo ""
echo "Japan Prefecture - Planar RFF (Matern 5/2)"
python train_gp_baselines.py \
    --method planar_rff \
    --dataset japan_prefecture \
    --dataset-dir "$PROJECT_DIR/datasets/japan_prefectures" \
    --kernel matern52 \
    --num-features 2000 \
    --num-epochs 100 \
    --batch-size 512 \
    --lr 0.01 \
    --label-fracs "$LABEL_FRACS" \
    --n-runs $N_RUNS \
    --seed $SEED \
    --csv-path "$RESULTS_DIR/japan_planar_rff_matern52.csv"

echo ""
echo "========================================"
echo "GP Baselines Complete"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
