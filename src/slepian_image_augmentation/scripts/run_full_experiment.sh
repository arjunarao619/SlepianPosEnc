#!/bin/bash
#SBATCH --job-name=slepian_exp
#SBATCH --output=slepian_exp_%j.out
#SBATCH --error=slepian_exp_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# =============================================================================
# Full Slepian vs SH Experiment
# =============================================================================
# Runs training for all combinations of:
#   - Regions: southflorida, dhaka, maharashtra, mexicocity
#   - Embeddings: alphaearth, galileo
#   - Slepian K values: 12, 24, 48, 96 (matching figure purple1-4)
#
# Usage:
#   ./scripts/run_full_experiment.sh              # Run all experiments
#   ./scripts/run_full_experiment.sh --dry-run    # Print commands without running
#   sbatch scripts/run_full_experiment.sh         # Submit as SLURM job
# =============================================================================

set -e

# Configuration
DATA_DIR="/scratch/local/arra4944_images/openbuildings"
SCRIPT_DIR="/projects/arra4944/SlepianPosEnc/src/slepian_image_augmentation"
RESULTS_DIR="${DATA_DIR}/results"

# Experiment parameters
REGIONS="southflorida dhaka maharashtra mexicocity"
EMBEDDINGS="alphaearth galileo"
K_VALUES="12 24 48 96"
L_SH=64
L_SLEPIAN=96
SEED=123

# Training parameters
EPOCHS=120
PATIENCE=20
BATCH_SIZE=512
LR=0.001

# Parse arguments
DRY_RUN=false
SINGLE_REGION=""
SINGLE_EMB=""
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --region=*)
            SINGLE_REGION="${arg#*=}"
            shift
            ;;
        --embedding=*)
            SINGLE_EMB="${arg#*=}"
            shift
            ;;
    esac
done

# Override if single region/embedding specified
if [ -n "$SINGLE_REGION" ]; then
    REGIONS="$SINGLE_REGION"
fi
if [ -n "$SINGLE_EMB" ]; then
    EMBEDDINGS="$SINGLE_EMB"
fi

# Activate environment
echo "=============================================="
echo "SLEPIAN VS SH FULL EXPERIMENT"
echo "=============================================="
echo "Regions: $REGIONS"
echo "Embeddings: $EMBEDDINGS"
echo "K values: $K_VALUES"
echo "L_SH: $L_SH, L_Slepian: $L_SLEPIAN"
echo "Results dir: $RESULTS_DIR"
echo "=============================================="

if [ "$DRY_RUN" = false ]; then
    echo "Activating environment..."
    source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
    mamba activate bg2

    # Create results directory
    mkdir -p "$RESULTS_DIR"
fi

cd "$SCRIPT_DIR"

# Count total experiments
total=0
for region in $REGIONS; do
    for emb in $EMBEDDINGS; do
        for K in $K_VALUES; do
            total=$((total + 1))
        done
    done
done

echo ""
echo "Total experiments to run: $total"
echo ""

# Run experiments
current=0
for region in $REGIONS; do
    for emb in $EMBEDDINGS; do
        for K in $K_VALUES; do
            current=$((current + 1))

            echo "=============================================="
            echo "[$current/$total] $region | $emb | K=$K"
            echo "=============================================="

            OUTPUT_FILE="${RESULTS_DIR}/${region}_${emb}_SH${L_SH}_Slep${L_SLEPIAN}_K${K}_seed${SEED}.csv"

            # Skip if already exists
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  [SKIP] Already exists: $OUTPUT_FILE"
                continue
            fi

            CMD="python train_slepian_vs_sh_multiscale_v2.py \
                --data-dir $DATA_DIR \
                --region $region \
                --embedding-type $emb \
                --L-sh $L_SH \
                --L-slepian $L_SLEPIAN \
                --K $K \
                --epochs $EPOCHS \
                --patience $PATIENCE \
                --batch-size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --csv-out $OUTPUT_FILE"

            if [ "$DRY_RUN" = true ]; then
                echo "  [DRY-RUN] Would run:"
                echo "  $CMD"
            else
                echo "  Running..."
                eval $CMD
                echo "  Done: $OUTPUT_FILE"
            fi

            echo ""
        done
    done
done

echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
echo ""

# List all result files
if [ "$DRY_RUN" = false ]; then
    echo "Generated files:"
    ls -la "$RESULTS_DIR"/*.csv 2>/dev/null || echo "No CSV files found"
fi
