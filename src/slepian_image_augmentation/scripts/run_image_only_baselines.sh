#!/bin/bash
# =============================================================================
# run_image_only_baselines.sh - Image-only baseline experiments
# =============================================================================
#
# Runs image embedding baseline experiments across multiple seeds.
# Uses the --image-only flag to skip SH and Slepian computations.
#
# Usage:
#   ./scripts/run_image_only_baselines.sh <data_dir>
#
# Example:
#   ./scripts/run_image_only_baselines.sh /path/to/data
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REGIONS="southflorida dhaka maharashtra mexicocity"
EMBEDDINGS="alphaearth galileo"
ARCH="mlp"
SEEDS="1234 0 42 1000 1"

# Training parameters
EPOCHS=120
PATIENCE=20
BATCH_SIZE=512
LR=0.001

# Parallel settings
MAX_PARALLEL_JOBS=20
NUM_WORKERS_PER_JOB=2

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_dir>"
    echo ""
    echo "Arguments:"
    echo "  data_dir   Directory containing prepared data (*_grid_ms.parquet files)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/data"
    echo ""
    echo "Experiment matrix (4 regions x 2 embeddings x 5 seeds = 40 total):"
    echo "  Regions:     $REGIONS"
    echo "  Embeddings:  $EMBEDDINGS"
    echo "  Seeds:       $SEEDS"
    echo "  Arch:        $ARCH"
    exit 1
fi

DATA_DIR="$1"
RESULTS_DIR="${DATA_DIR}/results_image_only"
JOBS_FILE="${SCRIPT_DIR}/scripts/.image_only_jobs.txt"

# =============================================================================
# VALIDATION
# =============================================================================
echo "=============================================="
echo "IMAGE-ONLY BASELINE EXPERIMENT"
echo "=============================================="
echo "Data dir:     $DATA_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Regions:      $REGIONS"
echo "Embeddings:   $EMBEDDINGS"
echo "Seeds:        $SEEDS"
echo "Arch:         $ARCH"
echo "Parallel:     $MAX_PARALLEL_JOBS jobs"
echo "=============================================="
echo ""

# Check data files exist
echo "Checking for prepared data..."
MISSING=0
for region in $REGIONS; do
    if [ -f "${DATA_DIR}/${region}_grid_ms.parquet" ]; then
        echo "  ✓ ${region}_grid_ms.parquet"
    else
        echo "  ✗ ${region}_grid_ms.parquet NOT FOUND"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Missing data files."
    exit 1
fi

# =============================================================================
# SETUP
# =============================================================================
echo ""
echo "Activating conda environment..."
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"

# =============================================================================
# GENERATE AND RUN EXPERIMENTS
# =============================================================================
echo ""
echo "=============================================="
echo "Generating Experiment Commands"
echo "=============================================="

> "$JOBS_FILE"
total=0
skipped=0

for region in $REGIONS; do
    for emb in $EMBEDDINGS; do
        for seed in $SEEDS; do
            OUTPUT_FILE="${RESULTS_DIR}/${region}_${emb}_imageonly_${ARCH}_seed${seed}.csv"

            if [ -f "$OUTPUT_FILE" ]; then
                skipped=$((skipped + 1))
                continue
            fi

            CMD="python train_slepian_vs_sh_multiscale_v3.py \
                --data-dir $DATA_DIR \
                --region $region \
                --embedding-type $emb \
                --arch $ARCH \
                --image-only \
                --epochs $EPOCHS \
                --patience $PATIENCE \
                --batch-size $BATCH_SIZE \
                --lr $LR \
                --num-workers $NUM_WORKERS_PER_JOB \
                --seed $seed \
                --csv-out $OUTPUT_FILE"

            echo "$CMD" | tr -s ' ' >> "$JOBS_FILE"
            total=$((total + 1))
        done
    done
done

echo "Experiments to run: $total"
echo "Already completed:  $skipped"

if [ "$total" -eq 0 ]; then
    echo ""
    echo "All experiments already complete!"
else
    echo ""
    echo "=============================================="
    echo "Running $total Experiments ($MAX_PARALLEL_JOBS parallel)"
    echo "=============================================="

    if command -v parallel &> /dev/null; then
        echo "Using GNU parallel..."
        parallel --jobs $MAX_PARALLEL_JOBS \
                 --bar \
                 --joblog "${RESULTS_DIR}/parallel_joblog.txt" \
                 --resume-failed \
                 --retries 1 \
                 --timeout 3600 \
                 < "$JOBS_FILE"
    else
        echo "GNU parallel not found. Running with xargs..."
        cat "$JOBS_FILE" | xargs -P $MAX_PARALLEL_JOBS -I {} bash -c '{}'
    fi
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=============================================="
echo "IMAGE-ONLY BASELINE EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Output files:"
ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null | wc -l | xargs -I {} echo "  {} CSV result files"
echo ""
echo "To view results:"
echo "  ls $RESULTS_DIR/"
echo "=============================================="
