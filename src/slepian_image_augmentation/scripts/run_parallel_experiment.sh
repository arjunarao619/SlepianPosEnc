#!/bin/bash
# =============================================================================
# run_parallel_experiment.sh - Parallel Slepian vs SH training
# =============================================================================
#
# Runs training experiments comparing Slepian vs Spherical Harmonic encodings.
# Uses precomputed design matrices for speed, runs 48 experiments in parallel.
#
# Prerequisites:
#   - Data prepared by prepare_data.sh (needs *_grid_ms.parquet files)
#
# Usage:
#   ./scripts/run_parallel_experiment.sh <data_dir>
#
# Example:
#   ./scripts/run_parallel_experiment.sh /scratch/local/openbuildings_v1
#
# This creates in <data_dir>:
#   - results/*.csv              (48 experiment result files)
#   - results/tikz_coordinates.tex
#   - design_matrices/           (precomputed SH/Slepian matrices)
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="/projects/arra4944/SlepianPosEnc/src/slepian_image_augmentation"
REGIONS="southflorida dhaka maharashtra mexicocity"
EMBEDDINGS="alphaearth galileo"
L_SH_VALUES="10 40"
L_SLEPIAN_VALUES="40 80 120"
ARCHS="linear mlp resmlp siren glu"
SEED=4200

# Training parameters
EPOCHS=120
PATIENCE=20
BATCH_SIZE=512
LR=0.001

# Parallel settings
MAX_PARALLEL_JOBS=30
NUM_WORKERS_PER_JOB=2
PRECOMPUTE_WORKERS=16

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
    echo "  $0 /scratch/local/openbuildings_v1"
    echo ""
    echo "Experiment matrix (4 regions x 2 embeddings x 2 L_SH x 4 L_Slepian x 5 archs = 320 total):"
    echo "  Regions:     $REGIONS"
    echo "  Embeddings:  $EMBEDDINGS"
    echo "  L_SH:        $L_SH_VALUES"
    echo "  L_Slepian:   $L_SLEPIAN_VALUES"
    echo "  Archs:       $ARCHS"
    exit 1
fi

DATA_DIR="$1"
RESULTS_DIR="${DATA_DIR}/results"
PRECOMPUTE_DIR="${DATA_DIR}/design_matrices"
JOBS_FILE="${SCRIPT_DIR}/scripts/.experiment_jobs.txt"

# =============================================================================
# VALIDATION
# =============================================================================
echo "=============================================="
echo "PARALLEL SLEPIAN VS SH EXPERIMENT"
echo "=============================================="
echo "Data dir:     $DATA_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Regions:      $REGIONS"
echo "Embeddings:   $EMBEDDINGS"
echo "L_SH:         $L_SH_VALUES"
echo "L_Slepian:    $L_SLEPIAN_VALUES"
echo "Archs:        $ARCHS"
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
    echo "ERROR: Missing data files. Run prepare_data.sh first:"
    echo "  ./scripts/prepare_data.sh $DATA_DIR <csv_dir>"
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
# STEP 1: Precompute Design Matrices
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 1/3: Precompute Design Matrices"
echo "=============================================="

python scripts/precompute_design_matrices.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$PRECOMPUTE_DIR" \
    --L-sh $L_SH_VALUES \
    --L-slepian $L_SLEPIAN_VALUES \
    --K-max 256 \
    --workers $PRECOMPUTE_WORKERS

# =============================================================================
# STEP 2: Generate and Run Experiments
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 2/3: Generate Experiment Commands"
echo "=============================================="

> "$JOBS_FILE"
total=0
skipped=0

for region in $REGIONS; do
    for emb in $EMBEDDINGS; do
        for L_sh in $L_SH_VALUES; do
            for L_slep in $L_SLEPIAN_VALUES; do
                for arch in $ARCHS; do
                    OUTPUT_FILE="${RESULTS_DIR}/${region}_${emb}_SH${L_sh}_Slep${L_slep}_${arch}_seed${SEED}.csv"

                    if [ -f "$OUTPUT_FILE" ]; then
                        skipped=$((skipped + 1))
                        continue
                    fi

                    CMD="python train_slepian_vs_sh_multiscale_v3.py \
                        --data-dir $DATA_DIR \
                        --region $region \
                        --embedding-type $emb \
                        --arch $arch \
                        --L-sh $L_sh \
                        --L-slepian $L_slep \
                        --precompute-dir $PRECOMPUTE_DIR \
                        --epochs $EPOCHS \
                        --patience $PATIENCE \
                        --batch-size $BATCH_SIZE \
                        --lr $LR \
                        --num-workers $NUM_WORKERS_PER_JOB \
                        --seed $SEED \
                        --csv-out $OUTPUT_FILE"

                    echo "$CMD" | tr -s ' ' >> "$JOBS_FILE"
                    total=$((total + 1))
                done
            done
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
    echo "STEP 3/3: Running $total Experiments ($MAX_PARALLEL_JOBS parallel)"
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

# # =============================================================================
# # STEP 4: Generate TikZ Output
# # =============================================================================
# echo ""
# echo "=============================================="
# echo "Generating TikZ Figure"
# echo "=============================================="

# python scripts/aggregate_results_to_tikz.py \
#     --results-dir "$RESULTS_DIR" \
#     --output "${RESULTS_DIR}/tikz_coordinates.tex"

# # =============================================================================
# # DONE
# # =============================================================================
# echo ""
# echo "=============================================="
# echo "EXPERIMENT COMPLETE"
# echo "=============================================="
# echo ""
# echo "Results directory: $RESULTS_DIR"
# echo ""
# echo "Output files:"
# ls -lh "$RESULTS_DIR"/*.csv 2>/dev/null | wc -l | xargs -I {} echo "  {} CSV result files"
# echo "  tikz_coordinates.tex"
# echo ""
# echo "To view results:"
# echo "  ls $RESULTS_DIR/"
# echo "=============================================="
