#!/bin/bash
# run_openbuildings_pipeline.sh
# Full pipeline for OpenBuildings experiments with AlphaEarth + Galileo embeddings
#
# Usage:
#   ./run_openbuildings_pipeline.sh [step] [region]
#
# Steps:
#   1 or galileo   - Generate Galileo embeddings for all regions
#   2 or merge     - Merge Galileo embeddings into original parquets
#   3 or prep      - Run prep_multiscale_targets for all regions
#   4 or train     - Run training with specified embedding type
#   all            - Run steps 1-3 (doesn't auto-run training)
#
# Examples:
#   ./run_openbuildings_pipeline.sh all                      # Steps 1-3 for all regions
#   ./run_openbuildings_pipeline.sh prep southflorida        # Just prep for one region
#   ./run_openbuildings_pipeline.sh train southflorida       # Train with default (alphaearth)

set -e

# Configuration
DATA_DIR="/scratch/local/arra4944_images/openbuildings"
SCRIPT_DIR="/projects/arra4944/SlepianPosEnc/src/slepian_image_augmentation"
REGIONS="southflorida dhaka maharashtra mexicocity"

# Parse arguments
STEP=${1:-"help"}
REGION=${2:-"all"}

# Activate environment
echo "Activating conda environment..."
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

cd "$SCRIPT_DIR"

case "$STEP" in
    1|galileo)
        echo "=========================================="
        echo "STEP 1: Generate Galileo Embeddings"
        echo "=========================================="
        python generate_galileo_all_regions.py --data-dir "$DATA_DIR"
        ;;

    2|merge)
        echo "=========================================="
        echo "STEP 2: Merge Galileo into Original Parquets"
        echo "=========================================="
        python merge_galileo_embeddings.py --data-dir "$DATA_DIR"
        ;;

    3|prep)
        echo "=========================================="
        echo "STEP 3: Prep Multi-Scale Targets"
        echo "=========================================="
        if [ "$REGION" = "all" ]; then
            for r in $REGIONS; do
                echo "--- Processing $r ---"
                python prep_multiscale_targets.py --data-dir "$DATA_DIR" --region "$r"
            done
        else
            python prep_multiscale_targets.py --data-dir "$DATA_DIR" --region "$REGION"
        fi
        ;;

    4|train)
        echo "=========================================="
        echo "STEP 4: Train Model"
        echo "=========================================="
        if [ "$REGION" = "all" ]; then
            echo "Please specify a region for training"
            exit 1
        fi
        EMB_TYPE=${3:-"alphaearth"}  # Third arg: alphaearth or galileo
        echo "Training $REGION with $EMB_TYPE embeddings..."
        python train_slepian_vs_sh_multiscale_v2.py \
            --data-dir "$DATA_DIR" \
            --region "$REGION" \
            --embedding-type "$EMB_TYPE" \
            --csv-out "$DATA_DIR/results/"
        ;;

    all)
        echo "=========================================="
        echo "RUNNING FULL PIPELINE (Steps 1-3)"
        echo "=========================================="

        echo ""
        echo "STEP 1: Generate Galileo Embeddings"
        echo "---"
        python generate_galileo_all_regions.py --data-dir "$DATA_DIR"

        echo ""
        echo "STEP 2: Merge Galileo into Original Parquets"
        echo "---"
        python merge_galileo_embeddings.py --data-dir "$DATA_DIR"

        echo ""
        echo "STEP 3: Prep Multi-Scale Targets"
        echo "---"
        for r in $REGIONS; do
            echo "--- Processing $r ---"
            python prep_multiscale_targets.py --data-dir "$DATA_DIR" --region "$r"
        done

        echo ""
        echo "=========================================="
        echo "PIPELINE COMPLETE!"
        echo "=========================================="
        echo ""
        echo "Next steps - run training:"
        echo "  ./run_openbuildings_pipeline.sh train southflorida alphaearth"
        echo "  ./run_openbuildings_pipeline.sh train southflorida galileo"
        ;;

    help|*)
        echo "OpenBuildings Pipeline"
        echo ""
        echo "Usage: $0 [step] [region] [embedding_type]"
        echo ""
        echo "Steps:"
        echo "  1, galileo  - Generate Galileo embeddings for all regions"
        echo "  2, merge    - Merge Galileo embeddings into original parquets"
        echo "  3, prep     - Run prep_multiscale_targets (specify region or 'all')"
        echo "  4, train    - Run training (requires region and embedding_type)"
        echo "  all         - Run steps 1-3 for all regions"
        echo ""
        echo "Examples:"
        echo "  $0 all                             # Full prep pipeline"
        echo "  $0 train southflorida alphaearth   # Train with AlphaEarth"
        echo "  $0 train southflorida galileo      # Train with Galileo"
        echo ""
        echo "Regions: $REGIONS"
        ;;
esac
