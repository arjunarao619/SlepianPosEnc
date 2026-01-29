#!/bin/bash
# =============================================================================
# prepare_data.sh - Full data preparation pipeline
# =============================================================================
#
# Prepares OpenBuildings data with AlphaEarth + Galileo embeddings.
#
# Prerequisites:
#   - OpenBuildings CSV files named: southflorida.csv, dhaka.csv, etc.
#   - Earth Engine access configured
#   - Galileo model weights in galileo/data/models/nano/
#
# Usage:
#   ./scripts/prepare_data.sh <output_dir> <csv_dir>
#
# Example:
#   ./scripts/prepare_data.sh /scratch/local/mydata /path/to/csvs
#
# This creates in <output_dir>:
#   - {region}_grid.parquet      (grid + AlphaEarth embeddings)
#   - {region}_metadata.json
#   - {region}_s2_patches.h5     (Sentinel-2 imagery)
#   - {region}_grid_ms.parquet   (final file with Galileo + multi-scale targets)
#   - {region}_grid_aux.json
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REGIONS="southflorida dhaka maharashtra mexicocity"
EE_PROJECT="sentinel-downloader-468517"
SAMPLES_PER_REGION=30000

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_dir> <csv_dir>"
    echo ""
    echo "Arguments:"
    echo "  output_dir   Directory to save all outputs"
    echo "  csv_dir      Directory containing OpenBuildings CSVs"
    echo "               (expects: southflorida.csv, dhaka.csv, maharashtra.csv, mexicocity.csv)"
    echo ""
    echo "Example:"
    echo "  $0 /scratch/local/openbuildings_v1 /data/openbuildings/csvs"
    exit 1
fi

OUTPUT_DIR="$1"
CSV_DIR="$2"

# =============================================================================
# VALIDATION
# =============================================================================
echo "=============================================="
echo "DATA PREPARATION PIPELINE"
echo "=============================================="
echo "Output dir:  $OUTPUT_DIR"
echo "CSV dir:     $CSV_DIR"
echo "Regions:     $REGIONS"
echo "=============================================="
echo ""

# Check CSVs exist
echo "Checking for CSV files..."
MISSING=0
for region in $REGIONS; do
    if [ -f "${CSV_DIR}/${region}.csv" ]; then
        echo "  ✓ ${region}.csv"
    else
        echo "  ✗ ${region}.csv NOT FOUND"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Missing CSV files. Please ensure all region CSVs are in $CSV_DIR"
    exit 1
fi

# =============================================================================
# SETUP
# =============================================================================
echo ""
echo "Activating conda environment..."
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"

# =============================================================================
# STEP 1: Create Grid + AlphaEarth Embeddings
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 1/5: Create Grid + AlphaEarth Embeddings"
echo "=============================================="

for region in $REGIONS; do
    if [ -f "${OUTPUT_DIR}/${region}_grid.parquet" ]; then
        echo "[$region] Already exists, skipping"
        continue
    fi

    echo "[$region] Creating grid with AlphaEarth embeddings..."
    python create_buildings_dataset.py \
        --csvs "${region}=${CSV_DIR}/${region}.csv" \
        --output-dir "$OUTPUT_DIR" \
        --samples-per-region $SAMPLES_PER_REGION \
        --ee-project "$EE_PROJECT"
done

# =============================================================================
# STEP 2: Download Sentinel-2 Patches
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 2/5: Download Sentinel-2 Patches"
echo "=============================================="

for region in $REGIONS; do
    if [ -f "${OUTPUT_DIR}/${region}_s2_patches.h5" ]; then
        echo "[$region] S2 patches exist, skipping"
        continue
    fi

    echo "[$region] Downloading Sentinel-2 patches..."
    python sentinel2_openbuildings_download.py \
        --data-dir "$OUTPUT_DIR" \
        --regions "$region" \
        --output-dir "$OUTPUT_DIR" \
        --ee-project "$EE_PROJECT"
done

# =============================================================================
# STEP 3: Generate Galileo Embeddings
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 3/5: Generate Galileo Embeddings"
echo "=============================================="

python generate_galileo_all_regions.py \
    --data-dir "$OUTPUT_DIR" \
    --model-size nano

# =============================================================================
# STEP 4: Merge Galileo into Grid Parquets
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 4/5: Merge Galileo Embeddings"
echo "=============================================="

python merge_galileo_embeddings.py \
    --data-dir "$OUTPUT_DIR"

# =============================================================================
# STEP 5: Prep Multi-Scale Targets
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 5/5: Prep Multi-Scale Targets"
echo "=============================================="

for region in $REGIONS; do
    if [ -f "${OUTPUT_DIR}/${region}_grid_ms.parquet" ]; then
        echo "[$region] Multi-scale parquet exists, skipping"
        continue
    fi

    echo "[$region] Creating multi-scale targets..."
    python prep_multiscale_targets.py \
        --data-dir "$OUTPUT_DIR" \
        --region "$region"
done

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=============================================="
echo "DATA PREPARATION COMPLETE"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files created per region:"
for region in $REGIONS; do
    echo ""
    echo "[$region]"
    ls -lh "${OUTPUT_DIR}/${region}"_* 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
done

echo ""
echo "=============================================="
echo "Next: Run parallel experiments"
echo "  ./scripts/run_parallel_experiment.sh $OUTPUT_DIR"
echo "=============================================="
