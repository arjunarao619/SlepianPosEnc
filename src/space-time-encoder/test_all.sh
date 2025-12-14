#!/bin/bash
# Directory containing the model weights
WEIGHTS_DIR="/projects/arra4944/SlepianEncoder/space-time-encoder/weights"
DATA_PATH="/scratch/local/arra4944_images/ace/"
RESULTS_FILE="regularized_models_results.txt"

# Create logs directory if it doesn't exist
mkdir -p logs

# Clear or create results file
echo "======================================" > $RESULTS_FILE
echo "Testing Regularized Models on ACE Dataset" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "======================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# List of regularized models (with _reg_ in name)
REG_MODELS=(
    "best_fourier_reg_seed120.pt"
    "best_fourier_reg_seed1234.pt"
    "best_fourier_reg_seed42.pt"
    "best_fourier_reg_seed55.pt"
    "best_legendre_reg_seed120.pt"
    "best_legendre_reg_seed1234.pt"
    "best_legendre_reg_seed42.pt"
    "best_legendre_reg_seed55.pt"
    "best_legendre_reg_seed5678.pt"
    "best_triangle_reg_seed120.pt"
    "best_triangle_reg_seed1234.pt"
    "best_triangle_reg_seed42.pt"
    "best_triangle_reg_seed55.pt"
    "best_triangle_reg_seed5678.pt"
)

# DPSS models (optimized versions)
DPSS_MODELS=(
"best_ste_dpss_NW15_K40_opt_seed42.pt"
"best_ste_dpss_NW24_K40_opt_seed42.pt"  
"best_ste_dpss_NW28_K40_opt_seed42.pt"
"best_ste_dpss_NW22_K40_opt.pt"
"best_ste_dpss_NW26_K40_opt.pt"
"best_ste_dpss_NW29_K40_opt.pt"
"best_ste_dpss_NW22_K40_opt_seed42.pt"
"best_ste_dpss_NW26_K40_opt_seed42.pt"  
"best_ste_dpss_NW29_K40_opt_seed42.pt"
"best_ste_dpss_NW24_K40_opt.pt"
"best_ste_dpss_NW28_K40_opt.pt"
"best_ste_dpss_NW30_K40_opt_seed42.pt"
)

# Combine all models to test
# ALL_MODELS=("${REG_MODELS[@]}" "${DPSS_MODELS[@]}")
ALL_MODELS=("${DPSS_MODELS[@]}")
# Counter for tracking progress
TOTAL=${#ALL_MODELS[@]}
CURRENT=0

echo "Testing ${TOTAL} models..."
echo ""

# Test each model
for MODEL in "${ALL_MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    MODEL_PATH="${WEIGHTS_DIR}/${MODEL}"
    
    echo "================================================" | tee -a $RESULTS_FILE
    echo "[${CURRENT}/${TOTAL}] Testing: ${MODEL}" | tee -a $RESULTS_FILE
    echo "================================================" | tee -a $RESULTS_FILE
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Model file not found: ${MODEL_PATH}" | tee -a $RESULTS_FILE
        echo "" >> $RESULTS_FILE
        continue
    fi
    
    # Run the test script
    python test_ace.py \
        --ckpt "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --batch_size 40000 \
        --num_workers 8 \
        --train_frac 0.01 \
        --val_frac 0.01 \
        --test_frac 0.01 \
        --device cuda \
        --per_var 2>&1 | tee -a $RESULTS_FILE
    
    echo "" >> $RESULTS_FILE
    echo "Completed: ${MODEL}" | tee -a $RESULTS_FILE
    echo "" >> $RESULTS_FILE
done

echo "================================================" | tee -a $RESULTS_FILE
echo "All models tested! Results saved to ${RESULTS_FILE}" | tee -a $RESULTS_FILE
echo "================================================" | tee -a $RESULTS_FILE

# Extract and summarize results
echo "" | tee -a $RESULTS_FILE
echo "======================================" | tee -a $RESULTS_FILE
echo "SUMMARY OF RESULTS" | tee -a $RESULTS_FILE
echo "======================================" | tee -a $RESULTS_FILE
grep -E "(Testing:|Test RMSE:|Test MAE:)" $RESULTS_FILE | \
    awk 'BEGIN {model=""} 
         /Testing:/ {model=$NF; gsub(/\[.*\]/, "", $0); print ""} 
         /Test RMSE:/ {rmse=$3; printf "%-50s RMSE: %s", model, rmse} 
         /Test MAE:/ {mae=$3; printf "  MAE: %s\n", mae}' | \
    tee -a $RESULTS_FILE

echo "" | tee -a $RESULTS_FILE
echo "Done!"