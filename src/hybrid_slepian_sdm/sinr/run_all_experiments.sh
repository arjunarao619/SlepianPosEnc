#!/bin/bash
set -e

# Activate environment (change this for your system)
source /curc/arm_sw/modules/idep/miniforge/24.11.2-1_setup.sh
mamba activate bg2

cd /projects/arra4944/SlepianPosEnc/src/hybrid_slepian_sdm/sinr

echo "========================================"
echo "Running all experiments"
echo "========================================"

# 1. Sin-cos baseline
echo ">>> sin_cos"
python train_and_evaluate_models.py sin_cos sin_cos

# 2. Spherical Harmonics for L=10,20,30,40
for L in 10 20 30 40; do
    echo ">>> sh_${L}"
    python train_and_evaluate_models.py sh_${L} sh --sh_L ${L}
done

# 3. Slepian-only (L_global=0) for L_regional=10,20,30,40
for L in 10 20 30 40; do
    echo ">>> slepian_only_l${L}"
    python train_and_evaluate_models.py slepian_only_l${L} slepian --L_global 0 --L_regional ${L}
done

# 4. Hybrid Slepian (L_global=10) for L_regional=10,20,30,40
for L in 10 20 30 40; do
    echo ">>> hybrid_l${L}"
    python train_and_evaluate_models.py hybrid_l${L} slepian --L_global 10 --L_regional ${L}
done

echo "========================================"
echo "All experiments complete!"
echo "Run 'python parse_results.py' to see summary."
echo "========================================"
