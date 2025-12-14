#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# User-tunable grid
# ---------------------------
NW_LIST=(15 22 24 26 28 29 30 31 32 34 34 35)      # DPSS time-bandwidth (W*N) choices
K_LIST=(40)             # requested DPSS basis count (code will cap at 2*NW-1)
OPT_LIST=(opt)        # 'base' uses DPSSTimeEncoder, 'opt' uses OptimizedDPSSTimeEncoder
REG_LIST=(reg noreg)       # 'reg' = STE-style orthoreg on, 'noreg' = all zero
SEEDS=(42 55 120)                # add more if you want multiple seeds

# Optional: adjust concentration threshold (passed into DPSSTimeEncoder)
DPS_CONC_THR="${DPS_CONC_THR:-0.85}"

# Paths
PY=python
TRAIN_FILE="train_dpss.py"

# CUDA device (optional): export CUDA_VISIBLE_DEVICES=0
: "${CUDA_VISIBLE_DEVICES:=}"

# ---------------------------
# Runner
# ---------------------------
run_one () {
  local NW="$1" K="$2" OPT="$3" REG="$4" SEED="$5"

  echo "=============================================================="
  echo "Run: NW=${NW}  K=${K}  OPT=${OPT}  REG=${REG}  SEED=${SEED}"
  echo "=============================================================="

  # Call train_dpss.py but override its module variables before main()
  # Args to Python: NW K OPT REG SEED DPS_CONC_THR
  ${PY} - "${NW}" "${K}" "${OPT}" "${REG}" "${SEED}" "${DPS_CONC_THR}" <<'PYCODE'
import sys, os, importlib, random
NW  = int(sys.argv[1])
K   = int(sys.argv[2])
OPT = sys.argv[3]         # 'base' or 'opt'
REG = sys.argv[4]         # 'reg' or 'noreg'
SEED= int(sys.argv[5])
THR = float(sys.argv[6])

m = importlib.import_module('train_dpss')

# Set seeds for reproducibility of splits / init
import numpy as np
import torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Override the SEED global variable (used for checkpoint filename)
m.SEED = SEED

# Single-run override of RUNS in the module
OPT_BOOL = (OPT.lower() == 'opt')
m.RUNS = [(NW, K, OPT_BOOL)]

# Override base regularization knobs if requested
if REG.lower() == 'noreg':
    m.BASE['ortho_weight_final'] = 0.0
    m.BASE['ortho_weight_space'] = 0.0
    m.BASE['ortho_weight_time']  = 0.0
    m.BASE['time_grad_penalty_weight'] = 0.0

# Pass concentration threshold
m.BASE['dpss_concentration_threshold'] = THR

# Optional: different output root per run? You can also chdir here if desired.
print(f"[driver] Using seed = {SEED}")
print(f"[driver] Using concentration threshold = {THR}")
print(f"[driver] Regularization: {('ON' if REG.lower()=='reg' else 'OFF')}")

# Launch
m.main()
PYCODE
}

# ---------------------------
# Sweep
# ---------------------------
for seed in "${SEEDS[@]}"; do
  for reg in "${REG_LIST[@]}"; do
    for nw in "${NW_LIST[@]}"; do
      for k in "${K_LIST[@]}"; do
        for opt in "${OPT_LIST[@]}"; do
          run_one "${nw}" "${k}" "${opt}" "${reg}" "${seed}"
        done
      done
    done
  done
done