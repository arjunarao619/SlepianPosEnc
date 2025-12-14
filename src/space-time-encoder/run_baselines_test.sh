#!/usr/bin/env bash
# eval_all_ace.sh
set -euo pipefail

CKPT_DIR=${1:-/projects/arra4944/SlepianEncoder/space-time-encoder}
DATA=${2:-/scratch/local/arra4944_images/ace/}
OUTDIR=${3:-logs_eval}
BS=${BATCH_SIZE:-40000}
NUM_WORKERS=${NUM_WORKERS:-$(($(nproc)-2))}
PER_VAR=${PER_VAR:-1}    # 1 to print per-variable metrics
STRICT=${STRICT:-0}      # 1 -> strict load

mkdir -p "$OUTDIR"

ts=$(date +"%Y%m%d_%H%M%S")
MASTER="$OUTDIR/master_${ts}.log"
touch "$MASTER"

echo "CKPT_DIR=$CKPT_DIR" | tee -a "$MASTER"
echo "DATA=$DATA"         | tee -a "$MASTER"
echo "OUTDIR=$OUTDIR"     | tee -a "$MASTER"
echo "BS=$BS NUM_WORKERS=$NUM_WORKERS" | tee -a "$MASTER"
echo "STRICT=$STRICT PER_VAR=$PER_VAR" | tee -a "$MASTER"
echo

for ckpt in "$CKPT_DIR"/*.pt; do
  [ -e "$ckpt" ] || continue
  base=$(basename "$ckpt")
  log="$OUTDIR/${base%.pt}.log"
  echo "==> Evaluating $base"
  {
    echo "===== $base ====="
    python test_ace.py \
      --ckpt "$ckpt" \
      --data_path "$DATA" \
      --batch_size "$BS" \
      --num_workers "$NUM_WORKERS" \
      $([ "$PER_VAR" = "1" ] && echo --per_var) \
      $([ "$STRICT" = "1" ] && echo --strict_load)
    echo
  } | tee "$log" | tee -a "$MASTER"
done

echo "All done. Logs in $OUTDIR/, master log: $MASTER"
