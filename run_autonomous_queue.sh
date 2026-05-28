#!/usr/bin/env bash
# Sequential protocol runs (GPU 0, venv Python, 14 epochs max).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
PY="${PY:-/home/shashwat/miniconda3/envs/venv/bin/python}"
SPLIT="splits/canonical_split.json"
MC_CSV="processed_image_arrays_multiclass/zone_training_table.csv"
MC_ROOT="processed_image_arrays_multiclass"

run_train() {
  local out="$1"
  shift
  mkdir -p "runs/${out}"
  echo "=== Starting ${out} at $(date -Is) ==="
  "$PY" train_convnext.py "$@" \
    --epochs 14 --batch-size 32 --num-workers 8 \
    --split-json "$SPLIT" \
    --output-dir "runs/${out}" \
    --wandb-project uveitis-per-zone \
    --wandb-run-name "${out}" \
    2>&1 | tee "runs/${out}/train.log"
  echo "=== Finished ${out} at $(date -Is) ==="
}

wait_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
}

# Usage: called after manual r15 or standalone
if [[ "${1:-}" == "from_r16" ]]; then
  run_train protocol_r16_convnext_bbfl_excl_t1 \
    --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --loss focal --focal-gamma 1.5 --class-weighting none \
    --exclude-tier1 --balanced-batches

  run_train protocol_r17_convnext_tier_conf_wt \
    --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --loss focal --focal-gamma 3.0 --class-weighting inverse \
    --tier-confidence-weights

  run_train protocol_r18_efficientnet_excl_t1 \
    --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --backbone efficientnet_b3 \
    --loss focal --focal-gamma 3.0 --class-weighting inverse \
    --exclude-tier1
fi
