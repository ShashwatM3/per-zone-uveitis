#!/usr/bin/env bash
# Phase 3: post-queue experiments (one change each vs r14 best), 14 epochs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/shashwat/miniconda3/envs/venv/bin/python}"
SPLIT="$ROOT/splits/canonical_split.json"
MC_CSV="$ROOT/processed_image_arrays_multiclass/zone_training_table.csv"
MC_ROOT="$ROOT/processed_image_arrays_multiclass"
LOG="$ROOT/runs/autonomous_protocol.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

retfound_weights_ready() {
  "$PY" -c "
from pathlib import Path
import torch
p = Path('$ROOT/weights/RETFound_mae_natureCFP.pth')
if not p.is_file() or p.stat().st_size < 500_000_000:
    raise SystemExit(1)
torch.load(p, map_location='cpu', weights_only=False)
" 2>/dev/null
}

run_convnext() {
  local name="$1"
  shift
  mkdir -p "$ROOT/runs/$name"
  if [[ -f "$ROOT/runs/$name/metrics.json" ]]; then
    log "SKIP $name (metrics.json exists)"
    return 0
  fi
  if pgrep -f "train_convnext.py.*${name}" >/dev/null 2>&1; then
    log "WAIT $name (already training)"
    while [[ ! -f "$ROOT/runs/$name/metrics.json" ]] && pgrep -f "train_convnext.py.*${name}" >/dev/null 2>&1; do
      sleep 30
    done
    if [[ -f "$ROOT/runs/$name/metrics.json" ]]; then
      return 0
    fi
  fi
  log "START $name"
  "$PY" "$ROOT/train_convnext.py" "$@" \
    --epochs 14 --batch-size 32 --num-workers 8 \
    --split-json "$SPLIT" \
    --output-dir "$ROOT/runs/$name" \
    --wandb-project uveitis-per-zone \
    --wandb-run-name "$name" \
    2>&1 | tee "$ROOT/runs/$name/train.log"
  "$PY" -c "
import json
m=json.load(open('runs/$name/metrics.json'))
print('$name', 'test F1', round(m['test']['macro_f1'],4))
" | tee -a "$LOG"
}

log "=== Phase 3 begin ==="

if [[ ! -f "$ROOT/runs/protocol_r19_retfound_excl_t1/metrics.json" ]] && retfound_weights_ready; then
  log "START protocol_r19_retfound_excl_t1 (local weights)"
  mkdir -p "$ROOT/runs/protocol_r19_retfound_excl_t1"
  "$PY" "$ROOT/train_retfound.py" \
    --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --loss focal --focal-gamma 3.0 --class-weighting inverse \
    --exclude-tier1 \
    --finetune "$ROOT/weights/RETFound_mae_natureCFP.pth" \
    --epochs 14 --batch-size 16 --num-workers 8 \
    --split-json "$SPLIT" \
    --output-dir "$ROOT/runs/protocol_r19_retfound_excl_t1" \
    --wandb-project uveitis-per-zone \
    --wandb-run-name protocol_r19_retfound_excl_t1 \
    2>&1 | tee "$ROOT/runs/protocol_r19_retfound_excl_t1/train.log" || log "r19 failed"
else
  log "SKIP protocol_r19 (done or weights not ready)"
fi

run_convnext protocol_r22_convnext_excl_t1_g25 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 2.5 --class-weighting inverse --exclude-tier1

run_convnext protocol_r23_convnext_excl_t1_cw_effective \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting effective --exclude-tier1

run_convnext protocol_r24_convnext_excl_t1_zone128 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 --zone-embed-dim 128

run_convnext protocol_r25_convnext_excl_t1_bbfl_g3 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 --balanced-batches

log "=== Phase 3 end ==="
