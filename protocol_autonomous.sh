#!/usr/bin/env bash
# Autonomous protocol runner — GPU 0, venv Python, 14 epochs max per run.
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

run_convnext() {
  local name="$1"
  shift
  mkdir -p "$ROOT/runs/$name"
  if [[ -f "$ROOT/runs/$name/metrics.json" ]]; then
    log "SKIP $name (metrics.json exists)"
    return 0
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

run_retfound() {
  local name="$1"
  shift
  mkdir -p "$ROOT/runs/$name"
  if [[ -f "$ROOT/runs/$name/metrics.json" ]]; then
    log "SKIP $name (metrics.json exists)"
    return 0
  fi
  if ! retfound_weights_ready; then
    log "SKIP $name (RETFound weights not ready at weights/RETFound_mae_natureCFP.pth)"
    return 0
  fi
  log "START $name (RETFound)"
  if ! "$PY" "$ROOT/train_retfound.py" "$@" \
    --finetune "$ROOT/weights/RETFound_mae_natureCFP.pth" \
    --epochs 14 --batch-size 16 --num-workers 8 \
    --split-json "$SPLIT" \
    --output-dir "$ROOT/runs/$name" \
    --wandb-project uveitis-per-zone \
    --wandb-run-name "$name" \
    2>&1 | tee "$ROOT/runs/$name/train.log"; then
    log "RETFound run $name failed"
    return 0
  fi
  "$PY" -c "
import json
m=json.load(open('runs/$name/metrics.json'))
print('$name', 'test F1', round(m['test']['macro_f1'],4))
" | tee -a "$LOG"
}

log "=== Autonomous protocol batch begin ==="

# Branch B2: BBFL-style balanced batches (one change vs r14: sampler + gamma 1.5)
run_convnext protocol_r16_convnext_bbfl_excl_t1 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 1.5 --class-weighting none \
  --exclude-tier1 --balanced-batches

# Branch C2b: tier confidence weights (one change vs r11: loss weighting not exclusion)
run_convnext protocol_r17_convnext_tier_conf_wt \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --tier-confidence-weights

# Branch A3 fallback: EfficientNet-B3 (one change: backbone)
run_convnext protocol_r18_efficientnet_excl_t1 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --backbone efficientnet_b3 \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1

# Branch A2: RETFound MAE CFP (local weights/RETFound_mae_natureCFP.pth when complete)
run_retfound protocol_r19_retfound_excl_t1 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1

# Re-run best ConvNeXt config with 14 epochs (sanity vs 20-epoch r14)
run_convnext protocol_r20_convnext_excl_t1_14ep \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1

# Branch D4: MixUp (Galdrán MICCAI 2021 Balanced-MixUp family; alpha=0.2)
run_convnext protocol_r21_convnext_mixup_excl_t1 \
  --csv "$MC_CSV" --data-root "$MC_ROOT" \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 --mixup-alpha 0.2

log "=== Autonomous protocol batch end ==="
