#!/usr/bin/env bash
# Continue autonomous protocol: RETFound r19 + phase3 (GPU 0, venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
PY="/home/shashwat/miniconda3/envs/venv/bin/python"
SPLIT="$ROOT/splits/canonical_split.json"
MC_CSV="$ROOT/processed_image_arrays_multiclass/zone_training_table.csv"
MC_ROOT="$ROOT/processed_image_arrays_multiclass"
WT="$ROOT/weights/RETFound_mae_natureCFP.pth"
LOG="$ROOT/runs/autonomous_continue.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

run_convnext() {
  local name="$1"
  shift
  mkdir -p "$ROOT/runs/$name"
  if [[ -f "$ROOT/runs/$name/metrics.json" ]]; then
    log "SKIP $name"
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
  "$PY" -c "import json; m=json.load(open('runs/$name/metrics.json')); print('$name test F1', round(m['test']['macro_f1'],4))" | tee -a "$LOG"
}

log "=== autonomous_continue begin ==="
log "Best baseline: protocol_r14 test F1=0.5599"

# Branch A2: RETFound (priority)
if [[ ! -f "$ROOT/runs/protocol_r19_retfound_excl_t1/metrics.json" ]]; then
  log "START protocol_r19_retfound_excl_t1"
  mkdir -p "$ROOT/runs/protocol_r19_retfound_excl_t1"
  rm -f "$ROOT/runs/protocol_r19_retfound_excl_t1/train.log"
  "$PY" "$ROOT/train_retfound.py" \
    --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --loss focal --focal-gamma 3.0 --class-weighting inverse \
    --exclude-tier1 \
    --finetune "$WT" \
    --epochs 14 --batch-size 16 --num-workers 8 \
    --split-json "$SPLIT" \
    --output-dir "$ROOT/runs/protocol_r19_retfound_excl_t1" \
    --wandb-project uveitis-per-zone \
    --wandb-run-name protocol_r19_retfound_excl_t1 \
    2>&1 | tee "$ROOT/runs/protocol_r19_retfound_excl_t1/train.log"
  "$PY" -c "import json; m=json.load(open('runs/protocol_r19_retfound_excl_t1/metrics.json')); print('r19 test F1', round(m['test']['macro_f1'],4))" | tee -a "$LOG"
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

# Threshold sweep on current best checkpoint (protocol step D2)
BEST_RUN=$( "$PY" -c "
import json
from pathlib import Path
best_f1, best_name = -1.0, None
for p in Path('runs').glob('protocol_r*/metrics.json'):
    m = json.load(p.open())
    f1 = m['test']['macro_f1']
    if f1 > best_f1:
        best_f1, best_name = f1, p.parent.name
print(best_name or 'protocol_r14_convnext_exclude_tier1')
")
CKPT="$ROOT/runs/$BEST_RUN/best.pt"
if [[ -f "$CKPT" ]]; then
  log "eval_convnext_sweep on $BEST_RUN"
  "$PY" "$ROOT/eval_convnext_sweep.py" \
    --checkpoint "$CKPT" \
    --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --split-json "$SPLIT" \
    --output-json "$ROOT/runs/$BEST_RUN/sweep_results.json" \
    2>&1 | tee -a "$LOG" || true
fi

log "=== autonomous_continue end ==="
