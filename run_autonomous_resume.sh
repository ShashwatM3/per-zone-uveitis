#!/usr/bin/env bash
# Resume after phase3 (r22-r25): r26+ → CV → sweep → more experiments until stop.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
PY="/home/shashwat/miniconda3/envs/venv/bin/python"
LOG="$ROOT/runs/autonomous_loop.log"
MC_CSV="$ROOT/processed_image_arrays_multiclass/zone_training_table.csv"
MC_ROOT="$ROOT/processed_image_arrays_multiclass"
SPLIT="$ROOT/splits/canonical_split.json"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_disk() {
  rm -rf "$ROOT"/wandb/run-* 2>/dev/null || true
  find "$ROOT/runs" -maxdepth 2 -name "checkpoint_epoch*.pt" -delete 2>/dev/null || true
}

run_one() {
  local name="$1"
  local epochs="${2:-14}"
  shift 2
  if [[ -f "$ROOT/runs/$name/metrics.json" ]]; then
    log "SKIP $name"
    return 0
  fi
  if pgrep -f "train_convnext.py.*${name}" >/dev/null 2>&1; then
    log "WAIT $name"
    while [[ ! -f "$ROOT/runs/$name/metrics.json" ]] && pgrep -f "train_convnext.py.*${name}" >/dev/null 2>&1; do
      sleep 45
    done
    return 0
  fi
  free_disk
  log "START $name (${epochs} ep)"
  mkdir -p "$ROOT/runs/$name"
  "$PY" "$ROOT/train_convnext.py" "$@" \
    --epochs "$epochs" --batch-size 32 --num-workers 8 \
    --split-json "$SPLIT" \
    --output-dir "$ROOT/runs/$name" \
    --wandb-project uveitis-per-zone \
    --wandb-run-name "$name" \
    2>&1 | tee "$ROOT/runs/$name/train.log"
  "$PY" -c "import json; m=json.load(open('runs/$name/metrics.json')); print('$name test F1', round(m['test']['macro_f1'],4))" | tee -a "$LOG"
}

log "=== autonomous_resume start ==="
BASE=(--csv "$MC_CSV" --data-root "$MC_ROOT" --loss focal --focal-gamma 3.0 --class-weighting inverse --exclude-tier1)

# Phase 3 follow-ups (protocol loop)
run_one protocol_r26_convnext_excl_t1_g35 14 "${BASE[@]}" --focal-gamma 3.5
run_one protocol_r27_convnext_excl_t1_lr5e5 14 "${BASE[@]}" --lr 5e-5
run_one protocol_r28_convnext_excl_t1_20ep 20 "${BASE[@]}"

# Threshold + TTA on best (protocol D2/D3)
BEST=$("$PY" -c "
import json
from pathlib import Path
best_f1, best_name = -1.0, 'protocol_r14_convnext_exclude_tier1'
for p in Path('runs').glob('protocol_r*/metrics.json'):
    m = json.load(p.open())
    f1 = m['test']['macro_f1']
    if f1 > best_f1:
        best_f1, best_name = f1, p.parent.name
print(best_name)
")
CKPT="$ROOT/runs/$BEST/best.pt"
if [[ -f "$CKPT" && ! -f "$ROOT/runs/$BEST/sweep_results.json" ]]; then
  log "eval_convnext_sweep on $BEST"
  "$PY" "$ROOT/eval_convnext_sweep.py" \
    --checkpoint "$CKPT" --csv "$MC_CSV" --data-root "$MC_ROOT" \
    --split-json "$SPLIT" \
    --output-json "$ROOT/runs/$BEST/sweep_results.json" 2>&1 | tee -a "$LOG" || true
fi

# 5-fold patient CV (protocol D6)
if [[ ! -f "$ROOT/runs/protocol_cv5_r14_config/cv_summary.json" ]]; then
  log "START 5-fold CV"
  free_disk
  WANDB_MODE=offline "$PY" "$ROOT/run_patient_cv.py" --epochs 14 2>&1 | tee -a "$LOG" || log "CV failed (see log)"
fi

# Ceiling check
"$PY" "$ROOT/scripts/check_protocol_ceiling.py" 2>&1 | tee -a "$LOG" || true

log "=== autonomous_resume end ==="
