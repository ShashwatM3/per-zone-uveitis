#!/usr/bin/env bash
# Wait for r19, then phase3 (r22-r25), then optional 5-fold CV.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
PY="/home/shashwat/miniconda3/envs/venv/bin/python"
LOG="$ROOT/runs/autonomous_watcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "watcher start"

# Wait for r19 (up to 4h)
for _ in $(seq 1 240); do
  if [[ -f "$ROOT/runs/protocol_r19_retfound_excl_t1/metrics.json" ]]; then
    log "r19 complete"
    break
  fi
  if ! pgrep -f "train_retfound.py.*protocol_r19" >/dev/null 2>&1; then
    if [[ ! -f "$ROOT/runs/protocol_r19_retfound_excl_t1/metrics.json" ]]; then
      log "r19 process ended without metrics — check train.log"
      tail -30 "$ROOT/runs/protocol_r19_retfound_excl_t1/train.log" 2>/dev/null | tee -a "$LOG" || true
    fi
    break
  fi
  sleep 60
done

# Phase 3 convnext sweeps (skips r19 if done)
bash "$ROOT/protocol_autonomous_phase3.sh" 2>&1 | tee -a "$LOG"

# 5-fold CV on best convnext config (protocol D6) if not done
if [[ ! -f "$ROOT/runs/protocol_cv5_r14_config/cv_summary.json" ]]; then
  log "START 5-fold patient CV"
  "$PY" "$ROOT/run_patient_cv.py" --epochs 14 2>&1 | tee -a "$LOG"
fi

log "watcher end"
