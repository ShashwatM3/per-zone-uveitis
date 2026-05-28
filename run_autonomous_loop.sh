#!/usr/bin/env bash
# Non-stop protocol runner: phase3 → CV5 → further sweeps until stop condition.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
PY="/home/shashwat/miniconda3/envs/venv/bin/python"
LOG="$ROOT/runs/autonomous_loop.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

free_disk() {
  rm -rf "$ROOT"/wandb/run-* 2>/dev/null || true
  find "$ROOT/runs" -name "checkpoint_epoch*.pt" -delete 2>/dev/null || true
}

log "=== autonomous_loop start ==="
free_disk

# Phase 3 (skip if done) + resume queue (r26+, CV, ceiling check)
if [[ ! -f "$ROOT/runs/protocol_r25_convnext_excl_t1_bbfl_g3/metrics.json" ]]; then
  bash "$ROOT/protocol_autonomous_phase3.sh" >> "$LOG" 2>&1
fi
bash "$ROOT/run_autonomous_resume.sh" >> "$LOG" 2>&1

log "=== autonomous_loop end ==="
