#!/usr/bin/env bash
# 3-fold patient-level CV — protocol_r14 config (ConvNeXt, exclude-tier1, focal g=3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU
PY="/home/shashwat/miniconda3/envs/venv/bin/python"
LOG="$ROOT/runs/protocol_cv3_r14_config/cv.log"
mkdir -p "$ROOT/runs/protocol_cv3_r14_config"
exec "$PY" "$ROOT/run_patient_cv.py" \
  --folds 3 \
  --epochs 20 \
  --output-dir "$ROOT/runs/protocol_cv3_r14_config" \
  2>&1 | tee -a "$LOG"
