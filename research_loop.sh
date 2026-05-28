#!/usr/bin/env bash
# Protocol runner: up to N training runs with venv Python + GPU 3.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/shashwat/miniconda3/envs/venv/bin/python
export CUDA_VISIBLE_DEVICES=3
SPLIT=splits/canonical_split.json
WANDB=--wandb-project=uveitis-per-zone
BEST_F1=0.525064655449321
RUN_NUM=${1:-0}
MAX_RUN=${2:-10}

run_convnext() {
  local name=$1; shift
  mkdir -p "runs/$name"
  $PY train_convnext.py \
    --split-json "$SPLIT" \
    --output-dir "runs/$name" \
    --wandb-run-name "$name" \
    $WANDB \
    "$@" \
    2>&1 | tee "runs/$name/train.log"
}

run_clip() {
  local name=$1; shift
  mkdir -p "runs/$name"
  $PY train_clip_convnext.py \
    --split-json "$SPLIT" \
    --output-dir "runs/$name" \
    --wandb-run-name "$name" \
    $WANDB \
    "$@" \
    2>&1 | tee "runs/$name/train.log"
}

report_metrics() {
  local name=$1
  $PY -c "
import json
from pathlib import Path
m=json.load(open('runs/$name/metrics.json'))
bv=m.get('best_val') or {}
t=m.get('test') or {}
pc=bv.get('per_class') or {}
print('METRICS', '$name', m.get('best_epoch'), bv.get('macro_f1'),
      pc.get('0',{}).get('recall'), pc.get('1',{}).get('recall'),
      t.get('macro_f1'), t.get('balanced_accuracy'))
"
}

echo "Research loop: starting from run $RUN_NUM, max $MAX_RUN"
