#!/usr/bin/env bash
# Periodic status for protocol_r26_retfound_correct_recipe
OUT="/home/shashwat/per-zone-uveitis/runs/protocol_r26_retfound_correct_recipe"
LOG="$OUT/monitor_status.log"
INTERVAL="${1:-300}"

while true; do
  {
    echo "======== $(date -Is) ========"
    df -h / | awk 'NR==2{printf "disk: %s used %s avail (%s)\n", $3, $4, $5}'
    du -sh /home/shashwat/per-zone-uveitis/runs "$OUT" 2>/dev/null
    if pgrep -f "train_retfound.*protocol_r26_retfound_correct_recipe" >/dev/null; then
      echo "training: RUNNING"
    else
      echo "training: STOPPED"
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | awk -F, '{print "gpu0:", $2, "mem", $3}'
    if [ -f "$OUT/metrics.json" ]; then
      /home/shashwat/miniconda3/envs/venv/bin/python -c "
import json
m=json.load(open('$OUT/metrics.json'))
t=m['test']
print('FINISHED test_f1', round(t['macro_f1'],4), 'c1_rec', round(t['per_class']['1']['recall'],4), 'best_ep', m.get('best_epoch'))
"
      break
    fi
    grep -E '^epoch=' "$OUT/train.log" 2>/dev/null | grep train_loss | tail -1
    ls -lh "$OUT/best.pt" 2>/dev/null || echo "best.pt: not yet"
  } >> "$LOG"
  sleep "$INTERVAL"
done
