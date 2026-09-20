#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/eval/20260816_qwen35base_mvopsd_v0_main
pkill -f AGENT_LOOP_TICK_mvopsd_base 2>/dev/null || true
nohup bash scripts/opsd/after_train_eval.sh 582080 20260816_qwen35base_mvopsd_v0_main 300 \
  > logs/eval/20260816_qwen35base_mvopsd_v0_main/after_train_eval.log 2>&1 &
echo "launched after_train_eval PID=$!"
