#!/usr/bin/env bash
# Offline VSI-Bench eval for llava_hound single-source arm @ step 100, max_tokens=4096.
#
#   setsid nohup bash scripts/opsd/run_llava_hound_step100_vsi4096.sh \
#     > logs/eval/llava_hound_step100_tok4096/driver.log 2>&1
#
# Waits for 8 GPUs to be free (e.g. after boxed_probe_tok8192_dual), then runs
# vLLM generation + rule scoring + judge via run_training_vsibench_eval.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

CKPT="checkpoints/20260821_mvopsd_single_llava_hound_main/global_step_100"
OUT_DIR="logs/vsi_train_eval/20260821_mvopsd_single_llava_hound_main/global_step_100"
STATUS="logs/eval/llava_hound_step100_tok4096/status.txt"
mkdir -p "$(dirname "$STATUS")" "$OUT_DIR"

echo "driver start $(date -Is)" | tee "$STATUS"

echo "[wait] waiting for GPUs to free (peak < 5000 MiB)" | tee -a "$STATUS"
for i in $(seq 1 720); do
  if ! pgrep -f 'run_boxed_probe_tok8192_dual.sh|eval_vsibench_vllm.py.*boxed_probe_tok8192' >/dev/null 2>&1; then
    peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    if (( peak < 5000 )); then
      echo "[wait] GPUs free after ${i} checks; peak=${peak} MiB $(date -Is)" | tee -a "$STATUS"
      break
    fi
  fi
  if (( i % 12 == 0 )); then
    peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    echo "[wait] still busy peak=${peak} MiB ($(date -Is))" | tee -a "$STATUS"
  fi
  sleep 10
done

peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if (( peak >= 5000 )); then
  echo "GPUs still occupied (peak=${peak} MiB); aborting" | tee -a "$STATUS"
  exit 2
fi

pkill -TERM -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -TERM -f 'VLLM::Worker_TP' 2>/dev/null || true
sleep 5

echo "[eval] starting vLLM VSI-Bench max_tokens=4096 -> ${OUT_DIR}" | tee -a "$STATUS"
FRAMES=32 \
MAX_TOKENS=4096 \
MAX_MODEL_LEN=16384 \
JUDGE=1 \
SCENES_PER_BATCH=24 \
bash scripts/opsd/run_training_vsibench_eval.sh "$CKPT" \
  2>&1 | tee "${OUT_DIR}/driver.log"

if [[ -f "${OUT_DIR}/summary.json" ]]; then
  python3 - "${OUT_DIR}/summary.json" <<'PY' | tee -a "$STATUS"
import json, sys

s = json.loads(open(sys.argv[1]).read())
rule, judged = s["rule_only"], s["judge_assisted"]
print("=== VSI-Bench @%d (llava_hound step 100) ===" % s["max_tokens"])
print(f"rule-only overall : {rule['overall']:.2f}  (answered {rule['answered_pct']:.2f}%)")
print(f"judge-assisted    : {judged['overall']:.2f}  (answered {judged['answered_pct']:.2f}%)")
print(f"truncated         : {s['truncated']}/{s['questions']} ({s['truncated_pct']:.2f}%)")
print(f"median out tokens : {s['output_tokens']['median']}")
for qtype, info in sorted(s["by_question_type"].items(), key=lambda kv: -kv[1]["rule_only_score"]):
    print(f"  {qtype:<28}{info['n']:>5}{info['rule_only_score']:>8.2f}")
PY
fi

echo "ALL DONE $(date -Is)" | tee -a "$STATUS"
