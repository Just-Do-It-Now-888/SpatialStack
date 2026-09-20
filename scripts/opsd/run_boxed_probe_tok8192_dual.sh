#!/usr/bin/env bash
# Boxed-format probe at max_tokens=8192 for the frozen base and a post-trained checkpoint.
#
#   setsid nohup bash scripts/opsd/run_boxed_probe_tok8192_dual.sh \
#     > logs/eval/boxed_probe_tok8192/driver.log 2>&1
#
# Each model runs the boxed arm on VSI-Bench only (5,130 rows).
# Full responses: vsi_boxed/samples.jsonl
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MAX_TOKENS=8192
export POOL_MAX_MODEL_LEN=20480
export VSI_MAX_MODEL_LEN=20480
export ARMS=boxed
export SKIP_POOL=1

BASE_MODEL="${BASE_MODEL:-./models/Qwen3.5-4B}"
POST_MODEL="${POST_MODEL:-./output/20260821_mvopsd_single_llava_hound_main_global_step_175_hf}"
OUT_BASE="${OUT_BASE:-logs/eval/boxed_probe_tok8192}"

run_one() {
  local tag="$1" model="$2"
  export MODEL="$model"
  export OUT_ROOT="${OUT_BASE}/${tag}"
  echo "=== ${tag} model=${model} max_tokens=${MAX_TOKENS} out=${OUT_ROOT} $(date -Is) ==="
  bash scripts/opsd/run_boxed_probe.sh
}

mkdir -p "$OUT_BASE"
echo "dual tok8192 driver start $(date -Is)" | tee "${OUT_BASE}/status.txt"

run_one base "$BASE_MODEL"
run_one post_step175 "$POST_MODEL"

echo "ALL MODELS DONE $(date -Is)" | tee -a "${OUT_BASE}/status.txt"
