#!/usr/bin/env bash
# Frame-budget curve for one model: VSI-Bench at several max_num_frames values.
#
#   scripts/opsd/eval_frame_budget.sh output/spatialstack_qwen35_novggt_aligned p0_7_baseline
#   FRAMES="1 2 4" scripts/opsd/eval_frame_budget.sh output/<exp>_hf <exp>
#
# The MV-OPSD claim is about the left half of this curve (低帧预算), so a single
# 32-frame number cannot confirm or refute it. MAX_PIXELS lets P0-8 compare the
# evaluation geometry (default, 300 tokens/frame on ScanNet) against the training
# geometry (196608 -> 192 tokens/frame).

# Deliberately not `set -e`: this runs unattended for hours, and one frame budget
# failing should not throw away the budgets that would have run after it.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${1:?usage: eval_frame_budget.sh <model_dir> [run_name]}"
RUN_NAME="${2:-$(basename "$MODEL")}"
FRAMES="${FRAMES:-1 2 4 8 16 32}"
MAX_PIXELS="${MAX_PIXELS:-}"
BENCHMARKS="${BENCHMARKS:-vsibench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval/${RUN_NAME}}"

# LESSON-003: offline flags inherited from training block evaluation downloads.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "no config.json in ${MODEL}; merge the checkpoint first (LESSON-002)" >&2
  exit 1
fi

failed=()
for frames in $FRAMES; do
  args="pretrained=${MODEL},use_flash_attention_2=true,max_num_frames=${frames},max_length=12800,disable_thinking=true"
  suffix="frames_${frames}"
  if [[ -n "$MAX_PIXELS" ]]; then
    args="${args},max_pixels=${MAX_PIXELS}"
    suffix="${suffix}_mp${MAX_PIXELS}"
  fi
  echo "=== ${RUN_NAME}: ${BENCHMARKS} @ ${frames} frames ($(date +%H:%M:%S)) ==="
  if ! MODEL_PATH="$MODEL" \
       MODEL_IMPL=qwen3_5 \
       MODEL_ARGS_BASE="$args" \
       OUTPUT_ROOT="${OUTPUT_ROOT}/${suffix}" \
       BENCHMARKS="$BENCHMARKS" \
       bash scripts/evaluation/eval.sh; then
    echo "FAILED: ${RUN_NAME} @ ${frames} frames" >&2
    failed+=("$frames")
  fi
done

if ((${#failed[@]})); then
  echo "frame budgets that failed for ${RUN_NAME}: ${failed[*]}" >&2
  exit 1
fi
echo "frame-budget curve complete for ${RUN_NAME}: ${FRAMES}"
