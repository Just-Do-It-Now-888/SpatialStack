#!/usr/bin/env bash
# Wait for a running MV-OPSD training job, then merge its final checkpoint and
# measure the VSI-Bench frame-budget curve for both the trained model and the
# SFT baseline it started from.
#
#   scripts/opsd/after_train_eval.sh <train_pid> [experiment_name] [final_step]
#
# The baseline curve is not optional. MV-OPSD claims an advantage at low frame
# budgets, and the only baseline number we have is 64.24 at 32 frames, so
# without the left half of the baseline curve the trained model's numbers cannot
# be read as better or worse than the starting point.
#
# Sentinels (grep-able, and used to wake the agent):
#   MVOPSD_CHAIN_TRAIN_EXITED, MVOPSD_CHAIN_ABORT, MVOPSD_CHAIN_MERGED,
#   MVOPSD_CHAIN_EVAL_DONE

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_PID="${1:?usage: after_train_eval.sh <train_pid> [experiment_name] [final_step]}"
EXPERIMENT_NAME="${2:-20260816_qwen35_mvopsd_v0_main}"
FINAL_STEP="${3:-300}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/output/spatialstack_qwen35_novggt_aligned}"
FRAMES="${FRAMES:-1 2 4 8 16 32}"

STEP_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}/global_step_${FINAL_STEP}"
HF_DIR="${PROJECT_ROOT}/output/${EXPERIMENT_NAME}_global_step_${FINAL_STEP}_hf"

echo "waiting for training pid ${TRAIN_PID} ($(date +%F\ %T))"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done
echo "MVOPSD_CHAIN_TRAIN_EXITED {\"pid\":${TRAIN_PID},\"at\":\"$(date +%F\ %T)\"}"

# Give the checkpoint writer a moment to flush after the process exits.
sleep 30

if [[ ! -d "${STEP_DIR}/actor" ]]; then
  echo "MVOPSD_CHAIN_ABORT {\"reason\":\"no final checkpoint\",\"expected\":\"${STEP_DIR}/actor\"}"
  echo "training did not reach step ${FINAL_STEP}; not evaluating a partial run" >&2
  ls -1 "${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}" 2>/dev/null >&2
  exit 1
fi

echo "=== merging ${STEP_DIR} ($(date +%T))"
if ! bash scripts/opsd/merge_mvopsd_ckpt.sh "$STEP_DIR" "$HF_DIR"; then
  echo "MVOPSD_CHAIN_ABORT {\"reason\":\"merge failed\",\"step_dir\":\"${STEP_DIR}\"}"
  exit 1
fi
echo "MVOPSD_CHAIN_MERGED {\"model\":\"${HF_DIR}\"}"

status=0
echo "=== frame-budget curve: trained model ($(date +%T))"
FRAMES="$FRAMES" bash scripts/opsd/eval_frame_budget.sh "$HF_DIR" "${EXPERIMENT_NAME}" || status=1

echo "=== frame-budget curve: SFT baseline ($(date +%T))"
FRAMES="$FRAMES" bash scripts/opsd/eval_frame_budget.sh "$BASE_MODEL" "sft_baseline_frame_curve" || status=1

echo "MVOPSD_CHAIN_EVAL_DONE {\"status\":${status},\"at\":\"$(date +%F\ %T)\"}"
exit "$status"
