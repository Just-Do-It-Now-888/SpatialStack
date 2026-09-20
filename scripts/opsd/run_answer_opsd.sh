#!/usr/bin/env bash
# Equal-view Answer-OPSD: student and teacher see the full album; only the
# teacher gets the ground-truth answer. Wraps scripts/opsd/run_mvopsd.sh and
# does not change mvopsd.yaml defaults.
#
#   SMOKE=1 bash scripts/opsd/run_answer_opsd.sh
#   bash scripts/opsd/run_answer_opsd.sh
#
# Privilege is answer_hint, not N/K views. data.max_prompt_length is raised so
# SPAR 32-view student prompts are not truncated (agent_loop error truncation).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

set +u
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sr_opsd
set -u
# shellcheck source=require_sr_opsd.sh
source "$(cd "$(dirname "$0")" && pwd)/require_sr_opsd.sh"
require_sr_opsd_training_env

EXPERIMENT_NAME="${EXPERIMENT_NAME:-20260919_spatialstack_answer_opsd_spar_fullviews}"
TRAIN_FILE="${TRAIN_FILE:-$PWD/data/mvopsd/parquet_answer_opsd_spar_fullviews/main_train.parquet}"
MODEL_PATH="${MODEL_PATH:-$PWD/output/spatialstack_qwen35_train}"
VAL_FILE="${VAL_FILE:-$PWD/data/eval/vsibench_verl/vsibench_val_boxed_lastline_sub480.parquet}"
SMOKE_VAL_FILE="${SMOKE_VAL_FILE:-$PWD/data/eval/vsibench_verl/vsibench_val_boxed_lastline_smoke8.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
# 32 views * 192 tokens plus question/answer hint. Matches teacher max_reprompt_len.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-12288}"

GPUS="${GPUS:-}"
if [[ -n "$GPUS" ]]; then
  NGPUS="$(awk -F, '{print NF}' <<<"$GPUS")"
  GPU_ENV=(CUDA_VISIBLE_DEVICES="$GPUS")
else
  NGPUS="${NGPUS:-8}"
  GPU_ENV=(-u CUDA_VISIBLE_DEVICES)
fi
if (( TRAIN_BATCH_SIZE % NGPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE is not divisible by NGPUS=$NGPUS" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "geo SFT weights missing at $MODEL_PATH (need config.json)" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "train parquet missing: $TRAIN_FILE" >&2
  echo "build with scripts/opsd/build_answer_opsd_full_views_plans.py then write_mvopsd_parquet.py --require-equal-views" >&2
  exit 1
fi

COMMON=(
  env "${GPU_ENV[@]}"
  ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
  AGENT_WORKERS="$NGPUS"
  MODEL_PATH="$MODEL_PATH"
  EXPERIMENT_NAME="$EXPERIMENT_NAME"
  TRAIN_FILE="$TRAIN_FILE"
  VAL_FILE="$VAL_FILE"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
  VAL_DO_SAMPLE=0
  JUDGE=0
)

ANSWER_OVERRIDES=(
  trainer.n_gpus_per_node="$NGPUS"
  trainer.resume_mode=disable
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH"
  actor_rollout_ref.actor.self_distillation.teacher_prompt_mode=answer_hint
  actor_rollout_ref.actor.self_distillation.teacher_enable_thinking=False
  actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher=False
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=12288
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE=1 WANDB_MODE="${WANDB_MODE:-offline}" SMOKE_VAL_FILE="$SMOKE_VAL_FILE" \
    SMOKE_BATCH="${SMOKE_BATCH:-$NGPUS}" "${COMMON[@]}" \
    bash scripts/opsd/run_mvopsd.sh main \
      "${ANSWER_OVERRIDES[@]}" \
      trainer.max_actor_ckpt_to_keep=1
  exit $?
fi

# One epoch: floor(train_rows / batch). Default assumes 54495 rows / 48, same
# holdout recipe as SPAR T32/S16; override TOTAL_STEPS after parquet is written.
TOTAL_STEPS="${TOTAL_STEPS:-1135}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-$TEST_FREQ}"

WANDB_MODE="${WANDB_MODE:-online}" "${COMMON[@]}" \
  TOTAL_STEPS="$TOTAL_STEPS" TEST_FREQ="$TEST_FREQ" SAVE_FREQ="$SAVE_FREQ" \
  bash scripts/opsd/run_mvopsd.sh main \
    "${ANSWER_OVERRIDES[@]}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-3}" \
    trainer.ckpt_keep_mode=best \
    trainer.ckpt_keep_metric=val-core/vsibench/overall/acc
