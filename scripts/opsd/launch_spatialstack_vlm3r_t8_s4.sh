#!/usr/bin/env bash
# SpatialStack geometry + VLM-3R Teacher-8 / Student-4 MV-OPSD.
# Formal run: WANDB_MODE=online. Smoke: WANDB_MODE=offline.
#
#   SMOKE=1 bash scripts/opsd/launch_spatialstack_vlm3r_t8_s4.sh
#   bash scripts/opsd/launch_spatialstack_vlm3r_t8_s4.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

set +u
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sr_opsd
set -u

# node-C ssh is root; Python netrc rejects a user-owned ~/.netrc.
# Prefer WANDB_API_KEY already in the environment, else /root/.netrc.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for _netrc in /root/.netrc "${HOME}/.netrc"; do
    if [[ -f "$_netrc" ]]; then
      WANDB_API_KEY="$(python3 -c "import netrc,sys
a=netrc.netrc(sys.argv[1]).authenticators('api.wandb.ai')
print(a[2] if a else '')" "$_netrc" 2>/dev/null || true)"
      if [[ -n "${WANDB_API_KEY:-}" ]]; then
        export WANDB_API_KEY
        break
      fi
    fi
  done
  unset _netrc
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-20260915_spatialstack_mvopsd_vlm3r_t8_s4}"
TRAIN_FILE="${TRAIN_FILE:-$PWD/data/mvopsd/parquet_vlm3r_t8_s4/main_train.parquet}"
MODEL_PATH="${MODEL_PATH:-$PWD/output/spatialstack_qwen35_train}"
VAL_FILE="${VAL_FILE:-$PWD/data/eval/vsibench_verl/vsibench_val_boxed_lastline_sub480.parquet}"
SMOKE_VAL_FILE="${SMOKE_VAL_FILE:-$PWD/data/eval/vsibench_verl/vsibench_val_boxed_lastline_smoke8.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"

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

COMMON=(
  env "${GPU_ENV[@]}"
  ROLLOUT_NAME=hf
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

if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE=1 WANDB_MODE=offline SMOKE_VAL_FILE="$SMOKE_VAL_FILE" \
    SMOKE_BATCH="${SMOKE_BATCH:-$NGPUS}" "${COMMON[@]}" \
    bash scripts/opsd/run_mvopsd.sh main \
      trainer.n_gpus_per_node="$NGPUS" \
      trainer.resume_mode=disable \
      actor_rollout_ref.actor.self_distillation.teacher_enable_thinking=False \
      trainer.max_actor_ckpt_to_keep=1
  exit $?
fi

# One epoch: floor(30493 / 48) = 635
TOTAL_STEPS="${TOTAL_STEPS:-635}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-$TEST_FREQ}"

WANDB_MODE=online "${COMMON[@]}" \
  TOTAL_STEPS="$TOTAL_STEPS" TEST_FREQ="$TEST_FREQ" SAVE_FREQ="$SAVE_FREQ" \
  bash scripts/opsd/run_mvopsd.sh main \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.self_distillation.teacher_enable_thinking=False \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
