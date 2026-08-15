#!/usr/bin/env bash
# Launch an MV-OPSD v0 training arm.
#
#   scripts/opsd/run_mvopsd.sh main            # teacher sees all N views
#   scripts/opsd/run_mvopsd.sh noprivilege     # teacher sees exactly the student's K views
#   SMOKE=1 scripts/opsd/run_mvopsd.sh main    # 2 steps on a tiny slice (P0-5)
#
# Both arms read parquet files generated from the same plan, so the only
# difference between them is the teacher's view set.

set -euo pipefail

ARM="${1:-main}"
shift || true
case "$ARM" in
  main|noprivilege) ;;
  *) echo "unknown arm: $ARM (expected main or noprivilege)" >&2; exit 1 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERL_ROOT="${PROJECT_ROOT}/verl_pkg"
CONFIG_DIR="${VERL_ROOT}/verl/trainer/config"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-20260816_qwen35_mvopsd_v0_${ARM}}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/mvopsd/parquet/${ARM}_train.parquet}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/output/spatialstack_qwen35_novggt_aligned}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"

MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=512
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

EXTRA_ARGS=()

# The smoke suffix has to land before the output paths are derived, or a throwaway
# run overwrites the real experiment's checkpoints and rollouts (LESSON-004).
if [[ "${SMOKE:-0}" == "1" ]]; then
  TOTAL_STEPS=2
  TRAIN_BATCH_SIZE="${SMOKE_BATCH:-8}"
  ROLLOUT_N=2
  SAVE_FREQ=-1
  # resume_mode=auto would pick up a leftover smoke checkpoint and continue from
  # its step count instead of validating the pipeline from scratch.
  EXTRA_ARGS+=(trainer.resume_mode=disable)
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
  echo "SMOKE mode: ${TOTAL_STEPS} steps, batch ${TRAIN_BATCH_SIZE}, n=${ROLLOUT_N}"
fi

CKPT_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/train/${EXPERIMENT_NAME}"
ROLLOUT_DIR="${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$ROLLOUT_DIR"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "train file not found: $TRAIN_FILE (run the stage 1-3 scripts first)" >&2
  exit 1
fi

# verl is also installed from ~/Documents/code/Vision-OPD-main via a .pth entry.
# Put this repo's copy first so the run matches the tree we version here.
export PYTHONPATH="${VERL_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

echo "arm=${ARM} experiment=${EXPERIMENT_NAME}"
echo "train_file=${TRAIN_FILE}"
echo "model=${MODEL_PATH}"
echo "steps=${TOTAL_STEPS} batch=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N}"

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name mvopsd \
    data.train_files="[\"${TRAIN_FILE}\"]" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
    critic.model.path="$MODEL_PATH" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.group_name=mvopsd_v0 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@" 2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
