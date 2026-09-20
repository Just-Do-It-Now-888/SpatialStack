#!/usr/bin/env bash
# SpatialStack geometry + SPAR3 nested K=1/K=2 MV-OPSD.
# ROLLOUT_NAME=hf; PYTHONPATH includes src/.
#
#   ARM=k1 SMOKE=1 bash scripts/opsd/launch_spatialstack_spar3_mvopsd.sh
#   ARM=k2 bash scripts/opsd/launch_spatialstack_spar3_mvopsd.sh
#
# GPUS pins a subset when the node is shared, e.g. GPUS=0,1,2,3,4,5 while another
# job holds 6-7. Both arms then need a batch divisible by that count, so
# TRAIN_BATCH_SIZE defaults to 48 (divisible by 6 and 8) rather than 64.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

# LESSON-041: SpatialStack / MV-OPSD training runs in sr_opsd.
# conda activate is incompatible with `set -u` (LESSON-008).
set +u
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sr_opsd
set -u

ARM="${ARM:?set ARM=k1 or k2}"
case "$ARM" in
  k1|k2) ;;
  *) echo "ARM must be k1 or k2" >&2; exit 1 ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-20260909_spatialstack_mvopsd_spar3_${ARM}}"
TRAIN_FILE="${TRAIN_FILE:-$PWD/data/mvopsd/parquet_spar3_${ARM}/main_train.parquet}"
MODEL_PATH="${MODEL_PATH:-$PWD/output/spatialstack_qwen35_train}"
# The in-training curve reads a fixed 480-row stratified slice, not the full
# 5,130 rows: HF generate costs about 53 s per batch of 8, i.e. ~9.4 h for the
# full set, which at test_freq=7 would be days of validation. Score the saved
# checkpoints on the full set offline instead.
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
  VAL_DO_SAMPLE=1 VAL_TEMPERATURE=1.0 VAL_TOP_P=0.8 VAL_N=1
  JUDGE=0
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE=1 SMOKE_VAL_FILE="$SMOKE_VAL_FILE" SMOKE_BATCH="${SMOKE_BATCH:-$NGPUS}" "${COMMON[@]}" \
    bash scripts/opsd/run_mvopsd.sh main \
      trainer.n_gpus_per_node="$NGPUS" \
      trainer.resume_mode=disable \
      actor_rollout_ref.actor.self_distillation.teacher_enable_thinking=False \
      trainer.max_actor_ckpt_to_keep=1
  exit $?
fi

# One epoch over the 4,064 matched rows: 63 steps at batch 64, 84 at batch 48.
# TEST_FREQ scales with it so both arms validate at the same sample counts.
TOTAL_STEPS="${TOTAL_STEPS:-84}"
TEST_FREQ="${TEST_FREQ:-7}"
SAVE_FREQ="${SAVE_FREQ:-$TEST_FREQ}"

"${COMMON[@]}" \
  TOTAL_STEPS="$TOTAL_STEPS" TEST_FREQ="$TEST_FREQ" SAVE_FREQ="$SAVE_FREQ" \
  bash scripts/opsd/run_mvopsd.sh main \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.self_distillation.teacher_enable_thinking=False \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
