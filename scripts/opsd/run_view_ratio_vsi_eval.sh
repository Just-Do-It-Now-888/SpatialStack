#!/usr/bin/env bash
# Merge selected MV-OPSD checkpoints and run VSI-Bench @ {1,2,4,8}.
#
#   # node-A (khalf)
#   EXPERIMENT=20260820_qwen35base_mvopsd_khalf_main STEPS="45 120 240" \
#     RUN_BASELINE=1 setsid nohup bash scripts/opsd/run_view_ratio_vsi_eval.sh \
#     > logs/eval/_launch/khalf_vsi.log 2>&1 &
#
#   # node-B (kquarter)
#   EXPERIMENT=20260820_qwen35base_mvopsd_kquarter_main STEPS="75 180 255" \
#     setsid nohup bash scripts/opsd/run_view_ratio_vsi_eval.sh \
#     > logs/eval/_launch/kquarter_vsi.log 2>&1 &

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Eval must run inside sr_opsd; ~/.local/bin/accelerate shadows the env binary.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-sr_opsd}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$EVAL_CONDA_ENV"
  set -u
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi
if ! python -c "import lmms_eval, accelerate" >/dev/null 2>&1; then
  echo "sr_opsd eval env not ready (need lmms_eval + accelerate)" >&2
  exit 1
fi

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
STEPS="${STEPS:?set STEPS (space-separated global steps)}"
FRAMES="${FRAMES:-32}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen3.5-4B}"
RUN_BASELINE="${RUN_BASELINE:-0}"

mkdir -p logs/eval/_launch

status=0

if [[ "$RUN_BASELINE" == "1" ]]; then
  echo "=== baseline VSI anchor: ${BASE_MODEL} ($(date +%F\ %T)) ==="
  if ! FRAMES="$FRAMES" bash scripts/opsd/eval_frame_budget.sh \
      "$BASE_MODEL" "20260820_qwen35base_vsi_anchor"; then
    status=1
  fi
fi

for step in $STEPS; do
  step_dir="checkpoints/${EXPERIMENT}/global_step_${step}"
  hf_dir="output/${EXPERIMENT}_global_step_${step}_hf"
  run_name="${EXPERIMENT}_step${step}"

  if [[ ! -d "${step_dir}/actor" ]]; then
    echo "MISSING checkpoint: ${step_dir}/actor" >&2
    status=1
    continue
  fi

  echo "=== merge ${step_dir} ($(date +%F\ %T)) ==="
  if ! bash scripts/opsd/merge_mvopsd_ckpt.sh "$step_dir" "$hf_dir"; then
    status=1
    continue
  fi

  echo "=== VSI eval ${run_name} @ {${FRAMES}} ($(date +%F\ %T)) ==="
  if ! FRAMES="$FRAMES" bash scripts/opsd/eval_frame_budget.sh "$hf_dir" "$run_name"; then
    status=1
  fi
done

echo "VSI_EVAL_DONE experiment=${EXPERIMENT} steps='${STEPS}' status=${status} at=$(date +%F\ %T)"
exit "$status"
