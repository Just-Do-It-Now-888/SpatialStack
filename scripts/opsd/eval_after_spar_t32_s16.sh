#!/usr/bin/env bash
# After 20260915 SPAR T32/S16 MV-OPSD exits, merge the latest complete FSDP
# actor and run full VSI boxed last-line (HF/lmms-eval).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
set +u
# shellcheck disable=SC1090
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate sr_opsd
set -u

CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/checkpoints/20260915_spatialstack_mvopsd_spar_t32_s16}"
BASE_MODEL="${BASE_MODEL:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
EVAL_PARENT="${EVAL_PARENT:-$PROJECT_ROOT/logs/eval/20260915_spatialstack_mvopsd_spar_t32_s16}"

latest=""
latest_n=-1
for dir in "$CKPT_ROOT"/global_step_*; do
  [[ -d "$dir" ]] || continue
  [[ -d "$dir/actor" ]] || continue
  n="${dir##*_}"
  if (( n > latest_n )); then
    latest_n=$n
    latest=$dir
  fi
done
if [[ -z "$latest" ]]; then
  echo "no complete actor checkpoint under $CKPT_ROOT" >&2
  exit 1
fi

HF_DIR="$PROJECT_ROOT/output/20260915_spatialstack_mvopsd_spar_t32_s16_global_step_${latest_n}_hf"
EVAL_ROOT="$EVAL_PARENT/step${latest_n}_vsibench_boxed"
mkdir -p "$EVAL_ROOT"

if [[ ! -f "$HF_DIR/config.json" ]]; then
  echo "[$(date +%F\ %T)] merge $latest -> $HF_DIR"
  BASE_MODEL="$BASE_MODEL" bash "$PROJECT_ROOT/scripts/opsd/merge_mvopsd_ckpt.sh" "$latest" "$HF_DIR"
fi

echo "[$(date +%F\ %T)] eval boxed $HF_DIR -> $EVAL_ROOT"
CKPT="$HF_DIR" EVAL_ROOT="$EVAL_ROOT" \
  bash "$PROJECT_ROOT/scripts/opsd/eval_spatialstack_geometry_vsi.sh" boxed \
  >"$EVAL_ROOT/chain.log" 2>&1
