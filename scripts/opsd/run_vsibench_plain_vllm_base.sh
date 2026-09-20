#!/usr/bin/env bash
# Base VSI-Bench original prompt through the vLLM harness (not lmms-eval).
# Matches spatialstack_plain: no boxed suffix, greedy 1024, answer_tail.
# Use conda sr_opsd (has both vllm and decord). System python has vllm but
# no decord (ISSUE-001 of 20260903_qwen35base_vsibench_plain_vllm).
#
#   bash scripts/opsd/run_vsibench_plain_vllm_base.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_DIR="${HF_DIR:-models/Qwen3.5-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval/20260903_qwen35base_vsibench_plain_vllm}"
mkdir -p "$OUTPUT_ROOT"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate sr_opsd
  set -u
fi

export PATH="${CONDA_PREFIX:-}/bin:${PATH}"
unset PYTHONPATH
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VSIBENCH_BOXED_PRIMARY=0

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "missing model: ${HF_DIR}/config.json" >&2
  exit 1
fi

if ! python -c "import vllm, decord" >/dev/null 2>&1; then
  echo "preflight failed: conda sr_opsd must import both vllm and decord" >&2
  python -c "import vllm, decord"
  exit 1
fi

echo "[$(date +%F\ %T)] VSI-Bench vLLM spatialstack_plain base"
echo "  python=$(command -v python)"
python -c "import vllm, decord; print('  vllm', vllm.__version__, 'decord', decord.__version__)"
echo "  model=${HF_DIR}"
echo "  output=${OUTPUT_ROOT}"

python scripts/opsd/tools/eval_vsibench_vllm.py \
  --model "$HF_DIR" \
  --protocol spatialstack_plain \
  --frames 32 \
  --max-tokens 1024 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization "${GPU_MEM:-0.9}" \
  --scenes-per-batch "${SCENES_PER_BATCH:-8}" \
  --output-dir "$OUTPUT_ROOT" \
  2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
