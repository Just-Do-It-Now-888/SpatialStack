#!/usr/bin/env bash
# VSI-Bench boxed last-line (spatialstack) on the published base.
# Default budget is 4096; set VSIBENCH_MAX_NEW_TOKENS to isolate prompt vs length.
#
#   bash scripts/opsd/run_vsibench_boxed_lastline_base.sh
#   VSIBENCH_MAX_NEW_TOKENS=1024 OUTPUT_ROOT=logs/eval/... bash scripts/opsd/run_vsibench_boxed_lastline_base.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_DIR="${HF_DIR:-models/Qwen3.5-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval/20260903_qwen35base_vsibench_boxed_lastline_tok1024}"
MASTER_PORT="${MASTER_PORT:-29533}"
export VSIBENCH_MAX_NEW_TOKENS="${VSIBENCH_MAX_NEW_TOKENS:-1024}"

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
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export VSIBENCH_PROTOCOL=spatialstack
export VSIBENCH_BOXED_PRIMARY=1
export MASTER_PORT

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "missing model: ${HF_DIR}/config.json" >&2
  exit 1
fi

echo "[$(date +%F\ %T)] VSI-Bench spatialstack boxed last-line"
echo "  model=${HF_DIR}"
echo "  protocol=${VSIBENCH_PROTOCOL}"
echo "  boxed_primary=${VSIBENCH_BOXED_PRIMARY}"
echo "  max_new_tokens=${VSIBENCH_MAX_NEW_TOKENS}"
echo "  output=${OUTPUT_ROOT}"

MODEL_PATH="$HF_DIR" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=${HF_DIR},use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
OUTPUT_PATH="$OUTPUT_ROOT" \
BENCHMARKS="vsibench" \
bash scripts/evaluation/eval.sh 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
