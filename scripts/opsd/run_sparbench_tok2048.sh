#!/usr/bin/env bash
# SPAR-Bench at max_new_tokens=2048 (until=[]). One arm per call.
#
#   ARM=base   bash scripts/opsd/run_sparbench_tok2048.sh
#   ARM=step55 bash scripts/opsd/run_sparbench_tok2048.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ARM="${ARM:?set ARM=base or ARM=step55}"
case "$ARM" in
  base)
    HF_DIR="models/Qwen3.5-4B"
    OUTPUT_ROOT="logs/eval/20260902_qwen35base_sparbench_tok2048"
    ;;
  step55)
    HF_DIR="output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf"
    OUTPUT_ROOT="logs/eval/20260902_spar3_k1_step55_sparbench_tok2048"
    ;;
  *)
    echo "unknown ARM=$ARM" >&2
    exit 1
    ;;
esac

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
# Cache is already on disk; do not handshake Hub (LESSON-003 reverse).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/sparbench_cache}"
export TOKENIZERS_PARALLELISM=false
export SPARBENCH_MAX_NEW_TOKENS="${SPARBENCH_MAX_NEW_TOKENS:-2048}"

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "missing model: ${HF_DIR}/config.json" >&2
  exit 1
fi

echo "[$(date +%F\ %T)] SPAR-Bench tok${SPARBENCH_MAX_NEW_TOKENS} ARM=${ARM}"
echo "  model=${HF_DIR}"
echo "  output=${OUTPUT_ROOT}"

MODEL_PATH="$HF_DIR" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=${HF_DIR},use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
OUTPUT_PATH="$OUTPUT_ROOT" \
BENCHMARKS="sparbench" \
MASTER_PORT="${MASTER_PORT:-29540}" \
bash scripts/evaluation/eval.sh 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"

status=${PIPESTATUS[0]}
echo "[$(date +%F\ %T)] finished ARM=${ARM} status=${status}"
exit "$status"
