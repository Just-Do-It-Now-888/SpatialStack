#!/usr/bin/env bash
# Full lmms-eval generalization for one SPAR3 best checkpoint.
#
#   ARM=k1 bash scripts/opsd/run_spar3_best_generalization_eval.sh
#   ARM=k2 bash scripts/opsd/run_spar3_best_generalization_eval.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ARM="${ARM:?set ARM=k1 or ARM=k2}"
case "$ARM" in
  k1)
    HF_DIR="output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf"
    OUTPUT_ROOT="logs/eval/20260831_mvopsd_spar3_k1_best_generalization"
    ;;
  k2)
    HF_DIR="output/20260831_mvopsd_spar3_k2_epoch1_nothink_global_step_15_hf"
    OUTPUT_ROOT="logs/eval/20260831_mvopsd_spar3_k2_best_generalization"
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
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTHONPATH
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/datasets}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "missing merged model: ${HF_DIR}/config.json" >&2
  exit 1
fi

echo "[$(date +%F\ %T)] starting generalization eval ARM=${ARM}"
echo "  model=${HF_DIR}"
echo "  output=${OUTPUT_ROOT}"

MODEL_PATH="$HF_DIR" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=${HF_DIR},use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
BENCHMARKS="cvbench,blink_spatial,sparbench,vsibench" \
bash scripts/evaluation/eval.sh 2>&1 | tee "${OUTPUT_ROOT}/run.log"

status=${PIPESTATUS[0]}
echo "[$(date +%F\ %T)] finished ARM=${ARM} status=${status}"
exit "$status"
