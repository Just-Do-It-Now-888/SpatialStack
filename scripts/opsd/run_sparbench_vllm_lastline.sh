#!/usr/bin/env bash
# SPAR-Bench vLLM lastline (no boxed), greedy @2048, TP=8.
#
#   ARM=base   bash scripts/opsd/run_sparbench_vllm_lastline.sh
#   ARM=step55 bash scripts/opsd/run_sparbench_vllm_lastline.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

ARM="${ARM:?set ARM=base or ARM=step55}"
case "$ARM" in
  base)
    HF_DIR="models/Qwen3.5-4B"
    OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval/20260906_sparbench_vllm_lastline/base}"
    ;;
  step55)
    HF_DIR="output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf"
    OUTPUT_ROOT="${OUTPUT_ROOT:-logs/eval/20260906_sparbench_vllm_lastline/spar3_step55}"
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
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SPARBENCH_PROTOCOL=lastline
export SPARBENCH_MAX_NEW_TOKENS="${SPARBENCH_MAX_NEW_TOKENS:-2048}"

if [[ ! -f "${HF_DIR}/config.json" ]]; then
  echo "missing model: ${HF_DIR}/config.json" >&2
  exit 1
fi

if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "preflight failed: vllm missing in sr_opsd" >&2
  python -c "import vllm"
  exit 1
fi

echo "[$(date +%F\ %T)] SPAR-Bench vLLM lastline ARM=${ARM}"
echo "  model=${HF_DIR}"
echo "  output=${OUTPUT_ROOT}"

python scripts/opsd/tools/eval_sparbench_vllm.py \
  --model "$HF_DIR" \
  --protocol lastline \
  --max-tokens "${SPARBENCH_MAX_NEW_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --tensor-parallel-size "${TP:-8}" \
  --output-dir "$OUTPUT_ROOT" \
  2>&1 | tee -a "${OUTPUT_ROOT}/run.log"

status=${PIPESTATUS[0]}
if [[ ! -f "${OUTPUT_ROOT}/summary.json" ]]; then
  echo "missing ${OUTPUT_ROOT}/summary.json" >&2
  exit 1
fi
echo "[$(date +%F\ %T)] finished ARM=${ARM} status=${status}"
exit "$status"
