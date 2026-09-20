#!/usr/bin/env bash
# SPAR-Bench vLLM boxed last-line (boxed-primary), greedy @2048, TP=8.
# Serial base then SPAR3 on the same 8 GPUs.
#
#   setsid nohup bash scripts/opsd/run_sparbench_vllm_boxed_chain.sh \
#     > logs/eval/20260908_sparbench_vllm_boxed/driver.log 2>&1

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAIN_ROOT="${CHAIN_ROOT:-logs/eval/20260908_sparbench_vllm_boxed}"
STATUS="${CHAIN_ROOT}/status.txt"
mkdir -p "$CHAIN_ROOT"

MAX_TOKENS="${MAX_TOKENS:-2048}"
MIN_FREE_MIB="${MIN_FREE_MIB:-80000}"
BASE_HF="${BASE_HF:-models/Qwen3.5-4B}"
SPAR3_HF="${SPAR3_HF:-output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf}"

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
export SPARBENCH_PROTOCOL=boxed
export SPARBENCH_MAX_NEW_TOKENS="$MAX_TOKENS"

log() {
  echo "[$(date +'%F %T')] $*" | tee -a "$STATUS"
}

wait_gpus_free() {
  local waited=0
  until [[ -z "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
                 awk -v m="$MIN_FREE_MIB" '$1 < m')" ]]; do
    if ((waited >= 900)); then
      log "ERROR cards still held after ${waited}s; refusing next arm"
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$STATUS"
      exit 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
  if ((waited > 0)); then
    log "cards free after ${waited}s"
  fi
}

pause_between_arms() {
  wait_gpus_free
  sleep 20
  wait_gpus_free
}

if ! python -c "import vllm" >/dev/null 2>&1; then
  log "preflight failed: need vllm in sr_opsd"
  exit 1
fi

for model in "$BASE_HF" "$SPAR3_HF"; do
  if [[ ! -f "${model}/config.json" ]]; then
    log "missing model: ${model}/config.json"
    exit 1
  fi
done

log "chain start host=$(hostname) python=$(command -v python) protocol=boxed tok=${MAX_TOKENS}"

run_arm() {
  local name="$1"
  local model="$2"
  local out="${CHAIN_ROOT}/${name}"
  mkdir -p "$out"
  if [[ -s "${out}/summary.json" ]]; then
    log "SKIP ${name} existing ${out}/summary.json"
    return 0
  fi
  pause_between_arms
  log "START ${name} model=${model}"
  set +e
  python scripts/opsd/tools/eval_sparbench_vllm.py \
    --model "$model" \
    --protocol boxed \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --tensor-parallel-size "${TP:-8}" \
    --output-dir "$out" \
    2>&1 | tee -a "${out}/run.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  if [[ "$rc" -ne 0 || ! -s "${out}/summary.json" ]]; then
    log "FAIL ${name} rc=${rc}"
    exit "$rc"
  fi
  log "DONE ${name} rc=0"
}

run_arm spar_base_boxed "$BASE_HF"
run_arm spar_spar3_boxed "$SPAR3_HF"

log "ALL DONE"
echo ALL DONE >>"$STATUS"
