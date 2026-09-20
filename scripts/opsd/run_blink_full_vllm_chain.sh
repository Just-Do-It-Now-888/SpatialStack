#!/usr/bin/env bash
# Serial vLLM BLINK full (14-task val): base vs SPAR3 K=1 step 55, original + spatialstack, greedy @1024.
#
#   setsid nohup bash scripts/opsd/run_blink_full_vllm_chain.sh \
#     > logs/eval/20260908_blink_full_vllm/driver.log 2>&1

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAIN_ROOT="${CHAIN_ROOT:-logs/eval/20260908_blink_full_vllm}"
STATUS="${CHAIN_ROOT}/status.txt"
mkdir -p "$CHAIN_ROOT"

MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM="${GPU_MEM:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MIN_FREE_MIB="${MIN_FREE_MIB:-80000}"
BASE_HF="${BASE_HF:-models/Qwen3.5-4B}"
SPAR3_HF="${SPAR3_HF:-output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf}"
LIMIT="${LIMIT:-}"
TASKS="${TASKS:-}"

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

if ! python -c "import vllm" >/dev/null 2>&1; then
  log "preflight failed: vllm missing in sr_opsd"
  python -c "import vllm"
  exit 1
fi

for cfg in "$BASE_HF/config.json" "$SPAR3_HF/config.json"; do
  if [[ ! -f "$cfg" ]]; then
    log "missing $cfg"
    exit 1
  fi
done

log "chain start host=$(hostname) python=$(command -v python) max_tokens=${MAX_TOKENS}"
python -c "import vllm; print('vllm', vllm.__version__)" | tee -a "$STATUS"

run_arm() {
  local name="$1"
  local model="$2"
  local protocol="$3"
  local out="${CHAIN_ROOT}/${name}"

  mkdir -p "$out"
  if [[ -s "${out}/summary.json" ]]; then
    log "SKIP ${name} existing ${out}/summary.json"
    return 0
  fi

  wait_gpus_free
  # First SPAR3 arm hit SIGBUS (rc=135) 17s after base_original NCCL destroy.
  sleep 20
  wait_gpus_free
  log "START ${name} model=${model} protocol=${protocol} tok=${MAX_TOKENS}"

  local args=(
    python scripts/opsd/tools/eval_blink_vllm.py
    --model "$model"
    --protocol "$protocol"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size 8
    --gpu-memory-utilization "$GPU_MEM"
    --batch-size "$BATCH_SIZE"
    --output-dir "$out"
  )
  if [[ -n "$TASKS" ]]; then
    args+=(--tasks "$TASKS")
  fi
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi

  set +e
  "${args[@]}" 2>&1 | tee -a "${out}/run.log"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [[ "$rc" -ne 0 || ! -s "${out}/summary.json" ]]; then
    log "FAIL ${name} rc=${rc} summary=${out}/summary.json"
    exit 1
  fi
  log "DONE ${name} rc=0"
}

run_arm base_original "$BASE_HF" original
run_arm spar3_original "$SPAR3_HF" original
run_arm base_spatialstack "$BASE_HF" spatialstack
run_arm spar3_spatialstack "$SPAR3_HF" spatialstack

log "ALL DONE"
exit 0
