#!/usr/bin/env bash
# Serial vLLM VSI-Bench @2048: base vs SPAR3 K=1 step 55, original + boxed, greedy + sample.
# Eight arms. Non-zero exit or missing summary.json stops the chain.
#
#   setsid nohup bash scripts/opsd/run_vsibench_vllm_tok2048_chain.sh \
#     > logs/eval/20260906_vsibench_vllm_tok2048/driver.log 2>&1

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAIN_ROOT="${CHAIN_ROOT:-logs/eval/20260906_vsibench_vllm_tok2048}"
STATUS="${CHAIN_ROOT}/status.txt"
mkdir -p "$CHAIN_ROOT"

BOXED_SUFFIX=' The final answer MUST BE put in \boxed{} on the last line of your response.'
SEED="${SEED:-20260904}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM="${GPU_MEM:-0.9}"
SCENES_PER_BATCH="${SCENES_PER_BATCH:-8}"
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

if ! python -c "import vllm, decord" >/dev/null 2>&1; then
  log "preflight failed: need vllm and decord in sr_opsd"
  python -c "import vllm, decord"
  exit 1
fi

for cfg in "$BASE_HF/config.json" "$SPAR3_HF/config.json"; do
  if [[ ! -f "$cfg" ]]; then
    log "missing $cfg"
    exit 1
  fi
done

log "chain start host=$(hostname) python=$(command -v python) max_tokens=${MAX_TOKENS} seed=${SEED}"
python -c "import vllm, decord; print('vllm', vllm.__version__, 'decord', decord.__version__)" | tee -a "$STATUS"

run_arm() {
  local name="$1"
  local model="$2"
  local protocol="$3"
  local sample="$4"
  local out="${CHAIN_ROOT}/${name}"

  mkdir -p "$out"
  if [[ -s "${out}/summary.json" ]]; then
    log "SKIP ${name} existing ${out}/summary.json"
    return 0
  fi

  wait_gpus_free
  log "START ${name} model=${model} protocol=${protocol} tok=${MAX_TOKENS} sample=${sample}"

  local args=(
    python scripts/opsd/tools/eval_vsibench_vllm.py
    --model "$model"
    --protocol "$protocol"
    --frames 32
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --tensor-parallel-size 8
    --gpu-memory-utilization "$GPU_MEM"
    --scenes-per-batch "$SCENES_PER_BATCH"
    --output-dir "$out"
  )
  if [[ "$protocol" == "spatialstack" ]]; then
    args+=(--prompt-suffix "$BOXED_SUFFIX" --boxed-primary)
  fi
  if [[ "$sample" == "1" ]]; then
    args+=(--do-sample --seed "$SEED")
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

run_arm base_plain_greedy "$BASE_HF" spatialstack_plain 0
run_arm spar3_plain_greedy "$SPAR3_HF" spatialstack_plain 0
run_arm base_boxed_greedy "$BASE_HF" spatialstack 0
run_arm spar3_boxed_greedy "$SPAR3_HF" spatialstack 0
run_arm base_plain_sample "$BASE_HF" spatialstack_plain 1
run_arm spar3_plain_sample "$SPAR3_HF" spatialstack_plain 1
run_arm base_boxed_sample "$BASE_HF" spatialstack 1
run_arm spar3_boxed_sample "$SPAR3_HF" spatialstack 1

python scripts/opsd/tools/summarize_vsibench_tok2048.py --chain-root "$CHAIN_ROOT" | tee -a "$STATUS"
log "ALL DONE"
exit 0
