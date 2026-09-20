#!/usr/bin/env bash
# Serial vLLM lastline prompt (SPAR wording, no \\boxed{}): VSI, BLINK, CV.
# Parser is last-line extract. original dumps are not regenerated.
#
#   setsid nohup bash scripts/opsd/run_lastline_prompt_chain.sh \
#     > logs/eval/20260908_lastline_prompt/driver.log 2>&1

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAIN_ROOT="${CHAIN_ROOT:-logs/eval/20260908_lastline_prompt}"
STATUS="${CHAIN_ROOT}/status.txt"
mkdir -p "$CHAIN_ROOT"

LASTLINE=' The final answer MUST BE put on the last line of your response.'
MAX_TOKENS_VSI="${MAX_TOKENS_VSI:-2048}"
MAX_TOKENS_BLINK="${MAX_TOKENS_BLINK:-1024}"
MAX_TOKENS_CV="${MAX_TOKENS_CV:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM="${GPU_MEM:-0.9}"
SCENES_PER_BATCH="${SCENES_PER_BATCH:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
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
export VSIBENCH_BOXED_PRIMARY=0

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

log "chain start host=$(hostname) python=$(command -v python) suffix=SPAR-lastline"
python -c "import vllm, decord; print('vllm', vllm.__version__, 'decord', decord.__version__)" | tee -a "$STATUS"

finish_arm() {
  local name="$1"
  local rc="$2"
  local summary="$3"
  if [[ "$rc" -ne 0 || ! -s "$summary" ]]; then
    log "FAIL ${name} rc=${rc} summary=${summary}"
    exit 1
  fi
  log "DONE ${name} rc=0"
}

run_vsi() {
  local name="$1"
  local model="$2"
  local out="${CHAIN_ROOT}/${name}"
  mkdir -p "$out"
  if [[ -s "${out}/summary.json" ]]; then
    log "SKIP ${name} existing ${out}/summary.json"
    return 0
  fi
  pause_between_arms
  log "START ${name} vsi lastline tok=${MAX_TOKENS_VSI}"
  set +e
  python scripts/opsd/tools/eval_vsibench_vllm.py \
    --model "$model" \
    --protocol spatialstack_plain \
    --prompt-suffix "$LASTLINE" \
    --frames 32 \
    --max-tokens "$MAX_TOKENS_VSI" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization "$GPU_MEM" \
    --scenes-per-batch "$SCENES_PER_BATCH" \
    --output-dir "$out" \
    2>&1 | tee -a "${out}/run.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  finish_arm "$name" "$rc" "${out}/summary.json"
}

run_blink() {
  local name="$1"
  local model="$2"
  local out="${CHAIN_ROOT}/${name}"
  mkdir -p "$out"
  if [[ -s "${out}/summary.json" ]]; then
    log "SKIP ${name} existing ${out}/summary.json"
    return 0
  fi
  pause_between_arms
  log "START ${name} blink lastline tok=${MAX_TOKENS_BLINK}"
  set +e
  python scripts/opsd/tools/eval_blink_vllm.py \
    --model "$model" \
    --protocol lastline \
    --max-tokens "$MAX_TOKENS_BLINK" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization "$GPU_MEM" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$out" \
    2>&1 | tee -a "${out}/run.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  finish_arm "$name" "$rc" "${out}/summary.json"
}

run_cv() {
  local name="$1"
  local model="$2"
  local out="${CHAIN_ROOT}/${name}"
  mkdir -p "$out"
  if [[ -s "${out}/summary.json" ]]; then
    log "SKIP ${name} existing ${out}/summary.json"
    return 0
  fi
  pause_between_arms
  log "START ${name} cv lastline tok=${MAX_TOKENS_CV}"
  set +e
  python scripts/opsd/tools/eval_cvbench_vllm.py \
    --model "$model" \
    --parser lastline \
    --prompt-suffix "$LASTLINE" \
    --max-tokens "$MAX_TOKENS_CV" \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --output-dir "$out" \
    2>&1 | tee -a "${out}/run.log"
  local rc="${PIPESTATUS[0]}"
  set -e
  finish_arm "$name" "$rc" "${out}/summary.json"
}

run_vsi vsi_base_lastline "$BASE_HF"
run_vsi vsi_spar3_lastline "$SPAR3_HF"
run_blink blink_base_lastline "$BASE_HF"
run_blink blink_spar3_lastline "$SPAR3_HF"
run_cv cv_base_lastline "$BASE_HF"
run_cv cv_spar3_lastline "$SPAR3_HF"

log "ALL DONE"
exit 0
