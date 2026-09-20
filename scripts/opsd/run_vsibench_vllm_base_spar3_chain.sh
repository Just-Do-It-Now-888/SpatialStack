#!/usr/bin/env bash
# Serial vLLM VSI-Bench queue: base vs SPAR3 K=1 step 55, greedy + sample.
# Skips base original greedy (already at logs/eval/20260903_qwen35base_vsibench_plain_vllm).
# One 8-GPU node; wait for free VRAM between arms. Non-zero exit stops the chain.
#
#   setsid nohup bash scripts/opsd/run_vsibench_vllm_base_spar3_chain.sh \
#     > logs/eval/20260904_vsibench_vllm_base_spar3_step55/driver.log 2>&1

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAIN_ROOT="${CHAIN_ROOT:-logs/eval/20260904_vsibench_vllm_base_spar3_step55}"
STATUS="${CHAIN_ROOT}/status.txt"
mkdir -p "$CHAIN_ROOT"

BOXED_SUFFIX=' The final answer MUST BE put in \boxed{} on the last line of your response.'
SEED="${SEED:-20260904}"
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

log "chain start host=$(hostname) python=$(command -v python)"
python -c "import vllm, decord; print('vllm', vllm.__version__, 'decord', decord.__version__)" | tee -a "$STATUS"

run_arm() {
  local name="$1"
  local model="$2"
  local protocol="$3"
  local max_tokens="$4"
  local sample="$5"
  local out="$6"
  shift 6

  mkdir -p "$out"
  wait_gpus_free
  log "START ${name} model=${model} protocol=${protocol} tok=${max_tokens} sample=${sample} out=${out}"

  local args=(
    python scripts/opsd/tools/eval_vsibench_vllm.py
    --model "$model"
    --protocol "$protocol"
    --frames 32
    --max-tokens "$max_tokens"
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

  log "END ${name} rc=${rc}"
  if [[ "$rc" -ne 0 ]]; then
    log "STOP chain after ${name}"
    exit "$rc"
  fi
}

# 1 SPAR3 original greedy
run_arm spar3_plain_greedy "$SPAR3_HF" spatialstack_plain 1024 0 \
  logs/eval/20260904_spar3_k1_step55_vsibench_plain_vllm
# 2 SPAR3 original sample
run_arm spar3_plain_sample "$SPAR3_HF" spatialstack_plain 1024 1 \
  logs/eval/20260904_spar3_k1_step55_vsibench_plain_vllm_sample
# 3 base boxed greedy
run_arm base_boxed_greedy "$BASE_HF" spatialstack 4096 0 \
  logs/eval/20260904_qwen35base_vsibench_boxed_lastline_vllm
# 4 base boxed sample
run_arm base_boxed_sample "$BASE_HF" spatialstack 4096 1 \
  logs/eval/20260904_qwen35base_vsibench_boxed_lastline_vllm_sample
# 5 SPAR3 boxed greedy
run_arm spar3_boxed_greedy "$SPAR3_HF" spatialstack 4096 0 \
  logs/eval/20260904_spar3_k1_step55_vsibench_boxed_lastline_vllm
# 6 SPAR3 boxed sample
run_arm spar3_boxed_sample "$SPAR3_HF" spatialstack 4096 1 \
  logs/eval/20260904_spar3_k1_step55_vsibench_boxed_lastline_vllm_sample
# 7 base original sample
run_arm base_plain_sample "$BASE_HF" spatialstack_plain 1024 1 \
  logs/eval/20260904_qwen35base_vsibench_plain_vllm_sample

log "ALL DONE"
exit 0
