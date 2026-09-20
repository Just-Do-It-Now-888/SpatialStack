#!/usr/bin/env bash
# Serial HF/lmms-eval sample queue: CV-Bench + BLINK-Spatial, base vs SPAR3 K=1 step 55.
# Decoding matches VSI `--do-sample`: t=1.0 / top_p=0.8 / seed=20260904, single sample.
# One 8-GPU node. Non-zero exit stops the chain.
#
#   setsid nohup bash scripts/opsd/run_cvbench_blink_sample_chain.sh \
#     > logs/eval/20260905_cvbench_blink_sample/driver.log 2>&1

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAIN_ROOT="${CHAIN_ROOT:-logs/eval/20260905_cvbench_blink_sample}"
STATUS="${CHAIN_ROOT}/status.txt"
mkdir -p "$CHAIN_ROOT"

SEED="${SEED:-20260904}"
MIN_FREE_MIB="${MIN_FREE_MIB:-80000}"
BASE_HF="${BASE_HF:-models/Qwen3.5-4B}"
SPAR3_HF="${SPAR3_HF:-output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29610}"
arm_index=0

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
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
# lmms-eval CV-Bench/BLINK arrow caches live in the project tree, not HF_HOME/datasets.
# 20260901 boxed last-line used this path; pointing at ~/.cache/.../datasets with
# HF_HUB_OFFLINE=1 raises OfflineModeIsEnabled for nyu-visionx/CV-Bench.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/cache/datasets}"

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

if ! python -c "import lmms_eval, decord, transformers" >/dev/null 2>&1; then
  log "preflight failed: need lmms_eval, decord, transformers in sr_opsd"
  python -c "import lmms_eval, decord, transformers"
  exit 1
fi

for cfg in "$BASE_HF/config.json" "$SPAR3_HF/config.json"; do
  if [[ ! -f "$cfg" ]]; then
    log "missing $cfg"
    exit 1
  fi
done

log "chain start host=$(hostname) python=$(command -v python) seed=${SEED}"
python -c "import transformers, decord; print('transformers', transformers.__version__, 'decord', decord.__version__)" | tee -a "$STATUS"

run_arm() {
  local name="$1"
  local hf="$2"
  local bench="$3"
  local out="${CHAIN_ROOT}/${name}"
  mkdir -p "$out"
  wait_gpus_free
  MASTER_PORT=$((MASTER_PORT_BASE + arm_index))
  arm_index=$((arm_index + 1))
  export MASTER_PORT
  log "START ${name} model=${hf} bench=${bench} port=${MASTER_PORT}"
  set +e
  MODEL_PATH="$hf" \
  MODEL_IMPL=qwen3_5 \
  MODEL_ARGS_BASE="pretrained=${hf},use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
  OUTPUT_ROOT="$out" \
  OUTPUT_PATH="$out" \
  BENCHMARKS="$bench" \
  bash scripts/evaluation/eval.sh >"${out}/run.log" 2>&1
  local rc=$?
  set -e
  local results
  results="$(find "$out" -name '*_results.json' -size +100c 2>/dev/null | head -n 1 || true)"
  if [[ "$rc" -ne 0 || -z "$results" ]]; then
    log "FAIL ${name} rc=${rc} results=${results:-none}  see ${out}/run.log"
    exit 1
  fi
  log "DONE ${name} rc=0 results=${results}"
}

export CVBENCH_DO_SAMPLE=1
export CVBENCH_TEMPERATURE=1.0
export CVBENCH_TOP_P=0.8
export CVBENCH_SEED="$SEED"
export BLINK_DO_SAMPLE=1
export BLINK_TEMPERATURE=1.0
export BLINK_TOP_P=0.8
export BLINK_SEED="$SEED"

# Boxed last-line first (comparable to greedy 88.07/85.05 and 65.99/68.92).
export CVBENCH_PROTOCOL=spatialstack
export CVBENCH_PARSER=boxed_lastline
run_arm cv_boxed_base "$BASE_HF" cvbench
run_arm cv_boxed_spar3 "$SPAR3_HF" cvbench
unset CVBENCH_PROTOCOL CVBENCH_PARSER

export BLINK_PROTOCOL=spatialstack
export VSIBENCH_BOXED_PRIMARY=1
run_arm blink_boxed_base "$BASE_HF" blink_spatial
run_arm blink_boxed_spar3 "$SPAR3_HF" blink_spatial
unset BLINK_PROTOCOL VSIBENCH_BOXED_PRIMARY

# original protocols (same greedy cells in the comparison report).
export CVBENCH_PROTOCOL=lmms_legacy
run_arm cv_orig_base "$BASE_HF" cvbench
run_arm cv_orig_spar3 "$SPAR3_HF" cvbench
unset CVBENCH_PROTOCOL

export BLINK_PROTOCOL=original
export VSIBENCH_BOXED_PRIMARY=0
run_arm blink_orig_base "$BASE_HF" blink_spatial
run_arm blink_orig_spar3 "$SPAR3_HF" blink_spatial
unset BLINK_PROTOCOL VSIBENCH_BOXED_PRIMARY

log "ALL DONE"
exit 0
