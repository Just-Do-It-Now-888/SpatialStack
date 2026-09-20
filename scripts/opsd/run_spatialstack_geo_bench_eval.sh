#!/usr/bin/env bash
# SpatialStack geometry cold-start offline eval (CV-Bench / BLINK / SPAR-Bench).
# Engine: HF lmms-eval in conda sr_opsd (geometry weights cannot use vLLM).
#
# Smoke (8 samples each):
#   SMOKE=1 bash scripts/opsd/run_spatialstack_geo_bench_eval.sh
#
# Full chain:
#   bash scripts/opsd/run_spatialstack_geo_bench_eval.sh
#
# Single benchmark:
#   BENCHMARKS=cvbench bash scripts/opsd/run_spatialstack_geo_bench_eval.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

SMOKE="${SMOKE:-0}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
VGGT="${VGGT:-$PROJECT_ROOT/models/VGGT-1B}"
EVAL_ID="${EVAL_ID:-20260914_spatialstack_geo_coldstart}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/logs/eval/$EVAL_ID}"
BENCHMARKS="${BENCHMARKS:-cvbench,blink_spatial,sparbench}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29614}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[ERROR] missing checkpoint: $MODEL_PATH/config.json" >&2
  exit 1
fi

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate sr_opsd
  set -u
fi

if [[ "$CONDA_PREFIX" != *"/envs/sr_opsd" ]]; then
  echo "[ERROR] expected conda env sr_opsd, got CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
  exit 1
fi

export PATH="$CONDA_PREFIX/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HOME="${HF_HOME:-/home/c30084464/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export MODEL_PATH MODEL_IMPL=qwen3_5
export MODEL_ARGS_BASE="pretrained=$MODEL_PATH,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true"
export OUTPUT_ROOT BENCHMARKS
export CUDA_VISIBLE_DEVICES="$GPUS"

# Align with MV-OPSD / base offline comparators. Callers may override
# (e.g. lastline SpatialStack eval).
export CVBENCH_PROTOCOL="${CVBENCH_PROTOCOL:-spatialstack}"
export BLINK_PROTOCOL="${BLINK_PROTOCOL:-spatialstack}"
export VSIBENCH_BOXED_PRIMARY="${VSIBENCH_BOXED_PRIMARY:-1}"
export SPARBENCH_PROTOCOL="${SPARBENCH_PROTOCOL:-boxed}"
export SPARBENCH_MAX_NEW_TOKENS="${SPARBENCH_MAX_NEW_TOKENS:-2048}"

# blink_spatial = 3 spatial subtasks only; blink = official 14-task val (1901 rows).
BLINK_BENCHMARK="${BLINK_BENCHMARK:-blink_spatial}"

if [[ "$SMOKE" == "1" ]]; then
  echo "[INFO] smoke mode: LIMIT=8, 1 GPU, all benchmarks"
  mkdir -p "$OUTPUT_ROOT"
  LOG="$OUTPUT_ROOT/smoke.log"
  exec > >(tee -a "$LOG") 2>&1
  echo "[$(date +%F\ %T)] spatialstack geo bench eval SMOKE=1"
  echo "[$(date +%F\ %T)] model=$MODEL_PATH conda=$CONDA_PREFIX"
  for bench in cvbench "$BLINK_BENCHMARK" sparbench; do
    echo "[$(date +%F\ %T)] === smoke $bench LIMIT=8 ==="
    env MASTER_PORT="$((MASTER_PORT + RANDOM % 100))" \
      LIMIT=8 PROCESSES_PER_MACHINE=1 CUDA_VISIBLE_DEVICES=0 \
      BENCHMARKS="$bench" OUTPUT_ROOT="$OUTPUT_ROOT" \
      bash "$PROJECT_ROOT/scripts/evaluation/eval.sh" || exit 1
  done
  echo "[$(date +%F\ %T)] smoke PASS"
  exit 0
fi

IFS=',' read -ra __gpus <<< "$GPUS"
echo "[INFO] full eval: ${#__gpus[@]} GPUs benchmarks=$BENCHMARKS"
mkdir -p "$OUTPUT_ROOT"
LOG="$OUTPUT_ROOT/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +%F\ %T)] spatialstack geo bench eval full"
echo "[$(date +%F\ %T)] model=$MODEL_PATH"
echo "[$(date +%F\ %T)] conda=$CONDA_PREFIX python=$(which python)"
echo "[$(date +%F\ %T)] GPUs=$GPUS MASTER_PORT=$MASTER_PORT"

env MASTER_PORT="$MASTER_PORT" PROCESSES_PER_MACHINE="${#__gpus[@]}" \
  env -u LIMIT \
  bash "$PROJECT_ROOT/scripts/evaluation/eval.sh"
rc=$?
echo "[$(date +%F\ %T)] eval.sh finished rc=$rc"
exit "$rc"
