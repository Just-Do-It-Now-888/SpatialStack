#!/usr/bin/env bash
# Merge SPAR T32/S16 historically best FSDP ckpts (450/600/1100) and run
# lastline CV-Bench / BLINK-Spatial / SPAR-Bench on each (HF lmms-eval, sr_opsd).
# Not comparable to boxed geo 66.59 / 86.06 / 68.86.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

export HOME="${HOME:-/home/c30084464}"
set +u
# shellcheck disable=SC1090
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sr_opsd
set -u

if [[ "$CONDA_PREFIX" != *"/envs/sr_opsd" ]]; then
  echo "expected sr_opsd, got CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
  exit 1
fi

CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/checkpoints/20260918_spatialstack_mvopsd_spar_t32_s16}"
BASE_MODEL="${BASE_MODEL:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
VGGT="${VGGT:-$PROJECT_ROOT/models/VGGT-1B}"
EVAL_PARENT="${EVAL_PARENT:-$PROJECT_ROOT/logs/eval/20260918_spatialstack_mvopsd_spar_t32_s16}"
STEPS="${STEPS:-450,600,1100}"
LOG="$EVAL_PARENT/best3_benches.log"
mkdir -p "$EVAL_PARENT"

export PYTHONPATH="$PROJECT_ROOT/src"
export QWEN35_ENV_ROOT="$CONDA_PREFIX"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export VSIBENCH_BOXED_PRIMARY=0
export CVBENCH_PROTOCOL=lastline
export CVBENCH_PARSER=lastline
export BLINK_PROTOCOL=lastline
export SPARBENCH_PROTOCOL=lastline
export SPARBENCH_MAX_NEW_TOKENS=2048
export MODEL_IMPL=qwen3_5

exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%F\ %T)] SPAR T32/S16 best3 lastline benches conda=$CONDA_PREFIX steps=$STEPS"

merge_one() {
  local step="$1"
  local step_dir="$CKPT_ROOT/global_step_${step}"
  local target="$PROJECT_ROOT/output/20260918_spatialstack_mvopsd_spar_t32_s16_global_step_${step}_hf"
  if [[ -f "$target/config.json" ]]; then
    echo "[$(date +%F\ %T)] skip merge step ${step}"
    return 0
  fi
  if [[ ! -d "$step_dir/actor" ]]; then
    echo "missing actor at $step_dir" >&2
    exit 1
  fi
  echo "[$(date +%F\ %T)] merge step ${step}"
  BASE_MODEL="$BASE_MODEL" \
    PYTHON_MERGE="$CONDA_PREFIX/bin/python" \
    bash "$PROJECT_ROOT/scripts/opsd/merge_mvopsd_ckpt.sh" "$step_dir" "$target"
}

bench_done() {
  local root="$1"
  local bench="$2"
  python - "$root" "$bench" <<'PY'
import json, sys
from pathlib import Path
root, bench = Path(sys.argv[1]), sys.argv[2]
keys = {
    "cvbench": "cvbench_score,none",
    "blink_spatial": "blink_spatial_score,none",
    "sparbench": "sparbench_score,none",
}
need = keys.get(bench)
cands = list(root.glob(f"**/{bench}/**/*_results.json")) + list(root.glob(f"**/*{bench}*_results.json"))
for p in cands:
    try:
        data = json.loads(p.read_text())
        block = (data.get("results") or {}).get(bench) or {}
        if need and need in block:
            sys.exit(0)
        if bench == "blink_spatial" and any("acc" in k.lower() or "score" in k.lower() for k in block):
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PY
}

run_bench() {
  local step="$1"
  local bench="$2"
  local port="$3"
  local ckpt="$PROJECT_ROOT/output/20260918_spatialstack_mvopsd_spar_t32_s16_global_step_${step}_hf"
  local eval_root="$EVAL_PARENT/step${step}_${bench}_lastline"
  mkdir -p "$eval_root"
  if bench_done "$eval_root" "$bench"; then
    echo "[$(date +%F\ %T)] skip ${bench} step ${step}"
    return 0
  fi
  echo "[$(date +%F\ %T)] ${bench} lastline step ${step}"
  set +e
  env \
    MODEL_PATH="$ckpt" \
    MODEL_ARGS_BASE="pretrained=$ckpt,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true" \
    OUTPUT_ROOT="$eval_root" \
    OUTPUT_PATH="$eval_root" \
    BENCHMARKS="$bench" \
    PROCESSES_PER_MACHINE=8 \
    MASTER_PORT="$port" \
    QWEN35_ENV_ROOT="$CONDA_PREFIX" \
    env -u LIMIT \
    bash "$PROJECT_ROOT/scripts/evaluation/eval.sh"
  local rc=$?
  set -e
  if bench_done "$eval_root" "$bench"; then
    echo "[$(date +%F\ %T)] ${bench} step ${step} ok"
    return 0
  fi
  echo "[$(date +%F\ %T)] ${bench} step ${step} missing results rc=$rc" >&2
  return 1
}

IFS=',' read -ra STEP_ARR <<< "$STEPS"
for step in "${STEP_ARR[@]}"; do
  merge_one "$step"
done

fail=0
port=29620
for step in "${STEP_ARR[@]}"; do
  for bench in cvbench blink_spatial sparbench; do
    run_bench "$step" "$bench" "$port" || fail=1
    port=$((port + 1))
  done
done

echo "[$(date +%F\ %T)] DONE SPAR best3 lastline benches fail=$fail"
exit "$fail"
