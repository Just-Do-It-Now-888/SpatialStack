#!/usr/bin/env bash
# Lastline CV / BLINK-Spatial / SPAR for SpatialStack geo SFT.
# Skip a bench if its results json already exists. Safe to run after the
# 20260918 VSI driver, or as a gap-fill if that driver died first.
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

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
VGGT="${VGGT:-$PROJECT_ROOT/models/VGGT-1B}"
EVAL_PARENT="${EVAL_PARENT:-$PROJECT_ROOT/logs/eval/20260918_spatialstack_geo_lastline}"
WAIT_PID="${WAIT_PID:-}"
LOG="$EVAL_PARENT/benches.log"
mkdir -p "$EVAL_PARENT"

if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date +%F\ %T)] wait pid $WAIT_PID before benches" | tee -a "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date +%F\ %T)] pid $WAIT_PID exited" | tee -a "$LOG"
fi

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "missing checkpoint $MODEL_PATH/config.json" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src"
export QWEN35_ENV_ROOT="$CONDA_PREFIX"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export CVBENCH_PROTOCOL=lastline
export CVBENCH_PARSER=lastline
export BLINK_PROTOCOL=lastline
export SPARBENCH_PROTOCOL=lastline
export SPARBENCH_MAX_NEW_TOKENS=2048
export VSIBENCH_BOXED_PRIMARY=0
export MODEL_IMPL=qwen3_5
export MODEL_PATH
export MODEL_ARGS_BASE="pretrained=$MODEL_PATH,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true"

exec >>"$LOG" 2>&1
echo "[$(date +%F\ %T)] geo lastline benches conda=$CONDA_PREFIX"

bench_done() {
  local bench="$1"
  local key="$2"
  python - "$EVAL_PARENT" "$bench" "$key" <<'PY'
import json, sys
from pathlib import Path
root, bench, key = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cands = list(root.glob(f"**/{bench}/**/*_results.json")) + list(root.glob(f"**/*{bench}*_results.json"))
for p in cands:
    try:
        data = json.loads(p.read_text())
        block = (data.get("results") or {}).get(bench) or {}
        if key in block:
            sys.exit(0)
        # blink_spatial dumps per-task keys; accept any blink_acc / overall
        if bench == "blink_spatial" and any("acc" in k.lower() or "score" in k.lower() for k in block):
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PY
}

run_one() {
  local bench="$1"
  local key="$2"
  local port="$3"
  if bench_done "$bench" "$key"; then
    echo "[$(date +%F\ %T)] skip $bench, results exist"
    return 0
  fi
  echo "[$(date +%F\ %T)] run $bench lastline"
  local out="$EVAL_PARENT/$bench"
  mkdir -p "$out"
  set +e
  env \
    OUTPUT_ROOT="$out" \
    OUTPUT_PATH="$out" \
    BENCHMARKS="$bench" \
    PROCESSES_PER_MACHINE=8 \
    MASTER_PORT="$port" \
    env -u LIMIT \
    bash "$PROJECT_ROOT/scripts/evaluation/eval.sh"
  local rc=$?
  set -e
  if bench_done "$bench" "$key"; then
    echo "[$(date +%F\ %T)] $bench ok"
    return 0
  fi
  echo "[$(date +%F\ %T)] $bench missing results rc=$rc" >&2
  return 1
}

fail=0
run_one cvbench "cvbench_score,none" 29614 || fail=1
run_one blink_spatial "blink_spatial_score,none" 29615 || fail=1
run_one sparbench "sparbench_score,none" 29616 || fail=1

echo "[$(date +%F\ %T)] DONE geo lastline benches fail=$fail"
exit "$fail"
