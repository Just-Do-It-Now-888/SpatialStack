#!/usr/bin/env bash
# SpatialStack geo SFT lastline offline eval (HF lmms-eval, conda sr_opsd).
# Matches VLM-3R last3 protocol: VSI lastline @4096 then CV / BLINK-Spatial / SPAR lastline.
#
#   setsid nohup env HOME=/home/c30084464 \
#     bash scripts/opsd/eval_spatialstack_geo_lastline.sh
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
LOG="$EVAL_PARENT/run.log"
mkdir -p "$EVAL_PARENT"

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
export VSIBENCH_PROTOCOL=lastline
export VSIBENCH_BOXED_PRIMARY=0
export CVBENCH_PROTOCOL=lastline
export CVBENCH_PARSER=lastline
export BLINK_PROTOCOL=lastline
export SPARBENCH_PROTOCOL=lastline
export SPARBENCH_MAX_NEW_TOKENS=2048
export MODEL_IMPL=qwen3_5
export MODEL_PATH
export MODEL_ARGS_BASE="pretrained=$MODEL_PATH,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true"

exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%F\ %T)] geo lastline conda=$CONDA_PREFIX model=$MODEL_PATH"

vsi_done() {
  local eval_root="$1"
  python - "$eval_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for p in list(root.glob("**/vsibench/**/*_results.json")) + list(root.glob("**/*_results.json")):
    try:
        data = json.loads(p.read_text())
        block = (data.get("results") or {}).get("vsibench") or {}
        if "vsibench_score,none" in block:
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PY
}

VSI_ROOT="$EVAL_PARENT/vsibench"
mkdir -p "$VSI_ROOT"
if vsi_done "$VSI_ROOT"; then
  echo "[$(date +%F\ %T)] skip VSI, results exist"
else
  echo "[$(date +%F\ %T)] VSI lastline @4096"
  env \
    OUTPUT_ROOT="$VSI_ROOT" \
    OUTPUT_PATH="$VSI_ROOT" \
    BENCHMARKS=vsibench \
    PROCESSES_PER_MACHINE=8 \
    MASTER_PORT="${MASTER_PORT:-29570}" \
    env -u LIMIT \
    bash "$PROJECT_ROOT/scripts/evaluation/eval.sh"
  if ! vsi_done "$VSI_ROOT"; then
    echo "VSI lastline failed" >&2
    exit 1
  fi
fi

python - "$VSI_ROOT" "$EVAL_PARENT/vsi_score.txt" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
best = None
path = None
for p in root.rglob("*_results.json"):
    data = json.loads(p.read_text())
    block = (data.get("results") or {}).get("vsibench") or {}
    if "vsibench_score,none" not in block:
        continue
    val = float(block["vsibench_score,none"])
    if best is None or val > best:
        best = val
        path = p
print(f"vsibench_score {best:.6f} file={path}")
Path(sys.argv[2]).write_text(f"vsibench_score={best:.6f}\nresults={path}\n")
PY

echo "[$(date +%F\ %T)] benches lastline"
MODEL_PATH="$MODEL_PATH" \
  VGGT="$VGGT" \
  EVAL_ID="20260918_spatialstack_geo_lastline" \
  OUTPUT_ROOT="$EVAL_PARENT" \
  BENCHMARKS="cvbench,blink_spatial,sparbench" \
  BLINK_BENCHMARK=blink_spatial \
  CVBENCH_PROTOCOL=lastline \
  CVBENCH_PARSER=lastline \
  BLINK_PROTOCOL=lastline \
  VSIBENCH_BOXED_PRIMARY=0 \
  SPARBENCH_PROTOCOL=lastline \
  SPARBENCH_MAX_NEW_TOKENS=2048 \
  bash "$PROJECT_ROOT/scripts/opsd/run_spatialstack_geo_bench_eval.sh"

echo "[$(date +%F\ %T)] DONE geo lastline"
