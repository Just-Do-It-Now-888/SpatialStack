#!/usr/bin/env bash
# Merge the last three greedy FSDP actors (550/600/635), run full VSI lastline
# @4096, pick the best overall, then CV-Bench / BLINK-Spatial / SPAR-Bench
# lastline (SPAR wording, no \\boxed{}).
#
# Geometry weights: HF lmms-eval in conda sr_opsd, not vLLM.
# BLINK is the 3-task spatial split. lastline scores are not comparable to boxed.
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

CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/checkpoints/20260915_spatialstack_mvopsd_vlm3r_t8_s4}"
BASE_MODEL="${BASE_MODEL:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
VGGT="${VGGT:-$PROJECT_ROOT/models/VGGT-1B}"
EVAL_PARENT="${EVAL_PARENT:-$PROJECT_ROOT/logs/eval/20260915_spatialstack_mvopsd_vlm3r_t8_s4}"
STEPS="${STEPS:-550,600,635}"
LOG="$EVAL_PARENT/last3_then_benches.log"
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
export VSIBENCH_PROTOCOL=lastline
export VSIBENCH_BOXED_PRIMARY=0
export CVBENCH_PROTOCOL=lastline
export CVBENCH_PARSER=lastline
export BLINK_PROTOCOL=lastline
export SPARBENCH_PROTOCOL=lastline
export SPARBENCH_MAX_NEW_TOKENS=2048
export MODEL_IMPL=qwen3_5

exec > >(tee -a "$LOG") 2>&1
echo "[$(date +%F\ %T)] last3 VSI lastline then benches conda=$CONDA_PREFIX protocol=lastline"

merge_one() {
  local step="$1"
  local step_dir="$CKPT_ROOT/global_step_${step}"
  local target="$PROJECT_ROOT/output/20260915_spatialstack_mvopsd_vlm3r_t8_s4_global_step_${step}_hf"
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

vsi_done() {
  local eval_root="$1"
  python - "$eval_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cands = list(root.glob("**/vsibench/**/*_results.json")) + list(root.glob("**/*_results.json"))
for p in cands:
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

run_vsi() {
  local step="$1"
  local ckpt="$PROJECT_ROOT/output/20260915_spatialstack_mvopsd_vlm3r_t8_s4_global_step_${step}_hf"
  local eval_root="$EVAL_PARENT/step${step}_vsibench_lastline"
  mkdir -p "$eval_root"
  if vsi_done "$eval_root"; then
    echo "[$(date +%F\ %T)] skip VSI step ${step}, results exist"
    return 0
  fi
  echo "[$(date +%F\ %T)] VSI lastline step ${step}"
  env \
    MODEL_PATH="$ckpt" \
    MODEL_ARGS_BASE="pretrained=$ckpt,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true" \
    OUTPUT_ROOT="$eval_root" \
    OUTPUT_PATH="$eval_root" \
    BENCHMARKS=vsibench \
    PROCESSES_PER_MACHINE=8 \
    MASTER_PORT="$((29540 + step % 50))" \
    QWEN35_ENV_ROOT="$CONDA_PREFIX" \
    env -u LIMIT \
    bash "$PROJECT_ROOT/scripts/evaluation/eval.sh"
  if ! vsi_done "$eval_root"; then
    echo "VSI lastline failed for step ${step}; not continuing" >&2
    exit 1
  fi
}

IFS=',' read -ra STEP_ARR <<< "$STEPS"
for step in "${STEP_ARR[@]}"; do
  merge_one "$step"
done
for step in "${STEP_ARR[@]}"; do
  run_vsi "$step"
done

python - "$EVAL_PARENT" "$STEPS" <<'PY'
import json, sys
from pathlib import Path
parent = Path(sys.argv[1])
steps = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
rows = []
for step in steps:
    root = parent / f"step{step}_vsibench_lastline"
    best = None
    for p in root.rglob("*_results.json"):
        try:
            data = json.loads(p.read_text())
            block = (data.get("results") or {}).get("vsibench") or {}
            if "vsibench_score,none" not in block:
                continue
            val = float(block["vsibench_score,none"])
            if best is None or val > best:
                best = val
        except Exception:
            pass
    if best is None:
        raise SystemExit(f"no vsibench_score under {root}")
    rows.append((step, best))
    print(f"step {step} vsibench_score {best:.6f}")
best_step, best_score = max(rows, key=lambda x: x[1])
(parent / "vsi_last3_scores.txt").write_text(
    "\n".join(f"step {s} {v:.6f}" for s, v in rows) + f"\nbest_step={best_step}\n"
)
(parent / "best_ckpt.txt").write_text(f"best_step={best_step}\nbest_score={best_score:.6f}\n")
print(f"BEST {best_step} {best_score:.6f}")
PY
best_step="$(awk -F= '/^best_step=/{print $2}' "$EVAL_PARENT/best_ckpt.txt")"
best_score="$(awk -F= '/^best_score=/{print $2}' "$EVAL_PARENT/best_ckpt.txt")"

echo "[$(date +%F\ %T)] best VSI step=${best_step} score=${best_score}"

BEST_HF="$PROJECT_ROOT/output/20260915_spatialstack_mvopsd_vlm3r_t8_s4_global_step_${best_step}_hf"
BENCH_ROOT="$EVAL_PARENT/step${best_step}_cv_blink_spar"
mkdir -p "$BENCH_ROOT"

echo "[$(date +%F\ %T)] benches on step ${best_step}"
MODEL_PATH="$BEST_HF" \
  VGGT="$VGGT" \
  EVAL_ID="20260915_spatialstack_mvopsd_vlm3r_t8_s4_step${best_step}" \
  OUTPUT_ROOT="$BENCH_ROOT" \
  BENCHMARKS="cvbench,blink_spatial,sparbench" \
  BLINK_BENCHMARK=blink_spatial \
  bash "$PROJECT_ROOT/scripts/opsd/run_spatialstack_geo_bench_eval.sh"

echo "[$(date +%F\ %T)] DONE best_step=${best_step} vsi=${best_score}"
