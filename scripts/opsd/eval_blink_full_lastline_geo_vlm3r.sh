#!/usr/bin/env bash
# BLINK 14-task val lastline: SpatialStack geo SFT then VLM-3R step 635.
# HF lmms-eval, conda sr_opsd, geometry weights (not vLLM).
#
# Smoke then full. If the blink group dies on VGGT fusion, continue per-task.
#
#   setsid nohup env HOME=/home/c30084464 \
#     bash scripts/opsd/eval_blink_full_lastline_geo_vlm3r.sh
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

GEO_MODEL="${GEO_MODEL:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
VLM_MODEL="${VLM_MODEL:-$PROJECT_ROOT/output/20260915_spatialstack_mvopsd_vlm3r_t8_s4_global_step_635_hf}"
VGGT="${VGGT:-$PROJECT_ROOT/models/VGGT-1B}"
EVAL_PARENT="${EVAL_PARENT:-$PROJECT_ROOT/logs/eval/20260918_blink_full_lastline}"
LOG="$EVAL_PARENT/driver.log"
mkdir -p "$EVAL_PARENT"

BLINK_TASKS=(
  blink_art_style
  blink_counting
  blink_forensic_detection
  blink_functional_correspondence
  blink_iq_test
  blink_jigsaw
  blink_multi_view_reasoning
  blink_object_localization
  blink_relative_depth
  blink_relative_reflectance
  blink_semantic_correspondence
  blink_spatial_relation
  blink_visual_correspondence
  blink_visual_similarity
)

export PYTHONPATH="$PROJECT_ROOT/src"
export QWEN35_ENV_ROOT="$CONDA_PREFIX"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export BLINK_PROTOCOL=lastline
export VSIBENCH_BOXED_PRIMARY=0
export MODEL_IMPL=qwen3_5

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] blink full lastline conda=$CONDA_PREFIX"

results_ok() {
  local eval_root="$1"
  local task="$2"
  python - "$eval_root" "$task" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
task = sys.argv[2]
key = f"{task}_acc,none" if task != "blink" else "blink_acc,none"
for p in root.rglob("*_results.json"):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    results = data.get("results") or {}
    if task == "blink":
        if "blink" in results and any(k.startswith("blink_acc") for k in results["blink"]):
            sys.exit(0)
        n = 0
        for name, block in results.items():
            if name.startswith("blink_") and isinstance(block, dict) and any(
                k.startswith("blink_acc") for k in block
            ):
                n += 1
        if n >= 14:
            sys.exit(0)
        continue
    block = results.get(task) or {}
    if any(k.startswith("blink_acc") for k in block):
        sys.exit(0)
sys.exit(1)
PY
}

run_eval() {
  local model="$1"
  local out="$2"
  local tasks_csv="$3"
  local port="$4"
  local limit="${5:-}"
  mkdir -p "$out"
  local extra=()
  if [[ -n "$limit" ]]; then
    extra+=(LIMIT="$limit" PROCESSES_PER_MACHINE=1 CUDA_VISIBLE_DEVICES=0)
  else
    extra+=(PROCESSES_PER_MACHINE=8)
    extra+=(LIMIT=)
  fi
  env \
    MODEL_PATH="$model" \
    MODEL_ARGS_BASE="pretrained=$model,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=$VGGT,disable_thinking=true" \
    OUTPUT_ROOT="$out" \
    OUTPUT_PATH="$out" \
    BENCHMARKS="$tasks_csv" \
    MASTER_PORT="$port" \
    "${extra[@]}" \
    bash "$PROJECT_ROOT/scripts/evaluation/eval.sh"
}

summarize_one() {
  local eval_root="$1"
  local json_out="$2"
  python - "$eval_root" "$json_out" <<'PY'
import json, statistics, sys
from pathlib import Path

SPATIAL = (
    "blink_multi_view_reasoning",
    "blink_relative_depth",
    "blink_spatial_relation",
)
root = Path(sys.argv[1])
out = Path(sys.argv[2])
acc = {}
answered = {}
n = {}
resp_lens = {}
failed = []
found_tasks = set()
for p in root.rglob("*_results.json"):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    results = data.get("results") or {}
    samples = data.get("samples") or {}
    for name, block in results.items():
        if not name.startswith("blink_") or not isinstance(block, dict):
            continue
        if not any(k.startswith("blink_acc") for k in block):
            continue
        val = None
        for k, v in block.items():
            if k.startswith("blink_acc"):
                val = float(v)
                break
        acc[name] = val
        found_tasks.add(name)
        rows = samples.get(name) or []
        n[name] = len(rows)
        ans = 0
        lengths = []
        for row in rows:
            blob = row.get("blink_acc") if isinstance(row, dict) else None
            if not isinstance(blob, dict):
                continue
            ans += int(blob.get("answered") or 0)
            pred = blob.get("pred") or ""
            lengths.append(len(pred))
        if rows:
            answered[name] = round(100.0 * ans / len(rows), 2)
        if lengths:
            resp_lens[name] = {
                "n": len(lengths),
                "median": int(statistics.median(lengths)),
                "p90": int(sorted(lengths)[max(0, int(0.9 * (len(lengths) - 1)))]),
                "max": max(lengths),
            }

payload = {
    "tasks": {k: {"acc": round(100.0 * v, 2), "answered": answered.get(k), "n": n.get(k), "resp_chars": resp_lens.get(k)} for k, v in sorted(acc.items())},
    "n_tasks_scored": len(acc),
    "macro_14": None,
    "spatial_macro": None,
    "spatial": {},
    "complete_14": len(acc) == 14,
}
if len(acc) == 14:
    payload["macro_14"] = round(sum(100.0 * v for v in acc.values()) / 14.0, 2)
if all(t in acc for t in SPATIAL):
    payload["spatial_macro"] = round(sum(100.0 * acc[t] for t in SPATIAL) / 3.0, 2)
    payload["spatial"] = {t: round(100.0 * acc[t], 2) for t in SPATIAL}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
}

run_model() {
  local tag="$1"
  local model="$2"
  local base_port="$3"
  local out="$EVAL_PARENT/$tag"
  mkdir -p "$out"
  if [[ ! -f "$model/config.json" ]]; then
    echo "missing $model/config.json" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] === model $tag $model ==="

  local smoke_dir="$out/smoke"
  local smoke_ok=0
  if [[ "${SKIP_SMOKE:-0}" != "1" && ! -f "$out/smoke.pass" ]]; then
    echo "[$(date '+%F %T')] smoke LIMIT=8 BENCHMARKS=blink"
    set +e
    run_eval "$model" "$smoke_dir" "blink" "$base_port" 8
    local rc=$?
    set -e
    if [[ $rc -eq 0 ]] && results_ok "$smoke_dir" blink; then
      smoke_ok=1
      echo pass > "$out/smoke.pass"
      echo "[$(date '+%F %T')] smoke PASS"
    else
      echo "[$(date '+%F %T')] smoke FAIL rc=$rc; switching to per-task"
      echo fail > "$out/smoke.fail"
      rm -f "$out/smoke.pass"
    fi
  elif [[ -f "$out/smoke.pass" ]]; then
    smoke_ok=1
    echo "[$(date '+%F %T')] skip smoke, smoke.pass exists"
  else
    echo "[$(date '+%F %T')] skip smoke (SKIP_SMOKE=1 or prior fail); per-task"
  fi

  if [[ "$smoke_ok" -eq 1 ]]; then
    local full="$out/full"
    if results_ok "$full" blink; then
      echo "[$(date '+%F %T')] skip full blink group, results exist"
    else
      echo "[$(date '+%F %T')] full BENCHMARKS=blink"
      set +e
      run_eval "$model" "$full" "blink" "$((base_port + 1))"
      local rc=$?
      set -e
      if [[ $rc -eq 0 ]] && results_ok "$full" blink; then
        echo "[$(date '+%F %T')] blink group complete"
      else
        echo "[$(date '+%F %T')] blink group failed rc=$rc; falling back to per-task"
        smoke_ok=0
      fi
    fi
  fi

  if [[ "$smoke_ok" -eq 0 ]]; then
    local tport="$base_port"
    for task in "${BLINK_TASKS[@]}"; do
      tport=$((tport + 1))
      local tdir="$out/per_task/$task"
      if results_ok "$tdir" "$task"; then
        echo "[$(date '+%F %T')] skip $task, results exist"
        continue
      fi
      echo "[$(date '+%F %T')] per-task $task"
      set +e
      run_eval "$model" "$tdir" "$task" "$tport"
      local rc=$?
      set -e
      if [[ $rc -ne 0 ]] || ! results_ok "$tdir" "$task"; then
        echo "[$(date '+%F %T')] FAIL $task rc=$rc"
        echo "$task rc=$rc $(date -Iseconds)" >> "$out/failed_tasks.txt"
      fi
    done
  fi

  if [[ -d "$out/full" ]] && results_ok "$out/full" blink; then
    summarize_one "$out/full" "$out/summary.json" || true
  else
    summarize_one "$out/per_task" "$out/summary.json" || true
  fi
}

run_model geo "$GEO_MODEL" 29630
run_model vlm3r635 "$VLM_MODEL" 29650

python - "$EVAL_PARENT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
combo = {}
for tag in ("geo", "vlm3r635"):
    p = root / tag / "summary.json"
    combo[tag] = json.loads(p.read_text()) if p.exists() else {"error": "missing summary"}
(root / "combined_summary.json").write_text(json.dumps(combo, indent=2) + "\n")
print((root / "combined_summary.json").read_text())
PY

echo "[$(date '+%F %T')] DONE blink full lastline"
