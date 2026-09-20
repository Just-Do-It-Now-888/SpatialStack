#!/usr/bin/env bash
# Build and run the teacher keyframe vs full-album probe.
#
#   # smoke (64 rows total across both parquets)
#   SMOKE=1 bash scripts/opsd/run_teacher_keyframe_probe.sh
#
#   # full run
#   bash scripts/opsd/run_teacher_keyframe_probe.sh
#
#   setsid nohup bash scripts/opsd/run_teacher_keyframe_probe.sh \
#     > logs/eval/teacher_keyframe_probe/driver.log 2>&1
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-logs/eval/teacher_keyframe_probe}"
PARQUET_ROOT="${PARQUET_ROOT:-data/mvopsd/parquet_keyframe_sweep}"
MODEL="${MODEL:-./models/Qwen3.5-4B}"
TP="${TP:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PY="${PY:-/home/c30084464/miniconda3/envs/sr_opsd/bin/python}"
SMOKE="${SMOKE:-0}"
SAMPLE_ROWS="${SAMPLE_ROWS:-64}"

mkdir -p "$OUT_ROOT"
STATUS="${OUT_ROOT}/status.txt"

echo "teacher keyframe probe $(date -Is) smoke=${SMOKE}" | tee -a "$STATUS"

echo "BUILD $(date -Is)" | tee -a "$STATUS"
if ! "$PY" scripts/opsd/build_teacher_keyframe_sweep.py --out "$PARQUET_ROOT" \
  > "${OUT_ROOT}/build.log" 2>&1; then
  # Builder only needs numpy/pandas; fall back to system python when sr_opsd lacks them.
  /usr/bin/python3 scripts/opsd/build_teacher_keyframe_sweep.py \
    --out "$PARQUET_ROOT" \
    > "${OUT_ROOT}/build.log" 2>&1
fi
tail -n 20 "${OUT_ROOT}/build.log" | tee -a "$STATUS"

run_job() {
  local name="$1" parquet="$2"
  local dir="${OUT_ROOT}/${name}"
  mkdir -p "$dir"
  echo "START ${name} $(date -Is)" | tee -a "$STATUS"
  local started=$SECONDS
  local sample_args=()
  if [[ "$SMOKE" == "1" ]]; then
    # Rows are emitted as full/key pairs; limit-rows preserves pairing, unlike
    # sample-rows which draws individual rows and breaks sweep_group alignment.
    sample_args=(--limit-rows "$SAMPLE_ROWS")
  fi
  set +e
  "$PY" scripts/opsd/tools/run_teacher_reliability.py \
    --parquet "$parquet" \
    --output-dir "$dir" \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${sample_args[@]}" \
    > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  echo "END   ${name} rc=${rc} elapsed=$((SECONDS - started))s $(date -Is)" | tee -a "$STATUS"
  if [[ $rc -ne 0 ]]; then
    tail -n 40 "${dir}/run.log" >&2
    exit "$rc"
  fi

  "$PY" scripts/opsd/tools/score_teacher_dump.py \
    --dump "${dir}/generations.jsonl" \
    --parquet "$parquet" \
    --group-by sweep_arm \
    > "${dir}/score.log" 2>&1
  tail -n 30 "${dir}/score.log" | tee -a "$STATUS"

  if [[ -s "${dir}/judge_queue.jsonl" ]]; then
    echo "JUDGE ${name} pending=$(wc -l < "${dir}/judge_queue.jsonl") $(date -Is)" | tee -a "$STATUS"
    echo "  (judge tier skipped in probe driver; run run_teacher_judging.sh manually if needed)" | tee -a "$STATUS"
  fi

  if [[ -f "${dir}/scored.jsonl" ]]; then
    set +e
    "$PY" scripts/opsd/tools/summarize_keyframe_probe.py \
      --scored "${dir}/scored.jsonl" \
      --out "${dir}/paired_summary.json" \
      | tee "${dir}/paired_summary.log"
    set -e
  fi
}

run_job spar32 "${PARQUET_ROOT}/sweep_spar32_keyframe.parquet"
run_job spar3 "${PARQUET_ROOT}/sweep_spar3_keyframe.parquet"

echo "ALL DONE $(date -Is)" | tee -a "$STATUS"
