#!/usr/bin/env bash
# Teacher accuracy curve: keep oracle key frames, then add 1..10 extras.
#
#   SMOKE=1 SAMPLE_ROWS=22 bash scripts/opsd/run_teacher_key_plus_probe.sh
#   bash scripts/opsd/run_teacher_key_plus_probe.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-logs/eval/teacher_key_plus_probe}"
PARQUET_ROOT="${PARQUET_ROOT:-data/mvopsd/parquet_key_plus_sweep}"
MODEL="${MODEL:-./models/Qwen3.5-4B}"
TP="${TP:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PY="${PY:-/home/c30084464/miniconda3/envs/sr_opsd/bin/python}"
SMOKE="${SMOKE:-0}"
SAMPLE_ROWS="${SAMPLE_ROWS:-22}"
NAME="${NAME:-spar32}"

mkdir -p "$OUT_ROOT"
STATUS="${OUT_ROOT}/status.txt"

echo "teacher key+extra probe $(date -Is) smoke=${SMOKE}" | tee -a "$STATUS"
echo "BUILD $(date -Is)" | tee -a "$STATUS"
if ! "$PY" scripts/opsd/build_teacher_key_plus_sweep.py --out "$PARQUET_ROOT" \
  > "${OUT_ROOT}/build.log" 2>&1; then
  /usr/bin/python3 scripts/opsd/build_teacher_key_plus_sweep.py --out "$PARQUET_ROOT" \
    > "${OUT_ROOT}/build.log" 2>&1
fi
tail -n 20 "${OUT_ROOT}/build.log" | tee -a "$STATUS"

parquet="${PARQUET_ROOT}/sweep_${NAME}_key_plus.parquet"
dir="${OUT_ROOT}/${NAME}"
mkdir -p "$dir"
echo "START ${NAME} $(date -Is)" | tee -a "$STATUS"
started=$SECONDS
sample_args=()
if [[ "$SMOKE" == "1" ]]; then
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
rc=$?
set -e
echo "END   ${NAME} rc=${rc} elapsed=$((SECONDS - started))s $(date -Is)" | tee -a "$STATUS"
if [[ $rc -ne 0 ]]; then
  tail -n 40 "${dir}/run.log" >&2
  exit "$rc"
fi

"$PY" scripts/opsd/tools/score_teacher_dump.py \
  --dump "${dir}/generations.jsonl" \
  --parquet "$parquet" \
  --group-by extra_added \
  > "${dir}/score.log" 2>&1
tail -n 40 "${dir}/score.log" | tee -a "$STATUS"

if [[ -s "${dir}/judge_queue.jsonl" ]]; then
  echo "JUDGE pending=$(wc -l < "${dir}/judge_queue.jsonl") (skipped in driver)" | tee -a "$STATUS"
fi

"$PY" scripts/opsd/tools/summarize_key_plus_probe.py \
  --scored "${dir}/scored.jsonl" \
  --out "${dir}/curve_summary.json" \
  | tee "${dir}/curve_summary.log"

echo "ALL DONE $(date -Is)" | tee -a "$STATUS"
