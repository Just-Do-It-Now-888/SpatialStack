#!/usr/bin/env bash
# Generate the frozen teacher's answers for the whole MV-OPSD training pool and
# for the two controlled view sweeps, one job after another on the same 8 cards.
#
#   setsid nohup bash scripts/opsd/run_teacher_reliability.sh \
#     > logs/eval/teacher_reliability/driver.log 2>&1
#
# Each job resumes from its own generations.jsonl, so an interrupted driver can
# simply be relaunched. Scoring is deliberately not part of this script: it is
# CPU-only, needs no cards, and the parsers will be revised after the first look
# at the dump.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-logs/eval/teacher_reliability}"
MODEL="${MODEL:-./models/Qwen3.5-4B}"
TP="${TP:-8}"
# The training environment, not conda sr_opsd: verl's vLLM lives in the system
# python + ~/.local (LESSON-007).
PY="${PY:-/usr/bin/python3}"

mkdir -p "$OUT_ROOT"
STATUS="${OUT_ROOT}/status.txt"

run_job() {
  local name="$1" parquet="$2"
  local dir="${OUT_ROOT}/${name}"
  mkdir -p "$dir"
  echo "START ${name} $(date -Is)" | tee -a "$STATUS"
  local started=$SECONDS
  set +e
  "$PY" scripts/opsd/tools/run_teacher_reliability.py \
    --parquet "$parquet" \
    --output-dir "$dir" \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  echo "END   ${name} rc=${rc} elapsed=$((SECONDS - started))s $(date -Is)" | tee -a "$STATUS"
  if [[ $rc -ne 0 ]]; then
    echo "--- last 40 lines of ${dir}/run.log ---" >&2
    tail -n 40 "${dir}/run.log" >&2
    # Abort the chain: a job that dies in its first minutes is a config fault,
    # and the later jobs would burn hours reproducing it (LESSON-007).
    exit "$rc"
  fi
}

echo "teacher reliability driver $(date -Is)" | tee -a "$STATUS"
run_job as_trained   data/mvopsd/parquet/main_train.parquet
run_job sweep_vlm3r  data/mvopsd/parquet_view_sweep/sweep_vlm3r.parquet
run_job sweep_spar32 data/mvopsd/parquet_view_sweep/sweep_spar32.parquet
echo "ALL DONE $(date -Is)" | tee -a "$STATUS"
