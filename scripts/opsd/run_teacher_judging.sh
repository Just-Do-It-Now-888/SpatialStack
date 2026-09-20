#!/usr/bin/env bash
# Score the teacher dumps and run the second grading tier over them.
#
#   setsid nohup bash scripts/opsd/run_teacher_judging.sh \
#     > logs/eval/teacher_reliability/judging.log 2>&1
#
# Run this only after run_teacher_reliability.sh has finished: gpt-oss-120b
# wants the same eight cards the generation is using. The rule tier alone is
# CPU-only and can be run any time with tools/score_teacher_dump.py.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-logs/eval/teacher_reliability}"
PY="${PY:-/usr/bin/python3}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${PROJECT_ROOT}/models/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
# Teacher responses run to 1024 tokens and the SPAR BEV questions are long, so
# the window has to hold both plus the extraction instruction.
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-16384}"
JUDGE_PID=""

STATUS="${OUT_ROOT}/judge_status.txt"
mkdir -p "$OUT_ROOT"

stop_judge() {
  [[ -n "$JUDGE_PID" ]] || return 0
  kill -- "-${JUDGE_PID}" 2>/dev/null || kill "$JUDGE_PID" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 "$JUDGE_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 -- "-${JUDGE_PID}" 2>/dev/null || true
  JUDGE_PID=""
}
trap stop_judge EXIT INT TERM

# --- rule tier first: it needs no cards and tells us how much judging is left
score_one() {
  local name="$1" parquet="$2" group="$3"
  local dir="${OUT_ROOT}/${name}"
  [[ -f "${dir}/generations.jsonl" ]] || { echo "skip ${name}: no dump" | tee -a "$STATUS"; return 1; }
  echo "RULE ${name} $(date -Is)" | tee -a "$STATUS"
  "$PY" scripts/opsd/tools/score_teacher_dump.py \
    --dump "${dir}/generations.jsonl" \
    --parquet "$parquet" \
    --group-by "$group" \
    > "${dir}/score.log" 2>&1
  tail -n 4 "${dir}/score.log"
}

score_one as_trained   data/mvopsd/parquet/main_train.parquet                 n_views_teacher || true
score_one sweep_vlm3r  data/mvopsd/parquet_view_sweep/sweep_vlm3r.parquet     sweep_views     || true
score_one sweep_spar32 data/mvopsd/parquet_view_sweep/sweep_spar32.parquet    sweep_views     || true

# --- judge tier
if [[ ! -f "${JUDGE_MODEL_PATH}/config.json" ]]; then
  echo "no judge model at ${JUDGE_MODEL_PATH}; rule-only scores stand" | tee -a "$STATUS"
  exit 0
fi

JUDGE_LOG="${OUT_ROOT}/judge_server_$(date +%Y%m%d_%H%M%S).log"
echo "starting gpt-oss-120b on port ${JUDGE_PORT} (log ${JUDGE_LOG})" | tee -a "$STATUS"
# MXFP4 needs Marlin in this image; triton_kernels crashes gpt-oss during the
# profile run (ISSUE-005). No sleep/wake dance here: nothing else wants the
# cards once generation is done.
VLLM_MXFP4_USE_MARLIN=1 setsid "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL_PATH" \
  --served-model-name judge \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size 8 \
  --max-model-len "$JUDGE_MAX_LEN" \
  --gpu-memory-utilization 0.85 \
  > "$JUDGE_LOG" 2>&1 &
JUDGE_PID=$!

waited=0
until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
    echo "--- last 40 lines of the judge log ---" >&2
    tail -n 40 "$JUDGE_LOG" >&2
    echo "judge server died during startup" >&2
    exit 1
  fi
  if (( waited >= 1800 )); then
    echo "judge server not healthy after ${waited}s" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done
echo "judge healthy after ${waited}s" | tee -a "$STATUS"

# The MXFP4 checkpoint can come up healthy and reply "!!!!!!!!" to everything
# (ISSUE-005). Check before spending an hour of calls on it.
"$PY" scripts/opsd/tools/check_judge_ready.py --api-base "http://127.0.0.1:${JUDGE_PORT}/v1" \
  || { echo "judge is not grading correctly; not sending the queue" | tee -a "$STATUS"; exit 1; }

judge_one() {
  local name="$1" group="$2"
  local dir="${OUT_ROOT}/${name}"
  [[ -f "${dir}/judge_queue.jsonl" ]] || return 0
  echo "JUDGE ${name} $(date -Is)" | tee -a "$STATUS"
  "$PY" scripts/opsd/tools/judge_teacher_dump.py \
    --dir "$dir" \
    --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --group-by "$group" \
    > "${dir}/judge.log" 2>&1
  echo "END   ${name} $(date -Is)" | tee -a "$STATUS"
  tail -n 30 "${dir}/judge.log"
}

judge_one as_trained   n_views_teacher
judge_one sweep_vlm3r  sweep_views
judge_one sweep_spar32 sweep_views
echo "ALL DONE $(date -Is)" | tee -a "$STATUS"
