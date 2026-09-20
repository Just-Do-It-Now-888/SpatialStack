#!/usr/bin/env bash
# Start the gpt-oss-120b judge and re-grade VSI-Bench answers already on disk.
#
#   bash scripts/opsd/tools/rejudge_vsibench.sh \
#     'logs/eval/20260820_qwen35base_vsi_anchor/frames_32/*/vsibench/*/*_samples_vsibench.jsonl'
#
# Mirrors scripts/opsd/eval_visionopd/rejudge.sh: generation is the expensive
# half and the responses are saved, so changing the grader must never cost
# another generation pass. Same judge model and serving flags as the CV-Bench
# path, so a VSI-Bench number and a CV-Bench number are graded by the same judge.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

(( $# )) || { echo "usage: $0 <sample_dump_glob> [more globs ...]" >&2; exit 1; }

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/gpt-oss-120b}"
JUDGE_GPUS="${JUDGE_GPUS:-4,5,6,7}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
# Must fit a whole graded response plus the prompt; VSI-Bench answers run to the
# 1024-token budget, so this is far more headroom than needed (ISSUE-008).
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-40960}"
GPU_UTIL="${GPU_UTIL:-0.85}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
NA_SCOPE="${NA_SCOPE:-unresolved}"
OUTPUT="${OUTPUT:-}"
PYTHON="${PYTHON:-/usr/bin/python3}"

export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"

LOG_DIR="logs/eval/judge"
mkdir -p "${LOG_DIR}"

[[ -f "${JUDGE_MODEL_PATH}/config.json" ]] || {
  echo "no config.json in JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}" >&2; exit 1; }

JUDGE_PID=""
cleanup() {
  [[ -z "${JUDGE_PID}" ]] && return 0
  # setsid put the server in its own group, so signal the whole tree; an
  # interrupted run otherwise leaves every card pinned by EngineCore workers.
  kill -- "-${JUDGE_PID}" 2>/dev/null || kill "${JUDGE_PID}" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "${JUDGE_PID}" 2>/dev/null || break
    sleep 1
  done
  kill -9 -- "-${JUDGE_PID}" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  return 0
}
trap cleanup EXIT INT TERM

judge_tp=$(awk -F, '{print NF}' <<< "${JUDGE_GPUS}")
echo "[$(date +%H:%M:%S)] starting judge (${JUDGE_MODEL_PATH}) on GPU ${JUDGE_GPUS}"
CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" setsid "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH}" --served-model-name judge --port "${JUDGE_PORT}" \
  --tensor-parallel-size "${judge_tp}" --max-model-len "${JUDGE_MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" --trust-remote-code \
  > "${LOG_DIR}/judge_server.log" 2>&1 &
JUDGE_PID=$!

waited=0
until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${JUDGE_PID}" 2>/dev/null || { tail -n 40 "${LOG_DIR}/judge_server.log" >&2; exit 1; }
  (( waited >= 1800 )) && { echo "judge not healthy after ${waited}s" >&2; exit 1; }
  sleep 5; waited=$((waited + 5))
done
echo "[$(date +%H:%M:%S)] judge healthy after ${waited}s"

# The grader needs lmms_eval, which only exists in sr_opsd; the judge server runs
# on the system python (LESSON-007).
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${EVAL_CONDA_ENV:-sr_opsd}"
  set -u
fi
unset PYTHONPATH

args=(--judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1/" --judge-model judge
      --parallel-workers "${PARALLEL_WORKERS}" --na-scope "${NA_SCOPE}"
      --reasoning-effort "${REASONING_EFFORT}")
[[ -n "${OUTPUT}" ]] && args+=(--output "${OUTPUT}")

python scripts/opsd/tools/judge_vsibench.py "$@" "${args[@]}"
status=$?
echo "REJUDGE_DONE status=${status} at=$(date +%F\ %T)"
exit "${status}"
