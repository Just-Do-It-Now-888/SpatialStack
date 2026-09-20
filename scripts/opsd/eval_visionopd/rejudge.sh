#!/usr/bin/env bash
# Re-grade existing answer files without re-running inference.
#
# Judging is the cheap half of the pipeline, and the answers are already on
# disk, so a change to the grader (a new tier, a different letter mode, a fixed
# prompt) should never cost another generation pass.
#
#   bash scripts/opsd/eval_visionopd/rejudge.sh mvopsd_v0_step50 mvopsd_v0_step300

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

BENCHMARK="${BENCHMARK:-cvbench}"
SEED="${SEED:-42}"
# Defaults mirror scripts/opsd/eval_visionopd/run_eval.sh, which mirrors upstream.
LETTER_MODE="${LETTER_MODE:-faithful}"
VERDICT_MODE="${VERDICT_MODE:-faithful}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/gpt-oss-120b}"
JUDGE_GPUS="${JUDGE_GPUS:-4,5,6,7}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
# Must fit a whole graded response; see the note in run_eval.sh (ISSUE-008).
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-40960}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
MAX_RESPONSE_CHARS="${MAX_RESPONSE_CHARS:-0}"
GPU_UTIL="${GPU_UTIL:-0.85}"
PYTHON="${PYTHON:-/usr/bin/python3}"

export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"

EVAL_DIR="scripts/opsd/eval_visionopd"
LOG_DIR="logs/eval_visionopd/rejudge"
mkdir -p "${LOG_DIR}"

(( $# )) || { echo "usage: $0 <run_tag> [run_tag ...]" >&2; exit 1; }

JUDGE_PID=""
cleanup() {
  [[ -z "${JUDGE_PID}" ]] && return 0
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
echo "[$(date +%H:%M:%S)] starting judge on GPU ${JUDGE_GPUS}"
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

for tag in "$@"; do
  answer="logs/eval_visionopd/${tag}/model_answer/${BENCHMARK}/${tag}_seed${SEED}_answer.jsonl"
  out="logs/eval_visionopd/${tag}/judge/${BENCHMARK}/${tag}_seed${SEED}_judge.json"
  if [[ ! -f "${answer}" ]]; then
    echo "SKIP ${tag}: no answers at ${answer}"
    continue
  fi
  echo
  echo "########## ${tag} ##########"
  "${PYTHON}" "${EVAL_DIR}/judge.py" \
    --benchmark "${BENCHMARK}" --answer-json "${answer}" --out-json "${out}" \
    --letter-mode "${LETTER_MODE}" --verdict-mode "${VERDICT_MODE}" \
    --parallel-workers "${PARALLEL_WORKERS}" \
    --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1/" --judge-model judge \
    --reasoning-effort "${REASONING_EFFORT}" \
    --max-response-chars "${MAX_RESPONSE_CHARS}" || continue
  "${PYTHON}" "${EVAL_DIR}/cal_acc.py" --benchmark "${BENCHMARK}" --judge-json "${out}" \
    | tee "logs/eval_visionopd/${tag}/${BENCHMARK}_score.txt"
done
