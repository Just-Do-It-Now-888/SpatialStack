#!/usr/bin/env bash
# Vision-OPD style evaluation: long generation, then graded by rules + LLM judge.
#
# Replaces the lmms_eval path for CV-Bench. That path capped generation at 16
# tokens and parsed the first [A-F] byte out of the surviving prose, which made
# MV-OPSD v0's whole CV-Bench curve a count of the word "Based" (ISSUE-003).
# Here the model runs to completion and a judge reads the finished answer.
#
#   POLICY_MODEL=output/..._hf RUN_TAG=v0_step300 \
#     bash scripts/opsd/eval_visionopd/run_eval.sh
#
# Everything is a preflight or a fail-fast: LESSON-007 was twelve chained eval
# jobs spinning for hours on an env that could not import the harness.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

POLICY_MODEL="${POLICY_MODEL:?POLICY_MODEL must be set (HF dir of the model under test)}"
RUN_TAG="${RUN_TAG:?RUN_TAG must be set (names the output dir)}"

BENCHMARK="${BENCHMARK:-cvbench}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"
LETTER_MODE="${LETTER_MODE:-faithful}"
VERDICT_MODE="${VERDICT_MODE:-faithful}"
MAX_RESPONSE_CHARS="${MAX_RESPONSE_CHARS:-0}"
# Vision-OPD's own default. At 64 the server reported "Running: 64 reqs,
# Waiting: 0" with 38% KV cache in use -- the client, not the GPU, was the
# limit, and a trained checkpoint that answers at length took hours.
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/gpt-oss-120b}"
NO_LLM_JUDGE="${NO_LLM_JUDGE:-0}"
# Upstream sends no gpt-oss knob. Empty omits it.
REASONING_EFFORT="${REASONING_EFFORT:-}"

POLICY_GPUS="${POLICY_GPUS:-0,1,2,3}"
JUDGE_GPUS="${JUDGE_GPUS:-4,5,6,7}"
POLICY_PORT="${POLICY_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
# Room for the image tokens plus a full-length answer.
POLICY_MAX_LEN="${POLICY_MAX_LEN:-40960}"
# Upstream never starts the judge -- it takes a JUDGE_API_BASE -- so the window
# is ours to size, and it has to hold a whole graded response plus the verdict.
# At 8192 every response longer than the window came back as an empty reply,
# which scores the row wrong: 272 of step 300's 294 unreadable verdicts were
# nothing but that (ISSUE-008). Sizing it from MAX_TOKENS keeps the judge able
# to read anything the policy is allowed to generate.
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-$((MAX_TOKENS + 8192))}"
MAX_IMAGES="${MAX_IMAGES:-32}"
GPU_UTIL="${GPU_UTIL:-0.85}"

PYTHON="${PYTHON:-/usr/bin/python3}"
EVAL_DIR="scripts/opsd/eval_visionopd"
DATA_DIR="data/eval/visionopd"
OUT_ROOT="logs/eval_visionopd/${RUN_TAG}"
SERVER_LOG_DIR="${OUT_ROOT}/server"
mkdir -p "${OUT_ROOT}" "${SERVER_LOG_DIR}"

POLICY_PID=""
JUDGE_PID=""

# A plain SIGTERM is not enough. vLLM ignores it while the engine is still
# allocating, and its EngineCore workers are separate processes that survive the
# API server, so an interrupted run used to leave every card pinned until the
# PIDs were hunted down and SIGKILLed by hand.
stop_server() {
  local pid="$1"
  [[ -z "${pid}" ]] && return 0
  kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null; return 0; }
    sleep 1
  done
  kill -9 -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null
  wait "${pid}" 2>/dev/null
  return 0
}

cleanup() {
  stop_server "${POLICY_PID}"
  stop_server "${JUDGE_PID}"
  # EngineCore workers are re-parented to init when the server goes away.
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  return 0
}
trap cleanup EXIT INT TERM

log() { echo "[$(date +%H:%M:%S)] $*"; }
# serve() is read through $(...), so anything it prints other than the PID would
# be captured as part of the PID.
logerr() { echo "[$(date +%H:%M:%S)] $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
log "preflight"
"${PYTHON}" - <<'PY' || die "python deps missing in ${PYTHON}"
import importlib.util, sys
missing = [m for m in ("vllm", "openai", "tqdm", "datasets") if not importlib.util.find_spec(m)]
if missing:
    print("missing:", missing, file=sys.stderr)
    sys.exit(1)
PY
[[ -f "${POLICY_MODEL}/config.json" ]] || die "no config.json in POLICY_MODEL=${POLICY_MODEL}"
if [[ "${NO_LLM_JUDGE}" != "1" ]]; then
  [[ -f "${JUDGE_MODEL_PATH}/config.json" ]] || die "no config.json in JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}"
fi

BENCH_SUFFIX=""
[[ "${LIMIT}" != "0" ]] && BENCH_SUFFIX="_${LIMIT}"
BENCH_JSON="${DATA_DIR}/${BENCHMARK}${BENCH_SUFFIX}.json"
if [[ ! -f "${BENCH_JSON}" ]]; then
  log "building ${BENCH_JSON}"
  "${PYTHON}" "${EVAL_DIR}/prepare_${BENCHMARK}.py" --limit "${LIMIT}" \
    || die "failed to build ${BENCH_JSON}"
fi
[[ -f "${BENCH_JSON}" ]] || die "benchmark json still missing: ${BENCH_JSON}"

# --- servers ---------------------------------------------------------------
# gpt-oss ships MXFP4 weights and vLLM 0.18 picks its Triton MoE kernels by
# default, but the triton_kernels build in this image has a different routing
# API ('tuple' object has no attribute 'mask'), so the engine dies during the
# profile run. Marlin reaches the same weights without triton_kernels.
export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"

serve() {
  local name="$1" model="$2" gpus="$3" port="$4" maxlen="$5" tp="$6"
  shift 6
  logerr "starting ${name} server on GPU ${gpus} port ${port}"
  # setsid gives the server its own process group, so cleanup can signal the
  # whole tree rather than just the parent.
  CUDA_VISIBLE_DEVICES="${gpus}" setsid "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${model}" \
    --served-model-name "${name}" \
    --port "${port}" \
    --tensor-parallel-size "${tp}" \
    --max-model-len "${maxlen}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --trust-remote-code \
    "$@" > "${SERVER_LOG_DIR}/${name}.log" 2>&1 &
  echo $!
}

wait_healthy() {
  local name="$1" port="$2" pid="$3" timeout="${4:-1800}"
  local waited=0
  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "--- last 40 lines of ${name} server log ---" >&2
      tail -n 40 "${SERVER_LOG_DIR}/${name}.log" >&2
      die "${name} server died during startup"
    fi
    if (( waited >= timeout )); then
      die "${name} server not healthy after ${timeout}s"
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "${name} server healthy after ${waited}s"
}

policy_tp=$(awk -F, '{print NF}' <<< "${POLICY_GPUS}")
POLICY_PID=$(serve policy "${POLICY_MODEL}" "${POLICY_GPUS}" "${POLICY_PORT}" \
  "${POLICY_MAX_LEN}" "${policy_tp}" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}}")

if [[ "${NO_LLM_JUDGE}" != "1" ]]; then
  judge_tp=$(awk -F, '{print NF}' <<< "${JUDGE_GPUS}")
  JUDGE_PID=$(serve judge "${JUDGE_MODEL_PATH}" "${JUDGE_GPUS}" "${JUDGE_PORT}" \
    "${JUDGE_MAX_LEN}" "${judge_tp}")
fi

wait_healthy policy "${POLICY_PORT}" "${POLICY_PID}"

# --- inference -------------------------------------------------------------
ANSWER_DIR="${OUT_ROOT}/model_answer"
MODEL_NAME="${RUN_TAG}_seed${SEED}"
ANSWER_JSON="${ANSWER_DIR}/${BENCHMARK}/${MODEL_NAME}_answer.jsonl"

log "inference: max_tokens=${MAX_TOKENS} thinking=${ENABLE_THINKING}"
"${PYTHON}" "${EVAL_DIR}/infer.py" \
  --benchmark "${BENCHMARK}" \
  --benchmark-json "${BENCH_JSON}" \
  --out-dir "${ANSWER_DIR}" \
  --model-name "${MODEL_NAME}" \
  --api-base "http://127.0.0.1:${POLICY_PORT}/v1/" \
  --model-id policy \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --seed "${SEED}" \
  --enable-thinking "${ENABLE_THINKING}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  || die "inference failed"

# The policy server is no longer needed; free the cards before judging.
if [[ -n "${POLICY_PID}" ]]; then
  kill "${POLICY_PID}" 2>/dev/null
  wait "${POLICY_PID}" 2>/dev/null
  POLICY_PID=""
fi

# --- judge -----------------------------------------------------------------
JUDGE_JSON="${OUT_ROOT}/judge/${BENCHMARK}/${MODEL_NAME}_judge.json"
JUDGE_ARGS=(
  --benchmark "${BENCHMARK}"
  --answer-json "${ANSWER_JSON}"
  --out-json "${JUDGE_JSON}"
  --letter-mode "${LETTER_MODE}"
  --verdict-mode "${VERDICT_MODE}"
  --max-response-chars "${MAX_RESPONSE_CHARS}"
  --parallel-workers "${PARALLEL_WORKERS}"
)
if [[ "${NO_LLM_JUDGE}" == "1" ]]; then
  JUDGE_ARGS+=(--no-llm-judge)
else
  wait_healthy judge "${JUDGE_PORT}" "${JUDGE_PID}"
  JUDGE_ARGS+=(
    --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1/"
    --judge-model judge
    --reasoning-effort "${REASONING_EFFORT}"
  )
fi

log "judging (letter-mode=${LETTER_MODE} verdict-mode=${VERDICT_MODE})"
"${PYTHON}" "${EVAL_DIR}/judge.py" "${JUDGE_ARGS[@]}" || die "judging failed"

# --- score -----------------------------------------------------------------
log "scoring"
"${PYTHON}" "${EVAL_DIR}/cal_acc.py" \
  --benchmark "${BENCHMARK}" \
  --judge-json "${JUDGE_JSON}" \
  | tee "${OUT_ROOT}/${BENCHMARK}_score.txt" \
  || die "scoring failed"

log "done: ${OUT_ROOT}"
