#!/usr/bin/env bash
# Judge-extract scoring for any VSI-Bench samples.jsonl dump.
#
#   bash scripts/opsd/run_judge_extract.sh <samples.jsonl> [out_dir]
#
# Starts gpt-oss-120b on JUDGE_GPUS, has it extract an option letter or a number
# (or NONE) from every response, scores those with the same lmms_eval metrics as
# the rule tier, then shuts the server down.
#
# This is the primary scoring tier as of 2026-08-23 (user decision): the rule
# parser produces false positives that survive to full marks -- id=168 scored
# 1.0 off the frame number 178 in an `Image N:` enumeration -- so rule scores
# are kept as a diagnostic column, not as the reported number.
#
#   JUDGE_GPUS=4,5,6,7   cards for the judge (TP is inferred from the count)
#   JUDGE_PORT=8001      must be free
#   PARALLEL_WORKERS=256 concurrent API requests
#
# Outputs under <out_dir> (default <dir of samples>/judge_extract):
#   samples.jsonl  every input row plus judge_parsed / judge_score /
#                  judge_answered / judge_failure_reason / rule_score_prev
#   summary.json   overall + per-type + delta_vs_rule
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

SAMPLES="${1:-}"
if [[ -z "$SAMPLES" ]]; then
  echo "usage: $0 <samples.jsonl> [out_dir]" >&2
  exit 1
fi
if [[ ! -s "$SAMPLES" ]]; then
  echo "no such samples file: ${SAMPLES}" >&2
  exit 1
fi
OUT_DIR="${2:-$(dirname "${SAMPLES}")/judge_extract}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/gpt-oss-120b}"
JUDGE_GPUS="${JUDGE_GPUS:-4,5,6,7}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-40960}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
PYTHON="${PYTHON:-/usr/bin/python3}"

[[ -f "${JUDGE_MODEL_PATH}/config.json" ]] || {
  echo "judge model missing: ${JUDGE_MODEL_PATH}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
echo "JUDGE_EXTRACT_START at=$(date +%F\ %T) samples=${SAMPLES} out=${OUT_DIR}"

export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"
judge_tp=$(awk -F, '{print NF}' <<< "${JUDGE_GPUS}")

JUDGE_PID=""
cleanup() {
  [[ -z "${JUDGE_PID}" ]] && return 0
  kill -- "-${JUDGE_PID}" 2>/dev/null || kill "${JUDGE_PID}" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "${JUDGE_PID}" 2>/dev/null || break
    sleep 1
  done
  # Only the judge's own process group. A broader pattern kill would also reap
  # the rollout engine of a training job sharing this node.
  kill -9 -- "-${JUDGE_PID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

# The server runs on the system python, which keeps vllm in the user site
# directory; `conda activate` (needed by the grader below) exports
# PYTHONNOUSERSITE=1, which hides exactly that directory (LESSON-007).
CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" setsid env -u PYTHONNOUSERSITE "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH}" --served-model-name judge --port "${JUDGE_PORT}" \
  --tensor-parallel-size "${judge_tp}" --max-model-len "${JUDGE_MAX_LEN}" \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  > "${OUT_DIR}/judge_server.log" 2>&1 &
JUDGE_PID=$!

waited=0
until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${JUDGE_PID}" 2>/dev/null || { tail -40 "${OUT_DIR}/judge_server.log"; exit 1; }
  (( waited >= 1800 )) && { echo "judge timeout"; exit 1; }
  sleep 5; waited=$((waited + 5))
done
echo "judge healthy after ${waited}s"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${EVAL_CONDA_ENV:-sr_opsd}"
  set -u
fi
unset PYTHONPATH

python scripts/opsd/tools/judge_vsibench_full_extract.py \
  "${SAMPLES}" \
  --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1/" \
  --judge-model judge \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --output-dir "${OUT_DIR}"
rc=$?

echo "JUDGE_EXTRACT_DONE rc=${rc} at=$(date +%F\ %T) out=${OUT_DIR}"
exit "${rc}"
