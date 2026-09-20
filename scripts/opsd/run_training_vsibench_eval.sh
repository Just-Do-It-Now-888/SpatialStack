#!/usr/bin/env bash
# Full VSI-Bench eval for one training checkpoint: vLLM generation then judge.
#
#   bash scripts/opsd/run_training_vsibench_eval.sh \
#     checkpoints/20260820_qwen35base_mvopsd_khalf_main/global_step_120
#
# Phase 1 uses all 8 GPUs for the policy (vLLM).  Phase 2 starts gpt-oss-120b
# on the same cards and grades the responses.
#
# Phase 2 has two modes, and since 2026-08-23 the default is the strict one
# (user decision, after reviewing rule/judge disagreements by hand):
#
#   JUDGE_MODE=extract   default. The judge extracts an option letter or a
#                        number (or NONE) from *every* response; those are the
#                        reported scores. Rule scores stay in the dump as a
#                        diagnostic column. Rules produce false positives that
#                        reach full marks -- id=168 scored 1.0 off the frame
#                        number 178 inside an `Image N:` enumeration -- so the
#                        rule number is an upper bound, not the result.
#                        Written to <out>/judge_extract/; the root summary.json
#                        gets a `primary` block pointing at it.
#   JUDGE_MODE=cascade   the old Vision-OPD behaviour: rules score first and
#                        only rows where no answer could be read reach the
#                        judge, which can therefore only add points. Kept to
#                        reproduce every number published before 2026-08-23.
#
#   JUDGE=0          skip phase 2 entirely (rule-only, faster)
#   FRAMES=32 MAX_TOKENS=4096   defaults match the spatialstack protocol
#   PROMPT_SUFFIX=' The final answer MUST BE put in \boxed{}.'  boxed eval prompt
#   For last-line + boxed (matches the rule parser): 
#   PROMPT_SUFFIX=' The final answer MUST BE put in \boxed{} on the last line of your response.'
#   BOXED_PRIMARY=1   closed \\boxed{} is the only answer site (in-training rule)
#   TRUST_BOXED=1 TRUST_TERSE=1 TRUST_MCA_TAIL=1   skip judge for boxes, terse, MCA tails
#   LIMIT_SCENES=4   first N scenes only; for checking the wiring, not for numbers
#   DO_SAMPLE=1      sample instead of greedy (TEMPERATURE / TOP_P / TOP_K /
#                    MIN_P / PRESENCE_PENALTY / SEED tune it; defaults 1.0 / 0.8
#                    / -1 / 0 / 0). Greedy is what every published number here
#                    was measured under, so a sampled run is a separate series.
#
# Outputs: logs/vsi_train_eval/<experiment>/<step>/
#   summary.json  overall + per-question-type score, answered rate, truncation,
#                 output-length percentiles, failure counts, rule vs judged
#   samples.jsonl every row: response, tokens, rule tier, final tier, tag
#   failures/     the same rows split by tag, for reading rather than counting:
#                 truncation_error, parse_error, factual_error, judge_recovered
#
# Phase 2 rewrites summary.json and samples.jsonl in place, so after a judged
# run summary.json carries both the rule-only and the judge-assisted number.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

FRAMES="${FRAMES:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TP="${TENSOR_PARALLEL_SIZE:-8}"
SCENES_PER_BATCH="${SCENES_PER_BATCH:-24}"
LIMIT_SCENES="${LIMIT_SCENES:-0}"
DO_SAMPLE="${DO_SAMPLE:-0}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"
BOXED_PRIMARY="${BOXED_PRIMARY:-0}"
TRUST_BOXED="${TRUST_BOXED:-1}"
TRUST_TERSE="${TRUST_TERSE:-1}"
TRUST_MCA_TAIL="${TRUST_MCA_TAIL:-1}"
JUDGE="${JUDGE:-1}"
JUDGE_MODE="${JUDGE_MODE:-extract}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-models/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
JUDGE_GPUS="${JUDGE_GPUS:-0,1,2,3,4,5,6,7}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-40960}"
PYTHON="${PYTHON:-/usr/bin/python3}"
CONDA_ENV="${EVAL_CONDA_ENV:-sr_opsd}"

HF_DIR=""
CKPT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf) HF_DIR="$2"; shift 2 ;;
    *) CKPT_DIR="${1%/}"; shift ;;
  esac
done

if [[ -z "$HF_DIR" && -z "$CKPT_DIR" ]]; then
  echo "usage: $0 <checkpoints/<exp>/global_step_N> | --hf <merged_hf_dir>" >&2
  exit 1
fi

if [[ -n "$CKPT_DIR" ]]; then
  EXPERIMENT="$(basename "$(dirname "$CKPT_DIR")")"
  STEP="$(basename "$CKPT_DIR")"
  HF_DIR="${HF_DIR:-output/${EXPERIMENT}_${STEP}_hf}"
  if [[ ! -d "$HF_DIR" ]]; then
    echo "[merge] ${CKPT_DIR} -> ${HF_DIR}"
    bash scripts/opsd/merge_mvopsd_ckpt.sh "$CKPT_DIR" "$HF_DIR"
  fi
else
  EXPERIMENT="$(basename "$HF_DIR")"
  STEP="$(basename "$HF_DIR" | grep -oE 'global_step_[0-9]+|step[0-9]+' | tail -1 || echo hf)"
fi

# The default path is derived from the checkpoint, so a second run of the same
# step overwrites the first one's samples.jsonl and summary.json in place.
# Anything that changes the protocol (decoding, prompt suffix, token budget)
# has to be given its own OUT_DIR or it destroys the archive it should be
# compared against.
OUT_DIR="${OUT_DIR:-logs/vsi_train_eval/${EXPERIMENT}/${STEP}}"
mkdir -p "$OUT_DIR"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SH" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV}"
  set -u
fi
unset PYTHONPATH

# Only forwarded when DO_SAMPLE=1: the eval refuses a sampling knob without
# --do-sample rather than silently running half a protocol.
DECODE_ARGS=()
if [[ "$DO_SAMPLE" == "1" ]]; then
  DECODE_ARGS+=(--do-sample)
  [[ -n "${TEMPERATURE:-}" ]] && DECODE_ARGS+=(--temperature "${TEMPERATURE}")
  [[ -n "${TOP_P:-}" ]] && DECODE_ARGS+=(--top-p "${TOP_P}")
  [[ -n "${TOP_K:-}" ]] && DECODE_ARGS+=(--top-k "${TOP_K}")
  [[ -n "${MIN_P:-}" ]] && DECODE_ARGS+=(--min-p "${MIN_P}")
  [[ -n "${PRESENCE_PENALTY:-}" ]] && DECODE_ARGS+=(--presence-penalty "${PRESENCE_PENALTY}")
  [[ -n "${SEED:-}" ]] && DECODE_ARGS+=(--seed "${SEED}")
fi

echo "=== phase 1: vLLM generation + rule scoring ==="
echo "model=${HF_DIR} -> ${OUT_DIR}"
echo "decoding args: ${DECODE_ARGS[*]:-greedy}"
PROMPT_ARGS=()
if [[ -n "$PROMPT_SUFFIX" ]]; then
  PROMPT_ARGS+=(--prompt-suffix "$PROMPT_SUFFIX")
  echo "prompt_suffix=${PROMPT_SUFFIX}"
fi
if [[ "$BOXED_PRIMARY" == "1" ]]; then
  PROMPT_ARGS+=(--boxed-primary)
  echo "boxed_primary=1"
fi
python scripts/opsd/tools/eval_vsibench_vllm.py \
  --model "${HF_DIR}" \
  --tensor-parallel-size "${TP}" \
  --scenes-per-batch "${SCENES_PER_BATCH}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --frames "${FRAMES}" \
  --limit-scenes "${LIMIT_SCENES}" \
  ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"} \
  ${DECODE_ARGS[@]+"${DECODE_ARGS[@]}"} \
  --output-dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/eval.log"

if [[ ! -s "${OUT_DIR}/samples.jsonl" ]]; then
  echo "phase 1 produced no samples; see ${OUT_DIR}/eval.log" >&2
  exit 1
fi

if [[ "$JUDGE" != "1" ]]; then
  echo "VSI_EVAL_DONE (rule-only) ${OUT_DIR}"
  exit 0
fi

[[ -f "${JUDGE_MODEL_PATH}/config.json" ]] || { echo "judge model missing: ${JUDGE_MODEL_PATH}" >&2; exit 1; }

# The policy engine sometimes lingers in shutdown after the script's last line.
# Starting the judge on cards it still holds fails on out-of-memory, so wait.
echo "[phase 1 -> 2] waiting for the policy to release GPU memory"
for _ in $(seq 1 60); do
  peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  (( peak < 5000 )) && break
  sleep 5
done
echo "[phase 1 -> 2] peak GPU memory now ${peak} MiB"

JUDGE_PID=""
cleanup() {
  [[ -z "${JUDGE_PID}" ]] && return 0
  kill -- "-${JUDGE_PID}" 2>/dev/null || kill "${JUDGE_PID}" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "${JUDGE_PID}" 2>/dev/null || break
    sleep 1
  done
  # Only the judge's own process group.  A broader pattern kill would also
  # reap the rollout engine of a training job sharing this node.
  kill -9 -- "-${JUDGE_PID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "=== phase 2: gpt-oss-120b judge (mode=${JUDGE_MODE}) ==="
export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"
judge_tp=$(awk -F, '{print NF}' <<< "${JUDGE_GPUS}")
# The judge server runs on the system python, which keeps vllm in the user
# site directory; `conda activate` (needed for phase 1) exports
# PYTHONNOUSERSITE=1, which hides exactly that directory.  Drop it for the
# server only -- phase 1 and the grader still run inside sr_opsd (LESSON-007).
CUDA_VISIBLE_DEVICES="${JUDGE_GPUS}" setsid env -u PYTHONNOUSERSITE "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH}" --served-model-name judge --port "${JUDGE_PORT}" \
  --tensor-parallel-size "${judge_tp}" --max-model-len "${JUDGE_MAX_LEN}" \
  --gpu-memory-utilization 0.85 --trust-remote-code \
  > "${OUT_DIR}/judge_server.log" 2>&1 &
JUDGE_PID=$!

waited=0
until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${JUDGE_PID}" 2>/dev/null || { tail -20 "${OUT_DIR}/judge_server.log" >&2; exit 1; }
  (( waited >= 1800 )) && { echo "judge timeout" >&2; exit 1; }
  sleep 5; waited=$((waited + 5))
done
echo "judge healthy after ${waited}s"

if [[ "$JUDGE_MODE" == "extract" ]]; then
  TRUST_ARGS=()
  [[ "$TRUST_BOXED" == "1" ]] && TRUST_ARGS+=(--trust-boxed) || TRUST_ARGS+=(--no-trust-boxed)
  [[ "$TRUST_TERSE" == "1" ]] && TRUST_ARGS+=(--trust-terse) || TRUST_ARGS+=(--no-trust-terse)
  [[ "$TRUST_MCA_TAIL" == "1" ]] && TRUST_ARGS+=(--trust-mca-tail) || TRUST_ARGS+=(--no-trust-mca-tail)
  python scripts/opsd/tools/judge_vsibench_full_extract.py \
    "${OUT_DIR}/samples.jsonl" \
    --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1/" \
    --judge-model judge \
    --parallel-workers "${JUDGE_PARALLEL_WORKERS:-256}" \
    --output-dir "${OUT_DIR}/judge_extract" \
    "${TRUST_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/judge_extract.log"

  # The root summary.json holds the rule tier, which is no longer the reported
  # number. Stamp the primary result into it so nothing downstream has to know
  # that the real score lives one directory deeper.
  python - "${OUT_DIR}" <<'PY' 2>&1 | tee -a "${OUT_DIR}/judge_extract.log"
import json, os, sys

out = sys.argv[1]
root = os.path.join(out, "summary.json")
extract = os.path.join(out, "judge_extract", "summary.json")
if not (os.path.exists(root) and os.path.exists(extract)):
    print("primary stamp skipped: missing summary.json")
    raise SystemExit(0)

with open(root) as fh:
    summary = json.load(fh)
with open(extract) as fh:
    judged = json.load(fh)

summary["scoring_primary"] = judged.get("scoring", "judge_extract_strict_v1")
summary["primary"] = {
    "tier": "judge_extract",
    "overall": judged["overall"]["score"],
    "answered_pct": judged["overall"]["answered_pct"],
    "summary_path": os.path.relpath(extract, out),
}
# Left in place on purpose: the rule number is a known upper bound (it reads
# frame numbers and stray letters as answers), so it stays visible next to the
# primary one rather than being dropped.
summary["rule_diagnostic"] = summary.get("rule_only")
with open(root, "w") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)

print(
    f"primary (judge-extract): {summary['primary']['overall']:.2f} "
    f"| answered {summary['primary']['answered_pct']:.2f}%"
)
print(f"rule diagnostic        : {summary['rule_diagnostic']['overall']:.2f}")
PY
else
  python scripts/opsd/tools/apply_judge_to_dump.py \
    "${OUT_DIR}/samples.jsonl" \
    --judge-api-base "http://127.0.0.1:${JUDGE_PORT}/v1/" \
    --judge-model judge \
    2>&1 | tee "${OUT_DIR}/judge.log"
fi

echo "VSI_EVAL_DONE (judge_mode=${JUDGE_MODE}) ${OUT_DIR} at=$(date +%F\ %T)"
