#!/usr/bin/env bash
# Does the frozen base model obey a format instruction it was never trained on?
#
#   setsid nohup bash scripts/opsd/run_boxed_probe.sh \
#     > logs/eval/boxed_probe/driver.log 2>&1
#
# Three prompt arms, run on the same model and the same questions:
#
#   plain   nothing appended -- the control, and byte-identical to the prompts
#           already archived, so the pool arm doubles as a parity gate
#   boxed   verl's geo3k instruction, minus the two sentences that ask the model
#           to open a <think> block our chat template has already closed
#           (examples/data_preprocess/geo3k.py)
#   answer  the format math_dapo.py actually reads by default: its
#           is_correct_minerva pattern is (?i)Answer\s*:\s*([^\n]+), and
#           \boxed{} is only reached under strict_box_verify=True, which verl's
#           own router never passes
#
# Set ARMS to a subset, e.g. ARMS="boxed" to skip plain/answer.
# Set SKIP_POOL=1 to skip the training-pool subset (VSI-Bench only).
#
# Two question sets, because compliance is not one number: VSI-Bench is the
# reported metric, the training pool is what the teacher is actually asked
# during distillation, and the teacher prompt has no system turn and a different
# image geometry.
#
# Generation only. Scoring is tools/summarize_boxed_probe.py so the parsers can
# be revised without paying for generation twice.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-logs/eval/boxed_probe}"
MODEL="${MODEL:-./models/Qwen3.5-4B}"
TP="${TP:-8}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
# Room for the longest prompt plus the full response budget (VSI val cap 11264 + 8192).
POOL_MAX_MODEL_LEN="${POOL_MAX_MODEL_LEN:-$((MAX_TOKENS + 11264))}"
VSI_MAX_MODEL_LEN="${VSI_MAX_MODEL_LEN:-$((MAX_TOKENS + 11264))}"
POOL_PARQUET="${POOL_PARQUET:-data/mvopsd/parquet/main_train.parquet}"
# Paired subset: row_index is assigned before subsampling, so the same seed
# draws the same rows in every arm and each row can also be looked up in the
# archived full dump.
POOL_ROWS="${POOL_ROWS:-3000}"
POOL_SEED="${POOL_SEED:-20260820}"

# The training environment, not conda sr_opsd: verl's vLLM lives in the system
# python + ~/.local (LESSON-007). The VSI harness is the other way round.
POOL_PY="${POOL_PY:-/usr/bin/python3}"

# ~/.local has to be importable, in both halves.
#
# For the pool arm that is the whole environment: with PYTHONNOUSERSITE=1
# inherited from the calling shell, /usr/bin/python3 cannot see transformers or
# vllm at all and the job dies on the first import.
#
# The VSI arm is the opposite case and is pinned separately in run_vsi_arm: every
# archived VSI-Bench number resolved conda sr_opsd's own transformers 5.3.0 and
# vllm 0.18.0, and letting ~/.local shadow them makes the image preprocessing
# depend on who launched the job (LESSON-003, LESSON-007). Resolved versions are
# printed into each run log rather than assumed.
export -n PYTHONNOUSERSITE || true
unset PYTHONNOUSERSITE
unset PYTHONPATH

BOXED_SUFFIX=' The final answer MUST BE put in \boxed{}.'
ANSWER_SUFFIX='
Remember to put your answer on its own line after "Answer:".'

# Space-separated subset of {plain,boxed,answer}. Default: all three.
ARMS="${ARMS:-plain boxed answer}"
SKIP_POOL="${SKIP_POOL:-0}"

mkdir -p "$OUT_ROOT"
STATUS="${OUT_ROOT}/status.txt"

log() { echo "$*" | tee -a "$STATUS"; }

# vLLM sizes its allocation as a fraction of *total* memory but fails if that
# much is not free, so a job that starts on a busy card dies during engine
# startup rather than degrading. Check before paying for model load.
require_free_gpu() {
  local need_mib="${1:-88000}"
  local worst
  worst=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  if (( worst < need_mib )); then
    log "ABORT $(date -Is): least-free GPU has ${worst} MiB, need ${need_mib} MiB"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a "$STATUS"
    exit 91
  fi
  log "gpu check ok: least-free GPU has ${worst} MiB"
}

# Killing the launcher leaves the engine and its eight workers orphaned, holding
# the whole card until someone notices (2026-08-23). Reap them on any exit.
reap_engines() {
  pkill -TERM -f 'VLLM::EngineCore' 2>/dev/null || true
  pkill -TERM -f 'VLLM::Worker_TP' 2>/dev/null || true
}
trap reap_engines EXIT INT TERM

run_pool_arm() {
  local name="$1" suffix="$2"
  local dir="${OUT_ROOT}/pool_${name}"
  mkdir -p "$dir"
  log "START pool_${name} $(date -Is)"
  require_free_gpu "${MIN_FREE_MIB:-88000}"
  local started=$SECONDS
  set +e
  "$POOL_PY" -c 'import transformers, vllm, pandas; print(f"env pool: transformers {transformers.__version__} from {transformers.__file__}\nenv pool: vllm {vllm.__version__}")' \
    > "${dir}/env.log" 2>&1 || { echo "pool arm cannot import transformers/vllm; see ${dir}/env.log" >&2; exit 90; }
  cat "${dir}/env.log"
  "$POOL_PY" scripts/opsd/tools/run_teacher_reliability.py \
    --parquet "$POOL_PARQUET" \
    --output-dir "$dir" \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$POOL_MAX_MODEL_LEN" \
    --sample-rows "$POOL_ROWS" \
    --sample-seed "$POOL_SEED" \
    --prompt-suffix "$suffix" \
    > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  log "END   pool_${name} rc=${rc} elapsed=$((SECONDS - started))s $(date -Is)"
  if [[ $rc -ne 0 ]]; then
    echo "--- last 40 lines of ${dir}/run.log ---" >&2
    tail -n 40 "${dir}/run.log" >&2
    exit "$rc"
  fi
}

run_vsi_arm() {
  local name="$1" suffix="$2"
  local dir="${OUT_ROOT}/vsi_${name}"
  mkdir -p "$dir"
  log "START vsi_${name} $(date -Is)"
  require_free_gpu "${MIN_FREE_MIB:-88000}"
  local started=$SECONDS
  set +e
  (
    set +u
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate sr_opsd
    set -u
    unset PYTHONPATH
    # conda-only resolution, which is what every archived VSI-Bench number used.
    # ~/.local currently also carries a vllm whose pydantic-core does not match
    # its pydantic, so letting the user site through breaks the import outright.
    export PYTHONNOUSERSITE=1
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    python -c 'import transformers, vllm, lmms_eval, decord; print(f"env vsi: transformers {transformers.__version__} from {transformers.__file__}\nenv vsi: vllm {vllm.__version__}")' || exit 90
    python scripts/opsd/tools/eval_vsibench_vllm.py \
      --model "$MODEL" \
      --tensor-parallel-size "$TP" \
      --max-tokens "$MAX_TOKENS" \
      --max-model-len "$VSI_MAX_MODEL_LEN" \
      --prompt-suffix "$suffix" \
      --output-dir "$dir"
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  log "END   vsi_${name} rc=${rc} elapsed=$((SECONDS - started))s $(date -Is)"
  if [[ $rc -ne 0 ]]; then
    echo "--- last 40 lines of ${dir}/run.log ---" >&2
    tail -n 40 "${dir}/run.log" >&2
    exit "$rc"
  fi
}

log "boxed probe driver $(date -Is) model=${MODEL} max_tokens=${MAX_TOKENS} arms=${ARMS} skip_pool=${SKIP_POOL} pool_rows=${POOL_ROWS}"

if [[ "$SKIP_POOL" != "1" ]]; then
  # Pool first: it is the cheap half, so a mistake in the prompt plumbing shows up
  # in minutes rather than after an hour of VSI-Bench.
  for arm in $ARMS; do
    case "$arm" in
      plain)  run_pool_arm plain  '' ;;
      boxed)  run_pool_arm boxed  "$BOXED_SUFFIX" ;;
      answer) run_pool_arm answer "$ANSWER_SUFFIX" ;;
      *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac
  done
fi

for arm in $ARMS; do
  case "$arm" in
    plain)  run_vsi_arm plain  '' ;;
    boxed)  run_vsi_arm boxed  "$BOXED_SUFFIX" ;;
    answer) run_vsi_arm answer "$ANSWER_SUFFIX" ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
  esac
done

log "ALL DONE $(date -Is)"
