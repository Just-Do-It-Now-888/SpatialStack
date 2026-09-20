#!/usr/bin/env bash
# Launch an MV-OPSD v0 training arm.
#
#   scripts/opsd/run_mvopsd.sh main            # teacher sees all N views
#   scripts/opsd/run_mvopsd.sh noprivilege     # teacher sees exactly the student's K views
#   SMOKE=1 scripts/opsd/run_mvopsd.sh main    # 2 steps on a tiny slice (P0-5)
#
# Both arms read parquet files generated from the same plan, so the only
# difference between them is the teacher's view set.

set -euo pipefail

# LESSON-041: subsequent training is conda sr_opsd (not vision-opd / ~/.local).
# shellcheck source=require_sr_opsd.sh
source "$(cd "$(dirname "$0")" && pwd)/require_sr_opsd.sh"
require_sr_opsd_training_env

ARM="${1:-main}"
shift || true
case "$ARM" in
  main|noprivilege) ;;
  *) echo "unknown arm: $ARM (expected main or noprivilege)" >&2; exit 1 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERL_ROOT="${PROJECT_ROOT}/verl_pkg"
CONFIG_DIR="${VERL_ROOT}/verl/trainer/config"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-20260816_qwen35base_mvopsd_v0_${ARM}}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/mvopsd/parquet/${ARM}_train.parquet}"
# Vision-OPD distills from the released checkpoint, not from a task-SFT one.
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-4B}"
# vllm (default) or hf. Geometry checkpoints cannot load in vLLM.
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
# The agent loop chunks each batch across its workers, so a worker count that
# does not divide the batch dies with "only support equal chunk". Keep it at the
# GPU count when running on a subset of a shared node.
AGENT_WORKERS="${AGENT_WORKERS:-8}"
ROLLOUT_N="${ROLLOUT_N:-1}"
# Which benchmark the in-training validation runs. TEST_FREQ=-1 restores the old
# train-then-evaluate flow for either.
#
# VSI_EVAL=1 (default from 2026-08-20) scores full VSI-Bench, 5,130 questions at
# 32 frames, in the loop. CV-Bench used to be the guardrail and was retired
# because it could not see the failure it existed to catch: the checkpoint with
# the best CV-Bench score of the whole run had collapsed by 19 points on VSI-Bench
# (LESSON-020). VSI_EVAL=0 restores CV-Bench.
VSI_EVAL="${VSI_EVAL:-1}"
# VAL_BENCHMARK is the selector; VSI_EVAL is kept as its legacy spelling so every
# archived launch command still reproduces byte for byte.
#
#   vsibench    5,130 questions at 32 frames (default)
#   cvbench     2,638 single-image rows
#   mindcube    1,050 tinybench rows, full view set -- the axis the
#               46.86 / 74.48 / 69.81 anchors live on
#   mindcube1v  the same 1,050 rows at one view -- the condition a K=1 MV-OPSD
#               run actually trains. Report it *alongside* mindcube, never
#               instead of it, and never as the same series.
if [[ -n "${VAL_BENCHMARK:-}" ]]; then
  case "$VAL_BENCHMARK" in
    vsibench|cvbench|mindcube|mindcube1v) ;;
    *) echo "unknown VAL_BENCHMARK: $VAL_BENCHMARK" >&2; exit 1 ;;
  esac
elif [[ "$VSI_EVAL" == "1" ]]; then
  VAL_BENCHMARK=vsibench
else
  VAL_BENCHMARK=cvbench
fi
TEST_FREQ="${TEST_FREQ:-15}"
# Default: save at the same steps we validate, so every eval point has a
# checkpoint for offline replay. Override with SAVE_FREQ=... when you need a
# coarser cadence; SMOKE=1 still forces SAVE_FREQ=-1 below.
if [[ ! -v SAVE_FREQ ]]; then
  if [[ "$TEST_FREQ" == "-1" ]]; then
    SAVE_FREQ=50
  else
    SAVE_FREQ="$TEST_FREQ"
  fi
fi
case "$VAL_BENCHMARK" in
  vsibench)
    # Default stays the plain parquet (LESSON-013). Boxed+last-line protocol:
    # VAL_FILE=$PWD/data/eval/vsibench_verl/vsibench_val_boxed_lastline.parquet
    VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/eval/vsibench_verl/vsibench_val.parquet}"
    ;;
  cvbench)
    VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/eval/cvbench_verl/cvbench_val.parquet}"
    ;;
  mindcube)
    VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/eval/mindcube_verl/mindcube_val.parquet}"
    ;;
  mindcube1v)
    VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/data/eval/mindcube_verl/mindcube_val_singleview.parquet}"
    ;;
esac
CVBENCH_PARSER="${CVBENCH_PARSER:-boxed_lastline}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE
# Default is the yaml (currently frozen). Set these to match Vision-OPD:
#   TEACHER_REGULARIZATION=ema TEACHER_UPDATE_RATE=0.05
# Unset keeps mvopsd.yaml so SPAR frozen runs stay bit-reproducible.
TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-}"

MAX_PROMPT_LENGTH=4096
# Training rollouts stay at 1024; validation uses a separate budget so in-loop
# VSI-Bench matches the offline protocol (4096) without blowing up PPO memory.
MAX_RESPONSE_LENGTH=1024
# Whether the caller chose this, so a per-benchmark default below can apply
# without overriding an explicit request.
VAL_RESPONSE_EXPLICIT=0
[[ -v VAL_MAX_RESPONSE_LENGTH ]] && VAL_RESPONSE_EXPLICIT=1
VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# A 32-frame VSI-Bench question tokenises to ~9.85k, against a training prompt
# peak of 3,340. Two separate limits have to make room for it, and only one of
# them fails loudly:
#
#   rollout.max_model_len       engine capacity, shared by training and
#                               validation. Too small and vLLM rejects the
#                               request outright.
#   rollout.val_kwargs.prompt_length   validation-only cap. Too small and the
#                               agent loop *silently truncates* the prompt
#                               (agent_loop.py:338-346), dropping most of the
#                               frames and scoring the model on a clip it never
#                               saw. Nothing in the logs would say so.
#
# data.max_prompt_length stays 4096 so nothing about the training path moves:
# rollout.prompt_length interpolates from it and still governs training rollouts.
VSI_VAL_PROMPT_LENGTH="${VSI_VAL_PROMPT_LENGTH:-11264}"

# MindCube needs its own cap, and it must come from a measurement rather than
# from "four images times 300 tokens plus text". tinybench ships nine image
# sizes, from 240x135 up to 1296x968, and the pixel floor min_pixels=200704
# *upscales* the small ones, so the widest full-view prompt in the 1,050 rows
# measures 3,859 tokens (median 1,045, p95 3,853) rather than the ~1,600 the
# nominal budget suggests. 4,608 clears that with room for the chat template;
# 3,072 would have silently dropped frames from the longest 5% of rows, which is
# the one validation failure that leaves no trace in the logs
# (agent_loop.py:338-346 truncates without complaining).
# Single view measures 1,386 at worst and rides the same cap.
MINDCUBE_VAL_PROMPT_LENGTH="${MINDCUBE_VAL_PROMPT_LENGTH:-4608}"

# Declared here so `set -u` is safe on the TEST_FREQ=-1 path, where no
# validation cap is ever computed.
VAL_PROMPT_LENGTH=""
EXTRA_ARGS=()
if [[ "$TEST_FREQ" != "-1" ]]; then
  case "$VAL_BENCHMARK" in
    vsibench) VAL_PROMPT_LENGTH="$VSI_VAL_PROMPT_LENGTH" ;;
    mindcube|mindcube1v)
      VAL_PROMPT_LENGTH="$MINDCUBE_VAL_PROMPT_LENGTH"
      # The anchors were produced with max_new_tokens=2048
      # (scripts/mindcube/eval_mindcube_qwen35.py). Keep it: the CoT arm's ~319
      # median needs far less, but a different cap is a different protocol, and
      # step 0 reproducing 74.48 / 69.81 is a hard gate. Watch
      # resp_tokens/p95 against it -- a run pressed against the cap is a run
      # whose accuracy is partly a truncation artefact (LESSON-020).
      [[ "$VAL_RESPONSE_EXPLICIT" == "0" ]] && VAL_MAX_RESPONSE_LENGTH=2048
      ;;
    *) VAL_PROMPT_LENGTH="" ;;
  esac
  if [[ -n "$VAL_PROMPT_LENGTH" ]]; then
    MAX_MODEL_LEN=$((VAL_PROMPT_LENGTH + VAL_MAX_RESPONSE_LENGTH))
    EXTRA_ARGS+=(
      "actor_rollout_ref.rollout.val_kwargs.prompt_length=${VAL_PROMPT_LENGTH}"
      "actor_rollout_ref.rollout.val_kwargs.response_length=${VAL_MAX_RESPONSE_LENGTH}"
    )
  fi
fi

# In-loop validation decodes greedily by default (mvopsd.yaml val_kwargs), which
# is what makes the curve comparable to the offline archives and to the 53.62
# base anchor. VAL_DO_SAMPLE=1 switches it to sampling, following upstream OPSD,
# whose eval script refuses to recommend greedy because it produces the endless
# repetitions we see as length explosion (LESSON-020). Doing so starts a new
# series: sampled points must not be plotted against greedy ones.
if [[ "${VAL_DO_SAMPLE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.8}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K:--1}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-1}"
  )
  echo "val decoding=SAMPLING t=${VAL_TEMPERATURE:-1.0} top_p=${VAL_TOP_P:-0.8} n=${VAL_N:-1}" \
    "-- not comparable to the greedy curve"
else
  echo "val decoding=greedy (default; comparable to offline archives)"
fi

# The smoke suffix has to land before the output paths are derived, or a throwaway
# run overwrites the real experiment's checkpoints and rollouts (LESSON-004).
if [[ "${SMOKE:-0}" == "1" ]]; then
  TOTAL_STEPS=2
  TRAIN_BATCH_SIZE="${SMOKE_BATCH:-8}"
  ROLLOUT_N="${ROLLOUT_N:-1}"
  SAVE_FREQ=-1
  TEST_FREQ=1
  # A smoke run has to exercise the validation path too, and it has to exercise
  # *this run's* validation path: swapping in CV-Bench would leave the selected
  # benchmark's parquet, scorer and metrics module untested, which is the half of
  # the pipeline a smoke run exists to check. SMOKE_VAL_FILE overrides.
  VAL_FILE="${SMOKE_VAL_FILE:-$VAL_FILE}"
  # resume_mode=auto would pick up a leftover smoke checkpoint and continue from
  # its step count instead of validating the pipeline from scratch.
  EXTRA_ARGS+=(trainer.resume_mode=disable)
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
  echo "SMOKE mode: ${TOTAL_STEPS} steps, batch ${TRAIN_BATCH_SIZE}, n=${ROLLOUT_N}"
fi

if [[ -n "$TEACHER_REGULARIZATION" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REGULARIZATION}")
fi
if [[ -n "$TEACHER_UPDATE_RATE" ]]; then
  EXTRA_ARGS+=("actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE}")
fi
if [[ -n "${VAL_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=("data.val_batch_size=${VAL_BATCH_SIZE}")
fi
if [[ "$ROLLOUT_NAME" == "vllm" ]]; then
  EXTRA_ARGS+=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False"
  )
fi

CKPT_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/train/${EXPERIMENT_NAME}"
ROLLOUT_DIR="${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}"
VAL_DIR="${PROJECT_ROOT}/logs/val/${EXPERIMENT_NAME}"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$ROLLOUT_DIR" "$VAL_DIR"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "train file not found: $TRAIN_FILE (run the stage 1-3 scripts first)" >&2
  exit 1
fi

if [[ "$TEST_FREQ" != "-1" && ! -f "$VAL_FILE" ]]; then
  echo "validation file not found: $VAL_FILE" >&2
  case "$VAL_BENCHMARK" in
    mindcube)   echo "build it with: python3 scripts/opsd/build_mindcube_val_parquet.py" >&2 ;;
    mindcube1v) echo "build it with: python3 scripts/opsd/build_mindcube_val_parquet.py --views 1 --tag singleview" >&2 ;;
    vsibench)   echo "build it with: python3 scripts/opsd/build_vsibench_val_parquet.py" >&2 ;;
    *)          echo "build it with: python3 scripts/opsd/build_cvbench_val_parquet.py" >&2 ;;
  esac
  echo "or disable in-training evaluation with TEST_FREQ=-1" >&2
  exit 1
fi

# wandb.init happens after Ray has brought up eight vLLM engines, so an
# unauthenticated run would burn several minutes before failing. Check up front.
if [[ "$WANDB_MODE" == "online" ]]; then
  if ! python3 -c "import netrc,os,sys; sys.exit(0 if os.environ.get('WANDB_API_KEY') or 'api.wandb.ai' in netrc.netrc().hosts else 1)" 2>/dev/null; then
    echo "wandb is set to online but no credentials were found." >&2
    echo "run 'wandb login', or export WANDB_API_KEY, or set WANDB_MODE=offline" >&2
    exit 1
  fi
fi

# verl is also installed from ~/Documents/code/Vision-OPD-main via a .pth entry.
# Put this repo's copy first so the run matches the tree we version here.
# src/ is required for SpatialStack geometry (qwen_vl.model).
export PYTHONPATH="${VERL_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=1

# From 2026-09-10, training is conda sr_opsd (LESSON-041). Older arms used
# system python3.12 + ~/.local; an unrelated conda env still exports
# PYTHONNOUSERSITE=1 and would hide ~/.local if someone bypassed the sr_opsd
# check. find_spec rather than a real import: this must not cost the ten
# seconds importing vllm takes. HF geometry rollout does not import vLLM.
if [[ "$ROLLOUT_NAME" == "vllm" ]]; then
  if ! python3 -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('vllm') else 1)"; then
    echo "python3 ($(command -v python3)) cannot see vllm." >&2
    if [[ -n "${PYTHONNOUSERSITE:-}" ]]; then
      echo "PYTHONNOUSERSITE=${PYTHONNOUSERSITE} is set, so ~/.local is being ignored." >&2
      echo "an active conda env sets it; relaunch with 'env -u PYTHONNOUSERSITE ...'" >&2
    fi
    exit 1
  fi
fi
echo "rollout=${ROLLOUT_NAME}"
export PYTHONUNBUFFERED=1
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

echo "arm=${ARM} experiment=${EXPERIMENT_NAME} rollout=${ROLLOUT_NAME}"
echo "train_file=${TRAIN_FILE}"
echo "model=${MODEL_PATH}"
echo "steps=${TOTAL_STEPS} batch=${TRAIN_BATCH_SIZE} rollout_n=${ROLLOUT_N} save_freq=${SAVE_FREQ}"
if [[ "$TEST_FREQ" == "-1" ]]; then
  echo "eval=disabled in verl loop (TEST_FREQ=-1)"
  echo "  score saved checkpoints offline with scripts/opsd/run_training_vsibench_eval.sh"
elif [[ "$VAL_BENCHMARK" == "vsibench" ]]; then
  echo "eval=vsibench every ${TEST_FREQ} steps from ${VAL_FILE}"
  echo "  val prompt cap ${VAL_PROMPT_LENGTH}, val response cap ${VAL_MAX_RESPONSE_LENGTH}, engine max_model_len ${MAX_MODEL_LEN}"
  echo "  watch val-core/vsibench/{overall/acc, answered/frac, resp_tokens/mean, resp_tokens/clip_ratio, resp_chars/median}"
elif [[ "$VAL_BENCHMARK" == mindcube* ]]; then
  echo "eval=${VAL_BENCHMARK} every ${TEST_FREQ} steps from ${VAL_FILE}"
  echo "  val prompt cap ${VAL_PROMPT_LENGTH} (widest row measures 3859), val response cap ${VAL_MAX_RESPONSE_LENGTH}, engine max_model_len ${MAX_MODEL_LEN}"
  echo "  watch val-core/${VAL_BENCHMARK}/{overall/acc, answered/frac, resp_tokens/{mean,median,p95}}"
  echo "  and val-aux/${VAL_BENCHMARK}/family/{among,around,rotation}/acc plus /nviews/{2,3,4}/acc"
  if [[ "$VAL_BENCHMARK" == "mindcube" ]]; then
    # In-loop validation generates with vLLM, so the gate is the vLLM-engine value
    # measured by probe_mindcube_recoverability.py, not the transformers anchor the
    # SFT round archived (74.48 / 69.81). The two differ by -0.19 / +0.76 pp.
    echo "  step 0 must land on this arm's vLLM anchor (74.29 answer-only / 70.57 CoT);"
    echo "  a miss means the environment differs from the anchor's, not that the method moved"
  else
    echo "  single-view series: NOT comparable to the full-view anchors; its own step-0"
    echo "  reference is the probe's single-view value (59.24 answer-only / 55.24 CoT)"
  fi
else
  echo "eval=cvbench every ${TEST_FREQ} steps from ${VAL_FILE} (parser=${CVBENCH_PARSER})"
fi
if [[ "$TEST_FREQ" != "-1" && "$SAVE_FREQ" == "$TEST_FREQ" ]]; then
  echo "checkpoint=aligned with eval (every ${SAVE_FREQ} steps)"
fi
echo "wandb=${WANDB_MODE} project=MV-OPSD run=${EXPERIMENT_NAME}"
if [[ -n "$TEACHER_REGULARIZATION" ]]; then
  echo "teacher=${TEACHER_REGULARIZATION} update_rate=${TEACHER_UPDATE_RATE:-yaml-default}"
  if [[ "$TEACHER_REGULARIZATION" == "ema" ]]; then
    echo "  watch self_distillation/teacher_probe: it MUST drift across steps (frozen is constant)"
  elif [[ "$TEACHER_REGULARIZATION" == "frozen" ]]; then
    echo "  watch self_distillation/teacher_probe: it MUST be identical across steps"
  fi
fi

# --- validation judge ------------------------------------------------------
# Optional second grading tier for in-training validation. Default is off:
# scores come from the rule tier only (vsibench_boxed_primary + spatialstack
# parser), matching the offline protocol we use without judge.
#
# JUDGE=1 starts gpt-oss-120b in vLLM sleep mode and enables
# trainer.validation_judge for rows the rules cannot read. See
# scripts/opsd/mvopsd_judge.py for the handshake.
JUDGE="${JUDGE:-0}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${PROJECT_ROOT}/models/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
# The judge only ever reads a CV-Bench question plus a response capped at
# MAX_RESPONSE_LENGTH, so this is far more room than it can use.
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-16384}"
JUDGE_PID=""

if [[ "$JUDGE" == "1" && ! -f "${JUDGE_MODEL_PATH}/config.json" ]]; then
  echo "validation judge requested but no model at ${JUDGE_MODEL_PATH}" >&2
  echo "download it, or run with JUDGE=0 (default) to score with rules only" >&2
  exit 1
fi
if [[ "$JUDGE" == "1" && "$TEST_FREQ" == "-1" ]]; then
  echo "TEST_FREQ=-1 means no in-training validation; skipping the judge"
  JUDGE=0
fi

stop_judge() {
  [[ -n "$JUDGE_PID" ]] || return 0
  # setsid put the server in its own group, so signal the whole tree: killing
  # the parent alone leaves the vLLM engine cores holding the cards.
  kill -- "-${JUDGE_PID}" 2>/dev/null || kill "$JUDGE_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$JUDGE_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 -- "-${JUDGE_PID}" 2>/dev/null || true
  JUDGE_PID=""
}
trap stop_judge EXIT INT TERM

if [[ "$JUDGE" == "1" ]]; then
  JUDGE_LOG="${LOG_DIR}/judge_$(date +%Y%m%d_%H%M%S).log"
  echo "judge=gpt-oss-120b on port ${JUDGE_PORT} (log ${JUDGE_LOG})"
  # /sleep and /wake_up are dev endpoints, and MXFP4 needs Marlin because this
  # image's triton_kernels build crashes gpt-oss during the profile run
  # (ISSUE-005). Starting before training also means vLLM profiles free cards.
  VLLM_SERVER_DEV_MODE=1 VLLM_MXFP4_USE_MARLIN=1 \
    setsid python3 -m vllm.entrypoints.openai.api_server \
      --model "$JUDGE_MODEL_PATH" \
      --served-model-name judge \
      --port "$JUDGE_PORT" \
      --tensor-parallel-size 8 \
      --max-model-len "$JUDGE_MAX_LEN" \
      --gpu-memory-utilization 0.25 \
      --enable-sleep-mode \
      --trust-remote-code > "$JUDGE_LOG" 2>&1 &
  JUDGE_PID=$!

  waited=0
  until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
      echo "--- last 40 lines of the judge log ---" >&2
      tail -n 40 "$JUDGE_LOG" >&2
      echo "judge server died during startup; rerun with JUDGE=0 to train without it" >&2
      exit 1
    fi
    if (( waited >= 1800 )); then
      echo "judge server not healthy after ${waited}s" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  # Hand the cards back before training profiles them, or vLLM sizes the
  # policy's KV cache against memory the judge is only borrowing. Level 1, not
  # 2: level 2 rebuilds the MXFP4 weights on wake and gets them wrong, coming
  # back healthy and answering "!!!!!!!!" (see mvopsd_judge.py).
  curl -sf -X POST "http://127.0.0.1:${JUDGE_PORT}/sleep?level=1" >/dev/null \
    || { echo "judge will not sleep; it would keep the GPUs training needs" >&2; exit 1; }
  echo "judge healthy after ${waited}s and asleep; it wakes only for unresolved rows"

  EXTRA_ARGS+=(
    trainer.validation_judge.enable=True
    "trainer.validation_judge.api_base=http://127.0.0.1:${JUDGE_PORT}/v1"
  )
else
  echo "judge=disabled; ${VAL_BENCHMARK} validation scored by rules only"
fi

# DRY_RUN=1 stops here, after every derived path and cap has been printed but
# before any GPU is touched. Added when the MindCube benchmark selector went in:
# the validation prompt cap is the one setting whose mistake is silent, so being
# able to read it back without a launch is worth the two lines.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: stopping before launch"
  exit 0
fi

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name mvopsd \
    data.train_files="[\"${TRAIN_FILE}\"]" \
    data.val_files="[\"${VAL_FILE}\"]" \
    custom_reward_function.reward_kwargs.cvbench_parser="${CVBENCH_PARSER}" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    critic.model.path="$MODEL_PATH" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.group_name=mvopsd_v0 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.validation_data_dir="$VAL_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    "$@" 2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
