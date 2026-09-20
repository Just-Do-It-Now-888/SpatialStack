#!/usr/bin/env bash
# Chain several MV-OPSD arms on one node, one after another.
#
#   LR=5.0e-6 TAG=lr5e6 scripts/opsd/run_lr_batch_sweep.sh
#   LR=1.0e-5 TAG=lr1e5 BATCHES="64 128" scripts/opsd/run_lr_batch_sweep.sh
#
# Every arm shares one LR and differs only in data.train_batch_size, which
# run_mvopsd.sh also applies to ppo_mini_batch_size, so each step stays a single
# on-policy update. LR is not an env var in run_mvopsd.sh, so it is passed as a
# hydra override; the launcher never sets actor.optim.lr itself, so there is no
# duplicate key.
#
# Deliberately no `set -e`: a mid-training crash in one arm must not silently
# cancel the arms behind it. Instead each arm's exit code is recorded, and the
# chain aborts only when an arm dies fast enough to look like an environment
# fault rather than a training failure (LESSON-007).

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

LR="${LR:?set LR, e.g. LR=5.0e-6}"
TAG="${TAG:?set TAG, e.g. TAG=lr5e6}"
BATCHES="${BATCHES:-64 128}"
STEPS="${STEPS:-300}"
FREQ="${FREQ:-15}"
PREFIX="${PREFIX:-20260818_qwen35base_mvopsd}"
# An arm that dies sooner than this never reached steady-state training, so the
# fault is almost certainly shared with every arm behind it.
MIN_HEALTHY_SECONDS="${MIN_HEALTHY_SECONDS:-1800}"
# 20 checkpoints x ~55 GB, plus room for rollout dumps and val dumps.
MIN_FREE_GB="${MIN_FREE_GB:-1300}"

SWEEP_LOG_DIR="${PROJECT_ROOT}/logs/train/_sweep"
mkdir -p "$SWEEP_LOG_DIR"
STATUS_FILE="${SWEEP_LOG_DIR}/${TAG}_status.txt"
: > "$STATUS_FILE"

log() { echo "[$(date '+%F %T %Z')] $*"; }

status() { echo "$*" >> "$STATUS_FILE"; }

free_gb() { df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9'; }

# run_mvopsd.sh parks gpt-oss-120b on port 8100 and its EXIT trap tears it down.
# The next arm starts its own, so wait for both the port and the cards before
# handing over, or the successor dies during judge startup or vLLM profiling.
wait_for_clean_node() {
  local waited=0
  while (( waited < 600 )); do
    local port_busy=0 mem_busy=0
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ':8100 '; then
      port_busy=1
    fi
    local maxmem
    maxmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    maxmem="${maxmem:-0}"
    (( maxmem > 2000 )) && mem_busy=1
    if (( port_busy == 0 && mem_busy == 0 )); then
      log "node is clean (port 8100 free, max GPU mem ${maxmem} MiB)"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  log "WARNING: node still busy after ${waited}s (port_busy/mem may block the next arm)"
  return 1
}

log "sweep tag=${TAG} lr=${LR} batches='${BATCHES}' steps=${STEPS} test_freq=${FREQ} save_freq=${FREQ}"
log "host=$(hostname) project=${PROJECT_ROOT} free=$(free_gb)GB"
status "sweep_start $(date '+%F %T') host=$(hostname) lr=${LR} batches='${BATCHES}'"

ARM_INDEX=0
for BATCH in $BATCHES; do
  ARM_INDEX=$((ARM_INDEX + 1))
  NAME="${PREFIX}_${TAG}_b${BATCH}_main"

  log "=============================================================="
  log "arm ${ARM_INDEX}: ${NAME}  (batch=${BATCH}, lr=${LR}, steps=${STEPS})"

  avail=$(free_gb)
  if (( avail < MIN_FREE_GB )); then
    log "ABORT arm ${ARM_INDEX}: only ${avail}GB free, need ${MIN_FREE_GB}GB for ${STEPS}/${FREQ} checkpoints"
    status "arm${ARM_INDEX} ${NAME} SKIPPED disk_only_${avail}GB"
    continue
  fi

  # ISSUE-101: resume_mode=auto silently continues from whatever is in the
  # checkpoint dir, so a non-empty dir means this is not the cold start we think.
  if [[ -n "$(ls -A "checkpoints/${NAME}" 2>/dev/null)" ]]; then
    log "ABORT arm ${ARM_INDEX}: checkpoints/${NAME} is not empty; refusing to resume into a fresh experiment"
    status "arm${ARM_INDEX} ${NAME} SKIPPED dirty_checkpoint_dir"
    continue
  fi

  wait_for_clean_node || true

  status "arm${ARM_INDEX} ${NAME} START $(date '+%F %T') batch=${BATCH} lr=${LR}"
  started=$(date +%s)

  EXPERIMENT_NAME="$NAME" \
  TRAIN_BATCH_SIZE="$BATCH" \
  TOTAL_STEPS="$STEPS" \
  TEST_FREQ="$FREQ" \
  SAVE_FREQ="$FREQ" \
    bash scripts/opsd/run_mvopsd.sh main \
      actor_rollout_ref.actor.optim.lr="$LR"
  rc=$?

  elapsed=$(( $(date +%s) - started ))
  log "arm ${ARM_INDEX} (${NAME}) exited rc=${rc} after ${elapsed}s ($((elapsed / 3600))h$(((elapsed % 3600) / 60))m)"
  status "arm${ARM_INDEX} ${NAME} END $(date '+%F %T') rc=${rc} elapsed_s=${elapsed}"

  if (( rc != 0 && elapsed < MIN_HEALTHY_SECONDS )); then
    log "ABORT chain: arm ${ARM_INDEX} failed after only ${elapsed}s, which looks like an environment fault."
    log "Fix it before the remaining arms burn the same way; see the arm's log under logs/train/${NAME}/."
    status "chain_aborted after arm${ARM_INDEX} fast_failure"
    exit "$rc"
  fi
done

log "sweep ${TAG} complete"
status "sweep_end $(date '+%F %T')"
