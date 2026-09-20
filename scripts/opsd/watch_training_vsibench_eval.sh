#!/usr/bin/env bash
# Watch a checkpoint directory and run VSI-Bench eval on each new global_step_*.
#
# Typical layout: node-A trains, node-B runs this after rsyncing checkpoints.
#
#   bash scripts/opsd/watch_training_vsibench_eval.sh \
#     checkpoints/20260820_qwen35base_mvopsd_khalf_main \
#     --poll 120
#
# Already-evaluated steps are skipped if logs/vsi_train_eval/<exp>/<step>/summary.json
# exists.  Pass --force to re-run.
#
# A step that fails evaluation gets a FAILED marker and is not retried, so one
# bad checkpoint cannot starve the queue by being re-attempted every poll.
# Delete the marker, or pass --force, to try it again.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

POLL=120
FORCE=0
CKPT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll) POLL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) CKPT_ROOT="${1%/}"; shift ;;
  esac
done

[[ -n "$CKPT_ROOT" ]] || { echo "usage: $0 <checkpoints/<experiment>> [--poll SEC] [--force]" >&2; exit 1; }
EXPERIMENT="$(basename "$CKPT_ROOT")"

SETTLE="${SETTLE:-60}"

# A checkpoint appears on disk shard by shard.  Reading one mid-save merges a
# torn set of weights, which shows up as a mysteriously bad score rather than
# an error, so require the full shard set and a quiet directory before starting.
checkpoint_ready() {
  local actor="$1/actor"
  [[ -d "$actor" && -f "${actor}/fsdp_config.json" ]] || return 1

  local first world_size shards
  first="$(ls "${actor}"/model_world_size_*_rank_0.pt 2>/dev/null | head -1)"
  [[ -n "$first" ]] || return 1
  world_size="$(basename "$first" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
  shards="$(ls "${actor}"/model_world_size_"${world_size}"_rank_*.pt 2>/dev/null | wc -l)"
  (( shards == world_size )) || return 1

  local newest age
  newest="$(find "${actor}" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)"
  [[ -n "$newest" ]] || return 1
  age=$(( $(date +%s) - ${newest%.*} ))
  (( age >= SETTLE ))
}

echo "watching ${CKPT_ROOT} every ${POLL}s (experiment=${EXPERIMENT}, settle=${SETTLE}s)"

while true; do
  # Numeric order, so the score series is produced in training order rather
  # than the lexicographic order the glob would give (step_100 before step_45).
  mapfile -t step_dirs < <(
    find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null |
      sort -t_ -k3 -n
  )
  for step_dir in "${step_dirs[@]}"; do
    step="$(basename "$step_dir")"
    out="logs/vsi_train_eval/${EXPERIMENT}/${step}"
    if [[ "$FORCE" -eq 0 && ( -f "${out}/summary.json" || -f "${out}/FAILED" ) ]]; then
      continue
    fi
    checkpoint_ready "$step_dir" || continue
    echo "[$(date +%H:%M:%S)] evaluating ${step}"
    if bash scripts/opsd/run_training_vsibench_eval.sh "$step_dir"; then
      echo "[$(date +%H:%M:%S)] done ${step}"
    else
      mkdir -p "$out"
      date +%F\ %T > "${out}/FAILED"
      echo "[$(date +%H:%M:%S)] eval failed for ${step}; marked FAILED, not retrying" >&2
    fi
  done
  sleep "$POLL"
done
