#!/usr/bin/env bash
# Wait for a running GPU job to exit and release the cards, then exec the next
# command.
#
#   setsid nohup bash scripts/opsd/tools/chain_after.sh 712336 80000 \
#     -- env FOO=bar bash scripts/opsd/run_mvopsd.sh main > queue.log 2>&1
#
# Takes the PID of the job itself, not of the shell that launched it: that shell
# outlives the job, so waiting on it would start too late or never. Find it with
# pgrep -f before queueing.
#
# Two conditions, both required. The process exiting says the job finished; free
# memory says its engine workers actually died. A vLLM run that is killed or
# crashes can leave orphan workers holding ~85 GB per card at 0% util for hours
# (ISSUE-403), and the next job would then fail its own memory gate long after
# the PID went away.
set -uo pipefail

WAIT_PID="${1:?PID of the job to wait for}"
MIN_FREE_MIB="${2:?minimum free MiB required on every GPU}"
shift 2
[[ "${1:-}" == "--" ]] && shift
(($#)) || { echo "nothing to run after the wait" >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

if ! kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "PID ${WAIT_PID} is already gone; check it was the right one before trusting this" >&2
fi

echo "waiting for PID ${WAIT_PID} to exit $(date -Is)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "PID ${WAIT_PID} gone $(date -Is); waiting for >= ${MIN_FREE_MIB} MiB free on every GPU"

waited=0
until [[ -z "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
               awk -v m="$MIN_FREE_MIB" '$1 < m')" ]]; do
  if ((waited >= 900)); then
    echo "cards still held after ${waited}s; refusing to start" >&2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
    exit 1
  fi
  sleep 30
  waited=$((waited + 30))
done

echo "cards free after ${waited}s; starting $* $(date -Is)"
exec "$@"
