#!/usr/bin/env bash
# Start the judging pass as soon as the generation driver reports ALL DONE.
#
#   setsid nohup bash scripts/opsd/tools/chain_teacher_judging.sh \
#     > logs/eval/teacher_reliability/judging.log 2>&1
#
# Watches the driver's status file rather than a PID: the shell that launched
# the driver outlives it, so waiting on that PID would start the judge either
# too early or never. gpt-oss-120b wants the same eight cards the generation is
# using, which is why this waits rather than running alongside.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
STATUS=logs/eval/teacher_reliability/status.txt

while true; do
  if grep -q "ALL DONE" "$STATUS" 2>/dev/null; then
    break
  fi
  if ! pgrep -f "bash scripts/opsd/run_teacher_reliability.sh" >/dev/null; then
    echo "generation driver is gone and never reported ALL DONE; not judging" >&2
    tail -n 5 "$STATUS" >&2
    exit 1
  fi
  sleep 60
done

echo "generation complete, starting the judging pass $(date -Is)"
exec bash scripts/opsd/run_teacher_judging.sh
