#!/usr/bin/env bash
# Full judge-extract scoring for llava_hound step-100 @4096 VSI-Bench eval.
#
# Kept so the command recorded in experiments/ACTIVE.md still reproduces the
# archived run. The logic now lives in run_judge_extract.sh, which takes any
# samples.jsonl -- there is one implementation, not two (LESSON-023).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SAMPLES="logs/vsi_train_eval/20260821_mvopsd_single_llava_hound_main/global_step_100/samples.jsonl"
OUT_DIR="logs/vsi_train_eval/20260821_mvopsd_single_llava_hound_main/global_step_100/judge_extract"
STATUS="logs/eval/judge_extract_step100/status.txt"
mkdir -p "$(dirname "$STATUS")"

exec > >(tee -a "${STATUS}") 2>&1
exec bash scripts/opsd/run_judge_extract.sh "${SAMPLES}" "${OUT_DIR}"
