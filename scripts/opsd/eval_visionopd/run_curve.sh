#!/usr/bin/env bash
# Re-measure MV-OPSD v0's whole CV-Bench curve under the Vision-OPD protocol.
#
# The curve on record (3.62 / 3.90 / 6.77 / 11.67 / 26.95 / 36.53 for steps
# 50..300) is void: it was produced by a 16-token generation cap plus a parser
# that read the "B" in "Based" as an answer, and a pure string rule reproduces
# it with zero residual (ISSUE-003). Step 0 is included because v0 never ran a
# baseline, which is why "the model is recovering" looked plausible at all.
#
#   bash scripts/opsd/eval_visionopd/run_curve.sh
#
# Sequential on purpose: each leg wants all eight cards, and run_eval.sh resumes
# from its own answer file, so a re-run after an interruption is cheap.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

CKPT_ROOT="output/20260816_qwen35base_mvopsd_v0_main_global_step"
# Two ends of the old curve only, per the 2026-08-17 decision. Under the void
# protocol these were its lowest and highest points (3.62 and 36.53), so if the
# rise was real they are where it has to show. Steps 100-250 and the step-0
# baseline stay merged under output/ and can be added by uncommenting.
LEGS=(
  "mvopsd_v0_step50:${CKPT_ROOT}_50_hf"
  "mvopsd_v0_step300:${CKPT_ROOT}_300_hf"
  # "base_qwen35_4b:models/Qwen3.5-4B"
  # "mvopsd_v0_step100:${CKPT_ROOT}_100_hf"
  # "mvopsd_v0_step150:${CKPT_ROOT}_150_hf"
  # "mvopsd_v0_step200:${CKPT_ROOT}_200_hf"
  # "mvopsd_v0_step250:${CKPT_ROOT}_250_hf"
)

SUMMARY="logs/eval_visionopd/curve_summary.txt"
mkdir -p "$(dirname "${SUMMARY}")"

for leg in "${LEGS[@]}"; do
  tag="${leg%%:*}"
  model="${leg#*:}"
  echo
  echo "########## ${tag} ##########"
  if [[ ! -f "${model}/config.json" ]]; then
    echo "SKIP ${tag}: no config.json in ${model}"
    continue
  fi
  POLICY_MODEL="${model}" RUN_TAG="${tag}" PARALLEL_WORKERS="${PARALLEL_WORKERS:-64}" \
    bash scripts/opsd/eval_visionopd/run_eval.sh
  status=$?
  echo "===== ${tag} EXIT ${status} ====="
done

echo
echo "########## curve ##########"
{
  printf "%-22s %10s %10s %10s\n" "run" "combined" "answered" "capped"
  for leg in "${LEGS[@]}"; do
    tag="${leg%%:*}"
    score_file="logs/eval_visionopd/${tag}/cvbench_score.txt"
    [[ -f "${score_file}" ]] || continue
    combined=$(grep -oP 'Vision-OPD, \(2D\+3D\)/2\):\s+\K[0-9.]+' "${score_file}" | head -1)
    answered=$(grep -oP 'option letter parsed:\s+\S+ = \K[0-9.]+' "${score_file}" | head -1)
    capped=$(grep -oP 'hit generation cap:\s+\S+ = \K[0-9.]+' "${score_file}" | head -1)
    printf "%-22s %10s %10s %10s\n" "${tag}" "${combined:-?}" "${answered:-?}%" "${capped:-?}%"
  done
} | tee "${SUMMARY}"
