#!/usr/bin/env bash
# Frozen-init single-view recoverability probe for the MindCube MV-OPSD round.
#
# Both arms run on separate GPUs of the SAME node on purpose: the two numbers are
# only comparable if they come from one engine build and one set of libraries.
# Measuring one arm per machine would fold the node-A/node-B environment drift
# (ISSUE-701 / ISSUE-702) into the very gap being measured.
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_DIR="${OUT_DIR:-output/probes}"
ARM_A_GPU="${ARM_A_GPU:-0}"
ARM_B_GPU="${ARM_B_GPU:-1}"
ARM_A_CKPT="${ARM_A_CKPT:-output/20260906_qwen35_mindcube_sft_answeronly/checkpoint-314}"
ARM_B_CKPT="${ARM_B_CKPT:-output/20260906_qwen35_mindcube_sft_cot/checkpoint-314}"
LIMIT="${LIMIT:-0}"

mkdir -p "$OUT_DIR"

launch() {
  local arm="$1" gpu="$2" ckpt="$3"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    env -u PYTHONNOUSERSITE /usr/bin/python3 scripts/opsd/probe_mindcube_recoverability.py \
      --model "$ckpt" \
      --limit "$LIMIT" \
      --out "$OUT_DIR/mindcube_recoverability_arm$arm.json" \
      > "$OUT_DIR/arm$arm.log" 2>&1 &
  echo "arm $arm -> GPU $gpu, pid $!, ckpt $ckpt"
}

launch A "$ARM_A_GPU" "$ARM_A_CKPT"
launch B "$ARM_B_GPU" "$ARM_B_CKPT"

FAILED=0
wait -n || FAILED=1
wait || FAILED=1
if [ "$FAILED" -ne 0 ]; then
  echo "at least one probe failed; see $OUT_DIR/armA.log and $OUT_DIR/armB.log" >&2
  exit 1
fi
echo "both probes finished; reports in $(pwd)/$OUT_DIR"
