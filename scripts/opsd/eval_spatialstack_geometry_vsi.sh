#!/usr/bin/env bash
# Full VSI-Bench boxed last-line eval for a SpatialStack geometry HF checkpoint.
# This is HuggingFace/lmms-eval, not vLLM: the geometry class cannot load in vLLM.
#
#   CKPT=output/20260909_spatialstack_mvopsd_spar3_k2_global_step_84_hf \
#     bash scripts/opsd/eval_spatialstack_geometry_vsi.sh boxed
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

CKPT="${CKPT:?set CKPT to a merged HF directory with config.json}"
VGGT="${VGGT:-$PROJECT_ROOT/models/VGGT-1B}"
EVAL_ROOT="${EVAL_ROOT:?set EVAL_ROOT}"
STAGE="${1:-boxed}"

exec env CKPT="$CKPT" VGGT="$VGGT" EVAL_ROOT="$EVAL_ROOT" \
  bash "$PROJECT_ROOT/logs/eval/20260909_spatialstack_qwen35_geo/run_dual.sh" "$STAGE"
