#!/usr/bin/env bash
# Merge remaining SpatialStack MV-OPSD FSDP actor weights and run full VSI
# boxed last-line (HF/lmms-eval, 5130) for every complete checkpoint:
#   k2 {70,77,84} and k1 {70,77,84}.
# k2 step 84 is assumed already running; this script waits for WAIT_PID then
# evaluates the rest. Skip smoke: the boxed path is already proven.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

WAIT_PID="${WAIT_PID:-2617825}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
BASE_MODEL="${BASE_MODEL:-$PROJECT_ROOT/output/spatialstack_qwen35_train}"
PYTHON_MERGE="${PYTHON_MERGE:-$HOME/miniconda3/envs/vision-opd/bin/python}"
LOG_ROOT="${EVAL_QUEUE_LOG:-$PROJECT_ROOT/logs/eval/20260909_spatialstack_mvopsd_spar3_student_k}"
mkdir -p "$LOG_ROOT"

merge_one() {
  local step_dir="$1"
  local target="$2"
  if [[ -f "$target/config.json" && -f "$target/model.safetensors" ]]; then
    echo "[$(date +%F\ %T)] skip merge, exists: $target"
    return 0
  fi
  echo "[$(date +%F\ %T)] merge $step_dir -> $target"
  BASE_MODEL="$BASE_MODEL" PYTHON_MERGE="$PYTHON_MERGE" \
    bash "$PROJECT_ROOT/scripts/opsd/merge_mvopsd_ckpt.sh" "$step_dir" "$target"
}

eval_boxed() {
  local ckpt="$1"
  local eval_root="$2"
  local port="${3:-29542}"
  if [[ -f "$eval_root/vsibench_boxed/run.log" ]] && \
     grep -q 'END arm=boxed rc=0' "$eval_root/chain.log" 2>/dev/null; then
    echo "[$(date +%F\ %T)] skip eval, already done: $eval_root"
    return 0
  fi
  echo "[$(date +%F\ %T)] eval boxed $ckpt -> $eval_root"
  mkdir -p "$eval_root"
  # run_dual.sh boxed does not wait for empty GPUs (needed: node-A shares ~26GB/card).
  CKPT="$ckpt" EVAL_ROOT="$eval_root" \
    bash "$PROJECT_ROOT/scripts/opsd/eval_spatialstack_geometry_vsi.sh" boxed \
    >"$eval_root/chain.log" 2>&1
}

echo "[$(date +%F\ %T)] start queue WAIT_PID=$WAIT_PID"

merge_one \
  "$PROJECT_ROOT/checkpoints/20260909_spatialstack_mvopsd_spar3_k2/global_step_77" \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k2_global_step_77_hf"
merge_one \
  "$PROJECT_ROOT/checkpoints/20260909_spatialstack_mvopsd_spar3_k2/global_step_70" \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k2_global_step_70_hf"

echo "[$(date +%F\ %T)] waiting for PID ${WAIT_PID}"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[$(date +%F\ %T)] PID ${WAIT_PID} gone; wait free>=${MIN_FREE_MIB} MiB/GPU"

waited=0
until [[ -z "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
               awk -v m="$MIN_FREE_MIB" '$1 < m')" ]]; do
  if ((waited >= 900)); then
    echo "cards still held after ${waited}s" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv >&2
    exit 1
  fi
  sleep 20
  waited=$((waited + 20))
done
echo "[$(date +%F\ %T)] cards free enough after ${waited}s"

# Remaining k2 then k1 (HF dirs must already be on this node).
eval_boxed \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k2_global_step_77_hf" \
  "$PROJECT_ROOT/logs/eval/20260909_spatialstack_mvopsd_spar3_k2_step77" \
  29551
eval_boxed \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k2_global_step_70_hf" \
  "$PROJECT_ROOT/logs/eval/20260909_spatialstack_mvopsd_spar3_k2_step70" \
  29552

# Pull k1 HF from node-B if this node does not have them yet.
for step in 84 77 70; do
  tgt="$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k1_global_step_${step}_hf"
  if [[ -f "$tgt/config.json" && -f "$tgt/model.safetensors" ]]; then
    echo "[$(date +%F\ %T)] k1 HF already local: $tgt"
    continue
  fi
  echo "[$(date +%F\ %T)] waiting for node-B merge of k1 step ${step}"
  until ssh -o ConnectTimeout=10 opsd2 "test -f /home/c30084464/Documents/code/SpatialStack_OPSD/output/20260909_spatialstack_mvopsd_spar3_k1_global_step_${step}_hf/model.safetensors"; do
    sleep 30
  done
  mkdir -p "$tgt"
  rsync -a opsd2:/home/c30084464/Documents/code/SpatialStack_OPSD/output/20260909_spatialstack_mvopsd_spar3_k1_global_step_${step}_hf/ "$tgt/"
done

eval_boxed \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k1_global_step_84_hf" \
  "$PROJECT_ROOT/logs/eval/20260909_spatialstack_mvopsd_spar3_k1_step84" \
  29553
eval_boxed \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k1_global_step_77_hf" \
  "$PROJECT_ROOT/logs/eval/20260909_spatialstack_mvopsd_spar3_k1_step77" \
  29554
eval_boxed \
  "$PROJECT_ROOT/output/20260909_spatialstack_mvopsd_spar3_k1_global_step_70_hf" \
  "$PROJECT_ROOT/logs/eval/20260909_spatialstack_mvopsd_spar3_k1_step70" \
  29555

echo "[$(date +%F\ %T)] all queued boxed evals finished"
