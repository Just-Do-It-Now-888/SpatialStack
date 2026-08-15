#!/usr/bin/env bash
# Train + eval only (extraction already complete).
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
source /home/c30084464/miniconda3/etc/profile.d/conda.sh
conda activate sr_opsd

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/c30084464/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/c30084464/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

mkdir -p ./output ./logs
log(){ echo "[traineval $(date +%H:%M:%S)] $*"; }

log "=== verify extracted data ==="
for d in structured3d scannet scannetpp rxr; do
  [ -d ./data/media/spar/$d ] && log "  spar/$d present" || { log "  spar/$d MISSING"; }
done

log "=== launching training ==="
MODEL_PATH=/home/c30084464/Documents/code/SpatialStack_OPSD/models/Qwen3.5-4B \
GEOMETRY_ENCODER_PATH=/home/c30084464/Documents/code/SpatialStack_OPSD/models/VGGT-1B \
USE_GEOMETRY_ENCODER=True \
FEATURE_FUSION_METHOD=deepstack_language_add \
GEOMETRY_ENCODER_LAYERS="11 17 23" \
GEOMETRY_FUSION_LAYERS="0 1 2" \
DATA_FLATTEN=False \
OUTPUT_DIR=./output/spatialstack_qwen35_train \
CACHE_DIR=/home/c30084464/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
bash scripts/train/train.sh
TRAIN_RC=$?
log "training finished rc=$TRAIN_RC"

CKPT=./output/spatialstack_qwen35_train
if [ ! -f "$CKPT/config.json" ]; then
    CKPT=$(ls -d ./output/spatialstack_qwen35_train/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
log "using checkpoint: $CKPT"
if [ ! -f "$CKPT/config.json" ]; then log "ERROR: no checkpoint. Aborting eval."; exit 1; fi

log "=== launching evaluation ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
MODEL_PATH="$CKPT" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=/home/c30084464/Documents/code/SpatialStack_OPSD/models/VGGT-1B,disable_thinking=true" \
OUTPUT_ROOT=logs/eval/spatialstack_qwen35_4b \
BENCHMARKS="vsibench,cvbench,blink_spatial,sparbench" \
bash scripts/evaluation/eval.sh
EVAL_RC=$?
log "evaluation finished rc=$EVAL_RC"
log "=== DONE. train_rc=$TRAIN_RC eval_rc=$EVAL_RC ==="
