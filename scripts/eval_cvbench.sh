#!/usr/bin/env bash
# Eval CV-Bench only, on the trained SpatialStack checkpoint.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
source /home/c30084464/miniconda3/etc/profile.d/conda.sh
conda activate sr_opsd

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/c30084464/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/c30084464/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false
mkdir -p ./output ./logs
log(){ echo "[cvbench $(date +%H:%M:%S)] $*"; }

CKPT=./output/spatialstack_qwen35_train
if [ ! -f "$CKPT/config.json" ]; then
    CKPT=$(ls -d ./output/spatialstack_qwen35_train/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
log "checkpoint: $CKPT"
[ -f "$CKPT/config.json" ] || { log "ERROR: no checkpoint."; exit 1; }

log "=== launching CV-Bench evaluation ==="
MODEL_PATH="$CKPT" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=/home/c30084464/Documents/code/SpatialStack_OPSD/models/VGGT-1B,disable_thinking=true" \
OUTPUT_ROOT=logs/eval/spatialstack_qwen35_4b \
BENCHMARKS="cvbench" \
bash scripts/evaluation/eval.sh
RC=$?
log "=== DONE cvbench_rc=$RC ==="
