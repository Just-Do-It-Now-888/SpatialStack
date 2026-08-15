#!/usr/bin/env bash
# Qwen3.5-4B training with the VGGT geometry encoder enabled.
# Qwen3.5 non-thinking formatting is applied automatically by data_qwen.py.
set -uo pipefail

cd /home/c30084464/Documents/code/SpatialStack_OPSD
source /home/c30084464/miniconda3/etc/profile.d/conda.sh
conda activate sr_opsd

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/c30084464/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/c30084464/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

OUT=./output/spatialstack_qwen35_vggt_aligned
mkdir -p "$OUT" ./logs
log(){ echo "[vggt-aligned $(date +%H:%M:%S)] $*"; }

log "=== environment check ==="
python - <<'PY'
import fla
import torch
import triton

print(
    f"torch {torch.__version__} | triton {triton.__version__} | "
    f"fla {fla.__version__} | gpus {torch.cuda.device_count()}"
)
assert torch.cuda.device_count() == 8, "expected 8 visible GPUs"
assert triton.__version__ == "3.6.0", "triton must be 3.6.0 to match torch pin and spatialstack-qwen35"
assert fla.__version__ == "0.4.1", "fla must be 0.4.1 to match spatialstack-qwen35"
PY
[ $? -ne 0 ] && { log "ERROR: environment check failed"; exit 1; }

log "=== data and model checks ==="
for path in \
    ./models/Qwen3.5-4B/config.json \
    ./models/VGGT-1B/config.json \
    ./data/media/spar/structured3d \
    ./data/media/spar/scannet \
    ./data/media/spar/scannetpp \
    ./data/media/spar/rxr
do
    [ -e "$path" ] || { log "ERROR: missing $path"; exit 1; }
    log "present: $path"
done

log "=== launching training (VGGT ENABLED, Qwen3.5 NON-THINKING) ==="
MODEL_PATH=/home/c30084464/Documents/code/SpatialStack_OPSD/models/Qwen3.5-4B \
GEOMETRY_ENCODER_PATH=/home/c30084464/Documents/code/SpatialStack_OPSD/models/VGGT-1B \
USE_GEOMETRY_ENCODER=True \
GEOMETRY_ENCODER_TYPE=vggt \
FEATURE_FUSION_METHOD=deepstack_language_add \
GEOMETRY_ENCODER_LAYERS="11 17 23" \
GEOMETRY_FUSION_LAYERS="0 1 2" \
DATA_FLATTEN=False \
OUTPUT_DIR="$OUT" \
CACHE_DIR=/home/c30084464/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
bash scripts/train/train.sh

TRAIN_RC=$?
log "=== training finished rc=$TRAIN_RC ==="

CKPT="$OUT"
if [ ! -f "$CKPT/config.json" ]; then
    CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
log "using checkpoint: $CKPT"
if [ ! -f "$CKPT/config.json" ]; then log "ERROR: no checkpoint, skipping eval"; exit 1; fi

log "=== launching evaluation (geometry encoder enabled) ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
MODEL_PATH="$CKPT" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true,geometry_encoder_path=/home/c30084464/Documents/code/SpatialStack_OPSD/models/VGGT-1B" \
OUTPUT_ROOT=logs/eval/spatialstack_qwen35_4b_vggt_aligned \
BENCHMARKS="vsibench,cvbench" \
bash scripts/evaluation/eval.sh
EVAL_RC=$?
log "=== DONE. train_rc=$TRAIN_RC eval_rc=$EVAL_RC ==="
