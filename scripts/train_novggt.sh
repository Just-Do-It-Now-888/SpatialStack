#!/usr/bin/env bash
# Qwen3.5-4B training with the VGGT geometry encoder disabled (ablation baseline).
# Non-thinking chat template is applied automatically by data_qwen.py for model_type qwen3.5.
set -uo pipefail
cd /home/c30084464/Documents/code/SpatialStack_OPSD
source /home/c30084464/miniconda3/etc/profile.d/conda.sh
conda activate sr_opsd

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/c30084464/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/c30084464/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

OUT=./output/spatialstack_qwen35_novggt_aligned
mkdir -p ./output ./logs "$OUT"
log(){ echo "[novggt $(date +%H:%M:%S)] $*"; }

log "=== env check ==="
python - <<'PY'
import torch, triton, fla
print(f"  torch {torch.__version__} | triton {triton.__version__} | fla {fla.__version__} | gpus {torch.cuda.device_count()}")
assert triton.__version__ == "3.6.0", "triton must be 3.6.0 to match torch pin and spatialstack-qwen35"
assert fla.__version__ == "0.4.1", "fla must be 0.4.1 to match spatialstack-qwen35"
PY
[ $? -ne 0 ] && { log "ERROR: env check failed"; exit 1; }

log "=== verify data ==="
for d in structured3d scannet scannetpp rxr; do
  [ -d ./data/media/spar/$d ] && log "  spar/$d present" || log "  spar/$d MISSING"
done

log "=== launching training (VGGT DISABLED) ==="
MODEL_PATH=/home/c30084464/Documents/code/SpatialStack_OPSD/models/Qwen3.5-4B \
USE_GEOMETRY_ENCODER=False \
DATA_FLATTEN=False \
OUTPUT_DIR="$OUT" \
CACHE_DIR=/home/c30084464/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
bash scripts/train/train.sh
TRAIN_RC=$?
log "training finished rc=$TRAIN_RC"

CKPT="$OUT"
if [ ! -f "$CKPT/config.json" ]; then
    CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
log "using checkpoint: $CKPT"
if [ ! -f "$CKPT/config.json" ]; then log "ERROR: no checkpoint, skipping eval"; exit 1; fi

log "=== launching evaluation (no geometry encoder) ==="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
MODEL_PATH="$CKPT" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
OUTPUT_ROOT=logs/eval/spatialstack_qwen35_4b_novggt_aligned \
BENCHMARKS="vsibench,cvbench" \
bash scripts/evaluation/eval.sh
EVAL_RC=$?
log "=== DONE. train_rc=$TRAIN_RC eval_rc=$EVAL_RC ==="
