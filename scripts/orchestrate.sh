#!/usr/bin/env bash
# Autonomous orchestration v2: verify SPAR chunks -> clean -> extract -> verify data -> train -> eval.
set -uo pipefail

cd /home/c30084464/Documents/code/SpatialStack_OPSD
source /home/c30084464/miniconda3/etc/profile.d/conda.sh
conda activate sr_opsd

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/c30084464/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/c30084464/.cache/huggingface/hub
export TOKENIZERS_PARALLELISM=false

SPAR_DIR=./data/media/spar
mkdir -p ./output

log(){ echo "[orchestrate $(date +%H:%M:%S)] $*"; }

log "=== orchestrate v2 start ==="

# ---------- 1. Wait for all 14 SPAR chunks at correct size ----------
EXPECTED_BIG=10737418240
EXPECTED_13=8996464724
log "waiting for all 14 SPAR chunks at correct size..."
while true; do
    ok=1
    for c in 00 01 02 03 04 05 06 07 08 09 10 11 12; do
        sz=$(stat -c %s "$SPAR_DIR/spar-$c.tar.gz" 2>/dev/null || echo 0)
        [ "$sz" -ge "$EXPECTED_BIG" ] || { ok=0; break; }
    done
    sz13=$(stat -c %s "$SPAR_DIR/spar-13.tar.gz" 2>/dev/null || echo 0)
    [ "$sz13" -ge "$EXPECTED_13" ] || ok=0
    n_curl=$(pgrep -f "curl -sL --connect" | wc -l)
    if [ "$ok" -eq 1 ] && [ "$n_curl" -eq 0 ]; then
        log "all 14 chunks verified at correct size."
        break
    fi
    log "chunks not ready (curl_procs=$n_curl); waiting 60s..."
    sleep 60
done

# ---------- 2. Clean partial extraction (keep .tar.gz) ----------
log "cleaning any partial SPAR extraction..."
find "$SPAR_DIR" -mindepth 1 -maxdepth 1 -type d ! -name ".cache" -exec rm -rf {} + 2>/dev/null || true
log "cleaned. remaining in spar dir:"; ls "$SPAR_DIR" | head -20

# ---------- 3. Extract SPAR ----------
log "extracting SPAR-7M (cat 14 chunks | pigz -dc | tar -xf)..."
mkdir -p ./data/media
(
  cd ./data/media
  cat \
    spar/spar-00.tar.gz spar/spar-01.tar.gz spar/spar-02.tar.gz spar/spar-03.tar.gz \
    spar/spar-04.tar.gz spar/spar-05.tar.gz spar/spar-06.tar.gz spar/spar-07.tar.gz \
    spar/spar-08.tar.gz spar/spar-09.tar.gz spar/spar-10.tar.gz spar/spar-11.tar.gz \
    spar/spar-12.tar.gz spar/spar-13.tar.gz \
  | pigz -dc | tar -xf - -C ./
)
EXTRACT_RC=$?
log "extraction rc=$EXTRACT_RC"
# flatten if extracted into spar/spar/*
if [ -d ./data/media/spar/spar ]; then
    log "flattening spar/spar/* -> spar/*"
    mv ./data/media/spar/spar/* ./data/media/spar/ 2>/dev/null || true
    rmdir ./data/media/spar/spar 2>/dev/null || true
fi
log "spar dir contents:"; ls ./data/media/spar/ | head -20
log "structured3d present: $([ -d ./data/media/spar/structured3d ] && echo yes || echo NO)"
log "scannet present: $([ -d ./data/media/spar/scannet ] && echo yes || echo NO)"

# ---------- 4. Verify all data paths ----------
log "verifying data paths..."
python - <<'PY'
import os, sys
checks = [
    ("spar_234k", "data/annotations/spar_234k.json", "data/media", "spar/structured3d/images"),
    ("spar_scannet", None, "data/media", "spar/scannet"),
    ("llava_hound_64k", "data/annotations/llava_hound_64k.json", "data/media", "llava_hound/frames"),
    ("vlm3r_scannet", "data/annotations/merged_qa_scannet_train.json", "data/vlm3r/media", "scannet/videos"),
    ("vsi_appr_order", "data/annotations/vsi_appearance_order_vsibench_scannet.json", "data/vsi_590k/media", "scannet"),
]
ok=True
for name, ann, dp, sub in checks:
    a = (ann is None) or os.path.exists(ann)
    full = os.path.join(dp, sub)
    d = os.path.isdir(full)
    print(f"  {name}: ann={a} media_dir={d} ({full})")
    if not (a and d): ok=False
print("VERIFY_OK" if ok else "VERIFY_FAIL")
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then log "data verification FAILED. Aborting."; exit 1; fi

# ---------- 5. Launch training ----------
log "launching training..."
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

# Locate checkpoint
CKPT=./output/spatialstack_qwen35_train
if [ ! -f "$CKPT/config.json" ]; then
    CKPT=$(ls -d ./output/spatialstack_qwen35_train/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
log "using checkpoint: $CKPT"
if [ ! -f "$CKPT/config.json" ]; then log "ERROR: no checkpoint. Aborting eval."; exit 1; fi

# ---------- 6. Launch evaluation ----------
log "launching evaluation..."
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/c30084464/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/c30084464/.cache/huggingface/hub
MODEL_PATH="$CKPT" \
MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=/home/c30084464/Documents/code/SpatialStack_OPSD/models/VGGT-1B,disable_thinking=true" \
OUTPUT_ROOT=logs/eval/spatialstack_qwen35_4b \
BENCHMARKS="vsibench,cvbench,blink_spatial,sparbench" \
bash scripts/evaluation/eval.sh
EVAL_RC=$?
log "evaluation finished rc=$EVAL_RC"
log "=== DONE. train_rc=$TRAIN_RC eval_rc=$EVAL_RC ==="
