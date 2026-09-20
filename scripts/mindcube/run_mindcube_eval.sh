#!/usr/bin/env bash
# Shard the MindCube tinybench eval across the local GPUs, one process per card.
#
#   MODEL=./models/Qwen3.5-4B OUT_DIR=logs/eval/foo bash scripts/mindcube/run_mindcube_eval.sh
#
# PYTHONNOUSERSITE is mandatory: ~/.local carries an opentelemetry install whose
# missing `zipp` breaks `import transformers` inside this conda env.
set -euo pipefail

MODEL="${MODEL:-./models/Qwen3.5-4B}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
# node-A has spatialstack-qwen35; node-B only has sr_opsd. Both are transformers
# 5.3.0 / torch 2.10.0+cu129 / py 3.12.13, so either gives comparable numbers.
# Hardcoding one path made every shard die instantly on the other node.
if [[ -z "${PYTHON:-}" ]]; then
    for cand in /home/c30084464/miniconda3/envs/{spatialstack-qwen35,sr_opsd}/bin/python; do
        [[ -x "$cand" ]] && { PYTHON="$cand"; break; }
    done
fi
: "${PYTHON:?no usable python found; set PYTHON=...}"
PYTHONNOUSERSITE=1 "$PYTHON" -c 'import transformers; assert transformers.__version__=="5.3.0", transformers.__version__' \
    || { echo ">>> transformers version differs from the anchor run; scores would not be comparable" >&2; exit 1; }

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES

mkdir -p "$OUT_DIR"
echo ">>> model=$MODEL out=$OUT_DIR shards=$NUM_SHARDS max_new_tokens=$MAX_NEW_TOKENS"

extra=()
[[ -n "$LIMIT_ROWS" ]] && extra+=(--limit "$LIMIT_ROWS")

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="$i" "$PYTHON" scripts/mindcube/eval_mindcube_qwen35.py \
        --model "$MODEL" \
        --out-dir "$OUT_DIR" \
        --shard-index "$i" \
        --num-shards "$NUM_SHARDS" \
        --device cuda:0 \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${extra[@]}" \
        > "$OUT_DIR/shard_$(printf '%02d' "$i").log" 2>&1 &
    pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
if [[ $rc -ne 0 ]]; then
    echo ">>> at least one shard failed; see $OUT_DIR/shard_*.log" >&2
    exit 1
fi

"$PYTHON" scripts/mindcube/eval_mindcube_qwen35.py \
    --out-dir "$OUT_DIR" --num-shards "$NUM_SHARDS" --merge-only \
    | tee "$OUT_DIR/summary.txt"
