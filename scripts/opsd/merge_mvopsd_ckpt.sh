#!/usr/bin/env bash
# Convert a verl FSDP checkpoint into a HuggingFace directory lmms-eval can load.
#
#   scripts/opsd/merge_mvopsd_ckpt.sh checkpoints/<experiment_id>/global_step_300
#
# The merged weights land in output/<experiment_id>_step<N>_hf together with the
# processor/tokenizer files copied from the base checkpoint, because
# verl's merger writes weights and config but the evaluation entry point also
# needs the chat template and processor config (LESSON-002).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STEP_DIR="${1:?usage: merge_mvopsd_ckpt.sh <checkpoints/<exp>/global_step_N> [target_dir]}"
STEP_DIR="${STEP_DIR%/}"
ACTOR_DIR="${STEP_DIR}/actor"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen3.5-4B}"

if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "actor checkpoint not found: $ACTOR_DIR" >&2
  exit 1
fi

EXPERIMENT_NAME="$(basename "$(dirname "$STEP_DIR")")"
STEP_NAME="$(basename "$STEP_DIR")"
TARGET_DIR="${2:-${PROJECT_ROOT}/output/${EXPERIMENT_NAME}_${STEP_NAME}_hf}"
mkdir -p "$TARGET_DIR"

export PYTHONPATH="${PROJECT_ROOT}/verl_pkg:${PYTHONPATH:-}"
if [[ -z "${PYTHON_MERGE:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_MERGE="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/sr_opsd/bin/python" ]]; then
    PYTHON_MERGE="${HOME}/miniconda3/envs/sr_opsd/bin/python"
  else
    PYTHON_MERGE=/usr/bin/python3
  fi
fi

echo "merging ${ACTOR_DIR} -> ${TARGET_DIR}"
"${PYTHON_MERGE}" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$TARGET_DIR"

for asset in chat_template.jinja processor_config.json preprocessor_config.json \
             video_preprocessor_config.json tokenizer_config.json tokenizer.json \
             vocab.json merges.txt generation_config.json; do
  if [[ -f "${BASE_MODEL}/${asset}" && ! -f "${TARGET_DIR}/${asset}" ]]; then
    cp "${BASE_MODEL}/${asset}" "${TARGET_DIR}/${asset}"
  fi
done

if [[ ! -f "${TARGET_DIR}/config.json" ]]; then
  echo "merge produced no config.json in ${TARGET_DIR}; evaluation will fail" >&2
  exit 1
fi
echo "merged model ready: ${TARGET_DIR}"
