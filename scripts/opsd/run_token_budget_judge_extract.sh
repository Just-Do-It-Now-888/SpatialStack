#!/usr/bin/env bash
# Token-budget ablation: same vLLM + judge-extract, only max_tokens differs (1024 vs 4096).
# Runs base @1024 then step100 @1024 serially. 4096 arms reuse archived judge-extract results.
#
#   setsid nohup bash scripts/opsd/run_token_budget_judge_extract.sh \
#     > logs/eval/token_budget_judge/driver.log 2>&1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ROOT="logs/eval/token_budget_judge"
mkdir -p "${ROOT}"

link_archive() {
  local name="$1" src="$2"
  local dst="${ROOT}/${name}"
  mkdir -p "${dst}"
  if [[ -f "${src}/summary.json" ]]; then
    ln -sfn "$(realpath "${src}")" "${dst}/judge_extract"
    echo "linked archive ${name} -> ${src}"
  else
    echo "WARN: missing archive ${src}" >&2
  fi
}

link_archive base_tok4096 logs/eval/decoding_ab/base_greedy/judge_extract
link_archive step100_tok4096 \
  logs/vsi_train_eval/20260821_mvopsd_single_llava_hound_main/global_step_100/judge_extract

run_arm_hf() {
  local tag="$1" hf_path="$2" max_tokens="$3" max_model_len="$4"
  local out="${ROOT}/${tag}"
  mkdir -p "${out}"
  echo "=== START ${tag} max_tokens=${max_tokens} at=$(date +%F\ %T) ===" | tee -a "${ROOT}/status.txt"
  if OUT_DIR="${out}" \
    FRAMES=32 \
    MAX_TOKENS="${max_tokens}" \
    MAX_MODEL_LEN="${max_model_len}" \
    SCENES_PER_BATCH=24 \
    JUDGE=1 \
    JUDGE_MODE=extract \
    bash scripts/opsd/run_training_vsibench_eval.sh --hf "${hf_path}"; then
    echo "=== END ${tag} rc=0 at=$(date +%F\ %T) ===" | tee -a "${ROOT}/status.txt"
    return 0
  fi
  echo "=== END ${tag} rc=1 at=$(date +%F\ %T) ===" | tee -a "${ROOT}/status.txt"
  return 1
}

run_arm_ckpt() {
  local tag="$1" ckpt="$2" max_tokens="$3" max_model_len="$4"
  local out="${ROOT}/${tag}"
  mkdir -p "${out}"
  echo "=== START ${tag} max_tokens=${max_tokens} at=$(date +%F\ %T) ===" | tee -a "${ROOT}/status.txt"
  if OUT_DIR="${out}" \
    FRAMES=32 \
    MAX_TOKENS="${max_tokens}" \
    MAX_MODEL_LEN="${max_model_len}" \
    SCENES_PER_BATCH=24 \
    JUDGE=1 \
    JUDGE_MODE=extract \
    bash scripts/opsd/run_training_vsibench_eval.sh "${ckpt}"; then
    echo "=== END ${tag} rc=0 at=$(date +%F\ %T) ===" | tee -a "${ROOT}/status.txt"
    return 0
  fi
  echo "=== END ${tag} rc=1 at=$(date +%F\ %T) ===" | tee -a "${ROOT}/status.txt"
  return 1
}

status=0
run_arm_ckpt step100_tok1024 \
  checkpoints/20260821_mvopsd_single_llava_hound_main/global_step_100 \
  1024 12288 || status=1
run_arm_hf base_tok1024 models/Qwen3.5-4B 1024 12288 || status=1

echo "TOKEN_BUDGET_JUDGE_DONE status=${status} at=$(date +%F\ %T)"
exit "${status}"
