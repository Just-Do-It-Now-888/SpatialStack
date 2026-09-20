#!/usr/bin/env bash
# LESSON-041: from 2026-09-10, all training launches must use conda sr_opsd.
# Source this file, then call require_sr_opsd_training_env.
# ALLOW_NON_SROPSD=1 is a documented one-off only; record it in the registry.

require_sr_opsd_training_env() {
  if [[ "${ALLOW_NON_SROPSD:-0}" == "1" ]]; then
    echo "[WARN] ALLOW_NON_SROPSD=1; skipping sr_opsd check. Record the exception in the registry." >&2
    return 0
  fi
  local py
  py="$(command -v python3 2>/dev/null || command -v python || true)"
  if [[ "$py" != *"/envs/sr_opsd/"* ]]; then
    echo "[ERROR] training must use conda sr_opsd (LESSON-041)." >&2
    echo "        python=${py:-none} CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-} CONDA_PREFIX=${CONDA_PREFIX:-}" >&2
    echo "        source ~/miniconda3/etc/profile.d/conda.sh && conda activate sr_opsd" >&2
    return 1
  fi
}
