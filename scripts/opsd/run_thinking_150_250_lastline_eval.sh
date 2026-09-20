#!/usr/bin/env bash
# Serial offline VSI-Bench for 20260827 thinking ckpts steps 150-250.
# Hard constraints: MAX_TOKENS=4096 and DO_SAMPLE=1 (t=1.0 / top_p=0.8).
#
#   setsid nohup bash scripts/opsd/run_thinking_150_250_lastline_eval.sh \
#     > logs/eval/20260828_thinking_150_250_lastline/driver.log 2>&1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ROOT="logs/eval/20260828_thinking_150_250_lastline"
mkdir -p "${ROOT}"
STATUS="${ROOT}/status.txt"

STEPS=(150 175 200 225 250)
EXP="20260827_mvopsd_vlm3r_epoch1_thinking"
SUFFIX=' The final answer MUST BE put in \boxed{} on the last line of your response.'

wait_gpu_free() {
  local peak=999999
  for _ in $(seq 1 60); do
    peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    (( peak < 5000 )) && return 0
    sleep 5
  done
  echo "WARN: GPU still ${peak} MiB after wait" | tee -a "${STATUS}"
  return 1
}

check_protocol() {
  local summary="$1" step="$2"
  python3 - "$summary" "$step" <<'PY'
import json, sys
path, step = sys.argv[1], sys.argv[2]
with open(path) as f:
    s = json.load(f)
dec = s.get("decoding") or {}
ok = True
def fail(msg):
    global ok
    ok = False
    print(f"PROTOCOL FAIL step {step}: {msg}", file=sys.stderr)
if int(s.get("max_tokens", 0)) != 4096:
    fail(f"max_tokens={s.get('max_tokens')!r} (want 4096)")
if dec.get("do_sample") is not True:
    fail(f"do_sample={dec.get('do_sample')!r} (want true)")
if abs(float(dec.get("temperature", -1)) - 1.0) > 1e-6:
    fail(f"temperature={dec.get('temperature')!r} (want 1.0)")
if abs(float(dec.get("top_p", -1)) - 0.8) > 1e-6:
    fail(f"top_p={dec.get('top_p')!r} (want 0.8)")
rule = (s.get("rule_only") or {}).get("overall")
print(f"PROTOCOL OK step {step} overall={rule} max_tokens={s.get('max_tokens')} do_sample={dec.get('do_sample')}")
sys.exit(0 if ok else 1)
PY
}

echo "=== DRIVER START at=$(date +%F\ %T) MAX_TOKENS=4096 DO_SAMPLE=1 ===" | tee "${STATUS}"

for step in "${STEPS[@]}"; do
  ckpt="checkpoints/${EXP}/global_step_${step}"
  out="logs/vsi_train_eval/${EXP}/global_step_${step}_tok4096_sample_boxed_lastline_rule"
  mkdir -p "${out}"
  echo "=== START global_step_${step} at=$(date +%F\ %T) ===" | tee -a "${STATUS}"
  wait_gpu_free || true
  if JUDGE=0 \
    MAX_TOKENS=4096 \
    MAX_MODEL_LEN=32768 \
    DO_SAMPLE=1 \
    TEMPERATURE=1.0 \
    TOP_P=0.8 \
    BOXED_PRIMARY=1 \
    FRAMES=32 \
    SCENES_PER_BATCH=24 \
    PROMPT_SUFFIX="${SUFFIX}" \
    OUT_DIR="${REPO_ROOT}/${out}" \
    bash scripts/opsd/run_training_vsibench_eval.sh "${REPO_ROOT}/${ckpt}"; then
    if ! check_protocol "${out}/summary.json" "${step}"; then
      echo "=== ABORT global_step_${step} protocol mismatch at=$(date +%F\ %T) ===" | tee -a "${STATUS}"
      exit 1
    fi
    echo "=== END global_step_${step} rc=0 at=$(date +%F\ %T) ===" | tee -a "${STATUS}"
  else
    echo "=== END global_step_${step} rc=1 at=$(date +%F\ %T) ===" | tee -a "${STATUS}"
    exit 1
  fi
done

echo "THINKING_150_250_LASTLINE_DONE at=$(date +%F\ %T)" | tee -a "${STATUS}"
exit 0
