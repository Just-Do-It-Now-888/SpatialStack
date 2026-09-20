#!/usr/bin/env python3
"""Exercise the validation judge's sleep/wake cycle against a live server.

    python3 scripts/opsd/tools/check_judge_handshake.py --port 8100

The in-training judge only pays off if it can hand the GPUs back: it shares the
cards with training, so a judge that wakes but does not release breaks the next
rollout rather than just a metric. That is not visible from the trainer's own
logs, so check it directly -- start the server the way run_mvopsd.sh does and
run this.

The check grades the same two rows before and after a sleep cycle, because
memory accounting alone will not catch the failure that matters most: sleep
level 2 discards the weights and reloads them on wake, and for this MXFP4
checkpoint the reload came back as garbage -- the server stayed healthy,
reported itself awake, held the right amount of memory, and answered every
prompt with "!!!!!!!!". A judge that silently scores like that would mark a
whole validation wrong. Only a verdict the judge got right before sleeping and
still gets right afterwards proves the cycle is safe.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def load_judge_module():
    path = os.path.join(REPO_ROOT, "scripts", "opsd", "mvopsd_judge.py")
    spec = importlib.util.spec_from_file_location("mvopsd_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_memory() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.split()
        used = [int(value) for value in out]
        return f"{sum(used) / 1024:.1f} GiB total, per card {used}"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def stage(label: str, judge) -> None:
    print(f"  {label:<22} is_sleeping={judge.is_sleeping()!s:<5} {gpu_memory()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--sleep-level", type=int, default=1)
    args = parser.parse_args()

    module = load_judge_module()
    judge = module.ValidationJudge(
        api_base=f"http://127.0.0.1:{args.port}/v1",
        sleep_level=args.sleep_level,
        parallel_workers=4,
    )

    if judge.is_sleeping() is None:
        print(f"no judge server answering on port {args.port}")
        print("start one the way run_mvopsd.sh does, with VLLM_SERVER_DEV_MODE=1 and --enable-sleep-mode")
        return 1

    # One row right, one row wrong, so a server that answers "Yes" to everything
    # reads as broken rather than as a working judge.
    questions = [
        "Which object is closer to the camera? (A) the chair (B) the lamp",
        "How many chairs are in the image? (A) 1 (B) 2 (C) 3 (D) 4",
    ]
    ground_truths = ["(B)", "(C)"]
    responses = ["After comparing the depths, Answer: B", "I count two chairs. Answer: B"]
    EXPECTED = [1.0, 0.0]

    def grade(label: str):
        verdicts, stats = judge.grade(questions, ground_truths, responses)
        mark = "ok" if verdicts == EXPECTED else "WRONG"
        print(f"  {label:<22} {verdicts} ({mark}, {stats['seconds']:.1f}s, unreadable={stats['unparsed']})")
        return verdicts

    print("stages:")
    stage("initial", judge)
    if judge.is_sleeping():
        print("  waking it, the baseline verdict has to come from a judge that is up")
        if not judge.wake():
            print("FAIL: the judge did not wake")
            return 1
        stage("after wake", judge)

    print("verdicts:")
    before = grade("before sleeping")
    if before != EXPECTED:
        print()
        print(f"the judge is wrong before any sleep cycle: got {before}, expected {EXPECTED}")
        print("fix the server or the prompt first; sleep/wake is not the suspect here")
        return 1

    print("stages:")
    started = time.time()
    judge.sleep()
    time.sleep(5)
    stage(f"after sleep ({time.time() - started:.0f}s)", judge)

    started = time.time()
    woke = judge.wake()
    stage(f"after wake ({time.time() - started:.0f}s)", judge)
    if not woke:
        print("FAIL: the judge did not wake back up")
        return 1

    print("verdicts:")
    after = grade("after wake")

    print()
    if after == EXPECTED:
        print(f"handshake ok at sleep level {args.sleep_level}: the judge releases the cards and")
        print("comes back grading the same way it did before, so validation can share the GPUs")
        return 0
    print(f"the sleep cycle broke the judge: {before} before, {after} after")
    print(f"sleep level {args.sleep_level} is not safe for this checkpoint -- level 1 keeps the")
    print("weights in host RAM instead of rebuilding them, which survives MXFP4 repacking")
    return 1


if __name__ == "__main__":
    sys.exit(main())
