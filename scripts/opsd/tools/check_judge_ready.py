#!/usr/bin/env python3
"""Confirm a judge server is grading, not just answering.

    python3 scripts/opsd/tools/check_judge_ready.py --api-base http://127.0.0.1:8100/v1

``check_judge_handshake.py`` is the check for a judge that shares its cards with
training and therefore has to sleep and wake; it needs the vLLM dev endpoints
and costs a sleep cycle.  This is the check for a judge that owns the cards: it
grades one row that is right and one that is wrong, which is enough to catch the
failure that matters -- the MXFP4 checkpoint can come up healthy and reply
"!!!!!!!!" to everything (ISSUE-005), and a judge like that would mark a whole
run wrong while looking fine.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

QUESTIONS = [
    "Which object is closer to the camera? (A) the chair (B) the lamp",
    "How many chairs are in the image? (A) 1 (B) 2 (C) 3 (D) 4",
]
GROUND_TRUTHS = ["(B)", "(C)"]
RESPONSES = ["After comparing the depths, Answer: B", "I count two chairs. Answer: B"]
EXPECTED = [1.0, 0.0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="judge")
    args = parser.parse_args()

    path = os.path.join(REPO_ROOT, "scripts", "opsd", "mvopsd_judge.py")
    spec = importlib.util.spec_from_file_location("mvopsd_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    judge = module.ValidationJudge(api_base=args.api_base, model=args.model, parallel_workers=4)
    verdicts, stats = judge.grade(QUESTIONS, GROUND_TRUTHS, RESPONSES)
    print(f"verdicts {verdicts} in {stats['seconds']:.1f}s (unreadable={stats['unparsed']})")
    if verdicts == EXPECTED:
        print("judge ready: it accepts a correct answer and rejects a wrong one")
        return 0
    print(f"judge NOT ready: expected {EXPECTED}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
