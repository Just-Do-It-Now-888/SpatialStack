#!/usr/bin/env python3
"""Compare VSI-Bench scoring when rule tier cannot parse an answer.

Protocol A (judge_fallback): rule_answered -> rule-primary score; else judge score.
Protocol B (rule_strict):     rule_answered -> rule-primary score; else 0.

Trusted rows (boxed / terse / mca_tail) always use rule-primary; they are excluded
from the 'unparsed' bucket because rule_answered is true there.

Usage::

    python3 scripts/opsd/tools/compare_unparsed_scoring.py \\
        logs/vsi_train_eval/.../judge_extract/samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, TOOLS_DIR)

from judge_vsibench_tri_insurance import TASK_UTILS, load_module  # noqa: E402
from judge_vsibench_full_extract import rule_primary_tier  # noqa: E402

SCORING = os.path.join(REPO_ROOT, "scripts", "opsd", "vsibench_scoring.py")


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_score_rule_primary(row: dict, scoring) -> tuple[float, bool]:
    score, answered, _ = rule_primary_tier(row, scoring)
    return float(score), bool(answered)


def build_scored_rows(rows: list[dict], scoring, task) -> tuple[list[dict], list[dict]]:
    judge_rows = []
    strict_rows = []
    for row in rows:
        rule_score, rule_answered = row_score_rule_primary(row, scoring)
        judge_score = float(row.get("judge_score", row.get("final_score", rule_score)))
        judge_answered = bool(row.get("judge_answered", row.get("final_answered", rule_answered)))
        trust = str(row.get("judge_trust") or "").strip()

        if rule_answered:
            score_a = score_b = rule_score
            answered_a = answered_b = True
        else:
            score_a = judge_score
            answered_a = judge_answered
            score_b = 0.0
            answered_b = False

        base = {
            "question_type": row["question_type"],
            "rule_answered": rule_answered,
            "judge_trust": trust,
            "truncated": bool(row.get("truncated")),
        }
        item_a = dict(base)
        item_b = dict(base)
        if row["question_type"] in task.MCA_QUESTION_TYPES:
            item_a["accuracy"] = score_a
            item_b["accuracy"] = score_b
        else:
            item_a["MRA:.5:.95:.05"] = score_a
            item_b["MRA:.5:.95:.05"] = score_b
        item_a["answered"] = answered_a
        item_b["answered"] = answered_b
        judge_rows.append(item_a)
        strict_rows.append(item_b)
    return judge_rows, strict_rows


def summarize(name: str, rows: list[dict], task) -> dict:
    return {
        "name": name,
        "overall": task.vsibench_aggregate_results([dict(r) for r in rows]),
        "answered_pct": task.vsibench_aggregate_answered(rows),
        "n": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="judge_extract/samples.jsonl (or post-judge samples.jsonl)")
    parser.add_argument("--json-out", default="", help="optional path to write comparison JSON")
    args = parser.parse_args()

    task = load_module(TASK_UTILS, "vsibench_task_utils_compare")
    scoring = load_module(SCORING, "vsibench_scoring_compare")
    rows = load_rows(args.samples)
    if not rows:
        raise SystemExit(f"no rows in {args.samples}")

    judge_rows, strict_rows = build_scored_rows(rows, scoring, task)
    unparsed = [r for r in rows if not row_score_rule_primary(r, scoring)[1]]
    unparsed_judge_recovered = sum(
        1 for r in unparsed if float(r.get("judge_score", 0.0)) >= 1.0 - 1e-9
    )
    unparsed_judge_answered = sum(1 for r in unparsed if bool(r.get("judge_answered", 0)))

    sum_judge = summarize("judge_fallback", judge_rows, task)
    sum_strict = summarize("rule_strict", strict_rows, task)

    result = {
        "samples": args.samples,
        "n": len(rows),
        "unparsed_rule": {
            "count": len(unparsed),
            "frac": len(unparsed) / len(rows),
            "judge_answered": unparsed_judge_answered,
            "judge_correct": unparsed_judge_recovered,
            "judge_recovered_frac": unparsed_judge_recovered / max(len(unparsed), 1),
        },
        "judge_fallback": sum_judge,
        "rule_strict": sum_strict,
        "delta_pp": sum_judge["overall"] - sum_strict["overall"],
        "answered_delta_pp": sum_judge["answered_pct"] - sum_strict["answered_pct"],
    }

    print(f"samples: {args.samples}")
    print(f"n={result['n']} | rule_unparsed={result['unparsed_rule']['count']} ({100*result['unparsed_rule']['frac']:.2f}%)")
    print(
        f"  unparsed bucket: judge answered {result['unparsed_rule']['judge_answered']}, "
        f"judge correct {result['unparsed_rule']['judge_correct']} "
        f"({100*result['unparsed_rule']['judge_recovered_frac']:.1f}% of unparsed)"
    )
    print()
    print(f"A judge_fallback (unparsed -> judge): overall {sum_judge['overall']:.2f} | answered {sum_judge['answered_pct']:.2f}%")
    print(f"B rule_strict     (unparsed -> 0):    overall {sum_strict['overall']:.2f} | answered {sum_strict['answered_pct']:.2f}%")
    print(f"delta (A - B): {result['delta_pp']:+.2f} pp overall, {result['answered_delta_pp']:+.2f} pp answered")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
