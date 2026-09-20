#!/usr/bin/env python3
"""Aggregate judged rows into CV-Bench scores, with the null hypotheses attached.

Beyond the Vision-OPD roll-up ``(acc_2D + acc_3D) / 2`` this prints two things
LESSON-011 says must never again be separated from a benchmark number:

* **response shape** -- how many rows the grader could resolve at all, how many
  hit the generation cap, how long the answers were. A score computed over rows
  that never contained an answer is not a score;
* **null hypotheses** -- what a constant predictor and a purely lexical rule
  ("did the response open with 'Based'?") would score on the same rows. MV-OPSD
  v0's entire CV-Bench curve was reproduced by that second rule with zero
  residual, and nothing in the pipeline noticed for a day.

The lmms_eval roll-up ``((ADE20K + COCO) / 2 + Omni3D) / 2`` is printed too.
It differs from Vision-OPD's whenever ADE20K and COCO have different counts, so
quoting one number against the other without saying which is a silent error.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

STANDALONE_OPTION = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")


def is_correct(item: dict) -> bool:
    return str(item.get("judge", "")).strip().lower() == "yes"


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(part: int, total: int) -> str:
    return f"{part}/{total} = {100 * part / total:.2f}%" if total else "0/0 = n/a"


def group_acc(rows, key, ok):
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(1 if ok(row) else 0)
    return {name: mean(values) for name, values in sorted(buckets.items())}, buckets


def visionopd_combined(rows, ok) -> float:
    by_type, _ = group_acc(rows, "type", ok)
    return 100 * (by_type.get("2D", 0.0) + by_type.get("3D", 0.0)) / 2


def lmms_combined(rows, ok) -> float:
    by_source, _ = group_acc(rows, "source", ok)
    acc_2d = (by_source.get("ADE20K", 0.0) + by_source.get("COCO", 0.0)) / 2
    return 100 * (acc_2d + by_source.get("Omni3D", 0.0)) / 2


def report_shape(rows) -> None:
    total = len(rows)
    capped = sum(1 for r in rows if r.get("finish_reason") == "length")
    errored = sum(1 for r in rows if str(r.get("model_answer", "")).startswith("["))
    resolved = sum(1 for r in rows if r.get("pred_option"))
    lengths = sorted(r["completion_tokens"] for r in rows if isinstance(r.get("completion_tokens"), int))

    print("=== response shape ===")
    print(f"  rows:                 {total}")
    print(f"  hit generation cap:   {pct(capped, total)}")
    print(f"  api/infra errors:     {pct(errored, total)}")
    print(f"  option letter parsed: {pct(resolved, total)}")
    if lengths:
        print(
            f"  completion tokens:    mean {mean(lengths):.0f}  p50 {lengths[len(lengths) // 2]}  "
            f"p99 {lengths[int(0.99 * (len(lengths) - 1))]}  max {lengths[-1]}"
        )
    if capped:
        print("  WARNING: capped responses are graded on a truncated prefix (ISSUE-003).")
    if resolved == 0:
        print("  WARNING: no row yielded an option letter. The score below is noise.")


def report_judge_hygiene(rows) -> None:
    """Count rows the upstream verdict rule scores wrong for a non-answer reason.

    ``is_correct`` is upstream's exact ``== "yes"``, so a judge that replied
    "Yes, the response is correct.", or did not reply at all, scores the row
    wrong no matter what the model wrote. That is the reported number and it
    stays that way; it just must not be invisible.
    """
    llm_rows = [r for r in rows if r.get("judge_source") == "llm"]
    if not llm_rows:
        return
    empty = sum(1 for r in llm_rows if not str(r.get("judge_raw", "")).strip())
    api_errors = sum(1 for r in llm_rows if r.get("judge_api_error"))
    off_protocol = sum(
        1
        for r in llm_rows
        if str(r.get("judge", "")).strip().lower() not in ("yes", "no")
    )
    lost = sum(
        1
        for r in llm_rows
        if str(r.get("judge", "")).strip().lower() != "yes"
        and r.get("judge_normalized") == "Yes"
    )
    print("\n=== judge reply hygiene ===")
    print(f"  llm-judged rows:              {len(llm_rows)}")
    print(f"  replies that are not Yes/No:  {pct(off_protocol, len(llm_rows))}")
    print(f"  empty replies:                {pct(empty, len(llm_rows))}")
    if api_errors:
        print(f"  judge call failures:          {api_errors}")
    if lost:
        print(f"  scored wrong but normalized to Yes: {lost}")


def report_nulls(rows) -> None:
    """Score rules that ignore the image. Anything they reproduce is not a result."""
    print("\n=== null hypotheses (image-independent rules) ===")

    def gold_letter(row):
        gt = str(row.get("response", ""))
        return gt[1] if len(gt) > 1 and gt.startswith("(") else gt.strip()[:1]

    best_const, best_letter = 0.0, ""
    for letter in "ABCDEF":
        score = visionopd_combined(rows, lambda r, c=letter: gold_letter(r) == c)
        if score > best_const:
            best_const, best_letter = score, letter
    print(f"  best constant predictor:  always '{best_letter}' -> {best_const:.2f}")

    based = visionopd_combined(
        rows,
        lambda r: str(r.get("model_answer", "")).strip().startswith("Based")
        and gold_letter(r) == "B",
    )
    print(f"  'opens with Based' -> B:  {based:.2f}   (this rule reproduced all of v0)")

    prefix_hits = sum(
        1
        for r in rows
        if not STANDALONE_OPTION.search(str(r.get("extracted_answer", "")))
        and re.search(r"[A-F]", str(r.get("extracted_answer", "")))
    )
    print(f"  rows whose only A-F letter is inside a word: {prefix_hits}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="cvbench")
    parser.add_argument("--judge-json", required=True)
    args = parser.parse_args()

    with open(args.judge_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    report_shape(rows)

    print("\n=== accuracy ===")
    by_type, type_rows = group_acc(rows, "type", is_correct)
    for name in sorted(by_type):
        n = len(type_rows[name])
        print(f"  {name:<10} {100 * by_type[name]:6.2f}%  (n={n})")

    by_task, task_rows = group_acc(rows, "task", is_correct)
    for name in sorted(by_task):
        print(f"    {name:<8} {100 * by_task[name]:6.2f}%  (n={len(task_rows[name])})")

    by_source, source_rows = group_acc(rows, "source", is_correct)
    for name in sorted(by_source):
        print(f"    {name:<8} {100 * by_source[name]:6.2f}%  (n={len(source_rows[name])})")

    vo = visionopd_combined(rows, is_correct)
    lm = lmms_combined(rows, is_correct)
    print(f"\n  {args.benchmark} combined (Vision-OPD, (2D+3D)/2):        {vo:.2f}")
    print(f"  {args.benchmark} combined (lmms_eval, ((ADE+COCO)/2+3D)/2): {lm:.2f}")

    tiers = defaultdict(lambda: [0, 0])
    for row in rows:
        tier = tiers[row.get("judge_source", "?")]
        tier[1] += 1
        tier[0] += 1 if is_correct(row) else 0
    print("\n=== verdict provenance ===")
    for name, (correct, total) in sorted(tiers.items()):
        print(f"  {name:<18} {pct(correct, total)}")

    report_judge_hygiene(rows)
    report_nulls(rows)


if __name__ == "__main__":
    main()
