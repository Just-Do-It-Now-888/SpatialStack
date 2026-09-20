#!/usr/bin/env python3
"""Summarize teacher accuracy vs extra frames added on top of key frames.

    python3 scripts/opsd/tools/summarize_key_plus_probe.py \
        --scored logs/eval/teacher_key_plus_probe/spar32/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def read_scored(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_acc(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return 100 * sum(float(row.get("score", 0.0)) for row in rows) / len(rows)


def mean_answered(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return 100 * sum(int(row.get("answered", 0)) for row in rows) / len(rows)


def bucket_table(rows: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field, ""))].append(row)
    out = {}
    for key, group in sorted(buckets.items(), key=lambda item: (len(item[0]), item[0]) if not _intish(item[0]) else (0, int(item[0]))):
        out[key] = {
            "rows": len(group),
            "acc": round(mean_acc(group), 2),
            "answered_pct": round(mean_answered(group), 1),
            "mean_views": round(sum(int(r.get("n_views_teacher") or 0) for r in group) / len(group), 2),
            "mean_extra": round(sum(int(r.get("extra_added") or 0) for r in group) / len(group), 2),
        }
    return out


def _intish(value: str) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def print_table(title: str, table: dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'key':<42}{'rows':>8}{'acc':>8}{'ans%':>8}{'views':>8}")
    for key, info in table.items():
        print(
            f"  {key:<42}{info['rows']:>8}{info['acc']:>8.2f}"
            f"{info['answered_pct']:>7.1f}%{info['mean_views']:>8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    scored_path = args.scored if os.path.isabs(args.scored) else os.path.join(REPO_ROOT, args.scored)
    rows = read_scored(scored_path)
    if not rows:
        raise SystemExit("empty scored dump")

    by_extra = bucket_table(rows, "extra_added")
    by_type = bucket_table(rows, "question_type")
    by_type_extra: dict[str, dict] = {}
    grouped: dict[tuple, list] = defaultdict(list)
    for row in rows:
        grouped[(row.get("question_type", ""), int(row.get("extra_added") or 0))].append(row)
    for (qtype, extra), group in sorted(grouped.items()):
        by_type_extra[f"{qtype}|+{extra}"] = {
            "rows": len(group),
            "acc": round(mean_acc(group), 2),
            "answered_pct": round(mean_answered(group), 1),
            "mean_views": round(sum(int(r.get("n_views_teacher") or 0) for r in group) / len(group), 2),
            "mean_extra": extra,
        }

    print(f"scored rows: {len(rows)}")
    print_table("accuracy vs extra frames (key + N)", by_extra)
    print_table("by question_type (all extras pooled)", by_type)
    print_table("by question_type × extra", by_type_extra)

    baseline = by_extra.get("0", {}).get("acc")
    plus10 = by_extra.get("10", {}).get("acc")
    if baseline is not None and plus10 is not None:
        print(f"\nkey+0 {baseline:.2f} → key+10 {plus10:.2f}  (Δ {plus10 - baseline:+.2f} pp)")

    summary = {
        "scored": os.path.relpath(scored_path, REPO_ROOT),
        "rows": len(rows),
        "by_extra_added": by_extra,
        "by_question_type": by_type,
        "by_question_type_and_extra": by_type_extra,
        "delta_plus10_vs_key": None if baseline is None or plus10 is None else round(plus10 - baseline, 2),
    }
    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
