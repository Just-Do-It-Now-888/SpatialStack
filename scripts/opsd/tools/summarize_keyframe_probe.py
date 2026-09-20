#!/usr/bin/env python3
"""Summarize paired full vs key teacher accuracy from a keyframe probe run.

    python3 scripts/opsd/tools/summarize_keyframe_probe.py \
        --scored logs/eval/teacher_keyframe_probe/spar32/scored.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

def read_scored(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_paired(rows: list[dict]) -> dict:
    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        arm = row.get("sweep_arm")
        group = row.get("sweep_group")
        if not arm or not group:
            continue
        by_group[group][arm] = row

    paired = []
    for group, arms in by_group.items():
        if "full" not in arms or "key" not in arms:
            continue
        full = arms["full"]
        key = arms["key"]
        paired.append(
            {
                "sweep_group": group,
                "question_type": full.get("question_type", ""),
                "source": full.get("source", ""),
                "full_score": float(full.get("score", 0.0)),
                "key_score": float(key.get("score", 0.0)),
                "full_answered": int(full.get("answered", 0)),
                "key_answered": int(key.get("answered", 0)),
                "n_views_full": int(full.get("n_views_teacher", 0)),
                "n_views_key": int(key.get("n_views_teacher", 0)),
                "delta": float(key.get("score", 0.0)) - float(full.get("score", 0.0)),
            }
        )
    return {"pairs": paired, "pair_count": len(paired)}


def _mean_score(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return 100 * sum(float(row[field]) for row in rows) / len(rows)


def table_by_key(rows: list[dict], key_field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key_field, ""))].append(row)

    out = {}
    for key, group_rows in sorted(buckets.items()):
        full_acc = _mean_score(group_rows, "full_score")
        key_acc = _mean_score(group_rows, "key_score")
        out[key] = {
            "pairs": len(group_rows),
            "full_acc": full_acc,
            "key_acc": key_acc,
            "delta_acc": key_acc - full_acc,
            "mean_key_views": sum(row["n_views_key"] for row in group_rows) / len(group_rows),
            "mean_full_views": sum(row["n_views_full"] for row in group_rows) / len(group_rows),
        }
    return out


def print_table(title: str, table: dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    print(
        f"  {'key':<40}{'pairs':>8}{'full':>8}{'key':>8}{'delta':>8}"
        f"{'k_views':>8}{'N_views':>8}"
    )
    for key, info in table.items():
        print(
            f"  {key:<40}{info['pairs']:>8}{info['full_acc']:>8.2f}"
            f"{info['key_acc']:>8.2f}{info['delta_acc']:>8.2f}"
            f"{info['mean_key_views']:>8.1f}{info['mean_full_views']:>8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", required=True, help="scored.jsonl from score_teacher_dump.py")
    parser.add_argument("--out", default=None, help="optional summary json path")
    args = parser.parse_args()

    scored_path = args.scored if os.path.isabs(args.scored) else os.path.join(REPO_ROOT, args.scored)
    rows = read_scored(scored_path)
    paired = aggregate_paired(rows)
    pairs = paired["pairs"]
    if not pairs:
        raise SystemExit("no paired full/key rows found; need sweep_group + sweep_arm on scored rows")

    overall = table_by_key(pairs, "source")
    by_type = table_by_key(pairs, "question_type")
    by_key_count = table_by_key(
        [{**row, "key_count": str(row["n_views_key"])} for row in pairs],
        "key_count",
    )

    print(f"paired rows: {len(pairs)} from {len(rows)} scored rows")
    print_table("overall by source", overall)
    print_table("by question_type", by_type)
    print_table("by key frame count", by_key_count)

    summary = {
        "scored": os.path.relpath(scored_path, REPO_ROOT),
        "pair_count": len(pairs),
        "overall": overall,
        "by_question_type": by_type,
        "by_key_frame_count": by_key_count,
    }
    out_path = args.out
    if out_path:
        out_path = out_path if os.path.isabs(out_path) else os.path.join(REPO_ROOT, out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
