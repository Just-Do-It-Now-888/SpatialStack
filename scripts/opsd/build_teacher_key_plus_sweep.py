#!/usr/bin/env python3
"""Build a nested key-plus-extra teacher sweep on SPAR 32-view.

Keeps ``required_views`` (oracle key frames) on every row, then adds 0..10
extra album frames.  The extras are nested: key ⊂ key+1 ⊂ … ⊂ key+10, so the
curve is a within-question comparison rather than a new random draw at each
budget.  Extra frames are chosen by farthest-point sampling in album index
space, covering the leftover timeline rather than clustering around the
markers.

Uses the same seed and spar32 sampling spec as ``build_teacher_keyframe_sweep.py``
so the 1,000 questions match the key-vs-full probe.

    python3 scripts/opsd/build_teacher_key_plus_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_teacher_keyframe_sweep import (  # noqa: E402
    ARMS,
    build_row,
    is_eligible,
    key_body_and_style,
    key_view_indices,
    select,
)
from mvopsd import sources as src  # noqa: E402

EXTRA_COUNTS = tuple(range(0, 11))
MAX_EXTRA = EXTRA_COUNTS[-1]


def remaining_views(record: dict) -> list[int]:
    required = set(record["required_views"])
    return [index for index in range(record["n_views"]) if index not in required]


def can_add_extras(record: dict, max_extra: int = MAX_EXTRA) -> bool:
    return is_eligible(record) and len(remaining_views(record)) >= max_extra


def extra_order(required: list[int], remaining: list[int], k: int) -> list[int]:
    """Farthest-point extras in album-index space, nested by prefix."""
    if k <= 0:
        return []
    selected = list(required)
    pool = list(remaining)
    extras: list[int] = []
    while len(extras) < k and pool:
        pick = max(pool, key=lambda frame: min(abs(frame - other) for other in selected))
        extras.append(pick)
        selected.append(pick)
        pool.remove(pick)
    return extras


def views_for_extra(record: dict, extra_n: int) -> list[int]:
    required = key_view_indices(record)
    extras = extra_order(required, remaining_views(record), extra_n)
    return sorted(required + extras)


def build_curve(record: dict, cache_root: str, group: str, extras: tuple[int, ...] = EXTRA_COUNTS) -> list[dict]:
    rows = []
    for extra_n in extras:
        view_indices = views_for_extra(record, extra_n)
        style, body = key_body_and_style(record, view_indices)
        arm = f"key+{extra_n}"
        row = build_row(record, view_indices, cache_root, arm, group, body, style)
        row["extra_info"]["extra_added"] = extra_n
        row["extra_info"]["n_key_frames"] = len(key_view_indices(record))
        row["extra_info"]["view_selection"] = "oracle_key_plus_farthest_extra"
        row["extra_info"]["index"] = f"{group}@{arm}"
        row["extra_info"]["sample_id"] = f"{group}@{arm}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="data/mvopsd/plan")
    parser.add_argument("--cache-root", default="data/mvopsd/views")
    parser.add_argument("--out", default="data/mvopsd/parquet_key_plus_sweep")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--arm", default="spar32")
    parser.add_argument("--max-extra", type=int, default=MAX_EXTRA)
    args = parser.parse_args()

    extras = tuple(range(0, args.max_extra + 1))
    spec = ARMS[args.arm]
    plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
    cache_root = os.path.join(src.REPO_ROOT, args.cache_root)
    out_dir = os.path.join(src.REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for position, line in enumerate(open(plan_path)):
        record = json.loads(line)
        record["plan_position"] = position
        records.append(record)
    print(f"read {len(records)} plan records")

    chosen = [record for record in select(records, spec, args.seed) if can_add_extras(record, args.max_extra)]
    if not chosen:
        raise SystemExit("no eligible questions with enough leftover frames")

    rows: list[dict] = []
    for record in chosen:
        group = f"{record['sample_id']}#{record['plan_position']}"
        curve = build_curve(record, cache_root, group, extras)
        if len(curve) != len(extras):
            raise AssertionError(f"{record['sample_id']}: expected {len(extras)} rows")
        rows.extend(curve)

    path = os.path.join(out_dir, f"sweep_{args.arm}_key_plus.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    groups = {row["extra_info"]["sweep_group"] for row in rows}
    by_extra = Counter(row["extra_info"]["extra_added"] for row in rows)
    by_type = Counter(row["extra_info"]["question_type"] for row in rows if row["extra_info"]["extra_added"] == 0)
    if len(groups) != len(chosen):
        raise AssertionError(f"{len(chosen)} questions collapsed into {len(groups)} groups")
    if len(set(by_extra.values())) != 1:
        raise AssertionError(f"unbalanced extra_added counts: {dict(by_extra)}")

    stats = {
        "seed": args.seed,
        "plan": args.plan,
        "arm": args.arm,
        "max_extra": args.max_extra,
        "questions": len(chosen),
        "rows": len(rows),
        "by_extra_added": {str(k): v for k, v in sorted(by_extra.items())},
        "by_question_type": dict(by_type),
        "mean_key_frames": float(np.mean([len(key_view_indices(r)) for r in chosen])),
        "parquet": os.path.relpath(path, src.REPO_ROOT),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(
        f"\n=== {args.arm} key+0..{args.max_extra} ===\n"
        f"  questions {len(chosen)}\n"
        f"  rows      {len(rows)}\n"
        f"  extras    {dict(by_extra)}\n"
        f"  by type   {dict(by_type)}\n"
        f"  -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
