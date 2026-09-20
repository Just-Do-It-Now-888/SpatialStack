#!/usr/bin/env python3
"""Nested SPAR 3-view student-K plans: teacher always sees 3, student sees 1 or 2.

The 2026-08-31 pair isolates student view count. SPAR 3-view albums are N=3, so
the only legal student budgets are K=1 and K=2. Most questions name two frames
(``required_views`` length 2); those cannot take K=1 without dropping a cited
view. This builder therefore keeps only the intersection that can support both
budgets -- otherwise the two arms would train on different questions and K
would be confounded with question type (the same trap as mixing sources).

The two arms are generated together so their subsets are nested: the K=1 views
are a subset of the K=2 views. Required views (marker frames, the BEV anchor)
are pinned into both budgets first.

    python3 scripts/opsd/build_spar3_student_k_plans.py

Pixels are untouched. Feed each output directory to write_mvopsd_parquet.py
``--arms main``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402

ARMS = ("k1", "k2")
BUDGETS = {"k1": 1, "k2": 2}


def nested_student_views(record: dict, rng: random.Random) -> dict[str, list[int]]:
    """Pin required views, then grow the student album from 1 to 2.

    Returns None when K=1 cannot hold every required view. Teacher indices stay
    the 3-view album already stored on the source plan.
    """
    teacher = list(record.get("teacher_view_indices") or range(record["n_views"]))
    required = sorted(set(record["required_views"]))
    if len(required) > BUDGETS["k1"]:
        return None
    if not set(required).issubset(set(teacher)):
        raise AssertionError(
            f"{record['sample_id']}: required {required} not inside teacher album {teacher}"
        )

    pool = [view for view in teacher if view not in set(required)]
    rng.shuffle(pool)
    ordered = required + pool
    subsets = {}
    for name, budget in BUDGETS.items():
        chosen = ordered[:budget]
        if len(chosen) != budget:
            raise AssertionError(
                f"{record['sample_id']}: expected {budget} student views, got {chosen}"
            )
        subsets[name] = sorted(chosen)
    if not set(subsets["k1"]).issubset(set(subsets["k2"])):
        raise AssertionError(f"{record['sample_id']}: K=1 is not nested inside K=2")
    return subsets


def retarget(record: dict, view_indices: list[int]) -> dict:
    """Copy the record with a new student subset and a matching question body."""
    out = dict(record)
    out["view_indices"] = view_indices
    out["k_views"] = len(view_indices)
    out["n_views_teacher_effective"] = len(out.get("teacher_view_indices") or range(out["n_views"]))
    body = record.get("teacher_body") or record["body"]
    out["student_body"] = (
        vw.renumber_frame_refs(body, {old: new for new, old in enumerate(view_indices)})
        if vw.referenced_views(body)
        else record["body"]
    )
    return out


def summarize(records: list[dict], arm: str) -> dict:
    k_values = [record["k_views"] for record in records]
    n_values = [record["n_views_teacher_effective"] for record in records]
    return {
        "arm": arm,
        "k_views": BUDGETS[arm],
        "total_samples": len(records),
        "kept_by_k": {str(key): value for key, value in sorted(Counter(k_values).items())},
        "kept_by_teacher_views": {str(key): value for key, value in sorted(Counter(n_values).items())},
        "question_types": dict(Counter(record["question_type"] or "" for record in records).most_common()),
        "mean_k": sum(k_values) / len(k_values),
        "mean_teacher_views": sum(n_values) / len(n_values),
        "mean_ratio": sum(n / k for n, k in zip(n_values, k_values)) / len(records),
        "unique_scenes": len({record["scene_id"] for record in records}),
        "unique_sample_ids": len({record["sample_id"] for record in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/mvopsd/plan_single_spar_3view")
    parser.add_argument("--out-prefix", default="data/mvopsd/plan_spar3")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
    records = [json.loads(line) for line in open(plan_path)]
    print(f"read {len(records)} records from {plan_path}")

    rng = random.Random(args.seed)
    per_arm: dict[str, list[dict]] = {name: [] for name in ARMS}
    skipped = Counter()
    for record in records:
        if record.get("source") != "spar_3view":
            skipped[f"source:{record.get('source')}"] += 1
            continue
        subsets = nested_student_views(record, rng)
        if subsets is None:
            skipped["required_views_exceed_k1"] += 1
            continue
        for name, view_indices in subsets.items():
            per_arm[name].append(retarget(record, view_indices))

    if not per_arm["k1"]:
        raise SystemExit(f"no legal K=1 samples; skipped {dict(skipped)}")
    if [r["sample_id"] for r in per_arm["k1"]] != [r["sample_id"] for r in per_arm["k2"]]:
        raise SystemExit("K=1 and K=2 plans drifted off the same sample order")

    for name, arm_records in per_arm.items():
        out_dir = os.path.join(src.REPO_ROOT, f"{args.out_prefix}_{name}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "plan.jsonl"), "w") as handle:
            for record in arm_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats = summarize(arm_records, name)
        stats.update({"seed": args.seed, "source_plan": args.plan, "skipped": dict(skipped)})
        with open(os.path.join(out_dir, "stats.json"), "w") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)

        print(f"\n=== arm {name} (student K={BUDGETS[name]}) -> {out_dir} ===")
        print(f"  samples          {stats['total_samples']}  (skipped {sum(skipped.values())}: {dict(skipped)})")
        print(f"  K distribution   {stats['kept_by_k']}")
        print(f"  teacher views    {stats['kept_by_teacher_views']}")
        print(f"  unique scenes    {stats['unique_scenes']}")
        print(f"  mean N/K         {stats['mean_ratio']:.2f}")
        for qtype, count in stats["question_types"].items():
            print(f"    {qtype:<40} {count:>5}")


if __name__ == "__main__":
    main()
