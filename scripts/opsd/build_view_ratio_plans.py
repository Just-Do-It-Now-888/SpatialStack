#!/usr/bin/env python3
"""Re-derive the student's view subset as a fixed fraction of the teacher's album.

The 2026-08-20 experiment fixes the student budget at N/2 and N/4 instead of
drawing K from {1,2,4}. Two things motivate the change:

* the old K draw was confounded with the data source -- spar_3view has N=3, so it
  can only ever produce K in {1,2}, which makes "few views" and "this source"
  the same variable in any aggregate read (see the per-K JSD regression in
  tools/jsd_by_k_regression.py, where source composition explains as much
  variance as K does);
* a fixed fraction makes the teacher/student gap a single constant per arm (2x
  or 4x) rather than a per-sample mixture ranging from 1.5x to 32x.

Only albums with N >= 8 survive, so that N/4 is still at least 2 views and the
gap is a real one. That drops spar_3view entirely, which is the point.

The two arms are generated together because their subsets are **nested**: the
N/4 views are a subset of the N/2 views, so the only difference between the arms
is the views that were added, not which views happened to be drawn.

    python3 scripts/opsd/build_view_ratio_plans.py

Pixels are untouched: every plan record already lists all N cached views, and
only `view_indices` (and the `Frame-N` renumbering that follows from it) changes.
Stage 2 does not need to run again. Feed the output to write_mvopsd_parquet.py.
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

ARMS = {"half": 2, "quarter": 4}


def nested_subsets(record: dict, rng: random.Random, divisors: dict[str, int]) -> dict[str, list[int]]:
    """One shuffled draw per record, sliced into nested subsets by budget.

    Drawing once and slicing is what makes the arms nested: the smallest budget
    takes the first views of the shuffled pool, the next budget keeps those and
    extends. Required views (SPAR marker frames, the BEV anchor) are pinned into
    every budget first, since dropping them changes the question.
    """
    n_views = record["n_views"]
    required = sorted(set(record["required_views"]))
    budgets = sorted({max(1, n_views // divisor) for divisor in divisors.values()})
    if budgets[0] < len(required):
        raise ValueError(
            f"{record['sample_id']}: smallest budget {budgets[0]} cannot hold "
            f"{len(required)} required views"
        )

    pool = [view for view in range(n_views) if view not in set(required)]
    rng.shuffle(pool)

    subsets = {}
    for name, divisor in divisors.items():
        budget = max(1, n_views // divisor)
        filler = pool[: budget - len(required)]
        subsets[name] = sorted(set(required) | set(filler))
        if len(subsets[name]) != budget:
            raise ValueError(f"{record['sample_id']}: expected {budget} views, got {len(subsets[name])}")
    return subsets


def retarget(record: dict, view_indices: list[int]) -> dict:
    """Copy the record with a new student subset and a matching question body."""
    out = dict(record)
    out["view_indices"] = view_indices
    out["k_views"] = len(view_indices)
    renumber = {old: new for new, old in enumerate(view_indices)}
    body = record["body"]
    # renumber_frame_refs raises if the body cites a view the student was not
    # given; required_views already contains every cited view, so a raise here
    # means the plan and the body disagree and must not be silently written out.
    out["student_body"] = vw.renumber_frame_refs(body, renumber) if vw.referenced_views(body) else body
    return out


def summarize(records: list[dict], arm: str, divisor: int) -> dict:
    k_values = [record["k_views"] for record in records]
    n_values = [record["n_views"] for record in records]
    ratios = [record["n_views"] / record["k_views"] for record in records]
    return {
        "arm": arm,
        "divisor": divisor,
        "total_samples": len(records),
        "kept_by_source": dict(Counter(record["source"] for record in records).most_common()),
        "kept_by_k": {str(key): value for key, value in sorted(Counter(k_values).items())},
        "kept_by_n": {str(key): value for key, value in sorted(Counter(n_values).items())},
        "mean_k": sum(k_values) / len(k_values),
        "mean_n": sum(n_values) / len(n_values),
        "mean_ratio": sum(ratios) / len(ratios),
        "student_visual_tokens_mean": sum(k_values) / len(k_values) * 192,
        "teacher_visual_tokens_mean": sum(n_values) / len(n_values) * 192,
        "max_student_views": max(k_values),
        "max_teacher_views": max(n_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/mvopsd/plan", help="source plan directory")
    parser.add_argument("--out-prefix", default="data/mvopsd/plan_k")
    parser.add_argument("--min-teacher-views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
    records = [json.loads(line) for line in open(plan_path)]
    print(f"read {len(records)} records from {plan_path}")

    kept = [record for record in records if record["n_views"] >= args.min_teacher_views]
    dropped = Counter(
        record["source"] for record in records if record["n_views"] < args.min_teacher_views
    )
    print(f"teacher album >= {args.min_teacher_views} views: {len(kept)} kept, {len(records) - len(kept)} dropped")
    for source, count in dropped.most_common():
        print(f"  dropped {source:<18} {count:>7}")

    # One RNG for the whole pass, advanced per record, so the draw is reproducible
    # from the seed alone and identical across both arms.
    rng = random.Random(args.seed)
    per_arm: dict[str, list[dict]] = {name: [] for name in ARMS}
    for record in kept:
        subsets = nested_subsets(record, rng, ARMS)
        smaller, larger = subsets["quarter"], subsets["half"]
        if not set(smaller).issubset(set(larger)):
            raise AssertionError(f"{record['sample_id']}: quarter subset is not nested inside half")
        for name, view_indices in subsets.items():
            per_arm[name].append(retarget(record, view_indices))

    for name, arm_records in per_arm.items():
        out_dir = os.path.join(src.REPO_ROOT, f"{args.out_prefix}_{name}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "plan.jsonl"), "w") as handle:
            for record in arm_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats = summarize(arm_records, name, ARMS[name])
        stats["seed"] = args.seed
        stats["source_plan"] = args.plan
        stats["min_teacher_views"] = args.min_teacher_views
        with open(os.path.join(out_dir, "stats.json"), "w") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)

        print(f"\n=== arm {name} (student = N/{ARMS[name]}) -> {out_dir} ===")
        print(f"  samples          {stats['total_samples']}")
        print(f"  K distribution   {stats['kept_by_k']}")
        print(f"  mean K           {stats['mean_k']:.2f}  ({stats['student_visual_tokens_mean']:.0f} visual tokens)")
        print(f"  mean N           {stats['mean_n']:.2f}  ({stats['teacher_visual_tokens_mean']:.0f} visual tokens)")
        print(f"  mean N/K         {stats['mean_ratio']:.2f}")
        print(f"  max K / max N    {stats['max_student_views']} / {stats['max_teacher_views']}")
        for source, count in stats["kept_by_source"].items():
            print(f"    {source:<18} {count:>7}  {100 * count / stats['total_samples']:5.1f}%")


if __name__ == "__main__":
    main()
