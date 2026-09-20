#!/usr/bin/env python3
"""Re-derive one plan per data source, with a capped teacher album and K ~ U{menu}.

The 2026-08-21 round trains one dataset at a time. Mixing them was hiding two
things at once: the teacher reliability audit measured 57.7% on llava_hound and
19.9% on spar_32view (report section 2.2), so a mixed pool averages a supervision
signal that differs by a factor of three between its halves, and every curve read
off it is a weighted sum whose weights nobody chose.

Two knobs move relative to `build_view_ratio_plans.py`:

* the teacher album is capped at `--max-teacher-views` (8) instead of being all N.
  The view sweep found 1->5 views buys +9.1 points on vlm3r and 8->32 buys little
  on spar32 while costing 4x the visual tokens (report section 3), so 8 is where
  the budget stops paying for itself.
* the student budget is drawn from `--k-menu` (2,3,4,5) per sample instead of
  being a fixed fraction of N. A fixed fraction ties K to the source, which is
  what made the earlier per-K reads unusable; a fixed menu makes K vary within a
  single source, which is the only way it can be read as K.

Pixels are untouched -- every view this writes was already materialised by stage
2, and only which of them each side sees changes. Stage 2 does not re-run.

    python3 scripts/opsd/build_single_source_plans.py --source llava_hound_64k
    python3 scripts/opsd/build_single_source_plans.py --source vlm3r_scannet

Feed the output directory to write_mvopsd_parquet.py --arms main.
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


class Excluded(Exception):
    """This sample cannot satisfy the requested budgets."""


def cap_teacher_album(record: dict, max_views: int, rng: random.Random) -> tuple[list[int], str]:
    """The teacher's view indices and the body renumbered to match them.

    Returns all N views unchanged when the album already fits, which is the case
    for every source except spar_32view. When it does not fit, the views the
    question names are pinned first: dropping a cited view would leave the
    teacher reading about a frame it was not shown, and renumber_frame_refs
    raises rather than emitting a dangling reference.
    """
    n_views = record["n_views"]
    if n_views <= max_views:
        return list(range(n_views)), record["body"]

    required = sorted(set(record["required_views"]))
    if len(required) > max_views:
        raise Excluded("required_views_exceed_teacher_cap")
    pool = [view for view in range(n_views) if view not in set(required)]
    rng.shuffle(pool)
    teacher_views = sorted(set(required) | set(pool[: max_views - len(required)]))

    body = record["body"]
    if vw.referenced_views(body):
        renumber = {old: new for new, old in enumerate(teacher_views)}
        body = vw.renumber_frame_refs(body, renumber)
    return teacher_views, body


def draw_student_views(
    teacher_views: list[int],
    required: set[int],
    k_menu: list[int],
    rng: random.Random,
) -> list[int]:
    """K views out of the teacher's album, K drawn uniformly from the legal menu.

    K is capped below the teacher's album size, not below N: with the album
    truncated to 8 the privilege gap is 8/K, and a student that saw all 8 would
    have no gap left to close.
    """
    options = [k for k in k_menu if max(1, len(required)) <= k < len(teacher_views)]
    if not options:
        raise Excluded("no_legal_k_for_this_album")
    budget = rng.choice(options)
    filler_pool = [view for view in teacher_views if view not in required]
    filler = rng.sample(filler_pool, budget - len(required))
    return sorted(required | set(filler))


def retarget(record: dict, max_views: int, k_menu: list[int], rng: random.Random) -> dict:
    teacher_views, teacher_body = cap_teacher_album(record, max_views, rng)
    required = set(record["required_views"])
    if not required.issubset(set(teacher_views)):
        raise AssertionError(f"{record['sample_id']}: required views lost while capping the album")

    student_views = draw_student_views(teacher_views, required, k_menu, rng)

    out = dict(record)
    # Indices into record["frames"], the same space view_indices already used, so
    # write_mvopsd_parquet resolves both sides against one cache listing.
    out["teacher_view_indices"] = teacher_views
    out["teacher_body"] = teacher_body
    out["view_indices"] = student_views
    out["k_views"] = len(student_views)
    out["n_views_teacher_effective"] = len(teacher_views)

    renumber = {old: new for new, old in enumerate(student_views)}
    out["student_body"] = (
        vw.renumber_frame_refs(record["body"], renumber)
        if vw.referenced_views(record["body"])
        else record["body"]
    )
    return out


def summarize(records: list[dict], source: str, max_views: int, k_menu: list[int]) -> dict:
    k_values = [record["k_views"] for record in records]
    n_values = [record["n_views_teacher_effective"] for record in records]
    ratios = [n / k for n, k in zip(n_values, k_values)]
    return {
        "source": source,
        "max_teacher_views": max_views,
        "k_menu": k_menu,
        "total_samples": len(records),
        "kept_by_k": {str(key): value for key, value in sorted(Counter(k_values).items())},
        "kept_by_teacher_views": {str(key): value for key, value in sorted(Counter(n_values).items())},
        "truncated_albums": sum(1 for r in records if r["n_views_teacher_effective"] < r["n_views"]),
        "mean_k": sum(k_values) / len(k_values),
        "mean_teacher_views": sum(n_values) / len(n_values),
        "mean_ratio": sum(ratios) / len(ratios),
        "student_visual_tokens_mean": sum(k_values) / len(k_values) * 192,
        "teacher_visual_tokens_mean": sum(n_values) / len(n_values) * 192,
        "teacher_visual_tokens_max": max(n_values) * 192,
        "question_types": dict(Counter(r["question_type"] or "" for r in records).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/mvopsd/plan", help="source plan directory")
    parser.add_argument("--source", required=True, help="the one data_source to keep")
    parser.add_argument("--out", default=None, help="defaults to data/mvopsd/plan_single_<source>")
    parser.add_argument("--max-teacher-views", type=int, default=8)
    parser.add_argument("--k-menu", default="2,3,4,5")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    k_menu = [int(value) for value in args.k_menu.split(",")]
    out_dir = os.path.join(
        src.REPO_ROOT, args.out or f"data/mvopsd/plan_single_{args.source}"
    )

    plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
    records = [json.loads(line) for line in open(plan_path)]
    available = Counter(record["source"] for record in records)
    if args.source not in available:
        raise SystemExit(f"unknown source {args.source!r}; plan holds {dict(available)}")
    mine = [record for record in records if record["source"] == args.source]
    print(f"read {len(records)} records; {len(mine)} from {args.source}")

    rng = random.Random(args.seed)
    kept, excluded = [], Counter()
    for record in mine:
        try:
            kept.append(retarget(record, args.max_teacher_views, k_menu, rng))
        except Excluded as reason:
            excluded[str(reason)] += 1
    if not kept:
        raise SystemExit(f"every sample was excluded: {dict(excluded)}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.jsonl"), "w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = summarize(kept, args.source, args.max_teacher_views, k_menu)
    stats.update({"seed": args.seed, "source_plan": args.plan, "excluded": dict(excluded)})
    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(f"\n=== {args.source} -> {out_dir} ===")
    print(f"  samples            {stats['total_samples']}  (excluded {sum(excluded.values())}: {dict(excluded)})")
    print(f"  K distribution     {stats['kept_by_k']}")
    print(f"  teacher views      {stats['kept_by_teacher_views']}  (truncated {stats['truncated_albums']})")
    print(f"  mean K             {stats['mean_k']:.2f}  ({stats['student_visual_tokens_mean']:.0f} visual tokens)")
    print(f"  mean teacher N     {stats['mean_teacher_views']:.2f}  ({stats['teacher_visual_tokens_mean']:.0f} visual tokens)")
    print(f"  mean N/K           {stats['mean_ratio']:.2f}")
    print(f"  peak teacher       {stats['teacher_visual_tokens_max']} visual tokens")


if __name__ == "__main__":
    main()
