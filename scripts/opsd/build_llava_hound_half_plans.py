#!/usr/bin/env python3
"""LLaVA-Hound MV-OPSD plan: Teacher all frames, Student half (no oracle keys).

K = n_views // 2. Student indices are linspace(0, n, K, endpoint=False), so an
8-view album becomes [0, 2, 4, 6]. Albums with n_views < 2 are skipped.

    python3 scripts/opsd/build_llava_hound_half_plans.py

Feed the output directory to write_mvopsd_parquet.py --arms main.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402

SOURCE = "llava_hound_64k"
DEFAULT_PLAN = "data/mvopsd/plan_single_llava_hound_64k"


class Excluded(Exception):
    """This sample cannot satisfy the requested budgets."""


def student_views_for(record: dict) -> list[int]:
    n_views = int(record["n_views"])
    if n_views < 2:
        raise Excluded("n_views_lt_2")
    budget = n_views // 2
    if budget < 1:
        raise Excluded("student_budget_lt_1")
    # linspace(0, n, K, endpoint=False): n=8 -> [0, 2, 4, 6]
    chosen = sorted({int(i * n_views / budget) for i in range(budget)})
    if len(chosen) != budget:
        raise Excluded(f"student_budget_mismatch:{len(chosen)}")
    if not set(chosen).issubset(set(range(n_views))):
        raise Excluded("student_views_outside_album")
    return chosen


def retarget(record: dict, student_views: list[int]) -> dict:
    n_views = record["n_views"]
    teacher_views = list(range(n_views))
    out = dict(record)
    out["teacher_view_indices"] = teacher_views
    out["teacher_body"] = record["body"]
    out["n_views_teacher_effective"] = n_views
    out["view_indices"] = student_views
    out["k_views"] = len(student_views)
    out["view_selection"] = "linspace_half"
    out["privilege_bucket"] = "half"
    body = record["body"]
    out["student_body"] = (
        vw.renumber_frame_refs(body, {old: new for new, old in enumerate(student_views)})
        if vw.referenced_views(body)
        else body
    )
    dangling = vw.referenced_views(out["student_body"]) - set(range(len(student_views)))
    if dangling:
        raise AssertionError(f"{record['sample_id']}: dangling student Frame refs {dangling}")
    return out


def summarize(records: list[dict]) -> dict:
    k_values = [record["k_views"] for record in records]
    n_values = [record["n_views_teacher_effective"] for record in records]
    return {
        "total_samples": len(records),
        "by_source": dict(Counter(record["source"] for record in records)),
        "kept_by_k": {str(key): value for key, value in sorted(Counter(k_values).items())},
        "kept_by_teacher_views": {str(key): value for key, value in sorted(Counter(n_values).items())},
        "mean_k": sum(k_values) / len(k_values),
        "mean_teacher_views": sum(n_values) / len(n_values),
        "mean_ratio": sum(n / k for n, k in zip(n_values, k_values)) / len(records),
        "unique_scenes": len({record["scene_id"] for record in records}),
        "question_types": dict(Counter(record["question_type"] or "" for record in records).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--out", default="data/mvopsd/plan_llava_hound_half")
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
    skipped = Counter()
    kept: list[dict] = []
    n_in = 0
    for line in open(plan_path):
        record = json.loads(line)
        n_in += 1
        if record.get("source") != SOURCE:
            skipped[f"source:{record.get('source')}"] += 1
            continue
        try:
            student_views = student_views_for(record)
        except Excluded as exc:
            skipped[str(exc)] += 1
            continue
        kept.append(retarget(record, student_views))
    print(f"read {n_in} from {plan_path}")

    if not kept:
        raise SystemExit(f"no samples kept; skipped {dict(skipped)}")

    out_dir = os.path.join(src.REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.jsonl"), "w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = summarize(kept)
    stats.update({"seed": args.seed, "skipped": dict(skipped), "source_plan": args.plan})
    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(f"\n=== {out_dir} ===")
    print(f"  samples          {stats['total_samples']}  skipped {sum(skipped.values())}: {dict(skipped)}")
    print(f"  K distribution   {stats['kept_by_k']}")
    print(f"  teacher views    {stats['kept_by_teacher_views']}")
    print(f"  mean N/K         {stats['mean_ratio']:.2f}")


if __name__ == "__main__":
    main()
