#!/usr/bin/env python3
"""SPAR full-pool MV-OPSD plan: Teacher 32 / Student 16 on spar_32view.

spar_3view albums only have three frames, so that half is Teacher=3 / Student=3
(keyframes are already inside the album). Do not reuse plan_spar_all: those
teacher albums were capped at 8.

Student 16 always contains oracle required_views. Leftover slots are filled by
farthest-point extras in album-index space (same extra_order as the 20260914
key-plus probe). Empty required_views pins frame 0 first.

    python3 scripts/opsd/build_spar_t32_s16_plans.py

Feed the output directory to write_mvopsd_parquet.py --arms main.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_teacher_key_plus_sweep import extra_order  # noqa: E402
from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402

STUDENT_BUDGET_32 = 16
SOURCE_PLANS = {
    "spar_3view": "data/mvopsd/plan_single_spar_3view",
    "spar_32view": "data/mvopsd/plan_single_spar_32view",
}


class Excluded(Exception):
    """This sample cannot satisfy the requested budgets."""


def key_frames(record: dict) -> list[int]:
    """Oracle keys; empty required_views pins the album origin."""
    required = sorted(set(record.get("required_views") or []))
    if not required:
        return [0]
    return required


def remaining_album(record: dict, keys: list[int]) -> list[int]:
    keyed = set(keys)
    return [index for index in range(record["n_views"]) if index not in keyed]


def student_views_for(record: dict) -> list[int]:
    n_views = record["n_views"]
    keys = key_frames(record)
    if not set(keys).issubset(set(range(n_views))):
        raise Excluded("required_views_outside_album")
    if n_views == 3:
        return list(range(3))
    if n_views != 32:
        raise Excluded(f"unexpected_n_views:{n_views}")
    if len(keys) > STUDENT_BUDGET_32:
        raise Excluded("required_views_exceed_student_16")
    extra_n = STUDENT_BUDGET_32 - len(keys)
    extras = extra_order(keys, remaining_album(record, keys), extra_n)
    chosen = sorted(set(keys) | set(extras))
    if len(chosen) != STUDENT_BUDGET_32:
        raise Excluded(f"student_budget_mismatch:{len(chosen)}")
    if not set(keys).issubset(set(chosen)):
        raise Excluded("dropped_keyframes")
    return chosen


def retarget(record: dict, student_views: list[int]) -> dict:
    """Full-N teacher album + student subset; rewrite Frame-N on the student side."""
    n_views = record["n_views"]
    teacher_views = list(range(n_views))
    out = dict(record)
    out["teacher_view_indices"] = teacher_views
    out["teacher_body"] = record["body"]
    out["n_views_teacher_effective"] = n_views
    out["view_indices"] = student_views
    out["k_views"] = len(student_views)
    if n_views == 32:
        out["view_selection"] = "oracle_key_plus_farthest_extra"
        out["privilege_bucket"] = "t32_s16"
    else:
        out["view_selection"] = "full_album"
        out["privilege_bucket"] = "t3_s3"
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


def load_plan(relpath: str) -> list[dict]:
    path = os.path.join(src.REPO_ROOT, relpath, "plan.jsonl")
    records = [json.loads(line) for line in open(path)]
    print(f"read {len(records)} from {path}")
    return records


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
    parser.add_argument("--out", default="data/mvopsd/plan_spar_t32_s16")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    skipped = Counter()
    kept: list[dict] = []
    for source, relpath in SOURCE_PLANS.items():
        for record in load_plan(relpath):
            if record.get("source") != source:
                skipped[f"source_mismatch:{record.get('source')}"] += 1
                continue
            try:
                student_views = student_views_for(record)
            except Excluded as exc:
                skipped[str(exc)] += 1
                continue
            kept.append(retarget(record, student_views))

    if not kept:
        raise SystemExit(f"no samples kept; skipped {dict(skipped)}")

    out_dir = os.path.join(src.REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.jsonl"), "w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = summarize(kept)
    stats.update({"seed": args.seed, "skipped": dict(skipped), "source_plans": SOURCE_PLANS})
    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(f"\n=== {out_dir} ===")
    print(f"  samples          {stats['total_samples']}  skipped {sum(skipped.values())}: {dict(skipped)}")
    print(f"  by source        {stats['by_source']}")
    print(f"  K distribution   {stats['kept_by_k']}")
    print(f"  teacher views    {stats['kept_by_teacher_views']}")
    print(f"  mean N/K         {stats['mean_ratio']:.2f}")


if __name__ == "__main__":
    main()
