#!/usr/bin/env python3
"""Equal-view Answer-OPSD plan: student and teacher see the full album.

Unlike SPAR T32/S16 or Hound-half, this builder does not subset views and does
not renumber Frame-N. Privilege is the ground-truth answer on the teacher
prompt, not view count. SPAR source plans are the first arm; Hound / VLM3R
can be added later with the same retarget.

    python3 scripts/opsd/build_answer_opsd_full_views_plans.py

Feed the output directory to write_mvopsd_parquet.py --arms main --require-equal-views.
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

SOURCE_PLANS = {
    "spar_3view": "data/mvopsd/plan_single_spar_3view",
    "spar_32view": "data/mvopsd/plan_single_spar_32view",
}


class Excluded(Exception):
    """This sample cannot use a full-album student/teacher pair."""


def retarget(record: dict) -> dict:
    """Full album on both sides; keep the original body numbering."""
    n_views = int(record["n_views"])
    if n_views < 1:
        raise Excluded("n_views_lt_1")
    if len(record["frames"]) != n_views:
        raise Excluded(f"frame_count_mismatch:{len(record['frames'])}!={n_views}")
    album = list(range(n_views))
    body = record["body"]
    dangling = vw.referenced_views(body) - set(album)
    if dangling:
        raise Excluded(f"dangling_frame_refs:{sorted(dangling)}")
    out = dict(record)
    out["teacher_view_indices"] = album
    out["view_indices"] = album
    out["k_views"] = n_views
    out["n_views_teacher_effective"] = n_views
    out["teacher_body"] = body
    out["student_body"] = body
    out["view_selection"] = "full_album"
    out["privilege_bucket"] = "answer_equal_views"
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
    parser.add_argument("--out", default="data/mvopsd/plan_answer_opsd_spar_fullviews")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    skipped = Counter()
    kept: list[dict] = []
    for source, relpath in SOURCE_PLANS.items():
        for record in load_plan(relpath):
            if record.get("source") != source:
                skipped[f"source_mismatch:{record.get('source')}"] += 1
                continue
            try:
                kept.append(retarget(record))
            except Excluded as exc:
                skipped[str(exc)] += 1

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
