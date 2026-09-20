#!/usr/bin/env python3
"""Build the controlled view sweep: same question, same album, different budget.

The as-trained table cannot answer "does the teacher get better with more
views", because in the training pool the view count and the data source are the
same variable -- N=3 is always spar_3view and N=32 is always spar_32view.  This
script removes the confound by holding the sample fixed and varying only how
many of its views the teacher is shown.

Two arms, chosen because their questions cite no specific frame
(``required_views`` is empty), so an album can be truncated to any budget
without the question referring to a view that is no longer there:

* ``vlm3r``  -- vlm3r_scannet, N=8, multiple choice, budgets 1..8.  Fully
  rule-scorable, so this arm gives a clean curve with no parser risk.
* ``spar32`` -- spar_32view restricted to its four frame-agnostic question
  types (obj_count, room_size and the two ``*_hard`` imagination types),
  N=32, budgets 1..32.  This is the only arm that reaches 32 views.

Views at each budget are **evenly spaced over the album**, not a nested random
draw.  A budget of 4 out of 32 should be the best 4-view look at the scene,
which is what uniform temporal sampling gives and what every frame-budget
evaluation in this project already does (``vsibench_eval_core.sample_frames``).
Nested random draws would instead compare a good 32-view album against a bad
4-view one and report the difference as a view-count effect.

    python3 scripts/opsd/build_teacher_view_sweep.py

Writes one parquet per arm plus a stats.json describing the draw.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402

ARMS = {
    "vlm3r": {
        "source": "vlm3r_scannet",
        "n_views": 8,
        "budgets": [1, 2, 3, 4, 5, 6, 7, 8],
        "question_types": None,
        "per_type": 0,
        "samples": 2500,
    },
    "spar32": {
        "source": "spar_32view",
        "n_views": 32,
        "budgets": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "question_types": [
            "obj_count",
            "room_size",
            "spatial_imagination_oc_video_hard",
            "spatial_imagination_oo_video_hard",
        ],
        # Balanced by type so a 2,869-row type does not drown a 803-row one in
        # the curve; the per-type curves are the interesting read anyway.
        "per_type": 500,
        "samples": 0,
    },
}


def even_views(n_views: int, budget: int) -> list[int]:
    """``budget`` view indices spread evenly over the album, in album order."""
    if budget >= n_views:
        return list(range(n_views))
    return sorted(set(np.linspace(0, n_views - 1, budget).astype(int).tolist()))


def build_row(record: dict, view_indices: list[int], cache_root: str, arm: str, group: str) -> dict:
    body = record["body"]
    if vw.referenced_views(body):
        raise AssertionError(
            f"{record['sample_id']}: question cites Frame refs; the sweep pool must be frame-agnostic"
        )
    prompt = vw.build_prompt(record["header_style"], len(view_indices), body)
    images = [{"path": os.path.join(cache_root, record["frames"][i]["cache"])} for i in view_indices]
    if prompt.count(vw.IMAGE_TOKEN) != len(images):
        raise AssertionError(f"{record['sample_id']}: placeholder/image mismatch")

    return {
        "data_source": record["source"],
        # Both columns carry the same content: the sweep has no student side,
        # and keeping the teacher_* names lets the same harness read it.
        "prompt": [{"role": "user", "content": prompt}],
        "teacher_prompt": [{"role": "user", "content": prompt}],
        "images": images,
        "teacher_images": images,
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": record["answer"]},
        "extra_info": {
            "index": f"{group}@{len(view_indices)}",
            "sample_id": f"{group}@{len(view_indices)}",
            # The group key: every budget of one question shares it, so the
            # curve is a within-question comparison rather than a between-sample
            # one. It carries the plan position because ``sample_id`` is scene +
            # index and collides across question types -- 674 ids in the plan
            # name two different questions, and grouping on it would average
            # two curves into one.
            "sweep_group": group,
            "sweep_views": len(view_indices),
            "sweep_arm": arm,
            "answer": record["answer"],
            "source": record["source"],
            "dataset": record["dataset"],
            "subset": record["subset"],
            "scene_id": record["scene_id"],
            "question_type": record["question_type"] or "",
            "n_views_teacher": len(view_indices),
            "n_views_album": record["n_views"],
            "k_views_student": len(view_indices),
            "view_indices_student": view_indices,
            "view_selection": "even_spaced_sweep",
            "privilege_bucket": "none",
            "answer_view_sensitive": bool(record["answer_view_sensitive"]),
        },
    }


def select(records: list[dict], spec: dict, seed: int) -> list[dict]:
    pool = [
        record
        for record in records
        if record["source"] == spec["source"]
        and record["n_views"] == spec["n_views"]
        and not record["required_views"]
        and (spec["question_types"] is None or record["question_type"] in spec["question_types"])
    ]
    rng = np.random.default_rng(seed)
    if spec["per_type"]:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for record in pool:
            by_type[record["question_type"] or ""].append(record)
        chosen: list[dict] = []
        for qtype in sorted(by_type):
            group = by_type[qtype]
            take = min(spec["per_type"], len(group))
            index = rng.choice(len(group), size=take, replace=False)
            chosen.extend(group[i] for i in sorted(index))
        return chosen
    take = min(spec["samples"], len(pool))
    index = rng.choice(len(pool), size=take, replace=False)
    return [pool[i] for i in sorted(index)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="data/mvopsd/plan")
    parser.add_argument("--cache-root", default="data/mvopsd/views")
    parser.add_argument("--out", default="data/mvopsd/parquet_view_sweep")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--arms", default="vlm3r,spar32")
    args = parser.parse_args()

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

    stats = {"seed": args.seed, "plan": args.plan, "arms": {}}
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        chosen = select(records, spec, args.seed)
        rows: list[dict] = []
        for record in chosen:
            group = f"{record['sample_id']}#{record['plan_position']}"
            for budget in spec["budgets"]:
                view_indices = even_views(record["n_views"], budget)
                if len(view_indices) != budget:
                    raise AssertionError(f"{record['sample_id']}: wanted {budget} views, got {len(view_indices)}")
                rows.append(build_row(record, view_indices, cache_root, arm, group))

        path = os.path.join(out_dir, f"sweep_{arm}.parquet")
        pd.DataFrame(rows).to_parquet(path, index=False)
        images = sum(len(row["teacher_images"]) for row in rows)
        by_type = Counter(row["extra_info"]["question_type"] for row in rows)
        groups = {row["extra_info"]["sweep_group"] for row in rows}
        if len(groups) != len(chosen):
            raise AssertionError(f"{arm}: {len(chosen)} questions collapsed into {len(groups)} sweep groups")
        stats["arms"][arm] = {
            "source": spec["source"],
            "budgets": spec["budgets"],
            "questions": len(chosen),
            "sweep_groups": len(groups),
            "rows": len(rows),
            "view_instances": images,
            "by_question_type": dict(by_type),
            "parquet": os.path.relpath(path, src.REPO_ROOT),
        }
        print(
            f"\n=== arm {arm} ({spec['source']}, budgets {spec['budgets']}) ===\n"
            f"  questions      {len(chosen)}\n"
            f"  rows           {len(rows)}\n"
            f"  view instances {images}\n"
            f"  by type        {dict(by_type)}\n"
            f"  -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)"
        )

    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
