#!/usr/bin/env python3
"""Build a paired teacher sweep: oracle key frames vs the full album.

The 20260820 view sweep varied budget with *evenly spaced* frames and excluded
any sample whose question cites a specific frame.  This builder holds the
question fixed and compares two teacher inputs on the same sample:

* ``full`` -- every frame in the album (``0 .. N-1``), prompt unchanged
* ``key``  -- only ``required_views`` (marked frames, body Frame-N refs, anchor)

    python3 scripts/opsd/build_teacher_keyframe_sweep.py

Writes one parquet per arm source plus ``stats.json``.
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
    "spar32": {
        "source": "spar_32view",
        "n_views": 32,
        "samples": 1000,
        "per_type": 250,
        "question_types": [
            "distance_prediction_oo_video",
            "spatial_imagination_oc_video",
            "spatial_imagination_oo_video",
            "distance_infer_center_oo_video",
        ],
    },
    "spar3": {
        "source": "spar_3view",
        "n_views": 3,
        "samples": 500,
        "per_type": 0,
        "question_types": None,
    },
}

SWEEP_ARMS = ("full", "key")


def is_eligible(record: dict) -> bool:
    required = record.get("required_views") or []
    n_views = record["n_views"]
    if not required or len(required) >= n_views:
        return False
    return all(0 <= view < n_views for view in required)


def key_view_indices(record: dict) -> list[int]:
    return sorted(set(record["required_views"]))


def key_body_and_style(record: dict, view_indices: list[int]) -> tuple[str, str]:
    body = record["body"]
    mapping = {old: new for new, old in enumerate(view_indices)}
    if vw.referenced_views(body):
        body = vw.renumber_frame_refs(body, mapping)
        style = "frame_labels"
    elif record["header_style"] == "frame_labels":
        style = "frame_labels"
    else:
        style = record["header_style"]
    return style, body


def build_row(
    record: dict,
    view_indices: list[int],
    cache_root: str,
    sweep_arm: str,
    group: str,
    body: str,
    header_style: str,
) -> dict:
    prompt = vw.build_prompt(header_style, len(view_indices), body)
    images = [{"path": os.path.join(cache_root, record["frames"][i]["cache"])} for i in view_indices]
    if prompt.count(vw.IMAGE_TOKEN) != len(images):
        raise AssertionError(
            f"{record['sample_id']} ({sweep_arm}): {prompt.count(vw.IMAGE_TOKEN)} placeholders "
            f"but {len(images)} images"
        )

    required = sorted(set(record.get("required_views") or []))
    return {
        "data_source": record["source"],
        "prompt": [{"role": "user", "content": prompt}],
        "teacher_prompt": [{"role": "user", "content": prompt}],
        "images": images,
        "teacher_images": images,
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": record["answer"]},
        "extra_info": {
            "index": f"{group}@{sweep_arm}",
            "sample_id": f"{group}@{sweep_arm}",
            "sweep_group": group,
            "sweep_arm": sweep_arm,
            "sweep_views": len(view_indices),
            "required_views": required,
            "view_indices_teacher": view_indices,
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
            "view_selection": "oracle_keyframes" if sweep_arm == "key" else "full_album",
            "privilege_bucket": "none",
            "answer_view_sensitive": bool(record["answer_view_sensitive"]),
        },
    }


def build_pair(record: dict, cache_root: str, group: str) -> list[dict]:
    full_indices = list(range(record["n_views"]))
    key_indices = key_view_indices(record)
    key_style, key_body = key_body_and_style(record, key_indices)
    return [
        build_row(
            record,
            full_indices,
            cache_root,
            "full",
            group,
            record["body"],
            record["header_style"],
        ),
        build_row(record, key_indices, cache_root, "key", group, key_body, key_style),
    ]


def select(records: list[dict], spec: dict, seed: int) -> list[dict]:
    pool = [
        record
        for record in records
        if record["source"] == spec["source"]
        and record["n_views"] == spec["n_views"]
        and is_eligible(record)
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
    parser.add_argument("--out", default="data/mvopsd/parquet_keyframe_sweep")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--arms", default="spar32,spar3")
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
        key_view_counts: list[int] = []
        for record in chosen:
            group = f"{record['sample_id']}#{record['plan_position']}"
            pair = build_pair(record, cache_root, group)
            rows.extend(pair)
            key_view_counts.append(len(key_view_indices(record)))

        path = os.path.join(out_dir, f"sweep_{arm}_keyframe.parquet")
        pd.DataFrame(rows).to_parquet(path, index=False)
        by_type = Counter(row["extra_info"]["question_type"] for row in rows)
        by_sweep_arm = Counter(row["extra_info"]["sweep_arm"] for row in rows)
        groups = {row["extra_info"]["sweep_group"] for row in rows}
        if len(groups) != len(chosen):
            raise AssertionError(f"{arm}: {len(chosen)} questions collapsed into {len(groups)} sweep groups")
        if by_sweep_arm["full"] != by_sweep_arm["key"]:
            raise AssertionError(f"{arm}: unbalanced full/key rows: {dict(by_sweep_arm)}")

        stats["arms"][arm] = {
            "source": spec["source"],
            "questions": len(chosen),
            "sweep_groups": len(groups),
            "rows": len(rows),
            "rows_per_arm": dict(by_sweep_arm),
            "view_instances": sum(len(row["teacher_images"]) for row in rows),
            "mean_key_frames": float(np.mean(key_view_counts)) if key_view_counts else 0.0,
            "mean_full_frames": spec["n_views"],
            "by_question_type": dict(by_type),
            "parquet": os.path.relpath(path, src.REPO_ROOT),
        }
        print(
            f"\n=== arm {arm} ({spec['source']}) ===\n"
            f"  questions       {len(chosen)}\n"
            f"  rows            {len(rows)} ({dict(by_sweep_arm)})\n"
            f"  mean key frames {stats['arms'][arm]['mean_key_frames']:.2f} "
            f"(full {spec['n_views']})\n"
            f"  by type         {dict(by_type)}\n"
            f"  -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)"
        )

    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
