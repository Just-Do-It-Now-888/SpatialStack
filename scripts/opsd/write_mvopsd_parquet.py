#!/usr/bin/env python3
"""Stage 3 of the MV-OPSD data pipeline: turn the plan into verl parquet files.

Two arms come out of the same plan, differing only in the teacher's view set:

* ``main``        - teacher sees all N views, student sees its K.
* ``noprivilege`` - teacher sees exactly the student's K views.

Because both arms share the sampling draw, the K draw and the view subsets,
their comparison isolates the multi-view privilege from on-policy
self-distillation itself. That is the v0 go/no-go criterion.

    python scripts/opsd/write_mvopsd_parquet.py --plan data/mvopsd/plan

Schema notes that are easy to get wrong:
* ``teacher_prompt`` content stays a plain string; ``_build_teacher_messages_from_template``
  skips list content outright (`ray_trainer.py:766`).
* the number of ``<image>`` placeholders must equal the image list length on both
  sides; both paths assert it.
* images are referenced by path, never inlined, to keep the parquet small.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402
from mvopsd.geometry import visual_tokens  # noqa: E402


def build_row(record: dict, cache_root: str, arm: str) -> dict:
    student_views = record["view_indices"]
    teacher_views = student_views if arm == "noprivilege" else list(range(record["n_views"]))

    student_prompt = vw.build_prompt(record["header_style"], len(student_views), record["student_body"])
    if arm == "noprivilege":
        teacher_prompt = student_prompt
    else:
        teacher_prompt = vw.build_prompt(record["header_style"], len(teacher_views), record["body"])

    def paths(view_indices):
        return [{"path": os.path.join(cache_root, record["frames"][i]["cache"])} for i in view_indices]

    student_images = paths(student_views)
    teacher_images = paths(teacher_views)

    if student_prompt.count(vw.IMAGE_TOKEN) != len(student_images):
        raise AssertionError(f"{record['sample_id']}: student placeholder/image mismatch")
    if teacher_prompt.count(vw.IMAGE_TOKEN) != len(teacher_images):
        raise AssertionError(f"{record['sample_id']}: teacher placeholder/image mismatch")

    return {
        "data_source": record["source"],
        "prompt": [{"role": "user", "content": student_prompt}],
        "teacher_prompt": [{"role": "user", "content": teacher_prompt}],
        "images": student_images,
        "teacher_images": teacher_images,
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": record["answer"]},
        "extra_info": {
            "index": record["sample_id"],
            "sample_id": record["sample_id"],
            "answer": record["answer"],
            "source": record["source"],
            "dataset": record["dataset"],
            "subset": record["subset"],
            "scene_id": record["scene_id"],
            "question_type": record["question_type"] or "",
            "n_views_teacher": len(teacher_views),
            "k_views_student": len(student_views),
            "view_indices_student": student_views,
            "view_selection": record["view_selection"],
            "privilege_bucket": "none" if arm == "noprivilege" else record["privilege_bucket"],
            "answer_view_sensitive": bool(record["answer_view_sensitive"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/mvopsd/plan")
    parser.add_argument("--cache-root", default="data/mvopsd/views")
    parser.add_argument("--out", default="data/mvopsd/parquet")
    parser.add_argument("--arms", default="main,noprivilege")
    parser.add_argument("--holdout-per-source", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-files", action="store_true", help="verify every referenced view exists")
    parser.add_argument("--per-view-tokens", type=int, default=192)
    parser.add_argument("--max-teacher-visual-tokens", type=int, default=8192)
    args = parser.parse_args()

    plan_dir = os.path.join(src.REPO_ROOT, args.plan)
    cache_root = os.path.join(src.REPO_ROOT, args.cache_root)
    out_dir = os.path.join(src.REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    records = [json.loads(line) for line in open(os.path.join(plan_dir, "plan.jsonl"))]
    if args.limit:
        records = records[: args.limit]

    # Hold out whole scenes, not samples: the same scene appears in many QAs.
    holdout_scenes: dict[str, set[str]] = {}
    if args.holdout_per_source:
        seen: dict[str, list[str]] = {}
        for record in records:
            seen.setdefault(record["source"], [])
            if record["scene_id"] not in seen[record["source"]]:
                seen[record["source"]].append(record["scene_id"])
        for source, scenes in seen.items():
            holdout_scenes[source] = set(scenes[: max(1, len(scenes) // 20)])

    train, holdout = [], []
    for record in records:
        if record["scene_id"] in holdout_scenes.get(record["source"], set()):
            holdout.append(record)
        else:
            train.append(record)

    if args.holdout_per_source:
        capped, per_source = [], Counter()
        for record in holdout:
            if per_source[record["source"]] < args.holdout_per_source:
                per_source[record["source"]] += 1
                capped.append(record)
            else:
                train.append(record)
        holdout = capped

    if args.check_files:
        verify_views(records, cache_root, args.max_teacher_visual_tokens)

    for arm in args.arms.split(","):
        for split, split_records in (("train", train), ("holdout", holdout)):
            if not split_records:
                continue
            rows = [build_row(record, cache_root, arm) for record in split_records]
            path = os.path.join(out_dir, f"{arm}_{split}.parquet")
            pd.DataFrame(rows).to_parquet(path, index=False)
            size_mb = os.path.getsize(path) / 1e6
            print(f"{path}: {len(rows)} rows, {size_mb:.1f} MB")

    report(train, holdout, args.per_view_tokens)


def verify_views(records, cache_root: str, max_teacher_visual_tokens: int) -> None:
    """Every view exists, is patch-aligned, and the teacher stays inside its context."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from mvopsd.geometry import assert_aligned

    wanted = sorted({os.path.join(cache_root, frame["cache"]) for record in records for frame in record["frames"]})

    def measure(path: str):
        try:
            with Image.open(path) as handle:
                size = handle.size
        except Exception as exc:
            return path, None, f"{type(exc).__name__}: {exc}"
        try:
            assert_aligned(size)
        except AssertionError as exc:
            return path, None, str(exc)
        return path, visual_tokens(size), None

    tokens_by_path, problems = {}, []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for path, tokens, problem in pool.map(measure, wanted):
            if problem:
                problems.append(f"{path}: {problem}")
            else:
                tokens_by_path[path] = tokens
    print(f"view files referenced: {len(wanted)}, unreadable or misaligned: {len(problems)}")
    if problems:
        for line in problems[:10]:
            print(f"  {line}")
        raise SystemExit("refusing to write parquet with bad views; rerun stage 2")

    worst = Counter()
    peak = 0
    for record in records:
        total = sum(tokens_by_path[os.path.join(cache_root, frame["cache"])] for frame in record["frames"])
        peak = max(peak, total)
        if total > max_teacher_visual_tokens:
            worst[record["source"]] += 1
    print(f"peak teacher visual tokens: {peak} (limit {max_teacher_visual_tokens})")
    if worst:
        raise SystemExit(
            f"teacher visual tokens exceed the limit for {dict(worst)}; "
            "raise self_distillation.max_reprompt_len or lower N before writing"
        )


def report(train, holdout, per_view_tokens: int) -> None:
    print("\n=== train split ===")
    total = len(train)
    for source, count in Counter(record["source"] for record in train).most_common():
        print(f"  {source:<18} {count:>7}  {100 * count / total:5.1f}%")
    print(f"  {'TOTAL':<18} {total:>7}")
    print(f"  holdout: {len(holdout)} samples")

    teacher_tokens = sum(record["n_views"] for record in train) / total * per_view_tokens
    student_tokens = sum(record["k_views"] for record in train) / total * per_view_tokens
    print(f"\nmean visual tokens/sample: student {student_tokens:.0f}, teacher {teacher_tokens:.0f}")
    print(f"max teacher visual tokens: {max(r['n_views'] for r in train) * per_view_tokens}")
    print(f"expected grid_thw per view: {visual_tokens((512, 384))} tokens at 512x384")


if __name__ == "__main__":
    main()
