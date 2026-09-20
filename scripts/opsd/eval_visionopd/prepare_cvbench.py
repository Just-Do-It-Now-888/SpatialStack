#!/usr/bin/env python3
"""Materialise CV-Bench in the Vision-OPD evaluation schema.

The lmms_eval path this replaces scored MV-OPSD v0 on a string artifact: with
``max_new_tokens=16`` every response was cut off before the answer, and
``extract_characters_regex`` then took the first ``[ABCDEF]`` anywhere in the
prose -- which is always the "B" in "**B**ased on the provided images". See
ISSUE-003 and LESSON-011. Vision-OPD instead lets the model finish (32768
tokens) and grades the finished text, so the prompt has to stop forcing a
bare-letter answer too.

Two deliberate departures from ``src/lmms_eval/tasks/cvbench``:

* the prompt is the dataset's own ``prompt`` field -- "... Select from the
  following choices.\\n(A) 3\\n(B) 2 ..." -- exactly as ``eval/cv_bench.json``
  in the Vision-OPD release. This drops both the ``These are frames of a
  video.`` prefix (false for single-image CV-Bench, and only ever present
  because of the falsy-``or`` bug in LESSON-010) and the
  ``Answer with the option's letter ... directly.`` suffix.
* the ground truth stays parenthesised, ``"(C)"``, because the Vision-OPD
  grader anchors on that form.

Numbers produced from this file are therefore NOT comparable with the
pre-2026-08-17 lmms_eval CV-Bench numbers. That is the point; those were not
measuring the model.

    python3 scripts/opsd/eval_visionopd/prepare_cvbench.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvopsd import sources as src  # noqa: E402


def build_item(doc: dict, image_path: str) -> dict:
    return {
        # infer.py keys its resume checkpoint on the first non-empty of
        # sample_uid/uid/index/question_id/id, so index must survive.
        "index": int(doc["idx"]),
        "question_id": int(doc["idx"]),
        "images": [image_path],
        "query": doc["prompt"],
        "response": doc["answer"],  # "(C)"
        "type": doc["type"],  # 2D / 3D
        "task": doc["task"],  # Count / Relation / Depth / Distance
        "source": doc["source"],  # ADE20K / COCO / Omni3D
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/datasets")
    parser.add_argument("--out-dir", default="data/eval/visionopd")
    parser.add_argument(
        "--image-dir",
        default="data/eval/cvbench_verl/images",
        help="reuses the PNGs build_cvbench_val_parquet.py already wrote, so the "
        "in-training validation set and this offline set read identical pixels",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="smoke builds; 0 keeps all 2638. Strided, not sliced: CV-Bench is "
        "ordered by subtask so a head slice is pure ADE20K/Count and never "
        "exercises the 2D/3D roll-up.",
    )
    args = parser.parse_args()

    import datasets

    out_dir = os.path.join(src.REPO_ROOT, args.out_dir)
    image_dir = os.path.join(src.REPO_ROOT, args.image_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    dataset = datasets.load_dataset(
        "nyu-visionx/CV-Bench",
        cache_dir=os.path.join(src.REPO_ROOT, args.cache_dir),
    )["test"]
    if args.limit and args.limit < len(dataset):
        stride = len(dataset) / args.limit
        dataset = dataset.select([int(i * stride) for i in range(args.limit)])

    items, written = [], 0
    for doc in dataset:
        image_path = os.path.join(image_dir, f"{int(doc['idx']):05d}.png")
        if not os.path.exists(image_path):
            doc["image"].convert("RGB").save(image_path, format="PNG")
            written += 1
        items.append(build_item(doc, image_path))

    suffix = f"_{args.limit}" if args.limit else ""
    path = os.path.join(out_dir, f"cvbench{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    print(f"{path}: {len(items)} items")
    print(f"images: {written} written, {len(items) - written} reused, in {image_dir}")

    print("\n=== composition ===")
    for key in ("type", "source", "task"):
        counts = Counter(item[key] for item in items)
        print(f"  {key:<8} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    gold = Counter(item["response"] for item in items)
    print("  gold     " + "  ".join(f"{k}={v}" for k, v in sorted(gold.items())))
    print(
        "\nA constant-B predictor scores 42.57 under the old lmms_eval roll-up.\n"
        "Any new number near that deserves the same scrutiny ISSUE-003 applied."
    )
    print(f"\nprompt sample:\n---\n{items[0]['query']}\n---")


if __name__ == "__main__":
    main()
