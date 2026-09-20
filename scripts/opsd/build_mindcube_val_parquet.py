#!/usr/bin/env python3
"""Materialise MindCube tinybench as a verl validation set, in two view budgets.

    # full view set -- the axis the 46.86 / 74.48 / 69.81 anchors live on
    python3 scripts/opsd/build_mindcube_val_parquet.py

    # single view -- the condition training actually targets
    python3 scripts/opsd/build_mindcube_val_parquet.py --views 1 --tag singleview

Both files must exist and both must be reported. A full-view gain is not evidence
that single-view spatial reasoning improved, and the reverse is equally true, so
the two series are never plotted as one line. The cautionary case is in-repo:
`20260831_mvopsd_spar3_student_k`'s K=1 arm gained +2.96 pp on its in-training
validation while the same checkpoint scored *below* base on CV-Bench (85.05 vs
88.07), VSI original (47.87 vs 52.75) and SPAR-Bench (13.87 vs 39.88).

The prompt is not rebuilt for the full-view file: tinybench ships the
`input_prompt` that produced the anchors, and it is copied verbatim under the
same system turn the offline script sends. The single-view file runs that prompt
through the same rewriter the training pool uses, so the val condition and the
train condition describe their view budget the same way.

Images point at the original MindCube jpgs rather than a cache. `fetch_image`
applies `smart_resize` with the min/max pixels written below, which is exactly
what the offline script's processor does to the same file, so a cache would only
add a JPEG round-trip between the two paths.

Read the result split by family *and* by view count, never pooled. Training is
74% four-view while tinybench is 41/33/26% four/three/two-view, and `rotation` is
trained mostly on four-view rows but evaluated entirely on three-view ones.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mindcube_scoring as mcs  # noqa: E402
from mvopsd import mindcube as mc  # noqa: E402
from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402

DEFAULT_ROWS = (
    "/home/c30084464/Documents/code/3DThinker/MindCube-main/data/prompts/general/"
    "MindCube_tinybench_raw_qa.jsonl"
)
DEFAULT_IMAGE_ROOT = "/home/c30084464/Documents/code/3DThinker/MindCube-main/data"

# scripts/mindcube/eval_mindcube_qwen35.py:213-214, which is where the anchors
# come from, and identical to the training pool's caps. verl has no min_pixels
# knob: it forwards whatever the parquet says to qwen_vl_utils.fetch_image.
MIN_PIXELS = 256 * 28 * 28  # 200704
MAX_PIXELS = 1605632

# Only anchor-free rows draw a view, so the seed matters for 26 of the 1,050 rows.
# Fixed anyway: the single-view series is read step over step, and a val set that
# reshuffled between builds would put noise on that comparison.
DEFAULT_SEED = 20260907

# data_qwen.py gives every SFT sample this system turn and the offline evaluation
# sends it too, so both val files carry it or the measurement is not the anchor's.
SYSTEM_PROMPT = "You are a helpful assistant."

# Full view set and single view get different prefixes so they cannot be summed
# by accident. The training pool's data_sources use underscores
# (`mindcube_among_4view`), so neither prefix can ever collide with a train row.
FULLVIEW_PREFIX = "mindcube"
SINGLEVIEW_PREFIX = "mindcube1v"


def build_row(row: dict, image_root: str, views: int, rng: random.Random) -> dict:
    images = row["images"]
    n_views = len(images)
    family = mcs.family_of(row["id"])

    if views == 0 or views == n_views:
        # Verbatim. Not "rewritten with K=N", because byte-equality with the
        # anchor's prompt is the point of this file.
        body = row["input_prompt"]
        view_indices = list(range(n_views))
        prefix, template, matches = FULLVIEW_PREFIX, "verbatim", True
    else:
        question = mc.parse_question(row["input_prompt"], n_views)
        view_indices = choose_views(question, views, rng)
        body = mc.rewrite_for_views(question, view_indices)
        prefix, template = SINGLEVIEW_PREFIX, question.template
        # False only for the pair template, which keeps its two-view wording
        # above a single image by protocol decision -- the same accepted
        # inconsistency the training pool carries for 1,302 rows.
        matches = question.rewritable

    paths = [os.path.join(image_root, images[index]) for index in view_indices]
    prompt = vw.IMAGE_TOKEN * len(paths) + body

    return {
        # Family and view count both go in the key: `mindcube_metrics` parses
        # them back out, and a pooled average would hide that `rotation` is
        # evaluated only on three-view rows.
        "data_source": f"{prefix}/{family}/{n_views}view",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "images": [
            {"path": path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS} for path in paths
        ],
        "ability": "spatial_reasoning",
        # `gt_answer` is already the bare letter `mindcube_process_results`
        # compares against, so no normalisation happens here.
        "reward_model": {"style": "rule", "ground_truth": row["gt_answer"]},
        "extra_info": {
            "index": row["id"],
            "id": row["id"],
            "answer": row["gt_answer"],
            "benchmark": "mindcube_tinybench",
            "family": family,
            "n_views_available": n_views,
            "k_views_shown": len(paths),
            "view_indices_shown": view_indices,
            "template": template,
            "prompt_matches_view_count": bool(matches),
        },
    }


def choose_views(question: mc.MindCubeQuestion, views: int, rng: random.Random) -> list[int]:
    """The views the student keeps, by the same rule the training pool uses.

    Anchored rows keep their anchor, anchor-free rows draw uniformly rather than
    always taking view 0 -- otherwise the anchor-free subset becomes "front view
    only" and the view budget is confounded with viewpoint. The pair template has
    no anchor and no rewrite, so it takes the leading views to avoid adding
    variance to rows that carry no recoverable signal either way.
    """
    if not question.rewritable:
        return list(range(views))
    if question.anchor is not None:
        kept = {question.anchor}
    else:
        kept = set()
    pool = [index for index in range(question.n_views) if index not in kept]
    rng.shuffle(pool)
    while len(kept) < views:
        kept.add(pool.pop())
    return sorted(kept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", default="data/eval/mindcube_verl")
    parser.add_argument(
        "--views",
        type=int,
        default=0,
        help="views shown to the model; 0 keeps the full set and the prompt verbatim",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="filename infix, e.g. --tag singleview -> mindcube_val_singleview.parquet",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=0, help="smoke builds only")
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.rows)]
    if args.limit:
        rows = rows[: args.limit]
    if args.views and args.views != 1:
        # The rotation premise can only be dropped wholesale; a surviving pair
        # would need a premise rewritten for it, which the builder refuses to
        # invent. Failing here beats emitting a prompt that lies about the camera.
        raise SystemExit(f"--views must be 0 (full) or 1; got {args.views}")

    rng = random.Random(args.seed)
    built = [build_row(row, args.image_root, args.views, rng) for row in rows]

    missing = [
        image["path"]
        for row in built
        for image in row["images"]
        if not os.path.exists(image["path"])
    ]
    if missing:
        for path in missing[:5]:
            print(f"  missing image: {path}")
        raise SystemExit(f"{len(missing)} referenced images do not exist")

    for row in built:
        if row["prompt"][1]["content"].count(vw.IMAGE_TOKEN) != len(row["images"]):
            raise SystemExit(f"{row['extra_info']['id']}: placeholder/image mismatch")

    out_dir = os.path.join(src.REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
    suffix = f"_{args.limit}" if args.limit else ""
    path = os.path.join(out_dir, f"mindcube_val{tag}{suffix}.parquet")
    pd.DataFrame(built).to_parquet(path, index=False)

    print(f"{path}: {len(built)} rows, {os.path.getsize(path) / 1e6:.1f} MB")
    report(built, args.views)


def report(rows: list[dict], views: int) -> None:
    print(f"\n=== validation composition (views={'all' if views == 0 else views}) ===")
    counts = Counter(row["data_source"] for row in rows)
    for data_source, count in sorted(counts.items()):
        print(f"  {data_source:<28} {count:>5}")
    print(f"  {'TOTAL':<28} {len(rows):>5}")

    shown = Counter(row["extra_info"]["k_views_shown"] for row in rows)
    print(f"\n  images per row: {dict(sorted(shown.items()))}")
    print(f"  gold letters:   {dict(sorted(Counter(r['extra_info']['answer'] for r in rows).items()))}")
    inconsistent = sum(1 for row in rows if not row["extra_info"]["prompt_matches_view_count"])
    if inconsistent:
        print(
            f"  rows whose prompt references views they were not shown: {inconsistent}"
            f" ({100 * inconsistent / len(rows):.1f}%) -- the pair template, by protocol"
        )
    print(
        "\nverl reports one val-core/<data_source>/acc/mean@1 per line above;\n"
        "mindcube_metrics adds the overall / per-family / per-N roll-up that the\n"
        "46.86 / 74.48 / 69.81 anchors are quoted on."
    )


if __name__ == "__main__":
    main()
