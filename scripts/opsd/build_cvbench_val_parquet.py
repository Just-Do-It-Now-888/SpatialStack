#!/usr/bin/env python3
"""Materialise CV-Bench as a verl validation set so it can be scored inside the
training loop instead of hours after it.

    python3 scripts/opsd/build_cvbench_val_parquet.py
    python3 scripts/opsd/build_cvbench_val_parquet.py --prompt-style native

By default the prompt follows ``src/lmms_eval/tasks/cvbench`` (``A. option``
layout, no spurious video preamble). Pass ``--prompt-style native`` to keep the
dataset's own ``(A) option`` wording used before 2026-08-18.

Scoring uses ``scripts/opsd/cvbench_scoring.py``: boxed last-line by default,
with ``word_boundary`` (SPAR3 84.82 dumps) and ``lmms_legacy`` via
``CVBENCH_PARSER``. The official 2D/3D/combined roll-up is in
``verl/trainer/ppo/cvbench_metrics.py``.

Do not overwrite an existing ``cvbench_val.parquet`` that was built without the
boxed last-line suffix. Rebuild into a new file with ``--tag boxed_lastline``.

Images are written out as PNG rather than re-encoded to JPEG: the arrow cache
holds the original PNGs, and a lossy round-trip would put a second difference
between this path and the offline one.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from cvbench_scoring import (  # noqa: E402
    CVBENCH_PROMPT_STYLES,
    DEFAULT_PROMPT_STYLE,
    build_cvbench_prompt,
    normalize_prompt_style,
)

# lmms_eval/models/qwen3_5.py:124-125. Kept as literals rather than imported
# because lmms_eval is not installed in the training env (LESSON-007).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1605632

# Generation stays short in the training loop -- 32768-token rollouts every ten
# steps would cost more than the training -- so the model still has to be told
# to answer with a bare letter.
POST_PROMPT = "Answer with the option's letter from the given choices directly."


def build_row(doc: dict, image_path: str, prompt_style: str) -> dict:
    # doc["answer"] is parenthesised, e.g. "(C)"; cvbench_process_results
    # compares against answer[1], so the ground truth is the bare letter.
    answer = doc["answer"][1]

    return {
        # Both halves matter downstream: process_validation_metrics groups on
        # this string, and cvbench_val_metrics parses source/task back out of it
        # to weight the 2D/3D combination.
        "data_source": f"cvbench/{doc['source']}/{doc['task']}",
        "prompt": [{"role": "user", "content": "<image>" + build_cvbench_prompt(doc, prompt_style)}],
        "images": [{"path": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}],
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": int(doc["idx"]),
            "answer": answer,
            "type": doc["type"],
            "task": doc["task"],
            "source": doc["source"],
            "benchmark": "cvbench",
            "prompt_style": prompt_style,
            "choices": list(doc["choices"]),
            "parser": "boxed_lastline",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache/datasets")
    parser.add_argument("--out-dir", default="data/eval/cvbench_verl")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="smoke builds; 0 keeps all 2638 rows. Sampled with a stride rather "
        "than sliced, because CV-Bench is ordered by subtask and a head slice "
        "would be entirely ADE20K/Count -- never exercising the 2D/3D roll-up.",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=CVBENCH_PROMPT_STYLES,
        default=DEFAULT_PROMPT_STYLE,
        help="lmms_eval uses question + 'Options:' + 'A. ...'; native uses the "
        "dataset's '(A) ...' prompt field.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="filename infix so a boxed-last-line rebuild does not overwrite "
        "cvbench_val.parquet. Example: --tag boxed_lastline -> "
        "cvbench_val_boxed_lastline.parquet",
    )
    args = parser.parse_args()
    prompt_style = normalize_prompt_style(args.prompt_style)

    import datasets

    out_dir = os.path.join(src.REPO_ROOT, args.out_dir)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # The hub is unreachable from this machine; the arrow cache under
    # cache/datasets was populated by an earlier lmms_eval run.
    dataset = datasets.load_dataset(
        "nyu-visionx/CV-Bench",
        cache_dir=os.path.join(src.REPO_ROOT, args.cache_dir),
    )["test"]
    if args.limit and args.limit < len(dataset):
        stride = len(dataset) / args.limit
        dataset = dataset.select([int(i * stride) for i in range(args.limit)])

    rows, written = [], 0
    for doc in dataset:
        image_path = os.path.join(image_dir, f"{int(doc['idx']):05d}.png")
        if args.overwrite_images or not os.path.exists(image_path):
            doc["image"].convert("RGB").save(image_path, format="PNG")
            written += 1
        rows.append(build_row(doc, image_path, prompt_style))

    suffix = f"_{args.limit}" if args.limit else ""
    tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
    path = os.path.join(out_dir, f"cvbench_val{tag}{suffix}.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)

    print(f"{path}: {len(rows)} rows, {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"prompt_style={prompt_style}")
    print(f"images: {written} written, {len(rows) - written} reused, in {image_dir}")
    report(rows)


def report(rows: list[dict]) -> None:
    from collections import Counter

    counts = Counter(row["data_source"] for row in rows)
    print("\n=== validation composition ===")
    for data_source, count in sorted(counts.items()):
        print(f"  {data_source:<28} {count:>5}")
    print(f"  {'TOTAL':<28} {len(rows):>5}")
    print(
        "\nverl reports one val-core/<data_source>/acc/mean@1 series per line above;\n"
        "cvbench_val_metrics adds the official 2D / 3D / combined roll-up."
    )


if __name__ == "__main__":
    main()
