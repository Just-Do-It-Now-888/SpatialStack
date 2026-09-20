#!/usr/bin/env python3
"""Materialise VSI-Bench as a verl validation set so it can be scored inside the
training loop instead of hours after it.

    python3 scripts/opsd/build_vsibench_val_parquet.py
    python3 scripts/opsd/build_vsibench_val_parquet.py --limit-scenes 8

This replaces CV-Bench as the in-training guardrail.  CV-Bench is a single
image answered in a handful of tokens, and it stayed at its best score while
the same checkpoint collapsed by 19 points on 32-frame video, so it could not
see the failure it was there to catch (LESSON-020).

The 32 frames per scene are sampled exactly the way the offline path samples
them (``vsibench_eval_core.sample_frames``: evenly spaced over the whole clip)
and written as PNG rather than JPEG, so the in-training number differs from the
offline number only by the engine, not by the pixels.

Frames are shared by every question about the same scene, so this writes
288 x 32 = 9,216 images, not 5,130 x 32.

Prompt, options layout and post-prompt come from
``src/lmms_eval/tasks/vsibench/utils.py``; scoring is dispatched on the
``vsibench/`` data_source prefix in ``scripts/opsd/mvopsd_reward.py`` and rolled
up into the official score by ``verl/trainer/ppo/vsibench_metrics.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vsibench_eval_core as core  # noqa: E402

# lmms_eval/models/qwen3_5.py:124-125, same caps the offline run uses.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1605632

# lmms_eval/models/qwen3_5.py builds every request with this system turn.
SYSTEM_PROMPT = "You are a helpful assistant."

from vsibench_scoring import BOXED_LASTLINE_SUFFIX  # noqa: E402


def build_row(
    doc: dict,
    frame_paths: list[str],
    context: str,
    index: int,
    prompt_suffix: str = "",
) -> dict:
    options = doc.get("options") or []
    return {
        # process_validation_metrics groups on this string, and
        # vsibench_metrics parses the question type back out of it to rebuild
        # the unweighted per-type mean that defines the VSI-Bench score.
        "data_source": f"vsibench/{doc['question_type']}",
        "prompt": [
            # The offline path sends this system turn (lmms_eval/models/qwen3_5.py),
            # and Qwen3.5's chat template does not supply one of its own, so
            # leaving it out here is not a formatting detail: the base model
            # scored 51.61 without it against 52.58 with it, and its responses
            # grew a heavier rambling tail (mean 375 -> 558 characters).
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>" * len(frame_paths) + context},
        ],
        "images": [
            {"path": path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
            for path in frame_paths
        ],
        "ability": "spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": str(doc["ground_truth"])},
        "extra_info": {
            "index": index,
            "id": str(doc.get("id", "")),
            "answer": str(doc["ground_truth"]),
            "benchmark": "vsibench",
            "question_type": doc["question_type"],
            "dataset": doc["dataset"],
            "scene_name": doc["scene_name"],
            # The multiple-choice parser restricts itself to the letters the
            # question actually offers, so a stray "C" in prose cannot be a vote
            # in a two-option question.  JSON rather than a list column: verl
            # hands extra_info through as a plain dict and a nested list makes
            # the arrow schema fragile for no gain.
            "options_json": json.dumps(options, ensure_ascii=False),
            "num_options": len(options),
            # Which prompt variant produced this row. Two val parquets now exist
            # (plain and boxed) and their scores are not comparable, so the
            # variant travels with the data rather than living only in the
            # filename: a curve read from the wrong file would otherwise look
            # like a model change.
            "prompt_suffix": prompt_suffix,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/eval/vsibench_verl")
    parser.add_argument("--snapshot", default=core.DEFAULT_SNAPSHOT)
    parser.add_argument("--video-root", default=core.DEFAULT_VIDEO_ROOT)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=0,
        help="smoke builds; 0 keeps all 288 scenes / 5,130 questions",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help=(
            r"appended verbatim to every question. Recommended boxed+last-line form: "
            r"' The final answer MUST BE put in \boxed{} on the last line of your response.' "
            "Default empty, so the file this script has always written is byte-identical (LESSON-013)."
        ),
    )
    parser.add_argument(
        "--tag",
        default="",
        help="filename tag, e.g. --tag boxed_lastline writes vsibench_val_boxed_lastline.parquet",
    )
    args = parser.parse_args()

    # A suffix without a tag would overwrite the plain val parquet that every
    # published in-training number was measured on, and the two are not
    # comparable. Refuse rather than silently replacing it.
    if args.prompt_suffix and not args.tag:
        raise SystemExit("--prompt-suffix requires --tag so the plain val parquet is not overwritten")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    out_dir = os.path.join(repo_root, args.out_dir)
    frame_root = os.path.join(out_dir, "frames")
    os.makedirs(frame_root, exist_ok=True)

    os.environ.setdefault("VSIBENCH_PROTOCOL", "spatialstack")
    task = core.load_module(core.TASK_UTILS, "vsibench_task_utils")

    docs = core.load_docs(args.snapshot)
    by_scene: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in docs:
        by_scene[(doc["dataset"], doc["scene_name"])].append(doc)
    scenes = sorted(by_scene)
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]

    rows: list[dict] = []
    written = 0
    reused = 0
    skipped: list[tuple[str, str]] = []

    for position, key in enumerate(scenes):
        dataset, scene_name = key
        scene_dir = os.path.join(frame_root, dataset, scene_name)
        expected = [os.path.join(scene_dir, f"frame_{i:02d}.png") for i in range(args.frames)]
        have_all = all(os.path.exists(path) for path in expected)

        if args.overwrite_images or not have_all:
            video_path = core.video_path_for(by_scene[key][0], args.video_root)
            try:
                frames = core.sample_frames(video_path, args.frames)
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP {dataset}/{scene_name}: {exc}", file=sys.stderr)
                skipped.append(key)
                continue
            os.makedirs(scene_dir, exist_ok=True)
            # A clip shorter than the frame budget yields fewer frames; the
            # prompt must carry exactly as many <image> tags as there are files.
            expected = [os.path.join(scene_dir, f"frame_{i:02d}.png") for i in range(len(frames))]
            for image, path in zip(frames, expected):
                image.save(path, format="PNG")
            written += len(frames)
        else:
            reused += len(expected)

        for doc in by_scene[key]:
            context = task.vsibench_doc_to_text_plain(doc, core.LMMS_KWARGS) + args.prompt_suffix
            rows.append(build_row(doc, expected, context, len(rows), prompt_suffix=args.prompt_suffix))

        if (position + 1) % 25 == 0:
            print(f"  [{position + 1}/{len(scenes)}] scenes, {len(rows)} questions", flush=True)

    parts = [f"_{args.tag}" if args.tag else "", f"_{args.limit_scenes}scenes" if args.limit_scenes else ""]
    path = os.path.join(out_dir, f"vsibench_val{''.join(parts)}.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)

    print(f"\n{path}: {len(rows)} rows, {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"prompt_suffix={args.prompt_suffix!r}")
    if rows:
        print(f"last user turn ends: ...{rows[0]['prompt'][1]['content'][-90:]!r}")
    print(f"frames: {written} written, {reused} reused, in {frame_root}")
    if skipped:
        print(f"skipped {len(skipped)} scenes with unreadable video: {skipped[:5]}")
    report(rows)


def report(rows: list[dict]) -> None:
    counts = Counter(row["data_source"] for row in rows)
    print("\n=== validation composition ===")
    for data_source, count in sorted(counts.items()):
        print(f"  {data_source:<34} {count:>5}")
    print(f"  {'TOTAL':<34} {len(rows):>5}")
    print(
        "\nverl reports one val-core/<data_source>/acc/mean@1 series per line above;\n"
        "vsibench_metrics adds the official roll-up: the three object_rel_direction\n"
        "difficulties are merged first, then the per-type scores are averaged\n"
        "unweighted, so a 1,000-row type does not outvote a 200-row one."
    )


if __name__ == "__main__":
    main()
