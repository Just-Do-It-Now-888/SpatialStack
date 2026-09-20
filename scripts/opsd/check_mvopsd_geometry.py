#!/usr/bin/env python3
"""Verify the two claims the MV-OPSD attribution depends on.

1. verl's student path and verl's teacher path see the *same pixels* for the
   same cached view. They do not share code: the student goes through
   ``process_image`` -> ``fetch_image`` -> ``smart_resize``
   (`utils/dataset/vision_utils.py:23`), the teacher through a bare
   ``Image.open`` (`trainer/ppo/ray_trainer.py:714`). If they disagree,
   resolution becomes a second privileged axis and no result is attributable.

2. A cached view matches what SpatialStack SFT fed the model, so the student
   starts in-distribution with its own initialisation.

    python scripts/opsd/check_mvopsd_geometry.py --samples 200

Failing (1) is fatal; (2) is expected to differ only by JPEG quantisation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import transformers
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from mvopsd import sources as src  # noqa: E402
from mvopsd.geometry import ALIGN_FACTOR, grid_thw, visual_tokens  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/mvopsd/plan")
    parser.add_argument("--cache-root", default="data/mvopsd/views")
    parser.add_argument("--model", default="output/spatialstack_qwen35_novggt_aligned")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--compare-sft", action="store_true", help="also re-run the SFT image pipeline")
    parser.add_argument(
        "--parquet",
        default="",
        help="check the image dicts the trainer will actually load, pixel caps included, "
        "instead of reconstructing them from the plan. Pools that write min_pixels/"
        "max_pixels must be checked this way or the caps go untested.",
    )
    args = parser.parse_args()

    from verl.utils.dataset.vision_utils import process_image

    processor = transformers.AutoProcessor.from_pretrained(os.path.join(src.REPO_ROOT, args.model))
    image_processor = processor.image_processor
    print(
        f"processor: patch_size={image_processor.patch_size} merge_size={image_processor.merge_size} "
        f"-> alignment factor {image_processor.patch_size * image_processor.merge_size}"
    )
    if image_processor.patch_size * image_processor.merge_size != ALIGN_FACTOR:
        raise SystemExit(f"expected alignment factor {ALIGN_FACTOR}")

    cache_root = os.path.join(src.REPO_ROOT, args.cache_root)
    if args.parquet:
        records = records_from_parquet(os.path.join(src.REPO_ROOT, args.parquet))
    else:
        plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
        with open(plan_path) as handle:
            records = [json.loads(line) for _, line in zip(range(20000), handle)]
        for record in records:
            for frame in record["frames"]:
                frame["image"] = {"path": os.path.join(cache_root, frame["cache"])}
    random.Random(0).shuffle(records)

    token_hist: dict[int, int] = {}
    mismatches, sft_deltas, checked = [], [], 0
    max_teacher_tokens = 0

    for record in records:
        if checked >= args.samples:
            break
        teacher_tokens = 0
        for view_index, frame in enumerate(record["frames"]):
            image = dict(frame["image"])
            path = image["path"]

            student_image = process_image(image, image_patch_size=args.image_patch_size)
            with Image.open(path) as handle:  # verl's teacher path, verbatim
                teacher_image = handle.convert("RGB")

            if student_image.size != teacher_image.size:
                mismatches.append(f"{path}: student {student_image.size} vs teacher {teacher_image.size}")
                continue
            delta = np.abs(
                np.asarray(student_image, dtype=np.int16) - np.asarray(teacher_image, dtype=np.int16)
            ).max()
            if delta != 0:
                mismatches.append(f"{path}: identical size but pixels differ by up to {delta}")

            tokens = visual_tokens(student_image.size)
            token_hist[tokens] = token_hist.get(tokens, 0) + 1
            teacher_tokens += tokens

            expected = grid_thw(student_image.size)
            actual = tuple(
                image_processor(student_image, return_tensors="pt")["image_grid_thw"][0].tolist()
            )
            if actual != expected:
                mismatches.append(f"{path}: image_grid_thw {actual} != expected {expected}")

            compare_this = (
                args.compare_sft
                and view_index in record["view_indices"]
                and not frame["marked"]
                and frame["frame_index"] is None  # mp4 sources have no still image to re-run
            )
            if compare_this:
                sft_deltas.append(sft_pixel_delta(frame["src"], path, image_processor))

        max_teacher_tokens = max(max_teacher_tokens, teacher_tokens)
        checked += 1

    print(f"\nchecked {checked} samples / {sum(token_hist.values())} views")
    print("tokens per view:")
    for tokens, count in sorted(token_hist.items()):
        print(f"  {tokens:>4} tokens: {count:>6} views ({100 * count / sum(token_hist.values()):.1f}%)")
    print(f"max teacher visual tokens over the sampled records: {max_teacher_tokens}")

    if sft_deltas:
        deltas = np.array(sft_deltas)
        print(
            f"\nSFT pipeline vs cached view: mean abs pixel delta {deltas.mean():.2f}, "
            f"p95 {np.percentile(deltas, 95):.2f}, max {deltas.max():.2f} (JPEG quantisation only)"
        )

    if mismatches:
        print(f"\nFAILED: {len(mismatches)} mismatches")
        for line in mismatches[:10]:
            print(f"  {line}")
        raise SystemExit(1)
    print("\nOK: verl's student and teacher image paths agree on every checked view")


def records_from_parquet(path: str) -> list[dict]:
    """Reshape parquet rows into the frame list the checker walks.

    The teacher album is the superset here, so it is what gets checked; the
    student's views are a subset of the same files and are flagged for the
    optional SFT comparison.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    records = []
    for row in frame.itertuples(index=False):
        student = {image["path"] for image in row.images}
        records.append(
            {
                "frames": [
                    {
                        "image": {k: v for k, v in image.items()},
                        "cache": image["path"],
                        "src": image["path"],
                        "marked": False,
                        # A parquet row has no still-vs-video provenance left, but
                        # every path in it is already an extracted still.
                        "frame_index": None,
                    }
                    for image in row.teacher_images
                ],
                "view_indices": [
                    index
                    for index, image in enumerate(row.teacher_images)
                    if image["path"] in student
                ],
            }
        )
    return records


def sft_pixel_delta(source_path: str, cache_path: str, image_processor) -> float:
    """Mean absolute difference between SFT's own preprocessing and our cache."""
    from qwen_vl.data.utils import load_and_preprocess_images

    reference = load_and_preprocess_images([source_path])[0]
    _, height, width = reference.shape
    height -= height % ALIGN_FACTOR
    width -= width % ALIGN_FACTOR
    reference = reference[:, :height, :width]

    with Image.open(cache_path) as handle:
        cached = torch.from_numpy(np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    if cached.shape != reference.shape:
        raise AssertionError(f"{cache_path}: cached {tuple(cached.shape)} vs SFT {tuple(reference.shape)}")
    return float((cached - reference).abs().mean() * 255)


if __name__ == "__main__":
    main()
