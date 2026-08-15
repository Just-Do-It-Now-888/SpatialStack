"""Measure how much supervision survives model_max_length truncation for SPAR samples.

Reproduces the SFT dataloader path (LazySupervisedDataset._get_item + collator slicing)
on real annotations, grouped by how many views a sample carries.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# SPAR is image-only; decord is imported at module scope but never exercised here.
try:
    import decord  # noqa: F401
except ImportError:
    import types

    stub = types.ModuleType("decord")
    stub.VideoReader = None
    stub.cpu = None
    sys.modules["decord"] = stub

from qwen_vl.data.data_qwen import LazySupervisedDataset, get_rope_index_35  # noqa: E402


class DataArgs:
    pass


def build_dataset(model_path, args):
    processor = transformers.AutoProcessor.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = args.model_max_length

    data_args = DataArgs()
    data_args.model_type = "qwen3.5"
    data_args.image_processor = processor.image_processor
    data_args.max_pixels = args.max_pixels
    data_args.min_pixels = args.min_pixels
    data_args.base_interval = 2
    data_args.video_max_frames = 8
    data_args.video_min_frames = 4
    data_args.video_max_frame_pixels = 1664 * 28 * 28
    data_args.video_min_frame_pixels = 256 * 28 * 28
    data_args.data_flatten = False
    data_args.image_grid_pinpoints = None

    data_args.image_processor.max_pixels = data_args.max_pixels
    data_args.image_processor.min_pixels = data_args.min_pixels
    data_args.image_processor.size["longest_edge"] = data_args.max_pixels
    data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    ds = LazySupervisedDataset.__new__(LazySupervisedDataset)
    ds.tokenizer = tokenizer
    ds.data_args = data_args
    ds.model_type = data_args.model_type
    ds.get_rope_index = get_rope_index_35
    ds.video_max_total_pixels = data_args.video_max_frame_pixels
    ds.video_min_total_pixels = data_args.video_min_frame_pixels
    ds.list_data_dict = []
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="output/spatialstack_qwen35_novggt_aligned")
    ap.add_argument("--annotation", default="data/annotations/spar_234k.json")
    ap.add_argument("--media_root", default="data/media")
    ap.add_argument("--model_max_length", type=int, default=12800)
    ap.add_argument("--max_pixels", type=int, default=576 * 28 * 28)
    ap.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    ap.add_argument("--per_group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = build_dataset(args.model_path, args)

    annotations = json.load(open(args.annotation))
    by_views = defaultdict(list)
    for ann in annotations:
        by_views[len(ann.get("images", []))].append(ann)

    random.seed(args.seed)
    max_len = args.model_max_length

    print(f"{'views':>6} {'n':>4} {'total_len':>10} {'label_tok':>10} "
          f"{'kept_len':>9} {'kept_label':>11} {'lost%':>7}")
    print("-" * 66)

    for n_views in sorted(by_views):
        pool = by_views[n_views]
        picks = random.sample(pool, min(args.per_group, len(pool)))
        rows = []
        for ann in picks:
            sample = json.loads(json.dumps(ann))
            sample["data_path"] = args.media_root
            sample["tag"] = "3d"
            ds.list_data_dict = [sample]
            try:
                item = ds._get_item(0)
            except Exception as exc:  # noqa: BLE001
                print(f"{n_views:>6} skipped: {type(exc).__name__}: {exc}")
                break

            input_ids = item["input_ids"]
            labels = item["labels"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)

            total_len = input_ids.shape[-1]
            label_tok = int((labels != -100).sum())
            kept_labels = int((labels[:, :max_len] != -100).sum())
            rows.append((total_len, label_tok, min(total_len, max_len), kept_labels))

        if not rows:
            continue
        n = len(rows)
        avg = [sum(r[i] for r in rows) / n for i in range(4)]
        lost = 100.0 * (1 - avg[3] / avg[1]) if avg[1] else float("nan")
        print(f"{n_views:>6} {n:>4} {avg[0]:>10.0f} {avg[1]:>10.1f} "
              f"{avg[2]:>9.0f} {avg[3]:>11.1f} {lost:>6.1f}%")


if __name__ == "__main__":
    main()
