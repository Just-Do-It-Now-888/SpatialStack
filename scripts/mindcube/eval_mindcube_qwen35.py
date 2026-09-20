#!/usr/bin/env python3
"""Score a Qwen3.5 checkpoint on the local MindCube tinybench split.

Reads the official `MindCube_tinybench_raw_qa.jsonl`, which already carries the
`input_prompt` used at eval time, so the prompt is not rebuilt here. Model
loading, image preprocessing and decoding mirror
`src/lmms_eval/models/qwen3_5.py`; scoring reuses `mindcube_process_results`
from `src/lmms_eval/tasks/mindcube/utils.py` so there is only one definition of
the answer parser and the among/around/rotation routing.

The lmms-eval `mindcube_tiny` task is deliberately not used: it points at a
remote HF dataset that is not cached on this box.

Run one process per GPU with --shard-index / --num-shards, then merge with
--merge-only.
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

DEFAULT_ROWS = (
    "/home/c30084464/Documents/code/3DThinker/MindCube-main/data/prompts/general/"
    "MindCube_tinybench_raw_qa.jsonl"
)
DEFAULT_IMAGE_ROOT = "/home/c30084464/Documents/code/3DThinker/MindCube-main/data"


def load_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def aggregate(records):
    from lmms_eval.tasks.mindcube.utils import (
        mindcube_aggregate_among_results,
        mindcube_aggregate_around_results,
        mindcube_aggregate_results,
        mindcube_aggregate_rotation_results,
    )

    overall = [{"score": r["score"]} for r in records]
    typed = [{"score": r["score"], "type": r["type"]} for r in records]
    lengths = [r["output_tokens"] for r in records]
    counts = {}
    for r in records:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    return {
        "n": len(records),
        "overall": 100 * mindcube_aggregate_results(overall),
        "among": 100 * mindcube_aggregate_among_results(typed),
        "around": 100 * mindcube_aggregate_around_results(typed),
        "rotation": 100 * mindcube_aggregate_rotation_results(typed),
        "type_counts": counts,
        # An unreadable answer and a wrong answer look identical in accuracy
        # alone, so both are always reported next to it.
        "unreadable_frac": 100 * sum(1 for r in records if r["pred"] is None) / max(len(records), 1),
        "output_tokens_median": statistics.median(lengths) if lengths else 0,
        "output_tokens_mean": sum(lengths) / max(len(lengths), 1),
        "output_tokens_max": max(lengths) if lengths else 0,
        "hit_cap_frac": 100 * sum(1 for r in records if r["hit_cap"]) / max(len(records), 1),
        "visual_tokens_per_image_median": statistics.median(
            [t for r in records for t in r["visual_tokens"]] or [0]
        ),
    }


def merge(out_dir, num_shards):
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.jsonl")))
    if len(shards) != num_shards:
        print(f"FAIL: expected {num_shards} shards, found {len(shards)}", file=sys.stderr)
        sys.exit(1)
    records = []
    for path in shards:
        with open(path) as fh:
            records += [json.loads(line) for line in fh]
    seen = {r["id"] for r in records}
    if len(seen) != len(records):
        print(f"FAIL: duplicate ids after merge ({len(records)} rows, {len(seen)} ids)", file=sys.stderr)
        sys.exit(1)
    records.sort(key=lambda r: r["id"])

    samples = os.path.join(out_dir, "samples.jsonl")
    with open(samples, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate(records)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nmerged {len(records)} rows -> {samples}")
    print(f"  overall  {summary['overall']:.2f}")
    print(f"  among    {summary['among']:.2f}   around {summary['around']:.2f}   "
          f"rotation {summary['rotation']:.2f}")
    print(f"  type counts        {summary['type_counts']}")
    print(f"  unreadable         {summary['unreadable_frac']:.2f}%")
    print(f"  output tokens      median {summary['output_tokens_median']:.0f}  "
          f"mean {summary['output_tokens_mean']:.1f}  max {summary['output_tokens_max']}")
    print(f"  hit max_new_tokens {summary['hit_cap_frac']:.2f}%")
    print(f"  visual tok/image   median {summary['visual_tokens_per_image_median']:.0f}")
    return summary


def run_shard(args):
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

    from lmms_eval.tasks.mindcube.utils import extract_answer, mindcube_process_results

    rows = load_rows(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    shard = rows[args.shard_index :: args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard)} of {len(rows)} rows")

    config = AutoConfig.from_pretrained(args.model)
    if getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False):
        print("FAIL: this checkpoint wants the geometry encoder, which this script does not load",
              file=sys.stderr)
        sys.exit(1)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels, min_pixels=args.min_pixels, padding_side="left"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard_{args.shard_index:02d}.jsonl")
    started = time.time()

    with open(out_path, "w") as fh:
        for n, row in enumerate(shard, 1):
            images = [
                Image.open(os.path.join(args.image_root, p)).convert("RGB")
                for p in row["images"]
            ]
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": row["input_prompt"]})
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = processor(text=text, images=images, videos=None, padding=True, return_tensors="pt")
            grid = inputs["image_grid_thw"].tolist()
            merge_size = processor.image_processor.merge_size
            visual_tokens = [(t * h * w) // merge_size**2 for t, h, w in grid]
            inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            output_ids = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            trimmed = output_ids[0][inputs["input_ids"].shape[-1] :]
            response = processor.batch_decode(
                [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            scored = mindcube_process_results(row, [response])
            record = {
                "id": row["id"],
                "gt_answer": row["gt_answer"],
                "pred": extract_answer(response.strip()),
                "score": scored["overall_accuracy"]["score"],
                "type": scored["around_accuracy"]["type"],
                "response": response,
                "output_tokens": int(trimmed.shape[-1]),
                "hit_cap": int(trimmed.shape[-1]) >= args.max_new_tokens,
                "num_images": len(images),
                "image_grid_thw": grid,
                "visual_tokens": visual_tokens,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(shard):
                rate = (time.time() - started) / n
                print(f"  [{args.shard_index}] {n}/{len(shard)}  {rate:.2f} s/row", flush=True)

    print(f"shard {args.shard_index} done -> {out_path} ({time.time()-started:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Qwen3.5-4B")
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048)
    ap.add_argument("--max-pixels", dest="max_pixels", type=int, default=1605632)
    ap.add_argument("--min-pixels", dest="min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N rows (0 = all); smoke tests only")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        merge(args.out_dir, args.num_shards)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
