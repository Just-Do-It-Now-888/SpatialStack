#!/usr/bin/env python3
"""Gate the MindCube arms before training: visual budget and loss masking.

Builds `LazySupervisedDataset` exactly the way `train_qwen.py` does, then for one
2-, 3- and 4-image sample reports

  * image_grid_thw and visual tokens per image -- must match what
    scripts/mindcube/eval_mindcube_qwen35.py produces for the same image, or the
    use_vggt_image_preprocess flag did not reach the dataset and training would
    silently fall back to cropping a quarter off every portrait image;
  * the rendered prompt, so a nested `<think>` block would be visible;
  * exactly which tokens carry loss (labels != -100).

Usage:
  python scripts/mindcube/check_mindcube_pipeline.py --dataset mindcube_cot
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

IGNORE_INDEX = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Qwen3.5-4B")
    ap.add_argument("--dataset", default="mindcube_answeronly")
    ap.add_argument("--max-pixels", dest="max_pixels", type=int, default=1605632)
    ap.add_argument("--min-pixels", dest="min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--use-vggt-preprocess", default="false")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer

    from qwen_vl.data.data_qwen import LazySupervisedDataset
    from qwen_vl.train.argument import DataArguments

    use_vggt = args.use_vggt_preprocess.lower() in ("1", "true", "yes")

    # Mirror train_qwen.py's Qwen3.5 branch.
    data_args = DataArguments(
        dataset_use=args.dataset,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        use_vggt_image_preprocess=use_vggt,
    )
    processor = AutoProcessor.from_pretrained(args.model, padding_side="right")
    data_args.image_processor = processor.image_processor
    data_args.processor = processor
    data_args.model_type = "qwen3.5"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")

    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    print(f"\ndataset={args.dataset}  rows={len(dataset)}  "
          f"use_vggt_image_preprocess={use_vggt}  "
          f"max_pixels={args.max_pixels} min_pixels={args.min_pixels}")

    merge_size = processor.image_processor.merge_size

    # One sample per image count, so the 2/3/4-image shapes all get exercised.
    picks = {}
    for i, row in enumerate(dataset.list_data_dict):
        n = len(row["images"])
        picks.setdefault(n, i)
        if len(picks) == 3:
            break

    failures = []
    for n in sorted(picks):
        i = picks[n]
        row = dataset.list_data_dict[i]
        item = dataset[i]

        grid = [g.tolist() for g in item["image_grid_thw"]]
        vis = [(t * h * w) // merge_size**2 for t, h, w in grid]

        # What the eval script would produce for the same first image.
        path = os.path.join(row["data_path"], row["images"][0])
        pil = Image.open(path).convert("RGB")
        eval_proc = AutoProcessor.from_pretrained(
            args.model, max_pixels=args.max_pixels, min_pixels=args.min_pixels
        )
        eg = eval_proc(images=[pil], text=["<|vision_start|><|image_pad|><|vision_end|>"],
                       return_tensors="pt")["image_grid_thw"][0].tolist()
        eval_vis = (eg[0] * eg[1] * eg[2]) // merge_size**2

        text = tokenizer.decode(item["input_ids"])
        supervised = [t for t in item["labels"].tolist() if t != IGNORE_INDEX]
        target_text = tokenizer.decode(supervised)
        nested = text.count("<think>") > 1 or text.count("</think>") > 1

        print(f"\n--- {n} 图样本  id={row['id']}")
        print(f"  原图尺寸           {pil.size}")
        print(f"  训练侧 grid        {grid}")
        print(f"  训练侧 视觉 token  {vis}  合计 {sum(vis)}")
        print(f"  评测侧 grid/token  {eg} / {eval_vis}")
        print(f"  序列总长           {len(item['input_ids'])}  受监督 {len(supervised)}")
        print(f"  <think> 出现次数   {text.count('<think>')}  </think> {text.count('</think>')}")
        print(f"  受监督文本         {target_text[:80]!r}"
              f"{' ... ' + repr(target_text[-40:]) if len(target_text) > 80 else ''}")

        if vis[0] != eval_vis:
            failures.append(f"{n} 图样本: 训练侧 {vis[0]} != 评测侧 {eval_vis} token/图")
        if nested:
            failures.append(f"{n} 图样本: 渲染文本里有嵌套 <think> 块")
        # The end-of-turn token is supervised too, so strip it before checking
        # that the answer block is the last thing the model is taught to emit.
        body = target_text.replace("<|im_end|>", "").rstrip()
        if not body.endswith("</answer>"):
            failures.append(f"{n} 图样本: 受监督文本未以 </answer> 结尾")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        sys.exit(1)
    print("门禁通过：视觉 token 训练/评测一致、无嵌套 think、监督目标以 </answer> 收尾")


if __name__ == "__main__":
    main()
