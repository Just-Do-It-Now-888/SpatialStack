#!/usr/bin/env python3
"""Verify no row in an MV-OPSD parquet exceeds the rollout prompt-length cap.

This has to be checked offline because the training config deliberately refuses to
paper over it: `data.filter_overlong_prompts: False` (re-running the processor over
80k multi-image prompts at startup costs hours) and `data.truncation: error`. The
cap enforced at rollout time is `actor_rollout_ref.rollout.prompt_length`, which
resolves to `data.max_prompt_length`; `agent_loop.py::_truncate_prompt_ids` raises
on the first row above it. So one overlong sample does not degrade quality, it
kills the run at whatever step that sample happens to be drawn.

Token counts are computed without decoding pixels: a row's length is its text plus
the visual tokens of its views, and the views' sizes come from the cached files'
headers.

    length = len(tokenize(chat_template(messages))) + sum(view_tokens) - n_images

The template already emits one <|image_pad|> per image, hence the subtraction; the
processor expands each into that view's token count. View sizes are not uniform --
SPAR and vlm3r are all 512x384 (192 tokens) but llava_hound's video frames run
512x256 to 512x512, i.e. 128 to 256 tokens -- so they have to be measured rather
than assumed. `--verify-top` re-runs the real processor with the real images on
the longest rows to confirm the identity rather than trusting it.

    python3 scripts/opsd/tools/check_prompt_lengths.py \
        data/mvopsd/parquet_k_half/main_train.parquet --limit-tokens 4096

    # the teacher column renders with thinking on and has its own, larger cap
    python3 scripts/opsd/tools/check_prompt_lengths.py \
        data/mvopsd/parquet_single_vlm3r_scannet/main_train.parquet \
        --prompt-key teacher_prompt --image-key teacher_images \
        --enable-thinking --limit-tokens 12288
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mvopsd import sources as src  # noqa: E402

IMAGE_TOKEN = "<image>"


def build_messages(prompt, n_images: int) -> list:
    """Mirror RLHFDataset._build_messages: split the text on <image> placeholders."""
    messages = []
    seen = 0
    for message in prompt:
        content = message["content"]
        parts = [part for part in re.split(r"(<image>)", content) if part != ""]
        content_list = []
        for part in parts:
            if part == IMAGE_TOKEN:
                # The image itself never reaches the tokenizer; only the template's
                # placeholder does, so a dummy entry is enough for length purposes.
                content_list.append({"type": "image", "image": "dummy"})
                seen += 1
            else:
                content_list.append({"type": "text", "text": part})
        messages.append({"role": message["role"], "content": content_list})
    if seen != n_images:
        raise AssertionError(f"prompt has {seen} <image> placeholders but {n_images} images")
    return messages


def measure_views(frame, image_key: str) -> dict[str, int]:
    """Visual tokens per referenced view, read from the cached files' headers."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from mvopsd.geometry import visual_tokens

    wanted = sorted({entry["path"] for row in frame[image_key] if row is not None for entry in row})

    def measure(path: str) -> tuple[str, int]:
        with Image.open(path) as handle:
            return path, visual_tokens(handle.size)

    with ThreadPoolExecutor(max_workers=32) as pool:
        return dict(pool.map(measure, wanted))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+")
    parser.add_argument("--model", default="models/Qwen3.5-4B")
    parser.add_argument("--limit-tokens", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--image-key", default="images")
    # The student and every validation pass render with enable_thinking=False,
    # which closes the reasoning block in the generation prompt. The teacher
    # renders with True (actor.self_distillation.teacher_enable_thinking), which
    # leaves it open and is a few tokens shorter -- pass this when checking a
    # teacher_prompt / teacher_images column so the count is the real one.
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--verify-top", type=int, default=5, help="re-check the N longest rows with real images")
    args = parser.parse_args()

    from transformers import AutoProcessor, AutoTokenizer

    model_path = os.path.join(src.REPO_ROOT, args.model)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    overall_bad = 0
    for path in args.parquet:
        full = path if os.path.isabs(path) else os.path.join(src.REPO_ROOT, path)
        frame = pd.read_parquet(full)
        print(f"\n=== {path} ({len(frame)} rows) ===")

        # Per-view token counts come from the cached views themselves, not from a
        # constant. SPAR and vlm3r views are all 512x384 (192 tokens) so a
        # constant happened to be exact there, but llava_hound is video frames
        # and carries eleven distinct sizes from 512x256 to 512x512, i.e. 128 to
        # 256 tokens per view. Assuming 192 understated the longest rows by 30%
        # and reported every one of them as a MISMATCH, which reads as "the data
        # is broken" when the estimator was.
        tokens_by_path = measure_views(frame, args.image_key)

        lengths = []
        for _, row in frame.iterrows():
            images = row[args.image_key] if row[args.image_key] is not None else []
            messages = build_messages(row[args.prompt_key], len(images))
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=args.enable_thinking
            )
            base = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            # Each <image> placeholder already contributes one token to `base`.
            lengths.append(base + sum(tokens_by_path[entry["path"]] - 1 for entry in images))

        series = pd.Series(lengths)
        print(f"  prompt tokens: min {series.min()}  p50 {int(series.median())}  "
              f"p99 {int(series.quantile(0.99))}  max {series.max()}")
        print(f"  cap {args.limit_tokens} (rollout.prompt_length) -> "
              f"headroom at max: {args.limit_tokens - series.max()}")

        over_prompt = int((series > args.limit_tokens).sum())
        over_model = int((series + args.response_length > args.max_model_len).sum())
        print(f"  rows over prompt cap        : {over_prompt}")
        print(f"  rows over max_model_len-{args.response_length}: {over_model}"
              f"  (prompt + {args.response_length} response vs {args.max_model_len})")
        overall_bad += over_prompt + over_model

        # Longest rows get the real treatment: load the actual views and let the
        # processor count, so the analytic expansion above is checked not assumed.
        if args.verify_top:
            from PIL import Image

            worst = series.nlargest(args.verify_top).index
            print(f"  verifying the {len(worst)} longest rows with the real processor:")
            for index in worst:
                row = frame.iloc[index]
                images = [Image.open(entry["path"]).convert("RGB") for entry in row[args.image_key]]
                messages = build_messages(row[args.prompt_key], len(images))
                for message in messages:
                    pointer = 0
                    for item in message["content"]:
                        if item.get("type") == "image":
                            item["image"] = images[pointer]
                            pointer += 1
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=args.enable_thinking
                )
                actual = processor(text=[text], images=images, return_tensors="pt")["input_ids"].shape[-1]
                mark = "ok" if actual == lengths[index] else "MISMATCH"
                print(f"    {row['extra_info']['sample_id']:<40} images {len(images):>2}  "
                      f"estimated {lengths[index]:>5}  actual {actual:>5}  {mark}")
                if actual != lengths[index]:
                    overall_bad += 1

    if overall_bad:
        sys.exit(f"\n{overall_bad} problem(s) found; do not launch training on this data")
    print("\nall rows fit within the prompt cap and the analytic length matches the processor")


if __name__ == "__main__":
    main()
