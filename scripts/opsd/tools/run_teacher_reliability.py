#!/usr/bin/env python3
"""Generate the frozen teacher's answers to the MV-OPSD training questions.

Full training pool, teacher seeing every view exactly as rollout8 fed it:

    python3 scripts/opsd/tools/run_teacher_reliability.py \
        --parquet data/mvopsd/parquet/main_train.parquet \
        --output-dir logs/eval/teacher_reliability/as_trained

Controlled view sweep (album truncated to a fixed budget, question unchanged):

    python3 scripts/opsd/tools/run_teacher_reliability.py \
        --parquet data/mvopsd/parquet_view_sweep/sweep.parquet \
        --output-dir logs/eval/teacher_reliability/sweep

Writes ``generations.jsonl`` and ``gen_config.json``. Scoring is a separate
step (``tools/score_teacher_dump.py``) so the parsers can be revised without
paying for generation twice.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

from teacher_eval_core import GenConfig, run_generation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="./models/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--limit-images", type=int, default=40)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--images-per-chunk", type=int, default=6000)
    parser.add_argument("--limit-rows", type=int, default=0, help="keep only the first N rows")
    parser.add_argument("--sample-rows", type=int, default=0, help="smoke runs; random draw of N rows")
    parser.add_argument("--sample-seed", type=int, default=20260820)
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help=r"appended verbatim to the last user turn, e.g. ' The final answer MUST BE put in \boxed{}.'",
    )
    parser.add_argument("--prompt-column", default="teacher_prompt")
    parser.add_argument("--image-column", default="teacher_images")
    parser.add_argument("--no-resume", action="store_true", help="ignore and overwrite an existing dump")
    args = parser.parse_args()

    cfg = GenConfig(
        parquet=args.parquet,
        model=args.model,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        limit_images=args.limit_images,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        images_per_chunk=args.images_per_chunk,
        limit_rows=args.limit_rows,
        sample_rows=args.sample_rows,
        sample_seed=args.sample_seed,
        prompt_suffix=args.prompt_suffix,
        prompt_column=args.prompt_column,
        image_column=args.image_column,
        resume=not args.no_resume,
    )
    path = run_generation(cfg)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
