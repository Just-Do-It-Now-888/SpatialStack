#!/usr/bin/env python3
"""VSI-Bench through vLLM with rule scoring, optional judge, and failure diagnostics.

    python3 scripts/opsd/tools/eval_vsibench_vllm.py \
        --model output/20260820_qwen35base_mvopsd_khalf_main_global_step_120_hf \
        --tensor-parallel-size 8 \
        --output-dir logs/eval/vllm/khalf_step120

With judge (Vision-OPD verdict cascade on rule-rejected rows only):

    python3 scripts/opsd/tools/eval_vsibench_vllm.py \
        --model models/Qwen3.5-4B \
        --judge-api-base http://127.0.0.1:8100/v1 \
        --output-dir logs/eval/vllm/base_judged

Outputs under ``output-dir``:
  summary.json          overall + per-question-type + failure counts
  samples.jsonl         every row with rule/final scores and failure_reason
  failures/*.jsonl      truncation / parse / factual / judge_recovered subsets
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

from vsibench_eval_core import EvalConfig, format_report, run_eval, write_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--scenes-per-batch", type=int, default=8)
    parser.add_argument(
        "--protocol",
        default="spatialstack",
        choices=("spatialstack", "spatialstack_plain", "lmms_legacy"),
        help="spatialstack_plain is the unsuffixed original prompt (answer_tail, no boxed-primary). "
        "Empty --prompt-suffix is still required; this flag only selects the parser/budget pairing.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help=r"appended verbatim to every question, e.g. ' The final answer MUST BE put in \boxed{}.'",
    )
    parser.add_argument(
        "--boxed-primary",
        action="store_true",
        help="score a closed \\boxed{} only; no box falls back to the last-line parser "
        "(same as in-training vsibench_boxed_primary). Off by default.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="sample instead of greedy; defaults to temperature 1.0 / top_p 0.8 "
        "(Qwen3 non-thinking guidance, as upstream OPSD's eval uses). Greedy is "
        "the protocol every published number here was measured under, so a "
        "sampled run is a separate series, not a continuation of that curve.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="per-request vLLM seed, so a sampled run replays")
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-parallel-workers", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for summary.json, samples.jsonl, failures/",
    )
    # Back-compat alias
    parser.add_argument("--output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.output:
        output_dir = args.output.replace(".json", "") if args.output.endswith(".json") else args.output
    if not output_dir:
        raise SystemExit("--output-dir is required")

    cfg = EvalConfig(
        model=args.model,
        frames=args.frames,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scenes_per_batch=args.scenes_per_batch,
        limit_scenes=args.limit_scenes,
        protocol=args.protocol,
        prompt_suffix=args.prompt_suffix,
        boxed_primary=args.boxed_primary,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        judge_api_base=args.judge_api_base,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        judge_parallel_workers=args.judge_parallel_workers,
        judge_reasoning_effort=args.reasoning_effort,
    )
    if args.snapshot:
        cfg.snapshot = args.snapshot
    if args.video_root:
        cfg.video_root = args.video_root

    records, summary = run_eval(cfg)
    out = output_dir if os.path.isabs(output_dir) else os.path.join(REPO_ROOT, output_dir)
    write_results(records, summary, out)

    print(format_report(summary))
    print(f"\nsaved: {out}/")


if __name__ == "__main__":
    main()
