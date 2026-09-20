"""Re-grade a VSI-Bench sample dump with the LLM judge and rewrite the report.

Phase 2 of the training-time eval.  Generation (phase 1) holds every GPU, so
the judge cannot run in the same process; this reads the dump phase 1 wrote and
finishes the cascade afterwards.  It calls the same core functions as phase 1,
so the judged report and the rule-only report are produced by one scorer.

The judge only sees rows the rule tier could not settle, so it can raise the
score but never lower it.

    python scripts/opsd/tools/apply_judge_to_dump.py logs/.../samples.jsonl \
        --judge-api-base http://127.0.0.1:8100/v1/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import vsibench_eval_core as core  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="samples.jsonl written by phase 1")
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-parallel-workers", type=int, default=256)
    parser.add_argument("--judge-reasoning-effort", default="")
    parser.add_argument(
        "--mca-scope",
        default="unreadable",
        choices=("unreadable", "rule_wrong"),
        help="which multiple-choice rows reach the judge; see build_judge_requests",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to the directory holding samples.jsonl (rewritten in place)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.samples))
    summary_path = os.path.join(output_dir, "summary.json")
    previous = {}
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            previous = json.load(handle)

    with open(args.samples) as handle:
        records = [core.record_from_dict(json.loads(line)) for line in handle if line.strip()]
    if not records:
        raise SystemExit(f"no samples in {args.samples}")

    cfg = core.EvalConfig(
        model=previous.get("model", "unknown"),
        frames=int(previous.get("frames", 32)),
        max_tokens=int(previous.get("max_tokens", 4096)),
        protocol=previous.get("protocol", "spatialstack"),
        judge_api_base=args.judge_api_base,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        judge_parallel_workers=args.judge_parallel_workers,
        judge_reasoning_effort=args.judge_reasoning_effort,
        judge_mca_scope=args.mca_scope,
    )
    os.environ["VSIBENCH_PROTOCOL"] = cfg.protocol
    os.environ["VSIBENCH_MAX_NEW_TOKENS"] = str(cfg.max_tokens)

    task = core.load_module(core.TASK_UTILS, "vsibench_task_utils")
    upstream = core.load_module(core.VISIONOPD_JUDGE, "visionopd_judge")
    scoring = core.load_module(core.SCORING, "vsibench_scoring")

    pending, kinds, prompts = core.build_judge_requests(records, task, upstream, mca_scope=args.mca_scope)
    print(f"rows={len(records)} sent_to_judge={len(pending)} "
          f"(mca={kinds.count('mca')}, na={kinds.count('na')})", flush=True)

    started = time.perf_counter()
    stats = core.apply_judge(records, pending, kinds, prompts, cfg, task, upstream, scoring)
    judge_seconds = time.perf_counter() - started

    generation_seconds = float(previous.get("wall_seconds", 0.0))
    summary = core.build_summary(records, task, cfg, generation_seconds or judge_seconds, stats)
    summary["judge_seconds"] = judge_seconds
    summary["wall_seconds"] = generation_seconds
    summary["questions_per_second"] = previous.get("questions_per_second", summary["questions_per_second"])
    core.write_results(records, summary, output_dir)

    print(core.format_report(summary))
    print(f"\njudge time: {judge_seconds / 60:.1f} min\nsaved: {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
