#!/usr/bin/env python3
"""How many VSI-Bench responses hit the token budget, and would a judge help?

    # an lmms_eval sample dump
    python3 scripts/opsd/tools/analyze_vsibench_responses.py \
        logs/eval/<run>/frames_32/vsibench/*/*_samples_vsibench.jsonl

    # or the dump from eval_vsibench_vllm.py
    python3 scripts/opsd/tools/analyze_vsibench_responses.py logs/eval/vllm/<run>_samples.jsonl

The score alone cannot separate three different failures, and they call for
different fixes:

* the answer never arrived because generation stopped at the budget -> raise the
  budget;
* the answer arrived in a shape the rule parser does not read -> fix the parser,
  or add a judge;
* the answer arrived, was read, and was wrong -> the model is wrong, which is
  the only one of the three that is actually about spatial ability.

lmms_eval dumps carry no ``finish_reason``, so truncation is inferred from the
token count reaching the budget. That is exact when the tokenizer is available
and the budget is known; both are checked below rather than assumed.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

TASK_UTILS = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench", "utils.py")


def load_task_utils(protocol: str):
    os.environ["VSIBENCH_PROTOCOL"] = protocol
    spec = importlib.util.spec_from_file_location(f"vsibench_utils_{protocol}", TASK_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: str):
    """Normalise an lmms_eval dump and a vLLM dump into the same shape."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "doc" in raw:  # lmms_eval
                doc = raw["doc"]
                rows.append(
                    {
                        "doc": doc,
                        "response": raw["filtered_resps"][0],
                        "finish_reason": None,
                        "output_tokens": None,
                    }
                )
            else:  # eval_vsibench_vllm.py
                rows.append(
                    {
                        "doc": {
                            "id": raw.get("id"),
                            "question_type": raw["question_type"],
                            "ground_truth": raw["ground_truth"],
                            "options": raw.get("options"),
                        },
                        "response": raw["response"],
                        "finish_reason": raw.get("finish_reason"),
                        "output_tokens": raw.get("output_tokens"),
                    }
                )
    return rows


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(p: float) -> int:
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    return {"median": at(0.5), "p90": at(0.9), "p95": at(0.95), "p99": at(0.99), "max": ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="sample dumps (globs allowed)")
    parser.add_argument("--budget", type=int, default=1024, help="the max_new_tokens the run used")
    parser.add_argument("--tokenizer", default=None, help="model dir; enables exact token counts")
    parser.add_argument("--protocol", default="spatialstack", choices=("spatialstack", "lmms_legacy"))
    parser.add_argument("--show", type=int, default=5, help="unparsed responses to print per file")
    args = parser.parse_args()

    task = load_task_utils(args.protocol)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    paths: list[str] = []
    for pattern in args.paths:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded or [pattern])

    for path in paths:
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            continue
        rows = read_rows(path)
        if not rows:
            print(f"empty: {path}", file=sys.stderr)
            continue

        token_counts: list[int] = []
        truncated_flags: list[bool] = []
        scored = []
        for row in rows:
            response = row["response"]
            if row["output_tokens"] is not None:
                n_tokens = row["output_tokens"]
            elif tokenizer is not None:
                n_tokens = len(tokenizer.encode(response, add_special_tokens=False))
            else:
                n_tokens = None
            if n_tokens is not None:
                token_counts.append(n_tokens)

            if row["finish_reason"] is not None:
                is_truncated = row["finish_reason"] == "length"
            elif n_tokens is not None:
                # lmms_eval strips <think> blocks before dumping, so a truncated
                # response can come back under budget. Treat "within 2 tokens of
                # the budget" as truncated and report it as a lower bound.
                is_truncated = n_tokens >= args.budget - 2
            else:
                is_truncated = False
            truncated_flags.append(is_truncated)

            result = task.vsibench_process_results(dict(row["doc"]), [response])["vsibench_score"]
            scored.append((row, result, is_truncated, n_tokens))

        n = len(scored)
        overall = task.vsibench_aggregate_results([dict(r) for _, r, _, _ in scored])
        answered = task.vsibench_aggregate_answered([dict(r) for _, r, _, _ in scored])
        n_truncated = sum(truncated_flags)
        unparsed = [(row, res, cut, tok) for row, res, cut, tok in scored if not res.get("answered")]
        needs_judge = [(row, res, cut, tok) for row, res, cut, tok in scored if not res.get("answered") or cut]

        print(f"\n{'=' * 78}\n{os.path.relpath(path, REPO_ROOT)}")
        print(f"  questions {n} | overall {overall:.2f} | answered {answered:.2f}%")

        if token_counts:
            stats = percentiles(token_counts)
            print(
                "  output tokens: "
                + " | ".join(f"{k} {v}" for k, v in stats.items())
                + f"  (budget {args.budget})"
            )
            exactness = "exact" if rows[0]["finish_reason"] is not None else "inferred from token count"
            print(f"  truncated: {n_truncated} / {n} ({100 * n_truncated / n:.2f}%) [{exactness}]")
        else:
            print("  output tokens: unavailable (pass --tokenizer <model dir> for exact counts)")

        print(f"  unparsed by the rule parser: {len(unparsed)} / {n} ({100 * len(unparsed) / n:.2f}%)")
        print(f"  unparsed or truncated      : {len(needs_judge)} / {n} ({100 * len(needs_judge) / n:.2f}%)")

        if unparsed:
            print("    unparsed by question type:", dict(Counter(r["doc"]["question_type"] for r, _, _, _ in unparsed)))
            print(f"    of which truncated: {sum(cut for _, _, cut, _ in unparsed)}")

        if token_counts and n_truncated:
            by_type = defaultdict(lambda: [0, 0])
            for row, _, cut, _ in scored:
                stats = by_type[row["doc"]["question_type"]]
                stats[0] += 1
                stats[1] += int(cut)
            print("    truncation by question type:")
            for question_type in sorted(by_type):
                count, cut = by_type[question_type]
                if cut:
                    print(f"      {question_type:<32} {cut:>5} / {count:<5} ({100 * cut / count:5.1f}%)")

        for row, _, cut, tok in unparsed[: args.show]:
            snippet = row["response"].replace("\n", " ")[:150]
            print(f"    [{row['doc']['question_type']}, tokens={tok}, truncated={cut}] {snippet!r}")


if __name__ == "__main__":
    main()
