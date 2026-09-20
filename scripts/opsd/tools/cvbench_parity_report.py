#!/usr/bin/env python3
"""Decompose the gap between our in-training CV-Bench score and lmms_eval's.

Two different things get conflated when the two numbers disagree:

1. the parser -- our word-boundary extractor vs lmms_eval's first-[A-F] rule;
2. the inference -- lmms_eval generates 16 tokens through transformers with its
   own prompt, we generate 1024 through the training process's vLLM engines.

The first is measurable for free: ``logs/val/<experiment>/<step>.jsonl`` holds
every in-training generation, so both parsers can be run over the same text.
Only the second needs a GPU. Reporting them together is what makes a diff
actionable instead of a single unexplained delta.

    # parser-only decomposition on the in-training generations
    python3 scripts/opsd/tools/cvbench_parity_report.py \
        --val-dump logs/val/20260817_qwen35base_mvopsd_v1_main/300.jsonl

    # add lmms_eval's own inference once the offline run has finished
    python3 scripts/opsd/tools/cvbench_parity_report.py \
        --val-dump logs/val/20260817_qwen35base_mvopsd_v1_main/300.jsonl \
        --lmms-samples logs/eval/<run>/**/cvbench/**/*_samples_cvbench.jsonl

The val dump carries no data_source, so source/task are recovered positionally
from the val parquet. ``data.validation_shuffle: False`` is what makes that
sound; the alignment is asserted on the ground truths rather than assumed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))

from cvbench_scoring import score_cvbench_row  # noqa: E402
from verl.trainer.ppo.cvbench_metrics import compute_cvbench_metrics  # noqa: E402

DEFAULT_PARQUET = "data/eval/cvbench_verl/cvbench_val.parquet"
PARSERS = ("boxed_lastline", "word_boundary", "lmms_legacy")


def load_val_dump(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def load_data_sources(parquet_path: str, ground_truths: list[str]) -> list[str]:
    """Recover per-row data_source positionally, asserting the order matches."""
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    if len(frame) != len(ground_truths):
        raise SystemExit(
            f"row count mismatch: parquet has {len(frame)}, dump has {len(ground_truths)}; "
            "the dump was produced against a different val file"
        )

    parquet_gts = [str(row["ground_truth"]) for row in frame["reward_model"]]
    mismatched = [i for i, (a, b) in enumerate(zip(parquet_gts, ground_truths)) if a != b]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} ground truths disagree with the parquet at the same index "
            f"(first at {mismatched[:5]}); rows are not positionally aligned"
        )
    return [str(source) for source in frame["data_source"]]


def score_all(data_sources: list[str], predictions: list[str], ground_truths: list[str], parser: str) -> dict:
    accuracies, answered = [], []
    for prediction, target in zip(predictions, ground_truths):
        accuracy, was_answered = score_cvbench_row(prediction or "", target or "", parser=parser)
        accuracies.append(accuracy)
        answered.append(was_answered)
    return compute_cvbench_metrics(data_sources, accuracies, answered)


def load_lmms_samples(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(pattern, recursive=True))
    paths = [path for path in paths if os.path.getsize(path) > 100_000]
    if not paths:
        raise SystemExit(f"no lmms_eval sample files matched {pattern!r}")
    if len(paths) > 1:
        print(f"note: {len(paths)} sample files matched; using the newest ({paths[-1]})")
    with open(paths[-1]) as handle:
        return [json.loads(line) for line in handle]


def score_lmms_samples(samples: list[dict], parser: str) -> dict:
    data_sources, accuracies, answered = [], [], []
    for sample in samples:
        doc = sample["doc"]
        response = sample["filtered_resps"][0]
        gold = doc["answer"][1]
        accuracy, was_answered = score_cvbench_row(response, gold, parser=parser)
        data_sources.append(f"cvbench/{doc['source']}/{doc['task']}")
        accuracies.append(accuracy)
        answered.append(was_answered)
    return compute_cvbench_metrics(data_sources, accuracies, answered)


def pct(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "     -" if value is None else f"{100 * value:6.2f}"


def report(columns: list[tuple[str, dict]]) -> None:
    rows = [
        ("combined", "val-core/cvbench/combined/acc"),
        ("2D", "val-aux/cvbench/2d/acc"),
        ("3D", "val-aux/cvbench/3d/acc"),
        ("  ADE20K", "val-aux/cvbench/source/ADE20K/acc"),
        ("  COCO", "val-aux/cvbench/source/COCO/acc"),
        ("  Omni3D", "val-aux/cvbench/source/Omni3D/acc"),
        ("micro", "val-aux/cvbench/micro/acc"),
        ("answered", "val-core/cvbench/answered/frac"),
    ]

    header = f"{'metric':<12}" + "".join(f"{label:>22}" for label, _ in columns)
    print("\n" + header)
    print("-" * len(header))
    for label, key in rows:
        line = f"{label:<12}"
        for _, metrics in columns:
            line += f"{pct(metrics, key):>22}"
        print(line)

    counts = {label: metrics.get("val-aux/cvbench/num_samples") for label, metrics in columns}
    print(f"\nrows scored: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dump", required=True, help="logs/val/<experiment>/<step>.jsonl")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--lmms-samples", default=None, help="glob for *_samples_cvbench.jsonl")
    args = parser.parse_args()

    dump_path = args.val_dump if os.path.isabs(args.val_dump) else os.path.join(REPO_ROOT, args.val_dump)
    parquet_path = args.parquet if os.path.isabs(args.parquet) else os.path.join(REPO_ROOT, args.parquet)

    dump = load_val_dump(dump_path)
    predictions = [row.get("output", "") for row in dump]
    ground_truths = [str(row.get("gts", "")) for row in dump]
    data_sources = load_data_sources(parquet_path, ground_truths)

    step = dump[0].get("step") if dump else "?"
    print(f"in-training dump: {os.path.relpath(dump_path, REPO_ROOT)} (step {step}, {len(dump)} rows)")

    columns = [
        ("ours (word_bnd)", score_all(data_sources, predictions, ground_truths, "word_boundary")),
        ("same gen, legacy", score_all(data_sources, predictions, ground_truths, "lmms_legacy")),
    ]

    # Cross-check against what the trainer logged, so a scoring change here cannot
    # silently redefine the number the run was actually steered by.
    logged = [float(row["acc"]) for row in dump if "acc" in row]
    if len(logged) == len(dump):
        logged_metrics = compute_cvbench_metrics(
            data_sources, logged, [float(row.get("answered", 1.0)) for row in dump]
        )
        combined = logged_metrics.get("val-core/cvbench/combined/acc")
        recomputed = columns[0][1].get("val-core/cvbench/combined/acc")
        if combined is not None and recomputed is not None:
            delta = 100 * abs(combined - recomputed)
            status = "matches" if delta < 0.005 else f"DIFFERS by {delta:.2f}pp"
            print(f"trainer-logged combined: {100 * combined:.2f} ({status} the recomputation)")

    if args.lmms_samples:
        samples = load_lmms_samples(args.lmms_samples)
        columns.append(("lmms_eval infer", score_lmms_samples(samples, "lmms_legacy")))
        columns.append(("lmms gen, word_bnd", score_lmms_samples(samples, "word_boundary")))

    report(columns)

    print(
        "\nreading the columns:\n"
        "  ours vs 'same gen, legacy' isolates the parser: identical generations, different rule.\n"
        "  'lmms_eval infer' is the number lmms_eval would publish (16-token generations, its parser).\n"
        "  'lmms gen, word_bnd' re-scores those short generations with our parser, separating\n"
        "  the generation-length effect from the parser effect."
    )


if __name__ == "__main__":
    main()
