#!/usr/bin/env python3
"""Scan student rollouts for answer-hint leakage.

Looks at dumped generation texts (json/jsonl) and flags:
* the OPSD hint template phrasing ("reference solution", "Hint")
* exact copies of the row's ground_truth (when a sidecar parquet or json has answers)

    python3 scripts/opsd/tools/scan_answer_leakage.py \
        rollouts/20260919_spatialstack_answer_opsd_spar_fullviews \
        --answers-parquet data/mvopsd/parquet_answer_opsd_spar_fullviews/main_train.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

HINT_PATTERNS = [
    re.compile(r"reference solution", re.IGNORECASE),
    re.compile(r"\bHint\b"),
    re.compile(r"After understanding the reference solution", re.IGNORECASE),
]


def iter_json_records(path: str):
    if path.endswith(".jsonl"):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            yield item
    elif isinstance(payload, dict):
        for key in ("samples", "generations", "data"):
            if isinstance(payload.get(key), list):
                for item in payload[key]:
                    yield item
                return
        yield payload


def response_text(record: dict) -> str:
    for key in ("response", "output", "generation", "pred", "text", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def sample_id_of(record: dict) -> str | None:
    extra = record.get("extra_info") or {}
    if isinstance(extra, dict):
        for key in ("sample_id", "index"):
            if extra.get(key):
                return str(extra[key])
    for key in ("sample_id", "index", "uid"):
        if record.get(key):
            return str(record[key])
    return None


def load_answers(parquet_path: str) -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path, columns=["extra_info"])
    answers = {}
    for extra in frame["extra_info"]:
        if not isinstance(extra, dict):
            continue
        sid = str(extra.get("sample_id") or extra.get("index") or "")
        answer = str(extra.get("answer") or "").strip()
        if sid and answer:
            answers[sid] = answer
    return answers


def scan_file(path: str, answers: dict[str, str]) -> Counter:
    counts = Counter()
    for record in iter_json_records(path):
        text = response_text(record)
        if not text:
            continue
        counts["responses"] += 1
        for pattern in HINT_PATTERNS:
            if pattern.search(text):
                counts[f"pattern:{pattern.pattern}"] += 1
                counts["hint_pattern"] += 1
                break
        sid = sample_id_of(record)
        answer = answers.get(sid or "") or str(record.get("ground_truth") or "").strip()
        if answer and len(answer) >= 8 and answer in text:
            counts["exact_gt_copy"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="rollout/validation dump file or directory")
    parser.add_argument("--answers-parquet", default="")
    parser.add_argument("--glob", default="*.jsonl")
    args = parser.parse_args()

    answers = load_answers(args.answers_parquet) if args.answers_parquet else {}
    files = []
    if os.path.isfile(args.root):
        files = [args.root]
    else:
        import glob

        files = sorted(glob.glob(os.path.join(args.root, "**", args.glob), recursive=True))
        files += sorted(glob.glob(os.path.join(args.root, "**", "*.json"), recursive=True))
        files = sorted(set(files))

    total = Counter()
    for path in files:
        try:
            counts = scan_file(path, answers)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue
        if counts["responses"]:
            print(f"{path}: {dict(counts)}")
            total.update(counts)

    n = total["responses"]
    print("\n=== total ===")
    print(f"  responses     {n}")
    if n:
        print(f"  hint_pattern  {total['hint_pattern']}  ({100 * total['hint_pattern'] / n:.2f}%)")
        print(f"  exact_gt_copy {total['exact_gt_copy']}  ({100 * total['exact_gt_copy'] / n:.2f}%)")
    if n and (total["hint_pattern"] or total["exact_gt_copy"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
