#!/usr/bin/env python3
"""Print VSI @2048 vLLM scores from the tok2048 chain summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = (
    "base_plain_greedy",
    "spar3_plain_greedy",
    "base_boxed_greedy",
    "spar3_boxed_greedy",
    "base_plain_sample",
    "spar3_plain_sample",
    "base_boxed_sample",
    "spar3_boxed_sample",
)

DIR_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(x: float) -> float:
    return round(100.0 * x, 2) if x <= 1.5 else round(x, 2)


def overall(data: dict) -> float:
    return round(data["rule_only"]["overall"], 2)


def answered(data: dict) -> float:
    return round(data["rule_only"]["answered_pct"], 2)


def direction(data: dict) -> float:
    types = data["by_question_type"]
    return round(sum(types[k]["score"] for k in DIR_TYPES) / 3.0, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-root", default="logs/eval/20260906_vsibench_vllm_tok2048")
    args = parser.parse_args()
    root = Path(args.chain_root)
    rows = {}
    for name in ARMS:
        summary = root / name / "summary.json"
        if not summary.is_file():
            print(f"missing {summary}")
            continue
        rows[name] = load(summary)

    print("=== VSI vLLM max_tokens=2048 ===")
    for name in ARMS:
        if name not in rows:
            continue
        d = rows[name]
        print(
            f"{name}: overall={overall(d):.2f} answered={answered(d):.2f}% "
            f"truncated={d.get('truncated_pct', 0):.2f}%"
        )

    if {"base_plain_greedy", "spar3_plain_greedy"} <= rows.keys():
        print("\noriginal / greedy types (percent):")
        base = rows["base_plain_greedy"]["by_question_type"]
        spar = rows["spar3_plain_greedy"]["by_question_type"]
        keys = [
            "object_counting",
            "object_abs_distance",
            "object_size_estimation",
            "room_size_estimation",
            "object_rel_distance",
            "route_planning",
            "obj_appearance_order",
        ]
        for key in keys:
            b, s = base[key]["score"], spar[key]["score"]
            print(f"  {key}: {b:.2f} {s:.2f} {s - b:+.2f}")
        bd, sd = direction(rows["base_plain_greedy"]), direction(rows["spar3_plain_greedy"])
        print(f"  object_rel_direction: {bd:.2f} {sd:.2f} {sd - bd:+.2f}")


if __name__ == "__main__":
    main()
