#!/usr/bin/env python3
"""Top-K checkpoint retention by validation score, not recency."""

from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))

from verl.utils.checkpoint.best_ckpt_retention import (  # noqa: E402
    apply_best_retention,
    rank_steps,
    resolve_score,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not condition:
        raise AssertionError(name)


def test_rank_prefers_score_then_later_step():
    keep, drop = rank_steps({50: 0.61, 100: 0.64, 150: 0.64, 200: 0.62}, max_to_keep=3)
    check("keep top scores", keep == [150, 100, 200], str(keep))
    check("drop worst", drop == [50], str(drop))


def test_tie_keeps_larger_step():
    keep, drop = rank_steps({10: 0.5, 20: 0.5, 30: 0.4}, max_to_keep=2)
    check("tie -> later step", keep == [20, 10], str(keep))
    check("low score dropped", drop == [30], str(drop))


def test_resolve_inherits_last_score():
    stored, inherited = resolve_score(None, 0.63)
    check("inherit value", stored == 0.63)
    check("inherited flag", inherited is True)
    stored, inherited = resolve_score(0.7, 0.63)
    check("explicit wins", stored == 0.7 and inherited is False)


def test_apply_deletes_dirs_outside_topk(tmp_path: str) -> None:
    def touch(step: int) -> None:
        os.makedirs(os.path.join(tmp_path, f"global_step_{step}", "actor"), exist_ok=True)

    touch(50)
    apply_best_retention(tmp_path, 50, 0.60, max_to_keep=3, metric="val-core/vsibench/overall/acc")
    touch(100)
    apply_best_retention(tmp_path, 100, 0.64, max_to_keep=3)
    touch(150)
    apply_best_retention(tmp_path, 150, 0.62, max_to_keep=3)
    touch(200)
    result = apply_best_retention(tmp_path, 200, 0.65, max_to_keep=3)
    remaining = sorted(
        int(name.split("_")[-1])
        for name in os.listdir(tmp_path)
        if name.startswith("global_step_")
    )
    check("kept three dirs", remaining == [100, 150, 200], str(remaining))
    check("result keep", result["keep"] == [200, 100, 150], str(result["keep"]))
    with open(os.path.join(tmp_path, "ckpt_scores.json"), encoding="utf-8") as handle:
        payload = json.load(handle)
    check("json dropped 50", "50" not in payload["steps"])
    check("json kept 200", payload["steps"]["200"] == 0.65)


if __name__ == "__main__":
    test_rank_prefers_score_then_later_step()
    test_tie_keeps_larger_step()
    test_resolve_inherits_last_score()
    with tempfile.TemporaryDirectory() as tmp:
        test_apply_deletes_dirs_outside_topk(tmp)
    print("all ok")
