"""Correctness score for MV-OPSD training data, mirroring VSI-Bench's metrics.

v0 is reward-free: the vopd policy loss ignores advantages entirely
(`dp_actor.py:1113`), so this score never steers the update. It exists because
verl still runs the reward manager, and because a per-step accuracy metric on
the training distribution is the cheapest diagnostic we get. The GRPO ablation
arm in Phase 1 uses the same function as a real reward.

Scoring follows `src/lmms_eval/tasks/vsibench/utils.py`: exact match on the
option letter for multiple choice, mean relative accuracy (MRA .5:.95:.05) for
numeric answers, and 0.0 for free-form text we cannot verify.
"""

from __future__ import annotations

import re

MRA_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]

_OPTION_RE = re.compile(r"\b([A-D])\b")
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _first_token(text: str) -> str:
    return text.strip().split(" ")[0].rstrip(".").strip()


def _to_float(text: str):
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _mean_relative_accuracy(prediction: float, target: float) -> float:
    if target == 0:
        return float(prediction == 0)
    error = abs(prediction - target) / abs(target)
    hits = sum(1 for threshold in MRA_THRESHOLDS if error <= 1 - threshold)
    return hits / len(MRA_THRESHOLDS)


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs) -> float:
    prediction = (solution_str or "").strip()
    target = (ground_truth or "").strip()
    if not prediction or not target:
        return 0.0

    # Multiple choice: the ground truth is a bare option letter.
    if len(target) == 1 and target.upper() in "ABCD":
        match = _OPTION_RE.search(prediction.upper())
        return float(match is not None and match.group(1) == target.upper())

    target_value = _to_float(target)
    if target_value is not None and len(target.split()) <= 3:
        predicted_value = _to_float(_first_token(prediction)) or _to_float(prediction)
        if predicted_value is None:
            return 0.0
        return _mean_relative_accuracy(predicted_value, target_value)

    # Free-form spatial descriptions (SPAR BEV lists, llava_hound captions) are
    # not rule-verifiable; reporting 0 keeps the metric honest rather than noisy.
    return 0.0
