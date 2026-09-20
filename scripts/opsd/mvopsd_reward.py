"""Correctness score for MV-OPSD training data, mirroring VSI-Bench's metrics.

v0 is reward-free: the vopd policy loss ignores advantages entirely
(`dp_actor.py:1113`), so this score never steers the update. It exists because
verl still runs the reward manager, and because a per-step accuracy metric on
the training distribution is the cheapest diagnostic we get. The GRPO ablation
arm in Phase 1 uses the same function as a real reward.

Scoring follows `src/lmms_eval/tasks/vsibench/utils.py`: exact match on the
option letter for multiple choice, mean relative accuracy (MRA .5:.95:.05) for
numeric answers, and 0.0 for free-form text we cannot verify.

The same function also scores two in-training validation sets, dispatched on the
data_source prefix:

* `cvbench/` -- built by `build_cvbench_val_parquet.py`. CV-Bench needs its own
  extractor because it runs to six options (A-F) and the offline aggregate we
  compare against is produced by a specific parser; reusing the training branch
  would quietly report a different number than `logs/eval/.../cvbench/`.
* `vsibench/` -- built by `build_vsibench_val_parquet.py`, the guardrail from
  2026-08-20 on. This branch delegates to the same `vsibench_scoring.py` parser
  and the same metric definitions the offline path uses, so the in-training
  curve and the offline number are the same measurement on the same protocol.
  The training branch below cannot be reused for it: that one reads the first
  token and only knows options A-D, which on a model that reasons before
  answering scores a correct response as wrong (LESSON-018).
* `mindcube/` and `mindcube1v/` -- built by `build_mindcube_val_parquet.py`, the
  full-view and single-view tinybench sets. Both delegate to
  `mindcube_scoring.py`, which is the offline anchor script's own parser, so the
  in-training curve and the 46.86 / 74.48 / 69.81 anchors are the same
  measurement.
* `mindcube_` (underscore, no slash) -- MindCube *training* rows, whose
  data_source is `mindcube_among_4view` and friends. They route to the same
  extractor rather than to the generic branch below. The generic one takes the
  first `\\b[A-D]\\b` anywhere in the response, which on `<answer>C. Curtain</answer>`
  happens to work but on a CoT arm that reasons first does not; having the
  training diagnostic and the validation score disagree about what counts as a
  correct answer is how a step-0 mismatch becomes unattributable.
"""

from __future__ import annotations

import json
import os
import re
import sys

# verl loads this file by path from inside a Ray worker
# (import_utils.load_module -> spec_from_file_location), and that does not put the
# file's own directory on sys.path. The sibling import below therefore fails in the
# worker even though it resolves for the tests and tools, which prepend this
# directory themselves.
_OPSD_DIR = os.path.dirname(os.path.abspath(__file__))
if _OPSD_DIR not in sys.path:
    sys.path.insert(0, _OPSD_DIR)

from cvbench_scoring import (  # noqa: E402
    DEFAULT_PARSER,
    extract_cvbench_option,
    normalize_parser,
    score_cvbench_row,
)
from mindcube_scoring import score_mindcube_row  # noqa: E402
from vsibench_scoring import (  # noqa: E402
    MRA_THRESHOLDS,
    apply_boxed_primary,
    extract_boxed,
    extract_vsibench_number,
    extract_vsibench_option,
    is_terse_answer,
    is_trustable_mca_tail,
    mean_relative_accuracy as _mean_relative_accuracy,
    normalize_boxed,
)

_OPTION_RE = re.compile(r"\b([A-D])\b")
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")

CVBENCH_DATA_SOURCE_PREFIX = "cvbench/"
VSIBENCH_DATA_SOURCE_PREFIX = "vsibench/"
# Validation: "mindcube/among/4view", "mindcube1v/rotation/3view".
# Training:   "mindcube_among_4view", "mindcube_pair_2view".
# One tuple covers all three because the training names use an underscore where
# the validation names use a slash, so no prefix here can match the other's rows.
MINDCUBE_DATA_SOURCE_PREFIXES = ("mindcube/", "mindcube1v/", "mindcube_")

# src/lmms_eval/tasks/vsibench/utils.py:29-42.
VSIBENCH_MCA_TYPES = frozenset(
    {
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    }
)
VSIBENCH_NA_TYPES = frozenset(
    {
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    }
)

# Backward-compatible alias for tests and callers that imported the private name.
_extract_cvbench_option = extract_cvbench_option


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


def _mvopsd_score(prediction: str, target: str) -> float:
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


def _vsibench_options(extra_info) -> list:
    """The option strings for this row, or [] when the question has none."""
    if not isinstance(extra_info, dict):
        return []
    raw = extra_info.get("options_json")
    if not raw:
        return []
    try:
        options = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return list(options) if isinstance(options, list) else []


def _score_vsibench_row(
    data_source: str,
    prediction: str,
    target: str,
    extra_info,
    boxed_primary: bool = False,
) -> tuple[float, float, float]:
    """Score one VSI-Bench validation row exactly as the offline path does.

    Multiple choice is exact match on the option letter; numerical is
    ``MRA:.5:.95:.05``.  ``answered`` is kept separate from the score because a
    response the parser cannot read and a response that is simply wrong both
    score zero, and only the first one means the metric has stopped measuring
    spatial ability (LESSON-011).

    ``boxed_primary`` narrows where the answer may be read from. When the prompt
    asks for ``\\boxed{}`` and the model complied, the box *is* the answer, so
    scanning the surrounding 300 tokens of reasoning can only introduce false
    positives -- that is how a truncated ``Image 178: Kitchen.`` enumeration once
    scored full marks against a 173 cm gold. Rows with no closed box fall back to
    the ordinary parser, which only trusts an explicit "the answer is X" or a
    letter/number on the **last line** that looks like a verdict -- including
    "would be C" / "I choose C" / "so C", but not a stray letter in "a guess".

    Returns ``(score, answered, boxed_present)``.
    """
    question_type = ""
    if isinstance(extra_info, dict):
        question_type = str(extra_info.get("question_type") or "")
    if not question_type:
        question_type = data_source[len(VSIBENCH_DATA_SOURCE_PREFIX) :]

    scored_text, boxed_hit = apply_boxed_primary(prediction, boxed_primary)
    boxed_present = 1.0 if boxed_hit else 0.0

    if question_type in VSIBENCH_MCA_TYPES:
        parsed = extract_vsibench_option(scored_text, _vsibench_options(extra_info))
        if not parsed:
            return 0.0, 0.0, boxed_present
        return float(parsed == target.strip().upper()), 1.0, boxed_present

    if question_type in VSIBENCH_NA_TYPES:
        parsed = extract_vsibench_number(scored_text)
        if parsed is None:
            return 0.0, 0.0, boxed_present
        target_value = _to_float(target)
        if target_value is None:
            return 0.0, 1.0, boxed_present
        return _mean_relative_accuracy(parsed, target_value), 1.0, boxed_present

    # An unknown type must not be silently scored zero: that reads as "the model
    # got it wrong" when it actually means the validation set and this file
    # disagree about what types exist.
    raise ValueError(f"unknown VSI-Bench question type {question_type!r} from data_source {data_source!r}")


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs) -> dict:
    """Return {"score", "acc", "answered"} for every row, benchmark and training alike.

    The dict is unconditional on purpose. NaiveRewardManager only collects
    reward_extra_info for rows whose score is a dict, and `_validate` asserts
    every extra-info list is as long as the batch, so returning a float for some
    data sources and a dict for others would crash validation the moment a
    single batch mixed them. "acc" is the key `process_validation_metrics`
    promotes to the `val-core/` section.

    "answered" is the format-collapse alarm. A model that stops emitting option
    letters scores near zero for a reason that has nothing to do with spatial
    ability, and v0 burned eight hours plus a day of analysis before anyone
    could tell the two apart (LESSON-011).
    """
    prediction = (solution_str or "").strip()
    target = (ground_truth or "").strip()
    boxed_present = 0.0

    if not prediction or not target:
        score, answered = 0.0, 0.0
    elif (data_source or "").startswith(CVBENCH_DATA_SOURCE_PREFIX):
        cvbench_parser = normalize_parser(kwargs.get("cvbench_parser", DEFAULT_PARSER))
        extra = extra_info or {}
        score, answered = score_cvbench_row(
            prediction,
            target,
            parser=cvbench_parser,
            choices=extra.get("choices"),
        )
        boxed_present = 1.0 if extract_boxed(prediction) is not None else 0.0
    elif (data_source or "").startswith(MINDCUBE_DATA_SOURCE_PREFIXES):
        score, answered = score_mindcube_row(prediction, target)
        # MindCube never asks for \boxed{}; it asks for <answer>C. Curtain</answer>.
        # The key is still reported because NaiveRewardManager requires every row's
        # extra-info dict to carry the same keys (see the note at the return).
        boxed_present = 1.0 if extract_boxed(prediction) is not None else 0.0
    elif (data_source or "").startswith(VSIBENCH_DATA_SOURCE_PREFIX):
        score, answered, boxed_present = _score_vsibench_row(
            data_source,
            prediction,
            target,
            extra_info,
            boxed_primary=bool(kwargs.get("vsibench_boxed_primary", False)),
        )
    else:
        score = _mvopsd_score(prediction, target)
        answered = 1.0 if prediction else 0.0

    # Response length is the earliest signal of the repetitive degeneration that
    # cost this project a training run: the median went from 4 tokens to ~200
    # while CV-Bench still looked healthy (LESSON-020). Characters rather than
    # tokens because this function never sees the tokenizer.
    #
    # boxed_present is reported for every row, including the ones where the key
    # is meaningless: NaiveRewardManager only gathers reward_extra_info for rows
    # whose score is a dict and `_validate` asserts every extra-info list is as
    # long as the batch, so a key that appears on some rows and not others breaks
    # validation as soon as one batch mixes data sources.
    return {
        "score": score,
        "acc": score,
        "answered": answered,
        "resp_chars": float(len(prediction)),
        "boxed_present": boxed_present,
    }
