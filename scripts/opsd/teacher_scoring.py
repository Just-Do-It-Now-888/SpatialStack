"""Grade a teacher generation dump: rule tier, judge queue, aggregation.

The dump holds the response only, so the question and its options are recovered
by joining back to the source parquet on ``row_index``.  Four branches, chosen
by data source and by the shape of the gold answer:

* bare option letter -> the VSI-Bench option reader, restricted to the letters
  the question actually offers (LESSON-018);
* bare number        -> ``MRA:.5:.95:.05``;
* SPAR prose         -> ``spar_scoring``;
* llava_hound        -> no rule tier at all.  These are open captions with no
  structured target, so every row goes to the judge.

Three counters are kept apart on purpose and must stay apart in any report:

``gold_parsed``  the gold itself could be read.  A row that fails here is
                 excluded from the denominator, not counted as a miss.
``answered``     the response could be read.  "Could not read" and "read it and
                 it was wrong" both score zero and mean opposite things
                 (LESSON-011, LESSON-018).
``score``        correctness, once both sides were read.
"""

from __future__ import annotations

import os
import re
import string
import sys
from collections import defaultdict

_OPSD_DIR = os.path.dirname(os.path.abspath(__file__))
if _OPSD_DIR not in sys.path:
    sys.path.insert(0, _OPSD_DIR)

import spar_scoring as spar  # noqa: E402
from vsibench_scoring import extract_vsibench_number, extract_vsibench_option  # noqa: E402

SPAR_SOURCES = frozenset({"spar_3view", "spar_32view"})
# Open-ended captions: there is no structured answer to parse, so the rule tier
# has nothing to offer and every row is a judge row.
FREEFORM_SOURCES = frozenset({"llava_hound_64k"})

_LETTER_GOLD_RE = re.compile(r"^[A-Da-d]$")
_NUMBER_GOLD_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
# "A. window" at a line start, or "A. 0.7 B. 1.3" run together on one line.
_OPTION_MARK_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])[.)]\s+")


def options_from_prompt(prompt: str) -> list[str]:
    """The offered options as ``["A. window", "B. door"]``, or [] when there are none.

    Both layouts in the pool are handled: vlm3r/vsi put one per line under
    ``Options:``, SPAR runs them together inside the question sentence.  The
    letters must be consecutive from A, otherwise a sentence like "Frame A. "
    could masquerade as an option block.
    """
    marks = list(_OPTION_MARK_RE.finditer(prompt or ""))
    if len(marks) < 2:
        return []
    # Keep the last consecutive A,B,C... run: the question text may contain an
    # earlier stray marker.
    best: list[re.Match] = []
    current: list[re.Match] = []
    for mark in marks:
        expected = string.ascii_uppercase[len(current)]
        if mark.group(1) == expected:
            current.append(mark)
        elif mark.group(1) == "A":
            current = [mark]
        else:
            current = []
        if len(current) > len(best):
            best = list(current)
    if len(best) < 2:
        return []

    options = []
    for index, mark in enumerate(best):
        end = best[index + 1].start() if index + 1 < len(best) else len(prompt)
        body = prompt[mark.end() : end].strip()
        # Trailing instruction lines belong to neither option.
        body = re.split(r"\s*(?:Your answer can only include|Answer with the option)", body)[0].strip()
        options.append(f"{mark.group(1)}. {body}")
    return options


def gold_kind(source: str, gold: str) -> str:
    gold = (gold or "").strip()
    if source in FREEFORM_SOURCES:
        return "freeform"
    if _LETTER_GOLD_RE.match(gold):
        return "letter"
    if _NUMBER_GOLD_RE.match(gold):
        return "number"
    if source in SPAR_SOURCES:
        return "spar"
    return "freeform"


def score_row(row: dict, prompt: str) -> dict:
    """Rule-tier verdict for one dump row."""
    source = row.get("source") or row.get("data_source") or ""
    gold = str(row.get("ground_truth", "")).strip()
    response = row.get("response", "") or ""
    question_type = row.get("question_type") or ""
    kind = gold_kind(source, gold)

    result = {
        "kind": kind,
        "family": kind,
        "gold_parsed": 1,
        "gold_repr": gold,
        "answered": 0,
        "parsed_repr": "",
        "score": 0.0,
        "axes": {},
        "axes_missing": [],
        "judge_kind": "",
    }

    if kind == "freeform":
        result["gold_parsed"] = 1
        result["judge_kind"] = "semantic"
        return result

    if kind == "letter":
        options = options_from_prompt(prompt)
        letter = extract_vsibench_option(response, options or None)
        result["family"] = "mcq"
        result["parsed_repr"] = letter
        if not letter:
            result["judge_kind"] = "option"
            return result
        result["answered"] = 1
        result["score"] = float(letter == gold.upper())
        return result

    if kind == "number":
        value = extract_vsibench_number(response)
        result["family"] = "numeric"
        if value is None:
            result["judge_kind"] = "number"
            return result
        result["answered"] = 1
        result["parsed_repr"] = f"{value:g}"
        result["score"] = spar.mean_relative_accuracy(value, float(gold))
        return result

    scored = spar.score_row(question_type, gold, response, question=prompt)
    result.update(
        {
            "family": scored["family"],
            "gold_parsed": scored["gold_parsed"],
            "gold_repr": scored["gold_repr"],
            "answered": scored["answered"],
            "parsed_repr": scored["parsed_repr"],
            "score": scored["score"],
            "axes": scored["axes"],
            "axes_missing": scored["axes_missing"],
        }
    )
    if result["gold_parsed"] and not result["answered"]:
        result["judge_kind"] = {
            "numeric": "number",
            "direction": "direction",
            "compare_pair": "compare",
            "compare_set": "compare",
            "yesno": "yesno",
            "bev": "",  # a coordinate list is not something to ask a judge for
        }.get(scored["family"], "")
    return result


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def _bucket() -> dict:
    return {
        "rows": 0,
        "gold_parsed": 0,
        "answered": 0,
        "score_sum": 0.0,
        "trunc": 0,
        "tokens": [],
        # Direction rows are graded all-or-nothing over two or three axes, and
        # the teacher usually names only one of them. Without these counters a
        # score near zero cannot be told apart from a score near zero for a
        # completely different reason: "it said left when the answer was right"
        # and "it never said left or right at all".
        "dir_rows": 0,
        "dir_complete": 0,
        "dir_axes_stated": 0,
        "dir_axes_correct": 0,
    }


def _finish(bucket: dict) -> dict:
    scorable = bucket["gold_parsed"]
    tokens = sorted(bucket["tokens"])
    direction = {}
    if bucket["dir_rows"]:
        direction = {
            "direction_rows_answered": bucket["dir_rows"],
            # Share of answered direction rows that named every axis the gold states.
            "direction_complete_pct": 100 * bucket["dir_complete"] / bucket["dir_rows"],
            # Accuracy over the axes the response actually named, which is the
            # teacher's directional sense with the incompleteness taken out.
            "direction_axis_acc": (
                100 * bucket["dir_axes_correct"] / bucket["dir_axes_stated"]
                if bucket["dir_axes_stated"]
                else 0.0
            ),
        }
    return {
        **direction,
        "rows": bucket["rows"],
        "scorable": scorable,
        "excluded_gold_unparsed": bucket["rows"] - scorable,
        # Denominator is the scorable rows, not every row: a row whose gold
        # cannot be read is not evidence about the teacher either way.
        "acc": 100 * bucket["score_sum"] / scorable if scorable else 0.0,
        "answered_pct": 100 * bucket["answered"] / scorable if scorable else 0.0,
        # Accuracy among the rows the parser could actually read, which
        # separates "the teacher is wrong" from "we cannot tell".
        "acc_of_answered": 100 * bucket["score_sum"] / bucket["answered"] if bucket["answered"] else 0.0,
        "truncated_pct": 100 * bucket["trunc"] / bucket["rows"] if bucket["rows"] else 0.0,
        "median_output_tokens": tokens[len(tokens) // 2] if tokens else 0,
    }


def aggregate(records: list[dict], keys: tuple[str, ...]) -> dict:
    """Group scored rows by the given fields and roll each group up."""
    buckets: dict[tuple, dict] = defaultdict(_bucket)
    for record in records:
        key = tuple(record.get(field) for field in keys)
        bucket = buckets[key]
        bucket["rows"] += 1
        bucket["trunc"] += int(record.get("truncated", 0))
        bucket["tokens"].append(int(record.get("output_tokens", 0)))
        if not record.get("gold_parsed"):
            continue
        bucket["gold_parsed"] += 1
        bucket["answered"] += int(record.get("answered", 0))
        bucket["score_sum"] += float(record.get("score", 0.0))
        if record.get("family") == "direction" and record.get("answered"):
            axes = record.get("axes") or {}
            bucket["dir_rows"] += 1
            bucket["dir_complete"] += int(not record.get("axes_missing"))
            bucket["dir_axes_stated"] += len(axes)
            bucket["dir_axes_correct"] += sum(int(value) for value in axes.values())
    return {
        ("|".join(str(part) for part in key) if len(key) > 1 else str(key[0])): _finish(bucket)
        for key, bucket in sorted(buckets.items(), key=lambda item: item[0])
    }
