#!/usr/bin/env python3
"""Tests for the teacher-dump rule tier and for the judge's reply handling.

The judge half is tested with synthetic replies rather than a live server: the
thing that breaks is the contract between the instruction we send and the
parser that reads the answer back, and that does not need a GPU to check.

    python3 scripts/opsd/tests/test_teacher_scoring.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "tools")))

import teacher_scoring as ts  # noqa: E402


def _load_judge_tool():
    path = os.path.abspath(os.path.join(_HERE, "..", "tools", "judge_teacher_dump.py"))
    spec = importlib.util.spec_from_file_location("judge_teacher_dump", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge_tool = _load_judge_tool()


# --------------------------------------------------------------------------
# option blocks
# --------------------------------------------------------------------------


def test_options_one_per_line():
    prompt = (
        "Which object appears first?\nOptions:\nA. window\nB. door\nC. chair\nD. table\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert ts.options_from_prompt(prompt) == ["A. window", "B. door", "C. chair", "D. table"]


def test_options_inline_in_the_sentence():
    prompt = (
        "In meters, what is the distance between the red and blue points? "
        "Pick the right response from the available choices. A. 0.7 B. 1.3 C. 1.6 D. 1.0 "
        "Your answer can only include one of options A, B, C or D."
    )
    assert ts.options_from_prompt(prompt) == ["A. 0.7", "B. 1.3", "C. 1.6", "D. 1.0"]


def test_two_option_question():
    prompt = (
        "which is closer to magazine (red point): the green point or the blue point? "
        "A. clothes (green point) B. cabinet (blue point) "
        "Your answer can only include one of options A, B."
    )
    assert ts.options_from_prompt(prompt) == ["A. clothes (green point)", "B. cabinet (blue point)"]


def test_no_option_block():
    assert ts.options_from_prompt("Describe what happens in the video.") == []
    # A lone "A." in prose is not an option block.
    assert ts.options_from_prompt("Frame A. shows the door.") == []


def test_letters_must_start_at_a():
    """A stray "B." mid-sentence cannot start an option block."""
    prompt = "See B. the door and C. the chair."
    assert ts.options_from_prompt(prompt) == []


# --------------------------------------------------------------------------
# rule tier routing
# --------------------------------------------------------------------------


def test_mcq_letter_restricted_to_offered_options():
    prompt = "Which is closer? A. the chair B. the lamp Your answer can only include one of options A, B."
    row = {"source": "vlm3r_scannet", "ground_truth": "B", "response": "Considering the depth, the answer is B."}
    result = ts.score_row(row, prompt)
    assert result["family"] == "mcq" and result["answered"] == 1 and result["score"] == 1.0
    # "D" is not on offer, so it cannot be read as a selection.
    row["response"] = "The answer is D."
    blind = ts.score_row(row, prompt)
    assert blind["answered"] == 0 and blind["judge_kind"] == "option"


def test_freeform_always_goes_to_the_judge():
    row = {"source": "llava_hound_64k", "ground_truth": "A man walks along the beach.", "response": "A man walks."}
    result = ts.score_row(row, "Describe the video.")
    assert result["judge_kind"] == "semantic" and result["answered"] == 0


def test_bare_number_gold_uses_mra():
    row = {"source": "vlm3r_scannet", "ground_truth": "52", "response": "Therefore the length is 52 cm."}
    assert ts.score_row(row, "How long is it?")["score"] == 1.0


def test_spar_prose_gold_routes_to_spar_scoring():
    row = {
        "source": "spar_3view",
        "question_type": "distance_prediction_oc_mv",
        "ground_truth": "chair (red point) lies at a distance of about 2.8 meters from the observer.",
        "response": "Therefore the distance is 2.8 meters.",
    }
    result = ts.score_row(row, "What is the distance to the chair?")
    assert result["family"] == "numeric" and result["score"] == 1.0


def test_unreadable_rows_carry_a_judge_kind():
    row = {
        "source": "spar_3view",
        "question_type": "obj_spatial_relation_oc_mv",
        "ground_truth": "the chair is to the left and below the table.",
        "response": "I cannot tell from these images.",
    }
    result = ts.score_row(row, "Where is the chair?")
    assert result["answered"] == 0 and result["judge_kind"] == "direction"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_unparsable_gold_leaves_the_denominator():
    records = [
        {"n": 3, "gold_parsed": 1, "answered": 1, "score": 1.0, "output_tokens": 10, "truncated": 0},
        {"n": 3, "gold_parsed": 1, "answered": 0, "score": 0.0, "output_tokens": 10, "truncated": 0},
        {"n": 3, "gold_parsed": 0, "answered": 0, "score": 0.0, "output_tokens": 10, "truncated": 0},
    ]
    table = ts.aggregate(records, ("n",))["3"]
    assert table["rows"] == 3 and table["scorable"] == 2 and table["excluded_gold_unparsed"] == 1
    # 1 correct of 2 scorable rows; the unreadable answer is a miss, the
    # unreadable gold is not evidence at all.
    assert table["acc"] == 50.0
    assert table["answered_pct"] == 50.0
    # ...and among the rows that were read, the teacher was right every time.
    assert table["acc_of_answered"] == 100.0


# --------------------------------------------------------------------------
# judge replies
# --------------------------------------------------------------------------


def test_judge_direction_reply_is_parsed():
    reply = "horizontal=left\nvertical=below\ndepth=unspecified"
    assert judge_tool.parse_direction_reply(reply) == {"horizontal": "left", "vertical": "below"}


def test_judge_direction_scores_against_the_gold():
    row = {
        "judge_kind": "direction",
        "question_type": "obj_spatial_relation_oc_mv",
        "ground_truth": "the chair is to the left and below the table.",
    }
    hit = judge_tool.rescore(row, "horizontal=left\nvertical=below\ndepth=unspecified")
    assert hit["answered"] == 1 and hit["score"] == 1.0
    miss = judge_tool.rescore(row, "horizontal=right\nvertical=below\ndepth=unspecified")
    assert miss["answered"] == 1 and miss["score"] == 0.0
    partial = judge_tool.rescore(row, "horizontal=unspecified\nvertical=below\ndepth=unspecified")
    assert partial["answered"] == 1 and partial["score"] == 0.0


def test_judge_none_reply_stays_unanswered():
    row = {"judge_kind": "number", "question_type": "obj_count", "ground_truth": "3"}
    assert judge_tool.rescore(row, "NONE")["answered"] == 0
    assert judge_tool.rescore(row, "")["answered"] == 0


def test_judge_number_reply_uses_mra():
    row = {"judge_kind": "number", "question_type": "obj_count", "ground_truth": "3"}
    assert judge_tool.rescore(row, "3")["score"] == 1.0
    assert 0.0 < judge_tool.rescore(row, "4")["score"] < 1.0


def test_judge_option_reply_restricted_to_offered_letters():
    row = {"judge_kind": "option", "ground_truth": "B", "options": ["A. chair", "B. lamp"]}
    assert judge_tool.rescore(row, "B")["score"] == 1.0
    assert judge_tool.rescore(row, "A")["score"] == 0.0


def test_judge_semantic_verdict():
    row = {"judge_kind": "semantic", "ground_truth": "A man walks along the beach."}
    assert judge_tool.rescore(row, "Yes")["score"] == 1.0
    assert judge_tool.rescore(row, "No")["score"] == 0.0
    assert judge_tool.rescore(row, "maybe, hard to say either way")["answered"] == 0


def test_judge_prompts_name_the_expected_format():
    row = {
        "judge_kind": "direction",
        "question": "Where is the chair?",
        "response": "It is below.",
        "ground_truth": "left and below",
    }
    prompt = judge_tool.build_prompt(row, 6000)
    assert "horizontal=<left|right|unspecified>" in prompt
    # The extraction tiers must not show the judge the gold, or it stops
    # reporting what the response said and starts reporting the right answer.
    assert "left and below" not in prompt

    semantic = judge_tool.build_prompt(dict(row, judge_kind="semantic"), 6000)
    assert "left and below" in semantic, "the semantic tier does need the gold"


def test_long_responses_are_trimmed_from_the_front():
    row = {"judge_kind": "number", "question": "q", "response": "START" + "x" * 9000 + "END"}
    prompt = judge_tool.build_prompt(row, 500)
    assert "END" in prompt and "START" not in prompt and "[truncated]" in prompt


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
