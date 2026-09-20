#!/usr/bin/env python3
"""Regression tests for the SPAR answer parsers.

The load-bearing check is ``test_gold_scores_itself``: every gold string in the
training pool is fed back in as if the model had produced it, and must score
1.0.  A parser that reads the gold one way and the response another would
otherwise mark correct answers wrong, and on prose answers there is no other
cheap way to notice.

    python3 scripts/opsd/tests/test_spar_scoring.py
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import spar_scoring as sp  # noqa: E402
import vsibench_scoring as vs  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
POOL = os.path.join(REPO_ROOT, "data", "mvopsd", "parquet", "main_train.parquet")

# Gold strings whose own axes contradict each other; measured, not assumed.
KNOWN_UNPARSED_GOLD = 7


def _pool_rows():
    import pandas as pd

    frame = pd.read_parquet(POOL, columns=["data_source", "reward_model", "extra_info", "teacher_prompt"])
    frame = frame[frame["data_source"].isin(["spar_3view", "spar_32view"])]
    for record in frame.to_dict(orient="records"):
        gold = str(record["reward_model"]["ground_truth"]).strip()
        if len(gold) == 1 and gold.upper() in "ABCD":
            continue
        if re.fullmatch(r"-?\d+(\.\d+)?", gold):
            continue
        yield (
            record["extra_info"].get("question_type", ""),
            gold,
            record["teacher_prompt"][0]["content"],
        )


def test_gold_parses():
    rows = list(_pool_rows())
    failures = Counter()
    for question_type, gold, _ in rows:
        parsed = sp.parse_gold(question_type, gold)
        if parsed.ambiguous or parsed.is_empty():
            failures[question_type] += 1
    total = sum(failures.values())
    assert total <= KNOWN_UNPARSED_GOLD, f"gold parse regressed: {total} failures, {dict(failures)}"
    print(f"  gold parsed: {len(rows) - total}/{len(rows)} ({100 * (1 - total / len(rows)):.2f}%), {dict(failures)}")


def test_gold_scores_itself():
    """Feeding the gold back as the response must score 1.0 on every family."""
    rows = list(_pool_rows())
    mismatches = Counter()
    checked = Counter()
    for question_type, gold, question in rows:
        parsed = sp.parse_gold(question_type, gold)
        if parsed.ambiguous or parsed.is_empty():
            continue
        result = sp.score_row(question_type, gold, gold, question=question)
        checked[result["family"]] += 1
        if not (result["answered"] and result["score"] >= 1.0 - 1e-9):
            mismatches[f"{question_type}/{result['family']}"] += 1
    for family, count in sorted(checked.items()):
        print(f"  {family:<14} {count:>6} self-scored")
    assert not mismatches, f"gold did not score itself: {dict(mismatches)}"


def test_before_after_split():
    """Only the post-move state answers an imagination question."""
    gold = (
        "Initially, Object35 is perceived as to the left by the observer. It might also seem "
        "to the front. After moving to Object0 and orienting toward Object24, Object35 is "
        "described as to the right below. It appears to the front."
    )
    parsed = sp.parse_gold("spatial_imagination_oc_mv", gold)
    assert parsed.axes == {"horizontal": "right", "vertical": "below"}, parsed.axes
    assert sp.score_row("spatial_imagination_oc_mv", gold, "It is to the right and below.")["score"] == 1.0
    assert sp.score_row("spatial_imagination_oc_mv", gold, "It is to the left and below.")["score"] == 0.0


def test_observer_clause_ignored():
    """"in front of the observer" is not part of the answer, and both poles may appear."""
    gold = (
        "trash can (Object2) shows up to the left and below relative to tray (Object3). "
        "It might seem farther. The trash can (Object2) is in front of the new observer. "
        "The tray (Object3) is behind the new observer."
    )
    # "in front of" and "behind" both appear, for the two objects; neither is
    # the answer, so the axes come out clean.
    parsed = sp.parse_gold("spatial_imagination_oo_video_hard", gold)
    assert parsed.axes == {"horizontal": "left", "vertical": "below", "depth": "farther"}, parsed.axes
    assert not parsed.ambiguous


def test_depth_axis_only_for_object_to_object():
    gold = "the Object25(red bbox) can be spotted to the left and above the Object6(blue bbox). It may be farther."
    parsed = sp.parse_gold("obj_spatial_relation_oo_mv", gold)
    assert parsed.axes == {"horizontal": "left", "vertical": "above", "depth": "farther"}, parsed.axes


def test_partial_direction_is_answered_but_wrong():
    """Naming one of two axes is an incomplete answer, not an unreadable one.

    The teacher does this constantly -- the gold says "left below" and it
    replies "below" -- so routing these to the judge would send tens of
    thousands of rows for it to reach the same verdict.
    """
    gold = "the chair is to the left and below the table."
    partial = sp.score_row("obj_spatial_relation_oc_mv", gold, "The chair is below the table.")
    assert partial["gold_parsed"] == 1
    assert partial["answered"] == 1
    assert partial["score"] == 0.0
    assert partial["axes_missing"] == ["horizontal"]

    silent = sp.score_row("obj_spatial_relation_oc_mv", gold, "I cannot tell from these images.")
    assert silent["answered"] == 0, "a response with no axis at all is unreadable"


def test_compare_set_reads_the_object_name():
    """The teacher answers with the object, not the label the question gave it."""
    question = (
        "Which item in kitchen cabinets (in Frame-1, point0), cabinet (in Frame-1, point1), "
        "sofa chair (in Frame-24, point2), pot (in Frame-3, point3), has the farthest distance to couch?"
    )
    gold = "The object farthest to couch is sofa chair (Object2), which is 4.2 meters away."
    hit = sp.score_row("distance_infer_center_oo_video", gold, "Therefore, the sofa chair is the farthest.", question)
    assert hit["answered"] == 1 and hit["score"] == 1.0, hit
    miss = sp.score_row("distance_infer_center_oo_video", gold, "Therefore, the pot is the farthest.", question)
    assert miss["answered"] == 1 and miss["score"] == 0.0, miss


def test_compare_set_wrong_end_cannot_match():
    """With more than two candidates, naming the closest does not answer "farthest"."""
    question = (
        "Which item in a (in Frame-1, point0), b (in Frame-1, point1), c (in Frame-2, point2), "
        "d (in Frame-3, point3), has the farthest distance to couch?"
    )
    gold = "The object farthest to couch is c (Object2)."
    result = sp.score_row("distance_infer_center_oo_video", gold, "Object0 is the closest to the couch.", question)
    assert result["answered"] == 1 and result["score"] == 0.0


def test_spelled_out_counts():
    gold = "In that recorded segment, we see 2 cooking pot stationed in the room."
    assert sp.score_row("obj_count", gold, "Therefore, there are two cooking pots.")["score"] == 1.0
    assert sp.score_row("obj_count", gold, "Therefore, there are five cooking pots.")["score"] < 1.0


def test_frame_labels_are_not_numbers():
    """"Frame-4" must not read as the number -4 (see vsibench_scoring._NUMBER_RE).

    Every multi-view prompt labels its images Frame-0..Frame-N and the teacher
    cites those labels while explaining, so a counting answer whose last clause
    is a frame reference used to be scored against a negative count.
    """
    gold = "Looking at the footage, we discover 2 clothes occupying that room."
    cited = (
        "- In Frame-3, there is a pile of clothes on the washing machine.\n"
        "- In Frame-4, there is another pile on the dryer.\n"
        "Therefore, there are two distinct piles of clothes occupying the room."
    )
    result = sp.score_row("obj_count", gold, cited)
    assert result["parsed_repr"] == "2", result
    assert result["score"] == 1.0, result

    # The digits of a frame label must not surface even when nothing else can.
    only_labels = "The clothes appear in Frame-18 and Frame-23."
    assert sp.score_row("obj_count", gold, only_labels)["answered"] == 0

    # A real negative number still parses.
    assert vs.extract_vsibench_number("The offset is -3.5 metres.") == -3.5


def test_scene_ids_are_not_numbers():
    assert vs.extract_vsibench_number("Recorded in scene0555_00.") is None


def test_sentence_final_number_is_readable():
    """A number that ends a sentence used to parse as None (see vsibench_scoring._NUMBER_RE)."""
    gold = "From the scenario captured, there are 2 pipe in the room."
    result = sp.score_row("obj_count", gold, "Looking at the frames, the pipes in the room number 2.")
    assert result["answered"] == 1, result


def test_compare_pair_polarity_reconciles():
    """The gold and the response may state the same fact with opposite polarity."""
    question = "which is closer to magazine (red point): the green point or the blue point?"
    gold = "With distances of 2.1 meters (wardrobe (green point)) and 1.0 meters (socket (blue point)), the wardrobe (green point) is evidently farther to light switch."
    # gold says green is farther, so blue is the closer one the question asks for
    assert sp.score_row("distance_infer_center_oo_mv", gold, "The blue point is closer.", question)["score"] == 1.0
    assert sp.score_row("distance_infer_center_oo_mv", gold, "The green point is farther.", question)["score"] == 1.0
    assert sp.score_row("distance_infer_center_oo_mv", gold, "The green point is closer.", question)["score"] == 0.0


def test_compare_pair_bare_colour():
    question = "which is closer to chair: the green point or the blue point?"
    assert sp.score_row("distance_infer_center_oo_mv", "dumbbell (green point)", "green point", question)["score"] == 1.0
    assert sp.score_row("distance_infer_center_oo_mv", "dumbbell (green point)", "blue point", question)["score"] == 0.0


def test_yes_no_rows():
    result = sp.score_row("obj_spatial_relation_oo_mv", "No", "Based on the 3D centers, the answer is No.")
    assert result["family"] == "yesno" and result["answered"] == 1 and result["score"] == 1.0
    assert sp.score_row("obj_spatial_relation_oo_mv", "No", "Yes, it is above.")["score"] == 0.0


def test_image_label_lines_are_not_answers():
    """Trailing ``Image N:`` enumeration must not supply the numeric answer."""
    cited = (
        "Image 1 shows the chair. The wall is 9.0 meters wide. "
        "Therefore the distance is 2.8 meters."
    )
    tail = "\n".join(f"* Image {i}: Kitchen." for i in range(170, 179))
    assert vs.extract_vsibench_number(cited + "\n" + tail) == 2.8

    only_enum = tail
    assert vs.extract_vsibench_number(only_enum) is None

    reasoning = "Let's assume it's a standard 2-seater sofa. Length ~150-160cm.\n"
    assert vs.extract_vsibench_number(reasoning + tail) is None

    frame_list = (
        "Looking around the room.\n"
        "-   The table is visible in several frames (e.g., frame 3, 4, 5, 6, 7, 8, 9, 10, 11,"
    )
    assert vs.extract_vsibench_number(frame_list) is None


    gold = "chair (red point) lies at a distance of about 2.8 meters from the observer."
    exact = sp.score_row("distance_prediction_oc_mv", gold, "Therefore the distance is 2.8 meters.")
    assert exact["score"] == 1.0 and exact["answered"] == 1
    graded = sp.score_row("distance_prediction_oc_mv", gold, "Therefore the distance is 3.5 meters.")
    assert 0.0 < graded["score"] < 1.0, graded["score"]
    # A number buried in the reasoning must not outvote the conclusion.
    misread = sp.score_row(
        "distance_prediction_oc_mv",
        gold,
        "Image 1 shows the chair. The wall is 9.0 meters wide. Therefore the distance is 2.8 meters.",
    )
    assert misread["score"] == 1.0


def test_unreadable_response_is_unanswered():
    gold = "chair (red point) lies at a distance of about 2.8 meters from the observer."
    result = sp.score_row("distance_prediction_oc_mv", gold, "I cannot tell from these images.")
    assert result["answered"] == 0 and result["score"] == 0.0


def test_bev_counts_missing_objects_against_the_answer():
    gold = "Object0 :(-0.3,1.4). Object1 :(-1.1,2.0). Object2 :(-0.8,2.4)."
    full = sp.score_row("spatial_imagination_map_mv", gold, gold)
    assert full["score"] == 1.0
    partial = sp.score_row("spatial_imagination_map_mv", gold, "Object0 :(-0.3,1.4).")
    assert abs(partial["score"] - 1 / 3) < 1e-9, partial["score"]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"\n== {test.__name__}")
        test()
        print("   ok")
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
