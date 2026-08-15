"""Unit tests for the MV-OPSD view subsetting and prompt rewriting.

Run with ``python -m pytest scripts/opsd/tests/test_mvopsd_data.py`` or plain
``python scripts/opsd/tests/test_mvopsd_data.py``.

The renumbering test is the important one: a stale ``Frame-N`` in the question
after subsetting is a silent semantic corruption, not a crash.
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mvopsd import views as vw
from mvopsd.geometry import ALIGN_FACTOR, grid_thw, resize_to_sft_geometry, sft_view_size, visual_tokens
from mvopsd.markers import MarkerError, marked_view_indices


# --- geometry -----------------------------------------------------------------

@pytest.mark.parametrize(
    "source_size,expected",
    [
        ((1920, 1440), (512, 384)),  # SPAR ScanNet 4:3
        ((1296, 968), (512, 384)),   # SPAR ScanNet++
        ((640, 480), (512, 384)),    # ScanNet video frames
    ],
)
def test_sft_geometry_matches_sft(source_size, expected):
    assert sft_view_size(*source_size) == expected
    resized = resize_to_sft_geometry(Image.new("RGB", source_size, "gray"))
    assert resized.size == expected


def test_cached_views_are_patch_aligned():
    for size in [(1920, 1440), (1296, 968), (640, 480), (1920, 1080), (960, 720)]:
        width, height = resize_to_sft_geometry(Image.new("RGB", size, "gray")).size
        assert width % ALIGN_FACTOR == 0 and height % ALIGN_FACTOR == 0, size


def test_smart_resize_is_identity_on_cached_views():
    """verl's student path must not touch what the teacher path reads verbatim."""
    from qwen_vl_utils.vision_process import IMAGE_MAX_TOKEN_NUM, IMAGE_MIN_TOKEN_NUM, smart_resize

    width, height = sft_view_size(1920, 1440)
    resized = smart_resize(
        height,
        width,
        factor=ALIGN_FACTOR,
        min_pixels=IMAGE_MIN_TOKEN_NUM * ALIGN_FACTOR**2,
        max_pixels=IMAGE_MAX_TOKEN_NUM * ALIGN_FACTOR**2,
    )
    assert resized == (height, width)
    assert visual_tokens((width, height)) == 192
    assert grid_thw((width, height)) == (1, 24, 32)


# --- prompt surgery -----------------------------------------------------------

SPAR32 = "Frame-0: <image>\nFrame-1: <image>\nFrame-2: <image>\nFrame-3: <image>\nHow far is A (in Frame-1) from B (in Frame-3)?"
SPAR3 = "<image>\n<image>\n<image>\n\nUsing the first image as the main viewpoint, where is the sofa?"


def test_split_image_header_keeps_body_verbatim():
    header, body = vw.split_image_header(SPAR32)
    assert header.count(vw.IMAGE_TOKEN) == 4
    assert body == "\nHow far is A (in Frame-1) from B (in Frame-3)?"
    assert vw.has_frame_labels(header)

    header, body = vw.split_image_header(SPAR3)
    assert not vw.has_frame_labels(header)
    assert body.startswith("\n\nUsing the first image")


def test_full_budget_prompt_reproduces_sft_text():
    """With K = N the rebuilt prompt must equal what the SFT dataloader fed."""
    for original in (SPAR32, SPAR3):
        header, body = vw.split_image_header(original)
        style = "frame_labels" if vw.has_frame_labels(header) else "bare"
        rebuilt = vw.build_prompt(style, original.count(vw.IMAGE_TOKEN), body)
        assert rebuilt == vw.collapse_image_newlines(original)


def test_renumbering_follows_the_student_view_order():
    _, body = vw.split_image_header(SPAR32)
    student = vw.renumber_frame_refs(body, {1: 0, 3: 1})
    assert "Frame-0" in student and "Frame-1" in student
    assert "Frame-3" not in student


def test_renumbering_refuses_to_leave_a_dangling_reference():
    _, body = vw.split_image_header(SPAR32)
    with pytest.raises(KeyError):
        vw.renumber_frame_refs(body, {1: 0})


def test_anchor_phrase_detection():
    assert vw.needs_anchor_view("Using the first image as the main viewpoint, ...")
    assert not vw.needs_anchor_view("How far is the chair from the table?")


# --- view selection -----------------------------------------------------------

def test_choose_views_keeps_required_and_stays_under_n():
    rng = random.Random(0)
    for _ in range(200):
        views = vw.choose_views(rng, 32, {5, 9})
        assert views is not None
        assert {5, 9} <= set(views)
        assert len(views) in (2, 4) and len(views) < 32
        assert views == sorted(views)


def test_choose_views_rejects_zero_privilege_and_oversized_requirements():
    rng = random.Random(0)
    assert vw.choose_views(rng, 3, {0, 1, 2}) is None  # student would see everything
    assert vw.choose_views(rng, 32, {1, 2, 3, 4, 5}) is None  # K_min above the menu
    assert len(vw.choose_views(rng, 2, ())) == 1  # N=2 still leaves a 2 -> 1 gap


def test_view_budget_distribution_covers_the_menu():
    rng = random.Random(1)
    budgets = {len(vw.choose_views(rng, 8, ())) for _ in range(200)}
    assert budgets == {1, 2, 4}


# --- SPAR markers -------------------------------------------------------------

def test_marker_discovery_reports_the_views_the_draw_function_touches():
    info = {
        "type": "distance_prediction_oo_mv",
        "point_img_idx": [[2, 0]],
        "red_point": [[100, 100]],
        "blue_point": [[200, 200]],
    }
    assert marked_view_indices(info, 3) == {0, 2}

    info_none = {"type": "obj_count"}
    assert marked_view_indices(info_none, 32) == set()


def test_marker_discovery_rejects_out_of_range_and_malformed_metadata():
    with pytest.raises(MarkerError):
        marked_view_indices({"type": "distance_prediction_oo_mv", "point_img_idx": [[9, 0]]}, 3)
    with pytest.raises(MarkerError):
        marked_view_indices({"type": "not_a_real_type"}, 3)


# --- end to end against the real annotations ----------------------------------

PLAN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "mvopsd", "plan", "plan.jsonl"
)


@pytest.mark.skipif(not os.path.exists(PLAN_PATH), reason="plan not built yet")
def test_plan_records_are_self_consistent():
    with open(PLAN_PATH) as handle:
        records = [json.loads(line) for _, line in zip(range(2000), handle)]

    for record in records:
        student = vw.build_prompt(record["header_style"], record["k_views"], record["student_body"])
        teacher = vw.build_prompt(record["header_style"], record["n_views"], record["body"])

        assert student.count(vw.IMAGE_TOKEN) == record["k_views"] == len(record["view_indices"])
        assert teacher.count(vw.IMAGE_TOKEN) == record["n_views"] == len(record["frames"])
        assert record["k_views"] < record["n_views"], "every sample must carry a real view gap"
        assert set(record["required_views"]) <= set(record["view_indices"])
        assert set(record["marked_views"]) <= set(record["view_indices"])

        # The student's question may only differ from the teacher's in frame numbering.
        assert vw.FRAME_REF_RE.sub("F", record["student_body"]) == vw.FRAME_REF_RE.sub("F", record["body"])
        assert vw.referenced_views(record["student_body"]) <= set(range(record["k_views"]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
