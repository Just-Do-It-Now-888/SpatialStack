"""Tests for the teacher keyframe sweep builder."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from build_teacher_keyframe_sweep import (  # noqa: E402
    build_pair,
    is_eligible,
    key_body_and_style,
    key_view_indices,
)
from mvopsd import views as vw  # noqa: E402


def _frame_record(**overrides):
    frames = [{"cache": f"spar/frame_{i}.jpg", "marked": i in {1, 3}} for i in range(4)]
    record = {
        "sample_id": "spar4/test",
        "source": "spar_32view",
        "dataset": "spar_234k",
        "subset": "scannet",
        "scene_id": "scene0001_00",
        "n_views": 4,
        "required_views": [1, 3],
        "marked_views": [1, 3],
        "header_style": "frame_labels",
        "body": "\nHow far is A (in Frame-1) from B (in Frame-3)?",
        "answer": "about 2 meters",
        "answer_view_sensitive": False,
        "question_type": "distance_prediction_oo_video",
        "frames": frames,
        "plan_position": 7,
    }
    record.update(overrides)
    return record


def test_is_eligible_requires_strict_subset():
    assert is_eligible(_frame_record())
    assert not is_eligible(_frame_record(required_views=[]))
    assert not is_eligible(_frame_record(required_views=[0, 1, 2, 3]))
    assert not is_eligible(_frame_record(required_views=[9]))


def test_key_body_renumbers_frame_refs():
    record = _frame_record()
    indices = key_view_indices(record)
    style, body = key_body_and_style(record, indices)
    assert indices == [1, 3]
    assert style == "frame_labels"
    assert body == "\nHow far is A (in Frame-0) from B (in Frame-1)?"


def test_build_pair_matches_placeholders_and_groups():
    record = _frame_record()
    group = f"{record['sample_id']}#{record['plan_position']}"
    rows = build_pair(record, "/tmp/cache", group)

    assert len(rows) == 2
    assert {row["extra_info"]["sweep_arm"] for row in rows} == {"full", "key"}
    assert all(row["extra_info"]["sweep_group"] == group for row in rows)

    full_row, key_row = rows
    assert full_row["extra_info"]["n_views_teacher"] == 4
    assert key_row["extra_info"]["n_views_teacher"] == 2
    assert full_row["teacher_prompt"][0]["content"].count(vw.IMAGE_TOKEN) == 4
    assert key_row["teacher_prompt"][0]["content"].count(vw.IMAGE_TOKEN) == 2
    assert "Frame-0" in key_row["teacher_prompt"][0]["content"]
    assert "Frame-3" not in key_row["teacher_prompt"][0]["content"]


def test_bare_header_stays_bare_without_frame_refs():
    record = _frame_record(
        header_style="bare",
        body="\nUsing the first image as the main viewpoint, where is the sofa?",
        required_views=[0, 2],
    )
    style, body = key_body_and_style(record, [0, 2])
    assert style == "bare"
    assert body == record["body"]


PLAN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "data",
    "mvopsd",
    "plan",
    "plan.jsonl",
)


@pytest.mark.skipif(not os.path.exists(PLAN_PATH), reason="plan not built yet")
def test_real_plan_pairs_are_self_consistent():
    eligible = []
    with open(PLAN_PATH) as handle:
        for position, line in enumerate(handle):
            record = json.loads(line)
            record["plan_position"] = position
            if record["source"] in {"spar_3view", "spar_32view"} and is_eligible(record):
                eligible.append(record)
            if len(eligible) >= 20:
                break

    assert eligible
    for record in eligible:
        rows = build_pair(record, "/tmp/cache", f"{record['sample_id']}#{record['plan_position']}")
        full_row, key_row = rows
        assert full_row["extra_info"]["sweep_views"] == record["n_views"]
        assert key_row["extra_info"]["sweep_views"] == len(record["required_views"])
        assert key_row["extra_info"]["sweep_views"] < full_row["extra_info"]["sweep_views"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
