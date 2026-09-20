"""Tests for the nested key-plus-extra teacher sweep."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from build_teacher_key_plus_sweep import (  # noqa: E402
    extra_order,
    remaining_views,
    views_for_extra,
    build_curve,
    can_add_extras,
)
from mvopsd import views as vw  # noqa: E402


def _record(**overrides):
    frames = [{"cache": f"spar/frame_{i}.jpg", "marked": i in {2, 10}} for i in range(32)]
    record = {
        "sample_id": "spar32/test",
        "source": "spar_32view",
        "dataset": "spar_234k",
        "subset": "scannet",
        "scene_id": "scene0001_00",
        "n_views": 32,
        "required_views": [2, 10],
        "marked_views": [2, 10],
        "header_style": "frame_labels",
        "body": "\nHow far is A (in Frame-2) from B (in Frame-10)?",
        "answer": "about 2 meters",
        "answer_view_sensitive": False,
        "question_type": "distance_prediction_oo_video",
        "frames": frames,
        "plan_position": 3,
    }
    record.update(overrides)
    return record


def test_extra_order_is_nested_and_avoids_keys():
    required = [2, 10]
    remaining = [i for i in range(32) if i not in required]
    extras = extra_order(required, remaining, 10)
    assert len(extras) == 10
    assert len(set(extras)) == 10
    assert not set(extras) & set(required)
    for k in range(1, 11):
        prefix = extra_order(required, remaining, k)
        assert prefix == extras[:k]


def test_views_always_keep_key_frames():
    record = _record()
    for extra_n in range(0, 11):
        views = views_for_extra(record, extra_n)
        assert set(record["required_views"]) <= set(views)
        assert len(views) == len(record["required_views"]) + extra_n
        assert views == sorted(views)


def test_build_curve_renumbers_and_pairs():
    record = _record()
    group = f"{record['sample_id']}#{record['plan_position']}"
    rows = build_curve(record, "/tmp/cache", group)
    assert len(rows) == 11
    assert [row["extra_info"]["extra_added"] for row in rows] == list(range(11))
    groups = {row["extra_info"]["sweep_group"] for row in rows}
    assert groups == {group}

    key_only = rows[0]
    plus_ten = rows[10]
    assert key_only["extra_info"]["n_views_teacher"] == 2
    assert plus_ten["extra_info"]["n_views_teacher"] == 12
    prompt = key_only["teacher_prompt"][0]["content"]
    assert prompt.count(vw.IMAGE_TOKEN) == 2
    assert "Frame-0" in prompt and "Frame-1" in prompt
    assert "Frame-10" not in prompt

    smaller = set(key_only["extra_info"]["view_indices_teacher"])
    for row in rows[1:]:
        assert smaller <= set(row["extra_info"]["view_indices_teacher"])
        smaller = set(row["extra_info"]["view_indices_teacher"])


def test_cannot_add_extras_when_album_too_small():
    record = _record(n_views=8, required_views=[0, 1, 2], frames=[{"cache": f"f{i}.jpg"} for i in range(8)])
    assert not can_add_extras(record, max_extra=10)
    assert remaining_views(record) == [3, 4, 5, 6, 7]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
