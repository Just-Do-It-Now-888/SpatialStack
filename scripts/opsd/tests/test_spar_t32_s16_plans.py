"""Teacher-32 / Student-16 SPAR plan builder."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from build_spar_t32_s16_plans import (  # noqa: E402
    Excluded,
    key_frames,
    retarget,
    student_views_for,
)
from mvopsd import views as vw  # noqa: E402


def _spar32(**overrides):
    record = {
        "sample_id": "spar32/test",
        "source": "spar_32view",
        "n_views": 32,
        "required_views": [2, 10],
        "body": "How far is A (in Frame-2) from B (in Frame-10)?",
        "teacher_body": "renumbered cap-8 body should be ignored",
        "teacher_view_indices": [0, 2, 10, 13, 21, 25, 26, 28],
        "header_style": "frame_labels",
        "view_selection": "oracle_guided",
        "privilege_bucket": "old",
        "frames": [{"cache": f"f{i}.jpg"} for i in range(32)],
        "question_type": "distance_prediction_oo_video",
        "scene_id": "scene0001_00",
    }
    record.update(overrides)
    return record


def _spar3(**overrides):
    record = {
        "sample_id": "spar3/test",
        "source": "spar_3view",
        "n_views": 3,
        "required_views": [0, 2],
        "body": "In Frame-0 and Frame-2, which is closer?",
        "teacher_view_indices": [0, 1, 2],
        "header_style": "frame_labels",
        "frames": [{"cache": f"f{i}.jpg"} for i in range(3)],
        "question_type": "spatial_imagination_oc_video",
        "scene_id": "scene0002_00",
    }
    record.update(overrides)
    return record


def test_spar32_student_is_16_and_keeps_keys():
    record = _spar32()
    views = student_views_for(record)
    assert len(views) == 16
    assert set(record["required_views"]) <= set(views)
    assert views == sorted(views)


def test_spar32_empty_required_pins_frame_zero():
    record = _spar32(required_views=[])
    assert key_frames(record) == [0]
    views = student_views_for(record)
    assert 0 in views
    assert len(views) == 16


def test_spar32_too_many_keys_excluded():
    record = _spar32(required_views=list(range(17)))
    try:
        student_views_for(record)
    except Excluded as exc:
        assert "required_views_exceed_student_16" in str(exc)
    else:
        raise AssertionError("expected Excluded")


def test_spar32_retarget_resets_capped_teacher():
    record = _spar32()
    out = retarget(record, student_views_for(record))
    assert out["teacher_view_indices"] == list(range(32))
    assert out["n_views_teacher_effective"] == 32
    assert out["teacher_body"] == record["body"]
    assert out["k_views"] == 16
    assert out["view_selection"] == "oracle_key_plus_farthest_extra"
    refs = vw.referenced_views(out["student_body"])
    assert refs <= set(range(16))
    assert "Frame-2" not in out["student_body"] or 2 in out["view_indices"]
    student_map = {old: new for new, old in enumerate(out["view_indices"])}
    assert f"Frame-{student_map[2]}" in out["student_body"]
    assert f"Frame-{student_map[10]}" in out["student_body"]


def test_spar3_full_album_both_sides():
    record = _spar3()
    views = student_views_for(record)
    assert views == [0, 1, 2]
    out = retarget(record, views)
    assert out["teacher_view_indices"] == [0, 1, 2]
    assert out["k_views"] == 3
    assert out["privilege_bucket"] == "t3_s3"
    assert set(record["required_views"]) <= set(out["view_indices"])


if __name__ == "__main__":
    test_spar32_student_is_16_and_keeps_keys()
    test_spar32_empty_required_pins_frame_zero()
    test_spar32_too_many_keys_excluded()
    test_spar32_retarget_resets_capped_teacher()
    test_spar3_full_album_both_sides()
    print("ok")
