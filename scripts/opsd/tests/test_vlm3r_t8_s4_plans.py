"""Teacher-8 / Student-4 VLM-3R plan builder."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from build_vlm3r_t8_s4_plans import (  # noqa: E402
    Excluded,
    key_frames,
    retarget,
    student_views_for,
)
from mvopsd import views as vw  # noqa: E402


def _vlm8(**overrides):
    record = {
        "sample_id": "vlm3r/test",
        "source": "vlm3r_scannet",
        "n_views": 8,
        "required_views": [1, 6],
        "body": "Which option matches Frame-1 and Frame-6?",
        "teacher_body": "old capped body",
        "teacher_view_indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "header_style": "frame_labels",
        "frames": [{"cache": f"f{i}.jpg"} for i in range(8)],
        "question_type": "mcq",
        "scene_id": "scene0001_00",
    }
    record.update(overrides)
    return record


def test_student_is_4_and_keeps_keys():
    views = student_views_for(_vlm8())
    assert len(views) == 4
    assert {1, 6} <= set(views)
    assert views == sorted(views)


def test_empty_required_pins_frame_zero():
    record = _vlm8(required_views=[])
    assert key_frames(record) == [0]
    views = student_views_for(record)
    assert 0 in views
    assert len(views) == 4


def test_too_many_keys_excluded():
    try:
        student_views_for(_vlm8(required_views=[0, 1, 2, 3, 4]))
    except Excluded as exc:
        assert "required_views_exceed_student_4" in str(exc)
    else:
        raise AssertionError("expected Excluded")


def test_n_views_le_4_excluded():
    try:
        student_views_for(_vlm8(n_views=4, required_views=[], frames=[{"cache": f"f{i}.jpg"} for i in range(4)]))
    except Excluded as exc:
        assert "n_views_le_student_budget" in str(exc)
    else:
        raise AssertionError("expected Excluded")


def test_retarget_full_teacher_and_renumber():
    record = _vlm8()
    out = retarget(record, student_views_for(record))
    assert out["teacher_view_indices"] == list(range(8))
    assert out["teacher_body"] == record["body"]
    assert out["k_views"] == 4
    assert out["privilege_bucket"] == "t8_s4"
    refs = vw.referenced_views(out["student_body"])
    assert refs <= set(range(4))
    mapping = {old: new for new, old in enumerate(out["view_indices"])}
    assert f"Frame-{mapping[1]}" in out["student_body"]
    assert f"Frame-{mapping[6]}" in out["student_body"]


if __name__ == "__main__":
    test_student_is_4_and_keeps_keys()
    test_empty_required_pins_frame_zero()
    test_too_many_keys_excluded()
    test_n_views_le_4_excluded()
    test_retarget_full_teacher_and_renumber()
    print("ok")
