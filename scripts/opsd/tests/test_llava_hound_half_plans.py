"""Teacher-full / Student-half LLaVA-Hound plan builder."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from build_llava_hound_half_plans import Excluded, retarget, student_views_for  # noqa: E402
from mvopsd import views as vw  # noqa: E402


def _hound(n_views=8, **overrides):
    record = {
        "sample_id": "llava_hound_64k/test",
        "source": "llava_hound_64k",
        "n_views": n_views,
        "required_views": [],
        "body": "Describe the video.",
        "teacher_body": "old capped body",
        "teacher_view_indices": list(range(n_views)),
        "header_style": "concat",
        "frames": [{"cache": f"f{i}.jpg"} for i in range(n_views)],
        "question_type": None,
        "scene_id": "1013431592",
    }
    record.update(overrides)
    return record


def test_eight_views_even_half():
    assert student_views_for(_hound(8)) == [0, 2, 4, 6]


def test_four_views_kept():
    assert student_views_for(_hound(4)) == [0, 2]


def test_seven_views_floor_half():
    views = student_views_for(_hound(7))
    assert len(views) == 3
    assert views == sorted(views)


def test_n_views_lt_2_excluded():
    try:
        student_views_for(_hound(1))
    except Excluded as exc:
        assert "n_views_lt_2" in str(exc)
    else:
        raise AssertionError("expected Excluded")


def test_retarget_full_teacher_no_key_pin():
    record = _hound(8)
    out = retarget(record, student_views_for(record))
    assert out["teacher_view_indices"] == list(range(8))
    assert out["teacher_body"] == record["body"]
    assert out["k_views"] == 4
    assert out["view_selection"] == "linspace_half"
    assert out["privilege_bucket"] == "half"
    assert vw.referenced_views(out["student_body"]) == set()


if __name__ == "__main__":
    test_eight_views_even_half()
    test_four_views_kept()
    test_seven_views_floor_half()
    test_n_views_lt_2_excluded()
    test_retarget_full_teacher_no_key_pin()
    print("ok")
