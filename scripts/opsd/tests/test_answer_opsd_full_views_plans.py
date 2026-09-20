"""Equal-view Answer-OPSD plan builder."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from build_answer_opsd_full_views_plans import Excluded, retarget  # noqa: E402
from mvopsd import views as vw  # noqa: E402


def _spar32(**overrides):
    record = {
        "sample_id": "spar32/test",
        "source": "spar_32view",
        "n_views": 32,
        "required_views": [2, 10],
        "body": "How far is A (in Frame-2) from B (in Frame-10)?",
        "student_body": "How far is A (in Frame-0) from B (in Frame-1)?",
        "teacher_body": "renumbered cap-8 body should be ignored",
        "teacher_view_indices": [0, 2, 10, 13, 21, 25, 26, 28],
        "view_indices": [2, 10],
        "k_views": 2,
        "header_style": "frame_labels",
        "view_selection": "oracle_guided",
        "privilege_bucket": "old",
        "frames": [{"cache": f"f{i}.jpg"} for i in range(32)],
        "question_type": "distance_prediction_oo_video",
        "scene_id": "scene0001_00",
        "answer": "2.4 meters",
        "dataset": "spar",
        "subset": "32view",
        "answer_view_sensitive": False,
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
        "view_indices": [0, 2],
        "k_views": 2,
        "header_style": "frame_labels",
        "frames": [{"cache": f"f{i}.jpg"} for i in range(3)],
        "question_type": "spatial_imagination_oc_video",
        "scene_id": "scene0002_00",
        "answer": "A",
        "dataset": "spar",
        "subset": "3view",
        "answer_view_sensitive": False,
    }
    record.update(overrides)
    return record


def test_spar32_full_album_keeps_original_body():
    record = _spar32()
    out = retarget(record)
    assert out["view_indices"] == list(range(32))
    assert out["teacher_view_indices"] == list(range(32))
    assert out["k_views"] == 32
    assert out["n_views_teacher_effective"] == 32
    assert out["student_body"] == record["body"]
    assert out["teacher_body"] == record["body"]
    assert out["view_selection"] == "full_album"
    assert out["privilege_bucket"] == "answer_equal_views"
    assert vw.referenced_views(out["student_body"]) <= set(range(32))
    assert "Frame-2" in out["student_body"]
    assert "Frame-10" in out["student_body"]
    student = vw.build_prompt(out["header_style"], out["k_views"], out["student_body"])
    teacher = vw.build_prompt(out["header_style"], out["n_views"], out["teacher_body"])
    assert student.count(vw.IMAGE_TOKEN) == 32
    assert teacher.count(vw.IMAGE_TOKEN) == 32
    assert student == teacher


def test_spar3_full_album():
    out = retarget(_spar3())
    assert out["view_indices"] == [0, 1, 2]
    assert out["teacher_view_indices"] == [0, 1, 2]
    assert out["k_views"] == 3
    assert out["student_body"] == _spar3()["body"]


def test_dangling_frame_excluded():
    try:
        retarget(_spar32(body="See Frame-99.", n_views=32))
    except Excluded as exc:
        assert "dangling_frame_refs" in str(exc)
    else:
        raise AssertionError("expected Excluded")


def test_empty_album_excluded():
    try:
        retarget(_spar32(n_views=0, frames=[], body="No frames."))
    except Excluded as exc:
        assert "n_views_lt_1" in str(exc)
    else:
        raise AssertionError("expected Excluded")


def test_build_row_equal_views():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from write_mvopsd_parquet import build_row

    out = retarget(_spar3())
    row = build_row(out, cache_root="/tmp/views", arm="main")
    extra = row["extra_info"]
    assert extra["n_views_teacher"] == extra["k_views_student"] == 3
    assert extra["privilege_bucket"] == "answer_equal_views"
    assert extra["answer"] == "A"
    assert len(row["images"]) == len(row["teacher_images"]) == 3
    assert row["prompt"] == row["teacher_prompt"]
    assert row["reward_model"]["ground_truth"] == "A"


if __name__ == "__main__":
    test_spar32_full_album_keeps_original_body()
    test_spar3_full_album()
    test_dangling_frame_excluded()
    test_empty_album_excluded()
    test_build_row_equal_views()
    print("ok")
