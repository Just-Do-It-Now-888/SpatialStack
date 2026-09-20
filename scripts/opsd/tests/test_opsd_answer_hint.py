"""Answer-hint teacher message construction and equal-view guards."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "verl_pkg"))

from verl.utils.opsd_answer_hint import (  # noqa: E402
    DEFAULT_ANSWER_HINT_TEMPLATE,
    assert_equal_view_counts,
    count_images_in_messages,
    prepare_opsd_teacher_messages,
    resolve_ground_truth_answer,
)


def _student_messages(n_images=2, question="Where is the chair?"):
    content = []
    for i in range(n_images):
        content.append({"type": "image", "path": f"f{i}.jpg"})
        content.append({"type": "text", "text": f" Frame-{i}."})
    content.append({"type": "text", "text": f" {question}"})
    return [{"role": "user", "content": content}]


def test_prepare_appends_answer_only_on_teacher():
    answer = "B"
    student = _student_messages()
    teacher = prepare_opsd_teacher_messages(student, answer)
    student_text = "".join(item.get("text", "") for item in student[0]["content"] if item.get("type") == "text")
    teacher_text = "".join(item.get("text", "") for item in teacher[0]["content"] if item.get("type") == "text")
    assert answer not in student_text
    assert "reference solution" not in student_text
    assert answer in teacher_text
    assert "reference solution" in teacher_text
    assert count_images_in_messages(student) == count_images_in_messages(teacher) == 2
    assert student[0]["content"][-1]["text"] != teacher[0]["content"][-1]["text"]


def test_prepare_string_content():
    messages = [{"role": "user", "content": "<image><image> What color?"}]
    teacher = prepare_opsd_teacher_messages(messages, "red")
    assert messages[0]["content"].endswith("What color?")
    assert "red" in teacher[0]["content"]
    assert teacher[0]["content"].startswith("<image><image> What color?")


def test_resolve_ground_truth_prefers_reward_model():
    assert resolve_ground_truth_answer({"ground_truth": "A"}, {"answer": "B"}) == "A"
    assert resolve_ground_truth_answer(None, {"answer": " C "}) == "C"
    assert resolve_ground_truth_answer({"ground_truth": "  "}, {"answer": ""}) is None
    assert resolve_ground_truth_answer(None, None) is None


def test_assert_equal_view_counts_ok():
    extra = {"n_views_teacher": 32, "k_views_student": 32}
    assert_equal_view_counts(extra, 32, 32, sample_idx=0)


def test_assert_equal_view_counts_rejects_n_k_gap():
    extra = {"n_views_teacher": 32, "k_views_student": 16}
    try:
        assert_equal_view_counts(extra, 16, 16, sample_idx=3)
    except ValueError as exc:
        assert "n_views_teacher" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_assert_equal_view_counts_rejects_image_mismatch():
    extra = {"n_views_teacher": 3, "k_views_student": 3}
    try:
        assert_equal_view_counts(extra, 3, 2, sample_idx=1)
    except ValueError as exc:
        assert "same number of images" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_default_template_formats_answer():
    text = DEFAULT_ANSWER_HINT_TEMPLATE.format(answer="north of the sofa")
    assert "north of the sofa" in text
    assert "{answer}" not in text


if __name__ == "__main__":
    test_prepare_appends_answer_only_on_teacher()
    test_prepare_string_content()
    test_resolve_ground_truth_prefers_reward_model()
    test_assert_equal_view_counts_ok()
    test_assert_equal_view_counts_rejects_n_k_gap()
    test_assert_equal_view_counts_rejects_image_mismatch()
    test_default_template_formats_answer()
    print("ok")
