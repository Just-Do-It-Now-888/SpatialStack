"""Answer-hint teacher construction for equal-view OPSD.

Kept independent of RayPPOTrainer so unit tests do not import the full PPO stack.
The trainer calls these helpers from the `teacher_prompt_mode=answer_hint` path.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

DEFAULT_ANSWER_HINT_TEMPLATE = (
    "\n\nHere is a reference solution to this problem:\n{answer}\n\n"
    "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
)


def resolve_ground_truth_answer(reward_model_info: Any, extra_info: Any) -> Optional[str]:
    answer = None
    if isinstance(reward_model_info, dict):
        answer = reward_model_info.get("ground_truth", None)
    if answer is None and isinstance(extra_info, dict):
        answer = extra_info.get("answer", None)
    if answer is None:
        return None
    text = str(answer).strip()
    return text if text else None


def count_images_in_messages(messages: list[dict]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
    return count


def prepare_opsd_teacher_messages(
    raw_prompt_messages: list[dict],
    answer: str,
    hint_template: str = DEFAULT_ANSWER_HINT_TEMPLATE,
) -> list[dict]:
    teacher_messages = deepcopy(raw_prompt_messages)
    if not teacher_messages:
        raise ValueError("raw_prompt_messages is empty; cannot append an answer hint")
    last_msg = teacher_messages[-1]
    content = last_msg["content"]
    hint_suffix = hint_template.format(answer=answer)

    if isinstance(content, list):
        teacher_messages[-1]["content"] = list(content) + [{"type": "text", "text": hint_suffix}]
    elif isinstance(content, str):
        teacher_messages[-1]["content"] = content + hint_suffix
    else:
        raise TypeError(f"Unsupported message content type: {type(content)}")
    return teacher_messages


def assert_equal_view_counts(
    extra_info: Any,
    student_n_images: int,
    teacher_n_images: int,
    sample_idx: int,
) -> None:
    if student_n_images != teacher_n_images:
        raise ValueError(
            "answer_hint requires the teacher to see the same number of images as the student, "
            f"got student={student_n_images} teacher={teacher_n_images} sample_idx={sample_idx}"
        )
    if not isinstance(extra_info, dict):
        return
    n_teacher = extra_info.get("n_views_teacher")
    k_student = extra_info.get("k_views_student")
    if n_teacher is None or k_student is None:
        return
    n_teacher_i = int(n_teacher)
    k_student_i = int(k_student)
    if n_teacher_i != k_student_i:
        raise ValueError(
            "answer_hint requires extra_info.n_views_teacher == k_views_student, "
            f"got teacher={n_teacher_i} student={k_student_i} sample_idx={sample_idx}"
        )
    if student_n_images and student_n_images != k_student_i:
        raise ValueError(
            "answer_hint image count does not match extra_info.k_views_student, "
            f"got images={student_n_images} k_views_student={k_student_i} sample_idx={sample_idx}"
        )
