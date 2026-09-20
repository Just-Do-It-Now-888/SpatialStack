#!/usr/bin/env python3
"""Regression tests for terse-answer and MCA-tail trust routing."""

from __future__ import annotations

import os
import sys

_OPSD = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _OPSD not in sys.path:
    sys.path.insert(0, _OPSD)

from vsibench_scoring import extract_boxed, is_terse_answer, is_trustable_mca_tail  # noqa: E402

OPTIONS = [
    "A. back-left",
    "B. back-right",
    "C. front-left",
    "D. front-right",
]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    check(is_terse_answer("173"), "bare integer")
    check(is_terse_answer("2.8"), "bare float")
    check(is_terse_answer("B"), "bare letter")
    check(is_terse_answer("b"), "bare letter lower")
    check(not is_terse_answer("The answer is 173"), "framed answer is not terse")
    check(not is_terse_answer(""), "empty")
    check(not is_terse_answer("Image 178: Kitchen"), "enumeration is not terse")
    check(is_terse_answer("221"), "short wrong number still terse")
    check(
        is_terse_answer("\n\n\n\n215"),
        "thinking wrapper stripped before terse check",
    )
    check(extract_boxed(r"\boxed{200}") is not None, "boxed still separate")
    check(not is_terse_answer(r"\boxed{200}"), "boxed string alone is not terse whole-string")

    cot_tail_c = (
        "Based on this logical deduction, the stool is to the front-left of the viewer.\n\nC"
    )
    check(is_terse_answer(cot_tail_c), "last-line bare letter after CoT")
    check(
        is_terse_answer("To determine the count of pipes is 0.\n\n4"),
        "last-line lone number after CoT",
    )
    check(
        not is_terse_answer("Image 178: Kitchen."),
        "last-line image label is not terse",
    )

    check(is_trustable_mca_tail("C. front-left", OPTIONS), "letter+body matches option C")
    check(
        not is_trustable_mca_tail("A. front-left", OPTIONS),
        "letter/body mismatch is not trusted",
    )
    check(is_trustable_mca_tail(cot_tail_c, OPTIONS), "bare last-line C is trusted")
    check(is_trustable_mca_tail("(C)", OPTIONS), "lone parenthesised letter is trusted")
    check(
        not is_trustable_mca_tail("Then the answer would be C.", OPTIONS),
        "hypothetical would-be C is not trusted",
    )
    check(is_trustable_mca_tail("front-left", OPTIONS), "unique option body on last line")
    check(
        not is_trustable_mca_tail("C. front-left", None),
        "letter+body needs options for consistency",
    )
    print("test_terse_trust: OK")


if __name__ == "__main__":
    main()
