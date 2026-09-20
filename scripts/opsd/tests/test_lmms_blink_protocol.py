#!/usr/bin/env python3
"""Guard the two BLINK protocols in src/lmms_eval/tasks/blink.

    python3 scripts/opsd/tests/test_lmms_blink_protocol.py

Default stays ``original`` (letter-only pre_prompt + start-of-string letter)
so the 13.05 dump remains one env-var away. ``spatialstack`` is the VSI-Bench
MCA protocol: boxed last-line suffix, no blank-line stop, boxed-primary +
last-line option extract. The two must not share a dump.
"""

from __future__ import annotations

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

TASK_DIR = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "blink")
TASK_YAML = os.path.join(TASK_DIR, "blink_multi_view_reasoning.yaml")

failures: list[str] = []

DOC = {
    "idx": 0,
    "prompt": "Which image is taken from a viewpoint closer to the object?\n(A) Image A\n(B) Image B",
    "choices": ["Image A", "Image B"],
    "answer": "(A)",
    "sub_task": "multi_view_reasoning",
}

LETTER_PRE = (
    "Return exactly one uppercase option letter from the given choices ({}). "
    "Do not output any explanation, punctuation, or extra text.\n"
)
MCA_POST = "Answer with the option's letter from the given choices directly."


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def load_task_utils(protocol: str):
    os.environ["BLINK_PROTOCOL"] = protocol
    spec = importlib.util.spec_from_file_location(f"blink_utils_{protocol}", os.path.join(TASK_DIR, "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task_config(protocol: str) -> dict:
    os.environ["BLINK_PROTOCOL"] = protocol
    from lmms_eval.utils import load_yaml_config

    return load_yaml_config(TASK_YAML)


def test_generation_budget() -> None:
    print("\n[1] the answer budget follows the protocol")
    for key in ("BLINK_DO_SAMPLE", "BLINK_TEMPERATURE", "BLINK_TOP_P", "BLINK_SEED"):
        os.environ.pop(key, None)
    original = load_task_config("original")["generation_kwargs"]
    ours = load_task_config("spatialstack")["generation_kwargs"]

    check(original["max_new_tokens"] == 1024, f"original generates up to {original['max_new_tokens']} tokens")
    check(ours["max_new_tokens"] == 1024, f"spatialstack generates up to {ours['max_new_tokens']} tokens")
    for name, kwargs in (("original", original), ("spatialstack", ours)):
        check(
            kwargs["do_sample"] is False and kwargs["temperature"] == 0,
            f"{name} decoding stays greedy",
        )

    check(ours.get("until") == [], "spatialstack stops only at the token budget")
    check(original.get("until") is None, "original still inherits the yaml-default until")

    os.environ["BLINK_DO_SAMPLE"] = "1"
    os.environ["BLINK_SEED"] = "20260904"
    try:
        sampled = load_task_utils("spatialstack").BLINK_GENERATION_KWARGS
    finally:
        os.environ.pop("BLINK_DO_SAMPLE", None)
        os.environ.pop("BLINK_SEED", None)
    check(
        sampled["do_sample"] is True
        and sampled["temperature"] == 1.0
        and sampled["top_p"] == 0.8
        and sampled["seed"] == 20260904
        and sampled.get("until") == [],
        "BLINK_DO_SAMPLE=1 is t=1.0 / top_p=0.8 / seed, until still empty",
    )

    metrics = [entry["metric"] for entry in load_task_config("original")["metric_list"]]
    check(metrics == ["blink_acc"], f"metrics reported: {metrics}")


def test_prompt() -> None:
    print("\n[2] prompt wording")
    from vsibench_scoring import BOXED_LASTLINE_SUFFIX

    kwargs = {"pre_prompt": LETTER_PRE, "mca_post_prompt": MCA_POST, "post_prompt": ""}
    original = load_task_utils("original").blink_doc_to_text(DOC, kwargs)
    ours = load_task_utils("spatialstack").blink_doc_to_text(DOC, kwargs)

    check(original.startswith(LETTER_PRE.format("A, B")), "original keeps the letter-only pre_prompt")
    check(r"\boxed{}" not in original and "last line" not in original, "original does not grow the boxed suffix")
    check(DOC["prompt"] in original, "original still concatenates the dataset prompt")

    check(not ours.startswith("Return exactly one uppercase"), "spatialstack drops the letter-only pre_prompt")
    check(r"\boxed{}" in ours and "last line" in ours, "spatialstack asks for a boxed answer on the last line")
    check(ours.endswith(MCA_POST + BOXED_LASTLINE_SUFFIX), "boxed suffix is concatenated on the MCA line, as in VSI val")
    check("\nThe final answer" not in ours, "boxed suffix is not a separate prompt line")
    check(DOC["prompt"] in ours, "spatialstack still includes the dataset prompt")


def test_parsing() -> None:
    print("\n[3] reading the answer out of a response")
    prose = "To determine which viewpoint is closer, we need to compare the two images."
    based = "Based on the two images, the answer is B."
    last_letter = "Therefore the closer view is Image A.\n\n(A)"
    boxed = "reasoning\n\\boxed{A}"

    original = load_task_utils("original")
    ours = load_task_utils("spatialstack")

    orig_prose = original.blink_process_results(dict(DOC), [prose])["blink_acc"]
    check(orig_prose["pred_parsed"] == "T", f"original: 'To determine...' reads as {orig_prose['pred_parsed']!r}")

    orig_based = original.blink_process_results(dict(DOC), [based])["blink_acc"]
    check(orig_based["pred_parsed"] == "B", f"original: 'Based on...' reads as {orig_based['pred_parsed']!r}")

    orig_last = original.blink_process_results(dict(DOC), [last_letter])["blink_acc"]
    check(orig_last["pred_parsed"] == "T", "original still only looks at the start of the string")

    stack_prose = ours.blink_process_results(dict(DOC), [prose])["blink_acc"]
    check(stack_prose["pred_parsed"] == "", f"spatialstack: cut-off CoT is unanswered ({stack_prose['pred_parsed']!r})")
    check(stack_prose["answered"] == 0, "spatialstack: answered flag on the cut-off response is 0")

    stack_last = ours.blink_process_results(dict(DOC), [last_letter])["blink_acc"]
    check(stack_last["pred_parsed"] == "A", f"spatialstack: last-line (A) reads as {stack_last['pred_parsed']!r}")
    check(stack_last["is_correct"] is True, "spatialstack last-line (A) is scored correct")

    stack_boxed = ours.blink_process_results(dict(DOC), [boxed])["blink_acc"]
    check(stack_boxed["pred_parsed"] == "A", f"spatialstack: \\boxed{{A}} reads as {stack_boxed['pred_parsed']!r}")
    check(stack_boxed["boxed_present"] == 1, "spatialstack records a closed box")


def test_default_is_original() -> None:
    print("\n[4] default protocol is original")
    os.environ.pop("BLINK_PROTOCOL", None)
    spec = importlib.util.spec_from_file_location("blink_utils_default", os.path.join(TASK_DIR, "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check(module.BLINK_PROTOCOL == "original", f"unset env defaults to {module.BLINK_PROTOCOL!r}")


def main() -> None:
    original = os.environ.get("BLINK_PROTOCOL")
    original_sample = {
        k: os.environ.get(k) for k in ("BLINK_DO_SAMPLE", "BLINK_TEMPERATURE", "BLINK_TOP_P", "BLINK_SEED")
    }
    try:
        test_generation_budget()
        test_prompt()
        test_parsing()
        test_default_is_original()
    finally:
        if original is None:
            os.environ.pop("BLINK_PROTOCOL", None)
        else:
            os.environ["BLINK_PROTOCOL"] = original
        for key, value in original_sample.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
