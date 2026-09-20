#!/usr/bin/env python3
r"""Guard the \boxed{} answer channel.

    python3 scripts/opsd/tests/test_boxed_extraction.py

The box is a *diagnostic* column: it records whether the model put its answer
where the prompt asked, and what the run would score if only the box counted.
It never changes the reported VSI-Bench number, so what has to stay true is
narrower than for the main parser but not weaker:

1. the box is read with a brace-balanced scan, so a nested `\boxed{\frac{1}{2}}`
   survives and an unclosed `\boxed{4` reads as absent rather than as "4";
2. a box that is followed by more text is still found. Upstream scores only the
   last 300 characters, which is right for MATH-500 and wrong here: our failure
   mode is a response that answers and then keeps going to the token budget;
3. the gold answer, wrapped in a box, scores 1.0 for every row of VSI-Bench
   (LESSON-021 -- a scorer that cannot recognise its own gold is measuring
   something else);
4. strings the prompt itself introduces (`Frame-4`, `Object169`, `scene0555_00`)
   never become an answer (LESSON-023);
5. a response with no box is answered=0 and scores 0 in the box column, and is
   left untouched in the rule column. "Did not comply" and "answered wrongly"
   have to stay distinguishable, which is the whole reason for the probe;
6. the vLLM harness prompt suffix is empty by default, so every number already
   published from that harness is reproducible without passing a flag
   (LESSON-013). lmms-eval ``spatialstack`` now appends the boxed last-line
   suffix; rebuilds of the val parquet still concatenate it via ``--prompt-suffix``.
"""

from __future__ import annotations

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

TASK_DIR = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench")

failures: list[str] = []

MCA_DOC = {
    "question_type": "object_rel_direction_easy",
    "question": "If I stand by the sofa and face the tv, is the door on my left or right?",
    "options": ["A. right", "B. left"],
    "ground_truth": "B",
}
NA_DOC = {
    "question_type": "room_size_estimation",
    "question": "What is the size of this room (in square meters)?",
    "options": None,
    "ground_truth": "30.9",
}


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def load_task_utils():
    os.environ["VSIBENCH_PROTOCOL"] = "spatialstack"
    spec = importlib.util.spec_from_file_location("vsibench_utils_boxed", os.path.join(TASK_DIR, "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extraction() -> None:
    print("\n[1] finding the box")
    import vsibench_scoring as scoring

    cases = [
        ("plain", r"The room is about 30.9 square meters, so \boxed{30.9}.", "30.9"),
        ("nested braces", r"\boxed{\frac{1}{2}} is the ratio.", r"\frac{1}{2}"),
        ("units inside", r"\boxed{30.9 \text{ m}^2}", r"30.9 \text{ m}^2"),
        ("no box", "The answer is 30.9 square meters.", None),
        ("unclosed box", r"I estimate \boxed{30.9", None),
        ("last of several", r"maybe \boxed{12}, or \boxed{30.9} on reflection", "30.9"),
        ("empty box", r"\boxed{}", ""),
    ]
    for name, text, want in cases:
        got = scoring.extract_boxed(text)
        check(got == want, f"{name}: reads {got!r} (want {want!r})")


def test_box_survives_a_rambling_tail() -> None:
    """Upstream's 300-character window would miss this; ours must not."""
    print("\n[2] a box followed by 1,000 characters of enumeration")
    import vsibench_scoring as scoring

    tail = " ".join(f"Image {i} shows another portion of the wall." for i in range(1, 60))
    text = r"Counting the chairs across the frames gives four. \boxed{4} " + tail
    check(len(text) > 1500, f"the tail is {len(text)} characters, well past a 300-character window")
    check(scoring.extract_boxed(text) == "4", f"still reads {scoring.extract_boxed(text)!r}")

    # And the converse: the trailing prose must not supply an answer of its own
    # when there is no box at all.
    check(scoring.extract_boxed(tail) is None, "the enumeration alone yields no box")


def test_prompt_strings_are_not_answers() -> None:
    print("\n[3] strings the prompt introduces are not boxes")
    import vsibench_scoring as scoring

    for name, text in (
        ("frame label", "In Frame-4 there is another pile of clothes."),
        ("object id", "At a depth of about 6.7 meters, Object169's center is visible."),
        ("scene id", "The video is scene0555_00 from the ScanNet split."),
        ("literal word", "I will put the boxed answer at the end."),
        ("latex without box", r"The ratio is \frac{1}{2}."),
    ):
        got = scoring.extract_boxed(text)
        check(got is None, f"{name}: yields no box (got {got!r})")


def test_gold_self_consistency() -> None:
    """Every VSI-Bench gold answer, boxed, must score 1.0."""
    print("\n[4] gold answers wrapped in a box score 1.0")
    import vsibench_eval_core as core
    import vsibench_scoring as scoring

    task = load_task_utils()
    if not os.path.isdir(core.DEFAULT_SNAPSHOT):
        print("  skip (VSI-Bench snapshot not present)")
        return

    docs = core.load_docs(core.DEFAULT_SNAPSHOT)
    bad: list[tuple] = []
    for doc in docs:
        response = "\\boxed{" + str(doc["ground_truth"]) + "}"
        content = scoring.normalize_boxed(scoring.extract_boxed(response))
        scored = task.vsibench_process_results(dict(doc), [content])["vsibench_score"]
        metric = "accuracy" if doc["question_type"] in task.MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
        if scored.get(metric, 0.0) < 1.0 - 1e-9:
            bad.append((doc["question_type"], doc["ground_truth"], scored.get("parsed_answer")))
    check(
        not bad,
        f"all {len(docs)} gold answers round-trip through the box "
        + (f"({len(bad)} failed, e.g. {bad[:3]})" if bad else ""),
    )

    # The same check for the shapes a model actually writes, which the bare gold
    # does not cover: an option letter with the option text next to it, and a
    # number carrying its unit.
    for name, doc, response, metric in (
        ("letter with text", MCA_DOC, r"\boxed{B. left}", "accuracy"),
        ("option text only", MCA_DOC, r"\boxed{left}", "accuracy"),
        ("bold letter", MCA_DOC, r"\boxed{\textbf{B}}", "accuracy"),
        ("number with unit", NA_DOC, r"\boxed{30.9 square meters}", "MRA:.5:.95:.05"),
        ("latex unit", NA_DOC, r"\boxed{30.9\,\text{m}^2}", "MRA:.5:.95:.05"),
        ("latex unit braced exponent", NA_DOC, r"\boxed{30.9\ \mathrm{m}^{2}}", "MRA:.5:.95:.05"),
        ("dollar delimited", NA_DOC, r"\boxed{$30.9$}", "MRA:.5:.95:.05"),
        ("approx sign", NA_DOC, r"\boxed{\approx 30.9}", "MRA:.5:.95:.05"),
    ):
        content = scoring.normalize_boxed(scoring.extract_boxed(response))
        scored = task.vsibench_process_results(dict(doc), [content])["vsibench_score"]
        check(
            scored.get(metric, 0.0) >= 1.0 - 1e-9,
            f"{name}: {response} normalises to {content!r} and scores {scored.get(metric)}",
        )

    # A fraction has to be divided out rather than read as its denominator.
    print("\n[4b] fractions")
    for response, want in ((r"\boxed{\frac{1}{2}}", 0.5), (r"\boxed{\dfrac{3}{4}}", 0.75)):
        content = scoring.normalize_boxed(scoring.extract_boxed(response))
        got = scoring.extract_vsibench_number(content)
        check(got == want, f"{response} normalises to {content!r} and reads {got!r} (want {want})")


def test_non_compliance_is_visible() -> None:
    print("\n[5] no box is answered=0, and the rule column is untouched")
    import vsibench_scoring as scoring

    task = load_task_utils()
    # A response that answers correctly but ignores the format. The rule parser
    # must still read it; the box column must record it as absent.
    text = "The room is roughly 30.9 square meters."
    check(scoring.extract_boxed(text) is None, "no box found")

    boxed = task.vsibench_process_results(dict(NA_DOC), [""])["vsibench_score"]
    check(
        boxed["answered"] == 0 and boxed["MRA:.5:.95:.05"] == 0.0,
        "the box column scores it 0 with answered=0",
    )
    rule = task.vsibench_process_results(dict(NA_DOC), [text])["vsibench_score"]
    check(
        rule["answered"] == 1 and rule["MRA:.5:.95:.05"] == 1.0,
        "the rule column still scores it 1.0",
    )


def test_suffix_defaults_off() -> None:
    print("\n[6] the vLLM prompt suffix is opt-in; lmms-eval spatialstack is not")
    import vsibench_eval_core as core
    from vsibench_scoring import BOXED_LASTLINE_SUFFIX

    cfg = core.EvalConfig(model="unused")
    check(cfg.prompt_suffix == "", f"EvalConfig().prompt_suffix is {cfg.prompt_suffix!r}")

    task = load_task_utils()
    for doc in (MCA_DOC, NA_DOC):
        plain = task.vsibench_doc_to_text_plain(dict(doc), core.LMMS_KWARGS)
        check(
            r"\boxed{}" not in plain,
            f"{doc['question_type']}: the plain prompt has no boxed suffix",
        )
        check(
            task.vsibench_doc_to_text(dict(doc), core.LMMS_KWARGS) == plain + BOXED_LASTLINE_SUFFIX,
            f"{doc['question_type']}: lmms-eval spatialstack appends the val suffix",
        )

    suffix = r" The final answer MUST BE put in \boxed{}."
    withsuffix = task.vsibench_doc_to_text_plain(dict(NA_DOC), core.LMMS_KWARGS) + suffix
    check(
        withsuffix.endswith(r"single word or phrase. The final answer MUST BE put in \boxed{}."),
        "the vLLM suffix still lands after the lmms_eval post prompt",
    )


def test_decoding_defaults_greedy() -> None:
    print("\n[7] sampling is opt-in and never half-applied")
    import vsibench_eval_core as core

    greedy = core.resolve_decoding(core.EvalConfig(model="unused"))
    check(greedy["do_sample"] is False, "EvalConfig() does not sample")
    check(
        greedy["temperature"] == 0.0 and greedy["top_p"] == 1.0 and greedy["top_k"] == -1,
        "the default knobs are the greedy ones every published number used",
    )

    sampled = core.resolve_decoding(core.EvalConfig(model="unused", do_sample=True))
    check(
        sampled["temperature"] == 1.0 and sampled["top_p"] == 0.8,
        "--do-sample alone gives Qwen3 non-thinking guidance (t=1.0, top_p=0.8)",
    )

    tuned = core.resolve_decoding(
        core.EvalConfig(model="unused", do_sample=True, temperature=0.6, top_p=0.95, top_k=20, seed=7)
    )
    check(
        (tuned["temperature"], tuned["top_p"], tuned["top_k"], tuned["seed"]) == (0.6, 0.95, 20, 7),
        "explicit knobs win over the mode defaults",
    )

    try:
        core.resolve_decoding(core.EvalConfig(model="unused", temperature=1.0))
    except ValueError:
        check(True, "a sampling knob without do_sample raises instead of being ignored")
    else:
        check(False, "a sampling knob without do_sample raises instead of being ignored")


def main() -> None:
    original = os.environ.get("VSIBENCH_PROTOCOL")
    try:
        test_extraction()
        test_box_survives_a_rambling_tail()
        test_prompt_strings_are_not_answers()
        test_gold_self_consistency()
        test_non_compliance_is_visible()
        test_suffix_defaults_off()
        test_decoding_defaults_greedy()
    finally:
        if original is None:
            os.environ.pop("VSIBENCH_PROTOCOL", None)
        else:
            os.environ["VSIBENCH_PROTOCOL"] = original

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
