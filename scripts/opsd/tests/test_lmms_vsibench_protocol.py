#!/usr/bin/env python3
"""Guard the two VSI-Bench protocols in src/lmms_eval/tasks/vsibench.

    python3 scripts/opsd/tests/test_lmms_vsibench_protocol.py

The offline task now defaults to the in-training val protocol (4096-token
budget, boxed last-line suffix, boxed-primary extract), with upstream reachable
via VSIBENCH_PROTOCOL=lmms_legacy. What has to stay true:

1. the answer budget moves with the parser, so no run pairs a 16-token
   generation with a parser that expects a full answer;
2. the default prompt appends the same boxed last-line suffix as
   vsibench_val_boxed_lastline.parquet, on the MCA/NA instruction line;
3. a closed ``\\boxed{}`` is scored first; otherwise last-line answer_tail;
4. multiple choice is restricted to the letters the question actually offers;
5. **the metrics are still lmms_eval's** -- exact match for the six MCA types,
   MRA:.5:.95:.05 for the four NA types, and an unweighted mean over question
   types.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

TASK_DIR = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench")
TASK_YAML = os.path.join(TASK_DIR, "vsibench.yaml")
OFFLINE_GLOB = "logs/eval/*/*/vsibench/*/*_samples_vsibench.jsonl"

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


def load_task_utils(protocol: str):
    """Import the task's utils.py fresh, the way the yaml loader does."""
    os.environ["VSIBENCH_PROTOCOL"] = protocol
    spec = importlib.util.spec_from_file_location(f"vsibench_utils_{protocol}", os.path.join(TASK_DIR, "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task_config(protocol: str) -> dict:
    os.environ["VSIBENCH_PROTOCOL"] = protocol
    from lmms_eval.utils import load_yaml_config

    return load_yaml_config(TASK_YAML)


def test_generation_budget() -> None:
    print("\n[1] the answer budget follows the protocol")
    ours = load_task_config("spatialstack")["generation_kwargs"]
    lastline = load_task_config("lastline")["generation_kwargs"]
    legacy = load_task_config("lmms_legacy")["generation_kwargs"]
    plain = load_task_config("spatialstack_plain")["generation_kwargs"]

    check(ours["max_new_tokens"] == 4096, f"default protocol generates up to {ours['max_new_tokens']} tokens")
    check(lastline["max_new_tokens"] == 4096, f"lastline generates up to {lastline['max_new_tokens']} tokens")
    check(legacy["max_new_tokens"] == 16, f"lmms_legacy generates up to {legacy['max_new_tokens']} tokens")
    check(plain["max_new_tokens"] == 1024, f"spatialstack_plain generates up to {plain['max_new_tokens']} tokens")
    for name, kwargs in (("default", ours), ("lastline", lastline), ("lmms_legacy", legacy), ("spatialstack_plain", plain)):
        check(kwargs["do_sample"] is False and kwargs["temperature"] == 0, f"{name} decoding stays greedy")
    check(ours.get("until") == [], "the default protocol stops only at the token budget")
    check(lastline.get("until") == [], "lastline also clears until")
    check(plain.get("until") == [], "spatialstack_plain also clears until")
    check(legacy.get("until") is None, "lmms_legacy still inherits upstream's blank-line stop")

    metrics = [entry["metric"] for entry in load_task_config("spatialstack")["metric_list"]]
    check(metrics == ["vsibench_score", "vsibench_answered"], f"metrics reported: {metrics}")

    os.environ["VSIBENCH_MAX_NEW_TOKENS"] = "4096"
    try:
        stretched = load_task_utils("lmms_legacy")
    finally:
        os.environ.pop("VSIBENCH_MAX_NEW_TOKENS", None)
    check(
        stretched.VSIBENCH_GENERATION_KWARGS["max_new_tokens"] == 4096
        and stretched.VSIBENCH_PROTOCOL == "lmms_legacy",
        "VSIBENCH_MAX_NEW_TOKENS stretches lmms_legacy without changing its parser",
    )


def test_prompt() -> None:
    print("\n[1b] prompt wording matches the boxed last-line val parquet")
    from vsibench_eval_core import LMMS_KWARGS
    from vsibench_scoring import BOXED_LASTLINE_SUFFIX

    ours = load_task_utils("spatialstack")
    legacy = load_task_utils("lmms_legacy")
    mca = ours.vsibench_doc_to_text(MCA_DOC, LMMS_KWARGS)
    na = ours.vsibench_doc_to_text(NA_DOC, LMMS_KWARGS)
    mca_plain = ours.vsibench_doc_to_text_plain(MCA_DOC, LMMS_KWARGS)
    na_plain = ours.vsibench_doc_to_text_plain(NA_DOC, LMMS_KWARGS)

    check(mca == mca_plain + BOXED_LASTLINE_SUFFIX, "MCA boxed suffix is concatenated, not a new line")
    check(na == na_plain + BOXED_LASTLINE_SUFFIX, "NA boxed suffix is concatenated, not a new line")
    check("\nThe final answer" not in mca and "\nThe final answer" not in na, "suffix is not its own prompt line")
    check(
        mca.splitlines()[-1].endswith(BOXED_LASTLINE_SUFFIX.strip()),
        "MCA last line is the post-prompt plus the boxed instruction",
    )
    check(
        r"\boxed{}" not in legacy.vsibench_doc_to_text(MCA_DOC, LMMS_KWARGS),
        "lmms_legacy does not grow the boxed last-line suffix",
    )
    check(
        legacy.vsibench_doc_to_text(MCA_DOC, LMMS_KWARGS) == mca_plain,
        "lmms_legacy prompt is the unsuffixed spatialstack prompt",
    )
    plain = load_task_utils("spatialstack_plain")
    check(
        plain.vsibench_doc_to_text(MCA_DOC, LMMS_KWARGS) == mca_plain,
        "spatialstack_plain is the original unsuffixed prompt",
    )
    check(
        r"\boxed{}" not in plain.vsibench_doc_to_text(MCA_DOC, LMMS_KWARGS),
        "spatialstack_plain does not add boxed last-line",
    )
    from vsibench_scoring import LASTLINE_SUFFIX

    lastline = load_task_utils("lastline")
    mca_last = lastline.vsibench_doc_to_text(MCA_DOC, LMMS_KWARGS)
    check(mca_last == mca_plain + LASTLINE_SUFFIX, "lastline uses SPAR wording, concatenated")
    check(r"\boxed{}" not in mca_last, "lastline prompt does not mention boxed")
    lastline_scored = lastline.vsibench_process_results(dict(MCA_DOC), ["\\boxed{B}\nA"])["vsibench_score"]
    check(
        lastline_scored["parsed_answer"] == "A",
        "lastline scores the last line, not a closed box above it",
    )

    parquet_path = os.path.join(REPO_ROOT, "data/eval/vsibench_verl/vsibench_val_boxed_lastline.parquet")
    if not os.path.isfile(parquet_path):
        print("  skip (boxed last-line val parquet not built)")
        return
    import pandas as pd

    row = pd.read_parquet(parquet_path).iloc[0].to_dict()
    user = row["prompt"][1]["content"]
    check(user.endswith(BOXED_LASTLINE_SUFFIX), "val parquet user turn ends with the same suffix bytes")
    check(BOXED_LASTLINE_SUFFIX in mca, "offline spatialstack asks for the same boxed last-line")


def test_boxed_primary() -> None:
    print("\n[1c] closed \\boxed{} is the answer site")
    ours = load_task_utils("spatialstack")
    boxed_mca = "The answer is A.\n\\boxed{B}"
    mca = ours.vsibench_process_results(dict(MCA_DOC), [boxed_mca])["vsibench_score"]
    check(mca["parsed_answer"] == "B", f"boxed B wins over last-line answer-is A (got {mca['parsed_answer']!r})")
    check(mca["boxed_present"] == 1, "boxed_present is set when the box is closed")

    boxed_na = "* Image 178: Kitchen.\n\\boxed{30.9}"
    na = ours.vsibench_process_results(dict(NA_DOC), [boxed_na])["vsibench_score"]
    check(na["parsed_answer"] == "30.9", f"boxed 30.9 is not the Image index (got {na['parsed_answer']!r})")

    last_line = ours.vsibench_process_results(dict(MCA_DOC), ["reasoning\n\nB"])["vsibench_score"]
    check(last_line["parsed_answer"] == "B" and last_line["boxed_present"] == 0, "no box still reads a last-line letter")

    original = os.environ.get("VSIBENCH_BOXED_PRIMARY")
    os.environ["VSIBENCH_BOXED_PRIMARY"] = "0"
    try:
        off = load_task_utils("spatialstack")
        fallback = off.vsibench_process_results(dict(MCA_DOC), [boxed_mca])["vsibench_score"]
        check(
            fallback["parsed_answer"] != "B" or fallback["boxed_present"] == 0,
            "VSIBENCH_BOXED_PRIMARY=0 does not take the box as the only site",
        )
        check(fallback["parsed_answer"] != "B", f"without boxed-primary last line is {fallback['parsed_answer']!r}, not B")
    finally:
        if original is None:
            os.environ.pop("VSIBENCH_BOXED_PRIMARY", None)
        else:
            os.environ["VSIBENCH_BOXED_PRIMARY"] = original


def test_mca_parsing() -> None:
    print("\n[2] reading a letter out of a reasoning response")
    ours = load_task_utils("spatialstack")
    legacy = load_task_utils("lmms_legacy")

    cot = (
        "Based on the frames, the sofa is against the far wall and the tv is opposite it.\n"
        "Facing the tv, the door appears on the same side as the window.\n"
        "The answer is B."
    )
    trailing_line = "Looking at the layout, the door is behind me and to one side.\n\n(B)"
    option_text = "Standing by the sofa facing the tv, the door is on my left"

    for name, response, want in (
        ("chain of thought", cot, "B"),
        ("verdict on its own line", trailing_line, "B"),
        ("answers with the option text", option_text, "B"),
    ):
        got = ours.vsibench_process_results(dict(MCA_DOC), [response])["vsibench_score"]
        check(got["parsed_answer"] == want, f"default: {name} reads as {got['parsed_answer']!r} (want {want!r})")
        check(got["accuracy"] == 1.0, f"default: {name} scores correct")

    stale = legacy.vsibench_process_results(dict(MCA_DOC), [cot])["vsibench_score"]
    check(
        stale["parsed_answer"] == "Based" and stale["accuracy"] == 0.0,
        f"lmms_legacy still reads the first token ({stale['parsed_answer']!r}) and scores 0",
    )

    print("\n[3] a letter the question does not offer is not a selection")
    # Two options, so only A and B exist. "C" here is a coordinate label, and
    # upstream's A-F style scan would happily return it.
    noise = "The camera passes marker C on the wall, then the door.\nAnswer: A"
    got = ours.vsibench_process_results(dict(MCA_DOC), [noise])["vsibench_score"]
    check(got["parsed_answer"] == "A", f"a two-option question reads {got['parsed_answer']!r}, not 'C'")

    blank = ours.vsibench_process_results(dict(MCA_DOC), ["I cannot tell from these frames."])["vsibench_score"]
    check(
        blank["parsed_answer"] == "" and blank["answered"] == 0,
        f"an unreadable response is answered=0, not a silent wrong answer (got {blank['parsed_answer']!r})",
    )

    print("\n[3b] truncated CoT must not treat English 'a' as option A")
    long_cot = (
        "The sofa is on the left of the frame. A desk is below the TV.\n"
        "There is a fan (image 15).\n"
        "The stove is in the kitchen. The kitchen"
    )
    stray = ours.vsibench_process_results(dict(MCA_DOC), [long_cot])["vsibench_score"]
    check(
        stray["parsed_answer"] == "" and stray["answered"] == 0,
        f"truncated CoT is unanswered, not letter {stray['parsed_answer']!r}",
    )

    print("\n[3c] mid-CoT answer/option phrases do not override the last line")
  # Hypothetical branches name A/C in the middle; only the last line counts.
    hypotheticals = (
        "Let's assume the piano is to the left. Then the answer is C. left.\n"
        "Let's assume the piano is to the right. Then the answer is A. right.\n"
        "Which one is it?\n\n"
        "Then the answer would be C."
    )
    hypo = ours.vsibench_process_results(dict(MCA_DOC), [hypotheticals])["vsibench_score"]
    check(
        hypo["parsed_answer"] == "" and hypo["answered"] == 0,
        f"mid-CoT 'answer is A/C' is ignored when the last line is not a selection "
        f"(got {hypo['parsed_answer']!r})",
    )

    option_chatter = (
        "None of the options match.\n"
        "Only option C has this as the fourth item, which is wrong.\n"
        "I will bet on option A being the intended answer.\n"
        "Order: microwave, door, refrigerator, backpack."
    )
    chatter = ours.vsibench_process_results(dict(MCA_DOC), [option_chatter])["vsibench_score"]
    check(
        chatter["parsed_answer"] == "" and chatter["answered"] == 0,
        f"mid-CoT 'option A/C' chatter is ignored (got {chatter['parsed_answer']!r})",
    )

    settled = (
        "Option C is wrong. Option A is wrong too.\n"
        "After checking every frame, the answer is B."
    )
    settled_score = ours.vsibench_process_results(dict(MCA_DOC), [settled])["vsibench_score"]
    check(
        settled_score["parsed_answer"] == "B" and settled_score["accuracy"] == 1.0,
        "an explicit answer on the last line still scores",
    )

    print("\n[3d] last-line selection phrases vs stray letters")
    three = dict(MCA_DOC)
    three["options"] = ["A. right", "B. back", "C. left"]
    three["ground_truth"] = "A"
    would_be = (
        "Let's assume the piano is to the left. Then the answer is C. left.\n"
        "Let's assume the piano is to the right. Then the answer is A. right.\n"
        "Then the answer would be C."
    )
    would = ours.vsibench_process_results(three, [would_be])["vsibench_score"]
    check(
        would["parsed_answer"] == "C" and would["answered"] == 1,
        f"three-option 'would be C' reads C (got {would['parsed_answer']!r})",
    )
    choose = ours.vsibench_process_results(three, ["After checking the frames, I choose C."])["vsibench_score"]
    check(choose["parsed_answer"] == "C", f"I choose C reads C (got {choose['parsed_answer']!r})")
    so_c = ours.vsibench_process_results(three, ["The piano is on the left, so C."])["vsibench_score"]
    check(so_c["parsed_answer"] == "C", f"so C. reads C (got {so_c['parsed_answer']!r})")
    lone = ours.vsibench_process_results(three, ["reasoning\n\nC"])["vsibench_score"]
    check(lone["parsed_answer"] == "C" and lone["answered"] == 1, "a lone last-line C still scores")

    order_doc = {
        "question_type": "obj_appearance_order",
        "question": "What is the order?",
        "options": [
            "A. telephone, bed, suitcase, laptop",
            "B. laptop, bed, suitcase, telephone",
            "C. telephone, suitcase, laptop, bed",
            "D. bed, suitcase, laptop, telephone",
        ],
        "ground_truth": "A",
    }
    initials = (
        "None of the options match.\n"
        "I am stuck. Let's make a guess based on the most common errors. "
        "Perhaps the order in the question is T, L, B, S,"
    )
    guess = ours.vsibench_process_results(order_doc, [initials])["vsibench_score"]
    check(
        guess["parsed_answer"] == "" and guess["answered"] == 0,
        f"truncated 'T, L, B, S' / 'a guess' is unanswered (got {guess['parsed_answer']!r})",
    )


def test_na_parsing() -> None:
    print("\n[4] reading a number out of a reasoning response")
    ours = load_task_utils("spatialstack")
    legacy = load_task_utils("lmms_legacy")

    cot = (
        "The room looks about 6 meters long and 5 meters wide from the frames.\n"
        "6 x 5 = 30, adjusting slightly for the alcove.\n"
        "The answer is 30.9 square meters."
    )
    bare_tail = "Estimating from the walls and the visible furniture, roughly:\n\n30.9"
    thousands = "Final answer: 1,024"

    for name, response, want in (
        ("chain of thought", cot, 30.9),
        ("number on its own line", bare_tail, 30.9),
        ("thousands separator", thousands, 1024.0),
    ):
        parsed = ours._shared().extract_vsibench_number(response)
        check(parsed == want, f"default: {name} reads as {parsed!r} (want {want!r})")

    scored = ours.vsibench_process_results(dict(NA_DOC), [cot])["vsibench_score"]
    check(scored["MRA:.5:.95:.05"] == 1.0, f"an exact numeric answer gets full MRA (got {scored['MRA:.5:.95:.05']})")

    stale = legacy.vsibench_process_results(dict(NA_DOC), [cot])["vsibench_score"]
    check(
        stale["MRA:.5:.95:.05"] == 0.0 and stale["answered"] == 0,
        "lmms_legacy cannot parse the same response and records the worst-case MRA",
    )

    unreadable = ours.vsibench_process_results(dict(NA_DOC), ["I am not able to estimate this."])["vsibench_score"]
    check(
        unreadable["answered"] == 0 and unreadable["MRA:.5:.95:.05"] == 0.0,
        "a response with no number is answered=0 and scores the worst-case MRA",
    )

    outline = (
        "1. Identify the sofa.\n"
        "4. Estimate the distance:\n"
        "Let's look at Image `image"
    )
    outline_parsed = ours._shared().extract_vsibench_number(outline)
    check(
        outline_parsed is None,
        f"an outline heading is not a quantity (got {outline_parsed!r})",
    )

    print("\n[4b] mid-CoT numeric answer phrases do not override the last line")
    mid_number = (
        "The alcove looks about 6 by 5, so the answer is 30.9 square meters.\n"
        "Wait, that used the wrong wall. Rechecking the frames.\n"
        "Still not sure; maybe 33."
    )
    mid_parsed = ours._shared().extract_vsibench_number(mid_number)
    check(mid_parsed == 33.0, f"last-line number wins over mid-CoT answer-is (got {mid_parsed!r})")

    print("\n[4c] a truncated frame-index list is not a distance")
    frame_dump = (
        "Looking around the room.\n"
        "-   The **table** is visible in several frames (e.g., frame 3, 4, 5, 6, 7, "
        "8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,"
    )
    frame_parsed = ours._shared().extract_vsibench_number(frame_dump)
    check(frame_parsed is None, f"frame enumeration last line is unanswered (got {frame_parsed!r})")
    still_distance = ours._shared().extract_vsibench_number(
        "Therefore the distance is 2.8 meters.\n* Image 178: Kitchen."
    )
    check(still_distance == 2.8, f"Image-label tail still reads the conclusion (got {still_distance!r})")


def test_metrics_unchanged() -> None:
    """The parser changed; the metric definition did not."""
    print("\n[5] metrics are still lmms_eval's")
    ours = load_task_utils("spatialstack")
    legacy = load_task_utils("lmms_legacy")

    # MRA is a mean over ten thresholds, not a hit/miss, so a near-miss has to
    # land strictly between 0 and 1 exactly as upstream defines it.
    near = ours._shared().extract_vsibench_number("The answer is 33.")
    doc = dict(NA_DOC)
    scored = ours.vsibench_process_results(doc, ["The answer is 33."])["vsibench_score"]
    manual = ours.mean_relative_accuracy(near, 30.9, start=.5, end=.95, interval=.05)
    check(abs(scored["MRA:.5:.95:.05"] - manual) < 1e-12, "MRA comes from upstream's mean_relative_accuracy")
    check(0.0 < manual < 1.0, f"a near miss is partial credit ({manual:.3f}), not hit/miss")

    rows = [
        {"question_type": "object_rel_direction_easy", "accuracy": 1.0, "answered": 1},
        {"question_type": "object_rel_direction_medium", "accuracy": 0.0, "answered": 1},
        {"question_type": "object_rel_direction_hard", "accuracy": 0.5, "answered": 1},
        {"question_type": "route_planning", "accuracy": 1.0, "answered": 1},
        {"question_type": "room_size_estimation", "MRA:.5:.95:.05": 0.5, "answered": 0},
    ]
    ours_overall = ours.vsibench_aggregate_results([dict(r) for r in rows])
    legacy_overall = legacy.vsibench_aggregate_results([dict(r) for r in rows])
    check(abs(ours_overall - legacy_overall) < 1e-12, "aggregation is identical under both protocols")
    # direction easy/medium/hard collapse into one entry, so the mean is over
    # {direction 0.5, route 1.0, room 0.5} -- an unweighted mean over question
    # types, not over rows.
    check(abs(ours_overall - 100 * (0.5 + 1.0 + 0.5) / 3) < 1e-9, f"overall = {ours_overall:.4f}")
    check(
        abs(ours.vsibench_aggregate_answered(rows) - 80.0) < 1e-9,
        "answered is reported separately and does not enter the score",
    )


def test_replay() -> None:
    """Short archived answers must score the same under both protocols."""
    print("\n[6] replay of archived offline generations")
    paths = [
        p
        for p in glob.glob(os.path.join(REPO_ROOT, OFFLINE_GLOB))
        if os.path.getsize(p) > 1_000_000 and "boxed_lastline" not in p
    ]
    if not paths:
        print("  skip (no archived samples)")
        return

    ours = load_task_utils("spatialstack")
    legacy = load_task_utils("lmms_legacy")
    print(f"  {'run':<40} {'default':>8} {'answered':>9} {'legacy':>8}")
    for path in sorted(paths):
        with open(path) as handle:
            samples = [json.loads(line) for line in handle]
        our_rows, legacy_rows = [], []
        for sample in samples:
            response = sample["filtered_resps"][0]
            our_rows.append(ours.vsibench_process_results(dict(sample["doc"]), [response])["vsibench_score"])
            legacy_rows.append(legacy.vsibench_process_results(dict(sample["doc"]), [response])["vsibench_score"])
        our_score = ours.vsibench_aggregate_results(our_rows)
        legacy_score = legacy.vsibench_aggregate_results(legacy_rows)
        name = os.path.relpath(path, REPO_ROOT).split("/")[2][-38:]
        print(
            f"  {name:<40} {our_score:>8.2f} {ours.vsibench_aggregate_answered(our_rows):>8.1f}% {legacy_score:>8.2f}"
        )
        # These runs were generated under the 16-token protocol, so the answers
        # are already bare. A parser that only relocates where it looks must not
        # move a number that was never ambiguous.
        check(
            abs(our_score - legacy_score) < 0.5,
            f"{name}: short answers score the same either way ({our_score:.2f} vs {legacy_score:.2f})",
        )


def main() -> None:
    original = os.environ.get("VSIBENCH_PROTOCOL")
    original_boxed = os.environ.get("VSIBENCH_BOXED_PRIMARY")
    try:
        test_generation_budget()
        test_prompt()
        test_boxed_primary()
        test_mca_parsing()
        test_na_parsing()
        test_metrics_unchanged()
        test_replay()
    finally:
        if original is None:
            os.environ.pop("VSIBENCH_PROTOCOL", None)
        else:
            os.environ["VSIBENCH_PROTOCOL"] = original
        if original_boxed is None:
            os.environ.pop("VSIBENCH_BOXED_PRIMARY", None)
        else:
            os.environ["VSIBENCH_BOXED_PRIMARY"] = original_boxed

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
