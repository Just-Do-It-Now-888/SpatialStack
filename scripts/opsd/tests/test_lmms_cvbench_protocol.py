#!/usr/bin/env python3
"""Guard the two CV-Bench protocols in src/lmms_eval/tasks/cvbench.

    python3 scripts/opsd/tests/test_lmms_cvbench_protocol.py

The offline task now defaults to the protocol the training loop uses, with
upstream's reachable via CVBENCH_PROTOCOL=lmms_legacy. What has to stay true:

1. the answer budget moves with the protocol, so no run can pair a 16-token
   generation with the parser that expects a full answer -- the mismatch that
   made one checkpoint read 83.57 in training and 5.00 offline;
2. the default prompt drops upstream's "These are frames of a video." and
   asks for ``\\boxed{}`` on the last line (same wording as VSI-Bench);
   an older ``cvbench_val.parquet`` without that suffix is left as-is;
3. lmms_legacy still reproduces upstream verbatim, preamble quirk included;
4. the 2D/3D roll-up is untouched by any of this.
"""

from __future__ import annotations

import importlib.util
import json
import glob
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

TASK_DIR = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "cvbench")
TASK_YAML = os.path.join(TASK_DIR, "cvbench.yaml")
OFFLINE_GLOB = "logs/eval/*/*/cvbench/*/*_samples_cvbench.jsonl"

failures: list[str] = []

DOC = {
    "idx": 0,
    "question": "How many organs are in the image?",
    "choices": ["3", "2", "1", "0"],
    "prompt": "How many organs are in the image? Select from the following choices.\n(A) 3\n(B) 2\n(C) 1\n(D) 0",
    "answer": "(C)",
    "type": "2D",
    "task": "Count",
    "source": "ADE20K",
}
POST_PROMPT = "Answer with the option's letter from the given choices directly."
BOTTLE = (
    "The options are A. 0, B. 12, C. 16, D. 20, E. 24, F. 28.\n"
    "There are many bottles on the shelves.\n"
    "C"
)


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def load_task_utils(protocol: str):
    """Import the task's utils.py fresh, the way the yaml loader does."""
    os.environ["CVBENCH_PROTOCOL"] = protocol
    spec = importlib.util.spec_from_file_location(f"cvbench_utils_{protocol}", os.path.join(TASK_DIR, "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task_config(protocol: str) -> dict:
    os.environ["CVBENCH_PROTOCOL"] = protocol
    from lmms_eval.utils import load_yaml_config

    return load_yaml_config(TASK_YAML)


def test_generation_budget() -> None:
    print("\n[1] the answer budget follows the protocol")
    for key in ("CVBENCH_DO_SAMPLE", "CVBENCH_TEMPERATURE", "CVBENCH_TOP_P", "CVBENCH_SEED"):
        os.environ.pop(key, None)
    ours = load_task_config("spatialstack")["generation_kwargs"]
    lastline = load_task_config("lastline")["generation_kwargs"]
    legacy = load_task_config("lmms_legacy")["generation_kwargs"]

    check(ours["max_new_tokens"] == 1024, f"default protocol generates up to {ours['max_new_tokens']} tokens")
    check(lastline["max_new_tokens"] == 1024, f"lastline generates up to {lastline['max_new_tokens']} tokens")
    check(legacy["max_new_tokens"] == 16, f"lmms_legacy generates up to {legacy['max_new_tokens']} tokens")
    for name, kwargs in (("default", ours), ("lastline", lastline), ("lmms_legacy", legacy)):
        check(
            kwargs["do_sample"] is False and kwargs["temperature"] == 0,
            f"{name} decoding stays greedy",
        )

    check(ours.get("until") == [], "the default protocol stops only at the token budget")
    check(legacy.get("until") is None, "lmms_legacy still inherits upstream's blank-line stop")

    metrics = [entry["metric"] for entry in load_task_config("spatialstack")["metric_list"]]
    check(metrics == ["cvbench_score", "cvbench_answered"], f"metrics reported: {metrics}")

    # The deliberate mix: upstream's prompt and parser at our answer length,
    # which is how you tell a changed model apart from a changed parser.
    os.environ["CVBENCH_MAX_NEW_TOKENS"] = "1024"
    try:
        stretched = load_task_utils("lmms_legacy")
    finally:
        os.environ.pop("CVBENCH_MAX_NEW_TOKENS", None)
    check(
        stretched.CVBENCH_GENERATION_KWARGS["max_new_tokens"] == 1024
        and stretched.CVBENCH_PROTOCOL == "lmms_legacy",
        "CVBENCH_MAX_NEW_TOKENS stretches lmms_legacy without changing its parser",
    )

    os.environ["CVBENCH_DO_SAMPLE"] = "1"
    os.environ["CVBENCH_SEED"] = "20260904"
    try:
        sampled = load_task_utils("spatialstack").CVBENCH_GENERATION_KWARGS
    finally:
        os.environ.pop("CVBENCH_DO_SAMPLE", None)
        os.environ.pop("CVBENCH_SEED", None)
    check(
        sampled["do_sample"] is True
        and sampled["temperature"] == 1.0
        and sampled["top_p"] == 0.8
        and sampled["seed"] == 20260904
        and sampled["max_new_tokens"] == 1024,
        "CVBENCH_DO_SAMPLE=1 is t=1.0 / top_p=0.8 / seed, budget unchanged",
    )


def test_prompt() -> None:
    print("\n[2] prompt wording")
    from cvbench_scoring import BOXED_LASTLINE_SUFFIX, build_cvbench_prompt

    ours = load_task_utils("spatialstack").cvbench_doc_to_text(DOC, {"pre_prompt": "", "mca_post_prompt": POST_PROMPT})
    legacy = load_task_utils("lmms_legacy").cvbench_doc_to_text(DOC, {"pre_prompt": "", "mca_post_prompt": POST_PROMPT})

    check("These are frames of a video." not in ours, "the default protocol drops the video preamble")
    check(ours == build_cvbench_prompt(DOC), "the default prompt comes from the shared builder, not a copy of it")
    check(r"\boxed{}" in ours and "last line" in ours, "the default prompt asks for a boxed answer on the last line")
    check(
        ours.splitlines()[-1] == POST_PROMPT + BOXED_LASTLINE_SUFFIX,
        "boxed suffix is concatenated on the MCA line, as in VSI val",
    )
    check("\nThe final answer" not in ours, "boxed suffix is not a separate prompt line")
    check(
        r"\boxed{}" not in legacy and "last line" not in legacy,
        "lmms_legacy does not grow the boxed last-line suffix",
    )
    check(
        legacy.startswith("These are frames of a video.\n"),
        "lmms_legacy keeps the preamble upstream actually sends",
    )
    check("Options:\nA. 3" in legacy, "lmms_legacy keeps upstream's 'A. 3' option layout")
    check(POST_PROMPT in legacy, "lmms_legacy keeps the MCA post-prompt")

    from cvbench_scoring import LASTLINE_SUFFIX

    last = load_task_utils("lastline").cvbench_doc_to_text(DOC, {"pre_prompt": "", "mca_post_prompt": POST_PROMPT})
    check(r"\boxed{}" not in last, "lastline prompt does not mention boxed")
    check(
        last.splitlines()[-1] == POST_PROMPT + LASTLINE_SUFFIX,
        "lastline suffix is concatenated on the MCA line",
    )
    last_score = load_task_utils("lastline").cvbench_process_results(dict(DOC), ["\\boxed{A}\nC"])
    check(last_score["cvbench_score"]["pred_answer"] == "C", "lastline CV scores the last line, not the box")


def test_prompt_matches_val_parquet() -> None:
    """The offline default is only comparable to the curve if both ask the same question."""
    print("\n[3] offline default vs the validation set the trainer runs")
    parquet_path = os.path.join(REPO_ROOT, "data/eval/cvbench_verl/cvbench_val.parquet")
    if not os.path.exists(parquet_path):
        print("  skip (build the val parquet first)")
        return

    import pandas as pd

    from cvbench_scoring import build_cvbench_prompt

    frame = pd.read_parquet(parquet_path)
    row = frame.iloc[0]
    trainer_prompt = row["prompt"][0]["content"].replace("<image>", "", 1)
    style = row["extra_info"].get("prompt_style")

    if r"\boxed{}" not in trainer_prompt:
        print("  note this parquet predates boxed last-line; it is not rebuilt")
        print(f"       trainer last line: {trainer_prompt.splitlines()[-1]!r}")
        print(f"       offline last line: {build_cvbench_prompt(DOC).splitlines()[-1]!r}")
        check(r"\boxed{}" in build_cvbench_prompt(DOC), "new builder still emits the boxed last-line suffix")
        return

    if style is None:
        # Built before build_cvbench_val_parquet.py grew --prompt-style, i.e. with
        # the dataset's own "(A) 3" wording. Not a failure -- rebuilding it would
        # change the protocol of a run in flight -- but the offline default then
        # asks a differently worded question than the curve it is compared to.
        print("  note this parquet predates --prompt-style, so it is the native '(A) 3' wording")
        print(f"       trainer: {trainer_prompt.splitlines()[0][:70]!r}")
        print(f"       offline: {build_cvbench_prompt(DOC).splitlines()[0][:70]!r}")
        print("       rebuild with --prompt-style lmms_eval to close this gap between runs")
        return

    os.environ["CVBENCH_PROMPT_STYLE"] = style
    try:
        offline = load_task_utils("spatialstack").cvbench_doc_to_text(DOC, None)
    finally:
        os.environ.pop("CVBENCH_PROMPT_STYLE", None)
    check(offline == trainer_prompt, f"offline and trainer ask the same question under style {style!r}")


def test_parsing() -> None:
    print("\n[4] reading the answer out of a response")
    prose = "Based on the provided images, we can observe the following:\n\n-   The"
    answered = "Therefore, the count is 1.\n\n(C)"

    for protocol, expect_prose, expect_answered in (("spatialstack", "", "C"), ("lmms_legacy", "B", "C")):
        utils = load_task_utils(protocol)
        cut = utils.cvbench_process_results(dict(DOC), [prose])["cvbench_score"]
        full = utils.cvbench_process_results(dict(DOC), [answered])["cvbench_score"]
        check(
            cut["pred_answer"] == expect_prose,
            f"{protocol}: a cut-off response reads as {cut['pred_answer']!r} (want {expect_prose!r})",
        )
        check(
            full["pred_answer"] == expect_answered,
            f"{protocol}: a complete response reads as {full['pred_answer']!r} (want {expect_answered!r})",
        )
        check(
            cut["answered"] == (1 if expect_prose else 0),
            f"{protocol}: answered flag on the cut-off response is {cut['answered']}",
        )

    from cvbench_scoring import extract_cvbench_option

    bottle_doc = {
        **DOC,
        "question": "How many bottles are in the image?",
        "choices": ["0", "12", "16", "20", "24", "28"],
        "answer": "(C)",
    }
    utils = load_task_utils("spatialstack")
    bottle = utils.cvbench_process_results(dict(bottle_doc), [BOTTLE])["cvbench_score"]
    check(bottle["pred_answer"] == "C", f"bottle recitation last-line C reads as {bottle['pred_answer']!r}")

    boxed = utils.cvbench_process_results(dict(bottle_doc), ["reasoning\n\\boxed{C}"])["cvbench_score"]
    check(boxed["pred_answer"] == "C", f"\\boxed{{C}} on the last line reads as {boxed['pred_answer']!r}")

    paren = utils.cvbench_process_results(dict(bottle_doc), ["reasoning\n(C)"])["cvbench_score"]
    check(paren["pred_answer"] == "C", f"last-line (C) with no box reads as {paren['pred_answer']!r}")

    none = utils.cvbench_process_results(dict(bottle_doc), ["There are many bottles on the shelves."])["cvbench_score"]
    check(none["pred_answer"] == "", f"no last-line option is unanswered ({none['pred_answer']!r})")

    os.environ["CVBENCH_PARSER"] = "word_boundary"
    try:
        check(
            extract_cvbench_option(BOTTLE, choices=bottle_doc["choices"]) == "A",
            "CVBENCH_PARSER=word_boundary still takes the first A from the bottle recitation",
        )
        wb = load_task_utils("spatialstack").cvbench_process_results(dict(bottle_doc), [BOTTLE])["cvbench_score"]
        check(wb["pred_answer"] == "A", f"word_boundary through the task still reads {wb['pred_answer']!r}")
    finally:
        os.environ.pop("CVBENCH_PARSER", None)


def test_rollup() -> None:
    print("\n[5] 2D/3D roll-up is unchanged")
    utils = load_task_utils("spatialstack")
    rows = [
        {"source": "ADE20K", "task": "Count", "result": 1, "answered": 1},
        {"source": "COCO", "task": "Count", "result": 0, "answered": 1},
        {"source": "COCO", "task": "Count", "result": 0, "answered": 1},
        {"source": "COCO", "task": "Count", "result": 0, "answered": 1},
        {"source": "Omni3D", "task": "Depth", "result": 1, "answered": 1},
        {"source": "Omni3D", "task": "Depth", "result": 1, "answered": 0},
    ]
    combined = utils.cvbench_aggregate_results(rows)
    answered = utils.cvbench_aggregate_answered(rows)
    check(abs(combined - 75.0) < 1e-6, f"combined = {combined} (a row average would be 50.0)")
    check(abs(answered - 100 * 5 / 6) < 1e-6, f"answered = {answered:.2f}%")


def test_replay() -> None:
    print("\n[6] replay of archived offline generations")
    paths = [p for p in glob.glob(os.path.join(REPO_ROOT, OFFLINE_GLOB)) if os.path.getsize(p) > 1_000_000]
    if not paths:
        print("  skip (no archived samples)")
        return

    ours = load_task_utils("spatialstack")
    legacy = load_task_utils("lmms_legacy")
    print(f"  {'run':<44} {'default':>8} {'answered':>9} {'legacy':>8}")
    for path in sorted(paths):
        with open(path) as handle:
            samples = [json.loads(line) for line in handle]
        our_rows, legacy_rows = [], []
        for sample in samples:
            response = sample["filtered_resps"][0]
            our_rows.append(ours.cvbench_process_results(dict(sample["doc"]), [response])["cvbench_score"])
            legacy_rows.append(legacy.cvbench_process_results(dict(sample["doc"]), [response])["cvbench_score"])
        name = os.path.relpath(path, REPO_ROOT).split("/")[2][-40:]
        print(
            f"  {name:<44} {ours.cvbench_aggregate_results(our_rows):>8.2f} "
            f"{ours.cvbench_aggregate_answered(our_rows):>8.1f}% {legacy.cvbench_aggregate_results(legacy_rows):>8.2f}"
        )
    check(True, f"replayed {len(paths)} archived run(s) through both protocols")


def main() -> None:
    original = os.environ.get("CVBENCH_PROTOCOL")
    original_parser = os.environ.get("CVBENCH_PARSER")
    original_sample = {
        k: os.environ.get(k) for k in ("CVBENCH_DO_SAMPLE", "CVBENCH_TEMPERATURE", "CVBENCH_TOP_P", "CVBENCH_SEED")
    }
    try:
        test_generation_budget()
        test_prompt()
        test_prompt_matches_val_parquet()
        test_parsing()
        test_rollup()
        test_replay()
    finally:
        if original is None:
            os.environ.pop("CVBENCH_PROTOCOL", None)
        else:
            os.environ["CVBENCH_PROTOCOL"] = original
        if original_parser is None:
            os.environ.pop("CVBENCH_PARSER", None)
        else:
            os.environ["CVBENCH_PARSER"] = original_parser
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
