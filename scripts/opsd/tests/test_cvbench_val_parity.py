#!/usr/bin/env python3
"""Guard the CV-Bench scoring path against the failure that voided MV-OPSD v0.

What is asserted:

1. the hardened extractor never invents an option letter out of prose;
2. ``lmms_legacy`` still reproduces the old first-[A-F] rule when requested;
3. replaying v0's archived generations shows the legacy parser scoring high while
   the default parser reports that almost nothing was answered;
4. the 2D/3D roll-up arithmetic still matches CV-Bench's published weighting;
5. lmms_eval-style prompts built into the val parquet match ``cvbench_scoring``.

    python3 scripts/opsd/tests/test_cvbench_val_parity.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))

from cvbench_scoring import build_cvbench_prompt, extract_cvbench_option  # noqa: E402
from mvopsd_reward import compute_score  # noqa: E402
from verl.trainer.ppo.cvbench_metrics import compute_cvbench_metrics  # noqa: E402

OFFLINE_GLOB = "logs/eval/*/frames_32/*/cvbench/*/*_samples_cvbench.jsonl"
VAL_PARQUET = "data/eval/cvbench_verl/cvbench_val.parquet"
OFFLINE_JSON = "data/eval/visionopd/cvbench.json"
TOLERANCE = 1e-6

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def scores_of(row: dict) -> dict:
    """The verdict fields only.

    `compute_score` also returns diagnostics such as `resp_chars`, which are
    read by the metric layer and may grow. Comparing whole dicts made this test
    fail for a new diagnostic column rather than for a scoring change.
    """
    return {key: row[key] for key in ("score", "acc", "answered")}


def test_extractor() -> None:
    print("\n[1] option extraction")
    cases = [
        ("(C)", "C", "parenthesised option"),
        ("C", "C", "bare letter"),
        ("C. 1", "C", "letter with content"),
        ("Answer: B", "B", "letter after a label"),
        ("The answer is D.", "D", "letter after a stripped prefix"),
        ("Based on the provided images, we can observe the following:", "", "prose opening with 'Based'"),
        ("To determine which object is closer to the camera", "", "prose with no option"),
        ("Both cushions are visible in the frame", "", "prose containing no standalone A-F"),
    ]
    for text, expected, label in cases:
        got = extract_cvbench_option(text, parser="word_boundary")
        check(got == expected, f"{label}: {text[:44]!r} -> {got!r} (want {expected!r})")

    check(
        extract_cvbench_option("Based on the provided images, we can observe", parser="lmms_legacy") == "B",
        "the legacy parser really did read 'Based' as a vote for B",
    )

    bottle = "The options are A. 0, B. 12, C. 16.\nC"
    choices = ["0", "12", "16"]
    check(
        extract_cvbench_option(bottle, parser="boxed_lastline", choices=choices) == "C",
        "boxed_lastline reads last-line C after reciting A. 0",
    )
    check(
        extract_cvbench_option("so the answer is\n\\boxed{C}", parser="boxed_lastline", choices=choices) == "C",
        "boxed_lastline reads a closed box",
    )
    check(
        extract_cvbench_option(bottle, parser="word_boundary") == "A",
        "word_boundary still takes A from the same recitation",
    )


def test_parser_switch() -> None:
    print("\n[2] parser switch via reward_kwargs")
    prose = "Based on the provided images, there are a total of"
    default_row = compute_score(
        data_source="cvbench/COCO/Count",
        solution_str=prose,
        ground_truth="B",
    )
    legacy_row = compute_score(
        data_source="cvbench/COCO/Count",
        solution_str=prose,
        ground_truth="B",
        cvbench_parser="lmms_legacy",
    )
    check(scores_of(default_row) == {"score": 0.0, "acc": 0.0, "answered": 0.0}, f"default parser: {default_row}")
    check(scores_of(legacy_row) == {"score": 1.0, "acc": 1.0, "answered": 1.0}, f"legacy parser: {legacy_row}")


def test_answered_flag() -> None:
    print("\n[3] answered flag")
    answered_row = compute_score(data_source="cvbench/COCO/Count", solution_str="(B)", ground_truth="B")
    prose_row = compute_score(
        data_source="cvbench/COCO/Count",
        solution_str="Based on the provided images, there are a total of",
        ground_truth="B",
    )
    check(scores_of(answered_row) == {"score": 1.0, "acc": 1.0, "answered": 1.0}, f"a real answer: {answered_row}")
    check(scores_of(prose_row) == {"score": 0.0, "acc": 0.0, "answered": 0.0}, f"unanswered prose: {prose_row}")


def test_rollup() -> None:
    print("\n[4] 2D/3D roll-up weighting")
    sources = ["cvbench/ADE20K/Count"] + ["cvbench/COCO/Count"] * 3 + ["cvbench/Omni3D/Depth"] * 2
    accuracies = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    metrics = compute_cvbench_metrics(sources, accuracies, answered=[1.0] * 5 + [0.0])

    check(abs(metrics["val-aux/cvbench/2d/acc"] - 0.5) < TOLERANCE, f"2D = {metrics['val-aux/cvbench/2d/acc']}")
    check(abs(metrics["val-aux/cvbench/3d/acc"] - 1.0) < TOLERANCE, f"3D = {metrics['val-aux/cvbench/3d/acc']}")
    check(
        abs(metrics["val-core/cvbench/combined/acc"] - 0.75) < TOLERANCE,
        f"combined = {metrics['val-core/cvbench/combined/acc']} (micro would be 0.5)",
    )
    check(
        abs(metrics["val-core/cvbench/answered/frac"] - 5 / 6) < TOLERANCE,
        f"answered fraction = {metrics['val-core/cvbench/answered/frac']}",
    )


def combined_from_rows(rows) -> float:
    sources, accuracies = zip(*rows) if rows else ((), ())
    metrics = compute_cvbench_metrics(list(sources), list(accuracies))
    return metrics.get("val-core/cvbench/combined/acc", 0.0)


def test_v0_replay() -> None:
    print("\n[5] replay of v0's archived generations")
    matches = [
        path
        for path in glob.glob(os.path.join(REPO_ROOT, OFFLINE_GLOB))
        if os.path.getsize(path) > 1_000_000
    ]
    if not matches:
        print("  skip (no archived offline samples)")
        return

    print(f"  {'checkpoint':<34} {'legacy':>8} {'hardened':>9} {'answered':>9}")
    for path in sorted(matches):
        with open(path) as handle:
            samples = [json.loads(line) for line in handle]

        legacy_rows, new_rows, answered = [], [], 0.0
        for sample in samples:
            doc = sample["doc"]
            response = sample["filtered_resps"][0]
            data_source = f"cvbench/{doc['source']}/{doc['task']}"
            gold = doc["answer"][1]
            legacy_rows.append(
                (
                    data_source,
                    compute_score(
                        data_source=data_source,
                        solution_str=response,
                        ground_truth=gold,
                        cvbench_parser="lmms_legacy",
                    )["acc"],
                )
            )
            result = compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=gold,
                cvbench_parser="word_boundary",
            )
            new_rows.append((data_source, result["acc"]))
            answered += result["answered"]

        legacy = 100 * combined_from_rows(legacy_rows)
        new = 100 * combined_from_rows(new_rows)
        frac = answered / len(samples)
        name = os.path.relpath(path, REPO_ROOT).split("/")[2][-24:]
        print(f"  {name:<34} {legacy:>8.2f} {new:>9.2f} {100 * frac:>8.1f}%")

        check(frac < 0.05, f"{name}: nearly nothing was answered ({100 * frac:.1f}%)")
        check(
            legacy > new + 3.0,
            f"{name}: legacy awarded {legacy:.2f} where the hardened parser awards {new:.2f}",
        )


def test_prompt_alignment() -> None:
    print("\n[6] in-training prompt vs cvbench_scoring")
    parquet_path = os.path.join(REPO_ROOT, VAL_PARQUET)
    if not os.path.exists(parquet_path):
        print("  skip (build the val parquet first)")
        return

    import datasets
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    sample_prompt = frame.iloc[0]["prompt"][0]["content"] if len(frame) else ""
    if r"\boxed{}" not in sample_prompt:
        print("  note existing parquet predates boxed last-line; skip byte-match against the new builder")
        return

    prompt_style = "native"
    if len(frame):
        prompt_style = frame.iloc[0]["extra_info"].get("prompt_style", "native")

    if prompt_style == "native":
        json_path = os.path.join(REPO_ROOT, OFFLINE_JSON)
        if not os.path.exists(json_path):
            print("  skip native check (offline json missing)")
            return
        from build_cvbench_val_parquet import POST_PROMPT

        by_index = {int(row["extra_info"]["index"]): row for _, row in frame.iterrows()}
        with open(json_path) as handle:
            offline = {int(item["index"]): item for item in json.load(handle)}
        mismatches = 0
        for index, item in offline.items():
            row = by_index.get(index)
            if row is None:
                continue
            ours = row["prompt"][0]["content"].replace("<image>", "", 1)
            if ours != f"{item['query']}\n{POST_PROMPT}":
                mismatches += 1
        check(
            mismatches == 0,
            f"{len(by_index) - mismatches}/{len(by_index)} native prompts match eval_visionopd",
        )
        return

    dataset = datasets.load_dataset(
        "nyu-visionx/CV-Bench",
        cache_dir=os.path.join(REPO_ROOT, "cache/datasets"),
    )["test"]
    by_index = {int(doc["idx"]): doc for doc in dataset}
    mismatches = 0
    for _, row in frame.iterrows():
        index = int(row["extra_info"]["index"])
        doc = by_index.get(index)
        if doc is None:
            continue
        ours = row["prompt"][0]["content"].replace("<image>", "", 1)
        expected = build_cvbench_prompt(doc, "lmms_eval")
        if ours != expected:
            if mismatches < 3:
                print(f"  idx={index}\n    expected: {expected!r}\n    parquet:  {ours!r}")
            mismatches += 1
    check(
        mismatches == 0,
        f"{len(frame) - mismatches}/{len(frame)} lmms_eval prompts match cvbench_scoring",
    )


def main() -> None:
    test_extractor()
    test_parser_switch()
    test_answered_flag()
    test_rollup()
    test_v0_replay()
    test_prompt_alignment()

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
