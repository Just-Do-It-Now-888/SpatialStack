#!/usr/bin/env python3
r"""Guard the boxed-primary scoring tier.

    python3 scripts/opsd/tests/test_boxed_primary_scoring.py

From 2026-08-26 the in-training VSI-Bench score may read the answer from inside
``\boxed{}`` instead of scanning the whole response (user decision). That moves
where the reported number comes from, so what has to stay true is:

1. off by default -- every curve published before today reproduces without
   setting the flag (LESSON-013);
2. with a box, the box decides, and prose outside it cannot contribute. This is
   the point of the tier: on a boxed prompt the answer is one token wrapped in
   ~300 tokens of reasoning, which is the shape that produces tail-scan false
   positives;
3. with no box, the ordinary parser still runs. A compliance failure must not
   depress the score below what the model actually answered -- that would make
   the metric measure formatting rather than spatial ability;
4. ``boxed_present`` is reported for every row regardless of data source, because
   NaiveRewardManager only gathers reward_extra_info for dict scores and
   ``_validate`` asserts every extra-info list is batch-length, so a key that
   comes and goes breaks validation on the first mixed batch;
5. the gold answer, boxed, still scores 1.0 for every VSI-Bench row (LESSON-021);
6. strings the prompt itself introduces never become the answer (LESSON-023).
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

import mvopsd_reward as reward  # noqa: E402
import vsibench_scoring as scoring  # noqa: E402

failures: list[str] = []

MCA_SOURCE = "vsibench/object_rel_direction_easy"
NA_SOURCE = "vsibench/room_size_estimation"
MCA_EXTRA = {
    "question_type": "object_rel_direction_easy",
    "options_json": '["A. right", "B. left"]',
    "num_options": 2,
}
NA_EXTRA = {"question_type": "room_size_estimation"}


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def score(source, response, gt, extra, boxed_primary):
    return reward.compute_score(
        data_source=source,
        solution_str=response,
        ground_truth=gt,
        extra_info=extra,
        vsibench_boxed_primary=boxed_primary,
    )


def test_default_is_off() -> None:
    print("\n[1] boxed-primary is opt-in")
    # A response whose box and whose prose disagree separates the two tiers: the
    # default must follow the prose (the old behaviour), the flag the box.
    response = r"Measuring across the frames I get 12 square meters. \boxed{30.9}"
    default = reward.compute_score(
        data_source=NA_SOURCE, solution_str=response, ground_truth="30.9", extra_info=NA_EXTRA
    )
    check(
        default["boxed_present"] == 0.0,
        f"without the flag boxed_present stays 0 (got {default['boxed_present']})",
    )
    # The tail scan reads 30.9 here too (it is the last number), so assert on the
    # flag rather than on the score, which would not distinguish the tiers.
    flagged = score(NA_SOURCE, response, "30.9", NA_EXTRA, True)
    check(flagged["boxed_present"] == 1.0, "with the flag the box is detected")


def test_box_decides() -> None:
    print("\n[2] when a box is present, only the box is scored")
    # Correct answer in the box, wrong number after it. Upstream's math_dapo
    # scores solution_str[-300:], which would read the trailing prose; ours must
    # not, because our degenerate responses ramble past the answer.
    tail = " ".join(f"Image {i}: Kitchen." for i in range(1, 40))
    response = r"The room is roughly 30.9 square meters. \boxed{30.9} " + tail
    boxed = score(NA_SOURCE, response, "30.9", NA_EXTRA, True)
    check(boxed["acc"] >= 0.999, f"box wins over a 40-line enumeration tail (acc {boxed['acc']})")
    check(boxed["boxed_present"] == 1.0, "boxed_present is 1")

    plain = score(NA_SOURCE, response, "30.9", NA_EXTRA, False)
    check(
        plain["acc"] < boxed["acc"],
        f"without the flag the same response scores lower ({plain['acc']} < {boxed['acc']}) "
        "-- this is the false-positive shape the tier exists to remove",
    )

    # Multiple choice: the box holds the letter, the prose contradicts it.
    mca_response = r"The door appears to be on the right. \boxed{B}"
    got = score(MCA_SOURCE, mca_response, "B", MCA_EXTRA, True)
    check(got["acc"] == 1.0 and got["answered"] == 1.0, f"MCA reads B from the box (acc {got['acc']})")

    wrong_box = score(MCA_SOURCE, r"Clearly it is B. \boxed{A}", "B", MCA_EXTRA, True)
    check(
        wrong_box["acc"] == 0.0 and wrong_box["answered"] == 1.0,
        "a wrong box is answered=1 score=0, not unresolved",
    )


def test_no_box_falls_back() -> None:
    print("\n[3] no box falls back to the ordinary parser")
    for name, source, response, gt, extra, want in (
        ("NA correct prose", NA_SOURCE, "The room is roughly 30.9 square meters.", "30.9", NA_EXTRA, 1.0),
        ("MCA correct prose", MCA_SOURCE, "The answer is B.", "B", MCA_EXTRA, 1.0),
    ):
        got = score(source, response, gt, extra, True)
        check(
            got["acc"] >= want - 1e-9 and got["answered"] == 1.0,
            f"{name}: scores {got['acc']} with answered=1 despite no box",
        )
        check(got["boxed_present"] == 0.0, f"{name}: boxed_present is 0")

    # An unclosed box is a truncated answer, not an answer. It must fall back
    # rather than score the fragment.
    truncated = score(NA_SOURCE, r"I estimate the area to be \boxed{30.9", "30.9", NA_EXTRA, True)
    check(truncated["boxed_present"] == 0.0, "an unclosed box does not count as compliance")


def test_key_is_always_present() -> None:
    print("\n[4] boxed_present exists on every row, every data source")
    rows = [
        ("vsibench", NA_SOURCE, "30.9", "30.9", NA_EXTRA),
        ("cvbench", "cvbench/2d", "(B) 3", "B", {"question_type": "count"}),
        ("training pool", "spar_3view", "about 2.5 meters", "2.5", {}),
        ("empty response", NA_SOURCE, "", "30.9", NA_EXTRA),
    ]
    for name, source, response, gt, extra in rows:
        got = score(source, response, gt, extra, True)
        check(
            set(got) == {"score", "acc", "answered", "resp_chars", "boxed_present"},
            f"{name}: key set is {sorted(got)}",
        )


def test_gold_self_consistency() -> None:
    print("\n[5] every gold answer, boxed, scores 1.0 through this path")
    import vsibench_eval_core as core

    if not os.path.isdir(core.DEFAULT_SNAPSHOT):
        print("  skip (VSI-Bench snapshot not present)")
        return
    docs = core.load_docs(core.DEFAULT_SNAPSHOT)
    bad = []
    for doc in docs:
        options = doc.get("options") or []
        extra = {
            "question_type": doc["question_type"],
            "options_json": __import__("json").dumps(options, ensure_ascii=False),
            "num_options": len(options),
        }
        response = "\\boxed{" + str(doc["ground_truth"]) + "}"
        got = score(f"vsibench/{doc['question_type']}", response, str(doc["ground_truth"]), extra, True)
        if got["acc"] < 1.0 - 1e-9 or got["boxed_present"] != 1.0:
            bad.append((doc["question_type"], doc["ground_truth"], got["acc"]))
    check(not bad, f"all {len(docs)} gold answers round-trip (failures: {bad[:3]})")


def test_prompt_strings_are_not_answers() -> None:
    print("\n[6] prompt-introduced strings do not become answers")
    for name, response in (
        ("frame label", "In Frame-4 there is another pile of clothes."),
        ("image enumeration", "Image 178: Kitchen.\nImage 179: Kitchen."),
        ("object id", "Object169's center is visible."),
    ):
        # No box at all, so this exercises the fallback -- the tier must not make
        # these parse *better* than before.
        got = score(NA_SOURCE, response, "173", NA_EXTRA, True)
        check(got["boxed_present"] == 0.0, f"{name}: no box detected")
        check(
            scoring.extract_boxed(response) is None,
            f"{name}: extract_boxed yields None",
        )


def main() -> None:
    test_default_is_off()
    test_box_decides()
    test_no_box_falls_back()
    test_key_is_always_present()
    test_gold_self_consistency()
    test_prompt_strings_are_not_answers()

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
