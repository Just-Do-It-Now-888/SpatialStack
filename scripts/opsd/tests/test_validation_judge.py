#!/usr/bin/env python3
"""Tests for the in-training judge tier.

    python3 scripts/opsd/tests/test_validation_judge.py

The tier's whole reason to exist is scoring the rows the rules cannot read, but
it does that by putting the policy to sleep mid-validation and paging a 61 GB
model onto the same cards. That trade is only acceptable if every way it can go
wrong ends in the rule-only score training already had, so most of what is
checked here is failure behaviour: an unreachable judge, a judge that will not
wake, a judge that raises mid-grade, and a judge that answers with something
other than yes or no all have to leave the run standing.
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "verl_pkg"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        FAILURES.append(message)


class FakeRollout:
    """The agent loop manager, which must not be touched from the judge tier.

    It sleeps its own engines at the end of every `generate_sequences` and wakes
    them at the start of the next one. A second sleep from this side runs a
    level-2 sleep over weights vLLM has already released and dies in
    `buffer.cpu()` with CUDA "invalid argument", which is how this was found.
    """

    def __init__(self):
        self.slept = 0

    def sleep(self):
        self.slept += 1
        raise RuntimeError("CUDA error: invalid argument (the engines are already asleep)")


class FakeJudge:
    """Stands in for ValidationJudge, recording the handshake it was put through."""

    def __init__(self, verdicts=None, min_rows=16, wake_ok=True, raise_on_grade=False, answers=None):
        self.verdicts = verdicts
        self.min_rows = min_rows
        self.wake_ok = wake_ok
        self.raise_on_grade = raise_on_grade
        self.woke = 0
        self.slept = 0
        self.graded_rows = None
        # Strict mode: list of ("mca", letter) / ("na", value) / (kind, None).
        self.answers = answers
        self.extract_calls = 0

    def wake(self):
        self.woke += 1
        return self.wake_ok

    def sleep(self):
        self.slept += 1
        return True

    def grade(self, questions, ground_truths, responses):
        if self.raise_on_grade:
            raise RuntimeError("judge exploded")
        self.graded_rows = list(zip(questions, ground_truths, responses))
        stats = {"rows": len(questions), "yes": 0, "no": 0, "unparsed": 0, "seconds": 1.0}
        for verdict in self.verdicts:
            if verdict == 1.0:
                stats["yes"] += 1
            elif verdict == 0.0:
                stats["no"] += 1
            else:
                stats["unparsed"] += 1
        return list(self.verdicts), stats

    def extract_answers(self, questions, responses, kinds, options=None):
        if self.raise_on_grade:
            raise RuntimeError("judge exploded")
        self.extract_calls += 1
        self.graded_rows = list(zip(questions, responses, kinds))
        answers = list(self.answers)
        stats = {
            "rows": len(questions),
            "found": sum(1 for _, v in answers if v is not None),
            "none": sum(1 for _, v in answers if v is None),
            "errors": 0,
            "seconds": 1.0,
        }
        return answers, stats


def bind_vsibench_tier(trainer) -> None:
    """Give the stub the real VSI-Bench tier, which `_maybe_judge_validation` calls first.

    The rows here are CV-Bench, so that tier bows out immediately -- but it is
    on the code path, and stubbing it out would hide a change in what it does to
    the CV-Bench rows on the way through.
    """
    trainer._load_opsd_module = RayPPOTrainer._load_opsd_module
    trainer._judge_vsibench_rows = types.MethodType(RayPPOTrainer._judge_vsibench_rows, trainer)
    # Both VSI-Bench modes have to be bound: `_judge_vsibench_rows` dispatches to
    # the strict one, and a stub that lacks it fails with AttributeError rather
    # than telling you the dispatch changed.
    trainer._judge_vsibench_strict = types.MethodType(RayPPOTrainer._judge_vsibench_strict, trainer)


def run_tier(judge, answered, accuracies, sources=None, n=None, audit=False, mode="cascade", gts=None):
    """Invoke the trainer method against stubs and return (metrics, state).

    `gts` defaults to CV-Bench's "(A)" form because that is what these rows are.
    VSI-Bench ground truth is a bare letter -- `data/eval/vsibench_verl` stores
    'C', not '(C)' -- so the strict tests below pass their own.
    """
    n = n or len(answered)
    sources = sources or ["cvbench/2D_ADE20K"] * n
    rollout = FakeRollout()
    trainer = types.SimpleNamespace(
        _get_validation_judge=lambda: judge,
        async_rollout_manager=rollout,
        _validation_judge_audit=audit,
        _validation_judge_mode=mode,
    )
    bind_vsibench_tier(trainer)
    extra = {"acc": list(accuracies), "answered": list(answered)}
    metrics = RayPPOTrainer._maybe_judge_validation(
        trainer,
        inputs=[f"question {i}" for i in range(n)],
        outputs=[f"response {i}" for i in range(n)],
        gts=gts or [f"({chr(65 + i % 4)})" for i in range(n)],
        # _validate hands over a list of per-batch arrays, not a flat list.
        data_sources=[sources[: n // 2], sources[n // 2 :]],
        reward_extra_infos_dict=extra,
    )
    return metrics, extra, rollout


def test_disabled():
    print("\n[disabled] no judge configured -> tier is inert")
    metrics, extra, rollout = run_tier(None, [0.0] * 40, [0.0] * 40)
    check(metrics == {}, "reports nothing")
    check(rollout.slept == 0, "the rollout engines are left alone")
    check(extra["acc"] == [0.0] * 40, "scores are untouched")


def test_below_threshold():
    print("\n[threshold] a handful of unresolved rows is not worth 61 GB")
    answered = [1.0] * 96 + [0.0] * 4
    judge = FakeJudge(verdicts=[1.0] * 4, min_rows=16)
    metrics, extra, rollout = run_tier(judge, answered, [0.0] * 100)
    check(judge.woke == 0, "judge stays asleep")
    check(rollout.slept == 0, "the rollout engines are left alone")
    check(metrics["val-aux/cvbench/judge/pending"] == 4, "the 4 unresolved rows are still reported")
    check(extra["acc"] == [0.0] * 100, "rule scores stand")


def test_grades_only_unresolved():
    print("\n[selection] only rule-unresolved CV-Bench rows reach the judge")
    # Rows 0-19 answered and correct, 20-39 unresolved, 40-59 answered and wrong.
    answered = [1.0] * 20 + [0.0] * 20 + [1.0] * 20
    accuracies = [1.0] * 20 + [0.0] * 20 + [0.0] * 20
    judge = FakeJudge(verdicts=[1.0] * 15 + [0.0] * 5)
    metrics, extra, rollout = run_tier(judge, answered, accuracies)

    check(judge.woke == 1 and judge.slept == 1, "judge wakes once and goes back to sleep")
    check(rollout.slept == 0, "the trainer never sleeps the rollout engines itself")
    check(len(judge.graded_rows) == 20, "exactly the 20 unresolved rows are sent")
    check(judge.graded_rows[0][0] == "question 20", "the sent rows are the unresolved ones")
    check(extra["acc"][:20] == [1.0] * 20, "already-correct rows keep their score")
    check(extra["acc"][20:35] == [1.0] * 15, "judge promotes the rows it read as correct")
    check(extra["acc"][35:40] == [0.0] * 5, "judge leaves the rows it read as wrong")
    check(extra["acc"][40:] == [0.0] * 20, "answered-but-wrong rows are not re-graded")
    check(extra["answered"] == answered, "answered stays a record of what the rules read")
    check(metrics["val-aux/cvbench/judge/graded"] == 20, "graded count is reported")


def test_non_cvbench_rows_ignored():
    print("\n[selection] rows from another benchmark are left alone")
    sources = ["vsibench/obj_count"] * 30 + ["cvbench/3D_Omni3D"] * 30
    judge = FakeJudge(verdicts=[1.0] * 30)
    metrics, extra, _ = run_tier(judge, [0.0] * 60, [0.0] * 60, sources=sources)
    check(metrics["val-aux/cvbench/judge/pending"] == 30, "only the CV-Bench half is pending")
    check(extra["acc"][:30] == [0.0] * 30, "the other benchmark is untouched")
    check(extra["acc"][30:] == [1.0] * 30, "the CV-Bench half is re-graded")


def test_unparsed_keeps_rule_score():
    print("\n[verdicts] an unreadable judge reply must not look like a wrong answer")
    judge = FakeJudge(verdicts=[1.0] * 10 + [None] * 10)
    metrics, extra, _ = run_tier(judge, [0.0] * 20, [0.0] * 20)
    check(extra["acc"][10:] == [0.0] * 10, "unreadable rows keep the rule verdict")
    check(metrics["val-aux/cvbench/judge/unparsed"] == 10, "and are counted separately")
    check(metrics["val-aux/cvbench/judge/graded"] == 10, "graded counts only what was read")


def test_wake_failure_is_survivable():
    print("\n[failure] a judge that will not wake degrades to the rule score")
    judge = FakeJudge(verdicts=[1.0] * 20, wake_ok=False)
    metrics, extra, rollout = run_tier(judge, [0.0] * 20, [0.0] * 20)
    check(extra["acc"] == [0.0] * 20, "scores fall back to the rules")
    check(metrics["val-aux/cvbench/judge/woken"] == 0.0, "the failure is visible in the metrics")
    check(judge.slept == 0, "a judge that never woke is not told to sleep")


def test_grade_exception_is_survivable():
    print("\n[failure] a judge that raises mid-grade still hands the cards back")
    judge = FakeJudge(verdicts=[], raise_on_grade=True)
    metrics, extra, _ = run_tier(judge, [0.0] * 20, [0.0] * 20)
    check(extra["acc"] == [0.0] * 20, "scores fall back to the rules")
    check(judge.slept == 1, "judge is put to sleep anyway, or the next rollout OOMs")


def test_audit_mode_measures_without_scoring():
    print("\n[audit] auditing the rule tier must not become the score")
    # Rows 0-9 the rules read as correct, 10-19 the rules could not read.
    answered = [1.0] * 10 + [0.0] * 10
    accuracies = [1.0] * 10 + [0.0] * 10
    # The judge disagrees with the rules on 3 of the 10 audited rows, and
    # resolves the 10 unread ones as correct.
    verdicts = [1.0] * 7 + [0.0] * 3 + [1.0] * 10
    judge = FakeJudge(verdicts=verdicts)
    metrics, extra, _ = run_tier(judge, answered, accuracies, audit=True)

    check(len(judge.graded_rows) == 20, "every CV-Bench row is sent, not just the unread ones")
    check(extra["acc"][:10] == [1.0] * 10, "rule-scored rows keep the rule verdict")
    check(extra["acc"][10:] == [1.0] * 10, "unread rows are still the only ones re-scored")
    check(metrics["val-aux/cvbench/judge/graded"] == 10, "graded counts only the rows it may overwrite")
    check(metrics["val-aux/cvbench/judge/audited"] == 10, "audited counts the rest")
    check(
        abs(metrics["val-aux/cvbench/judge/rule_disagree_frac"] - 0.3) < 1e-9,
        "the 30% rule/judge disagreement is reported",
    )


def test_strict_grades_every_row_and_can_lower():
    print("\n[strict] extract mode grades all VSI rows and is allowed to lower them")
    # Two MCA rows the rules scored full marks, two the rules scored zero.
    sources = ["vsibench/object_rel_direction_easy"] * 4
    answered = [1.0, 1.0, 0.0, 0.0]
    accuracies = [1.0, 1.0, 0.0, 0.0]
    # The judge finds no answer in row 0 (the `Image N:` false positive it
    # exists to catch), agrees on row 1, and reads row 3 correctly.
    judge = FakeJudge(answers=[("mca", None), ("mca", "B"), ("mca", None), ("mca", "D")])
    metrics, extra, _ = run_tier(
        judge, answered, accuracies, sources=sources, mode="extract",
        gts=["A", "B", "C", "D"],
    )
    check(judge.extract_calls == 1, "the strict path is taken, not the cascade")
    check(len(judge.graded_rows) == 4, "all 4 rows are sent, including the ones rules read")
    check(extra["acc"][0] == 0.0, "a rule false positive is lowered to 0 (the point of this tier)")
    check(extra["acc"][1] == 1.0, "a row both tiers agree on keeps full marks")
    check(extra["acc"][3] == 1.0, "a row the rules missed is raised")
    check(metrics["val-aux/vsibench/judge/lowered"] == 1, "the lowering is counted")
    check(metrics["val-aux/vsibench/judge/raised"] == 1, "the raising is counted")
    check(metrics["val-aux/vsibench/judge/strict"] == 1.0, "the mode is visible in the metrics")
    check(
        metrics["val-core/vsibench/rule_only/acc"] is not None,
        "the rule tier is still reported next to the primary number",
    )
    check(extra["answered"][0] == 0.0, "a row the judge could not read counts as unanswered")
    check(extra["answered"][3] == 1.0, "a row the judge read counts as answered")


def test_strict_wake_failure_says_so():
    print("\n[strict] a judge that will not wake must not silently report rule scores")
    sources = ["vsibench/object_rel_direction_easy"] * 4
    judge = FakeJudge(answers=[], wake_ok=False)
    metrics, extra, _ = run_tier(
        judge, [1.0] * 4, [1.0] * 4, sources=sources, mode="extract"
    )
    check(extra["acc"] == [1.0] * 4, "scores fall back to the rules")
    check(
        metrics.get("val-aux/vsibench/judge/fellback_to_rules") == 1.0,
        "and the point is flagged as a different tier, not left to look normal",
    )
    check(judge.slept == 0, "a judge that never woke is not told to sleep")


def test_missing_answered_series():
    print("\n[compat] a reward manager without `answered` cannot be judged")
    rollout = FakeRollout()
    judge = FakeJudge(verdicts=[1.0])
    trainer = types.SimpleNamespace(
        _get_validation_judge=lambda: judge,
        async_rollout_manager=rollout,
        _validation_judge_mode="cascade",
    )
    bind_vsibench_tier(trainer)
    metrics = RayPPOTrainer._maybe_judge_validation(
        trainer,
        inputs=["q"],
        outputs=["r"],
        gts=["(A)"],
        data_sources=[["cvbench/2D_ADE20K"]],
        reward_extra_infos_dict={"acc": [0.0]},
    )
    check(metrics == {}, "the tier bows out instead of raising")
    check(judge.woke == 0, "and does not wake the judge")


def test_tracker_metrics_drop_judge_series() -> None:
    print("\n[tracker] judge aux metrics are not published to wandb")
    raw = {
        "val-core/vsibench/overall/acc": 0.45,
        "val-core/vsibench/rule_only/acc": 0.46,
        "val-aux/vsibench/judge/pending": 12.0,
        "val-aux/cvbench/judge/graded": 5.0,
        "actor/loss": 0.1,
    }
    published = RayPPOTrainer._tracker_metrics(raw)
    check("val-core/vsibench/overall/acc" in published, "overall acc is kept")
    check("actor/loss" in published, "training metrics are kept")
    check(not any("/judge/" in key for key in published), "judge aux series are dropped")
    check(not any("rule_only" in key for key in published), "rule_only diagnostic is dropped")


if __name__ == "__main__":
    test_disabled()
    test_below_threshold()
    test_grades_only_unresolved()
    test_non_cvbench_rows_ignored()
    test_unparsed_keeps_rule_score()
    test_wake_failure_is_survivable()
    test_grade_exception_is_survivable()
    test_audit_mode_measures_without_scoring()
    test_strict_grades_every_row_and_can_lower()
    test_strict_wake_failure_says_so()
    test_missing_answered_series()
    test_tracker_metrics_drop_judge_series()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("all checks passed")
