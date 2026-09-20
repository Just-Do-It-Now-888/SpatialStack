#!/usr/bin/env python3
"""Tests for per-step sample provenance and the frozen teacher.

    python3 scripts/opsd/tests/test_sample_provenance.py

Both features exist to answer questions after the fact, which is exactly when
they are too late to fix. The provenance columns are what lets a jump in the
CV-Bench curve be attributed to the data mix rather than to the method, and
`teacher_probe` is what lets "the teacher is frozen" be checked rather than
assumed (LESSON-014). So what is checked here is that they stay correct when the
batch is not the happy path: a dataset without `extra_info`, a reward-free arm
whose scores are all zero, and an EMA teacher that must still move.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "verl_pkg"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

import torch  # noqa: E402

from verl.trainer.ppo.sample_provenance import (  # noqa: E402
    compute_sample_provenance_metrics,
    extract_provenance,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {message}")
    if not condition:
        FAILURES.append(message)


class FakeBatch:
    """The parts of DataProto the provenance helpers read."""

    def __init__(self, non_tensor_batch, batch=None):
        self.non_tensor_batch = non_tensor_batch
        self.batch = batch or {}


def make_batch(scores=None):
    """Two prompts x two rollouts, drawn from two sources with different K."""
    extra_info = [
        {
            "sample_id": "vlm3r_scannet/aaa",
            "source": "vlm3r_scannet",
            "subset": "scannet",
            "scene_id": "scene0548_00",
            "k_views_student": 1,
            "n_views_teacher": 8,
            "privilege_bucket": "strong",
            "view_selection": "pure_random",
            "answer_view_sensitive": False,
        },
        {
            "sample_id": "vlm3r_scannet/aaa",
            "source": "vlm3r_scannet",
            "subset": "scannet",
            "scene_id": "scene0548_00",
            "k_views_student": 1,
            "n_views_teacher": 8,
            "privilege_bucket": "strong",
            "view_selection": "pure_random",
            "answer_view_sensitive": False,
        },
        {
            "sample_id": "spar/bbb",
            "source": "spar",
            "subset": "spar7m",
            "scene_id": "scene0011_00",
            "k_views_student": 4,
            "n_views_teacher": 32,
            "privilege_bucket": "strong",
            "view_selection": "coverage",
            "answer_view_sensitive": True,
        },
        {
            "sample_id": "spar/bbb",
            "source": "spar",
            "subset": "spar7m",
            "scene_id": "scene0011_00",
            "k_views_student": 4,
            "n_views_teacher": 32,
            "privilege_bucket": "strong",
            "view_selection": "coverage",
            "answer_view_sensitive": True,
        },
    ]
    tensors = {
        # 10, 10, 30, 30 response tokens.
        "response_mask": torch.tensor(
            [[1] * 10 + [0] * 20, [1] * 10 + [0] * 20, [1] * 30, [1] * 30],
            dtype=torch.long,
        ),
    }
    if scores is not None:
        tensors["token_level_scores"] = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    return FakeBatch(
        {
            "extra_info": extra_info,
            "uid": ["uid-a", "uid-a", "uid-b", "uid-b"],
            "data_source": ["vlm3r_scannet", "vlm3r_scannet", "spar", "spar"],
        },
        tensors,
    )


def test_extract_provenance() -> None:
    print("extract_provenance")
    columns = extract_provenance(make_batch())

    check(
        all(len(values) == 4 for values in columns.values()),
        "every column is as long as the batch, so _dump_generations keeps them",
    )
    check(columns["sample_id"][0] == "vlm3r_scannet/aaa", "sample_id survives to the dump")
    check(columns["k_views_student"] == [1, 1, 4, 4], "the student's view budget is recorded per row")
    check(columns["n_views_teacher"] == [8, 8, 32, 32], "the teacher's view budget is recorded per row")
    check(
        columns["uid"] == ["uid-a", "uid-a", "uid-b", "uid-b"],
        "uid is carried so the four rollouts of one prompt can be grouped",
    )
    check(
        "answer_view_sensitive" in columns,
        "a column of all-False values is still emitted, since False is informative here",
    )

    empty = extract_provenance(FakeBatch({"uid": ["a"]}))
    check(
        empty == {"uid": ["a"]},
        "a dataset without extra_info degrades to what verl dumped before, not a crash",
    )


def test_composition_metrics() -> None:
    print("compute_sample_provenance_metrics")
    metrics = compute_sample_provenance_metrics(make_batch())

    check(metrics["data/step/num_sequences"] == 4, "sequence count is the repeated batch size")
    check(metrics["data/step/num_prompts"] == 2, "prompt count divides out rollout.n")
    check(metrics["data/step/num_unique_samples"] == 2, "unique samples are counted, not rows")
    check(
        metrics["data/step/source/vlm3r_scannet/frac"] == 0.5
        and metrics["data/step/source/spar/frac"] == 0.5,
        "source fractions sum over the batch",
    )
    check(metrics["data/step/k_views_student/mean"] == 2.5, "mean K reflects the sampled budget mix")
    check(
        metrics["data/step/answer_view_sensitive/frac"] == 0.5,
        "the view-sensitive fraction says how much of the batch the teacher can help on",
    )
    check(
        metrics["data/step/source/vlm3r_scannet/response_length/mean"] == 10.0
        and metrics["data/step/source/spar/response_length/mean"] == 30.0,
        "response length is split by source, which is where prose drift would first show",
    )
    check(
        not any(key.endswith("/score/mean") for key in metrics),
        "an all-zero reward emits no per-source score series (the v0 arm is reward-free)",
    )

    scored = compute_sample_provenance_metrics(make_batch(scores=[0.0, 1.0, 0.0, 0.0]))
    check(
        scored["data/step/source/vlm3r_scannet/score/mean"] == 0.5,
        "a non-trivial reward does get broken out by source",
    )

    check(
        compute_sample_provenance_metrics(FakeBatch({})) == {},
        "a batch without extra_info yields no metrics rather than raising",
    )


class FakeParam:
    def __init__(self, value):
        self.data = torch.full((8,), float(value))

    @property
    def device(self):
        return self.data.device


class FakeModule:
    def __init__(self, value):
        self._params = [FakeParam(value)]

    def parameters(self):
        return iter(self._params)

    def buffers(self):
        return iter([])


class FakeConfig(dict):
    """Stands in for the actor config, which is attribute-accessed here."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def make_actor(teacher_regularization, teacher_value=2.0, student_value=10.0):
    actor = object.__new__(DataParallelPPOActor)
    actor.config = FakeConfig(
        policy_loss={"loss_mode": "vopd"},
        self_distillation=FakeConfig(
            teacher_model_source="legacy",
            teacher_regularization=teacher_regularization,
            teacher_update_rate=0.5,
        ),
    )
    actor.teacher_module = FakeModule(teacher_value)
    actor.actor_module = FakeModule(student_value)
    return actor


def test_frozen_teacher() -> None:
    print("frozen teacher")
    frozen = make_actor("frozen")
    before = frozen._teacher_probe()
    frozen._update_teacher()
    after = frozen._teacher_probe()
    check(before == after == 16.0, "frozen leaves the teacher weights bit-identical")

    ema = make_actor("ema")
    ema_before = ema._teacher_probe()
    ema._update_teacher()
    ema_after = ema._teacher_probe()
    check(ema_before == 16.0 and ema_after == 48.0, "ema still moves the teacher toward the student")
    check(
        ema_before != ema_after,
        "teacher_probe changes under ema, so a constant probe really does mean frozen",
    )

    no_teacher = make_actor("frozen")
    no_teacher.teacher_module = None
    check(no_teacher._teacher_probe() is None, "no teacher module reports no probe instead of raising")


def test_config_accepts_frozen() -> None:
    print("config validation")
    from verl.workers.config.actor import SelfDistillationConfig

    config = SelfDistillationConfig(
        teacher_regularization="frozen",
        teacher_update_rate=0.0,
        teacher_always_on=True,
        teacher_image_key="teacher_images",
    )
    check(config.teacher_regularization == "frozen", "'frozen' passes validation")

    try:
        SelfDistillationConfig(
            teacher_regularization="frozen",
            teacher_model_source="current",
            teacher_always_on=True,
            teacher_image_key="teacher_images",
        )
    except ValueError:
        check(True, "frozen + teacher_model_source='current' is rejected, having no teacher to freeze")
    else:
        check(False, "frozen + teacher_model_source='current' is rejected, having no teacher to freeze")


def main() -> int:
    test_extract_provenance()
    test_composition_metrics()
    test_frozen_teacher()
    test_config_accepts_frozen()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
