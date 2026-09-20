#!/usr/bin/env python3
"""Where each side's `enable_thinking` comes from, and what it renders to.

The three sides render their prompts through three different code paths, and
only one config key separates them, so a regression here is silent: the run
would train and score normally while the teacher's targets came from the wrong
mode. What this pins down:

* student rollouts -- `RLHFDataset` / `AgentLoopWorker` use
  `data.apply_chat_template_kwargs`;
* in-loop validation -- the same kwargs, because `_validate` goes through the
  same agent loop;
* teacher -- `RayPPOTrainer._teacher_apply_chat_template_kwargs`, which is
  allowed to override `enable_thinking` and nothing else.

On Qwen3 templates the difference is visible in the generation prompt: thinking
off pre-closes the block (`<think>\n\n</think>\n\n`), thinking on leaves it open
(`<think>\n`).

This file used to assert that the teacher had thinking *on*, which is what
mvopsd.yaml shipped between 2026-08-21 and 2026-08-26. That setting was then
shown to be the sole cause of the response-length explosion
(`20260824_teacher_thinking_off`, LESSON-029) and the config moved to
non-thinking on both sides. So the assertions are split in two:

* the *mechanism* -- rendering with True leaves the block open and with False
  pre-closes it -- is invariant and tested against both values explicitly;
* the *configured* value is tested separately, so flipping it is a deliberate
  one-line change here rather than something a run can do silently.

    python3 scripts/opsd/tests/test_thinking_split.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))
sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize_config_dir  # noqa: E402

from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402

CONFIG_DIR = os.path.join(REPO_ROOT, "verl_pkg", "verl", "trainer", "config")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "Qwen3.5-4B")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def load_config():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="mvopsd")


def main() -> None:
    config = load_config()
    student_kwargs = dict(config.data.apply_chat_template_kwargs or {})
    # _teacher_apply_chat_template_kwargs only reads config, so it can be called
    # without building a trainer (which would need Ray and eight GPUs).
    teacher_kwargs = RayPPOTrainer._teacher_apply_chat_template_kwargs(
        type("_ConfigOnly", (), {"config": config})()
    )

    print("mvopsd.yaml resolves to the requested split")
    check(
        "student prompts disable thinking",
        student_kwargs.get("enable_thinking") is False,
        f"{student_kwargs}",
    )
    # Both sides non-thinking since 2026-08-26. Teacher thinking on was the sole
    # cause of the length explosion in the 20260821 arms: flipping just this
    # field moved VSI rule_only 20.85 -> 52.57 at step 25 (LESSON-029). Change
    # this expectation only together with mvopsd.yaml and a registry entry.
    check(
        "teacher prompts also disable thinking (homomorphic with the student)",
        teacher_kwargs.get("enable_thinking") is False,
        f"{teacher_kwargs}",
    )
    check(
        "the override touches enable_thinking and nothing else",
        {k: v for k, v in teacher_kwargs.items() if k != "enable_thinking"}
        == {k: v for k, v in student_kwargs.items() if k != "enable_thinking"},
    )

    print("\nrollout.n is 1")
    check(
        "one rollout per prompt",
        int(config.actor_rollout_ref.rollout.n) == 1,
        f"n={config.actor_rollout_ref.rollout.n}",
    )
    check(
        "one mini-batch per step keeps the update on-policy",
        int(config.data.train_batch_size) == int(config.actor_rollout_ref.actor.ppo_mini_batch_size),
        f"batch {config.data.train_batch_size} vs mini-batch "
        f"{config.actor_rollout_ref.actor.ppo_mini_batch_size}",
    )

    print("\nnull inherits the student's setting rather than defaulting to on")
    check(
        "teacher_enable_thinking=null gives the student's value",
        RayPPOTrainer._teacher_apply_chat_template_kwargs(
            type("_ConfigOnly", (), {"config": compose_with_teacher_thinking(None)})()
        ).get("enable_thinking")
        is False,
    )
    check(
        "teacher_enable_thinking=True is still honoured when asked for",
        RayPPOTrainer._teacher_apply_chat_template_kwargs(
            type("_ConfigOnly", (), {"config": compose_with_teacher_thinking(True)})()
        ).get("enable_thinking")
        is True,
    )

    if not os.path.exists(os.path.join(MODEL_PATH, "chat_template.jinja")):
        print(f"\nskip template rendering: {MODEL_PATH} not present")
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        messages = [{"role": "user", "content": "How many chairs are in the room?"}]

        def render(**kwargs) -> str:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **kwargs
            )

        # The mechanism, tested against both values rather than against whatever
        # the config currently says. This is what must never change silently:
        # the flag's only effect is whether the reasoning block is left open.
        print("\nthe flag's rendering effect, both values")
        closed = render(enable_thinking=False)
        opened = render(enable_thinking=True)
        check(
            "enable_thinking=False pre-closes the reasoning block",
            closed.endswith("<think>\n\n</think>\n\n"),
            repr(closed[-24:]),
        )
        check(
            "enable_thinking=True leaves it open",
            opened.endswith("<think>\n") and "</think>" not in opened,
            repr(opened[-24:]),
        )
        check(
            "only the tail differs, so the question itself is identical",
            closed[: closed.index("<think>")] == opened[: opened.index("<think>")],
        )

        # And the configured sides, which under the current decision must render
        # to the same string: the teacher scores the student's own tokens, so a
        # difference here means the two are conditioned on different states.
        print("\nthe configured sides render identically (both non-thinking)")
        student = render(**student_kwargs)
        teacher = render(**teacher_kwargs)
        check(
            "student and teacher generation prompts are byte-identical",
            student == teacher,
            f"student {student[-24:]!r} vs teacher {teacher[-24:]!r}",
        )
        student_len = len(tokenizer(student, add_special_tokens=False)["input_ids"])
        teacher_len = len(tokenizer(teacher, add_special_tokens=False)["input_ids"])
        print(f"  prompt tokens: student {student_len}, teacher {teacher_len}")

    # The conditional used to sit inside the print call, so the "OK" branch
    # evaluated to a string that was never written anywhere and a passing run
    # printed nothing at all.
    print(f"\nFAILED: {', '.join(failures)}" if failures else "\nOK")
    raise SystemExit(1 if failures else 0)


def compose_with_teacher_thinking(value):
    override = "null" if value is None else str(value)
    # Hydra parses True/False from the string form, so str(True) -> "True" is
    # correct here and does not need quoting.
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="mvopsd",
            overrides=[f"actor_rollout_ref.actor.self_distillation.teacher_enable_thinking={override}"],
        )


if __name__ == "__main__":
    main()
