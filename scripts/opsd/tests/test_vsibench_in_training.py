#!/usr/bin/env python3
"""The in-training VSI-Bench guardrail must report the same number as offline.

The point of scoring VSI-Bench inside the training loop is to watch the same
quantity the offline report publishes. That only holds if two independent code
paths agree:

* offline -- `src/lmms_eval/tasks/vsibench/utils.py` (spatialstack = boxed
  last-line prompt + boxed-primary extract);
* in-training -- `scripts/opsd/mvopsd_reward.py` with
  ``vsibench_boxed_primary=True``, rolled up by
  `verl/trainer/ppo/vsibench_metrics.py`.

This replays an archived unboxed base-model run (5,130 responses). The
historical vLLM harness number 52.58 used a different prompt and is not this
protocol. The two paths here must still agree with each other.

    python3 scripts/opsd/tests/test_vsibench_in_training.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))

os.environ.setdefault("VSIBENCH_PROTOCOL", "spatialstack")

import mvopsd_reward  # noqa: E402
from verl.trainer.ppo.vsibench_metrics import compute_vsibench_metrics  # noqa: E402

ARCHIVE = os.path.join(REPO_ROOT, "logs", "eval", "vllm", "base_anchor_vllm_samples.jsonl")
TOLERANCE = 0.05

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def offline_reference():
    """Score the archive with lmms_eval's own functions, as the offline run did."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vsibench_task_utils",
        os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench", "utils.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = []
    for line in open(ARCHIVE):
        raw = json.loads(line)
        doc = {
            "question_type": raw["question_type"],
            "ground_truth": raw["ground_truth"],
            "options": raw.get("options"),
        }
        rows.append(module.vsibench_process_results(doc, [raw["response"]])["vsibench_score"])
    return module.vsibench_aggregate_results(rows)


def load_tokenizer():
    """Qwen3.5's tokenizer, or None if the weights directory is not on this box.

    Only needed for the token-length series; the accuracy checks below do not
    depend on it, so a missing model downgrades one check instead of failing.
    """
    model_path = os.path.join(REPO_ROOT, "models", "Qwen3.5-4B")
    if not os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def in_training(rows, tokenizer=None):
    data_sources, accuracies, answered, lengths, tokens = [], [], [], [], []
    for raw in rows:
        data_source = f"vsibench/{raw['question_type']}"
        options = raw.get("options") or []
        result = mvopsd_reward.compute_score(
            data_source=data_source,
            solution_str=raw["response"],
            ground_truth=str(raw["ground_truth"]),
            extra_info={
                "question_type": raw["question_type"],
                "options_json": json.dumps(options),
                "num_options": len(options),
            },
            vsibench_boxed_primary=True,
        )
        data_sources.append(data_source)
        accuracies.append(result["acc"])
        answered.append(result["answered"])
        lengths.append(result["resp_chars"])
        if tokenizer is not None:
            tokens.append(len(tokenizer(raw["response"], add_special_tokens=False)["input_ids"]))
    return compute_vsibench_metrics(data_sources, accuracies, answered, lengths, tokens or None)


def main() -> None:
    if not os.path.exists(ARCHIVE):
        print(f"archive missing: {ARCHIVE}")
        raise SystemExit(1)

    rows = [json.loads(line) for line in open(ARCHIVE)]
    print(f"replaying {len(rows)} archived base-model responses\n")

    print("in-training path vs lmms_eval scoring the same responses")
    metrics = in_training(rows, tokenizer=load_tokenizer())
    overall = metrics["val-core/vsibench/overall/acc"] * 100
    reference = offline_reference()
    check(
        "overall matches lmms_eval recomputed on this archive",
        abs(overall - reference) <= TOLERANCE,
        f"{overall:.2f} vs {reference:.2f}",
    )

    print("\nthe roll-up is the unweighted per-type mean, not the micro average")
    micro = metrics["val-aux/vsibench/micro/acc"] * 100
    check(
        "micro average differs, so the weighting is actually being applied",
        abs(micro - overall) > 0.1,
        f"micro {micro:.2f} vs overall {overall:.2f}",
    )
    check(
        "the three direction difficulties are merged into one vote",
        int(metrics["val-aux/vsibench/num_types"]) == 8,
        f"{int(metrics['val-aux/vsibench/num_types'])} types",
    )

    print("\ndiagnostics that separate a wrong answer from a missing one")
    check(
        "answered fraction is reported",
        0.97 <= metrics["val-core/vsibench/answered/frac"] <= 1.0,
        f"{metrics['val-core/vsibench/answered/frac'] * 100:.2f}%",
    )
    check(
        "response length is reported and matches the base model's terse answers",
        metrics["val-core/vsibench/resp_chars/median"] < 40,
        f"median {metrics['val-core/vsibench/resp_chars/median']:.0f} chars",
    )
    if "val-core/vsibench/resp_tokens/mean" in metrics:
        # The token mean is what the training curve reports for its own
        # rollouts, so it has to exist on the validation side too or the two
        # cannot be read together.
        mean_tokens = metrics["val-core/vsibench/resp_tokens/mean"]
        median_tokens = metrics["val-aux/vsibench/resp_tokens/median"]
        check(
            "response length is also reported in tokens, the training curve's unit",
            median_tokens < 10 <= mean_tokens,
            f"mean {mean_tokens:.1f} tokens, median {median_tokens:.0f}",
        )
        # On this archive the base model answers in 3 tokens at the median but
        # averages ~97, because a small minority writes full reasoning and 2% of
        # rows hit the 1024 cap. Neither statistic alone describes the run: a
        # mean that stays put can hide the median doubling, and a stable median
        # can hide a growing tail. Both are reported for that reason.
        check(
            "the mean is far above the median, so a tail is already present at step 0",
            mean_tokens > 5 * max(median_tokens, 1.0),
            f"mean/median = {mean_tokens / max(median_tokens, 1.0):.0f}x",
        )
    else:
        print("  skip token length: models/Qwen3.5-4B tokenizer not available here")

    print("\nrows from another benchmark are ignored rather than mis-scored")
    mixed = compute_vsibench_metrics(["cvbench/COCO/Count", "cvbench/ADE20K/Count"], [1.0, 0.0])
    check("a CV-Bench-only batch produces no VSI metrics", mixed == {})

    print("\nper question type")
    for key in sorted(metrics):
        if key.startswith("val-aux/vsibench/type/") and key.endswith("/acc"):
            name = key[len("val-aux/vsibench/type/") : -len("/acc")]
            n = metrics.get(f"val-aux/vsibench/type/{name}/num_samples", "-")
            print(f"  {name:<32} {metrics[key] * 100:>6.2f}   n={n}")

    print("\nFAILED: " + ", ".join(failures) if failures else "\nOK")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
