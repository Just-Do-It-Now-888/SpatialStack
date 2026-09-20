#!/usr/bin/env python3
"""Did the 2026-08-21 config changes move the primary metric?

`val-core/vsibench/overall/acc` is the number every training decision will be
read off. Three things changed underneath it on 2026-08-21 -- rollout.n 4->1, a
new `resp_tokens` column threaded through the validation metric path, and
`enable_thinking` split between teacher and student -- and none of them is
*supposed* to move it. "Supposed to" is not evidence, and the failure mode here
is silent: ISSUE-203 already showed that a val prompt drifting by one turn moves
the score by ~1 point with nothing in the logs to say so.

So this replays the archived in-training validation dump (5,130 real rows from
`20260820_qwen35base_mvopsd_khalf_pipecheck` step 0) through the live metric code
and checks invariance rather than plausibility:

1. row alignment -- the dump has no `data_source`, so it is recovered by index
   from the val parquet. Every per-type number depends on that being right.
2. the score is a pure function of (data_source, acc): feeding an absurd
   `resp_tokens` column must not perturb it by one bit.
3. the roll-up is still VSI-Bench's own: direction merged to one vote, unweighted
   mean over 8 types, cross-checked against lmms_eval's aggregator.
4. `val-core` gains exactly one key and loses none.
5. `_response_token_lengths` counts what verl's training-side
   `response_length/mean` counts, so the two series share an axis.
6. the student/validation prompt still renders byte-identically to the one the
   archived scores were produced from.

    python3 scripts/opsd/tests/test_val_metric_invariance.py
"""

from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("VSIBENCH_PROTOCOL", "spatialstack")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from verl.protocol import DataProto  # noqa: E402
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask  # noqa: E402
from verl.trainer.ppo.vsibench_metrics import DIRECTION_TYPES, compute_vsibench_metrics  # noqa: E402

VAL_PARQUET = os.path.join(REPO_ROOT, "data", "eval", "vsibench_verl", "vsibench_val.parquet")
VAL_DUMP = os.path.join(
    REPO_ROOT, "logs", "val", "20260820_qwen35base_mvopsd_khalf_pipecheck", "0.jsonl"
)
MODEL_PATH = os.path.join(REPO_ROOT, "models", "Qwen3.5-4B")
CONFIG_DIR = os.path.join(REPO_ROOT, "verl_pkg", "verl", "trainer", "config")

CORE = "val-core/vsibench/overall/acc"

# What the trainer itself logged for this dump, from
# logs/train/20260820_qwen35base_mvopsd_khalf_pipecheck/train_20260820_173529.log:6276
# (step 0, the released Qwen3.5-4B on the in-training harness). Replaying the
# dump has to land on this exactly, or the harness below is not measuring the
# same thing the training curve does and every check after it is vacuous.
TRAINER_LOGGED_STEP0 = 0.5362459737332027

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def val_metrics(data_sources, uids, extra):
    """The exact call the trainer makes at the end of every validation pass."""
    return RayPPOTrainer._val_metrics_update(object(), np.array(data_sources), uids, extra, [])


# ---------------------------------------------------------------- [1] alignment


def load_replay():
    frame = pd.read_parquet(VAL_PARQUET)
    rows = [json.loads(line) for line in open(VAL_DUMP)]
    print(f"\n[1] row alignment ({len(rows)} dumped rows vs {len(frame)} parquet rows)")
    check("the dump and the val parquet have the same length", len(rows) == len(frame))

    data_sources = [str(source) for source in frame["data_source"]]
    # data.validation_shuffle is False, so dump row i is parquet row i. If that
    # ever stops holding, the ground truths stop matching and every per-type
    # number silently becomes a mixture of the wrong question types.
    mismatch = sum(
        1
        for i, row in enumerate(rows)
        if str(row["gts"]) != str(frame.iloc[i]["reward_model"]["ground_truth"])
    )
    check(
        "every dumped ground truth matches its parquet row, so the recovery is sound",
        mismatch == 0,
        f"{mismatch} mismatched",
    )
    return frame, rows, data_sources


# --------------------------------------------- [2] purity and [4] key inventory


def test_purity(rows, data_sources):
    print("\n[2] the score is a pure function of (data_source, acc)")
    uids = [str(i) for i in range(len(rows))]
    base_extra = {
        "reward": [row["reward"] for row in rows],
        "acc": [row["acc"] for row in rows],
        "answered": [row["answered"] for row in rows],
        "resp_chars": [row["resp_chars"] for row in rows],
    }

    before = val_metrics(data_sources, uids, dict(base_extra))
    # The replay uses the `acc` column the trainer already wrote, so this is a
    # check on the aggregation only. It has to hold for the invariance checks
    # below to mean anything; the scoring layer is checked in [3b].
    check(
        "replaying the dumped scores reproduces the value the trainer logged, bit for bit",
        before[CORE] == TRAINER_LOGGED_STEP0,
        f"{before[CORE]!r} vs {TRAINER_LOGGED_STEP0!r}",
    )

    # Not a plausible column: an adversarial one. If the score is genuinely
    # independent of it, absurd values must not shift a single bit.
    random.seed(0)
    absurd = dict(base_extra)
    absurd["resp_tokens"] = [random.choice([0.0, 1.0, 1024.0, 1e9]) for _ in rows]
    after = val_metrics(data_sources, uids, absurd)

    check(
        "overall is bit-identical with an adversarial resp_tokens column",
        after[CORE] == before[CORE],
        f"{after[CORE] * 100:.6f} vs {before[CORE] * 100:.6f}",
    )
    check(
        "so is every per-type score",
        all(
            after[key] == before[key]
            for key in before
            if key.startswith("val-aux/vsibench/type/") and key.endswith("/acc")
        ),
    )
    check(
        "so is answered/frac",
        after["val-core/vsibench/answered/frac"] == before["val-core/vsibench/answered/frac"],
    )

    print("\n[4] val-core gains exactly one key and loses none")
    core_before = {key for key in before if key.startswith("val-core/")}
    core_after = {key for key in after if key.startswith("val-core/")}
    check("nothing was dropped from val-core", core_before <= core_after, f"lost {core_before - core_after}")
    check(
        "the only addition is the requested mean",
        core_after - core_before == {"val-core/vsibench/resp_tokens/mean"},
        f"added {sorted(core_after - core_before)}",
    )
    # A validation pass reports one response per prompt, so the core selector in
    # _val_metrics_update has to be looking at mean@1. rollout.n moves the
    # training batch, never this.
    per_source = [key for key in after if key.endswith("/acc/mean@1") and "vsibench" in key]
    check(
        "per-data_source means are reported at @1, unaffected by rollout.n",
        len(per_source) == 10,
        f"{len(per_source)} data sources at mean@1",
    )
    return before


# ------------------------------------------------------------------ [3] roll-up


def test_rollup(before, rows, data_sources):
    print("\n[3] the roll-up is still VSI-Bench's own definition")
    by_type: dict[str, list[float]] = {}
    for source, row in zip(data_sources, rows):
        by_type.setdefault(source.split("/", 1)[1], []).append(float(row["acc"]))

    def mean(values):
        return sum(values) / len(values)

    direction = mean([mean(by_type[name]) for name in DIRECTION_TYPES])
    per_type = [mean(values) for name, values in by_type.items() if name not in DIRECTION_TYPES]
    hand_rolled = mean(per_type + [direction])
    check(
        "recomputing the unweighted 8-type mean by hand reproduces it exactly",
        abs(hand_rolled - before[CORE]) < 1e-12,
        f"{hand_rolled * 100:.6f} vs {before[CORE] * 100:.6f}",
    )
    check("8 types after merging the three direction difficulties", int(before["val-aux/vsibench/num_types"]) == 8)

    micro = mean([float(row["acc"]) for row in rows])
    check(
        "the micro average is a different number, so the weighting is real",
        abs(micro - before[CORE]) > 1e-3,
        f"micro {micro * 100:.2f} vs overall {before[CORE] * 100:.2f}",
    )

def test_upstream_agrees(frame, rows, data_sources, before):
    """Re-score the same generations through both pipelines and compare.

    Not a re-average of the dumped `acc` column: both sides start from the raw
    response text, so this covers the parsers and the MRA arithmetic as well as
    the roll-up. `vsibench_process_results` / `vsibench_aggregate_results` also
    build their rows in a per-metric shape the in-training path never touches.
    """
    print("\n[3b] the in-training scorer and lmms_eval agree on the same generations")
    import importlib.util

    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    spec = importlib.util.spec_from_file_location(
        "vsibench_task_utils", os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench", "utils.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import mvopsd_reward

    upstream_rows, ours_acc = [], []
    for i, row in enumerate(rows):
        extra = frame.iloc[i]["extra_info"]
        truth = frame.iloc[i]["reward_model"]["ground_truth"]
        options = json.loads(extra["options_json"] or "[]")
        upstream_rows.append(
            module.vsibench_process_results(
                {"question_type": extra["question_type"], "ground_truth": truth, "options": options},
                [row["output"]],
            )["vsibench_score"]
        )
        ours_acc.append(
            mvopsd_reward.compute_score(
                data_source=data_sources[i],
                solution_str=row["output"],
                ground_truth=str(truth),
                extra_info={
                    "question_type": extra["question_type"],
                    "options_json": extra["options_json"],
                    "num_options": len(options),
                },
            )["acc"]
        )

    upstream = module.vsibench_aggregate_results(upstream_rows)
    ours = compute_vsibench_metrics(data_sources, ours_acc)["val-core/vsibench/overall/acc"] * 100
    disagreements = sum(
        1
        for i, up in enumerate(upstream_rows)
        if abs(float(up.get("accuracy", up.get("MRA:.5:.95:.05"))) - float(ours_acc[i])) > 1e-9
    )
    check(
        "the two pipelines score every one of the 5,130 rows identically",
        disagreements == 0,
        f"{disagreements} rows differ",
    )
    check(
        "and therefore report the same overall",
        abs(upstream - ours) < 0.01,
        f"ours {ours:.4f} vs lmms_eval {upstream:.4f}",
    )
    print(
        f"  note: the archived column was scored before the MRA fix "
        f"({before[CORE] * 100:.2f}); rescoring the same text now gives {ours:.2f}"
    )

    # The gap this check first caught was 8 rows of MRA boundary arithmetic, so
    # pin the arithmetic itself rather than only its 5,130-row average. The
    # cases below all sit exactly on a threshold.
    import vsibench_scoring

    boundary = [(1.1, 1.0), (0.9, 1.0), (9.0, 10.0), (1.7, 2.0), (18.0, 20.0), (2.3, 2.0)]
    bad = [
        (pred, target, vsibench_scoring.mean_relative_accuracy(pred, target), reference)
        for pred, target in boundary
        for reference in [module.mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05)]
        if abs(vsibench_scoring.mean_relative_accuracy(pred, target) - reference) > 1e-12
    ]
    check(
        "MRA agrees with upstream on values sitting exactly on a threshold",
        not bad,
        "; ".join(f"{p}/{t}: ours {o} vs {r}" for p, t, o, r in bad) if bad else f"{len(boundary)} cases",
    )


# ------------------------------------------------------- [5] token length units


def test_token_lengths():
    print("\n[5] resp_tokens counts what the training curve counts")
    prompt_len, response_len = 6, 8
    # Prompts left-padded, responses right-padded: verl's layout. Row 2 is an
    # empty response, row 3 fills the budget.
    real = [3, 1, 0, response_len]
    attention_mask = torch.zeros((4, prompt_len + response_len), dtype=torch.long)
    for i, length in enumerate(real):
        attention_mask[i, prompt_len - 2 :] = 0
        attention_mask[i, prompt_len - 4 : prompt_len] = 1  # 4 real prompt tokens
        attention_mask[i, prompt_len : prompt_len + length] = 1
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.zeros((4, response_len), dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": torch.zeros((4, prompt_len + response_len), dtype=torch.long),
        }
    )

    got = RayPPOTrainer._response_token_lengths(batch)
    check("padding is excluded and every row is exact", got == [float(x) for x in real], f"{got} vs {real}")
    # The same quantity verl reduces into response_length/mean for the training
    # rollouts. If these ever diverge, the two curves stop sharing an axis.
    reference = compute_response_mask(batch).sum(-1).float().tolist()
    check("identical to verl's own compute_response_mask reduction", got == reference, f"{got} vs {reference}")

    metrics = compute_vsibench_metrics(
        ["vsibench/object_counting"] * 4,
        [1.0, 0.0, 0.0, 1.0],
        resp_tokens=real,
        max_response_tokens=response_len,
    )
    check(
        "the reported mean is the plain mean of the per-row token counts",
        abs(metrics["val-core/vsibench/resp_tokens/mean"] - sum(real) / len(real)) < 1e-12,
        f"{metrics['val-core/vsibench/resp_tokens/mean']}",
    )
    check(
        "clip_ratio matches the training-side definition (length == budget)",
        abs(metrics["val-core/vsibench/resp_tokens/clip_ratio"] - 0.25) < 1e-12,
        f"{metrics['val-core/vsibench/resp_tokens/clip_ratio']}",
    )
    check(
        "a batch with no VSI rows still reports nothing",
        compute_vsibench_metrics(["cvbench/COCO/Count"], [1.0], resp_tokens=[5.0]) == {},
    )


# ------------------------------------------------------- [6] the scored prompt


def test_prompt_unchanged(frame, rows):
    print("\n[6] the prompt the score is computed on has not moved")
    if not os.path.exists(os.path.join(MODEL_PATH, "chat_template.jinja")):
        print(f"  skip: {MODEL_PATH} not present")
        return

    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="mvopsd")
    student_kwargs = dict(config.data.apply_chat_template_kwargs or {})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # RLHFDataset splits the text on <image> placeholders; the images themselves
    # never reach the tokenizer, and their tokens are special so the archived
    # `input` (decoded with skip_special_tokens=True) dropped them too.
    def render(prompt_messages):
        messages = []
        for message in prompt_messages:
            content = message["content"].replace("<image>", "")
            messages.append({"role": message["role"], "content": content})
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **student_kwargs
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids, skip_special_tokens=True)

    sample = [0, 1, len(rows) // 2, len(rows) - 1]
    diffs = [i for i in sample if render(frame.iloc[i]["prompt"]) != rows[i]["input"]]
    check(
        "renders byte-identically to the archived scored prompt",
        not diffs,
        f"differs at rows {diffs}" if diffs else f"{len(sample)} rows checked",
    )
    check(
        "the reasoning block is still pre-closed on the student/validation side",
        render(frame.iloc[0]["prompt"]).endswith("<think>\n\n</think>\n\n"),
        repr(render(frame.iloc[0]["prompt"])[-22:]),
    )
    check(
        "the system turn is still present (the ISSUE-203 regression)",
        render(frame.iloc[0]["prompt"]).startswith("system\n"),
    )


def main() -> None:
    for path in (VAL_PARQUET, VAL_DUMP):
        if not os.path.exists(path):
            print(f"missing artifact: {path}")
            raise SystemExit(1)

    frame, rows, data_sources = load_replay()
    before = test_purity(rows, data_sources)
    test_rollup(before, rows, data_sources)
    test_upstream_agrees(frame, rows, data_sources, before)
    test_token_lengths()
    test_prompt_unchanged(frame, rows)

    print("\nFAILED: " + ", ".join(failures) if failures else "\nOK")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
