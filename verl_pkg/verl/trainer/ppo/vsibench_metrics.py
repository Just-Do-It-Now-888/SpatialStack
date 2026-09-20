# Copyright 2026 SpatialStack_OPSD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Roll VSI-Bench's per-type validation scores up into its official score.

`process_validation_metrics` reports one mean per `data_source`, which for
VSI-Bench yields ten numbers and no headline. VSI-Bench's own score is not the
micro average over the 5,130 rows: `vsibench_aggregate_results` merges the three
`object_rel_direction` difficulties into one value, then takes an **unweighted**
mean over the resulting eight question types. Row counts differ by more than 5x
between types, so the micro average is a different number and would not be
comparable to the offline `logs/vsi_train_eval/` results -- which is the whole
point of running this in the loop.

Alongside the score this reports the two series that tell a wrong answer apart
from a missing one: the fraction of rows an answer could be read from, and the
response length. The checkpoint that made this guardrail necessary kept its
CV-Bench score while its median VSI response went from 4 tokens to ~200 and a
fifth of its answers never arrived (LESSON-020).

Validation also reports ``resp_tokens/clip_ratio``: the fraction of rows whose
generated length equals the response budget, using the same definition as the
training curve's ``response_length/clip_ratio``. On a boxed prompt, a rising
clip ratio often means truncated answers and lost ``\\boxed{}`` closures.

Everything here is a fraction; multiply by 100 for the percentage lmms_eval
prints.
"""

from collections import defaultdict

VSIBENCH_PREFIX = "vsibench/"

# src/lmms_eval/tasks/vsibench/utils.py:260-271 averages the three difficulties
# into a single object_rel_direction score before the per-type mean is taken.
DIRECTION_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
)
DIRECTION_MERGED = "object_rel_direction"


def _mean(values):
    return sum(values) / len(values) if values else None


def compute_vsibench_metrics(
    data_sources,
    accuracies,
    answered=None,
    resp_chars=None,
    resp_tokens=None,
    max_response_tokens=None,
) -> dict:
    """Rebuild VSI-Bench's official score from per-row accuracies.

    Args:
        data_sources: per-row `vsibench/<question_type>` strings, as written by
            scripts/opsd/build_vsibench_val_parquet.py. Rows from other
            benchmarks are ignored, so a mixed validation set is safe.
        accuracies: per-row score aligned with `data_sources` -- 0/1 for the
            multiple-choice types, MRA in [0, 1] for the numerical ones.
        answered: optional per-row 0/1 flag for "an answer could be read out of
            the response at all".
        resp_chars: optional per-row response length in decoded characters.
            Reported as a mean and a median because the degeneration this
            guardrail exists to catch shows up as a long tail well before it
            shows up in the score.
        resp_tokens: optional per-row response length in generated tokens. The
            same unit as the training curve's `response_length/mean`, so the two
            can be read on one axis; characters cannot be compared to it.
        max_response_tokens: optional validation response budget. When set
            together with `resp_tokens`, reports `val-core/vsibench/resp_tokens/
            clip_ratio` -- the fraction of rows whose length equals this cap,
            matching `response_length/clip_ratio` on the training rollouts.

    Returns:
        A metric dict, empty when the validation set holds no VSI-Bench rows.
    """
    by_type = defaultdict(list)
    every = []
    answered_rows = []
    length_rows = []
    token_rows = []
    answered = list(answered) if answered is not None else None
    resp_chars = list(resp_chars) if resp_chars is not None else None
    resp_tokens = list(resp_tokens) if resp_tokens is not None else None

    for index, (data_source, accuracy) in enumerate(zip(data_sources, accuracies)):
        source = str(data_source)
        if not source.startswith(VSIBENCH_PREFIX):
            continue
        question_type = source[len(VSIBENCH_PREFIX) :]
        if not question_type:
            continue
        accuracy = float(accuracy)
        by_type[question_type].append(accuracy)
        every.append(accuracy)
        if answered is not None and index < len(answered):
            answered_rows.append(float(answered[index]))
        if resp_chars is not None and index < len(resp_chars):
            length_rows.append(float(resp_chars[index]))
        if resp_tokens is not None and index < len(resp_tokens):
            token_rows.append(float(resp_tokens[index]))

    if not every:
        return {}

    metrics = {
        # Micro average over every row. Not VSI-Bench's published score, but it
        # moves with any subtask and so is the least surprising collapse alarm.
        "val-aux/vsibench/micro/acc": _mean(every),
        "val-aux/vsibench/num_samples": len(every),
    }
    for question_type, values in by_type.items():
        metrics[f"val-aux/vsibench/type/{question_type}/acc"] = _mean(values)
        metrics[f"val-aux/vsibench/type/{question_type}/num_samples"] = len(values)

    if answered_rows:
        # Watch this next to the score. A drop here means the model stopped
        # producing readable answers, and the score below stops being a
        # measurement of spatial ability.
        metrics["val-core/vsibench/answered/frac"] = _mean(answered_rows)
    if length_rows:
        ordered = sorted(length_rows)
        metrics["val-core/vsibench/resp_chars/median"] = ordered[len(ordered) // 2]
        metrics["val-aux/vsibench/resp_chars/mean"] = _mean(length_rows)
        metrics["val-aux/vsibench/resp_chars/p95"] = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        metrics["val-aux/vsibench/resp_chars/max"] = ordered[-1]
    if token_rows:
        ordered = sorted(token_rows)
        # The mean is the headline here, not the median: it is the one number
        # that moves as soon as any part of the distribution grows a tail, and
        # it is directly comparable to the training rollouts' response length.
        # The median is kept beside it because a mean alone cannot say whether
        # the whole distribution shifted or a minority ran away.
        metrics["val-core/vsibench/resp_tokens/mean"] = _mean(token_rows)
        metrics["val-aux/vsibench/resp_tokens/median"] = ordered[len(ordered) // 2]
        metrics["val-aux/vsibench/resp_tokens/p95"] = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
        metrics["val-aux/vsibench/resp_tokens/max"] = ordered[-1]
        if max_response_tokens is not None:
            cap = float(max_response_tokens)
            # metric_utils.compute_data_metrics uses equality against the budget,
            # not >=: a row one token short was not truncated.
            clipped = sum(1 for value in token_rows if value == cap)
            metrics["val-core/vsibench/resp_tokens/clip_ratio"] = clipped / len(token_rows)

    # Merge the three direction difficulties before averaging, or direction
    # would carry three of the ten votes instead of one.
    per_type = {}
    direction = [_mean(by_type[name]) for name in DIRECTION_TYPES if by_type.get(name)]
    for question_type, values in by_type.items():
        if question_type in DIRECTION_TYPES:
            continue
        per_type[question_type] = _mean(values)
    if direction:
        per_type[DIRECTION_MERGED] = _mean(direction)
        metrics[f"val-aux/vsibench/type/{DIRECTION_MERGED}/acc"] = per_type[DIRECTION_MERGED]

    if per_type:
        # The series to early-stop on: directly comparable to `overall` in
        # logs/vsi_train_eval/<exp>/<step>/summary.json.
        metrics["val-core/vsibench/overall/acc"] = _mean(list(per_type.values()))
        metrics["val-aux/vsibench/num_types"] = len(per_type)

    return metrics
