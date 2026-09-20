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
"""Roll MindCube tinybench's per-group validation accuracies up onto the anchor axis.

Unlike CV-Bench, MindCube's published score needs no weighting: its
`mindcube_aggregate_results` is a plain mean over all rows, and the per-family
numbers are plain means within a family. So the roll-up here is arithmetic
rather than a reconstruction -- what it buys is that the headline lands on one
key, comparable digit for digit against the offline anchors (46.86 base,
74.48 answer-only SFT, 69.81 CoT SFT), instead of being spread across the five
`data_source` buckets `process_validation_metrics` reports.

Two view budgets are reported side by side and must never be added together:

* `mindcube/...`   -- full view set, the axis the anchors live on.
* `mindcube1v/...` -- one view, the condition training actually targets.

A gain on one is not evidence about the other. `20260831_mvopsd_spar3_student_k`
is the in-repo case where an in-training gain of +2.96 pp coexisted with the same
checkpoint scoring below base on three held-out benchmarks.

Per-N series exist because the train and eval view distributions differ: training
is 74% four-view, tinybench is 41/33/26% four/three/two-view, and `rotation` is
trained mostly on four-view rows but evaluated *entirely* on three-view ones. A
pooled average cannot show that.

Everything is a fraction; multiply by 100 for the percentage the anchors are
quoted in.
"""

from collections import defaultdict

# Validation prefixes only. The training pool's data_sources are
# `mindcube_among_4view` (underscore), so training rows never reach these
# buckets even when a batch mixes them.
MINDCUBE_VIEW_BUDGETS = {
    "mindcube/": "mindcube",
    "mindcube1v/": "mindcube1v",
}

FAMILIES = ("among", "around", "rotation")


def _mean(values):
    return sum(values) / len(values) if values else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _budget_of(data_source: str):
    """Return (metric namespace, family, n_views) or None for a non-MindCube row.

    `mindcube1v/` is checked before `mindcube/` would ever match it -- they differ
    at the character before the slash, so the lookup is unambiguous, but the
    ordering matters if a third budget is ever added with a shared stem.
    """
    text = str(data_source)
    for prefix, namespace in MINDCUBE_VIEW_BUDGETS.items():
        if not text.startswith(prefix):
            continue
        parts = text[len(prefix) :].split("/")
        if len(parts) != 2:
            return None
        family, view_tag = parts
        if not view_tag.endswith("view"):
            return None
        try:
            n_views = int(view_tag[: -len("view")])
        except ValueError:
            return None
        return namespace, family, n_views
    return None


def compute_mindcube_metrics(
    data_sources,
    accuracies,
    answered=None,
    resp_chars=None,
    resp_tokens=None,
) -> dict:
    """Roll MindCube validation rows up per view budget, family and view count.

    Args:
        data_sources: per-row `mindcube/<family>/<n>view` or
            `mindcube1v/<family>/<n>view` strings, as written by
            scripts/opsd/build_mindcube_val_parquet.py. Rows from other
            benchmarks and from the training pool are ignored, so a mixed
            validation set is safe.
        accuracies: per-row 0/1 correctness aligned with `data_sources`.
        answered: optional per-row 0/1 "an option letter could be read at all".
            Reported separately because a wrong answer and an unreadable one both
            score zero, and only the second means the accuracy has stopped
            measuring spatial ability (LESSON-011).
        resp_chars / resp_tokens: optional per-row response lengths. Mean, median
            and p95 are all reported: a rising p95 over a flat median is the early
            signature of the repetitive degeneration that cost this project a run
            (LESSON-020, re-confirmed by LESSON-029). The answer-only arm starts
            near 11 tokens, so it has the most room to blow up.

    Returns:
        A metric dict, empty when the validation set holds no MindCube rows.
    """
    accuracies = list(accuracies)
    answered = list(answered) if answered is not None else None
    resp_chars = list(resp_chars) if resp_chars is not None else None
    resp_tokens = list(resp_tokens) if resp_tokens is not None else None

    # namespace -> bucket -> values
    overall = defaultdict(list)
    by_family = defaultdict(lambda: defaultdict(list))
    by_views = defaultdict(lambda: defaultdict(list))
    by_family_views = defaultdict(lambda: defaultdict(list))
    answered_rows = defaultdict(list)
    chars = defaultdict(list)
    tokens = defaultdict(list)

    for index, data_source in enumerate(data_sources):
        parsed = _budget_of(data_source)
        if parsed is None or index >= len(accuracies):
            continue
        namespace, family, n_views = parsed
        accuracy = float(accuracies[index])
        overall[namespace].append(accuracy)
        by_family[namespace][family].append(accuracy)
        by_views[namespace][n_views].append(accuracy)
        by_family_views[namespace][(family, n_views)].append(accuracy)
        if answered is not None and index < len(answered):
            answered_rows[namespace].append(float(answered[index]))
        if resp_chars is not None and index < len(resp_chars):
            chars[namespace].append(float(resp_chars[index]))
        if resp_tokens is not None and index < len(resp_tokens):
            tokens[namespace].append(float(resp_tokens[index]))

    metrics = {}
    for namespace, values in overall.items():
        if not values:
            continue
        # The series to read against the anchors. MindCube's own overall score is
        # this plain mean, so no reweighting stands between the two numbers.
        metrics[f"val-core/{namespace}/overall/acc"] = _mean(values)
        metrics[f"val-aux/{namespace}/num_samples"] = len(values)

        for family in FAMILIES:
            family_values = by_family[namespace].get(family)
            if family_values:
                metrics[f"val-aux/{namespace}/family/{family}/acc"] = _mean(family_values)

        for n_views, view_values in sorted(by_views[namespace].items()):
            metrics[f"val-aux/{namespace}/nviews/{n_views}/acc"] = _mean(view_values)

        for (family, n_views), cell in sorted(by_family_views[namespace].items()):
            metrics[f"val-aux/{namespace}/family/{family}/{n_views}view/acc"] = _mean(cell)

        if answered_rows[namespace]:
            metrics[f"val-core/{namespace}/answered/frac"] = _mean(answered_rows[namespace])
        if chars[namespace]:
            metrics[f"val-aux/{namespace}/resp_chars/median"] = _median(chars[namespace])
        if tokens[namespace]:
            metrics[f"val-core/{namespace}/resp_tokens/mean"] = _mean(tokens[namespace])
            metrics[f"val-core/{namespace}/resp_tokens/median"] = _median(tokens[namespace])
            metrics[f"val-core/{namespace}/resp_tokens/p95"] = _percentile(tokens[namespace], 0.95)

    return metrics
