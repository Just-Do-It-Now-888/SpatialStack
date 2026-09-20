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
"""Roll CV-Bench's per-group validation accuracies up into its official score.

`process_validation_metrics` reports one mean per `data_source` and stops there,
which for CV-Bench yields six numbers and no headline. CV-Bench's own score is
not the mean of those six, and it is not the micro average over the 2,638 rows
either: `cvbench_aggregate_results` averages ADE20K with COCO to get 2D, takes
Omni3D as 3D, then averages 2D with 3D, so the 1,200 Omni3D rows carry half the
weight. Without this roll-up the in-training curve could not be compared against
the offline `logs/eval/.../cvbench/` numbers, which is the whole point of
running it.

Everything here is expressed as a fraction; multiply by 100 for the percentage
lmms_eval prints.
"""

from collections import defaultdict

CVBENCH_PREFIX = "cvbench/"

# src/lmms_eval/tasks/cvbench/utils.py:70-75.
SOURCES_2D = ("ADE20K", "COCO")
SOURCES_3D = ("Omni3D",)


def _mean(values):
    return sum(values) / len(values) if values else None


def compute_cvbench_metrics(data_sources, accuracies, answered=None) -> dict:
    """Rebuild CV-Bench's 2D / 3D / combined score from per-row accuracies.

    Args:
        data_sources: per-row `cvbench/<source>/<task>` strings, as written by
            scripts/opsd/build_cvbench_val_parquet.py. Rows from other
            benchmarks are ignored, so a mixed validation set is safe.
        accuracies: per-row 0/1 correctness aligned with `data_sources`.
        answered: optional per-row 0/1 flag for "the response contained an
            option letter at all". Reported as its own series because accuracy
            alone cannot distinguish a model that answers wrongly from one that
            stopped answering the question -- the distinction that cost MV-OPSD
            v0 a full training run (LESSON-011).

    Returns:
        A metric dict, empty when the validation set holds no CV-Bench rows.
    """
    by_source = defaultdict(list)
    by_task = defaultdict(list)
    every = []
    answered_rows = []
    answered = list(answered) if answered is not None else None

    for index, (data_source, accuracy) in enumerate(zip(data_sources, accuracies)):
        if not str(data_source).startswith(CVBENCH_PREFIX):
            continue
        parts = str(data_source).split("/")
        if len(parts) != 3:
            continue
        _, source, task = parts
        accuracy = float(accuracy)
        by_source[source].append(accuracy)
        by_task[task].append(accuracy)
        every.append(accuracy)
        if answered is not None and index < len(answered):
            answered_rows.append(float(answered[index]))

    if not every:
        return {}

    metrics = {
        # Micro average over every row. Not CV-Bench's published score, but the
        # least surprising thing to watch for a collapse, since it moves with
        # any subtask.
        "val-aux/cvbench/micro/acc": _mean(every),
        "val-aux/cvbench/num_samples": len(every),
    }
    if answered_rows:
        # Watch this alongside the combined score. A drop here means the model
        # stopped selecting options, and the accuracy below stops being a
        # measurement of anything.
        metrics["val-core/cvbench/answered/frac"] = _mean(answered_rows)
    for source, values in by_source.items():
        metrics[f"val-aux/cvbench/source/{source}/acc"] = _mean(values)
    for task, values in by_task.items():
        metrics[f"val-aux/cvbench/task/{task}/acc"] = _mean(values)

    accuracy_2d = _mean([_mean(by_source[s]) for s in SOURCES_2D if by_source.get(s)])
    accuracy_3d = _mean([_mean(by_source[s]) for s in SOURCES_3D if by_source.get(s)])

    if accuracy_2d is not None:
        metrics["val-aux/cvbench/2d/acc"] = accuracy_2d
    if accuracy_3d is not None:
        metrics["val-aux/cvbench/3d/acc"] = accuracy_3d
    if accuracy_2d is not None and accuracy_3d is not None:
        # The series to early-stop on: directly comparable to the combined score
        # in experiments/reports/*.md.
        metrics["val-core/cvbench/combined/acc"] = (accuracy_2d + accuracy_3d) / 2

    return metrics
