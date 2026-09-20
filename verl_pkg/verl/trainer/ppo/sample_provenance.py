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
"""Record which training samples each step actually consumed.

verl's rollout dump writes the decoded prompt, the response and the score, but
nothing that identifies the row: two ScanNet questions about the same scene are
indistinguishable in it, and the K/N view budget that defines the privileged
axis does not appear at all. That is enough to watch a curve move and not enough
to say what moved it.

MV-OPSD's parquet already carries the identifiers in `extra_info`
(`write_mvopsd_parquet.py`): `sample_id`, `source`, `subset`, `scene_id`, the
student's `k_views_student` against the teacher's `n_views_teacher`, and the
`privilege_bucket` those two imply. This module surfaces them in two places:

* per row in the rollout JSONL, so any step can be traced back to exact samples;
* per step as scalar metrics, so the batch composition sits next to the
  CV-Bench curve in wandb and a jump can be checked against the data mix that
  produced it rather than assumed to be the method.

Both are read-only views over the batch. Nothing here changes training.
"""

from collections import defaultdict
from typing import Any, Optional

# Row-level identity written into the rollout dump. `index` is omitted because
# it duplicates `sample_id` in the current parquet; add it here if that changes.
PROVENANCE_FIELDS = (
    "sample_id",
    "source",
    "subset",
    "scene_id",
    "k_views_student",
    "n_views_teacher",
    "privilege_bucket",
    "view_selection",
    "answer_view_sensitive",
    "question_type",
)

# Categorical fields that become per-step composition fractions. Keeping this
# separate from PROVENANCE_FIELDS is deliberate: `sample_id` and `scene_id` are
# near-unique per row and would emit thousands of near-zero wandb series.
COMPOSITION_FIELDS = (
    "source",
    "k_views_student",
    "n_views_teacher",
    "privilege_bucket",
    "view_selection",
)

METRIC_PREFIX = "data/step"


def _extra_info_rows(batch) -> Optional[list]:
    rows = batch.non_tensor_batch.get("extra_info")
    if rows is None:
        return None
    return [row if isinstance(row, dict) else {} for row in rows]


def _mean(values):
    return sum(values) / len(values) if values else None


def extract_provenance(batch) -> dict[str, list[Any]]:
    """Per-row identity columns for the rollout JSONL, aligned with the batch.

    A dataset without `extra_info` simply contributes no identity columns, so
    this stays safe on non-MV-OPSD data.

    `uid` is included because it is the only way to tell the rollout group
    apart: with `rollout.n=4` the same prompt appears on four consecutive rows,
    and per-row scores are only interpretable within a group.
    """
    columns: dict[str, list[Any]] = {}

    rows = _extra_info_rows(batch)
    if rows is not None:
        for field in PROVENANCE_FIELDS:
            values = [row.get(field) for row in rows]
            if any(value is not None for value in values):
                columns[field] = values

    for key in ("uid", "data_source"):
        series = batch.non_tensor_batch.get(key)
        if series is not None:
            columns[key] = list(series)

    return columns


def compute_sample_provenance_metrics(batch) -> dict[str, Any]:
    """Per-step batch composition, plus response length and score broken out by source.

    The length split is the load-bearing one. MV-OPSD v0's answers drifted from
    option letters to ~170-word prose, and with a single global mean there was
    no way to see whether that started in one data source and spread or happened
    everywhere at once (LESSON-011). Per-source series make that visible while
    the run is still cheap to stop.

    Scores are reported only when they exist and are not identically zero: v0's
    arm is reward-free by construction (`mvopsd_reward.py`), so a per-source
    breakdown of constant zeros would be noise. See LESSON-012.
    """
    rows = _extra_info_rows(batch)
    if rows is None:
        return {}

    num_sequences = len(rows)
    if num_sequences == 0:
        return {}

    metrics: dict[str, Any] = {f"{METRIC_PREFIX}/num_sequences": num_sequences}

    uids = batch.non_tensor_batch.get("uid")
    if uids is not None:
        metrics[f"{METRIC_PREFIX}/num_prompts"] = len(set(uids))

    sample_ids = [row.get("sample_id") for row in rows if row.get("sample_id") is not None]
    if sample_ids:
        metrics[f"{METRIC_PREFIX}/num_unique_samples"] = len(set(sample_ids))

    for field in COMPOSITION_FIELDS:
        counts: dict[Any, int] = defaultdict(int)
        for row in rows:
            value = row.get(field)
            if value is not None:
                counts[value] += 1
        for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
            metrics[f"{METRIC_PREFIX}/{field}/{value}/frac"] = count / num_sequences

    for field in ("k_views_student", "n_views_teacher"):
        values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float))]
        if values:
            metrics[f"{METRIC_PREFIX}/{field}/mean"] = _mean(values)

    sensitive = [
        float(bool(row.get("answer_view_sensitive")))
        for row in rows
        if row.get("answer_view_sensitive") is not None
    ]
    if sensitive:
        # The fraction of the batch where the extra views can change the answer,
        # i.e. where the privileged teacher has anything to teach at all.
        metrics[f"{METRIC_PREFIX}/answer_view_sensitive/frac"] = _mean(sensitive)

    metrics.update(_per_source_metrics(batch, rows))
    return metrics


def _per_source_metrics(batch, rows) -> dict[str, Any]:
    sources = [row.get("source") for row in rows]
    if not any(source is not None for source in sources):
        return {}

    response_mask = batch.batch.get("response_mask")
    lengths = response_mask.sum(-1).float().cpu().tolist() if response_mask is not None else None

    token_level_scores = batch.batch.get("token_level_scores")
    scores = token_level_scores.sum(-1).float().cpu().tolist() if token_level_scores is not None else None
    if scores is not None and not any(score != 0.0 for score in scores):
        scores = None

    lengths_by_source = defaultdict(list)
    scores_by_source = defaultdict(list)
    for index, source in enumerate(sources):
        if source is None:
            continue
        if lengths is not None and index < len(lengths):
            lengths_by_source[source].append(lengths[index])
        if scores is not None and index < len(scores):
            scores_by_source[source].append(scores[index])

    metrics: dict[str, Any] = {}
    for source, values in lengths_by_source.items():
        metrics[f"{METRIC_PREFIX}/source/{source}/response_length/mean"] = _mean(values)
    for source, values in scores_by_source.items():
        metrics[f"{METRIC_PREFIX}/source/{source}/score/mean"] = _mean(values)
    return metrics
