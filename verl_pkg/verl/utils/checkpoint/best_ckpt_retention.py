# Copyright 2026 SpatialStack_OPSD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or governing law is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Keep the historically highest-scoring checkpoints instead of the newest ones."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any

SCORES_FILENAME = "ckpt_scores.json"
STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")


def scores_path(ckpt_root: str) -> str:
    return os.path.join(ckpt_root, SCORES_FILENAME)


def load_scores(ckpt_root: str) -> dict[str, Any]:
    path = scores_path(ckpt_root)
    if not os.path.isfile(path):
        return {"metric": None, "last_scored": None, "steps": {}}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("metric", None)
    data.setdefault("last_scored", None)
    data.setdefault("steps", {})
    return data


def save_scores(ckpt_root: str, data: dict[str, Any]) -> None:
    os.makedirs(ckpt_root, exist_ok=True)
    tmp_path = scores_path(ckpt_root) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, scores_path(ckpt_root))


def parse_step_from_dirname(name: str) -> int | None:
    match = STEP_DIR_RE.match(name)
    return int(match.group(1)) if match else None


def list_step_dirs(ckpt_root: str) -> dict[int, str]:
    if not os.path.isdir(ckpt_root):
        return {}
    found: dict[int, str] = {}
    for name in os.listdir(ckpt_root):
        step = parse_step_from_dirname(name)
        if step is None:
            continue
        path = os.path.join(ckpt_root, name)
        if os.path.isdir(path):
            found[step] = path
    return found


def rank_steps(step_scores: dict[int, float], max_to_keep: int) -> tuple[list[int], list[int]]:
    """Return (keep, drop). Higher score wins; ties keep the larger step."""
    if max_to_keep is None or max_to_keep <= 0:
        ordered = sorted(step_scores, reverse=True)
        return ordered, []
    ranked = sorted(step_scores.items(), key=lambda item: (-item[1], -item[0]))
    keep = [step for step, _ in ranked[:max_to_keep]]
    drop = [step for step, _ in ranked[max_to_keep:]]
    return keep, drop


def resolve_score(score: float | None, last_scored: float | None) -> tuple[float | None, bool]:
    """Return (score_to_store, inherited). None means this step cannot compete."""
    if score is not None:
        return float(score), False
    if last_scored is not None:
        return float(last_scored), True
    return None, False


def apply_best_retention(
    ckpt_root: str,
    step: int,
    score: float | None,
    max_to_keep: int,
    metric: str | None = None,
) -> dict[str, Any]:
    """Record this step's score and delete checkpoint dirs outside the top-K.

    Steps with no score and no previous score stay on disk and are not used to
    evict a scored checkpoint. The next scored save will drop those extras if
    they still sit outside the top-K scored set.
    """
    data = load_scores(ckpt_root)
    if metric:
        data["metric"] = metric
    stored, inherited = resolve_score(score, data.get("last_scored"))
    if stored is not None:
        data["steps"][str(step)] = stored
        if score is not None:
            data["last_scored"] = float(score)

    step_dirs = list_step_dirs(ckpt_root)
    scored: dict[int, float] = {}
    for existing_step in step_dirs:
        raw = data["steps"].get(str(existing_step))
        if raw is None:
            continue
        scored[existing_step] = float(raw)

    keep, drop = rank_steps(scored, max_to_keep)
    if max_to_keep and max_to_keep > 0 and len(keep) >= max_to_keep:
        for extra_step in list(step_dirs):
            if extra_step not in keep:
                drop.append(extra_step)
        drop = sorted(set(drop))

    dropped_paths: list[str] = []
    for drop_step in drop:
        path = step_dirs.get(drop_step)
        if path is None:
            continue
        print(f"Checkpoint manager remove previous save local path: {os.path.abspath(path)}")
        shutil.rmtree(path, ignore_errors=True)
        dropped_paths.append(path)
        data["steps"].pop(str(drop_step), None)
        step_dirs.pop(drop_step, None)

    save_scores(ckpt_root, data)
    kept_paths = [step_dirs[s] for s in keep if s in step_dirs]
    return {
        "keep": keep,
        "drop": drop,
        "kept_paths": kept_paths,
        "dropped_paths": dropped_paths,
        "score": stored,
        "inherited": inherited,
        "metric": data.get("metric"),
    }
