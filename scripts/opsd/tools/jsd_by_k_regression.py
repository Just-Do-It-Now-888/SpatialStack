#!/usr/bin/env python3
"""Estimate per-K distillation JSD from an existing run's console log. No GPU.

The trainer logs one scalar JSD per step (`actor/vopd_loss`, a token-mean over the
whole batch) and, separately, the batch's composition over the student view budget
(`data/step/k_views_student/<K>/frac`) and over data source. The composition varies
randomly from step to step because the dataloader shuffles, so the per-K JSD can be
recovered cross-sectionally without instrumenting the loss itself:

    jsd_t = sum_K b_K * frac_K,t

The catch is that jsd_t also decays over training (0.041 -> 0.016), and that trend
would swamp the composition effect. So the trend is removed with a centred rolling
median before fitting, and the fit is on deviations: within a short window, steps
that happened to draw more K=1 samples are compared against steps that drew fewer.

    python3 scripts/opsd/tools/jsd_by_k_regression.py logs/train/<exp>/train_*.log

Reported alongside the K fit is the same fit on data-source fractions, because K and
source are correlated in this dataset (spar_3view can only ever be K in {1,2}); if
source explains the same variance, the K coefficients cannot be read as a view effect.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

STEP_RE = re.compile(r"step:(\d+) - ")
JSD_RE = re.compile(r"actor/vopd_loss:([0-9.eE+-]+)")
K_FRAC_RE = re.compile(r"data/step/k_views_student/(\d+)/frac:([0-9.eE+-]+)")
SOURCE_FRAC_RE = re.compile(r"data/step/source/([A-Za-z0-9_]+)/frac:([0-9.eE+-]+)")
LEN_RE = re.compile(r"response_length/mean:([0-9.eE+-]+)")


def parse(path: str) -> list[dict]:
    steps: dict[int, dict] = {}
    for line in open(path, errors="replace"):
        step_match = STEP_RE.search(line)
        jsd_match = JSD_RE.search(line)
        if not step_match or not jsd_match:
            continue
        record = steps.setdefault(int(step_match.group(1)), {"k": {}, "source": {}})
        record["step"] = int(step_match.group(1))
        record["jsd"] = float(jsd_match.group(1))
        length = LEN_RE.search(line)
        if length:
            record["length"] = float(length.group(1))
        for key, value in K_FRAC_RE.findall(line):
            record["k"][int(key)] = float(value)
        for key, value in SOURCE_FRAC_RE.findall(line):
            record["source"][key] = float(value)
    return [steps[key] for key in sorted(steps) if steps[key]["k"]]


def detrend(values: np.ndarray, window: int) -> np.ndarray:
    """Subtract a centred rolling median, so only within-window variation survives."""
    half = window // 2
    trend = np.array(
        [np.median(values[max(0, i - half) : min(len(values), i + half + 1)]) for i in range(len(values))]
    )
    return values - trend


def fit(design: np.ndarray, target: np.ndarray, names: list[str], labels: np.ndarray) -> None:
    """Least squares on deviations from the rolling trend, reported per column."""
    centred = design - design.mean(axis=0)
    coef, *_ = np.linalg.lstsq(centred, target, rcond=None)
    predicted = centred @ coef
    residual = target - predicted
    r2 = 1.0 - residual.var() / target.var() if target.var() > 0 else float("nan")

    # Bootstrap over steps: the point estimates are meaningless without a sense of
    # whether a 300-step log can resolve them at all.
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        pick = rng.integers(0, len(target), len(target))
        try:
            drawn, *_ = np.linalg.lstsq(centred[pick], target[pick], rcond=None)
        except np.linalg.LinAlgError:
            continue
        draws.append(drawn)
    spread = np.percentile(np.array(draws), [2.5, 97.5], axis=0)

    print(f"  R^2 on detrended JSD: {r2:.3f}")
    for index, name in enumerate(names):
        mean_share = labels[:, index].mean()
        print(
            f"  {name:<22} slope {coef[index]:+.5f}  "
            f"95% CI [{spread[0, index]:+.5f}, {spread[1, index]:+.5f}]  "
            f"(mean batch share {mean_share:.1%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--window", type=int, default=15, help="rolling-median window for detrending")
    args = parser.parse_args()

    records = parse(args.log)
    if len(records) < 50:
        sys.exit(f"only {len(records)} usable steps in {args.log}")

    jsd = np.array([record["jsd"] for record in records])
    print(f"{len(records)} steps parsed from {args.log}")
    print(f"JSD: step {records[0]['step']} = {jsd[0]:.4f}, "
          f"first-10 mean = {jsd[:10].mean():.4f}, last-50 mean = {jsd[-50:].mean():.4f}, "
          f"min = {jsd.min():.4f}  (ln2 = {np.log(2):.4f})")
    lengths = [record.get("length") for record in records if record.get("length")]
    if lengths:
        print(f"response_length/mean: first-10 {np.mean(lengths[:10]):.0f} -> last-50 {np.mean(lengths[-50:]):.0f}")

    target = detrend(jsd, args.window)

    k_keys = sorted({key for record in records for key in record["k"]})
    k_design = np.array([[record["k"].get(key, 0.0) for key in k_keys] for record in records])
    print(f"\n=== JSD vs student view budget K (detrended, window {args.window}) ===")
    fit(k_design, target, [f"frac(K={key})" for key in k_keys], k_design)

    source_keys = sorted({key for record in records for key in record["source"]})
    if source_keys:
        source_design = np.array([[record["source"].get(key, 0.0) for key in source_keys] for record in records])
        print("\n=== same fit on data-source composition (collinearity check) ===")
        fit(source_design, target, [f"frac({key})" for key in source_keys], source_design)
    else:
        print("\nno data/step/source/*/frac series in this log; collinearity check skipped")


if __name__ == "__main__":
    main()
