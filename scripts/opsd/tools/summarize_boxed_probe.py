#!/usr/bin/env python3
r"""Compliance, length and score for the three prompt arms of the boxed probe.

    python3 scripts/opsd/tools/summarize_boxed_probe.py

Answers three questions, in this order, because the later ones are meaningless
if the earlier ones fail:

1. **Is the comparison paired?** The ``plain`` arms carry no suffix, so their
   prompts are byte-identical to the archived runs. If ``pool_plain`` does not
   reproduce the archived ``as_trained`` responses on the rows they share, the
   environment moved and no arm-to-arm difference can be attributed to the
   prompt (LESSON-010).
2. **Does the model comply?** The fraction of responses that actually contain a
   closed ``\boxed{...}`` (or an ``Answer:`` line), reported per source, since
   a quarter of the training pool is free-form description where a boxed answer
   is not a meaningful request.
3. **What does it cost?** Response length and score, rule-tier only. A format
   that is obeyed but shortens every answer into a guess is not a win, and the
   score has to be read next to the compliance number rather than instead of it
   (LESSON-011, LESSON-020).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd", "tools"))

import teacher_scoring as ts  # noqa: E402
import vsibench_scoring as scoring  # noqa: E402

PROBE_ROOT = os.environ.get(
    "PROBE_ROOT",
    os.path.join(REPO_ROOT, "logs", "eval", "boxed_probe"),
)
MAX_TOKENS_LABEL = int(os.environ.get("MAX_TOKENS", "1024"))
ARCHIVE = os.path.join(REPO_ROOT, "logs", "eval", "teacher_reliability", "as_trained", "generations.jsonl")
POOL_PARQUET = os.path.join(REPO_ROOT, "data", "mvopsd", "parquet", "main_train.parquet")
ARMS = ("plain", "boxed", "answer")

# math_dapo.is_correct_minerva's own pattern, so "complied with the Answer:
# format" means exactly "the scorer verl actually runs would find it".
ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def _row_stats(rows: list[dict], prompts: dict[int, str]) -> dict:
    lengths = [int(row.get("output_tokens", 0)) for row in rows]
    scores, answered = [], 0
    for row in rows:
        if prompts:
            verdict = ts.score_row(dict(row), prompts.get(int(row["row_index"]), ""))
            scores.append(verdict["score"])
            answered += verdict["answered"]
    return {
        "n": len(rows),
        "score": 100 * sum(scores) / len(scores) if scores else float("nan"),
        "answered_pct": 100 * answered / len(rows) if prompts else float("nan"),
        "median_tokens": percentile(lengths, 0.5),
        "p95_tokens": percentile(lengths, 0.95),
        "cap_pct": 100 * sum(1 for row in rows if row.get("truncated")) / max(len(rows), 1),
        "boxed_pct": 100 * sum(1 for row in rows if scoring.extract_boxed(row.get("response", "")) is not None)
        / max(len(rows), 1),
    }


def parity_gate(prompts: dict[int, str]) -> None:
    print("=" * 100)
    print("[1] gate: does the plain arm reproduce the archived run?")
    print("=" * 100)
    archived = read_jsonl(ARCHIVE)
    replay = read_jsonl(os.path.join(PROBE_ROOT, "pool_plain", "generations.jsonl"))
    if not archived or not replay:
        print("  skip (need both logs/eval/teacher_reliability/as_trained and pool_plain)")
        return
    by_index = {int(row["row_index"]): row for row in archived}
    shared = [row for row in replay if int(row["row_index"]) in by_index]
    identical = sum(
        1 for row in shared if row.get("response", "") == by_index[int(row["row_index"])].get("response", "")
    )
    print(f"  rows shared with the archive : {len(shared):,} of {len(replay):,} replayed")
    print(f"  byte-identical responses     : {identical:,} ({100 * identical / max(len(shared), 1):.2f}%)")
    print(
        "\n  Byte equality is the wrong gate and this number is reported only to say so.\n"
        "  Greedy decoding fixes the argmax, not the logits: vLLM's batched kernels are\n"
        "  not batch-invariant, this job packs 3,000 rows where the archive packed\n"
        "  124,306, and one differing logit early in a response changes every token\n"
        "  after it. The divergences look like that -- same opening clause, then a\n"
        "  paraphrase -- rather than like a different prompt.\n"
        "  What has to hold instead is that the two agree in distribution on the same\n"
        "  rows. If they do, an arm-to-arm difference larger than this drift is a\n"
        "  prompt effect; if they do not, nothing here is attributable."
    )
    archive_subset = [by_index[int(row["row_index"])] for row in shared]
    left, right = _row_stats(archive_subset, prompts), _row_stats(shared, prompts)
    print(f"\n  {'quantity':<22}{'archive':>12}{'replay':>12}{'delta':>10}")
    for key, label, fmt in (
        ("score", "rule score", "{:.2f}"),
        ("answered_pct", "answered %", "{:.2f}"),
        ("median_tokens", "median tokens", "{:.0f}"),
        ("p95_tokens", "p95 tokens", "{:.0f}"),
        ("cap_pct", f"hit {MAX_TOKENS_LABEL}-token cap %", "{:.2f}"),
        ("boxed_pct", "spontaneous boxed %", "{:.2f}"),
    ):
        a, b = left[key], right[key]
        print(f"  {label:<22}{fmt.format(a):>12}{fmt.format(b):>12}{b - a:>+10.2f}")


def pool_arms(prompts: dict[int, str]) -> None:
    print()
    print("=" * 100)
    print("[2] training pool: compliance, length, rule-tier score")
    print("=" * 100)
    per_arm: dict[str, dict] = {}
    per_source: dict[str, dict[str, float]] = defaultdict(dict)
    for arm in ARMS:
        rows = read_jsonl(os.path.join(PROBE_ROOT, f"pool_{arm}", "generations.jsonl"))
        if not rows:
            continue
        stats = _row_stats(rows, prompts)
        stats["answer_pct"] = 100 * sum(1 for row in rows if ANSWER_RE.search(row.get("response", ""))) / len(rows)
        per_arm[arm] = stats
        by_source: dict[str, list] = defaultdict(list)
        for row in rows:
            has_box = scoring.extract_boxed(row.get("response", "")) is not None
            by_source[row.get("source") or row.get("data_source") or "?"].append(has_box)
        for source, hits in by_source.items():
            per_source[source][arm] = 100 * sum(hits) / len(hits)

    if not per_arm:
        print("  skip (no pool arms yet)")
        return
    print(
        f"  {'arm':<8}{'n':>7}{'boxed%':>9}{'Answer:%':>10}{'med tok':>9}{'p95 tok':>9}"
        f"{'cap%':>8}{'score':>8}{'answered':>10}"
    )
    for arm, info in per_arm.items():
        print(
            f"  {arm:<8}{info['n']:>7,}{info['boxed_pct']:>8.2f}%{info['answer_pct']:>9.2f}%"
            f"{info['median_tokens']:>9,}{info['p95_tokens']:>9,}{info['cap_pct']:>7.2f}%"
            f"{info['score']:>8.2f}{info['answered_pct']:>9.2f}%"
        )

    print(f"\n  boxed compliance by source ({'/'.join(ARMS)}):")
    for source in sorted(per_source):
        cells = "  ".join(f"{per_source[source].get(arm, float('nan')):>6.2f}%" for arm in ARMS)
        print(f"    {source:<24}{cells}")


def vsi_arms() -> None:
    print()
    print("=" * 100)
    print("[3] VSI-Bench: compliance, length, score")
    print("=" * 100)
    rows_out = []
    for arm in ARMS:
        path = os.path.join(PROBE_ROOT, f"vsi_{arm}", "summary.json")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            summary = json.load(handle)
        boxed = summary.get("boxed", {})
        rows_out.append(
            (
                arm,
                summary["questions"],
                summary["rule_only"]["overall"],
                summary["rule_only"]["answered_pct"],
                boxed.get("present_pct", float("nan")),
                boxed.get("overall", float("nan")),
                summary["output_tokens"]["median"],
                summary["truncated_pct"],
            )
        )
    if not rows_out:
        print("  skip (no VSI arms yet)")
        return
    print(
        f"  {'arm':<8}{'n':>7}{'rule score':>12}{'answered':>10}"
        f"{'boxed%':>9}{'box score':>11}{'med tok':>9}{'trunc':>8}"
    )
    for arm, n, score, answered, boxed_pct, boxed_score, med, trunc in rows_out:
        print(
            f"  {arm:<8}{n:>7,}{score:>12.2f}{answered:>9.2f}%{boxed_pct:>8.2f}%"
            f"{boxed_score:>11.2f}{med:>9}{trunc:>7.2f}%"
        )

    # Per-question-type compliance, because the four numerical types are where
    # an unreadable answer currently hides as a wrong one (LESSON-018).
    for arm in ARMS:
        path = os.path.join(PROBE_ROOT, f"vsi_{arm}", "summary.json")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            summary = json.load(handle)
        if not summary.get("prompt_suffix"):
            continue
        print(f"\n  {arm}: boxed compliance by question type")
        for qtype, info in summary["by_question_type"].items():
            print(
                f"    {qtype:<30}{info.get('boxed_present_pct', 0.0):>7.2f}%"
                f"   box score {info.get('boxed_score', 0.0):>6.2f}   rule {info['score']:>6.2f}"
            )


def failure_shapes() -> None:
    """What non-compliant responses look like, on the arm that asked for a box."""
    path = os.path.join(PROBE_ROOT, "vsi_boxed", "samples.jsonl")
    rows = read_jsonl(path)
    if not rows:
        return
    print()
    print("=" * 100)
    print("[4] shape of non-compliance on vsi_boxed")
    print("=" * 100)
    missing = [row for row in rows if not row.get("boxed_present")]
    print(f"  responses with no closed box: {len(missing):,} of {len(rows):,}")
    if not missing:
        return
    tags = Counter(row.get("failure_reason", "") for row in missing)
    print(f"  their failure tags: {dict(tags)}")
    truncated = sum(1 for row in missing if row.get("truncated"))
    print(f"  of which truncated: {truncated:,} ({100 * truncated / len(missing):.1f}%)")
    unclosed = sum(1 for row in missing if "\\boxed{" in (row.get("response") or ""))
    print(f"  wrote '\\boxed{{' but never closed it: {unclosed:,}")
    print("\n  two examples (tail of the response):")
    for row in missing[:2]:
        print(f"    [{row.get('question_type')}] {(row.get('response') or '')[-160:]!r}")


def main() -> None:
    try:
        from score_teacher_dump import load_prompts

        prompts = load_prompts(POOL_PARQUET)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot load prompts ({exc}); pool scores will be skipped")
        prompts = {}

    parity_gate(prompts)
    pool_arms(prompts)
    vsi_arms()
    failure_shapes()


if __name__ == "__main__":
    main()
