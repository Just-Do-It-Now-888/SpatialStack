#!/usr/bin/env python3
"""Score a teacher generation dump and print the tables the experiment asked for.

    python3 scripts/opsd/tools/score_teacher_dump.py \
        --dump logs/eval/teacher_reliability/as_trained/generations.jsonl \
        --parquet data/mvopsd/parquet/main_train.parquet

Writes ``scored.jsonl`` (every row with its rule verdict), ``summary.json``
(roll-ups by teacher view count, by source, by question type) and
``judge_queue.jsonl`` (the rows the rule tier declined, ready for
``judge_teacher_dump.py``) next to the dump.

The by-view-count table is printed with its confound spelled out: in the
training pool N=3 is always spar_3view and N=32 is always spar_32view, so that
column is a source column wearing a view-count label.  The sweep dumps are the
ones that answer the view-count question, and they are scored by the same code
path with ``--group-by sweep_views``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

import teacher_scoring as ts  # noqa: E402


def load_prompts(parquet: str) -> dict[int, str]:
    import pandas as pd

    path = parquet if os.path.isabs(parquet) else os.path.join(REPO_ROOT, parquet)
    frame = pd.read_parquet(path, columns=["teacher_prompt"]).reset_index(drop=True)
    return {
        index: str(messages[0]["content"])
        for index, messages in enumerate(frame["teacher_prompt"])
    }


def read_dump(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_table(title: str, table: dict, key_label: str, note: str = "") -> str:
    lines = [f"\n=== {title} ==="]
    if note:
        lines.append(note)
    lines.append(
        f"  {key_label:<38}{'rows':>8}{'scorable':>10}{'acc':>8}"
        f"{'answered':>10}{'acc|ans':>9}{'trunc':>8}{'med tok':>9}"
    )
    for key, info in table.items():
        lines.append(
            f"  {key:<38}{info['rows']:>8}{info['scorable']:>10}{info['acc']:>8.2f}"
            f"{info['answered_pct']:>9.1f}%{info['acc_of_answered']:>9.2f}"
            f"{info['truncated_pct']:>7.1f}%{info['median_output_tokens']:>9}"
        )
    return "\n".join(lines)


def format_direction_table(table: dict) -> str:
    """Split the direction score into "said too little" and "said the wrong thing".

    A direction row is right only if every axis the gold names is matched, so a
    teacher that answers "it is above" to a left/above question scores zero
    exactly like one that answers "it is right". These two columns tell them
    apart: `complete` is how often it named every axis, `axis acc` is how often
    the axes it did name were correct.
    """
    rows = {key: info for key, info in table.items() if info.get("direction_rows_answered")}
    if not rows:
        return ""
    lines = [
        "\n=== direction rows: completeness vs. correctness ===",
        "  a direction row scores 1 only if every gold axis is matched, so these",
        "  columns separate an incomplete answer from a wrong one",
        f"  {'source|question_type':<38}{'answered':>10}{'complete':>10}{'axis acc':>10}{'exact':>9}",
    ]
    for key, info in rows.items():
        lines.append(
            f"  {key:<38}{info['direction_rows_answered']:>10}"
            f"{info['direction_complete_pct']:>9.1f}%{info['direction_axis_acc']:>10.2f}"
            f"{info['acc_of_answered']:>9.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out-dir", default=None, help="defaults to the dump's directory")
    parser.add_argument(
        "--group-by",
        default="n_views_teacher",
        help="primary grouping field: n_views_teacher for the as-trained run, sweep_views for a sweep",
    )
    parser.add_argument(
        "--audit-fraction",
        type=float,
        default=0.02,
        help="share of rule-accepted rows kept for the judge's reverse audit",
    )
    args = parser.parse_args()

    dump_path = args.dump if os.path.isabs(args.dump) else os.path.join(REPO_ROOT, args.dump)
    out_dir = args.out_dir or os.path.dirname(dump_path)
    os.makedirs(out_dir, exist_ok=True)

    prompts = load_prompts(args.parquet)
    rows = read_dump(dump_path)
    print(f"{len(rows)} generated rows, {len(prompts)} prompts")

    scored: list[dict] = []
    judge_queue: list[dict] = []
    # Rows the rules did read, kept in the judge's own input format so the
    # reverse audit can re-extract them later without reopening the dump
    # (LESSON-019: a judge that contradicts the parser where the parser worked
    # is not to be trusted where it did not).
    audit_pool: list[dict] = []
    audit_rng = random.Random(20260821)
    for row in rows:
        prompt = prompts.get(int(row["row_index"]), "")
        verdict = ts.score_row(row, prompt)
        record = dict(row)
        record.pop("response", None)
        record.update(
            {
                "family": verdict["family"],
                "gold_parsed": verdict["gold_parsed"],
                "gold_repr": verdict["gold_repr"],
                "answered": verdict["answered"],
                "parsed_repr": verdict["parsed_repr"],
                "score": verdict["score"],
                "rule_score": verdict["score"],
                "rule_answered": verdict["answered"],
                "judge_kind": verdict["judge_kind"],
                "axes": verdict["axes"],
                "axes_missing": verdict.get("axes_missing", []),
            }
        )
        scored.append(record)
        payload = {
            "row_index": row["row_index"],
            "sample_id": row.get("sample_id", ""),
            "source": row.get("source", ""),
            "question_type": row.get("question_type", ""),
            "judge_kind": verdict["judge_kind"],
            "question": prompt.replace("<image>", ""),
            "ground_truth": row.get("ground_truth", ""),
            "response": row.get("response", ""),
            "options": ts.options_from_prompt(prompt),
        }
        if verdict["judge_kind"]:
            judge_queue.append(payload)
        elif verdict["answered"] and audit_rng.random() < args.audit_fraction:
            audit_pool.append(dict(payload, rule_score=verdict["score"], family=verdict["family"]))

    by_views = ts.aggregate(scored, (args.group_by,))
    by_source = ts.aggregate(scored, ("source",))
    by_type = ts.aggregate(scored, ("source", "question_type"))
    by_views_source = ts.aggregate(scored, (args.group_by, "source"))
    overall = ts.aggregate(scored, ("family",))

    summary = {
        "dump": os.path.relpath(dump_path, REPO_ROOT),
        "parquet": args.parquet,
        "group_by": args.group_by,
        "rows": len(scored),
        "judge_pending": len(judge_queue),
        "judge_pending_by_kind": ts.aggregate(judge_queue, ("judge_kind",)),
        "by_group": by_views,
        "by_source": by_source,
        "by_question_type": by_type,
        "by_group_and_source": by_views_source,
        "by_family": overall,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "scored.jsonl"), "w") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "judge_queue.jsonl"), "w") as handle:
        for record in judge_queue:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "audit_pool.jsonl"), "w") as handle:
        for record in audit_pool:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    note = ""
    if args.group_by == "n_views_teacher":
        note = (
            "  NOTE: in the training pool the view count is confounded with the source\n"
            "  (N=3 is always spar_3view, N=32 always spar_32view). Read the sweep dumps\n"
            "  for the view-count effect, not this table."
        )
    print(format_table(f"by {args.group_by}", by_views, args.group_by, note))
    print(format_table("by source", by_source, "source"))
    print(format_table("by question type", by_type, "source|question_type"))
    print(format_table("by answer family", overall, "family"))
    print(format_direction_table(by_type))
    print(
        f"\nrule tier declined {len(judge_queue)} rows "
        f"({100 * len(judge_queue) / max(len(scored), 1):.1f}%); queued in {out_dir}/judge_queue.jsonl"
    )
    print(f"saved: {out_dir}/summary.json, scored.jsonl, judge_queue.jsonl")


if __name__ == "__main__":
    main()
