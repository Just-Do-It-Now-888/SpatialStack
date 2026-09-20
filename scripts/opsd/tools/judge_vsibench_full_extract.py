#!/usr/bin/env python3
"""Judge-extract scoring for every row in a VSI-Bench samples.jsonl dump.

The judge only extracts a structured answer (option letter or number, or NONE).
Scores are computed with the same lmms_eval metrics as the rule tier (exact match
for multiple choice, MRA for numerical types).

    python scripts/opsd/tools/judge_vsibench_full_extract.py \\
        logs/vsi_train_eval/.../samples.jsonl \\
        --judge-api-base http://127.0.0.1:8001/v1/ \\
        --output-dir logs/vsi_train_eval/.../global_step_100/judge_extract
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, TOOLS_DIR)

import vsibench_eval_core as core  # noqa: E402
from judge_vsibench_tri_insurance import (  # noqa: E402
    TASK_UTILS,
    build_mca_prompt,
    build_na_prompt,
    load_module,
    parse_mca_reply,
    parse_na_reply,
)
from judge_vsibench_tri_insurance import VISIONOPD_JUDGE, SCORING  # noqa: E402


def load_samples(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(row: dict, parsed_mca: str, parsed_na: float | None, task) -> tuple[float, bool, str]:
    qtype = row["question_type"]
    if qtype in task.MCA_QUESTION_TYPES:
        letter = parsed_mca
        answered = bool(letter)
        score = 1.0 if letter and letter == str(row["ground_truth"]).strip().upper() else 0.0
        return score, answered, letter
    if parsed_na is None:
        return 0.0, False, ""
    target = task.to_float(row["ground_truth"])
    score = task.mean_relative_accuracy(parsed_na, target, start=0.5, end=0.95, interval=0.05)
    return score, True, str(parsed_na)


def rule_primary_tier(row: dict, scoring) -> tuple[float, bool, str]:
    """Mirror in-training boxed-primary: box if present, else full-text rule."""
    boxed = row.get("boxed_content")
    if boxed is None and row.get("response"):
        boxed = scoring.extract_boxed(row.get("response", ""))
    if boxed is not None:
        return (
            float(row.get("boxed_score", 0.0)),
            bool(row.get("boxed_answered", 0)),
            str(boxed),
        )
    return (
        float(row.get("rule_score", row.get("final_score", 0.0))),
        bool(row.get("rule_answered", 0)),
        str(row.get("rule_parsed", "")),
    )


def trust_kind(
    row: dict,
    *,
    trust_boxed: bool,
    trust_terse: bool,
    trust_mca_tail: bool,
    scoring,
    mca_types: frozenset[str] | set[str] | None = None,
) -> str:
    response = row.get("response", "")
    boxed = row.get("boxed_content")
    if boxed is None and response:
        boxed = scoring.extract_boxed(response)
    if trust_boxed and boxed is not None:
        return "boxed"
    if trust_terse and scoring.is_terse_answer(response):
        return "terse"
    qtype = row.get("question_type", "")
    if trust_mca_tail and mca_types and qtype in mca_types:
        if scoring.is_trustable_mca_tail(response, row.get("options") or []):
            return "mca_tail"
    return ""


def trusted_judge_fields(row: dict, trust_label: str, scoring, task) -> dict:
    score, answered, parsed = rule_primary_tier(row, scoring)
    return {
        "judge_extract_reply": f"[rule_trusted:{trust_label}]",
        "judge_parsed": parsed,
        "judge_answered": int(answered),
        "judge_score": score,
        "judge_failure_reason": failure_reason(row, answered, score, task),
        "judge_trust": trust_label,
    }


def failure_reason(row: dict, answered: bool, score: float, task) -> str:
    truncated = bool(row.get("truncated"))
    qtype = row["question_type"]
    if score >= 1.0 - 1e-9:
        return "correct"
    if truncated:
        return "truncation_error"
    if not answered:
        return "parse_error"
    if qtype in task.MCA_QUESTION_TYPES:
        return "factual_error"
    return "partial_error" if score > 0.0 else "factual_error"


def build_summary(rows: list[dict], task, cfg: dict, elapsed: float, stats: dict) -> dict:
    scored_rows = []
    for row in rows:
        item = {
            "question_type": row["question_type"],
            "answered": bool(row["judge_answered"]),
            "parsed_answer": row["judge_parsed"],
        }
        if row["question_type"] in task.MCA_QUESTION_TYPES:
            item["accuracy"] = row["judge_score"]
        else:
            item["MRA:.5:.95:.05"] = row["judge_score"]
        scored_rows.append(item)

    by_type = {}
    for qtype in sorted({r["question_type"] for r in rows}):
        subset = [r for r in rows if r["question_type"] == qtype]
        by_type[qtype] = {
            "n": len(subset),
            "score": 100 * sum(r["judge_score"] for r in subset) / max(len(subset), 1),
            "answered_pct": 100 * sum(1 for r in subset if r["judge_answered"]) / max(len(subset), 1),
            "truncated_pct": 100 * sum(1 for r in subset if r.get("truncated")) / max(len(subset), 1),
            "failure_reasons": dict(
                __import__("collections").Counter(r["judge_failure_reason"] for r in subset)
            ),
        }

    lengths = sorted(int(r.get("output_tokens", 0)) for r in rows)

    def pct(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

    return {
        "scoring": stats.get("scoring", "judge_extract_strict_v1"),
        "model": cfg.get("model", "unknown"),
        "protocol": cfg.get("protocol", "spatialstack"),
        "frames": cfg.get("frames", 32),
        "max_tokens": cfg.get("max_tokens", 4096),
        "questions": len(rows),
        "judge_seconds": elapsed,
        "judge_model": stats.get("judge_model"),
        "judge_api_errors": stats.get("api_errors", 0),
        "judge_unreadable": stats.get("unreadable", 0),
        "trust": stats.get("trust"),
        "full_response": True,
        "overall": {
            "score": task.vsibench_aggregate_results([dict(r) for r in scored_rows]),
            "answered_pct": task.vsibench_aggregate_answered(scored_rows),
        },
        "by_question_type": by_type,
        "failure_reasons": dict(__import__("collections").Counter(r["judge_failure_reason"] for r in rows)),
        "output_tokens": {
            "median": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": pct(1.0),
        },
        "truncated": sum(1 for r in rows if r.get("truncated")),
        "truncated_pct": 100 * sum(1 for r in rows if r.get("truncated")) / max(len(rows), 1),
        "delta_vs_rule": stats.get("delta_vs_rule"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="samples.jsonl from offline eval")
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--parallel-workers", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: grade only the first N rows")
    parser.add_argument(
        "--trust-boxed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rows with a closed \\boxed{} keep the rule-primary score and skip the judge",
    )
    parser.add_argument(
        "--trust-terse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whole-string or last-line number/letter rows keep the rule score and skip the judge",
    )
    parser.add_argument(
        "--trust-mca-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MCA rows whose last line is C / C. option / unique option body skip the judge",
    )
    args = parser.parse_args()

    samples_path = os.path.abspath(args.samples)
    output_dir = os.path.abspath(args.output_dir)
    summary_path = os.path.join(os.path.dirname(samples_path), "summary.json")
    previous = {}
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            previous = json.load(handle)

    protocol = args.protocol or previous.get("protocol", "spatialstack")
    os.environ["VSIBENCH_PROTOCOL"] = protocol

    task = load_module(TASK_UTILS, "vsibench_task_utils_full_judge")
    upstream = load_module(VISIONOPD_JUDGE, "visionopd_judge_full_judge")
    scoring = load_module(SCORING, "vsibench_scoring_full_judge")

    rows = load_samples(samples_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no samples in {samples_path}")

    pending_rows: list[dict] = []
    pending_kinds: list[str] = []
    pending_prompts: list[str] = []
    boxed_trusted = terse_trusted = mca_tail_trusted = 0
    for row in rows:
        label = trust_kind(
            row,
            trust_boxed=args.trust_boxed,
            trust_terse=args.trust_terse,
            trust_mca_tail=args.trust_mca_tail,
            scoring=scoring,
            mca_types=task.MCA_QUESTION_TYPES,
        )
        if label == "boxed":
            boxed_trusted += 1
        elif label == "terse":
            terse_trusted += 1
        elif label == "mca_tail":
            mca_tail_trusted += 1
        if label:
            continue
        response = row.get("response", "")
        if row["question_type"] in task.MCA_QUESTION_TYPES:
            pending_prompts.append(build_mca_prompt(row["question"], row.get("options") or [], response))
            pending_kinds.append("mca")
        else:
            pending_prompts.append(build_na_prompt(row["question"], response))
            pending_kinds.append("na")
        pending_rows.append(row)

    print(
        f"judge-extract: {len(rows)} rows | trusted boxed={boxed_trusted} terse={terse_trusted} "
        f"mca_tail={mca_tail_trusted} "
        f"| pending={len(pending_rows)} ({sum(k=='mca' for k in pending_kinds)} mca, "
        f"{sum(k=='na' for k in pending_kinds)} na)",
        flush=True,
    )
    started = time.perf_counter()
    from argparse import Namespace

    replies: list[tuple[str, str]] = []
    if pending_prompts:
        replies = upstream.judge_via_api(
            pending_prompts,
            Namespace(
                judge_api_key=args.judge_api_key,
                judge_api_base=args.judge_api_base,
                judge_model=args.judge_model,
                judge_max_tokens=args.judge_max_tokens,
                reasoning_effort=args.reasoning_effort,
                parallel_workers=args.parallel_workers,
            ),
        )
    elapsed = time.perf_counter() - started

    out_rows = []
    api_errors = unreadable = 0
    rule_sum = judge_sum = 0.0
    pending_iter = iter(zip(pending_rows, pending_kinds, replies))
    for row in rows:
        label = trust_kind(
            row,
            trust_boxed=args.trust_boxed,
            trust_terse=args.trust_terse,
            trust_mca_tail=args.trust_mca_tail,
            scoring=scoring,
            mca_types=task.MCA_QUESTION_TYPES,
        )
        if label:
            fields = trusted_judge_fields(row, label, scoring, task)
            judge_reply = fields["judge_extract_reply"]
        else:
            pending_row, kind, (content, reasoning) = next(pending_iter)
            assert pending_row is row
            reply = content or reasoning
            if reasoning == "[JUDGE_API_ERROR]":
                api_errors += 1
                judge_parsed = ""
                judge_answered = False
                judge_score = float(row.get("rule_score", row.get("final_score", 0.0)))
                judge_reply = "[JUDGE_API_ERROR]"
                fields = {
                    "judge_extract_reply": judge_reply,
                    "judge_parsed": judge_parsed,
                    "judge_answered": int(judge_answered),
                    "judge_score": judge_score,
                    "judge_failure_reason": failure_reason(row, judge_answered, judge_score, task),
                    "judge_trust": "",
                }
            elif kind == "mca":
                letter = parse_mca_reply(reply, row.get("options"), scoring)
                judge_score, judge_answered, judge_parsed = score_row(row, letter, None, task)
                if not judge_answered and reply and reply.strip().upper() != "NONE":
                    unreadable += 1
                judge_reply = (reply or "")[:500]
                fields = {
                    "judge_extract_reply": judge_reply,
                    "judge_parsed": judge_parsed,
                    "judge_answered": int(judge_answered),
                    "judge_score": judge_score,
                    "judge_failure_reason": failure_reason(row, judge_answered, judge_score, task),
                    "judge_trust": "",
                }
            else:
                number = parse_na_reply(reply, scoring)
                judge_score, judge_answered, judge_parsed = score_row(row, "", number, task)
                if not judge_answered and reply and reply.strip().upper() != "NONE":
                    unreadable += 1
                judge_reply = (reply or "")[:500]
                fields = {
                    "judge_extract_reply": judge_reply,
                    "judge_parsed": judge_parsed,
                    "judge_answered": int(judge_answered),
                    "judge_score": judge_score,
                    "judge_failure_reason": failure_reason(row, judge_answered, judge_score, task),
                    "judge_trust": "",
                }

        judge_score = float(fields["judge_score"])
        judge_answered = bool(fields["judge_answered"])
        judge_parsed = fields["judge_parsed"]
        rule_score = float(row.get("rule_score", row.get("final_score", 0.0)))
        rule_sum += rule_score
        judge_sum += judge_score

        out = dict(row)
        out.update(
            {
                **fields,
                "rule_score_prev": rule_score,
                "final_score_prev": float(row.get("final_score", rule_score)),
            }
        )
        out_rows.append(out)

    os.makedirs(output_dir, exist_ok=True)
    samples_out = os.path.join(output_dir, "samples.jsonl")
    with open(samples_out, "w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    scoring_label = "judge_extract_strict_v2_trust"
    if not args.trust_boxed and not args.trust_terse and not args.trust_mca_tail:
        scoring_label = "judge_extract_strict_v1"
    stats = {
        "scoring": scoring_label,
        "judge_model": args.judge_model,
        "api_errors": api_errors,
        "unreadable": unreadable,
        "trust": {
            "boxed": boxed_trusted,
            "terse": terse_trusted,
            "mca_tail": mca_tail_trusted,
            "rule_trusted": boxed_trusted + terse_trusted + mca_tail_trusted,
            "pending": len(pending_rows),
        },
        "delta_vs_rule": {
            "rule_only_overall": 100 * rule_sum / len(rows),
            "judge_extract_overall": 100 * judge_sum / len(rows),
            "delta_pp": 100 * (judge_sum - rule_sum) / len(rows),
        },
    }
    summary = build_summary(out_rows, task, previous, elapsed, stats)
    summary["source_samples"] = samples_path
    summary["previous_rule_assisted"] = previous.get("judge_assisted", previous.get("rule_only"))
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    overall = summary["overall"]["score"]
    answered = summary["overall"]["answered_pct"]
    print(
        f"\n=== judge-extract ({len(rows)} rows, {elapsed/60:.1f} min) ===\n"
        f"overall {overall:.2f} | answered {answered:.2f}%\n"
        f"vs rule-only {stats['delta_vs_rule']['rule_only_overall']:.2f} "
        f"({stats['delta_vs_rule']['delta_pp']:+.2f} pp)\n"
        f"api_errors {api_errors} unreadable {unreadable}\n"
        f"saved: {output_dir}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
