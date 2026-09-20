"""Re-score a VSI-Bench samples.jsonl dump without re-running generation.

Reads each stored response, re-parses with the current ``vsibench_scoring``
rules, optionally re-applies stored judge verdicts (no API calls), and writes an
updated ``samples.jsonl`` + ``summary.json``.

Accepts the vLLM/eval-core dump (top-level ``question_type`` + ``response``)
and lmms-eval ``*_samples_vsibench.jsonl`` (``doc`` + ``resps``).

    python scripts/opsd/tools/rescore_vsibench_samples.py \\
        logs/vsi_train_eval/.../global_step_100/samples.jsonl \\
        --output-dir logs/vsi_train_eval/.../global_step_100/rescore
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import vsibench_eval_core as core  # noqa: E402

_MCA_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}


def _is_lmms_eval_row(row: dict) -> bool:
    if "question_type" in row and "response" in row:
        return False
    return isinstance(row.get("doc"), dict) or "vsibench_score" in row or "filtered_resps" in row


def _lmms_response(row: dict) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0] if not isinstance(filtered[0], list) else filtered[0][0])
    resps = row.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    doc = row.get("vsibench_score") or row.get("doc") or {}
    return str(doc.get("prediction") or "")


def _lmms_stored_rule(doc: dict) -> tuple[float, bool, str]:
    question_type = str(doc.get("question_type", ""))
    if question_type in _MCA_TYPES:
        score = float(doc.get("accuracy") or 0.0)
    else:
        score = float(doc.get("MRA:.5:.95:.05") or 0.0)
    answered = bool(int(doc.get("answered") or 0)) if str(doc.get("answered", "")).strip() != "" else bool(doc.get("parsed_answer"))
    parsed = "" if doc.get("parsed_answer") is None else str(doc.get("parsed_answer"))
    return score, answered, parsed


def record_from_lmms_eval(row: dict) -> core.SampleRecord:
    doc = row.get("vsibench_score") or row.get("doc") or {}
    question_type = doc["question_type"]
    score, answered, parsed = _lmms_stored_rule(doc)
    response = _lmms_response(row)
    return core.SampleRecord(
        id=doc.get("id", row.get("doc_id")),
        dataset=str(doc.get("dataset", "")),
        scene_name=str(doc.get("scene_name", "")),
        question_type=question_type,
        question=str(doc.get("question", "")),
        ground_truth=str(doc.get("ground_truth", "")),
        options=doc.get("options"),
        response=response,
        output_tokens=int(row.get("output_tokens", 0) or 0),
        truncated=bool(row.get("truncated", 0)),
        finish_reason=str(row.get("finish_reason", "")),
        rule_parsed=parsed,
        rule_answered=answered,
        rule_score=score,
        final_parsed=parsed,
        final_answered=answered,
        final_score=score,
        failure_reason="",
        judge_used=False,
        judge_verdict="",
        judge_source="rule",
    )


def load_sample_records(samples_path: str) -> list[core.SampleRecord]:
    records: list[core.SampleRecord] = []
    with open(samples_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if _is_lmms_eval_row(row):
                records.append(record_from_lmms_eval(row))
            else:
                records.append(core.record_from_dict(row))
    return records


def _rule_metric(scored: dict, question_type: str, task) -> tuple[float, bool, str]:
    if question_type in task.MCA_QUESTION_TYPES:
        return scored.get("accuracy", 0.0), bool(scored.get("answered")), str(scored.get("parsed_answer", ""))
    mra = scored.get("MRA:.5:.95:.05", 0.0)
    return mra, bool(scored.get("answered")), str(scored.get("parsed_answer", ""))


def _apply_stored_judge(rec: core.SampleRecord, task, scoring) -> bool:
    """Re-derive final tier from a stored judge verdict. Returns judge_recovered."""
    if not rec.judge_used:
        rec.final_score = rec.rule_score
        rec.final_answered = rec.rule_answered
        rec.final_parsed = rec.rule_parsed
        return False

    rule_correct = core._is_correct(rec.rule_score, rec.question_type, task)
    if rec.judge_source == "judge_verdict":
        new_score = 1.0 if rec.judge_verdict == "Yes" else 0.0
        rec.final_parsed = rec.judge_verdict
    elif rec.judge_source == "judge_extract":
        number = scoring._to_float(rec.judge_verdict)
        if number is None:
            rec.final_score = rec.rule_score
            rec.final_answered = rec.rule_answered
            rec.final_parsed = rec.rule_parsed
            return False
        target = task.to_float(rec.ground_truth)
        new_score = task.mean_relative_accuracy(number, target, start=0.5, end=0.95, interval=0.05)
        rec.final_parsed = str(number)
    else:
        rec.final_score = rec.rule_score
        rec.final_answered = rec.rule_answered
        rec.final_parsed = rec.rule_parsed
        return False

    rec.final_score = new_score
    rec.final_answered = True
    new_correct = core._is_correct(new_score, rec.question_type, task)
    return not rule_correct and new_correct


def rescore_records(
    records: list[core.SampleRecord],
    task,
    scoring,
    boxed_primary: bool = False,
) -> dict:
    changed = 0
    for rec in records:
        doc = {
            "question_type": rec.question_type,
            "ground_truth": rec.ground_truth,
            "options": rec.options,
        }
        scored = task.vsibench_process_results(dict(doc), [rec.response])["vsibench_score"]
        rule_score, rule_answered, rule_parsed = _rule_metric(scored, rec.question_type, task)

        boxed_content = scoring.extract_boxed(rec.response)
        boxed_scored = task.vsibench_process_results(
            dict(doc), [scoring.normalize_boxed(boxed_content)]
        )["vsibench_score"]
        boxed_score, boxed_answered, boxed_parsed = _rule_metric(boxed_scored, rec.question_type, task)
        if boxed_primary and boxed_content is not None:
            rule_score, rule_answered, rule_parsed = boxed_score, boxed_answered, boxed_parsed

        if (
            abs(rule_score - rec.rule_score) > 1e-9
            or rule_answered != rec.rule_answered
            or rule_parsed != rec.rule_parsed
        ):
            changed += 1

        rec.rule_score = rule_score
        rec.rule_answered = rule_answered
        rec.rule_parsed = rule_parsed
        rec.boxed_content = boxed_content
        rec.boxed_answered = boxed_answered
        rec.boxed_score = boxed_score

        judge_recovered = _apply_stored_judge(rec, task, scoring)
        rec.failure_reason = core.classify_failure(
            truncated=rec.truncated,
            final_answered=rec.final_answered,
            score=rec.final_score,
            judge_recovered=judge_recovered,
        )
    return {"rows_changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="samples.jsonl from an offline or training-time eval")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to write rescored artifacts (default: <samples_dir>/rescore)",
    )
    parser.add_argument("--protocol", default=None, help="defaults to summary.json protocol or spatialstack")
    parser.add_argument(
        "--boxed-primary",
        action="store_true",
        help="if set, a closed box is the only answer site (same as eval --boxed-primary). "
        "Otherwise inherit summary.json boxed_primary.",
    )
    parser.add_argument(
        "--no-boxed-primary",
        action="store_true",
        help="force full-text rule scoring even if summary.json had boxed_primary",
    )
    args = parser.parse_args()

    samples_path = os.path.abspath(args.samples)
    output_dir = args.output_dir or os.path.join(os.path.dirname(samples_path), "rescore")
    summary_path = os.path.join(os.path.dirname(samples_path), "summary.json")
    previous: dict = {}
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            previous = json.load(handle)
    results_json = samples_path.replace("_samples_vsibench.jsonl", "_results.json")
    if not previous and os.path.exists(results_json):
        with open(results_json) as handle:
            results = json.load(handle)
        vsibench = (results.get("results") or {}).get("vsibench") or {}
        gen = ((results.get("configs") or {}).get("vsibench") or {}).get("generation_kwargs") or {}
        previous = {
            "model": "models/Qwen3.5-4B",
            "frames": 32,
            "max_tokens": int(gen.get("max_new_tokens") or 1024),
            "protocol": "spatialstack",
            "boxed_primary": False,
            "rule_only": {
                "overall": float(vsibench.get("vsibench_score,none") or 0.0),
                "answered_pct": float(vsibench.get("vsibench_answered,none") or 0.0),
            },
        }

    records = load_sample_records(samples_path)
    if not records:
        raise SystemExit(f"no samples in {samples_path}")

    protocol = args.protocol or previous.get("protocol", "spatialstack")
    if args.no_boxed_primary:
        boxed_primary = False
    elif args.boxed_primary:
        boxed_primary = True
    else:
        boxed_primary = bool(previous.get("boxed_primary", False))
    os.environ["VSIBENCH_PROTOCOL"] = protocol
    os.environ["VSIBENCH_MAX_NEW_TOKENS"] = str(previous.get("max_tokens", 4096))
    os.environ["VSIBENCH_BOXED_PRIMARY"] = "1" if boxed_primary else "0"

    # Force reload of the shared scorer so edits to vsibench_scoring.py apply.
    task = core.load_module(core.TASK_UTILS, "vsibench_task_utils_rescore")
    task._shared_module = None  # type: ignore[attr-defined]
    scoring = core.load_module(core.SCORING, "vsibench_scoring_rescore")

    started = time.perf_counter()
    stats = rescore_records(records, task, scoring, boxed_primary=boxed_primary)
    judge_stats = {
        "judged": sum(1 for r in records if r.judge_used),
        "recovered": sum(
            1
            for r in records
            if r.judge_used
            and core._is_correct(r.final_score, r.question_type, task)
            and not core._is_correct(r.rule_score, r.question_type, task)
        ),
    }
    cfg = core.EvalConfig(
        model=previous.get("model", "unknown"),
        frames=int(previous.get("frames", 32)),
        max_tokens=int(previous.get("max_tokens", 4096)),
        protocol=protocol,
        boxed_primary=boxed_primary,
        judge_api_base="stored" if judge_stats["judged"] else None,
    )
    summary = core.build_summary(records, task, cfg, float(previous.get("wall_seconds", 0.0)), judge_stats)
    summary["rescore_seconds"] = time.perf_counter() - started
    summary["rescore_source"] = samples_path
    summary["rescore_rows_changed"] = stats["rows_changed"]
    source_overall = (previous.get("rule_only") or {}).get("overall")
    if source_overall is not None:
        summary["delta_vs_source_overall"] = summary["rule_only"]["overall"] - float(source_overall)
    if previous:
        summary["previous"] = {
            "rule_only": previous.get("rule_only", {}),
            "judge_assisted": previous.get("judge_assisted", {}),
            "failure_reasons": previous.get("failure_reasons", {}),
        }

    core.write_results(records, summary, output_dir)
    print(core.format_report(summary))
    print(
        f"\nrescore: {stats['rows_changed']} rule rows changed | "
        f"saved: {output_dir}/",
        flush=True,
    )
    if previous:
        old_rule = previous.get("rule_only", {}).get("overall")
        old_judge = previous.get("judge_assisted", {}).get("overall")
        if old_rule is not None:
            print(
                f"delta rule-only: {summary['rule_only']['overall'] - old_rule:+.2f} "
                f"({old_rule:.2f} -> {summary['rule_only']['overall']:.2f})",
                flush=True,
            )
        if old_judge is not None:
            print(
                f"delta judge-assisted: {summary['judge_assisted']['overall'] - old_judge:+.2f} "
                f"({old_judge:.2f} -> {summary['judge_assisted']['overall']:.2f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
