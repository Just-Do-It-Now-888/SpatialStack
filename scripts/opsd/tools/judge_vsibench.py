#!/usr/bin/env python3
"""Vision-OPD-style LLM judging for VSI-Bench, applied to answers already on disk.

    python3 scripts/opsd/tools/judge_vsibench.py \
        logs/eval/<run>/frames_32/vsibench/*/*_samples_vsibench.jsonl \
        --judge-api-base http://127.0.0.1:8001/v1/ --judge-model judge \
        --output logs/eval/judge/<run>.json

Generation is the expensive half and the responses are already saved, so adding a
judge must never cost another generation pass. This reads a sample dump from
either harness (lmms_eval or ``eval_vsibench_vllm.py``) and re-grades it.

The cascade follows ``Vision-OPD-main/eval/judge_qwenlm.py``, whose defining
property is that **the rule tier can only mark a row correct, never wrong**.
Anything not confirmed by a rule goes to the judge, including plainly wrong
answers -- which is why upstream sends 25% of HR-Bench and 89% of ZoomBench to
the judge rather than the ~2% one might expect from parse failures alone.

One deliberate divergence, because copying upstream exactly here would silently
change the metric:

* **Multiple choice** is upstream's design unchanged. The rule tier accepts a
  letter that matches the gold option; everything else gets the Yes/No judge,
  and the verdict maps onto exact-match accuracy.
* **Numerical** questions are scored by ``MRA:.5:.95:.05``, a mean over ten
  thresholds -- a graded value, not a hit or miss. A Yes/No judge would collapse
  it into binary accuracy while still being reported under the MRA name. So here
  the judge is used as an *extractor*: it is asked which number the response
  settles on, and MRA is then computed from that number by lmms_eval's own
  function. The metric definition is untouched.

Rule-only and judge-assisted scores are both reported. The rule-only number is
what previous runs were scored with, so keeping it visible is what makes the two
comparable instead of silently superseded.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import string
import sys
from argparse import Namespace
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

TASK_UTILS = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench", "utils.py")
VISIONOPD_JUDGE = os.path.join(REPO_ROOT, "scripts", "opsd", "eval_visionopd", "judge.py")
SCORING = os.path.join(REPO_ROOT, "scripts", "opsd", "vsibench_scoring.py")
DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/"
    "bdcadb3fea447621a828a24911801faba3587c12"
)

# Upstream's wording, reused verbatim so the multiple-choice half stays faithful.
# (Imported from judge.py rather than copied; this constant is only the fallback.)
MCA_EXTRACT_TEMPLATE = (
    "Read the response below and report which answer option it selects.\n"
    "The question is: {question}\n"
    "The options are:\n{options}\n"
    "The response is: {response}\n"
    "Reply with only the option letter. If the response does not settle on one of "
    "the listed options, reply exactly NONE."
)

NA_EXTRACT_TEMPLATE = (
    "Read the response below and report the single final numeric answer it gives.\n"
    "The question is: {question}\n"
    "The response is: {response}\n"
    "Reply with only the number, in digits, with no units, no ranges and no other text. "
    "If the response does not settle on a numeric answer, reply exactly NONE."
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gold_answer(doc: dict) -> str:
    """The gold option as "C. right", not the bare "C".

    lmms_eval stores VSI-Bench's ground truth as a lone letter. Handing that to
    the judge asks it to confirm an answer whose meaning it was never told, and
    the reverse audit showed it then rejects correct responses at roughly 10%.
    Vision-OPD does not hit this because its own gold field carries the option
    text, which is what ``extract_gt_option`` is written to strip off.
    """
    letter = str(doc.get("ground_truth", "")).strip()
    for index, raw in enumerate(doc.get("options") or []):
        text = str(raw).strip()
        match = re.match(r"\s*([A-Za-z])\s*[.):]\s*(.*)", text)
        found, body = (match.group(1).upper(), match.group(2).strip()) if match else (
            string.ascii_uppercase[index],
            text,
        )
        if found == letter.upper():
            return f"{letter}. {body}" if body else letter
    return letter


def read_dump(path: str, questions_by_id: dict):
    """Normalise an lmms_eval dump and a vLLM dump into one shape."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "doc" in raw:
                doc = raw["doc"]
                rows.append(
                    {
                        "doc": doc,
                        "question": doc.get("question", ""),
                        "response": raw["filtered_resps"][0],
                        "truncated": None,
                    }
                )
            else:
                doc = {
                    "id": raw.get("id"),
                    "question_type": raw["question_type"],
                    "ground_truth": raw["ground_truth"],
                    "options": raw.get("options"),
                }
                rows.append(
                    {
                        "doc": doc,
                        # Older vLLM dumps predate the question field; recover it
                        # from the dataset so the judge is not asked to grade an
                        # answer without seeing the question.
                        "question": raw.get("question") or questions_by_id.get(raw.get("id"), ""),
                        "response": raw["response"],
                        "truncated": raw.get("truncated"),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="sample dumps (globs allowed)")
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--parallel-workers", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument(
        "--mca-mode",
        choices=("extract", "verdict"),
        default="extract",
        help="how the judge handles multiple choice. 'extract' asks which option "
        "the response selects and lets lmms_eval's exact match grade it, so the "
        "metric keeps its published definition and the judge can move a row "
        "either way. 'verdict' is Vision-OPD's Yes/No cascade, kept for "
        "comparison: it grades whether the prose means the same as the gold "
        "answer, which is a different question from which option was chosen, and "
        "because the rule tier is accept-only it can only raise the score",
    )
    parser.add_argument(
        "--na-scope",
        choices=("unresolved", "all"),
        default="unresolved",
        help="which numerical rows the judge extracts from. 'unresolved' trusts the "
        "rule parser when it found a number and asks the judge only when it did "
        "not; 'all' asks the judge for every numerical row, which is the closer "
        "analogue of upstream sending everything unconfirmed to the judge",
    )
    parser.add_argument(
        "--audit-accepted",
        type=int,
        default=0,
        help="also send this many rule-accepted multiple-choice rows to the judge. "
        "The cascade is accept-only, so the judge never sees a row the rule called "
        "correct and can therefore only raise the score -- a lenient judge inflates "
        "it invisibly. This samples the other direction to measure that: any 'No' on "
        "an accepted row is a disagreement the normal run would never surface. "
        "Audit rows are reported separately and never change the score.",
    )
    parser.add_argument("--audit-seed", type=int, default=42)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run only the reverse audit, skipping the real grading pass. The "
        "audit is the cheap half and is what needs iterating on, so this avoids "
        "paying for the full judge round every time",
    )
    parser.add_argument("--max-response-chars", type=int, default=0, help="0 sends the whole response")
    parser.add_argument("--protocol", default="spatialstack", choices=("spatialstack", "lmms_legacy"))
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--dry-run", action="store_true", help="report how many rows would be judged, then stop")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["VSIBENCH_PROTOCOL"] = args.protocol
    task = load_module(TASK_UTILS, "vsibench_task_utils")
    upstream = load_module(VISIONOPD_JUDGE, "visionopd_judge")
    scoring = load_module(SCORING, "vsibench_scoring")

    questions_by_id = {}
    jsonl = os.path.join(args.snapshot, "test.jsonl")
    if os.path.exists(jsonl):
        with open(jsonl) as handle:
            for line in handle:
                doc = json.loads(line)
                questions_by_id[doc["id"]] = doc["question"]

    paths: list[str] = []
    for pattern in args.paths:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded or [pattern])

    for path in paths:
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            continue
        rows = read_dump(path, questions_by_id)
        if not rows:
            continue

        print(f"\n{'=' * 78}\n{os.path.relpath(path, REPO_ROOT)}  ({len(rows)} rows)")

        # --- tier 1: rules ---------------------------------------------------
        rule_rows, pending, prompts, kinds = [], [], [], []
        for index, row in enumerate(rows):
            doc = row["doc"]
            response = row["response"]
            scored = task.vsibench_process_results(dict(doc), [response])["vsibench_score"]
            rule_rows.append(scored)

            if args.audit_only:
                continue

            question_type = doc["question_type"]
            shown = response
            if args.max_response_chars and len(shown) > args.max_response_chars:
                shown = "...[truncated]...\n" + shown[-args.max_response_chars :]

            if question_type in task.MCA_QUESTION_TYPES:
                if args.mca_mode == "extract":
                    # Every row, not just the rejected ones. Extraction is
                    # symmetric -- it can move a row either way -- so unlike the
                    # verdict cascade it does not bias the score upward.
                    pending.append(index)
                    kinds.append("mca_extract")
                    listed = "\n".join(str(o) for o in (doc.get("options") or []))
                    prompts.append(
                        MCA_EXTRACT_TEMPLATE.format(
                            question=row["question"], options=listed, response=shown
                        )
                    )
                    continue
                # Upstream's rule tier: a matching letter is accepted, anything
                # else -- wrong letter or no letter -- goes to the judge.
                if scored.get("accuracy") == 1.0:
                    continue
                pending.append(index)
                kinds.append("mca")
                prompts.append(
                    upstream.PROMPT_TEMPLATE.format(
                        question=row["question"], gt=gold_answer(doc), response=shown
                    )
                )
            else:
                if args.na_scope == "unresolved" and scored.get("answered"):
                    continue
                pending.append(index)
                kinds.append("na")
                prompts.append(NA_EXTRACT_TEMPLATE.format(question=row["question"], response=shown))

        # Rows the rule tier accepted, sampled for the reverse check. Collected
        # here so the audit shares one judge round-trip with the real grading.
        audit_index: list[int] = []
        if args.audit_accepted:
            import random

            accepted = [
                i
                for i, (row, scored) in enumerate(zip(rows, rule_rows))
                if row["doc"]["question_type"] in task.MCA_QUESTION_TYPES and scored.get("accuracy") == 1.0
            ]
            rng = random.Random(args.audit_seed)
            audit_index = rng.sample(accepted, min(args.audit_accepted, len(accepted)))
            for index in audit_index:
                row = rows[index]
                shown = row["response"]
                if args.max_response_chars and len(shown) > args.max_response_chars:
                    shown = "...[truncated]...\n" + shown[-args.max_response_chars :]
                prompts.append(
                    upstream.PROMPT_TEMPLATE.format(
                        question=row["question"], gt=gold_answer(row["doc"]), response=shown
                    )
                )
                pending.append(index)
                kinds.append("audit")

        rule_overall = task.vsibench_aggregate_results([dict(r) for r in rule_rows])
        rule_answered = task.vsibench_aggregate_answered(rule_rows)
        n_mca = sum(1 for k in kinds if k in ("mca", "mca_extract"))
        n_na = sum(1 for k in kinds if k == "na")
        n_audit = sum(1 for k in kinds if k == "audit")
        print(f"  rule-only : overall {rule_overall:.2f} | answered {rule_answered:.2f}%")
        print(
            f"  to judge  : {n_mca + n_na} / {len(rows)} ({100 * (n_mca + n_na) / len(rows):.1f}%)"
            f"  [{n_mca} multiple-choice {args.mca_mode}s, {n_na} numeric extractions]"
            + (f" + {n_audit} audit rows" if n_audit else "")
        )

        if args.dry_run:
            continue
        if not pending:
            print("  nothing for the judge")
            continue
        if not args.judge_api_base:
            raise SystemExit("--judge-api-base is required (or pass --dry-run)")

        # --- tier 2: judge ---------------------------------------------------
        replies = upstream.judge_via_api(
            prompts,
            Namespace(
                judge_api_key=args.judge_api_key,
                judge_api_base=args.judge_api_base,
                judge_model=args.judge_model,
                judge_max_tokens=args.judge_max_tokens,
                reasoning_effort=args.reasoning_effort,
                parallel_workers=args.parallel_workers,
            ),
        )

        judged_rows = [dict(r) for r in rule_rows]
        audit_disagreements: list[dict] = []
        sources = Counter()
        flips = {"mca_to_correct": 0, "mca_to_wrong": 0, "na_recovered": 0, "na_changed": 0}
        api_errors = 0
        unreadable = 0

        for offset, (content, reasoning) in enumerate(replies):
            index = pending[offset]
            doc = rows[index]["doc"]
            reply = content or reasoning
            if reasoning == "[JUDGE_API_ERROR]":
                api_errors += 1

            if kinds[offset] == "audit":
                verdict = upstream.normalize_verdict(reply)
                sources[f"audit_{verdict.lower()}"] += 1
                if verdict == "No":
                    # Kept in full so the disagreements can be read by hand.
                    # Whether these are parser luck or judge error is the whole
                    # question, and it cannot be settled from counts alone.
                    audit_disagreements.append(
                        {
                            "id": doc.get("id"),
                            "question_type": doc["question_type"],
                            "question": rows[index]["question"],
                            "options": doc.get("options"),
                            "ground_truth": doc["ground_truth"],
                            "rule_parsed": rule_rows[index].get("parsed_answer"),
                            "judge_reply": reply,
                            "response": rows[index]["response"],
                        }
                    )
                continue

            if kinds[offset] == "mca_extract":
                letter = ""
                if reply and reply.strip().upper() != "NONE":
                    letter = scoring.extract_vsibench_option(reply, doc.get("options"))
                if not letter:
                    unreadable += 1
                    sources["mca_extract_none"] += 1
                    continue
                before = judged_rows[index].get("accuracy", 0.0)
                after = 1.0 if letter == str(doc["ground_truth"]).strip().upper() else 0.0
                judged_rows[index]["accuracy"] = after
                judged_rows[index]["parsed_answer"] = letter
                judged_rows[index]["answered"] = 1
                sources["mca_extract"] += 1
                if after > before:
                    flips["mca_to_correct"] += 1
                elif after < before:
                    flips["mca_to_wrong"] += 1
                continue

            if kinds[offset] == "mca":
                verdict = upstream.normalize_verdict(reply)
                if verdict == "UNPARSED":
                    unreadable += 1
                    sources["mca_judge_unreadable"] += 1
                    continue
                before = judged_rows[index].get("accuracy", 0.0)
                after = 1.0 if verdict == "Yes" else 0.0
                judged_rows[index]["accuracy"] = after
                sources["mca_judge"] += 1
                if after > before:
                    flips["mca_to_correct"] += 1
                elif after < before:
                    flips["mca_to_wrong"] += 1
            else:
                number = None
                if reply and reply.strip().upper() != "NONE":
                    number = scoring.extract_vsibench_number(reply)
                if number is None:
                    unreadable += 1
                    sources["na_judge_no_number"] += 1
                    continue
                target = task.to_float(doc["ground_truth"])
                before = judged_rows[index].get("MRA:.5:.95:.05", 0.0)
                after = task.mean_relative_accuracy(number, target, start=0.5, end=0.95, interval=0.05)
                judged_rows[index]["MRA:.5:.95:.05"] = after
                judged_rows[index]["answered"] = 1
                sources["na_judge"] += 1
                if not rule_rows[index].get("answered"):
                    flips["na_recovered"] += 1
                elif abs(after - before) > 1e-9:
                    flips["na_changed"] += 1

        judged_overall = task.vsibench_aggregate_results([dict(r) for r in judged_rows])
        judged_answered = task.vsibench_aggregate_answered(judged_rows)

        if not args.audit_only:
            print(f"\n  judge-assisted: overall {judged_overall:.2f} | answered {judged_answered:.2f}%")
            print(f"  delta vs rule-only: {judged_overall - rule_overall:+.2f}")
            print(f"  verdict tiers: {dict(sources)}")
            print(
                f"  flips: mca wrong->correct {flips['mca_to_correct']}, "
                f"mca correct->wrong {flips['mca_to_wrong']}, "
                f"na recovered {flips['na_recovered']}, na value changed {flips['na_changed']}"
            )
        if unreadable:
            print(f"  WARNING: {unreadable} judge replies unusable; those rows keep their rule score")
        if api_errors:
            print(f"  WARNING: {api_errors} judge calls failed")

        if n_audit:
            agree = sources.get("audit_yes", 0)
            disagree = sources.get("audit_no", 0)
            print(
                f"\n  reverse audit on {n_audit} rule-accepted rows: "
                f"judge agrees {agree}, disagrees {disagree}"
                + (f", unreadable {sources.get('audit_unparsed', 0)}" if sources.get("audit_unparsed") else "")
            )
            if disagree:
                # The cascade cannot act on these, so they do not change the
                # score. They bound how much of the +delta above is the judge
                # being lenient rather than the parser being wrong.
                print(
                    f"  -> the judge would overturn {100 * disagree / n_audit:.1f}% of rows the rule accepted; "
                    "treat the gain above as an upper bound by roughly that fraction"
                )

        if args.audit_only:
            if args.output:
                out_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_ROOT, args.output)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as handle:
                    json.dump(
                        {
                            "source": os.path.relpath(path, REPO_ROOT),
                            "audit_rows": n_audit,
                            "agree": sources.get("audit_yes", 0),
                            "disagree": sources.get("audit_no", 0),
                            "disagreements": audit_disagreements,
                        },
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"\n  saved: {out_path}")
            continue

        by_type = defaultdict(lambda: [0.0, 0.0, 0])
        for rule, judged, row in zip(rule_rows, judged_rows, rows):
            key = row["doc"]["question_type"]
            metric = "accuracy" if key in task.MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
            entry = by_type[key]
            entry[0] += rule.get(metric, 0.0)
            entry[1] += judged.get(metric, 0.0)
            entry[2] += 1
        print(f"\n  {'question type':<32} {'rule':>8} {'judged':>8} {'delta':>7}")
        for key in sorted(by_type):
            rule_sum, judged_sum, count = by_type[key]
            r, j = 100 * rule_sum / count, 100 * judged_sum / count
            print(f"  {key:<32} {r:>8.2f} {j:>8.2f} {j - r:>+7.2f}")

        if args.output:
            out_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_ROOT, args.output)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as handle:
                json.dump(
                    {
                        "source": os.path.relpath(path, REPO_ROOT),
                        "protocol": args.protocol,
                        "mca_mode": args.mca_mode,
                        "na_scope": args.na_scope,
                        "judge_model": args.judge_model,
                        "rows": len(rows),
                        "judged_rows": len(pending),
                        "rule_only": {"overall": rule_overall, "answered": rule_answered},
                        "judge_assisted": {"overall": judged_overall, "answered": judged_answered},
                        "tiers": dict(sources),
                        "flips": flips,
                        "judge_unreadable": unreadable,
                        "judge_api_errors": api_errors,
                        "by_question_type": {
                            k: {"rule": 100 * v[0] / v[2], "judged": 100 * v[1] / v[2], "n": v[2]}
                            for k, v in sorted(by_type.items())
                        },
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
