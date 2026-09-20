#!/usr/bin/env python3
"""Second grading tier for a teacher dump: gpt-oss-120b reads what the rules could not.

    python3 scripts/opsd/tools/judge_teacher_dump.py \
        --dir logs/eval/teacher_reliability/as_trained \
        --judge-api-base http://127.0.0.1:8100/v1

The judge is used for **extraction**, not for verdicts, everywhere a structured
answer exists: it reads the number, the option letter, the direction tuple or
the named candidate out of the teacher's prose, and this project's own metric
then grades it.  A Yes/No judge would replace MRA with a binary and would decide
correctness by its own standard rather than the benchmark's, which is how the
Vision-OPD cascade ended up overturning ~10% of rule-accepted rows (LESSON-019).

``llava_hound`` is the one exception.  Its gold is an open caption, so there is
nothing to extract and the judge has to give a semantic verdict.  Because that
tier cannot be checked against a rule tier, it is checked against itself:

* **reverse audit** -- rows the rules already scored are re-extracted by the
  judge and the disagreement rate is reported.  A judge that contradicts the
  parser on rows the parser could read is not trustworthy on rows it could not.
* **calibration probe** -- the semantic judge is shown the gold as if it were
  the response (it must say Yes) and a caption from a different sample (it must
  say No).  Both error rates are reported next to the score.

Only rows the rule tier declined are re-scored, so the judge can raise the
number and never lower it; the rule-only number stays in the summary beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from argparse import Namespace
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

import spar_scoring as spar  # noqa: E402
import teacher_scoring as ts  # noqa: E402
from eval_visionopd.judge import PROMPT_TEMPLATE, judge_via_api, normalize_verdict  # noqa: E402

EXTRACT_HEADER = (
    "You are reading one answer out of a model's response. Do not solve the question "
    "yourself and do not judge whether the response is right: report only what the "
    "response says.\n\nThe question is: {question}\n\nThe response is: {response}\n\n"
)

EXTRACT_INSTRUCTION = {
    "number": (
        "Reply with the single final number the response gives, in digits, with no units "
        "and no other text. If the response settles on no number, reply exactly NONE."
    ),
    "option": (
        "Reply with the single option letter the response selects, and nothing else. "
        "If the response selects no option, reply exactly NONE."
    ),
    "direction": (
        "Report the direction the response gives for the object asked about, on exactly "
        "three lines and nothing else:\n"
        "horizontal=<left|right|unspecified>\n"
        "vertical=<above|below|unspecified>\n"
        "depth=<closer|farther|unspecified>\n"
        "Use 'unspecified' for any axis the response does not state. If the response "
        "describes a position before and after the observer moves, report the position "
        "after the move."
    ),
    "compare": (
        "Reply with the single item the response names as its answer, copied from the "
        "question's own wording for that item, and nothing else. If the response names "
        "no item, reply exactly NONE."
    ),
    "yesno": (
        "Reply with Yes or No, whichever the response gives as its answer, and nothing "
        "else. If the response gives neither, reply exactly NONE."
    ),
}

# A judge window has to hold the question (up to 32 view placeholders are
# stripped, but the SPAR BEV prompts are long) plus the response.
DEFAULT_MAX_RESPONSE_CHARS = 6000


def _trim(text: str, limit: int) -> str:
    text = text or ""
    if limit and len(text) > limit:
        return "...[truncated]...\n" + text[-limit:]
    return text


def build_prompt(row: dict, max_response_chars: int) -> str:
    question = _trim(row.get("question", ""), 4000)
    response = _trim(row.get("response", ""), max_response_chars)
    kind = row["judge_kind"]
    if kind == "semantic":
        return PROMPT_TEMPLATE.format(question=question, gt=row.get("ground_truth", ""), response=response)
    return EXTRACT_HEADER.format(question=question, response=response) + EXTRACT_INSTRUCTION[kind]


def parse_direction_reply(reply: str) -> dict[str, str]:
    axes: dict[str, str] = {}
    for line in (reply or "").splitlines():
        if "=" not in line:
            continue
        axis, _, value = line.partition("=")
        axis, value = axis.strip().lower(), value.strip().lower().strip(".")
        if axis in spar.AXES and value in spar.AXES[axis]:
            axes[axis] = value
    return axes


def rescore(row: dict, reply: str) -> dict:
    """Grade an extraction reply with the same metric the rule tier uses."""
    kind = row["judge_kind"]
    gold = str(row.get("ground_truth", "")).strip()
    question_type = row.get("question_type") or ""
    out = {"answered": 0, "score": 0.0, "parsed_repr": "", "judge_verdict": (reply or "").strip()[:200]}
    text = (reply or "").strip()
    if not text or text.upper().startswith("NONE"):
        return out

    if kind == "semantic":
        verdict = normalize_verdict(text)
        out["judge_verdict"] = verdict
        if verdict == "UNPARSED":
            return out
        out["answered"] = 1
        out["score"] = float(verdict == "Yes")
        return out

    if kind == "number":
        from vsibench_scoring import extract_vsibench_number

        value = extract_vsibench_number(text)
        if value is None:
            return out
        target = spar.parse_gold(question_type, gold).number if question_type in spar.NUMERIC_TYPES else None
        if target is None:
            try:
                target = float(gold)
            except ValueError:
                return out
        out["answered"] = 1
        out["parsed_repr"] = f"{value:g}"
        out["score"] = spar.mean_relative_accuracy(value, target)
        return out

    if kind == "option":
        from vsibench_scoring import extract_vsibench_option

        letter = extract_vsibench_option(text, row.get("options") or None)
        if not letter:
            return out
        out["answered"] = 1
        out["parsed_repr"] = letter
        out["score"] = float(letter == gold.upper())
        return out

    if kind == "direction":
        gold_axes = spar.parse_gold(question_type, gold)
        if gold_axes.ambiguous or not gold_axes.axes:
            return out
        axes = parse_direction_reply(text)
        if not axes:
            return out
        out["answered"] = 1
        out["parsed_repr"] = "+".join(f"{a}:{v}" for a, v in sorted(axes.items()))
        out["score"] = float(
            all(axes.get(axis) == value for axis, value in gold_axes.axes.items())
        )
        return out

    if kind == "compare":
        family = spar.family_of(question_type)
        scored = spar.score_row(question_type, gold, text, question=row.get("question", ""))
        out["answered"] = scored["answered"]
        out["parsed_repr"] = scored["parsed_repr"]
        out["score"] = scored["score"]
        out["family"] = family
        return out

    if kind == "yesno":
        verdict = normalize_verdict(text)
        if verdict == "UNPARSED":
            return out
        out["answered"] = 1
        out["parsed_repr"] = verdict.lower()
        out["score"] = float(verdict.lower() == gold.strip().lower())
        return out

    return out


def call_judge(rows: list[dict], args, max_response_chars: int) -> list[str]:
    if not rows:
        return []
    prompts = [build_prompt(row, max_response_chars) for row in rows]
    replies = judge_via_api(
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
    return [content or reasoning for content, reasoning in replies]


KIND_FOR_FAMILY = {
    "numeric": "number",
    "mcq": "option",
    "direction": "direction",
    "compare_pair": "compare",
    "compare_set": "compare",
    "yesno": "yesno",
}


def reverse_audit(audit_pool: list[dict], args, sample: int, max_chars: int) -> dict:
    """Re-extract rows the rules already read, and count how often the judge disagrees."""
    usable = [row for row in audit_pool if row.get("family") in KIND_FOR_FAMILY]
    if not usable:
        return {"sampled": 0}
    rng = random.Random(20260821)
    picked = rng.sample(usable, min(sample, len(usable)))
    probes = [dict(row, judge_kind=KIND_FOR_FAMILY[row["family"]]) for row in picked]

    replies = call_judge(probes, args, max_chars)
    disagree = Counter()
    # A numeric row is scored by MRA, so "disagreement" is a difference in a
    # continuous number: reading 2.3 where the rule read 2.34 counts the same as
    # reading a completely different quantity. Keep the pairs so the two can be
    # told apart instead of reporting one rate that conflates them.
    material = Counter()
    examples: list[dict] = []
    unreadable = 0
    for probe, reply in zip(probes, replies):
        verdict = rescore(probe, reply)
        if not verdict["answered"]:
            unreadable += 1
            continue
        gap = abs(verdict["score"] - float(probe["rule_score"]))
        if gap <= 1e-9:
            continue
        disagree[probe["family"]] += 1
        if gap > 0.25:
            material[probe["family"]] += 1
        if len(examples) < 40:
            examples.append(
                {
                    "family": probe["family"],
                    "question_type": probe.get("question_type", ""),
                    "gold": probe.get("ground_truth", "")[:160],
                    "rule_score": round(float(probe["rule_score"]), 4),
                    "judge_score": round(verdict["score"], 4),
                    "judge_read": verdict.get("parsed_repr", ""),
                    "gap": round(gap, 4),
                }
            )
    total_read = len(probes) - unreadable
    return {
        "sampled": len(probes),
        "judge_unreadable": unreadable,
        "disagreed_with_rule": sum(disagree.values()),
        "disagreement_pct": 100 * sum(disagree.values()) / max(total_read, 1),
        "disagreement_by_family": dict(disagree),
        # Only gaps big enough to flip a verdict rather than nudge an MRA.
        "material_disagreement": sum(material.values()),
        "material_pct": 100 * sum(material.values()) / max(total_read, 1),
        "material_by_family": dict(material),
        "examples": examples,
    }


def calibration_probe(queue: list[dict], args, sample: int, max_chars: int) -> dict:
    """The semantic judge on answers that are certainly right and certainly wrong."""
    semantic = [row for row in queue if row["judge_kind"] == "semantic" and row.get("ground_truth")]
    if len(semantic) < 4:
        return {"sampled": 0}
    rng = random.Random(20260821)
    picked = rng.sample(semantic, min(sample, len(semantic)))
    shuffled = picked[1:] + picked[:1]

    positives = [dict(row, response=row["ground_truth"]) for row in picked]
    negatives = [dict(row, response=other["ground_truth"]) for row, other in zip(picked, shuffled)]
    replies = call_judge(positives + negatives, args, max_chars)
    half = len(positives)
    verdicts = [normalize_verdict(reply) for reply in replies]
    return {
        "sampled": half,
        "gold_as_response_said_no": sum(1 for verdict in verdicts[:half] if verdict == "No"),
        "gold_as_response_unparsed": sum(1 for verdict in verdicts[:half] if verdict == "UNPARSED"),
        "other_caption_said_yes": sum(1 for verdict in verdicts[half:] if verdict == "Yes"),
        "other_caption_unparsed": sum(1 for verdict in verdicts[half:] if verdict == "UNPARSED"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="directory holding scored.jsonl and judge_queue.jsonl")
    parser.add_argument("--judge-api-base", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--parallel-workers", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--max-response-chars", type=int, default=DEFAULT_MAX_RESPONSE_CHARS)
    parser.add_argument("--group-by", default="n_views_teacher")
    parser.add_argument("--audit-sample", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0, help="smoke runs; judge only the first N queued rows")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="re-run just the reverse audit against an existing summary, leaving the scores alone",
    )
    args = parser.parse_args()

    work_dir = args.dir if os.path.isabs(args.dir) else os.path.join(REPO_ROOT, args.dir)
    def read_jsonl(name: str) -> list[dict]:
        path = os.path.join(work_dir, name)
        if not os.path.exists(path):
            return []
        return [json.loads(line) for line in open(path) if line.strip()]

    queue = read_jsonl("judge_queue.jsonl")
    audit_pool = read_jsonl("audit_pool.jsonl")

    if args.audit_only:
        audit = reverse_audit(audit_pool, args, args.audit_sample, args.max_response_chars)
        summary_path = os.path.join(work_dir, "summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        summary.setdefault("judge", {})["reverse_audit"] = audit
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in audit.items() if k != "examples"}, indent=2))
        for example in audit["examples"]:
            print(f"  {example}")
        return

    scored = read_jsonl("scored.jsonl")
    if args.limit:
        queue = queue[: args.limit]
    print(f"{len(scored)} scored rows, {len(queue)} queued for the judge")
    print(f"  by kind: {dict(Counter(row['judge_kind'] for row in queue))}")

    replies = call_judge(queue, args, args.max_response_chars)
    verdicts = {}
    stats = Counter()
    for row, reply in zip(queue, replies):
        result = rescore(row, reply)
        verdicts[int(row["row_index"])] = result
        stats[f"{row['judge_kind']}/{'read' if result['answered'] else 'unreadable'}"] += 1

    recovered = 0
    for row in scored:
        result = verdicts.get(int(row["row_index"]))
        if result is None:
            row["judge_used"] = 0
            continue
        row["judge_used"] = 1
        row["judge_verdict"] = result["judge_verdict"]
        if result["answered"]:
            row["answered"] = 1
            row["parsed_repr"] = result["parsed_repr"] or row.get("parsed_repr", "")
            row["score"] = result["score"]
            recovered += int(result["score"] > 0)

    audit = reverse_audit(audit_pool, args, args.audit_sample, args.max_response_chars)
    probe = calibration_probe(queue, args, min(args.audit_sample, 200), args.max_response_chars)

    summary_path = os.path.join(work_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    summary["judge"] = {
        "model": args.judge_model,
        "queued": len(queue),
        "read_by_judge": sum(count for key, count in stats.items() if key.endswith("/read")),
        "still_unreadable": sum(count for key, count in stats.items() if key.endswith("/unreadable")),
        "by_kind": dict(stats),
        "rows_scoring_above_zero": recovered,
        "reverse_audit": audit,
        "semantic_calibration": probe,
    }
    summary["judge_assisted"] = {
        "by_group": ts.aggregate(scored, (args.group_by,)),
        "by_source": ts.aggregate(scored, ("source",)),
        "by_question_type": ts.aggregate(scored, ("source", "question_type")),
        "by_family": ts.aggregate(scored, ("family",)),
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(work_dir, "scored_judged.jsonl"), "w") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\njudge read {summary['judge']['read_by_judge']} of {len(queue)} queued rows")
    print(f"  by kind: {dict(stats)}")
    print(f"  reverse audit: {audit}")
    print(f"  semantic calibration: {probe}")
    for title, key in (("by " + args.group_by, "by_group"), ("by source", "by_source"), ("by family", "by_family")):
        table = summary["judge_assisted"][key]
        print(f"\n=== {title} (judge-assisted) ===")
        print(f"  {'key':<38}{'rows':>8}{'acc':>8}{'answered':>10}{'acc|ans':>9}")
        for name, info in table.items():
            print(
                f"  {name:<38}{info['rows']:>8}{info['acc']:>8.2f}"
                f"{info['answered_pct']:>9.1f}%{info['acc_of_answered']:>9.2f}"
            )
    print(f"\nsaved: {summary_path}, {work_dir}/scored_judged.jsonl")


if __name__ == "__main__":
    main()
