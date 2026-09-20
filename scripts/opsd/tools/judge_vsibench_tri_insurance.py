#!/usr/bin/env python3
"""Three insurance checks before full judge-extract scoring on VSI-Bench.

1. Calibration probe  -- synthetic responses with known expected extractions.
2. Reverse audit      -- rows the rule tier marked correct; does strict judge agree?
3. Spot-check panel   -- stratified real rows (truncation, size estimation, id=168).

    python scripts/opsd/tools/judge_vsibench_tri_insurance.py \\
        logs/vsi_train_eval/.../samples.jsonl \\
        --judge-api-base http://127.0.0.1:8001/v1/ \\
        --output logs/eval/judge/tri_insurance_step100.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
from argparse import Namespace
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

TASK_UTILS = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench", "utils.py")
VISIONOPD_JUDGE = os.path.join(REPO_ROOT, "scripts", "opsd", "eval_visionopd", "judge.py")
SCORING = os.path.join(REPO_ROOT, "scripts", "opsd", "vsibench_scoring.py")

MCA_HEADER = (
    "You are a grading assistant. Do not solve the question yourself and do not "
    "judge whether the response is correct. Report only what final option the "
    "model's response settles on.\n\n"
)

MCA_RULES = (
    "Rules:\n"
    "1. Count only an explicit final choice (e.g. 'The answer is B', or a lone letter on the last line).\n"
    "2. Mentions of options while reasoning do not count unless the response converges on one.\n"
    "3. If the tail is mostly 'Image N:' or 'Frame-N' enumeration, or the response was cut off "
    "without a final option, reply exactly NONE.\n"
    "4. Reply with one option letter from the listed choices, or exactly NONE. No other text.\n"
)

NA_HEADER = (
    "You are a grading assistant. Do not solve the question yourself and do not "
    "judge whether the response is correct. Report only the single final numeric "
    "answer the model's response settles on.\n\n"
)

NA_RULES = (
    "Rules:\n"
    "1. There must be a final-answer reading (e.g. 'the answer is 173', 'approximately 2.8 meters').\n"
    "2. Ranges or multiple candidates ('150-160 cm', 'maybe 180 or 200') are not a final answer -> NONE.\n"
    "3. Numbers in 'Image 178:' or 'Frame-12' labels are frame indices, not measurements -> ignore them.\n"
    "4. If truncated into frame-by-frame listing with no settled number, reply exactly NONE.\n"
    "5. Reply with one number in digits (e.g. 173 or 2.8) or exactly NONE. No units or other text.\n"
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trim_response(text: str, head: int, tail: int) -> str:
    text = text or ""
    if len(text) <= head + tail + 64:
        return text
    return text[:head] + "\n...[middle omitted]...\n" + text[-tail:]


def build_mca_prompt(question: str, options: list[str], response: str) -> str:
    listed = "\n".join(str(o) for o in options)
    return (
        f"{MCA_HEADER}"
        f"Question: {question}\n"
        f"Options:\n{listed}\n\n"
        f"Model response:\n{response}\n\n"
        f"{MCA_RULES}"
    )


def build_na_prompt(question: str, response: str) -> str:
    return (
        f"{NA_HEADER}"
        f"Question: {question}\n\n"
        f"Model response:\n{response}\n\n"
        f"{NA_RULES}"
    )


def parse_mca_reply(reply: str, options: list[str] | None, scoring) -> str:
    text = (reply or "").strip()
    if not text or text.upper() == "NONE":
        return ""
    letter = scoring.extract_vsibench_option(text, options)
    if letter:
        return letter.upper()
    match = re.search(r"\b([A-F])\b", text.upper())
    return match.group(1) if match else ""


def parse_na_reply(reply: str, scoring) -> float | None:
    text = (reply or "").strip()
    if not text or text.upper() == "NONE":
        return None
    return scoring.extract_vsibench_number(text)


def calibration_cases() -> list[dict]:
    enum_tail = "\n".join(f"* Image {i}: Kitchen." for i in range(170, 179))
    return [
        {
            "id": "cal_mca_clear",
            "kind": "mca",
            "question": "If I stand by the sofa and face the tv, is the door on my left or right?",
            "options": ["A. right", "B. left"],
            "ground_truth": "B",
            "response": "After checking the layout, the door is on my left.\nThe answer is B.",
            "expect": "B",
        },
        {
            "id": "cal_mca_wrong",
            "kind": "mca",
            "question": "If I stand by the sofa and face the tv, is the door on my left or right?",
            "options": ["A. right", "B. left"],
            "ground_truth": "B",
            "response": "The answer is A.",
            "expect": "A",
        },
        {
            "id": "cal_mca_enum_tail",
            "kind": "mca",
            "question": "If I stand by the sofa and face the tv, is the door on my left or right?",
            "options": ["A. right", "B. left"],
            "ground_truth": "B",
            "response": "I considered both sides...\nThe answer is B.\n" + enum_tail,
            "expect": "B",
        },
        {
            "id": "cal_mca_no_final",
            "kind": "mca",
            "question": "If I stand by the sofa and face the tv, is the door on my left or right?",
            "options": ["A. right", "B. left"],
            "ground_truth": "B",
            "response": "I looked at many frames but cannot tell left from right.\n" + enum_tail,
            "expect": "NONE",
        },
        {
            "id": "cal_na_clear",
            "kind": "na",
            "question": "What is the distance in meters?",
            "ground_truth": "2.8",
            "response": "The wall is 9 meters wide. Therefore the distance is 2.8 meters.",
            "expect": "2.8",
        },
        {
            "id": "cal_na_wrong",
            "kind": "na",
            "question": "What is the distance in meters?",
            "ground_truth": "2.8",
            "response": "Therefore the distance is 3.5 meters.",
            "expect": "3.5",
        },
        {
            "id": "cal_na_range_only",
            "kind": "na",
            "question": "What is the length of the sofa in cm?",
            "ground_truth": "173",
            "response": "A standard 2-seater is about 150-160 cm long.\n" + enum_tail,
            "expect": "NONE",
        },
        {
            "id": "cal_na_enum_only",
            "kind": "na",
            "question": "What is the length of the sofa in cm?",
            "ground_truth": "173",
            "response": enum_tail,
            "expect": "NONE",
        },
        {
            "id": "cal_na_frame_not_answer",
            "kind": "na",
            "question": "How many pipes are in the room?",
            "ground_truth": "2",
            "response": "Pipes appear in Frame-18 and Frame-23. Therefore there are two pipes.",
            "expect": "2",
        },
    ]


def load_samples(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_spot_check(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("id") == 168:
            by_bucket["id168"].append(row)
        if row.get("question_type") == "object_size_estimation" and row.get("truncated"):
            by_bucket["size_trunc"].append(row)
        if row.get("truncated") and row.get("final_score", 0) >= 0.999:
            by_bucket["trunc_full"].append(row)
        if row.get("truncated") and not row.get("final_answered"):
            by_bucket["trunc_unanswered"].append(row)
        if row.get("failure_reason") == "correct":
            by_bucket["correct"].append(row)

    picked: list[dict] = []
    seen: set = set()

    def add(row: dict) -> None:
        key = row["id"]
        if key in seen:
            return
        seen.add(key)
        picked.append(row)

    for row in by_bucket["id168"]:
        add(row)

    quotas = [
        ("size_trunc", 15),
        ("trunc_full", 10),
        ("trunc_unanswered", 10),
        ("correct", 10),
    ]
    for bucket, n in quotas:
        pool = by_bucket[bucket]
        if not pool:
            continue
        for row in rng.sample(pool, min(n, len(pool))):
            add(row)

    remaining = [r for r in rows if r["id"] not in seen]
    rng.shuffle(remaining)
    for row in remaining:
        if len(picked) >= 50:
            break
        add(row)
    return picked[:50]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="samples.jsonl from offline eval")
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="judge")
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--audit-size", type=int, default=200)
    parser.add_argument("--audit-seed", type=int, default=42)
    parser.add_argument("--spot-size", type=int, default=50)
    parser.add_argument("--spot-seed", type=int, default=7)
    parser.add_argument("--response-head-chars", type=int, default=800)
    parser.add_argument("--response-tail-chars", type=int, default=2500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="spatialstack")
    args = parser.parse_args()

    os.environ["VSIBENCH_PROTOCOL"] = args.protocol
    task = load_module(TASK_UTILS, "vsibench_task_utils_tri")
    upstream = load_module(VISIONOPD_JUDGE, "visionopd_judge_tri")
    scoring = load_module(SCORING, "vsibench_scoring_tri")

    samples = load_samples(args.samples)

    # --- build all prompts ---------------------------------------------------
    jobs: list[dict] = []

    for case in calibration_cases():
        response = case["response"]
        if case["kind"] == "mca":
            prompt = build_mca_prompt(case["question"], case["options"], response)
        else:
            prompt = build_na_prompt(case["question"], response)
        jobs.append({"phase": "calibration", "case": case, "prompt": prompt})

    rng = random.Random(args.audit_seed)
    audit_pool = [
        r
        for r in samples
        if r["question_type"] in task.MCA_QUESTION_TYPES and float(r.get("rule_score", 0)) >= 0.999
    ]
    audit_rows = rng.sample(audit_pool, min(args.audit_size, len(audit_pool)))
    for row in audit_rows:
        shown = trim_response(row["response"], args.response_head_chars, args.response_tail_chars)
        if row["question_type"] in task.MCA_QUESTION_TYPES:
            prompt = build_mca_prompt(row["question"], row.get("options") or [], shown)
            kind = "mca"
        else:
            prompt = build_na_prompt(row["question"], shown)
            kind = "na"
        jobs.append({"phase": "reverse_audit", "row": row, "kind": kind, "prompt": prompt})

    spot_rows = pick_spot_check(samples, args.spot_seed)[: args.spot_size]
    for row in spot_rows:
        shown = trim_response(row["response"], args.response_head_chars, args.response_tail_chars)
        if row["question_type"] in task.MCA_QUESTION_TYPES:
            prompt = build_mca_prompt(row["question"], row.get("options") or [], shown)
            kind = "mca"
        else:
            prompt = build_na_prompt(row["question"], shown)
            kind = "na"
        jobs.append({"phase": "spot_check", "row": row, "kind": kind, "prompt": prompt})

    print(
        f"tri-insurance: calibration={sum(1 for j in jobs if j['phase']=='calibration')} "
        f"audit={len(audit_rows)} spot={len(spot_rows)} total_calls={len(jobs)}",
        flush=True,
    )

    replies = upstream.judge_via_api(
        [j["prompt"] for j in jobs],
        Namespace(
            judge_api_key=args.judge_api_key,
            judge_api_base=args.judge_api_base,
            judge_model=args.judge_model,
            judge_max_tokens=args.judge_max_tokens,
            reasoning_effort=args.reasoning_effort,
            parallel_workers=args.parallel_workers,
        ),
    )

    # --- grade calibration ---------------------------------------------------
    cal_results = []
    cal_pass = cal_fail = cal_api = 0
    for job, (content, reasoning) in zip(jobs, replies):
        if job["phase"] != "calibration":
            continue
        case = job["case"]
        reply = content or reasoning
        if reasoning == "[JUDGE_API_ERROR]":
            cal_api += 1
            cal_results.append({**case, "reply": reply, "parsed": "", "ok": False, "reason": "api_error"})
            continue
        if case["kind"] == "mca":
            parsed = parse_mca_reply(reply, case.get("options"), scoring) or "NONE"
        else:
            num = parse_na_reply(reply, scoring)
            parsed = "NONE" if num is None else str(num)
        expect = case["expect"]
        if case["kind"] == "na" and expect != "NONE":
            try:
                ok = num is not None and abs(float(parsed) - float(expect)) < 1e-6
            except (TypeError, ValueError):
                ok = False
        else:
            ok = parsed == expect
        if ok:
            cal_pass += 1
        else:
            cal_fail += 1
        cal_results.append(
            {**case, "reply": (reply or "")[:200], "parsed": parsed, "ok": ok, "reason": "" if ok else "mismatch"}
        )

    # --- reverse audit -------------------------------------------------------
    audit_results = []
    audit_agree = audit_disagree = audit_none = audit_api = 0
    for job, (content, reasoning) in zip(jobs, replies):
        if job["phase"] != "reverse_audit":
            continue
        row = job["row"]
        reply = content or reasoning
        if reasoning == "[JUDGE_API_ERROR]":
            audit_api += 1
            continue
        rule_parsed = str(row.get("rule_parsed", "")).strip().upper()
        if job["kind"] == "mca":
            judge_parsed = parse_mca_reply(reply, row.get("options"), scoring)
            if not judge_parsed:
                audit_none += 1
                agree = False
            else:
                agree = judge_parsed.upper() == rule_parsed
        else:
            num = parse_na_reply(reply, scoring)
            judge_parsed = "" if num is None else str(num)
            if num is None:
                audit_none += 1
                agree = False
            else:
                agree = judge_parsed == rule_parsed
        if agree:
            audit_agree += 1
        else:
            audit_disagree += 1
            audit_results.append(
                {
                    "id": row["id"],
                    "question_type": row["question_type"],
                    "ground_truth": row["ground_truth"],
                    "rule_parsed": row.get("rule_parsed"),
                    "judge_parsed": judge_parsed or "NONE",
                    "truncated": row.get("truncated"),
                    "reply": (reply or "")[:200],
                }
            )

    # --- spot check summary --------------------------------------------------
    spot_results = []
    spot_by_tag = Counter()
    for job, (content, reasoning) in zip(jobs, replies):
        if job["phase"] != "spot_check":
            continue
        row = job["row"]
        reply = content or reasoning
        qtype = row["question_type"]
        if job["kind"] == "mca":
            parsed = parse_mca_reply(reply, row.get("options"), scoring) or "NONE"
            gt = str(row["ground_truth"]).strip().upper()
            score = 1.0 if parsed == gt else 0.0
        else:
            num = parse_na_reply(reply, scoring)
            parsed = "NONE" if num is None else str(num)
            if num is None:
                score = 0.0
            else:
                score = task.mean_relative_accuracy(
                    num, task.to_float(row["ground_truth"]), start=0.5, end=0.95, interval=0.05
                )
        rule_score = float(row.get("rule_score", 0))
        tag = "unchanged"
        if parsed == "NONE" and row.get("rule_answered"):
            tag = "judge_none_rule_answered"
        elif parsed != "NONE" and not row.get("rule_answered"):
            tag = "judge_found_rule_unanswered"
        elif abs(score - rule_score) > 1e-6:
            tag = "score_changed"
        spot_by_tag[tag] += 1
        spot_results.append(
            {
                "id": row["id"],
                "question_type": qtype,
                "ground_truth": row["ground_truth"],
                "truncated": int(row.get("truncated", 0)),
                "rule_parsed": row.get("rule_parsed"),
                "rule_score": rule_score,
                "judge_parsed": parsed,
                "judge_score": score,
                "failure_reason": row.get("failure_reason"),
                "tag": tag,
                "reply": (reply or "")[:160],
            }
        )

    report = {
        "samples": os.path.abspath(args.samples),
        "judge_model": args.judge_model,
        "prompt": "strict_extract_v1",
        "calibration": {
            "total": len(cal_results),
            "pass": cal_pass,
            "fail": cal_fail,
            "api_errors": cal_api,
            "pass_rate": cal_pass / max(len(cal_results) - cal_api, 1),
            "cases": cal_results,
            "ready": cal_fail == 0 and cal_api == 0,
        },
        "reverse_audit": {
            "pool_rule_correct_mca": len(audit_pool),
            "sampled": len(audit_rows),
            "agree": audit_agree,
            "disagree": audit_disagree,
            "judge_none": audit_none,
            "api_errors": audit_api,
            "disagree_rate": audit_disagree / max(len(audit_rows) - audit_api, 1),
            "disagreements": audit_results[:30],
            "ready": audit_disagree / max(len(audit_rows) - audit_api, 1) <= 0.05,
        },
        "spot_check": {
            "sampled": len(spot_results),
            "tags": dict(spot_by_tag),
            "rows": spot_results,
        },
        "overall_ready": False,
    }
    report["overall_ready"] = report["calibration"]["ready"] and report["reverse_audit"]["ready"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n=== calibration ===")
    print(f"  pass {cal_pass}/{len(cal_results)}  fail {cal_fail}  api_errors {cal_api}")
    for row in cal_results:
        if not row["ok"]:
            print(f"  FAIL {row['id']}: expect={row['expect']} got={row['parsed']} reply={row['reply']!r}")

    print("\n=== reverse audit (rule-correct MCA) ===")
    denom = max(len(audit_rows) - audit_api, 1)
    print(
        f"  agree {audit_agree}/{denom}  disagree {audit_disagree} ({100*audit_disagree/denom:.1f}%)  "
        f"judge_none {audit_none}  api_errors {audit_api}"
    )
    for row in audit_results[:8]:
        print(
            f"  id={row['id']} rule={row['rule_parsed']} judge={row['judge_parsed']} "
            f"gt={row['ground_truth']} trunc={row['truncated']}"
        )

    print("\n=== spot check (50 rows) ===")
    print(f"  tags: {dict(spot_by_tag)}")
    for row in spot_results:
        if row["id"] == 168 or row["tag"] != "unchanged":
            print(
                f"  id={row['id']} {row['question_type']} rule={row['rule_parsed']}({row['rule_score']}) "
                f"judge={row['judge_parsed']}({row['judge_score']:.2f}) tag={row['tag']}"
            )

    print(f"\noverall_ready={report['overall_ready']}  saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
