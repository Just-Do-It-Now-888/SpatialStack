#!/usr/bin/env python3
"""Grade generated answers the Vision-OPD way: rules first, LLM judge for the rest.

Port of ``Vision-OPD-main/eval/judge_qwenlm.py``. **The defaults reproduce
upstream**, so numbers from this script are comparable with Vision-OPD's own:

* ``--letter-mode faithful``  -- upstream's ``extract_first_option``, ending in
  ``re.search(r"([A-Z])", text)``;
* ``--max-response-chars 0``  -- the whole response goes to the judge, so the
  judge server must be deployed with a context window that fits it. Upstream
  does not start the judge at all, it takes a ``JUDGE_API_BASE``, so the window
  is a deployment decision; ``run_eval.sh`` sizes it from ``MAX_TOKENS``;
* ``--verdict-mode faithful`` -- the judge's reply is stored verbatim, and
  ``cal_acc.py`` counts a row correct only on an exact ``"yes"``, as upstream
  does;
* ``--reasoning-effort ''``   -- no gpt-oss knob is sent.

Two escape hatches exist because MV-OPSD v0 was scored for a day on a string
artifact (ISSUE-003, LESSON-011), and neither is on by default:

* ``--letter-mode strict`` requires the option letter to stand alone, rejecting
  the "B" of "**B**ased on the provided images". Against CV-Bench's 41.8%-B gold
  distribution that spurious match is worth about 40 free points, though here it
  can only ever promote a row to correct -- everything else falls through to the
  LLM judge, which reads the response;
* ``--verdict-mode normalized`` accepts "Yes, the response is correct." and
  falls back to ``reasoning_content`` when a reasoning judge leaves ``content``
  empty, flagging anything still unreadable as ``UNPARSED`` instead of silently
  scoring it wrong.

Both modes are always evaluated and the disagreement counts are printed, so the
size of each deviation stays visible without changing the reported score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

MCQ_BENCHMARKS = {"cvbench", "cv-bench", "blink-spatial", "mmstar", "vstar"}

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)

# Anchored at the start, so it reads the gold "(C)" / "C." form only.
GT_OPTION = re.compile(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))")
PAREN_OPTION = re.compile(r"\(([A-Z])\)")
PUNCT_OPTION = re.compile(r"([A-Z])[\.\)\s]")
BARE_OPTION = re.compile(r"([A-Z])")
# Standalone A-F: not glued to other letters. Rejects the "B" of "Based".
STANDALONE_OPTION = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")


def extract_gt_option(gt: str) -> str:
    if not isinstance(gt, str) or not gt:
        return ""
    match = GT_OPTION.match(gt.strip())
    return match.group(1) if match else ""


def extract_pred_option(text: str, mode: str) -> str:
    """Upstream's extract_first_option, with an optional word-boundary tier."""
    if not text:
        return ""
    match = PAREN_OPTION.search(text)
    if match:
        return match.group(1)
    match = PUNCT_OPTION.search(text)
    if match:
        return match.group(1)
    if mode == "strict":
        match = STANDALONE_OPTION.search(text)
        return match.group(1) if match else ""
    match = BARE_OPTION.search(text)
    return match.group(1) if match else ""


def extract_answer(raw: str) -> str:
    """Narrow a full response down to its answer-bearing part, as upstream does."""
    if not isinstance(raw, str):
        return ""
    # Not upstream's, but a no-op unless the server inlines a chain of thought,
    # which only happens in thinking mode. Evaluation runs non-thinking.
    think_end = raw.rfind("</think>")
    if think_end != -1:
        raw = raw[think_end + len("</think>") :].strip()
    start = raw.find("<answer>")
    if start != -1:
        end = raw.find("</answer>", start)
        if end != -1:
            return raw[start + len("<answer>") : end].strip()
    if "Answer:" in raw:
        return raw[raw.find("Answer:") :].strip()
    return raw.strip()


def normalize_verdict(text: str) -> str:
    """Map a judge reply onto Yes / No, flagging anything unreadable."""
    if not isinstance(text, str) or not text.strip():
        return "UNPARSED"
    head = text.strip().lstrip("*# ").lower()
    if head.startswith("yes"):
        return "Yes"
    if head.startswith("no"):
        return "No"
    tail = head[-16:]
    if "yes" in tail and "no" not in tail:
        return "Yes"
    if "no" in tail and "yes" not in tail:
        return "No"
    return "UNPARSED"


def judge_via_api(prompts, args):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_api_base, timeout=1800)
            thread_local.client = client
        return client

    extra = {}
    if args.reasoning_effort:
        extra["extra_body"] = {"reasoning_effort": args.reasoning_effort}

    # (content, reasoning_content) per prompt. Upstream keeps only the first;
    # the second is recorded so an empty content is diagnosable rather than
    # indistinguishable from a judge that genuinely said nothing.
    results = [("", "")] * len(prompts)

    def call_one(idx: int, prompt: str):
        for attempt in range(3):
            try:
                resp = get_client().chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=args.judge_max_tokens,
                    **extra,
                )
                message = resp.choices[0].message
                content = (message.content or "").strip()
                reasoning = (getattr(message, "reasoning_content", None) or "").strip()
                return idx, (content, reasoning)
            except Exception:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(1.0)
        # Upstream's failure verdict. It scores the row wrong, which is why the
        # caller flags it: an unreachable judge must not look like a bad answer.
        return idx, ("No", "[JUDGE_API_ERROR]")

    with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
        futures = [pool.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="llm judge"):
            idx, pair = future.result()
            results[idx] = pair
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--answer-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--letter-mode",
        choices=["strict", "faithful"],
        default="faithful",
        help="faithful reproduces upstream's extract_first_option; strict requires "
        "the option letter to stand alone, rejecting the 'B' of 'Based'",
    )
    parser.add_argument(
        "--verdict-mode",
        choices=["faithful", "normalized"],
        default="faithful",
        help="faithful stores the judge reply verbatim, as upstream does; "
        "normalized maps prefixed replies onto Yes/No, reads reasoning_content "
        "when content is empty, and marks the rest UNPARSED",
    )
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=0,
        help="0 sends the whole response, as upstream does; the judge server must "
        "then have a context window that fits it. A positive value keeps only "
        "that much of the tail, where a conclusion would be -- needed only when "
        "the judge window is smaller than the responses being graded.",
    )
    parser.add_argument("--reasoning-effort", default="", help="gpt-oss knob; empty to omit")
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="rules only; unresolved rows count as wrong and are reported separately",
    )
    args = parser.parse_args()

    rows = []
    with open(args.answer_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    try:
        from mathruler.grader import grade_answer

        has_mathruler = True
    except ImportError:
        grade_answer = None
        has_mathruler = False

    is_mcq = args.benchmark.lower() in MCQ_BENCHMARKS
    pending, prompts = [], []
    letter_disagreements = 0

    for i, item in enumerate(tqdm(rows, desc="rule grading")):
        raw = item.get("model_answer") or ""
        extracted = extract_answer(raw)
        gt = item.get("response", "")
        item["extracted_answer"] = extracted

        correct = False
        if has_mathruler and extracted:
            try:
                correct = bool(grade_answer(gt, extracted))
            except Exception:  # noqa: BLE001 - grader is third-party and input is free text
                correct = False

        letter_ok = False
        if not correct and is_mcq:
            gt_option = extract_gt_option(gt)
            pred = extract_pred_option(extracted, args.letter_mode)
            other = extract_pred_option(extracted, "faithful" if args.letter_mode == "strict" else "strict")
            letter_ok = bool(gt_option and pred and gt_option == pred)
            if bool(gt_option and other and gt_option == other) != letter_ok:
                letter_disagreements += 1
                item["letter_mode_disagreement"] = True
            item["pred_option"] = pred

        if correct:
            item["judge"] = "Yes"
            item["judge_source"] = "mathruler"
        elif letter_ok:
            item["judge"] = "Yes"
            item["judge_source"] = "letter"
        else:
            question = str(item.get("query", "")).replace("<image>", "")
            shown = extracted
            if args.max_response_chars and len(shown) > args.max_response_chars:
                shown = "...[truncated]...\n" + shown[-args.max_response_chars :]
                item["judge_saw_truncated_response"] = True
            pending.append(i)
            prompts.append(PROMPT_TEMPLATE.format(question=question, gt=gt, response=shown))

    if prompts and not args.no_llm_judge:
        if not args.judge_api_base or not args.judge_model:
            raise SystemExit(
                "--judge-api-base and --judge-model are required unless --no-llm-judge is set"
            )
        print(f"llm judge on {len(prompts)} unresolved rows")
        replies = judge_via_api(prompts, args)
        for offset, (content, reasoning) in enumerate(replies):
            item = rows[pending[offset]]
            item["judge_raw"] = content
            item["judge_reasoning"] = reasoning
            normalized = normalize_verdict(content or reasoning)
            item["judge_normalized"] = normalized
            item["judge"] = content if args.verdict_mode == "faithful" else normalized
            item["judge_source"] = "llm"
            if reasoning == "[JUDGE_API_ERROR]":
                item["judge_api_error"] = True
    else:
        for i in pending:
            rows[i]["judge"] = "No"
            rows[i]["judge_source"] = "rule_unresolved"

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    sources = Counter(item.get("judge_source", "?") for item in rows)
    scored_yes = sum(1 for item in rows if str(item.get("judge", "")).strip().lower() == "yes")
    print(f"\nsaved {len(rows)} rows -> {args.out_json}")
    print(f"  verdict by tier: {dict(sources)}")
    print(f"  scored correct:  {scored_yes}/{len(rows)}")

    # Everything below is a deviation counter, not a verdict. The score above is
    # produced by the upstream defaults; these say what the two hardenings would
    # have changed, so their size is visible without being silently applied.
    print(f"  letter-mode={args.letter_mode}: {letter_disagreements} rows would flip "
          "under the other mode")

    llm_rows = [item for item in rows if item.get("judge_source") == "llm"]
    if llm_rows:
        empty_content = sum(1 for item in llm_rows if not str(item.get("judge_raw", "")).strip())
        recovered = sum(
            1
            for item in llm_rows
            if not str(item.get("judge_raw", "")).strip()
            and str(item.get("judge_reasoning", "")).strip()
        )
        api_errors = sum(1 for item in llm_rows if item.get("judge_api_error"))
        verdict_flips = sum(
            1
            for item in llm_rows
            if (str(item.get("judge_raw", "")).strip().lower() == "yes")
            != (item.get("judge_normalized") == "Yes")
        )
        unparsed = sum(1 for item in llm_rows if item.get("judge_normalized") == "UNPARSED")
        print(f"  verdict-mode={args.verdict_mode}: {verdict_flips} rows would flip under the other mode")
        print(f"  judge replies with empty content: {empty_content} "
              f"(of which {recovered} had reasoning_content)")
        print(f"  judge replies still unreadable either way: {unparsed}")
        if api_errors:
            print(f"  WARNING: {api_errors} judge calls failed and score as wrong (upstream behaviour)")
        if empty_content:
            print(
                "  WARNING: an empty judge reply scores the row wrong. If the count is "
                "large the judge context is probably too small for these responses."
            )


if __name__ == "__main__":
    main()
