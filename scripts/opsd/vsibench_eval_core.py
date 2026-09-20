"""VSI-Bench evaluation core: vLLM generation, rule scoring, optional judge, diagnostics.

Used by ``tools/eval_vsibench_vllm.py`` and the training-time checkpoint eval
(``run_training_vsibench_eval.sh``).  One module so generation, judge cascade,
and failure tagging cannot drift between offline and training-time runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

TASK_UTILS = os.path.join(REPO_ROOT, "src", "lmms_eval", "tasks", "vsibench", "utils.py")
VISIONOPD_JUDGE = os.path.join(REPO_ROOT, "scripts", "opsd", "eval_visionopd", "judge.py")
SCORING = os.path.join(REPO_ROOT, "scripts", "opsd", "vsibench_scoring.py")
DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/"
    "bdcadb3fea447621a828a24911801faba3587c12"
)
DEFAULT_VIDEO_ROOT = os.path.expanduser("~/.cache/huggingface/vsibench")

LMMS_KWARGS = {
    "pre_prompt": "",
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "na_post_prompt": "Please answer the question using a single word or phrase.",
}

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
    import re
    import string

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


@dataclass
class EvalConfig:
    model: str
    frames: int = 32
    max_tokens: int = 4096
    max_model_len: int = 16384
    max_pixels: int = 1605632
    min_pixels: int = 256 * 28 * 28
    tensor_parallel_size: int = 8
    gpu_memory_utilization: float = 0.9
    scenes_per_batch: int = 8
    snapshot: str = DEFAULT_SNAPSHOT
    video_root: str = DEFAULT_VIDEO_ROOT
    limit_scenes: int = 0
    protocol: str = "spatialstack"
    # Appended verbatim to the end of every question, after the lmms_eval post
    # prompt. Empty by default so the rendered prompt stays byte-identical to
    # every number already published from this harness (LESSON-013); the boxed
    # probe passes r" The final answer MUST BE put in \boxed{}." here.
    prompt_suffix: str = ""
    # Match in-training ``vsibench_boxed_primary``: a closed ``\\boxed{}`` is
    # the only answer site; otherwise the last-line / "answer is" parser runs
    # on the full response. Off by default so published offline numbers stay
    # full-text rule scores (LESSON-013).
    boxed_primary: bool = False
    # Decoding. Greedy by default, which is what every VSI-Bench number this
    # harness has published was measured under (LESSON-013): flipping the
    # default would silently move the base anchor 52.58 and every arm's curve.
    #
    # Sampling exists because greedy is a known cause of the repetition loops
    # this project keeps hitting -- 90.62% of step-100 responses ran to the
    # 4096-token cap enumerating `Image N:` (LESSON-020). Upstream OPSD's own
    # eval script warns against greedy for exactly this reason and evaluates at
    # temperature 1.0, averaging over 12 samples per question
    # (siyan-zhao/OPSD eval/evaluate_math.py, eval/run_eval.sh).
    #
    # A single sampled pass is not the same estimator as OPSD's Avg@12: it has
    # per-question variance that greedy does not, so one sampled run is
    # evidence about the failure mode, not a benchmark number to publish next
    # to the greedy curve.
    do_sample: bool = False
    # None means "take the default for the chosen mode": greedy uses
    # temperature 0 / top_p 1, sampling uses Qwen3 non-thinking guidance
    # (temperature 1.0, top_p 0.8), matching the student's enable_thinking=False
    # rendering used everywhere in this harness.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    # vLLM seeds per request; set it so a sampled run can be replayed.
    seed: int | None = None
    # Judge: Vision-OPD verdict cascade on rule-rejected rows only.
    judge_api_base: str | None = None
    judge_api_key: str = "EMPTY"
    judge_model: str = "judge"
    judge_max_tokens: int = 2048
    judge_parallel_workers: int = 256
    judge_reasoning_effort: str = ""
    judge_mca_scope: str = "unreadable"


def resolve_decoding(cfg: EvalConfig) -> dict:
    """The decoding knobs actually sent to vLLM, with the mode defaults filled in.

    Rejects a sampling knob set while ``do_sample`` is off instead of ignoring
    it: half a protocol scores a model nobody configured, which is the failure
    the VSIBENCH_PROTOCOL switch was introduced to prevent.
    """
    if not cfg.do_sample:
        conflicts = [
            name
            for name, value in (
                ("temperature", cfg.temperature),
                ("top_p", cfg.top_p),
                ("top_k", cfg.top_k if cfg.top_k != -1 else None),
                ("min_p", cfg.min_p or None),
                ("presence_penalty", cfg.presence_penalty or None),
            )
            if value is not None
        ]
        if conflicts:
            raise ValueError(
                f"do_sample is off but {', '.join(conflicts)} was set; "
                "pass --do-sample to sample, or drop these to stay greedy"
            )
        return {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "seed": cfg.seed,
        }
    return {
        "do_sample": True,
        "temperature": 1.0 if cfg.temperature is None else cfg.temperature,
        "top_p": 0.8 if cfg.top_p is None else cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "presence_penalty": cfg.presence_penalty,
        "seed": cfg.seed,
    }


@dataclass
class SampleRecord:
    id: Any
    dataset: str
    scene_name: str
    question_type: str
    question: str
    ground_truth: str
    options: list | None
    response: str
    output_tokens: int
    truncated: bool
    finish_reason: str
    # Rule tier
    rule_parsed: str
    rule_answered: bool
    rule_score: float
    # Final tier (after judge if applicable)
    final_parsed: str
    final_answered: bool
    final_score: float
    # Diagnostics
    failure_reason: str
    judge_used: bool = False
    judge_verdict: str = ""
    judge_source: str = "rule"
    # \boxed{} compliance, reported alongside the score and never folded into
    # it. ``boxed_content`` is None when the response contains no closed box.
    # ``boxed_score`` is what this row would earn if only the box counted, so a
    # non-complying row scores 0 there while keeping its rule score.
    boxed_content: str | None = None
    boxed_answered: bool = False
    boxed_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "scene_name": self.scene_name,
            "question_type": self.question_type,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "options": self.options,
            "response": self.response,
            "output_tokens": self.output_tokens,
            "truncated": int(self.truncated),
            "finish_reason": self.finish_reason,
            "rule_parsed": self.rule_parsed,
            "rule_answered": int(self.rule_answered),
            "rule_score": self.rule_score,
            "final_parsed": self.final_parsed,
            "final_answered": int(self.final_answered),
            "final_score": self.final_score,
            "failure_reason": self.failure_reason,
            "judge_used": self.judge_used,
            "judge_verdict": self.judge_verdict,
            "judge_source": self.judge_source,
            "boxed_present": int(self.boxed_content is not None),
            "boxed_content": self.boxed_content,
            "boxed_answered": int(self.boxed_answered),
            "boxed_score": self.boxed_score,
        }


def load_docs(snapshot: str) -> list[dict]:
    path = os.path.join(snapshot, "test.jsonl")
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def video_path_for(doc: dict, video_root: str) -> str:
    return os.path.join(video_root, doc["dataset"], doc["scene_name"] + ".mp4")


def sample_frames(video_path: str, max_num_frames: int):
    import decord
    import numpy as np
    from PIL import Image

    reader = decord.VideoReader(video_path)
    frame_count = len(reader)
    if frame_count <= max_num_frames:
        indices = np.arange(frame_count)
    else:
        indices = np.linspace(0, frame_count - 1, max_num_frames).astype(int)
    return [Image.fromarray(reader[i].asnumpy()).convert("RGB") for i in indices]


def _rule_metric(scored: dict, question_type: str, task) -> tuple[float, bool, str]:
    if question_type in task.MCA_QUESTION_TYPES:
        return scored.get("accuracy", 0.0), bool(scored.get("answered")), str(scored.get("parsed_answer", ""))
    mra = scored.get("MRA:.5:.95:.05", 0.0)
    return mra, bool(scored.get("answered")), str(scored.get("parsed_answer", ""))


def _is_correct(score: float, question_type: str, task) -> bool:
    if question_type in task.MCA_QUESTION_TYPES:
        return score >= 1.0 - 1e-9
    return score > 0.0


FAILURE_TAGS = ("truncation_error", "parse_error", "partial_error", "factual_error", "judge_recovered")


def classify_failure(
    *,
    truncated: bool,
    final_answered: bool,
    score: float,
    judge_recovered: bool,
) -> str:
    """Why a row did not earn full credit.

    Order matters: a truncated answer is blamed on truncation even if a number
    could still be read out of the fragment, because the missing tail is the
    thing to fix.  ``partial_error`` only ever applies to the numerical types,
    where the metric is graded -- a multiple-choice score is 0 or 1, so a
    partially right option does not exist.
    """
    if score >= 1.0 - 1e-9:
        return "judge_recovered" if judge_recovered else "correct"
    if truncated:
        return "truncation_error"
    if not final_answered:
        return "parse_error"
    if score > 0.0:
        return "partial_error"
    return "factual_error"


MCA_JUDGE_SCOPES = ("unreadable", "rule_wrong")


def build_judge_requests(records: list[SampleRecord], task, upstream, mca_scope: str = "unreadable"):
    """Rows the rule tier could not settle, as judge prompts.

    ``mca_scope`` decides what "could not settle" means for multiple choice:

    * ``unreadable`` (default) -- only rows where no option letter could be read
      at all. A letter the parser read and scored wrong is a wrong answer, not
      an unresolved one. This is also what the in-training guardrail sends
      (``ray_trainer._judge_vsibench_rows``), so the two report the same number.
    * ``rule_wrong`` -- every row the rules scored wrong, which is Vision-OPD's
      cascade. It raised the base model by +1.5 while overturning ~10% of
      rule-accepted rows in a reverse audit, so it measures something looser
      than the benchmark does (LESSON-019). Kept for comparison runs.

    Numerical is always extraction-only, on rows no number could be read from:
    MRA is a mean over ten thresholds and a Yes/No verdict would flatten it into
    a binary accuracy still labelled MRA.

    Either way a rule-accepted row is never re-opened, so the judge can raise
    the score but never lower it.
    """
    if mca_scope not in MCA_JUDGE_SCOPES:
        raise ValueError(f"unknown mca_scope {mca_scope!r}; expected one of {MCA_JUDGE_SCOPES}")

    pending: list[int] = []
    kinds: list[str] = []
    prompts: list[str] = []
    for index, rec in enumerate(records):
        if rec.question_type in task.MCA_QUESTION_TYPES:
            settled = rec.rule_answered if mca_scope == "unreadable" else _is_correct(
                rec.rule_score, rec.question_type, task
            )
            if settled:
                continue
            doc = {
                "question_type": rec.question_type,
                "ground_truth": rec.ground_truth,
                "options": rec.options,
            }
            pending.append(index)
            kinds.append("mca")
            prompts.append(
                upstream.PROMPT_TEMPLATE.format(
                    question=rec.question, gt=gold_answer(doc), response=rec.response
                )
            )
        else:
            if rec.rule_answered:
                continue
            pending.append(index)
            kinds.append("na")
            prompts.append(NA_EXTRACT_TEMPLATE.format(question=rec.question, response=rec.response))
    return pending, kinds, prompts


def record_from_dict(row: dict) -> SampleRecord:
    return SampleRecord(
        id=row.get("id"),
        dataset=row.get("dataset", ""),
        scene_name=row.get("scene_name", ""),
        question_type=row["question_type"],
        question=row.get("question", ""),
        ground_truth=str(row.get("ground_truth", "")),
        options=row.get("options"),
        response=row.get("response", ""),
        output_tokens=int(row.get("output_tokens", 0)),
        truncated=bool(row.get("truncated", 0)),
        finish_reason=row.get("finish_reason", ""),
        rule_parsed=row.get("rule_parsed", ""),
        rule_answered=bool(row.get("rule_answered", 0)),
        rule_score=float(row.get("rule_score", 0.0)),
        final_parsed=row.get("final_parsed", row.get("rule_parsed", "")),
        final_answered=bool(row.get("final_answered", row.get("rule_answered", 0))),
        final_score=float(row.get("final_score", row.get("rule_score", 0.0))),
        failure_reason=row.get("failure_reason", ""),
        judge_used=bool(row.get("judge_used", False)),
        judge_verdict=row.get("judge_verdict", ""),
        judge_source=row.get("judge_source", "rule"),
    )


def apply_judge(
    records: list[SampleRecord],
    pending: list[int],
    kinds: list[str],
    prompts: list[str],
    cfg: EvalConfig,
    task,
    upstream,
    scoring,
) -> dict:
    if not prompts or not cfg.judge_api_base:
        return {"judged": 0, "recovered": 0, "api_errors": 0}

    replies = upstream.judge_via_api(
        prompts,
        Namespace(
            judge_api_key=cfg.judge_api_key,
            judge_api_base=cfg.judge_api_base,
            judge_model=cfg.judge_model,
            judge_max_tokens=cfg.judge_max_tokens,
            reasoning_effort=cfg.judge_reasoning_effort,
            parallel_workers=cfg.judge_parallel_workers,
        ),
    )

    stats = {"judged": len(pending), "recovered": 0, "api_errors": 0, "unparsed": 0}
    for offset, (content, reasoning) in enumerate(replies):
        index = pending[offset]
        rec = records[index]
        doc = {
            "question_type": rec.question_type,
            "ground_truth": rec.ground_truth,
            "options": rec.options,
        }
        reply = content or reasoning
        if reasoning == "[JUDGE_API_ERROR]":
            stats["api_errors"] += 1
            continue

        rec.judge_used = True
        if kinds[offset] == "mca":
            verdict = upstream.normalize_verdict(reply)
            rec.judge_verdict = verdict
            if verdict == "UNPARSED":
                stats["unparsed"] += 1
                rec.judge_source = "judge_unparsed"
                continue
            new_score = 1.0 if verdict == "Yes" else 0.0
            rec.judge_source = "judge_verdict"
        else:
            number = None
            if reply and reply.strip().upper() != "NONE":
                number = scoring.extract_vsibench_number(reply)
            if number is None:
                stats["unparsed"] += 1
                rec.judge_verdict = "NONE"
                rec.judge_source = "judge_no_number"
                continue
            target = task.to_float(rec.ground_truth)
            new_score = task.mean_relative_accuracy(
                number, target, start=0.5, end=0.95, interval=0.05
            )
            rec.final_parsed = str(number)
            rec.judge_verdict = str(number)
            rec.judge_source = "judge_extract"

        rule_correct = _is_correct(rec.rule_score, rec.question_type, task)
        new_correct = _is_correct(new_score, rec.question_type, task)
        if not rule_correct and new_correct:
            stats["recovered"] += 1

        rec.final_score = new_score
        rec.final_answered = True
        judge_recovered = not rule_correct and new_correct
        rec.failure_reason = classify_failure(
            truncated=rec.truncated,
            final_answered=True,
            score=new_score,
            judge_recovered=judge_recovered,
        )

    return stats


def run_eval(cfg: EvalConfig, *, progress: bool = True) -> tuple[list[SampleRecord], dict]:
    """Generate, score, optionally judge, return per-sample records and summary."""
    os.environ["VSIBENCH_PROTOCOL"] = cfg.protocol
    os.environ["VSIBENCH_MAX_NEW_TOKENS"] = str(cfg.max_tokens)

    task = load_module(TASK_UTILS, "vsibench_task_utils")
    upstream = load_module(VISIONOPD_JUDGE, "visionopd_judge")
    scoring = load_module(SCORING, "vsibench_scoring")

    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    docs = load_docs(cfg.snapshot)
    by_scene: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for doc in docs:
        by_scene[(doc["dataset"], doc["scene_name"])].append(doc)
    scenes = sorted(by_scene)
    if cfg.limit_scenes:
        scenes = scenes[: cfg.limit_scenes]

    processor = AutoProcessor.from_pretrained(
        cfg.model, max_pixels=cfg.max_pixels, min_pixels=cfg.min_pixels, trust_remote_code=True
    )
    llm = LLM(
        model=cfg.model,
        trust_remote_code=True,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        limit_mm_per_prompt={"image": cfg.frames},
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        mm_processor_kwargs={"max_pixels": cfg.max_pixels, "min_pixels": cfg.min_pixels},
    )
    decoding = resolve_decoding(cfg)
    sampling = SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=decoding["temperature"],
        top_p=decoding["top_p"],
        top_k=decoding["top_k"],
        min_p=decoding["min_p"],
        presence_penalty=decoding["presence_penalty"],
        seed=decoding["seed"],
    )
    if progress:
        print(f"decoding: {decoding}", flush=True)

    def build_batch(scene_keys):
        requests, metas = [], []
        for key in scene_keys:
            scene_docs = by_scene[key]
            path = video_path_for(scene_docs[0], cfg.video_root)
            try:
                frames = sample_frames(path, cfg.frames)
            except Exception as exc:
                print(f"  SKIP scene {key}: {exc}", file=sys.stderr)
                continue
            for doc in scene_docs:
                context = task.vsibench_doc_to_text_plain(doc, LMMS_KWARGS) + cfg.prompt_suffix
                content = [{"type": "image", "image": frame} for frame in frames]
                content.append({"type": "text", "text": context})
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                requests.append({"prompt": prompt, "multi_modal_data": {"image": frames}})
                metas.append(doc)
        return requests, metas

    records: list[SampleRecord] = []
    started = time.perf_counter()
    done = 0
    total = sum(len(by_scene[s]) for s in scenes)
    batches = [scenes[i : i + cfg.scenes_per_batch] for i in range(0, len(scenes), cfg.scenes_per_batch)]

    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        pending_batch = prefetcher.submit(build_batch, batches[0]) if batches else None
        for index, batch in enumerate(batches):
            requests, metas = pending_batch.result()
            pending_batch = (
                prefetcher.submit(build_batch, batches[index + 1]) if index + 1 < len(batches) else None
            )
            if not requests:
                continue
            outputs = llm.generate(requests, sampling_params=sampling)
            for output, doc in zip(outputs, metas):
                completion = output.outputs[0]
                text = completion.text
                n_tokens = len(completion.token_ids)
                truncated = completion.finish_reason == "length"
                scored = task.vsibench_process_results(dict(doc), [text])["vsibench_score"]
                qtype = doc["question_type"]
                rule_score, rule_answered, rule_parsed = _rule_metric(scored, qtype, task)
                # Score the box through the same upstream path, with the box
                # contents standing in for the whole response. Feeding "" for a
                # non-complying row is what makes it count as unanswered here
                # rather than silently inheriting the rule parse.
                boxed_content = scoring.extract_boxed(text)
                boxed_scored = task.vsibench_process_results(
                    dict(doc), [scoring.normalize_boxed(boxed_content)]
                )["vsibench_score"]
                boxed_score, boxed_answered, boxed_parsed = _rule_metric(boxed_scored, qtype, task)
                if cfg.boxed_primary and boxed_content is not None:
                    rule_score, rule_answered, rule_parsed = boxed_score, boxed_answered, boxed_parsed
                rec = SampleRecord(
                    id=doc["id"],
                    dataset=doc["dataset"],
                    scene_name=doc["scene_name"],
                    question_type=qtype,
                    question=doc["question"],
                    ground_truth=str(doc["ground_truth"]),
                    options=doc.get("options"),
                    response=text,
                    output_tokens=n_tokens,
                    truncated=truncated,
                    finish_reason=completion.finish_reason,
                    rule_parsed=rule_parsed,
                    rule_answered=rule_answered,
                    rule_score=rule_score,
                    final_parsed=rule_parsed,
                    final_answered=rule_answered,
                    final_score=rule_score,
                    failure_reason=classify_failure(
                        truncated=truncated,
                        final_answered=rule_answered,
                        score=rule_score,
                        judge_recovered=False,
                    ),
                    boxed_content=boxed_content,
                    boxed_answered=boxed_answered,
                    boxed_score=boxed_score,
                )
                records.append(rec)
            done += len(requests)
            if progress:
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"[{index + 1}/{len(batches)}] {done}/{total} questions "
                    f"| {rate:.2f} q/s | elapsed {elapsed / 60:.1f}m",
                    flush=True,
                )

    judge_pending, judge_kinds, judge_prompts = build_judge_requests(
        records, task, upstream, mca_scope=cfg.judge_mca_scope
    )
    judge_stats = apply_judge(records, judge_pending, judge_kinds, judge_prompts, cfg, task, upstream, scoring)

    if not records:
        raise SystemExit(
            "no VSI-Bench rows were scored; every scene was skipped. "
            "Typical cause: the eval python cannot import decord "
            "(system python + ~/.local vllm does not include it; use conda sr_opsd)."
        )

    elapsed = time.perf_counter() - started
    summary = build_summary(records, task, cfg, elapsed, judge_stats)
    return records, summary


def build_summary(
    records: list[SampleRecord],
    task,
    cfg: EvalConfig,
    elapsed: float,
    judge_stats: dict,
) -> dict:
    n = len(records)
    scored_rows = []
    for rec in records:
        row = {
            "question_type": rec.question_type,
            "answered": rec.final_answered,
            "parsed_answer": rec.final_parsed,
        }
        if rec.question_type in task.MCA_QUESTION_TYPES:
            row["accuracy"] = rec.final_score
        else:
            row["MRA:.5:.95:.05"] = rec.final_score
        scored_rows.append(row)

    overall = task.vsibench_aggregate_results([dict(r) for r in scored_rows])
    answered_pct = task.vsibench_aggregate_answered(scored_rows)

    # Per-type score is the plain mean of that type's metric, which is exactly
    # the per-type value the upstream aggregator averages into `overall`.  The
    # three object_rel_direction difficulties are reported separately here;
    # `overall` still merges them first, as upstream defines it.
    by_type: dict[str, dict] = {}
    for qtype in sorted({r.question_type for r in records}):
        subset = [r for r in records if r.question_type == qtype]
        by_type[qtype] = {
            "n": len(subset),
            "score": 100 * sum(r.final_score for r in subset) / max(len(subset), 1),
            "rule_only_score": 100 * sum(r.rule_score for r in subset) / max(len(subset), 1),
            "answered_pct": 100 * sum(1 for r in subset if r.final_answered) / max(len(subset), 1),
            "truncated": sum(1 for r in subset if r.truncated),
            "truncated_pct": 100 * sum(1 for r in subset if r.truncated) / max(len(subset), 1),
            "median_output_tokens": sorted(r.output_tokens for r in subset)[len(subset) // 2],
            "failure_reasons": dict(Counter(r.failure_reason for r in subset)),
            "boxed_present_pct": 100
            * sum(1 for r in subset if r.boxed_content is not None)
            / max(len(subset), 1),
            "boxed_score": 100 * sum(r.boxed_score for r in subset) / max(len(subset), 1),
        }

    lengths = sorted(r.output_tokens for r in records)

    def pct(p: float) -> int:
        if not lengths:
            return 0
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)]

    failure_counts = Counter(r.failure_reason for r in records)
    rule_overall_rows = []
    for rec in records:
        row = {"question_type": rec.question_type, "answered": rec.rule_answered}
        if rec.question_type in task.MCA_QUESTION_TYPES:
            row["accuracy"] = rec.rule_score
        else:
            row["MRA:.5:.95:.05"] = rec.rule_score
        rule_overall_rows.append(row)

    boxed_rows = []
    for rec in records:
        row = {"question_type": rec.question_type, "answered": rec.boxed_answered}
        if rec.question_type in task.MCA_QUESTION_TYPES:
            row["accuracy"] = rec.boxed_score
        else:
            row["MRA:.5:.95:.05"] = rec.boxed_score
        boxed_rows.append(row)
    boxed_present = sum(1 for r in records if r.boxed_content is not None)

    return {
        "model": cfg.model,
        "protocol": cfg.protocol,
        "frames": cfg.frames,
        "max_tokens": cfg.max_tokens,
        "prompt_suffix": cfg.prompt_suffix,
        "boxed_primary": cfg.boxed_primary,
        "decoding": resolve_decoding(cfg),
        "questions": n,
        "wall_seconds": elapsed,
        "questions_per_second": n / elapsed if elapsed else 0.0,
        "rule_only": {
            "overall": task.vsibench_aggregate_results([dict(r) for r in rule_overall_rows]),
            "answered_pct": task.vsibench_aggregate_answered(rule_overall_rows),
        },
        "judge_assisted": {
            "overall": overall,
            "answered_pct": answered_pct,
            "enabled": bool(cfg.judge_api_base),
            "pending_rows": judge_stats.get("judged", 0),
            "recovered": judge_stats.get("recovered", 0),
            "judge_unparsed": judge_stats.get("unparsed", 0),
            "judge_api_errors": judge_stats.get("api_errors", 0),
        },
        # Diagnostic only: `overall` here is what the run would score if the box
        # were the only place an answer could live. It is not a VSI-Bench number
        # for this model unless the prompt actually asked for a box.
        "boxed": {
            "present": boxed_present,
            "present_pct": 100 * boxed_present / max(n, 1),
            "overall": task.vsibench_aggregate_results([dict(r) for r in boxed_rows]),
            "answered_pct": task.vsibench_aggregate_answered(boxed_rows),
        },
        "by_question_type": by_type,
        "failure_reasons": dict(failure_counts),
        "output_tokens": {
            "median": pct(0.5),
            "p90": pct(0.9),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": pct(1.0),
        },
        "truncated": sum(1 for r in records if r.truncated),
        "truncated_pct": 100 * sum(1 for r in records if r.truncated) / max(n, 1),
    }


def format_report(summary: dict) -> str:
    rule = summary["rule_only"]
    judged = summary["judge_assisted"]
    tokens = summary["output_tokens"]
    boxed = summary.get("boxed", {})
    suffix = summary.get("prompt_suffix", "")
    decoding = summary.get("decoding", {"do_sample": False})
    if decoding.get("do_sample"):
        decode_line = (
            f"sampling t={decoding['temperature']} top_p={decoding['top_p']} "
            f"top_k={decoding['top_k']} min_p={decoding['min_p']} "
            f"presence={decoding['presence_penalty']} seed={decoding['seed']}"
        )
    else:
        decode_line = "greedy"
    lines = [
        f"\n=== VSI-Bench ({summary['protocol']}, {summary['frames']} frames, "
        f"{summary['max_tokens']} tokens) ===",
        f"decoding      : {decode_line}",
        f"prompt suffix : {suffix!r}" if suffix else "prompt suffix : (none)",
        f"rule-only     : {rule['overall']:.2f} | answered {rule['answered_pct']:.2f}%",
        f"judge-assisted: {judged['overall']:.2f} | answered {judged['answered_pct']:.2f}% "
        f"(+{judged['overall'] - rule['overall']:.2f})",
        f"wall          : {summary['wall_seconds'] / 60:.1f} min "
        f"({summary['questions_per_second']:.2f} q/s)",
        f"truncated     : {summary['truncated']} / {summary['questions']} "
        f"({summary['truncated_pct']:.2f}%)",
        f"out tokens    : median {tokens['median']}, p90 {tokens['p90']}, "
        f"p99 {tokens['p99']}, max {tokens['max']}",
        f"failure tags  : {summary['failure_reasons']}",
    ]
    if boxed:
        lines.append(
            f"boxed         : present {boxed['present']} / {summary['questions']} "
            f"({boxed['present_pct']:.2f}%) | box-only score {boxed['overall']:.2f} "
            f"| box-only answered {boxed['answered_pct']:.2f}%"
        )
    if judged["enabled"]:
        lines.append(
            f"judge         : {judged['pending_rows']} rows sent, "
            f"{judged['recovered']} recovered, {judged['judge_unparsed']} unreadable, "
            f"{judged['judge_api_errors']} api errors"
        )
    lines.append(
        f"\n  {'question type':<28}{'n':>5}{'score':>8}{'rule':>8}"
        f"{'answered':>10}{'trunc':>8}{'med tok':>9}{'boxed':>8}{'boxsc':>8}  failures"
    )
    for qtype, info in summary["by_question_type"].items():
        fails = ", ".join(f"{k}={v}" for k, v in sorted(info["failure_reasons"].items()))
        lines.append(
            f"  {qtype:<28}{info['n']:>5}{info['score']:>8.2f}{info['rule_only_score']:>8.2f}"
            f"{info['answered_pct']:>9.2f}%{info['truncated_pct']:>7.1f}%"
            f"{info['median_output_tokens']:>9}{info.get('boxed_present_pct', 0.0):>7.1f}%"
            f"{info.get('boxed_score', 0.0):>8.2f}  {fails}"
        )
    return "\n".join(lines)


def write_results(records: list[SampleRecord], summary: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    samples_path = os.path.join(output_dir, "samples.jsonl")
    with open(samples_path, "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    failures_dir = os.path.join(output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)
    for reason in FAILURE_TAGS:
        subset = [r.to_dict() for r in records if r.failure_reason == reason]
        if subset:
            with open(os.path.join(failures_dir, f"{reason}.jsonl"), "w") as handle:
                for row in subset:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
