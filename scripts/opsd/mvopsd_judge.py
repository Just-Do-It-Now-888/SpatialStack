#!/usr/bin/env python3
"""LLM judge for verl's in-training validation, time-sharing the GPUs with training.

The offline protocol (`scripts/opsd/eval_visionopd/`) grades CV-Bench with a
gpt-oss-120b judge. The in-training curve used rules only, so the two could
disagree exactly when it mattered: a model that stops emitting bare option
letters scores zero under rules while the judge would still read its answer.
MV-OPSD v0 lost a run to a version of that gap (LESSON-011).

Judge and policy cannot both hold the cards -- gpt-oss-120b is 61 GB and
training fills all eight -- so they take turns instead of splitting the machine.
verl already opens the window: ``AgentLoopManager.generate_sequences`` sleeps
the rollout engines as its last step and wakes them at the top of the next call,
so validation ends with the cards free and training resumes on its own.

    validation generates (policy awake)
      -> agent loop sleeps the rollout engines
      -> rules resolve what they can
      -> if anything is left: judge.wake(), grade, judge.sleep()
      -> training resumes, policy wakes itself on the next generate_sequences()

Nothing here sleeps the policy. Doing so is not a harmless double-check: it
walks a level-2 sleep over weights vLLM has already released and dies in
``buffer.cpu()`` with CUDA "invalid argument".

The judge stays asleep the rest of the time, holding no device memory. With the
in-training prompt still forcing a direct answer, the rule tier resolves nearly
every row, so in a healthy run the judge is never woken at all -- and it wakes
precisely when the model starts rambling, which is the failure worth catching.

Verdicts come from the offline judge's own functions, loaded from that file
rather than reimplemented, so the in-training and offline numbers cannot drift
apart through a copy that was only fixed on one side.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

_OPSD_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_JUDGE = os.path.join(_OPSD_DIR, "eval_visionopd", "judge.py")
# The strict extract-only prompts and their parsers. Loaded from the offline
# tool rather than restated here: the in-training and offline primary numbers
# have to be the same measurement, and a second copy of a prompt is a copy that
# gets fixed on one side only (LESSON-023).
_STRICT_JUDGE = os.path.join(_OPSD_DIR, "tools", "judge_vsibench_tri_insurance.py")
_SCORING = os.path.join(_OPSD_DIR, "vsibench_scoring.py")

# Kept byte-identical to NA_EXTRACT_TEMPLATE in scripts/opsd/vsibench_eval_core.py:
# the in-training and offline judges must ask the same question of the same
# response, or the two curves stop being the same measurement.
NUMBER_EXTRACTION_TEMPLATE = (
    "Read the response below and report the single final numeric answer it gives.\n"
    "The question is: {question}\n"
    "The response is: {response}\n"
    "Reply with only the number, in digits, with no units, no ranges and no other text. "
    "If the response does not settle on a numeric answer, reply exactly NONE."
)

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _parse_number(reply: str) -> Optional[float]:
    if not reply or reply.strip().upper() == "NONE":
        return None
    match = _NUMBER_RE.search(reply)
    if match is None:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def _load_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_offline_judge():
    """Import scripts/opsd/eval_visionopd/judge.py, which is a CLI script, not a module."""
    return _load_path(_OFFLINE_JUDGE, "mvopsd_offline_judge")


class ValidationJudge:
    """Grades the rows the rule tier could not, waking the judge only when needed.

    Every failure path here is non-fatal. A judge that will not start, will not
    wake, or returns nonsense must degrade to the rule-only score that training
    already had: losing the extra signal costs a metric, while raising through
    `_validate` would cost the run.
    """

    def __init__(
        self,
        api_base: str,
        model: str = "judge",
        api_key: str = "EMPTY",
        max_tokens: int = 2048,
        parallel_workers: int = 256,
        min_rows: int = 16,
        # Level 1 parks the weights in host RAM; level 2 drops them and rebuilds
        # from disk. Level 2 is the cheaper-looking option and it is wrong here:
        # this MXFP4 checkpoint comes back from it answering "!!!!!!!!" while
        # still reporting itself healthy and awake, so a whole validation would
        # be marked wrong by a judge that looks fine. Level 1 costs 61 GB of the
        # node's 1.6 TB and restores the repacked weights verbatim.
        sleep_level: int = 1,
        wake_timeout_s: float = 600.0,
        request_timeout_s: float = 1800.0,
        reasoning_effort: str = "",
    ) -> None:
        self.api_base = api_base.rstrip("/") + "/"
        self.server_root = self.api_base[: self.api_base.rfind("/v1/")] if "/v1/" in self.api_base else self.api_base
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.parallel_workers = parallel_workers
        # Waking a 61 GB model to grade three rows costs more than the three rows
        # are worth; below this many the rule verdict stands and answered/frac
        # still reports that the rows went unresolved.
        self.min_rows = min_rows
        self.sleep_level = sleep_level
        self.wake_timeout_s = wake_timeout_s
        self.request_timeout_s = request_timeout_s
        self.reasoning_effort = reasoning_effort
        self._offline = _load_offline_judge()
        self._strict = None
        self._scoring = None
        self._local = threading.local()

    def _strict_modules(self):
        """The extract-only prompts/parsers, loaded lazily so cascade runs never pay for them."""
        if self._strict is None:
            self._strict = _load_path(_STRICT_JUDGE, "mvopsd_strict_judge")
            self._scoring = _load_path(_SCORING, "mvopsd_strict_scoring")
        return self._strict, self._scoring

    # --- lifecycle ---------------------------------------------------------

    def _post(self, path: str, timeout: float, **params) -> bool:
        import requests

        try:
            response = requests.post(f"{self.server_root}{path}", params=params or None, timeout=timeout)
            return response.status_code < 400
        except Exception as exc:  # noqa: BLE001 - any transport failure is non-fatal
            logger.warning("validation judge: %s failed: %s", path, exc)
            return False

    def is_sleeping(self, timeout: float = 10.0) -> Optional[bool]:
        """None when the server cannot be reached at all.

        `/health` is not the question to ask: a sleeping engine still answers it
        200, so it would report a judge with no weights on the GPUs as ready.
        """
        import requests

        try:
            response = requests.get(f"{self.server_root}/is_sleeping", timeout=timeout)
            if response.status_code >= 400:
                return None
            return bool(response.json().get("is_sleeping"))
        except Exception:  # noqa: BLE001
            return None

    def wake(self) -> bool:
        """Bring the judge's weights back onto the GPUs. Requires VLLM_SERVER_DEV_MODE=1."""
        if not self._post("/wake_up", timeout=self.wake_timeout_s):
            return False
        # The endpoint awaits the engine, but says itself that it can return
        # before the wake lands under frontend multiprocessing, so confirm it.
        deadline = time.time() + self.wake_timeout_s
        while time.time() < deadline:
            if self.is_sleeping() is False:
                return True
            time.sleep(2.0)
        logger.warning("validation judge: still asleep %.0fs after wake_up", self.wake_timeout_s)
        return False

    def sleep(self) -> bool:
        return self._post("/sleep", timeout=self.wake_timeout_s, level=self.sleep_level)

    # --- grading -----------------------------------------------------------

    def _client(self):
        from openai import OpenAI

        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.request_timeout_s)
            self._local.client = client
        return client

    def _ask(self, prompt: str) -> str:
        extra = {}
        if self.reasoning_effort:
            extra["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        for attempt in range(3):
            try:
                response = self._client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=self.max_tokens,
                    **extra,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception:  # noqa: BLE001 - the judge is best-effort
                if attempt < 2:
                    time.sleep(1.0)
        return ""

    def grade(self, questions, ground_truths, responses) -> tuple[list[Optional[float]], dict]:
        """Return per-row 1.0/0.0 (None when ungraded) plus a stats dict.

        The verdict rule is the offline one: an exact "yes" and nothing else.
        Replies that are neither yes nor no are counted separately rather than
        folded into the score, because a judge failure and a wrong answer must
        never look alike (LESSON-013).
        """
        stats = {"rows": len(questions), "yes": 0, "no": 0, "unparsed": 0, "seconds": 0.0}
        if not questions:
            return [], stats

        prompts = [
            self._offline.PROMPT_TEMPLATE.format(
                question=str(question).replace("<image>", ""),
                gt=gt,
                response=self._offline.extract_answer(response),
            )
            for question, gt, response in zip(questions, ground_truths, responses)
        ]

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            replies = list(pool.map(self._ask, prompts))
        stats["seconds"] = time.time() - started

        verdicts: list[Optional[float]] = []
        for reply in replies:
            normalized = reply.strip().lower()
            if normalized == "yes":
                stats["yes"] += 1
                verdicts.append(1.0)
            elif normalized == "no":
                stats["no"] += 1
                verdicts.append(0.0)
            else:
                # Unreadable: leave the rule verdict in place instead of inventing one.
                stats["unparsed"] += 1
                verdicts.append(None)
        return verdicts, stats

    def extract_answers(self, questions, responses, kinds, options=None):
        """Extract-only grading for *every* row: an option letter, a number, or nothing.

        This is the strict tier (`validation_judge.mode: extract`, the default
        since 2026-08-23). Unlike `grade`, it is authoritative rather than
        one-sided: the caller replaces the rule score with what comes back here,
        including downwards. That is the point of it -- the rule parser reads
        frame indices out of `Image N:` enumerations and scores them as full
        marks, so a tier that can only add points cannot correct it.

        Returns ``(answers, stats)`` where each answer is ``("mca", letter)``,
        ``("na", value)`` or ``(kind, None)`` when the judge found no settled
        answer. ``None`` from an API failure is indistinguishable here from
        ``None`` meaning "the response never answered", so the two are counted
        separately in ``stats``.
        """
        stats = {"rows": len(questions), "found": 0, "none": 0, "errors": 0, "seconds": 0.0}
        if not questions:
            return [], stats

        strict, scoring = self._strict_modules()
        opts = options or [None] * len(questions)
        prompts = []
        for question, response, kind, option in zip(questions, responses, kinds, opts):
            text = str(question).replace("<image>", "")
            body = self._offline.extract_answer(response)
            if kind == "mca":
                prompts.append(strict.build_mca_prompt(text, option or [], body))
            else:
                prompts.append(strict.build_na_prompt(text, body))

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            replies = list(pool.map(self._ask, prompts))
        stats["seconds"] = time.time() - started

        answers = []
        for reply, kind, option in zip(replies, kinds, opts):
            if not reply:
                # `_ask` returns "" after exhausting its retries, which is an
                # infrastructure failure, not a model that failed to answer.
                stats["errors"] += 1
                answers.append((kind, None))
                continue
            if kind == "mca":
                letter = strict.parse_mca_reply(reply, option, scoring)
                value = letter or None
            else:
                value = strict.parse_na_reply(reply, scoring)
            if value is None:
                stats["none"] += 1
            else:
                stats["found"] += 1
            answers.append((kind, value))
        return answers, stats

    def extract_numbers(self, questions, responses) -> tuple[list[Optional[float]], dict]:
        """Read the final numeric answer out of each response, or None.

        VSI-Bench's numerical types are scored by ``MRA:.5:.95:.05``, a mean over
        ten relative-error thresholds. Grading them with the Yes/No verdict above
        would collapse that continuous metric into a binary one while still
        reporting it under the MRA name (LESSON-019), so the judge is asked only
        to find the number and the caller applies the real metric to it.
        """
        stats = {"rows": len(questions), "found": 0, "none": 0, "seconds": 0.0}
        if not questions:
            return [], stats

        prompts = [
            NUMBER_EXTRACTION_TEMPLATE.format(
                question=str(question).replace("<image>", ""),
                response=self._offline.extract_answer(response),
            )
            for question, response in zip(questions, responses)
        ]

        started = time.time()
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            replies = list(pool.map(self._ask, prompts))
        stats["seconds"] = time.time() - started

        numbers: list[Optional[float]] = []
        for reply in replies:
            value = _parse_number(reply)
            if value is None:
                stats["none"] += 1
            else:
                stats["found"] += 1
            numbers.append(value)
        return numbers, stats
