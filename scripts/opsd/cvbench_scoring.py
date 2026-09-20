"""Shared CV-Bench prompt building and answer parsing for in-training eval.

Prompts follow ``src/lmms_eval/tasks/cvbench`` when ``prompt_style=lmms_eval``.
The official combined metric lives in ``verl/trainer/ppo/cvbench_metrics.py``;
this module only builds prompts and extracts option letters.

Parser selection (via ``custom_reward_function.reward_kwargs.cvbench_parser`` or
``CVBENCH_PARSER``):

* ``boxed_lastline`` (default): closed ``\\boxed{}`` is the only source; otherwise
  last-line option extraction from ``vsibench_scoring.extract_vsibench_option``.
* ``lastline``: last-line option extract only; used with the SPAR lastline prompt
  (no ``\\boxed{}``).
* ``word_boundary``: full-text scan used for the 84.82/84.84 SPAR3 dumps.
* ``lmms_legacy``: ``extract_characters_regex`` from lmms_eval, kept behind an
  if-else so older scores remain reproducible.
"""

from __future__ import annotations

import os
import re
import string
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_SRC_LMMS = os.path.join(_REPO_ROOT, "src")
_OPSD_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_LMMS not in sys.path:
    sys.path.insert(0, _SRC_LMMS)
if _OPSD_DIR not in sys.path:
    sys.path.insert(0, _OPSD_DIR)

from vsibench_scoring import (  # noqa: E402
    BOXED_LASTLINE_SUFFIX,
    LASTLINE_SUFFIX,
    extract_boxed,
    extract_vsibench_option,
    normalize_boxed,
)

LMMS_MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NATIVE_POST_PROMPT = LMMS_MCA_POST_PROMPT

CVBENCH_PARSERS = ("boxed_lastline", "word_boundary", "lmms_legacy", "lastline")
CVBENCH_PROMPT_STYLES = ("lmms_eval", "native")
DEFAULT_PARSER = "boxed_lastline"
DEFAULT_PROMPT_STYLE = "lmms_eval"

_CVBENCH_PAREN_OPTION_RE = re.compile(r"\(([A-F])\)")
_CVBENCH_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")
_CVBENCH_ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
)

try:
    from lmms_eval.tasks.cvbench.utils import (  # type: ignore[import-not-found]
        extract_characters_regex as _lmms_extract_characters_regex,
    )

    _HAS_LMMS_EVAL = True
except ImportError:
    _HAS_LMMS_EVAL = False

    def _lmms_extract_characters_regex(s: str) -> str:
        s = s.strip()
        for answer_prefix in _CVBENCH_ANSWER_PREFIXES:
            s = s.replace(answer_prefix, "")
        if len(s.split()) > 10 and not re.search(r"[ABCDEF]", s):
            return ""
        match = re.search(r"[ABCDEF]", s)
        return match[0] if match else ""


def normalize_parser(parser: str | None) -> str:
    value = (parser or os.environ.get("CVBENCH_PARSER") or DEFAULT_PARSER).strip()
    if value not in CVBENCH_PARSERS:
        raise ValueError(f"unknown cvbench parser {value!r}; expected one of {CVBENCH_PARSERS}")
    return value


def normalize_prompt_style(style: str | None) -> str:
    value = (style or os.environ.get("CVBENCH_PROMPT_STYLE") or DEFAULT_PROMPT_STYLE).strip()
    if value not in CVBENCH_PROMPT_STYLES:
        raise ValueError(f"unknown cvbench prompt style {value!r}; expected one of {CVBENCH_PROMPT_STYLES}")
    return value


def _lmms_doc_to_text_fixed(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """``cvbench_doc_to_text`` with ``pre_prompt: ""`` honoured.

    lmms_eval uses ``kwargs.get("pre_prompt", "") or "These are frames..."``, so
    an empty string falls through to the video default (LESSON-010). CV-Bench's
    yaml sets ``pre_prompt: ""`` on purpose; we skip that default here.
    """
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    if pre_prompt is None:
        pre_prompt = ""
    post_prompt = kwargs.get("mca_post_prompt") or LMMS_MCA_POST_PROMPT
    question = doc["question"]
    chars = string.ascii_uppercase
    options = "Options:\n" + "\n".join(f"{chars[i]}. {choice}" for i, choice in enumerate(doc["choices"]))
    suffix = kwargs.get("answer_suffix")
    if suffix is None:
        suffix = BOXED_LASTLINE_SUFFIX
    closing = (post_prompt or "") + suffix
    parts = [part for part in (pre_prompt, question, options, closing) if part]
    return "\n".join(parts)


def _protocol_suffix(protocol: str | None = None) -> str:
    if (protocol or "").strip() == "lastline":
        return LASTLINE_SUFFIX
    return BOXED_LASTLINE_SUFFIX


def build_cvbench_prompt(doc: dict, style: str | None = None, protocol: str | None = None) -> str:
    """Return the user-visible CV-Bench question text (without the image token)."""
    prompt_style = normalize_prompt_style(style)
    suffix = _protocol_suffix(protocol)
    if prompt_style == "native":
        return "\n".join([doc["prompt"], NATIVE_POST_PROMPT + suffix])
    lmms_kwargs = {
        "pre_prompt": "",
        "mca_post_prompt": LMMS_MCA_POST_PROMPT,
        "answer_suffix": suffix,
    }
    return _lmms_doc_to_text_fixed(doc, lmms_kwargs)


def _extract_word_boundary(text: str) -> str:
    cleaned = text or ""
    for prefix in _CVBENCH_ANSWER_PREFIXES:
        cleaned = cleaned.replace(prefix, "")

    match = _CVBENCH_PAREN_OPTION_RE.search(cleaned)
    if match:
        return match.group(1)
    match = _CVBENCH_OPTION_RE.search(cleaned)
    return match.group(1) if match else ""


def extract_cvbench_option(text: str, parser: str | None = None, choices: list | None = None) -> str:
    """Return the selected option letter, or "" when none can be read."""
    parser_name = normalize_parser(parser)
    if parser_name == "lmms_legacy":
        return _lmms_extract_characters_regex(text)
    if parser_name == "word_boundary":
        return _extract_word_boundary(text)
    if parser_name == "lastline":
        return extract_vsibench_option(text, options=choices)

    boxed = extract_boxed(text)
    if boxed is not None:
        return extract_vsibench_option(normalize_boxed(boxed), options=choices)
    return extract_vsibench_option(text, options=choices)


def score_cvbench_row(
    prediction: str,
    target: str,
    parser: str | None = None,
    choices: list | None = None,
) -> tuple[float, float]:
    """Return (accuracy, answered) for one CV-Bench row."""
    option = extract_cvbench_option(prediction, parser=parser, choices=choices)
    gold = target.strip().upper()
    return float(option == gold), float(bool(option))
