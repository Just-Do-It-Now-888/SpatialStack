"""Answer extraction for VSI-Bench when the model reasons before answering.

Upstream's parser is ``pred.split(' ')[0]``: it takes the first whitespace token
and nothing else. That is correct for the protocol it ships with (16 new tokens,
"answer with the option's letter directly"), and wrong for any checkpoint that
writes a chain of thought first -- the first token is then a word like "Based",
which scores 0 on multiple choice and parses to ``None`` on the numerical half,
where ``None`` is recorded as the worst possible MRA rather than as a parse
failure. CV-Bench showed the size of this effect: the same step-300 weights read
83.57 under a parser that finds the answer and 47.11 under one that reads the
first capital letter it sees (LESSON-017).

Only the extraction changes here. Per-question-type metrics (exact match for the
six multiple-choice types, ``MRA:.5:.95:.05`` for the four numerical ones) and
the unweighted mean over question types stay exactly as lmms_eval defines them,
so a number produced with this parser is still a VSI-Bench number.

Two rules, selected by ``VSIBENCH_PARSER``:

* ``answer_tail`` (default): read a last-line selection phrase -- "the answer
  is X", "would be X", "I choose X", "so X" -- or a verdict line the same
  forms ``is_trustable_mca_tail`` uses (lone ``C``, ``C. left``, unique option
  body, or a lone ``(C)``). Do not take a stray letter out of "a guess" or
  "T, L, B, S".
* ``lmms_legacy``: upstream's first-token rule, kept so published numbers stay
  reproducible.

The in-training val parquet (``vsibench_val_boxed_lastline.parquet``) also
asks for ``\\boxed{}`` on the last line and scores a closed box first
(``vsibench_boxed_primary``). Offline lmms-eval ``spatialstack`` uses the same
suffix and the same boxed-primary extract; ``VSIBENCH_BOXED_PRIMARY=0``
restores last-line-only scoring on archived generations.
"""

from __future__ import annotations

import os
import re
import string

VSIBENCH_PARSERS = ("answer_tail", "lmms_legacy")
DEFAULT_PARSER = "answer_tail"

# Same bytes as the in-training val parquet: a leading space concatenated onto
# the MCA/NA instruction, not a new line.
BOXED_LASTLINE_SUFFIX = (
    r" The final answer MUST BE put in \boxed{} on the last line of your response."
)
# SPAR-Bench lastline wording. original vs lastline prompt pairs use this, not
# BOXED_LASTLINE_SUFFIX; the parser is last-line extract with boxed-primary off.
LASTLINE_SUFFIX = (
    " The final answer MUST BE put on the last line of your response."
)

# Last-line selection phrases. Last match on that line wins. ``would be``
# counts; stray letters in "T, L, B, S" do not (no bare-letter scan).
_ANSWER_LETTER_RE = re.compile(
    r"(?:"
    r"(?:final\s+)?(?:answer|option|choice|selection)\s*(?:is|:|=|would\s+be)?\s*"
    r"|"
    r"would\s+be\s+"
    r"|"
    r"(?:I\s+)?(?:choose|pick|select)\s+"
    r"|"
    r"(?:I\s+)?go\s+with\s+"
    r"|"
    r"(?<![A-Za-z])(?:so|thus|therefore|hence)\s+"
    r")"
    r"[\*\s]*\(?\s*([A-Z])\s*\)?(?![A-Za-z])",
    re.IGNORECASE,
)
_WHOLE_PAREN_LETTER_RE = re.compile(r"^\(\s*([A-Za-z])\s*\)$")

# Numbers with optional sign, thousands separators and decimals.
#
# The two lookbehinds decide when a digit run is a quantity rather than part of
# an identifier. `(?<![\w.])` keeps a version-like "1.2.3" from being read as
# two numbers and keeps "scene0555_00" from yielding 555. `(?<!\w-)` is what
# stops "Frame-4" from parsing as the number -4: the hyphen in a frame label
# glues a token together, it is not a minus sign. Multi-view prompts label every
# image "Frame-N" and the teacher cites those labels constantly, so without this
# a counting answer that ends "...visible in Frame-18." was read as -18.
#
# `(?<!\^)` and `(?<!\^\{)` cover a third way a digit can fail to be a quantity:
# as a LaTeX exponent. Asking for the answer in \boxed{} invites LaTeX, and the
# units of two of the four numerical types are areas and volumes, so
# `\boxed{30.9\,\text{m}^2}` is an ordinary thing for a model to write. Since
# the numeric fallback takes the *last* number on the line, the exponent won
# and a correct 30.9 was scored as the answer 2 -- the same shape of bug as
# `Frame-4` reading as -4, introduced by the format we ourselves asked for.
#
# The lookahead rejects a following digit or a following ".<digit>" but *not* a
# bare trailing dot: `(?![\d.])` also rejected every number that ends a
# sentence, so "Therefore, the count of pipes is 0." parsed to None and was
# reported as unreadable rather than as the answer 0.
_NUMBER_RE = re.compile(r"(?<![\w.])(?<!\w-)(?<!\^)(?<!\^\{)[-+]?\d[\d,]*(?:\.\d+)?(?!\d)(?!\.\d)")
_ANSWER_NUMBER_RE = re.compile(
    r"(?:final\s+)?(?:answer|value|result|estimate|count|total|distance|size|area)\s*"
    r"(?:is|:|=|of)?\s*(?:about|approximately|around|roughly|~)?\s*"
    r"[\*\s]*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Degenerate long responses often end in a frame-by-frame dump labelled "Image N:"
# (or "* Image N:"). The numeric fallback used to read that index as the answer:
# a sofa-size question truncated at "Image 178: Kitchen." scored 178 cm against a
# 173 cm gold and passed every MRA threshold. Skip these lines in the tail scan;
# an answer that cites "Image 1 shows the chair. ... Therefore 2.8 meters." still
# reads from the conclusion line above the enumeration block.
_IMAGE_LABEL_LINE_RE = re.compile(r"^\s*[\*\-\u2022]?\s*Image\s+\d+\s*[:.]", re.IGNORECASE)
# "4. Estimate the distance:" is an outline heading, not the quantity 4.
_OUTLINE_HEADING_RE = re.compile(r"^\s*[\*\-\u2022]?\s*\d+\.\s")


def normalize_parser(parser: str | None) -> str:
    value = (parser or os.environ.get("VSIBENCH_PARSER") or DEFAULT_PARSER).strip()
    if value not in VSIBENCH_PARSERS:
        raise ValueError(f"unknown vsibench parser {value!r}; expected one of {VSIBENCH_PARSERS}")
    return value


def boxed_primary_enabled(flag: bool | None = None) -> bool:
    """Whether a closed ``\\boxed{}`` is the only answer site.

    Matches ``custom_reward_function.reward_kwargs.vsibench_boxed_primary``.
    ``flag`` wins when the trainer passes it explicitly; otherwise
    ``VSIBENCH_BOXED_PRIMARY`` (default on) is the offline switch.
    """
    if flag is not None:
        return bool(flag)
    value = (os.environ.get("VSIBENCH_BOXED_PRIMARY") or "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def apply_boxed_primary(text: str, boxed_primary: bool | None = None) -> tuple[str, bool]:
    """Return ``(text to parse, box was closed)``.

    When boxed-primary is on and the response has a closed box, only the
    normalised box contents are scored -- the surrounding CoT, including a
    truncated ``Image N:`` dump, cannot become the answer.
    """
    if not boxed_primary_enabled(boxed_primary):
        return text, False
    content = extract_boxed(text)
    if content is None:
        return text, False
    return normalize_boxed(content), True


def legacy_fuzzy_matching(pred: str) -> str:
    """Upstream's ``fuzzy_matching``, character for character."""
    return (pred or "").split(" ")[0].rstrip(".").strip()


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK_RE.sub(" ", text or "")


_TERSE_NUMBER_RE = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?$")
_TERSE_LETTER_RE = re.compile(r"^[A-F]$", re.IGNORECASE)
_MCA_TAIL_RE = re.compile(r"^\s*([A-F])\s*[.):]\s*(.*?)\s*$", re.IGNORECASE)


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in _strip_thinking(text).splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _is_junk_tail_line(line: str) -> bool:
    return bool(_IMAGE_LABEL_LINE_RE.match(line) or _OUTLINE_HEADING_RE.match(line))


_FRAME_LIST_RE = re.compile(r"\bframes?\s+\d", re.IGNORECASE)
_ENUM_INT_THRESHOLD = 5


def _is_integer_enumeration_line(line: str) -> bool:
    """True for ``frame 3, 4, 5, ...`` dumps, not a lone quantity like ``2.8``."""
    stripped = (line or "").strip()
    if not stripped or _TERSE_NUMBER_RE.match(stripped):
        return False
    numbers = _NUMBER_RE.findall(stripped)
    if len(numbers) >= _ENUM_INT_THRESHOLD:
        return True
    return bool(_FRAME_LIST_RE.search(stripped) and len(numbers) >= 2)


def _line_for_answer_parse(lines: list[str]) -> str:
    """Line to read an explicit answer from.

    Normally the last non-empty line. When the response ends in a truncated
    frame dump (``Image N:``) or outline heading, walk up one step so a
    conclusion written just above the dump still counts -- but do not keep
    scanning for incidental numbers in earlier CoT.
    """
    if not lines:
        return ""
    if not _is_junk_tail_line(lines[-1]):
        return lines[-1]
    for line in reversed(lines[:-1]):
        if not _is_junk_tail_line(line):
            return line
    return ""


def _is_terse_token(token: str) -> bool:
    if not token:
        return False
    if _IMAGE_LABEL_LINE_RE.match(token):
        return False
    return bool(_TERSE_NUMBER_RE.match(token) or _TERSE_LETTER_RE.match(token))


def is_terse_answer(text: str) -> bool:
    """True when the whole response or its last line is a lone number/letter.

    Used with ``trust_terse``: an unambiguous terse answer is treated like a
    closed ``\\boxed{}`` and scored by the rule tier without sending the row to
    the judge. No truncation gate -- a truncated ``Image 178:`` tail that ends
    in a lone digit is still terse and is trusted on purpose.
    """
    cleaned = _strip_thinking(text or "").strip()
    if not cleaned:
        return False
    if _is_terse_token(cleaned):
        return True
    return _is_terse_token(_last_nonempty_line(text))


def is_trustable_mca_tail(text: str, options: list[str] | None = None) -> bool:
    """True when the last line is an unambiguous multiple-choice verdict.

    Trusts ``C``, ``(C)``, ``C. front-left``, or a last line that uniquely
    matches one option body. Letter+body rows require the body to match that
    letter's option text so ``A. front-left`` is not trusted when ``front-left``
    belongs to C. Uses the same last-line walk as the scorer (skip ``Image N:``).
    """
    cleaned = _strip_thinking(text or "").strip()
    if not cleaned:
        return False
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return bool(_letter_from_trustable_tail(_line_for_answer_parse(lines), options))


def _letter_from_trustable_tail(tail: str, options: list[str] | None) -> str:
    """Option letter if ``tail`` is a verdict line, else ``""``."""
    if not tail:
        return ""
    allowed = _valid_letters(len(options) if options else None)

    if _TERSE_LETTER_RE.match(tail):
        letter = tail.upper()
        return letter if letter in allowed else ""

    paren = _WHOLE_PAREN_LETTER_RE.match(tail)
    if paren:
        letter = paren.group(1).upper()
        return letter if letter in allowed else ""

    match = _MCA_TAIL_RE.match(tail)
    if match:
        letter = match.group(1).upper()
        body = match.group(2).strip()
        if letter not in allowed:
            return ""
        if not options:
            return ""
        bodies = _option_bodies(options)
        expected = bodies.get(letter, "")
        if not body:
            return letter if (expected or letter in bodies) else ""
        if expected and body.lower() == expected.lower():
            return letter
        return ""

    return _match_option_body(tail, options)


def extract_boxed(text: str) -> str | None:
    """Contents of the last well-formed ``\\boxed{...}``, or None if there is none.

    Brace-balanced scan rather than a regex, ported from verl's
    ``verl/utils/reward_score/math_dapo.py::last_boxed_only_string`` so a nested
    ``\\boxed{\\frac{1}{2}}`` is not cut at the first ``}``.

    Two deliberate departures from that file, both because our failure mode is
    not MATH-500's:

    * No trailing window. Upstream scores ``solution_str[-300:]`` and then
      ``pred[-100:]``, which is right when the answer is always the last thing
      written. Our degenerate responses run to the full 1,024-token budget, and a
      model that answers and *then* keeps rambling would be recorded as having
      given no answer at all.
    * Returns the string, not a verdict. Scoring stays with the per-question-type
      metrics in this module -- exact match for the six multiple-choice types,
      ``MRA:.5:.95:.05`` for the four numerical ones. Upstream's ``compute_score``
      is exact-match only, and routing our numerical half through it would
      collapse a continuous metric into a binary one while still reporting it
      under the MRA name (LESSON-019).
    """
    haystack = text or ""
    start = haystack.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for index in range(start + len("\\boxed"), len(haystack)):
        char = haystack[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return haystack[start + len("\\boxed{") : index]
    # Unclosed brace: the response was truncated part-way through the answer.
    return None


_TEX_WRAPPER_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_TEX_SPACING_RE = re.compile(r"\\[,;:!]|\\\s|~|\$|\\left|\\right")
_TEX_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?")
_TEX_FRAC_RE = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_PLAIN_RATIO_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)\s*$")


def normalize_boxed(content: str | None) -> str:
    """Strip LaTeX decoration from box contents so the answer is readable.

    Only ever applied to what was inside a ``\\boxed{}``, never to free prose.
    Inside a box every character is part of the answer, so decoration can be
    removed without the risk that makes this unsafe on a full response.

    Upstream (``math_dapo.normalize_final_answer``) does the same job with a
    hand-listed table -- ``REMOVED_EXPRESSIONS`` contains the literal strings
    ``"\\text{}^2"``, ``"\\text{}^3"``, ``"cm"``, ``"meters"``, ``"square"``.
    That table is grown from the cases MATH-500 happened to contain, so it
    covers ``^2`` and ``^3`` but not ``^4``, and only after a separate
    substitution has rewritten ``\\text{m}``. Matching by structure instead
    (unwrap ``\\text{...}``, drop spacing macros, resolve ``\\frac``) does not
    need to know which units the benchmark uses.
    """
    if content is None:
        return ""
    text = content
    for _ in range(3):  # nested \text{\mathrm{...}} is rare but cheap to allow
        replaced = _TEX_WRAPPER_RE.sub(r"\1", text)
        if replaced == text:
            break
        text = replaced
    text = _TEX_DEGREE_RE.sub(" ", text)
    text = _TEX_FRAC_RE.sub(r"\1/\2", text)
    text = _TEX_SPACING_RE.sub(" ", text)
    text = text.replace("\\%", "%").strip()

    # A bare ratio has to be divided out: the numeric fallback reads the last
    # number on the line, so "1/2" would otherwise be read as the answer 2.
    ratio = _PLAIN_RATIO_RE.match(text)
    if ratio:
        numerator, denominator = _to_float(ratio.group(1)), _to_float(ratio.group(2))
        if numerator is not None and denominator not in (None, 0.0):
            quotient = numerator / denominator
            return str(int(quotient)) if quotient == int(quotient) else repr(quotient)
    return text


def _valid_letters(num_options: int | None) -> str:
    # Falling back to A-F rather than to "any letter": VSI-Bench never offers
    # more than four options, and an unbounded range would let a "T" from prose
    # count as a selection.
    if not num_options or num_options < 1:
        return string.ascii_uppercase[:6]
    return string.ascii_uppercase[: min(num_options, 26)]


def _last_match(pattern: re.Pattern[str], text: str, allowed: str) -> str:
    found = ""
    for match in pattern.finditer(text):
        letter = match.group(1).upper()
        if letter in allowed:
            found = letter
    return found


def _option_bodies(options: list[str] | None) -> dict[str, str]:
    """Map option letter -> option text, for ``['A. right', 'B. left']`` input."""
    bodies: dict[str, str] = {}
    for index, raw in enumerate(options or []):
        text = str(raw)
        match = re.match(r"\s*([A-Za-z])\s*[.):]\s*(.*)", text)
        if match:
            bodies[match.group(1).upper()] = match.group(2).strip()
        else:
            bodies[string.ascii_uppercase[index]] = text.strip()
    return bodies


def _match_option_body(text: str, options: list[str] | None) -> str:
    """Letter of the one option whose text the last line contains uniquely.

    A model that ignores the "answer with the letter" instruction often answers
    with the option itself ("back-left"). Last-line only: scanning a 200-char
    window of CoT matches "sofa on the left" as a vote for A.left.
    """
    bodies = _option_bodies(options)
    if not bodies:
        return ""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    tail = (lines[-1] if lines else "").lower()
    if not tail:
        return ""
    exact_hits = [letter for letter, body in bodies.items() if body and body.lower() == tail]
    if len(exact_hits) == 1:
        return exact_hits[0]
    hits = [letter for letter, body in bodies.items() if body and body.lower() in tail]
    return hits[0] if len(hits) == 1 else ""


def extract_vsibench_option(
    text: str,
    options: list[str] | None = None,
    parser: str | None = None,
) -> str:
    """Return the selected option letter, or "" when none can be read."""
    if normalize_parser(parser) == "lmms_legacy":
        return legacy_fuzzy_matching(text)

    cleaned = _strip_thinking(text).strip()
    if not cleaned:
        return ""
    allowed = _valid_letters(len(options) if options else None)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    last = _line_for_answer_parse(lines)

    letter = _last_match(_ANSWER_LETTER_RE, last, allowed)
    if letter:
        return letter
    return _letter_from_trustable_tail(last, options)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


# The ten MRA thresholds, built the way upstream builds them and not the way the
# name ".5:.95:.05" reads. `src/lmms_eval/tasks/vsibench/utils.py:180-184` does
#
#     num_pts = (end - start) / interval + 2     -> 10.999999999999998
#     conf_intervs = np.linspace(start, end, int(num_pts))
#
# so the count is 10 (the +2 is cancelled by the truncation of a value that just
# misses 11), and two of the ten thresholds carry float error that a decimal
# `0.5 + 0.05 * i` does not: at index 8 upstream's `1 - threshold` is
# 0.10000000000000009 where the decimal form gives 0.09999999999999998. A
# relative error of "exactly" 0.1 evaluates to 0.10000000000000009, so it lands
# inside the band upstream and outside the decimal one. That is not a rounding
# curiosity -- it decides whole rows: 1.1 against 1.0, 9 against 10, 1.7 against
# 2.0, 18 against 20 all sit exactly on a threshold, and they are common answers.
# On the 5,130-row VSI-Bench validation set at step 0 it moved 8 rows and 0.02
# points, always downward, which is enough to make the in-training curve
# non-comparable to the published number it exists to track (LESSON-013).
#
# Reproduced with the same linspace so the two agree bit for bit rather than
# approximately.
def _mra_thresholds() -> tuple[float, ...]:
    import numpy as np

    start, end, interval = 0.5, 0.95, 0.05
    return tuple(np.linspace(start, end, int((end - start) / interval + 2)).tolist())


MRA_THRESHOLDS = _mra_thresholds()


def mean_relative_accuracy(prediction: float, target: float) -> float:
    """VSI-Bench's ``MRA:.5:.95:.05``, bit-identical to lmms_eval's version.

    The single definition for the whole repo: the in-training reward, the offline
    harness and SPAR's numeric types all have to agree, or the same answer scores
    differently depending on which code path measured it (LESSON-023).
    """
    if target == 0:
        return float(prediction == 0)
    error = abs(prediction - target) / abs(target)
    return sum(1 for threshold in MRA_THRESHOLDS if error <= 1 - threshold) / len(MRA_THRESHOLDS)


def extract_vsibench_number(text: str, parser: str | None = None) -> float | None:
    """Return the answer as a float, or ``None`` when none can be read."""
    if normalize_parser(parser) == "lmms_legacy":
        return _to_float(legacy_fuzzy_matching(text))

    cleaned = _strip_thinking(text).strip()
    if not cleaned:
        return None

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None
    line = _line_for_answer_parse(lines)
    if not line:
        return None

    matches = list(_ANSWER_NUMBER_RE.finditer(line))
    if matches:
        value = _to_float(matches[-1].group(1))
        if value is not None:
            return value

    numbers = _NUMBER_RE.findall(line)
    if not numbers:
        return None
    # Bare-number fallback only on a strict last line, or on a lone number just
    # above a truncated ``Image N:`` tail -- not on speculative CoT with
    # incidental measurements, and not on ``frame 3, 4, 5, ...`` dumps.
    if _is_junk_tail_line(lines[-1]) and not _TERSE_NUMBER_RE.match(line):
        return None
    if _is_integer_enumeration_line(line):
        return None
    return _to_float(numbers[-1])
