"""Read answers out of SPAR's templated ground truth and out of the model's prose.

``mvopsd_reward._mvopsd_score`` only understands a bare option letter or a bare
number, which covers 30.1% of the MV-OPSD training pool and 0% of the SPAR free
text.  Running the teacher against that scorer would report "the teacher is
wrong about almost everything at N=3 and N=32", when in fact those two buckets
are exactly the two SPAR sources whose answers it cannot read.

SPAR's answers are template-generated, so the *gold* side parses reliably.  The
*response* side does not: the teacher writes several hundred tokens of prose.
The rule tier here is therefore deliberately conservative -- it answers only
when the reading is unambiguous -- and everything it declines is handed to the
judge in extraction mode, which is where a judge belongs (LESSON-019).

Question types fall into five families:

``numeric``       one distance / depth / count / area, graded by MRA .5:.95:.05,
                  the same metric VSI-Bench uses for its numerical types.
``direction``     a tuple over three independent axes: horizontal (left|right),
                  vertical (above|below) and, for object-to-object questions,
                  relative depth (closer|farther).
``compare_pair``  which of the green / blue point is the closer (or farther) one.
``compare_set``   which ``ObjectN`` of several is the closest (or farthest).
``bev``           a bird's-eye-view coordinate list.  Reported separately and
                  never folded into the headline accuracy: the tolerance below
                  is our choice, not SPAR's published metric.

Two deliberate exclusions from the direction family, both because the gold
itself is ambiguous rather than because the parser is weak:

* the ``front``/``behind`` clause is dropped.  It describes where the *observer*
  stands relative to the objects ("The trash can and the tray are in front of
  the observer"), not the answer, and a single gold string routinely contains
  both words for its two objects.
* the ``*_imagination_*_mv`` types state the position **before and after** the
  observer moves, and the question asks only for the state after.  When a naive
  scan sees both ``left`` and ``right`` the text is re-read from the last
  move marker; if it is still ambiguous the row is reported as gold-unparsed
  rather than scored on a coin flip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# One definition of MRA for the whole repo. SPAR's numeric types are graded on
# VSI-Bench's scale on purpose, so they have to use VSI-Bench's exact arithmetic
# too -- a second copy drifts on the threshold boundaries (see the comment on
# vsibench_scoring.MRA_THRESHOLDS).
from vsibench_scoring import MRA_THRESHOLDS, mean_relative_accuracy  # noqa: F401

NUMERIC_TYPES = frozenset(
    {
        "depth_prediction_oc_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
        "distance_prediction_oo_video",
        "obj_count",
        "room_size",
    }
)
DIRECTION_TYPES = frozenset(
    {
        "obj_spatial_relation_oc_mv",
        "obj_spatial_relation_oo_mv",
        "spatial_imagination_oc_mv",
        "spatial_imagination_oo_mv",
        "spatial_imagination_oc_video",
        "spatial_imagination_oc_video_hard",
        "spatial_imagination_oo_video",
        "spatial_imagination_oo_video_hard",
    }
)
COMPARE_PAIR_TYPES = frozenset({"distance_infer_center_oo_mv"})
COMPARE_SET_TYPES = frozenset({"distance_infer_center_oo_video"})
BEV_TYPES = frozenset({"spatial_imagination_map_mv"})

SPAR_TYPES = NUMERIC_TYPES | DIRECTION_TYPES | COMPARE_PAIR_TYPES | COMPARE_SET_TYPES | BEV_TYPES

# Numeric types whose answer carries a unit, as opposed to obj_count.
METRIC_TYPES = NUMERIC_TYPES - {"obj_count"}

# Only the object-to-object types carry a relative-depth clause; asking for one
# in an observer-to-object answer would mark every gold string incomplete.
DEPTH_AXIS_TYPES = frozenset(
    {
        "obj_spatial_relation_oo_mv",
        "spatial_imagination_oo_mv",
        "spatial_imagination_oo_video",
        "spatial_imagination_oo_video_hard",
    }
)

AXES: dict[str, tuple[str, str]] = {
    "horizontal": ("left", "right"),
    "vertical": ("above", "below"),
    "depth": ("closer", "farther"),
}

# Deliberately the same reader the response side uses. A second, laxer copy used
# to live here and read 45 out of "2 rj45 outlet" while the response side read 2,
# which marks a correct answer wrong; the gold and the response have to be read
# by one definition or they drift apart silently.
from vsibench_scoring import _NUMBER_RE  # noqa: E402

# "After the move to X", "Upon reaching X", "Once the observer moves", "Following the move"
_POST_MOVE_RE = re.compile(r"\b(?:after|upon|once|following)\b", re.IGNORECASE)
# "The X and the Y are in front of the observer." / "X is behind the new observer."
_OBSERVER_CLAUSE_RE = re.compile(
    r"[^.]*\b(?:in\s+front\s+of|behind)\s+the\s+(?:new\s+)?observer\b[^.]*\.?",
    re.IGNORECASE,
)
_BEV_ENTRY_RE = re.compile(r"Object\s*(\d+)\s*:?\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")

# Phrases after which a model states its conclusion. Used to find the sentence
# that carries the answer inside several hundred tokens of reasoning.
_CONCLUSION_RE = re.compile(
    r"\b(?:the\s+answer\s+is|therefore|thus|in\s+conclusion|to\s+summarize|so,|final\s+answer|hence)\b",
    re.IGNORECASE,
)


@dataclass
class Parsed:
    """What could be read out of a gold string or a response."""

    kind: str
    number: float | None = None
    axes: dict[str, str] = field(default_factory=dict)
    choice: str = ""
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    ambiguous: bool = False

    def is_empty(self) -> bool:
        return self.number is None and not self.axes and not self.choice and not self.points

    def describe(self) -> str:
        if self.number is not None:
            return f"{self.number:g}"
        if self.axes:
            return "+".join(f"{axis}:{value}" for axis, value in sorted(self.axes.items()))
        if self.choice:
            return self.choice
        if self.points:
            return f"bev[{len(self.points)}]"
        return ""


def family_of(question_type: str) -> str:
    if question_type in NUMERIC_TYPES:
        return "numeric"
    if question_type in DIRECTION_TYPES:
        return "direction"
    if question_type in COMPARE_PAIR_TYPES:
        return "compare_pair"
    if question_type in COMPARE_SET_TYPES:
        return "compare_set"
    if question_type in BEV_TYPES:
        return "bev"
    return ""


# --------------------------------------------------------------------------
# gold side
# --------------------------------------------------------------------------


# The measured quantity is always stated with its unit ("0.8 meters", "30
# square meters"). Reading the first number instead would answer "50" for
# "The red point (Object50) ... have a distance of 0.8 meters", and the object
# ids appear in most of the distance templates.
_GOLD_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:square\s+)?meters?\b", re.IGNORECASE)


def _gold_number(gold: str) -> float | None:
    with_unit = _GOLD_UNIT_RE.findall(gold or "")
    if with_unit:
        return float(with_unit[-1])
    numbers = _NUMBER_RE.findall(gold or "")
    return float(numbers[-1]) if numbers else None


def _axes_in(text: str, want_depth: bool) -> tuple[dict[str, str], bool]:
    axes: dict[str, str] = {}
    ambiguous = False
    for axis, (low, high) in AXES.items():
        if axis == "depth" and not want_depth:
            continue
        seen = [word for word in (low, high) if re.search(rf"\b{word}\b", text, re.IGNORECASE)]
        if len(seen) == 1:
            axes[axis] = seen[0]
        elif len(seen) == 2:
            ambiguous = True
    return axes, ambiguous


def parse_direction(text: str, question_type: str) -> Parsed:
    want_depth = question_type in DEPTH_AXIS_TYPES
    stripped = _OBSERVER_CLAUSE_RE.sub(" ", text)
    axes, ambiguous = _axes_in(stripped, want_depth)
    if not ambiguous:
        return Parsed(kind="direction", axes=axes)

    # Both poles of some axis appear: the gold describes the position before and
    # after the observer moved, and only the second one is the answer.
    markers = list(_POST_MOVE_RE.finditer(stripped))
    if markers:
        tail = stripped[markers[-1].start() :]
        axes, ambiguous = _axes_in(tail, want_depth)
        if not ambiguous:
            return Parsed(kind="direction", axes=axes)
    return Parsed(kind="direction", ambiguous=True)


def effective_family(question_type: str, gold: str) -> str:
    """The family this *row* is scored as, which is not always its type's family.

    360 ``obj_spatial_relation_oo_mv`` rows are phrased as "Is X above Y? Only
    answer Yes/No" and their gold is the bare word, so they are a yes/no row
    inside a direction type.
    """
    if (gold or "").strip().lower() in ("yes", "no"):
        return "yesno"
    return family_of(question_type)


def parse_gold(question_type: str, gold: str) -> Parsed:
    gold = (gold or "").strip()
    family = effective_family(question_type, gold)

    if family == "yesno":
        return Parsed(kind="yesno", choice=gold.lower())

    if family == "numeric":
        return Parsed(kind="numeric", number=_gold_number(gold))

    if family == "direction":
        return parse_direction(gold, question_type)

    if family == "compare_pair":
        # "the table (blue point) is farther to chair" -- the colour is the
        # answer, but the template states it with either polarity, so the
        # polarity has to travel with it and be resolved against the question.
        colour, polarity = _colour_and_polarity(gold)
        if not colour:
            return Parsed(kind="compare_pair", ambiguous=True)
        return Parsed(kind="compare_pair", choice=f"{colour}:{polarity}")

    if family == "compare_set":
        obj, polarity = _object_and_polarity(gold)
        if not obj:
            return Parsed(kind="compare_set", ambiguous=True)
        return Parsed(kind="compare_set", choice=f"{obj}:{polarity}")

    if family == "bev":
        points = {
            int(index): (float(x), float(y)) for index, x, y in _BEV_ENTRY_RE.findall(gold)
        }
        return Parsed(kind="bev", points=points, ambiguous=not points)

    return Parsed(kind="unknown", ambiguous=True)


def _colour_and_polarity(text: str) -> tuple[str, str]:
    """The colour named as the answer, and the polarity it was named with.

    The templates recite both distances before stating the verdict, and some of
    them state it twice with opposite polarity ("... is farther ...  The stove
    (green point) is closer ..."), so the verdict is read from the **last**
    sentence that carries a polarity word, and inside it from the colour nearest
    to that word.  Some answers are the bare colour with no polarity at all
    ("dumbbell (green point)"); those name the answer directly and come back
    with an empty polarity.
    """
    lowered = text.lower()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", lowered) if s.strip()]
    for sentence in reversed(sentences):
        polarity_hits = [
            match
            for match in re.finditer(r"\b(closer|farther)\b", sentence)
        ]
        if not polarity_hits:
            continue
        verdict = polarity_hits[-1]
        colour_hits = [match for match in re.finditer(r"\b(green|blue)\s+point\b", sentence)]
        if not colour_hits:
            continue
        nearest = min(colour_hits, key=lambda match: abs(match.start() - verdict.start()))
        return nearest.group(1), verdict.group(1)

    colours = {match.group(1) for match in re.finditer(r"\b(green|blue)\s+point\b", lowered)}
    if len(colours) == 1:
        return colours.pop(), ""
    return "", ""


def _object_and_polarity(text: str) -> tuple[str, str]:
    """The ``ObjectN`` named as the answer, and whether it was named closest or farthest.

    The gold writes ``Object2`` and the question writes ``point2`` for the same
    thing, and responses copy either one, so both spellings normalise to the
    index.
    """
    lowered = text.lower()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", lowered) if s.strip()]
    for sentence in reversed(sentences):
        verdicts = list(re.finditer(r"\b(closest|farthest|closer|farther)\b", sentence))
        objects = list(re.finditer(r"(?:object|point)\s*(\d+)", sentence))
        if not objects:
            continue
        polarity = ""
        if verdicts:
            polarity = "farthest" if verdicts[-1].group(1).startswith("far") else "closest"
            nearest = min(objects, key=lambda match: abs(match.start() - verdicts[-1].start()))
        elif len({match.group(1) for match in objects}) == 1:
            nearest = objects[0]
        else:
            continue
        return f"Object{nearest.group(1)}", polarity
    return "", ""


# --------------------------------------------------------------------------
# response side
# --------------------------------------------------------------------------


# "clothes (green point)", "kitchen counter (in Frame-31, point2)". The name is
# kept short and word-like so a whole clause cannot be swallowed as a name.
_PAIR_CANDIDATE_RE = re.compile(
    r"([A-Za-z][A-Za-z \-']{0,40}?)\s*\(\s*(green|blue)\s+point\s*\)", re.IGNORECASE
)
_SET_CANDIDATE_RE = re.compile(
    r"([A-Za-z][A-Za-z \-']{0,40}?)\s*\(\s*(?:in\s+Frame-\d+\s*,\s*)?(?:point|object)\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def candidates_from_question(question: str, family: str) -> dict[str, str]:
    """Map each candidate's plain name to its identifier, as the question names them.

    The teacher frequently answers with the object's name ("Therefore, the sofa
    chair is the farthest") instead of the ``point1`` label the question
    attached to it. Without this map those answers read as unreadable.
    """
    pairs: dict[str, str] = {}
    if family == "compare_pair":
        for name, colour in _PAIR_CANDIDATE_RE.findall(question or ""):
            pairs[name.strip().lower()] = colour.lower()
    else:
        for name, index in _SET_CANDIDATE_RE.findall(question or ""):
            pairs[name.strip().lower()] = f"Object{index}"
    # A name shared by two candidates cannot identify either of them.
    counts: dict[str, set[str]] = {}
    for name, value in pairs.items():
        counts.setdefault(name, set()).add(value)
    return {name: next(iter(values)) for name, values in counts.items() if len(values) == 1 and name}


def _value_by_name(text: str, candidates: dict[str, str]) -> str:
    """The candidate whose name appears last in the text, when only one does."""
    best_position = -1
    best_value = ""
    for name, value in candidates.items():
        for match in re.finditer(rf"\b{re.escape(name)}\b", text or "", re.IGNORECASE):
            if match.start() > best_position:
                best_position, best_value = match.start(), value
    return best_value


def _answer_segment(response: str) -> str:
    """The part of a long response most likely to hold its conclusion.

    Preference order: text after an explicit conclusion marker, then the last
    non-empty line.  Scanning the whole response instead would let a direction
    word from the reasoning ("the chair is on the left, but...") outvote the
    conclusion.
    """
    text = (response or "").strip()
    if not text:
        return ""
    markers = list(_CONCLUSION_RE.finditer(text))
    if markers:
        tail = text[markers[-1].start() :]
        if tail.strip():
            return tail
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


# Counting answers are routinely spelled out ("there are two ovens"), and
# `obj_count` is a sixth of the SPAR 32-view pool. Only small integers: nothing
# above twenty is ever written as a word in these responses, and a wider table
# would start matching prose ("a hundred things to look at").
_WORD_NUMBERS = {
    "zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE)


def _number_from_words(text: str) -> float | None:
    hits = _WORD_NUMBER_RE.findall(text or "")
    if not hits:
        return None
    return float(_WORD_NUMBERS[hits[-1].lower()])


def parse_response(question_type: str, response: str, family: str = "", question: str = "") -> Parsed:
    family = family or family_of(question_type)
    text = (response or "").strip()
    if not text:
        return Parsed(kind=family or "unknown")

    if family == "yesno":
        segment = _answer_segment(text)
        for candidate in (segment, text):
            hits = list(re.finditer(r"\b(yes|no)\b", candidate, re.IGNORECASE))
            if hits:
                return Parsed(kind="yesno", choice=hits[-1].group(1).lower())
        return Parsed(kind="yesno")

    if family == "numeric":
        # The same reader VSI-Bench's numerical types use, so "the teacher could
        # not be read" means the same thing on both benchmarks -- but for the
        # types measured in metres, a number carrying the unit outranks it. Its
        # fallback is "the last number on the last line", and these answers
        # routinely end with an object id ("...is 6.7 meters, Object169").
        from vsibench_scoring import extract_vsibench_number

        segment = _answer_segment(text)
        value = None
        if question_type in METRIC_TYPES:
            with_unit = _GOLD_UNIT_RE.findall(segment) or _GOLD_UNIT_RE.findall(text)
            if with_unit:
                value = float(with_unit[-1])
        if value is None:
            value = extract_vsibench_number(text)
        if value is None:
            value = _number_from_words(segment)
        if value is None:
            return Parsed(kind="numeric")
        return Parsed(kind="numeric", number=value)

    if family == "direction":
        # Same reader as the gold side, including the before/after fallback: a
        # response that narrates both states must be resolved the way the gold
        # is, or a correct answer scores as unreadable.
        return parse_direction(_answer_segment(text), question_type)

    if family in ("compare_pair", "compare_set"):
        read = _colour_and_polarity if family == "compare_pair" else _object_and_polarity
        segment = _answer_segment(text)
        value, polarity = read(segment)
        if not value:
            value, polarity = read(text)
        if not value:
            # The response named the object rather than its label.
            candidates = candidates_from_question(question, family)
            value = _value_by_name(segment, candidates) or _value_by_name(text, candidates)
            polarity = ""
            for scope in (segment, text):
                hits = re.findall(r"\b(closest|farthest|closer|farther)\b", scope, re.IGNORECASE)
                if hits:
                    polarity = "farthest" if hits[-1].lower().startswith("far") else "closest"
                    break
        if not value:
            return Parsed(kind=family)
        return Parsed(kind=family, choice=f"{value}:{polarity}")

    if family == "bev":
        points = {int(i): (float(x), float(y)) for i, x, y in _BEV_ENTRY_RE.findall(text)}
        return Parsed(kind="bev", points=points)

    return Parsed(kind="unknown")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

# A BEV point counts as placed if it lands within this radius of the gold
# coordinate. SPAR does not publish this metric, so bev never enters the
# headline accuracy -- it is reported on its own line.
BEV_TOLERANCE_M = 0.5


_QUESTION_POLARITY_RE = re.compile(r"\b(closest|farthest|closer|farther)\b", re.IGNORECASE)


def question_polarity(question: str) -> str:
    """Whether the question asks for the closer/closest or the farther/farthest one."""
    hits = _QUESTION_POLARITY_RE.findall(question or "")
    if not hits:
        return ""
    return "far" if hits[0].lower().startswith("far") else "close"


def _resolve_choice(choice: str, asked: str, alternatives: tuple[str, ...]) -> str:
    """Turn a ``value:polarity`` statement into the value that answers the question.

    "the blue point is closer" and "the green point is farther" state the same
    fact, and the gold and the response each pick their own phrasing.  With
    exactly two candidates, a statement made with the opposite polarity to the
    question names the *other* candidate.  A statement with no polarity word
    ("dumbbell (green point)") already names the answer.

    With more than two candidates the flip is not defined -- "X is the farthest"
    does not say which one is the closest -- so the verdict is marked as naming
    the wrong end and can never match.
    """
    value, polarity = choice.rsplit(":", 1)
    if not polarity or not asked:
        return value
    if polarity.startswith("far") == (asked == "far"):
        return value
    others = [item for item in alternatives if item != value]
    return others[0] if len(others) == 1 else f"~{value}"


def score_row(question_type: str, gold: str, response: str, question: str = "") -> dict:
    """Grade one SPAR row.

    ``answered`` is 0 when nothing could be read from the response, which is a
    different failure from answering wrongly and must never be folded into a
    zero score (LESSON-018).  ``gold_parsed`` is 0 when the gold itself is
    ambiguous; those rows are excluded from the denominator rather than counted
    as misses.

    ``question`` is optional but needed by the comparison families, which state
    their answer with whichever polarity the template felt like using.
    """
    family = effective_family(question_type, gold)
    gold_parsed = parse_gold(question_type, gold)
    result = {
        "family": family,
        "gold_parsed": int(not gold_parsed.ambiguous and not gold_parsed.is_empty()),
        "gold_repr": gold_parsed.describe(),
        "answered": 0,
        "parsed_repr": "",
        "score": 0.0,
        "axes": {},
        "axes_missing": [],
    }
    if not result["gold_parsed"]:
        return result

    predicted = parse_response(question_type, response, family=family, question=question)
    result["parsed_repr"] = predicted.describe()
    if predicted.is_empty():
        return result

    if family == "yesno":
        result["answered"] = 1
        result["score"] = float(predicted.choice == gold_parsed.choice)
        return result

    if family == "numeric":
        result["answered"] = 1
        result["score"] = mean_relative_accuracy(predicted.number, gold_parsed.number)
        return result

    if family == "direction":
        # A response that names some of the gold's axes but not all of them has
        # been read -- it just did not answer the whole question, which is a
        # wrong answer and not an unreadable one. Only a response with no axis
        # at all counts as unanswered and goes to the judge.
        per_axis = {}
        missing = []
        for axis, gold_value in gold_parsed.axes.items():
            if axis in predicted.axes:
                per_axis[axis] = int(predicted.axes[axis] == gold_value)
            else:
                missing.append(axis)
        result["axes"] = per_axis
        result["axes_missing"] = missing
        if not per_axis:
            return result
        result["answered"] = 1
        result["score"] = float(not missing and all(per_axis.values()))
        return result

    if family in ("compare_pair", "compare_set"):
        asked = question_polarity(question)
        if family == "compare_pair":
            alternatives = ("green", "blue")
        else:
            alternatives = tuple(sorted(set(candidates_from_question(question, family).values())))
        gold_value = _resolve_choice(gold_parsed.choice, asked, alternatives)
        pred_value = _resolve_choice(predicted.choice, asked, alternatives)
        result["answered"] = 1
        result["gold_repr"] = gold_value
        result["parsed_repr"] = pred_value
        result["score"] = float(gold_value == pred_value)
        return result

    if family == "bev":
        shared = set(gold_parsed.points) & set(predicted.points)
        if not shared:
            return result
        result["answered"] = 1
        hits = 0
        for index in shared:
            gx, gy = gold_parsed.points[index]
            px, py = predicted.points[index]
            hits += ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5 <= BEV_TOLERANCE_M
        # Objects the response never placed count against it; otherwise a model
        # that answers for one object out of seven scores like a perfect one.
        result["score"] = hits / len(gold_parsed.points)
        return result

    return result
