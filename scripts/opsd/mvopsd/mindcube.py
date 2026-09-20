"""MindCube prompt surgery: make the prose describe the views the student gets.

MindCube encodes the album size in English rather than in a frame label. A row
says "these four images (image 1, 2, 3, and 4) ... from different viewpoints
(front, left, back, and right)" and then names one of them, so handing the
student a subset without touching the text leaves it counting images it cannot
see. That is not a cosmetic problem: the count, the enumeration, the viewpoint
list and the named anchor are all load-bearing, and the rotation family adds a
premise sentence about how the camera moved *between* views.

Two rules shape everything here.

1. **Rebuild the boilerplate, never substitute into it.** Option text carries the
   same words the boilerplate does -- "A. Two photographs", "Leather loveseat
   with three seat cushions" -- so a global "four" -> "one" pass would rewrite
   262 answers. Every template below is therefore parsed into (prefix, clause)
   and only the prefix is regenerated.

2. **Strip the premise before counting anchors.** The rotation premise cites
   image numbers ("Image 2 was taken after turning the camera 90 degrees...")
   that have to go at K=1 regardless, because a one-view student cannot be told
   how the camera moved between views it never sees. Counting references on the
   raw text makes 504 legal rows look like they name two views.

The pair-displacement template is deliberately exempt. "in which direction did I
move from the first view to the second view?" is a question *about* the pair, so
no rewrite makes it answerable from one view; per the 2026-09-07 protocol those
1,302 rows keep their wording verbatim and are marked as such in the plan, with
the incoherence accepted as a known cost rather than papered over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from . import views as vw

QUESTION_MARKER = "[Question]"

# The [Task] / [Answer Instruction] block is byte-identical across all 9,999
# training rows and all 1,050 tinybench rows, and it too claims a plural album.
TASK_PREAMBLE_PLURAL = (
    "Your task is to analyze the spatial arrangement of objects in the scene by "
    "examining the provided images, which show the scene from different viewpoints."
)
TASK_PREAMBLE_SINGULAR = (
    "Your task is to analyze the spatial arrangement of objects in the scene by "
    "examining the provided image, which shows the scene from one viewpoint."
)

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


class MindCubeRewriteError(Exception):
    """The question does not match any known template, or cannot be rewritten."""


def enumerate_images(count: int) -> str:
    """MindCube's own enumeration style: "image 1", "image 1 and 2", "image 1, 2, and 3"."""
    if count < 1:
        raise MindCubeRewriteError(f"cannot enumerate {count} images")
    if count == 1:
        return "image 1"
    if count == 2:
        return "image 1 and 2"
    body = ", ".join(str(index) for index in range(1, count))
    return f"image {body}, and {count}"


def join_labels(labels: Sequence[str]) -> str:
    """"front", "front and left", "front, left, and right"."""
    labels = list(labels)
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _number_word(count: int) -> str:
    if count not in NUMBER_WORDS:
        raise MindCubeRewriteError(f"no number word for {count} views")
    return NUMBER_WORDS[count]


def _split_label_list(text: str) -> tuple[str, ...]:
    """"front, left, back, and right" -> ("front", "left", "back", "right").

    The list is positional: image 1 is the first label. For the `among` family
    the image filenames independently encode the same labels in the same order
    (front_301.jpg, left_052.jpg, ...) and agree on all 7,055 rows, which is what
    licenses reading the other families' lists positionally too -- their
    filenames carry no labels at all.
    """
    parts = [part.strip() for part in text.replace(", and ", ", ").replace(" and ", ", ").split(",")]
    return tuple(part for part in parts if part)


# --- templates ----------------------------------------------------------------
#
# Seven prefixes cover all 9,999 rows exactly (verified by
# test_mindcube_templates_cover_the_whole_pool). Each entry owns a matcher and a
# builder that regenerates the prefix for K views.

AMONG4_RE = re.compile(
    r"^Based on these four images \(image 1, 2, 3, and 4\) showing (?P<obj>.+?) from different "
    r"viewpoints \((?P<labels>[^)]*)\), with each camera aligned with room walls and partially "
    r"capturing the surroundings: "
)
AROUND3_RE = re.compile(
    r"^Based on these three images \(image 1, 2, and 3\) showing the same scene from different "
    r"viewpoints \((?P<labels>[^)]*)\): "
)
PAIR_RE = re.compile(r"^Based on these two views showing the same scene: ")
ROT_TURN_RE = re.compile(
    r"^These (?P<word>two|three|four) images \([^)]*\) show the same scene from "
    r"(?:two |three )?different viewpoints\. (?P<premise>.+?) "
    r"Based on these (?:two|three|four) images: "
)
ROT_OPPOSING_RE = re.compile(
    r"^Based on these two opposing views \((?P<labels>[^)]*)\) of the same scene captured "
    r"during rotation: "
)


@dataclass(frozen=True)
class MindCubeQuestion:
    """A parsed question: constant preamble, album boilerplate, and the question."""

    template: str
    # Everything up to the first character of the question itself, i.e. the
    # [Task] / [Answer Instruction] block, the "[Question]" marker and the
    # newlines between them. Carried verbatim so reassembly is byte-exact.
    head: str
    # The album boilerplate as it appears in the source, i.e. what _build_prefix
    # regenerates. Kept so the pair passthrough can return the question verbatim.
    prefix: str
    clause: str
    n_views: int
    # 0-based view index the clause names, or None when it names none. Derived
    # after the premise is removed, so rotation rows read as single-anchor.
    anchor: Optional[int]
    labels: tuple[str, ...] = ()
    obj: str = ""

    @property
    def rewritable(self) -> bool:
        """False only for the pair-displacement template, which keeps its wording."""
        return self.template != "pair"


def parse_question(body: str, n_views: int) -> MindCubeQuestion:
    """Split the text after the ``<image>`` header into preamble + boilerplate + question.

    ``body`` is what ``vw.split_image_header`` returns: "\\n[Task]...[Question]\\n<question>".
    """
    if QUESTION_MARKER not in body:
        raise MindCubeRewriteError("body has no [Question] section")
    marker_end = body.index(QUESTION_MARKER) + len(QUESTION_MARKER)
    question_start = marker_end + len(body[marker_end:]) - len(body[marker_end:].lstrip("\n"))
    head, question = body[:question_start], body[question_start:]

    for name, pattern in (
        ("among4", AMONG4_RE),
        ("around3", AROUND3_RE),
        ("pair", PAIR_RE),
        ("rotation_turn", ROT_TURN_RE),
        ("rotation_opposing", ROT_OPPOSING_RE),
    ):
        match = pattern.match(question)
        if match is None:
            continue
        clause = question[match.end() :]
        groups = match.groupdict()
        labels = _split_label_list(groups["labels"]) if groups.get("labels") else ()
        if labels and len(labels) != n_views:
            raise MindCubeRewriteError(
                f"viewpoint list {labels} does not match n_views={n_views}"
            )
        anchors = sorted(vw.referenced_images(clause))
        if len(anchors) > 1:
            raise MindCubeRewriteError(f"clause names {len(anchors)} views: {anchors}")
        if anchors and not 0 <= anchors[0] < n_views:
            raise MindCubeRewriteError(f"clause names view {anchors[0]} outside n_views={n_views}")
        return MindCubeQuestion(
            template=name,
            head=head,
            prefix=match.group(0),
            clause=clause,
            n_views=n_views,
            anchor=anchors[0] if anchors else None,
            labels=labels,
            obj=groups.get("obj") or "",
        )

    raise MindCubeRewriteError(f"no template matches: {question[:120]!r}")


def _build_prefix(question: MindCubeQuestion, view_indices: Sequence[int]) -> str:
    count = len(view_indices)
    kept_labels = [question.labels[index] for index in view_indices] if question.labels else []

    if question.template == "among4":
        if count == 1:
            return (
                f"Based on this image (image 1) showing {question.obj} from one viewpoint "
                f"({kept_labels[0]}), with the camera aligned with room walls and partially "
                "capturing the surroundings: "
            )
        return (
            f"Based on these {_number_word(count)} images ({enumerate_images(count)}) showing "
            f"{question.obj} from different viewpoints ({join_labels(kept_labels)}), with each "
            "camera aligned with room walls and partially capturing the surroundings: "
        )

    if question.template == "around3":
        if count == 1:
            return (
                f"Based on this image (image 1) showing the same scene from one viewpoint "
                f"({kept_labels[0]}): "
            )
        return (
            f"Based on these {_number_word(count)} images ({enumerate_images(count)}) showing the "
            f"same scene from different viewpoints ({join_labels(kept_labels)}): "
        )

    if question.template == "rotation_turn":
        if count == 1:
            # The premise describes motion *between* views and is deleted, not
            # rewritten: nothing true about a rotation can be said to a student
            # holding one frame. What survives -- "facing the direction shown in
            # image 1, turn 90 degrees left, what is to my right?" -- is still a
            # well-posed mental-rotation question, just a harder one.
            return "This image (image 1) shows the scene from one viewpoint. Based on this image: "
        raise MindCubeRewriteError(
            "rotation premise can only be dropped wholesale (K=1); K>1 would need a "
            "premise rewritten for the surviving pair, which this builder does not do"
        )

    if question.template == "rotation_opposing":
        if count == 1:
            return (
                f"Based on this view ({kept_labels[0]}) of the same scene captured during "
                "rotation: "
            )
        return (
            f"Based on these {_number_word(count)} opposing views ({join_labels(kept_labels)}) of "
            "the same scene captured during rotation: "
        )

    raise MindCubeRewriteError(f"no prefix builder for template {question.template!r}")


def rewrite_for_views(question: MindCubeQuestion, view_indices: Sequence[int]) -> str:
    """Return the student's body text for ``view_indices`` (0-based, sorted, subset of N).

    Raises rather than emitting a prompt that references a view the student was
    not given -- the same contract ``renumber_frame_refs`` has.
    """
    view_indices = list(view_indices)
    if sorted(view_indices) != view_indices or len(set(view_indices)) != len(view_indices):
        raise MindCubeRewriteError(f"view_indices must be sorted and unique, got {view_indices}")
    if not view_indices or view_indices[-1] >= question.n_views:
        raise MindCubeRewriteError(
            f"view_indices {view_indices} out of range for n_views={question.n_views}"
        )

    if not question.rewritable:
        # Deliberate passthrough: the pair-displacement rows keep the two-view
        # wording while receiving fewer images (protocol decision 2026-09-07), so
        # for these rows the prompt describes views the student cannot see and
        # the label is not recoverable from its input. Asserted verbatim by
        # test_pair_template_keeps_its_stale_wording -- a later reader who
        # "fixes" this is changing the experiment, not repairing a bug.
        return question.head + question.prefix + question.clause

    if question.anchor is not None and question.anchor not in view_indices:
        raise MindCubeRewriteError(
            f"question names view {question.anchor + 1}, which is not in the student's views "
            f"{[index + 1 for index in view_indices]}"
        )

    mapping = {old: new for new, old in enumerate(view_indices)}
    clause = vw.renumber_image_refs(question.clause, mapping)
    if vw.referenced_images(clause) - set(range(len(view_indices))):
        raise MindCubeRewriteError(f"dangling image reference survived rewriting: {clause[:120]!r}")

    head = question.head
    if len(view_indices) == 1:
        # The [Task] block also promises a plural album ("the provided images,
        # which show the scene from different viewpoints"). Left alone it would be
        # false on every single student row, so it follows the same count rule as
        # the question prefix. This is the one place where the student's
        # boilerplate leaves the distribution the SFT checkpoint was trained on.
        if TASK_PREAMBLE_PLURAL not in head:
            raise MindCubeRewriteError("preamble is not the known MindCube [Task] block")
        head = head.replace(TASK_PREAMBLE_PLURAL, TASK_PREAMBLE_SINGULAR)

    return head + _build_prefix(question, view_indices) + clause


# --- gold answers -------------------------------------------------------------

ANSWER_RE = re.compile(r"^<answer>\s*([A-E])\.\s(?P<text>.*)</answer>$", re.DOTALL)
OPTION_RE = re.compile(r"(?:^|\s)([A-E])\.\s")


def parse_options(clause: str) -> tuple[str, dict[str, str]]:
    """Split "<question> A. x B. y" into the question and {letter: text}.

    Only a run starting at A and stepping one letter at a time counts, so option
    text that happens to read "C. something" cannot open a phantom option.
    """
    letters, spans = [], []
    for match in OPTION_RE.finditer(clause):
        letter = match.group(1)
        if letter != chr(ord("A") + len(letters)):
            continue
        letters.append(letter)
        spans.append((match.start(1), match.end()))
    if not letters:
        return clause.strip(), {}
    options = {}
    for index, (letter, (start, end)) in enumerate(zip(letters, spans)):
        stop = spans[index + 1][0] if index + 1 < len(spans) else len(clause)
        options[letter] = clause[end:stop].strip()
    return clause[: spans[0][0]].strip(), options


def answer_letter(gold: str) -> str:
    """``<answer>C. Curtain</answer>`` -> ``C``.

    Raises on anything else so a format drift in the annotations surfaces at pool
    build time rather than as a pool of silently zero-scoring rows.
    """
    match = ANSWER_RE.match(gold.strip())
    if match is None:
        raise MindCubeRewriteError(f"gold answer is not <answer>X. ...</answer>: {gold[:80]!r}")
    return match.group(1)


def answer_text(gold: str) -> str:
    """``<answer>C. Curtain</answer>`` -> ``Curtain``, for the option cross-check."""
    match = ANSWER_RE.match(gold.strip())
    if match is None:
        raise MindCubeRewriteError(f"gold answer is not <answer>X. ...</answer>: {gold[:80]!r}")
    return match.group("text").strip()
