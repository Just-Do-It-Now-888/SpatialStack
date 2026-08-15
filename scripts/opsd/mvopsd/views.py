"""View subsetting and prompt rewriting.

Student and teacher must differ in exactly one thing: how many views they see.
Everything else -- question wording, marker pixels, view ordering, per-view
resolution -- has to survive the subsetting untouched, which is what the
renumbering and anchor rules below are for.
"""

from __future__ import annotations

import random
import re
from typing import Iterable, Optional, Sequence

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

FRAME_REF_RE = re.compile(r"Frame-(\d+)")
# SPAR's BEV-style questions define the world origin as the observer position of
# the first image; dropping view 0 silently invalidates the ground truth.
ANCHOR_PHRASES = ("first image", "first view", "main viewpoint")

DEFAULT_K_MENU = (1, 2, 4)


def split_image_header(text: str) -> tuple[str, str]:
    """Split a conversation value into its leading image block and the question body."""
    if IMAGE_TOKEN not in text:
        raise ValueError("conversation value contains no <image> placeholder")
    end = text.rindex(IMAGE_TOKEN) + len(IMAGE_TOKEN)
    header, body = text[:end], text[end:]
    if IMAGE_TOKEN in body:
        raise ValueError("unexpected <image> placeholder after the image header")
    return header, body


def has_frame_labels(header: str) -> bool:
    return header.lstrip().startswith("Frame-0:")


def referenced_views(text: str) -> set[int]:
    return {int(match) for match in FRAME_REF_RE.findall(text)}


def needs_anchor_view(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ANCHOR_PHRASES)


def renumber_frame_refs(text: str, mapping: dict[int, int]) -> str:
    """Rewrite ``Frame-N`` references to the student's view numbering.

    Raises if the body mentions a view the student was not given: leaving a
    dangling frame number in the question is a silent semantic corruption.
    """

    def replace(match: re.Match) -> str:
        old = int(match.group(1))
        if old not in mapping:
            raise KeyError(f"question references Frame-{old}, which is not in the student's views")
        return f"Frame-{mapping[old]}"

    return FRAME_REF_RE.sub(replace, text)


def choose_views(
    rng: random.Random,
    n_views: int,
    required: Iterable[int] = (),
    k_menu: Sequence[int] = DEFAULT_K_MENU,
) -> Optional[list[int]]:
    """Pick the student's view subset, or None when no legal budget exists.

    ``required`` holds views the question cannot lose (marker frames, the anchor
    view). ``k < n_views`` is strict, so every surviving sample carries a real
    teacher advantage.
    """
    required = set(required)
    if any(not 0 <= view < n_views for view in required):
        raise ValueError(f"required views {sorted(required)} out of range for n_views={n_views}")
    k_min = max(1, len(required))
    options = [k for k in k_menu if k_min <= k < n_views]
    if not options:
        return None
    k = rng.choice(options)
    filler_pool = [view for view in range(n_views) if view not in required]
    filler = rng.sample(filler_pool, k - len(required))
    return sorted(required | set(filler))


def render_header(style: str, count: int, view_indices: Optional[Sequence[int]] = None) -> str:
    """Rebuild the image block for ``count`` views in the source's own format."""
    if style == "frame_labels":
        return "\n".join(f"Frame-{i}: {IMAGE_TOKEN}" for i in range(count))
    if style == "bare":
        return "\n".join([IMAGE_TOKEN] * count)
    if style == "concat":
        return IMAGE_TOKEN * count
    raise ValueError(f"unknown header style: {style}")


def collapse_image_newlines(text: str) -> str:
    """Apply the SFT dataloader's ``<image>\\n`` -> ``<image>`` normalisation.

    `data_qwen.py:_get_item` does this to every sample, so the SFT checkpoint has
    only ever seen the collapsed form.
    """
    return text.replace(f"{IMAGE_TOKEN}\n", IMAGE_TOKEN)


def build_prompt(style: str, count: int, body: str) -> str:
    return collapse_image_newlines(render_header(style, count) + body)
