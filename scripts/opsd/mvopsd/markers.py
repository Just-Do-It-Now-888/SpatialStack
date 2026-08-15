"""Discover and reproduce the SPAR visual markers that SFT draws on the fly.

SPAR questions refer to objects through coloured points/boxes that
`src/qwen_vl/data/draw_marker.py` paints onto specific views inside the SFT
dataloader ("the shower curtain (in Frame-23) (red point)"). verl has no such
hook, so the markers have to be baked into the cached views. That in turn makes
a marked view sample-specific and pins it into the student's view subset: drop
it and the question loses its referent.

Rather than restating which views each of the ~30 draw functions touches, the
helpers below run the real draw function against a recording list and observe
the indices it reads or writes.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Iterable, Optional

from PIL import Image

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from qwen_vl.data.draw_marker import DRAW_FUNCTIONS  # noqa: E402


class MarkerError(RuntimeError):
    """The sample's marker metadata is unusable (missing key, bad index, ...)."""


class _RecordingList(list):
    """A list that remembers which positions a draw function touched."""

    def __init__(self, items: Iterable[Any]):
        super().__init__(items)
        self.touched: set[int] = set()

    def __getitem__(self, index):
        if isinstance(index, int):
            self.touched.add(index % len(self) if index < 0 else index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.touched.add(index % len(self) if index < 0 else index)
        return super().__setitem__(index, value)


class LazyImageList(_RecordingList):
    """Loads a view from disk only when the draw function actually asks for it."""

    def __init__(self, paths: list[str], loader: Callable[[str], Image.Image]):
        super().__init__([None] * len(paths))
        self._paths = paths
        self._loader = loader

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if value is None and isinstance(index, int):
            value = self._loader(self._paths[index])
            list.__setitem__(self, index, value)
        return value

    def loaded(self) -> dict[int, Image.Image]:
        return {i: value for i, value in enumerate(list.__iter__(self)) if value is not None}


def parse_spar_info(annotation: dict) -> Optional[dict]:
    """``spar_info`` is stored as a JSON string inside the annotation JSON."""
    raw = annotation.get("spar_info")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def draw_markers(images: _RecordingList, info: dict) -> set[int]:
    """Run the SFT draw function for this sample; return the touched view indices."""
    task_type = info.get("type")
    if task_type not in DRAW_FUNCTIONS:
        raise MarkerError(f"unknown spar_info type: {task_type!r}")
    try:
        DRAW_FUNCTIONS[task_type](images, info)
    except MarkerError:
        raise
    except Exception as exc:  # malformed annotation: missing colour key, index out of range
        raise MarkerError(f"{task_type}: {type(exc).__name__}: {exc}") from exc
    bad = [i for i in images.touched if not 0 <= i < len(images)]
    if bad:
        raise MarkerError(f"{task_type}: marker references views {bad} outside 0..{len(images) - 1}")
    return set(images.touched)


def marked_view_indices(info: dict, n_views: int) -> set[int]:
    """Which views this sample paints markers on, without touching the real files."""
    probe = _RecordingList([Image.new("RGB", (64, 48), "black") for _ in range(n_views)])
    return draw_markers(probe, info)
