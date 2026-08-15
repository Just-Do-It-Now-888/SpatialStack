"""Source registry and the SFT frame-sampling rule.

Sampling rates and frame selection are copied from the SFT run
(`scripts/train/train.sh`: ``spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,
vsi_appr_order%50``, ``--base_interval 2 --video_min_frames 4
--video_max_frames 8``) so that "we start from the same draw SFT started from"
is a checkable claim rather than a slogan.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# SFT's read_video_images() defaults, as passed by scripts/train/train.sh.
BASE_INTERVAL = 2
VIDEO_MIN_FRAMES = 4
VIDEO_MAX_FRAMES = 8

# VSI-Bench test scene that also appears in SPAR's ScanNet split (plan section 6.6).
LEAKED_SCENES = frozenset({"scene0685_00"})


@dataclass(frozen=True)
class Source:
    name: str
    annotation: str
    media_root: str
    sampling_rate: float
    sft_expected: Optional[int] = None  # samples SFT actually trained on, from its log

    def annotation_path(self) -> str:
        return os.path.join(REPO_ROOT, self.annotation)

    def media_path(self, relative: str) -> str:
        return os.path.join(REPO_ROOT, self.media_root, relative)


SOURCES: dict[str, Source] = {
    "spar_234k": Source(
        name="spar_234k",
        annotation="data/annotations/spar_234k.json",
        media_root="data/media",
        sampling_rate=0.60,
        sft_expected=140566,
    ),
    "llava_hound_64k": Source(
        name="llava_hound_64k",
        annotation="data/annotations/llava_hound_64k.json",
        media_root="data/media",
        sampling_rate=0.60,
        sft_expected=38250,
    ),
    "vlm3r_scannet": Source(
        name="vlm3r_scannet",
        annotation="data/annotations/merged_qa_scannet_train.json",
        media_root="data/vlm3r/media",
        sampling_rate=0.60,
        sft_expected=31067,
    ),
    "vsi_appr_order": Source(
        name="vsi_appr_order",
        annotation="data/annotations/vsi_appearance_order_vsibench_scannet.json",
        media_root="data/vsi_590k/media",
        sampling_rate=0.50,
        sft_expected=1909,
    ),
}


def sample_like_sft(annotations: list, rate: float, seed: int) -> list:
    """SFT draws ``int(len * rate)`` items with an unseeded ``random.sample``.

    We keep the count identical and only add a seed, so the draw becomes a data
    artifact instead of a per-rank runtime accident.
    """
    if rate >= 1.0:
        return list(annotations)
    return random.Random(seed).sample(annotations, int(len(annotations) * rate))


def sft_frame_indices(total_frames: int, fps: float) -> list[int]:
    """The frame indices SFT's ``read_video_images`` would pick."""
    video_length = total_frames / fps
    num_to_sample = round(video_length / BASE_INTERVAL)
    target = min(max(num_to_sample, VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES)
    indices = np.linspace(0, total_frames - 1, target, dtype=int)
    return np.unique(indices).tolist()


def probe_video(path: str) -> tuple[int, float]:
    """Return (total_frames, average_fps); matches decord's view of the file."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        total = stream.frames
        if not total:
            total = sum(1 for _ in container.decode(video=0))
    if total <= 0:
        raise ValueError(f"could not determine frame count for {path}")
    return total, fps


def list_frame_dir(path: str) -> list[str]:
    """Frame files of a pre-extracted video directory, in SFT's sorted order."""
    names = sorted(name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)))
    return [os.path.join(path, name) for name in names]
