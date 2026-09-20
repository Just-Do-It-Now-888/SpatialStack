"""Reproduce, offline, the per-view geometry that SpatialStack SFT feeds the model.

SFT never uses Qwen's ``max_pixels`` for image inputs: ``prepare_image_inputs``
(`src/qwen_vl/data/utils.py`) first runs the VGGT helper
``load_and_preprocess_images`` (width -> 518, height rounded to a multiple of 14,
centre crop to 518), then crops width/height down to a multiple of
``patch_size * merge_size``. For Qwen3.5 that factor is 32, which yields 384x512
(192 tokens) for 4:3 sources.

verl's teacher branch opens images with a bare ``Image.open`` and its student
branch runs ``smart_resize``. Both are identity transforms only when the file on
disk is already aligned to the factor and inside the pixel bounds, so the cache
this module writes is what makes teacher and student pixel-identical.
"""

from __future__ import annotations

import math

from PIL import Image

VGGT_TARGET_SIZE = 518
VGGT_ROUND_FACTOR = 14
# Qwen3.5 vision tower: patch_size=16, spatial_merge_size=2.
ALIGN_FACTOR = 32
MERGE_SIZE = 2
PATCH_SIZE = 16

# qwen_vl_utils.smart_resize bounds (IMAGE_MIN_TOKEN_NUM / IMAGE_MAX_TOKEN_NUM
# times the patch factor squared). verl's student path applies them; a cached
# view outside the range would be rescaled on the student side only.
SMART_RESIZE_MIN_PIXELS = 4 * ALIGN_FACTOR * ALIGN_FACTOR
SMART_RESIZE_MAX_PIXELS = 16384 * ALIGN_FACTOR * ALIGN_FACTOR


def sft_view_size(width: int, height: int) -> tuple[int, int]:
    """Return the (width, height) SFT would end up feeding the vision tower."""
    new_width = VGGT_TARGET_SIZE
    new_height = round(height * (new_width / width) / VGGT_ROUND_FACTOR) * VGGT_ROUND_FACTOR
    if new_height > VGGT_TARGET_SIZE:
        new_height = VGGT_TARGET_SIZE
    new_width -= new_width % ALIGN_FACTOR
    new_height -= new_height % ALIGN_FACTOR
    return new_width, new_height


def resize_to_sft_geometry(image: Image.Image) -> Image.Image:
    """Apply the SFT resize chain to a PIL image and return the aligned result."""
    image = image.convert("RGB")
    width, height = image.size
    new_width = VGGT_TARGET_SIZE
    new_height = round(height * (new_width / width) / VGGT_ROUND_FACTOR) * VGGT_ROUND_FACTOR
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    if new_height > VGGT_TARGET_SIZE:
        top = (new_height - VGGT_TARGET_SIZE) // 2
        image = image.crop((0, top, new_width, top + VGGT_TARGET_SIZE))

    width, height = image.size
    width -= width % ALIGN_FACTOR
    height -= height % ALIGN_FACTOR
    image = image.crop((0, 0, width, height))
    assert_aligned(image.size)
    return image


def pixel_budget_size(width: int, height: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """``smart_resize`` at the Qwen3.5 alignment factor: scale to fit, never crop.

    This is the transform ``resize_to_sft_geometry`` is *not*. The VGGT chain
    centre-crops height, which costs 24.5% of the frame on a 480x640 portrait
    image and so was switched off for the MindCube SFT round (registry
    20260906_qwen35_mindcube_sft). But simply copying the source instead is also
    wrong, and silently so: MindCube ships 480x640 (79% of views), 2016x1512,
    4032x3024 and six other sizes, and verl's two image paths disagree about
    them. The student runs ``fetch_image`` with the parquet's min/max pixels; the
    teacher runs a bare ``Image.open`` and inherits the processor's own bounds
    (65536 / 16777216). On a 4032x3024 source that is 1,568 student tokens
    against 11,844 teacher tokens per view -- resolution becomes a second
    privileged axis, and four such views overrun ``max_reprompt_len`` and get
    truncated with no error.

    Writing the cache at the student's own budget makes both paths identity
    transforms, which is the invariant this module exists to hold. It also
    reproduces what SFT fed: its processor applies the same rule at the same
    factor with the same bounds.
    """
    factor = ALIGN_FACTOR
    height_bar = max(factor, _round_by_factor(height, factor))
    width_bar = max(factor, _round_by_factor(width, factor))
    if height_bar * width_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        height_bar = max(factor, _floor_by_factor(height / beta, factor))
        width_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif height_bar * width_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        height_bar = _ceil_by_factor(height * beta, factor)
        width_bar = _ceil_by_factor(width * beta, factor)
    return width_bar, height_bar


def resize_to_pixel_budget(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """Resize into ``[min_pixels, max_pixels]`` with both sides aligned, no crop."""
    image = image.convert("RGB")
    width, height = pixel_budget_size(*image.size, min_pixels=min_pixels, max_pixels=max_pixels)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    assert_aligned(image.size)
    return image


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def assert_aligned(size: tuple[int, int]) -> None:
    width, height = size
    if width % ALIGN_FACTOR or height % ALIGN_FACTOR:
        raise AssertionError(
            f"cached view {width}x{height} is not a multiple of {ALIGN_FACTOR}; "
            "verl's student path would rescale it while the teacher path would not"
        )
    pixels = width * height
    if not SMART_RESIZE_MIN_PIXELS <= pixels <= SMART_RESIZE_MAX_PIXELS:
        raise AssertionError(
            f"cached view {width}x{height} ({pixels} px) is outside the smart_resize "
            f"pass-through range [{SMART_RESIZE_MIN_PIXELS}, {SMART_RESIZE_MAX_PIXELS}]"
        )


def visual_tokens(size: tuple[int, int]) -> int:
    width, height = size
    return (width // PATCH_SIZE) * (height // PATCH_SIZE) // (MERGE_SIZE**2)


def grid_thw(size: tuple[int, int]) -> tuple[int, int, int]:
    """Expected ``image_grid_thw`` for a cached view, for the P0-5 assertion."""
    width, height = size
    return 1, height // PATCH_SIZE, width // PATCH_SIZE
