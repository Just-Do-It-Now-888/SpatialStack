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
