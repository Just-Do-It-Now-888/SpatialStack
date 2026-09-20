"""Build SpatialStack VGGT tensors for MV-OPSD student and teacher forwards.

Copied from lmms-eval's Qwen3.5 geometry helper so training workers do not
import lmms_eval. Views are resized to image_grid_thw * patch_size.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image


# VGGT's own patch size, not the Qwen vision tower's 16. The grid dims come from
# the vision tower, so a view is resized to grid * 14 -- src/qwen_vl/data/utils.py
# GEOMETRY_ENCODER_PATCH_SIZE and lmms_eval build_qwen3_5_geometry_inputs both
# use 14, and vggt/layers/patch_embed.py asserts on it.
GEOMETRY_ENCODER_PATCH_SIZE = 14


def geometry_patch_size(config=None) -> int:
    return GEOMETRY_ENCODER_PATCH_SIZE


def build_geometry_encoder_inputs(
    images: Iterable[Image.Image],
    image_grid_thw: torch.Tensor,
    patch_size: int = GEOMETRY_ENCODER_PATCH_SIZE,
) -> list[torch.Tensor]:
    geometry_tensors = []
    max_height = 0
    max_width = 0
    image_list = list(images)
    if not image_list:
        return []

    for image, grid in zip(image_list, image_grid_thw, strict=True):
        _, grid_h, grid_w = [int(v) for v in grid.tolist()]
        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        resized = rgb.resize((target_width, target_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        geometry_tensors.append(tensor)
        max_height = max(max_height, target_height)
        max_width = max(max_width, target_width)

    padded = []
    for tensor in geometry_tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        padded.append(tensor)
    return padded


def attach_geometry_encoder_inputs(
    multi_modal_inputs: dict[str, Any],
    images: Optional[list[Image.Image]],
    config=None,
    patch_size: Optional[int] = None,
) -> dict[str, Any]:
    if not images:
        return multi_modal_inputs
    grid = multi_modal_inputs.get("image_grid_thw")
    if grid is None:
        return multi_modal_inputs
    if grid.dim() == 1:
        grid = grid.unsqueeze(0)
    size = patch_size if patch_size is not None else geometry_patch_size(config)
    tensors = build_geometry_encoder_inputs(images, grid, patch_size=size)
    if not tensors:
        return multi_modal_inputs
    multi_modal_inputs["geometry_encoder_inputs"] = torch.stack(tensors)
    return multi_modal_inputs


def freeze_geometry_encoder(module) -> None:
    encoder = getattr(module, "geometry_encoder", None)
    inner = getattr(module, "model", None)
    if encoder is None and inner is not None:
        encoder = getattr(inner, "geometry_encoder", None)
    if encoder is None:
        return
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
