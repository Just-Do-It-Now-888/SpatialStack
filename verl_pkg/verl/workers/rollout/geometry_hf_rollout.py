"""HF generate rollout for SpatialStack geometry models (no vLLM).

Each FSDP rank holds a full-weight generate copy, hybrid-synced from the actor
the same way vLLM is. AgentLoop talks to the worker's generate() method.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generator, Optional

import torch
from PIL import Image
from transformers import GenerationConfig

from verl.utils.qwen35_geometry import (
    attach_geometry_encoder_inputs,
    freeze_geometry_encoder,
)
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica, TokenOutput


def _as_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if "image" in image and isinstance(image["image"], Image.Image):
            return image["image"].convert("RGB")
        if "path" in image:
            return Image.open(image["path"]).convert("RGB")
        if "bytes" in image:
            from io import BytesIO

            return Image.open(BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"unsupported image type {type(image)}")


class GeometryHFAsyncRollout(BaseRollout):
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config, model_config, device_mesh)
        self.model = None
        self.processor = getattr(model_config, "processor", None) or getattr(model_config, "hf_processor", None)
        self.tokenizer = getattr(model_config, "tokenizer", None) or getattr(model_config, "hf_tokenizer", None)
        self._device = torch.device("cpu")
        self._on_gpu = False
        self._load_cpu_model()

    def _model_path(self) -> str:
        path = getattr(self.model_config, "local_path", None) or getattr(self.model_config, "path", None)
        if path is None:
            path = getattr(self.config, "model_path", None)
        if not path:
            raise ValueError("Geometry HF rollout needs actor_rollout_ref.model.path")
        return str(path)

    def _load_cpu_model(self) -> None:
        from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

        local_path = self._model_path()
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
        geo_path = getattr(config, "geometry_encoder_path", None)
        self.model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            geometry_encoder_path=geo_path,
            trust_remote_code=True,
            attn_implementation={"": "flash_attention_2", "model.visual": "sdpa"},
        )
        freeze_geometry_encoder(self.model)
        self.model.eval()
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    async def resume(self, tags: list[str]):
        if self.model is None:
            return
        if not self._on_gpu:
            self._device = torch.device("cuda")
            self.model.to(self._device)
            self._on_gpu = True

    async def release(self):
        if self.model is None or not self._on_gpu:
            return
        self.model.to("cpu")
        self._device = torch.device("cpu")
        self._on_gpu = False
        torch.cuda.empty_cache()

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        if self.model is None:
            return
        named = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        with torch.no_grad():
            for name, tensor in weights:
                target = named.get(name)
                if target is None:
                    target = buffers.get(name)
                if target is None:
                    continue
                target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs,
    ) -> TokenOutput:
        if self.model is None:
            raise RuntimeError("geometry HF rollout model is not loaded")
        if not self._on_gpu:
            await self.resume(tags=["weights"])

        sampling = dict(sampling_params)
        max_new = int(sampling.pop("max_tokens", None) or sampling.pop("max_new_tokens", None) or self.config.response_length)
        temperature = float(sampling.get("temperature", self.config.temperature))
        top_p = float(sampling.get("top_p", self.config.top_p))
        do_sample = temperature > 0

        images = [_as_pil(img) for img in (image_data or [])]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        if images and self.processor is not None:
            # Do not decode prompt_ids: they already contain expanded image-pad tokens.
            # Qwen3 VL processor then treats those as extra image slots and indexes
            # past image_grid_thw (32-frame VSI: IndexError index 32 / size 32).
            placeholder = "".join(["<image>"] * len(images))
            processed = self.processor(
                text=[placeholder],
                images=images,
                videos=None,
                return_tensors="pt",
            )
            pixel_values = processed.get("pixel_values")
            image_grid_thw = processed.get("image_grid_thw")
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values.to(self._device)
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw.to(self._device)
            mm = {k: v for k, v in processed.items() if k not in {"input_ids", "attention_mask"}}
            attach_geometry_encoder_inputs(mm, images, config=getattr(self.model, "config", None))
            geo = mm.get("geometry_encoder_inputs")
            if geo is not None:
                gen_kwargs["geometry_encoder_inputs"] = [geo.to(self._device)]

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
        eos_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)
        response = output_ids[0, input_ids.shape[1] :].tolist()
        return TokenOutput(token_ids=response, log_probs=None)


class GeometryHFReplica(RolloutReplica):
    def get_ray_class_with_init_args(self):
        raise NotImplementedError("Geometry HF rollout only supports hybrid mode on the actor workers")

    async def launch_servers(self):
        self.servers = list(self.workers)
        self._server_handle = self.workers[0]
        self._server_address = f"geometry-hf://replica{self.replica_rank}"

    async def clear_kv_cache(self):
        return
