"""Smallest end-to-end check of geometry-model HF generate.

The MV-OPSD rollout cannot use vLLM, so generate() is on the critical path and
each failure there costs a full smoke launch. This runs one prompt with real VSI
frames on one GPU: enough to catch shape and transformers-version breakage.

  PYTHONPATH=verl_pkg:src:. python scripts/opsd/tools/check_geometry_generate.py
"""

from __future__ import annotations

import sys

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry
from verl.utils.qwen35_geometry import build_geometry_encoder_inputs, geometry_patch_size

MODEL = "output/spatialstack_qwen35_train"
VAL = "data/eval/vsibench_verl/vsibench_val_boxed_lastline.parquet"
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def main() -> int:
    row = pd.read_parquet(VAL).iloc[0]
    images = [Image.open(d["path"]).convert("RGB") for d in row["images"]][:N_FRAMES]

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    processed = processor(
        text=["".join(["<image>"] * len(images))], images=images, return_tensors="pt"
    )
    geo = torch.stack(
        build_geometry_encoder_inputs(
            images, processed["image_grid_thw"], patch_size=geometry_patch_size()
        )
    )
    print(f"frames={len(images)} grid={processed['image_grid_thw'][0].tolist()} geo={tuple(geo.shape)}")

    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        geometry_encoder_path="models/VGGT-1B",
        trust_remote_code=True,
        attn_implementation={"": "flash_attention_2", "model.visual": "sdpa"},
    ).to("cuda").eval()

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": "How many chairs are there? Answer with a number."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = processor(text=[prompt], images=images, return_tensors="pt")

    with torch.inference_mode():
        out = model.generate(
            input_ids=enc["input_ids"].cuda(),
            attention_mask=enc["attention_mask"].cuda(),
            pixel_values=enc["pixel_values"].cuda(),
            image_grid_thw=enc["image_grid_thw"].cuda(),
            geometry_encoder_inputs=[geo.cuda()],
            max_new_tokens=24,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[0, enc["input_ids"].shape[1] :]
    print("OK generated:", repr(tokenizer.decode(new, skip_special_tokens=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
