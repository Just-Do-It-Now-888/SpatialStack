"""Ask the frozen MV-OPSD teacher the training questions and record what it says.

Self-distillation copies the teacher's N-view posterior onto the student. That
only transfers spatial ability if the posterior is right, and nothing in the
training loop ever checks: ``vopd_loss`` measures agreement with the teacher,
not agreement with the ground truth (LESSON-012). This module supplies the
missing measurement by generating from the teacher's own inputs.

The point of the exercise is destroyed by any protocol drift, so the prompt is
built the way ``ray_trainer._build_teacher_prompt_inputs`` builds it, not the
way the benchmark harnesses build theirs:

* messages come from the parquet's ``teacher_prompt`` column verbatim, so there
  is **no system turn**. VSI-Bench adds "You are a helpful assistant." and it
  moved the base model by ~1 point (ISSUE-203), but the teacher has never seen
  one and adding it here would measure a model the training never used.
* ``enable_thinking=False``, from ``data.apply_chat_template_kwargs``.
* images are handed to the processor at their cached size with **no**
  min_pixels/max_pixels override. The cache is already patch-aligned at 512x384
  (192 visual tokens per view) and the released preprocessor_config leaves
  anything between 65,536 and 16.7M pixels alone, so training does no resizing
  and neither can this.

Generation is the expensive, irreversible half and scoring is neither, so this
module only generates and dumps. ``tools/score_teacher_dump.py`` grades the dump
afterwards and can be re-run as the parsers improve.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Teacher prompts peak at 6,540 tokens (32 views x 192 + question); the training
# config caps them at max_reprompt_len=12288 and never truncates in practice.
DEFAULT_MAX_MODEL_LEN = 12288
# data.max_response_length, i.e. the budget the student had. The teacher is not
# the student, but a shorter budget here would report "teacher cannot answer"
# for rows that merely ran out of room.
DEFAULT_MAX_TOKENS = 1024
# actor_rollout_ref.rollout.limit_images, so a 32-view album fits with headroom.
DEFAULT_LIMIT_IMAGES = 40

# PIL decodes each cached view to ~0.6 MB, so a chunk is sized by image count
# rather than row count: 8-view and 32-view rows would otherwise differ 4x in
# resident memory for the same batch size.
DEFAULT_IMAGES_PER_CHUNK = 6000


@dataclass
class GenConfig:
    parquet: str
    model: str = "./models/Qwen3.5-4B"
    output_dir: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    limit_images: int = DEFAULT_LIMIT_IMAGES
    tensor_parallel_size: int = 8
    gpu_memory_utilization: float = 0.85
    images_per_chunk: int = DEFAULT_IMAGES_PER_CHUNK
    limit_rows: int = 0
    # Smoke runs: a random draw rather than the first N rows, which are all one
    # source and one view count and so exercise none of the interesting paths.
    sample_rows: int = 0
    sample_seed: int = 20260820
    # Views the teacher is shown. "all" reproduces training. An integer runs the
    # controlled sweep: the same question with a truncated album.
    prompt_column: str = "teacher_prompt"
    image_column: str = "teacher_images"
    enable_thinking: bool = False
    # Appended verbatim to the last message's text, after the question. Empty
    # leaves every prompt byte-identical to the runs already archived under
    # logs/eval/teacher_reliability/, so a dump taken with a suffix can be
    # compared row by row against those as a paired control.
    prompt_suffix: str = ""
    seed: int = 20260816
    resume: bool = True
    carry_fields: list[str] = field(
        default_factory=lambda: [
            "sample_id",
            "source",
            "dataset",
            "subset",
            "scene_id",
            "question_type",
            "n_views_teacher",
            "k_views_student",
            "view_selection",
            "privilege_bucket",
            "answer_view_sensitive",
            "sweep_views",
            "sweep_group",
            "sweep_arm",
            "required_views",
            "view_indices_teacher",
            "extra_added",
            "n_key_frames",
        ]
    )


def load_rows(cfg: GenConfig) -> list[dict]:
    """Flatten the parquet into plain dicts: prompt text, image paths, gold, tags."""
    import pandas as pd

    path = cfg.parquet if os.path.isabs(cfg.parquet) else os.path.join(REPO_ROOT, cfg.parquet)
    frame = pd.read_parquet(path)
    if cfg.limit_rows:
        frame = frame.iloc[: cfg.limit_rows]
    # row_index is assigned before subsampling so a smoke dump and the full dump
    # use the same key space and can be compared row by row.
    frame = frame.reset_index(drop=True)
    keep = None
    if cfg.sample_rows and cfg.sample_rows < len(frame):
        keep = set(frame.sample(n=cfg.sample_rows, random_state=cfg.sample_seed).index)

    rows: list[dict] = []
    for position, record in enumerate(frame.to_dict(orient="records")):
        if keep is not None and position not in keep:
            continue
        messages = [dict(message) for message in record[cfg.prompt_column]]
        if cfg.prompt_suffix:
            messages[-1]["content"] = str(messages[-1]["content"]) + cfg.prompt_suffix
        images = [str(image["path"]) for image in record[cfg.image_column]]
        extra = dict(record.get("extra_info") or {})
        placeholders = sum(str(message.get("content", "")).count("<image>") for message in messages)
        if placeholders != len(images):
            raise AssertionError(
                f"row {position} ({extra.get('sample_id')}): {placeholders} <image> "
                f"placeholders but {len(images)} images"
            )
        row = {
            "row_index": position,
            "data_source": record["data_source"],
            "messages": messages,
            "image_paths": images,
            "ground_truth": str((record.get("reward_model") or {}).get("ground_truth", "")),
            "n_images": len(images),
        }
        for key in cfg.carry_fields:
            if key in extra:
                value = extra[key]
                if hasattr(value, "tolist"):
                    row[key] = value.tolist()
                elif hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
                    row[key] = value.item()
                else:
                    row[key] = value
        row.setdefault("sample_id", f"row{position}")
        rows.append(row)
    return rows


def chunk_rows(rows: list[dict], images_per_chunk: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    budget = 0
    for row in rows:
        if current and budget + row["n_images"] > images_per_chunk:
            chunks.append(current)
            current, budget = [], 0
        current.append(row)
        budget += row["n_images"]
    if current:
        chunks.append(current)
    return chunks


def done_ids(path: str) -> set[int]:
    """Rows already generated, so an interrupted run resumes instead of restarting.

    Keyed on ``row_index`` rather than ``sample_id``: 661 of the 124,306 pool
    rows share a sample_id (the same SPAR question drawn under two view
    subsets), and resuming on that key would silently drop the duplicates.
    """
    if not os.path.exists(path):
        return set()
    seen: set[int] = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(int(json.loads(line)["row_index"]))
            except (ValueError, KeyError):
                # A run killed mid-write leaves one partial line; everything
                # before it is still usable.
                continue
    return seen


def _messages_with_images(messages: list[dict], images: list) -> list[dict]:
    """Splice PIL images into ``<image>`` placeholders, as ray_trainer.py:769 does."""
    import re

    out: list[dict] = []
    offset = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            out.append(message)
            continue
        parts: list[dict] = []
        for segment in (piece for piece in re.split(r"(<image>)", content) if piece != ""):
            if segment == "<image>":
                parts.append({"type": "image", "image": images[offset]})
                offset += 1
            else:
                parts.append({"type": "text", "text": segment})
        out.append({"role": message.get("role", "user"), "content": parts})
    if offset != len(images):
        raise ValueError(f"spliced {offset} images but got {len(images)}")
    return out


def run_generation(cfg: GenConfig, *, progress: bool = True) -> str:
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out_dir = cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(REPO_ROOT, cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    dump_path = os.path.join(out_dir, "generations.jsonl")

    rows = load_rows(cfg)
    total_rows = len(rows)
    if cfg.resume:
        already = done_ids(dump_path)
        if already:
            rows = [row for row in rows if row["row_index"] not in already]
            print(f"resume: {len(already)} rows already generated, {len(rows)} to go", flush=True)
    else:
        if os.path.exists(dump_path):
            os.remove(dump_path)

    if not rows:
        print("nothing to generate", flush=True)
        return dump_path

    # No min_pixels/max_pixels: training passes none, and the cached views are
    # already the size the teacher saw.
    processor = AutoProcessor.from_pretrained(cfg.model, trust_remote_code=True)
    llm = LLM(
        model=cfg.model,
        trust_remote_code=True,
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        limit_mm_per_prompt={"image": cfg.limit_images},
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        seed=cfg.seed,
    )
    sampling = SamplingParams(max_tokens=cfg.max_tokens, temperature=0.0, top_p=1.0)

    def build_chunk(chunk: list[dict]):
        requests = []
        for row in chunk:
            images = [Image.open(path).convert("RGB") for path in row["image_paths"]]
            messages = _messages_with_images(row["messages"], images)
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=cfg.enable_thinking,
            )
            requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        return requests

    chunks = chunk_rows(rows, cfg.images_per_chunk)
    started = time.perf_counter()
    done = 0

    with open(dump_path, "a") as handle, ThreadPoolExecutor(max_workers=1) as prefetcher:
        pending = prefetcher.submit(build_chunk, chunks[0])
        for index, chunk in enumerate(chunks):
            requests = pending.result()
            pending = prefetcher.submit(build_chunk, chunks[index + 1]) if index + 1 < len(chunks) else None
            outputs = llm.generate(requests, sampling_params=sampling)
            for output, row in zip(outputs, chunk):
                completion = output.outputs[0]
                record = {key: value for key, value in row.items() if key not in ("messages", "image_paths")}
                record["prompt_tokens"] = len(output.prompt_token_ids or [])
                record["response"] = completion.text
                record["output_tokens"] = len(completion.token_ids)
                record["finish_reason"] = completion.finish_reason
                record["truncated"] = int(completion.finish_reason == "length")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done += len(chunk)
            if progress:
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (len(rows) - done) / rate / 60 if rate else 0.0
                print(
                    f"[{index + 1}/{len(chunks)}] {done}/{len(rows)} rows "
                    f"({total_rows} total) | {rate:.1f} row/s | "
                    f"elapsed {elapsed / 60:.1f}m | eta {remaining:.1f}m",
                    flush=True,
                )

    with open(os.path.join(out_dir, "gen_config.json"), "w") as handle:
        json.dump(
            {
                "parquet": cfg.parquet,
                "model": cfg.model,
                "max_tokens": cfg.max_tokens,
                "max_model_len": cfg.max_model_len,
                "prompt_column": cfg.prompt_column,
                "image_column": cfg.image_column,
                "enable_thinking": cfg.enable_thinking,
                "prompt_suffix": cfg.prompt_suffix,
                "sample_rows": cfg.sample_rows,
                "sample_seed": cfg.sample_seed,
                "system_turn": None,
                "tensor_parallel_size": cfg.tensor_parallel_size,
                "rows": total_rows,
                "wall_seconds": time.perf_counter() - started,
            },
            handle,
            indent=2,
        )
    return dump_path
