#!/usr/bin/env python3
"""BLINK (14-task val) through vLLM. Do not mix with HF/lmms-eval spatial dumps.

    python3 scripts/opsd/tools/eval_blink_vllm.py \
        --model models/Qwen3.5-4B \
        --protocol spatialstack \
        --tensor-parallel-size 8 \
        --output-dir logs/eval/20260908_blink_full_vllm/base_spatialstack
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

CACHE_ROOT = os.path.join(REPO_ROOT, "cache", "datasets", "BLINK-Benchmark__BLINK")

TASKS = [
    ("blink_art_style", "Art_Style"),
    ("blink_counting", "Counting"),
    ("blink_forensic_detection", "Forensic_Detection"),
    ("blink_functional_correspondence", "Functional_Correspondence"),
    ("blink_iq_test", "IQ_Test"),
    ("blink_jigsaw", "Jigsaw"),
    ("blink_multi_view_reasoning", "Multi-view_Reasoning"),
    ("blink_object_localization", "Object_Localization"),
    ("blink_relative_depth", "Relative_Depth"),
    ("blink_relative_reflectance", "Relative_Reflectance"),
    ("blink_semantic_correspondence", "Semantic_Correspondence"),
    ("blink_spatial_relation", "Spatial_Relation"),
    ("blink_visual_correspondence", "Visual_Correspondence"),
    ("blink_visual_similarity", "Visual_Similarity"),
]
SPATIAL_TASKS = (
    "blink_multi_view_reasoning",
    "blink_relative_depth",
    "blink_spatial_relation",
)
LMMS_KWARGS = {
    "pre_prompt": (
        "Return exactly one uppercase option letter from the given choices ({}). "
        "Do not output any explanation, punctuation, or extra text.\n"
    ),
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "post_prompt": "",
}


def load_val(config_name: str, cache_root: str):
    from datasets import Dataset, DatasetDict, load_from_disk

    path = os.path.join(cache_root, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing BLINK cache {path}")
    loaded = load_from_disk(path)
    if isinstance(loaded, Dataset):
        return loaded
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"unexpected BLINK cache type {type(loaded)} at {path}")
    if "val" in loaded:
        return loaded["val"]
    split = next(iter(loaded.keys()))
    return loaded[split]


def aggregate(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    def task_stats(task_rows: list[dict]) -> dict:
        n = len(task_rows)
        acc = sum(r["score"] for r in task_rows) / n if n else 0.0
        answered = sum(int(r.get("answered") or 0) for r in task_rows)
        trunc = sum(int(r.get("truncated") or 0) for r in task_rows)
        boxed = sum(int(r.get("boxed_present") or 0) for r in task_rows)
        return {
            "n": n,
            "acc": round(100 * acc, 2),
            "answered_pct": round(100 * answered / n, 2) if n else 0.0,
            "truncated_pct": round(100 * trunc / n, 2) if n else 0.0,
            "boxed_present_pct": round(100 * boxed / n, 2) if n else 0.0,
        }

    by_task_out = {task: task_stats(by_task[task]) for task in sorted(by_task)}
    task_accs = [v["acc"] for v in by_task_out.values()]
    spatial_accs = [by_task_out[t]["acc"] for t in SPATIAL_TASKS if t in by_task_out]
    n = len(rows)
    trunc = sum(int(r.get("truncated") or 0) for r in rows)
    answered = sum(int(r.get("answered") or 0) for r in rows)
    boxed = sum(int(r.get("boxed_present") or 0) for r in rows)
    return {
        "n": n,
        "n_tasks": len(by_task_out),
        "overall": round(sum(task_accs) / len(task_accs), 2) if task_accs else 0.0,
        "spatial_macro": round(sum(spatial_accs) / len(spatial_accs), 2) if spatial_accs else None,
        "truncated": trunc,
        "truncated_pct": round(100 * trunc / n, 2) if n else 0.0,
        "answered_pct": round(100 * answered / n, 2) if n else 0.0,
        "boxed_present_pct": round(100 * boxed / n, 2) if n else 0.0,
        "by_task": by_task_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", default="spatialstack", choices=("original", "spatialstack", "lastline"))
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--tasks", default="", help="comma-separated HF config names; default all 14")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--limit-images", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="per-task cap; 0 = all")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.environ["BLINK_PROTOCOL"] = args.protocol
    from lmms_eval.tasks.blink import utils as blink
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
    task_list = [(tid, cfg) for tid, cfg in TASKS if not wanted or cfg in wanted or tid in wanted]
    if not task_list:
        raise SystemExit(f"no tasks matched {args.tasks!r}")

    examples: list[tuple[str, str, dict]] = []
    for task_id, config_name in task_list:
        ds = load_val(config_name, args.cache_root)
        n = len(ds) if not args.limit else min(args.limit, len(ds))
        for i in range(n):
            examples.append((task_id, config_name, ds[i]))
    # BLINK val is at most 4 images/row (evalscope stats). Do not decode the
    # whole set before the engine starts — that plus a second vLLM init caused
    # SIGBUS (rc=135) when the previous arm had just torn down NCCL.
    limit_mm = max(int(args.limit_images), 1)
    max_images = limit_mm

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    n = len(examples)

    processor = AutoProcessor.from_pretrained(
        args.model,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        trust_remote_code=True,
    )
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": limit_mm},
        gpu_memory_utilization=args.gpu_memory_utilization,
        mm_processor_kwargs={"max_pixels": args.max_pixels, "min_pixels": args.min_pixels},
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    def build_batch(start: int, end: int):
        requests, metas = [], []
        for i in range(start, end):
            task_id, config_name, doc = examples[i]
            images = blink.blink_doc_to_visual(doc)
            if len(images) > args.limit_images:
                raise RuntimeError(
                    f"{doc.get('idx')} has {len(images)} images > --limit-images {args.limit_images}"
                )
            context = blink.blink_doc_to_text(doc, LMMS_KWARGS)
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": context})
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            req = {"prompt": prompt}
            if images:
                req["multi_modal_data"] = {"image": images}
            requests.append(req)
            metas.append(
                {
                    "id": doc["idx"],
                    "task": task_id,
                    "config": config_name,
                    "sub_task": doc.get("sub_task"),
                    "answer": doc["answer"],
                    "choices": list(doc.get("choices") or []),
                    "nimg": len(images),
                    "prompt": context,
                    "doc": {
                        "idx": doc["idx"],
                        "answer": doc["answer"],
                        "choices": list(doc.get("choices") or []),
                        "sub_task": doc.get("sub_task"),
                        "prompt": doc.get("prompt"),
                    },
                }
            )
        return requests, metas

    samples_path = os.path.join(out_dir, "samples.jsonl")
    rows: list[dict] = []
    started = time.perf_counter()
    indices = list(range(0, n, args.batch_size))
    print(
        f"[blink-vllm] protocol={args.protocol} n={n} tasks={len(task_list)} "
        f"max_images={max_images} max_tokens={args.max_tokens} tp={args.tensor_parallel_size} "
        f"model={args.model}",
        flush=True,
    )

    with open(samples_path, "w") as handle, ThreadPoolExecutor(max_workers=1) as pool:
        first = indices[0] if indices else None
        pending = (
            pool.submit(build_batch, first, min(first + args.batch_size, n))
            if first is not None
            else None
        )
        for bi, start in enumerate(indices):
            requests, metas = pending.result()
            nxt = indices[bi + 1] if bi + 1 < len(indices) else None
            pending = (
                pool.submit(build_batch, nxt, min(nxt + args.batch_size, n)) if nxt is not None else None
            )
            outputs = llm.generate(requests, sampling_params=sampling)
            for output, meta in zip(outputs, metas):
                completion = output.outputs[0]
                text = completion.text
                truncated = int(completion.finish_reason == "length")
                scored = blink.blink_process_results(meta["doc"], [text])["blink_acc"]
                rec = {
                    "id": meta["id"],
                    "task": meta["task"],
                    "config": meta["config"],
                    "sub_task": meta["sub_task"],
                    "answer": meta["answer"],
                    "choices": meta["choices"],
                    "nimg": meta["nimg"],
                    "prompt": meta["prompt"],
                    "response": text,
                    "parsed": scored.get("pred_parsed"),
                    "score": int(bool(scored.get("is_correct"))),
                    "answered": int(scored.get("answered") or 0),
                    "boxed_present": int(scored.get("boxed_present") or 0),
                    "truncated": truncated,
                    "finish_reason": completion.finish_reason,
                    "output_tokens": len(completion.token_ids or []),
                }
                rows.append(rec)
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            done = len(rows)
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            eta = (n - done) / rate / 60 if rate else 0.0
            print(
                f"[{bi + 1}/{len(indices)}] {done}/{n} "
                f"{rate:.2f} q/s elapsed {elapsed / 60:.1f}m eta {eta:.1f}m",
                flush=True,
            )

    summary = aggregate(rows)
    summary.update(
        {
            "protocol": args.protocol,
            "engine": "vllm",
            "model": args.model,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "disable_thinking": True,
            "max_images": max_images,
            "elapsed_sec": round(time.perf_counter() - started, 1),
        }
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("n", "n_tasks", "overall", "spatial_macro", "truncated_pct", "answered_pct")
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
