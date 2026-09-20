#!/usr/bin/env python3
"""SPAR-Bench through vLLM. Default protocol is lastline (no \\boxed{}).

    python3 scripts/opsd/tools/eval_sparbench_vllm.py \
        --model models/Qwen3.5-4B \
        --tensor-parallel-size 8 \
        --output-dir logs/eval/20260906_sparbench_vllm_lastline/base

Do not mix lastline scores with official HF 39.88 / 13.87, or boxed with lastline.
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

from PIL import Image

DEFAULT_ARROW = os.path.join(
    REPO_ROOT,
    "sparbench_cache/jasonzhango___spar-bench/default/0.0.0",
    "ee122877c25c8bb08539b07e06d872152c9968f1",
)
LMMS_KWARGS = {
    "pre_prompt": "",
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "na_post_prompt": "Please answer the question using a single word or phrase.",
}


def load_ds(arrow_dir: str):
    from datasets import Dataset, concatenate_datasets

    files = sorted(Path(arrow_dir).glob("spar-bench-test-*.arrow"))
    if not files:
        raise FileNotFoundError(f"no SPAR-Bench arrows in {arrow_dir}")
    return concatenate_datasets([Dataset.from_file(str(f)) for f in files])


def as_images(raw) -> list[Image.Image]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    out = []
    for im in raw:
        if not isinstance(im, Image.Image):
            im = Image.open(im)
        out.append(im.convert("RGB"))
    return out


def aggregate(rows: list[dict]) -> dict:
    from lmms_eval.tasks.sparbench.utils import High, Low, Middle

    by_task: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    trunc = 0
    boxed_present = 0
    tokens: list[int] = []
    for row in rows:
        by_task[row["task"]].append(row["score"])
        by_kind[row["kind"]].append(row["score"])
        trunc += int(row.get("truncated") or 0)
        boxed_present += int(bool(row.get("boxed_present")))
        tokens.append(int(row.get("output_tokens") or 0))
    task_mean = {t: (sum(v) / len(v)) for t, v in by_task.items()}
    overall = (sum(task_mean.values()) / len(task_mean)) if task_mean else 0.0
    fam_means: dict[str, list[float]] = defaultdict(list)
    for task, mean in task_mean.items():
        if task in Low:
            fam_means["Low"].append(mean)
        elif task in Middle:
            fam_means["Middle"].append(mean)
        elif task in High:
            fam_means["High"].append(mean)
        else:
            fam_means["Other"].append(mean)
    tokens_sorted = sorted(tokens)
    median_tokens = tokens_sorted[len(tokens_sorted) // 2] if tokens_sorted else 0
    return {
        "n": len(rows),
        "overall": round(100 * overall, 4),
        "truncated": trunc,
        "truncated_pct": round(100 * trunc / len(rows), 2) if rows else 0.0,
        "boxed_present": boxed_present,
        "boxed_present_pct": round(100 * boxed_present / len(rows), 2) if rows else 0.0,
        "output_tokens_median": median_tokens,
        "by_kind": {
            k: {"n": len(v), "mean": round(100 * sum(v) / len(v), 2)}
            for k, v in sorted(by_kind.items())
        },
        "by_family": {
            fam: {"task_mean": round(100 * sum(v) / len(v), 2), "n_tasks": len(v)}
            for fam, v in fam_means.items()
            if v
        },
        "by_task": {
            t: {"n": len(by_task[t]), "mean": round(100 * task_mean[t], 2)}
            for t in sorted(task_mean, key=lambda x: task_mean[x])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arrow-dir", default=DEFAULT_ARROW)
    parser.add_argument("--protocol", default="lastline", choices=("official", "lastline", "boxed"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--limit-images", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.environ["SPARBENCH_PROTOCOL"] = args.protocol
    from lmms_eval.tasks.sparbench import utils as spar
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    ds = load_ds(args.arrow_dir)
    n = len(ds) if not args.limit else min(args.limit, len(ds))
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(REPO_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

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
        limit_mm_per_prompt={"image": args.limit_images},
        gpu_memory_utilization=args.gpu_memory_utilization,
        mm_processor_kwargs={"max_pixels": args.max_pixels, "min_pixels": args.min_pixels},
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    def build_batch(start: int, end: int):
        requests, metas = [], []
        for i in range(start, end):
            doc = ds[i]
            images = as_images(doc.get("image"))
            if len(images) > args.limit_images:
                images = images[: args.limit_images]
            context = spar.sparbench_doc_to_text(doc, LMMS_KWARGS, protocol=args.protocol)
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
            mm = {"image": images} if images else None
            req = {"prompt": prompt}
            if mm:
                req["multi_modal_data"] = mm
            requests.append(req)
            metas.append(
                {
                    "id": int(doc["id"]),
                    "task": doc["task"],
                    "answer": doc["answer"],
                    "question": doc.get("question") or "",
                    "nimg": len(images),
                    "prompt": context,
                }
            )
        return requests, metas

    samples_path = os.path.join(out_dir, "samples.jsonl")
    rows: list[dict] = []
    started = time.perf_counter()
    indices = list(range(0, n, args.batch_size))
    print(
        f"[sparbench-vllm] protocol={args.protocol} n={n} max_tokens={args.max_tokens} "
        f"tp={args.tensor_parallel_size} model={args.model}",
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
                metric, parsed, kind = spar.score_sparbench_prediction(
                    meta["task"], text, meta["answer"], protocol=args.protocol
                )
                boxed_content = spar._shared().extract_boxed(text)
                rec = {
                    **meta,
                    "response": text,
                    "parsed": parsed,
                    "score": float(metric),
                    "kind": kind,
                    "truncated": truncated,
                    "finish_reason": completion.finish_reason,
                    "output_tokens": len(completion.token_ids or []),
                    "boxed_present": boxed_content is not None,
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
            "prompt_suffix": (
                spar.boxed_suffix()
                if args.protocol == "boxed"
                else (spar.LASTLINE_SUFFIX if args.protocol == "lastline" else "")
            ),
            "elapsed_sec": round(time.perf_counter() - started, 1),
        }
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({k: summary[k] for k in ("n", "overall", "truncated_pct", "by_kind")}, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
