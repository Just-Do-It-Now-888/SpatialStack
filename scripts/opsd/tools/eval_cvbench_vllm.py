#!/usr/bin/env python3
"""CV-Bench eval through vLLM, matching MV-OPSD in-training validation settings.

Uses the same val parquet, greedy decoding, max_tokens=1024, and enable_thinking=False
as ``mvopsd.yaml`` / ``run_mvopsd.sh``.

    python3 scripts/opsd/tools/eval_cvbench_vllm.py \
        --model models/Qwen3.5-4B \
        --output logs/eval/qwen35base_vllm_cvbench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))

from cvbench_scoring import score_cvbench_row  # noqa: E402
from verl.trainer.ppo.cvbench_metrics import compute_cvbench_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(REPO_ROOT, "models/Qwen3.5-4B"))
    parser.add_argument("--parquet", default=os.path.join(REPO_ROOT, "data/eval/cvbench_verl/cvbench_val.parquet"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument(
        "--parser",
        default="boxed_lastline",
        choices=("boxed_lastline", "word_boundary", "lmms_legacy", "lastline"),
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="appended to the parquet question text, e.g. SPAR lastline suffix",
    )
    parser.add_argument("--output-dir", default=None, help="writes summary.json and samples.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import pandas as pd
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    frame = pd.read_parquet(args.parquet)
    if args.limit and args.limit < len(frame):
        frame = frame.head(args.limit)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    requests = []
    meta = []
    for _, row in frame.iterrows():
        image_info = row["images"][0]
        image_path = image_info["path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(REPO_ROOT, image_path)
        question = row["prompt"][0]["content"].replace("<image>", "", 1)
        if args.prompt_suffix:
            question = question.rstrip() + args.prompt_suffix
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image = Image.open(image_path).convert("RGB")
        requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        meta.append(
            {
                "data_source": row["data_source"],
                "ground_truth": row["reward_model"]["ground_truth"],
                "question": question,
            }
        )

    print(f"generating {len(requests)} rows with vLLM (max_tokens={args.max_tokens})...")
    outputs = llm.generate(requests, sampling_params=sampling)

    data_sources, accuracies, answered, rows_out = [], [], [], []
    for output, info in zip(outputs, meta):
        completion = output.outputs[0]
        text = completion.text
        accuracy, was_answered = score_cvbench_row(text, info["ground_truth"], parser=args.parser)
        data_sources.append(info["data_source"])
        accuracies.append(accuracy)
        answered.append(was_answered)
        rows_out.append(
            {
                "data_source": info["data_source"],
                "ground_truth": info["ground_truth"],
                "prompt": info.get("question"),
                "response": text,
                "output": text,
                "acc": accuracy,
                "answered": was_answered,
                "truncated": int(completion.finish_reason == "length"),
                "finish_reason": completion.finish_reason,
                "output_tokens": len(completion.token_ids or []),
            }
        )

    metrics = compute_cvbench_metrics(data_sources, accuracies, answered)
    combined = metrics.get("val-core/cvbench/combined/acc")
    answered_frac = metrics.get("val-core/cvbench/answered/frac")
    acc_2d = metrics.get("val-aux/cvbench/2d/acc")
    acc_3d = metrics.get("val-aux/cvbench/3d/acc")
    n = len(rows_out)
    trunc = sum(int(r["truncated"]) for r in rows_out)
    print(f"\n=== vLLM CV-Bench ({args.parser}) ===")
    print(f"combined: {100 * combined:.2f}%" if combined is not None else "combined: n/a")
    print(f"answered: {100 * answered_frac:.2f}%" if answered_frac is not None else "answered: n/a")
    for key in sorted(metrics):
        if key.startswith("val-"):
            print(f"  {key}: {metrics[key]:.4f}")

    summary = {
        "n": n,
        "parser": args.parser,
        "prompt_suffix": args.prompt_suffix,
        "engine": "vllm",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "overall": round(100 * combined, 2) if combined is not None else None,
        "acc_2d": round(100 * acc_2d, 2) if acc_2d is not None else None,
        "acc_3d": round(100 * acc_3d, 2) if acc_3d is not None else None,
        "answered_pct": round(100 * answered_frac, 2) if answered_frac is not None else None,
        "truncated_pct": round(100 * trunc / n, 2) if n else 0.0,
        "metrics": {k: metrics[k] for k in sorted(metrics)},
    }

    if args.output_dir:
        out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(REPO_ROOT, args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        samples_path = os.path.join(out_dir, "samples.jsonl")
        with open(samples_path, "w") as handle:
            for rec in rows_out:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"\nsaved: {out_dir}/summary.json")

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_ROOT, args.output)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump({"metrics": metrics, "rows": rows_out, "summary": summary}, handle, ensure_ascii=False, indent=2)
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
