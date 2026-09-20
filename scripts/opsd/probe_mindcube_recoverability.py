#!/usr/bin/env python3
"""How much of the K=1 supervision is answerable at all? Measure, do not gate.

    python3 scripts/opsd/probe_mindcube_recoverability.py \
        --model output/20260906_qwen35_mindcube_sft_answeronly/checkpoint-314 \
        --out logs/eval/20260907_mindcube_mvopsd_k1/probe_nodeA.json

Runs the frozen initialisation over both MindCube val parquets -- full view set
and single view -- and reports the gap per family and per view count.

Why this exists, given the pool is already fixed by decision: MindCube's own
premise is that each camera "partially captures the surroundings", so the object
a question asks about may simply not be in the one frame the student keeps. Where
the evidence is absent the student can only learn a prior over answer letters,
and `vopd_loss` will fall contentedly while nothing is learned (LESSON-022; the
`spar_32view` imagination pool was abandoned once its teacher was found scoring
1.3%). This number is what any later gain has to be read against, so it is
recorded in the registry before the run rather than reconstructed after it.

Three outputs earn their place:

* **coverage x accuracy-on-answered.** A family can lose accuracy either by
  answering wrongly or by stopping to answer, and only the first is a statement
  about spatial reasoning (LESSON-011/018).
* **the predicted letter distribution against the gold one.** This is the direct
  evidence for "it is fitting a letter prior". A bucket whose predictions collapse
  onto one or two letters is answering from the prior no matter what its accuracy
  looks like.
* **the full-view score under vLLM.** The 74.48 / 69.81 anchors were produced by
  `transformers` generate; in-loop validation runs vLLM, and the two are known to
  differ by a small amount on this stack. Measuring the vLLM full-view number here
  is what gives the step-0 smoke gate a value it can actually be checked against,
  instead of an anchor from a different engine.

Expect `pair` rows at chance by construction -- their question is about a
transition between two views and they are shown one -- and `rotation` near it,
since its premise sentence is deleted at K=1. The teacher side needs no probe:
the frozen teacher is this same checkpoint with all views, which is the
full-view column below and the reason this round is not the failed SPAR attempt.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))
sys.path.insert(0, os.path.join(REPO_ROOT, "verl_pkg"))

from mindcube_scoring import extract_answer, score_mindcube_row  # noqa: E402

# The caps the pool, the val files and the anchor script all use. Passed to the
# processor as well as applied to the pixels, so vLLM's own preprocessing is an
# identity transform on what it is handed.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1605632


def load_requests(parquet_path: str, processor, limit: int = 0):
    """Turn a val parquet into vLLM requests, reusing verl's own image path.

    The images go through `verl.utils.dataset.vision_utils.process_image` rather
    than a bare `Image.open`, so the pixels here are the pixels in-loop
    validation will see -- the parquet's min/max caps included. Skipping that is
    how a probe ends up measuring a different resolution than the run it is
    meant to predict.
    """
    import pandas as pd

    from verl.utils.dataset.vision_utils import process_image

    frame = pd.read_parquet(parquet_path)
    if limit and limit < len(frame):
        frame = frame.head(limit)

    patch_size = processor.image_processor.patch_size
    requests, meta = [], []
    for row in frame.itertuples(index=False):
        images = [process_image(dict(image), image_patch_size=patch_size) for image in row.images]

        system_turn = [turn for turn in row.prompt if turn["role"] == "system"]
        user_turn = next(turn for turn in row.prompt if turn["role"] == "user")
        question = user_turn["content"].replace("<image>", "")

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": question})
        messages = [{"role": turn["role"], "content": turn["content"]} for turn in system_turn]
        messages.append({"role": "user", "content": content})

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        info = dict(row.extra_info)
        info["ground_truth"] = row.reward_model["ground_truth"]
        info["data_source"] = row.data_source
        meta.append(info)
    return requests, meta


def summarize(records: list[dict]) -> dict:
    """Overall, per family, per N and per (family, N), each decomposed."""

    def block(rows: list[dict]) -> dict:
        if not rows:
            return {}
        answered = [row for row in rows if row["answered"]]
        lengths = [row["output_tokens"] for row in rows]
        return {
            "n": len(rows),
            "acc": 100 * sum(row["acc"] for row in rows) / len(rows),
            # acc = coverage x acc_on_answered. Reported separately because a
            # bucket that stops answering and a bucket that answers wrongly are
            # different findings that look identical in `acc` alone.
            "coverage": 100 * len(answered) / len(rows),
            "acc_on_answered": (
                100 * sum(row["acc"] for row in answered) / len(answered) if answered else None
            ),
            # A collapse onto one or two letters is the signature of answering
            # from a prior rather than from the image.
            "pred_letters": dict(sorted(Counter(row["pred"] for row in rows).items(), key=str)),
            "gold_letters": dict(sorted(Counter(row["ground_truth"] for row in rows).items())),
            "output_tokens_median": statistics.median(lengths),
            "output_tokens_p95": sorted(lengths)[int(0.95 * len(lengths))],
        }

    by_family = defaultdict(list)
    by_views = defaultdict(list)
    by_cell = defaultdict(list)
    by_consistency = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
        by_views[int(record["n_views_available"])].append(record)
        by_cell[f"{record['family']}_{record['n_views_available']}view"].append(record)
        key = "prompt_matches_views" if record["prompt_matches_view_count"] else "pair_inconsistent"
        by_consistency[key].append(record)

    return {
        "overall": block(records),
        "by_family": {name: block(rows) for name, rows in sorted(by_family.items())},
        "by_n_views": {str(n): block(rows) for n, rows in sorted(by_views.items())},
        "by_family_and_n": {name: block(rows) for name, rows in sorted(by_cell.items())},
        # The headline with the knowingly-unanswerable rows removed. Every gain
        # this experiment reports has to be quoted both ways.
        "by_prompt_consistency": {name: block(rows) for name, rows in sorted(by_consistency.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--fullview-parquet", default="data/eval/mindcube_verl/mindcube_val.parquet"
    )
    parser.add_argument(
        "--singleview-parquet", default="data/eval/mindcube_verl/mindcube_val_singleview.parquet"
    )
    # 2048 is the anchor script's max_new_tokens; a different cap is a different
    # protocol, and the CoT arm's ~319-token median leaves plenty of headroom.
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=6656)  # 4608 prompt + 2048 response
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0, help="smoke runs only")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model_path = args.model if os.path.isabs(args.model) else os.path.join(REPO_ROOT, args.model)
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    budgets = {}
    for name, rel in (("fullview", args.fullview_parquet), ("singleview", args.singleview_parquet)):
        path = rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            raise SystemExit(f"missing {name} parquet: {path} (run build_mindcube_val_parquet.py)")
        budgets[name] = load_requests(path, processor, args.limit)
        print(f"{name}: {len(budgets[name][0])} rows from {path}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 4},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    )
    # Greedy, matching in-loop validation's default and the offline anchors.
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    report = {"model": model_path, "max_tokens": args.max_tokens, "budgets": {}}
    dumps = {}
    for name, (requests, meta) in budgets.items():
        print(f"\ngenerating {name} ({len(requests)} rows, greedy)...")
        outputs = llm.generate(requests, sampling_params=sampling)

        records: list[dict] = []
        for output, info in zip(outputs, meta):
            text = output.outputs[0].text
            accuracy, answered = score_mindcube_row(text, info["ground_truth"])
            records.append(
                {
                    "id": info["id"],
                    "family": info["family"],
                    "n_views_available": int(info["n_views_available"]),
                    "k_views_shown": int(info["k_views_shown"]),
                    "prompt_matches_view_count": bool(info["prompt_matches_view_count"]),
                    "ground_truth": info["ground_truth"],
                    "pred": extract_answer(text.strip()),
                    "acc": accuracy,
                    "answered": answered,
                    "output_tokens": len(output.outputs[0].token_ids),
                    "response": text,
                }
            )
        dumps[name] = records

    # prompt_matches_view_count is True for every full-view row, so bucketing each
    # budget by its own copy would compare a 1,050-row full-view bucket against a
    # 909-row single-view one and drop pair_inconsistent from the full-view side
    # entirely. The label is a property of the row under the K=1 protocol, so take
    # it from the single-view build and apply it to both budgets.
    consistency = {
        row["id"]: row["prompt_matches_view_count"] for row in dumps.get("singleview", [])
    }
    for name, records in dumps.items():
        missing = [row["id"] for row in records if row["id"] not in consistency]
        if missing:
            raise SystemExit(
                f"{name} has {len(missing)} rows absent from the single-view build "
                f"(e.g. {missing[:3]}); the two parquets must cover the same ids"
            )
        for row in records:
            row["prompt_matches_view_count"] = consistency[row["id"]]
        report["budgets"][name] = summarize(records)

    print_report(report)

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        rows_path = out_path.replace(".json", "_rows.json")
        with open(rows_path, "w") as handle:
            json.dump(dumps, handle, ensure_ascii=False, indent=2)
        print(f"\nsaved: {out_path}\n       {rows_path}")


def print_report(report: dict) -> None:
    full = report["budgets"].get("fullview", {})
    single = report["budgets"].get("singleview", {})

    print("\n=== single-view answer recoverability of the frozen init ===")
    print(f"model: {report['model']}")
    print(
        "\nThe full-view column is also the frozen teacher's accuracy, and it is the\n"
        "vLLM-engine value the step-0 in-loop gate should be checked against -- the\n"
        "74.48 / 69.81 anchors came from transformers generate, not vLLM.\n"
    )
    header = f"{'bucket':<26} {'n':>5} {'full':>7} {'1view':>7} {'gap':>7} {'cover':>7} {'on-ans':>7}"
    print(header)
    print("-" * len(header))

    def line(label: str, a: dict, b: dict) -> None:
        if not b:
            return
        if not a:
            print(f"{label:<26} {b['n']:>5} {'MISSING full-view counterpart':>7}")
            return
        on_answered = b["acc_on_answered"]
        on_answered_text = "    n/a" if on_answered is None else f"{on_answered:>6.2f}%"
        print(
            f"{label:<26} {b['n']:>5} {a['acc']:>6.2f}% {b['acc']:>6.2f}% "
            f"{b['acc'] - a['acc']:>+6.2f} {b['coverage']:>6.2f}% {on_answered_text}"
        )

    line("OVERALL", full.get("overall", {}), single.get("overall", {}))
    print()
    for group in ("by_family", "by_n_views", "by_family_and_n", "by_prompt_consistency"):
        for key in single.get(group, {}):
            line(f"  {group[3:]}: {key}", full.get(group, {}).get(key, {}), single[group][key])
        print()

    for name, budget in report["budgets"].items():
        overall = budget.get("overall", {})
        if not overall:
            continue
        print(
            f"{name:<11} predicted letters {overall['pred_letters']}\n"
            f"{'':<11} gold letters      {overall['gold_letters']}\n"
            f"{'':<11} output tokens     median {overall['output_tokens_median']} "
            f"p95 {overall['output_tokens_p95']}"
        )
    print(
        "\nRead the `pair_inconsistent` row as the floor this design accepts: those\n"
        "prompts ask about a transition between two views while showing one, so\n"
        "whatever they score is a letter prior, not spatial reasoning."
    )


if __name__ == "__main__":
    main()
