#!/usr/bin/env python3
"""Build LLaVA-style MindCube SFT annotations for two supervision arms.

Arm A (answer-only): target is `<answer>{letter}. {option text}</answer>`.
Arm B (text CoT):    target is the 3DThinker reasoning body followed by the same
                     `<answer>...</answer>` line.

Both files hold the same questions in the same order, so the only variable
between the two arms is the target text.

Sources
  MindCube_train.jsonl        authoritative gt_answer + original image paths
  MindCube_tinybench_raw_qa   the eval-time prompt template (asserted constant)
  idx.jsonl                   answer-only target, prompt with <image> tokens
  data_output3d_*.jsonl       CoT target (`text_output`)

`idx` is an index into a 3DThinker-internal ordering that does not correspond to
any file we have, so rows are joined to MindCube_train by the composite key
(question, image sub-paths). That key is unique across all 10000 train rows,
while `question` alone is not (8104/10000 distinct).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

THINKER = "/home/c30084464/Documents/code/3DThinker"
TRAIN = f"{THINKER}/MindCube-main/data/raw/MindCube_train.jsonl"
TINYBENCH = f"{THINKER}/MindCube-main/data/prompts/general/MindCube_tinybench_raw_qa.jsonl"
IDX = f"{THINKER}/data/idx.jsonl"
COT = f"{THINKER}/data/data_output3d_begin_10k_resized.jsonl"
IMAGE_ROOT = f"{THINKER}/MindCube-main/data"

COT_RE = re.compile(r"^<output_3D>\n<think>(.*?)</think>\n(<answer>.*?</answer>)$", re.S)
ANSWER_RE = re.compile(r"^<answer>(.*)</answer>$", re.S)


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def image_subpath(path):
    """Strip the image-root prefix so resized and original paths share a key."""
    for marker in ("other_all_image_resize/", "other_all_image/"):
        if marker in path:
            return path.split(marker, 1)[1]
    raise ValueError(f"unrecognised image path: {path}")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/mindcube")
    ap.add_argument("--dry-run", action="store_true", help="run every check but write nothing")
    args = ap.parse_args()

    train = load_jsonl(TRAIN)
    tiny = load_jsonl(TINYBENCH)
    idx_rows = load_jsonl(IDX)
    cot_rows = load_jsonl(COT)
    print(f"loaded train={len(train)} tinybench={len(tiny)} idx={len(idx_rows)} cot={len(cot_rows)}")

    # ---- the prompt template must be a single constant shared with eval ----
    templates = Counter()
    for row in tiny:
        prompt, question = row["input_prompt"], row["question"]
        if not prompt.endswith(question):
            fail(f"tinybench {row['id']}: input_prompt does not end with question")
        templates[prompt[: len(prompt) - len(question)]] += 1
    if len(templates) != 1:
        fail(f"tinybench holds {len(templates)} distinct prompt templates, expected 1")
    template = next(iter(templates))
    print(f"prompt template: single constant over {len(tiny)}/{len(tiny)} tinybench rows")

    # ---- join key into the authoritative train split ----
    def train_key(row):
        return (row["question"], tuple(image_subpath(p) for p in row["images"]))

    train_map = {}
    for row in train:
        key = train_key(row)
        if key in train_map:
            fail(f"composite key is not unique in train: {row['id']}")
        train_map[key] = row
    print(f"composite key unique over {len(train_map)}/{len(train)} train rows")

    cot_by_idx = {row["idx"]: row for row in cot_rows}
    if set(cot_by_idx) != {row["idx"] for row in idx_rows}:
        fail("idx.jsonl and the CoT file do not cover the same idx set")

    answeronly, cot_out, dropped = [], [], []
    shapes = Counter()

    for row in sorted(idx_rows, key=lambda r: r["idx"]):
        i = row["idx"]
        question = row["text_input"]
        subpaths = [image_subpath(p) for p in row["image_input"]]
        gold = train_map.get((question, tuple(subpaths)))
        if gold is None:
            fail(f"idx={i}: no train row matches the composite key")

        # prompt must equal <image>*N + shared template + question, verbatim
        want_prompt = "<image>\n" * len(subpaths) + template + question
        if row["mindcube_input"] != want_prompt:
            fail(f"idx={i}: mindcube_input does not match the tinybench template")

        # answer-only target must equal the archived mindcube_output, and its
        # letter must agree with the authoritative gt_answer
        target_a = f"<answer>{row['answer']}</answer>"
        if row["mindcube_output"] != target_a:
            fail(f"idx={i}: mindcube_output != <answer>{{answer}}</answer>")
        letter = row["answer"].split(".")[0].strip()
        if letter != gold["gt_answer"].strip():
            fail(f"idx={i} ({gold['id']}): answer letter {letter!r} != gt_answer {gold['gt_answer']!r}")

        # CoT target: drop the <output_3D> marker and both <think> tags, because
        # the training chat template already opens and closes a think block for
        # every assistant turn -- keeping these would nest one inside another.
        match = COT_RE.match(cot_by_idx[i]["text_output"])
        if match is None:
            fail(f"idx={i}: CoT text_output does not match the expected structure")
        # Normalise the surrounding whitespace: the archived bodies start with a
        # space, a newline or a letter depending on the row, and end with zero to
        # two newlines. Left as-is the model would have to learn that noise.
        body, answer_block = match.group(1).strip(), match.group(2)
        inner = ANSWER_RE.match(answer_block).group(1)
        if inner != row["answer"]:
            dropped.append((i, gold["id"], inner, row["answer"]))
            continue

        for path in subpaths:
            if not os.path.exists(os.path.join(IMAGE_ROOT, "other_all_image", path)):
                fail(f"idx={i}: missing image other_all_image/{path}")

        images = [f"other_all_image/{p}" for p in subpaths]
        prompt = "<image>\n" * len(images) + template + question
        answeronly.append({
            "id": gold["id"],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": target_a},
            ],
            "images": images,
        })
        cot_out.append({
            "id": gold["id"],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": f"{body}\n{answer_block}"},
            ],
            "images": images,
        })
        shapes[len(images)] += 1

    # ---- cross-arm invariants ----
    if len(answeronly) != len(cot_out):
        fail(f"arm sizes differ: {len(answeronly)} vs {len(cot_out)}")
    for a, b in zip(answeronly, cot_out):
        if a["id"] != b["id"] or a["conversations"][0]["value"] != b["conversations"][0]["value"]:
            fail(f"arms are not row-aligned at id={a['id']}")
    for rec in cot_out:
        target = rec["conversations"][1]["value"]
        for tag in ("<output_3D>", "<think>", "</think>"):
            if tag in target:
                fail(f"{rec['id']}: CoT target still carries {tag}")
        if not target.endswith("</answer>"):
            fail(f"{rec['id']}: CoT target does not end with </answer>")

    print(f"\nkept {len(answeronly)} / {len(idx_rows)} rows")
    print(f"image counts: {dict(sorted(shapes.items()))}")
    for i, rid, inner, gold_answer in dropped:
        print(f"dropped idx={i} ({rid}): CoT answer {inner[:60]!r} != gt {gold_answer!r}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("mindcube_train_answeronly.json", answeronly),
                       ("mindcube_train_cot.json", cot_out)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
