"""诊断 CV-Bench 分数随 step 变化的成因：格式恢复 vs 能力恢复。

对每个 checkpoint 的 lmms_eval 样本文件统计：
  - 输出形态（是否直接给选项字母、是否是散文前言、是否被 max_new_tokens 截断）
  - 官方解析器 extract_characters_regex 的行为（含从散文单词里误抓 A-F 的情况）
  - 在「解析成功」子集上的条件正确率，用于区分格式与能力
"""

import glob
import json
import re
import string
from collections import Counter

STEPS = [50, 100, 150, 200, 250, 300]

PATTERNS = {
    50: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step50/**/*samples_cvbench.jsonl",
    100: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step100/**/*samples_cvbench.jsonl",
    150: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step150/**/*samples_cvbench.jsonl",
    200: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step200/**/*samples_cvbench.jsonl",
    250: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step250/**/*samples_cvbench.jsonl",
    300: "logs/eval/20260816_qwen35base_mvopsd_v0_main/**/*samples_cvbench.jsonl",
}


def extract_characters_regex(s):
    """官方 lmms_eval 解析器，逐字复制。"""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search(r"[ABCDEF]", s):
        return ""
    matches = re.search(r"[ABCDEF]", s)
    if matches is None:
        return ""
    return matches[0]


CLEAN_LETTER = re.compile(r"^\s*\(?([A-F])\)?\s*[.:,)]?\s*$")
LEADING_LETTER = re.compile(r"^\s*\(?([A-F])\)?\s*[.:,)\s]")


def classify(resp):
    """返回 (形态标签, 干净字母或 None)。"""
    s = resp.strip()
    if not s:
        return "empty", None
    m = CLEAN_LETTER.match(s)
    if m:
        return "bare_letter", m.group(1)
    m = LEADING_LETTER.match(s)
    if m:
        return "letter_then_text", m.group(1)
    return "prose", None


def load(step):
    paths = glob.glob(PATTERNS[step], recursive=True)
    assert len(paths) == 1, (step, paths)
    rows = []
    with open(paths[0]) as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "resp": r["filtered_resps"][0],
                    "gold": r["target"][1],
                    "task": r["doc"]["task"],
                    "source": r["doc"]["source"],
                    "n_choices": len(r["doc"]["choices"]),
                    "logged_pred": r["doc"]["pred_answer"],
                    "logged_result": r["doc"]["result"],
                }
            )
    return rows


def combined(rows, key):
    """官方聚合：((ADE+COCO)/2 + Omni3D)/2。key 取每行的 0/1 判定函数。"""
    by_src = {}
    for src in ("ADE20K", "COCO", "Omni3D"):
        sub = [key(r) for r in rows if r["source"] == src]
        by_src[src] = sum(sub) / len(sub) if sub else 0.0
    acc2d = (by_src["ADE20K"] + by_src["COCO"]) / 2
    acc3d = by_src["Omni3D"]
    return 100 * (acc2d + acc3d) / 2, 100 * acc2d, 100 * acc3d


def main():
    print(f"{'step':>5} {'combined':>9} {'bare':>7} {'lead':>7} {'prose':>7} "
          f"{'parse_fail':>11} {'spurious':>9} {'cond_acc':>9} {'words':>7}")
    print("-" * 82)
    details = {}
    for step in STEPS:
        rows = load(step)
        n = len(rows)
        shapes = Counter()
        preds = Counter()
        spurious = 0
        parse_fail = 0
        clean_rows = []
        words = 0
        for r in rows:
            shape, letter = classify(r["resp"])
            shapes[shape] += 1
            words += len(r["resp"].split())
            p = extract_characters_regex(r["resp"])
            r["pred"] = p
            preds[p or "<none>"] += 1
            if p == "":
                parse_fail += 1
            # 解析出了字母，但模型输出根本不是在给选项 → 从散文里误抓
            if p != "" and shape == "prose":
                spurious += 1
            if shape in ("bare_letter", "letter_then_text"):
                r["intended"] = letter
                clean_rows.append(r)

        comb, a2d, a3d = combined(rows, lambda r: 1 if r["pred"] == r["gold"] else 0)
        cond = combined(clean_rows, lambda r: 1 if r["intended"] == r["gold"] else 0)[0] if clean_rows else float("nan")

        print(f"{step:>5} {comb:>9.2f} {100*shapes['bare_letter']/n:>6.1f}% "
              f"{100*shapes['letter_then_text']/n:>6.1f}% {100*shapes['prose']/n:>6.1f}% "
              f"{100*parse_fail/n:>10.1f}% {100*spurious/n:>8.1f}% "
              f"{cond:>8.2f}% {words/n:>7.1f}")
        details[step] = (rows, preds, shapes, len(clean_rows))

    print("\n\n=== 解析出的答案字母分布（含误抓） ===")
    print(f"{'step':>5} " + " ".join(f"{c:>7}" for c in "ABCDEF") + f"{'<none>':>8}")
    for step in STEPS:
        _, preds, _, _ = details[step]
        n = sum(preds.values())
        print(f"{step:>5} " + " ".join(f"{100*preds.get(c,0)/n:>6.1f}%" for c in "ABCDEF")
              + f"{100*preds.get('<none>',0)/n:>7.1f}%")

    print("\n\n=== 各 step 最常见的回答开头（前 40 字符） ===")
    for step in STEPS:
        rows, _, _, n_clean = details[step]
        heads = Counter(r["resp"].strip()[:40].replace("\n", "\\n") for r in rows)
        print(f"\n--- step {step}（可判定回答 {n_clean}/{len(rows)}）---")
        for head, c in heads.most_common(6):
            print(f"  {100*c/len(rows):>5.1f}%  {head!r}")

    print("\n\n=== 仅在「模型明确给出字母」的子集上算分（去掉格式因素） ===")
    print(f"{'step':>5} {'子集占比':>9} {'子集内正确率':>13} {'该子集的随机基线':>16}")
    for step in STEPS:
        rows, _, _, _ = details[step]
        clean = []
        for r in rows:
            shape, letter = classify(r["resp"])
            if shape in ("bare_letter", "letter_then_text"):
                r["intended"] = letter
                clean.append(r)
        if not clean:
            print(f"{step:>5} {'0.0%':>9} {'n/a':>13} {'n/a':>16}")
            continue
        acc = 100 * sum(1 for r in clean if r["intended"] == r["gold"]) / len(clean)
        rand = 100 * sum(1 / r["n_choices"] for r in clean) / len(clean)
        print(f"{step:>5} {100*len(clean)/len(rows):>8.1f}% {acc:>12.2f}% {rand:>15.2f}%")


if __name__ == "__main__":
    main()
