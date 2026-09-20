"""验证 CV-Bench 分数上升是否等价于「回答以 Based 开头的频率」上升。

假设：max_new_tokens=16 让所有回答都在给出选项前被截断，
官方解析器 re.search(r"[ABCDEF]", s) 于是只能从散文单词里抓大写字母，
而唯一能命中的高频词是句首的 "Based"。若成立，则
    实测分数 == 常量 B 预测器的分数 × P(以 Based 开头)
"""

import glob
import json
import re
from collections import Counter

PATTERNS = {
    50: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step50/**/*samples_cvbench.jsonl",
    100: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step100/**/*samples_cvbench.jsonl",
    150: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step150/**/*samples_cvbench.jsonl",
    200: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step200/**/*samples_cvbench.jsonl",
    250: "logs/eval/20260816_qwen35base_mvopsd_v0_main_step250/**/*samples_cvbench.jsonl",
    300: "logs/eval/20260816_qwen35base_mvopsd_v0_main/**/*samples_cvbench.jsonl",
}


def load(step):
    (path,) = glob.glob(PATTERNS[step], recursive=True)
    return [json.loads(l) for l in open(path)]


def combined(rows, ok):
    acc = {}
    for src in ("ADE20K", "COCO", "Omni3D"):
        sub = [ok(r) for r in rows if r["doc"]["source"] == src]
        acc[src] = sum(sub) / len(sub) if sub else 0.0
    return 100 * (((acc["ADE20K"] + acc["COCO"]) / 2) + acc["Omni3D"]) / 2


rows0 = load(300)
const_b = combined(rows0, lambda r: 1 if r["target"][1] == "B" else 0)
gold = Counter(r["target"][1] for r in rows0)
n = len(rows0)

print("金标答案字母分布：",
      {k: f"{100*v/n:.1f}%" for k, v in sorted(gold.items())})
print(f"「永远答 B」这个常量预测器的 combined 分数：{const_b:.2f}\n")

print(f"{'step':>5} {'实测分数':>9} {'含大写A-F的回答':>15} {'以Based开头':>12} "
      f"{'预测=常量B×P(Based)':>20} {'含A-F但非首词':>14}")
print("-" * 88)
for step, path in PATTERNS.items():
    rows = load(step)
    n = len(rows)
    based = sum(1 for r in rows if r["filtered_resps"][0].strip().startswith("Based"))
    hasAF = sum(1 for r in rows if re.search(r"[ABCDEF]", r["filtered_resps"][0]))
    other = hasAF - based
    measured = combined(rows, lambda r: 1 if r["doc"]["result"] else 0)
    pred = const_b * based / n
    print(f"{step:>5} {measured:>9.2f} {100*hasAF/n:>14.1f}% {100*based/n:>11.1f}% "
          f"{pred:>19.2f} {100*other/n:>13.1f}%")

print("\n=== 被截断的比例（回答末尾不是句号/完整句） ===")
for step in PATTERNS:
    rows = load(step)
    n = len(rows)
    trunc = sum(1 for r in rows if not r["filtered_resps"][0].rstrip().endswith((".", "!", "?")))
    print(f"  step {step:>3}: {100*trunc/n:>5.1f}% 的回答在 16 token 处被硬截断")

print("\n=== 含大写 A-F 的回答里，那个字母来自哪个词 ===")
for step in (50, 300):
    rows = load(step)
    words = Counter()
    for r in rows:
        s = r["filtered_resps"][0]
        m = re.search(r"[ABCDEF]", s)
        if m:
            i = m.start()
            j = i
            while j < len(s) and s[j].isalpha():
                j += 1
            words[s[i:j]] += 1
    print(f"  step {step}: {words.most_common(8)}")
