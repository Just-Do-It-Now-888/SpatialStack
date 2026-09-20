# MindCube 实验结果汇总

- 最后更新：2026-09-08
- 模型：Qwen3.5-4B（`models/Qwen3.5-4B`）
- 评测集：MindCube **tinybench**（1,050 题；among 600 / around 250 / rotation 200）
- 相关 registry：
  - [`20260906_qwen35_mindcube_sft.yaml`](../registry/20260906_qwen35_mindcube_sft.yaml)（SFT）
  - [`20260907_mindcube_mvopsd_k1.yaml`](../registry/20260907_mindcube_mvopsd_k1.yaml)（frozen K=1 MV-OPSD）
  - [`20260907_mindcube_mvopsd_k1_ema.yaml`](../registry/20260907_mindcube_mvopsd_k1_ema.yaml)（EMA teacher，进行中）
  - [`20260908_qwen35base_mindcube_zeroshot.yaml`](../registry/20260908_qwen35base_mindcube_zeroshot.yaml)（基座复测）

---

## 1. 核心结论

在官方 `input_prompt`、greedy、`disable_thinking` 协议下：

1. **Zero-shot 基座 overall 46.86**（unreadable 0%），为可复现能力下限，非 scorer 失效。
2. **In-domain SFT 将准确率提升至 70+**：answer-only **74.48** > CoT **69.81**（McNemar +4.67 pp，p=5.5e-3）；A 在 314 步仍未饱和，B 约 0.25 epoch 即 plateau。
3. **满视图 vs 单视图存在 ~15–20 pp 的 view-information gap**（coverage 100%，gap 全来自 conditional accuracy 下降）。
4. **Frozen K=1 MV-OPSD 未实现单视图→多视图迁移**：Arm A 最佳 step 30 仅 +1.14 pp，终点 **68.76**（相对 step 0 −5.62）；Arm B **71.81 → 68.38**。`vopd_loss` 下降不是 task accuracy 的有效 surrogate。

**不可声称：** 跨 benchmark 空间能力提升；CoT 推理过程正确；K=1 已完成 N-view→1-view 知识压缩；相对 3DThinker 62.19% 的方法优劣。SPAR3 K=1 step 55 **未测** MindCube。

---

## 2. 评测与训练协议

| 项目 | 设定 |
| --- | --- |
| 题面 | 官方 `input_prompt`（`[Task]` / `[Answer Instruction]` / `[Question]`）；**不走** VSI/BLINK 的 original/lastline/`\boxed{}` |
| 解码 | greedy；`max_new_tokens=2048`；`enable_thinking=false` |
| 判分 | `lmms_eval.tasks.mindcube.utils.mindcube_process_results`（官方 `extract_answer`） |
| 图像 | 训练与评测均 **bypass VGGT 竖图裁切**；`min_pixels=200704` / `max_pixels=1605632`；中位 **300 visual token/图** |
| 引擎 | 离线 SFT/基座：**transformers 5.3.0** 八卡分片；训练内 MV-OPSD：**vLLM 0.18.0** 满视图 tinybench |

**读数纪律**

- transformers 与 vLLM 同 checkpoint 引擎差 ≤0.8 pp（74.48 vs 74.29 等），**趋势可比，Δ 不宜当精确效应量**。
- 主看 task family 与 `nviews/{3,4}`；`nviews/2` 几乎全为 evidence-missing pair，无 view privilege。
- 任何 OPSD headline 须同时报 **剔除 `mindcube_pair_2view` 后** 的 `prompt_matches_views` 桶。

---

## 3. Zero-shot 基座

**实验：** `20260908_qwen35base_mindcube_zeroshot`（与 `20260906` 锚点同协议复测，逐项相同）

| 指标 | 数值 |
| --- | --- |
| overall | **46.86** |
| among / around / rotation | 42.50 / 58.80 / 45.00 |
| unreadable | **0.00%** |
| 输出 token（中位 / 均值 / max） | 288 / 352.5 / 2048 |
| 撞 cap | 2.00% |
| visual token/图（中位） | 300 |

**产物：** `logs/eval/20260908_qwen35base_mindcube_zeroshot/`（历史锚点 `logs/eval/20260906_qwen35_mindcube_sft/baseline/`）

**对照（非受控）：** 3DThinker-S1（Qwen2.5-VL-3B，同协议）**62.19**（among 62.33 / around 72.00 / rotation 49.50）。本基座低 15.3 pp，主差在 among。

---

## 4. SFT（answer-only vs CoT）

**实验：** `20260906_qwen35_mindcube_sft`  
**数据：** 9,999 条/臂，逐行同题同序；2 epoch（314 步）；lr 1e-5；唯一变量为监督目标。

| 臂 | 监督 | 中位目标 token | checkpoint | overall @314 |
| --- | --- | --- | --- | --- |
| A | `<answer>…</answer>` | ~9 | `output/..._answeronly/checkpoint-314` | **74.48** |
| B | 文本 CoT + 同一 answer 行 | ~381 | `output/..._cot/checkpoint-314` | **69.81** |

### 4.1 训练曲线（tinybench，greedy）

| step | A overall | A among | B overall | B among | A 中位 token | B 中位 token |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 46.86 | 42.50 | 46.86 | 42.50 | 288 | 288 |
| 78 | 52.00 | 53.00 | **64.86** | 62.67 | 10 | 322 |
| 156 | 58.10 | 59.17 | 68.86 | 65.83 | 11 | 313 |
| 234 | 72.67 | 78.67 | 68.38 | 67.33 | 11 | 328 |
| **314** | **74.48** | **80.00** | **69.81** | **70.00** | 11 | 319 |

- **A：** +27.6 pp（46.86→74.48）；收敛慢、314 步未饱和；生成长度塌至 ~11 token（format collapse 至 answer-only）。
- **B：** +22.95 pp；~78 步接近平台，之后三点在噪声内；长度保持 313–328 token。
- 两条 token-level CE 基数差 ~40×，**loss 不可并列**。

### 4.2 终点配对检验（step 314，McNemar，n=1050）

| 对比 | Δ (pp) | p | 备注 |
| --- | --- | --- | --- |
| A vs B overall | **+4.67** | 5.5e-3 | only-A 174 / only-B 125 |
| among (n=600) | +10.00 | 1.7e-6 | A 显著更好 |
| around (n=250) | −7.60 | 2.6e-2 | B 更好，n 较小 |
| rotation (n=200) | +4.00 | 0.44 | 不显著 |

**Oracle 并集（两 checkpoint 取每题最优）：** **86.38%**（相对最优单模型 +11.9 pp）；错误集部分互补。

### 4.3 解释与边界

- tinybench 只判选项字母；A 将 capacity 用于 answer token 条件分布，对当前 metric 最优。
- B 保留长 CoT，around 略优，但 overall 更低且更早 plateau——**answer-only vs CoT supervision** 的 trade-off；不能从分数推出 CoT 推理正确（仅校验结尾与 gt 一致）。
- train vs tinybench 图片/场景零重叠 → **同分布 test-time 泛化**，非泄漏，亦非 OOD 空间推理提升。
- **曲线图：** [`figs/mindcube_qwen35_sft_curve.png`](figs/mindcube_qwen35_sft_curve.png)

---

## 5. View recoverability（蒸馏上界）

**实验：** `20260907_mindcube_mvopsd_k1` gate-0；冻结 SFT-314；同机 vLLM；full-N vs K=1 single-view。

| init | full-view | 1-view | gap | coverage | 中位 token (full / 1v) |
| --- | --- | --- | --- | --- | --- |
| A (answer-only) | **74.29** | 59.24 | **−15.05** | 100% | 11 / 10 |
| B (CoT) | **70.57** | 55.24 | **−15.33** | 100% | 319 / 261 |

- 相对 HF 锚点：A −0.19 pp / B +0.76 pp → 引擎差可忽略。
- gap 全为 **accuracy-on-answered** 下降，无 abstention。

### 5.1 Evidence-missing pair 子集

| 范围 | 占比 | 现象 |
| --- | --- | --- |
| 训练池 pair 行 | 13%（1,302/9,999） | 题面 two-view、输入单图 |
| tinybench pair 行 | 26%（274/1,050） | val 被该桶稀释更重 |

Arm A 在 274 行 pair 上：full **47.45** → 1-view **47.81**（+0.36）——teacher 亦未利用第二视图；该子集蒸馏退化为 label prior 自确认。

**剔 pair 后 `prompt_matches_views`（776 行）：** gap **−20.49**（A）/ **−19.72**（B）——K=1 distillation 应对齐的 view-information gap。

**产物：** `output/probes/mindcube_recoverability_arm{A,B}.json`

---

## 6. Frozen MV-OPSD K=1

**实验：** `20260907_mindcube_mvopsd_k1`  
**设定：** teacher 全 N 视图 / student K=1；frozen teacher；9,999 行训练池；156 步；训练内满视图 tinybench（vLLM greedy）。

### 6.1 Arm A（answer-only init，node-B）

| 指标 | 数值 |
| --- | --- |
| step 0 overall | 74.38 |
| **best（step 30）** | **75.52**（+1.14 vs step 0） |
| best among / nviews-3 | 80.00 / 76.52 |
| final（step 156） | **68.76**（−5.62 vs step 0） |
| 保留 ckpt | `checkpoints/20260907_mindcube_mvopsd_k1_armA/global_step_30` |

**逐步 overall（%）：** 74.38 → 74.95 → 74.57 → **75.52** → 73.81 → 73.43 → 70.48 → 71.62 → 69.14 → 68.19 → 69.14 → 69.43 → 67.90 → 68.48 → 68.67 → 68.57 → **68.76**

`nviews/4` 自 step 0 的 92.34 持续下降（step 30 已 90.49）。

### 6.2 Arm B（CoT init）

| 指标 | 数值 |
| --- | --- |
| step 0 → final | **71.81 → 68.38** |

逐步序列未按 Arm A 密度写入 registry；方向与 A 一致（in-domain full-view accuracy 回退）。

### 6.3 解释

Student 在 **view-budget mismatch**（1 vs N）下对齐冻结 multi-view teacher posterior；早期微弱 uplift 后 full-view accuracy 被冲刷。`vopd_loss` 仅反映 student–teacher 对齐（LESSON-012），**不能作为方法有效性判据**。

### 6.4 EMA teacher（进行中，不纳入终点结论）

**实验：** `20260907_mindcube_mvopsd_k1_ema`；相对 frozen 单变量 `teacher_regularization=ema`、`teacher_update_rate=0.05`。  
SMOKE 已通过（`teacher_probe` 跨步漂移）。正式双臂曲线截至本报告编写时未收口。

---

## 7. 实验索引与产物

| 阶段 | experiment_id | 关键产物 |
| --- | --- | --- |
| 基座锚点 / 复测 | `20260906_qwen35_mindcube_sft`（baseline） / `20260908_qwen35base_mindcube_zeroshot` | `logs/eval/20260906_.../baseline/`；`logs/eval/20260908_qwen35base_mindcube_zeroshot/` |
| SFT | `20260906_qwen35_mindcube_sft` | 权重已于 2026-09-09 从两机删除；评测 `logs/eval/20260906_qwen35_mindcube_sft/` 仍在 |
| MV-OPSD frozen | `20260907_mindcube_mvopsd_k1` | 权重已删；`logs/val/` 与探针 `output/probes/` 仍在 |
| MV-OPSD EMA | `20260907_mindcube_mvopsd_k1_ema` | 权重已删；`logs/train/ema_arm{A,B}_driver.log` 仍在 |

---

## 8. 可写 / 不可写（对外口径）

| 可写 | 不可写 |
| --- | --- |
| 基座 46.86 可复现；answer-only SFT → 74.48（未饱和） | 跨 benchmark 空间能力提升 |
| CoT SFT → 69.81，更早 plateau | CoT 推理过程事实正确 |
| 剔 pair 后 view gap ~20 pp；K=1 frozen distillation 净效应为负 | K=1 已完成 N-view→1-view 压缩 |
| A/B 在 among vs around 上互补（oracle 86.38%） | 相对 3DThinker 62.19 的优劣 |
| SPAR3 未测 MindCube | 将 SPAR/VSI 协议分数与 MindCube 并列 |
