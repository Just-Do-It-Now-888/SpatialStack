# OPSD

## 基本信息

- 论文标题：Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
- 作者：Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, Aditya Grover
- 年份与会议：2026；arXiv 预印本
- arXiv/DOI：arXiv:2601.18734；DOI: 10.48550/arXiv.2601.18734
- 论文链接：https://arxiv.org/abs/2601.18734
- 代码：https://github.com/siyan-zhao/OPSD
- 本地文件：无
- 阅读状态：已完成（论文正文公式与完整消融表待核对；公开 README/代码入口已读）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-09-19
- 最后更新：2026-09-19

## 一句话总结

OPSD 用同一模型充当学生与教师：学生只看题目，教师额外看到 ground-truth solution，并在学生自己的 on-policy 轨迹上做 token 级分布匹配；公开实现以 TRL GOLD 为底座，主结果使用固定教师（LoRA）与 JSD，后续加入 per-token KL clipping 以抑制风格 token。

## 核心问题

Off-policy SFT 模仿教师轨迹，但部署时学生走自己的状态分布。论文研究：

1. 能否不引入外部更强教师，只靠同一模型在不同上下文下形成 teacher/student 不对称；
2. 教师特权是否可以是答案/题解，而两边的问题文本保持同一套输入；
3. 在学生自己采样的 token 上做 dense distillation，是否比模仿教师生成轨迹更贴近部署状态。

## 方法

### 整体流程

公开 README（[siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)，2026-03-18 更新）将方法定义为：

1. 单一模型同时扮演 student 与 teacher，条件不同。
2. Student 只看到 problem。
3. Teacher 在同一 problem 上额外看到 ground-truth solution。
4. Student 生成 on-policy completion；teacher 不另采一条答案，而是沿学生轨迹做 token-level distribution matching。

### 关键模块

**答案特权，而非视觉特权。** Teacher/student 的非答案输入相同；不对称只来自 reference solution。这与 Vision-OPD 的 crop/full-image 轴、本仓库 MV-OPSD 的 N/K 视图轴都不同。

**固定教师。** `--fixed_teacher` 默认 False；主结果使用固定初始策略，且当前实现依赖 LoRA（`--use_peft`）。README 注明：若关闭 PEFT，教师会随每步更新，训练可能不稳定。

**JSD 与采样目标。** 主目标是 full-vocabulary JSD；`--beta` 插值 forward/reverse KL（0=forward，1=reverse）。`--use_tinker_loss` 改为 sampled-token policy-gradient，更省显存，但当时没有 clip，可能不稳定。

**Per-token point-wise KL clipping。** 2026-03-18 更新：风格 token（如 wait/think）的 KL 可比数学 token 高 6–15 倍并主导梯度。`--jsd_token_clip` 在对词表求和前截断逐 token 贡献；截断后 loss 可为负。

### 损失函数与训练目标

公开参数（README “Key OPSD arguments”）：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--fixed_teacher` | False | 把教师钉在 step-0 策略；主结果使用，且需 PEFT |
| `--use_tinker_loss` | False | sampled-token PG，替代 full-vocab JSD |
| `--max_completion_length` | — | 学生蒸馏生成长度；主实验 1024 |
| `--beta` | — | JSD 混合：0=forward KL，1=reverse KL |
| `--jsd_token_clip` | 0.05 | 点式截断 JSD 贡献 |
| `--reason_first` | False | 蒸馏前给教师附加显式 rationalization |

精确 JSD 公式、top-\(K\) 是否使用、以及 clip 的符号约定以论文/代码为准，正文未在本笔记中逐式核对。

### 训练数据与训练流程

- 公开脚本以数学推理（AIME/HMMT）为主，不是多视图空间数据。
- 代码入口：`opsd_train.py` / `opsd_trainer.py` / `data_collator.py`；另有 SFT/GRPO baseline。
- 构建在 TRL 实验性 GOLD trainer 上。
- 2026-03-18 修复 chat template 与 DeepSpeed ZeRO-2；作者称修复后 OPSD 提升更明显，尤其 Qwen3-1.7B。

## 实验设置

- 模型：公开表覆盖 Qwen3-1.7B（thinking）、以及 1.7B/4B/8B non-thinking。
- 数据集：AIME24、AIME25、HMMT25。
- Baseline：SFT、GRPO（脚本 `run_sft.sh` / `run_grpo.sh`）。
- 指标：Avg@12；thinking 评测 temperature=1.0、max new tokens=38912、12 samples。
- 关键超参数（README 表，非论文正文核对）：thinking 主实验 completion 1024；non-thinking 的 `jsd_token_clip` 为 1e-7（8B）或 1e-6（4B/1.7B）。
- 算力声明：Qwen3-1.7B 训练约 15 分钟 / 4×H100，峰值约在 100 steps 内。

## 实验结论

公开 README 的 thinking-mode Qwen3-1.7B Avg@12（单 seed）：

- AIME24：Base 51.5 → step 100 57.2
- AIME25：Base 36.7 → step 50 43.9，step 100 41.1
- HMMT25：Base 23.1 → step 100 29.2

Non-thinking 在 8B 上 step 50 相对 base 提升很大（AIME24 26.4→49.7），之后部分回落。作者承认单 seed Avg@12 有波动，建议多 seed。

这些数字来自 GitHub README，不是本笔记对 arXiv 表的逐格核对。

## 局限性

- 特权是**不可部署的答案/题解**；学生若复制 hint 句式，会学到捷径而非推理。ViCuR 后续把这一点做成显式风险（hint leakage）。
- 主结果依赖 LoRA 固定教师；与本仓库 frozen full-weight teacher 不是同一实现。
- 公开评测是数学竞赛，不是空间/多视图。短选项或短数字答案的特权强度低于完整题解。
- 单 seed；曲线非单调（AIME25 thinking、8B non-thinking 后期回落）。
- `jsd_token_clip` 使 loss 可负，监控时不能把“loss 变负”直接读成崩溃或变好。

## 可复用到当前项目的内容

1. **特权轴可以是答案而不是视图数。** 等视图 Answer-OPSD 把 student/teacher 都放到完整相册，只给 teacher 接 GT。
2. **输入视图数量必须相同**，否则答案特权与 N/K 视图特权混杂，无法归因。
3. Student on-policy rollout + teacher rescoring 已由本仓库 `loss_mode=vopd` 实现；缺的是 `teacher_prompt_mode=answer_hint` 的数据与空答案守卫。
4. 必须扫描 student 是否复述 “reference solution” / Hint / 原文 GT（ViCuR E-008）。
5. 本轮不默认搬 `--jsd_token_clip`：仓库已有 top-K JSD 与 IS clip。出现风格词或 hint 主导后再加。
6. SPAR 的 GT 常是短段落或选项，不是 AIME 式完整题解；特权更弱，也更容易被整句抄写。

## 与其他论文的相同点和冲突

- 与 [Vision-OPD](vision-opd.md)：同为 on-policy self-distillation；Vision-OPD 用 crop 特权且主配置不消费答案，OPSD 用答案特权且两侧非答案输入相同。
- 与本仓库 MV-OPSD：原先唯一特权轴是 teacher N / student K；Answer-OPSD 关掉该轴。
- 与 [ViCuR](vicur.md)：ViCuR 认为答案特权不可恢复，并报告 hint leakage；本方案是对该风险的直接复现，而不是否定它。
- 与 [ViGOS](vigos.md)：ViGOS 保留答案但推迟到 description 之后；本方案把答案放进整个 teacher prefix。
- 与 [VCSD](vcsd.md)：VCSD 的 answer-hint OPSD 是其 baseline；VCSD 主张用图像 contrast 代替答案。

## 证据

### E-001

- 结论：OPSD 用同一模型、不同上下文（学生无答案、教师有答案）在学生 on-policy 轨迹上做 token 级分布匹配。
- 类型：作者讨论（公开 README Overview）
- 定位：https://github.com/siyan-zhao/OPSD README Overview
- 必要引用：无
- 备注：论文公式待核对。

### E-002

- 结论：教师与学生的非答案输入相同；特权是 reference solution，不是更多视觉证据。
- 类型：作者讨论
- 定位：README Overview；Key arguments 中无 crop/view 开关
- 必要引用：无
- 备注：这是与 MV-OPSD 视图特权正交的轴。

### E-003

- 结论：主结果使用固定教师，当前代码路径依赖 LoRA；关闭 PEFT 时教师会每步更新。
- 类型：作者讨论
- 定位：README `--fixed_teacher`
- 必要引用：无
- 备注：本仓库用 frozen full-weight copy，语义相近但实现不同。

### E-004

- 结论：风格 token 的 KL 可高于数学 token 6–15 倍；点式 JSD/KL clip 用于稳定训练。
- 类型：作者讨论
- 定位：README Updates 2026-03-18；`--jsd_token_clip`
- 必要引用：无
- 备注：clip 后 loss 可为负。

### E-005

- 结论：公开 Qwen3-1.7B thinking 表在 AIME24/HMMT25 上 step 100 高于 base，AIME25 峰值在 step 50。
- 类型：论文结论（README 转载；arXiv 表待核对）
- 定位：README “Thinking Mode Eval”
- 必要引用：无
- 备注：单 seed Avg@12。

### E-006

- 结论：答案特权在后续视觉 OPD 中被报告为 shortcut 风险，而不是默认安全的教师信号。
- 类型：个人推断
- 定位：见 [ViCuR E-004、E-008](vicur.md#e-004)
- 必要引用：无
- 备注：本仓库等视图 Answer-OPSD 必须测泄漏，不能只看 vopd_loss。

## 待验证问题

- [ ] arXiv 正文中 JSD 公式、\(\beta\) 取值与 GOLD trainer 对应关系。
- [ ] 固定教师是否必须 LoRA，全量 frozen copy 是否复现主表。
- [ ] 短答案（选项/数字）相对完整题解的特权强度。
- [ ] 多视图空间任务上，答案特权是提升空间推理还是只诱发抄答案。

## 阅读日志

### 2026-09-19

- 阅读范围：GitHub README（Overview、Updates、Key arguments、thinking/non-thinking 表）与引用信息 arXiv:2601.18734。
- 新增认识：原版 OPSD 的不对称是答案上下文，输入题面相同；可直接映射为“等视图 + teacher 接 GT”。
- 修正内容：无。
- 下一步：用 SPAR 全相册等视图数据打开仓库已有 `teacher_prompt_mode=answer_hint`；扫描 hint/GT 泄漏。
