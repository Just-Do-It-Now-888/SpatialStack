# 论文索引

此文件提供论文知识库的快速入口。详细内容保存在 `papers/`，不要在索引中重复完整笔记。

## 正在阅读

暂无。

## 已完成

- [Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation](papers/vision-opd.md)
  - 标识：arXiv:2605.18740
  - 年份与会议：2026；arXiv 预印本（v4）
  - 状态：已完成（实现细节有待核对项）
  - 一句话摘要：以同一 MLLM 的证据裁剪视图为教师、带框全图视图为学生，在学生 rollout 上做 token 级 JSD 自蒸馏，无需答案标签或推理时 zoom 即提升细粒度视觉理解。
  - 最后更新：2026-08-13

- [VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation](papers/vold.md)
  - 标识：arXiv:2510.23497
  - 年份与会议：2025；arXiv 预印本（v3）
  - 状态：已完成（实现细节有待核对项）
  - 一句话摘要：先用同一文本教师的轨迹对齐 VLM 学生，再联合 GRPO、on-policy token 蒸馏与错误轨迹 KL masking，仅用文本训练提升视觉推理。
  - 最后更新：2026-08-13
- [Visual-Advantage On-Policy Distillation for Vision-Language Models](papers/va-opd.md)
  - 标识：arXiv:2605.21924
  - 年份与会议：2026；arXiv 预印本（v1）
  - 状态：已完成（部分训练超参数待核对）
  - 一句话摘要：用教师在原图与细节退化图上的 token log-prob 差识别视觉关键 token，并从 rollout 与 token 两级重分配 OPD 梯度。
  - 最后更新：2026-08-13
- [Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding](papers/vgs.md)
  - 标识：arXiv:2606.00564
  - 年份与会议：2026；ICML 2026 Spotlight
  - 状态：已完成（部分公式实现语义待核对）
  - 一句话摘要：将 OPD 分解为语言先验与视觉 grounding 目标，通过 Visual Gradient Steering 优先视觉梯度并保持语言能力。
  - 最后更新：2026-08-13
- [ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation](papers/vicur.md)
  - 标识：arXiv:2606.05718
  - 年份与会议：2026；预印本，under review
  - 状态：已完成（cue 构造与模块条件化细节待核对）
  - 一句话摘要：用可从视觉输入恢复的 cue 替代答案特权，并以 sink-token cross-attention 帮助学生内部聚合相关证据。
  - 最后更新：2026-08-13
- [Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation](papers/vigos.md)
  - 标识：arXiv:2606.19120
  - 年份与会议：2026；arXiv 预印本（v2）
  - 状态：已完成（fallback 占比与 PALR 解释有待核对）
  - 一句话摘要：让学生先描述视觉证据再推理，并按 description/reasoning/invalid segment 路由 image-only、answer-privileged 与 reference teacher，以延后答案特权介入。
  - 最后更新：2026-08-13
- [V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning](papers/v-zero.md)
  - 标识：arXiv:2606.25319
  - 年份与会议：2026；arXiv 预印本（v1）
  - 状态：已完成（crop 数据来源、消融语义与 checkpoint 选择待核对）
  - 一句话摘要：比较目标 crop 与随机负 crop 对 sibling rollouts 的 teacher support，以组内视觉证据优势门控 positive-crop OPD，无需文本答案标签。
  - 最后更新：2026-08-13
- [Visual Contrastive Self-Distillation](papers/vcsd.md)
  - 标识：arXiv:2607.21556
  - 年份与会议：2026；arXiv 预印本（v1）
  - 状态：已完成（EMA 系数语义、termination tokens 与评测配置待核对）
  - 一句话摘要：比较 EMA teacher 的原图与内容擦除图全词表分布，在 plausible support 内重塑视觉增强 target，以 forward KL 实现无答案、无 crop 的 OPSD。
  - 最后更新：2026-08-13
- [Correcting What You Cannot See: Credit Assignment for Perception Distillation in Multimodal Reasoners](papers/pcd.md)
  - 标识：arXiv:2607.28336
  - 年份与会议：2026；arXiv 预印本（v2）
  - 状态：已完成（主配置、KL normalization 与标签语义待核对）
  - 一句话摘要：用共享 perception 的多 reasoning 成功率与 perception teacher disagreement 的乘积，门控感知段蒸馏并保留 reasoning RL。
  - 最后更新：2026-08-13
- [VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation](papers/vad.md)
  - 标识：arXiv:2607.28590
  - 年份与会议：2026；arXiv 预印本（v1）
  - 状态：已完成（坐标几何、语义分析与训练配置待核对）
  - 一句话摘要：将 privileged teacher correction 投影到清晰/退化证据的 signed intervention direction，并围绕 student 重构 support/refutation target。
  - 最后更新：2026-08-13
- [Distill What the Student Can See: Fisher-Projected On-Policy Distillation for Vision-Language Models](papers/fp-opd.md)
  - 标识：arXiv:2608.01263
  - 年份与会议：2026；arXiv 预印本（v2）
  - 状态：已完成（优化配置、计算成本与 tangent 实现待核对）
  - 一句话摘要：用视觉 feature perturbations 估计 student 局部响应空间，在 Fisher metric 下投影 teacher correction并构造 capacity-aware OPD target。
  - 最后更新：2026-08-13
- [When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation](papers/sa-opd.md)
  - 标识：arXiv:2608.03632
  - 年份与会议：2026；arXiv 预印本（v1）
  - 状态：已完成（no-prompt implementation、FLMR 单位与动态阈值待核对）
  - 一句话摘要：比较完整输入与 no-prompt 条件下的 sampled teacher–student divergence，过滤低输入依赖且高影响的 token supervision。
  - 最后更新：2026-08-13
- [OPD-V: Visual On-Policy Self-Distillation with Modality Balance](papers/opd-v.md)
  - 标识：arXiv:2608.05131
  - 年份与会议：2026；arXiv 预印本（v2）
  - 状态：已完成（mask配置与 rollout-correction实现待核对）
  - 一句话摘要：用 zoom/masked-crop EMA teachers的正 sampled-token margin构造视觉 trust region，并蒸馏zoom-teacher distribution。
  - 最后更新：2026-08-13

## 待读

暂无。

## 索引条目格式

```markdown
- [论文标题](papers/<method-name>.md)
  - 标识：arXiv / DOI / URL
  - 年份与会议：
  - 状态：待读 / 在读 / 已完成 / 待核对
  - 一句话摘要：
  - 最后更新：YYYY-MM-DD
```

