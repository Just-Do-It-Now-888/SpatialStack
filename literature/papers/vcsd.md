# VCSD

## 基本信息

- 论文标题：Visual Contrastive Self-Distillation
- 作者：Yijun Liang, Yunjie Tian, Yijiang Li, Yuqi Jia, Furong Huang, Tianyi Zhou, Di Fu
- 年份与会议：2026；arXiv 预印本（v1）
- arXiv/DOI：arXiv:2607.21556；DOI: 10.48550/arXiv.2607.21556
- 论文链接：https://arxiv.org/abs/2607.21556
- 代码：未提供
- 本地文件：无
- 阅读状态：已完成（EMA 系数语义、termination tokens 与评测配置待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

VCSD 不给 self-teacher 答案、crop 或外部标注，而是在每个 student prefix 上比较 EMA teacher 的原图与内容擦除图 full-vocabulary 分布，用 log-ratio 在原图 plausible support 内重塑 teacher target，再以 forward KL 蒸馏；它跨六个 Qwen 规模稳定提高七项平均分，但“PMI/visual reward”解释依赖 control image 近似无视觉条件，且 EMA update rate 0.05 的实现语义不明确。

## 核心问题

OPSD 虽不需要外部 teacher，但如果 student 和 self-teacher 在同一 prefix 上看到完全相同的信息，两者预测可能过于接近，teacher target 缺少新增信息。已有方法用 answer/rationale、crop、cue 等 teacher-only privilege 制造不对称，却引入答案泄漏、区域标注或额外数据 pipeline。

论文研究：

1. 能否只由原始 prompt-image 本身构造 self-distillation 所需的 target asymmetry；
2. 原图与 content-erased control 下的 same-model distribution difference 能否识别 image-dependent candidate tokens；
3. 如何防止低概率 token 因大 likelihood ratio 被错误放大；
4. 将视觉 contrast 直接写入 full-distribution target，是否比 answer-hint OPSD 更稳定有效。

## 方法

### 整体流程

1. Student \(\pi_\theta\) 在原图 \(J\) 与 prompt \(P\) 上采样 on-policy response \(y\)。
2. EMA teacher \(\pi_\phi\) 在每个固定 student prefix \(y_{<t}\) 上执行两次 forward：
   - 原图 \(J\)；
   - 同尺寸黑色 RGB control image \(J_{\mathrm{ctrl}}\)。
3. 对全词表每个 candidate \(v\) 计算原图/control log-prob difference。
4. 只保留原图 teacher probability 不低于其最大概率 \(\beta\) 倍的 candidates。
5. 用 visual contrast 指数重加权原图 teacher distribution，形成 detached target \(q_t^*\)。
6. 对 student 的全分布做 forward KL；每次 student update 后更新 EMA teacher。
7. 推理时仅保留 student，不需要 control image、teacher 或额外分支。

### Visual conditioning contrast

默认 control 是与原图同尺寸的全黑 RGB image，以保持 resolution、multimodal interface、preprocessing path 和 visual-token count，同时移除 instance-specific content。

\[
p_{\phi,t}^{J}(v)
=\pi_\phi(v\mid P,J,y_{<t}),\qquad
p_{\phi,t}^{0}(v)
=\pi_\phi(v\mid P,J_{\mathrm{ctrl}},y_{<t}).
\]

\[
\Delta_t(v)
=\log p_{\phi,t}^{J}(v)
-\log p_{\phi,t}^{0}(v).
\]

正值表示该 candidate 在原图下获得更多 teacher support。Control distribution 仅作 reference，不直接作为蒸馏 target。

### Plausibility support 与 target shaping

Relative support：

\[
\mathcal S_t(\beta)
=\left\{
v:p_{\phi,t}^{J}(v)
\ge \beta\max_u p_{\phi,t}^{J}(u)
\right\}.
\]

Contrast-shaped target：

\[
q_t^*(v)
=\frac{
\mathbf1[v\in\mathcal S_t(\beta)]
p_{\phi,t}^{J}(v)
\exp(\alpha\widetilde\Delta_t(v))
}{
\sum_{u\in\mathcal S_t(\beta)}
p_{\phi,t}^{J}(u)
\exp(\alpha\widetilde\Delta_t(u))
}.
\]

对非 termination token：

\[
\log \tilde q_t^*(v)
=(1+\alpha)\log p_{\phi,t}^{J}(v)
-\alpha\log p_{\phi,t}^{0}(v).
\]

因此默认 \(\alpha=1\) 时，未归一化 target 相当于
\((p_\phi^J(v))^2/p_\phi^0(v)\)。原图分布提供 plausibility anchor，contrast 负责视觉 sharpening。指定的 sequence-termination token 令 \(\widetilde\Delta=0\)，但论文未列明具体 token 集合。

### Distillation 与 EMA

\[
\mathcal L_{\mathrm{VCSD}}
=T_{\mathrm{KD}}^2\,
\mathbb E\left[
\frac1{|y|}\sum_t
D_{\mathrm{KL}}\left(
\operatorname{sg}[q_t^*]\|p_{\theta,t}
\right)
\right].
\]

Teacher 公式为：

\[
\phi\leftarrow\mu\phi+(1-\mu)\theta.
\]

实验部分另称 EMA “update rate” \(\rho=0.05\)，但未说明 \(\rho=\mu\) 还是 \(\rho=1-\mu\)。若前者，teacher 每步吸收 95% student；若后者，则常规 decay 为 0.95，二者差异很大。

### 理论解释

论文将

\[
r_t^{\mathrm{vis}}(v)=\widetilde\Delta_t(v)
\]

视为 implicit visual-evidence reward，并证明 \(q_t^*\) 是 support-restricted KL-regularized one-step policy improvement 的唯一解：

\[
q_t^*
=\arg\max_{q\in\Pi(\mathcal S_t)}
\left\{
\alpha\mathbb E_q[r_t^{\mathrm{vis}}]
-D_{\mathrm{KL}}(q\|\bar p_{\phi,t}^{J})
\right\}.
\]

该闭式解证明是标准 exponential tilting 恒等式，说明 target 对所定义 reward 最优，但不证明该 reward 与任务正确性一致。

论文还把 \(\Delta_t(v)\) 解释为 conditional PMI 的近似：

\[
\operatorname{PMI}(v;J\mid H_t)
=\log\frac{p(v\mid J,H_t)}{p(v\mid H_t)}.
\]

只有当 black/control-image prediction 可靠近似 \(p(v\mid H_t)\) 时该解释成立；实验仅证明不同 control 的平均结果接近，未验证分布层面的 PMI 近似误差。

## 训练数据与训练流程

- 数据：ViRL39K single-image dataset。
- 模型：Qwen3-VL 2B/4B/8B；Qwen3.5 2B/4B/9B。
- 对照：原始 base、published answer-hint OPSD、VCSD。
- 默认 \(\alpha=1.0,\beta=0.1,T_{\mathrm{KD}}=2\)。
- Uniform response-position weights。
- EMA update rate：\(\rho=0.05\)，具体与公式 \(\mu\) 的映射 `待核对`。
- AdamW；batch 32 prompts；每 prompt \(n=8\) rollouts。
- LR \(2\times10^{-6}\)；10 warmup steps 后 constant LR。
- 固定 90 optimization steps；8×NVIDIA B200。
- 未报告 rollout/evaluation temperature、top-p/top-k、max prompt/response length、weight decay、gradient clipping、precision、随机种子与 wall-clock。

## 实验设置

- General visual perception：BLINK、MMStar
- Visual mathematics：MathVista
- Fine-grained/high-resolution：V*Bench、HRBench4K、HRBench8K
- Hallucination：HallusionBench，其表项先平均 aAcc/fAcc/qAcc
- Aggregate Acc：上述七个 benchmark 的无权平均
- 消融主要在 Qwen3-VL-2B 上进行

## 实验结论

1. **六个模型规模的 aggregate 均高于 base 与 answer-hint OPSD。** 相对 base 增益为 +1.86 至 +4.77 points。
2. **Qwen3-VL：** 2B 62.27→67.04，4B 71.30→73.16，8B 72.51→76.26。这里起点是 base，不是 OPSD；对应 OPSD 分别为 64.89、71.40、73.72。
3. **Qwen3.5：** 2B 68.61→71.51，4B 73.94→76.77，9B 74.97→79.24；answer-hint OPSD aggregate 分别为 66.18、73.92、74.73，两个规模低于 base，一个近乎持平。
4. **并非每项 benchmark 都提高。** 例如 Qwen3-VL-4B 的 BLINK/MMStar 低于 base；Qwen3.5-2B 的 HR8K 低于 base；Qwen3.5-9B 的 V*Bench 低于 answer-hint OPSD。
5. **Plausibility support 对长训练稳定性重要。** \(\beta=0\) 与 0.1 早期接近，之后 unrestricted target 持续退化；但论文未给 figure 的精确数值或 support cardinality。
6. **Moderate contrast 最好。** \(\alpha=0\) 保留 support-renormalized original-image target，但比 \(\alpha=1\) 低 2.33 aggregate points；\(\alpha\in[1,1.5]\) 相差不足 1 point，\(\alpha=2\) 退化到接近 no-contrast。
7. **Forward KL 明显优于 reverse KL。** Aggregate：forward KL 67.04、JSD 66.25、reverse KL 64.77。
8. **Control choice 的 aggregate 较稳健。** Gaussian noise 67.14、black 67.04、blur 66.36、no-image 66.24；black 并非平均最佳。
9. **Control choice 的逐任务差异仍明显。** 例如 MathVista 63.10–67.20、HallusionBench 51.25–56.31，因此不能理解为所有行为对 control 不敏感。
10. **原图 anchor 主要抑制 language drift。** 有/无 anchor aggregate 为 67.04/66.78，仅差 0.26；论文用 non-target-language rollout 比例显示 anchor 更稳定，但未提供文本数值。
11. **VCSD 训练曲线优于 OPSD。** 论文称 aggregate 每个评测 step 均更高、late-stage degradation 更小；仅展示同一 run，无多种子。

## 局限性

### 论文明确承认

论文未设置独立 limitations section。方法分析承认：

- 单纯 likelihood contrast 会放大原图下仍极低概率的 token，必须用 plausible support 约束；
- contrast 太强会压过原图 teacher anchor 并降低性能；
- EMA 递归 target 会累积 distortion，无 support 时长期训练退化。

### 由实验设计可直接确认

- 只验证 Qwen3-VL/Qwen3.5 与 ViRL39K，跨模型家族和视频/多图/3D 场景未知。
- 无多训练随机种子、标准差或显著性检验。
- 仅 90 steps、8×B200；未报告 wall-clock 和相对普通 OPSD 的额外 teacher-forward 成本。
- 每个 student prefix 需要 original/control 两次 EMA teacher distribution；“无外部 teacher”不等于低训练成本。
- “Full-distribution target”实际在 \(\mathcal S_t(\beta)\) 外严格置零，是 support-restricted vocabulary target。
- 没有直接测 student 对 image intervention 的 causal sensitivity、grounding localization 或视觉依赖与正确性的相关性。
- Answer-hint OPSD 在 Qwen3.5 上明显退化，VCSD 的优势部分来自对照较弱；仍需和原图-only EMA self-distillation、VA-OPD、VGS 做同 recipe 对照。
- Alpha/support/control 消融仅报告 2B，不能确认超参数跨规模稳定。

### 个人分析

- **EMA 系数存在关键复现歧义。** 方法公式使用 decay \(\mu\)，实验使用 update rate \(\rho=0.05\)；若没有代码，无法知道 teacher 是 0.95 old + 0.05 student，还是 0.05 old + 0.95 student。
- **PMI 解释是条件性的。** Black image 并不等价于 marginalizing image；它仍产生视觉 token、位置编码和 out-of-distribution activation。No-image control 又改变 token count/path，因此只能作为 intervention proxy。
- **“无 visual evidence signal”表述需收紧。** VCSD 不需要外部 crop/region annotation，但 original-vs-erased distribution difference 本身就是自动构造的视觉证据信号。
- **Visual reward 不保证 correctness。** 某 token 对特定图像高度敏感，不代表其描述正确；teacher 的视觉误读也会被平方式 sharpening 放大。
- **Support threshold 是熵相关的硬截断。** \(\beta\max p\) 在低熵位置可能只留极少 token，在高熵位置保留很多；论文未报告 support size、tail mass 或 hard cutoff sensitivity。
- **Aggregate control robustness 掩盖任务差异。** Noise/black/blur/no-image 平均接近，但各 benchmark 波动可达 4–5 points，说明 control 干预并非行为等价。
- **摘要数值衔接容易误读。** “Outperforms matched OPSD”后列出的 62.27→67.04 等起点实际是 base，而非 OPSD；真正 2B OPSD→VCSD 是 64.89→67.04。
- **理论结论较弱。** Remark 1 只证明给定 hand-defined reward 和 reference distribution 后 exponential-tilted target 是 KL-regularized optimum，不证明 content contrast 是无偏视觉信息或最优训练信号。
- **Uniform position loss 仍可能稀释视觉关键 token。** 虽然 full-vocabulary target 在每个位置被重塑，但 response 中大量语言 scaffold 仍参与平均；可结合 VA-OPD 的 token selection。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可用原场景与 content-erased/structure-erased control 的 same-teacher distribution difference 自动构造 target asymmetry，无需答案或局部标注。
- 用原场景 teacher distribution 限制 plausible support，可避免纯 likelihood ratio 放大异常低概率动作/token。
- 将 spatial contrast 写入完整 target distribution，比只在 sampled token 上给标量 gate 提供更密集监督。
- EMA teacher 可保持 self-distillation target 与 student 同源，推理时无额外分支。

### 个人建议

- 对空间输入分别构造 RGB erased、depth erased、object-ID shuffled、topology broken、coordinate masked controls，得到 factor-specific contrast，而不是单一黑图。
- Control 必须保持 token count、位置编码与 preprocessing path，并增加 in-distribution matched control，降低 OOD artifact。
- 对 target support 记录 cardinality、retained mass 和 entropy；比较 hard threshold、top-\(K\)+tail 与 soft support。
- 把 visual/spatial contrast 与 correctness verifier 联合，避免 sharpening visually sensitive but wrong token。
- 明确 EMA 公式并对 decay 0.95/0.99 与 near-current teacher 做消融；记录 teacher-student KL 和 target drift。

## 与其他论文的相同点和冲突

### 与 [VA-OPD](va-opd.md)

- 两者都比较视觉充分/受限条件下 teacher distribution，并认为视觉相关监督集中且语言 prior 会稀释它。
- VA-OPD 用 sampled-token log-prob difference 做 rollout weighting 和 high/low token grouped KL；VCSD 对全词表 candidate 计算 contrast，直接重塑 teacher target。
- VA-OPD 使用固定更大外部 teacher 与 pixelated image；VCSD 使用 EMA self-teacher 与 same-size content-erased image，无外部模型。

### 与 [VGS](vgs.md)

- 两者最接近的公式都是 image-conditioned 与 text/no-image distribution 的 log-ratio/information gain。
- VGS 保留 student text-only prior、注入固定 teacher visual gain，并从 gradient geometry 角度 steering；VCSD 以 EMA teacher 原图分布为 anchor，对 support 内 target 做 exponential tilting，再用 forward KL。
- VGS 明确加入 language-preservation loss；VCSD 依靠 original-image anchor 和 support 抑制 language drift。

### 与 [Vision-OPD](vision-opd.md)

- 都是 EMA/self-teacher OPSD。Vision-OPD 用 evidence crop 制造 teacher 输入优势；VCSD 不需区域 crop，用原图/control contrast制造 target asymmetry。
- Vision-OPD 依赖数据构造但 teacher 视觉证据更明确；VCSD 标注成本低，却依赖 model-internal contrast 是否忠实反映视觉内容。

### 与 [V-Zero](v-zero.md)

- 两者都用 paired visual conditions 且不消费 textual answer label。V-Zero 用 target/random crops 给 sibling trajectory 加权；VCSD 用 original/content-erased full-vocabulary contrast重塑每个 token target。
- V-Zero 依赖 region crop 与外部 27B teacher；VCSD 无 region annotation、使用 EMA self-teacher。
- V-Zero 主要解决 trajectory selection；VCSD 主要解决 token-distribution target construction，二者可组合。

### 与 [ViCuR](vicur.md) 和 [ViGOS](vigos.md)

- ViCuR 用离线 visual cue 替代 answer privilege；ViGOS 保留答案但延迟介入；VCSD 完全不消费答案或 cue，仅从输入 intervention 构造不对称。
- VCSD 无额外 inference module/token；ViCuR 增加 prefill branch，ViGOS 增加 description generation。
- VCSD 避免 privilege provenance 问题，但引入 control validity、EMA drift 与 self-reinforcement 风险。

### 与 [VOLD](vold.md)

- VOLD 用外部文本 teacher、SFT alignment 和 verifier 迁移 reasoning；VCSD 用同源视觉 EMA teacher，不做 correctness gating。
- 两者可组合：VOLD 的 correctness mask 可防止 VCSD 强化 visually sensitive but wrong trajectories。

### 与 [PCD](pcd.md)

- VCSD 构造“应该学习什么”视觉 target；PCD 判断“当前失败是否应纠正 perception”并分配 trajectory weight。
- VCSD 无 answer verifier，但可能强化 image-sensitive error；PCD 使用 verifier 和强 teacher gap，可作为外层 gate，但不再是 answer-label-free。

### 与 [VAD](vad.md)

- 两者都在 candidate distribution 层使用视觉条件 log-ratio重构 target。VCSD 对 EMA teacher 原图/control contrast做 exponential tilt；VAD 将完整 privileged correction投影到 frozen teacher crop/degraded direction。
- VCSD 不需 region annotation；VAD 的 intervention更聚焦，但受 student top-\(K\) support和一维 projection geometry限制。

### 与 [FP-OPD](fp-opd.md)

- 两者都围绕 detached student重构 distribution target。VCSD 用 EMA teacher visual contrast做 sharpening；FP-OPD 将外部 teacher gap投影到 student Fisher probe span。

### 与 [SA-OPD](sa-opd.md)

- 两者都使用input control。VCSD以black-image contrast塑造candidate target；SA-OPD以no-prompt disagreement gap删除token位置，均需审计control-induced OOD shift。

### 与 [OPD-V](opd-v.md)

- 两者都使用EMA teacher与visual control。VCSD用original/black-image contrast塑造target；OPD-V用zoom/masked-crop contrast筛选并加权位置。

### 综合定位

十三篇论文中，VCSD 新增“无需辅助 privilege 的 target asymmetry”路线。它位于 VA-OPD/VGS 的视觉 distribution decomposition 与 Vision-OPD 的 EMA self-distillation 交叉点：不选择 rollout 或单独 steering gradient，而是直接构造 support-restricted visually sharpened target。

## 证据

### E-001

- 结论：VCSD 只用 original image 与同尺寸 content-erased control 构造 self-teacher asymmetry。
- 类型：论文结论
- 定位：§1；§3.1–3.2；式 (2)–(6)
- 必要引用：默认 control 为 black RGB，保持 visual-token count 与 preprocessing path。
- 备注：不依赖外部答案/crop，不等于没有自动视觉信号。

### E-002

- 结论：VCSD 在 original-image plausible support 内用 conditional log-ratio重塑 teacher target。
- 类型：论文结论
- 定位：§3.3；式 (7)–(9)
- 必要引用：默认 \(\alpha=1,\beta=0.1\)。
- 备注：support 外 target probability 为 0。

### E-003

- 结论：实际目标是 full-distribution forward KL，teacher 通过 EMA 更新。
- 类型：论文结论
- 定位：§3.4；式 (10)
- 必要引用：\(T_{\mathrm{KD}}=2\)，每 prompt 8 rollouts。
- 备注：EMA update rate 语义待核对。

### E-004

- 结论：Contrast-shaped target 是给定 visual reward 下 support-restricted KL-regularized improvement 的唯一解。
- 类型：论文结论
- 定位：§3.5；Remark 1；式 (11)–(14)；附录 A
- 必要引用：无
- 备注：不证明该 reward 与 ground-truth correctness 一致。

### E-005

- 结论：\(\Delta\) 只在 control prediction 近似无 instance-specific visual information 时近似 conditional PMI。
- 类型：论文条件性结论
- 定位：§3.5；式 (15)–(16)
- 必要引用：无
- 备注：black/noise/blur 均非严格 image marginal。

### E-006

- 结论：VCSD 在六个模型规模的七项 aggregate 上均超过 base 与 answer-hint OPSD。
- 类型：论文结论
- 定位：§4.2；表 1
- 必要引用：相对 base +1.86 至 +4.77。
- 备注：并非每个 benchmark 单调提高。

### E-007

- 结论：摘要列出的 62.27→67.04 等起点是 base，不是 matched OPSD。
- 类型：个人核对
- 定位：摘要；表 1
- 必要引用：Qwen3-VL-2B OPSD 为 64.89。
- 备注：不影响 VCSD 超过 OPSD 的事实，但表述容易误读。

### E-008

- 结论：\(\alpha=0\) 相对 \(\alpha=1\) 低 2.33 points，说明 contrast shaping 有独立贡献。
- 类型：论文结论
- 定位：§4.4；图 3(b)
- 必要引用：\(\alpha=0\) 仍保留 support-renormalized original-image target。
- 备注：未给所有点的精确表格。

### E-009

- 结论：Forward KL 在同 target 下高于 JSD 和 reverse KL。
- 类型：论文结论
- 定位：§4.5；表 2
- 必要引用：67.04 vs 66.25 vs 64.77。
- 备注：仅 Qwen3-VL-2B。

### E-010

- 结论：不同 control 的 aggregate 接近，但逐 benchmark 差异明显。
- 类型：个人核对
- 定位：§4.6；表 3
- 必要引用：aggregate 66.24–67.14；MathVista 63.10–67.20；HalluB 51.25–56.31。
- 备注：支持平均鲁棒性，不支持行为等价。

### E-011

- 结论：Original-image anchor 对 aggregate 贡献小，但降低 non-target-language drift。
- 类型：论文结论
- 定位：§4.7；表 4；图 3(c)
- 必要引用：67.04 vs 66.78。
- 备注：drift 曲线没有文本数值。

### E-012

- 结论：EMA decay/update rate 的实现语义不明确。
- 类型：个人核对
- 定位：§3.4；§4.1
- 必要引用：公式 \(\phi\leftarrow\mu\phi+(1-\mu)\theta\)；实验称 \(\rho=0.05\)。
- 备注：需源码确定 \(\mu=0.95\) 还是 0.05。

## 待验证问题

- [ ] EMA 的 \(\rho=0.05\) 对应 old-teacher decay 还是 new-student update fraction？
- [ ] Designated sequence-termination tokens 具体包含哪些 token？
- [ ] Plausibility support 的平均 cardinality、retained mass 和 entropy 分布是多少？
- [ ] Black/noise/blur control 与真正 text-only marginal 的 distribution distance 多大？
- [ ] VCSD contrast 与 token correctness、region grounding、student intervention sensitivity 的相关性如何？
- [ ] 多随机种子下 1–4 point aggregate 增益是否稳定？
- [ ] 相对原图-only EMA OPSD、VA-OPD、VGS 的同 recipe 结果如何？
- [ ] 视频、多图、深度和空间图输入应使用何种 content-erased control？
- [ ] 训练 wall-clock 与两次 full-vocabulary teacher forward 的显存成本是多少？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，正文 §1–5、式 (1)–(17)、表 1–4、图 1–6；附录 A、式 (18)–(29)。
- 新增认识：Self-distillation 的不对称不必来自 teacher-only privilege，也可来自同输入的受控条件差，并直接形成 vocabulary-level target shaping。
- 修正内容：未把 content contrast 当作严格 PMI，未把 visual reward 当作 correctness reward，区分 support-restricted target 与真正无截断 full-vocabulary target。
- 下一步：核对 EMA 实现、support 统计和 control validity，并与 VGS/VA-OPD 做统一公式及实验对照。
