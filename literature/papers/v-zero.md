# V-Zero

## 基本信息

- 论文标题：V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating for Fine-Grained Visual Reasoning
- 作者：Haoxiang Sun, Zhihang Yi, Langxuan Deng, Yuhao Zhou, Peiqi Jia, Jian Zhao, Li Yuan, Jiancheng Lv, Tao Wang
- 年份与会议：2026；arXiv 预印本（v1）
- arXiv/DOI：arXiv:2606.25319；DOI: 10.48550/arXiv.2606.25319
- 论文链接：https://arxiv.org/abs/2606.25319
- 代码与数据：https://github.com/eVI-group-SCU/V-Zero（论文称将发布）
- 本地文件：无
- 阅读状态：已完成（crop 数据来源、消融语义与 checkpoint 选择待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

V-Zero 让学生从全图采样 8 条 sibling rollouts，再让 27B 教师分别在目标区域 crop 与随机负 crop 条件下重评分，以组内标准化的正负证据 log-prob 差给每条轨迹分配 \(0\!-\!2\) 权重，并只蒸馏 positive-crop teacher；它无需文本答案标签或推理时视觉工具，但门控衡量的是相对视觉证据支持而非答案正确性，且现有消融没有干净分离 positive crop target 与 contrastive gate 的贡献。

## 核心问题

标准 OPD 在 student-generated prefixes 上提供密集 token correction，但把每条 rollout 近似等权处理。若学生已进入错误推理路径，teacher 只能在该 prefix 上给局部 next-token target，并不直接判断整条 trajectory 是否可靠。

论文研究：

1. OPD 能否解释为 student/teacher views 之间的 negative-free stop-gradient alignment；
2. 在不使用 annotated textual answer labels、verifiable reward 或 agentic visual search 的前提下，如何获得 trajectory-level discrimination；
3. 问题相关 crop 与无关 negative crop 的 teacher support 差，能否作为视觉证据依赖代理；
4. 用该代理门控 positive-view OPD，能否提升细粒度全图推理并保留 OOD 能力。

## 方法

### 整体流程

1. 训练样本包含 full image、question 与 question-relevant regional crop \(z^+\)；另从目标区域外生成 equal-size random negative crop \(z^-\)。
2. 4B student 只看 full image 和 question，每个 prompt 采样 \(G=8\) 条 sibling trajectories。
3. 27B teacher 在同一 student token prefix 上做两次 replay：
   - full image + positive target crop；
   - full image + negative random crop。
4. 对每个 sampled token 计算两种 teacher 条件下的 log-prob 差，并沿 trajectory 平均得到 evidence score。
5. 在同一 prompt 的 8 条 sibling rollouts 内做 mean/std 标准化，得到 relative evidence advantage。
6. 将 \(1+\) advantage clip 到 \([0,2]\)，作为 stop-gradient trajectory gate。
7. Negative view 只用于算权重；最终 token target 始终是 positive-crop teacher，以 gate 加权 sampled-token reverse-KL surrogate。
8. 推理时 student 只看 full image 和 question，不使用 crop 或视觉工具。

### OPD 的 alignment 解释

在 student-induced state 上：

\[
q_s^{(g,k)}=\pi_s(\cdot\mid x,y_{<k}^{(g)}),\qquad
q_t^{(g,k)}=\operatorname{sg}\!\left[
\pi_t(\cdot\mid x,z,y_{<k}^{(g)})
\right].
\]

论文将 OPD 写为 asymmetric negative-free alignment：

\[
\ell_{\mathrm{align}}^{(g,k)}
=d(q_s^{(g,k)},q_t^{(g,k)}).
\]

“Negative-free”表示标准 OPD 只有一个 teacher target view，没有显式 trajectory contrast；这是一种重述和设计视角，并非证明 OPD 存在统一理论性能上界。

### Contrastive evidence gating

Teacher sampled-token log-prob：

\[
\ell_{+,k}^{(g)}
=\log\pi_t(y_k^{(g)}\mid x,z^+,y_{<k}^{(g)}),
\]

\[
\ell_{-,k}^{(g)}
=\log\pi_t(y_k^{(g)}\mid x,z^-,y_{<k}^{(g)}).
\]

Token evidence gap 与 trajectory score：

\[
\Delta_k^{(g)}=\ell_{+,k}^{(g)}-\ell_{-,k}^{(g)},\qquad
p^{(g)}=\frac1{T_g}\sum_k\Delta_k^{(g)}.
\]

同 prompt 组内标准化：

\[
a^{(g)}
=\frac{p^{(g)}-\mu_x}{\sigma_x+\epsilon}.
\]

Stop-gradient gate：

\[
w^{(g)}
=\operatorname{sg}\!\left[
\operatorname{clip}(1+a^{(g)},w_{\min},w_{\max})
\right],
\quad w_{\min}=0,\;w_{\max}=2.
\]

该分数只表示 rollout 相对 siblings 更受 target crop 支持。它不检查最终答案，也不能保证高分 rollout 正确。

### V-Zero objective

Positive-view full-vocabulary reverse KL：

\[
D_{\mathrm{KL},+}^{(g,k)}
=D_{\mathrm{KL}}\!\left(
\pi_s(\cdot\mid x,y_{<k}^{(g)})
\middle\|
\pi_t(\cdot\mid x,z^+,y_{<k}^{(g)})
\right).
\]

实际 sampled-token detached log-ratio：

\[
\widehat d_{\mathrm{KL},+}^{(g,k)}
=\operatorname{sg}\!\left[
\log\frac{\pi_s(y_k^{(g)}\mid x,y_{<k}^{(g)})}
{\pi_t(y_k^{(g)}\mid x,z^+,y_{<k}^{(g)})}
\right].
\]

训练 surrogate：

\[
\widetilde{\mathcal L}_{\mathrm{V\text{-}Zero}}
=\frac1G\sum_g w^{(g)}\frac1{T_g}\sum_k
\widehat d_{\mathrm{KL},+}^{(g,k)}
\log\pi_s(y_k^{(g)}\mid x,y_{<k}^{(g)}).
\]

Gate 与 sampled log-ratio 均 detached；negative teacher 不作为 distillation target。

## 训练数据与训练流程

- 数据：Zooming without Zooming（ZwZ）整理的 23K 样本；每条含 full image、question、question-relevant regional crop。
- Negative crop：将 full image 下采样 2 倍后，从 positive target region 外随机裁出同尺寸区域，并预写入训练数据。
- Student：Qwen3.5-4B；teacher：Qwen3.5-27B；另消融 9B teacher。
- 框架：VeRL；主训练 8×NVIDIA RTX PRO 6000 96G。
- Batch size 32，PPO mini-batch size 16，\(G=8\)。
- Max prompt 25,000；max response 2,048；LR \(10^{-6}\)。
- 使用 VeRL 默认 sampled-token reverse-KL OPD estimator。
- Gate clip：\([0,2]\)。
- 主结果选择 step-60 checkpoint；未报告总 epoch、optimizer、scheduler、rollout sampling temperature/top-p、teacher/student precision、随机种子，均 `待核对`。

## 实验设置

- 细粒度/高分辨率：VStar、HR-Bench 4K/8K、ZoomBench full-image、MME-RealWorld
- OOD generalization：MMStar
- 对照：Qwen3-VL/Qwen3.5 base models；DeepEyes、Pixel-Reasoner、Thyme、DeepEyesV2；ZwZ
- 主要受控对照：Qwen3.5-4B base vs V-Zero-4B
- 跨系统对照使用不同 backbone、训练数据和硬件，只能说明竞争力，不能归因于 V-Zero

## 实验结论

1. **相对 4B backbone，六项平均提高约 3.2 分。** Qwen3.5-4B 为 73.7，V-Zero 为 76.9；论文按底层未舍入值称平均 +3.1。
2. **四项细粒度 benchmark 均提升。** 表 1 中 VStar +4.7、HR-4K +3.4、HR-8K +2.5、ZoomBench +5.6；正文写 HR-8K +2.0、ZoomBench +5.5，对应消融表的 82.1/57.7，而非主表的 82.6/57.8。
3. **通用/OOD 能力未明显退化。** MME-RealWorld 69.2→69.8，MMStar 71.8→74.4；但只有单次结果，无方差。
4. **相对视觉推理系统具有竞争力。** V-Zero 在 HR-4K、HR-8K、ZoomBench、MMStar 表中最高，但这些方法 backbone 与 recipe 不同。
5. **所谓 gating ablation 提高 perception average。** `None` 为 78.0，V-Zero 为 79.2；VStar、HR-4K、ZoomBench 提升，HR-8K 82.4→82.1 略降。
6. **Random-positive/random-negative 退化明显。** `Rand.` perception average 72.5，说明无关 crop 不能替代任务相关 positive evidence；但这不单独证明 contrastive gate 必要。
7. **更大 teacher 并非逐项更优。** 9B→4B 在 VStar、HR-8K 更高，27B→4B 在 HR-4K、ZoomBench 更高；平均仅 78.9→79.2。
8. **更大 sibling group 有益但成本更高。** \(G=4\)→8 的 perception average 为 78.1→79.2，主要收益来自 ZoomBench 54.1→57.7。
9. **Step 60 最优但训练不单调。** Perception average 从 step 0 的 75.3 到 step 60 的 79.2，step 70 回落到 77.8；不同 benchmark 最佳 step 不同。
10. **Wall-clock 较短。** V-Zero 在 8×RTX PRO 6000 上训练 4.8 h；论文对比 ZwZ 的 8×H100 约 1 天和 DeepEyes 的约 2 天，称 \(>5\times\)/\(>10\times\)。

## 局限性

### 论文明确承认

论文没有单列 limitations section。正文较谨慎地承认跨系统比较不是 backbone-matched controlled ablation，并指出 8K setting 中 gate 收益不明显、不同 benchmark 最佳 checkpoint 不同。

### 由实验设计可直接确认

- 只训练 Qwen3.5-4B，教师仅比较 9B/27B；跨架构和更大/更小 student 未验证。
- 没有多训练随机种子、标准差、置信区间或显著性检验。
- Main checkpoint 选 step 60，但未说明独立 validation set；若根据报告 test benchmarks 选择，会产生 checkpoint-selection bias。
- 与 SFT/RL 的 \(5\times/10\times\) 是跨硬件、跨算法的 wall-clock 比较，不是控制后的 FLOPs、GPU-hours 或同质量 time-to-target。
- 23K 数据虽无 textual answer label，但包含 question-relevant region crop，仍需要区域级特权监督；“label-free”仅限答案文本。
- Crop provenance 未说明：目标区域由人工标注、模型生成、答案辅助还是 ZwZ pipeline 构造，当前无法审计。
- Negative crop 只保证从 target region 外随机取样，可能仍含相关证据；positive crop 也可能缺少全局关系。
- Negative 是从 2× downsampled image 中裁取，而 positive 是目标区域 crop；contrast 同时改变区域相关性与图像质量/尺度。
- 每个 prompt 需要 \(G\) 次 student rollout、positive/negative 两套 teacher replay，训练推理次数较多；论文未给标准 OPD 的同硬件成本对照。

### 个人分析

- **Gate 不是 correctness discriminator。** \(p^{(g)}\) 测的是 positive crop 相对 negative crop 对该 token sequence 的支持。视觉证据充分但结论错误的 rollout 仍可能高分；它不能替代 VOLD 式 verifier。
- **Gate 只有组内相对意义。** 标准化强制 sibling advantage 均值接近 0；若 8 条都错，仍会产生高低权重；若 8 条都好，也会相互压低。称其为 rollout reliability 过强，更准确是 relative evidence support。
- **缺少关键的 `positive-only, no-gate` 消融。** 表 2 的 `None` 标为无 positive/negative evidence，`Rand.` 同时改变两个 crop；没有明确比较“同一个 positive-crop teacher target、但所有 \(w=1\)”与完整 V-Zero。因此无法把 78.0→79.2 单独归因于 contrastive gating，而非 positive target。
- **与 VA-OPD 高度相近。** 两者都用 teacher 在视觉充分/受限条件下的 sampled-token log-prob 差，再把 token 差汇总为 trajectory weight。V-Zero 的主要变化是 target/random crop、同 prompt sibling normalization 和 positive-crop OPD target。论文未引用更早发布的 VA-OPD，可能与投稿时间接近，但“缺少 trajectory discrimination”的新颖性主张需结合该并行工作理解。
- **“Negative-free alignment”主要是概念重述。** Stop-gradient asymmetric matching 能解释 OPD 形式，但没有严格证明缺少 negatives 就构成性能 ceiling，也没有推导 contrastive gate 的正确性或一致性。
- **长度平均会稀释稀疏视觉 token。** \(p^{(g)}\) 对整条 response 的 \(\Delta_k\) 平均，长语言 scaffold 会稀释局部 evidence token；可与 VA-OPD token grouping 结合。
- **Negative construction 存在混杂。** 若 negative crop 因下采样更模糊，\(\Delta\) 可能衡量清晰度而非 task relevance；需 equal-resolution、hard negative 和 region-swap 对照。
- **Step-60 主表存在数值不一致。** 表 1 HR-8K/ZoomBench 为 82.6/57.8，表 2–5 的主配置为 82.1/57.7，正文增量也使用后者，需核对是否来自不同 evaluation run。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可从同一场景构造 positive local map/object subgraph 与 negative region，比较 teacher 对 student trajectory 的支持差。
- 不需要答案 verifier，也能用视觉/空间证据形成 trajectory-level gate，再保留 token-level OPD 的密集监督。
- Sibling rollout 的组内对比可消除部分 prompt 难度与分数尺度差异。
- 训练时使用局部特权视图、部署时保持完整场景输入，可避免推理工具调用。

### 个人建议

- 将 gate 明确命名为 spatial-evidence advantage，不当作 correctness advantage；与任务 verifier 组成 evidence×correctness 二维门控。
- Positive/negative 必须控制分辨率、面积、压缩和全局上下文，只改变空间证据相关性；增加邻近 hard negative、同类对象负例和关系破坏负例。
- 同时计算 token-level 与 trajectory-level gate，避免坐标、方向、对象 ID 等少数关键 token 被长度平均稀释。
- 必做四组正交消融：full-image OPD、positive-crop OPD、positive-crop+random gate、positive-crop+hard-negative gate。
- 若所有 siblings 的 absolute evidence score 都低，应整体关闭蒸馏，而不是组内强制选优；可结合 teacher confidence 和绝对 margin threshold。
- 记录 region provenance，确认 positive crop 不由答案、真值轨迹或部署不可恢复字段构造。

## 与其他论文的相同点和冲突

### 与 [Vision-OPD](vision-opd.md)

- 两者都用 question-relevant crop 作为训练时 teacher-side privilege，student 在完整图像上 rollout，推理时不需要 crop。
- Vision-OPD 是同 backbone crop/full-image self-distillation，强调 frozen/EMA teacher 与 top-\(K\) JSD；V-Zero 使用更大 27B fixed teacher、8 条 sibling rollouts 与 sampled-token reverse KL。
- V-Zero 额外引入 negative crop 和 trajectory gate，但没有干净消融 positive crop target 与 gate；Vision-OPD 更直接检验 crop teacher 本身。

### 与 [VA-OPD](va-opd.md)

- 两者机制最接近：都计算 teacher 在视觉充分/受限条件下对 student-sampled token 的 log-prob 差，沿 trajectory 聚合并重加权 OPD。
- VA-OPD 使用 original vs pixelated image，做 rollout softmax weighting 和 high/low token grouped KL；V-Zero 使用 target crop vs random crop，做同 prompt z-score 与 \([0,2]\) clipping，但只做 trajectory gate。
- 两者都不能判断答案正确性，可能强化视觉依赖但错误的 rollout；V-Zero 的 sibling normalization 进一步只有相对含义。

### 与 [VOLD](vold.md)

- 两者都对 trajectory 做选择性蒸馏。VOLD 用 exact-match reward mask，仅纠正错误 rollout；V-Zero 无答案 label，用视觉 evidence support 调整权重。
- Correctness 与 evidence dependence 互补：VOLD 可能保护语言 shortcut 的正确轨迹，V-Zero 可能强化 grounded but wrong 轨迹。

### 与 [VGS](vgs.md)

- V-Zero 决定“哪条 trajectory 多学”，VGS 决定视觉/语言目标“沿哪个梯度方向学”；二者可以组合。
- V-Zero 的 positive/negative sampled-token difference 不构造独立 visual target；VGS 使用 full-distribution image/text-only information gain。

### 与 [ViCuR](vicur.md) 和 [ViGOS](vigos.md)

- 三者都试图避免 OPD 过度依赖文本 shortcut。ViCuR 改 privilege 类型，ViGOS 控制答案 privilege 的介入阶段，V-Zero 完全不用 textual answer target、改用区域证据对比。
- ViCuR 的 cue 与 V-Zero 的 positive crop 都需 provenance audit；ViGOS 不需 region annotation，但需要显式 description 和 full-reference fallback。
- V-Zero 推理接口最接近标准 full-image response；ViCuR 增加 prefill module，ViGOS 增加生成 token。

### 综合定位

与 [VCSD](vcsd.md) 相比，两者都不消费 textual answer label，并使用 paired visual conditions。V-Zero 依赖 target/random crop 与外部 teacher，输出 trajectory gate；VCSD 使用 original/content-erased input 与 EMA teacher，输出 vocabulary-level shaped target。

与 [PCD](pcd.md) 相比，两者都进行 trajectory-level allocation。V-Zero 无答案 verifier，只衡量 sibling-relative crop support；PCD 用答案 reward 与 perception teacher gap定位可纠正 failure stage。

与 [VAD](vad.md) 相比，两者都使用 positive/negative visual views。V-Zero 将 contrast 变成 trajectory gate且继续匹配完整 positive teacher；VAD 将 contrast 变成 signed candidate direction并重构 target。

与 [FP-OPD](fp-opd.md) 相比，V-Zero 用 sibling contrast分配 trajectory weight；FP-OPD 不筛 trajectory，而在每个 prefix过滤 teacher correction的 Fisher direction。

与 [SA-OPD](sa-opd.md) 相比，两者都无需答案label。V-Zero用positive/negative crop做trajectory gate；SA-OPD用full/no-prompt disagreement做token deletion mask。

与 [OPD-V](opd-v.md) 相比，两者都用paired visual teachers；V-Zero做sibling trajectory-relative gating，OPD-V做独立token positive-margin gating。

十三篇论文中，V-Zero 新增“无答案的轨迹级视觉证据门控”维度。它位于 Vision-OPD 的 privileged crop 与 VA-OPD 的 visual-dependency weighting 交叉点：region crop 提供 teacher target，positive-negative contrast 提供 rollout weight。

## 证据

### E-001

- 结论：V-Zero 将标准 OPD 解释为 student/teacher views 上的 negative-free stop-gradient alignment。
- 类型：论文结论
- 定位：“Revisiting OPD”；式 (1)–(13)
- 必要引用：无
- 备注：属于概念重述，未严格证明性能 ceiling。

### E-002

- 结论：V-Zero 用 positive/negative crop teacher log-prob 差构造组内 trajectory evidence gate。
- 类型：论文结论
- 定位：Method；算法 1；式 (14)–(20)
- 必要引用：\(G=8\)，z-score 后 \(w=\mathrm{clip}(1+a,0,2)\)。
- 备注：是 relative evidence support，不是 correctness reward。

### E-003

- 结论：Negative view 只决定 gate，最终蒸馏 target 是 positive-crop teacher。
- 类型：论文结论
- 定位：式 (21)–(24)
- 必要引用：gate 与 sampled reverse-KL log-ratio均 stop-gradient。
- 备注：Student rollout 与 inference 都只用 full image。

### E-004

- 结论：V-Zero 不使用 annotated textual answer labels，但依赖 question-relevant region crop。
- 类型：论文结论
- 定位：摘要；Experiment Setup “Training Dataset”；图 4
- 必要引用：23K ZwZ data；crop 仅训练使用。
- 备注：“Answer-label-free”不等于无区域监督。

### E-005

- 结论：V-Zero 相对 Qwen3.5-4B 在六项平均上约提高 3.2 分。
- 类型：论文结论
- 定位：表 1；Main Results
- 必要引用：73.7→76.9。
- 备注：论文正文称 +3.1，可能使用未舍入底层值。

### E-006

- 结论：V-Zero 在四项细粒度 benchmark 均高于 base，并提高 MMStar。
- 类型：论文结论
- 定位：表 1
- 必要引用：VStar 89.0、HR-4K 87.8、HR-8K 82.6、ZoomBench 57.8、MMStar 74.4。
- 备注：无多种子或显著性检验。

### E-007

- 结论：表 2 不能单独识别 contrastive gate 的净贡献。
- 类型：个人核对
- 定位：Ablation Study；表 2
- 必要引用：`None` 无正/负 view；不存在 positive target + \(w=1\) 行。
- 备注：需要正交消融 positive teacher condition 与 gate。

### E-008

- 结论：Gate 只提供 sibling-relative evidence ranking，不能检测 absolute correctness。
- 类型：个人推断
- 定位：式 (18)–(20)
- 必要引用：同 prompt mean/std normalization，无答案或 verifier。
- 备注：all-wrong/all-good groups 仍被相对重加权。

### E-009

- 结论：Positive/negative contrast 混合了区域相关性与分辨率差异。
- 类型：个人核对
- 定位：Student Rollouts and Teacher Evidence Views；Training Dataset
- 必要引用：negative 从 2× downsampled full image 中裁取。
- 备注：缺少 equal-resolution hard-negative control。

### E-010

- 结论：主表与消融表的 step-60 主配置数值不一致。
- 类型：个人核对
- 定位：表 1–5；Main Results
- 必要引用：HR-8K 82.6 vs 82.1；ZoomBench 57.8 vs 57.7。
- 备注：正文增量对应后者，`待核对`。

### E-011

- 结论：V-Zero wall-clock 为 8×RTX PRO 6000 上 4.8 h。
- 类型：论文结论
- 定位：Training Cost
- 必要引用：相对 ZwZ/DeepEyes 声称 \(>5\times/>10\times\)。
- 备注：跨硬件、跨 recipe，不是受控算力效率比较。

### E-012

- 结论：Step 60 的 perception average 最优，step 70 回落。
- 类型：论文结论
- 定位：表 5
- 必要引用：75.3→79.2→77.8。
- 备注：未说明独立 validation checkpoint selection。

## 待验证问题

- [ ] 23K positive crop 如何构造？是否使用答案、人工框、teacher rationale 或检测标注？
- [ ] `None` 行究竟是 full-image standard OPD，还是 positive teacher 但 \(w=1\)？
- [ ] 增加 positive-only no-gate baseline，分离 crop target 与 contrastive gate。
- [ ] Equal-resolution random/hard-negative 下增益是否保留？
- [ ] Gate 与最终 correctness、student visual sensitivity 的相关性是多少？
- [ ] All-wrong sibling group 中 gate 是否会强化错误轨迹？
- [ ] 加 absolute evidence threshold 是否优于纯组内 z-score？
- [ ] 主表 82.6/57.8 与消融表 82.1/57.7 的差异来源是什么？
- [ ] Step 60 是否用 test benchmark 选择？多 seed 结果是否稳定？
- [ ] 相对同硬件 standard OPD 的 teacher FLOPs、wall-clock 与最终性能如何？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，正文全部章节、式 (1)–(24)、算法 1、表 1–5、图 1–4。
- 新增认识：在没有答案 verifier 时，可用 paired privileged evidence 对 sibling rollouts 做相对 gating；但 visual evidence dependence 与 correctness 必须分开。
- 修正内容：未把 answer-label-free 解释为 annotation-free，未把 gate 解释为 correctness reward，未把跨硬件 speedup 当作严格计算效率结论。
- 下一步：核对 crop provenance 和 `None` 消融，实现 positive-only、equal-resolution hard-negative 与 absolute-threshold 对照。
