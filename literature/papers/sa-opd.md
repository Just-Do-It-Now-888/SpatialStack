# SA-OPD

## 基本信息

- 论文标题：When Teachers Mislead: Spurious-Signal-Aware On-Policy Distillation
- 作者：Yinuo Jiang, Yongjie Ye, Zhou Tao, Xiang Zhuang, Qiang Zhang, Huajun Chen, Tiankai Li
- 年份与会议：2026；arXiv 预印本（v1）
- arXiv/DOI：arXiv:2608.03632；DOI: 10.48550/arXiv.2608.03632
- 论文链接：https://arxiv.org/abs/2608.03632
- 代码：https://github.com/jjjyinuo/SA-OPD（当前仓库页面未展示可核对实现）
- 本地文件：`uploads/2608.03632v1-0.md`
- 阅读状态：已完成（no-prompt implementation、FLMR 单位与动态阈值待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

SA-OPD 比较 sampled-token teacher–student divergence 在完整输入与 no-prompt 条件下的变化，将“变化小且原 divergence 极大”的位置从 full-vocabulary reverse-KL 中过滤，并用 FLMR 控制删除的监督质量；它在 LLM/VLM 对照中稳定优于 Vanilla OPD 和若干 selective baselines，但该 proxy 识别的是 disagreement 对输入移除的敏感性，不是 teacher grounding 本身，且存在 teacher/student effect cancellation、prefix leakage 与 no-prompt OOD 等可识别性问题。

## 核心问题

Dense OPD 默认每个 teacher signal 都值得学习。已有 selective OPD 关注：

- token entropy；
- teacher confidence；
- teacher–student divergence；
- local learnability；
- trajectory quality。

但强 divergence 可能来自输入无关的语言先验、格式偏好或固定 reasoning template。此类 signal 可能产生大更新，却与具体任务改进方向弱对齐。论文研究：

1. 如何定义 input-grounded 与 prior-driven OPD signal；
2. 如何用无需标签的反事实近似 token groundedness；
3. 如何只过滤低 groundedness 且高 optimization impact 的 token；
4. 如何约束过滤强度以避免删除过多有效监督。

## 方法

### Sampled-token divergence

Student 从当前 policy 生成：

\[
y\sim\pi_\theta(\cdot\mid x).
\]

Vanilla OPD 在 student prefixes 上优化：

\[
\mathcal L_{\mathrm{OPD}}
=\frac1L\sum_t
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x,y_{<t})
\|
\pi_T(\cdot\mid x,y_{<t})
\right).
\]

对 sampled token \(y_t\) 定义：

\[
A_t
=\log\pi_\theta(y_t\mid x,y_{<t})
-\log\pi_T(y_t\mid x,y_{<t}).
\]

将 \(A_t\) stop-gradient 时，单样本 score-function update 写成：

\[
g_t^{\mathrm{OPD}}
=-A_t\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t}).
\]

这是 reverse-KL gradient 的 sampled estimator 视角；实际训练在保留位置仍计算 full-vocabulary KL。

### Conceptual signal decomposition

论文概念上写：

\[
A_t=A_t^{\mathrm{grd}}+A_t^{\mathrm{prior}},
\]

并将 input-groundedness 定义为：

\[
\mathrm{IG}_t
=I(X;A_t\mid Y_{<t}).
\]

Prior-induced update 被假设具有较大 second-moment energy，但与 input-specific ideal update近似零期望 alignment，从而降低 effective gradient SNR。

附录进一步定义：

\[
A_t^{\mathrm{prior}}
=\mathbb E[A_t\mid C_t],
\qquad
A_t^{\mathrm{grd}}
=A_t-A_t^{\mathrm{prior}},
\]

其中 \(C_t=(Y_{<t},Y_t)\)。Theorem 1 在 score direction 对输入弱依赖的条件下上界 prior-gradient 与 input-specific ideal gradient 的 normalized alignment。Theorem 2 在 nuisance update是 martingale difference 的假设下证明，相对 signal-only trajectory 的 expected squared parameter drift为：

\[
\mathbb E\|\theta_K-\bar\theta_K\|^2
=\eta^2\sum_{k=0}^{K-1}\mathbb E[v_k].
\]

这些结论是条件性解释，不证明实际 filtered tokens满足相应假设。

### No-prompt groundedness proxy

对同一 rollout prefix，计算：

\[
A_t^{\mathrm{full}}
=
\log\pi_\theta(y_t\mid x,y_{<t})
-
\log\pi_T(y_t\mid x,y_{<t}),
\]

\[
A_t^{\mathrm{res}}
=
\log\pi_\theta(y_t\mid \varnothing,y_{<t})
-
\log\pi_T(y_t\mid \varnothing,y_{<t}).
\]

Input-Grounding Gap：

\[
\Delta_t^{\mathrm{IG}}
=
\left|
A_t^{\mathrm{full}}-A_t^{\mathrm{res}}
\right|.
\]

低 \(\Delta_t^{IG}\) 被解释为 teacher–student divergence 在删除输入后几乎不变，因此可能由 generic prior/template主导。

需要严格区分：

\[
\Delta_t^{IG}
=
\left|
(\Delta\log p_\theta)
-
(\Delta\log p_T)
\right|.
\]

它是 teacher 与 student 输入响应差的差，而不是 teacher 自身 input dependence。两者都强烈依赖输入但变化相同，也会得到低 gap。

### Two-factor token filtering

理论 threshold：

\[
\mathrm{Filtered}_t
=
\mathbf1[\Delta_t^{IG}<\tau_{IG}]
\mathbf1[|A_t^{full}|>\tau_A].
\]

实际使用 batch-relative quantiles：

\[
\mathcal F(p_1,p_2)
=
\operatorname{Bottom}_{p_1}(\Delta^{IG})
\cap
\operatorname{Top}_{p_2}(|A^{full}|).
\]

默认初始 \(p_1=0.2,p_2=0.3\)。附录实现将 high-impact set写成 \(A^{full}\) 的 top/bottom \(p_2\) tails，与 \(|A^{full}|\) top quantile应近似但在 tie/比例定义上需核对。

### Filtered Loss-Mass Ratio

\[
\mathrm{FLMR}(\mathcal F)
=
\frac{
\sum_{t\in\mathcal F}|A_t^{full}|
}{
\sum_{t\in\mathcal V}|A_t^{full}|+\epsilon
}.
\]

动态调节 \(p_1,p_2\)，使：

\[
\mathrm{FLMR}(\mathcal F)\le\beta.
\]

Visual understanding 使用动态约束；visual/math reasoning 未使用。论文表中 \(\beta=1.8\)，正文称约 1.8pp。由于 FLMR 按定义属于 \([0,1]\)，若 1.8 按无量纲值解释则约束恒真；实际应很可能是 1.8%/0.018，但当前文本单位不一致。

最终只在保留 token positions 上做 reverse KL：

\[
\mathcal L_{\mathrm{SA}}
=
\frac1{|\mathcal V\setminus\mathcal F|}
\sum_{t\notin\mathcal F}
D_{\mathrm{KL}}
(\pi_{\theta,t}\|\pi_{T,t}).
\]

## 训练数据与训练流程

- Framework：verl；PyTorch 2.10；CUDA 12.9；Python 3.12。
- Hardware：8×NVIDIA H20。
- Main LLM：Qwen3-4B-Instruct→Qwen3-1.7B。
- Main VLM：Qwen3.5-35B-A3B→Qwen3.5-2B。
- Scale checks：
  - DeepSeek-R1-0528-Qwen3-8B→Qwen3-1.7B；
  - Qwen3.5-9B→Qwen3.5-2B。
- LLM data：DeepMath difficulty ≥6 后随机取 30%，约 7K。
- Visual understanding：VERO-600K 的 Captioning & IF、Grounding、Counting & Search各取 10%。
- Visual reasoning：MMRL30k 取 10%。
- Batch size 128；AdamW；LR \(10^{-6}\)。
- Steps：visual understanding 700；visual reasoning 270；math reasoning 160。
- Max prompt：VLM 12000，LLM 2048。
- Max response：VLM 4096，LLM 8192。
- 初始 \(p_1,p_2=(0.2,0.3)\)。
- Visual understanding 的 FLMR bound 报为 1.8；其余两项为 `-`。
- 未报告 rollout 数、训练 sampling 参数、weight decay、LR schedule、warmup、gradient clipping、seed 与 checkpoint selection。

## 实验设置

- Visual understanding：EvoChart、MMIFEval、CountQA
- Visual reasoning：MathVision、Geo3K、MathVista-mini（正文简称 MathVista）
- LLM math：Math500、AMC 2023、AIME 2024/2025、MinervaMATH
- Zero-shot evaluation
- Sampling：temperature 1.0，top-p 0.95
- Generation：visual understanding 1024；visual reasoning 4096；LLM evaluation 18000
- Metrics：Math500/视觉任务主要为 accuracy/exact match；AMC/AIME/Minerva 为 Avg@8
- Baselines：Vanilla OPD、ExOPD、TIP、FiRe-OPD

## 实验结论

1. **Main VLM 六项均领先已有 OPD rows。** Qwen3.5-35B-A3B→2B 下，SA-OPD visual-understanding average 54.0 vs OPD 50.5；visual-reasoning average 63.5 vs OPD 60.4。
2. **Visual understanding 增益由 CountQA显著驱动。** EvoChart/MMIFEval/CountQA 相对 OPD为 +1.6/+1.7/+7.2；三项等权平均 +3.5。
3. **Visual reasoning 相对 OPD均提升。** MathVision/Geo3K/MathVista-mini 为 +1.2/+5.0/+3.2。
4. **Main LLM average +1.9 over OPD。** 28.5→30.4；相对最佳 selective baseline TIP 29.3 为 +1.1。
5. **LLM 单项并非全部独占最高。** AMC23 44.7 与 ExOPD并列；AIME25 8.3 与 FiRe并列；AIME24 11.7 与 TIP并列。
6. **跨 teacher scale仍为正。** 9B→2B VLM 六项均超过 OPD；R1-8B→1.7B LLM五项均超过 OPD，但表 3 未提供 aggregate和不确定性。
7. **Two-factor intersection 优于单因素。** Geo3K/MathVista：base 67.2/69.0，divergence-only 69.8/69.6，groundedness-only 69.0/70.3，SA-OPD 72.2/72.1。
8. **Random filtering无稳定收益。** 67.6/67.8；MathVista低于 base，支持收益不只是减少 token。
9. **Teacher-logprob replacement低于 disagreement proxy。** 69.6/71.5 vs SA 72.2/72.1；但只测两项，无统计检验。
10. **Filtered tokens主要是 content words。** 500 visual-understanding examples中，1967个 filtered positions的 69.9%为 content words，20.7%为 punctuation/format。
11. **Prefix repetition很常见。** 60.2% content tokens此前出现，27.9%属于 repeated bigram，13.9%属于 repeated trigram；各属性可能重叠，不能相加解释。
12. **VLM filtered mass更持久。** 论文图示 math FLMR早期降近零，VLM保持非零；但 FLMR由方法自身 proxy定义，不能独立证明 spurious contamination。
13. **额外开销为 2.64%–7.53%。** Math 5.19→5.54h，visual understanding 4.38→4.71h，visual reasoning 1.89→1.94h；无需额外 backward或参数。

## 局限性

### 论文明确承认或设计中明确处理

- Aggressive filtering可能误删有用 perceptual supervision，因此使用 FLMR约束。
- No-prompt scoring需要额外 teacher/student forward，但测得 overhead较小。

### 由实验设计可直接确认

- 无多 seed、方差、置信区间或显著性检验。
- 只测试 Qwen3/Qwen3.5 student family，LLM/VLM各一类训练 domain。
- VLM训练子集只按 10%描述，未给 exact sample counts、sampling seed与去重情况。
- “same compute budget”与 SA 额外 2.64%–7.53% wall-clock不完全一致；更准确是相同 steps/data/model budget。
- 没有与 Token Teachability、VA-OPD、VGS、VAD、FP-OPD等相近 signal-selection/target-filtering方法直接比较。
- Ablation只覆盖 Geo3K与MathVista-mini。
- Main VLM performance只报告单次 sampled evaluation，没有 greedy或重评稳定性。

### 个人分析

- **Proxy 测的是 disagreement groundedness，不是 teacher groundedness。** \(\Delta^{IG}\) 同时包含 teacher和student在输入删除前后的变化；低值可来自两者输入效应相互抵消。
- **Input dependence 不等于 task correctness。** 高 \(\Delta^{IG}\) 可能来自 image/prompt-conditioned hallucination、spurious correlation或格式变化；论文理论假定 grounded component更有正 alignment，但没有由定义推出。
- **No-prompt 是强 OOD intervention。** 保留完整 response prefix却删除原问题/图像，会形成语义不连贯条件；distribution shift可放大或缩小 divergence，而非纯粹移除 task evidence。
- **Response prefix 会泄露输入。** 越靠后，前缀可能已包含对象、数字、问题重述和中间结论；no-prompt model可从 \(y_{<t}\) 恢复 task信息，使真正 grounded token出现低 gap。
- **Sampled \(A_t\) 与 full-KL impact不等价。** Filter由单个 sampled token log-ratio决定，训练损失却是整个 vocabulary KL；\(|A_t|\) 不是该位置的 KL、gradient norm或curvature-adjusted impact。
- **FLMR 不是实际 loss mass。** 它使用 \(\sum|A_t|\)，而非 \(\sum D_{KL}\) 或梯度范数；名称可能高估其优化含义。
- **\(\beta=1.8\) 单位自相矛盾。** Ratio不可能超过1；应明确是 1.8%、百分点还是实现中的百分数标度。
- **理论是条件性而非识别证明。** Theorem 1依赖 score weak-input-dependence，Theorem 2依赖 martingale zero-mean nuisance；论文未测这些假设在 filtered set是否成立。
- **“Spurious”缺少外部 ground truth。** 训练收益和重复 token统计支持 filter有用，但没有人工/causal标注验证 filtered token确实错误或无关。
- **Batch-relative quantile使语义漂移。** 同一个 absolute groundedness/impact在不同 batch、task或训练阶段可能得到不同决定。
- **Loss renormalization改变剩余 token权重。** 过滤后除以 retained count，因此 SA-OPD既删除 token，也提高其余 token的平均权重；需与 mask-only固定分母区分。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 不应只根据 divergence大小选择 supervision，还应测该 disagreement是否依赖当前空间输入。
- 双条件 gate比仅看低 groundedness或高 divergence更稳健。
- FLMR式预算可限制一次过滤掉的高权重监督，适合不同空间任务动态校准。
- No-prompt/control scoring无需额外模型或标签，工程成本相对较低。

### 个人建议

- 不用完全 no-prompt；构造 matched spatial controls：保留问题文本，仅移除图像、对象位置、深度、坐标或关系边。
- 分别测 teacher 与 student input effect：

\[
\Delta_T=\log p_T^{full}-\log p_T^{ctrl},\quad
\Delta_S=\log p_S^{full}-\log p_S^{ctrl},
\]

再报告 disagreement effect \(\Delta_S-\Delta_T\)，避免 cancellation被隐藏。
- 对 response prefix做 leakage audit：按 token position、已出现实体/数字、问题重述程度分桶。
- 用 full-KL difference、Fisher gradient norm或VAD-style candidate vector替代 sampled-token \(|A_t|\) impact。
- 以固定原 token数归一化做消融，区分删除与 retained-token upweighting。
- 建立小规模人工/程序化 causal ground truth，测 filtered-token precision/recall，而不只看最终 benchmark。

## 与其他论文的相同点和冲突

### 与 [VA-OPD](va-opd.md)

- 两者都比较 full/control输入下的 teacher–student signal并做 token-level allocation。
- VA-OPD 用 teacher original/pixelated visual log-prob difference识别视觉依赖并增权；SA-OPD 用 full/no-prompt disagreement difference识别低 input dependence并删除。
- VA-OPD只看 sampled-token positive visual advantage；SA-OPD同时过滤 divergence正负 tails，但其 proxy混合 teacher与student response。

### 与 [VGS](vgs.md)

- 两者都试图隔离 language-prior supervision。VGS 用 teacher multimodal/text-only distribution构造 visual target并 steering gradient；SA-OPD 用 no-prompt disagreement筛 token。
- VGS保留并重定向视觉 correction；SA-OPD直接丢弃可疑位置，工程更简单但信息损失更大。

### 与 [VAD](vad.md)

- 两者都用 input intervention分析 teacher correction来源。VAD固定 teacher并用 evidence-present/degraded candidate vector做 signed target reconstruction；SA-OPD比较 teacher–student sampled divergence并做位置过滤。
- VAD更接近 source attribution但依赖 region/degradation；SA-OPD无需区域 privilege但存在 teacher/student cancellation与no-prompt OOD。

### 与 [FP-OPD](fp-opd.md)

- SA-OPD决定哪些位置不学习；FP-OPD决定每个位置 teacher gap中哪些方向与 student probe span兼容。
- 两者都避免盲目复制完整 teacher target，可组合为 groundedness gate × Fisher-projected target，但额外 forward成本会叠加。

### 与 [VCSD](vcsd.md) 和 [V-Zero](v-zero.md)

- 三者都依赖 original/control contrast。VCSD塑造 candidate target，V-Zero产生trajectory-relative evidence gate，SA-OPD产生token deletion mask。
- Control validity决定结论：black image、negative crop和no-prompt分别引入不同 distribution shift。

### 与 [PCD](pcd.md)、[ViGOS](vigos.md) 和 [ViCuR](vicur.md)

- PCD定位 perception failure，ViGOS控制 privilege介入阶段，ViCuR改善 cue recovery；SA-OPD过滤 input-insensitive high-impact supervision。
- 对空间任务可先按阶段和recoverability判断，再做 matched-control groundedness过滤。

### 与 [Vision-OPD](vision-opd.md) 和 [VOLD](vold.md)

- Vision-OPD/VOLD主要相信 teacher target并控制 view/state；SA-OPD明确假设 teacher在 student prefix上也可能输出 template-biased signal。
- SA-OPD补充了 teacher reliability维度，但没有 correctness verifier，不能识别 grounded yet wrong teacher。

### 与 [OPD-V](opd-v.md)

- 两者都处理输入证据相对语言先验的可靠性。SA-OPD删除full/no-prompt disagreement近似不变的位置；OPD-V选择zoom比masked crop更支持的位置。
- SA-OPD的proxy混合teacher/student input effects；OPD-V的proxy混合zoom、context loss与mask artifact，均不是可识别modality decomposition。

### 综合定位

十三篇论文中，SA-OPD 新增“input-grounded supervision reliability”维度：高 confidence、divergence、learnability或local realizability都不足以保证 teacher signal真正依赖当前任务输入；但 no-prompt disagreement gap仍只是可干预 proxy，不是 spuriousness的可识别分解。

## 证据

### E-001

- 结论：SA-OPD 用完整输入与 no-prompt 下 sampled teacher–student divergence差作为 input-grounding proxy。
- 类型：论文结论
- 定位：§3.2；式 (12)–(14)
- 必要引用：保持同一 student response prefix。
- 备注：衡量 disagreement sensitivity，不是 teacher-only grounding。

### E-002

- 结论：Filter取低 groundedness与高 absolute divergence的交集。
- 类型：论文结论
- 定位：式 (15)–(16)；算法 1
- 必要引用：默认 \(p_1=0.2,p_2=0.3\)。
- 备注：batch-relative quantile。

### E-003

- 结论：FLMR限制 filtered sampled-log-ratio mass。
- 类型：论文结论
- 定位：式 (17)–(18)、(59)；算法 2
- 必要引用：visual understanding报 \(\beta=1.8\)。
- 备注：比例与百分点单位不一致。

### E-004

- 结论：最终只在 retained positions优化 full-vocabulary reverse KL。
- 类型：论文结论
- 定位：式 (19)
- 必要引用：按 retained-token count重新归一化。
- 备注：filter selection signal与训练 loss粒度不同。

### E-005

- 结论：Main VLM visual understanding/reasoning averages为 54.0/63.5。
- 类型：论文结论
- 定位：表 1；§4.2
- 必要引用：OPD为 50.5/60.4。
- 备注：无多 seed。

### E-006

- 结论：Main LLM average为30.4，OPD为28.5。
- 类型：论文结论
- 定位：表 2
- 必要引用：五项best或tied-best。
- 备注：不同 benchmark混合 Acc@1与Avg@8。

### E-007

- 结论：Intersection filter在两个 ablation tasks均高于单因素与random。
- 类型：论文结论
- 定位：表 4；§4.3
- 必要引用：Geo3K 72.2，MathVista 72.1。
- 备注：只测两项。

### E-008

- 结论：额外 no-prompt teacher/student scoring增加2.64%–7.53% wall-clock。
- 类型：论文结论
- 定位：附录 D；表 8
- 必要引用：8×H20，无额外 backward。
- 备注：不同 task overhead差异较大。

### E-009

- 结论：低 disagreement gap不能识别 teacher自身是否input-grounded。
- 类型：个人推断
- 定位：式 (12)–(14)
- 必要引用：gap是 student input effect减teacher input effect的绝对值。
- 备注：存在 cancellation。

### E-010

- 结论：Response prefix可能向 no-prompt condition泄露原输入。
- 类型：个人推断
- 定位：式 (13)；算法 1
- 必要引用：删除 \(x\) 但保留完整 \(y_{<t}\)。
- 备注：需按位置和prefix content审计。

### E-011

- 结论：\(|A_t|\)-based FLMR不等于 full-KL loss mass或gradient mass。
- 类型：个人核对
- 定位：式 (17)、(19)
- 必要引用：filter用 sampled token log-ratio，loss用full distribution KL。
- 备注：术语需收紧。

### E-012

- 结论：理论结果依赖未实证验证的弱输入依赖与martingale nuisance假设。
- 类型：个人核对
- 定位：附录 A；Theorem 1–2
- 必要引用：\(\kappa_t\ll1\) 与 \(\mathbb E[\xi_k\mid\mathcal H_k]=0\)。
- 备注：不能作为 proxy正确性的无条件证明。

## 待验证问题

- [ ] No-prompt 对 VLM具体删除 image、question、system prompt中的哪些部分？
- [ ] \(\beta=1.8\) 实际是 0.018、1.8%还是其他内部标度？
- [ ] 动态 FLMR 的 \([\beta_{\min},\beta_{\max}]\) 如何设置？
- [ ] Teacher-only、student-only与disagreement groundedness哪个预测最终收益最好？
- [ ] Prefix leakage随 token position如何变化？
- [ ] Sampled \(|A_t|\) 与 full KL、gradient norm的相关性是多少？
- [ ] Filtered tokens经人工/causal标注后 spurious precision是多少？
- [ ] 固定原 denominator后收益是否保持？
- [ ] 多 seed、greedy与sampling重评下结果是否稳定？
- [ ] Matched visual/spatial controls是否优于完全 no-prompt？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，正文 §1–5、式 (1)–(19)、表 1–5、图 1–4；附录 A–E、Theorem 1–2、算法 1–2、表 6–8。
- 新增认识：Selective OPD除 confidence/divergence/learnability外，还需区分 supervision对当前输入的依赖性。
- 修正内容：未把 disagreement gap当作teacher grounding，未把 \(|A_t|\) FLMR当作精确KL/gradient mass，明确记录 \(\beta\) 单位矛盾与prefix leakage。
- 下一步：实现 matched spatial controls、teacher/student effect decomposition与causal filtered-token audit。
