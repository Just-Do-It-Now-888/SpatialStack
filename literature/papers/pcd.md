# PCD

## 基本信息

- 论文标题：Correcting What You Cannot See: Credit Assignment for Perception Distillation in Multimodal Reasoners
- 作者：Feng Xiong, Leyan Xue, Hongyu Lin
- 年份与会议：2026；arXiv 预印本（v2）
- arXiv/DOI：arXiv:2607.28336；DOI: 10.48550/arXiv.2607.28336
- 论文链接：https://arxiv.org/abs/2607.28336
- 代码：未提供公开仓库链接
- 本地文件：无
- 阅读状态：已完成（主配置、KL normalization 与标签语义待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

PCD 将 response 拆成 perception 与 reasoning 两段，每个 prompt 先采 2 个 perception、再各采 4 个共享该 perception 的 reasoning，用下游失败率 \(1-\mathrm{PSR}\) 与 perception-span teacher disagreement 的乘积识别“失败且可由教师纠正”的感知轨迹，并在保持 perception 权重均值为 1 的前提下重分配 OPD；它改善视觉数学迁移，但仍依赖答案 verifier，且主表配置与 VPPO 消融标签存在明显不一致。

## 核心问题

完整答案失败可能来自两个阶段：

1. perception 没有提取足够视觉证据；
2. perception 基本正确，但后续 reasoning 失败。

Outcome reward 只能看到最终成败，uniform OPD 则无差别纠正所有 perception。仅为同一 perception 采多个 reasoning 并计算 Perception Success Rate（PSR）仍不够：低 PSR 可能是 perception 差，也可能只是题难或当前 reasoner 弱。

论文研究：

1. 如何在没有 perception ground-truth label 时识别 teacher-correctable perception failure；
2. 下游 failure 与 teacher–student disagreement 能否作为互补 witness；
3. 如何只重分配 perception distillation，而不改变 reasoning RL 的公式和总权重尺度；
4. 分离 perception/reasoning sampling 是否能建立更低方差的 per-perception credit unit。

## 方法

### Perception–reasoning factorization

\[
\pi_\theta(y,z\mid x)
=\pi_p(z\mid x)\pi_r(y\mid x,z),
\]

其中 \(z\) 是带 delimiter 的 perception span，\(y\) 是 reasoning span。

每个 prompt 先采样 \(a\) 个 perceptions：

\[
z_i\sim\pi_p(\cdot\mid x),
\]

再对每个固定 \(z_i\) 采样 \(b\) 条 reasoning：

\[
y_{ij}\sim\pi_r(\cdot\mid x,z_i).
\]

默认 \(a=2,b=4\)，总预算仍为 8 条 completed trajectories。所有 8 条进入同一 DAPO reward group，但同一 perception 下的四条 trajectory 共享 prefix。

### Perception Success Rate

\[
\mathrm{PSR}_i
=\frac1b\sum_{j=1}^bR(x,z_i,y_{ij}).
\]

它是当前 reasoner 条件下 perception value 的 Monte Carlo estimate。对 Bernoulli reward，条件方差为 \(V(z_i)(1-V(z_i))/b\)。

论文用简化模型

\[
V(z)\approx \rho(x)q(z)
\]

说明 reward 只能识别 reasoning success \(\rho\) 与 perception sufficiency \(q\) 的乘积，无法仅从 PSR 分离二者。这证明的是该建模假设下的 non-identifiability，不是对任意现实 reward process 的普遍因果定理。

### Teacher disagreement

Perception-span detached disagreement：

\[
\mathrm{KL}_i
=\left\langle
\log\pi_\theta^{\mathrm{old}}(t)
-\log\pi_T(t)
\right\rangle_{t\in z_i}.
\]

实现使用与 distillation 相同的 top-\(k=64\) support 和 log-prob clamp，先在 trajectory 内平均，再在共享 perception 的 \(b\) 条 continuations 间平均，clamp 为非负并归一化到
\(\widetilde{\mathrm{KL}}_i\in[0,1]\)。配置给出 normalization threshold 0.3，但正文未写出精确映射公式。

### Two-witness deficiency gate

\[
d_i
=(1-\mathrm{PSR}_i)\widetilde{\mathrm{KL}}_i.
\]

解释：

- 高 PSR：不强化 perception correction；
- 低 PSR、低 KL：teacher 也基本同意，可能是 reasoning 难或 shared perception error；
- 低 PSR、高 KL：更像 teacher-correctable perception deficiency。

论文以 conditional-independence 下 binary witness likelihood ratios 的相乘，动机化 continuous product。它没有证明 PSR 与 KL 在现实数据中条件独立，也没有把 \(d_i\) 校准为 posterior probability。

在满足 \(g(a,0)=g(0,b)=0,g(1,1)=1\) 的 bilinear gate 中，乘法 \(g(a,b)=ab\) 唯一；这是由预先选定的 AND boundary conditions 得到的代数结论，不排除 nonlinear 或非零单 witness gate。

### Mean-preserving weighting

\[
\bar w_i=w_{\mathrm{base}}+\alpha d_i,\qquad
w_i=\frac{\bar w_i}
{\frac1N\sum_k\bar w_k}.
\]

默认 \(w_{\mathrm{base}}=1,\alpha=1\)，最终 perception-level weights 均值为 1。

Perception loss：

\[
\mathcal L_{\mathrm{aware}}
=\frac{
\sum_{i,j,t}w_i m^{\mathrm{aw}}_{ijt}\ell_{ijt}
}{
\sum_{i,j,t}m^{\mathrm{aw}}_{ijt}
}.
\]

总目标：

\[
\mathcal L
=\lambda_{\mathrm{aw}}\mathcal L_{\mathrm{aware}}
+\lambda_{\mathrm{cot}}\mathcal L_{\mathrm{cot}}^{\mathrm{DAPO}},
\]

其中 \(\lambda_{\mathrm{aw}}=0.1,\lambda_{\mathrm{cot}}=1.0\)。PCD 不改变 DAPO advantage 公式，只改变 perception sampling structure 与 distillation weights。

论文证明：若假设“额外 teacher supervision 的局部收益正比于 \(d_i\)”，则在 zero-mean、L2-bounded weight perturbation 下，最优一阶方向为 \(d_i-\bar d\)。该结论依赖关键收益假设，本身不能验证 \(d_i\) 是正确 utility。

## 训练数据与训练流程

- 模型：Qwen3-VL-8B→2B；Qwen3-VL-32B→8B。
- 数据：Geo3K（div format），训练 2 epochs。
- Reward：可验证最终答案；reasoning 使用 DAPO-style RL。
- Rollout：\(a=2,b=4\)，每 prompt 共 8 trajectories；perception 最长 512 tokens，总 response 最长 2048。
- Sampling：temperature 1.0，top-p 1.0。
- Teacher：top-\(k=64\) token probabilities，tensor parallel 4。
- Train batch 128，mini-batch 32，actor LR \(10^{-6}\)，10-step warmup，grad clip 1.0。
- DAPO clip：low 0.2、high 0.28、dual-clip \(c=10\)。
- KL normalization threshold 0.3，log-prob clamp -10。
- Format bonus 0.1；repetition/malformed penalties 各 0.5。
- Hardware：8×B200；4 student/rollout GPUs + 4 teacher-inference GPUs。
- Validation：每 5 steps；checkpoint 由 held-out validation 选择。

## 实验设置

- ID：Geo3K
- Near-OOD：MathVerse、MathVista、MATH-Vision、We-Math
- OOD：LogicVista、MMMU-Pro、MMStar
- 评测：每问题 8 个独立 samples，Avg@8；temperature 1.0，top-p 1.0
- Macro average：八个 dataset 等权，不按样本量加权
- Baselines：Base、same-size DAPO、standard OPD；2B 提供组件消融

## 实验结论

1. **8B→2B 相对 OPD macro +2.78。** 44.50→47.28；但只比 DAPO 46.60 高 0.68，且无多种子显著性。
2. **32B→8B 相对 OPD macro +4.28。** 56.94→61.22；相对 base 56.92 也有明显提升。
3. **2B 收益主要集中于视觉数学。** 相对 OPD，Geo3K +11.30、四项 Near-OOD mean +2.99，而三项 OOD mean -0.33。
4. **8B transfer 更广，但 teacher size 与 student size 同时变化。** Geo3K +17.14、Near-OOD +3.43、OOD +1.15；不能据此单独归因于 teacher 更强。
5. **不是逐项提高。** 2B PCD 相对 OPD 在 LogicVista -1.95、MMStar -0.86；8B 在 LogicVista 仅 +0.06 且低于 base 0.22。
6. **权重行为符合预设 AND gate。** 256 个 Geo3K perceptions 中，高 failure+高 disagreement 象限平均权重 1.36，其他象限 0.85–0.92。该图验证实现行为，不证明这种分配导致准确率提升。
7. **消融中的 adaptive weighting 有明显贡献。** `Full` held-out average 49.28；去 PCD weight 为 47.06（-2.22），保持 separated rollout 与 divided objectives。
8. **Separated rollout 有独立贡献。** 去掉后为 48.40（-0.88）；说明 shared-perception continuations 有益，但也说明 sampling 改变不只是“保持 reasoning objective 不变”。
9. **完整 perception distillation 略高于 VPPO token selection。** 49.28 vs 48.88（+0.40）；作者认为视觉 token filter 可能漏掉 connective/implicit visual token。
10. **运行时间并未增加。** 8B→2B：OPD 183.9 s/step，PCD 132.5；32B→8B：417 vs 412。2B 加速部分来自只蒸馏较短 perception span，token 数不等价。

## 局限性

### 论文明确承认

- 缺少 PSR-only、KL-only、additive-fusion 与 multiplicative gate 的 matched runs，因此尚不能把增益归因于乘法 interaction。
- 每种方法只使用一个 selected checkpoint；缺少 multi-seed intervals。
- PSR 随 policy 非平稳；固定 rollout budget 下增大 \(b\) 会降低 PSR variance，却减少 perception diversity。
- Teacher disagreement 只有在 teacher perception 更可靠时才有意义；shared errors 得到低 KL，confidently wrong teacher 会造成有害 correction。
- KL normalization threshold 需按 teacher size 的 training-trace quantile 校准。
- 只验证一个模型家族与 Geo3K 训练域；ID/OOD 是任务相似度描述，不是形式化距离。

### 由实验设计可直接确认

- “Label-free”仅指不需要 perception label；完整方法仍依赖 Geo3K answer/verifier reward 与 DAPO，不能称为 answer-label-free。
- \(b=4\) 时 PSR 只有 \(\{0,0.25,0.5,0.75,1\}\) 五种值，单 perception estimate 仍较粗。
- Qualitative cases 特意筛选“OPD 全错、PCD 正确”，只能展示成功模式，不能估计典型错误率或因果机制。
- Standard OPD 与 PCD 主对照同时改变 rollout structure、span-specific objective、distillation scope 和 weighting；只有内部消融能分离部分因素。
- 没有验证 perception span 是否只写视觉观察；它可能包含推理、答案暗示或语言模板，使 KL 不完全对应 perception disagreement。
- Teacher disagreement 可能来自表达风格、长度或置信度，而不只是视觉错误。

### 个人分析

- **主表配置与消融表发生内部不一致。** 主表 2B PCD 的 Geo3K/MMMU-Pro/We-Math/MathVista/MathVerse/MathVision/MMStar 数值与表 2 的 `Full w/ VPPO-Distill` 完全相同，而不同于 `Full (ours)`。但正文把 PCD 默认定义为 complete perception-span distillation，并称 VPPO 是 substitute。需核对主结果到底使用哪一配置。
- **“Fixed supervision budget”只在 perception-level 均值上成立。** \(N^{-1}\sum_iw_i=1\)，但式 (14) 按 token 求和且不先对每个 perception 归一化。若 perception 长度不同，\(\sum_iw_i|z_i|\) 不一定等于 \(\sum_i|z_i|\)，实际 token-weighted teacher loss scale仍会变化。
- **Separated rollout 改变 sampling distribution。** 虽然 DAPO advantage 公式和总 trajectory 数不变，2×4 tree 引入 shared-prefix correlation，与 8 条独立 joint rollouts 的 reward/gradient variance不同；不能理解为 reasoning optimization 完全不变。
- **乘法优越性尚未被实验证明。** Proposition 3 只说明在所选 bilinear AND axioms 下乘法唯一；论文自己也承认缺少 single-witness/additive 对照。
- **“Optimal reallocation”带有循环假设。** Proposition 4 先假设 supervision benefit 正比于 \(d_i\)，再证明沿 \(d-\bar d\) 最优；真正待验证的正是 \(d_i\) 是否代表 correction benefit。
- **低 KL 可能是 shared blindness。** Student 与 teacher 同时看不见或误读时 PCD 不纠正；这是一种保守选择，不是感知充分的证据。
- **高 KL 可能不是可纠正视觉差。** Teacher 与 student 的语言风格、tokenization 或 reasoning leakage 都可能扩大 perception-span KL。
- **Batch normalization 引入跨 prompt coupling。** Weight 在 optimization batch 的所有 perceptions 上归一化，一个 batch 中困难题比例会改变其他 prompt 的绝对权重。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可将轨迹拆为 scene observation/state extraction 与 planning/reasoning，并分别使用 distillation 和 outcome RL。
- 对同一空间 observation 采多条 planner continuation，可估计该 observation 在当前 planner 下的 downstream value。
- Outcome failure 与 teacher disagreement 的 conjunction 可作为保守的 correction gate，避免对所有失败感知一律强制模仿。
- Mean normalization 能在样本层面分离 selective allocation 与全局 distillation coefficient。

### 个人建议

- 不把 PSR 称为 perception correctness；同时记录 task difficulty、teacher confidence 和 shared-error probes。
- 使用 per-perception normalized loss 后再乘 \(w_i\)，才能严格保持 token-length independent supervision budget。
- 做完整 \(2\times2\) 消融：failure-only、KL-only、additive、multiplicative，并至少多 seed。
- 对 perception span 做结构约束或事实 verifier，禁止写答案和长 reasoning；分别测视觉事实 KL 与语言形式 KL。
- 在 all-wrong group 中增加 visual intervention/crop/cue teacher，检测 teacher-student shared blindness，而不是仅因低 KL 关闭 correction。
- 比较 \(a\times b\) 分配：1×8、2×4、4×2、8×1，在固定生成 token/FLOPs 下分析 perception diversity 与 PSR variance。

## 与其他论文的相同点和冲突

### 与 [VOLD](vold.md)

- 两者都联合 outcome RL 与 OPD，并选择性使用 teacher signal。
- VOLD 只在整条 rollout reward=0 时蒸馏，未区分 failure source；PCD 进一步用 perception-span KL 判断失败是否可能来自可纠正感知。
- VOLD 保护正确轨迹上的新 reasoning；PCD 不改 reasoning objective，只重分配 perception supervision。

### 与 [ViGOS](vigos.md)

- 两者都显式分离 perception/description 与 reasoning span，并使用不同 objective/teacher role。
- ViGOS 用 image-only teacher 监督 description、answer teacher 监督 reasoning；PCD 用强视觉 teacher 只蒸馏 perception、DAPO 优化 reasoning。
- ViGOS 处理 answer privilege 介入阶段；PCD 处理 outcome failure 应归因于 perception 还是 reasoning。

### 与 [VA-OPD](va-opd.md) 和 [V-Zero](v-zero.md)

- 三者都做 trajectory-level supervision allocation。VA-OPD/V-Zero 衡量 visual evidence dependence；PCD 将 downstream failure 与 teacher disagreement 相乘，目标是 teacher-correctable deficiency。
- VA-OPD/V-Zero 不需要 correctness reward，却可能强化 grounded-but-wrong trajectory；PCD 使用 verifier，但仍不能从 PSR 单独识别 failure source。
- PCD 可与 token-level visual weighting相乘，形成 trajectory deficiency × token saliency。

### 与 [Vision-OPD](vision-opd.md)

- Vision-OPD 用 privileged crop 改善 perception teacher target；PCD 假设已有更强 teacher，并决定哪些 perception 应获得更多 target correction。
- 两者互补：crop/full-image teacher disagreement可作为 PCD 的第二 witness，但需避免 crop 丢失全局关系。

### 与 [VGS](vgs.md) 和 [VCSD](vcsd.md)

- VGS/VCSD 从视觉条件 difference 构造 visual target；PCD 从 teacher–student perception KL 与 downstream reward判断是否应加强该 target。
- VGS 控制 gradient direction，VCSD 控制 vocabulary target，PCD 控制 perception trajectory weight，分别对应 how/what/when-to-correct。

### 与 [ViCuR](vicur.md)

- ViCuR 关注 teacher privilege 是否可由部署输入恢复；PCD 关注 teacher correction 应分配给哪个失败 perception。
- ViCuR cue/recovery 可降低 shared blindness；PCD gate 则可避免对 teacher-aligned perception 过度纠正。

### 综合定位

与 [VAD](vad.md) 相比，PCD 判断 perception 是否是可纠正 failure stage，VAD 决定被纠正位置具体采用哪些 intervention-attributable signed target shifts；二者对应 when 与 what。

与 [FP-OPD](fp-opd.md) 相比，PCD 决定是否纠正 perception，FP-OPD 决定 teacher correction中哪些方向与 student当前 response geometry兼容；可组合为 stage gate × capacity-aware target。

与 [SA-OPD](sa-opd.md) 相比，PCD用outcome×teacher gap定位可纠正perception failure；SA-OPD不用答案，只过滤对输入不敏感的高影响token。

与 [OPD-V](opd-v.md) 相比，PCD用outcome witness保护非perception failure；OPD-V正margin不判断student token是否正确，可用PCD gate补足。

十三篇论文中，PCD 新增“跨阶段 credit assignment”维度：不只问哪些 token/trajectory 依赖视觉，而是结合 downstream failure 与 teacher disagreement，判断 perception 是否是当前应被纠正的阶段。

## 证据

### E-001

- 结论：PCD 使用 \(a=2,b=4\) separated rollout，把多个 reasoning outcomes归并到共享 perception。
- 类型：论文结论
- 定位：§3.2；式 (4)–(7)；附录 B/C
- 必要引用：八条 trajectories 仍在同一 DAPO group。
- 备注：改变 rollout correlation，不只是 bookkeeping。

### E-002

- 结论：PSR 单独不能在简化模型下区分 perception sufficiency 与 reasoning difficulty。
- 类型：论文结论
- 定位：§3.3；Proposition 1；式 (8)
- 必要引用：\(V\approx\rho q\)。
- 备注：依赖二因素乘积模型和 insufficient-perception success \(\varepsilon\approx0\)。

### E-003

- 结论：PCD deficiency 是 failure 与 perception teacher disagreement 的乘积。
- 类型：论文结论
- 定位：§3.4；式 (9)–(12)
- 必要引用：\(d=(1-\mathrm{PSR})\widetilde{\mathrm{KL}}\)。
- 备注：是 soft AND surrogate，不是 calibrated posterior。

### E-004

- 结论：PCD 将 perception weights 归一化到样本均值 1。
- 类型：论文结论
- 定位：§3.5；式 (13)–(16)
- 必要引用：默认 base weight/boost 均为 1。
- 备注：不严格保证 token-weighted loss budget 不变。

### E-005

- 结论：PCD 只加权 perception distillation，reasoning 使用 DAPO。
- 类型：论文结论
- 定位：§3.6；式 (17)；算法 1
- 必要引用：\(\lambda_{\mathrm{aw}}=0.1,\lambda_{\mathrm{cot}}=1\)。
- 备注：公式不变，但 separated sampling 改变 trajectory distribution。

### E-006

- 结论：相对 OPD，PCD macro 在 8B→2B 和 32B→8B 分别 +2.78/+4.28。
- 类型：论文结论
- 定位：§4.2；表 1；附录表 6
- 必要引用：44.50→47.28；56.94→61.22。
- 备注：2B OOD mean -0.33，收益并不均匀。

### E-007

- 结论：去 PCD weight 与去 separated rollout 的 held-out average 分别下降 2.22/0.88。
- 类型：论文结论
- 定位：§4.4；表 2
- 必要引用：49.28→47.06/48.40。
- 备注：无 multi-seed；缺少 single-witness/additive gate。

### E-008

- 结论：论文没有实证隔离 multiplication 相对 PSR-only、KL-only、addition 的优势。
- 类型：论文明确局限
- 定位：§4.5 “Attribution”；附录 G
- 必要引用：无
- 备注：机制图只证明实现权重符合乘法形状。

### E-009

- 结论：“Label-free”不表示无答案 label/verifier。
- 类型：个人核对
- 定位：摘要；§3.1、§3.6；§4.1；附录 C
- 必要引用：Geo3K verifiable reward、DAPO reasoning objective。
- 备注：准确含义是无 perception labels 或 learned perception gate。

### E-010

- 结论：主表 2B PCD 数值等于消融表的 `Full w/ VPPO-Distill`，不等于 `Full (ours)`。
- 类型：个人核对
- 定位：表 1 与表 2
- 必要引用：主表 Geo3K 42.12、held-out shared entries对应 48.88；`Full` 为 Geo3K 43.12、held-out 49.28。
- 备注：默认 PCD 配置定义存在矛盾，`待核对`。

### E-011

- 结论：Perception-level mean normalization 不严格保持 token-level teacher budget。
- 类型：个人推断
- 定位：式 (13)–(14)
- 必要引用：\(\frac1N\sum_iw_i=1\)，但 loss 按可变数量 token 求和。
- 备注：perception 长度相同时近似成立。

### E-012

- 结论：PCD 2B 每 step 比 OPD 快，但 token scope 不等价。
- 类型：论文结论
- 定位：附录 C.4；表 7
- 必要引用：183.9→132.5 s/step；32B→8B 为 417→412。
- 备注：2B PCD 只蒸馏较短 perception span。

## 待验证问题

- [ ] 主表 PCD 为何与 `Full w/ VPPO-Distill` 完全相同，而非 `Full (ours)`？
- [ ] \(\widetilde{\mathrm{KL}}\) 的 threshold=0.3 具体归一化公式是什么？
- [ ] PSR-only、KL-only、addition、multiplication 在多 seed 下谁真正有效？
- [ ] Perception span 长度变化后，实际 token-weighted teacher loss scale 是否保持？
- [ ] Perception span 是否泄漏 reasoning 或 final answer？
- [ ] PCD weight 与人工判定 perception error 的 precision/recall 如何？
- [ ] Teacher shared blindness 和 confidently wrong teacher 各造成多少 false negative/positive？
- [ ] 在固定 token/FLOPs 下，\(a\times b\) 的最佳 perception-diversity/PSR-variance trade-off 是什么？
- [ ] 跨非数学、视频、3D 与空间规划数据是否仍有效？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v2 全文，正文 §1–5、式 (1)–(17)、算法 1、表 1–2、图 1–3；附录 A–G、式 (18)、算法 2、表 3–7、图 4。
- 新增认识：Multimodal OPD 不仅需要视觉 token/trajectory selection，还需判断 failure 应归因到 perception 还是 reasoning；reward 与 teacher disagreement提供不同 witness。
- 修正内容：未把 PCD 称为 answer-label-free，未把 PSR 当 perception correctness，未把 bilinear uniqueness 当作 multiplication 的经验优越性证明。
- 下一步：核对主配置矛盾，完成 single-witness/additive 消融，并严格控制 token-weighted supervision budget。
