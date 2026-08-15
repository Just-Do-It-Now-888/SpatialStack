# ViCuR

## 基本信息

- 论文标题：ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation
- 作者：Kanghui Tian, Siyuan Liu, Ziang Yan, Sheng Xia, Shuai Dong, Yi Wang
- 年份与会议：2026；预印本，under review
- arXiv/DOI：arXiv:2606.05718；DOI: 10.48550/arXiv.2606.05718
- 论文链接：https://arxiv.org/abs/2606.05718
- 代码：https://github.com/tiankanghui/ViCuR
- 本地文件：无
- 阅读状态：已完成（cue 构造与模块条件化细节待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

ViCuR 用描述图像/视频中问题相关证据的 visual cue 替代答案或 rationale privilege，让教师依赖理论上可从推理输入恢复的信息，并在学生中加入仅 prefill 激活的 sink-token→visual-token 专用 cross-attention；相对 answer-based OPSD 或 vanilla OPD，它提高总体平均表现，但收益主要来自视觉 cue 本身，cue recovery 模块的跨域增益并不稳定。

## 核心问题

Privileged-teacher OPD 允许教师看到训练时额外信息，但常见的答案、rationale 或 verifier feedback 在推理时不可得。学生被要求模仿一个依赖自己无法访问变量的 teacher policy，可能学习答案提示格式和 shortcut，而不是视觉 grounding。

论文研究：

1. 特权信息是否应从 answer-side privilege 改为来自原始视觉输入的、理论上可恢复的 visual cue；
2. 学生不显式接收 cue text 时，如何从视觉 token 内部聚合对应证据；
3. privilege 设计的收益能否同时适用于同 backbone OPSD 和更强外部教师 OPD；
4. visual cue 是否能减少训练时 hint pattern 泄漏，并改善跨域迁移。

## 方法

### 整体流程

1. 标准学生输入为 \(X=(I,Q)\)，包含图像/视频和问题。
2. 训练时为教师额外提供 visual cue \(S\)：只描述与问题相关的可见证据，不直接给答案、定理或求解步骤。
3. 学生在标准输入 \(X\) 上生成 on-policy trajectory \(\hat y\)；教师在 \(X,S\) 上重评分相同 student prefix。
4. 学生每隔五个 transformer layer 插入 cue recovery branch。序列开头的 sink token 作为单 query，对当前层全部 visual tokens 做专用 cross-attention，并把聚合结果残差写回 sink state。
5. 后续 question/answer token 通过普通 causal attention 读取更新后的 sink state；模块不生成 cue text，也没有 cue reconstruction auxiliary loss。
6. 用 sampled-token OPD 的 teacher−student log-prob advantage 和 PPO-style clipping 联合训练学生与 recovery module。
7. 推理时输入接口仍为普通 \(I,Q\)；recovery branch 只在 prefill 运行，不在 autoregressive decoding step 重复执行。

### 关键模块

**Recoverable privilege。** 答案特权 \(R\) 通常满足
\(I(Y_t;R\mid X,Y_{<t})>0\)，意味着 teacher next-token behavior 依赖学生推理时不可访问的信息。论文理想化 visual cue 为确定映射 \(S=f(X)\)，于是：

\[
H(S\mid X)=0,\qquad
I(Y_t;S\mid X,Y_{<t})=0.
\]

这只说明 cue 的信息来源没有超出推理输入，并不保证学生能实际提取，也不保证现实 cue generator 真正是确定且不使用外部知识的 \(f(X)\)。

**Sink-based cue recovery。** 在选定层 \(\ell\)：

\[
q^{(\ell)}=h_{\mathrm{sink}}^{(\ell)}W_Q^{(\ell)},\quad
K^{(\ell)}=V^{(\ell)}W_K^{(\ell)},\quad
U^{(\ell)}=V^{(\ell)}W_V^{(\ell)},
\]

\[
\tilde h_{\mathrm{sink}}^{(\ell)}
=h_{\mathrm{sink}}^{(\ell)}
+\mathrm{Attn}(q^{(\ell)},K^{(\ell)},U^{(\ell)})W_O^{(\ell)}.
\]

这些 \(W_Q,W_K,W_V,W_O\) 是 recovery-specific 参数，不复用原 self-attention。Qwen3-VL 使用 system prompt 前的第一个 `<|im_start|>` 作为 sink token。

**隐式而非显式 cue 对齐。** 模块只通过最终 teacher-shaped OPD loss 获得梯度；没有 cue text 生成或 hidden-state matching。论文把内部表示写为 \(\hat S=g(V,Q;\theta_{\mathrm{sink}})\)。

### 损失函数与训练目标

对 student-sampled token \(\hat y_n\)，定义：

\[
A_n
=\log p_T(\hat y_n\mid X,S,\hat y_{<n})
-\log p_\theta(\hat y_n\mid X,\hat S,\hat y_{<n}).
\]

\(A_n\) 被 stop-gradient。简化后的 surrogate（式 7）为：

\[
\mathcal L_d(\theta)
=-\mathbb E\left[
\frac1{|\hat y|}\sum_n
\mathrm{sg}[A_n]\,
\log p_\theta(\hat y_n\mid X,\hat S,\hat y_{<n})
\right].
\]

实际实现使用 old log-probability、importance ratio 和 clipping 的 PPO-style objective。每个 prompt 只采一条 student rollout。

论文附录另分析 full-vocabulary forward KL：

\[
D_{\mathrm{KL}}\left(
p_T(\cdot\mid X,S,Y_{<t})
\|p_\theta(\cdot\mid X,\hat S,Y_{<t})
\right),
\]

但明确说明实际训练使用 sampled-token objective，不能把附录 KL 当作实现。

### 训练数据与训练流程

- 主训练：Vision R1 train；5 epochs，275 steps，batch size 128，LR \(10^{-6}\)，max prompt 2048，max response 4096。
- 消融：Geometry3K train；100 epochs，batch size 128，LR \(10^{-6}\)，max prompt 1024，max response 2048。
- Student：Qwen3-VL-2B-Instruct、Qwen3-VL-8B-Instruct。
- OPSD：teacher 与 student 同 backbone，但 teacher 额外接收 privilege。
- OPD：2B student 使用 Qwen3-VL-8B teacher；8B student 使用 Qwen3-VL-32B teacher。
- GRPO baseline：每 prompt 5 rollouts；distillation 方法每 prompt 1 rollout。
- Recovery branch：每五层插入一次；2B 增加 100.7M 参数（4.52%），8B 增加 536.9M（5.77%）。
- 框架与硬件：VeRL；单节点 8×H200，distillation 分配 4 student + 4 teacher GPU。
- 评测采样：temperature 0.7，top-p 0.8，top-k 20，presence penalty 1.5，max new tokens 4096。
- 论文未说明 visual cue 的具体生成模型、prompt、采样配置、过滤规则及是否访问答案/标注；optimizer、weight decay、训练 rollout temperature 与 PPO clipping 超参数也未报告，均 `待核对`。

## 实验设置

- 模型：Qwen3-VL-2B/8B students；OPD 教师为 8B/32B
- 数据集：训练 Vision R1；消融 Geometry3K
- 评测：Vision R1-Test；DynaMath、MathVista、WeMath、MathVerse；MMMU-Val、Video-MME
- Baseline：Base、GRPO、answer-based OPSD、vanilla stronger-teacher OPD
- 指标：各 benchmark 官方分数及七项简单平均；消融使用最后 100 training steps 的平均 accuracy
- 泛化划分：Vision R1-Test 为 in-domain；四个数学集为 near-domain；MMMU/Video-MME 为 out-of-domain

## 实验结论

1. **相对 answer-based OPSD，总体平均提高。** 2B 为 44.88→46.07（+1.19），8B 为 58.15→59.39（+1.24）。
2. **“七个 benchmark 一致提升”只对 8B OPSD 成立。** 2B OPSD+ViCuR 在 DynaMath（-0.77）、MMMU（-0.34）、Video-MME（-0.1）低于 OPSD；因此摘要应理解为 overall average consistently improves，而非每个 benchmark 都提高。
3. **Answer-based OPSD 本身会退化。** OPSD 低于 base：2B 为 44.88 vs 45.42，8B 为 58.15 vs 60.76。ViCuR 使 2B 超过 base（46.07），但 8B 仍低于 base（59.39 vs 60.76），只能说缩小退化。
4. **相对 stronger-teacher OPD，平均提高。** 2B 为 46.93→47.57（+0.64），8B 为 63.88→64.96（+1.08）。
5. **Stronger-teacher OPD 也不是逐项提高。** 2B 在 DynaMath -0.49、MathVista -0.8；8B 在 MathVista -1.4。作者将 MathVista 下降归因于 arithmetic/word-problem 子集更依赖数值推理而非 grounding。
6. **Visual cue 是主要收益来源。** Geometry3K OPSD 中，仅将 answer privilege 换成 cue，2B/8B 分别 +2.80/+4.36；只加 recovery module、仍用 answer teacher 时仅 +1.17/+0.09；完整方法为 +4.80/+4.65。
7. **Recovery module 跨域收益不稳定。** 附录表 5 中 visual-cue-only 经常高于 full ViCuR，例如 2B OPD 的 MathVista geometry reasoning 75.73 vs 74.06；作者认为较大模块可能过拟合源域 cue pattern。
8. **教师规模与 cue 设计具有互补性，但受 student capacity 限制。** Geometry3K 四种配置均提高；最大增益是 32B→8B 的 +6.88。对 2B student，8B teacher 的 +3.35 反而低于 self-distillation 的 +4.80。
9. **Visual cue 减少显式 hint leakage。** Answer-based OPSD 中 student 输出“Hint”的样本/次数随 epoch 急升；ViCuR 显著减少，但并未归零。该指标只测词面复制，不能覆盖隐式 shortcut。
10. **训练成本小，prefill 有明确开销。** 主训练总时间与 baseline 接近；推理时 small-image prefill 增加约 19–25%，large-image 增加约 4–7%。模块不在 autoregressive decoding 重复执行。
11. **Attention visualization 与案例支持机制，但不是因果证据。** Heatmap 覆盖几何标签、弦和图表多区域；论文自己也承认不能据此证明每层精确选择正确证据。

## 局限性

### 论文明确承认

- 方法依赖训练 visual cue 质量；cue generator 若遗漏证据、引入非视觉信息或描述含糊，会削弱 teacher supervision。
- Recovery module 增加参数与优化复杂度，跨模型规模收益不单调；更大 student 可能需要参数高效设计、分阶段训练或额外正则化。

### 由实验设计可直接确认

- 全部实验限于 Qwen3-VL 家族，跨架构泛化未知。
- 主训练偏重 Vision R1 数学 reasoning；虽包含 MMMU 和 Video-MME out-of-domain 评测，但开放式 grounding、长视频、3D 与交互任务未验证。
- 没有多训练随机种子、标准差或显著性检验；0.1–1 point 差异不宜过度解释。
- “不改变 inference interface”只表示输入输出协议不变；模型架构与 checkpoint 已改变，且 prefill latency 和参数量明确增加。
- 8B recovery module 增加 536.9M 参数，称为“lightweight”需要结合 5.77% 参数和部署约束理解。
- 总生成时间受输出长度巨大变化影响，不能用其比较模块计算开销；更可靠的是 prefill 与 per-token decoding 指标。
- 代码仓库当前公开页面未提供可核对实现内容，关键细节仍依赖后续发布。

### 个人分析

- **实际 cue 构造缺失是最大复现问题。** 论文证明只在 \(S=f(X)\) 的理想确定映射下成立，但未报告 cue generator。若生成器使用答案、rationale、标注或外部知识，所谓 recoverable privilege 可能仍含 answer-side leakage。
- **信息论命题本身近乎定义性结论。** 一旦假设 \(S\) 完全由 \(X\) 确定，\(I(Y;S\mid X)=0\) 自动成立；它不能证明现实 cue 更容易被有限容量 student 恢复，也不能量化实际 \(H(S\mid X)\)。
- **Sink query 可能并非逐样本 question-conditioned。** Sink token 位于 system prompt 前，按 causal 顺序看不到后续问题；专用 cross-attention 的 query 来自该 sink state，visual token 通常也位于问题之前。训练 loss 可让参数学习跨数据集的通用视觉选择，但这不等于推理时 query 随当前 \(Q\) 动态变化。论文把表示写为 \(g(V,Q)\) 并称 question influence 来自 optimization，需与真正的 per-sample conditioning 区分。
- **“\(A_n>0\) reinforces current evidence representation”解释过强。** Policy-gradient 更新提高 sampled token log-prob，但对内部向量的具体方向取决于 Jacobian；不能仅凭 advantage 正负断言保留了哪类视觉证据。
- **Teacher cue 与 recovery module 的贡献未在主七 benchmark 上完整正交消融。** 核心组件表主要使用 Geometry3K；跨域附录显示 recovery 可能负贡献。
- **论文内部有一处表述不一致。** Limitations 称 8B student 在 Geometry3K 上 full ViCuR 略低于 visual-cue-only，但表 2 为 57.05 > 56.76；可能实际指附录跨域结果，需作者澄清。
- Attention map 是机制一致性证据，不是 cue recovery 的因果验证；应做 attention intervention、sink ablation 或 cue-target probing。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 的 privileged teacher 应优先接收可从部署输入恢复的空间证据，例如对象关系、局部拓扑、可见 landmark 或轨迹片段，而不是直接答案/完整规划。
- 可用训练时 cue 引导教师，但保持 student 输入为部署标准接口，从而把“提供答案”改为“突出证据”。
- 在 prefill 阶段增加单 query 对空间 token 的聚合模块，可以不增加每个 autoregressive decoding step 的模块计算。
- 应记录 privilege leakage：例如学生是否复述 teacher-only 标签、`Hint`、真值坐标或规划注释。

### 个人建议

- Cue 构造必须做 provenance audit：记录生成模型、prompt、是否可见答案/标注、采样 seed、过滤规则，并自动检查 cue 中答案和推理定理泄漏。
- 让 sink query 真正条件于当前问题：把 query token 放在 question 后，或先编码 question summary 再 cross-attend visual/spatial tokens；与论文的固定前置 sink 做对照。
- 对 cue recoverability 进行直接测量：训练 probe 从 student 输入/hidden state 预测 cue facts，或做视觉证据遮挡后的性能下降，而不只看最终答案。
- 在 SpatialStack 中比较 answer privilege、visual/spatial cue、真值局部地图、对象图和无 privilege，并控制 cue 长度与 teacher confidence。
- Recovery module 应与简单参数匹配 baseline 比较，例如同参数 LoRA/MLP、额外 self-attention 或 question-conditioned pooling，排除纯容量收益。

## 与其他论文的相同点和冲突

### 与 [Vision-OPD](vision-opd.md)

- 两者都把 visual-side privileged information 仅放在训练教师侧，并在 student rollout prefix 上蒸馏；部署时学生仍接受标准视觉输入。
- Vision-OPD 的 privilege 是 evidence-centered crop，并给 student 带框全图与显式空间约束；ViCuR 的 privilege 是 visual cue text，student 不接收框或 cue text，但增加 recovery architecture。
- Vision-OPD 的 crop 直接提升教师感知，信息源明确但依赖区域标注；ViCuR cue 更灵活，可描述多区域或视频事件，但生成流程与 recoverability 更难验证。
- ViCuR 明确指出 answer/rationale privilege 的 train-test gap，并把 Vision-OPD 归为 answer-agnostic evidence-centered crop 路线；两者共同支持“privilege 设计与 teacher strength 同等重要”。

### 与 [VA-OPD](va-opd.md) 和 [VGS](vgs.md)

- 三者都针对 OPD 中视觉 grounding 不足，但层级不同：ViCuR 改变 teacher privilege 和 student architecture；VA-OPD 选择高视觉依赖 rollout/token；VGS 分解语言/视觉目标并 steering 梯度。
- VA-OPD/VGS 从教师在不同视觉条件下的 distribution 差自动得到连续视觉依赖信号；ViCuR 依赖离线 cue text，监督可解释但生成成本和泄漏风险更高。
- ViCuR recovery 可与 VA weighting 或 VGS 组合，但应先证明 sink 表示真正按当前问题选择证据。

### 与 [VOLD](vold.md)

- VOLD 用 outcome verifier 避免教师干扰正确路径，并通过同教师 SFT 解决 state mismatch；ViCuR 不用 reward，重点解决 privilege mismatch。
- VOLD 的 teacher privilege 是更强文本推理能力，视觉塔冻结；ViCuR 直接改造学生视觉证据聚合，更接近感知/grounding 侧迁移。

### 与 [ViGOS](vigos.md)

- 两者都针对 answer/rationale privilege 引发的视觉 shortcut。ViCuR 用 visual cue 替换答案特权；ViGOS 保留 reference solution，但禁止它直接监督 description token。
- ViCuR 用 latent sink recovery，增加 prefill 参数但不生成 cue；ViGOS 用显式 description 作为 grounding interface，不加模块但增加推理 token。
- ViCuR 需审计 cue provenance；ViGOS 需审计 description factuality 与 invalid fallback 是否绕过分阶段 routing。

### 与 [V-Zero](v-zero.md)

- 两者都避免直接使用 textual answer label 作为主要 teacher privilege。ViCuR 使用视觉 cue text；V-Zero 使用 target/random region crops。
- ViCuR 训练 student recovery architecture；V-Zero 不改 inference architecture，而用 crop contrast 给 sibling rollouts 加权。二者都必须审计 privilege 的构造 provenance。

### 与 [VCSD](vcsd.md)

- 两者都避免 answer/rationale privilege。ViCuR 仍需离线 visual cue 与 recovery module；VCSD 只需原始图文，通过 content-erased control 自动构造 target asymmetry。
- ViCuR 风险是 cue provenance 与 recoverability；VCSD 风险是 control validity、EMA target drift 与视觉错误自强化。

### 与 [PCD](pcd.md)

- ViCuR 改善 student 恢复视觉 evidence 的通道；PCD 在已有 teacher 下识别哪些失败 perception 值得加强纠正。
- ViCuR 可减少 shared blindness，PCD 的 failure×disagreement gate可避免对 teacher-aligned perception 过度模仿。

### 与 [VAD](vad.md)

- ViCuR 改善 evidence recovery architecture；VAD 过滤并重构 teacher correction 中与特定视觉 intervention 对齐的部分。
- 两者都依赖训练时视觉 privilege；ViCuR 需审计 cue provenance，VAD 需审计 crop/degradation proxy 与 projection support。

### 与 [FP-OPD](fp-opd.md)

- ViCuR 通过 recovery module扩大 student可用视觉证据；FP-OPD 接受当前 probe-estimated response geometry并过滤 teacher gap，分别代表扩容与目标适配。

### 与 [SA-OPD](sa-opd.md)

- ViCuR增强从推理输入恢复cue的通道；SA-OPD过滤对输入移除不敏感的监督。二者分别处理evidence recovery与teacher-signal reliability。

### 与 [OPD-V](opd-v.md)

- ViCuR通过architecture恢复cue；OPD-V用zoom/mask margin选择视觉敏感token。前者扩大能力，后者分配监督。

### 综合定位

十三篇论文给出 OPD 的十三类设计约束：VOLD 的状态对齐与正确性选择、Vision-OPD 的特权视图与时间稳定性、VA-OPD 的 token 稀疏性、VGS 的模态梯度冲突、ViCuR 的 privilege recoverability 与学生证据恢复通道、ViGOS 的 privilege 介入阶段与 teacher routing、V-Zero 的 sibling-relative evidence gating、VCSD 的 input-conditioned target asymmetry、PCD 的 cross-stage credit assignment、VAD 的 correction-source attribution、FP-OPD 的 student-local compatibility、SA-OPD 的 input-grounded reliability，以及 OPD-V 的 modality-balance trust region。

## 证据

### E-001

- 结论：ViCuR 用 visual cue 替代 answer/rationale privilege，并将其理想化为 \(S=f(X)\)。
- 类型：论文结论
- 定位：§3.1；式 (1)–(2)；附录 F.1、式 (9)–(22)
- 必要引用：确定映射假设下 \(I(Y_t;S\mid X,Y_{<t})=0\)。
- 备注：作者明确承认现实 cue 可能随机，gap 仅减少而非消除。

### E-002

- 结论：Cue recovery 使用前置 sink token 对 visual tokens 的专用 cross-attention，每五层插入，结果残差写回 sink state。
- 类型：论文结论
- 定位：§3.2–3.4；式 (3)–(5)；图 2；附录 A
- 必要引用：无
- 备注：当前问题是否能逐样本影响 sink query `待核对`。

### E-003

- 结论：实际训练使用 sampled-token teacher−student log-prob advantage 和 PPO-style clipped objective。
- 类型：论文结论
- 定位：§3.3；式 (6)–(8)；附录 F.2
- 必要引用：每 prompt 1 rollout，advantage stop-gradient。
- 备注：附录 F.3 的 full-vocabulary KL 仅为分析 counterpart。

### E-004

- 结论：相对 answer-based OPSD，ViCuR 的总体平均提高 1.19/1.24 points。
- 类型：论文结论
- 定位：§4.1；表 1
- 必要引用：2B 44.88→46.07；8B 58.15→59.39。
- 备注：2B 有三个单项下降；8B ViCuR 仍低于 base。

### E-005

- 结论：相对 stronger-teacher OPD，ViCuR 的总体平均提高 0.64/1.08 points。
- 类型：论文结论
- 定位：§4.1；表 1
- 必要引用：2B 46.93→47.57；8B 63.88→64.96。
- 备注：MathVista 在两个规模均下降。

### E-006

- 结论：Visual cue 是主要增益来源，recovery module 单独贡献较小。
- 类型：论文结论
- 定位：§4.2；表 2
- 必要引用：cue-only +2.80/+4.36；recovery-only +1.17/+0.09；full +4.80/+4.65。
- 备注：该消融在 Geometry3K OPSD setting。

### E-007

- 结论：Recovery module 增加 4.52%/5.77% 参数，只在 prefill 激活。
- 类型：论文结论
- 定位：§4.2；附录 B.4；表 6–7
- 必要引用：small-image prefill +19–25%，large-image +4–7%。
- 备注：输入接口不变不等于架构和延迟不变。

### E-008

- 结论：Visual cue privilege 减少 answer-based OPSD 的显式 “Hint” 泄漏。
- 类型：论文结论
- 定位：§4.2；图 4；附录 D、表 8、图 9
- 必要引用：无
- 备注：只测词面 “Hint”，不是完整 shortcut 指标。

### E-009

- 结论：论文没有提供实际 visual cue 生成 pipeline。
- 类型：个人核对
- 定位：§3.1；§4 “Data”；附录 A、C
- 必要引用：只称 cue 从 Vision R1 输入构造，并展示示例。
- 备注：generator、prompt、答案可见性、过滤与随机性均 `待核对`。

### E-010

- 结论：前置 sink cross-attention 的 per-sample question conditioning 不明确。
- 类型：个人推断
- 定位：§3.2；式 (3)–(5)；附录 A
- 必要引用：sink 是 system prompt 前第一个 token；query 来自 sink state，作者称 \(Q\) 仅通过 optimization 隐式影响。
- 备注：训练梯度依赖问题不等于推理 query 读取当前问题。

### E-011

- 结论：Limitations 对 8B Geometry3K full/cue-only 排序的描述与表 2 不一致。
- 类型：个人核对
- 定位：§4.2 表 2；§6
- 必要引用：表 2 为 full 57.05 > cue-only 56.76，限制部分称 full 略差。
- 备注：可能实际指附录表 5 的跨域结果，`待核对`。

## 待验证问题

- [ ] Visual cue 由什么模型、prompt 和数据字段生成？是否可见答案、rationale 或标注？
- [ ] Sink query 在推理时是否真正条件于当前问题？因果 attention 路径是什么？
- [ ] 用 question-conditioned sink/late query 替代前置固定 sink 是否更有效？
- [ ] Cue 的实际 \(H(S\mid X)\) 与 answer leakage 如何测量？
- [ ] Recovery module 相对等参数 LoRA/MLP/attention baseline 的净收益是多少？
- [ ] 用 intervention 而非 attention heatmap 验证 sink 是否因果聚合正确证据。
- [ ] 多随机种子下 +0.1–1 point 的结果是否显著？
- [ ] §6 与表 2 的 8B Geometry3K 排序矛盾如何解释？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，正文 §1–6、式 (1)–(8)、表 1–2、图 1–4；附录 A–F、式 (9)–(35)、表 3–8、图 5–11。
- 新增认识：OPD 不仅要控制 teacher strength，还要保证 privilege 的信息来源可由部署输入恢复，并为 student 提供适合恢复该证据的内部通道。
- 修正内容：未把“recoverable”视为现实已证明性质，未把“consistently improves”解释为所有 benchmark 均提升，也未把 attention visualization 当作因果 grounding 证据。
- 下一步：核对 cue 生成代码与 sink 因果路径，并设计 answer privilege、visual cue、crop、自动 visual dependency signal 的统一对照。
