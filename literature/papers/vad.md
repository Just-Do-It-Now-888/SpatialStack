# VAD

## 基本信息

- 论文标题：VAD: Attributing Visual Evidence for Target Reconstruction in Multimodal On-Policy Distillation
- 作者：Kangning Zhang, Yixing Li, Shuai Shao, Qingyao Li, Zhengxi Lu, Zhiyuan Yao, Shijian Wang, Jianghao Lin, Wenxiang Jiao, Yuan Lu, Weiwen Liu, Weinan Zhang, Yong Yu
- 年份与会议：2026；arXiv 预印本（v1）
- arXiv/DOI：arXiv:2607.28590；DOI: 10.48550/arXiv.2607.28590
- 论文链接：https://arxiv.org/abs/2607.28590
- 代码：https://github.com/DeepExperience/VAD_Multimodal_OPD
- 模型：https://huggingface.co/zhangkangning/VAD_for_Qwen3.5-4b；https://huggingface.co/zhangkangning/VAD_for_Qwen3.5-9b
- 本地文件：无
- 阅读状态：已完成（坐标几何、语义分析与训练配置待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

VAD 在每个 student prefix 上比较固定 teacher 的清晰证据 crop 与同区域退化 crop，以 centered log-prob difference 构造带正负号的视觉 intervention direction，再把完整 teacher correction 投影到该方向、围绕当前 student 重构 support/refutation target，并以弱 direct-teacher regularizer 保持语言稳定；它在严格匹配的 4B/9B 对照中领先，但归因只是一维、support-dependent 的 proxy projection，不是可识别的视觉因果分解。

## 核心问题

Privileged-view OPD 直接模仿 crop teacher 的完整 next-token distribution，其中同时包含：

- 视觉证据带来的 correction；
- 语言 prior、格式和表达偏好；
- teacher/student 模型差异；
- crop/degradation artifact。

VA-OPD、V-Zero 通过 visual contrast 决定哪些 token/trajectory 多学，但局部 target 仍是完整 evidence-present teacher distribution。VAD 进一步研究：

1. 如何估计 teacher correction 中与受控视觉 intervention 对齐的分量；
2. 如何同时表达视觉证据对候选 token 的支持与反驳；
3. 如何围绕 student current distribution 重构 target，而不是复制 source-mixed teacher；
4. 如何在视觉 target 与语言/格式稳定性间平衡。

## 方法

### On-policy counterfactual views

Student 用 full image \(x^0\) 生成 rollout。固定初始 checkpoint teacher 在相同 student prefix 上评估：

\[
p_S^0=\pi_\theta(\cdot\mid x^0,y_{<t}),\quad
p_T^+=\pi_{\bar\theta}(\cdot\mid x^+,y_{<t}),\quad
p_T^-=\pi_{\bar\theta}(\cdot\mid x^-,y_{<t}).
\]

\(x^+\) 是相关区域 2× crop，\(x^-\) 是同一区域先 0.1× bilinear downsample、再 nearest-neighbor upsample 的 evidence-degraded crop。Teacher 参数和文本 prefix 固定。

实现只在 student top-\(K\) candidate set \(V_t\) 上运算，主训练 \(K=100\)，并加入 tail bucket。Centered log probability：

\[
\phi_t(p)
=\log(p[V_t]+\epsilon)
-\operatorname{mean}\log(p[V_t]+\epsilon).
\]

Centering 去掉共同 offset，保留该离散 support 上的 pairwise log-odds。

### Correction attribution

完整 privileged correction 与 intervention response：

\[
r_t=\phi_t(p_T^+)-\phi_t(p_S^0),\qquad
u_t=\phi_t(p_T^+)-\phi_t(p_T^-).
\]

\(u_t(i)>0\) 表示清晰证据相对退化证据提高 candidate \(i\) 的相对 odds，\(u_t(i)<0\) 表示 refutation。

One-sided regularized projection：

\[
\beta_t
=\frac{[\langle r_t,u_t\rangle]_+}
{\|u_t\|_2^2+\zeta},
\]

\[
r_t^{\mathrm{vis}}=\beta_tu_t,\qquad
r_t^{\mathrm{res}}=r_t-r_t^{\mathrm{vis}}.
\]

当完整 correction 与 intervention direction 内积不正时，视觉 correction 置零。Residual 只表示 proxy-unexplained，不等于纯语言分量。

One-sided student-anchored target：

\[
q_{T,t}^{\mathrm{one}}
=\operatorname{softmax}
\left(
\phi_t(p_S^0)+
\operatorname{clip}(r_t^{\mathrm{vis}},-c,c)
\right).
\]

### Budgeted support/refutation

将 intervention direction 分成：

\[
u_t^+=[u_t]_+,\qquad
u_t^-=[u_t]_-.
\]

Agreement scores 与 branch shares：

\[
s_t^\pm=[\langle r_t,u_t^\pm\rangle]_+,
\quad
Z_t=s_t^++s_t^-+\epsilon,
\]

\[
\omega_t^+
=\min(s_t^+/Z_t,\tau_+),\qquad
\omega_t^-=s_t^-/Z_t.
\]

以 one-sided projection norm
\(B_t=\|r_t^{\mathrm{vis}}\|_2\)
作为总视觉 correction budget：

\[
r_t^{\mathrm{VAD}}
=B_t\left(
\omega_t^+\frac{u_t^+}{\|u_t^+\|_2+\epsilon}
+\omega_t^-\frac{u_t^-}{\|u_t^-\|_2+\epsilon}
\right).
\]

\[
q_{T,t}^{\mathrm{VAD}}
=\operatorname{softmax}
\left(
\phi_t(p_S^0)
+\operatorname{clip}(r_t^{\mathrm{VAD}},-c,c)
\right).
\]

只 cap support branch；被截去的质量不重新分配。4B/9B 的 \(\tau_+\) 分别为 0.8/0.7。

需要注意：负 coordinate 只降低相对 log-odds，不保证该 token 的绝对 probability 必然下降；softmax normalization 取决于所有 coordinates。论文附录明确给出这一边界。

### Training objective

Primary target 使用 JSD：

\[
\mathcal L_{\mathrm{vis}}
=\frac1{|\mathcal T|}\sum_t
D_{\mathrm{JS}}
\left(
\operatorname{sg}[q_{T,t}^{\mathrm{VAD}}],
p_S^0
\right).
\]

Visual-only 训练出现 response 变长、重复、延迟回答、格式和 stopping 不稳定，因此加入 weak direct-teacher regularizer：

\[
\rho_t
=\frac{\|r_t^{\mathrm{VAD}}\|_2}
{\|r_t\|_2+\epsilon},
\qquad
a_t=\operatorname{sg}[\operatorname{clip}(1-\rho_t,0,1)],
\]

\[
\mathcal L_{\mathrm{reg}}
=\frac1{|\mathcal T|}\sum_t
a_tD_{\mathrm{JS}}
\left(
\operatorname{sg}[p_T^+],p_S^0
\right),
\]

\[
\mathcal L
=\mathcal L_{\mathrm{vis}}
+\lambda\mathcal L_{\mathrm{reg}},
\qquad \lambda=0.1.
\]

当 visual-attributed fraction 较低时，source-mixed privileged teacher regularizer反而更强；这是稳定性折中，不是完全排除 unattributed correction。

Student copy 用于 target reconstruction 时 stop-gradient。推理只保留 full-image student。

## 训练数据与训练流程

- 模型：Qwen3.5-4B、Qwen3.5-9B；teacher 是各自初始模型的 frozen copy。
- 数据：Vision-OPD 发布的 6,241 条 synthetic VQA examples。
- 输入：带相关区域标记的 full image、2× evidence crop、同 crop 的 0.1× downsample→upsample degraded view。
- Candidate support：student top-100 + tail bucket。
- Batch size 96；每 prompt 8 rollouts；LR \(2\times10^{-6}\)。
- Projection stabilizer \(\zeta=10^{-3}\)；coordinate clip \(c=20\)。
- \(\lambda=0.1\)；\(\tau_+=0.8\)（4B）、0.7（9B）。
- Controlled methods 使用相同 Qwen3.5 initialization、6,241 examples、rollout budget 和 update budget；baseline hyperparameters 另行调优。
- 论文未集中报告 optimizer、scheduler、warmup、rollout temperature/top-p、max response length、总训练 update/epoch、checkpoint selection 和随机种子，需结合代码核对。
- 训练效率表只统计前 65 steps（1 epoch）的 core timer。

## 实验设置

- 主 benchmark：VStar、ZoomBench、HRBench-4K/8K、MME-RealWorld EN/CN
- Held-out：MMVP、CV-Bench、MMStar、POPE
- 主指标：六项或四项无权平均
- Evaluation：官方 Vision-OPD pipeline；主六项使用 GPT-OSS-120B judge
- Controlled baselines：Base、GRPO、VA-OPD、V-Zero、Vision-OPD、Decomposed OPD
- Cross-family context：agentic、闭源和超大开源模型；只能作能力背景

## 实验结论

1. **VAD 在 matched Avg\(_6\) 上领先。** 4B 为 78.32，最佳其他方法 Vision-OPD 75.92，领先 2.40；9B 为 79.93，最佳其他方法 V-Zero 77.13，领先 2.80。
2. **相对 direct Vision-OPD 提升 2.40/3.05。** 相对 Decomposed OPD 提升 2.95/2.88，相对 VA-OPD 提升 3.40/3.26。
3. **逐项优势较广。** 4B 在六项 controlled rows 全部最高；9B 五项最高，仅 ZoomBench 比 VA-OPD 低 0.35。
4. **Cross-family “小模型胜大模型”不是受控结论。** 4B Avg\(_6\) 超过表中 Gemini 3.1 Pro 和 Qwen3.5-397B，但架构、prompt、数据和评测输出均不同。
5. **Branch-aware reconstruction 是主要组件。** 4B Direct 75.92、scalar-shrunk 76.19、one-sided no-reg 77.06、VAD no-reg 78.06、Full VAD 78.32。
6. **Support/refutation branch separation 的净增益约 1.0。** 在都无 regularizer 时，one-sided 77.06→VAD 78.06。
7. **Weak teacher regularizer 增益较小但改善稳定性。** VAD no-reg→full 为 +0.26；one-sided no-reg→reg 为 +0.46。Visual-only drift 的长度/重复数据未量化。
8. **Offline answer-token effect 只有方向性证据。** Correct support 7.52→7.89、wrong suppression 6.29→6.61，但 confidence intervals 重叠，不能声称 pairwise 显著。
9. **Held-out aggregate 基本保持 base，而非明确提升。** 4B +0.24、9B +0.23；无多 seed/significance。单项 MMStar 仍低于 base：4B -2.74、9B -1.67。
10. **JSD aggregate 最好。** 4B JSD/F-KL/R-KL 为 78.32/77.62/77.85；9B 为 79.93/79.21/79.56。
11. **训练成本与 Vision-/VA-OPD 接近。** 4B VAD 67.9 GPU-hours vs Vision 65.5、VA 65.9；9B 为 96.3 vs 86.7、92.7。V-Zero 更慢。
12. **Semantic enrichment 支持 proxy interpretation。** Visual/object/decision category 在 \(r^{\mathrm{vis}}\) top-5 composition 为 42.0%，相对完整 correction 26.7%、residual 17.7%；但 taxonomy 与标注流程未充分说明。

## 局限性

### 论文明确承认

- 单一 view pair 只提供一个 contrastive vector，可能偏置 compositional evidence；多 view 或 learned directional basis可能更完整。
- Projection 只产生 semantic enrichment，不是 identifiable separation；attributed component 仍可含 nonvisual teacher effects，residual 也保持 source-mixed。

### 由实验设计可直接确认

- 仅 Qwen3.5 4B/9B、单一 Vision-OPD synthetic dataset 和细粒度图像任务；视频、多图、3D、开放空间推理未知。
- 无多训练随机种子、标准差或显著性检验；step-time 标准差是跨 step，不是跨 run。
- 主评测依赖 GPT-OSS-120B judge；未报告 judge agreement、重评稳定性或人工抽检。
- 训练 full image 含 relevant-region mark，不能证明完全无提示的自主视觉搜索能力。
- Student top-100 support 可能遗漏 teacher 强烈支持但 student 尚未考虑的正确 token；tail bucket 无法恢复其独立方向。
- Baseline 虽匹配数据和 update budget，但论文未完整列出各方法具体超参数；“further tuned”可能引入不同调参强度。
- Semantic token category 的定义、标注者/模型、样本量与一致性未在正文/附录充分交代。

### 个人分析

- **Projection geometry 是设计选择，不是自然归因。** Euclidean inner product 对 candidate support、uniform coordinate weighting、centering和 tail bucket敏感；没有证明它对应 Fisher geometry、probability mass 或 causal contribution。
- **一维 span 只能捕获共线 correction。** \(r_t^{\mathrm{vis}}=\beta u_t\) 假设可归因 correction 沿单一 intervention vector；多个对象、属性和关系可能需要多维 basis。
- **Positive projection 会丢弃 anti-aligned case。** 若 \(\langle r,u\rangle\le0\)，VAD 不施加 visual shift；这可能避免冲突 teacher，也可能漏掉因 support 截断、噪声或复杂 evidence 导致的有效 correction。
- **Top-\(K\) 是能力瓶颈。** Target 围绕 student support 重构非常保守，但 correct alternative 若不在 student top-100，只能进入 tail aggregate，无法被单独提升。
- **Diagnostic 的 23% 结论近乎由 quantile 定义。** 它计算“top-quartile teacher corrections 中也属于 top-quartile alignment 的比例”；若两者独立，期望就是 25%。23.2%/22.8% 主要说明无正关联，不能直接证明其余约 77% 都是 source-mixed，也不是 absolute alignment rate。
- **Diagnostic 使用 VAD 自己定义的 \(\rho\) 证明 VAD 动机，存在 operational circularity。** 该 proxy 若不完备，低 alignment 不能证明 correction 非视觉。
- **Degradation response 不等于纯视觉证据方向。** 0.1× down/up 还会引入 blur、aliasing 和 teacher behavior shift；它只隔离该 intervention 的响应。
- **“Support/refutation”是相对 odds 语义。** Negative shift 不保证 absolute probability suppression，论文附录也承认需与 log-normalizer 比较。
- **Regularizer 部分重新引入 source-mixed target。** 当 attributed fraction 低时 \(a_t\) 高，恰好更依赖完整 crop teacher；稳定性与 attribution purity 有明确 trade-off。
- **Held-out preservation 证据较弱。** +0.23/+0.24 远小于潜在 sampling/judge variance，只能说未观察到 aggregate 明显退化。
- **作者元数据存在页面差异。** arXiv 摘要页列 12 位作者，HTML 全文署名还包含 Shijian Wang，共 13 位，需以最终 PDF/版本元数据核对。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可将 privileged teacher correction 与“空间证据 present/removed” teacher response 对比，避免直接复制完整 source-mixed target。
- Signed contrast 同时表达证据支持与反驳，适合纠正错误对象、方向、距离和拓扑候选，而不只奖励正 visual advantage。
- Student-anchored target 只改变 intervention-supported odds，推理时不需要额外 view 或 teacher。
- Weak direct-teacher anchor 可稳定语言、格式和 stopping，但应单独报告其贡献。

### 个人建议

- 用多个正交 spatial interventions 构造 basis：对象移除、坐标遮挡、深度擦除、关系边断开、局部地图替换；做 ridge/NNLS 多维投影，而非单向量。
- 在 Fisher-weighted 或 probability-weighted geometry 中比较 projection，避免低概率 token 与 tail bucket被等权放大。
- Candidate support 使用 teacher/student union top-\(K\)+tail，或动态扩展 correct/high-intervention candidates，降低 student-support盲区。
- 分开测 relative-odds refutation 与 absolute probability suppression，不用负 coordinate 直接宣称 token 被压低。
- 对 region mark 做 dropout/退火，并在无框 full scene 上评测自主定位。
- 将 VAD target reconstruction 与 PCD stage gate结合：先判断 perception 是否可纠正，再决定重构哪些 signed spatial corrections。

## 与其他论文的相同点和冲突

### 与 [Vision-OPD](vision-opd.md)

- 使用相同 6,241 数据、full-image student、crop teacher 与 student prefixes。Vision-OPD 直接匹配完整 crop-teacher target；VAD 只保留与 crop/degraded intervention 对齐的 correction，并围绕 student 重构 target。
- VAD 更能排除 source-mixed teacher preference，但依赖 projection proxy；Vision-OPD 更简单、teacher information覆盖更完整。

### 与 [VA-OPD](va-opd.md)

- 两者都使用 evidence-present/removed teacher log-prob contrast。
- VA-OPD rectifies sampled-token positive advantage，用于 rollout/token weighting，local target仍是完整 teacher；VAD 使用全 candidate signed contrast，重构 local target并显式处理 refutation。
- VA-OPD 可选择“哪里多学”，VAD选择“这个位置具体学什么 odds shift”。

### 与 [V-Zero](v-zero.md)

- V-Zero 用 positive/negative view difference 做 sibling trajectory gate，VAD 用同类 contrast 做 candidate-level signed target reconstruction。
- V-Zero 需要 group-relative rollout comparison；VAD 每个 prefix 可独立构造 target。二者都依赖 intervention validity。

### 与 [VGS](vgs.md)

- 两者都重构 visual target，而不只是 reweight teacher loss。
- VGS 用 student text prior + teacher multimodal/text-only information gain，并 steering gradient；VAD 投影 teacher-to-student correction到 same-teacher crop/degraded direction，再以 student full-image distribution为 anchor。
- VGS 处理 language/visual gradient direction；VAD 处理 privileged correction 的 source attribution。

### 与 [VCSD](vcsd.md)

- 两者都对 candidate-level conditional log-ratio做 target shaping，并以 student/current distribution提供保守 anchor。
- VCSD 使用 EMA teacher original/content-erased contrast，无 region annotation；VAD 使用 frozen crop/degraded teacher，并额外要求完整 correction 与 contrast direction对齐。
- VCSD exponential tilt 整个 plausible support；VAD 做 one-dimensional projection、signed branch budget 和 weak direct-teacher regularization。

### 与 [PCD](pcd.md)

- VAD 回答 perception correction 的内容；PCD 回答当前失败是否应纠正 perception。
- 二者可串联为 stage-level deficiency weight × token-distribution attributed target，但都依赖 teacher 比 student 更可靠。

### 与 [ViCuR](vicur.md)、[ViGOS](vigos.md) 和 [VOLD](vold.md)

- ViCuR 改善 evidence recovery，ViGOS 控制 privilege 介入阶段，VOLD 用 correctness mask保护成功轨迹；VAD专注每个 token 的 target source attribution。
- VAD 不使用 answer privilege 或 outcome reward，但依赖 region-mark/crop privilege；recoverability 与无框部署仍需审计。

### 与 [FP-OPD](fp-opd.md)

- 两者都投影完整 teacher correction并围绕 detached student重构 target。VAD 使用 evidence intervention direction做 source attribution；FP-OPD 使用 student probe span做 local compatibility。
- FP-OPD 的 Fisher metric比 VAD uniform Euclidean coordinates更贴合 KL，但其四个 input probes不等于 parameter trainability；二者可结合 task-specific basis与 Fisher projection。

### 与 [SA-OPD](sa-opd.md)

- 两者都通过control分析correction来源。VAD固定teacher并重构candidate direction；SA-OPD比较teacher/student sampled divergence并过滤位置，后者更轻但有effect cancellation。

### 与 [OPD-V](opd-v.md)

- 两者都用crop/degraded paired teachers。OPD-V以sampled-token正margin筛选位置并复制zoom target；VAD用全candidate signed contrast重构support/refutation target。

### 综合定位

十三篇论文中，VAD 新增“teacher correction attribution”维度：不仅决定哪里、何时、以多大权重蒸馏，也显式重构 teacher correction 中由特定视觉 intervention支持或反驳的 candidate-level方向。

## 证据

### E-001

- 结论：VAD 用 evidence-present/degraded same-teacher contrast定义 signed proxy \(u_t\)。
- 类型：论文结论
- 定位：§3.1–3.2；式 (1)–(3)
- 必要引用：centered log probabilities；student top-\(K\) shared support。
- 备注：proxy 只对应所选 intervention。

### E-002

- 结论：VAD 将完整 teacher correction 单向投影到 \(u_t\)，residual 不被解释为纯语言。
- 类型：论文结论
- 定位：式 (4)–(5)
- 必要引用：\(\beta=[\langle r,u\rangle]_+/(\|u\|^2+\zeta)\)。
- 备注：是一维 Euclidean projection。

### E-003

- 结论：VAD 将视觉预算分给 support/refutation branches，并围绕 student 构造 target。
- 类型：论文结论
- 定位：§3.3；式 (6)–(7)
- 必要引用：只 cap positive branch；\(B=\|r^{vis}\|_2\)。
- 备注：signed coordinate 表示 relative odds，不保证 absolute probability变化。

### E-004

- 结论：Full VAD 使用 reconstructed-target JSD 加弱 direct-teacher JSD。
- 类型：论文结论
- 定位：§3.4；式 (8)–(10)
- 必要引用：\(\lambda=0.1\)，regularizer weight \(a_t=1-\rho_t\)。
- 备注：attribution fraction低时 source-mixed anchor更强。

### E-005

- 结论：Matched Avg\(_6\) 中 VAD 在 4B/9B 分别为 78.32/79.93。
- 类型：论文结论
- 定位：表 1；§4.2
- 必要引用：领先 best alternative 2.40/2.80。
- 备注：无多 seed。

### E-006

- 结论：Branch-aware target 与 weak regularizer均有贡献。
- 类型：论文结论
- 定位：表 2；§4.4
- 必要引用：one-sided no-reg 77.06，VAD no-reg 78.06，Full 78.32。
- 备注：regularizer净增益较小。

### E-007

- 结论：Offline correct-token support/wrong-token suppression趋势改善，但 CI 重叠。
- 类型：论文结论
- 定位：图 4；§4.4
- 必要引用：7.52→7.89；6.29→6.61。
- 备注：不能支持 pairwise statistical superiority。

### E-008

- 结论：Held-out Avg\(_4\) 与 base 基本持平。
- 类型：论文结论
- 定位：表 3；§4.5
- 必要引用：4B +0.24，9B +0.23。
- 备注：MMStar 单项仍下降，无显著性分析。

### E-009

- 结论：单 view-pair projection不是 identifiable visual decomposition。
- 类型：论文明确局限
- 定位：§5
- 必要引用：attributed 与 residual 均可能 source-mixed。
- 备注：需要 multiple views/directional basis。

### E-010

- 结论：23.2%/22.8% diagnostic 是双 top-quartile overlap，不是 absolute visual-attribution rate。
- 类型：个人核对
- 定位：Introduction；附录 A
- 必要引用：分母为 top-quartile \(D_t\)，分子再要求 top-quartile \(\rho_t\)。
- 备注：独立变量期望 overlap 约 25%，不能据此断言其余 77% 纯非视觉。

### E-011

- 结论：VAD projection 依赖 student top-\(K\) support 和 Euclidean coordinate geometry。
- 类型：个人推断
- 定位：式 (2)–(7)；§4.1
- 必要引用：student top-100 + tail，uniform centering与L2 projection。
- 备注：teacher-only correct candidate可能被 tail 合并。

### E-012

- 结论：VAD 训练成本只略高于 Vision-/VA-OPD。
- 类型：论文结论
- 定位：附录 C；表 5
- 必要引用：4B 67.9 GPU-hours；9B 96.3。
- 备注：统计前 65 steps，不含评测、checkpoint和 judge。

### E-013

- 结论：arXiv 摘要页与 HTML 全文作者数不一致。
- 类型：个人核对
- 定位：arXiv metadata；HTML title block
- 必要引用：摘要页 12 位，全文另含 Shijian Wang。
- 备注：最终引用前需核对 PDF/更新版本。

## 待验证问题

- [ ] Student top-100 之外的 teacher-supported correct token 如何处理？
- [ ] Fisher/probability-weighted projection 是否优于 uniform Euclidean projection？
- [ ] 多 view、多对象、多关系 basis 能否提高 compositional attribution？
- [ ] Negative branch 对 token absolute probability 的实际 suppression rate 是多少？
- [ ] 23% diagnostic 相对 independence baseline、cluster CI 和其他 attribution proxy 表现如何？
- [ ] Semantic token taxonomy 由谁标注，样本量与一致性是多少？
- [ ] Region mark dropout 后，无框 full-image 推理是否仍保持收益？
- [ ] 多随机种子与 judge 重评下 2–3 point 主结果是否稳定？
- [ ] Weak regularizer 的稳定收益与重新引入 source-mixed correction 如何权衡？
- [ ] 作者列表应以哪个版本为准？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，正文 §1–5、式 (1)–(10)、算法 1、表 1–4、图 1–5；附录 A–D、表 5–6。
- 新增认识：Visual contrast 不只能作 scalar weight，也可用于 teacher-correction attribution 和 signed candidate-level target reconstruction。
- 修正内容：未把 residual 当纯语言成分，未把负 coordinate 当绝对概率下降，未把双 top-quartile overlap 当作绝对视觉归因率。
- 下一步：核对代码中的 support/tail 与 projection geometry，实现 multi-direction spatial attribution 和无框评测。
