# VGS

## 基本信息

- 论文标题：Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding
- 作者：Hee Suk Yoon, Eunseop Yoon, Jaehyun Jang, SooHwan Eom, Ji Woo Hong, Mark Hasegawa-Johnson, Qi Dai, Chong Luo, Chang D. Yoo
- 年份与会议：2026；ICML 2026 Spotlight
- arXiv/DOI：arXiv:2606.00564；DOI: 10.48550/arXiv.2606.00564
- 论文链接：https://arxiv.org/abs/2606.00564
- 代码：https://github.com/hee-suk-yoon/Decomposed_OPD
- 本地文件：无
- 阅读状态：已完成（部分公式实现语义待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

VGS 将多模态 OPD 拆成语言先验匹配与视觉信息增益匹配，发现二者在高视觉依赖 token 上梯度近乎正交，于是在标准 reverse-KL 上额外注入视觉梯度、保持总梯度范数，并用高视觉依赖 token 上的语言保持项防止遗忘；相对标准 OPD，它主要在视觉依赖强的任务上提高 2B/4B 学生性能。

## 核心问题

标准 VLM on-policy distillation 直接匹配师生在图文条件下的完整 token 分布，把语言推理风格和视觉 grounding 混成一个目标。论文研究：

1. 标准多模态 OPD 的梯度中，语言先验匹配和视觉 grounding 各自贡献什么；
2. 两类梯度在视觉关键 token 上是否一致；
3. 若视觉 grounding 是小 VLM 的主要瓶颈，能否不改变 student rollout、不过度增大参数步长，仅通过调整梯度方向增强视觉学习；
4. 强化视觉梯度时，如何避免破坏已经较成熟的语言能力。

## 方法

### 整体流程

1. 学生在图像 \(I\) 与问题 \(x\) 上生成 on-policy trajectory \(\tau\)。
2. 教师和学生均在两种条件下打分同一 prefix：图文条件 \((I,x)\) 与 text-only 条件 \(x\)。
3. 标准 OPD：匹配师生图文条件下的 reverse KL。
4. 语言目标：匹配师生 text-only distribution。
5. 视觉目标：构造 \(q_T^*\)，保留学生 text-only language prior，同时注入教师由“有图—无图”logit 差表示的 visual information gain。
6. VGS：在标准 OPD 梯度上叠加 \(\gamma\) 倍视觉目标梯度，并用 \(\eta\) 把更新范数缩放回标准 OPD 水平。
7. LP：仅在 VDS 最高 30% token 上加入小权重 language-preservation loss，抑制视觉梯度对语言梯度产生负投影。

### 关键模块

**语言—视觉概率分解。** 论文用 Bayes rule 写出：

\[
\log p(\tau\mid I,x)
=\log p(\tau\mid x)
+\log p(I\mid\tau,x)
-\log p(I\mid x).
\]

其中 \(\log p(\tau\mid x)\) 是语言先验；\(\log p(\tau\mid I,x)-\log p(\tau\mid x)\) 是 visual information gain。

**视觉目标分布。** 作者希望只迁移教师视觉 likelihood，同时保留学生自己的语言 prior：

\[
q_T^*(\tau\mid I,x)
\propto p_S^\theta(\tau\mid x)\,q_T(I\mid\tau,x).
\]

实际 token logits 由学生 text-only logits 加上教师“图文 logits−文本 logits”构造，再 softmax 归一化：

\[
\log q_T^*
=\log p_S^\theta(\tau\mid x)
+\log q_T(\tau\mid I,x)
-\log q_T(\tau\mid x)
-\log Z^*.
\]

**Visual Dependency Score。** 分析和 LP gating 使用：

\[
\mathrm{VDS}_t
=D_{\mathrm{KL}}\!\left(
q_T(\cdot\mid\tau_{<t},I,x)
\|q_T(\cdot\mid\tau_{<t},x)
\right).
\]

VDS 是教师完整 token distribution 对图像的敏感度，与 VA-OPD 对 sampled token 的正 log-prob 差不同。

**梯度几何。** 按 VDS 将 token 分成 10 个等频 bin。语言梯度和视觉梯度的夹角从低 VDS bin 的约 \(60^\circ\) 增至最高 bin 的约 \(92^\circ\)；标准梯度在最高 bin 距视觉梯度约 \(42^\circ\)、距语言梯度约 \(50^\circ\)。作者将其解释为标准 OPD 被动采取固定折中方向。

**Language Preservation。** 在 top 30% VDS token 上加入 text-only teacher/student reverse KL。因为最高 VDS 区域视觉与语言梯度可能成钝角，单纯加强视觉目标会提高 language loss。

### 损失函数与训练目标

标准图文 reverse-KL（式 3–4）：

\[
\ell_{\mathrm{Standard}}(\tau)
=\frac1{|\tau|}\sum_t
D_{\mathrm{KL}}\!\left(
p_S^\theta(\cdot\mid\tau_{<t},I,x)
\|q_T(\cdot\mid\tau_{<t},I,x)
\right).
\]

语言先验目标（式 6–7）：

\[
\ell_{\mathrm{Lang}}(\tau)
=\frac1{|\tau|}\sum_t
D_{\mathrm{KL}}\!\left(
p_S^\theta(\cdot\mid\tau_{<t},x)
\|q_T(\cdot\mid\tau_{<t},x)
\right).
\]

视觉 grounding 目标（式 11）：

\[
\ell_{\mathrm{Vis}}(\tau)
=\frac1{|\tau|}\sum_t
D_{\mathrm{KL}}\!\left(
p_S^\theta(\cdot\mid\tau_{<t},I,x)
\|q_T^*(\cdot\mid\tau_{<t},I,x)
\right).
\]

VGS 与梯度范数缩放（式 13–15）：

\[
\ell_{\mathrm{VGS}}
=\ell_{\mathrm{Standard}}+\gamma\ell_{\mathrm{Vis}},
\qquad
\eta_{\mathrm{VGS}}
=\frac{\|\nabla\mathcal L_{\mathrm{Standard}}\|_2}
{\|\nabla\mathcal L_{\mathrm{Standard}}
+\gamma\nabla\mathcal L_{\mathrm{Vis}}\|_2}.
\]

LP 与最终目标（式 16–18）：

\[
\ell_{\mathrm{VGS-LP}}
=\ell_{\mathrm{Standard}}
+\gamma\ell_{\mathrm{Vis}}
+\lambda\ell_{\mathrm{LP}}.
\]

默认 \(\gamma=2.0,\lambda=0.01\)。实际训练不逐 step 动态计算 \(\eta\)，而使用预估常数：2B 为 0.41，4B 为 0.36。

RL 联合版本（式 22）将无 KL penalty 的 GRPO surrogate 与 VGS-LP 按 \(\alpha=0.3\) 混合。

### 训练数据与训练流程

- 数据：Vision-SR1-47K，共约 47k 个图像、问题、可验证答案 triplet。
- 教师：Qwen3-VL-8B-Instruct，在同一数据上用 GRPO 训练 2 epochs；AdamW，LR \(10^{-6}\)，weight decay \(10^{-2}\)，constant schedule，global batch 128，rollout group 8，temperature 1.0，top-p 0.99，max response 2048，不冻结 vision encoder，不使用 KL loss。
- 学生：Qwen3-VL-2B-Instruct、Qwen3-VL-4B-Instruct，从标准 instruct checkpoint 初始化，向 GRPO teacher 蒸馏。
- 学生 OPD：1 epoch，AdamW，LR \(10^{-6}\)，weight decay \(10^{-2}\)，constant schedule，global batch 512，temperature 1.0，top-p 1.0，max input 16384，max response 2048，不冻结 vision encoder。
- VGS：\(\gamma=2.0,\lambda=0.01\)；2B/4B 的固定 \(\eta\) 分别为 0.41/0.36。
- 硬件：单节点 8×NVIDIA A100 80GB。
- 统一 prompt：显式 `<reason>...</reason>` 推理，最终答案放入 `\boxed{}`。
- 训练成本：相对 Standard OPD，VGS 因额外 text-only forward 使单 step 时间约增至 1.375×。

## 实验设置

- 模型：Qwen3-VL-8B GRPO teacher；Qwen3-VL-2B/4B student
- 数据集：训练 Vision-SR1-47K；评测 MMMU-Pro-4、LogicVista、MathVerse-VD、MathVerse-VO、VisualPuzzles、MathVision、VlmsAreBlind；补充评测 Geo3K、WeMath、MATH500、AIME25、OlympiadBench
- Baseline：初始学生、Standard OPD；RL 实验另比较 GRPO、GRPO+Standard OPD
- 指标：greedy Acc@1；temperature 1.0 下 16 次独立生成的平均准确率 Acc@16
- 关键超参数：\(\gamma=2.0,\lambda=0.01,\alpha=0.3\)，LP/VDS threshold 为第 70 percentile

## 实验结论

1. **纯 OPD 中平均性能稳定提升。** 8B→2B 的 Acc@1 平均由 Standard OPD 43.74 提至 VGS 46.10（+2.37），Acc@16 为 45.07→46.14（+1.07）；8B→4B 的 Acc@1 为 56.64→58.12（+1.56），Acc@16 为 56.86→57.27（+0.50）。
2. **收益在视觉依赖强的任务上更明显。** 2B Acc@1 的 VisualPuzzles、LogicVista、VlmsAreBlind 分别 +3.68、+3.35、+2.26；附录 D 显示 high-vision-dependency benchmark 对视觉 steering 最敏感，低视觉依赖和 text-only benchmark 差异很小。
3. **“所有 benchmark 均提升”并不严格成立。** 表 1 的 4B VisualPuzzles Acc@1 为 40.75→40.31（-0.44），虽然对应 Acc@16 +0.05；论文摘要和 §6 的“across all seven benchmarks”应限定到总体趋势或 Acc@16。
4. **视觉 steering 优于语言 steering。** 增大 visual \(\gamma\) 提高七项平均准确率；反向增强 language objective 则低于 Standard OPD，支持“该设置下视觉 grounding 是更大瓶颈”。
5. **LP 防止高 VDS token 上语言先验发散。** 仅 VGS、\(\gamma=2\) 时 high-VDS language loss 上升；加入小权重 LP 后 language loss 保持稳定，同时视觉 loss 下降更快。
6. **与 GRPO 结合仍有平均收益。** GRPO+Standard OPD 到 GRPO+VGS 的 Acc@1 平均 45.41→47.20（+1.79），Acc@16 45.22→46.57（+1.35）。
7. **RL 联合结果也非每个单项 Acc@1 都提升。** MathVerse-VO 为 58.03→57.80（-0.23），VlmsAreBlind 为 51.90→51.74（-0.16）；Acc@16 则七项均提高。
8. **VGS 约束生成长度。** 纯 GRPO 出现 length explosion；Standard OPD 与 VGS 都把长度拉向教师平均值，而 VGS 在类似长度下准确率更高。
9. **Token-adaptive VGS 小幅提高平均结果。** 按 VDS 分段设置 \(\gamma_t\in\{0,\gamma/2,\gamma\}\) 后，2B Acc@1 平均由固定 VGS 46.10 升至 46.53；但 LogicVista、MathVision、VlmsAreBlind 低于固定 VGS，不是逐项提升。

## 局限性

### 论文明确承认

- 额外 text-only forward 带来约 1.375× 单 step 训练时间。
- 方法依赖教师多模态与 text-only distribution 之间存在可靠差异；若教师本身视觉盲、\(\nabla\mathcal L_{\mathrm{Vis}}\approx0\)，VGS 退化为 Standard OPD，不能创造教师没有的视觉能力。

### 由实验设计可直接确认

- 只验证 Qwen3-VL 同家族、单一 8B teacher 和 2B/4B students；跨架构泛化未知。
- 教师先在与学生相同的 Vision-SR1-47K 上 GRPO，未测试基础 instruct teacher 或训练/蒸馏数据分离时的效果。
- 主表没有多训练随机种子的均值、标准差或显著性检验；“significantly”主要表示数值提升而非统计显著。
- 学生和教师 vision encoder 都不冻结，无法区分视觉提升来自视觉塔、projector 还是语言解码器。
- 主要任务仍是视觉数学、逻辑和短答案 reasoning；开放式 grounding、检测、视频、3D 与交互规划未验证。
- 固定 \(\eta\) 依赖模型架构和 \(\gamma\)，换模型或训练阶段可能需要先做额外梯度统计。

### 个人分析

- Bayes 分解对完整 sequence probability 成立，但实现用每一步局部 softmax 构造 \(q_T^*\)。论文没有严格证明“逐 token 归一化后的 target KL”等价于 sequence-level visual likelihood matching；这是合理构造，但“数学等价”表述可能强于实际自回归实现证据。
- \(q_T^*\) 含当前学生的 text-only distribution。若该分支参与反向传播，视觉 target 本身依赖 \(\theta\)，会出现 target leakage 或额外梯度路径；论文没有明确说明构造 \(q_T^*\) 时是否对 student text-only logits stop-gradient，需核对代码。
- 式 (15) 的 \(\eta\) 由梯度定义。若在式 (14) 中直接作为可微量，求导会引入二阶项，不能简单保证更新范数相等；实际固定常数规避了该问题，但理论公式应明确 stop-gradient。
- “语言/视觉梯度近乎正交”主要来自一个模型家族和训练设置，且最高 bin 约 \(92^\circ\) 已是轻微冲突，不只是独立。不能直接视为所有 VLM 的结构性质。
- VDS 以教师有图/无图 forward KL 衡量视觉依赖，可能混合真正 grounding、图像触发的风格变化和 teacher calibration；与准确视觉证据没有必然等价关系。
- LP 只在 top 30% VDS token 激活，但这些 token 是否恰好是语言遗忘风险最高的位置由当前梯度图支持，跨任务阈值稳定性未知。
- Appendix E 式 (24) 在 token 求和内部仍写 trajectory-level \(\ell_{\mathrm{Standard}}(\tau)\)、\(\ell_{\mathrm{Vis}}(\tau)\)、\(\ell_{\mathrm{LP}}(\tau)\)，而非 token-level loss；按字面会重复整条轨迹损失，疑似记号错误，`待核对`。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可通过“完整空间输入 vs text-only/移除空间状态输入”的教师 distribution 差，构造空间信息增益 target，而不只做普通 teacher KL。
- 可直接记录 language gradient、spatial grounding gradient 的 cosine/angle，判断训练瓶颈是推理先验还是空间感知，而不是预设二者同权。
- 对视觉/空间目标加权时应保持总梯度范数，避免把方向收益与等效 learning-rate 增大混淆。
- 在高空间依赖 token 上增加轻量 language-preservation regularizer，可抑制空间 steering 对推理流畅性的破坏。
- VGS 可作为 GRPO 的视觉正则器，让 outcome reward 管正确性，让分解蒸馏管 grounding。

### 个人建议

- 将 text-only counterfactual 扩展为多种空间反事实：移除图像、深度、对象关系图、位姿、局部地图，分别构造 factor-specific target 与梯度。
- 对 \(q_T^*\) 中的 student language logits 做 stop-gradient 与不做 stop-gradient 的明确消融，并记录梯度流。
- 不要直接照搬 top 30% VDS；先分析 SpatialStack 中 VDS、正确性、teacher entropy 与梯度冲突的联合分布，再设 gate。
- 同时比较三种干预：VA-OPD 的 token weighting、VGS 的 gradient steering、二者组合，确认收益来自“选对 token”还是“选对方向”。
- 对每个训练阶段动态抽样估计 \(\eta\)，验证固定常数是否持续稳定；至少报告未归一化、固定归一化和 stop-gradient 动态归一化。

## 与其他论文的相同点和冲突

### 与 [VA-OPD](va-opd.md)

- 两者都指出 Standard OPD 会过度受语言信号影响，视觉 grounding 需要显式增强；都用教师有图/视觉信息受限条件的分布差衡量 token 视觉依赖。
- VA-OPD 的 VA 是 sampled token 在原图/像素化图下的正 log-prob 差，并通过 rollout weighting 与 high/low-token grouped KL 分配 loss mass；VGS 的 VDS 是教师完整 distribution 在有图/text-only 条件下的 forward KL，并通过构造视觉 target 改变梯度方向。
- VA-OPD 更像“选择哪些 token/轨迹”；VGS 更像“这些位置沿哪个模态子空间更新”。Appendix E 的 adaptive VGS 开始引入 token 级选择，两者边界因此部分重合。
- VA-OPD 的退化图保留全局布局，主要探测细节；VGS 直接移除全部图像，视觉差更强但可能混入全局语义、风格和格式变化。

### 与 [VOLD](vold.md)

- 两者都能与 GRPO 共享 student rollout 并联合优化；VOLD 用 reward mask 保护正确轨迹，VGS 用视觉分解确保 RL reasoning 保持 grounding。
- VOLD 依赖 text-only teacher，目标是把语言推理迁移给 VLM；VGS 依赖 visual teacher 的有图/无图差，目标是增强视觉 grounding。
- VOLD 说明 OPD 前需 teacher/student state alignment；VGS 通过统一 prompt 保持结构对齐，但未做冷启动对齐程度消融。

### 与 [Vision-OPD](vision-opd.md)

- Vision-OPD 用同模型 privileged crop teacher 解决局部证据在全图中难以聚焦；VGS 用跨规模教师的有图/text-only信息差解决标准 loss 的模态梯度折中。
- Vision-OPD 的关键稳定性是 frozen/EMA teacher；VGS 使用固定 8B teacher，重点转向视觉/语言梯度冲突。
- 二者可组合：privileged crop 提供更可靠视觉 likelihood，VGS 再把该视觉信息从语言先验中分离；但 crop 丢失全局关系时仍需 teacher-confidence gate。

### 综合定位

与 [ViCuR](vicur.md) 相比，VGS 从 teacher 有图/text-only distribution 自动构造视觉目标，不需要离线 cue text；ViCuR 则显式改变 privilege 类型并增加 student recovery architecture。ViCuR 解决“教师依据的信息能否由部署输入恢复”，VGS 解决“已有视觉监督沿什么梯度方向更新”，二者处于互补层级。

与 [ViGOS](vigos.md) 相比，VGS 在同一 token 上分解并 steering language/visual gradient；ViGOS 先按 description/reasoning 阶段选择不同 teacher。前者控制更新方向，后者控制 privilege 进入时间，可进一步组合。

与 [V-Zero](v-zero.md) 相比，V-Zero 通过 crop contrast 选择哪条 sibling trajectory 多学，VGS 通过 image/text-only decomposition 决定沿哪个模态方向更新；trajectory gate 与 gradient steering 是互补层级。

与 [VCSD](vcsd.md) 相比，两者都用 image-conditioned 与 no-content/text-only distribution difference。VGS 构造 visual target 并 steering gradient，VCSD 对 EMA teacher target 做 support-restricted exponential shaping；两者对 control distribution 的信息论解释都依赖 intervention validity。

与 [PCD](pcd.md) 相比，PCD 决定哪个 perception trajectory 应被纠正，VGS 决定视觉纠正沿哪个梯度方向执行；前者提供 when/stage credit，后者提供 how-to-update。

与 [VAD](vad.md) 相比，两者都重构 visual target。VGS 注入 teacher multimodal/text-only information gain并 steering gradient；VAD 将 privileged correction投影到 crop/degraded direction并预算 support/refutation。

与 [FP-OPD](fp-opd.md) 相比，两者都改变 correction direction；VGS 在 parameter-gradient层 steering，FP-OPD 在 output-distribution层做 Fisher target projection，但后者的 input tangent不直接等于参数可训练空间。

与 [SA-OPD](sa-opd.md) 相比，两者都针对language-prior contamination；VGS重定向visual gradient，SA-OPD删除no-prompt-invariant高影响位置。

与 [OPD-V](opd-v.md) 相比，两者都处理modality imbalance；VGS在gradient层分解language/visual目标，OPD-V用zoom/mask正margin筛token，但未直接优化attention balance。

十三篇论文分别补足 OPD 的不同条件：VOLD 处理 state alignment 与 correctness selection；Vision-OPD 处理 privileged view 与 teacher temporal stability；VA-OPD 处理视觉 token 的监督稀疏性；VGS 处理语言/视觉目标的梯度方向冲突；ViCuR 处理 privilege recoverability 与学生证据恢复通道；ViGOS 处理 privilege intervention stage 与 teacher routing；V-Zero 处理无答案标签时的 sibling-relative evidence gating；VCSD 处理无辅助 privilege 时的 target asymmetry 与递归 support 控制；PCD 处理 perception/reasoning 间的 credit assignment；VAD 处理 privileged correction attribution；FP-OPD 处理 correction 与 student local response geometry的兼容性；SA-OPD 处理 input-grounded supervision reliability；OPD-V 处理modality-balance trust region。

## 证据

### E-001

- 结论：VGS 将标准多模态 OPD 分为 text-only language prior 与基于教师 visual information gain 的视觉目标。
- 类型：论文结论
- 定位：§3.1；式 (5)–(11)；图 2
- 必要引用：无
- 备注：逐 token 构造与 sequence-level 数学等价性的严格性仍需核对。

### E-002

- 结论：语言与视觉梯度的夹角随 VDS 增大，从约 \(60^\circ\) 增至最高 bin 的约 \(92^\circ\)。
- 类型：论文结论
- 定位：§3.2；图 3(a)
- 必要引用：无
- 备注：结果来自当前模型/数据设置。

### E-003

- 结论：VGS 用 standard loss + visual loss 旋转梯度，并用 \(\eta\) 保持与 standard gradient 相同范数。
- 类型：论文结论
- 定位：§4；式 (13)–(15)；附录 A、图 7
- 必要引用：主实验固定 \(\eta_{2B}=0.41,\eta_{4B}=0.36\)。
- 备注：动态公式的 stop-gradient 语义未说明。

### E-004

- 结论：LP 在 top 30% VDS token 上保持教师语言先验，防止视觉 steering 引发语言 unlearning。
- 类型：论文结论
- 定位：§4；式 (16)–(18)；图 4
- 必要引用：默认 \(\lambda=0.01\)。
- 备注：threshold 跨设置稳定性未知。

### E-005

- 结论：VGS 相对 Standard OPD 提高 2B/4B 学生的平均 Acc@1 和 Acc@16。
- 类型：论文结论
- 定位：§6；表 1
- 必要引用：2B 平均 +2.37/+1.07；4B 平均 +1.56/+0.50。
- 备注：4B VisualPuzzles Acc@1 下降 0.44，不能表述为每个指标均提升。

### E-006

- 结论：视觉 steering 提高平均准确率，语言 steering 则降低平均准确率。
- 类型：论文结论
- 定位：§7；图 5
- 必要引用：无
- 备注：支持当前设置下的 asymmetric maturity hypothesis。

### E-007

- 结论：GRPO+VGS 平均优于 GRPO+Standard OPD。
- 类型：论文结论
- 定位：§8；式 (22)；表 2；图 6
- 必要引用：Acc@1 45.41→47.20，Acc@16 45.22→46.57。
- 备注：两个单项 Acc@1 略降，Acc@16 均升。

### E-008

- 结论：VGS 的额外 text-only forward 使单 step 训练时间约为 Standard OPD 的 1.375×。
- 类型：论文结论
- 定位：附录 C；图 9；附录 G
- 必要引用：无
- 备注：未给绝对吞吐、总 wall-clock 或能耗。

### E-009

- 结论：VGS 的收益集中在高视觉依赖 benchmark，低视觉依赖与 text-only benchmark 基本不变。
- 类型：论文结论
- 定位：附录 D；图 10
- 必要引用：无
- 备注：图中未给完整精确数值。

### E-010

- 结论：adaptive token-level VGS 的平均 Acc@1 略高于固定 VGS。
- 类型：论文结论
- 定位：附录 E；式 (24)–(25)；表 5
- 必要引用：46.10→46.53。
- 备注：式 (24) 的 token/trajectory loss 记号疑似错误，且并非每项提升。

### E-011

- 结论：\(q_T^*\) 中 student language prior 的梯度处理方式未明确。
- 类型：个人核对
- 定位：§3.1；式 (9)–(11)
- 必要引用：\(q_T^*\) 显式依赖 \(p_S^\theta(\tau\mid x)\)。
- 备注：是否 stop-gradient 将改变优化目标，`待核对`。

### E-012

- 结论：论文“across all benchmarks”的表述与表 1/2 的部分 Acc@1 数值不一致。
- 类型：个人核对
- 定位：摘要；§6；表 1；§8；表 2
- 必要引用：4B VisualPuzzles -0.44；RL 联合 MathVerse-VO -0.23、VlmsAreBlind -0.16。
- 备注：平均值与 Acc@16 仍总体提高。

## 待验证问题

- [ ] 构造 \(q_T^*\) 时 student text-only logits 是否 stop-gradient？不同处理如何影响结果？
- [ ] 动态 \(\eta\) 是否 stop-gradient？固定 \(\eta\) 跨训练阶段、模型与 \(\gamma\) 是否稳定？
- [ ] 逐 token target 是否严格等价于 sequence-level visual likelihood matching？
- [ ] 语言/视觉梯度近正交是否可跨模型家族、数据和层级复现？
- [ ] VDS 是否真正定位正确视觉证据，而非图像触发的风格或 calibration 差异？
- [ ] VA token weighting、VGS gradient steering 及二者组合哪个更有效？
- [ ] Appendix E 式 (24) 的实际实现是 token-level loss 还是重复 trajectory-level loss？
- [ ] 多训练随机种子下平均增益和小幅单项差异是否显著？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，正文 §1–11、式 (1)–(22)、表 1–2、图 1–6；附录 A–G、式 (23)–(25)、表 3–5、图 7–10。
- 新增认识：VLM OPD 不仅有 token 重要性不均，还存在语言先验与视觉 grounding 的梯度方向冲突；控制梯度方向和范数可与 token gating、reward masking 互补。
- 修正内容：未将“所有 benchmark 均提升”“数学等价”“无语言退化”按摘要表述直接接受；记录了表格例外、target gradient 和 adaptive formula 疑点。
- 下一步：核对公开代码中 \(q_T^*\)、stop-gradient、\(\eta\) 与 adaptive VGS 的实现，并在 SpatialStack 上做模态梯度角度诊断。
