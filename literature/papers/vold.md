# VOLD

## 基本信息

- 论文标题：VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation
- 作者：Walid Bousselham, Hilde Kuehne, Cordelia Schmid
- 年份与会议：2025；arXiv 预印本（v3 更新于 2026-06-03，未注明正式会议）
- arXiv/DOI：arXiv:2510.23497；DOI: 10.48550/arXiv.2510.23497
- 论文链接：https://arxiv.org/abs/2510.23497
- 本地文件：无
- 阅读状态：已完成（实现细节有待核对项）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

VOLD 先用文本教师生成的推理轨迹对 VLM 学生做 SFT 冷启动，使两者策略分布对齐，再在纯文本数学题上复用学生 rollout，同时进行 GRPO 和教师 token 级蒸馏；实验表明，对齐是跨模态 on-policy distillation 生效的必要前提，而只对错误轨迹施加蒸馏可减少教师与 RL 正确信号的冲突。

## 核心问题

高质量视觉推理数据稀缺，但文本推理数据和文本教师易于扩展。论文研究：能否不使用图文推理训练数据，仅依靠文本教师、文本推理题和 RL，把推理能力迁移到保留视觉输入能力的 VLM；以及在 student rollout 上进行 token 级教师蒸馏时，怎样避免 teacher/student 分布错位和蒸馏信号干扰学生已找到的正确路径。

## 方法

### 整体流程

1. 学生为 Qwen2.5-VL-3B-Instruct，教师默认为 Qwen3-8B，二者共享 tokenizer 和词表；训练期间冻结视觉编码器，只更新语言模型参数。
2. Stage 1（策略对齐）：用教师回答 Mixture-of-Thoughts（MoT）中的 prompt，构造 MoT-Teacher-8B；学生对教师轨迹做 SFT。目的不是直接提高任务准确率，而是让 student rollout 进入教师熟悉、低熵且能提供有效 token 分布指导的状态区域。
3. Stage 2（统一 RL + 蒸馏）：学生在纯文本 orz-57k 数学题上生成同一批 on-policy rollout。一方面用答案 exact match 的二值 reward 计算 GRPO，另一方面让固定教师在学生生成的每个 prefix 上给出 token 级分布指导。
4. Reward-guided KL masking：正确轨迹不做蒸馏，错误轨迹才接受教师指导，从而保留学生自行探索到的正确推理路径。
5. 推理时直接在视觉推理 benchmark 上 zero-shot 测试，不再做图文微调。

### 关键模块

**策略对齐冷启动。** On-policy 蒸馏访问的是学生自己产生的 prefix。若学生策略与教师相距过远，这些 prefix 对教师而言可能是分布外状态，教师分布会变得弥散，梯度弱或方差大。论文用同一教师生成的 SFT 轨迹建立“breadcrumb trail”。补充实验显示约 3000 SFT steps 后收益趋于饱和。

**共享 rollout 的统一训练。** GRPO 和 on-policy distillation 都需要学生 rollout，因此复用同一批轨迹。GRPO 提供稀疏、序列级、可验证奖励，教师提供稠密、token 级指导。

**Reward-guided KL masking。** 用二值 reward 构造 \(1-r(\tau)\) 掩码：回答正确时 KL 项为 0，回答错误时启用教师指导。论文报告最终训练 reward 约为：GRPO 0.51、无 masking 的 VOLD 0.56、带 masking 的 VOLD 0.58。

### 损失函数与训练目标

Stage 1 的 SFT 目标（式 4）：

\[
\mathcal{L}_{\mathrm{SFT}}
=-\mathbb{E}_{(q,\tau^*)\sim\mathcal{D}_{teacher}}
\sum_t\log\pi_\theta(y_t^*\mid q,y_{<t}^*).
\]

Stage 2 先以组内 reward 标准化得到 GRPO advantage：

\[
A_i=\frac{r_i-\bar r}{\sigma_r+\delta}.
\]

论文将 token 级蒸馏写为（式 3）：

\[
\mathcal{L}_{\mathrm{RKL}}
=\mathbb{E}_{q,\tau\sim\pi_\theta}
\sum_t D_{\mathrm{KL}}\!\left(\pi_\phi(\cdot\mid h_t)\|
\pi_\theta(\cdot\mid h_t)\right).
\]

最终使用 masked objective（式 6）：

\[
\mathcal{L}_{\mathrm{VOLD\text{-}masked}}
=\mathcal{L}_{\mathrm{GRPO}}
+\beta\mathbb{E}_{q,\tau\sim\pi_\theta}
\left[(1-r(\tau))\sum_t
D_{\mathrm{KL}}(\pi_\phi\|\pi_\theta)\right].
\]

重要实现疑点：论文称式 (3) 为 reverse KL，并写成 \(D_{\mathrm{KL}}(\pi_\phi\|\pi_\theta)\)，但又称用从学生策略采到的单 token 和 “k2” estimator 近似。直接从学生分布采样通常对应估计 \(D_{\mathrm{KL}}(\pi_\theta\|\pi_\phi)\)；若要估计文中所写方向，需要完整教师分布或重要性修正。仅凭论文当前描述无法确认实际代码采用哪个方向，故标记为 `待核对`。

### 训练数据与训练流程

- SFT：MoT 含约 350k 条数学、代码、科学验证轨迹。作者保留其 prompt，重新用 Qwen3-8B 生成轨迹；只过滤超过 8192 token 的样本，不验证答案正确性。
- SFT 超参数：4000 steps（约 5 epochs），batch size 256，learning rate \(5\times10^{-5}\)，max length 8192，AdamW，weight decay \(10^{-2}\)，150 warmup steps，cosine schedule；32 张 A100。
- RL：orz-57k 纯文本数学题，ground-truth exact match 二值 reward；60 steps，5 rollouts/prompt，batch size 256，learning rate \(6\times10^{-6}\)，max length 8192，AdamW，5 warmup steps；4 张 A100。
- GRPO 非对称 clipping：下界 0.2、上界 0.3。
- KL 系数存在文内矛盾：正文 §4.1 写 \(\beta=0.1\)，补充材料表 4 写 \(1\times10^{-3}\)，实际值 `待核对`。
- RL 期间用视觉几何数据集 Geo3K 做 validation，监测纯文本训练向视觉任务的迁移。

## 实验设置

- 模型：Qwen2.5-VL-3B-Instruct（3.75B）学生；Qwen3-8B 默认教师，另消融 4B/14B 教师
- 数据集：训练使用 MoT prompt + 教师轨迹、orz-57k；主评测使用 MMMU-Pro、MMStar、MathVision、MathVista、MathVerse、DynaMath、WeMath、LogicVista；补充评测 Geo3K、MME、HallusionBench
- Baseline：原始 Qwen2.5-VL-3B；复现 X-Reasoner；VLM-R1 3B-Math；VLAA-Thinker；以及 SFT-only、SFT+GRPO、错配 SFT 来源等内部消融
- 指标：各 benchmark accuracy；MME 分数；RL training reward
- 关键超参数：见“训练数据与训练流程”；推理使用 vLLM + VLMEvalKit，答案由 GPT-4o-mini 抽取

## 实验结论

1. **相对同底座模型，纯文本训练能改善多数视觉推理任务。** VOLD 相对 base 在 MMMU-Pro 为 32.0 vs 27.1，MathVision 为 28.0 vs 21.9，MathVerse 为 37.9 vs 31.2，DynaMath 为 50.7 vs 42.7，LogicVista 为 45.0 vs 40.3；但 MMStar 略降（55.2 vs 55.9），MathVista 仅小幅升高（61.9 vs 61.2）。
2. **相对同为纯文本训练的 X-Reasoner 复现，VOLD 在推理密集任务上更强。** MathVision 28.0 vs 24.4、LogicVista 45.0 vs 41.1；但 MMStar 相同（55.2），因此不能概括为所有 benchmark 均显著提升。
3. **同教师 SFT 对齐是 on-policy distillation 生效的关键。** 使用 DeepSeek-R1 轨迹的原始 MoT 做 SFT 时，加入蒸馏几乎无收益（例如 MathVision 24.4→24.5、LogicVista 41.1→41.2）；使用 Qwen3-8B 自身生成轨迹对齐后，完整 VOLD 达到 28.0 和 45.0。
4. **SFT 本身会暂时损害性能，但为后续蒸馏创造条件。** Teacher-MoT SFT-only 的多项指标低于 base；SFT+RL 有所恢复；完整 VOLD 最优。作者归因于 SFT 轨迹未验证、含错误答案。
5. **视觉推理提升伴随感知能力下降。** MMStar perception 从 58.4 降至 54.4，而 reasoning 从 53.5 升至 55.9；MME perception 从 1564 降至 1530，reasoning 从 594 升至 738。HallusionBench 则从 46.3 升至 49.8。
6. **教师越大并非越好。** 8B 通常优于 4B，但 14B 对 3B 学生没有稳定增益，作者认为学生容量形成瓶颈。
7. **VOLD 可作为后续视觉 RL 初始化。** VOLD 后再用图文数据做 RL，MathVista/MathVerse 达到 63.4/38.7，高于仅文本 VOLD 的 61.9/37.9 和从底座开始图文 RL 的 61.0/36.4。
8. **蒸馏收益不局限于 GRPO。** 补充材料图 6 显示，加入 OPD 后 GRPO 和 GSPO 的 Geo3K validation accuracy 均改善。

## 局限性

### 论文已呈现或可由实验直接确认

- 教师与学生必须共享 tokenizer 和 vocabulary，限制了可用模型组合。
- 训练仍需同时运行 3.75B 学生与最高 14B 教师；论文称 rollout 可复用、额外开销较小，但未报告 wall-clock、吞吐、显存或总 FLOPs，无法量化“virtually no additional cost”。
- SFT 使用未验证的教师轨迹，SFT-only 性能明显下降。
- 推理强化导致部分感知指标退化，不能把总体提升理解为通用视觉能力无损增强。
- 主实验仅采用一个学生模型家族与规模；跨架构、不同视觉编码器、不同 tokenizer 的泛化未验证。
- 主结论主要来自数学、几何与逻辑型视觉 benchmark，对开放式图像理解、视频、空间交互任务的适用性未知。

### 个人分析

- “跨模态迁移”主要发生在共享语言解码器的推理策略中；视觉塔全程冻结，论文没有证明新的视觉表征或视觉 grounding 能力被学到。更准确的解释是：保留已有视觉感知能力，同时提升语言侧推理后处理。
- SFT 来源同时改变了“与教师的匹配程度”和“轨迹质量/过滤方式”。虽然表 2 支持分布对齐假设，但未用可测的 KL、teacher perplexity 或固定质量数据做严格控制，因此因果证据仍不完全。
- 主结果中的 X-Reasoner 是作者自行复现且无官方 checkpoint，复现差异可能影响比较。
- 依赖 GPT-4o-mini 做答案抽取引入外部闭源评测组件；论文未报告抽取误差或多次评测方差。
- 仅训练 60 个 RL steps，且多项结果未报告随机种子、标准差和显著性检验；小幅差异（如 0.1–1.0 point）不应过度解读。
- 式 (3) 的 KL 方向、命名和采样估计描述可能不一致；在代码公开前，复现者应优先核对真实实现。

## 可复用到当前项目的内容

### 论文直接支持

- 若 SpatialStack_OPSD 的目标也是把强文本教师的空间推理迁移给 VLM，可以采用“两阶段对齐 + 在线指导”，先让学生模仿同一个教师的输出风格，再进入 RL，避免直接 OPD 产生无效或高方差梯度。
- 在有可验证最终答案的空间任务中，用同一 student rollout 同时计算 outcome reward 与 token-level teacher guidance，可减少重复采样。
- 对正确轨迹关闭教师 KL、仅纠正失败轨迹，适合保留学生探索出的、不同于教师的有效空间推理路径。
- 冻结视觉塔可作为低风险第一版实验，以检验提升是否来自语言侧空间推理，而不破坏已有视觉特征。
- 训练过程中增加一个视觉空间验证集，即使 RL 数据全是文本，也能实时观察是否发生目标域迁移。

### 个人建议

- 在项目中先做四组最小消融：RL-only、错配教师 SFT+RL+OPD、同教师 SFT+RL、同教师 SFT+RL+masked OPD；同时直接测 student rollout 在 teacher 下的 token NLL/entropy，验证“对齐阈值”而非只看 SFT steps。
- 对空间任务，二值最终答案可能过稀疏。可研究把 masking 从 response-level 扩展为步骤级置信度或可验证中间状态，但这超出本文证据。
- 同时报告空间推理和基础视觉感知，防止 reasoning gain 掩盖 perception regression。
- 实现前必须确定 KL 的实际方向与估计器，并对正文 \(\beta\) 冲突做小规模 sweep。

## 与其他论文的相同点和冲突

与 [Vision-OPD](vision-opd.md)、[VA-OPD](va-opd.md)、[VGS](vgs.md)、[ViCuR](vicur.md)、[ViGOS](vigos.md)、[V-Zero](v-zero.md)、[VCSD](vcsd.md)、[PCD](pcd.md)、[VAD](vad.md)、[FP-OPD](fp-opd.md)、[SA-OPD](sa-opd.md) 和 [OPD-V](opd-v.md) 一样，VOLD 都在 student-generated prefix 上施加 token 级监督；十三者共同表明“on-policy”本身不足，还需控制状态对齐、teacher 稳定性、监督分配、模态梯度、privilege gap、介入阶段、轨迹证据质量、递归 target 漂移、跨阶段 credit、correction attribution、student compatibility、input-grounded reliability 与 modality balance。

- VOLD 解决跨模型状态错配：用同教师轨迹 SFT 冷启动，并用 reward mask 只纠正失败轨迹。
- Vision-OPD 解决同源 self-teacher 的时间漂移：用 frozen/EMA target，并通过特权 crop 向带框全图学生迁移细粒度感知。
- VA-OPD 解决视觉监督的 token 稀疏性：按原图/退化图 teacher log-prob 差识别高 VA token，再做 rollout reweighting 与 grouped KL。
- VGS 解决语言/视觉目标的梯度折中：构造 visual information-gain target，转向视觉梯度，并用 LP 保持语言先验。
- ViCuR 解决 teacher privilege 的部署可恢复性：以视觉 cue 替代答案/rationale，并增加学生证据聚合通道。
- ViGOS 解决 answer privilege 的阶段路由：先用 image-only teacher 监督视觉描述，再让 answer teacher 监督推理，并以格式 fallback 处理无效 rollout。
- V-Zero 解决无答案标签时的 trajectory weighting：用目标/负 crop 对比得到 sibling-relative evidence gate，但它不判断答案是否正确。
- VCSD 解决无辅助 privilege 时的 target asymmetry：用 EMA teacher 原图/control contrast 重塑 vocabulary target，但同样缺少 correctness 判断。
- PCD 延伸 VOLD 的 correctness selection：把 final failure 与 perception teacher disagreement结合，尝试区分感知失败和后续 reasoning 失败。
- VAD 不做 reward mask，而是把 privileged correction 投影到视觉 intervention direction，决定失败位置具体应学习哪种 signed target shift。
- FP-OPD 将 teacher gap 投影到 student probe-estimated Fisher response span，处理 teacher target相对小 student的局部兼容性。
- SA-OPD 进一步质疑大 divergence是否依赖当前输入，用 no-prompt disagreement gap过滤可疑高影响 token。
- OPD-V 用zoom/masked-crop teacher正margin构造modality-balance trust region，但不含VOLD式correctness protection。
- VOLD 的 text-only teacher 无法直接计算 visual advantage；若 SpatialStack 使用视觉教师，可把 VOLD 的 correctness masking 与 VA-OPD 的 visual-dependency weighting 组合。
- VOLD 与 VGS 都验证了 GRPO+OPD；前者按 correctness 决定是否模仿，后者把 grounding 作为与 outcome reward 互补的梯度方向约束。
- VOLD 中 8B→14B text teacher 的绝对收益饱和，而 VA-OPD 中 4B→32B visual teacher 相对 Standard OPD 的增量扩大；由于学生、模态、数据和比较量不同，不能视为直接矛盾。

## 证据

### E-001

- 结论：VOLD 是 SFT 对齐后再联合 GRPO 与 on-policy distillation 的两阶段纯文本训练框架。
- 类型：论文结论
- 定位：摘要；§1；图 2；§3.2
- 必要引用：无
- 备注：视觉编码器在两个阶段均冻结。

### E-002

- 结论：同一教师生成的 SFT 轨迹所建立的策略对齐，是后续 OPD 获益的关键前提。
- 类型：论文结论
- 定位：§3.2 “Why Alignment Is Necessary”；§4.3；表 2；补充材料 §7.1、图 4
- 必要引用：使用原始 MoT 时 SFT+RL 与 SFT+RL+Distillation 几乎相同；VOLD 完整配置明显更高。
- 备注：未直接报告 teacher/student divergence 测量，因果机制仍有待更严格控制实验。

### E-003

- 结论：masked VOLD 只对 reward=0 的失败轨迹施加教师 KL，对成功轨迹关闭蒸馏。
- 类型：论文结论
- 定位：§3.2 “Reward-Guided KL Masking”；式 (6)
- 必要引用：无
- 备注：补充材料 §7.3、图 5 支持其优于不 masking 的版本。

### E-004

- 结论：VOLD 相对 Qwen2.5-VL-3B base 和 X-Reasoner 在多数推理 benchmark 上改善，但不是所有指标都提升。
- 类型：论文结论
- 定位：§4.2；表 1
- 必要引用：MathVision 28.0，LogicVista 45.0；MMStar 55.2，低于 base 55.9。
- 备注：X-Reasoner 由作者复现。

### E-005

- 结论：SFT-only 会降低性能；加入 GRPO 后恢复，完整 RL+OPD 最强。
- 类型：论文结论
- 定位：§4.4；表 3
- 必要引用：MathVision 18.6→24.0→28.0；LogicVista 28.9→38.3→45.0。
- 备注：作者将 SFT 退化归因于未过滤的错误教师轨迹。

### E-006

- 结论：文本训练提升 reasoning 的同时会轻微损害部分 perception 指标。
- 类型：论文结论
- 定位：补充材料 §7.4；表 6
- 必要引用：MMStar perception 58.4→54.4，reasoning 53.5→55.9。
- 备注：同类退化也见于图文训练 baseline，但不能据此证明机制完全相同。

### E-007

- 结论：8B 到 14B 教师的收益饱和。
- 类型：论文结论
- 定位：补充材料 §7.2；表 5
- 必要引用：8B 与 14B 在各 benchmark 上互有胜负，无一致提升。
- 备注：学生容量瓶颈是作者解释，不是直接验证。

### E-008

- 结论：论文超参数对 KL 系数的记录矛盾。
- 类型：个人核对
- 定位：正文 §4.1 “Training & Implementation Details”；补充材料表 4
- 必要引用：正文为 \(\beta=0.1\)，表 4 为 \(1\times10^{-3}\)。
- 备注：`待核对`。

### E-009

- 结论：论文所写 KL 方向与单个 student-sampled token 的 k2 估计描述可能不匹配。
- 类型：个人推断
- 定位：§3.1 式 (3) 及紧随其后的实现说明
- 必要引用：式 (3) 写 \(D_{\mathrm{KL}}(\pi_\phi\|\pi_\theta)\)，采样轨迹来自 \(\pi_\theta\)。
- 备注：需要源码或作者澄清，`待核对`。

### E-010

- 结论：VOLD checkpoint 继续做 image-based RL，可进一步提高 MathVista 和 MathVerse。
- 类型：论文结论
- 定位：补充材料 §7.5；表 7
- 必要引用：VOLD+image RL 为 63.4/38.7，VOLD 为 61.9/37.9。
- 备注：只报告两个 benchmark。

## 待验证问题

- [ ] 实际实现使用的是 \(D_{\mathrm{KL}}(\pi_\phi\|\pi_\theta)\) 还是 \(D_{\mathrm{KL}}(\pi_\theta\|\pi_\phi)\)？k2 estimator 如何构造？
- [ ] RL 阶段的真实 KL coefficient 是 0.1 还是 \(10^{-3}\)？
- [ ] 用直接 divergence 指标而非 SFT steps 衡量时，OPD 开始获益是否存在稳定阈值？
- [ ] 在多随机种子下，表 1 中 0.1–1.0 point 的差异是否显著？
- [ ] 在空间定位、3D/视频推理与开放式视觉问答上是否仍能实现纯文本到视觉的推理迁移？
- [ ] 是否可在不共享 tokenizer 的教师/学生之间，用 top-k logit 映射、隐藏状态或序列级反馈实现同类方法？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v3 全文，包括正文 §1–5、补充材料 §6–7、式 (1)–(6)、表 1–7、图 1–6。
- 新增认识：OPD 的有效性依赖 teacher/student 在 student-visited states 上的预对齐；错误轨迹 masking 是协调教师模仿与 RL 探索的关键设计。
- 修正内容：未将“所有 benchmark 均提升”“蒸馏几乎无额外成本”或“跨模态提升等于视觉表征改善”作为既定事实。
- 下一步：核对公开代码中的 KL 方向、估计器与 \(\beta\)，并阅读 X-Reasoner、KDRL 形成跨论文综合。
