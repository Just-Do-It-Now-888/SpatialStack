# ViGOS

## 基本信息

- 论文标题：Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation
- 作者：Sihan Wang, Xiyao Liu, Lianqing Liu, Zhi Han
- 年份与会议：2026；arXiv 预印本（v2，2026-06-19，未注明正式会议）
- arXiv/DOI：arXiv:2606.19120；DOI: 10.48550/arXiv.2606.19120
- 论文链接：https://arxiv.org/abs/2606.19120
- 项目主页：https://oedosoldier.github.io/ViGOS/
- 本地文件：无
- 阅读状态：已完成（fallback 占比与 PALR 解释有待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

ViGOS 强制学生先生成视觉描述、再推理和回答，并把描述 token 交给 image-only teacher、推理/答案 token 交给 answer-privileged teacher、格式无效 rollout 交给 reference fallback；它没有消除答案特权，而是延后其介入，从而在基本保持 OPSD 平均能力的同时显著提高视觉先验冲突测试中的 image-grounded behavior。

## 核心问题

标准 multimodal OPSD 用同一个、已知 reference answer/solution 的 teacher 监督学生整条 rollout。由于 MLLM 往往比图像更容易利用文本，teacher 可能在模型明确读取视觉证据前就把 reasoning 拉向已知答案，产生“答案兼容但视觉依据薄弱”的 rationale。

论文研究：

1. 如何保留 answer privilege 对推理的帮助，同时阻止它直接监督最早的视觉陈述；
2. 显式“先看图、后推理”的 trajectory segmentation 是否能降低答案驱动的 dense correction；
3. perception、reasoning 和 malformed-output recovery 是否需要不同 teacher context；
4. 该设计能否保持普通 benchmark 上 OPSD 的收益，并在图像与语言先验冲突时更依赖视觉证据。

## 方法

### 整体流程

1. 学生只输入图像 \(I\)、问题 \(x\) 和结构化输出 prompt，生成一条 on-policy trajectory：
   \[
   y=(d,r,a),
   \]
   其中 \(d\) 为 `<description>` 视觉描述，\(r\) 为 `<reasoning>` 推理，\(a\) 为 `\boxed{}` 最终答案。
2. Parser 检查 delimiter、非空 description/reasoning 和可解析 boxed answer，构造 \(\mathcal T_d,\mathcal T_r,\mathcal T_a\) token masks。
3. 若 rollout 格式有效：
   - image-only perception teacher 只监督 \(d\)；
   - privileged reasoning teacher 看到 image、question、reference target，只监督 \(r,a\)。
4. 若格式无效：不用不可靠的 segment masks，改由 full privileged reference teacher 对整条 rollout 提供 reverse-KL fallback。
5. 三个 teacher role 都是同一初始 MLLM 的 frozen detached copy，只是外部 context 不同；所有 teacher 都评分同一 student prefix，不生成 replacement trajectory。
6. 推理时移除 teacher 和 reference target，但学生仍使用结构化 prompt 并生成 description→reasoning→answer。

### 关键模块

**分段 teacher routing。** Student context：

\[
c_{\mathrm{stu}}=(I,x,\pi_{\mathrm{out}}).
\]

Teacher contexts：

\[
c_{\mathrm{img}}=I,\quad
c_{\mathrm{rea}}=(I,x,a^*,\pi_{\mathrm{rea}}),\quad
c_{\mathrm{ref}}=(I,x,a^*,\pi_{\mathrm{out}}).
\]

Image-only 是指没有额外 question、answer options 或 reference target；teacher 仍看到 student-generated prefix \(h_t\)，后者可能间接包含问题相关内容。

**视觉描述作为 grounding interface。** Description 不是额外标注，而是学生生成的中间文本。后续 reasoning teacher 在相同 prefix 上评分，因此可以读取学生先前写出的视觉证据，即使描述不完美。

**格式有效性路由。** Wrong answer 只要格式可解析仍属于 valid rollout；invalid 仅由格式决定，不是 correctness verifier。

**PALR 诊断。** Privileged Answer Leakage Rate 在固定 rollout/prefix 上分别替换 reference answer 和 image，比较 observed token 的 teacher log-prob 变化。它衡量 active teacher correction 中 answer-vs-image sensitivity 的比例，而不是完整 causal attribution。

### 损失函数与训练目标

有效 rollout 的 perception forward KL：

\[
\mathcal L_{\mathrm{perc}}
=\mathbb E\left[
(1-m_{\mathrm{inv}})
\sum_{t\in\mathcal T_d}
D_{\mathrm{KL}}(q_{\mathrm{img},t}\|p_{\theta,t})
\right].
\]

有效 rollout 的 reasoning/answer forward KL：

\[
\mathcal L_{\mathrm{rea}}
=\mathbb E\left[
(1-m_{\mathrm{inv}})
\sum_{t\in\mathcal T_r\cup\mathcal T_a}
D_{\mathrm{KL}}(q_{\mathrm{rea},t}\|p_{\theta,t})
\right].
\]

无效 rollout 的 reference reverse KL：

\[
\mathcal L_{\mathrm{ref}}
=\mathbb E\left[
m_{\mathrm{inv}}
\sum_{t\in\mathcal T_y}
D_{\mathrm{KL}}(p_{\theta,t}\|q_{\mathrm{ref},t})
\right].
\]

总目标：

\[
\mathcal L_{\mathrm{ViGOS}}
=\lambda_{\mathrm{perc}}\mathcal L_{\mathrm{perc}}
+\lambda_{\mathrm{rea}}\mathcal L_{\mathrm{rea}}
+\lambda_{\mathrm{ref}}\mathcal L_{\mathrm{ref}}.
\]

默认 \(\lambda_{\mathrm{perc}}=1,\lambda_{\mathrm{rea}}=1,\lambda_{\mathrm{ref}}=2\)。每个 active KL sum 按对应 segment token 数归一化；KL clipping 为 0.05。

PALR 对 token group \(G\)：

\[
\mathrm{PALR}(G)
=\frac{\sum_{t\in G}s_tc_t^A}
{\sum_{t\in G}s_t(c_t^A+c_t^I)+\epsilon},
\]

其中 \(s_t\) 是 active teacher 与 student 对 observed token 的 log-prob correction magnitude；\(c_t^A\) 使用 wrong-answer teacher mixture，\(c_t^I\) 使用 mismatched-next image。

## 训练数据与训练流程

- 模型：Qwen2.5-VL-3B-Instruct、Qwen2.5-VL-7B-Instruct；self-distillation teacher 与 student 同初始化规模。
- 数据：Vision-SR1-47K，训练 1 epoch。
- 训练：8×A100，effective batch 32，Fused AdamW，LR \(5\times10^{-6}\)，linear scheduler，max grad norm 0.1，bf16，ZeRO-2。
- 参数高效训练：LoRA rank 64，alpha 128，dropout 0.05；target 为 q/k/v/o projection 与 gate/up/down projection。表中 7B target modules 单元格为空，可能是排版合并，`待核对`。
- 长度：max prompt 32768，max completion 4096。
- Rollout：temperature 1.1，top-p 0.95，top-k 20。
- Distillation temperature 1.0，KL clipping 0.05。
- 评测：每题 5 samples，temperature 1.0，top-p 0.9，top-k 20，seed 42；Pass@5 与 Avg@5。

## 实验设置

- 主控对照：相同 Qwen2.5-VL backbone、Vision-SR1-47K 数据和训练预算下的 Baseline、vanilla OPSD、ViGOS
- 参考模型：Visionary-R1-3B、Vision-R1-7B；数据与 recipe 不同，只作参考
- 主 benchmark：MM-Vet、MMMU、MMMU-Pro、MathVerse、MathVista、MMSI、RealWorldQA、CV-Bench
- Shortcut benchmark：ViLP-F、ViLP-P；Score 衡量图像与先验冲突时的视觉答案准确率，Prior 衡量先验一致问题上的保持
- PALR：每规模 1000 个诊断样本；最终 valid rollouts 为 3B 909、7B 919

## 实验结论

1. **相对原始 backbone，ViGOS 提升明显。** 八项平均 Pass@5/Avg@5：3B 从 60.86/27.91 到 71.97/41.35；7B 从 68.13/45.38 到 75.60/50.99。
2. **相对 vanilla OPSD，普通 benchmark 上只是“基本保持”，不是全面优于。** 3B 平均 Pass@5 约低 0.11、Avg@5 高约 0.22；7B Pass@5 高约 0.99、Avg@5 低约 0.13。
3. **逐 benchmark 也有较多下降。** 3B ViGOS 相对 OPSD 在 MM-Vet、MMMU-Pro、MathVerse 及部分 Avg@5 上下降；7B 在 MMMU-Pro/MathVerse/MathVista/MMSI/CV-Bench 的 Avg@5 下降。论文对此的准确说法应是保留主要收益，而非逐项胜出。
4. **Shortcut-sensitive ViLP 上稳定改善。** 相对 OPSD，3B ViLP-F/P Score 为 67.17/66.83→70.17/69.50；7B 为 58.00/57.00→62.67/61.67，同时 Prior 基本保持。
5. **PALR 显著下降。** Reasoning+answer segment 从 17.26%→6.33%（3B）、26.01%→7.56%（7B）；full rollout 从 5.59%→3.07%、7.55%→3.72%。
6. **Description PALR=0 是构造结果。** ViGOS description teacher 不接收答案，所以 \(c_t^A\) 被直接置 0；这验证 routing 生效，但不能证明 description 没有其他 shortcut。
7. **Perception loss 主要影响 prior conflict。** 去掉 perception loss：Overall 74.91→74.81，仅 -0.10；ViLP 69.84→67.58，-2.26。说明普通平均指标对 grounding shortcut 不敏感。
8. **Reasoning loss 的边际主表贡献也小。** 去掉 reasoning loss Overall 74.71，ViLP 69.42；但没有多种子，0.2–0.4 point 差异显著性未知。
9. **Fallback 对格式与 ViLP 很重要。** 去掉独立 reference teacher 后 Overall 73.60、ViLP 63.25；reverse-KL fallback 略高于 forward KL。
10. **Structured prompt 本身贡献很大。** 同 prompt 下，zero-shot 3B Baseline 的平均 Pass@5/Avg@5 已达 71.05/39.83，而普通 prompt baseline 是 60.86/27.91。ViGOS 进一步达到 71.97/41.35，说明主要总增益不能全部归因于训练目标。
11. **Same-prompt OPSD 反而退化。** 同一 description→reasoning prompt 下 OPSD 平均 70.13/37.20，低于 zero-shot baseline；ViGOS Avg@5 在八项均高于同 prompt OPSD，支持“分段 supervision 而非仅格式”解释。

## 局限性

### 论文明确承认

- 学生生成的 visual description 可能不完整或错误，错误会进入后续 reasoning prefix。
- Image-only teacher 可能产生过于通用的描述。
- 训练需要额外 teacher forward passes。

### 由实验设计可直接确认

- 只验证 Qwen2.5-VL 3B/7B 与 Vision-SR1-47K，跨家族和非数学数据泛化未知。
- 推理必须生成 description 和 reasoning，改变输出协议并增加 token 成本；论文没有报告推理 latency、平均长度或相对普通回答的成本。
- 全部 teacher 是同一初始模型的 frozen copy；更强 external teacher、EMA teacher 和跨架构场景未验证。
- 无多训练随机种子、置信区间或显著性检验；多数 ViGOS-vs-OPSD 普通 benchmark 差异很小。
- PALR 使用 Gemini 3.1 Flash-Lite 生成 wrong answers，引入外部闭源诊断组件。
- PALR 的 mismatched-next image 不一定是语义匹配的最优反事实，且只比较 answer/image 两类支持，忽略 question wording、format 和通用语言 prior。
- 训练期间 invalid rollout 比例与随 step 变化没有报告，无法判断 \(\lambda_{\mathrm{ref}}=2\) 的 full privileged fallback 实际占多大训练质量。

### 个人分析

- **ViGOS 延后而非消除 answer privilege。** Reasoning teacher 仍看到完整 reference solution；PALR reasoning-answer 仍为 6.33%/7.56%。方法控制 privilege 进入位置，不保证后续 rationale 不再 answer-compatible。
- **Image-only teacher 并非完全 question-independent。** 它不接收外部问题，但能看到 student prefix \(h_t\)，该 prefix 是学生在问题条件下生成的；后续 description token 可间接携带问题信息。第一批 description token 则可能缺乏任务选择性。
- **Description 是可编辑文本瓶颈。** 若视觉证据无法自然语言化、需要精确坐标/密集关系，强制先写 description 可能丢信息；后续 reasoning 仍可看原图，但 teacher routing 强化了文本中介。
- **PALR full-rollout 下降部分来自机械稀释。** Description token 的 answer sensitivity 被定义为 0，再与 reasoning token 合并，自然会压低 full-rollout ratio；更有价值的是 \(\mathcal T_{ra}\) 仍明显下降。
- **PALR 不是因果 shortcut 完整归因。** Positive-part、observed-token scoring、wrong-answer mixture 与 mismatched image 都会改变数值；它更适合作为受控诊断而非绝对泄漏率。
- **Fallback ablation 有混杂。** “w/o Ref teacher”并非简单让 invalid rollout 无 loss，而是让 perception/reasoning teachers 监督所有 token；结果同时改变 teacher role 和是否训练，不能单独量化 fallback 必要性。
- **Invalid fallback 可能绕过解耦。** 无效 rollout 上 full privileged teacher 监督整条轨迹，恰好恢复 vanilla shortcut 路径。若早期 invalid rate 高，它可能主导训练；论文未报告该比例。
- **普通 benchmark 结果主要证明“不损失太多”。** ViGOS 的核心证据是 ViLP、PALR 和 same-prompt comparison，而不是表 1 对 OPSD 的普遍准确率优势。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可把 trajectory 显式拆成“空间观测描述→空间推理→动作/答案”，并为不同 segment 使用不同 teacher context。
- 感知 segment 的 teacher 不应看到最终答案或完整规划，以免把结果反向泄漏到早期空间状态描述。
- Outcome/reference teacher 可保留在后段推理或仅作格式 recovery，而不是监督所有 token。
- 可构造 PALR 类反事实指标：替换目标答案、图像、局部地图或对象关系，测 dense correction 更依赖哪种信息。

### 个人建议

- 除自然语言 description 外，增加结构化空间 state：对象表、坐标、关系边、可见性和置信度；避免文本瓶颈丢失精确几何。
- 记录 valid/invalid rollout rate、三项 loss token 数和梯度占比；若 invalid 较多，应先用无答案 format teacher 或独立 grammar loss，而不是 full answer teacher。
- 将 PALR 扩展为 answer/image/question/spatial-state 多源 attribution，并使用难负例或语义匹配 image counterfactual。
- 比较“生成 description”与 latent perception token/cue recovery；前者可解释但增加推理 token，后者更高效。
- 对描述事实做 verifier 或与图像反事实 consistency 检查，防止错误描述被后续 reasoning 固化。

## 与其他论文的相同点和冲突

### 与 [ViCuR](vicur.md)

- 两者都识别 answer/rationale privilege 的 train-test mismatch 与 shortcut，并主张先强化 inference-accessible visual evidence。
- ViCuR 用 visual cue 完全替换 answer privilege，并增加 student sink recovery module；ViGOS 保留 answer teacher，但只在 description 之后监督 reasoning/answer。
- ViCuR 的学生不显式生成 cue，输入输出接口基本不变但增加 prefill 参数；ViGOS 无新模型模块，但推理时必须生成 description，增加 token 与延迟。
- ViCuR 的关键风险是 cue provenance 和 sink 是否 question-conditioned；ViGOS 的关键风险是 description 错误传播与 invalid fallback 绕过解耦。

### 与 [Vision-OPD](vision-opd.md)

- 两者都用同 backbone 的不同 teacher context 做 segment/condition-specific supervision。Vision-OPD 以 crop teacher 改善细粒度感知，ViGOS 以 image-only teacher 阻断答案对 description 的直接影响。
- Vision-OPD student 使用带框全图，ViGOS student 使用普通 image+question，但被要求输出显式 description。
- Vision-OPD 采用 EMA/JSD；ViGOS 使用 frozen teachers，valid segment 为 forward KL、invalid fallback 为 reverse KL。

### 与 [VA-OPD](va-opd.md) 和 [VGS](vgs.md)

- ViGOS 通过人工 trajectory segmentation 决定不同 teacher 的作用范围；VA-OPD 通过 visual advantage 自动识别高视觉依赖 token；VGS 通过 distribution decomposition 改变视觉/语言梯度方向。
- ViGOS 的 description mask 可与 VA/VDS 进一步细分，避免 description 中大量通用语言 token 仍稀释视觉监督。
- VGS 的 language-preservation 思路与 ViGOS 相反但互补：ViGOS 限制答案/语言信号过早介入，VGS 在视觉 steering 过强时保护语言能力。

### 与 [VOLD](vold.md)

- 两者都使用 selective teacher guidance：VOLD 按 rollout correctness 屏蔽 KL，ViGOS 按 trajectory segment 与 format validity 路由 teacher。
- VOLD 以 GRPO verifier 确认正确性；ViGOS 不用 reward，wrong answer 仍视为 valid，只控制 supervision source。

### 综合定位

与 [V-Zero](v-zero.md) 相比，ViGOS 保留 reference answer 并按 description/reasoning 阶段路由 teacher；V-Zero 不用答案 label，按 target/random crop support 对 sibling trajectories 加权。前者控制 privilege 进入时间，后者控制不同 rollout 的学习强度。

与 [VCSD](vcsd.md) 相比，VCSD 完全移除 answer/cue privilege，以原图/control contrast 重塑 EMA target；ViGOS 保留答案 teacher 以指导后段推理。VCSD 更简洁，但没有 ViGOS 的显式 perception→reasoning interface。

与 [PCD](pcd.md) 相比，两者都显式拆分 perception/description 与 reasoning。ViGOS 按阶段选择 teacher context；PCD 用多 continuation reward×teacher gap判断 perception 是否应被加强纠正。

与 [VAD](vad.md) 相比，ViGOS 控制 answer privilege 在何时进入；VAD 不使用答案 teacher，控制每个位置 privileged visual correction 中具体哪些方向进入 target。

与 [FP-OPD](fp-opd.md) 相比，ViGOS 决定 privileged signal在哪个阶段进入；FP-OPD 决定该阶段的 teacher correction是否落在 student probe-estimated response geometry内。

与 [SA-OPD](sa-opd.md) 相比，ViGOS控制privilege进入阶段；SA-OPD检查该阶段teacher–student disagreement是否随任务输入变化。

与 [OPD-V](opd-v.md) 相比，ViGOS按perception/reasoning阶段路由teacher；OPD-V不分阶段，按zoom/mask token margin路由监督。

十三篇论文中，ViGOS 新增“时间/阶段路由”维度：除状态对齐、teacher 稳定性、token 稀疏性、梯度方向、privilege recoverability、trajectory evidence、recursive target control、cross-stage credit、correction attribution、student compatibility、input-grounded reliability 和 modality balance 外，还需决定 privileged signal 在推理链的哪个阶段进入。

## 证据

### E-001

- 结论：ViGOS 将 student rollout 强制拆为 description、reasoning、answer，并用三个 frozen teacher roles 按 segment/validity 路由。
- 类型：论文结论
- 定位：§3.1–3.3；图 3；式 (12)–(21)
- 必要引用：valid 时 image teacher→description、privileged teacher→reasoning/answer；invalid 时 reference teacher→全序列。
- 备注：Wrong answer 只要格式有效仍走 segment teachers。

### E-002

- 结论：Image-only perception teacher 不接收外部 question、options 或 reference target。
- 类型：论文结论
- 定位：§3.1；式 (13)–(15)；附录 E.1
- 必要引用：无
- 备注：它仍看到 student prefix，因此不是对问题信息绝对独立。

### E-003

- 结论：PALR 显示 ViGOS 的 reasoning-answer correction 更少依赖 privileged answer。
- 类型：论文结论
- 定位：§2.3、§3.4；式 (7)–(10)；图 2；附录 A、表 A.I
- 必要引用：3B 17.26%→6.33%；7B 26.01%→7.56%。
- 备注：PALR 是特定 counterfactual diagnostic，不是完整 causal attribution。

### E-004

- 结论：ViGOS 相对 backbone 显著提升，但相对 OPSD 的普通 benchmark 平均性能基本持平。
- 类型：论文结论
- 定位：§4.2；表 1
- 必要引用：3B 对 OPSD Pass/Avg 约 -0.11/+0.22；7B 约 +0.99/-0.13。
- 备注：主张是保持 OPSD gains，不是全面超过 OPSD。

### E-005

- 结论：ViGOS 在四个 ViLP Score 设置均高于 OPSD，并基本保持 Prior。
- 类型：论文结论
- 定位：§4.3；表 2；图 4
- 必要引用：3B Score +3.0/+2.67；7B +4.67/+4.67。
- 备注：这是论文最直接的 shortcut-resilience 证据。

### E-006

- 结论：Perception loss、reasoning loss 和 reference fallback 的 full setting 在报告消融中最好。
- 类型：论文结论
- 定位：§4.4；表 3
- 必要引用：去 perception 时 ViLP -2.26；去独立 ref teacher 时 ViLP -6.59。
- 备注：前两项 Overall 差异小；ref ablation 改为 segment teacher 全序列监督，存在混杂。

### E-007

- 结论：Structured prompt 自身解释了相当部分 image-grounded benchmark 增益，但不能解释 ViGOS 相对同 prompt OPSD 的优势。
- 类型：论文结论
- 定位：附录 C；表 A.II
- 必要引用：同 prompt Baseline 71.05/39.83，OPSD 70.13/37.20，ViGOS 71.97/41.35。
- 备注：普通 prompt baseline 为 60.86/27.91。

### E-008

- 结论：Description segment 的 PALR=0 是 metric construction 的直接结果。
- 类型：个人核对
- 定位：§2.3 式 (8)；附录 A、表 A.I
- 必要引用：active teacher 不接收 \(a^*\) 时 \(c_t^A=0\)。
- 备注：不能由此推出所有 description token 都视觉 grounded。

### E-009

- 结论：论文未报告 invalid rollout 比例和 reference fallback 的实际训练占比。
- 类型：个人核对
- 定位：§3.2–3.3；式 (17)、(20)–(21)；附录 E.2
- 必要引用：invalid 时 \(\lambda_{\mathrm{ref}}=2\) 的 full privileged teacher 监督所有 token。
- 备注：高 invalid rate 可能绕过 segment decoupling，`待核对`。

### E-010

- 结论：ViGOS 改变推理输出协议并可能增加显著 inference token 成本。
- 类型：个人核对
- 定位：§3.1；附录 E.1、E.3
- 必要引用：必须输出 description、reasoning 和 answer，总长度上限 4096。
- 备注：论文未报告 latency 或输出长度对照。

## 待验证问题

- [ ] 每个训练 step/epoch 的 valid rollout rate、三项 loss token 占比和梯度占比是多少？
- [ ] 用无答案 grammar teacher 替代 full privileged fallback 是否可避免 invalid 分支 shortcut？
- [ ] Image-only teacher 的描述是否 task-relevant，尤其第一批 description token？
- [ ] Description factuality 如何验证，错误描述对后续 reasoning 的传播率是多少？
- [ ] PALR 对 wrong-answer generator、wrong-answer 数量与 image counterfactual 选择是否稳健？
- [ ] 用 latent perception token 或结构化 scene graph 替代自然语言 description 是否更准确高效？
- [ ] ViGOS 的推理 token 数、wall-clock latency 与显存开销是多少？
- [ ] 多训练随机种子下 ViGOS 与 OPSD 的小幅主 benchmark 差异是否显著？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v2 全文，正文 §1–5、式 (1)–(22)、表 1–3、图 1–4；附录 A–E、式 (A1)–(A8)、表 A.I–A.V。
- 新增认识：答案 privilege 不必完全移除，也可通过 trajectory stage routing 延后介入；普通平均 benchmark 可能对 grounding shortcut 不敏感，需 prior-conflict 与 counterfactual diagnostics。
- 修正内容：未把 PALR 当作完整因果泄漏率，未把 ViGOS 描述为全面超过 OPSD，并区分 structured prompt 收益、segment supervision 收益与 fallback 风险。
- 下一步：统计 invalid routing 占比，复现 PALR sensitivity，并比较显式 description、visual cue recovery 与 latent spatial state 三种 grounding interface。
