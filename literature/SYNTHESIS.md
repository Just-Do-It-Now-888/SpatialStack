# Vision-OPD 跨论文综合

此文件只记录经过多篇论文对照后形成的综合认识。单篇论文内容保存在 `papers/`。

## 问题定义与研究目标

当前十四条 OPD 路线都试图把“训练时可用、部署时不可得或代价较高的条件”内化到学生的标准推理策略：

- [OPSD](papers/opsd.md)（E-001、E-002）是文本侧源头：同一 LLM，学生只看题目，教师额外看 ground-truth solution，在学生 on-policy 轨迹上做 token 级 JSD。

- [VOLD](papers/vold.md)（E-001、E-002）用更强文本 LLM 的推理分布，提升 VLM 共享语言解码器的推理能力。
- [Vision-OPD](papers/vision-opd.md)（E-001、E-002）用同一 MLLM 在证据裁剪视图下的优势，提升其带框全图条件下的细粒度感知。
- [VA-OPD](papers/va-opd.md)（E-001–E-004）用视觉教师在原图/细节退化图上的反事实 log-prob 差，识别并放大 student rollout 中稀疏的视觉关键 token 监督。
- [VGS](papers/vgs.md)（E-001–E-004）把标准 OPD 分解为语言先验与视觉信息增益目标，通过改变梯度方向优先修复视觉 grounding，并保持更新范数与语言能力。
- [ViCuR](papers/vicur.md)（E-001–E-003）用来自推理输入的 visual cue 替代答案/rationale privilege，并用 sink-token cross-attention 建立学生侧证据恢复通道。
- [ViGOS](papers/vigos.md)（E-001–E-003）保留 answer privilege，但将其延后到显式视觉描述之后，并按 trajectory segment 路由 image-only、reasoning 与 fallback teacher。
- [V-Zero](papers/v-zero.md)（E-002–E-004）不用文本答案 label，以目标 crop/随机负 crop 的 teacher support 差对 sibling rollouts 做组内相对 evidence gating，再蒸馏 positive-crop teacher。
- [VCSD](papers/vcsd.md)（E-001–E-003）不使用外部 teacher 或辅助 privilege，以 EMA teacher 的原图/content-erased full-vocabulary contrast 在 plausible support 内重塑自蒸馏 target。
- [PCD](papers/pcd.md)（E-001–E-005）把 response 分成 perception/reasoning，以共享 perception 的多 continuation reward 与 teacher disagreement 共同判断 perception 是否是当前可纠正的失败阶段。
- [VAD](papers/vad.md)（E-001–E-004）将完整 privileged-teacher correction 投影到 evidence-present/degraded intervention direction，并围绕 current student 重构 signed support/refutation target。
- [FP-OPD](papers/fp-opd.md)（E-002–E-004）用 student visual feature perturbations估计低秩 output-response span，在 reverse-KL Fisher metric下投影 teacher correction并重构 student-anchored target。
- [SA-OPD](papers/sa-opd.md)（E-001–E-004）比较完整输入/no-prompt下的 sampled teacher–student divergence，过滤输入敏感性低但 optimization impact高的 token positions。
- [OPD-V](papers/opd-v.md)（E-001–E-004）用zoom/masked-crop EMA teachers的正sampled-token margin定义trust region，并在其中加权蒸馏zoom-teacher JSD target。

共同目标不是简单复制教师答案，而是在学生实际会访问的生成状态上获得密集 token 级指导，减少 off-policy SFT 的 prefix/state distribution mismatch。

## 方法分类

### 按教师来源

- **同模型答案特权教师：** OPSD 的 teacher/student 共享同一 LLM；教师额外读取 reference solution，学生不读，非答案输入相同。
- **跨模型能力教师：** VOLD 的教师为固定 Qwen3-8B，学生为 Qwen2.5-VL-3B；先通过同教师轨迹 SFT 缩小策略差距。
- **同模型特权条件教师：** Vision-OPD 的 teacher/student 从同一 Qwen3.5 checkpoint 出发，分别看 crop 与全图；通过 frozen/EMA target 保持教师稳定。
- **跨规模视觉教师：** VA-OPD 使用 Qwen3-VL-4B/8B/32B 教师指导 2B 学生，并额外用退化图 teacher pass 估计 token 视觉依赖。
- **跨规模视觉教师与模态分解：** VGS 使用 GRPO 后训练的 Qwen3-VL-8B 教师指导 2B/4B 学生，以有图/text-only distribution 差分离语言与视觉目标。
- **可恢复 privilege 教师：** ViCuR 同时覆盖同 backbone OPSD 和更强教师 OPD；teacher 额外接收 visual cue，student 只接收标准图文输入。
- **分阶段多角色教师：** ViGOS 的三个 teacher role 来自同一 frozen backbone，分别使用 image-only、answer-privileged reasoning 和 full reference context。
- **跨规模对比证据教师：** V-Zero 用 Qwen3.5-27B teacher 指导 4B student，并对每条 rollout 分别执行 positive/negative crop replay。
- **输入对比 EMA 自教师：** VCSD 的 teacher 是 student EMA，在相同 prefix 上分别看原图和同尺寸内容擦除图，不接收 answer/crop/cue。
- **跨规模感知纠错教师：** PCD 用 Qwen3-VL 8B→2B、32B→8B，并只在 perception span 使用 teacher top-\(k\) supervision；reasoning 由 DAPO reward 优化。
- **固定同规模 attribution 教师：** VAD 的 teacher 是 Qwen3.5-4B/9B 初始 checkpoint 的 frozen copy，分别看清晰 crop 与同区域退化 crop。
- **跨规模 capacity teacher：** FP-OPD 使用 Qwen3-VL 8B/8B-GRPO→2B 与 32B→8B，teacher/student 看相同原图，额外 probes只作用于 student。
- **跨规模 reliability teacher：** SA-OPD 在 Qwen3/Qwen3.5 LLM与VLM跨规模 teacher/student上测试，通过额外 no-prompt scoring审计 dense supervision。
- **双视图 EMA modality teacher：** OPD-V 的positive/negative teachers共享EMA参数，分别看evidence crop与其随机矩形mask版本；student看带cue原图。

### 按特权信息

- **答案/题解特权：** OPSD 的教师看到 ground-truth solution，学生只看到问题；两边非答案 token 相同。
- **语言推理特权：** VOLD 的教师规模更大、文本推理能力更强。
- **视觉证据特权：** Vision-OPD 的教师看到隔离且放大 2 倍的 evidence-centered crop。
- **视觉反事实特权：** VA-OPD 的教师同时比较原图与细节退化图，从预测差异中提供视觉依赖度，而原图 teacher distribution 仍作为 KL target。
- **视觉信息增益特权：** VGS 用教师“有图 logits−text-only logits”构造 visual target，并保留学生自身 text-only language prior。
- **视觉 cue 特权：** ViCuR 用文字描述问题相关视觉证据，强调 cue 的来源应包含在推理输入中，而不是直接泄露答案或 reasoning shortcut。
- **阶段受限答案特权：** ViGOS 不替换 reference answer，而是禁止它直接监督 description token，只允许其进入后续 reasoning/answer 或 invalid-format fallback。
- **区域对比特权：** V-Zero 的 positive crop 指向问题相关区域，negative crop 从目标区域外随机采样；不使用 textual answer，但依赖 region-level supervision。
- **无辅助 privilege 的条件对比：** VCSD 只从原始 image-question 构造 black/noise/blur/no-image control，以 intervention contrast 提供 target asymmetry。
- **无感知标签但有答案 verifier：** PCD 不需要 perception ground truth，却依赖 Geo3K 可验证最终答案估计 PSR，不能归为 answer-label-free。
- **区域干预归因特权：** VAD 依赖 marked full image、2× relevant crop 与 0.1× degraded crop，估计特定 intervention 可解释的 correction direction。
- **无额外 privilege 的 student geometry：** FP-OPD 不需要 crop/cue/答案，将 student feature perturbation responses当作当前视觉可达空间 proxy。
- **Residual-context intervention：** SA-OPD 删除 task input但保留 student response prefix，以 disagreement变化作为 input-groundedness proxy；无需外部标签，但 control可能 OOD且prefix可泄露输入。
- **Modality-balance intervention：** OPD-V 用zoom/mask view改变视觉证据突出程度，并以attention-ratio correlation解释其modality balance；训练实际只使用log-prob contrast。

### 按优化组合

- **RL + selective OPD：** VOLD 复用同一 student rollout，同时计算 GRPO 与 token KL，并只对 reward=0 的错误轨迹蒸馏。
- **纯分布匹配 OPD：** Vision-OPD 不使用 reward，直接对所有 student prefixes 做 top-\(K\) JSD。
- **视觉依赖加权 OPD：** VA-OPD 不使用 reward，按轨迹平均 VA 重加权 rollout，并将 high/low-VA token 分组归一化后计算 reverse KL。
- **模态梯度 steering：** VGS 在 standard reverse-KL 上增加 visual target loss，用固定范数缩放改变更新方向，并在 top-VDS token 上加入 language preservation；亦可作为 GRPO regularizer。
- **Privilege redesign + recovery architecture：** ViCuR 使用 sampled-token PPO-style OPD，并在 student prefill 中增加 sink-to-visual cross-attention，不增加 autoregressive branch。
- **Trajectory segmentation + teacher routing：** ViGOS 强制 description→reasoning→answer 格式；valid segment 使用 forward KL，invalid rollout 使用 full privileged reverse-KL fallback。
- **Sibling-relative evidence gating：** V-Zero 对同 prompt 的 \(G=8\) rollouts 做 positive-negative evidence score z-normalization，以 \([0,2]\) gate 加权 positive-view sampled reverse KL。
- **Vocabulary-level target shaping：** VCSD 在原图 plausible support 内按 \((p^J)^{1+\alpha}/(p^0)^\alpha\) 重塑 EMA teacher target，并用 full-distribution forward KL。
- **Stage-specific credit assignment：** PCD 用 \(d=(1-\mathrm{PSR})\widetilde{\mathrm{KL}}\) 给 perception trajectory 分配 mean-normalized weight，reasoning objective 保持 DAPO 形式。
- **Candidate-level correction reconstruction：** VAD 在 student top-\(K\)+tail 坐标中做 one-sided projection、support/refutation budget，再以 JSD 蒸馏 student-anchored target。
- **Fisher-projected target：** FP-OPD 在 full vocabulary 中用 \(K=4\) probes构造 Fisher Gram system，以 projected gap替换完整 teacher gap并继续优化 reverse KL。
- **Token deletion mask：** SA-OPD 用低 \(\Delta^{IG}\) 与高 \(|A^{full}|\) 的 quantile intersection删除位置，并在 retained positions继续做full-vocabulary reverse KL。
- **Positive-margin trust region：** OPD-V 同时用 \(\delta^{MB}>0\)作hard gate、用margin作soft weight，target是positive zoom teacher top-100+tail JSD。

## 数据构造与监督信号

- OPSD 使用数学题的 reference solution 作为教师上下文，学生 prompt 不含该 solution；蒸馏在学生 on-policy completion 上进行（[OPSD](papers/opsd.md)，E-001、E-002）。公开评测为 AIME/HMMT，不是空间多视图。
- VOLD 使用大规模纯文本推理 prompt/轨迹：约 350K MoT 教师轨迹用于 SFT，orz-57k 数学题及 exact-match answer 用于 RL。监督同时包含序列级二值 reward 和外部教师 token 分布（[VOLD](papers/vold.md)，E-001、E-003）。
- Vision-OPD 使用 6.2K 个全图—crop—问题三元组。主方法不消费答案标签，但依赖对象检测/分割区域、bounding box、crop，以及 Qwen3.5-397B 生成问题（[Vision-OPD](papers/vision-opd.md)，E-003、E-010）。
- VA-OPD 使用 Geometry3K/ViRL39K 图文数学题，不需要额外标注或 reward；监督由视觉教师的原图 KL target 与原图/退化图反事实 VA 共同构成（[VA-OPD](papers/va-opd.md)，E-001、E-004）。
- VGS 使用 Vision-SR1-47K；先在同一数据上把 8B teacher 做 2 epochs GRPO，再对 2B/4B student 做 1 epoch OPD。监督包含教师原始图文分布、text-only 分布及二者构成的 visual target（[VGS](papers/vgs.md)，E-001、E-005）。
- ViCuR 使用 Vision R1 与 Geometry3K；teacher-side cue 是视觉证据文字，student 不接收 cue text。论文未公开 cue generator、prompt、答案可见性和过滤流程，因此“recoverable”仍需 provenance audit（[ViCuR](papers/vicur.md)，E-001、E-009）。
- ViGOS 使用 Vision-SR1-47K 的 reference answer/solution；description 由 student on-policy 生成而非离线标注。Image-only teacher 不接收外部 question/answer，但可读取 student prefix；reasoning teacher 读取完整 reference target（[ViGOS](papers/vigos.md)，E-001、E-002）。
- V-Zero 使用 ZwZ 的 23K full-image/question/target-crop 数据，并额外生成随机负 crop。它无 textual answer label，但 positive crop 的构造 provenance 未说明，因此不是 annotation-free（[V-Zero](papers/v-zero.md)，E-004、E-009）。
- VCSD 使用 ViRL39K，但训练目标只消费 image-question；answer-hint OPSD baseline 才读取 reference answer。监督来自 same-model 原图/control distribution difference，无额外 region 标注（[VCSD](papers/vcsd.md)，E-001、E-006）。
- PCD 使用 Geo3K 的 answer verifier 和强教师，不消费 perception label。每个 prompt 生成 2 个 perceptions、每个 perception 共享 4 个 reasoning continuations，以最终 reward 估计 PSR（[PCD](papers/pcd.md)，E-001、E-009）。
- VAD 复用 Vision-OPD 的 6,241 条 marked-full-image/crop 数据，并构造同 crop 的细节退化 view；无需答案 reward，但依赖 region-level privilege（[VAD](papers/vad.md)，E-001、E-005）。
- FP-OPD 使用 Geo3K 原始图文训练，无 region/answer privilege；teacher correction 来自更大模型，capacity filter来自 student top/bottom/left/right visual-feature probes（[FP-OPD](papers/fp-opd.md)，E-002、E-005）。
- SA-OPD 的 VLM数据来自 VERO-600K 视觉理解子集与 MMRL30k visual reasoning子集；LLM用约7K DeepMath。它不需标签型 privilege，但每个 rollout需 full/no-prompt双条件评分（[SA-OPD](papers/sa-opd.md)，E-001、E-008）。
- OPD-V 再次使用Vision-OPD 6,241条数据中的bbox crop，在crop上随机mask构造negative view；verified target是否实际进入method teacher prompt不清楚（[OPD-V](papers/opd-v.md)，E-001、E-011）。

因此，“label-free”应细分：Vision-OPD 无需**答案标签与 reward verifier**，但并非没有数据构造监督；其 privileged crop 和空间框本身是区域级信息。

## Reward、Advantage 与优化目标

- VOLD 使用 GRPO group-normalized advantage，并以 \((1-r(\tau))\) mask 控制教师 KL，只纠正失败轨迹。该设计允许学生保留不同于教师但已被 verifier 判定正确的路径（[VOLD](papers/vold.md)，E-003）。
- Vision-OPD 没有 reward/advantage。主目标是 student rollout 上的 top-100 \(\mathrm{JSD}_{0.5}\)，带 tail probability；每个 token 都有 dense gradient（[Vision-OPD](papers/vision-opd.md)，E-004、E-008）。
- VA-OPD 的 visual advantage 不是 RL advantage，而是 teacher-scored counterfactual sensitivity；它按“是否依赖视觉细节”分配梯度，不评价答案是否正确。默认 top 20% token 与其余 token 各占一半组级 loss，使单个 high-VA token 权重为 low-VA token 的 4 倍（[VA-OPD](papers/va-opd.md)，E-001、E-004）。
- VGS 的 VDS 是教师有图与 text-only full distribution 的 forward KL；visual target 则保留学生 language prior、注入教师 visual information gain。它不只重加权 loss，而是显式把更新方向转向 visual gradient（[VGS](papers/vgs.md)，E-001–E-004）。
- ViCuR 的 sampled-token advantage 是 cue-conditioned teacher 与 student 对已采 token 的 log-prob 差；与 VA/VDS 不同，它衡量模仿差距而非视觉依赖度。优势 stop-gradient，实际使用 importance ratio 与 clipping（[ViCuR](papers/vicur.md)，E-003）。
- ViGOS 不使用 reward 或 sampled-token advantage。它按格式有效性与 token segment 选择 teacher：description/reasoning 用 forward KL，invalid 全序列用 reverse KL；因此选择轴是**推理阶段**而非 correctness 或视觉依赖度（[ViGOS](papers/vigos.md)，E-001）。
- V-Zero 的 evidence advantage 也不是 RL correctness advantage：它对 positive/negative crop 的 sampled-token teacher log-prob 差做 trajectory 平均与 sibling z-score，只表示**组内相对视觉证据支持**（[V-Zero](papers/v-zero.md)，E-002、E-008）。
- VCSD 不对 rollout 或 sampled token 给标量 advantage，而把每个 candidate 的 conditional log-ratio 当作 implicit visual reward，解析地构造 support-restricted target。该 reward 仍衡量 image sensitivity，不保证 correctness（[VCSD](papers/vcsd.md)，E-002、E-004、E-005）。
- PCD 同时使用 correctness 与 teacher gap，但只把 correctness 作为 downstream failure witness；低 PSR 本身不能区分感知不足与 reasoning difficulty，乘积 gate 也只是未校准的 conservative surrogate（[PCD](papers/pcd.md)，E-002、E-003）。
- VAD 不产生 scalar advantage，而把 teacher-to-student correction 在 same-teacher intervention response 上投影；signed direction同时处理视觉支持与 refutation，但只在所选坐标和 intervention 下有意义（[VAD](papers/vad.md)，E-001–E-003）。
- FP-OPD 同样不使用 advantage。它在 student-Fisher geometry下过滤 teacher correction；该 filter衡量 probe-induced input sensitivity，而不是 outcome correctness或参数更新可训练性（[FP-OPD](papers/fp-opd.md)，E-003、E-010）。
- SA-OPD 用 sampled-token log-ratio的绝对值近似 optimization impact，并以其 full/no-prompt差近似 groundedness；实际 loss仍是位置级full-vocabulary KL，因此 proxy与优化量并不完全同粒度（[SA-OPD](papers/sa-opd.md)，E-001、E-011）。
- OPD-V 用positive/negative teacher对sampled token的log-prob差作为selection/weight；负margin被丢弃，正margin位置匹配完整positive distribution（[OPD-V](papers/opd-v.md)，E-002、E-003）。
- 两者都说明只在 sampled token 上给标量信号不一定充分：Vision-OPD 的 top-\(K\) logits 比 sampled-token policy-gradient objective 平均高 1.06 点；VOLD 则用 teacher token distribution 补足 outcome reward 的稀疏性。

需注意两篇论文均存在 divergence 复现疑点：VOLD 的公式方向与 student-sampled k2 描述可能不匹配；Vision-OPD 的式 (3)/(4) 与算法 1 参数顺序相反（[VOLD](papers/vold.md)，E-009；[Vision-OPD](papers/vision-opd.md)，E-012）。

## 训练流程与稳定性

十三篇论文揭示了互补的有效性与稳定性条件：

1. **状态空间对齐：** VOLD 中，若学生轨迹对固定教师而言太偏离分布，teacher guidance 几乎无收益；来自同一教师的 SFT cold start 是 OPD 生效前提（[VOLD](papers/vold.md)，E-002）。
2. **目标时间稳定性：** Vision-OPD 中，teacher/student 初始参数相同且只改变视觉条件，但若 current policy 同步充当教师会崩溃；frozen、trust-region 或 EMA teacher 是必要的（[Vision-OPD](papers/vision-opd.md)，E-007）。
3. **任务相关梯度密度：** VA-OPD 中约 top 10% token 承载 93% VA mass；若 uniform KL 让关键 token 被语言脚手架稀释，准确率可提高而视觉依赖几乎不变（[VA-OPD](papers/va-opd.md)，E-002、E-003、E-008）。
4. **模态梯度方向：** VGS 中语言与视觉梯度随 VDS 增大趋向正交甚至轻微冲突；仅增加视觉梯度会造成语言 unlearning，需范数归一化与 LP regularizer（[VGS](papers/vgs.md)，E-002–E-004）。
5. **Privilege recoverability：** ViCuR 表明 answer-based OPSD 可低于 base 且出现 hint pattern 泄漏；visual cue 能缩小该退化，但现实 cue 是否完全来自推理输入取决于生成流程（[ViCuR](papers/vicur.md)，E-004、E-008、E-009）。
6. **Privilege intervention stage：** ViGOS 表明即使保留答案教师，也可通过“先描述视觉、后接收答案指导”降低 prior-conflict error；但 invalid rollout 上 full privileged fallback 可能绕过该解耦，其实际占比未报告（[ViGOS](papers/vigos.md)，E-003、E-005、E-009）。
7. **Trajectory-level evidence discrimination：** V-Zero 用 sibling-relative crop support 弥补 uniform OPD 的轨迹等权问题；但 relative evidence 不等于 absolute correctness，all-wrong group 仍会产生高权重轨迹（[V-Zero](papers/v-zero.md)，E-002、E-008）。
8. **Recursive target support：** VCSD 表明 EMA target 反复重塑时，unrestricted likelihood ratio 会积累 distortion；原图 plausible support、适中 contrast 和 anchor 用于限制自强化漂移（[VCSD](papers/vcsd.md)，E-002、E-008、E-011）。
9. **Cross-stage credit assignment：** PCD 表明 final failure 不能直接归因于 perception；需结合多 reasoning outcomes 与 teacher disagreement，但 shared teacher-student blindness 和错误教师仍不可识别（[PCD](papers/pcd.md)，E-002、E-003）。
10. **Correction-source attribution：** VAD 表明 privileged teacher correction 可先按 intervention-aligned direction重构再蒸馏；但单 view-pair projection 只产生 semantic enrichment，不构成 identifiable decomposition（[VAD](papers/vad.md)，E-002、E-009）。
11. **Student-local response compatibility：** FP-OPD 表明完整 teacher target 未必最易拟合；Fisher projection可过滤 probe space外 correction，但 input tangent不等同于 parameter trainability（[FP-OPD](papers/fp-opd.md)，E-001、E-010）。
12. **Input-grounded supervision reliability：** SA-OPD 表明高 divergence/learnability仍可能对应输入不敏感的监督；但 disagreement sensitivity不等于teacher自身grounding（[SA-OPD](papers/sa-opd.md)，E-001、E-009）。
13. **Modality-balance trust region：** OPD-V 表明zoom/mask正margin可作为有效token gate，但实证识别的是所选视觉operations的敏感性，而非visual/text因果贡献（[OPD-V](papers/opd-v.md)，E-002、E-010）。

综合来看，OPD 需要同时控制 student 相对 teacher competence region 的**状态漂移**、teacher target 的**时间漂移**、有限梯度被无关 token 稀释的**监督密度**、语言/视觉目标的**梯度方向冲突**、训练特权相对部署输入的**可恢复性**、privileged signal 进入推理链的**阶段与路由**、rollout 的**绝对正确性与相对证据支持**、自教师 target 的**递归 sharpening 与 support 漂移**、最终 failure 在 perception/reasoning 间的**跨阶段归因**、teacher correction 的**来源归因与 target reconstruction**、correction 相对 student 当前 response geometry 的**局部兼容性**、teacher disagreement对当前输入的**grounded reliability**，以及视觉证据相对文本先验的**模态分配**。

## 评测任务与指标

- VOLD 主要评测数学、几何、逻辑视觉推理，并补充 perception/reasoning 子项，适合观察纯文本推理后训练是否迁移到视觉任务以及是否损伤感知。
- Vision-OPD 主要评测小目标、缩放层级、高分辨率和真实场景细节，并以 MMVP、CV-Bench、MMStar、POPE 检查 holdout 能力。
- VA-OPD 评测视觉数学、hallucination、图解理解与 OCR，并直接跟踪 teacher-scored rollout VA；但该 VA 仍不是学生内部 grounding 的直接测量。
- VGS 评测视觉数学、逻辑、视觉谜题与 text-only holdout，并同时分析梯度角度、visual/language loss 和生成长度。
- ViCuR 评测视觉数学、MMMU 与 Video-MME，并按 in/near/out-of-domain 划分；还报告 hint leakage、prefill/decoding latency 与参数增量。
- ViGOS 评测八项通用/学术/数学/空间 benchmark，并用 ViLP 的图像—语言先验冲突、PALR 反事实诊断和 same-prompt control 区分 grounding、答案泄漏与结构化 prompt 收益。
- V-Zero 聚焦 VStar、HR-Bench、ZoomBench、MME-RealWorld 的细粒度/高分辨率感知，并以 MMStar 检查 OOD；它报告 wall-clock，但没有 grounding intervention 或 gate-correctness correlation。
- VCSD 覆盖 BLINK、MMStar、MathVista、V*Bench、HRBench4K/8K、HallusionBench，并跨 Qwen3-VL/Qwen3.5 六个规模验证 aggregate；但 grounding 证据主要是单案例与 model-internal contrast visualization。
- PCD 以 Geo3K、四项视觉数学 Near-OOD 和 LogicVista/MMMU-Pro/MMStar OOD 做 Avg@8；它明确报告 2B OOD mean 略降，并使用 held-out validation 选 checkpoint。
- VAD 使用 VStar、ZoomBench、HRBench、MME-RealWorld 主评测与 MMVP/CV-Bench/MMStar/POPE held-out；主评测依赖 GPT-OSS-120B judge，并提供 token semantics 与 offline target effect。
- FP-OPD 使用四项视觉数学与 MMMU/HallusionBench/MMStar，并在 32B→8B 同报 greedy/Avg@8；但没有训练成本与多 seed。
- SA-OPD 同时评测视觉理解、视觉推理与纯文本数学，并报告2.64%–7.53%额外wall-clock；但无多seed。
- OPD-V使用V*/Zoom/HR/MME-RealWorld六项等权平均并报告4B/9B step latency；judge身份未报告，v2 HTML主表转换失败但逐项值可由v1表核对。
- 十三者都缺少多训练随机种子和完整显著性检验，0.1–1.0 point 的差异不能作为稳定优势。
- 当前证据仍未覆盖无框自主定位、3D/视频空间推理、交互式规划等 SpatialStack 关键场景。

## 共同结论

1. 在 student-generated prefixes 上进行 token 级指导，比只模仿 teacher-generated trajectories 更贴近部署状态分布。
2. 教师不一定需要在所有方面更强；只要在某个训练时条件下具有可靠优势，就能形成 privileged supervision。优势可以来自更强模型，也可以来自同模型的更易视觉输入。
3. Dense OPD 并非自动稳定或有效：教师要在 student states 上有可靠分布，且 target 更新不能与学生无约束共同漂移。
4. 最可信的结论来自同 backbone、同数据的训练策略对照，而非跨模型 SOTA 表。
5. 能力提升可能伴随边界条件：VOLD 出现部分 perception regression；Vision-OPD 的提升依赖带框全图，尚未证明无提示自主定位。
6. Token 监督价值高度不均匀；应依据 correctness、teacher confidence、视觉依赖或空间约束满足度做 masking、gating 或分组归一化，而不是默认每个 token 等权。
7. 任务准确率提升不自动等于视觉 grounding 增强；需要感知分项、特权—部署 gap、student counterfactual sensitivity 等独立诊断。
8. “选对监督位置”和“选对更新方向”是两个独立层级：VA-OPD 证明 token/rollout 重要性不均，VGS 进一步证明视觉与语言目标可近乎正交。
9. 增强视觉目标需要配套语言保持或梯度约束；否则高视觉依赖位置可能产生 destructive interference。
10. Teacher privilege 的信息来源与 teacher strength 同样重要；answer/rationale privilege 可能产生不可恢复目标和格式泄漏，而视觉 privilege 也必须审计其生成 provenance。
11. “输入接口不变”不等于零部署成本：student architecture、prefill latency 与参数量仍可能变化。
12. Privilege 不只有“是否使用”的选择，还需控制“何时使用”：先形成受视觉约束的中间状态，再引入答案指导，可降低 early shortcut，但中间状态错误和 fallback 路径也需审计。
13. 显式“先描述再推理”prompt 本身可带来很大收益；训练方法比较必须使用相同输出格式，避免把 prompt effect 归因于 distillation objective。
14. Answer-label-free 不等于 supervision-free：目标 crop、box、cue、对象图仍是区域或结构标签，必须报告 provenance、构造成本和部署可恢复性。
15. Visual evidence advantage 与 correctness 是独立轴；相对证据支持可以筛出 grounded rollout，但不能保证结论正确，尤其在 all-wrong sibling group 中。
16. Teacher/student 不对称不一定需要额外 privilege；受控输入 intervention 也能形成 target asymmetry，但其有效性取决于 control 是否近似“移除目标信息而不引入新分布偏移”。
17. 将 contrast 写入 target distribution 比 scalar weighting 更密集，但递归 self-distillation 必须限制 support 与 shaping strength，防止极低概率 token 或语言漂移被持续放大。
18. Outcome failure 不能直接定位感知错误；同一 perception 下多 reasoning rollouts 只能估计当前 reasoner 的 downstream value，还需独立感知 witness。
19. “Label-free”必须说明 label 类型：无 perception annotation 不代表无 answer/verifier，避免把 PCD 与 VCSD/V-Zero 的无文本答案训练混为一谈。
20. Visual contrast 的更细用途是重构 correction direction，而不只选择监督位置或强度；但“视觉归因”必须说明 candidate support、projection geometry 与 intervention basis。
21. Signed refutation 应按 relative odds 与 absolute probability 分开验证；负 logit shift 经 softmax normalization 后不必然降低绝对概率。
22. Teacher correction 是否“适配 student”应区分三种概念：当前 input sensitivity、参数更新可训练性、有限预算内 fitability；三者不能互换。
23. Fisher metric提供与 KL 一致的局部几何，但 projection basis仍决定保留什么；好的 metric不能弥补不完整 probes。
24. Input-groundedness需区分 teacher input effect、student input effect与二者 disagreement effect；只看最后一项会隐藏 cancellation。
25. Residual/no-input control必须审计 OOD shift与prefix leakage；保留的response prefix本身可能恢复已删除任务信息。
26. Attention ratio与logit intervention sensitivity都不能自动解释为modality causal contribution；需要干预或attribution验证。
27. Token margin同时做gate和weight时，必须控制selected fraction、weight scale与loss denominator，否则effective learning rate随batch漂移。

## 分歧与适用条件

- **是否需要 verifier：** VOLD 需要 exact-match reward，并利用 reward mask 协调探索与模仿；Vision-OPD 不需要答案或 verifier，但要求 crop teacher 在所有训练 token 上足够可靠。
- **是否需要预对齐：** VOLD 的跨模型 teacher/student 需要 SFT alignment；Vision-OPD 参数同源，无需 SFT，但需要 frozen/EMA teacher regularization。
- **训练能力位置：** VOLD 冻结视觉塔，改善主要位于语言侧推理；Vision-OPD 直接优化不同视觉条件下的行为，但哪些模块更新尚待代码核对。
- **部署输入条件：** VOLD 可直接接收一般视觉 benchmark 输入；Vision-OPD 训练学生看带框全图和显式空间约束，迁移到无框输入的有效性未知。
- **监督风险：** VOLD 的教师可能给错误推理，因此只在 verifier 判错时指导；Vision-OPD 的 crop 通常更易，但 crop 可能丢失必要全局关系，论文未设计 teacher-confidence gate。
- **视觉依赖与正确性：** VA-OPD 能识别视觉相关 token，却不判断 rollout 是否正确；高 VA 错误轨迹也可能被加权。VOLD 的 reward mask 与 VA-OPD 的视觉权重在概念上互补。
- **教师规模：** VOLD 的 3B 学生在 8B→14B text teacher 上收益趋于饱和；VA-OPD 的 2B 学生在 4B→32B visual teacher 上，相对 Standard OPD 的增量继续增大。任务、教师模态和比较量均不同，不能据此建立统一 scaling law。
- **视觉反事实：** VA-OPD 用 pixelation 保留全局布局、偏向细节依赖；VGS 完全移除图像，覆盖全局与局部视觉信息，但更可能混入格式/风格变化；Vision-OPD 则直接提供目标 crop，监督最强但依赖区域标注。
- **“全面提升”边界：** VGS 的平均 Acc@1/Acc@16 均提升，但部分单项 greedy accuracy 下降；应区分平均趋势、采样鲁棒性与逐 benchmark 单调性。
- **Privilege 可恢复性：** Vision-OPD 的 crop 明确来自原图但依赖区域标注；ViCuR cue 更灵活，却可能由随机外部模型生成。只有在 cue 确为 \(S=f(X)\) 时其零信息 gap 命题才严格成立。
- **学生架构：** VOLD、VA-OPD、VGS 主要改变 loss；Vision-OPD 改变 teacher 条件；ViCuR 还增加 4.52–5.77% recovery 参数和 4–25% prefill latency。
- **是否保留答案特权：** ViCuR 尝试用可恢复 visual cue 替代 answer privilege；ViGOS 保留完整 reference solution，但只让它监督 description 之后的 token。前者依赖 cue provenance，后者依赖 description fidelity 与 routing 不被 fallback 绕过。
- **推理部署成本：** ViCuR 不要求生成 cue text，但增加 prefill 模块；ViGOS 不增加模型模块，却要求生成 description→reasoning→answer，论文未报告额外 token latency。
- **PALR 的解释边界：** ViGOS description PALR=0 由“不把答案给 active teacher”直接构造；full-rollout 降低部分受零 answer-sensitivity description token 稀释。Reasoning-answer PALR 与 ViLP 改善是更有信息量的证据。
- **轨迹门控语义：** VOLD 使用 absolute outcome correctness；VA-OPD/V-Zero 使用 teacher-side visual dependence。V-Zero 还做组内标准化，因此只能比较 siblings，无法判断整个 group 是否值得蒸馏。
- **Crop contrast 的混杂：** Vision-OPD 主要比较 target crop/full image；V-Zero 的 positive/negative 还同时改变区域和分辨率。没有 equal-resolution hard-negative 对照时，gate 可能部分响应清晰度。
- **消融可识别性：** V-Zero 缺少 positive-crop target + uniform \(w=1\) 对照，无法把 positive privilege 与 contrastive gate 的净贡献完全分开。
- **有图/无图差的用途：** VA-OPD 用于 token/rollout weighting，VGS 用于 visual target 与 gradient steering，VCSD 用于 vocabulary target exponential tilting。三者信号来源相似，但优化作用点不同。
- **Control validity：** VGS 的 text-only pass 改变 multimodal path；VCSD 的 black/noise/blur 保留 path 却可能是 OOD 输入。两者都不是严格的 image marginal，PMI/information-gain 解释均为近似。
- **EMA 时间尺度：** Vision-OPD 与 VCSD 都依赖 EMA target；VCSD 公式 decay \(\mu\) 与实验“update rate \(\rho=0.05\)”映射未说明，是关键复现项。
- **Correctness 的粒度：** VOLD 在整条失败 trajectory 上蒸馏；PCD 将 failure 归并到共享 perception，再用 KL 判断是否加强 perception correction。前者简单，后者更细但依赖分段格式与 teacher calibration。
- **Weight budget：** PCD 的 perception-level mean weight 为 1，但 token 长度可变时不严格保持 token-weighted teacher loss；应区分样本预算与 token/FLOP 预算。
- **理论与实证：** PCD 的 bilinear uniqueness 和 first-order optimal direction 都建立在设计公理/收益假设上；缺少 PSR-only、KL-only、addition 对照，尚不能证明乘法经验最优。
- **Target reconstruction 层级：** VCSD 对 support 内 conditional contrast 做 exponential tilt；VGS 用 visual information gain 构造 target并 steering；VAD 先把完整 teacher correction 投影到 intervention direction，再重构 signed target。
- **Attribution geometry：** VAD 在 student top-\(K\)+tail 的 centered-log Euclidean space 中投影。该方向对 support 和度量敏感，不能直接等同于概率空间或 causal attribution。
- **稳定 anchor 的反作用：** VAD/VCSD 都需要原 teacher/policy anchor防止语言漂移；anchor 同时会重新引入被视觉分解试图排除的 source-mixed correction。
- **Projection 目标差异：** VAD 投影到 same-teacher evidence intervention方向，强调来源归因；FP-OPD 投影到 student perturbation span，强调能力兼容。前者 proxy更 task-specific，后者 metric更 principled。
- **Reachability 定义：** FP-OPD 的 \(J_Z\ell_\theta\) 是 input-feature tangent，训练更新却沿 \(J_\theta\ell_\theta\)。称 projected target“可训练实现”仍需 parameter-space或实际 optimization证据。
- **简单 weakening baseline：** FP-OPD 的动机实验显示 \(\alpha=0.5\) 优于完整 target，却未在七项主表比较 tuned scalar interpolation；projection增益尚未与 target norm reduction完全分离。
- **Groundedness vs correctness：** SA-OPD 假定input-dependent signal更可能有用，但 grounded teacher error仍可高 \(\Delta^{IG}\)；低 gap也可能来自teacher/student相同的真实输入响应。
- **Impact proxy粒度：** SA-OPD用sampled-token \(|A_t|\)选择位置，却将该位置full-vocabulary KL称为loss mass；二者不是同一个量。
- **Control条件：** SA-OPD no-prompt、VCSD black image、V-Zero negative crop与VAD degraded crop都需要matched-control audit，不能把contrast直接解释为纯视觉或纯输入因果效应。
- **Modality-balance解释：** OPD-V的attention ratio只用于相关性动机，objective只用zoom/mask log-prob margin；更稳妥的术语是visual-intervention trust region。
- **错误token强化：** OPD-V/VA-OPD的positive sampled-token margin不含correctness，可能增权image-sensitive but wrong tokens；VOLD correctness mask或PCD outcome witness可补足。
- **效率口径：** OPD-V实际step time更低，但同时改变response length、teacher prompt construction与input length；不能等价为双teacher方法per-token更便宜。

## 可用于 SpatialStack_OPSD 的设计

建议构造统一的“特权空间状态 OPD”最小方案：

1. 学生使用部署条件下的完整场景；教师使用目标 crop、局部 occupancy map、真值对象图或真值位姿中的一种。
2. rollout 必须由学生生成；教师只在学生 prefix 上重评分。
3. 先测 teacher 在 student states 上的 NLL、entropy、top-\(K\) overlap 和任务正确率。若跨模型差距大，采用 VOLD 式同教师 SFT cold start；若同模型 teacher 动态更新，采用 frozen 或 EMA target。
4. 若有 verifier，只对失败轨迹或 teacher 明显更可信的 token 蒸馏；若无 verifier，可使用 teacher entropy、跨视图一致性或空间约束满足度做 confidence gate。
5. 实现 top-\(K\)+tail logits 蒸馏时，先按本项目分布选择 \(K\)，并显式核对 divergence 方向。
6. 必做分解消融：无特权信息、仅框、crop 不放大、crop 放大、无框学生、frozen teacher、EMA teacher、sampled-token、top-\(K\)。
7. 同时报告 privileged-to-deployment gap、空间推理、基础感知、无框定位及训练/推理成本。
8. 增加 factor-specific spatial advantage：分别遮挡区域、移除深度、扰动对象 ID、破坏拓扑边或模糊坐标文本，识别坐标、方向、距离和对象关系 token。
9. 同时使用 correctness 与 spatial advantage 双轴门控：保护已正确的新路径，强化真正依赖空间证据的 token；并直接测 student 在原始/反事实输入下的 log-prob 差，避免只用 teacher proxy 证明 grounding。
10. 增加 gradient-level 诊断：分别计算 language、spatial target 与 outcome objective 的梯度 norm/cosine；比较 token weighting、gradient steering、PCGrad/投影和 Pareto weighting。
11. 构造 spatial target 时明确 stop-gradient：若 target 含 student text-only prior，必须对其梯度路径做单元测试与消融；同时保持 steered gradient norm，避免混淆方向和 learning-rate 效应。
12. 建立 privilege provenance 规范：记录 cue/crop/对象图的生成器、可见字段、是否访问答案、随机性和过滤规则，并自动扫描答案、定理、真值坐标及 teacher-only 标签泄漏。等视图 Answer-OPSD 必须把答案特权与 N/K 视图特权分成两臂，并扫描 Hint/GT 复述。
13. 若使用 recovery token，应让 query 在因果路径上真正读取当前问题；比较前置固定 sink、question 后 sink、question-summary query 与等参数 pooling baseline。
14. 把空间 rollout 拆成结构化 observation→relation inference→plan/action，并按阶段限制 teacher 可见字段；感知/观测 teacher 不应访问最终答案、真值路径或动作标签。
15. 对所有 routing 分支记录样本率、token 数、loss 和梯度占比；invalid-format fallback 优先使用无答案 grammar teacher，避免 full privileged teacher 在训练早期重新监督整条轨迹。
16. 使用同一结构化 prompt 比较 zero-shot、vanilla OPD 与 staged OPD，并单独报告输出长度和 latency；否则无法区分 prompt、supervision routing 与额外计算的贡献。
17. 构造 positive/negative spatial evidence 时严格匹配分辨率、面积和全局上下文；加入邻近 hard negative、同类对象负例、关系边扰动与地图拓扑破坏。
18. 同时使用 absolute evidence threshold、sibling-relative advantage 和 correctness verifier；当整个 group 的证据分数都低时关闭蒸馏，而不是强制组内选优。
19. 对 crop-gated OPD 做正交消融：full-image teacher、positive-crop teacher、positive+random gate、positive+hard-negative gate；并报告 token/trajectory gate 与空间关键 token 的对应关系。
20. 为每类空间因素设计 matched control，并报告 intervention 后 token count、position encoding、support size、retained mass 与 teacher entropy，避免把 OOD artifact 当作空间信息。
21. 比较三种相同 contrast 的使用方式：scalar token/trajectory weighting、visual target+gradient steering、support-restricted target shaping；统一 teacher、数据与算力后判断真正收益来源。
22. EMA self-distillation 必须明确 old-teacher decay 与 new-student update fraction，并跟踪 teacher-student KL、target support 和 language drift 随 step 的变化。
23. 对同一 perception 采多条 planner continuations，估计 downstream value；同时加入视觉 intervention、teacher confidence 或事实 verifier，避免把困难 reasoning 错判成 perception failure。
24. Perception correction 必做 PSR-only、KL-only、additive、multiplicative 与 learned/calibrated gate 多 seed 对照，并报告人工 perception-error precision/recall。
25. 使用 per-perception normalized token loss 后再乘 trajectory weight，严格区分 selective allocation 与可变 perception 长度造成的 loss-scale变化。
26. 对 spatial teacher correction 做 multi-intervention basis attribution：对象、深度、坐标、拓扑分别构造方向，并比较 Euclidean、probability-weighted 与 Fisher projection。
27. Candidate support 至少使用 student/teacher union top-\(K\)+tail；报告正确 token 落在 support 外的比例，避免 conservative student anchor 阻止新候选进入。
28. 对 signed refutation 同时报告 pairwise odds improvement、absolute wrong-token suppression 和最终 correctness，并审计 degradation artifact。
29. 在同一 teacher gap上比较 scalar shrink、VAD intervention projection、FP Fisher input projection与 parameter-Jacobian projection，并匹配 projected norm。
30. 对 tangent method报告 probe coverage、Gram rank/condition number、projected energy ratio、finite-difference linearity和额外 GPU-hours。
31. 使用 adapter/LoRA parameter JVP估计 trainability tangent，检验其与 visual input tangent及实际 one-step fitting improvement的相关性。
32. 对 teacher/student分别计算 full-control log-prob effect，再比较 disagreement effect；报告 cancellation rate。
33. 按response position、实体/数字已出现情况审计prefix leakage，并比较no-prompt、image-only removal、relation removal等matched controls。
34. 用full-KL差、Fisher gradient norm和sampled \(|A_t|\)分别构造impact gate，区分token deletion与retained-token renormalization。
35. 对zoom/mask trust region加入semantic mask、correctness gate与negative-margin refutation，并报告wrong-token selection precision。
36. 比较all-token、selected-count、valid-count与weight-sum normalization，匹配gradient norm后评估margin weighting净收益。
37. 固定sequence length和batch token数测双teachertokens/s，同时单独报告行为性response shortening带来的end-to-end收益。

## 综合结论的证据索引

每条综合结论应链接到至少一篇论文笔记中的具体证据。

- **S-001：OPD 的核心价值是学生状态上的密集监督。** 证据：[VOLD E-001](papers/vold.md#e-001)；[Vision-OPD E-002](papers/vision-opd.md#e-002)。
- **S-002：教师可靠性需要状态对齐和时间稳定性。** 证据：[VOLD E-002](papers/vold.md#e-002)；[Vision-OPD E-007](papers/vision-opd.md#e-007)。
- **S-003：特权监督既可来自更强模型，也可来自更易输入。** 证据：[VOLD E-001](papers/vold.md#e-001)；[Vision-OPD E-001、E-002](papers/vision-opd.md#e-001)。
- **S-004：选择性蒸馏与 teacher confidence gating 值得统一研究。** 直接证据：[VOLD E-003](papers/vold.md#e-003)；研究推断：[Vision-OPD E-007](papers/vision-opd.md#e-007)。
- **S-005：当前 OPD 公式与实现方向均需源码核对。** 证据：[VOLD E-008、E-009](papers/vold.md#e-008)；[Vision-OPD E-012](papers/vision-opd.md#e-012)。
- **S-006：视觉监督集中在少数 token，uniform KL 会稀释关键梯度。** 证据：[VA-OPD E-002、E-003](papers/va-opd.md#e-002)。
- **S-007：准确率提升与视觉依赖增强必须分开测量。** 证据：[VOLD E-006](papers/vold.md#e-006)；[VA-OPD E-008](papers/va-opd.md#e-008)。
- **S-008：correctness masking 与 visual-dependency weighting 是互补选择轴。** 直接证据：[VOLD E-003](papers/vold.md#e-003)、[VA-OPD E-004](papers/va-opd.md#e-004)；联合使用属于待验证推断。
- **S-009：高视觉依赖 token 上语言与视觉梯度可能近乎正交或冲突。** 证据：[VGS E-002、E-004](papers/vgs.md#e-002)。
- **S-010：视觉 grounding 干预可分为“监督位置选择”与“梯度方向 steering”。** 证据：[VA-OPD E-004](papers/va-opd.md#e-004)、[VGS E-003](papers/vgs.md#e-003)。
- **S-011：视觉 steering 可与 outcome RL 联合，但逐项收益并非始终单调。** 证据：[VGS E-007、E-012](papers/vgs.md#e-007)；[VOLD E-003](papers/vold.md#e-003)。
- **S-012：Privilege 设计会影响 OPD 的部署匹配与 shortcut leakage。** 证据：[OPSD E-002、E-006](papers/opsd.md#e-002)；[ViCuR E-001、E-004、E-008](papers/vicur.md#e-001)；[Vision-OPD E-010](papers/vision-opd.md#e-010)。
- **S-013：视觉 cue 本身的贡献通常大于额外 recovery architecture。** 证据：[ViCuR E-006](papers/vicur.md#e-006)。
- **S-014：Recoverable privilege 是条件性结论，依赖 cue provenance。** 论文结论：[ViCuR E-001](papers/vicur.md#e-001)；核对项：[ViCuR E-009](papers/vicur.md#e-009)。
- **S-015：限制答案 privilege 的介入阶段可改善先验冲突下的视觉依赖。** 证据：[ViGOS E-003、E-005](papers/vigos.md#e-003)。
- **S-016：结构化“先描述后推理”prompt 是强混杂，必须做 same-prompt control。** 证据：[ViGOS E-007](papers/vigos.md#e-007)。
- **S-017：Format fallback 可能成为 privilege routing 的旁路，需报告实际触发占比。** 论文设计：[ViGOS E-001](papers/vigos.md#e-001)；核对项：[ViGOS E-009](papers/vigos.md#e-009)。
- **S-018：无答案标签的视觉对比可提供 trajectory-level evidence gate，但不等于 correctness。** 证据：[V-Zero E-002、E-008](papers/v-zero.md#e-002)。
- **S-019：Answer-label-free 方法仍可能依赖高成本区域级特权监督。** 证据：[V-Zero E-004](papers/v-zero.md#e-004)；[Vision-OPD E-010](papers/vision-opd.md#e-010)。
- **S-020：Positive crop target 与 contrastive gate 必须通过正交消融分离。** 核对项：[V-Zero E-007](papers/v-zero.md#e-007)。
- **S-021：无需外部 privilege，也可由 matched input conditioning 构造 OPSD target asymmetry。** 证据：[VCSD E-001、E-006](papers/vcsd.md#e-001)。
- **S-022：Visual contrast 可直接重塑 full-distribution target，但需 plausible support 防止递归 distortion。** 证据：[VCSD E-002、E-008](papers/vcsd.md#e-002)。
- **S-023：Content-erased contrast 只是 conditional PMI 的代理，其解释取决于 control validity。** 条件性证据：[VCSD E-005、E-010](papers/vcsd.md#e-005)；[VGS E-010](papers/vgs.md#e-010)。
- **S-024：Final-answer reward 无法单独识别 perception failure，需要额外 witness。** 证据：[PCD E-002、E-003](papers/pcd.md#e-002)。
- **S-025：Perception-level failure×disagreement weighting可改善 aggregate，但乘法形式尚未被 single-witness 对照隔离。** 端点证据：[PCD E-007](papers/pcd.md#e-007)；局限：[PCD E-008](papers/pcd.md#e-008)。
- **S-026：Label-free 必须按 supervision 类型细分。** [PCD E-009](papers/pcd.md#e-009) 仍需答案 verifier；[VCSD E-001](papers/vcsd.md#e-001) 与 [V-Zero E-004](papers/v-zero.md#e-004) 不消费 textual answer label。
- **S-027：Visual contrast 可用于重构 candidate-level target，而不只加权 loss。** 证据：[VAD E-001–E-006](papers/vad.md#e-001)；对照：[VA-OPD E-004](papers/va-opd.md#e-004)、[V-Zero E-002](papers/v-zero.md#e-002)。
- **S-028：单 view-pair projection 只提供 proxy-aligned semantic enrichment，不是可识别视觉分解。** 证据：[VAD E-009、E-011](papers/vad.md#e-009)。
- **S-029：Signed coordinate 只直接控制 relative odds，不保证 absolute probability suppression。** 证据：[VAD E-003](papers/vad.md#e-003)。
- **S-030：完整 teacher target 可能不如中间 target易拟合，但这不自动证明 correction超出 student capacity。** 证据：[FP-OPD E-001、E-009](papers/fp-opd.md#e-001)。
- **S-031：Fisher geometry显著优于 Euclidean projection，但 probe basis仍只近似 student response space。** 证据：[FP-OPD E-007、E-008、E-010](papers/fp-opd.md#e-007)。
- **S-032：Input-response tangent 与 parameter-update realizability必须区分。** 核对项：[FP-OPD E-010](papers/fp-opd.md#e-010)。
- **S-033：No-prompt disagreement gap只测teacher/student相对输入效应之差，不识别teacher grounding。** 证据：[SA-OPD E-001、E-009](papers/sa-opd.md#e-001)。
- **S-034：低groundedness与高impact的intersection优于两个single-axis filters，但证据仅来自两个ablation benchmarks。** 证据：[SA-OPD E-007](papers/sa-opd.md#e-007)。
- **S-035：Sampled log-ratio mass不能直接称为full-KL或gradient mass。** 核对项：[SA-OPD E-011](papers/sa-opd.md#e-011)。
- **S-036：Zoom/mask正margin提供有效token allocation，但不直接识别modality balance。** 证据：[OPD-V E-002、E-010](papers/opd-v.md#e-002)。
- **S-037：OPD-V matched 4B average比Vision-OPD高2.91，但无多seed且跨论文baseline数值不一致。** 证据：[OPD-V E-005](papers/opd-v.md#e-005)。
- **S-038：OPD-V wall-clock下降来自更短response与更轻teacher construction，非固定token算法成本。** 证据：[OPD-V E-009](papers/opd-v.md#e-009)。

