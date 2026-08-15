# 未解决问题与研究假设

记录跨论文阅读中尚未解决、需要继续查阅或通过实验验证的问题。

## 高优先级

### Q-001：VOLD 实际采用的 KL 方向、估计器与系数是什么？

- 状态：open
- 来源论文：[VOLD](papers/vold.md)（E-008、E-009）
- 当前认识：式 (3) 写为 \(D_{\mathrm{KL}}(\pi_\phi\|\pi_\theta)\)，但论文称用 student-sampled 单 token 的 “k2” estimator；正文 §4.1 写 \(\beta=0.1\)，补充材料表 4 写 \(\beta=10^{-3}\)。
- 相互冲突的证据：公式方向与采样描述可能不匹配；同一版本的两个超参数位置直接冲突。
- 需要查阅：作者公开代码、配置文件、训练日志或作者勘误。
- 可验证实验：分别实现两个 KL 方向并 sweep \(\beta\in\{10^{-3},10^{-2},10^{-1}\}\)，比较训练稳定性、Geo3K 与文本 reward。
- 结论：待核对。
- 最后更新：2026-08-13

### Q-002：如何直接测量 on-policy distillation 生效所需的 teacher/student 对齐阈值？

- 状态：experiment
- 来源论文：[VOLD](papers/vold.md)（E-002）
- 当前认识：VOLD 用 SFT steps 间接表示对齐程度，并观察到约 3000 steps 后 OPD 收益趋于饱和；论文没有直接报告策略 divergence。
- 相互冲突的证据：暂无；但 SFT steps 同时改变分布匹配、学生能力和轨迹记忆，不能等价于纯粹的对齐程度。
- 需要查阅：on-policy distillation、sequence-level distribution matching 与 state-distribution shift 的测量方法。
- 可验证实验：在固定 student rollout 上测 teacher NLL、token KL、teacher entropy、top-k overlap，并拟合这些指标与 OPD 相对 GRPO 增益的关系。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-004：Vision-OPD 的 divergence、EMA 与 top-\(K\) 实际如何实现？

- 状态：open
- 来源论文：[Vision-OPD](papers/vision-opd.md)（E-004、E-012）
- 当前认识：主配置使用 top-100 \(\mathrm{JSD}_{0.5}\) 和 EMA teacher（\(\alpha=0.05\)）；式 (3)/(4) 写 \(D(p_T\|p_S)\)，算法 1 第 7 行写 \(D(p_S\|\operatorname{stopgrad}(p_T))\)。JSD 主配置对顺序不敏感，但表 4 的 forward/reverse KL 对顺序敏感。
- 相互冲突的证据：公式与算法的参数顺序直接相反；论文未给 EMA 更新公式、更新时机、参数覆盖范围和 top-\(K\) tail bucket 细节。
- 需要查阅：作者公开代码、训练配置、teacher update 与 distillation loss 实现。
- 可验证实验：对固定 logits 单元测试 forward KL、reverse KL、JSD 和 top-\(K\)+tail；记录 EMA 前后参数并复现实验表 3、表 4。
- 结论：待核对。
- 最后更新：2026-08-13

### Q-005：特权视图 OPD 能否迁移到没有框或目标坐标的全场景输入？

- 状态：experiment
- 来源论文：[Vision-OPD](papers/vision-opd.md)（E-010、E-011）
- 当前认识：论文学生看叠加 bounding box 的全图，问题还附加显式空间约束；现有结果证明模型能从带定位提示的全图恢复 crop-visible evidence。
- 相互冲突的证据：论文将结果表述为学会从 full image 聚焦细粒度证据，但训练学生输入已明确标出证据区域，因此不能据此确认自主搜索或定位能力。
- 需要查阅：公开数据预处理与 benchmark inference 代码，确认训练和评测时 bounding box 的使用方式。
- 可验证实验：比较无框全图、仅框、仅文本坐标、crop、crop+2 倍放大五组，并在训练中对框提示做 dropout 或退火。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-007：VA-OPD 提升的是学生真实视觉依赖，还是教师代理指标上的视觉依赖？

- 状态：experiment
- 来源论文：[VA-OPD](papers/va-opd.md)（E-001、E-008）
- 当前认识：VA-OPD 训练后，教师在 student rollout 上评分的 mean VA 从 0.07 增至 0.16，且 MathVerse accuracy 提升；但 VA 完全由教师的原图/退化图 log-prob 差定义。
- 相互冲突的证据：论文将 rollout VA 上升解释为学生更依赖视觉细节，同时在 limitations 中承认 VA 不是学生 attention、feature grounding 或内部感知过程的直接测量。
- 需要查阅：causal mediation、input ablation、counterfactual VLM grounding 与视觉信息依赖指标。
- 可验证实验：直接计算 student 在原图/退化图/无图条件下的 token log-prob 差，并测试图像替换、局部遮挡后的准确率下降；与 teacher-scored VA 做相关和干预分析。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-009：VGS 的 visual target、stop-gradient 与梯度归一化实际如何实现？

- 状态：open
- 来源论文：[VGS](papers/vgs.md)（E-003、E-010、E-011）
- 当前认识：\(q_T^*\) 显式使用当前 student 的 text-only prior；\(\eta\) 在理论上由梯度 norm 定义，但主实验使用预估常数；adaptive VGS 式 (24) 在 token 求和中写入 trajectory-level loss。
- 相互冲突的证据：若 student prior 不 stop-gradient，target 会向 student 分支反传；若动态 \(\eta\) 不 stop-gradient，会产生二阶项；式 (24) 按字面会重复整条 trajectory loss。论文未明确这些实现语义。
- 需要查阅：公开代码中的 visual target construction、detach、loss reduction、gradient scaling 与 adaptive-VGS implementation。
- 可验证实验：固定 logits 做梯度单元测试；比较 detach/no-detach target、固定/dynamic-stopgrad \(\eta\)、token/trajectory adaptive loss，并复现表 1、表 5。
- 结论：待核对。
- 最后更新：2026-08-13

### Q-010：ViCuR 的 visual cue 是否真正只含推理输入中可恢复的信息？

- 状态：open
- 来源论文：[ViCuR](papers/vicur.md)（E-001、E-009）
- 当前认识：理论在确定映射 \(S=f(X)\) 下得到 \(I(Y_t;S\mid X,Y_{<t})=0\)，但现实 cue 可能由外部模型随机生成；论文未公开生成器、prompt、答案可见性和过滤规则。
- 相互冲突的证据：论文将 cue 描述为 inference-recoverable，同时承认实际 \(H(S\mid X)>0\)；缺少 provenance 无法排除 cue 使用答案、标注、外部知识或不可见推理。
- 需要查阅：公开代码与数据字段、cue generation prompt、生成模型输入权限、随机参数、过滤与 leakage audit。
- 可验证实验：对 cue 做答案/定理/标注泄漏扫描；分别仅凭 \(X\) 和凭 \(X+\)外部知识重建 cue；比较 cue-conditioned teacher 与 marginal teacher 的 conditional information gap。
- 结论：待核对。
- 最后更新：2026-08-13

### Q-011：前置 sink token 能否真正按当前问题恢复视觉 cue？

- 状态：experiment
- 来源论文：[ViCuR](papers/vicur.md)（E-002、E-010）
- 当前认识：sink 位于 system prompt 前，专用 query 来自 sink hidden state；它可 cross-attend visual tokens，但按 causal 顺序通常看不到后续问题。作者称问题通过 question-dependent training loss “隐式”影响模块。
- 相互冲突的证据：跨样本训练梯度能学习通用视觉聚合参数，但不等于推理时 query 随当前 \(Q\) 动态变化；论文却将表示写为 \(g(V,Q)\) 并称其聚合 task-relevant evidence。
- 需要查阅：Qwen3-VL token/layout、视觉 token 在各层是否可读取 question、recovery branch 的实际 attention mask 和执行顺序。
- 可验证实验：比较前置 sink、question 后 sink、question-summary query、无问题 query 和等参数 pooling；做 question swap、attention intervention 与 cue probe。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-012：ViGOS 的 invalid fallback 会在多大程度上绕过 perception/reasoning 解耦？

- 状态：open
- 来源论文：[ViGOS](papers/vigos.md)（E-001、E-009）
- 当前认识：valid rollout 按 description/reasoning 路由 teacher；invalid rollout 则由看到完整 reference solution 的 teacher 对全序列做 reverse KL，且 \(\lambda_{\mathrm{ref}}=2\)。
- 相互冲突的证据：fallback 被解释为有限的格式恢复信号，但论文未报告训练期间 invalid rate、fallback token 占比或梯度占比；如果早期格式失败多，full privileged path 可能重新主导训练。
- 需要查阅：公开训练日志、parser 实现、每 step routing 统计与 fallback loss scale。
- 可验证实验：记录三条分支的样本/token/gradient 占比；比较 full reference、无答案 format teacher、grammar loss、直接丢弃 invalid rollout 和 staged format warm-up。
- 结论：待核对与实验。
- 最后更新：2026-08-13

### Q-013：PALR 能否可靠测量 privileged-answer shortcut，而不只是反映 teacher routing 的构造？

- 状态：experiment
- 来源论文：[ViGOS](papers/vigos.md)（E-003、E-008）
- 当前认识：PALR 用 wrong-answer mixture 与 mismatched-next image 比较 observed token 的 answer/image sensitivity；ViGOS description teacher 不接收答案，因此该 segment 的 answer sensitivity 和 PALR 被定义为 0。
- 相互冲突的证据：reasoning-answer PALR 与 ViLP 同向变化支持其诊断价值；但 full-rollout PALR 会被零 answer-sensitivity description token 稀释，数值还依赖 Gemini 生成的错误答案、positive-part 和 image counterfactual。
- 需要查阅：causal attribution、conditional mutual information、modality reliance 与 counterfactual benchmark 设计。
- 可验证实验：替换 wrong-answer generator、难度和混合方式；使用语义匹配图像、局部遮挡和 question counterfactual；检验 PALR 与 ViLP、student image sensitivity 及干预后性能的相关性。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-014：V-Zero 的 contrastive gate 是否提供了独立于 positive crop teacher 的净收益？

- 状态：open
- 来源论文：[V-Zero](papers/v-zero.md)（E-007）
- 当前认识：完整方法使用 positive crop 作为 distillation target，并用 positive/negative crop difference 计算 gate；表 2 的 `None` 没有正负 evidence view，`Rand.` 又同时替换 positive 与 negative。
- 相互冲突的证据：完整方法高于 `None` 1.2 perception-average points，但缺少“positive-crop teacher + 所有 \(w=1\)”行，因此差异可能同时来自更强 teacher target 和 gate。
- 需要查阅：公开代码、表 2 中 `None` 的精确定义、作者补充消融。
- 可验证实验：固定 positive-crop target，比较 uniform weight、random gate、positive-negative gate、shuffled gate 与 oracle correctness gate。
- 结论：待核对与实验。
- 最后更新：2026-08-13

### Q-015：组内 relative evidence advantage 如何避免强化 all-wrong 或视觉相关但结论错误的轨迹？

- 状态：experiment
- 来源论文：[V-Zero](papers/v-zero.md)（E-002、E-008）；[VA-OPD](papers/va-opd.md)（E-001、E-008）；[VOLD](papers/vold.md)（E-003）
- 当前认识：V-Zero 对 sibling scores 做 z-normalization，即使整个 group 都错或证据都弱，也会产生相对高权重；VA-OPD 同样衡量 visual dependence 而非 correctness；VOLD 的 verifier 则只提供 correctness。
- 相互冲突的证据：V-Zero 的 benchmark 提升说明相对 gate 有经验价值，但论文未报告 gate 与答案正确率、absolute crop margin 或 student grounding 的相关性。
- 需要查阅：group-relative reward degeneracy、positive-unlabeled ranking、confidence calibration 与 multi-objective gating。
- 可验证实验：按 correctness×absolute evidence×relative rank 分桶；比较 z-score、absolute threshold、二者联合、all-low group rejection 及 correctness×evidence 二维 gate。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-016：VCSD 的 EMA update rate 0.05 在实现中究竟对应哪个方向？

- 状态：open
- 来源论文：[VCSD](papers/vcsd.md)（E-003、E-012）
- 当前认识：方法写 \(\phi\leftarrow\mu\phi+(1-\mu)\theta\)，实验则称 EMA update rate \(\rho=0.05\)，未定义两者关系。
- 相互冲突的证据：若 \(\rho=\mu=0.05\)，teacher 每步几乎替换为当前 student；若 \(\rho=1-\mu=0.05\)，则 conventional EMA decay 为 0.95。两者 target drift 和稳定性完全不同。
- 需要查阅：公开代码、optimizer step 后的 EMA update、参数日志或作者勘误。
- 可验证实验：对单参数 checkpoint 做一步 update 单元测试；比较 decay 0.05/0.95/0.99 下 teacher-student KL、support drift 与七项性能。
- 结论：待核对。
- 最后更新：2026-08-13

### Q-017：Content-erased control 能否近似“无 instance-specific visual information”的条件分布？

- 状态：experiment
- 来源论文：[VCSD](papers/vcsd.md)（E-005、E-010）；[VGS](papers/vgs.md)（E-001、E-010）
- 当前认识：VCSD 用 black/noise/blur/no-image prediction 近似 \(p(v\mid H_t)\)，VGS 使用 text-only teacher；这些 intervention 分别引入 OOD visual tokens 或改变 multimodal computation path。
- 相互冲突的证据：VCSD 四种 control 的 aggregate 接近，支持平均鲁棒性；但逐 benchmark 波动达 4–5 points，且没有直接比较各 control 与真实 image-marginal distribution。
- 需要查阅：causal feature ablation、conditional mutual information estimation、in-distribution image counterfactual 与 modality dropout。
- 可验证实验：构造 token/path matched 的 learned-null image、dataset-matched irrelevant image 与多 control ensemble；比较 distribution KL、contrast ranking、grounding intervention 和最终性能。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-018：PCD 的乘法 gate 是否优于 single-witness 或 additive fusion？

- 状态：open
- 来源论文：[PCD](papers/pcd.md)（E-003、E-007、E-008）
- 当前认识：PCD 用 \((1-\mathrm{PSR})\widetilde{\mathrm{KL}}\) 实现 soft AND；去掉所有 adaptive weight 会使 held-out average 下降 2.22 points。
- 相互冲突的证据：机制图证明实现符合乘法形状，但论文没有 PSR-only、KL-only、addition 的 matched runs；bilinear uniqueness 仅在预设 zero-axis boundary conditions 下成立。
- 需要查阅：公开代码、补充实验、multi-view credit assignment 与 calibrated evidence fusion。
- 可验证实验：固定 separated rollout、teacher loss 与 mean normalization，比较 uniform、PSR、KL、sum、product、min、log-odds fusion，并做多 seed 与人工 perception-error calibration。
- 结论：待核对与实验。
- 最后更新：2026-08-13

### Q-019：PCD 主表使用的究竟是完整 perception distillation 还是 VPPO-Distill？

- 状态：open
- 来源论文：[PCD](papers/pcd.md)（E-010）
- 当前认识：主表 2B PCD 的七个共享 benchmark 数值与消融表 `Full w/ VPPO-Distill` 完全相同，而 `Full (ours)` 使用完整 perception span 且 held-out average 更高。
- 相互冲突的证据：方法与消融正文把完整 perception distillation 定义为 default full method，把 VPPO-Distill 称为 substitute；主结果数值却对应 substitute。
- 需要查阅：作者训练配置、checkpoint 名称、表格生成脚本或勘误。
- 可验证实验：按两种 token mask 复现主表，并核对论文所报 47.28 macro 的具体 checkpoint/config。
- 结论：待核对。
- 最后更新：2026-08-13

### Q-020：VAD 的视觉归因是否对 candidate support 与 projection geometry 稳健？

- 状态：experiment
- 来源论文：[VAD](papers/vad.md)（E-002、E-011）
- 当前认识：VAD 在 student top-100+tail 的 centered log-prob coordinates 中，用 uniform Euclidean inner product 将完整 teacher correction投影到单个 intervention vector。
- 相互冲突的证据：matched benchmark 与语义 enrichment 支持该设计有经验价值；但 correct teacher candidate 若不在 student top-\(K\) 会被 tail 合并，投影结果也可能随 support、centering和coordinate weighting改变。
- 需要查阅：information geometry、Fisher projection、logit attribution、top-\(K\) distillation 与 sparse support expansion。
- 可验证实验：比较 student/teacher/union top-\(K\)、不同 \(K\)、tail/no-tail，以及 Euclidean、probability-weighted、Fisher-metric projection；跟踪 correct-token support recall。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-021：单个 evidence-present/degraded direction 能否表示组合式空间证据？

- 状态：experiment
- 来源论文：[VAD](papers/vad.md)（E-001、E-009）
- 当前认识：VAD 每个 prefix 只有一个 \(u_t\)，可归因 correction 被限制在它的一维 span；论文明确承认单 view pair 可能偏置 compositional evidence。
- 相互冲突的证据：六项 benchmark 提升和 token semantic enrichment 说明单方向有效；但多个对象、属性、深度与关系可能产生非共线 correction，residual 仍 source-mixed。
- 需要查阅：multi-view causal subspace、concept activation vectors、low-rank attribution basis 与 non-negative projection。
- 可验证实验：为对象移除、坐标遮挡、depth erased、relation broken 构造多方向 basis，比较 ridge/NNLS/orthogonal projection 与单向量 VAD。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-022：FP-OPD 的 input visual tangent 是否真的刻画参数训练可实现性？

- 状态：experiment
- 来源论文：[FP-OPD](papers/fp-opd.md)（E-002、E-010）
- 当前认识：FP-OPD 用 \(J_Z\ell_\theta\) 的四个 finite-difference directions近似 student visual response space，但实际 distillation通过更新参数 \(\theta\) 改变固定输入上的分布。
- 相互冲突的证据：Fisher-projected target在 benchmark 上优于 OPD；但 input sensitivity 与 \(J_\theta\ell_\theta\) parameter-update tangent不是同一空间，论文没有直接测 projected component的实际 fitability。
- 需要查阅：neural tangent kernel、parameter Jacobian、influence functions、natural gradient 与 representation controllability。
- 可验证实验：构造 LoRA/visual-adapter parameter JVP basis，比较 input tangent、parameter tangent和实际 one-step/one-epoch target-gap removal；测 principal angles与性能相关性。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-023：FP-OPD 的收益是否只是来自减弱 teacher target，而非选择正确方向？

- 状态：open
- 来源论文：[FP-OPD](papers/fp-opd.md)（E-001、E-009）
- 当前认识：论文动机实验中 \(\alpha=0.5\) target比完整 \(\alpha=1\) 更易拟合且 Geometry score更高，但主结果没有 tuned scalar interpolation baseline。
- 相互冲突的证据：Fisher projection远高于 Euclidean projection，支持 geometry重要；然而未匹配 projected norm，无法分离“方向选择”与“整体 target变弱”。
- 需要查阅：target interpolation实现、完整 alpha sweep、projected Fisher norm和代码配置。
- 可验证实验：对每个 prefix用 scalar shrink匹配 FP projected Fisher norm，并比较 fixed/global alpha、adaptive norm-only shrink、Fisher direction projection与random direction control。
- 结论：待核对与实验。
- 最后更新：2026-08-13

### Q-024：SA-OPD 的 no-prompt disagreement gap 能否识别 teacher signal 的真实 input-groundedness？

- 状态：experiment
- 来源论文：[SA-OPD](papers/sa-opd.md)（E-001、E-009、E-010）
- 当前认识：\(\Delta^{IG}\) 是 teacher与student的输入效应之差；低值既可能表示双方都不依赖输入，也可能表示双方以相近幅度依赖输入。保留的response prefix还可能泄露已删除输入。
- 相互冲突的证据：Intersection filter在两个ablation benchmark优于单因素，但论文没有teacher-only grounding标注或causal precision/recall。
- 需要查阅：no-prompt具体模板、VLM image/question removal方式、token-position diagnostics与定性样本。
- 可验证实验：分别计算teacher/student full-control effect，构造2×2 cancellation taxonomy；按position、实体/数字泄露分桶，并用程序化空间任务提供ground-truth relevant interventions。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-025：SA-OPD 的 FLMR 是否真正控制 OPD loss/gradient mass？

- 状态：open
- 来源论文：[SA-OPD](papers/sa-opd.md)（E-003、E-004、E-011）
- 当前认识：FLMR累加sampled-token \(|A_t|\)，而最终训练使用position-level full-vocabulary reverse KL；两者通常相关但不相等。论文报告的 \(\beta=1.8\) 与ratio范围也存在单位歧义。
- 相互冲突的证据：动态约束改善CountQA且额外成本较小；但没有报告 \(|A_t|\) 与KL/gradient norm相关性，也未给 \([\beta_{\min},\beta_{\max}]\)。
- 需要查阅：实现代码、内部percent scaling、dynamic binary-search参数和loss normalization。
- 可验证实验：比较sampled \(|A|\)、full KL、Fisher gradient norm三种mass budget；加入固定原denominator与retained-count denominator对照。
- 结论：待核对与实验。
- 最后更新：2026-08-13

### Q-026：OPD-V 的 zoom/mask margin 是否真的测量 modality balance？

- 状态：experiment
- 来源论文：[OPD-V](papers/opd-v.md)（E-001、E-002、E-010）
- 当前认识：训练只比较同一EMA teacher在zoom crop与随机masked crop上对sampled token的log probability；attention ratio只作相关性分析。该margin更直接表示特定visual intervention sensitivity。
- 相互冲突的证据：正margin与attention-ratio gap、correctness相关，且最终性能强；但zoom改变scale/global context，mask可能不覆盖evidence并引入black artifact，attention mass也不等于causal contribution。
- 需要查阅：mask sampler、attention layer/head aggregation、5K分析数据组成与误差条。
- 可验证实验：加入semantic evidence removal、irrelevant-region mask、texture-preserving inpainting和text-only controls；用causal attention intervention或logit attribution验证margin与真实modality contribution的相关性。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-027：OPD-V 的 margin gate、margin weight 与 loss normalization各自贡献多少？

- 状态：open
- 来源论文：[OPD-V](papers/opd-v.md)（E-003、E-007、E-012）
- 当前认识：\(\delta^{MB}\)同时决定是否选择和连续权重，loss却除以全部valid tokens；附录另列threshold 2.0但公式未体现。
- 相互冲突的证据：Dual-teacher结果显著高于single-teacher variants；然而未分离binary gate、weight scale、normalization和clipping。
- 需要查阅：rollout correction实现、negative-only ablation objective和训练日志中的margin分布。
- 可验证实验：做gate-only、weight-only、sign×constant、clipped weight、weight-sum normalization与gradient-norm-matched controls。
- 结论：待核对与实验。
- 最后更新：2026-08-13

## 一般问题

### Q-003：纯文本推理后训练在提升空间推理时，如何避免视觉感知退化？

- 状态：open
- 来源论文：[VOLD](papers/vold.md)（E-006）
- 当前认识：VOLD 的 MMStar/MME reasoning 提升，但 perception 子项下降；冻结视觉塔并不能完全防止语言解码器侧的感知能力退化。
- 相互冲突的证据：HallusionBench 和 MME 总分提升，说明退化并非所有通用视觉指标上一致发生。
- 需要查阅：reasoning-focused VLM post-training 中的 perception-reasoning trade-off，以及 rehearsal、multi-objective regularization。
- 可验证实验：在文本 RL 中混入少量视觉感知 replay，或增加相对 base 的视觉输出 KL，并同时评测空间 reasoning 与 perception。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-006：如何统一控制 OPD 的状态漂移、教师时间漂移、监督密度、梯度冲突、privilege gap、介入阶段、轨迹证据质量、递归 target 漂移、跨阶段 credit、correction attribution、student compatibility、input-grounded reliability 与 modality balance？

- 状态：open
- 来源论文：[VOLD](papers/vold.md)（E-002、E-003）；[Vision-OPD](papers/vision-opd.md)（E-007）；[VA-OPD](papers/va-opd.md)（E-002–E-004）；[VGS](papers/vgs.md)（E-002–E-004）；[ViCuR](papers/vicur.md)（E-001、E-008）；[ViGOS](papers/vigos.md)（E-001、E-009）；[V-Zero](papers/v-zero.md)（E-002、E-008）；[VCSD](papers/vcsd.md)（E-002、E-012）；[PCD](papers/pcd.md)（E-002、E-003）；[VAD](papers/vad.md)（E-001–E-003）；[FP-OPD](papers/fp-opd.md)（E-002、E-010）；[SA-OPD](papers/sa-opd.md)（E-001、E-009）；[OPD-V](papers/opd-v.md)（E-002、E-010）
- 当前认识：前十篇分别控制 state、time、density、gradient、recoverability、stage、evidence、recursive target、cross-stage credit与source attribution；FP-OPD新增student compatibility，SA-OPD新增input-grounded reliability，OPD-V新增modality-balance trust region。
- 相互冲突的证据：十三篇论文尚无系统联合；各类 gate/target 都可能受 teacher error、intervention validity、support truncation、probe incompleteness、proxy cancellation、margin scaling 与目标冲突影响。
- 需要查阅：EMA self-distillation、confidence gating、token importance、gradient surgery、learning with privileged information、information leakage 与 mixture-of-teachers routing。
- 可验证实验：alignment × teacher update × token selection × gradient control × privilege type（answer/cue/crop/none）× intervention stage × target support 消融；跟踪 NLL、gradient cosine、leakage、routing 占比、support drift、student counterfactual sensitivity 和任务 gap。
- 结论：待实验。
- 最后更新：2026-08-13

### Q-008：Correctness 与 visual/spatial advantage 应如何联合控制蒸馏？

- 状态：open
- 来源论文：[VOLD](papers/vold.md)（E-003）；[VA-OPD](papers/va-opd.md)（E-004）；[VGS](papers/vgs.md)（E-003、E-007）；[V-Zero](papers/v-zero.md)（E-002、E-008）；[PCD](papers/pcd.md)（E-002、E-003）；[VAD](papers/vad.md)（E-001–E-003）
- 当前认识：VOLD 用 reward mask；VA-OPD 选择视觉依赖位置；VGS 改变视觉梯度；V-Zero 使用 relative crop support；PCD 定位 failure stage；VAD 重构 intervention-attributable signed correction。Correctness、证据支持、监督位置、更新方向、failure stage 和 target content 是互补控制轴。
- 相互冲突的证据：高 VA 轨迹可能仍然错误，单独按 VA softmax 会提高其权重；而正确轨迹也可能依赖语言捷径，单独 reward masking 无法增强视觉 grounding。
- 需要查阅：multi-objective distillation、advantage-conditioned imitation、per-token verifier 与 visual grounding reward。
- 可验证实验：按 correctness×absolute evidence×relative evidence rank 分桶，并比较乘法门控、all-low group rejection、分层 loss、VGS、gradient projection 与 Pareto weighting；评测答案准确率、student counterfactual sensitivity 和语言保持。
- 结论：待实验。
- 最后更新：2026-08-13

## 已解决

暂无。问题解决后移到这里，保留答案和证据，不要直接删除。

## 条目格式

```markdown
### Q-001：问题描述

- 状态：open / reading / experiment / resolved
- 来源论文：
- 当前认识：
- 相互冲突的证据：
- 需要查阅：
- 可验证实验：
- 结论：
- 最后更新：YYYY-MM-DD
```

