# Vision-OPD

## 基本信息

- 论文标题：Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
- 作者：Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, Yaojie Lu
- 年份与会议：2026；arXiv 预印本（v4，2026-06-02；未注明正式会议）
- arXiv/DOI：arXiv:2605.18740；DOI: 10.48550/arXiv.2605.18740
- 论文链接：https://arxiv.org/abs/2605.18740
- 本地文件：无
- 阅读状态：已完成（实现细节有待核对项）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

Vision-OPD 用同一 MLLM 的“局部裁剪视图策略”作为教师、“带框全图视图策略”作为学生，在学生自己生成的前缀上进行 top-100、token 级 JSD 自蒸馏；仅用 6.2K 个全图—裁剪—问题三元组，无需答案标签、奖励验证器或推理时缩放工具，即可显著缩小细粒度视觉任务中的区域—全局感知差距。

## 核心问题

细粒度视觉问答中的关键证据通常只占全图很小区域。论文首先验证：同一 MLLM 看证据中心裁剪时，准确率比看全图高 18–22 个点，说明不少失败来自“全局上下文中无法聚焦已具备识别能力的局部证据”，而不一定是局部识别能力缺失。

论文研究能否把推理时 crop/zoom 的优势在训练阶段内化：不调用更强答案教师，不依赖答案标签、reward verifier 或推理时图像工具，仅以模型自身在特权裁剪视图下的 token 分布，指导标准全图输入策略。

## 方法

### 整体流程

1. 从原始图像中识别、分割候选对象，只保留面积比小于阈值 \(\tau\) 的小区域。
2. 用 Qwen3.5-397B 针对候选区域生成可由该区域独立回答的细粒度问题。
3. 在原图上叠加候选区域的红色边界框，并在问题中追加“只关注红框内对象”等空间约束，形成学生输入 \(x,q\)；将区域裁剪并放大 2 倍，形成教师输入 \(x',q\)。
4. 从同一 MLLM 初始化两份条件策略：学生看带框全图，教师看证据裁剪。学生先生成 on-policy response \(y\)。
5. 教师与学生在相同的学生前缀 \(y_{<n}\) 上计算下一 token 分布，以 token 级 divergence 训练学生；教师分支 stop-gradient，并通过 EMA 正则化更新。
6. 推理时只保留标准全图单次前向，不再裁剪、搜索或重复编码图像。

### 关键模块

**区域—全局配对监督。** 教师和学生的语言模型起点相同，能力差异来自视觉条件：教师的裁剪视图隔离并放大证据，学生的全图视图更接近真实推理条件。这是 learning with privileged information，而非“大模型教小模型”。

**学生轨迹上的密集监督。** 轨迹由全图学生生成，教师只重评分同一前缀，避免用教师轨迹 SFT 时的 prefix/state distribution mismatch。每个位置都有分布级信号，不会像二值 RLVR 那样在组内全对或全错时失去 advantage。

**Top-\(K\) logits distillation。** 主配置保留学生 top-100 token 及对应教师 logits，并增加尾部概率项，以降低全词表蒸馏的显存成本。作者称其场景中 top-100 以外概率质量通常小于 \(10^{-13}\)。表 6 显示 top-\(K\) 比仅对 student-sampled token 做 policy-gradient-style shaping 的平均分高 1.06 点（79.68 vs 78.62）。

**教师正则化。** 当前学生直接作为教师会共同适应并彻底崩溃；冻结初始教师、trust-region、EMA 都能稳定训练。主配置采用 EMA，更新系数 \(\alpha=0.05\)，平均分略高于冻结教师（79.68 vs 79.40）。

### 损失函数与训练目标

传统 off-policy supervised distillation（式 1）在数据集轨迹前缀上匹配教师和学生：

\[
\mathcal{L}_{\mathrm{SD}}(\theta)
=\mathbb{E}_{(x,y)\sim\mathcal S}
\left[\frac{1}{|y|}\sum_t
D\!\left(p_T(\cdot\mid y_{<t},x)\|p_S(\cdot\mid y_{<t},x)\right)\right].
\]

通用 OPD（式 2）改为学生自己采样的轨迹 \(\hat y\sim p_S\)。Vision-OPD 对一个学生 response 的平均 divergence 定义为（式 3）：

\[
D(p_T\|p_S)(y\mid x,x',q)
=\frac{1}{|y|}\sum_{n=1}^{|y|}
D\!\left(
p_T(\cdot\mid x',q,y_{<n})
\|p_S(\cdot\mid x,q,y_{<n})
\right).
\]

总目标（式 4）为：

\[
\mathcal L_{\mathrm{Vision\text{-}OPD}}(\theta)
=\mathbb E_{(x,x',q)\sim\mathcal D}
\mathbb E_{y\sim p_S(\cdot\mid x,q)}
\left[D(p_T\|p_S)(y\mid x,x',q)\right].
\]

主实验使用 \(\mathrm{JSD}_{0.5}\)：

\[
\mathrm{JSD}_{\beta}(p_T\|p_S)
=\beta D_{\mathrm{KL}}(p_T\|m)
+(1-\beta)D_{\mathrm{KL}}(p_S\|m),\quad
m=\beta p_T+(1-\beta)p_S.
\]

只对学生分支反向传播，教师作为 stop-gradient target。需注意：式 (3)/(4) 写 \(D(p_T\|p_S)\)，算法 1 第 7 行却写 \(D(p_S\|\operatorname{stopgrad}(p_T))\)。对主配置 \(\mathrm{JSD}_{0.5}\) 两者等价，但对 forward/reverse KL 不等价，源码实现方向仍需核对。

### 训练数据与训练流程

- 数据由全图 \(x\)、2 倍放大的证据裁剪 \(x'\)、细粒度问题 \(q\) 组成，共 6.2K 个三元组。
- 问题由 Qwen3.5-397B 基于候选区域生成；候选区域来自对象识别与分割，并按面积比 \(<\tau\) 过滤。
- 为 SFT、GRPO、DAPO、OPSD 对照组生成答案标签时，Qwen3.5-397B 对区域多次作答，仅保留严格多数共识 \(>0.75\) 的问题；Vision-OPD 本身不使用这些答案标签。
- 学生/教师 backbone：Qwen3.5-4B 与 Qwen3.5-9B，non-thinking mode。
- divergence：top-100 logits 近似的 \(\mathrm{JSD}_{0.5}\)，包含 tail-probability term。
- 教师：EMA regularization，更新系数 \(\alpha=0.05\)。
- 最长 on-policy generation：1024 tokens；训练 1 epoch。
- 论文未报告 optimizer、learning rate、batch size、图像候选区域阈值 \(\tau\)、采样温度及完整算力配置，复现仍依赖代码。

## 实验设置

- 模型：Qwen3.5-4B/9B；均使用 non-thinking mode
- 训练数据：6.2K 个合成全图—裁剪—问题三元组
- 细粒度数据集：V* Bench、ZoomBench、HR-Bench 4K/8K、MME-RealWorld EN/CN
- Holdout 数据集：MMVP、CV-Bench、MMStar、POPE
- Baseline：原始 Qwen3.5；SFT on Self-Teacher；GRPO；DAPO；OPSD；DeepEyes、Thyme、DeepEyesV2、SenseNova-MARS 等 agentic 方法；多种开源与闭源 MLLM
- 指标：各 benchmark accuracy（%）、六项细粒度 benchmark 宏平均、ZoomBench 平均单样本耗时的倒数
- 关键超参数：JSD \(\beta=0.5\)，top-\(K=100\)，EMA \(\alpha=0.05\)，rollout max length 1024，1 epoch

## 实验结论

1. **同底座各细粒度任务均提升。** 9B 平均分由 73.37 提至 79.68（+6.31）；4B 由 70.68 提至 77.07（+6.39）。9B 在 V* Bench/ZoomBench 上分别由 82.72/52.07 提至 94.76/65.80。
2. **主表中 9B 的平均分高于列出的更大开源、闭源及 agentic 模型。** Vision-OPD-9B 平均 79.68；Qwen3.5-397B 为 77.44，Gemini-3.1-Pro 为 79.25，SenseNova-MARS 为 73.05。该比较受模型版本、评测设置和是否多步推理等差异影响，不等于通用能力全面超过。
3. **相同训练数据和 backbone 下，Vision-OPD 优于 SFT、GRPO、DAPO、OPSD。** 对 9B，四项细粒度任务均最高；尤其 ZoomBench 为 65.80，而 SFT/GRPO/DAPO/OPSD 分别为 58.46/57.51/55.62/57.51。
4. **Holdout 未显示灾难性遗忘。** 9B 在 MMVP、CV-Bench、MMStar、POPE 上相对 vanilla 分别变化 +0.34、+0.11、+0.13、+0.25；相比之下 SFT 与 RLVR 多项明显下降。但这些小幅正差异未报告随机种子和误差，稳妥结论是“基本保持”，不是已证明普遍改善。
5. **教师稳定化是必要条件。** 不正则化、用 current policy 当 teacher 时六项平均仅 0.59；冻结初始策略、trust-region、EMA 分别为 79.40、79.22、79.68。
6. **JSD 略优于两个 KL 方向。** JSD、forward KL、reverse KL 平均为 79.68、78.70、78.53；仅单次结果且差距小于 1.2 点。
7. **更长 rollout 和更密集的分布监督更有效。** 1024 token 优于 512 token（79.68 vs 78.62）；top-\(K\) logits 优于 sampled-token distillation（79.68 vs 78.62）。
8. **区域—全局差距随训练缩小。** 图 5 显示 4B/9B 的 gap 持续下降，终点低于列出的更大及闭源模型；正文未给出图中精确数值。
9. **单次前向比多步 zoom agent 更快。** 附录图 6 报告 Vision-OPD-9B 在所比 agentic baseline 中推理速度最快，但未提供图中具体 wall-clock 数值、硬件和吞吐表。

## 局限性

### 论文已呈现或可直接确认

- 训练仍需要证据中心 crop、全图 bounding box 和问题三元组；“无标签”仅指不需要答案标签或 reward verifier，不表示无需区域级特权信息或外部数据合成模型。
- 问题生成使用 Qwen3.5-397B，因此“no external teacher”准确地指没有外部答案分布教师；完整 pipeline 仍依赖一个大型外部模型生成训练问题。
- 主实验只覆盖 Qwen3.5 的 4B、9B 两个规模，跨架构和不同视觉 tokenizer 的泛化未验证。
- 训练细节不完整，缺少 optimizer、learning rate、batch size、\(\tau\)、采样配置、EMA 精确定义和训练资源。
- 细粒度评测均是选择式或短答案型；开放式描述、无显式框提示、视频与 3D 空间任务未验证。
- 未报告多随机种子、置信区间或显著性检验；小于约 1 point 的消融或 holdout 差异不宜过度解释。

### 个人分析

- 学生输入不是未经标注的自然全图，而是带红色框且问题含显式空间约束。方法证明的是“从带框全图中读取小证据”的内化，不足以证明模型学会在没有定位提示时自行搜索相关区域。
- crop 被放大 2 倍，师生差异同时包含上下文隔离、空间分辨率和尺度变化；当前实验无法分离哪一项是主要监督来源。
- 所谓 self-distillation 并非完全静态的“同一模型两个输入”：实际稳定版本维护冻结或 EMA teacher。EMA 的参数更新顺序和 teacher 是否含独立视觉/语言参数副本会影响复现成本。
- 表 1 跨模型比较可能受输入分辨率、prompt、模型发布日期和评测 harness 差异影响；最可靠证据是表 2 的同数据、同 backbone 对照。
- Top-100 以外概率质量小于 \(10^{-13}\) 对大词表语言模型而言异常极端，需确认是未归一化截断方式、数值精度现象还是特定低温分布。
- 训练只进行 1 epoch 且仅 6.2K 数据，结果很强，但缺少数据量曲线、训练方差和独立复现，尚不能判断扩展规律。
- 算法 1 与式 (3)/(4) 的 divergence 参数顺序不一致；主 JSD 配置掩盖了这一差异，但表 4 的 forward/reverse KL 名称必须结合源码确认。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可把“更易求解的特权空间视图”作为同模型教师条件，例如目标裁剪、局部地图、可见对象子图或带真值位姿的视图；学生仍使用部署时的完整场景输入。
- 在学生自己的空间推理轨迹上，让特权视图教师提供 token 级 soft target，可避免直接 SFT 特权视图回答造成的 state-distribution mismatch。
- 若全词表 logits 蒸馏显存过高，可实现 top-\(K\)+tail mass；但需要先测本项目分布下 top-\(K\) 覆盖率，不能直接照搬 \(K=100\)。
- 自蒸馏教师必须稳定化。最小实验应至少比较 frozen initialization 与 EMA teacher，不能直接用同步 current policy 作为 target。
- 同时跟踪“特权输入准确率—部署输入准确率”的 gap，比只看最终 benchmark 更能判断知识是否从易视图转移到难视图。

### 个人建议

- 构造四组分解实验：完整场景；场景+框；纯 crop；crop+放大，分别测准确率和 teacher entropy，以区分定位、上下文干扰与分辨率收益。
- 若项目部署时没有红框，应增加无框学生组，或把框/目标坐标逐步退火掉；否则训练目标与真实“自主聚焦”目标不一致。
- 对照 VOLD 的失败轨迹 masking，可研究只在 crop teacher 比 global student 更可信时蒸馏，例如按 teacher entropy、区域—全局 logit margin 或可验证空间状态门控；本文没有验证该扩展。
- 复现时记录 EMA 更新公式、参数副本范围、top-\(K\) union 规则、tail bucket 定义和 KL 方向，并报告峰值显存与额外前向成本。
- 评测需同时包含细粒度空间感知、无框定位、通用视觉和长链空间推理，防止只学会服从红框提示。

## 与其他论文的相同点和冲突

与 [VOLD](vold.md) 的共同点：

- 都在 student-generated prefixes 上施加 token 级分布监督，以降低 off-policy imitation 的 exposure bias。
- 都强调 dense teacher guidance 可补足稀疏序列奖励；两者都需要控制 teacher/student 训练动态，避免无效或不稳定指导。
- 都把部署时不可得或不便使用的信息只放在训练教师侧：VOLD 是更强文本模型的推理能力，Vision-OPD 是 evidence-centered crop。

关键差异：

- VOLD 是跨模型、文本教师到 VLM 学生的能力迁移，需要同一教师轨迹 SFT 对齐，再联合 GRPO，并依赖答案 verifier；Vision-OPD 是同源模型的跨视觉条件自蒸馏，不使用 RL reward 或答案标签。
- VOLD 仅对错误轨迹蒸馏以避免干扰正确探索；Vision-OPD 对所有学生 token 做 JSD，但通过 EMA/frozen teacher 防止共同崩溃。
- VOLD 的视觉塔冻结，主要迁移语言侧推理；Vision-OPD 直接改变全图条件下的细粒度视觉行为，但论文未说明哪些模块冻结。
- VOLD 的 teacher/student 策略差距过大时指导失效，需要 SFT alignment；Vision-OPD 初始 teacher/student 参数相同，仅输入条件不同，但同步更新反而崩溃，需要时间尺度上的 teacher regularization。

与 [VA-OPD](va-opd.md) 的关系：

- 两者都使用视觉条件差构造无需答案 reward 的特权监督。Vision-OPD 让同模型教师看隔离、放大的 evidence crop；VA-OPD 让跨规模视觉教师比较原图和细节退化图，并将差值用作 token/rollout 权重。
- Vision-OPD 直接改变 teacher target 的视觉条件，重点解决全图—局部证据 gap；VA-OPD 的 KL target 始终来自原图教师，退化图只用于识别哪些 token 的视觉监督最重要。
- Vision-OPD 需要 bounding box/crop 和 teacher 时间稳定化；VA-OPD 不需要区域标注且教师固定，但每条 rollout 多一次退化图 teacher forward。
- Vision-OPD 对所有 token 做 top-100 JSD；VA-OPD 显示 uniform token 平均会稀释视觉关键位置。把 VA grouped weighting 应用于 crop-teacher JSD 是合理研究方向，但尚无直接实验证据。

与 [VGS](vgs.md) 的关系：

- 两者都从视觉条件差中构造 privileged supervision。Vision-OPD 使用 evidence crop/full-image 差，VGS 使用 image/text-only distribution 差。
- Vision-OPD 重点控制 self-teacher 的时间稳定性；VGS 使用固定 teacher，重点分析 language/visual gradient 的正交与冲突。
- VGS 的 gradient steering 可用于 crop teacher，但 crop 丢失全局关系时，应先判断 privileged teacher 是否真的更可靠。

与 [ViCuR](vicur.md) 的关系：

- 两者都主张用 visual-side evidence privilege 代替答案/rationale privilege。Vision-OPD 使用 evidence crop，ViCuR 使用视觉 cue text。
- Vision-OPD 的 crop 来源明确但 student 依赖红框；ViCuR student 无框、无 cue text，但增加 sink recovery module，且 cue generation provenance 尚不清楚。
- ViCuR 的 recoverability 框架可用于审计 Vision-OPD：crop/box 确实来自原图，但“带框学生”与无框部署之间仍可能存在输入接口 gap。

与 [ViGOS](vigos.md) 的关系：

- 两者都使用同 backbone 的视觉条件 teacher 并保持 student-prefix scoring。Vision-OPD 让 teacher 看 evidence crop；ViGOS 的 perception teacher 只看原图，不接收外部 question/answer。
- Vision-OPD 通过更易视觉输入提高 teacher 感知可靠性；ViGOS 通过 description-first routing 阻止答案特权过早介入。可组合为 crop/image teacher→description、answer teacher→reasoning。

与 [V-Zero](v-zero.md) 的关系：

- 两者都使用 question-relevant crop 作为 teacher-side privilege，student 与推理阶段只看全图。
- V-Zero 增加 random negative crop 和 sibling trajectory gate，但使用更大外部教师；其消融尚未完全分离 positive crop target 与 gate。Vision-OPD 则重点研究同模型 crop/full-image distillation 与 teacher 时间稳定性。

与 [VCSD](vcsd.md) 的关系：

- 两者都使用 EMA self-teacher。Vision-OPD 通过 evidence crop 制造 teacher 输入优势；VCSD 通过原图/content-erased full-vocabulary contrast制造不对称，不需要 region annotation。
- 两者都依赖 EMA 时间尺度；VCSD 还需用 plausible support 防止 contrast-shaped target 递归漂移。

与 [PCD](pcd.md) 的关系：

- Vision-OPD 改善 perception teacher target，PCD 则根据 downstream failure 与 teacher gap 决定哪些 perception 应获得更强 target correction。
- Privileged crop teacher 可作为 PCD 的 disagreement witness，但 crop 缺少全局信息时需要 teacher-confidence gate。

与 [VAD](vad.md) 的关系：

- 两者使用同一 6,241 数据与 crop/full-image setup。Vision-OPD 直接匹配完整 crop teacher；VAD 只重构与 crop/degraded intervention 对齐的 signed correction，并以弱 direct teacher作稳定 anchor。

与 [FP-OPD](fp-opd.md) 的关系：

- FP-OPD 不使用 crop privilege，而把外部 teacher gap投影到 student visual-feature probes张成的 Fisher response space；它处理的是 target capacity mismatch。

与 [SA-OPD](sa-opd.md) 的关系：

- SA-OPD 不引入 privileged crop，而用 no-prompt control过滤输入不敏感的高divergence位置；其control更轻，但不如region intervention聚焦。

与 [OPD-V](opd-v.md) 的关系：

- OPD-V复用Vision-OPD crop/data，并增加masked-crop teacher，以正sampled-token margin筛选和加权crop-teacher JSD。

十三篇已读论文共同提示：OPD 的有效性不只取决于“是否 on-policy”，还取决于状态可靠性、teacher 时间稳定性、监督密度、模态梯度方向、privilege 可恢复性、介入阶段、轨迹证据质量、递归 target 漂移、跨阶段 credit、correction attribution、student-local compatibility、input-grounded reliability 及 modality balance。

## 证据

### E-001

- 结论：多个 MLLM 在 ZoomBench 上存在 18–22 个点的区域输入相对全图输入优势。
- 类型：论文结论
- 定位：§3.1；图 2、图 3
- 必要引用：正文明确报告 regional-input accuracy consistently exceeds global-input accuracy by 18–22 points。
- 备注：图中各模型精确数值未在正文表格列出。

### E-002

- 结论：Vision-OPD 用同一 MLLM 的 crop-conditioned teacher 在学生全图 rollout 的每个 prefix 上提供 token 级分布监督。
- 类型：论文结论
- 定位：§3.2；图 4；式 (3)、式 (4)；算法 1
- 必要引用：无
- 备注：梯度仅通过学生分支。

### E-003

- 结论：Vision-OPD 的 6.2K 训练样本是全图、证据裁剪与问题三元组，主方法不使用生成的答案标签。
- 类型：论文结论
- 定位：§3.2 数据构造段落
- 必要引用：答案生成和 \(>0.75\) 共识过滤用于 SFT、RLVR、OPSD 对照。
- 备注：问题本身由外部 Qwen3.5-397B 生成。

### E-004

- 结论：主训练配置为 top-100 \(\mathrm{JSD}_{0.5}\)、EMA teacher、1024 token rollout、1 epoch。
- 类型：论文结论
- 定位：§4.1；§4.3.1–§4.3.4
- 必要引用：EMA 更新系数 \(\alpha=0.05\) 见 §4.3.1。
- 备注：其余优化超参数未报告。

### E-005

- 结论：Vision-OPD-4B/9B 相对相同 Qwen3.5 base 的六项细粒度平均分分别提升 6.39/6.31 点。
- 类型：论文结论
- 定位：§4.2.1；表 1
- 必要引用：4B 70.68→77.07；9B 73.37→79.68。
- 备注：六项包括 HR-Bench 与 MME-RW 的不同分辨率/语言子项。

### E-006

- 结论：同数据同 backbone 下，Vision-OPD 在报告的细粒度任务上优于 SFT、GRPO、DAPO、OPSD，并基本保持 holdout 能力。
- 类型：论文结论
- 定位：§4.2.2；表 2
- 必要引用：9B holdout 相对 vanilla 的变化均在 +0.11 至 +0.34 之间。
- 备注：无多随机种子，不能确认这些小幅 holdout 改善是否显著。

### E-007

- 结论：不正则化的 current-policy teacher 导致近零准确率崩溃，EMA teacher 最佳。
- 类型：论文结论
- 定位：§4.3.1；表 3
- 必要引用：Current Policy 平均 0.59；Initial 79.40；Trust-Region 79.22；EMA 79.68。
- 备注：EMA 相对 frozen 的优势仅 0.28 点，稳定性必要不等于 EMA 显著优于 frozen。

### E-008

- 结论：JSD、1024-token rollout 和 top-\(K\) logits 分别优于对应 KL、512-token 和 sampled-token 变体。
- 类型：论文结论
- 定位：§4.3.2–§4.3.4；表 4–6
- 必要引用：JSD 79.68；forward/reverse KL 78.70/78.53；1024/512 为 79.68/78.62；top-\(K\)/sampled-token 为 79.68/78.62。
- 备注：均为 Qwen3.5-9B 单组报告结果。

### E-009

- 结论：Vision-OPD 训练期间区域—全局准确率差距持续缩小。
- 类型：论文结论
- 定位：§4.3.5；图 5
- 必要引用：无
- 备注：正文没有提供精确终点值。

### E-010

- 结论：“无需外部教师”不等于数据 pipeline 不使用外部大模型或区域级特权信息。
- 类型：个人核对
- 定位：摘要；§3.2
- 必要引用：Qwen3.5-397B 用作 question generator；训练输入包含检测/分割所得 bounding box 和 crop。
- 备注：论文的准确主张是无需 external teacher model 提供蒸馏答案分布。

### E-011

- 结论：论文未证明无框全图上的自主证据定位能力。
- 类型：个人推断
- 定位：§3.2；图 4
- 必要引用：学生全图叠加 bounding box，问题追加显式空间约束。
- 备注：benchmark 原始输入是否均含同类框提示需结合数据处理代码核对。

### E-012

- 结论：公式与算法对 divergence 参数顺序的记录不一致。
- 类型：个人核对
- 定位：式 (3)、式 (4)；算法 1 第 7 行
- 必要引用：前者写 \(D(p_T\|p_S)\)，后者写 \(D(p_S\|\operatorname{stopgrad}(p_T))\)。
- 备注：对 \(\mathrm{JSD}_{0.5}\) 无影响；forward/reverse KL 实现方向 `待核对`。

## 待验证问题

- [ ] 源码中 forward/reverse KL 的实际参数顺序是什么？表 4 的命名是否与实现一致？
- [ ] EMA teacher 的精确更新公式、更新时机和覆盖模块是什么？视觉塔、projector、LLM 是否全部训练？
- [ ] 去掉学生输入的 bounding box 和问题空间约束后，收益能保留多少？
- [ ] crop 隔离上下文与 2 倍放大分辨率分别贡献多少？
- [ ] top-100 以外概率质量小于 \(10^{-13}\) 如何测得？不同温度、任务和模型下是否成立？
- [ ] 多随机种子下 JSD 相对 KL、EMA 相对 frozen teacher 的小幅优势是否显著？
- [ ] 训练成本相对 SFT/RLVR 如何？双视觉前向和 top-\(K\) teacher scoring 的显存、吞吐代价是多少？
- [ ] 在 SpatialStack 的无框空间定位、3D/视频和长链规划任务中，特权视图 OPD 是否仍有效？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v4 全文，包括正文 §1–6、式 (1)–(4)、算法 1、表 1–6、图 1–5，以及附录 A–C、图 6、表 7。
- 新增认识：同一模型可利用输入条件造成的能力差作为 privileged teacher；对自蒸馏而言，teacher target 的时间稳定性与 student-state 对齐同样关键。
- 修正内容：将“无标签”限定为不使用答案标签和 verifier；未把带框全图结果解释为已学会无提示自主定位；记录公式与算法的 divergence 顺序不一致。
- 下一步：核对公开代码中的 teacher EMA、模块冻结、top-\(K\) tail 实现与 KL 方向，并做无框/不放大/纯裁剪分解实验设计。
