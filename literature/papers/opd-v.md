# OPD-V

## 基本信息

- 论文标题：OPD-V: Visual On-Policy Self-Distillation with Modality Balance
- 作者：Aniri, Jinhe Bi, Peng Liao, Zengjie Jin, Volker Tresp, Fei Shen, Yunpu Ma, Tat-Seng Chua
- 年份与会议：2026；arXiv 预印本（v2）
- arXiv/DOI：arXiv:2608.05131；DOI: 10.48550/arXiv.2608.05131
- 论文链接：https://arxiv.org/abs/2608.05131
- 代码：https://github.com/aniri15/OPD-V
- 本地文件：无
- 阅读状态：已完成（mask配置与 rollout-correction实现待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

OPD-V 用同一 EMA teacher 分别查看 evidence-centered zoom crop 与其随机矩形 mask 版本，以 sampled-token 正 log-prob margin选择并加权位置，再把 zoom-teacher top-100+tail distribution通过JSD蒸馏到原图student；其 matched 4B平均分和wall-clock结果很强，但“modality balance”实际由视觉干预敏感性间接代理，mask validity、错误token强化、loss尺度与效率比较仍存在关键混杂。

## 核心问题

视觉 OPSD 常给 teacher额外crop、visual thought或答案，但强文本先验仍可能主导预测，使 privileged visual information未被充分利用。论文将这种现象定义为 Modality Imbalance，并研究：

1. 能否把visual/text modality balance本身作为training privilege；
2. 如何用显式图像条件产生不同程度的modality imbalance；
3. 如何从positive/negative teacher差异中选择视觉相关token；
4. 这种选择能否同时提高准确率、控制response length并降低实际训练时延。

## 方法

### Three matched image conditions

Student在Original Image \(I\) 上生成：

\[
y\sim p_\theta(\cdot\mid I,x).
\]

同一EMA teacher在相同query和student prefix下使用：

\[
q_t^+
=p_{\bar\theta}(\cdot\mid I^{zoom},x,y_{<t}),
\]

\[
q_t^-
=p_{\bar\theta}(\cdot\mid I^{mask},x,y_{<t}).
\]

\(I^{zoom}\) 是Vision-OPD数据提供的evidence-centered bbox crop并放大；\(I^{mask}\) 在该crop上随机选择矩形区域替换为黑色。Positive teacher提供target，negative teacher只用于token selection/weighting。

### Modality-Balance Attention Ratio

论文用完整response上的visual/text attention mass ratio诊断modality imbalance：

\[
\rho(c;y)
=
\frac{\sum_t\alpha_t^V(c)}
{\sum_t\alpha_t^T(c)+\epsilon}.
\]

5K样本分析显示negative teacher、original-image student、positive teacher的ratio依次增大，且positive/negative logit margin越大时attention-ratio gap与student correctness越高。

该attention ratio只用于动机分析，不进入训练loss。Attention mass也不等同于causal modality contribution，尤其论文没有完整说明layer/head aggregation和Qwen3.5 hybrid layers的处理。

### Modality-Balance Logits Margin

对student sampled token \(y_t\)：

\[
\delta_t^{MB}
=
\log q_t^+(y_t)
-
\log q_t^-(y_t).
\]

Trust region：

\[
\mathcal R_{MB}(y)
=
\{t:\delta_t^{MB}>0\}.
\]

它保留zoom teacher比masked-crop teacher更支持student token的位置。更准确地说，这是selected visual intervention sensitivity，而不是visual/text contribution的直接分解。

### OPD-V objective

\[
\mathcal L_{\mathrm{OPD-V}}
=
\mathbb E\left[
\frac1{\sum_t r_t}
\sum_{t\in\mathcal R_{MB}}
r_t\delta_t^{MB}
D_{JS}(q_t^+,p_{\theta,t})
\right].
\]

特点：

- 只使用正margin；
- margin既决定binary selection又连续加权；
- 分母是所有valid generated tokens，不是selected count或weight sum；
- teacher、margin和target均detached；
- negative teacher不作为distribution target。

因此selected fraction和margin scale会共同改变总gradient scale。附录超参数表另列“rollout correction threshold=2.0”，但正文公式与算法均使用未clip的 \(\delta_t^{MB}\)，是否实际clip及其位置 `待核对`。

### Top-\(K\)+tail JSD

Student top-100构成候选集合：

\[
\mathcal K_t=\operatorname{TopK}(p_{\theta,t},100).
\]

Student与positive teacher都保留这些candidate probabilities，并用单一tail bucket保存其余质量；在101维distribution上计算JSD。Negative teacher只需sampled token probability。

论文称top-100之外累计质量低于 \(10^{-13}\)。对一般LM token distribution而言该量异常小，需核对它是平均值、数值截断、特定位置统计还是表述错误。

### EMA teacher

\[
\bar\theta
\leftarrow
(1-\tau)\bar\theta+\tau\theta,
\qquad \tau=0.05.
\]

即每步保留0.95旧teacher，加入0.05当前student。

## 训练数据与训练流程

- 数据：Vision-OPD 6,241条synthetic visual reasoning samples。
- 每条含Original Image、text query、verified target、bbox evidence crop。
- Backbones：Qwen3.5-4B/9B、Qwen3-VL-4B/8B-Instruct。
- Hardware：1 node，4×H200。
- Batch size 48；actor mini-batch 48。
- 每prompt 8 rollouts。
- Max prompt 8192；max response 1024。
- 训练1 epoch；checkpoint/validation均禁用，使用最终训练状态。
- JSD，\(\beta=0.5\)；top-\(K=100\)+tail。
- EMA student coefficient 0.05。
- Rollout correction mode=Token，threshold=2.0，但正文未说明具体公式。
- Positive image field=`bbox_images`；negative view为zoom crop上的随机矩形black mask。
- 未报告optimizer、LR、scheduler、warmup、weight decay、gradient clipping、rollout temperature/top-p、mask面积/长宽比/位置分布和random seed。

## 实验设置

- Benchmarks：
  - V* Bench；
  - ZoomBench；
  - HR-Bench 4K/8K；
  - MME-RealWorld EN/CN。
- Average：六个benchmark/subset accuracy等权平均，不按32,162个样本加权。
- Evaluation：non-thinking；先deterministic matching，未解析样本交给Yes/No judge。
- Judge模型、prompt、agreement、重评稳定性未报告。
- Matched Qwen3.5-4B baselines：SFT、GRPO、OPSD、Vision-OPD、VA-OPD。
- Cross-model figure：Qwen3-VL-4B/8B、Qwen3.5-9B。
- Cross-family closed/open models只作背景，不是受控比较。

## 实验结论

1. **Matched Qwen3.5-4B平均80.01。** Base 64.30，绝对+15.71；最佳其他matched方法Vision-OPD 77.10，OPD-V +2.91。
2. **相对Vision-OPD六项均提高。** OPD-V在V*、Zoom、HR-4K、HR-8K、MME-EN、MME-CN分别为94.76、65.92、85.75、83.25、76.31、74.06；Vision-OPD为89.01、65.33、82.25、81.25、74.50、70.26。
3. **跨backbone均高于base。** Qwen3-VL-4B 68.00→74.14，8B 68.93→73.03，Qwen3.5-9B 69.75→77.63。
4. **跨backbone未给matched OPD逐项数值。** Figure 4显示Base/OPSD/OPD-V，但正文只集中报告base→OPD-V aggregate，不能仅凭摘要判断每项显著性。
5. **单teacher ablation支持组合收益。** Base 64.30，negative-only 71.98，positive-only 74.62，dual 80.01。
6. **但negative-only定义不够清楚。** Negative teacher按主方法只负责selection，仍需某个positive distribution作为target；论文未在正文明确该ablation的完整objective。
7. **Zoom+Mask是已测组合中最佳。** Repeat+Mask 75.95；Zoom+No Image 72.17、Zoom+Prune 73.74、Zoom+Blur 74.31、Zoom+Mask 80.01。
8. **Trust region长期约一半。** Warm-up后4B/9B平均selected fraction 53.8%/49.6%，说明gate未迅速退化为全选或全拒。
9. **Response length显著更短。** 最后10 steps：OPD-V 140.9 vs standard OPSD 553.8，-74.5%；但短response不是独立reasoning-quality指标。
10. **Policy entropy未塌缩。** 最终rolling means为4B 0.827、9B 0.761；没有base/OPSD同图直接阈值对照。
11. **Observed step latency更低。** 4B 352→240s（-31.8%）；9B 451→340s（-24.7%）。
12. **效率收益由trajectory length和teacher construction共同驱动。** 9B teacher preprocessing 79.2→15.3s；OPD-V双teacher combined forward 27.1s，低于OPSD单teacher40.4s。
13. **效率不是fixed-token算法成本优势。** OPD-V理论上多一个teacher pass；wall-clock比较同时改变response length、teacher prompt construction和输入长度，没有报告固定token吞吐。
14. **“4B超过闭源/超大模型”不是受控结论。** Prompt、judge、output mode、数据和模型版本均不同。

## 局限性

### 论文明确承认或设计中体现

- Full-vocabulary distillation内存高，因此采用student top-100+tail近似。
- 双teacher增加一次forward，但论文认为短rollout与图像式teacher输入抵消成本。

### 由实验设计可直接确认

- 无多训练seed、标准差、置信区间或显著性检验。
- 只测试Qwen3.5/Qwen3-VL和同一6,241数据；视频、多图、3D与开放空间导航未知。
- 训练full image可能含target-region cue，query含visual focus instruction；不能证明无框自主localization。
- Mask rectangle的面积、aspect ratio、采样分布与是否覆盖evidence未报告。
- 主评测含未知Yes/No judge，未报告judge一致性。
- Optimizer、LR、sampling和多数stability配置缺失。
- 没有与V-Zero、VCSD、VAD、VGS、SA-OPD或FP-OPD做matched比较。

### 个人分析

- **Modality balance是解释，不是训练量。** Loss只看zoom/mask sampled-token log-prob差，没有直接约束visual/text attention或causal contribution。
- **Attention ratio不等于信息使用。** Attention mass可受token数、layer/head选择、architecture和normalization影响；论文未证明ratio变化导致correctness变化。
- **Contrast混合多种因素。** Zoom crop改变scale和global context，black mask引入artifact且可能随机遮住无关区域；\(\delta^{MB}\)只隔离这组operations的model response。
- **Positive margin可能强化错误视觉token。** 若student token错误但对被mask像素敏感，仍会进入trust region并被zoom teacher JSD更新；无correctness或teacher-confidence gate。
- **Teacher target也可能缺全局信息。** Evidence crop强化局部细节但丢失scene context，尤其关系、计数和空间拓扑任务可能受损。
- **Objective scale不受控。** 分母固定为全部valid tokens，margin不按selected count/weight sum归一化；不同模型、step和mask draw的有效learning-rate会变化。
- **Threshold 2.0实现不透明。** 若是margin clipping，它是稳定性的关键组成，但论文公式/算法未体现。
- **Student-support top-\(K\)保守。** Teacher-only正确token若不在student top-100只能并入tail，无法被单独提升。
- **Top-100 tail \(<10^{-13}\) 可疑。** 对开放生成分布过于极端，需核对统计口径和BF16/softmax实现。
- **EMA使两teacher view difference随训练漂移。** 同一EMA参数有助于控制model difference，但target、margin和rollout同时变化，缺少frozen-teacher对照。
- **Efficiency comparison存在行为/实现混杂。** 更短输出是真实系统收益，但不能证明双teacher算法每token更便宜；需固定length与tokens/sec实验。
- **Verified target的角色前后不一致。** §2声明 \(a^\star\) 用于构造teacher supervision，但OPD-V equations、algorithm和image operations只使用 \((I,x)\)、zoom/mask和prefix；需确认答案是否实际进入teacher prompt。
- **跨论文matched数字不一致。** OPD-V报告Qwen3.5-4B Vision-OPD 77.10，而VAD同类六项表报告Vision-OPD 75.92；可能来自评测版本、judge或recipe差异，不能直接合并排名。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD可用positive/negative spatial views形成token trust region，只在证据增强比证据破坏更支持的位置蒸馏。
- Same-EMA paired teachers控制了teacher parameter差异，使contrast主要来自输入view。
- Positive target与negative selector分工明确，negative view无需作为模仿目标。
- 约一半token的长期选择率说明简单正margin gate可持续提供稀疏监督。

### 个人建议

- 用语义明确的空间干预替代随机黑mask：对象移除、关系边删除、深度/坐标擦除、target-preserving distractor mask。
- 加入correctness/teacher-confidence gate，避免强化image-sensitive wrong tokens。
- 分别报告raw margin、clip后margin、selected fraction和normalized weight；比较all-token、selected-count和weight-sum三种分母。
- 直接测causal modality contribution：image/text ablation log-ratio、integrated gradients或attention intervention，而不把attention ratio当证据使用量。
- 对fixed sequence length报告teacher/student tokens/s和GPU-hours，分离行为性短输出与算法开销。
- 用teacher/student union top-\(K\)加入high-margin teacher-only candidates。

## 与其他论文的相同点和冲突

### 与 [Vision-OPD](vision-opd.md)

- 两者使用同一6,241数据和evidence crop。Vision-OPD直接蒸馏crop EMA teacher；OPD-V增加masked-crop teacher，并只在positive margin位置蒸馏zoom target。
- OPD-V可视为Vision-OPD加contrastive token gate与margin weighting，但matched数字在不同论文间不完全一致。

### 与 [VA-OPD](va-opd.md)

- 两者都用visual contrast的sampled-token log-prob margin选择监督。
- VA-OPD比较original/pixelated teacher并对positive visual advantage加权；OPD-V比较zoom/masked-crop EMA teacher并将正margin定义为trust region。
- 两者都可能强化视觉相关但错误的token，且都未直接识别modality causal balance。

### 与 [V-Zero](v-zero.md)

- V-Zero在sibling trajectory层做positive/negative crop relative evidence gating；OPD-V在token层做zoom/mask正margin gate。
- OPD-V不做group-relative normalization，计算更直接；V-Zero使用trajectory competition但可能all-wrong。

### 与 [VCSD](vcsd.md)

- VCSD用original/content-erased EMA contrast塑造candidate distribution；OPD-V用zoom/mask contrast选择位置，并继续模仿完整positive teacher target。
- VCSD无需region annotation；OPD-V的bbox crop更聚焦但不能证明无框visual search。

### 与 [VAD](vad.md)

- VAD把crop/degraded contrast用于signed candidate-level target reconstruction；OPD-V只用sampled-token positive margin做selection/weighting。
- VAD能表达refutation，OPD-V忽略negative margins；OPD-V更简单但仍复制source-mixed zoom teacher。

### 与 [VGS](vgs.md) 和 [SA-OPD](sa-opd.md)

- 三者都关注文本先验压过输入证据。VGS重构visual target并steering gradient；SA-OPD删除no-prompt-invariant token；OPD-V选择zoom比mask更支持的位置。
- OPD-V的“modality balance”最接近input-intervention sensitivity，不能由attention correlation升级为可识别因果分解。

### 与 [FP-OPD](fp-opd.md) 和 [PCD](pcd.md)

- FP-OPD筛teacher gap direction的student-local compatibility；PCD筛perception failure stage；OPD-V筛positive visual-intervention token。
- 可组合为stage gate × evidence margin × Fisher-projected target，但必须控制多重gate导致的监督稀疏。

### 与 [ViCuR](vicur.md)、[ViGOS](vigos.md) 和 [VOLD](vold.md)

- ViCuR扩大evidence recovery，ViGOS路由privilege stage，VOLD用correctness mask保护成功轨迹；OPD-V用paired image conditions控制视觉相关token选择。
- OPD-V有verified target却未明确用于method teacher，和VOLD/ViGOS的answer-aware设计不同。

### 综合定位

十三篇论文新增“modality-balance trust region”维度：用positive/negative视觉view的sampled-token margin同时做hard selection与soft weighting；但其可靠结论是视觉干预敏感性有助于token allocation，而非已经直接测量或优化了模态平衡。

## 证据

### E-001

- 结论：OPD-V使用zoom positive teacher与masked-zoom negative teacher。
- 类型：论文结论
- 定位：§3.2；式 (6)；附录 B.2
- 必要引用：同一EMA参数、query和student prefix。
- 备注：mask rectangle配置未报告。

### E-002

- 结论：Positive sampled-token log-prob margin定义trust region。
- 类型：论文结论
- 定位：§3.3；式 (7)–(8)
- 必要引用：\(\delta^{MB}>0\)。
- 备注：是view sensitivity，不是直接modality attribution。

### E-003

- 结论：Margin同时选择并加权zoom-teacher JSD。
- 类型：论文结论
- 定位：§3.4；式 (9)；算法 1
- 必要引用：分母为所有valid tokens。
- 备注：未按selected count或weight sum归一化。

### E-004

- 结论：实现使用student top-100+tail JSD与EMA teacher。
- 类型：论文结论
- 定位：§4.1.1；附录 B.1、F
- 必要引用：\(\tau=0.05\)是current-student coefficient。
- 备注：teacher-only candidate受student support限制。

### E-005

- 结论：Matched Qwen3.5-4B average为80.01，Vision-OPD为77.10。
- 类型：论文结论
- 定位：表 1；§4.2
- 必要引用：相对base +15.71，相对Vision-OPD +2.91。
- 备注：v2 HTML主表转换失败，逐项值由可解析的v1表核对；无多seed。

### E-006

- 结论：四个backbone的base→OPD-V aggregate均提高。
- 类型：论文结论
- 定位：图 4；§4.2
- 必要引用：74.14/73.03/77.63等。
- 备注：跨backbone未完整报告matched方法表。

### E-007

- 结论：Dual teacher高于两个single-teacher variants。
- 类型：论文结论
- 定位：图 6(a)；§4.4
- 必要引用：71.98/74.62/80.01。
- 备注：negative-only objective定义需核对。

### E-008

- 结论：Zoom+Mask在测试的image-operation pairs中最好。
- 类型：论文结论
- 定位：图 6(b)；§4.4
- 必要引用：80.01 vs 72.17–75.95。
- 备注：不能证明mask因果隔离modality balance。

### E-009

- 结论：Observed step latency相对OPSD下降24.7%–31.8%。
- 类型：论文结论
- 定位：图 5；§4.3
- 必要引用：4B 352→240s，9B 451→340s。
- 备注：同时改变response length和teacher construction。

### E-010

- 结论：Attention-ratio correlation不足以识别modality contribution。
- 类型：个人核对
- 定位：图 1；式 (5)
- 必要引用：attention ratio不进入objective，aggregation细节不完整。
- 备注：训练信号实际是log-prob intervention contrast。

### E-011

- 结论：Verified target在method中的实际作用不清楚。
- 类型：个人核对
- 定位：§2.1–2.2 vs 式 (6)–(9)、算法 1
- 必要引用：前文声明 \(a^\star\) 构造teacher，OPD-V equations/algorithm未包含它。
- 备注：需核对prompt code。

### E-012

- 结论：Rollout correction threshold 2.0未出现在method公式或算法。
- 类型：个人核对
- 定位：附录表 4 vs 式 (9)、算法 1
- 必要引用：Policy correction mode=Token。
- 备注：可能是margin/ratio clipping，影响稳定性解释。

### E-013

- 结论：Top-100外质量低于 \(10^{-13}\) 的声明需要独立核验。
- 类型：个人核对
- 定位：§4.1.1
- 必要引用：student top-100+tail。
- 备注：统计口径未给出。

## 待验证问题

- [ ] Random mask的面积、长宽比、位置分布和evidence overlap是多少？
- [ ] Verified answer是否进入positive/negative teacher prompt？
- [ ] Rollout correction threshold 2.0具体clip哪个量？
- [ ] Negative-teacher-only ablation的target和loss如何定义？
- [ ] Attention ratio聚合哪些layers/heads，hybrid non-attention layers如何处理？
- [ ] Positive-margin wrong-token rate是多少？
- [ ] 按weight sum归一化后收益是否保持？
- [ ] Top-100 tail mass \(<10^{-13}\) 的统计口径是什么？
- [ ] 固定response length时OPD-V每token吞吐和GPU-hours如何？
- [ ] 无region cue/full-scene输入下收益是否保持？
- [ ] 多seed与judge重评后+2.91是否稳定？
- [ ] OPD-V与VAD/VCSD/V-Zero在同recipe下谁提供独立净收益？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v2 全文，正文 §1–5、式 (1)–(9)、图 1–6；附录 A–F、算法 1、式 (10)–(13)、表 2–4；主表逐项值另由可解析的v1 HTML核对。
- 新增认识：Visual contrast可被解释为modality-balance trust region，并同时承担token hard selection与soft weighting。
- 修正内容：未把attention ratio当作causal contribution，未把随机mask contrast当纯modality balance，分离了真实wall-clock收益与fixed-token算法效率。
- 下一步：核对代码中的answer prompt、mask sampler、threshold 2.0和top-K tail；实现semantic spatial intervention与correctness-aware margin gate。
