# FP-OPD

## 基本信息

- 论文标题：Distill What the Student Can See: Fisher-Projected On-Policy Distillation for Vision-Language Models
- 作者：Leyan Xue, Feng Xiong, Mingjun Ma, Changqing Zhang
- 年份与会议：2026；arXiv 预印本（v2）
- arXiv/DOI：arXiv:2608.01263；DOI: 10.48550/arXiv.2608.01263
- 论文链接：https://arxiv.org/abs/2608.01263
- 代码：未提供
- 本地文件：无
- 阅读状态：已完成（优化配置、计算成本与 tangent 实现待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

FP-OPD 用四个连续视觉 feature perturbations 估计 student 在当前 prefix 的低秩输出响应空间，再以 student Fisher metric 将完整 teacher–student log-prob gap 投影到该空间，围绕 detached student 构造 capacity-aware target并做 full-vocabulary reverse KL；它稳定超过 standard OPD，但所谓“locally realizable”实际只是对固定 probe-induced visual sensitivity 的近似，并不等价于参数训练可实现性。

## 核心问题

OPD 解决了 supervision state mismatch：teacher 在 student-generated prefixes 上评分。但它默认完整 teacher distribution 都适合 student。对于小 VLM，teacher correction 可能依赖 student 当前视觉 representation 未区分的细节，导致：

- teacher target 难以拟合；
- 过强 distillation 牺牲下游性能；
- 更大 teacher 不一定带来更好 student。

论文研究：

1. 如何定义 student 当前视觉 pathway 支持的局部 output-response space；
2. 如何在 reverse-KL 诱导的 Fisher geometry 下保留 teacher correction 的可达分量；
3. 投影后的 target 是否比完整 teacher target 更适合小 student；
4. 有限视觉 probes、finite-difference step 与 projection metric 如何影响结果。

## 方法

### Standard OPD correction

在 student prefix \(y_{<t}\)：

\[
p_\theta=p_\theta(\cdot\mid x,y_{<t}),\qquad
p_T=p_T(\cdot\mid x,y_{<t}).
\]

Standard OPD：

\[
\mathcal L_{\mathrm{OPD}}
=\frac1{|\mathcal T_y|}\sum_t
\mathrm{KL}(p_{\theta,t}\|p_{T,t}).
\]

定义 student-Fisher inner product：

\[
\langle a,b\rangle_p
=\sum_vp(v)a_vb_v,
\]

以及 \(p\)-weighted centering：

\[
\mathcal C_p[a]
=a-\left(\sum_vp(v)a_v\right)\mathbf1.
\]

完整 centered teacher correction：

\[
g
=\mathcal C_p[\ell_T-\ell_\theta].
\]

Softmax 对常数平移不变，因此：

\[
p_T
=\operatorname{softmax}(\ell_\theta+g).
\]

### 理论 visual tangent space

Student vision encoder 得到 visual embeddings
\(Z\in\mathbb R^{N\times d}\)。论文定义：

\[
\mathcal T_\theta(x,y_{<t})
=\overline{\operatorname{range}}
\left(
\left.\mathcal C_p\circ J_Z\ell_\theta
\right|_{\mathcal V_Z}
\right).
\]

它表示固定参数、固定 prefix 下，局部改变 visual representation 可诱导的一阶 output-score changes。需要区分：这是 input/feature perturbation tangent，不是对模型参数更新可学习空间的直接刻画。

### Finite-difference probes

将 visual tokens reshape 为 \(H\times W\) grid。对 spatial field \(m_k\)：

\[
\delta Z_k=m_k\odot(\bar Z-Z),
\]

\[
Z^{(k)}
=Z+\varepsilon_{\mathrm{fd}}\delta Z_k.
\]

即把选定位置的 visual features 以小步长推向 per-image mean，不删除 token。默认四个固定 binary fields 覆盖 top、bottom、left、right boundary bands，每个占对应轴的 25%。

Detached student finite-difference response：

\[
d_k
=\mathcal C_p\left[
\frac{\ell_\theta-\ell_\theta^{(k)}}
{\varepsilon_{\mathrm{fd}}}
\right].
\]

\[
D_K=[d_1,\ldots,d_K],\qquad
\widehat{\mathcal T}_{\theta,K}
=\operatorname{span}(D_K).
\]

默认 \(K=4,\varepsilon_{\mathrm{fd}}=0.05\)。

### Fisher projection

Reverse KL 的局部二阶展开：

\[
\mathrm{KL}\left(
p\|
\operatorname{softmax}(\log p+\delta)
\right)
=\frac12\langle\delta,\delta\rangle_p
+o(\|\delta\|^2).
\]

因此使用 Fisher metric 解：

\[
\widehat g^R
=\arg\min_{u\in\widehat{\mathcal T}_{\theta,K}}
\langle g-u,g-u\rangle_p.
\]

Gram matrix 与 alignment：

\[
A_{ij}=\langle d_i,d_j\rangle_p,\qquad
b_i=\langle d_i,g\rangle_p.
\]

Ridge：

\[
\lambda
=\rho\frac{\operatorname{tr}(A)}K
+\epsilon_{\mathrm{num}},
\qquad \rho=10^{-4},
\]

\[
c=(A+\lambda I)^{-1}b,\qquad
\widehat g^R=D_Kc.
\]

论文记 \(g^\perp=g-\widehat g^R\)。由于使用 ridge，残差一般不与 probe span 严格 Fisher-orthogonal：

\[
D_K^\top F(g-\widehat g^R)=\lambda c,
\]

除非 \(\lambda=0\) 或 \(c=0\)。因此“orthogonal/outside”应理解为 regularized remainder，而非精确正交分解。

### Capacity-aware target

\[
q^R
=\operatorname{softmax}
\left(
\operatorname{stopgrad}(\ell_\theta)
+\widehat g^R
\right).
\]

若 \(g\) 完全在 empirical probe span 内，恢复 teacher target；若与 probe span 正交，target 退化为 detached student，产生零更新。

最终：

\[
\mathcal L_{\mathrm{FP\text{-}OPD}}
=\frac1{|\mathcal T_y|}\sum_t
\mathrm{KL}(p_{\theta,t}\|q_t^R).
\]

Teacher、probe responses、projection 与 target 均 detached；仅 clean student distribution 接收梯度。无 reward、advantage 或 importance ratio。

## 训练数据与训练流程

- 模型：
  - Qwen3-VL-8B-Instruct→2B-Instruct；
  - Qwen3-VL-8B-Instruct-GRPO→2B-Instruct；
  - Qwen3-VL-32B-Instruct→8B-Instruct。
- 数据：Geo3K。
- 训练：2 epochs；step-wise linear LR decay；无 warmup；BF16。
- 每 GPU 1 example；gradient accumulation 8；每 prompt 4 on-policy rollouts。
- Max sequence 4096；max completion 2048。
- Probes：top/bottom/left/right boundary bands；\(K=4,\gamma=0.25,\varepsilon_{\mathrm{fd}}=0.05\)。
- Checkpoint：每 epoch 保存一次；论文未明确主表选哪个 checkpoint。
- 未报告 GPU 数量/型号、optimizer、初始 LR、weight decay、gradient clipping、有效 batch size、wall-clock 和 seed，均 `待核对`。

## 实验设置

- Task-near mathematical reasoning：WeMath、MathVista、MathVerse、MathVision
- General/OOD multimodal：MMMU、HallusionBench、MMStar
- 主评测：Avg@8，temperature 1.0，top-p 0.95，max generation 2048
- 补充：32B→8B 同时报 greedy Acc@1
- Baselines：Base、same-size GRPO、standard OPD
- 七项 Average 为 dataset 等权平均

## 实验结论

1. **8B→2B 相对 standard OPD 平均 +1.60。** 49.67→51.27；相对 base 48.50 为 +2.77。
2. **该设置七项均超过 base 和 standard OPD。** 相对 OPD：WeMath +2.55、MathVista +1.14、MathVerse +1.24、MathVision +2.24、MMMU +2.95、Hallusion +0.17、MMStar +0.92。
3. **但与 same-size GRPO 几乎持平。** FP-OPD 51.27 vs GRPO 51.26；且 WeMath、MathVision、MMStar 低于 GRPO。因此不能概括为优于 reward RL。
4. **更强 GRPO teacher 继续提高。** FP-OPD 为 52.78，相对同 teacher standard OPD 52.05 为 +0.73；七项均高于 direct student GRPO。
5. **32B→8B 平均 +0.80 over OPD。** 63.26→64.06；数学四项和 MMMU 提升，但 Hallusion -0.53、MMStar -0.21。
6. **Greedy 下仍有效。** 32B→8B greedy 平均 62.83→63.95（+1.12），六项提高，Hallusion -0.27。
7. **FD step 较稳健。** 0.025/0.05/0.10 的平均为 51.12/51.27/50.86。
8. **Probe 数量不单调。** \(K=2/4/8\) 为 51.16/51.27/51.09；四个结构 probe 仅略优。
9. **Fisher metric 是关键消融。** Fisher+visual 51.27，Euclidean+visual 42.48；但 8.79-point catastrophic drop 也可能包含 scale/numerical mismatch，不能仅凭单配置建立一般结论。
10. **Spatially structured probe 贡献较小。** Fisher+random 50.90，仅低于 visual probes 0.37。
11. **训练中 correction 变小。** 32B→8B 首/末 32 updates：median gap energy 0.299→0.169，target-shift JS 0.0987→0.0634；说明 model/target趋近，但不是 capacity correctness 的直接验证。
12. **Target scaling 支持“完整 teacher 未必最佳”。** 在 8B→2B 一 epoch 诊断中，\(\alpha=0.5\) 消除 prescribed KL gap 的 56.3%，\(\alpha=1\) 为 46.6%；MathVista Geometry 57.03 vs 55.29。

## 局限性

### 论文明确承认

- 未来需研究 adaptive tangent-space estimation。
- 当前多次 probe forward 增加计算，需更高效 projection strategy。

### 由实验设计可直接确认

- 只验证 Qwen3-VL 家族与 Geo3K；跨架构、非数学、视频、多图和空间规划未知。
- 无多 seed、标准差、置信区间或显著性检验；0.17–0.92 point 差异需谨慎。
- 未报告训练 hardware、wall-clock 或相对 OPD 的额外 FLOPs；每个 prefix 需要 \(K=4\) detached student probe forwards 和 position-wise solve。
- 训练超参数与 checkpoint selection 不完整，论文引用的“Appendix”在当前 HTML 中未提供对应完整配置。
- “ID”四项并非训练数据 Geo3K 本身，只是视觉数学 task-near benchmarks。
- 没有与最佳 uniform target interpolation、VGS、VCSD 或 VAD 做同 recipe 对照。

### 个人分析

- **Input tangent 不等于 trainability tangent。** \(J_Z\ell_\theta\) 描述固定参数模型对 visual features 变化的输出敏感度；distillation 实际通过参数 \(\theta\) 更新固定输入的行为。一个 teacher correction 可能无法由当前 input perturbation产生，却能通过参数训练学习，反之亦然。
- **“Capacity-aware”是 probe-aware proxy。** 理论 \(\mathcal V_Z\) 若允许任意高维 feature perturbation，其 output tangent 可能很大；实现却只取四个边界 band directions。保留的是这四个 probes 的 span，而不是 student 全部视觉 capacity。
- **固定边界 probes 与任务 evidence 未必对应。** 关键证据常在中央或任意局部；random probes 只低 0.37，也说明具体空间结构贡献有限。
- **Target-scaling 不能单独证明不可实现性。** 完整 target 在固定一 epoch 内拟合比例较低，可能来自优化 horizon、loss scale 或 over-regularization；论文未在主表比较调优后的 scalar \(\alpha\) baseline。
- **缺少最关键简单基线。** \(\alpha=0.5\) 已在 Geometry subset 优于 \(\alpha=1\)，但没有给其七项平均与 FP-OPD 的受控比较，无法判断 Fisher projection 相对“均匀减弱 teacher”净收益。
- **Ridge 后不是正交 projection。** 论文将 residual 记作 \(g^\perp\) 并称位于 probe space 外，但 ridge solution 只近似该性质。
- **Student Fisher metric 强化保守性。** 权重 \(p_\theta(v)\) 很小的 teacher-supported candidate 对 projection 影响弱，可能忽略 student 当前低估但正确的新 token。
- **Finite difference 是局部近似。** \(\varepsilon=0.05\) 经验稳定不等于 linearity；没有报告 approximation error、direction condition number 或 Gram rank。
- **Euclidean ablation 过度退化需排查数值尺度。** 同一 ridge \(\lambda\) 或 normalization 是否为 Euclidean geometry重新调优未说明。
- **结果相对 RL 的优势有限。** 普通 8B teacher 下平均仅比 same-size GRPO 高 0.01；核心证据应表述为优于 standard OPD，而不是全面优于其他 post-training。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可先估计 student 对空间输入扰动的 local response directions，再过滤 teacher correction。
- Fisher metric 与 reverse-KL local geometry一致，比 uniform Euclidean token-coordinate projection更合理。
- Student-anchored target 可在保留 on-policy/full-vocabulary OPD 的同时避免完整 teacher correction。
- 多 factor spatial probes 可形成小型 Gram system，position-wise 适应当前视觉和文本 context。

### 个人建议

- 将 input tangent 与 parameter tangent 对照：计算 visual adapter/LoRA 参数 Jacobian方向，直接估计训练可实现 correction。
- 用 evidence-aware multi-scale probes替代固定边界 band：对象 mask、中心/局部 patch、depth/coordinate/topology perturbation。
- 必须加入 tuned scalar interpolation baseline，并控制 projected correction norm，分离“选方向”与“减小 target strength”。
- 报告 Gram rank、condition number、projected energy ratio和 finite-difference linearity error。
- 使用 teacher/student union support或 full Fisher diagnostics，避免 student 低概率正确 token 被忽略。
- 评估低秩 basis cache、随机 sketch/JVP，降低每 prefix 四次额外 forward 成本。

## 与其他论文的相同点和冲突

### 与 [VAD](vad.md)

- 两者都围绕 detached student 重构 target，并投影完整 teacher correction。
- VAD 用同 teacher evidence-present/degraded direction和 uniform Euclidean geometry，目标是 source attribution；FP-OPD 用 student perturbation basis和 Fisher geometry，目标是 local realizability。
- FP-OPD 修正了 Euclidean geometry，但四个 input probes仍不是 parameter-learning capacity；VAD 的 intervention则更 task-specific。

### 与 [VGS](vgs.md)

- 两者都改变 teacher correction 的方向而非仅加权位置。VGS 在 parameter-gradient层 steering visual objective；FP-OPD 在 output-distribution层投影 teacher target。
- VGS 诊断 language/visual gradient conflict；FP-OPD 诊断 teacher correction是否落入 empirical student visual response span。

### 与 [VCSD](vcsd.md)

- 两者都以 detached student为 target anchor并做 candidate-distribution reconstruction。
- VCSD 用 EMA teacher original/control contrast进行 exponential shaping；FP-OPD 用外部 teacher gap在 student visual Fisher tangent上投影。
- VCSD 无外部 teacher但需 control validity；FP-OPD 无 privileged view但需 probe basis validity。

### 与 [VA-OPD](va-opd.md)、[V-Zero](v-zero.md) 和 [Vision-OPD](vision-opd.md)

- VA-OPD/V-Zero 选择视觉相关 token/trajectory，Vision-OPD 直接匹配 privileged target；FP-OPD 不做 selection，而过滤每个位置的 teacher correction方向。
- FP-OPD 不需要 crop/region annotation，但需要多次 student visual perturbation forward。

### 与 [PCD](pcd.md)、[ViGOS](vigos.md) 和 [VOLD](vold.md)

- PCD/ViGOS 处理 failure stage 与 teacher介入阶段，VOLD 处理 correctness/state alignment；FP-OPD 处理选定阶段内 teacher target 是否适配 student。
- PCD 的 stage weight可与 FP target相乘，但需避免同时过度削弱 supervision。

### 与 [ViCuR](vicur.md)

- ViCuR 增加 recovery architecture以扩大 student evidence extraction能力；FP-OPD 接受当前 response geometry并过滤超出 probe span的 correction。
- 前者尝试扩容，后者适配现有容量，构成互补策略。

### 与 [SA-OPD](sa-opd.md)

- SA-OPD先判断某位置的teacher–student disagreement是否随输入变化；FP-OPD再判断其direction是否落在student probe span内，可组合为position gate × projected target。
- SA-OPD受no-prompt cancellation影响，FP-OPD受probe incompleteness影响，二者proxy误差来源不同。

### 与 [OPD-V](opd-v.md)

- OPD-V用zoom/mask margin选择视觉敏感位置；FP-OPD用student Fisher probe span过滤target direction。前者处理where/weight，后者处理what is locally compatible。

### 综合定位

十三篇论文中，FP-OPD 新增“student-local realizability”维度：不仅判断 teacher correction 是否视觉相关、何时进入或来自何种证据，还要判断它是否落在 student 当前 probe-estimated response geometry内。

## 证据

### E-001

- 结论：完整 teacher target 在一 epoch 诊断中比 \(\alpha=0.5\) target 更难拟合且 Geometry performance更低。
- 类型：论文结论
- 定位：§1；图 1(c)
- 必要引用：gap removal 46.6% vs 56.3%；55.29 vs 57.03。
- 备注：不能单独排除 optimization-strength解释。

### E-002

- 结论：FP-OPD 用 visual feature finite differences估计 output tangent。
- 类型：论文结论
- 定位：§3.2–3.3；式 (8)–(11)
- 必要引用：默认四个 boundary bands，\(\varepsilon=0.05\)。
- 备注：是 input-response tangent，不是 parameter trainability tangent。

### E-003

- 结论：FP-OPD 在 student Fisher metric下投影 centered teacher gap。
- 类型：论文结论
- 定位：§3.4；式 (12)–(19)
- 必要引用：\(A=D^\top FD\)，ridge \(\rho=10^{-4}\)。
- 备注：ridge residual不严格正交。

### E-004

- 结论：Projected target 围绕 detached student构造，并使用 full-vocabulary reverse KL。
- 类型：论文结论
- 定位：§3.5；式 (20)–(22)
- 必要引用：无 reward/advantage/importance ratio。
- 备注：所有 target-side quantity detached。

### E-005

- 结论：8B→2B FP-OPD 平均比 base/OPD 高 2.77/1.60。
- 类型：论文结论
- 定位：表 1；§4.2
- 必要引用：48.50/49.67/51.27。
- 备注：与 GRPO 51.26 几乎相同。

### E-006

- 结论：32B→8B 相对 OPD 平均 +0.80，但两项 general benchmark下降。
- 类型：论文结论
- 定位：表 1–2
- 必要引用：63.26→64.06；Hallu -0.53、MMStar -0.21。
- 备注：不是七项一致提升。

### E-007

- 结论：Fisher projection显著高于 Euclidean projection。
- 类型：论文结论
- 定位：表 3；§4.3
- 必要引用：51.27 vs 42.48。
- 备注：需核对 Euclidean baseline的数值尺度与超参数公平性。

### E-008

- 结论：Structured visual probes只略高于 random probes。
- 类型：论文结论
- 定位：表 3
- 必要引用：51.27 vs 50.90。
- 备注：fixed boundary geometry贡献有限。

### E-009

- 结论：论文未将 FP-OPD 与 tuned scalar interpolation做完整七项对照。
- 类型：个人核对
- 定位：§1；表 1–3
- 必要引用：仅报告 \(\alpha=0.5/1\) 的 Geometry diagnostic。
- 备注：无法分离方向 projection 与简单 target weakening。

### E-010

- 结论：Visual input tangent不能直接代表 parameter-update realizability。
- 类型：个人推断
- 定位：式 (8)–(11)、(20)–(21)
- 必要引用：basis 来自 \(J_Z\ell_\theta\)，训练梯度更新 \(\theta\)。
- 备注：需 parameter-Jacobian或实际 fitability验证。

### E-011

- 结论：FP-OPD 增加 \(K\) 次 detached LM forward与 position-wise projection solve。
- 类型：论文结论/个人核对
- 定位：BuildBasis；§5
- 必要引用：默认 \(K=4\)；作者将效率优化列为 future work。
- 备注：未报告 wall-clock/GPU-hours。

## 待验证问题

- [ ] 主表 checkpoint 使用 epoch 1 还是 epoch 2，选择标准是什么？
- [ ] Optimizer、LR、hardware、有效 batch size与 wall-clock是多少？
- [ ] Tuned scalar interpolation在全部七项上能否达到 FP-OPD？
- [ ] Input visual tangent与 parameter-update tangent的 overlap多大？
- [ ] 四个 boundary probes对中央/小目标证据的覆盖率如何？
- [ ] Ridge residual的非正交程度与 Gram condition number是多少？
- [ ] Euclidean baseline是否重新调过 ridge和coordinate normalization？
- [ ] 多 seed下 +0.17–1.60 point增益是否稳定？
- [ ] 可否用 JVP/random sketch/cache降低四次额外 forward？
- [ ] 跨模型家族、视频、3D和空间规划是否有效？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v2 全文，正文 §1–5、式 (1)–(22)、表 1–3、图 1–3。
- 新增认识：OPD target mismatch不仅是视觉归因问题，也可从 student-local response geometry和 KL/Fisher metric解释。
- 修正内容：未把 input perturbation tangent等同于参数可训练容量，未把 ridge remainder称为严格正交 residual，未忽略 scalar weakening baseline。
- 下一步：补齐训练配置与成本，复现 scalar-interpolation/Fisher-projection对照，并设计 parameter-space spatial tangent。
