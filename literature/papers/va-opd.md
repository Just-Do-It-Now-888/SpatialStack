# VA-OPD

## 基本信息

- 论文标题：Visual-Advantage On-Policy Distillation for Vision-Language Models
- 作者：Ruiqi Liu, Xiaolei Lv, Gengsheng Li, Ximo Zhu, Zhiheng Wang, Zhengbo Zhang, Junkai Chen, Zhiheng Li, Bo Li, Jun Gao, Shu Wu
- 年份与会议：2026；arXiv 预印本（v1，2026-05-21，未注明正式会议）
- arXiv/DOI：arXiv:2605.21924；DOI: 10.48550/arXiv.2605.21924
- 论文链接：https://arxiv.org/abs/2605.21924
- 本地文件：无
- 阅读状态：已完成（部分训练超参数待核对）
- 阅读人/窗口：Cursor / SpatialStack_OPSD
- 首次记录：2026-08-13
- 最后更新：2026-08-13

## 一句话总结

VA-OPD 用教师在原图与细节退化图上的 token log-prob 差定义 visual advantage，识别稀疏的视觉关键 token，再从 rollout 和 token 两个层级重新分配 reverse-KL 梯度；相对均匀加权的标准 OPD，它在全部八个评测集上提高准确率，并使学生生成内容更多地落在教师判定为依赖视觉细节的位置。

## 核心问题

标准 VLM on-policy distillation 在学生自己的 rollout 上对每个 token 均匀计算教师—学生 KL，但响应中绝大多数 token 是语言模板或推理脚手架，真正由图像细节决定的 token 很少。均匀平均可能主要让学生模仿教师的语言表面分布，却没有强化视觉依赖。论文研究两个问题：

1. 如何以 token 粒度识别“预测该 token 是否真正需要精细视觉信息”；
2. 如何在不增加标注、reward model 或 student rollout 数量的情况下，把更多蒸馏梯度分配给视觉关键 token 和视觉参与度更高的轨迹。

## 方法

### 整体流程

1. 学生 VLM 在原图 \(v\) 和问题 \(q\) 上生成 \(K\) 条 on-policy rollout。
2. 教师在相同 prefix 下分别输入原图 \(v\) 与细节退化图 \(\tilde v\)，计算每个已生成 token 的 visual advantage（VA）。
3. Rollout-level：按每条轨迹的平均 VA 在同题的 \(K\) 条 sibling rollouts 内做 z-score 和 softmax，视觉依赖更强的轨迹获得更大权重。
4. Token-level：每条轨迹内按 VA 排序，前 20% 为 high-VA，其余为 low-VA；两组分别平均 reverse KL，再按 0.5/0.5 合并，避免少数视觉 token 被多数语言 token 稀释。
5. 训练只优化 VA 加权的教师—学生 KL，不使用任务 reward；推理阶段不需要 VA、退化图或教师。

### 关键模块

**Visual Advantage 反事实探针。** 对学生已生成 token \(y_t\)，比较教师在原图和退化图条件下的 log-prob：

\[
a_t=\max\left(
\log p_T(y_t\mid v,q,y_{<t})
-\log p_T(y_t\mid\tilde v,q,y_{<t}),0
\right).
\]

高 \(a_t\) 表示教师认为该 token 的可预测性依赖被退化操作破坏的视觉细节。负差值被截断为 0，作者认为这类位置主要是低置信噪声。

**视觉细节退化。** 图像先用 bilinear interpolation 下采样到原空间分辨率的 10%，再用 nearest-neighbor 恢复原尺寸。这会破坏数字、OCR、小符号等细节，同时尽量保留布局和颜色，并保持视觉 token 数一致，便于逐 token 对齐 log-prob。

**Rollout-level reweighting。** 对每条 rollout 的平均 VA 在同题 sibling group 内标准化，再经温度为 1 的 softmax 得到 \(w^{(k)}\)。它回答“优先学习哪条轨迹”。

**Token-level grouped KL。** 每条 rollout 的 top-\(p_v\) VA token 和其余 token 分组归一化，各自计算组内平均 KL。默认 \(p_v=0.2,\lambda=0.5\)，因此 high-VA 20% token 和 low-VA 80% token 各占一半组级 loss；按单 token 计算，high-VA token 的权重是 low-VA token 的 4 倍。

### 损失函数与训练目标

标准 OPD 使用学生到教师的 reverse KL（式 1）：

\[
\mathcal L_{\mathrm{KL}}
=\frac{1}{T}\sum_t
\mathrm{KL}\left(p_S(\cdot\mid v,q,y_{<t})
\|p_T(\cdot\mid v,q,y_{<t})\right).
\]

第 \(k\) 条 rollout 的平均 VA 与组内标准化（式 3）：

\[
\bar a^{(k)}=\frac{1}{T^{(k)}}\sum_ta_t^{(k)},\qquad
\hat z^{(k)}=\frac{\bar a^{(k)}-\mu}{\sigma+\epsilon}.
\]

Rollout 权重（式 4）：

\[
w^{(k)}
=\frac{\exp(\hat z^{(k)}/\tau)}
{\sum_j\exp(\hat z^{(j)}/\tau)}.
\]

Token-level grouped KL（式 5）：

\[
\mathcal L_{\mathrm{group}}^{(k)}
=\lambda\frac{1}{|V^{(k)}|}\sum_{t\in V^{(k)}}\mathrm{KL}_t
+(1-\lambda)\frac{1}{|L^{(k)}|}\sum_{t\in L^{(k)}}\mathrm{KL}_t.
\]

最终目标（式 6）是对同题的所有 rollout 按 \(w^{(k)}\) 加权：

\[
\mathcal L_{\mathrm{VA\text{-}OPD}}(x)
=\sum_{k=1}^{K}w^{(k)}\mathcal L_{\mathrm{group}}^{(k)}.
\]

退化图只参与 VA 计算；实际 KL target 始终是教师在原图上的分布。

### 训练数据与训练流程

- 主设置：Qwen3-VL-8B-Instruct 教师蒸馏 Qwen3-VL-2B-Instruct 学生。
- 规模实验：另用 Qwen3-VL-4B、32B 教师，学生固定为 2B。
- 训练数据：Geometry3K，约 2.1k problems；扩展实验使用 ViRL39K，约 39k problems。
- 训练：5 epochs，AdamW，\(K=4\) rollouts/prompt，batch size 16。
- VA-OPD：\(\tau=1.0\)、\(\lambda=0.5\)、\(p_v=0.2\)、pixelation ratio 0.10；所有配置固定，不逐配置调参。
- 计算：VA-OPD 相对标准 OPD 每条 rollout 增加一次教师在退化图上的 forward，不增加 student rollout、标注或 reward model；主效率实验使用 8 张 A100。
- arXiv HTML 正文未给出 learning rate、weight decay、max generation length、optimizer schedule、硬件型号和各模型具体并行配置；文中指向的“完整训练细节”位置为空，均标记为 `待核对`。

## 实验设置

- 模型：Qwen3-VL-2B 学生；Qwen3-VL-4B/8B/32B 教师
- 数据集：训练用 Geometry3K、ViRL39K；数学评测 WeMath、MathVista、MathVerse；视觉理解评测 HallusionBench、AI2D、MMMU、MMStar、OCRBench
- Baseline：Base、CoT-SFT、off-policy KD、Standard OPD、GRPO、PAPO
- 指标：temperature 1.0 下的 avg@8；遵循各 benchmark 官方指标，适用时使用 GPT-4o judge
- 关键超参数：5 epochs、batch 16、4 rollouts、\(\tau=1\)、\(\lambda=0.5\)、\(p_v=0.2\)、pixelation 0.10
- 公平性设置：同一配置内共享初始化、数据、优化器和评测协议；表 1 每种方法报告其 best checkpoint

## 实验结论

1. **视觉监督集中在少数 token。** 约 105k 个 held-out rollout token 上，top 10% token 承载约 93% 的 VA mass。屏蔽最高 VA 的 10% token 会使 MathVerse-mini 下降约 2.9 points；随机屏蔽同等比例仅下降约 0.3，屏蔽最低 VA 基本无影响。
2. **主设置下，VA-OPD 在八个 benchmark 上均高于 Standard OPD。** 数学平均 48.3 vs 45.4（+2.9），视觉平均 66.1 vs 64.6（+1.5）。单项增益为 WeMath +3.3、MathVista +2.7、MathVerse +2.8、HallusionBench +2.5、AI2D +2.4、MMMU +0.6、MMStar +0.2、OCRBench +1.7。
3. **相对 RL baseline，结果并非所有单项都占优。** VA-OPD 的数学平均和视觉平均最高，但 MMMU 51.5 低于 GRPO 52.9/PAPO 52.7；MMStar 59.9 高于其他方法；因此“全面优于”只适用于和 Standard OPD 的配对比较。
4. **教师规模越大，VA-OPD 相对 OPD 的增益越大。** Geometry3K 上，4B→2B、8B→2B、32B→2B 的数学/视觉平均增益分别为 +2.1/+0.8、+2.9/+1.5、+3.7/+2.0。
5. **数据扩大后相对增益没有被稀释。** 8B→2B 从 Geometry3K 扩展到 ViRL39K 后，VA-OPD 相对 OPD 的数学/视觉平均增益从 +2.9/+1.5 增至 +3.8/+2.5。
6. **两个粒度的组件都贡献增益。** 只做 rollout reweighting 或只做 token grouped KL 都不及完整 VA-OPD，但均恢复部分增益；完整方法在 MathVerse、HallusionBench、OCRBench 分别相对 OPD 提升 +2.8、+2.5、+1.7。
7. **Standard OPD 的准确率提升并未伴随本文 VA 指标明显上升。** MathVerse 从 19.6 升到 29.1 时，mean VA 仅 0.07→0.10；VA-OPD 达到 31.9，mean VA 为 0.16。
8. **达到同等准确率所需 wall-clock 更少。** 在 8×A100 的主设置中，VA-OPD 约 6.5 h 达到 Standard OPD 最终 MathVerse 29.1，而后者用 19.3 h；但 VA-OPD 每 step 多一次教师 forward，论文的 3× 是“达到目标准确率的时间”而不是单 step 加速。

## 局限性

### 论文明确承认

- VA 是“教师相对反事实敏感性”的代理指标，不是对学生内部 attention、feature grounding 或真实感知过程的直接测量。
- 增益集中在视觉密集任务；允许文本捷径的 MMStar 等 benchmark 上增益较小。

### 由实验设计可直接确认

- 只验证 Qwen3-VL 同家族的 2B 学生；跨模型家族、不同视觉编码器和不同 tokenizer 的泛化未知。
- 训练数据均为视觉数学/几何任务，方法对自然图像、视频、3D 与交互空间任务的效果尚未验证。
- 退化操作固定为 10% pixelation，主要针对 OCR、小符号和数值；不能保证适合颜色、计数、遮挡、深度、对象关系等不同视觉因素。
- 表 1 报告每种方法的 best checkpoint，且未报告多训练随机种子的均值/标准差。avg@8 反映推理采样平均，不等价于训练方差；MMStar +0.2、MMMU +0.6 等小差异的显著性未知。
- GPT-4o judge 在适用 benchmark 中引入闭源评测组件，未报告 judge variance。
- 相对标准 OPD，每条 rollout 需要额外一次教师 forward；32B 教师和大数据训练的绝对计算、显存与能耗未报告。
- 正文缺少若干完整训练超参数，当前版本复现信息不完整。

### 个人分析

- “student rollout VA 上升”意味着学生更倾向生成“教师认为依赖视觉细节”的 token，但 VA 仍完全由教师评分。它不能单独证明学生自身在有图/无图条件下的预测敏感性增强；需要直接测量 student counterfactual log-prob 或遮图后的性能下降。
- VA 把原图相对退化图的正 log-prob 差截断，忽略负向变化。此设计抗噪，但可能系统性高估正敏感性，且没有区分有益视觉依赖与对图像伪相关特征的依赖。
- Rollout reweighting 只看视觉依赖，不看答案正确性。高 VA 但错误或被误读的轨迹仍会获得更大权重；教师原图 KL 可能纠正它，但论文没有做 VA 与 correctness 的二维消融。
- 固定 top 20% 会强行把每条 rollout 的一部分 token 归为 high-VA，即使整条轨迹的绝对 VA 都接近 0。rollout-level 权重可降低低 VA 轨迹的整体贡献，但不能完全消除组内相对排序产生的伪高 VA token。
- 数据规模与教师规模的实验各自只覆盖少数点。“增益单调增长”是当前四个配置内的经验现象，不能外推为普遍 scaling law。

## 可复用到当前项目的内容

### 论文直接支持

- SpatialStack_OPSD 可把“视觉依赖”做成 token 级反事实量：教师分别观察原始空间输入和细节受损输入，比较同一 student token 的 log-prob。
- 对长空间推理链，不应让大量连接词、公式模板和通用推理 token 淹没坐标、方向、距离、对象编号等视觉/空间关键 token；可采用分组归一化 loss。
- 同一问题的多条 rollout 可按相对视觉参与度加权，优先蒸馏真正读取空间证据的推理路径。
- 该方法不改变推理路径，适合仅在训练阶段增加 teacher counterfactual pass。

### 个人建议

- 不要直接照搬单一 pixelation。应按 SpatialStack 输入定义构造多种反事实：遮挡局部区域、移除深度、打乱对象 ID、模糊坐标文本、破坏拓扑边或只保留全局布局，并分别计算 factor-specific advantage。
- 同时测 teacher-VA 与 student-VA：前者作为训练权重，后者作为真正的学生视觉依赖诊断；再配合遮图准确率，避免把教师代理指标当作学生 grounding 证据。
- 把 correctness/reward 与 VA 联合分组：优先“高视觉依赖且正确”的轨迹，对“高 VA 但错误”的轨迹使用教师纠错，而不是仅按 VA softmax。
- 当一条 rollout 的绝对 VA 总量低于阈值时退回 Standard OPD，避免 top-\(p_v\) 强制选择无实际视觉信号的 token。
- 对空间任务分别报告推理准确率、感知/定位准确率、反事实敏感性和无视觉输入性能。

## 与其他论文的相同点和冲突

### 与 [VOLD](vold.md) 的共同点

- 都在 student on-policy rollout 的 prefix 上查询教师，提供 token 级稠密监督。
- 都认为均匀、无条件地使用教师信号并非最优，需要根据轨迹状态选择或重加权蒸馏。
- 都复用 student rollout，推理时不需要教师。

### 与 [Vision-OPD](vision-opd.md) 的共同点

- 两者都用视觉条件差异构造训练时特权监督，不依赖任务 reward：Vision-OPD 比较同模型的证据 crop 与带框全图条件，VA-OPD 比较视觉教师的原图与细节退化图评分。
- 两者都直接在 student-generated prefix 上做分布蒸馏，推理时不再需要额外视觉视图或教师。
- 两者都针对细粒度视觉证据，但机制不同：Vision-OPD 用更易观察证据的 crop teacher 提供完整 token target；VA-OPD 用反事实差定位高视觉依赖 token，再重分配原图 teacher KL。

### 与 [VGS](vgs.md) 的共同点

- 两者都认为 Standard OPD 中语言信号会掩盖视觉 grounding，且都利用教师在视觉充分/视觉受限条件下的 distribution 差。
- VA-OPD 用 sampled-token log-prob 差识别重要轨迹/token，并重新分配 loss 权重；VGS 用 full-distribution KL 衡量视觉依赖，构造去语言先验的 visual target，并直接 steering 梯度方向。
- VA-OPD 回答“在哪些 token 上多学”，VGS 回答“沿哪个模态方向学”；VGS 的 adaptive token steering 与 VA-OPD 的 grouped weighting 具有直接组合空间。
- VA-OPD 的 pixelation 保留全局布局、偏向细节；VGS 的 text-only counterfactual 移除全部视觉信息，覆盖更广但更易混入图像触发的风格变化。

### 与 [ViCuR](vicur.md) 的共同点

- 两者都强调视觉 grounding，而非仅复制 teacher 输出；VA-OPD 自动发现视觉依赖 token，ViCuR 由离线 cue text 指明视觉证据并增加 recovery module。
- VA-OPD 无需 cue 标注但依赖 visual teacher 的反事实评分；ViCuR cue 可解释性更强，却必须审计 generator 是否访问答案或外部知识。
- ViCuR 可用 VA 对 cue-conditioned supervision 做 token weighting，但需先验证 sink query 真正按当前问题恢复证据。

### 与 [ViGOS](vigos.md) 的共同点

- 两者都反对 uniform、全轨迹使用单一 teacher signal。VA-OPD 按反事实视觉依赖选择 token/rollout；ViGOS 按 description/reasoning/invalid 阶段路由 teacher context。
- ViGOS 的固定 description mask 仍包含许多通用语言 token；VA weighting 可进一步在该 segment 内聚焦真正依赖图像的事实 token。

### 与 [V-Zero](v-zero.md) 的共同点

- 两者都用 teacher 在视觉充分/受限 view 下对 sampled token 的 log-prob 差形成 visual evidence signal，再从 trajectory 层重加权 OPD。
- VA-OPD 使用原图/像素化图并保留 token grouping；V-Zero 使用目标/随机 crop、同 prompt sibling z-score，只有 trajectory gate。两者的信号都不等于 correctness。

### 与 [VCSD](vcsd.md) 的共同点

- 两者都从原图与视觉受限条件的 teacher distribution difference 中提取视觉信号。
- VA-OPD 用固定外部教师，将差值用于 rollout/token weighting；VCSD 用 EMA self-teacher，对全词表差值做 support-restricted target shaping。前者选择监督位置，后者改变监督分布。

### 与 [PCD](pcd.md) 的共同点

- 两者都重新分配 OPD 监督。VA-OPD 依据 visual dependence 选择 rollout/token；PCD 依据 downstream failure×perception teacher gap 选择可纠正 perception。
- VA-OPD 不判断 correctness；PCD 使用 verifier，但其 PSR 仍混合感知充分性与 reasoning difficulty。

### 与 [VAD](vad.md) 的共同点

- 两者都使用 evidence-present/removed teacher contrast。VA-OPD 将正 sampled-token contrast用于监督加权；VAD 使用全 candidate signed contrast重构 local target并显式表达 refutation。
- VA-OPD 回答哪里多学，VAD 回答该位置具体改变哪些 candidate odds。

### 与 [FP-OPD](fp-opd.md) 的共同点

- FP-OPD 不用 visual advantage选择 token，而在 Fisher metric下把完整 teacher gap投影到 student visual probe span；它回答 teacher correction 是否与当前 student response geometry兼容。

### 与 [SA-OPD](sa-opd.md) 的共同点

- 两者都用full/control divergence做token allocation。VA-OPD增权视觉依赖位置；SA-OPD删除no-prompt后近似不变的高影响位置，但后者混合teacher/student input effects。

### 与 [OPD-V](opd-v.md) 的共同点

- 两者都用visual-view sampled-token正margin做selection/weighting。VA-OPD用original/pixelated views；OPD-V用zoom/masked-crop EMA teachers并蒸馏zoom target。

### 互补关系

- VOLD 使用文本教师和纯文本训练题，目标是把语言侧推理迁移给保留视觉能力的 VLM；VA-OPD 使用视觉教师和图文训练数据，目标是让蒸馏真正强化细粒度视觉依赖。
- VOLD 用 outcome reward 屏蔽正确轨迹的 KL，依据“是否答对”选择何时模仿；VA-OPD 不使用 reward，而用视觉反事实差异决定模仿哪条轨迹、哪些 token。
- VOLD 强调 OPD 前 teacher/student 分布对齐；VA-OPD 的效率曲线也显示前约 3 h 与 Standard OPD 重合，作者推测需先匹配表面分布后视觉信号才成为瓶颈，但未直接做冷启动对齐消融。
- Vision-OPD 强调 self-distillation target 的时间稳定性，证明同步 current-policy teacher 会崩溃；VA-OPD 使用固定的跨规模教师，规避了该问题，但未研究 EMA/self-teacher 版本。

### 潜在冲突与适用条件

- VOLD 的教师是 text-only，无法计算“教师有图 vs 退化图”的 VA，因此不能直接采用 VA-OPD；若项目采用视觉教师，两者可组合。
- VOLD 报告更大教师从 8B 到 14B 收益趋于饱和；VA-OPD 中相对 Standard OPD 的增益从 4B 到 32B 单调增加。二者并不构成直接矛盾：学生规模、教师类型、训练数据和比较量不同，前者看绝对终局性能，后者看视觉重加权相对标准 OPD 的增量。
- Vision-OPD 的 crop teacher 依赖明确区域特权信息，VA-OPD 的 pixelation 不要求区域标注，但只探测被降采样破坏的细节；前者更强、更局部，后者数据构造更轻但视觉因素覆盖更窄。
- 十三篇论文共同提示：有效 OPD 至少要考虑状态对齐、teacher 时间稳定性、监督密度、模态梯度方向、privilege 可恢复性、介入阶段、轨迹证据质量、递归 target 漂移、跨阶段 credit、correction attribution、student compatibility、input-grounded reliability 及 modality balance；只解决其中之一可能不足。

## 证据

### E-001

- 结论：VA 是教师在原图与细节退化图上对同一 student-generated token 的正向 log-prob 差。
- 类型：论文结论
- 定位：§2.1；式 (2)
- 必要引用：无
- 备注：它是 teacher-relative proxy，不是学生内部视觉 grounding 的直接测量。

### E-002

- 结论：视觉依赖信号高度稀疏，top 10% token 承载约 93% VA mass。
- 类型：论文结论
- 定位：§2.2；图 2(a)
- 必要引用：统计基于约 105k 个 held-out student-rollout token。
- 备注：数据来自 Standard OPD student。

### E-003

- 结论：屏蔽最高 VA 的 10% token 会显著破坏蒸馏效果。
- 类型：论文结论
- 定位：§2.2；图 2(b)
- 必要引用：MathVerse-mini 约 -2.9 points；随机屏蔽约 -0.3；最低 VA 屏蔽在 ±0.1 内。
- 备注：支持 high-VA token 承载有效视觉监督，而不仅是统计上数值较大。

### E-004

- 结论：VA-OPD 同时使用 rollout-level VA softmax weighting 和 token-level grouped KL。
- 类型：论文结论
- 定位：§3.1–3.3；式 (3)–(6)；图 3
- 必要引用：无
- 备注：默认 \(\tau=1,p_v=0.2,\lambda=0.5\)。

### E-005

- 结论：主设置中 VA-OPD 在八个 benchmark 上均高于 Standard OPD。
- 类型：论文结论
- 定位：§4.2；表 1
- 必要引用：Math Avg +2.9，Visual Avg +1.5。
- 备注：小幅单项差异未给训练方差；“均高于”不代表均高于所有 RL baseline。

### E-006

- 结论：VA-OPD 相对 Standard OPD 的增益随当前实验中的教师规模与数据规模增加。
- 类型：论文结论
- 定位：§4.3；表 2
- 必要引用：教师 4B/8B/32B 的 Math Avg 增益 +2.1/+2.9/+3.7；ViRL39K 上为 +3.8。
- 备注：实验点有限，不应外推为 scaling law。

### E-007

- 结论：完整方法优于任一单独组件。
- 类型：论文结论
- 定位：§4.4；图 4(a)
- 必要引用：无
- 备注：HTML 正文未给出单组件的完整数值表。

### E-008

- 结论：VA-OPD 训练时 accuracy 与 teacher-scored rollout VA 同时上升，而 Standard OPD 的 VA 近乎不变。
- 类型：论文结论
- 定位：§4.4；图 4(b)
- 必要引用：Standard OPD 19.6→29.1、VA 0.07→0.10；VA-OPD accuracy 31.9、VA 0.16。
- 备注：VA 由教师评分，不能单独证明学生内部 grounding。

### E-009

- 结论：VA-OPD 以约三分之一 wall-clock 达到 Standard OPD 的最终 MathVerse accuracy。
- 类型：论文结论
- 定位：§4.5；图 5
- 必要引用：约 6.5 h vs 19.3 h 达到 29.1。
- 备注：这是 time-to-accuracy，不是单 step 吞吐提升；VA-OPD 每 rollout 多一次教师 forward。

### E-010

- 结论：当前版本缺少部分复现所需训练细节。
- 类型：个人核对
- 定位：§4.1 “Training”；其“Full training, hyperparameter, and compute-fairness details are in .”引用为空
- 必要引用：无
- 备注：learning rate、generation length 等 `待核对`。

## 待验证问题

- [ ] 直接计算学生在原图/退化图下的 counterfactual log-prob 差时，是否也随 VA-OPD 上升？
- [ ] 高 VA 与答案正确性之间是什么关系？联合使用 reward 与 VA 是否优于单独 VA weighting？
- [ ] 当整条 rollout 的绝对 VA 很低时，固定 top 20% grouped KL 是否会放大噪声？
- [ ] 不同视觉退化分别能否捕获 OCR、颜色、计数、深度、遮挡、拓扑和对象关系？
- [ ] 多训练随机种子下，MMStar +0.2、MMMU +0.6 是否显著？
- [ ] 完整训练超参数和 32B 教师的绝对训练成本是什么？

## 阅读日志

### 2026-08-13

- 阅读范围：arXiv v1 全文，§1–7、式 (1)–(6)、表 1–2、图 1–6。
- 新增认识：视觉蒸馏监督不是均匀分布在推理链中；教师反事实评分可定位视觉关键 token，并通过分组归一化防止长语言脚手架稀释梯度。
- 修正内容：区分“教师判定的 rollout 视觉依赖”与“学生内部真实 grounding”；区分相对 Standard OPD 全面提升与相对所有 baseline 的单项结果。
- 下一步：核对补充训练配置；实验比较 teacher-VA、student-VA、reward 与多种空间反事实退化。
