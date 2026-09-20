# MV-OPSD 实验报告

> **多视角特权 On-Policy 自蒸馏用于 3D 空间推理**

| 项目 | 信息 |
|---|---|
| 实验状态 | 🔄 进行中 |
| 基座模型 | Qwen3.5-4B（发布权重，未经任务 SFT） |
| 训练框架 | verl（`loss_mode=vopd`） |
| 当前实验 | `20260817_qwen35base_mvopsd_v1`；离线对照含 SPAR3 K=1 step 55 |
| 最后更新 | 2026-09-03 |

---

## 1. 研究问题

3D 空间推理任务中，模型在推理时往往只能看到有限数量的视角（帧预算受限），而完整场景的几何信息分布在更多视角上。我们借鉴 Vision-OPD 的 on-policy 自蒸馏思路，将特权信号从「裁剪放大」改为「视角覆盖度」：

- **Teacher**：观察同一场景的 N 个视角；
- **Student**：仅观察其中 K 个视角，且 K ≪ N；
- 二者在 **student 自己采样出的同一条回答** 上做 token 级分布匹配。

目标是让 student 在有限视角预算下，仍能逼近多视角条件下的推理分布，从而在低帧预算评测（如 VSI-Bench @ 1/2/4 帧）上获得提升。

---

## 2. 方法

### 2.1 问题定义与符号

每条样本由一个多视角场景、一条问题和相应答案组成。本文使用以下符号：

- **V = {v₁, …, vₙ}**：一个场景可用的完整视角集合。每个 vᵢ 是一张图像或一帧视频。
- **N = |V|**：teacher 可见的视角总数。不同数据源的 N 不同，例如视频源通常为 8，SPAR 32-view 为 32。
- **S ⊂ V**：从完整视角集合中选出的 student 视角子集。
- **K = |S|**：student 实际可见的视角数，本实验从 {1, 2, 4} 中逐样本选取，并保证 K < N。
- **q**：输入问题文本。除图像占位符数量和视角编号外，teacher 与 student 的问题文本保持一致。
- **y = (y₁, …, yₗ)**：student 生成的完整回答 token 序列。
- **y_<t**：第 t 个位置之前已经生成的回答前缀，即 (y₁, …, yₜ₋₁)。

Student 根据有限视角 S 和问题 q 生成回答：

```text
y ~ π_student(· | S, q)
```

这里的 `~` 表示“从该概率分布中采样”。回答由 student 产生，因此训练状态来自当前策略实际会访问的生成路径，而不是数据集中预先写好的标准答案前缀，这就是 **on-policy** 的含义。

### 2.2 同一回答上的双条件分布

对于 student 已生成的同一条回答，teacher 不再生成另一条答案，而是在每个位置上执行 teacher forcing：

```text
P_t(x) = π_student(x | S, q, y_<t)
Q_t(x) = π_teacher(x | V, q, y_<t)
```

其中：

- **x**：词表中的任意候选 token，不是图像视角；
- **P_t(x)**：student 在有限视角下预测下一个 token 为 x 的概率；
- **Q_t(x)**：teacher 在完整视角下、面对相同回答前缀时预测 x 的概率；
- **t**：回答中的 token 位置。

两侧使用相同的 q 和 y_<t，因此 P_t 与 Q_t 的主要条件差异是可见视角数。这样可以逐 token 比较“有限视角预测”与“完整视角预测”，避免因 teacher 另行生成不同文本而失去位置对齐。

### 2.3 蒸馏目标

目标与 Vision-OPD 主实验一致：在每个回答 token 位置上，对 student 分布 **P_t** 与 teacher 分布 **Q_t** 计算 **JSD₀.₅**（广义 Jensen–Shannon divergence，α = 0.5）。

**实际比较的分布支撑集。** 词表规模很大，代码不会在每个位置保存完整词表概率。当前配置（`distillation_topk=100`，`distillation_add_tail=True`）下，每个位置先取 **student 概率最高的 100 个候选 token**，再让 teacher 对**同一组 100 个 token** 取 log-prob；词表中其余 token 的概率质量各合并为 **1 个尾部概率桶**（tail bucket）。因此 P_t 与 Q_t 在实现上都是 **101 维**分布（100 个显式候选 + 1 个“其余全部”），而不是完整词表上的分布。该 top-K + tail 近似沿用 Vision-OPD 主实验设置。

**JSD 中的“混合分布”是什么。** 在每个 token 位置、在上述 101 维支撑集上，JSD 公式会临时构造一个中间量：

```text
M_t = (1 − α)P_t + αQ_t
```

再计算：

```text
D_t = (1 − α)KL(P_t || M_t) + αKL(Q_t || M_t)
```

符号含义：

- **M_t**：student 与 teacher 在该位置上的**混合分布**，仅作为 JSD 计算中的数学中间量；
- **KL(A || B)**：概率分布 A 与 B 之间的 KL divergence；
- **D_t**：第 t 个 token 上的蒸馏损失；
- **α**：混合权重；本实验 **α = 0.5**，即对称的标准 JSD₀.₅。

需要强调：**M_t 不是额外训练或保存的“混合模型”**，也不产生新的 checkpoint；它只在反向传播时于当前 batch 的 log-prob 张量上按上式算出，用于得到 D_t。代码中 α = 0 对应 forward KL，α = 1 对应 reverse KL，介于两者之间为广义 JSD。选择 0.5 是为了在“贴近 teacher 认为可能的答案”与“避免 student 被单一模式过度拉动”之间折中。

**M_t 也不等同于 EMA teacher。** 前者在**输出概率空间**内、逐 token、仅服务于当前 loss；EMA 则在**模型参数空间**内跨训练步更新 teacher 权重。Vision-OPD 主配置同时使用 JSD₀.₅ 与 EMA teacher；本项目 v1 保留 JSD₀.₅，teacher 改为 frozen，因此不执行 EMA 参数更新。

完整 loss 按“单个 token → 单条回答 → 整个 batch”逐层计算。设 i 表示 batch 中第 i 条回答，t 表示该回答的第 t 个 token：

**第一步：计算该位置的 JSD。**

```text
M_i,t = (1 − α)P_i,t + αQ_i,t

KL_student_i,t = KL(P_i,t || M_i,t)
KL_teacher_i,t = KL(Q_i,t || M_i,t)

D_i,t = (1 − α)KL_student_i,t + αKL_teacher_i,t
```

其中 D_i,t 是该 token 位置上有限视角 student 与完整视角 teacher 的分布差异。

**第二步：计算策略变化的 importance-sampling 权重。**

```text
c_i,t = exp(log P_current(y_i,t) − log P_old(y_i,t))
c_i,t = min(c_i,t, is_clip)
```

当前配置中 `is_clip=2.0`。如果当前 actor 对已生成 token 的概率相较旧策略增长过大，该权重最多取 2.0，避免单个 token 的梯度被异常放大。

**第三步：加入 rollout engine 与训练 actor 之间的校正。**

```text
W_i,t = c_i,t × u_i,t
```

u_i,t 是可选的 rollout-correction 权重，用于校正 vLLM 生成策略与训练 actor 计算出的策略之间可能存在的概率偏差。若两者完全一致，则 u_i,t = 1。

**第四步：得到单个有效 token 的加权损失。**

```text
ℓ_i,t = mask_i,t × W_i,t × D_i,t
```

mask_i,t 在真实回答 token 上取 1，在 padding、prompt 或被排除位置上取 0。因此只有有效回答部分参与反向传播。

**第五步：把 batch 内所有 token 的损失聚合成一个标量。**

这一步由 `loss_agg_mode` 决定，两种模式的分母不同：

```text
token-mean（v1 之后采用，也是 verl 默认与 Vision-OPD 官方配置）：
    L = (Σ_i Σ_t ℓ_i,t) / (Σ_i Σ_t mask_i,t)

seq-mean-token-mean（v1 实际使用）：
    L_i = (Σ_t ℓ_i,t) / n_i ，其中 n_i = Σ_t mask_i,t
    L   = (Σ_i L_i) / B_valid ，其中 B_valid 为至少含一个有效 token 的回答数
```

区别在于**每条回答的权重**：`token-mean` 下一条回答的梯度占比正比于它的生成长度，`seq-mean-token-mean` 下所有回答等权、与长度无关。`agg_loss` 在分布式下会用 global 的 token/序列计数做归一化（含 `* dp_size` 的尺度补偿），但不改变上述数学目标。

概括而言，这个 loss **不直接判断回答是否正确，也不把生成结果与标准答案做交叉熵比较**。它在 student 自己生成的每一个回答前缀上，衡量“只看 K 个视角时的下一 token 分布”与“看完整 N 个视角时的下一 token 分布”有多大差异，并只更新 student，使有限视角模型逐渐逼近完整视角模型的预测。importance-sampling 和 rollout-correction 负责抑制策略版本差异带来的估计偏差。由于当前训练没有正确性 reward，如果完整视角 teacher 本身判断错误，这个 loss 也可能传递错误分布，因此最终有效性必须由外部 benchmark 验证。

### 2.4 关键超参数

**`distillation_topk=100` 与 `distillation_add_tail=True`**

二者共同定义上文 P_t、Q_t 的 **101 维支撑集**，是 JSD₀.₅ 在工程上的实现方式，而非与主目标无关的附加技巧：

1. 按 student 在该位置的 logits 取 top-100 索引；
2. teacher 对**相同 100 个 token** 取 log-prob（保证两侧在同一候选集上可比）；
3. 将 top-100 之外的全部概率质量合并为一个 tail bucket：在 log 空间用 `log(1 − Σ exp(log p_i))` 计算，使 100 个显式项与 tail 项之和为 1；
4. 在这 101 维分布上构造 M_t 并计算 JSD。

这样避免在每个位置物化完整词表分布，显著降低显存与通信；Vision-OPD 论文报告 top-100 以外概率质量通常可忽略。代价是：落在 tail 桶内的 token 不再被单独区分，只保留“其余总质量”这一条信息。

**`full_logit_distillation=True`**

表示损失比较的是上述候选集（+ tail）上的**完整分布**，而不只是 student 实际采样的那个 token。teacher 可传递“还有哪些替代 token 合理、各自概率多大”的软监督，这是分布蒸馏相对硬标签监督的主要区别。

**`is_clip=2.0`**

importance-sampling 权重定义为：

```text
c_t = min(exp(log P_current(y_t) − log P_old(y_t)), 2.0)
```

它衡量当前 actor 相对于生成或更新前策略，对实际生成 token y_t 的概率变化。上限 2.0 用于防止少数概率比值过大、放大梯度并破坏训练稳定性。它不是 PPO 的双边裁剪：这里仅限制最大值。当前实验每步只做一个 mini-batch 更新，训练接近严格 on-policy，因此 c_t 通常接近 1；该参数主要作为策略发生偏移时的安全保护。

### 2.5 受控变量：视角数量

本实验希望验证的因果变量是 **teacher 比 student 看见更多视角**。为保证结果可归因，其他输入条件尽量保持一致：

- teacher 与 student 使用同一个 Qwen3.5-4B 架构和 tokenizer；
- 每个视角均采用相同的 384×512 分辨率，即 192 个视觉 token；
- 两侧引用同一批预处理图像，不各自保存不同缩放版本；
- 问题文本保持一致，仅图像占位符数量和必要的视角编号不同；
- teacher 与 student 在同一条回答、同一回答前缀上计算分布。

若 teacher 同时拥有更多视角和更高分辨率，实验即无法判断收益来自场景覆盖度还是图像清晰度。因此，“每视角分辨率相同”不是工程细节，而是方法成立所需的对照条件。

### 2.6 Teacher 与训练信号设定

当前 v1 使用一个**训练期间保持不变的 teacher**。训练开始时，teacher 和 student 都来自同一个发布版 Qwen3.5-4B。此后只有 student 会继续学习，teacher 的参数始终不更新。

在每条训练样本上：

1. student 只看选出的 K 个视角，并生成回答；
2. teacher 看同一场景的完整 N 个视角，也阅读同一个问题；
3. teacher 不会看到标准答案，也不会得到额外的文字提示；
4. 系统比较两者在 student 回答过程中给出的词概率分布，并据此更新 student；
5. teacher 只提供学习目标，自身不会被反向更新。

例如，一条样本共有 8 张场景图，student 只看其中 2 张，teacher 则看完整 8 张。两者都回答同一个空间问题。训练目标不是要求 student 逐字复述 teacher 的最终回答，而是让 student 在生成每个词时的判断逐渐接近看过完整场景的 teacher。

所有符合条件的训练样本都会使用这一蒸馏信号，不会先根据回答正确与否进行筛选。如果一条样本缺少 teacher 所需的完整视角，训练会直接报错，以便及时发现数据问题，而不会悄悄改用另一种训练目标。

虽然数据中保存了标准答案，但本次训练**不使用标准答案计算损失，也不使用奖励模型给回答打分**。因此，student 的学习信号完全来自 teacher 的概率分布，而不是“答对得高分、答错得低分”或同组回答之间的优劣比较。

冻结 teacher 的好处是学习目标稳定：student 即使在训练过程中不断变化，teacher 的判断标准也不会跟着变化。但这并不意味着 teacher 一定正确。本方法依赖的假设是：同一个模型看到更多相关视角后，通常能比只看少量视角时做出更可靠的空间判断。该假设是否成立，最终仍要通过基线和评测结果验证。

训练过程中还会持续检查 teacher 的输出是否保持稳定，以确认它没有被意外更新。

---

## 3. 数据处理

### 3.1 数据源与规模

按 SpatialStack SFT 各源采样率（`60%/60%/60%/50%`）抽样后，仅保留能构造合法 N → K 视角落差的样本：

| 数据源 | 视角配置 | 采样后 | 保留 | 占比 |
|--------|---------|-------|------|------|
| llava_hound_64k | N = min(8, 可用帧数)，K ∈ {1, 2, 4} | 38,250 | 38,009 | 30.1% |
| SPAR 3-view | N = 3，K ∈ {1, 2} | — | 34,470 | 27.3% |
| vlm3r_scannet | N = 8，K ∈ {1, 2, 4} | 31,067 | 31,067 | 24.6% |
| SPAR 32-view | N = 32，K ∈ {1, 2, 4} | — | 21,025 | 16.6% |
| vsi_appr_order | N = 8，K ∈ {1, 2, 4} | 1,909 | 1,909 | 1.5% |
| **合计** | | | **126,480** | 100% |

SPAR 两档由同一份 `spar_234k` 标注按 N 拆出，采样后共 140,566 条，保留 55,495 条（39.5%）；过滤明细见 §3.2.1。`vlm3r_scannet` 与 `vsi_appr_order` 无一被过滤，因其视角由抽帧构造、答案不含视角编号。

实际训练 parquet：`data/mvopsd/parquet/main_train.parquet`（124,306 条）+ `main_holdout.parquet`（2,174 条留出），合计即上表 126,480。视角缓存 474,757 张图（24 GB），其中 91,128 张为样本专属的带标记帧（§3.2.2）。

> 计数以 `data/mvopsd/plan/stats.json` 为准。本表早期版本曾使用建库前的规划估值（合计 134,840），与实际产出不符，已更正。

### 3.2 三阶段数据管线

```text
标注 JSON → [Stage 1] 采样/过滤/视角选择 → [Stage 2] 抽帧+缩放 → [Stage 3] 写 Parquet
```

**Stage 1：采样、过滤与视角选择**

- 按固定种子复现 SpatialStack SFT 采样；
- 排除无法构造视角落差的样本（详见 §3.2.1）；
- 逐样本随机抽取 K ∈ {1, 2, 4}；
- SPAR 有标记题型：标记帧必须保留，并重写 `Frame-N` 引用（详见 §3.2.2）；
- 视频源：按照 SpatialStack 的规则均匀间隔抽帧。

#### 3.2.1 为什么要排除样本，以及排除了哪些

可以把一条多视角样本理解成一本含有 N 张照片的小相册：teacher 看完整相册，student 只看其中 K 张。本实验要求 **K < N**，同时要求 student 拿到的照片仍足以看懂并回答原问题。只要少给照片后问题变得无法回答，或者正确答案随照片编号发生变化，这条样本就不能使用。

采样后共排除 85,312 条（SPAR 85,071 + llava_hound 241），分四类：

| 类别 | 排除原因 | 条数 |
|---|---|---|
| **A. 无法在保持问题可回答的同时少给图片** | 只有一张图 | 45,336 |
| | 回答问题必须看到全部图片 | 13,781 |
| | 必须看到的图片多于 student 可接收的数量 | 2,741 |
| **B. 删图或重排图片会改变题目/答案** | 位置匹配 | 5,572 |
| | 选择离目标最近的视角 | 5,261 |
| | 视角变化判断 | 5,093 |
| | 相机运动判断 | 4,349 |
| | 出现顺序判断 | 1,512 |
| | 目标所在帧定位 | 1,407 |
| **C. 答案直接说“第几张图”** | 如“第二张图”一类回答 | 241 |
| **D. 评测集泄漏** | 与 VSI-Bench 场景重合 | 19 |

**A 类：无法在保持问题可回答的同时少给图片。**

- **45,336 条只有一张图。** teacher 和 student 都只能看这一张图，不可能形成“teacher 多看、student 少看”的对比，因此直接排除。
- **13,781 条必须看到全部图片才能回答。** 例如，一条样本共有 3 张图，而问题同时要求比较这 3 张图。若 student 少看任何一张，问题就无法回答；若三张都给 student，teacher 又没有多看到内容。因此这类样本也不能使用。
- **2,741 条需要保留的图片太多。** 本实验规定 student 每次看 1、2 或 4 张图。假设一条 8 图样本的问题必须保留其中 5 张，即使不必看到全部 8 张，也超过了 student 最多看 4 张的设置，所以无法纳入训练。

后两种情况合计 16,522 条。它们本身不一定有错误，只是不适合本实验的训练方式：既要让 student 少看图片，又不能删掉回答问题所必需的图片，这两个要求无法同时满足。

**B 类：删图或重新排序后，原问题或正确答案会改变。** 这 6 种 SPAR 题型共 23,194 条。例如：

- 问“目标出现在哪一帧”时，答案可能是“第 5 帧”。删掉或重排图片后，“第 5 帧”已经不是原来的图；
- 问“哪一个视角离目标最近”时，答案是从给出的视角中选一个。删掉部分视角后，候选集合和正确答案都可能改变；
- 问两个视角之间的变化、相机运动或位置对应关系时，student 必须同时看到指定的两张图。缺少任何一张，问题就无法回答。

因此，这类样本不能简单地从 N 张图中抽出 K 张继续使用。

**C 类：答案文字本身直接说“第几张图”。** B 类是根据问题类型整类排除；C 类则是从其他数据中额外发现的个别回答。在 `llava_hound_64k` 中有 241 条回答包含“第一张图”“第二个视角”或 `Frame-N` 等表述。比如原答案是“物体位于第二张图”，抽取和重排图片后，原来的第二张图可能变成第一张，也可能没有被选中。此时继续使用原答案会教给模型错误的图片编号，所以这些样本被排除。

**D 类：与视角无关的泄漏防护。** 另有 19 条样本与 VSI-Bench 评测场景重合，出于评测洁净度要求予以剔除；它们不属于视角落差过滤。

#### 3.2.2 SPAR 标记帧与 `Frame-N` 重写

**问题来源。** SPAR 的问题常通过彩色点或框指代物体，例如「the shower curtain (in Frame-23) (red point)」。这些标记并不在原始图像中，而是 SpatialStack 训练时动态添加。当前训练管线无法沿用这一在线处理方式，因此需要预先生成带标记的视角图像。

**后果一：带标记的视角必须保留给 student。** 同一张原图在不同样本中可能标记不同物体，因此带标记图像是样本专属的：474,757 张缓存图中有 91,128 张属于此类。更重要的是，一旦删去带标记帧，问题便失去明确的指代对象，所以该帧必须出现在 student 的 K 个视角中。若必须保留的标记帧过多，就无法同时维持 K < N，这也是前述结构性过滤的重要来源。

**后果二：student 的帧编号必须随视角子集同步更新。** student 只看到原始 N 个视角中的 K 个，保留下来的视角会按照新的顺序重新编号。因此，问题正文中的 `Frame-N` 引用也必须改成 student 实际看到的编号。例如，原问题引用 `Frame-23`，若该帧在 student 的 K 个视角中排在第二位，则问题应改写为 `Frame-1`。否则问题中的编号将与实际图像不对应。

**teacher 与 student 使用各自对应的帧编号。** teacher 看到完整 N 个视角，因此问题保留原编号；student 只看到 K 个视角，因此问题使用重排后的编号。两者表达的是同一个问题，只是编号分别与各自收到的图像对应。在无特权对照中，teacher 与 student 看到相同的 K 个视角，因此使用同一份问题文本。

**Stage 2：抽帧与图像标准化**

- ScanNet 视频预先抽帧，以便 teacher 按多图形式读取；
- 统一缩放至 **384×512**（192 token/视角），对齐 SpatialStack 几何；
- 图像尺寸满足 Qwen3.5 视觉编码所需的对齐条件。

**Stage 3：生成训练数据**

输出训练所需的 Parquet：student 侧保存 K 个视角及其问题，teacher 侧保存完整 N 个视角及对应问题；同时保留样本来源、student/teacher 视角数等实验追踪信息。标准答案仅为数据格式兼容而保留，不参与当前蒸馏损失。

同时生成 **无特权对照**：teacher 与 student 视角数相同（N = K），用于隔离「多视角特权」与「自蒸馏本身」的效应。

### 3.3 可判定性过滤

有些问题会用彩色点或方框标出被问的物体，但这个物体不一定出现在每一张图里。给 student 挑选 K 张图时，必须确保其中至少有一张能够看到这个物体。

例如，完整样本有 8 张图，问题询问“红点标出的椅子在桌子的哪一侧”，但这把椅子只出现在第 3、4 张图中。如果随机给 student 的图都来自其他位置，student 连被问的椅子都看不到，此时 teacher 的回答无法成为合理的学习目标。因此，挑选图片时必须保留第 3或第 4 张中的至少一张。

这一规则与 §3.2.1 的区别是：

- **§3.2.1 决定整条样本能不能用。** 如果无论怎样选图，都无法同时做到“teacher 多看”和“student 仍能回答”，就直接删除该样本。
- **本节决定一条可用样本应该选哪几张图。** 样本本身可以使用，但不能完全随机抽图；必须让 student 看见回答问题所必需的物体。

经过这两步后，保留下来的训练样本既能让 teacher 比 student 看到更多信息，也不会让 student 在完全看不到目标物体的情况下学习。

---

## 4. 实验设置

### 4.1 模型与训练超参

| 项目 | 设定 |
|------|------|
| 基座 | Qwen3.5-4B（`./models/Qwen3.5-4B`） |
| 学习率 | 2.0 × 10⁻⁶ |
| Batch size | 64 prompts × `rollout.n=4` = 256 序列/步 |
| 训练步数 | 300 步（约 19,200 样本，占池 14.2%） |
| `max_prompt_length` | 4096 |
| `max_response_length` | 1024 |
| `image_patch_size` | 16 |
| 视觉注意力 | sdpa（视觉塔） |
| GPU | 8× H20（~96 GB/卡），`gpu_memory_utilization=0.35` |
| 随机种子 | 20260816 |

### 4.2 训练内评测

CV-Bench（2,638 条）接入 verl 验证循环，训练前测 step 0 基座对照，之后周期性评测：

- 生成：**greedy**（`temperature=0`，`do_sample=false`）
- 指标：官方 2D/3D combined 聚合
- 辅助监控：`answered/frac`（回答格式）、`teacher_probe`（teacher 是否冻结）、`data/step/*`（批次数据源构成）

**基座对照（step 0，全量 CV-Bench）**：

| 指标 | 分数 |
|------|------|
| Combined | **86.96** |
| 2D | 82.43（ADE20K 76.78 / COCO 88.07） |
| 3D | 91.50（Omni3D） |

训练内分数是训练信号，不是可对外并列的 benchmark 口径：题面、生成长度上限与解析器均与官方协议不同，差异可达数十分，详见 §4.3.1。

### 4.3 离线评测协议

训练结束后在以下 benchmark 上评测：

| Benchmark | 输入形态 | 作用 |
|-----------|---------|------|
| **VSI-Bench** | 视频，帧预算 @ {1,2,4,8,16,32} | **主指标**（低帧预算曲线） |
| CV-Bench | 单图 | 单图空间能力 |
| BLINK-Spatial | 图像组 | 多视角推理 |
| SPAR-Bench | 多图 | 同域检查 |
| Video-MME 等 | — | 泛化与不退化护栏 |

帧预算曲线是核心主张：MV-OPSD 应把曲线左半段（低帧）抬高，右半段（高帧）不退化。

项目内有两条离线评测链，推理后端不同：

| 链 | 入口 | 推理后端 | 判分 |
|---|---|---|---|
| Vision-OPD 链 | `scripts/opsd/eval_visionopd/run_eval.sh` | vLLM 服务（被测模型 4 卡 + gpt-oss-120b 判官 4 卡） | 规则 + LLM 判官分级 |
| lmms_eval 链 | `scripts/evaluation/eval.sh` | HF transformers（accelerate 8 进程，非 vLLM） | 任务自带解析器 |

registry 指定前者为对外数字的准绳。但截至 2026-08-18，Vision-OPD 链只有冒烟产物，正式基座评测起了服务即中断，因此**当前所有在手的离线数字均来自 lmms_eval 链**。

#### 4.3.1 CV-Bench 分数强烈依赖评测协议

同一个 checkpoint 在不同协议下的分数相差可达数十分。协议由三项独立设定构成：题面措辞、生成长度上限、选项解析器。

| 题面 / 长度上限 / 解析器 | 基座 | v1 step 300 |
|---|---|---|
| 官方 / 16 / 官方（即 lmms_eval 原始协议） | 62.85 | 5.00 |
| 官方 / 1024 / 官方 | 73.83 | 44.20 |
| 官方 / 1024 / word_boundary | 85.67 | 79.86 |
| 本项目题面 / 1024 / word_boundary（**已作废**的旧离线默认） | ~~85.95~~ | — |
| **本项目题面 / 1024 / boxed last-line（当前离线默认）** | **88.07** | — |
| 训练内（native 题面 + vLLM，护栏曲线，非对外口径） | 86.96 | 83.57 |

**当前对外基座数字**：Qwen3.5-4B 零样本 CV-Bench **combined 88.07**（2D 83.55 / ADE 78.04 / COCO 89.07，3D 92.58，作答 99.62%）。产物：`logs/eval/20260901_qwen35base_cvbench_boxed_lastline/`，registry `20260901_qwen35base_cvbench_boxed_lastline`。旧的 **85.95** 是无 `\boxed{}` 后缀、全文 `word_boundary` 扫描的结果，协议已与 VSI boxed last-line 对齐后不再使用，不得再写成「基座 CV-Bench」。

**基座 62.85 → 86.96 的逐项拆解（2026-08-18，word_boundary / 训练内口径）**：放开长度 +11.0、换解析器 +11.8、题面措辞 +0.3、推理引擎与 native 排版 +1.0。长度与解析器各占约一半，只补其中一项都会得出错误结论。该拆解**不覆盖** 2026-09-01 的 boxed last-line 重测。

**同一次训练的退化幅度完全取决于口径**：

| 口径 | 基座 → step 300 |
|---|---|
| 官方协议 | −57.9 |
| 官方协议 + 1024 长度 | −29.6 |
| 再换本项目解析器 | −5.8 |
| 训练内 | −3.4 |

真实情况是两条并列的结论，不能只报其中一条：**空间推理小幅退化（约 3–6 分），作答格式严重退化**。训练后的模型不再直接给出选项字母，而是先写数百字推理再在末尾给答案；官方协议的 16 token 上限会把答案整段截掉，其"取全文第一个 `[A-F]`"的解析器随后把 `**B**ased on the provided images` 的 B 读成投票，5.00 这个分数本质上是"永远蒙 B"的结果。

三项差异的成因：

- **长度上限**：官方任务配置为 `max_new_tokens: 16`，训练内为 1024。
- **解析器**：官方取全文第一个 `[A-F]` 字节（含词内字母）；本项目先找 `(X)` 括号形式，再退到两侧不粘字母的独立大写 A–F。
- **题面**：官方配置将前缀设为空字符串，但代码以 `or` 兜底，导致每题实际都被加上 `These are frames of a video.`（CV-Bench 是静态单图）；本项目去掉该句。

#### 4.3.2 生成长度上限的影响

使用本项目解析器，改变长度上限后的 combined：

| 上限 | 基座（旧 word_boundary dump） | v1 step 300 | 被截断的行（基座 / step 300） |
|---|---|---|---|
| 1024 | 85.95 | 79.86 | 0% / 0% |
| **512** | **85.87** | **79.79** | 0.4% / 3.0% |
| **256** | **85.09** | **77.85** | 1.7% / 8.8% |
| 128 | 78.03 | 31.34 | 10.9% / 64.7% |
| 64 | 75.91 | 0.33 | 13.5% / 100% |

> 本表仍是 2026-08-18 对旧 word_boundary 生成做截断重打，**不是**当前 boxed last-line 基座 88.07。两列各自使用其唯一可用的 1024-token 生成（基座为本项目题面，step 300 为官方题面），故列间不可横向比较绝对值，只看纵向趋势。
>
> 该表由 1024-token 生成按 token 截断后重新打分得到。贪心解码下截断结果与真实短上限运行完全等价：将 step 300 的生成截至 16 token 后重打，得 0.04（本项目解析器）与 5.00（官方解析器），与真实 16-token 运行结果一位小数不差。

结论：**512 几乎无损**（两模型降幅均在 0.1 分以内），256 降 0.9–2.0 分尚可接受，128 以下出现断崖。断崖位置随模型作答长度移动——基座回答 token 数 p50=1 / p90=138，step 300 为 p50=142 / p90=242，故同一上限对两者杀伤截然不同。评测训练后的 checkpoint 时长度上限不应低于 256。

#### 4.3.3 工具与汇报口径

`src/lmms_eval/tasks/cvbench` 提供协议开关，题面、长度上限、停止条件与解析器由同一开关统一决定，避免跑出半套协议：

```bash
BENCHMARKS=cvbench bash scripts/evaluation/eval.sh                          # 默认：对齐训练内
CVBENCH_PROTOCOL=lmms_legacy BENCHMARKS=cvbench bash scripts/evaluation/eval.sh   # 复现官方数字
CVBENCH_MAX_NEW_TOKENS=512 BENCHMARKS=cvbench bash scripts/evaluation/eval.sh     # 省墙钟
```

同时新增 `cvbench_answered` 指标（能读出选项字母的行占比），用于区分"答错"与"不再按选项格式作答"。回归测试见 `scripts/opsd/tests/test_lmms_cvbench_protocol.py`。

**汇报要求**：CV-Bench 数字必须标注所用协议（题面、长度上限、解析器、推理后端）。跨协议数字之间不存在可折算的固定偏移——如上表所示，基座的解析器差异为 11.8 分而 step 300 为 35.7 分，差距随模型作答风格变化。

**题面排版的最后一处不一致（2026-08-18 已消除）**：训练内验证集 parquet 原先生成于协议改造之前，题面为数据集自带的 `(A) 3` 排版，而离线默认协议用 lmms_eval 的 `A. 3`。v1 与 rollout8 都跑完 300 步后重建了该文件，现两侧题面逐字一致（`data/eval/cvbench_verl/cvbench_val.parquet`，`extra_info.prompt_style = lmms_eval`，2638/2638 条经 `test_cvbench_val_parity.py` 逐条校验）。

v1 与 rollout8 实际使用的是 native `(A) 3` 排版（当时 `cvbench_val.parquet` 尚未重建）；该 native 备份已删除，复现这两条曲线需 `build_cvbench_val_parquet.py --prompt-style native`。两个排版下的分数不可混用比较——这正是 §4.3.1 的同一类问题。

#### 4.3.4 VSI-Bench（32 帧离线，基座 vs SPAR3 K=1 step 55）

test 5,130 题。overall = 8 类等权均值 ×100（方向 easy/medium/hard 先合并再进均值）。解码均为 **greedy**、`disable_thinking=true`、32 帧。lmms-eval 后端。

与 CV-Bench 一样，**题面 + 长度预算 + 解析器是一套协议，不可拆开加减**。当前有两套已跑完的离线协议：

| 协议名 | 题面 | 预算 | 解析 |
|---|---|---|---|
| **original**（`spatialstack_plain`） | 无 `\boxed{}` 后缀；MCA「直接写选项字母」、NA「一词或短语」 | 1024，`until=[]` | 末行 `answer_tail`，不开 boxed-primary |
| **boxed last-line**（默认 `spatialstack`） | 追加要求 `\boxed{}`、答案放最后一行 | 4096，`until=[]` | 有闭合框只认框内，否则末行 |

| 模型 | 协议 | 引擎 | token | overall | 作答 | Δ |
|---|---|---|---:|---:|---:|---:|
| 基座 | original | HF | 1024 | **52.75** | 98.13% | — |
| 基座 | original | vLLM | 1024 | **52.47** | 98.50% | vs HF **−0.28** |
| SPAR3 K=1 step 55 | original | HF | 1024 | **47.87** | 95.58% | vs 基座 HF **−4.88** |
| 基座 | boxed last-line | HF | 4096 | **48.06** | 95.75% | 与 original 不可减 |
| SPAR3 K=1 step 55 | boxed last-line | HF | 4096 | **49.84** | 98.25% | vs 基座 boxed **+1.78** |

Δ 只在同协议、同引擎、同 token 下有意义。**original 与 boxed 的 SPAR3−基座符号相反。** 训练内 SPAR3 val 是 boxed last-line + **sample**（约 46.5–49.23%），不可与上表 greedy 并列。vLLM 基座 52.47 相对历史锚点 52.58 为 −0.11（现行末行抽取）。

基座 original 52.75 是 2026-09-02 按 8-20 无后缀题面重新生成；当时源 dump 落盘 **53.40**，现行末行抽取重打为 52.75（Δ −0.65，主要在 `obj_appearance_order`）。对同一批 original 回答再套 boxed-primary 解析仍是 52.75（闭合框 0/5130）——**换解析变不出 48.06**，48 量级来自换题面后重新生成。

original 协议下题型分（百分制）：

| 题型 | 基座 | SPAR3 | Δ |
|---|---:|---:|---:|
| object_counting | 56.48 | 40.07 | −16.41 |
| object_abs_distance | 36.89 | 30.23 | −6.66 |
| object_size_estimation | 67.31 | 64.64 | −2.67 |
| room_size_estimation | 54.13 | 44.90 | −9.23 |
| object_rel_distance | 58.45 | 58.17 | −0.28 |
| object_rel_direction | 55.74 | 51.58 | −4.16 |
| route_planning | 35.05 | 33.51 | −1.54 |
| obj_appearance_order | 57.93 | 59.87 | **+1.94** |

跌分主要在计数与房间/绝对距离。本表 **不是** 1/2/4 帧曲线；帧预算扫描仍待做。

样本检查：[`experiments/viewers/vsibench_qwen35base_plain/index.html`](../viewers/vsibench_qwen35base_plain/index.html)（基座 original 题面 + 回复）。基座 original 回复中位约 3 字符，约 64% 为 ≤3 词短答。

Dump：`logs/eval/20260902_qwen35base_vsibench_plain/`，`logs/eval/20260903_qwen35base_vsibench_plain_vllm/`，`logs/eval/20260903_spar3_k1_step55_vsibench_plain/`，`logs/eval/20260901_qwen35base_vsibench_boxed_lastline/`，`logs/eval/20260831_mvopsd_spar3_k1_vsibench_boxed_lastline/`。对照总表见 [`评测对照_基座_vs_SPAR3_k1_step55.md`](评测对照_基座_vs_SPAR3_k1_step55.md)。

### 4.4 对照实验设计

| 实验 | 设定 | 目的 |
|------|------|------|
| **主实验（v1）** | teacher 使用 N 视角 / student 随机使用 K 视角，冻结 teacher | 完整 MV-OPSD |
| **无特权对照** | N = K（同一份数据） | 隔离多视角特权 vs 自蒸馏本身 |
| **rollout8 对照**（计划中） | `rollout.n: 4→8`（对齐 Vision-OPD） | 考察更多 on-policy 轨迹的影响 |

主判据：主实验在 VSI-Bench @ {1,2,4} 帧均值上 **高于无特权对照**（≥1.0 点），才说明多视角特权有效。

---

## 5. 实验进展

| 阶段 | 状态 | 说明 |
|------|------|------|
| 数据管线 | ✅ 完成 | 三阶段脚本跑通，parquet 与视角缓存就绪 |
| v0 训练（300 步，EMA teacher） | ✅ 完成 | 工程链路验证 |
| v1 训练（冻结 teacher + 训练内 CV-Bench） | 🔄 进行中 | 2026-08-17 启动，目标 300 步；最新 checkpoint：step 50 |
| v1 rollout8 对照 | ⏳ 排队 | v1 结束后启动，`rollout.n=8` |
| 离线六 benchmark 评测 | 🔄 部分完成 | VSI@32 帧两协议已出（§4.3.4）；1/2/4 帧曲线仍缺 |

详细运行状态见 [`experiments/ACTIVE.md`](../ACTIVE.md)；参数与产物路径见 [`experiments/registry/20260817_qwen35base_mvopsd_v1.yaml`](../registry/20260817_qwen35base_mvopsd_v1.yaml)。

---

## 6. 预期结论形式

若方法有效，应观察到：

1. VSI-Bench 低帧预算（1/2/4 帧）均值相对基座提升 ≥2.0 点；
2. 相对无特权对照提升 ≥1.0 点（证明多视角特权而非仅自蒸馏）；
3. VSI@32、CV-Bench 相对基座下降 ≤1.0 点（不退化）；
4. 收益集中在对视角数敏感的题型（计数、房间尺寸、appearance order 等）。

---

## 7. 相关文档

| 文档 | 路径 |
|------|------|
| 实验计划（完整版） | [`MV_OPSD_Experiment_Plan.md`](../../MV_OPSD_Experiment_Plan.md) |
| Vision-OPD 实现分析 | [`Vision_OPD_OPSD_Analysis.md`](../../Vision_OPD_OPSD_Analysis.md) |
| 当前实验状态 | [`experiments/ACTIVE.md`](../ACTIVE.md) |
| 基座 vs SPAR3 step 55 评测对照 | [`评测对照_基座_vs_SPAR3_k1_step55.md`](评测对照_基座_vs_SPAR3_k1_step55.md) |
| v1 实验注册表 | [`experiments/registry/20260817_qwen35base_mvopsd_v1.yaml`](../registry/20260817_qwen35base_mvopsd_v1.yaml) |
| 训练配置 | [`verl_pkg/verl/trainer/config/mvopsd.yaml`](../../verl_pkg/verl/trainer/config/mvopsd.yaml) |
