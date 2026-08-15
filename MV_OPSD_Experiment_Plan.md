# MV-OPSD 实验计划：多视角特权的 On-Policy 自蒸馏用于空间推理

> 目标任务：3D spatial reasoning
> 评测：VSI-Bench（主）、CV-Bench、BLINK-Spatial、SPAR-Bench、MMSI-Bench、Video-MME
> 基座：Qwen3.5-4B
> 训练框架：`verl_pkg/verl`（0.7.0.dev，已内置 vopd 补丁）
> 训练数据：**逐源照抄 SFT 的采样率（`%60/%60/%60/%50`）后，再取全部能构造视角落差的样本，共 134,840 条**（SPAR 47.2% / llava_hound 28.4% / VLM-3R ScanNet 23.0% / VSI-590K 1.4%），不针对任何 benchmark 重新设计
> 参考实现：Vision-OPD（见 `Vision_OPD_OPSD_Analysis.md`），**仅作参考，不照搬**
>
> 本文中凡标注 `待核对` 的条目为尚未由代码或实测证据确认的假设，不得当作既定结论使用。

---

## 1. 一句话方案

把 Vision-OPD 中"crop 特权视图教全图"的机制，替换为**"多视角特权教有限视角"**：teacher 观察同一 3D 场景的 $N$ 个视角，student 只观察其中 $K \ll N$ 个视角（且包含锚点视角），二者在 student 自己采样出的同一条 response 上做 token 级分布匹配，从而让 student 在视角预算受限时仍能复现多视角条件下的空间推理分布。

关键结论先行：

1. **verl 侧的多视角 teacher 路径已经可用，v0 无需改动 verl 核心代码。** `_get_gen_batch` 已把 `teacher_prompt` 列列入保留键（`verl_pkg/verl/trainer/ppo/ray_trainer.py:1512`），`_build_teacher_messages_from_template` 按 `teacher_prompt` 中 `<image>` 占位符数量插图（同文件 `761-793`），而不是要求与 student 的图数相等。因此 $N \neq K$ 只需在数据准备阶段多写一列。
2. **主要工作量在数据构造与评测协议，而不是算法实现。**
3. **最大的两个技术风险是 response 过短（监督 token 太少）和视觉 token 预算爆炸**，必须在 Phase 0 用实测数据先行排除。
4. **照抄 SFT 采样率后纳入全部能构造视角落差的数据，机制归因靠无特权对照而非 SFT 基线。** 采样率逐源照抄 SFT（`%60/%60/%60/%50`），但多视角过滤只砍 SPAR，占比仍从 66.4% 降到 47.2%，因此与 SFT 基线的差值不再是纯训练信号；主张一律以 $N{=}K$ 对照为准，它与主实验共用同一份数据（§6.5）。同时必须强制 `loss_agg_mode=seq-mean-token-mean`，否则 SPAR 3-view 会以 29.4% 的样本量拿走 57.2% 的梯度（§6.5.1）。

### 1.1 分阶段推进：先跑通 v0，再逐步加回复杂度

本文档 §5–§9 描述的是**目标状态**，不是第一次要跑的东西。那套设计（四源等比配比、随机视角预算、三级答案过滤、分池诊断、六 benchmark 评测）防御的都是"已经有结果之后才会暴露"的失效模式；在还不知道机制是否成立时先上全套，只会把第一次反馈推迟一到两周。

因此第一次训练用下面的 **v0 版本**，目标不是拿到可发表的数字，而是**证明管线通、并看到特权信号是否存在**。

**v0 的唯一特权轴是视角数量。** 分辨率、深度图、点云、bbox 一律不引入。teacher 与 student 的每视角分辨率必须严格相同，否则 teacher 的优势同时来自"看得更多"和"看得更清"，两个轴混在一起就无法归因。实现方式见下。

| 维度 | v0 | 目标状态（§5–§9） |
|---|---|---|
| 数据源 | **四源均照抄 SFT 采样率，再过滤出可用样本，134,840 条** | 同左 |
| 规模 | **固定步数预算（300 步 / 19.2k 样本，占池 14.2%），不跑满 1 epoch** | 跑满 1 epoch（2,107 步） |
| 视角 | **$N$：SPAR 32 或 3 / 视频源 8；$K$ 逐样本随机取 $\{1,2,4\}$**（SPAR 有标记题型受标记帧约束，见下） | 同左，可扩至 $K{=}8,16$ |
| 每视角分辨率 | **teacher = student = SFT，384×512（192 token）** | 同左 |
| 答案过滤 | 仅 `point_img_idx` / `bbox_img_idx` 集合相交 | 三级过滤（规则 + 模型 + 人工抽检） |
| 损失聚合 | `seq-mean-token-mean`（**必选，非可选**） | 同左 |
| 评测 | VSI-Bench @ {1, 2, 4, 16} + CV-Bench | 六 benchmark + 完整帧预算曲线 |
| 对照 | 主实验 + 无特权（$N{=}K$ 逐样本相等）两条 | 五条基线 + 11 轴消融 |
| 诊断 | 分池 / 分 $K$ / 分视角选择方式三组分桶 | 再加分特权强度分桶 |

评测预算必须覆盖 $\{1,2,4\}$，因为训练在这三个点上都有样本；只测 $K{=}4$ 会看不到 $K{=}1$ 是否学到了东西。

保留全部四源、只砍训练步数与诊断复杂度，好处是数据管线一次搭对，v1 扩规模时不必推倒重来；也不会因为先在单池上验证而得出无法外推的结论。

代价是 §6.5 的两个约束在 v0 就要承受：SPAR 3-view 的 $3\to1$ 弱特权占 29.4%，以及 `token-mean` 下 57.2% 的梯度会被 SPAR 3-view 吃掉。后者正是 **v0 必须把 `loss_agg_mode` 设为 `seq-mean-token-mean` 的原因**——这是全源配比下唯一不能留默认值的参数。

#### 数据就绪状态（已核实）

| 数据源 | 标注 | 媒体 | v0 前置工程 |
|---|---|---|---|
| `spar_234k` | `data/annotations/spar_234k.json` | `data/media/spar`（166 G，独立 jpg） | 仅缩放 |
| `llava_hound_64k` | `data/annotations/llava_hound_64k.json` | `data/media/llava_hound/frames`（149 G，301,751 个帧目录） | 仅缩放，已是帧 |
| `vlm3r_scannet` | `data/annotations/merged_qa_scannet_train.json` | `data/vlm3r/media/scannet/videos`（19 G，1,201 个 mp4） | **抽帧 + 缩放** |
| `vsi_appr_order` | `data/annotations/vsi_appearance_order_vsibench_scannet.json` | 软链至同一批 mp4 | 与上共用抽帧结果 |

四源标注与媒体均已就位，无需下载。

#### v0 只有一个前置脚本，且它不是可选项

抽帧与缩放本质是同一件事的两半，合并为**一个预处理脚本**。**按唯一文件去重后的实际工作量远小于样本×视角的展开量**：

| 来源 | 展开量 | 去重后 | 说明 |
|---|---|---|---|
| SPAR | 894,741 | **128,228** | 同场景的帧被大量复用 |
| `llava_hound_64k` | 306,000 | **306,000** | 每视频独立，已是帧目录 |
| `vlm3r_scannet` + `vsi_appr_order` | 263,808 | **9,608** | 1,201 场景 × 8 帧，两源共用 |
| 合计 | 1.46 M | **≈ 444 K** | |

ScanNet 用 ffmpeg 抽帧时直接 `-vf scale` 到目标尺寸，SPAR 与 llava_hound 用 PIL 过一遍。缓存约 15–20 GB，并行下数小时可完成。

两半都不是设计偏好，各自被一条硬约束逼出来：

**抽帧被 verl 逼出来。** `vlm3r_scannet` 与 `vsi_appr_order` 在磁盘上只有 mp4，而 teacher 侧构造输入时把 `videos=None` 写死（`verl_pkg/verl/trainer/ppo/ray_trainer.py:995`），teacher 物理上收不到视频。因此"包含全部四源"这个选择本身就蕴含"ScanNet 必须先变成帧"。唯一的规避方式是砍掉这两源。

**缩放被算术逼出来。** verl 的 teacher 路径不做任何缩放，实测 SPAR 原图为 1,693 token/视角（880–2,700），16 视角即 2.7 万 token，而 `model_max_length=12800`。**不缩放不是不够干净，是直接跑不起来。** 消除分辨率差异是做完这一步后顺带获得的，不是做它的动机。

#### 零分辨率差异不需要改 verl 代码，但需要改一个配置

verl 的两条图像路径确实不同：student 经 `process_image`（`verl_pkg/verl/utils/dataset/rl_dataset.py:207`）→ `fetch_image` 做 patch 对齐缩放；teacher 走裸 `Image.open`（`ray_trainer.py:713-728` 的 `_normalize_teacher_image`）。但 `smart_resize` 在**宽高已是对齐因子的整数倍、且总像素在界内时原样返回**，此时两条路径输出逐像素相同。

对齐因子 = `patch_size × merge_size`。Qwen3.5 实测为 $16 \times 2 = 32$（见下方几何实测），而 **verl 默认 `image_patch_size=14`，因子为 28**，两者不符会导致 student 被改、teacher 不变：

```
factor 28（verl 默认）: (384, 512) -> (392, 504)   ← 分辨率差异复活
factor 32（patch=16）:  (384, 512) -> (384, 512)   ← 两侧一致
```

因此 v0 需要两件事，都不涉及 verl 代码：

1. 预处理脚本把图落盘为 **384×512**（宽幅源图 288×512），并**断言输出宽高恒为 32 的倍数**；
2. verl 数据配置设 **`data.image_patch_size=16`**（`verl_pkg/verl/trainer/config/data/legacy_data.yaml:71`）。

漏掉任一条，分辨率差异都会静默复活。§6.4 的 `image_grid_thw` 逐视角断言就是拦这个的。

#### SFT 侧的抽帧行为（已核实，直接约束 v0 参数）

SpatialStack **有抽帧，但在 dataloader 里在线完成，不落盘**（`src/qwen_vl/data/data_qwen.py:433-441`，decord `VideoReader`）。且抽完即当图片用：`_get_item` 把 `<video>` 占位符替换为 $N$ 个 `<image>`，再令 `sources[0]["image"] = sources[0]["images"]`，后续完全走多图流程（同文件 `450-477`）。**这与本方案 teacher 需要的输入形态一致**，所以 v0 的离线抽帧不是另起炉灶，只是把 SFT 的在线步骤提前落盘（因为 verl 的 teacher 侧 `videos=None`）。

由此得到两条硬约束：

1. **视频源的 $N$ 不应超过 8。** `get_frame_indices`（同文件 `413-424`）取 `min(max(round(时长/2), 4), video_max_frames)`，而 `scripts/train/train.sh:126` 设 `--video_max_frames 8`。因此 SFT 基线在 `vlm3r_scannet` 与 `llava_hound` 上**从未见过超过 8 帧**。v0 若给 teacher 更多帧，teacher 一开始就处于初始策略的分布外，特权虽大但可能不可靠。**因此视频源取 $N{=}8$；SPAR 图像源不受此限（SFT 见过 32 图），取 $N{=}32$。**
2. **抽帧规则必须照抄 `np.linspace(0, total_frames-1, target_frames)` 均匀采样**，否则帧位置分布与 SFT 不一致，又多一个混淆变量。

#### SPAR 32-view 在 SFT 中**没有**被截断（原推断已被实测推翻）

此前基于 token 算术推断"SPAR 32-view 样本的答案被 `model_max_length` 截断、未提供监督信号"。**实测证伪。** 用 `scripts/opsd/check_spar_truncation.py` 走真实的 `_get_item` + collator 切片：

| views | 样本总长 | 答案 token | 截断后保留 | 丢失 |
|---|---|---|---|---|
| 1 | 291 | 34.7 | 34.7 | 0.0% |
| 2 | 430 | 38.7 | 38.7 | 0.0% |
| 3 | 718 | 38.5 | 38.5 | 0.0% |
| **32** | **6,506** | **33.3** | **33.3** | **0.0%** |

原推断错在每视角 token 数：估的是 540，实际只有 192。**因此不存在"基线在 32-view 形态上未经训练"这一有利前提，OPSD 的增益不能靠这条来解释。**

#### SFT 的真实每视角几何（实测，v0 必须对齐）

用 `scripts/opsd/probe_sft_image_geometry.py` 直接测 `prepare_image_inputs`，得到三条此前都搞错的事实：

1. **`patch_size=16`、`merge_size=2`，对齐因子是 32，不是 28。** 此前所有 token 算术与"28 对齐"的脚本纪律都要改成 32。
2. **`--max_pixels 576*28*28`（`train.sh:123`）对图像输入实际是失效参数。** `prepare_image_inputs`（`src/qwen_vl/data/utils.py:157-171`）先调 VGGT 的 `load_and_preprocess_images` 把图硬缩放到 518 px 宽，再裁到 32 的倍数，Qwen 的 `max_pixels` 根本没机会生效。
3. **SFT 实际喂给模型的是 384×512、192 token/视角**（4:3 源图；少量宽幅源图为 288×512、144 token）。SPAR 的 1920×1440 与 1296×968 两种原生分辨率都收敛到同一几何。

| 路径 | 每视角 token | 8 视角 | 16 视角 | 32 视角 |
|---|---|---|---|---|
| **SFT 实际（384×512）** | **192** | 1,536 | 3,072 | **6,144** |
| verl teacher 当前（裸 `Image.open`） | 1,693（880–2,700） | 13,544 | 27,088 | 54,176 |

**这直接改善了 v0 的视角预算**：按 SFT 的 192 token/视角，$N{=}32$ 的 teacher 只需 6,144 visual token，在 12,800 上下文里绰绰有余。因此 SPAR 图像源可直接取 $N{=}32$，配合 $K \in \{1,2,4\}$ 得到 8×–32× 的特权 gap。

同时也确认了缩放本身仍然必需：verl 的 teacher 路径不做缩放，16 视角就要 2.7 万 token，照样跑不起来。

#### $K$ 随机取 $\{1,2,4\}$，但 SPAR 的标记帧不可丢

student 视角数 $K$ 逐样本从 $\{1,2,4\}$ 随机抽取。两类数据源的约束完全不同，必须分开处理。

**视频源（`vlm3r_scannet`、`llava_hound_64k`、`vsi_appr_order`）：无约束。** 问题是自然语言表述，不绑定具体帧，$K$ 可在 $\{1,2,4\}$ 中自由抽取，视角**均匀随机采样**。这与评测端的行为一致（评测同样是 `np.linspace` 均匀抽帧），因此不引入训练/评测偏差。

**SPAR 分两类，不能一刀切。** 关键在于该题型是否用视觉标记指代物体。

**SPAR 无标记题型（`obj_count` / `appearance_order` / `room_size`，32-view 中 8,691 条）：与视频源同规则。** 这类问题问的是整个房间（"how many electrical panel can be found in the room?"），不指代特定物体，因此 `spar_info` 里没有 `point_img_idx`——**不是标注缺失，而是本就不需要**。它们是本方案质量最高的一批样本：无标记帧约束可纯随机采样、题型正是 VSI-Bench 的核心三类、$32 \to K$ 特权极强、问题文本不含 `Frame-N` 引用连重写都不需要。**按"无可见性标注即排除"处理会误伤它们，必须按题型白名单显式恢复。**

**SPAR 有标记题型：标记帧必须保留。** 被问物体是靠画在指定帧上的彩色标记指代的（`src/qwen_vl/data/draw_marker.py:105-160`，`draw_fn` 按 `point_img_idx` / `bbox_img_idx` 在对应帧上画红/蓝/绿点或框），且问题文本会直接点名帧号：

> "How far apart are the centers of shower curtain **(in Frame-23) (red point)** and towel **(in Frame-10) (blue point)**..."

丢掉 Frame-23 会同时造成两件事：问题里的帧号引用悬空，红点标记在任何视角中都不存在。**这不是"目标看不见"，而是问题失去指代对象、输入本身非法。** 全量 SPAR 中 38.7% 的样本在文本里显式提及彩色标记。

因此 SPAR 有标记题型的视角选择规则为：

1. 令 $K_{\min} = |\text{set}(\texttt{point\_img\_idx})|$，即标记帧的数量；
2. $K$ 从 $\{1,2,4\} \cap [K_{\min},\ N)$ 中随机抽取，该集合为空则排除该样本；
3. 选中的 $K$ 个视角 = **全部 $K_{\min}$ 个标记帧** + 其余 $K - K_{\min}$ 个从剩余视角中**均匀随机**抽取；
4. 视角重编号后，**必须同步重写问题文本中的 `Frame-N` 引用**，否则帧号与实际位置错位，会造成静默的语义污染。

第 3 条保留了随机性（非标记帧随机填充），同时保证输入合法。第 4 条是 SPAR 有标记题型特有的、极易遗漏的一步，必须写单元测试。

实测各池在此规则下的可用量与 $K$ 可选集（数值为按 SFT `%60` 抽样之后）：

| 池 | 抽样后 | 可用 | $K \in \{4\}$ | $K \in \{2,4\}$ | $K \in \{1,2,4\}$ |
|---|---|---|---|---|---|
| SPAR 32-view | 26,671 | **23,991（90.0%）** | 52.1% | 19.4% | 28.5% |
| SPAR 3-view | 53,775 | **39,623（73.7%）** | — | 66.5%（实为 $\{2\}$） | 33.5%（实为 $\{1,2\}$） |

排除项在两组之间完全不同，原因也不同：

| 排除原因 | 32-view | 3-view |
|---|---|---|
| $K_{\min} = N$，零特权（student 已看全，等同对照组） | 0 | **8,881**（$K_{\min}{=}3$） |
| $K_{\min} > 4$，超出 $K$ 上限 | **2,680**（$K_{\min}{=}5$） | 0 |
| 答案本身是视角编号（`distance_infer_center_oc_mv`） | 0 | **5,271** |

3-view 的 5,271 条问的是 "Which object appears closest to the observer? **Choose an image** showing the object"——答案是一个视角编号，删视角会直接改变正确答案，必须排除。32-view 的 2,680 条只是超出 $K$ 上限，若后续把 $K$ 菜单扩到 $\{1,2,4,8\}$ 即可全部恢复（届时 32-view 可用率达 100%），但 v0 不做。

**残留代价**：SPAR 有标记题型上 student 始终能看到标记帧，即视角选择部分是 oracle 引导的，与评测时的均匀采样不同。这一项无法在该题型上消除（否则输入非法）。但**视频源与 SPAR 无标记题型完全没有这个问题，两者合计占训练池 58.7%**。因此 §9.3 应分"oracle 引导 / 纯随机"两桶报告，前者衡量受控条件下的特权收益，后者才是与评测同分布的证据。

#### 这套几何与 VGGT 无关，v0 不上 VGGT 也照样继承

518 缩放虽然来自 VGGT 的工具函数 `load_and_preprocess_images`（`src/qwen_vl/data/utils.py:31`，`target_size=518` 是 VGGT/DINOv2 的原生输入尺寸），但 **`prepare_image_inputs` 在 `data_qwen.py:499` 是无条件调用的**，并未被 `use_geometry_encoder` 门控——门控只在同文件 `595` 行决定要不要把 `geometry_encoder_inputs` 放进 batch。

基线 `spatialstack_qwen35_novggt_aligned` 的 `config.json` 中确实没有任何 geometry 相关键（纯 Qwen3.5，`vision_config.patch_size=16`、`spatial_merge_size=2`），但它训练时每张图仍被压到 384×512。因此 **v0 不使用 VGGT 这一决定不改变上述几何结论**，它只是意味着我们继承了一个源自 VGGT 的历史遗留缩放。

#### SFT 自身存在训练/评测几何不一致（新发现）

评测端**不走** `load_and_preprocess_images`，而是直接用 `AutoProcessor(min_pixels=256*28*28, max_pixels=1605632)`（`src/lmms_eval/models/qwen3_5.py:124-125, 208-213`），并把视频帧同样当作 `{"type": "image"}` 送入（同文件 `314-347`）。抽帧规则两边一致（都是 `np.linspace` 均匀采样），但分辨率不一致：

| | 每帧 token | 32 帧 |
|---|---|---|
| SFT 训练（VGGT 518 缩放 → 512×384） | **192** | 6,144 |
| 评测（ScanNet 原生 640×480，未缩放） | **300** | 9,600 |

ScanNet 视频原生 640×480 = 307,200 px，正好落在评测的 $[200704,\ 1605632]$ 区间内，因此评测时完全不缩放。基线的 VSI-Bench 64.24 就是在 300 token/帧 下测得的。

**这个不一致不影响 v0 的判据**（主实验与无特权对照在同一几何下训练、同一几何下评测），但影响与 SFT 基线比较时的可归因性：若 v0 改用评测几何训练，收益里会混入"顺手修复了原有 train/eval 不一致"的成分。

**处理方式：用纯评测探针先行判定，不增加训练轮次**（P0-8，见 §13）。实测 `max_pixels=196608` 可在评测端精确复现训练几何：

```
max_pixels=1605632（评测默认） -> 640x480 = 300 token/帧
max_pixels= 196608            -> 512x384 = 192 token/帧
```

只需把 SFT 基线在两种 `max_pixels` 下各测一次 VSI-Bench。若差距在 0.5 分以内，说明该 checkpoint 对几何不敏感，选哪个都行——取 192 token 以换取更大的上下文余量；若 300 明显更优，则训练侧统一改用 300 token（$32 \times 300 = 9{,}600$，仍可容纳）。

#### 预处理的目标几何与一处必须改的配置

v0 的预处理脚本默认把所有视角**落盘为 384×512**（宽幅源图 288×512），即完全复刻 SFT 的几何。这样 student 侧与 SFT 初始化完全同分布，少一个混淆变量。**若 P0-8 判定评测几何显著更优，则改为 640×480（300 token/视角），脚本只需换一个目标尺寸参数。**

但仅此还不够。verl 的 `image_patch_size` 默认是 **14**（`verl_pkg/verl/utils/dataset/rl_dataset.py:109`），与 Qwen3.5 的 16 不符，后果是 student 侧的 `smart_resize` 会按因子 28 改动尺寸而 teacher 侧不变：

```
factor 28（verl 默认）: (384, 512) -> (392, 504)   ← student 被改，teacher 不变
factor 32（patch=16）:  (384, 512) -> (384, 512)   ← 两侧一致
```

**因此必须在 verl 数据配置中设 `data.image_patch_size=16`。** 这是配置项而非代码改动（`verl_pkg/verl/trainer/config/data/legacy_data.yaml:71`），v0 仍保持零 verl 代码修改。漏设这一项，分辨率差异会静默复活且极难发现——这正是 §6.4 那条 `image_grid_thw` 断言要拦的情况。

#### 推进顺序

- **v0**：上表配置，跑主实验 + 无特权对照两条。判据只有一条——**主实验在 VSI@4 上高于无特权对照**。达不到就回到 P0-2 查特权 gap，不要靠加数据或加复杂度补救。
- **v1**：加回随机视角预算、$N{=}32$、三级过滤与完整诊断，跑完整帧预算曲线与六 benchmark。这一版才产出主结果表（§9.1）。
- **v2**：§7.3 的消融轴，按优先级从上往下做。

---

## 2. 为什么不能照搬 Vision-OPD

| 维度 | Vision-OPD | 本项目（MV-OPSD） |
|---|---|---|
| 任务 | 细粒度 2D 视觉理解（单图找小目标） | 3D 空间推理（跨视角几何一致性、度量估计、拓扑关系） |
| 特权来源 | 空间**分辨率**（crop 放大证据区域） | 视角**覆盖度**（更多相机位姿 → 更完整的场景几何） |
| 特权标注成本 | 需要逐样本 bbox 标注 | **零额外标注**：多视角本来就在数据里，只需子采样 |
| 输入形态 | 单图 | 多图序列 / 视频帧序列 |
| student 是否知道证据位置 | 知道（全图带红框） | **不知道**：缺失的视角是真正不可见的信息 |
| 答案长度 | 较长 response（1024 token） | 分化极大：`vlm3r_scannet` 中位数 **1 token**，SPAR 3-view 达 **77 token**（§5 D2） |
| 部署收益 | 推理时不需要 crop 工具 | 推理时**帧预算更低**（延迟、显存、真实机器人场景） |

第三行和第五行是本方案相对 Vision-OPD 的两个实质优势与实质困难：特权信号免费获得，但 student 面临的是真正的信息缺失（information-theoretic gap），而不是 Vision-OPD 那种"信息都在、只是分辨率不够"的可恢复缺失。

这带来一个必须正视的理论问题，也是本方案最核心的科学问题：

> **当 teacher 拥有 student 在原理上无法获得的信息时，蒸馏应当传递什么？**

答案不应该是"让 student 猜出它看不见的东西"，而应该是：

- 传递**视角不变的空间先验**（物体尺度分布、房间尺度先验、常见布局），
- 传递**从局部视角外推全局结构的推理方式**，
- 并让 student 在无法确定时保持**恰当的不确定性**（这正是 JSD 相对 hard label 的价值所在）。

这一点会直接影响 §6 的数据筛选规则（必须剔除"student 视角下答案根本不可判定"的样本）和 §9 的诊断指标设计。相关文献侧的未决问题见 `literature/OPEN_QUESTIONS.md` 的 Q-005、Q-006、Q-008。

---

## 3. 方法形式化

设一个场景样本包含视角集合 $\mathcal{V} = \{v_1, \dots, v_N\}$（按时间/轨迹顺序），问题 $q$，答案 $a$。

- Student 视角子集：$\mathcal{S} \subset \mathcal{V}$，$|\mathcal{S}| = K$，且强制 $v_1 \in \mathcal{S}$（锚点约束，见 §5-D4）。
- Teacher 视角集：$\mathcal{T} = \mathcal{V}$，$|\mathcal{T}| = N$。

On-policy rollout 只由 student 产生：

$$
y \sim \pi_{\text{rollout}}(\cdot \mid \mathcal{S}, q)
$$

Teacher 用 EMA 权重 $\bar\theta$，在**同一条** $y$ 上做 teacher forcing，不重新生成：

$$
q_t(v) = \pi_{\bar\theta}(v \mid \mathcal{T}, q, y_{<t}),\qquad
p_t(v) = \pi_{\theta}(v \mid \mathcal{S}, q, y_{<t})
$$

损失沿用 verl 中已实现的 top-$k$ generalized JSD 加两级重要性采样校正（`verl_pkg/verl/trainer/ppo/core_algos.py:1085-1210`）：

$$
\mathcal{L} = \sum_t w_t^{\text{rollout}} \cdot r_t \cdot
\Big[(1-\alpha)\mathrm{KL}(p_t \| m_t) + \alpha \mathrm{KL}(q_t \| m_t)\Big],
\quad m_t = (1-\alpha)p_t + \alpha q_t
$$

Teacher 参数按 $\bar\theta \leftarrow (1-\beta)\bar\theta + \beta\theta$ 更新（`verl_pkg/verl/workers/actor/dp_actor.py:134-156`）。

**与 Vision-OPD 唯一的算法层差异是 teacher 的视觉条件构造方式。** 这是刻意的：v0 先只改变特权信号的语义，保持优化器一侧完全可比，才能干净地归因收益来源。

---

## 4. 现状盘点（已验证事实）

### 4.1 代码

| 事项 | 结论 | 证据 |
|---|---|---|
| verl 版本 | 0.7.0.dev，已含 vopd 补丁 | `verl_pkg/verl/version/version` |
| teacher 图列表 | 支持**多图列表**，不限单图 | `ray_trainer.py:1303-1309` |
| $N \neq K$ 支持 | **已支持**，通过 `teacher_prompt` 列 | `ray_trainer.py:1299-1301, 761-793, 795-803` |
| `teacher_prompt` 是否被 rollout 丢弃 | **不会**，已在保留键集合中 | `ray_trainer.py:1512` |
| student/teacher 视觉 token 数不同 | **允许**，按 `teacher_response_start_idx` 对齐 response | `dp_actor.py:208-226, 1012-1037` |
| 动态 micro-batch 分桶依据 | 仅按 student `attention_mask` 长度 | `seqlen_balancing.py:374-380` |
| teacher 侧 processor 调用 | `videos=None` 硬编码 | `ray_trainer.py:995` |
| teacher 图片分辨率控制 | **不走 `fetch_image`**，直接 `Image.open` 后交 processor | `ray_trainer.py:714-729` |
| student 图片分辨率控制 | 走 `fetch_image`，支持 dict 内 `max_pixels` | `vision_utils.py:23-35` |
| mRoPE / 3D 位置编码 | Qwen3.5 分支已处理 | `ray_trainer.py:1003-1060` |
| teacher 来源 | `legacy`(EMA ref) / `current` / `fixed` | `fsdp_workers.py:896-931`，`workers/config/actor.py:105-165` |
| checkpoint 转 HF | `verl/model_merger/fsdp_model_merger.py` | 已存在 |
| 现成 vopd 启动脚本 | **不存在**（`scripts/run_vision_opd.sh` 仅在分析文档中被引用） | `scripts/` 下无匹配 |

倒数第三行和第四行合起来意味着一个具体的实现陷阱：**student 与 teacher 的每视角分辨率由两条不同的代码路径决定**。若不处理，会引入"分辨率"这个混杂变量，破坏"唯一变量是视角数"的实验设计。解决方案见 §6.4。

### 4.2 数据

`data/train/spar_234k.json`（234,277 条）按图数分布：

| 图数 | 条数 | 问题形态 | 可用于视角子采样 |
|---|---|---|---|
| 32 | 44,606 | `Frame-i: <image>` 序列 + 空间关系/度量问题 | **是（首选池）** |
| 3 | 89,466 | BEV 构建，显式声明"first image 为主视角" | 有条件可用（见 §6.2） |
| 2 | 24,943 | `view_change_infer`，问"从第一张到第二张的相机移动" | **否，必须剔除** |
| 1 | 75,262 | 单图深度/关系判断 | 否（无多视角可降） |

其他数据源：

| 数据集 | 条数 | 形态 | 答案长度中位数 |
|---|---|---|---|
| `merged_qa_scannet_train.json` | 51,779 | ScanNet 视频 + VSI 风格问题 | 1 词 |
| `vsi_appearance_order_*.json` | 3,819 | ScanNet 视频，appearance order 选择题 | 1 词 |
| `llava_hound_64k.json` | 63,750 | 2D 视频描述 | 长文本 |

**32 视角 SPAR 子集的 `Frame-i: <image>` 格式是本方案最理想的数据形态**：视角数天然可变，问题文本不绑定具体视角索引语义，只需重编号即可。

ScanNet 视频池（合计 55,598 条）与 VSI-Bench 的任务分布最接近，是提升目标 benchmark 的主力，但需要离线抽帧成图片序列（verl teacher 路径不支持 video，见 4.1）。

### 4.3 模型与基线

`output/` 下目前只剩一个 checkpoint：

| Checkpoint | 存在 | VSI-Bench | CV-Bench |
|---|---|---|---|
| `spatialstack_qwen35_novggt_aligned` | **在盘（9.7 GB）** | 64.24 | 84.97 |
| `spatialstack_qwen35_vggt_aligned` | 已删除 | 67.65 | 84.91 |
| `spatialstack_qwen35_vggt_retrain` | 已删除 | 66.82 | 85.04 |
| `spatialstack_qwen35_novggt` | 已删除 | 63.01 | 85.44 |
| `spatialstack_qwen35_train` | 已删除 | 64.56 | 82.09 |

（评测结果 JSON 位于 `logs/eval/spatialstack_qwen35_4b*/`；对应实验记录见 `experiments/registry/20260813_qwen35_geo_baseline.yaml`。）

### 4.4 算力

- 本机 **8× H20（约 96 GB/卡）**，驱动 570.133.20。
- Slurm 参考脚本存在 8 节点×8 卡（训练）与 12 节点×4 卡（评测）配置，但本计划按**单节点 8 卡**做主路径设计。

---

## 5. 需要拍板的设计决策

以下 6 项决定实验的可行性与可解释性，建议在启动 Phase 0 前确认。每项给出推荐值与理由。

### D1. VGGT 几何编码器是否进入 RL 环？

**推荐：不进入。v0 使用纯视觉 Qwen3.5-4B。**

理由：SpatialStack 的几何模型是自定义类 `Qwen3_5ForConditionalGenerationWithGeometry`，vLLM rollout、FSDP 包装、`geometry_encoder_inputs` 在 rollout/teacher 两条路径上的透传均无现成支持，工程量远超本方案主线。而 `novggt_aligned`（VSI 64.24 / CV 84.97）已是可用且在盘的强起点，与 vggt 版本仅差 3.4 个 VSI 点。

代价：最终数字会低于 SpatialStack 的最好结果。但本实验的论证目标是**MV-OPSD 相对同一起点的增量**，不是刷 SOTA。几何编码器可作为后续正交工作。

### D2. Response 长度 / 是否开启 thinking？

用 Qwen3.5 tokenizer 实测各池的 GT 答案 token 数（每池随机采样 800–1500 条）：

| 数据池 | 中位数 | 均值 | 说明 |
|---|---|---|---|
| `vlm3r_scannet`（Pool B） | **1** | **1.4** | 极短，"2.3"、"C" 之类 |
| `vsi_appr_order`（Pool D） | **1** | **1.0** | 全部单 token |
| SPAR 32-view（Pool A） | 32 | 34.5 | 其中 `obj_count` 仅 13.1，`spatial_imagination_oo_video` 达 63.2 |
| SPAR 3-view BEV（Pool C） | 77 | 69.7 | 多行坐标列表，最长 |
| `llava_hound_64k`（Pool E） | 24 | 31.2 | 自由文本描述 |

**这修正了草稿中"SpatialStack 数据答案中位数 1 个词"的笼统说法。** 实际情况是分化极大：与 VSI-Bench 对齐度最高的 Pool B/D 恰恰是最短的（1 token），而 SPAR 系列反而有 32–77 token。这个分化本身会造成严重的梯度失衡，处理方式见 §6.5.1。

风险依然成立但更精确：**与 VSI-Bench 措辞最对齐的 `vlm3r_scannet` 的 rollout 只有 1–3 个 token**，OPSD 在其上退化为对单个答案 token 的软标签蒸馏，逐 token 分布匹配的优势基本消失（Vision-OPD 用的是 1024 token 的 response）。

三个选项：

| 选项 | 说明 | 风险 |
|---|---|---|
| A. 保持短答案 | 直接在 1–5 个 token 上做 JSD | 监督量极低；但等价于"带不确定性的软标签蒸馏"，仍可能有效，且完全匹配评测形态 |
| B. 开启 thinking | rollout 时 `enable_thinking=True`，在 CoT + 答案上蒸馏 | SFT checkpoint 是按直答训练的，是否还能产出有效 CoT `待核对`；训练/评测形态不一致 |
| C. 混合 | 短答案池走 A，SPAR/llava_hound 等长答案池天然提供密集 token 监督 | 实现简单，但需显式管理 token 质量配比（§6.5.1） |

**推荐：C，并在 Phase 0 实测后复核。** 保持 `enable_thinking=false`（与评测形态一致，见 `models/Qwen3.5-4B/chat_template.jinja:149-152`），靠长答案池提供 token 监督密度。

`待核对`：上表是 **GT 答案**长度，不是 rollout 长度。student 实际生成的 response 可能更长（尤其是未充分对齐时）。P0-3 必须测真实 rollout 分布，配比结论需据此复核。

### D3. 评测协议：帧预算曲线

**推荐：主指标改为"给定帧预算下的性能"，而非单点数字。**

导师的目标是"让 student 学会 limited view 的泛化"。如果 student 用 8 视角训练却在 32 帧下评测，训练/评测的视角预算不一致，收益会被稀释甚至反转。因此：

评测覆盖六个 benchmark，按视觉输入形态分两层。**帧预算维度只在视频类 benchmark 上成立**，因为它由 lmms-eval 的 `max_num_frames` 控制（`src/lmms_eval/models/qwen3_5.py:297-312`）；图像类 benchmark 的视觉输入条数由数据集本身固定。

**第一层：帧预算曲线（主指标）**

| Benchmark | 输入形态 | 作用 |
|---|---|---|
| VSI-Bench | 单个 mp4 | **主指标**，@ {1,2,4,8,16,32} 帧完整曲线 |
| Video-MME | 单个 mp4 | 通用视频理解，主要作**不退化护栏**（尤其对 Pool E 的取舍敏感） |

核心主张是 **MV-OPSD 把曲线的左半段抬起来**：低帧预算下收益最大，高帧预算下不退化。摘要引用的单点数字取低预算区间（1/2/4 帧）的平均提升。

**第二层：固定视觉输入（跨域泛化与不退化检查）**

| Benchmark | 输入形态 | 作用 |
|---|---|---|
| CV-Bench | **单图**（`tasks/cvbench/utils.py:17-21`） | 天然的 $K=1$ 极限情形，也是单图空间能力护栏 |
| BLINK-Spatial | 图像组 | 含 `blink_multi_view_reasoning` 子项，**与本方法机制最直接相关** |
| SPAR-Bench | 多图列表（`tasks/sparbench/utils.py:96-103`） | 同域检查（训练数据含 SPAR） |
| MMSI-Bench | 多图列表（`tasks/mmsibench/utils.py:25-37`） | 跨域多图空间推理 |

**不退化门槛**：VSI-Bench @ 32 帧、CV-Bench、Video-MME 相对 SFT 基线下降均不超过 1.0 点。

可选扩展：对 SPAR-Bench / MMSI-Bench 的图像列表做子采样，可为它们也构造预算维度，但需自定义 `doc_to_visual`，列为 Phase 3 可选项。

注意 VSI-Bench 总分是 8 个题型的**非加权平均**，而各题型样本数从 194（`route_planning`）到 953（`object_size_estimation`）差近 5 倍。小样本题型对总分的影响被放大、方差也大，**单次评测的 ±0.5 分波动不应解读为真实差异**。建议关键对比跑 2–3 个 seed，或至少报告分题型结果（§9.2）。

注意 VSI-Bench 总分是 8 个题型的**非加权平均**，而各题型样本数从 194（`route_planning`）到 953（`object_size_estimation`）差近 5 倍。小样本题型对总分的影响被放大、方差也大，**单次评测的 ±0.5 分波动不应解读为真实差异**。建议关键对比跑 2–3 个 seed，或至少报告分题型结果（§9.2）。

工程上这几乎零成本：帧预算就是 lmms-eval 的 `max_num_frames` 参数（`src/lmms_eval/models/qwen3_5.py:297-312`），改一个数即可。

### D4. Student 视角预算：固定 $K$ 还是随机 $K$？

**推荐：随机视角预算（stochastic view budget），$K$ 逐样本从分布中抽取，而不是全程固定。**

这一条修正了本方案早期草稿的一个内在矛盾：D3 把主指标定义为**帧预算曲线**，但若训练时固定 $K=4$，模型只会在 $K=4$ 处出现尖峰，曲线其余部分反而可能劣于基线。**指标是曲线，训练就该覆盖整条曲线。**

具体方案：

- 逐样本抽取 $K \sim \text{LogUniform}\{1, 2, 4, 8, 16\}$，teacher 恒定 $N=32$（或该样本可得的最大视角数）。
- teacher 始终保持最大视角，因此**特权 gap 随 $K$ 自动变化**：$K$ 越小 gap 越大、蒸馏信号越强。这是想要的性质，难样本自然获得更大梯度。
- 必须保证 $K < N$，否则该样本退化为无特权样本。

**关键实现优势：零代码改动。** verl 的 Parquet 是静态物化的，且本方案按 1 epoch 训练、每个 prompt 只被访问一次。因此"逐 step 随机抽 $K$"与"在数据准备阶段逐样本抽定 $K$"在期望意义上等价，不需要写自定义 Dataset 类。多 epoch 时再物化多份变体即可。

子采样的四条硬约束（无论 $K$ 是否随机都适用）：

- **均匀间隔**：从 $N$ 个视角中等间隔取 $K$ 个，保覆盖而非聚集，作为默认策略。
- **强制首视角**：SPAR 的 BEV 类问题以 "first image" 定义坐标原点（实测 3-view 池 3,000 条抽样中 1,750 条含 "first image"）。丢掉首视角会让 GT 失效。
- **保序**：空间/时序推理依赖视角顺序，不得打乱。
- **重编号**：`Frame-i:` 标签必须按 student 实际收到的视角重新编号为 `Frame-0..K-1`，否则会泄露"存在缺失帧"这一信息，并造成索引语义错乱。

**嵌套采样（推荐用于诊断）**：让同一问题在不同 $K$ 下的视角集合满足 $\mathcal{S}_1 \subset \mathcal{S}_2 \subset \mathcal{S}_4 \subset \cdots$。这保证单个问题的信息量随 $K$ 单调递增，使"预算-性能"曲线在样本级别可解释，也让 §9.3 的一致性诊断成立。

相关消融（Phase 2）：

| 变体 | 说明 |
|---|---|
| 固定 $K=4$ | **对照组**，验证随机预算是否真的买到了预算泛化 |
| 随机 $K$（默认） | 主方法 |
| 配对多预算 | 同一 (场景, 问题) 物化 2–3 个不同 $K$ 的行，直接施加预算不变性压力 |
| 连续 $K$ 帧 | 取轨迹上连续一段而非均匀采样，模拟真实机器人只看到局部轨迹，更难也更贴近部署 |
| 随机（非均匀）子采样 | 检验覆盖度是否真的重要 |

### D5. Reward-free 还是加正确性门控？

**推荐：v0 纯 reward-free（与 Vision-OPD 主路径一致），门控作为 v1 消融。**

理由：SpatialStack 数据有 GT，且 VSI-Bench 的 MRA / 选择题都可验证，加 reward 是可行的。但文献侧对"门控究竟贡献多少"存在明确的未决争议（`literature/OPEN_QUESTIONS.md` Q-008、Q-014、Q-015）。先跑纯蒸馏才能把该问题作为独立贡献点回答，而不是一开始就混在一起。

需要注意的空间推理特有风险：teacher 看到 32 视角也不保证答对度量类问题（绝对距离、房间尺寸）。若 teacher 本身错得离谱，reward-free 蒸馏会把错误传下去。因此 Phase 0 必须测 teacher 的实际准确率（P0-2）。

### D6. 视觉 token 预算

**推荐：离线预缩放所有视角图片，统一每视角 token 数，并使 teacher 总 token 数落在 `max_reprompt_len` 内。**

按 Qwen 约定，每视角 token 数 $\approx \text{pixels}/(28\times28)$。参考配置：

| 每视角 token | student $K=4$ | student $K=16$ | teacher $N=16$ | teacher $N=32$ |
|---|---|---|---|---|
| 128 | 512 | 2,048 | 2,048 | 4,096 |
| 256 | 1,024 | 4,096 | 4,096 | 8,192 |
| 512 | 2,048 | 8,192 | 8,192 | 16,384（超默认 `max_reprompt_len=10240`） |

**推荐起点：每视角 256 token，$N=32$，$K \in \{1,2,4,8,16\}$，`max_reprompt_len=12288`。** teacher 峰值约 8,192 视觉 token，留出文本与 response 余量。若 P0-6 显示显存吃紧，先降到每视角 128 token（teacher 4,096），而不是降 $N$——保住特权 gap 比保住分辨率更重要，因为分辨率是 Vision-OPD 的特权轴，不是本方案的。

`待核对`：lmms-eval 评测端 `qwen3_5.py` 默认 `max_pixels=1605632`（约 2048 token/图）配 32 帧，理论上远超 `max_length=12800`，实际必然发生了缩放或截断。**必须在 Phase 0 实测评测端的真实每帧 token 数，并让训练端与之对齐**，否则训练/评测的视觉预算不一致会成为最大的混杂因素。

---

## 6. 数据构造方案

### 6.1 目标产物

一个 verl 可消费的 Parquet，schema 如下：

```python
{
    "prompt": [{"role": "user", "content": "Frame-0: <image>\n...Frame-3: <image>\n<question>"}],
    "images":         [{"path": "/abs/path/view_00.jpg"}, ...],   # K 张，student
    "teacher_images": [{"path": "/abs/path/view_00.jpg"}, ...],   # N 张，teacher
    "teacher_prompt": [{"role": "user", "content": "Frame-0: <image>\n...Frame-15: <image>\n<question>"}],
    "reward_model": {"style": "none", "ground_truth": "<answer>"},
    "extra_info": {
        "sample_id": ..., "source": "spar32|scannet_video|spar_bev",
        "question_type": ..., "n_views_teacher": N, "k_views_student": K,
        "view_indices_student": [...], "scene_id": ...,
    },
}
```

**要点**：
- `images` / `teacher_images` 用 `{"path": ...}`（绝对路径），不内嵌 bytes，避免 Parquet 膨胀。
- `teacher_prompt` 的 content 必须是**字符串**且含 $N$ 个 `<image>`。`_build_teacher_messages_from_template` 只处理字符串 content，对 list content 会直接 `continue`（`ray_trainer.py:766-771`）。
- `prompt` 与 `teacher_prompt` 的**问题文本必须逐字相同**，仅 `Frame-i:` 标签行数不同。任何其他文本差异都会污染归因。
- `<image>` 数与图片列表长度必须严格相等，两条路径都有硬校验（`rl_dataset.py:_build_messages` 断言、`ray_trainer.py:788-792`）。

### 6.2 数据池：题型覆盖决定主次

先看一张覆盖矩阵。左列是 VSI-Bench 的 8 个题型及其样本数，右侧是各训练池的可用条数（VLM-3R 的题型由问题模板正则统计得出，措辞与 VSI-Bench 完全一致）：

| VSI-Bench 题型 | 测试样本 | Pool B<br>VLM-3R 视频 | Pool A<br>SPAR 32-view | Pool D<br>VSI-590K | Pool C<br>SPAR 3-view |
|---|---|---|---|---|---|
| `object_size_estimation` | 953 | **4,234** | — | — | — |
| `object_abs_distance` | 834 | **6,960** | 6,816 | — | 18,641 |
| `object_rel_distance` | 710 | **11,038** | 6,940 | — | 17,754 |
| `obj_appearance_order` | 618 | — | 2,439 | **3,819** | — |
| `object_counting` | 565 | **2,565** | 4,889 | — | — |
| `object_rel_direction` | 968 | **25,781** | 19,883 | — | 17,226 |
| `room_size_estimation` | 288 | **1,201** | 1,363 | — | — |
| `route_planning` | 194 | — | — | — | — |
| 合计可用 | 5,130 | 51,779 | 42,330 | 3,819 | ~89,466 |

**这张表是诊断工具，不是配比依据。** 训练配比由"纳入全部可构造视角落差的样本"这条规则决定（§6.5），本表的用途是在**解释结果**时判断某个题型的变化是有对应训练数据支撑的、还是零样本迁移。

从表中可读出三个对结果解释有用的事实：

- `merged_qa_scannet_train.json` 覆盖 8 个题型中的 6 个，措辞与评测完全一致，且**独占 `object_size_estimation`**（VSI-Bench 样本量最大的题型，953 条，SPAR 完全没有对应类型）。但按 SFT 配比它只占 14.67%，所以这些题型的训练暴露量其实很低。
- SPAR 独占 `obj_appearance_order` 的图像形态覆盖，并贡献 ScanNet++ 的 803 个场景。
- **`route_planning`（194 条，占 VSI-Bench 总分 1/8）在所有池中都无对应训练数据**，其变化一律属于零样本迁移，报告时必须注明。

另外，VSI-Bench 的 5,130 条中有 **1,601 条来自 ARKitScenes**（31%），而所有训练池都不含 ARKitScenes 场景。这部分是纯跨域泛化，也是天花板的一部分。

#### Pool A — SPAR 32-view（44,606 条）

`Frame-i: <image>` 格式，问题文本不绑定具体视角索引，直接可用。来源为 ScanNet 24,233 条（1,194 场景）+ ScanNet++ 20,373 条（803 场景）。**ScanNet++ 部分是 Pool B 完全没有的场景多样性来源**，这是 Pool A 不可替代的价值。

实测任务类型分布（`spar_info.type`，注意 `spar_info` 在 JSON 里是**字符串**，需二次 `json.loads`）：

| type | 条数 | 视角子采样后是否有效 | 处理 |
|---|---|---|---|
| `distance_infer_center_oo_video` | 6,940 | 目标可见即有效 | 用 `point_img_idx` 过滤 |
| `distance_prediction_oo_video` | 6,816 | 同上 | 用 `point_img_idx` 过滤 |
| `spatial_imagination_oc_video` | 5,509 | 同上 | 用 `point_img_idx` 过滤 |
| `spatial_imagination_oo_video` | 5,443 | 同上 | 用 `point_img_idx` 过滤 |
| `obj_count` | 4,889 | **答案可能不可恢复** | 单独分桶（见下） |
| `spatial_imagination_oc_video_hard` | 4,477 | 目标可见即有效 | 用 `point_img_idx` 过滤 |
| `spatial_imagination_oo_video_hard` | 4,454 | 同上 | 用 `point_img_idx` 过滤 |
| `appearance_order` | 2,439 | 仅当所有被问类别仍出现 | 条件保留 |
| `obj_frame_locate` | 2,276 | **问"物体在第几帧"，重编号后语义破坏** | **剔除** |
| `room_size` | 1,363 | 需从局部外推全局 | 保留（高价值） |

`point_img_idx` 字段（记录目标点出现在哪些视角）覆盖 35,915/44,606 条，即 **SPAR 32-view 子集的 80.5% 可用规则过滤**，`bbox_img_idx` 另覆盖 22,159 条。考虑到 SPAR 在 SFT 配比下占 66.4%，这使 §6.3 的一级过滤在最大的数据源上基本可行，无需依赖昂贵的模型过滤。

关于 `obj_count` 与 `room_size`：这两类在 student 视角下答案**确实可能不可完全恢复**，属于 §2 讨论的"真缺失"情形。但它们同时是 VSI-Bench 的核心题型（`object_counting`、`room_size_estimation`），也正是"从局部外推全局"这一能力的直接体现。**建议保留但单独分桶**，在 Phase 2 做"含/不含外推类样本"的消融——如果收益主要来自这一桶，说明模型学到的是外推能力；如果这一桶反而导致幻觉加剧（§9.3 的熵指标会显示），则剔除。

#### Pool B — VLM-3R ScanNet 视频（51,779 条，全量纳入 / 23.9%）

- 与 VSI-Bench 题型和措辞对齐度最高，是提升目标指标的主力。1,201 个 ScanNet 场景，视频已在盘（`data/vlm3r/media/scannet/videos/`，1,201 个 mp4，19 GB）。
- 需要**离线抽帧**：每场景均匀抽 $N=32$ 帧，预缩放后存为 jpg（§6.4）。抽帧结果可与 Pool D 共用（VSI-590K 是同一批 ScanNet 视频的符号链接）。
- 抽帧后按 `Frame-i: <image>` 重建 prompt。原始 prompt 开头是 "These are frames of a video."，需替换为多图格式，且 student/teacher 两侧**逐字一致**。
- 答案极短（中位数 1 词），受 D2 的短 response 问题影响最大。

**注意场景重合**：Pool A 的 ScanNet 部分（1,194 场景）与 Pool B（1,201 场景）来自同一批 ScanNet 场景，只是 QA 不同。因此两池叠加带来的是**题型多样性**而非场景多样性；真正的新场景来自 Pool A 的 ScanNet++（803 个）。

#### Pool D — VSI-590K appearance order（3,819 条）

独占覆盖 `obj_appearance_order`（VSI-Bench 618 条）。复用 Pool B 的抽帧结果，边际成本接近零，**建议全量纳入**。

需要特别注意：appearance order 问的是"各类别在视频中首次出现的先后顺序"，视角子采样**可能改变正确答案**。必须校验"被问及的所有类别在 student 视角子集中仍然出现，且相对顺序不变"，否则剔除。这是全部数据池中对视角子采样最敏感的题型。

#### Pool C — SPAR 3-view（89,466 条，条件可用）

全部为 `*_mv` 类型，主要有 `distance_prediction_oo_mv`(9,371)、`distance_prediction_oc_mv`(9,270)、`depth_prediction_oc_mv`(9,262)、`obj_spatial_relation_oo_mv`(9,067)、`distance_infer_center_oo_mv`(9,044)、`spatial_imagination_map_mv`(6,925) 等；`point_img_idx` 覆盖 63,530/89,466。

- 只允许"保留首视角、丢弃后续视角"的降级方式（$N=3 \to K=1$）。3,000 条抽样中有 1,750 条显式出现 "first image"，说明首视角是坐标系锚点，不可丢。
- 对 `spatial_imagination_map_mv`（BEV 构建，6,925 条）该操作**保持 GT 严格有效**：BEV 坐标以首视角为原点定义，丢掉辅助视角不改变答案，只是让任务从多视角变为单目——这正是一个理想的有限视角任务。
- BEV 答案是多行坐标列表（数十 token），**能显著缓解 D2 的短答案问题**，是 Pool C 中优先级最高的子类。
- 其余 `_mv` 类型需逐类确认"丢弃后两个视角后 GT 是否仍成立"，未确认的先不用。

**必须剔除**：
- SPAR 2-view `view_change_infer`（问题本身定义在两个视角的关系上，$K=1$ 时不可判定）。
- SPAR 1-view（无多视角可降）。
- llava_hound 中可用帧数少于 4 的视频（无法构造有意义的 $N/K$ 差距）。
- 任何 student 视角下**答案不可判定**的样本（见 6.3）。

### 6.3 可判定性过滤（关键步骤，不可省略）

若 student 的 $K$ 个视角里根本看不到被问及的物体，那么"正确答案"对 student 而言是不可及的，蒸馏会退化为强迫模型猜测，制造幻觉而非泛化能力。

**过滤方法（按成本递增）：**

1. **规则过滤**：SPAR 的 `spar_info` 含 `point_img_idx` 字段（记录每个目标点出现在哪个视角）。要求 student 视角集与 `point_img_idx` 的交集非空。这是最廉价且最可靠的过滤器，**应作为默认**。
2. **模型过滤**：用 SFT 基线在 student 视角下跑 $n$ 次采样，若 $n$ 次全错且 teacher 在 $N$ 视角下全对，标记为"高 gap"样本，单独分桶。这类样本既可能是最有价值的（真正的特权信号），也可能是不可判定的噪声，**建议先分桶观察，不要直接混入**。
3. **人工抽检**：每个来源随机抽 50 条，人工确认 student 视角下问题是否合理。

### 6.4 分辨率一致性（对应 4.1 的实现陷阱）

**这是整个方案的硬性约束：唯一的特权轴必须是视角数量，分辨率不得成为第二条特权轴。**

因为 student 图走 `fetch_image`、teacher 图走裸 `Image.open`，两条路径的缩放行为不同，**不处理时分辨率差异是默认存在的**。

但如 §1.1 所述，这不需要改 verl：`smart_resize` 在宽高已是 28 的整数倍、总像素落在 $[4 \times 28^2,\ 16384 \times 28^2]$ 内时原样返回，此时两条路径输出逐像素相同。因此只需在预处理脚本里保证：

- 所有落盘图片的宽高**恒为 28 的倍数**（脚本内断言，不依赖下游兜底）；
- 每视角 token 数统一（v0 为 288），teacher 与 student 引用**同一批文件**，而不是同一张图的两份副本。

第二条容易被忽略：若 teacher 与 student 各自持有一份独立缩放的副本，任何一次脚本改动都可能让两侧不同步。让两侧指向同一路径可以从结构上排除这类漂移。

**验收方式**：在 P0-5 的 smoke test 中打印 teacher 与 student 的 `image_grid_thw`，逐视角断言相等。这一条不通过，后续所有结果都不可归因。

### 6.5 规模与配比

**配比原则：先逐源照抄 SFT 的采样率，再纳入其中全部能构造视角落差的样本。不做人为配比，也不按任何 benchmark 的评分结构反向设计。**

两步顺序不能颠倒——必须**先按 SFT 的 `%60/%60/%60/%50` 抽，再过滤**。两种顺序的期望条数相同（随机采样与确定性过滤可交换），但只有"先抽后滤"才是对 SFT 取数流程的忠实复刻，可以直接陈述"我们与 SFT 从同一个抽样动作出发"。

不按 benchmark 配比的理由：评测是多 benchmark 的（VSI-Bench、CV-Bench、BLINK-Spatial、SPAR-Bench、MMSI-Bench、Video-MME），按其中任何一个的题型分布配数据都是 benchmark 过拟合，会让跨 benchmark 的泛化结论失去意义。

过滤规则只有一条、且完全由方法本身决定：**样本必须能构造出合法的 $N \to K$ 视角落差**。据此得到的训练池：

| 数据源 | 原始 | 采样率 | 抽样后 | 可用 | 占比 | 视角配置 |
|---|---|---|---|---|---|---|
| SPAR 3-view | 89,466 | 60% | 53,775 | **39,623** | 29.4% | $N{=}3$，$K \in \{1,2\}$ |
| `llava_hound_64k` | 63,750 | 60% | 38,250 | **38,250** | 28.4% | $N{=}\min(8,\text{可用帧})$，$K \in \{1,2,4\}$ |
| `vlm3r_scannet` | 51,779 | 60% | 31,067 | **31,067** | 23.0% | $N{=}8$，$K \in \{1,2,4\}$ |
| SPAR 32-view | 44,606 | 60% | 26,671 | **23,991** | 17.8% | $N{=}32$，$K \in \{1,2,4\}$ |
| `vsi_appr_order` | 3,819 | 50% | 1,909 | **1,909** | 1.4% | $N{=}8$，$K \in \{1,2,4\}$ |
| SPAR 1-view / 2-view | 100,205 | 60% | 60,123 | 0 | — | 无视角可降，全部排除 |
| **合计** | 353,625 | | 211,792 | **134,840** | 100% | |

SPAR 的排除分四类：单图（无视角可降）、双图 `view_change_infer`（降到 1 视角问题不成立）、$K_{\min}$ 触及零特权或超出 $K$ 上限、以及答案本身是视角编号的题型（详见 §1.1 的分组排除表）。**三个视频源 100% 存活，抽样后条数与 SFT 逐条相同**（38,250 / 31,067 / 1,909）。

**该配比只有在 `loss_agg_mode=seq-mean-token-mean` 下才成立**，否则实际梯度占比会与上表严重偏离，原因见 §6.5.1。

#### 与 SFT 实际用量的逐源对比

这是"照抄 SFT 设定"这一主张唯一可验证的地方，必须逐源核对（SFT 侧数字取自实际训练日志 `train.rank0.log:949-958`）：

| 数据源 | SFT 实际用 | 本方案 | 差 | 倍数 |
|---|---|---|---|---|
| `llava_hound_64k` | 38,250 | **38,250** | 0 | **1.00×** |
| `vlm3r_scannet` | 31,067 | **31,067** | 0 | **1.00×** |
| `vsi_appr_order` | 1,909 | **1,909** | 0 | **1.00×** |
| `spar_234k` | 140,566 | 63,614 | −76,952 | 0.45× |
| 合计 | 211,792 | 134,840 | −76,952 | 0.64× |

**三个视频源与 SFT 逐条相同，差异全部集中在 SPAR 一源**，且这 76,952 条的每一条都能给出排除理由：

| 排除类别 | 条数 | 占 SFT SPAR |
|---|---|---|
| 1-view 单图，无视角可降 | 45,086 | 32.1% |
| 2-view `view_change_infer`，降到 1 视角问题不成立 | 15,034 | 10.7% |
| 3-view $K_{\min}{=}3$，零特权（等同对照组） | 8,881 | 6.3% |
| 3-view 答案本身是视角编号（`distance_infer_center_oc_mv`） | 5,271 | 3.7% |
| 32-view $K_{\min}{=}5$，超出 $K$ 上限（可通过扩 $K$ 菜单恢复） | 2,680 | 1.9% |
| **合计排除** | **76,952** | **54.7%** |
| **可用** | **63,614** | 45.3% |

前两类合计 42.8% 是**方法的硬约束**（构造不出视角落差），与配比选择无关；后三类合计 11.9% 才是本方案的过滤规则所致，其中最后一类还是可恢复的。这张表应原样进论文，因为"为什么你的 SPAR 只有 SFT 的 45%"是必然被问到的问题。

#### 照抄采样率并不能恢复 SFT 的源级占比

这一点必须写清楚，否则容易被误认为已经解决：四个源乘的系数几乎相同（0.6/0.6/0.6/0.5），**相对占比基本不动**。

| 数据源 | SFT 占比 | 本方案占比 | 变化 |
|---|---|---|---|
| SPAR（合计） | 66.4% | 47.2% | −19.2 |
| `llava_hound_64k` | 18.1% | 28.4% | +10.3 |
| `vlm3r_scannet` | 14.7% | 23.0% | +8.3 |
| `vsi_appr_order` | 0.9% | 1.4% | +0.5 |

19.2 个百分点的偏移**主要来自 SPAR 那 60,123 条单图/双图样本天生不可用**，任何采样率都改不了。因此：

- **与 SFT 基线的差值不可单独归因于训练信号**，其中仍混有 SPAR 占比下降的成分；
- **隔离机制一律以无特权对照（$N{=}K$）为准**——对照组与主实验共用同一份 parquet，那一对比较里数据变量是严格钉死的。

照抄采样率买到的是**程序层面的对齐**：三个视频源条数与 SFT 逐条一致，SPAR 少掉的部分可精确归因于多视角可行性约束，而不是"我们多用或少用了数据"。这堵住了一个廉价的质疑点，代价为零——v0 只用到池的 14.2%。

#### 实现约束：采样必须可复现

SFT 的采样在 `data_qwen.py:230-234` 用无种子的全局 `random.sample`，且以 `print` 而非 `rank0_print` 输出，说明**每个 rank 各自独立抽样**。我们的预处理脚本必须**固定种子、在单进程内一次抽好并落盘为 parquet**，把抽样结果变成数据产物而不是运行时行为。

`待核对`：SFT 那 60% 具体是哪些样本已无法复原。这不影响本方案，但意味着"与 SFT 用同一批样本"这种更强的说法不成立，只能说"用同一组采样率"。

#### 弱特权样本占 29.4%

SPAR 3-view 的特权 gap 只有 $3 \to 1$ 或 $3 \to 2$，远弱于其余各源的 $8 \to K$ 与 $32 \to K$。它占池 29.4%，不会破坏实验，但会稀释平均信号强度。**必须按特权强度分桶记录 loss 与 teacher-student gap**（§9.3），否则无法判断收益来自哪一部分。若 v0 显示弱特权桶几乎无贡献，v1 再考虑降低其权重（§7.3 消融轴 5）。

#### 可选对照：题型分层配比

作为 Phase 2 的**一个明确标注为 benchmark-targeted 的对照臂**，可以按 VSI-Bench 的 8 类等权重新分层抽样（Pool B 中 `object_rel_direction` 占 49.8% 而评分权重仅 12.5%，`room_size_estimation` 占 2.3% 却同样是 12.5%，偏差极大）。

该臂回答的是"多少 headroom 来自数据策划、多少来自训练机制"，是一个正当的科学问题，但**其结果不得用于主结论，也不得与其他 benchmark 的泛化主张混用**。

### 6.5.1 token 质量失衡（严重，必须处理）

verl 的 actor 默认 `loss_agg_mode: token-mean`（`verl_pkg/verl/trainer/config/actor/actor.yaml:190`），其实现是把整个 global batch 的 masked loss 求和后除以**总 token 数**（`core_algos.py:1057-1060`）。这意味着**一条样本的梯度占比正比于它的 response token 数**，而不是样本数。

把 §6.5 的配比与 D2 实测的答案长度相乘：

| 池 | 样本数 | 样本占比 | 均值 token | token 质量 | **梯度占比** |
|---|---|---|---|---|---|
| SPAR 3-view | 39,623 | 29.4% | 69.7 | 2,761,723 | **57.2%** |
| `llava_hound_64k` | 38,250 | 28.4% | 31.2 | 1,193,400 | **24.7%** |
| SPAR 32-view | 23,991 | 17.8% | 34.5 | 827,690 | **17.1%** |
| `vlm3r_scannet` | 31,067 | 23.0% | 1.4 | 43,494 | **0.9%** |
| `vsi_appr_order` | 1,909 | 1.4% | 1.0 | 1,909 | **0.0%** |

**失衡与池的规模无关，等比缩放不改变任何占比**：29.4% 的 SPAR 3-view 样本拿走 **57.2%** 的梯度，而与 VSI-Bench 措辞完全对齐、样本量排第三的 `vlm3r_scannet` 只有 **0.9%**，`vsi_appr_order` 实际为 0。若不处理，这个实验优化的是 SPAR 的 BEV 坐标回归，与空间推理主张无关。

需要说明的是，**这不是配比选择引入的问题**——SpatialStack SFT 自身也存在同样的失衡（它同样用 token 级平均）。只是在 SFT 里这被当作既定事实接受了，而在这里必须显式处理，否则消融结论全部失真。

**解决方案（二选一或并用）：**

1. **改聚合模式：`actor.loss_agg_mode=seq-mean-token-mean`**（`core_algos.py:1067-1073`）。该模式先对每条序列做 token 平均，再对序列求平均，**使每条样本等权，与其长度无关**。这样 §6.5 的源级占比在梯度层面才真正等于 SFT 的源级占比。**推荐作为默认。**
2. **按 token 质量反向配比**：保持 `token-mean`，但按目标梯度占比反推样本数。这条路等于要么大幅重复采样短答案源、要么砍掉大半 SPAR 3-view，**会破坏 §6.5 "照抄 SFT 采样率"的前提，排除。**

因此只能选方案 1。副作用需注意：`seq-mean-token-mean` 下，一条 1-token 的样本，其单个 token 的 JSD 就是整条序列的损失，**方差高于长序列样本**。这是可接受的代价，但应在训练中监控 `vlm3r_scannet` 子集的 loss 方差。

无论选哪种，**都必须按池分别记录 loss 与梯度占比**，作为 §9.3 的诊断指标之一。这一项在 P0-5 的 smoke test 中就应该打通。

### 6.5.2 是否纳入 `llava_hound_64k`（Pool E）

`llava_hound_64k` 是 SFT 四个数据集中唯一未被本方案默认采用的。重新评估后的结论是**小比例纳入，不全量**。

支持纳入：

- 它占 SFT 数据的 18%，OPSD 属于后训练更新，完全撤掉有通用视频理解能力退化的风险（对应 `literature/OPEN_QUESTIONS.md` Q-003）。
- 视频形态，帧子采样机制同样适用（实测每视频帧目录中位数 20 帧、均值 29 帧、最少 9 帧）。

反对全量纳入：

- 特权语义不同：llava_hound 是动态 2D 视频，减帧减少的是**时间覆盖**，不是静态 3D 场景的**视角覆盖**。混入会削弱"多视角特权"这一主张的干净程度。
- 梯度占比偏高：全量纳入后它占样本 29.4%、占 `token-mean` 梯度 25.7%，是第二大来源。这一条已由 `seq-mean-token-mean` 处理（占比回落到样本占比 29.4%）。
- 每视频可用帧数中位数 20、最少 9，无法统一取 $N=8$，需按视频逐条设定 $N=\min(8, \text{可用帧数})$。

**结论：全量纳入 63,750 条**，与 §6.5"纳入全部可用数据"的原则一致；在 v1 增加"含/不含"的消融（消融轴 5），把它对主张的影响量化，而不是靠论证回避。它同时是 Video-MME 这个不退化护栏的直接对应项。

**训练量不等于数据量。** 训练池 134,840 条在 `total_batch=64` 下是 2,107 步、约 2.7 天/条实验。跑满虽已不算不可接受，但两条仍要 5–6 天才拿到第一个信号。因此 **v0 采用固定步数预算而非跑满 1 epoch**：300 步（19,200 样本，占池 14.2%，约 9.1 小时/条）。数据分布与全池一致（parquet 已打乱），只是不遍历完；若信号明确，直接从 checkpoint 续训到 1 epoch 即可，剩余 85% 仍未被看过，无需重建数据。Vision-OPD 用 6,241 条 × 1 epoch 就能产生效果，300 步是合理起点。

**留出集**：从每个池额外抽 500 条不参与训练，用于监控 teacher/student gap 的训练动态（§9.3）。留出集必须**按场景**而非按样本切分，否则同场景不同 QA 会造成泄露。

### 6.6 数据泄露检查（已完成，通过）

用 `logs/eval/.../\*samples_vsibench.jsonl` 中记录的 `doc.dataset` 与 `doc.scene_name` 还原 VSI-Bench 测试场景集合（scannet 88 个、scannetpp 50 个、arkitscenes 150 个），与各训练池的场景集合求交：

| 训练池 | 场景数 | 与 VSI-Bench 交集 |
|---|---|---|
| SPAR ScanNet（全部） | 1,204 | **1**（`scene0685_00`） |
| SPAR ScanNet++（全部） | 844 | **0** |
| SPAR structured3d | 3,071 | 0（VSI-Bench 不含该来源） |
| VLM-3R `merged_qa_scannet_train` | 1,201 | **0** |
| VSI-590K appearance order | 166 | **0** |

**结论：仅需在数据准备时排除 `scene0685_00`。** 该结果应写入 Phase 0 的 registry。注意 VSI-Bench 的 150 个 ARKitScenes 场景在所有训练池中都不存在，不构成泄露但构成域差。

---

## 7. 分阶段实验计划

### 7.1 Phase 0：可行性验证（不训练，约 1–2 天）

**目的：在投入训练算力前，排除会让整个方案失效的前提性风险。** 任何一项不通过都必须先解决再进入 Phase 1。

| ID | 任务 | 通过标准 | 对应风险 |
|---|---|---|---|
| P0-1 | 数据泄露核对：训练 scene_id ∩ VSI-Bench scene_id | 交集为空 | **已完成，通过**（仅需排除 `scene0685_00`，见 §6.6） |
| P0-2 | **特权 gap 测量**：SFT 基线在 $K \in \{1,2,4,8,16\}$ 与 $N=32$ 视角下，在留出集上的准确率与 GT 的 NLL，**分题型报告** | teacher($N$) 相对 student($K$) 有**显著且稳定**的 gap（建议 ≥5 个点或 NLL 差 ≥0.15） | **若 gap 不显著，方案在此终止**——没有特权就没有可蒸馏的信号 |
| P0-3 | Response 长度实测：基线在各池上 rollout，统计 response token 数分布 | 中位数 ≥8 token；否则触发 D2 的选项 B/C | 监督量不足 |
| P0-4 | 视觉 token 实测：分别测训练端与 lmms-eval 评测端的每帧真实 token 数 | 两端一致；teacher 总长 < `max_reprompt_len` | 训练/评测预算不一致 |
| P0-5 | 端到端 smoke test：50 条数据跑通 verl vopd 流程 | 无异常，`self_distillation/teacher_always_on_fraction=1.0` | 工程阻塞 |
| P0-6 | 显存与吞吐：测单步耗时与峰值显存 | 单步 < 5 分钟 | 算力预算 |
| P0-7 | 基线帧预算曲线：`novggt_aligned` 在 VSI-Bench @ {1,2,4,8,16,32} 帧 + CV-Bench | 得到完整参考曲线 | 无对照 |

**P0-2 是整个计划的 go/no-go 关卡。** 它直接对应 `Vision_OPD_OPSD_Analysis.md` §9.5 指出的风险（teacher 优势仅来自条件而非参数规模，条件若不够强则信号偏弱），在多视角设定下这个风险更高，因为 4 视角与 16 视角的差距未必有 crop 与全图的差距那么大。

**先跑 P0-7 再跑 P0-2**：P0-7（基线的帧预算曲线）本身就是特权 gap 在评测集上的直接测量，只需改 `max_num_frames` 反复调用现成评测脚本，成本远低于 P0-2，却能提前给出 go/no-go 的强信号。若基线在 4 帧与 32 帧之间的差距不到 3 个点，就应立刻重新考虑特权轴的选择，不必再做 P0-2。

P0-2/P0-7 同时产出本项目的一个独立可发表结论：**空间推理任务上，视角数量对 VLM 性能的边际收益曲线，以及该曲线的分题型差异**。

### 7.2 Phase 1：主实验（约 3–5 天）

统一配置：图像源 $N=32$、视频源 $N=8$、$K \sim \text{LogUniform}\{1,2,4,8,16\}$（不超过对应的 $N$）、每视角 token 数由 P0-8 判定（192 或 300）。

| 实验 ID | 设置 | 作用 |
|---|---|---|
| `20260815_qwen35_mvopsd_main` | teacher 32 视角 / student 随机预算，reward-free JSD | **主方法** |
| `20260815_qwen35_mvopsd_noprivilege` | **teacher 视角数与 student 相同**（$N=K$，逐样本相等），其余完全相同 | **最关键消融**：隔离"多视角特权"与"自蒸馏本身" |
| `20260815_qwen35_sft_viewdrop` | 同一批数据，纯 SFT，同样的随机预算 + GT 答案 | 隔离"只是在少视角数据上又训了一遍" |
| `20260815_qwen35_grpo_viewdrop` | 同一批数据，GRPO + 可验证 reward，同样的随机预算 | 隔离"只是做了 RL" |

四组共用完全相同的数据（含相同的 $K$ 抽样结果与随机种子），唯一差异是训练信号。这一点必须在数据准备阶段固化：**先物化一份带 $K$ 的数据集，四个实验共享同一个 Parquet**，`noprivilege` 组只替换 `teacher_images` 与 `teacher_prompt` 两列。

`mvopsd_noprivilege` 是四个里最重要的。它回答审稿人一定会问的问题：收益究竟来自多视角特权，还是来自 on-policy 自蒸馏这个训练范式本身？文献中同类工作正因缺少这一行对照而受质疑（见 `literature/OPEN_QUESTIONS.md` Q-014、Q-023）。

### 7.3 Phase 2：消融（约 5–7 天）

按优先级排序，算力不足时从上往下砍：

| 优先级 | 消融轴 | 取值 | 回答的问题 |
|---|---|---|---|
| 1 | **视角预算：随机 vs 固定** | 随机 $K$ / 固定 $K{=}4$ / 固定 $K{=}8$ | 随机预算是否真的买到了跨预算泛化？固定 $K$ 是否只在该点出现尖峰？（**直接对应 D4**） |
| 2 | 特权强度 $N$ | $N{=}K$（无特权）, 8, 16, 32 | 收益是否随特权 gap 单调增长？若不单调，说明存在最优 privilege gap |
| 3 | 视角采样策略 | 均匀 / 随机 / 连续 $K$ 帧 | 部署相关性；连续帧最贴近真实机器人，也最难 |
| 4 | 散度形式 $\alpha$ | 0(forward KL), 0.5(JSD), 1(reverse KL) | 面对 student 不可及的信息，mode-covering 还是 mode-seeking 更合适？**这在信息缺失场景下的答案可能与 Vision-OPD 不同** |
| 5 | 数据配比（**默认 = SFT 等比**） | SFT 等比 / 不含 `llava_hound` / SPAR 内部倾向 32-view / 题型分层 | 见下方说明 |
| 6 | 损失聚合模式 | `seq-mean-token-mean` / `token-mean` | 各池等权 vs 按 token 加权，对最终指标影响多大（§6.5.1） |
| 7 | 外推类样本 | 含/不含 `obj_count`+`room_size` | 真缺失样本是带来外推能力还是幻觉（§6.2） |
| 8 | 正确性门控 | 无 / 二值 / 连续（D5） | 是否需要过滤 teacher 的错误 |
| 9 | EMA 系数 $\beta$ | 0.01, 0.05, 0.2 | 目标漂移与跟随速度权衡 |
| 10 | 视角预算课程 | 随机 / 从大到小退火 | 课程学习是否缓解早期 gap 过大 |
| 11 | top-$k$ | 50, 100, 200 | 计算/信号权衡 |

第 5 项的四个取值性质不同，不能混为一谈：

- **SFT 等比**是主配置，也是所有主结论的唯一来源（§6.5）。
- **不含 `llava_hound`** 量化防遗忘 replay 的贡献，对应 Video-MME 护栏（§6.5.2）。
- **SPAR 内部倾向 32-view** 降低 3-view 权重，检验弱特权样本（$3\to1$ 或 $3\to2$，占 29.4%）是否真的贡献有限。这是机制探索，若结论为正可以进入正文讨论，但**不能替换主配置**，否则"照抄 SFT 采样率后纳入全部可用数据"这条可陈述的规则就失效了。
- **题型分层**是明确标注的 benchmark-targeted 对照，**其结果不得用于任何跨 benchmark 泛化主张**。

第 4 项值得特别关注。Vision-OPD 场景下信息是"可恢复的"，student 原则上能匹配 teacher；本方案中信息是"真缺失的"，强行 mode-seeking（reverse KL）可能诱发过度自信与幻觉，而 mode-covering（forward KL）可能更合适——**这是本方案区别于所有参考文献的一个原生科学问题**，应写入 `literature/OPEN_QUESTIONS.md`。

### 7.4 Phase 3：扩展（可选）

- 扩大数据到 30–50k，多 epoch。
- 引入 VGGT 几何编码器（D1 的后续）。
- 与 SpatialStack 的 vggt checkpoint 组合，冲最好数字。
- 跨 benchmark 泛化：BLINK-Spatial、SPAR-Bench、MMSI-Bench（评测脚本已支持）。

---

## 8. 训练配置

### 8.1 新增 Hydra 配置 `verl_pkg/verl/trainer/config/mvopsd.yaml`

以现有 `vopd.yaml` 为模板，关键改动是启用 `teacher_always_on` 与 `teacher_image_key`（现有 `vopd.yaml` 两者都没设，`opsd.yaml` 设的是 answer-hint 模式，都不是我们要的路径）：

```yaml
defaults:
  - ppo_trainer
  - user
  - _self_

max_model_len: 20480          # teacher: 16 views x 256 tok + text + response

actor_rollout_ref:
  model:
    path: ./output/spatialstack_qwen35_novggt_aligned
  actor:
    ppo_mini_batch_size: 32
    policy_loss:
      loss_mode: vopd
    self_distillation:
      teacher_always_on: True
      teacher_image_key: teacher_images
      teacher_prompt_mode: null            # 必须为 null，否则走 answer-hint
      teacher_model_source: legacy
      teacher_regularization: ema
      teacher_update_rate: 0.05
      full_logit_distillation: True
      distillation_topk: 100
      distillation_add_tail: True
      alpha: 0.5
      is_clip: 2.0
      max_reprompt_len: 12288
      fallback_to_policy_loss_on_missing_teacher: False
    loss_agg_mode: seq-mean-token-mean   # 关键：使各池按样本数等权，见 §6.5.1
    optim:
      lr: 2.0e-6
  rollout:
    n: 4
    calculate_log_probs: True
    limit_images: 40                       # 需 >= max(N, K)
    gpu_memory_utilization: 0.6            # teacher forward 需要预留

algorithm:
  adv_estimator: grpo                      # 仅用于跳过 critic，主路径不走 advantage
  rollout_correction:
    rollout_is: token
    rollout_is_threshold: 2.0

data:
  train_batch_size: 32
  image_key: images
  max_prompt_length: 4096
  max_response_length: 512

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  total_epochs: 1
  save_freq: 50
  test_freq: -1
  val_before_train: False
```

几个必须注意的点：

- `teacher_prompt_mode` 必须是 `null`。设为 `answer_hint` 会走文本答案提示分支，那是完全不同的方法（`ray_trainer.py:1168-1277`）。
- `fallback_to_policy_loss_on_missing_teacher: False` 让缺失 teacher 图**硬失败**而非静默降级。数据管线出错时我们要立刻知道，而不是训练出一个说不清在优化什么的模型。
- `rollout.n` 设 4 而非 Vision-OPD 的 8。主路径跳过 GRPO advantage，`n` 只增加 response 覆盖，多视角输入下生成成本远高于单图，8 不划算（`Vision_OPD_OPSD_Analysis.md` §9.6 也指出了这点）。但 GRPO 对照组（`grpo_4view`）需要 `n≥4` 才能算组内 advantage，保持 4 可让两组共用配置。
- `limit_images` 必须 ≥ $\max(N,K)$，否则 vLLM 会拒绝请求（`vllm_async_server.py:254-255`）。

### 8.2 需要新写的脚本

| 脚本 | 作用 | 预估工作量 |
|---|---|---|
| `scripts/opsd/extract_frames.py` | ScanNet 视频 → 均匀抽帧 + 预缩放落盘 | 半天 |
| `scripts/opsd/prepare_mvopsd_data.py` | 三池筛选 + 视角子采样 + 可判定性过滤 + 写 Parquet | 1–2 天 |
| `scripts/opsd/run_mvopsd.sh` | verl 启动封装（对标缺失的 `run_vision_opd.sh`） | 半天 |
| `scripts/opsd/merge_ckpt.py` | verl FSDP checkpoint → HF 格式（封装 `verl/model_merger`） | 半天 |
| `scripts/opsd/eval_frame_budget.sh` | 遍历 `max_num_frames` ∈ {1,2,4,8,16,32} 调 lmms-eval | 半天 |
| `scripts/opsd/measure_privilege_gap.py` | Phase 0 的 P0-2 特权 gap 测量 | 1 天 |

**verl 核心代码 v0 预期改动量为零。** 若 Phase 0 发现必须改，最可能的三处是：
1. `_normalize_teacher_image` 增加 `max_pixels` 支持（若不采用离线预缩放）；
2. teacher 侧 `videos=None` 放开（若要用原生 video token 而非多图）；
3. 动态 micro-batch 分桶改为按 $\max$(student, teacher) 长度（若 teacher 侧 OOM）。

### 8.3 评测流程

```bash
# 1) verl checkpoint -> HF
python scripts/opsd/merge_ckpt.py \
  --local_dir checkpoints/<experiment_id>/global_step_N/actor \
  --target_dir output/<experiment_id>_hf

# 2) 帧预算曲线（仅视频类 benchmark，max_num_frames 生效）
for NF in 1 2 4 8 16 32; do
  MODEL_PATH=output/<experiment_id>_hf \
  MODEL_IMPL=qwen3_5 \
  MODEL_ARGS_BASE="pretrained=output/<experiment_id>_hf,use_flash_attention_2=true,max_num_frames=${NF},max_length=12800,disable_thinking=true" \
  OUTPUT_ROOT=logs/eval/<experiment_id>/frames_${NF} \
  BENCHMARKS="vsibench" \
  bash scripts/evaluation/eval.sh
done

# 3) 固定视觉输入的 benchmark（无帧预算维度）
#    videomme 虽是视频，但只作 32 帧下的不退化护栏，不跑曲线
MODEL_PATH=output/<experiment_id>_hf MODEL_IMPL=qwen3_5 \
MODEL_ARGS_BASE="pretrained=output/<experiment_id>_hf,use_flash_attention_2=true,max_num_frames=32,max_length=12800,disable_thinking=true" \
OUTPUT_ROOT=logs/eval/<experiment_id>/generalization \
BENCHMARKS="cvbench blink_spatial sparbench mmsibench videomme" \
bash scripts/evaluation/eval.sh
```

`blink_spatial` 是 `src/lmms_eval/tasks/blink/blink_spatial.yaml` 定义的分组，其中 `blink_multi_view_reasoning` 子项与本方案机制最直接相关，应单独摘出报告。

注意评测前需取消 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`（`experiments/LESSONS.md` LESSON-003）。

---

## 9. 结果报告与诊断指标

### 9.1 主结果表

**表 1：帧预算曲线（主指标）**

| 方法 | VSI@1 | VSI@2 | VSI@4 | VSI@8 | VSI@16 | VSI@32 | 低预算均值<br>(1/2/4) |
|---|---|---|---|---|---|---|---|
| SFT 基线（`novggt_aligned`） | | | | | | 64.24 | |
| SFT + view-drop | | | | | | | |
| GRPO + view-drop | | | | | | | |
| MV-OPSD 无特权（$N=K$） | | | | | | | |
| MV-OPSD 固定 $K{=}4$ | | | | | | | |
| **MV-OPSD 随机预算（主）** | | | | | | | |

**表 2：跨 benchmark 泛化与不退化**

| 方法 | CV-Bench<br>(单图) | BLINK-Spatial<br>(多视角推理) | SPAR-Bench<br>(同域) | MMSI-Bench<br>(跨域多图) | Video-MME<br>(通用护栏) |
|---|---|---|---|---|---|
| SFT 基线（`novggt_aligned`） | 84.97 | | | | |
| MV-OPSD 无特权（$N=K$） | | | | | |
| **MV-OPSD 随机预算（主）** | | | | | |

表 2 是本方案不做 benchmark 导向配比的直接回报：因为训练数据分布未针对任何 benchmark 调整，这五列的结果可以作为真实泛化证据，而不是数据策划的产物。

**成功判据**（预注册，避免事后挑指标）：
1. MV-OPSD 在**低预算均值（1/2/4 帧）**上相对 SFT 基线 **≥ +2.0 点**；
2. MV-OPSD 在低预算均值上相对 `无特权` 对照 **≥ +1.0 点**（这一条才是"多视角特权有效"的证据）；
3. VSI@32 相对基线**下降不超过 1.0 点**；
4. CV-Bench 与 Video-MME 相对基线**下降均不超过 1.0 点**；
5. 随机预算相对 `固定 K=4` 在**非 4 帧的预算点**上整体更优（这一条验证 D4 的设计选择）；
6. BLINK-Spatial / SPAR-Bench / MMSI-Bench **至少两项不低于基线**（不要求提升，只要求不以牺牲跨域能力换取 VSI 曲线）。

第 2 条不成立时，即使第 1 条成立，也只能声称"on-policy 自蒸馏有效"，不能声称"多视角特权有效"。这个区分必须在结论中如实保留。

### 9.2 VSI-Bench 分题型报告

VSI-Bench 的总分是 8 个题型的**非加权平均**（`src/lmms_eval/tasks/vsibench/utils.py:136-191`），不同题型对视角数的敏感度差异很大。必须分项报告：

- 对视角数**高敏感**（预期收益最大）：`object_counting`、`obj_appearance_order`、`route_planning`、`room_size_estimation`——都需要全局场景覆盖。
- 对视角数**低敏感**：`object_size_estimation`、`object_abs_distance`——单视角加尺度先验即可估计。

如果收益集中在高敏感题型，就构成了"确实学到了从局部外推全局"的机制性证据，而不只是分数变高。这比总分更有说服力。

### 9.3 训练期诊断指标

这些指标用于在训练中判断机制是否按预期工作，而不是训练完才发现问题：

| 指标 | 来源 | 预期行为 |
|---|---|---|
| `self_distillation/teacher_always_on_fraction` | verl 已有 | 恒为 1.0，否则数据有问题 |
| **分池 loss 与梯度占比** | 自测 | 与 §6.5 的样本占比一致（29.4/28.4/23.0/17.8/1.4）；偏离即说明聚合模式失效（§6.5.1） |
| **按特权强度分桶的 loss 与 teacher-student gap** | 自测 | 强特权桶（$N\!\ge\!8$，70.6% 样本）与弱特权桶（SPAR 3-view，29.4% 样本）分开记录。若弱特权桶的 gap 从一开始就接近 0，说明这 29.4% 的数据几乎不产生蒸馏信号，需在 v1 消融轴 5 中处理 |
| **按 $K$ 分桶的 loss 与 gap** | 自测 | $K{=}1/2/4$ 三桶分开记录。gap 应随 $K$ 增大而单调减小；若 $K{=}1$ 桶的 loss 长期不降，说明该预算下信息缺口无法弥合，对应 §2 的"真缺失"假设 |
| **按视角选择方式分桶** | 自测 | oracle 引导桶（SPAR 有标记题型，41.3%）与纯随机桶（视频源 + SPAR 无标记题型，58.7%）分开记录。**只有纯随机桶与评测同分布**，跨 benchmark 泛化主张应主要以该桶为依据（§1.1） |
| JSD raw / weighted token mean | verl 已有 | 单调下降但不塌到 0 |
| 留出集上 teacher-student NLL gap | 自测 | **逐步缩小**——这是方法生效的直接证据 |
| student 在 $K$ 视角下的答案熵 | 自测 | 不应显著低于基线（下降过快 = 过度自信，对应 §7.3 第 4 项的风险） |
| response token 长度分布 | 自测 | 不应塌缩 |
| 视角敏感度：student 在 $K$ vs $N$ 视角下的输出差异 | 自测 | 应缩小，但**不应归零**（归零意味着模型开始忽略视觉输入） |

最后一项是重要的反向护栏。如果 student 学会了完全无视视觉输入、只靠语言先验作答，它在所有帧预算下的表现都会趋同——曲线变平——看起来像"低帧预算下变强了"，实际是能力退化。**必须同时检查曲线右端没有下降。**

---

## 10. 风险登记

| # | 风险 | 影响 | 缓解 | 状态 |
|---|---|---|---|---|
| R1 | 特权 gap 不足（16 视角未必显著强于 4 视角） | 方案失效 | P0-2 前置 go/no-go 关卡；必要时加大 $N/K$ 比或改用连续帧采样制造更大 gap | 待验证 |
| R2 | Response 过短，监督 token 不足 | 信号弱 | P0-3 实测；提高长答案样本占比；必要时启用 thinking | 待验证 |
| R3 | 训练/评测视觉 token 预算不一致 | 结果不可归因 | P0-4 实测两端并强制对齐 | 待验证 |
| R4 | student 视角下答案不可判定 → 学会幻觉 | 指标虚高但能力退化 | §6.3 三级过滤；§9.3 视角敏感度护栏 | 已设计 |
| R5 | teacher 在 32 视角下本身答错 | 传递错误 | P0-2 测 teacher 准确率；Phase 2 加正确性门控 | 已设计 |
| R6 | teacher forward（$N$ 视角）显存溢出 | 工程阻塞 | 降 $N$ / 降每视角 token / 改动态分桶依据 | 已设计 |
| R7 | 数据泄露（ScanNet train 与 VSI-Bench 场景重叠） | 结果无效 | P0-1 阻塞项 | 待验证 |
| R8 | 代码版本不可追溯 | 无法复现 | `experiments/LESSONS.md` LESSON-001 未解决；建议本方案启动前先 `git init` | **建议立即处理** |
| R9 | 只剩一个 SFT checkpoint，误删则无基线 | 全部对照失效 | 立即备份 `output/spatialstack_qwen35_novggt_aligned` | **建议立即处理** |
| R10 | 视角重编号错误导致索引语义泄露 | 静默污染 | 数据准备脚本加单元测试，人工抽检 50 条 | 已设计 |

R8 和 R9 与实验内容无关但优先级最高：当前项目无 Git 元数据（`experiments/LESSONS.md` LESSON-001），且四个历史 checkpoint 已有三个被删除。在启动一个跨越数周、包含十余次训练的计划之前，这两项应当先解决。

---

## 11. 算力与时间预算

单节点 8× H20（96 GB）为主路径。

| 阶段 | 训练轮次 | 预估 GPU·天 | 日历时间 |
|---|---|---|---|
| Phase 0 | 0（仅推理与实测） | 2–3 | 1–2 天 |
| Phase 1 | 4 | 8–16 | 3–5 天 |
| Phase 2 | 8–12 | 20–30 | 5–7 天 |
| 评测（贯穿） | — | 5–8 | 与训练并行 |
| 合计 | | **35–57 GPU·天** | **2–3 周** |

单步耗时由 **Vision-OPD 在本机 8 卡的实测锚点**外推：它 6,241 条 / `batch=96` / 1 epoch = **65 步，耗时 8 小时**，即 7.4 分钟/步。实测其每步视觉 token 为 2.63 M（student 全图中位 3,220 tok × 96 × 8，teacher crop 仅 198 tok），得端到端吞吐约 **355 K visual token/分钟**。

本方案的成本结构与之**完全反转**：

| | Vision-OPD | 本方案 |
|---|---|---|
| student 视觉 token | 3,220（单张全图） | 454（$K \in \{1,2,4\}$ × 192，按池加权） |
| teacher 视觉 token | 198（bbox crop） | 2,074（SPAR 32-view 6,144 / SPAR 3-view 576 / 视频 1,536 加权） |
| 每步序列数 | 768（96×8） | 256（64×4） |
| 每步视觉 token | 2,625 K（**94% 在 student**） | 647 K（**82% 在 teacher**） |
| 推算单步 | 7.4 分钟（实测） | **≈ 1.8 分钟** |

据此：**v0 的 300 步 ≈ 9.1 小时/条，两条不到一天**；跑满 1 epoch（134,840 条 = 2,107 步）约 64 小时 = **2.7 天/条**。

注意 teacher 均值 2,074 远低于 SPAR 32-view 单条的 6,144，因为占池 29.4% 的 SPAR 3-view 只有 $N{=}3$（576 token）。**按"SPAR 一律 $N{=}32$"估算会把成本高估约 60%**。

两点后续动作：

- **提速的杠杆在 $N$，不在 $K$ 或 `rollout.n`。** 82% 的开销在 teacher forward，把 SPAR 32-view 的 $N$ 降到 16 可让单步降至约 1.2 分钟，但同时削弱特权 gap，属于拿实验强度换时间，v0 不做。
- `待核对`：同一 prompt 的 4 条 rollout，teacher 侧视觉前缀完全相同，只有拼接的 response 不同。若 verl 的 teacher 分支只是对整个 batch 做朴素 `forward()`，6,144 个视觉 token 的前缀被重复算了 4 遍，在占 82% 的那一项上有接近 4× 的空间。P0-6 实测单步耗时时一并核查——若实测明显低于 1.8 分钟，说明已有复用。

---

## 12. 实验注册规范

按 `.cursor/rules/experiment-workflow.mdc` 与 `experiments/README.md`：

- 命名：`20260815_qwen35_mvopsd_<variant>`，例如 `_main`、`_noprivilege`、`_n32k4`、`_alpha0`。v0 的两条用 `_v0_main` 与 `_v0_noprivilege`，与后续 v1 的正式实验区分开，避免 v0 的探索性数字被误引。
- 每个设置一个 `experiments/registry/<experiment_id>.yaml`，训练与评测写在同一文件。
- 输出路径统一使用 `experiment_id`：`checkpoints/<id>/`、`logs/train/<id>/`、`logs/eval/<id>/`。
- 启动前在 `experiments/ACTIVE.md` 登记负责人、GPU、命令、作业 ID；当前 `20260813_qwen35_geo_baseline` 仍标记为"评测待运行"，需先澄清其状态再占用 GPU。
- Phase 0 的每项测量也建一个 `20260815_qwen35_mvopsd_phase0` registry，把 gap 测量结果写进 `results.metrics`——这些数字本身就是论文的一张图。

同时应向 `literature/OPEN_QUESTIONS.md` 新增两条本方案原生的问题：

- **Q-028**：当特权信息对 student 在原理上不可获得（真缺失，而非可恢复）时，OPSD 应采用 mode-covering 还是 mode-seeking 散度？（对应 §7.3 第 4 项）
- **Q-029**：空间推理中，视角数量带来的边际收益曲线形状如何，是否存在使蒸馏收益最大的最优 privilege gap？（对应 P0-2 与 §7.3 第 1 项）
- **Q-030**：当训练集中有相当比例的样本只具备弱特权（本方案中 29.4% 为 SPAR 3-view），这些样本是提供了有效的低强度信号，还是纯粹稀释梯度？（对应 §6.5 与 §7.3 消融轴 5）
- **Q-031**：SPAR 有标记题型上 student 必然能看到标记帧（否则问题失去指代对象），视频源与 SPAR 无标记题型则是纯均匀随机采样。这两种视角选择方式训练出的行为是否一致？若 oracle 引导桶收益显著高于纯随机桶，说明收益部分来自 oracle 视角提示而非有限视角泛化能力，主张需相应收窄（对应 §1.1 与 §9.3）。

---

## 13. 立即可执行的下一步

按 §1.1 的分阶段推进，下面只列 **v0** 所需的动作。§5–§9 的决策确认推迟到 v1。

**先做（与 GPU 无关，半天内可完成）**

1. 备份 `output/spatialstack_qwen35_novggt_aligned`（R9）。
2. `git init` 并首次提交，恢复版本可追溯性（R8 / LESSON-001）。
3. ~~实测 SPAR 32-view 是否被截断~~ **已完成，结论为否**（§1.1）：`scripts/opsd/check_spar_truncation.py` 测得 32-view 样本总长 6,506 token，答案 100% 保留。同时 `scripts/opsd/probe_sft_image_geometry.py` 测得 SFT 真实几何为 384×512、192 token/视角，对齐因子 32。

**v0 数据准备（两条脚本，零 verl 改动）**

4. **预处理脚本**：**先按 SFT 采样率 `%60/%60/%60/%50` 固定种子抽样（单进程，落盘留存抽中 id）**，再把抽中样本引用的图统一落盘为 **384×512（192 token/视角，复刻 SFT 几何）**。按唯一文件去重后约 44.4 万张（§1.1），缓存 15–20 GB。ScanNet 的 1,201 个 mp4 抽帧时直接 `-vf scale` 到位，**抽帧规则照抄 SFT 的 `np.linspace` 均匀采样**。**脚本必须断言输出宽高恒为 32 的倍数**，否则分辨率差异会静默复活。
5. **组装 parquet**：按 §6.5 得到 134,840 条，排除 `scene0685_00`（P0-1 已完成，§6.6）。逐样本抽 $K$：视频源与 **SPAR 无标记题型（`obj_count`/`appearance_order`/`room_size`，按题型白名单显式恢复）** 在 $\{1,2,4\}$ 中均匀抽取、视角均匀随机；SPAR 有标记题型在 $\{1,2,4\} \cap [K_{\min}, N)$ 中抽取，视角 = 全部标记帧 + 随机填充（§1.1）。SPAR 3-view 的 BEV 类样本强制保留首视角（§6.3），与标记帧取并集。**断言三个视频源的条数恰为 38,250 / 31,067 / 1,909**，与 SFT 逐条一致。
6. 写 `teacher_prompt` 列（SPAR 32 个 `<image>` 占位符，视频源 8 个）与 student `prompt` 列（占位符数 = 该样本的 $K$，逐样本不同），视角重编号。**SPAR 必须同步重写问题文本中的 `Frame-N` 引用**，这一步写单元测试，并每源人工抽检 10 条。
7. verl 数据配置设 **`data.image_patch_size=16`**，否则 student 侧尺寸会被按因子 28 改动（§1.1）。

**v0 训练与判读**

8. 跑 P0-5 smoke test，打印 teacher 与 student 的 `image_grid_thw` 逐视角断言相等，期望值 `(1, 24, 32)`（§6.4）。这一条不通过，后续结果全部不可归因。
9. 设 `actor.loss_agg_mode=seq-mean-token-mean`，**固定 300 步预算**（19,200 样本，占池 14.2%），跑两条：主实验（SPAR $N{=}32$、视频源 $N{=}8$，$K$ 逐样本随机）与无特权对照（$N{=}K$ 逐样本相等）。
10. **判据只有一条：主实验在 VSI-Bench @ {1, 2, 4} 帧的均值上高于无特权对照。** 之所以不用 SFT 基线作判据，是因为两者数据不同（SPAR 少 54.7%，见 §6.5），差值无法单独归因；而无特权对照与主实验共用同一份 parquet，是唯一干净的比较。

    **但 SFT 基线的三条曲线仍须同表报告**（P0-7 已产出，第 11 步）：主实验、无特权对照、SFT 基线。三者的相对位置区分了三种结局——主实验 > 对照 > SFT 说明特权与自蒸馏都有效；主实验 ≈ 对照 > SFT 说明只有自蒸馏有效，特权无贡献；两者均 < SFT 则说明数据缩减的损失盖过了训练信号，此时须先补"SFT 在同一份 134,840 条数据上重训"这一臂才能继续推进。
11. 同时记录 §9.3 的三组分桶（分池 / 分 $K$ / oracle 引导 vs 纯随机），它们不参与判据，但决定 v1 往哪个方向改。

**可并行（只调现成评测脚本，几 GPU·时）**

12. 执行 **P0-7**：SFT 基线的帧预算曲线。它与 v0 数据准备互不阻塞，且无论 v0 结果如何都要用到。有余力时把六个 benchmark 的基线数字一并跑齐，填进 §9.1 表 2。
13. 执行 **P0-8**（阻塞第 4 步的目标尺寸选择，但只需评测、零训练）：SFT 基线在 `max_pixels=1605632`（评测默认，300 token/帧）与 `max_pixels=196608`（复刻训练几何，192 token/帧）下各测一次 VSI-Bench @ {4, 16} 帧。差距 < 0.5 分则取 192 token 训练；否则取 300 token（§1.1）。

```bash
for MP in 1605632 196608; do
  for NF in 4 16; do
    MODEL_PATH=output/spatialstack_qwen35_novggt_aligned MODEL_IMPL=qwen3_5 \
    MODEL_ARGS_BASE="pretrained=output/spatialstack_qwen35_novggt_aligned,use_flash_attention_2=true,max_num_frames=${NF},max_pixels=${MP},max_length=12800,disable_thinking=true" \
    OUTPUT_ROOT=logs/eval/p0_8_geometry/mp${MP}_f${NF} BENCHMARKS="vsibench" \
    bash scripts/evaluation/eval.sh
  done
done
```

v0 判据不通过时，先回到 P0-2 测特权 gap，**不要靠加数据或加复杂度补救**。备选方向：加大视角跨度（$N{=}32, K{=}1$）、改用连续帧采样制造更强的覆盖度差异、或把特权从"视角数量"扩展为"视角数量 + 深度图/点云"的组合。
