# SpatialStack SPAR3 MV-OPSD（K=1 / K=2）实验结果

- **实验 ID**：`20260909_spatialstack_mvopsd_spar3_student_k`
- **Registry**：[`20260909_spatialstack_mvopsd_spar3_student_k.yaml`](../registry/20260909_spatialstack_mvopsd_spar3_student_k.yaml)
- **最后更新**：2026-09-14
- **状态**：训练与离线全量评测均已完成

## 设置

在 SpatialStack 几何融合冷启动权重（`output/spatialstack_qwen35_train`）上，复用 20260831 SPAR3 nested 协议做 MV-OPSD：teacher 固定 3 视角；student 分别为 **K=1**（node-B，6 卡）与 **K=2**（node-A，8 卡）。同题嵌套池 4,064 行，**84 步 ≈ 1 epoch**，`batch=48`，`TEST_FREQ=SAVE_FREQ=7`。推理引擎为 **HF generate**（非 vLLM），与历史 vLLM SPAR3 曲线**不可对减**。

## 评测口径

| 层级 | 题集 | 解码 | 用途 |
| --- | --- | --- | --- |
| 训练内 | sub480（分层子集） | sample t=1.0 / top_p=0.8 | 逐步监控；**仅相对本臂 step 0** |
| 离线 | 全量 5130 | greedy @4096，boxed last-line | 与 geo SFT **66.59** 可比 |

两套口径的绝对值**不可直接相减**（题集规模与解码均不同）。

## 结果

### 离线全量 VSI-Bench（主报）

同协议全量 boxed greedy；第一行是冷启动锚点。

| 臂 | step | overall | Δ vs geo | 作答率 |
| --- | --- | --- | --- | --- |
| geo SFT | 冷启动 | **66.59** | — | 100% |
| k2 | 70 | 66.82 | +0.23 | 100% |
| k2 | 77 | 66.85 | +0.26 | 100% |
| k2 | 84 | **66.87** | **+0.28** | 100% |
| k1 | 70 | 66.28 | −0.31 | 100% |
| k1 | 77 | 66.40 | −0.19 | 100% |
| k1 | 84 | 66.36 | −0.23 | 100% |

k2 全量最佳：**step 84 = 66.87**。k1 全量最佳：**step 77 = 66.40**（与 geo 差 −0.19 pp）。

### 训练内 sub480（辅报）

| step | k1 | k2 | k1−k2 |
| --- | --- | --- | --- |
| 0 | 64.16 | 63.91 | +0.25 |
| 70 | **67.16** | 65.04 | +2.12 |
| 77 | 62.82 | 65.69 | −2.87 |
| 84 | 61.00 | 64.11 | −3.11 |

相对本臂 step 0：k2 step 84 **+0.20 pp**；k1 step 84 **−3.16 pp**。

### geo 冷启动其它基准（HF lmms-eval，`sr_opsd`）

权重仍是 `output/spatialstack_qwen35_train`。实验 ID：`20260914_spatialstack_geo_coldstart_bench`。引擎与 VSI geo **66.59** 相同（HF，非 vLLM）。

| 基准 | 协议 | overall | 备注 |
| --- | --- | ---: | --- |
| VSI-Bench | boxed @4096 | **66.59** | 20260909 |
| CV-Bench | boxed @1024 | **86.06** | 作答 100% |
| **BLINK-Spatial** | boxed @1024 | **52.29** | **只报 3 任务** |
| SPAR-Bench | boxed @2048 | **68.86** | |

BLINK-Spatial 拆分：multi_view_reasoning **5.26** / relative_depth **83.06** / spatial_relation **68.53**。同协议基座 HF 为 **65.99**。

**不报** 14 任务全量 BLINK：几何融合在多图样本上 token 不对齐（`20260914_spatialstack_geo_blink_full`）。该数字不可与 vLLM 全量 14 任务（基座 62.67）并列。

## 结论

1. **MV-OPSD 没有在 VSI 上明显掉点。** 以 geo SFT 冷启动 **66.59** 为基线，k2 三个 checkpoint 都在 **66.82–66.87**（+0.23~+0.28 pp），末步 step 84 最高；k1 在 **66.28–66.40**（−0.19~−0.31 pp），略低于基线但差值不到 0.5 pp，更像小幅波动而不是训练崩坏。两臂作答率均为 100%，没有格式退化。

2. **K=2 明显优于 K=1。** 同一 step 下，k2 全量分数稳定高出 k1 约 **0.4–0.5 pp**（例如 step 84：66.87 vs 66.36）。训练前 sub480 上两臂 step 0 几乎一样（64.16 vs 63.91）；训完后 k2 全量随 step 单调上升（66.82→66.85→66.87），k1 则在 step 77 见顶（66.40）后回落到 66.36。

3. **训练内 sub480 不能用来选最佳 checkpoint。** sub480 用 sample 解码、题量只有 480，和离线全量 greedy 的走势多次不一致：k1 在 sub480 上 step 70 最高（67.16），但全量 step 70 只有 66.28；k2 在 sub480 上 step 84 偏低（64.11），全量却是 step 84 最好（66.87）。这说明训练曲线上的峰值不等于离线主指标上的最优 step。

4. **对照实验预期：** 希望 MV-OPSD 在 geo SFT 能力附近做小幅提升。**k2 基本符合**——全量始终不低于 geo SFT，末步还有 +0.28 pp 增益。**k1 未带来稳定提升**——全量最好也只到 66.40，仍低于 geo SFT；且 sub480 与全量排名相反，不能只看训练内曲线就下结论。
