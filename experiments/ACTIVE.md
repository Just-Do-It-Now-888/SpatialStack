# 当前实验与窗口协作状态

所有窗口在启动、停止或接管实验任务时更新此文件。这里只记录当前状态；详细参数、结果和问题保存在对应 registry YAML。

**对外汇报报告**：[`experiments/reports/MV-OPSD_实验报告.md`](reports/MV-OPSD_实验报告.md)（方法与数据管线）
**MindCube 结果汇总**：[`experiments/reports/MindCube_实验结果汇总.md`](reports/MindCube_实验结果汇总.md)（zero-shot / SFT / K=1 MV-OPSD）
**评测对照（基座 vs SPAR3 step 55）**：[`experiments/reports/评测对照_基座_vs_SPAR3_k1_step55.md`](reports/评测对照_基座_vs_SPAR3_k1_step55.md)
**近期结论稿**：[`experiments/reports/汇报_20260825_近期实验结论与影响.md`](reports/汇报_20260825_近期实验结论与影响.md)
（8.17–8.24，结论先行：监督可靠性、长度爆炸根因、解码与判分口径）
**阶段汇报稿**：[`experiments/reports/汇报_20260821_优化超参与视角设计.md`](reports/汇报_20260821_优化超参与视角设计.md)
（rollout / batch / LR / view 四个方向的横向结论，含全部 8 条 run 的曲线数字）

**训练环境（2026-09-10 起）**：**一律 conda `sr_opsd`**（LESSON-041）。
不要再用 `spatialstack-qwen35`、`vision-opd` 或系统 python + `~/.local` 启动新训练。
评测环境另记；启动前 python 路径须含 `/envs/sr_opsd/`。

## 机器与节点

自 2026-08-18 起本项目跨 DSW 实例并行。node-A/B 都是 8×H20（95 GiB/卡）、
驱动 570.133.20、Ubuntu 24.04.3；项目路径与家目录一致
（`/home/c30084464/Documents/code/SpatialStack_OPSD`）。
2026-09-12 起增加 **node-C**（同规格 8×H20）。从 node-A 互连走 pod 内网 22 端口，
**不要经 `60.205.212.164:996` / `:1000`**（DSW 公网网关不接受容器内回连）。

| 代号 | 主机名 | 内网 IP | SSH 别名 | 当前占用 |
| --- | --- | --- | --- | --- |
| node-A | `dsw-455344-95d655c8f-xm7dq` | `22.36.109.109` / `10.10.10.70` | （本机） | **`20260919` Answer-OPSD**：训练中 PID 2890470 |
| node-B | `dsw-455361-5459dbbfc9-gp4s6` | `22.36.104.74` / `10.10.10.69` | `ssh opsd2`（`c30084464`） | **0-5 外部作业约 45GB、6-7 各 91GB** |
| node-C | `dsw-455370-5559cfd88c-6gl5t` | `22.36.100.166` / `10.10.10.217` | `ssh opsd3` 或 `ssh nodec`（**root**） | **空闲**；Hound half lastline 已出分 |

### `20260919_spatialstack_answer_opsd_spar_fullviews`（**训练中**：等视图 Answer-OPSD，node-A）

- Registry：[`experiments/registry/20260919_spatialstack_answer_opsd_spar_fullviews.yaml`](registry/20260919_spatialstack_answer_opsd_spar_fullviews.yaml)
- 方法：student=teacher=全相册；仅 teacher 接 GT（`teacher_prompt_mode=answer_hint`）
- 数据：parquet **54495** train / 1000 holdout；N/K=1（3/3 与 32/32）
- 对照：`20260918` SPAR T32/S16 视图特权
- PID **2890470** / PGID **2890470** / python **2890519**
- 入口：`scripts/opsd/run_answer_opsd.sh`；conda **sr_opsd**；`ROLLOUT_NAME=hf`
- 日志：`logs/train/20260919_spatialstack_answer_opsd_spar_fullviews/train_20260919_164245.log`
- SMOKE：answer_hint_fraction=1.0，fallback=0，vopd_loss=0.00381，smoke8 64.17→65.42
- 泄漏：Hint 模板 0/16；1 条 exact GT 是正确短答 `curtain(Object2)`
- 不要抢 node-C（Hound half 已收口，8 卡空闲）
- 最后更新：2026-09-19 16:45 UTC+8

### `20260918_spatialstack_mvopsd_llava_hound_half`（**已完成**：Teacher 全帧 / Student 一半 lastline，node-C）

- Registry：[`experiments/registry/20260918_spatialstack_mvopsd_llava_hound_half.yaml`](registry/20260918_spatialstack_mvopsd_llava_hound_half.yaml)
- 训练 781 步结束；选 ckpt **750**
- **lastline**（SPAR 措辞，无 `\boxed{}`）：
  - VSI @4096：700 **66.32** / 750 **66.53** / 781 **66.34**
  - CV @1024：**84.98**
  - SPAR @2048：**68.51**
  - BLINK 10 项已出分（跳过 4 项 fusion）；**不报 14 宏**。spatial-3：9.77 / 81.45 / 69.23 → **53.48**
- 对照量级：geo 66.46 / 85.99 / 68.92；VLM-635 64.02 / 82.40 / 66.17（Student 规则不同）
- 链 **16:38 UTC+8** DONE；8 卡空闲
- wandb：[fzal1dfu](https://wandb.ai/slcheng/MV-OPSD/runs/fzal1dfu)
- 最后更新：2026-09-19 19:50 UTC+8

### `20260918_blink_full_lastline`（**已完成**：BLINK val lastline 10/14，node-C）

- Registry：[`experiments/registry/20260918_blink_full_lastline.yaml`](registry/20260918_blink_full_lastline.yaml)
- 口径：`BLINK_PROTOCOL=lastline` greedy @1024；HF；conda **sr_opsd**；逐任务
- **不报 14 项宏平均**（缺 art_style / functional_correspondence / semantic_correspondence / visual_correspondence，fusion 不对齐）
- 已出分 10 项（作答均为 100%，回复几乎单字母）：

| 任务 | geo | VLM-635 |
| --- | ---: | ---: |
| counting | 53.33 | 43.33 |
| forensic | 64.39 | 62.88 |
| iq_test | 24.67 | 22.00 |
| jigsaw | 72.00 | 70.67 |
| **MVR** | **5.26** | **6.77** |
| object_loc | 52.46 | 49.18 |
| relative_depth | 83.06 | 87.90 |
| relative_reflectance | 45.52 | 54.48 |
| spatial_relation | 68.53 | 68.53 |
| visual_similarity | 88.89 | 83.70 |
| **spatial-3 宏** | **52.29** | **54.40** |

- MVR：n=133，answered 100%，pred 仅 A/B（geo 77/56），不是截断/解析失败
- 不可与 vLLM 14 任务 boxed **62.67** 并列
- 最后更新：2026-09-18 16:16 UTC+8

### `20260918_spatialstack_geo_lastline`（**已完成**：geo SFT lastline，node-C）

- Registry：[`experiments/registry/20260918_spatialstack_geo_lastline.yaml`](registry/20260918_spatialstack_geo_lastline.yaml)
- 权重：`output/spatialstack_qwen35_train`；HF lmms-eval；conda **sr_opsd**；链 **12:21** 结束，8 卡空闲
- **lastline**（SPAR 措辞，无 `\boxed{}`）：
  - VSI @4096（5130 题，作答 100%）：**66.46**
  - CV @1024（2638 题，作答 100%）：**85.99**
  - BLINK-Spatial 3 任务：**52.29**（5.26 / 83.06 / 68.53）
  - SPAR @2048（7211 题）：**68.92**
- 同口径对 VLM-3R step 635：VSI **64.02**、CV **82.40**、SPAR **66.17**；635 的 BLINK lastline 仍缺
- **不可与** geo boxed VSI **66.59** 对减
- 最后更新：2026-09-18 14:23 UTC+8

### `20260915_spatialstack_mvopsd_vlm3r_t8_s4`（**评测已出分**：greedy last3 lastline，node-C）

- Registry：[`experiments/registry/20260915_spatialstack_mvopsd_vlm3r_t8_s4.yaml`](registry/20260915_spatialstack_mvopsd_vlm3r_t8_s4.yaml)
- 训练已完成（greedy 635）。离线口径 **lastline**（SPAR 措辞，无 `\boxed{}`）：merge 550/600/635 → 全量 VSI lastline @4096 → 最好的再跑 CV / BLINK-Spatial / SPAR
- 入口：`scripts/opsd/eval_vlm3r_t8_s4_last3_then_benches.sh`；conda **sr_opsd**；HF 非 vLLM；链 **2026-09-17 12:26** 结束（PID 1390889 已退出）
- 日志：`logs/eval/20260915_spatialstack_mvopsd_vlm3r_t8_s4/last3_then_benches.nohup.log`
- **VSI lastline @4096（5130 题，作答 100%）**：550 **63.55** / 600 **63.58** / 635 **64.02**（选 635）
- **step 635 三集**：CV lastline **82.40**（2638 题）；SPAR lastline **66.17**（7211 题）；**BLINK-Spatial 失败**（C 上 `BLINK_PROTOCOL` 当时只有 `original`/`spatialstack`，不认 `lastline`）
- 训练内 greedy sub480：step 0 **61.59** → step 635 **58.28**
- **不可与** geo VSI boxed **66.59**、geo CV boxed **86.06**、geo SPAR boxed **68.86** 对减
- 原始 plan 写的是 boxed VSI 对 geo 66.59；后改为 lastline，boxed 全量未完成
- 最后更新：2026-09-18 10:50 UTC+8

### `20260918_spatialstack_mvopsd_spar_t32_s16`（**已完成**：best-3 lastline CV/BLINK-Spatial/SPAR，node-A 已释放）

- Registry：[`experiments/registry/20260918_spatialstack_mvopsd_spar_t32_s16.yaml`](registry/20260918_spatialstack_mvopsd_spar_t32_s16.yaml)
- 训练：1135/1135；in-loop greedy VSI **450 68.20% / 1100 67.59% / 600 67.29%**（与全量 VSI lastline 不可并列）
- wandb：[ry2wrce6](https://wandb.ai/slcheng/MV-OPSD/runs/ry2wrce6)
- lastline 对照 geo SFT（`20260918_spatialstack_geo_lastline`）：CV **85.99** / BLINK-Spatial **52.29** / SPAR **68.92**
- SPAR T32/S16 lastline：

| ckpt | CV | BLINK-Spatial | SPAR |
| --- | ---: | ---: | ---: |
| geo SFT | 85.99 | 52.29 | 68.92 |
| step 450 | 84.48 | 52.81 | 68.37 |
| step 600 | 83.96 | 50.98 | 67.93 |
| step 1100 | 84.15 | 53.16 | 67.10 |

- BLINK 拆开（MVR / relative_depth / spatial_relation）：geo 5.26 / 83.06 / 68.53；450 3.01 / 85.48 / 69.93；600 5.26 / 79.84 / 67.83；1100 6.02 / 81.45 / 72.03
- 不可与 boxed 66.59 / 86.06 / 68.86 对减
- 最后更新：2026-09-19 16:11 UTC+8

### `20260915_spatialstack_mvopsd_spar_t32_s16`（**本机权重已删**：node-A 空闲）

- Registry：[`experiments/registry/20260915_spatialstack_mvopsd_spar_t32_s16.yaml`](registry/20260915_spatialstack_mvopsd_spar_t32_s16.yaml)
- sample 轮在约 step **950** 按用户要求停下；ckpt 曾归档 `..._valsample_aborted_20260916`
- greedy 重训：1135 步；PID **3521557** / PGID **3521496**；chain_after **3536334**
- 日志 `logs/train/20260915_spatialstack_mvopsd_spar_t32_s16/train_20260916_133559.log`
- wandb：**online** [9ipc8ihl](https://wandb.ai/slcheng/MV-OPSD/runs/9ipc8ihl)（旧 sample [02lqzpd7](https://wandb.ai/slcheng/MV-OPSD/runs/02lqzpd7) 勿并列）
- **2026-09-18 10:39 UTC+8**：用户要求删除本机 `checkpoints/` 与 `output/`（约 1.4T，含 SPAR FSDP、SFT `spatialstack_qwen35_train*`、HF merge）。目录已清空并重建为空。node-C 家目录非 live NFS，VLM last3 评测不受此次删除影响。
- 最后更新：2026-09-18 10:39 UTC+8

### `20260914_teacher_key_plus_probe`（**已完成**：关键帧 + 1..10 额外帧准确率曲线）

- Registry：[`experiments/registry/20260914_teacher_key_plus_probe.yaml`](registry/20260914_teacher_key_plus_probe.yaml)
- 窗口：cursor-window-mvopsd
- 模型：`models/Qwen3.5-4B`；conda **sr_opsd**；8 卡 TP=8；墙钟 **461 s**
- 设计：保留 `required_views`，farthest-point 嵌套补 1–10 帧；同一 1000 道 spar32 题
- **结论（规则层，每档 1000 题）**：
  - 整体：key+0 **21.22** → key+10 **21.21**（Δ **−0.01pp**）；中间档 19.9–21.4，无上升趋势
  - `distance_prediction_oo_video`：28.48 → 34.84（**+6.4pp**）
  - `distance_infer_center_oo_video`：38.80 → 37.60；想象题补帧无益
- 产物：`logs/eval/teacher_key_plus_probe/spar32/curve_summary.json`
- 最后更新：2026-09-14 19:32 UTC+8

### `20260914_teacher_keyframe_probe`（**已完成**：Teacher 关键帧 vs 全相册配对精度）

- Registry：[`experiments/registry/20260914_teacher_keyframe_probe.yaml`](registry/20260914_teacher_keyframe_probe.yaml)
- 窗口：cursor-window-mvopsd
- 模型：`models/Qwen3.5-4B`（冻结 teacher）；环境 **sr_opsd**；单卡 `CUDA_VISIBLE_DEVICES=0 TP=1 GPU_MEMORY_UTILIZATION=0.35`（8 卡被 GRPO 占用）
- 数据：`spar_32view` 1000 题 × 2 臂 + `spar_3view` 500 题 × 2 臂；关键帧 = `required_views`
- **结论（规则层判分，配对 Δacc = key − full）**：
  - **spar32**（32→3.1 帧）：full **21.98%** / key **21.36%**（**Δ −0.62pp**）→ 几乎无精度损失
  - **spar3**（3→1.9 帧）：full **31.29%** / key **34.08%**（**Δ +2.79pp**）→ 省帧极少，噪声主导
  - 分题型：spar32 `distance_prediction_oo_video` **−7.3pp**；`distance_infer_center_oo_video` **+4.4pp**
- 产物：`data/mvopsd/parquet_keyframe_sweep/`；`logs/eval/teacher_keyframe_probe/{spar32,spar3}/paired_summary.json`
- 复现：`bash scripts/opsd/run_teacher_keyframe_probe.sh`（冒烟 `SMOKE=1 SAMPLE_ROWS=64`）
- 最后更新：2026-09-14 18:49 UTC+8

### `20260914_spatialstack_geo_coldstart_bench`（**已完成**：CV / BLINK-Spatial / SPAR）

- Registry：[`experiments/registry/20260914_spatialstack_geo_coldstart_bench.yaml`](registry/20260914_spatialstack_geo_coldstart_bench.yaml)
- 窗口：cursor-window-spatialstack
- 权重：`output/spatialstack_qwen35_train`（对照 VSI geo **66.59**）
- 环境：**conda `sr_opsd`** + `PYTHONPATH=src`；HF lmms-eval（非 vLLM）
- **geo 冷启动实测**：
  - CV-Bench boxed @1024：**86.06**（作答 100%）
  - **BLINK-Spatial（仅 3 任务）**：**52.29**（5.26 / 83.06 / 68.53）；**不报 14 任务全量**
  - SPAR-Bench boxed @2048：**68.86**
- k1/k2 换训练源（LLaVA-Hound / VLM-3R）的 VSI 曲线为**模拟**；见报告 `SpatialStack_SPAR3_MV-OPSD_k1k2_实验结果.md`
- 产物：`logs/eval/20260914_spatialstack_geo_coldstart/20260914/{cvbench,blink_spatial,sparbench}/`
- 最后更新：2026-09-14 21:16 UTC+8

### `20260911_spatialstack_qwen35_geo_sroopsd`（**已完成**：训练 + lmms_eval 三口径）

- Registry：[`experiments/registry/20260911_spatialstack_qwen35_geo_sroopsd.yaml`](registry/20260911_spatialstack_qwen35_geo_sroopsd.yaml)
- 窗口：cursor-window-spatialstack
- **训练（sr_opsd）**：step **3308**，train_loss **0.5165**，墙钟 **33495 s**
- 权重：`output/spatialstack_qwen35_train_sroopsd/`
- **VSI lmms_eval 全量（5130 题，answered 100%）**：
  - boxed @4096：**66.21**（vs 20260909 spatialstack-qwen35 **66.59**，−0.38）
  - plain @1024：**66.30**（vs **66.63**，−0.33）
  - lmms_legacy @16：**66.30**（vs 论文带 **67.65**，−1.35；plain 与 lmms_legacy 逐题型相同，待核对是否短答形态导致）
- 产物：`logs/eval/20260911_spatialstack_qwen35_geo_sroopsd/{vsibench_boxed,vsibench_plain,vsibench_lmms_legacy}/`
- 8 卡已释放
- 最后更新：2026-09-12 16:05 UTC+8

### `20260909_spatialstack_mvopsd_spar3_student_k`（**已完成**：训练 + 全量评测；16:18）

- Registry：[`experiments/registry/20260909_spatialstack_mvopsd_spar3_student_k.yaml`](registry/20260909_spatialstack_mvopsd_spar3_student_k.yaml)
- **结果报告**：[`experiments/reports/SpatialStack_SPAR3_MV-OPSD_k1k2_实验结果.md`](reports/SpatialStack_SPAR3_MV-OPSD_k1k2_实验结果.md)
- 全量 VSI boxed greedy @4096（5130 题，对照 geo SFT **66.59**）：
  - k2：70 **66.82** / 77 **66.85** / 84 **66.87**（均 +0.2~+0.3 vs geo）
  - k1：70 **66.28** / 77 **66.40** / 84 **66.36**（均 −0.2~−0.3 vs geo）
- 训练内 sub480（sample）：k2 step 0→84 **63.91→64.11**；k1 **64.16→61.00**
- 产物：`logs/eval/20260909_spatialstack_mvopsd_spar3_k{1,2}_step{70,77,84}/`
- 最后更新：2026-09-10 16:30 UTC+8


### `20260909_spatialstack_qwen35_geo`（**训练已完成**；boxed/plain 已完成；**lmms_legacy 评测中**）

- Registry：[`experiments/registry/20260909_spatialstack_qwen35_geo.yaml`](registry/20260909_spatialstack_qwen35_geo.yaml)
- 窗口：cursor-window-spatialstack
- 1 epoch 冷启动结束：`global_step=3308` / `max_steps=3310`（epoch 0.9996）；`train_loss=0.5162`；墙钟 **33507 s（9.31 h）**
- 最终权重：`output/spatialstack_qwen35_train/`（`config.json` + `model.safetensors` 14.5G；架构 `Qwen3_5ForConditionalGenerationWithGeometry`）
- 中间 ckpt：`checkpoint-{1000,2000,3000}` 各 75G；整树 238G
- 进程已退出；8 卡 0 MiB。末尾仅有 NCCL `destroy_process_group` 警告，权重已写完。
- 评测：HF/lmms-eval。conda `spatialstack-qwen35` + `PYTHONPATH=src`。HF offline。
  已完成 boxed（`spatialstack` @4096）与 plain（`spatialstack_plain` @1024）。
  **进行中**：`VSIBENCH_PROTOCOL=lmms_legacy`（16 token + `split(' ')[0]`，论文/上游官方）。
- 产物：`logs/eval/20260909_spatialstack_qwen35_geo/{vsibench_boxed,vsibench_plain,vsibench_lmms_legacy,smoke_*}/`
- 驱动：`logs/eval/20260909_spatialstack_qwen35_geo/run_dual.sh lmms_all`；日志 `lmms_legacy_chain.log`；PGID **1961600**
- 对照：boxed vs 基座 **48.06**；plain vs 基座 **52.75**；lmms_legacy 只对论文 16-token（如 64.24 / 67.65），**不可与 66.59/66.63 对减**
- **VSI HF 结果（口径不可并列）**：
  - boxed last-line @4096：**66.59**（作答 100%；vs 基座 48.06，+18.53）
  - original/plain @1024：**66.63**（作答 100%；vs 基座 52.75，+13.88）
  - lmms_legacy @16：**跑评测中**
- 最后更新：2026-09-09 23:18 UTC+8

### 磁盘：MindCube 训练权重已清空（2026-09-09 10:08，两机）

按用户要求删除本项目 **MindCube SFT / MV-OPSD** 权重，评测日志与 parquet 未动。
未删 3DThinker Stage1、基座 `models/Qwen3.5-4B`、SPAR3 merge。

- node-A：`output/20260906_qwen35_mindcube_sft_{answeronly,cot}`（602G+9.7G）、
  `checkpoints/20260907_mindcube_mvopsd_k1_armB`（850G）、
  `checkpoints/20260907_mindcube_mvopsd_k1_ema_armA`（850G）、两个空壳目录
- node-B：同名 SFT（9.7G+602G）、`..._k1_armA`（54G）、`..._ema_armB`（213G）、两个 smoke 空壳
- 两机 `output/` 与 `checkpoints/` 下已无 `*mindcube*` 权重目录

### `20260908_sparbench_vllm_boxed`（**已完成**，node-A 8 卡已释放；22:29）

- Registry：[`experiments/registry/20260908_sparbench_vllm_boxed.yaml`](registry/20260908_sparbench_vllm_boxed.yaml)
- 窗口：cursor-window-mvopsd
- 题面 HF boxed 后缀；解析 boxed-primary；vLLM TP=8 greedy @2048
- 基座 **40.19**（截断 17.57%，框 82.5%，中位 411）；SPAR3 **42.65**（3.36% / 95.0% / 306）；Δ **+2.46**
- 与 lastline 39.72 / 42.16（+2.44）方向同，不可并列相减
- 对照表 SPAR 节已写入
- 产物：`logs/eval/20260908_sparbench_vllm_boxed/`
- 最后更新：2026-09-08 22:40 UTC+8

### `20260908_cvbench_vllm_boxed`（**已完成**，node-A 8 卡已释放；18:34）

- Registry：[`experiments/registry/20260908_cvbench_vllm_boxed.yaml`](registry/20260908_cvbench_vllm_boxed.yaml)
- 窗口：cursor-window-mvopsd
- 题面 HF boxed 后缀；解析 `boxed_lastline`；vLLM TP=8 greedy @1024
- 基座 **87.62**（2D 83.15 / 3D 92.08，作答 99.39%）；SPAR3 **85.23**（80.04 / 90.42，作答 99.81%）；Δ **−2.39**
- 相对 HF 同题面 88.07 / 85.05：引擎差 −0.45 / +0.18
- 不可与 lastline 71.04 / 83.84 对减
- 产物：`logs/eval/20260908_cvbench_vllm_boxed/`
- 最后更新：2026-09-08 18:35 UTC+8

### `20260901_qwen35base_cvbench_boxed_lastline` 现行解析重打（**已完成**，不占卡；18:10）

- 源 dump：`logs/eval/20260901_qwen35base_cvbench_boxed_lastline/`（HF boxed 生成，原 combined **88.07**）
- 现行 `parser=lastline`（不拆 `\boxed{}`）：**0.33**，作答 **0.91%**（2621 行末行是 `\boxed{X}`）
- 同 dump `boxed_lastline` 回放仍 **88.07**
- 产物：`logs/eval/20260901_qwen35base_cvbench_boxed_lastline/rescore_lastline/`
- 不可与 vLLM lastline 题面 71.04 并列
- 最后更新：2026-09-08 18:10 UTC+8

### `20260908_lastline_prompt`（**已完成**，node-A 8 卡已释放；17:16）

- Registry：[`experiments/registry/20260908_lastline_prompt.yaml`](registry/20260908_lastline_prompt.yaml)
- 窗口：cursor-window-mvopsd
- 题面后缀与 SPAR 相同（无 `\boxed{}`）；解析末行
- greedy vLLM：VSI **49.03 / 49.02（−0.01）**；BLINK 14 **63.43 / 64.94（+1.51）**；CV **71.04 / 83.84（+12.80）**（基座作答 78.81%）
- CV 是 vLLM，不可与 HF boxed 88.07 / 85.05 并列
- 对照表已写入；SPAR lastline 与 original 未重跑
- 产物：`logs/eval/20260908_lastline_prompt/`
- 最后更新：2026-09-08 17:20 UTC+8

### `20260908_qwen35base_mindcube_zeroshot`（**已完成**，node-A 8 卡已释放；14:58）

- Registry：[`experiments/registry/20260908_qwen35base_mindcube_zeroshot.yaml`](registry/20260908_qwen35base_mindcube_zeroshot.yaml)
- 窗口：cursor-window-mvopsd
- 模型：`models/Qwen3.5-4B` zero-shot
- 协议：MindCube tinybench 1050；官方 `input_prompt`；transformers 5.3.0 八卡分片；greedy 2048；`disable_thinking`
- overall **46.86** / among 42.50 / around 58.80 / rotation 45.00；不可读 0%；中位 288 token；撞 cap 2%；视觉 token/图中位 300
- 与 20260906 baseline **逐项相同**
- 产物：`logs/eval/20260908_qwen35base_mindcube_zeroshot/`
- 不可与 SFT 74.48/69.81 或训练内 vLLM 74.29/70.57 并列
- 最后更新：2026-09-08 14:58 UTC+8

### `20260908_blink_full_vllm`（**已完成**，node-A 8 卡已释放；13:34）

- Registry：[`experiments/registry/20260908_blink_full_vllm.yaml`](registry/20260908_blink_full_vllm.yaml)
- 窗口：cursor-window-mvopsd
- 协议：vLLM TP=8 greedy @1024；14 任务 val 1901 题宏平均
- boxed last-line（**归档**，含 `\boxed{}`）：**62.67 / 63.66（+0.99）**
- original **末行重打**（现行）：**64.98 / 61.98（−3.00）**；句首 64.82 / 4.10 不当能力差
- 现行口径与 SPAR 对齐：题面差只用 `The final answer MUST BE put on the last line of your response.`，不认 `\boxed{}`；lastline 题面见 `20260908_lastline_prompt`
- 对照表已加独立 vLLM 全量表，未与 HF spatial 65.99/68.92 混报
- 产物：`logs/eval/20260908_blink_full_vllm/`
- 最后更新：2026-09-08 13:35 UTC+8

（2026-09-07 23:43 磁盘：按用户要求在 node-B 删除 Model-Dowser 的 LLaVA/logs/Molmo-Finetune，
以及 `checkpoints/20260821_mvopsd_single_vlm3r_main/` 与
`checkpoints/20260831_mvopsd_spar3_k2_epoch1_nothink/`。根盘 `/` 从 4.5T/100% 降到 **2.5T/57%，剩余 2.0T**。）

（2026-09-07 12:5x 实测 `nvidia-smi`：两机原本都是 8 卡全空，
20260906 SFT 轮的评测已收口，上表此前的「8 卡占用」是过期记录。）

跨机访问：node-A 上 `ssh opsd2` 直连 node-B（用户 `c30084464`）；
`ssh opsd3` / `ssh nodec` 直连 node-C（用户 **root**；公网别名 `remote-server3` 仍可用）。
密钥均为 `~/.ssh/id_ed25519_transfer`。node-C 的 `c30084464` 账号尚未授权该公钥。

**不要经 `60.205.212.164:1000` 互连**——那是 DSW 的 SSH 网关，只接受外部客户端，
容器内部发起的 TCP 会超时（ICMP 却是通的，容易误判成认证问题）。走 pod 内网 IP 的
22 端口，实测 rsync 吞吐 600 MB/s。

### `20260907_mindcube_mvopsd_k1_ema`（**权重已删**；训练记录仍以 registry 为准）

- Registry：[`experiments/registry/20260907_mindcube_mvopsd_k1_ema.yaml`](registry/20260907_mindcube_mvopsd_k1_ema.yaml)
- 窗口：`cursor-window-mindcube-opsd`
- 相对 frozen 轮的唯一变量：`teacher_regularization=ema`、`teacher_update_rate=0.05`（Vision-OPD 同式）。
- 机器：**node-A = arm A**（answer-only SFT-314，conda `vision-opd`）；
  **node-B = arm B**（CoT SFT-314，系统 python3.12）。
- wandb：**online**，project `MV-OPSD`，run `20260907_mindcube_mvopsd_k1_ema_arm{A,B}`。
- **SMOKE PASS**（node-A arm A）：step-0 74.38；swap=1.0；fallback=0；
  `teacher_probe` 0.044563621 → 0.044563675（跨步漂移，EMA 生效）；median 11 token。
- 正式跑：156 步 / batch 64 / TEST_FREQ=SAVE_FREQ=10。
  arm A 日志 `logs/train/ema_armA_driver.log`；arm B 在 node-B `logs/train/ema_armB_driver.log`。
- 不要覆盖 frozen 轮 `checkpoints/20260907_mindcube_mvopsd_k1_arm{A,B}/`。
- **2026-09-09**：两机 EMA / frozen / SFT 权重目录均已删除。
- **2026-09-09**：两机 EMA / frozen / SFT 权重目录均已删除（见上节）。
- 下一步：盯 step-0 与 teacher_probe 漂移；step 10 看长度是否爆炸。
- 最后更新：2026-09-07 20:17 UTC+8

### `20260907_mindcube_mvopsd_k1`（**已完成**：arm A 在 node-B、arm B 在 node-A）

- Registry：[`experiments/registry/20260907_mindcube_mvopsd_k1.yaml`](registry/20260907_mindcube_mvopsd_k1.yaml)
- 窗口：`cursor-window-mindcube-opsd`
- 目标：MindCube 上一轮 MV-OPSD，**teacher 看全部 N 视图 / student 只看 1 视图（K=1）**。
  两条 arm 各从对应的最优 SFT checkpoint 起跑（answer-only 74.48 / CoT 69.81），
  唯一的**配置**变量是 SFT 监督类型。**注意**：两条 arm 蒸馏的 token 数差 30 倍
  （11 vs 319），所以「只有初始化不同」对有效监督不成立。
- **执行位置已改为单机 node-B（2026-09-07 14:50，见 ISSUE-702）**：原计划两机各跑一条 arm，
  但 node-A 环境在修复过程中被弄成两机混合状态而不可用。同机跑反而更严格——
  连 torch 构建 / CUDA / numpy 都相同，不再有设计外的跨机差异。
  arm A 的 checkpoint-314（9.7 GB）与全部数据产物已同步到 node-B，
  node-B 上 MindCube 测试 40 passed，探针端到端跑通。
- 数据：`data/mvopsd/parquet_mindcube_k1/main_train.parquet`，**9,999 行全量无过滤**
  （用户决策）。2,785 个去重视图，`data_source` 六个桶按 family × N 分。
  **13% 是刻意接受的证据缺失**：1,302 条 pair 位移题保留「these two views」原文却只给一张图，
  独立桶 `mindcube_pair_2view` 使其事后可分离；任何增益都要同时报剔除它之后的数字。
- 图像口径：**不过 VGGT 居中裁切**（竖图会丢 24.5% 高度，MindCube 93.5% 是竖图），
  改走 `pixel_budget` 缩放；`min_pixels=200704 / max_pixels=1605632` 写进 parquet 每个
  image dict。实测每图中位 **300 token**（与 SFT 轮逐字一致），teacher 峰值 6,120 < 12,288。
- 已过硬门禁：`check_mvopsd_geometry.py --parquet` 400 样本 / 1,434 视图上
  student 与 teacher 图像路径**逐像素一致**。
- **ISSUE-701（已修复）**：node-A 的 DSW 系统镜像被重建成 NGC nv25.11 / CUDA 13.0，
  `~/.local` 里 OPSD 训练用的 transformers 5.5.0 连同匹配的 regex 一起消失，
  qwen-vl-utils 退回 0.0.11（本仓 verl 需要 0.0.14 的签名）。已从 node-B 恢复同一份
  transformers 5.5.0 并对齐 regex / qwen-vl-utils。
- **⚠️ ISSUE-702：node-A 的 `~/.local` 现在不可用，其他窗口勿用该 python 环境。**
  修 vLLM 时把 node-B 整个 `site-packages` 覆盖同步了过去，而 user site 只能遮蔽
  dist-packages 不能替换它，结果是按 numpy 1.26 与 2.1 两套 ABI 混编的环境：
  pydantic-core 配对失败、scipy ufunc 报错，`pd.read_parquet` 直接崩。无备份、不可回滚。
  该机**系统层仍自洽**（`PYTHONNOUSERSITE=1`：numpy 2.1.0 / NGC torch 2.10.0a0 CUDA 13.0 /
  pandas / pyarrow 均正常），但缺 transformers 与 vllm。
  `~/.local` 的处置（移走留证据 / 重建 / 保留）待定；详见 registry 的 ISSUE-702 与 LESSON-039。
- gate-0 可恢复性测量**已完成**（node-B 同机同引擎）：full-view 复现锚点
  74.29 / 70.57（vLLM 口径，对 transformers 的 74.48 / 69.81 差 -0.19 / +0.76）；
  单视图 59.24 / 55.24，gap 约 -15 pp，覆盖率 100%。
  **274 行 pair 上 arm A 的全视图与单视图差 +0.36**——teacher 本来就没在用第二个视图，
  那 13% 不是「student 答不了」而是「没有可传的东西」；剔除后真实 gap 为 -20.49 / -19.72。
  用户已确认保持原协议不剔行（独立桶事后可分离）。
- **step-0 门禁值已改为 vLLM 口径 74.29 / 70.57**，训练内验证走 vLLM，
  照 transformers 锚点判断会误判（run_mvopsd.sh 的提示同步改了）。
- 资源：**两机并行，各 8 卡**（用户决策 2026-09-07 16:55 改为并行）。
  - **arm A（answer-only）在 node-B**，16:05 起，`logs/train/armA_driver.log`，
    解释器是系统 `/usr/bin/python3.12` + `~/.local`（LESSON-007 的老路径）。
  - **arm B（CoT）在 node-A**，16:56 起，`logs/train/armB_driver.log`，
    解释器是 **conda `vision-opd`**（见下条）。CoT 权重已从 node-B rsync 到本机
    `output/20260906_qwen35_mindcube_sft_cot/checkpoint-314`（9.7 GB / 17 秒 / 593 MB·s⁻¹）。
  - 机器与 arm 的对应关系与最初设计相反（原为 node-A=armA），
    因为 arm A 在 node-B 上已跑到 74/156，换机要丢弃约 55 分钟计算而无科学收益。
- **ISSUE-702 的正解：node-A 有 conda 环境 `vision-opd`，与 node-B 训练环境数值等价。**
  之前整场环境折腾是修错了对象——OPSD 脚本走「系统 python3.12 + ~/.local」（LESSON-007），
  而这个名字就叫 vision-opd 的 conda 环境一直是好的：
  torch 2.10.0+cu128 / vllm 0.18.0 / transformers 5.5.0 / numpy 1.26.4 / triton 3.6.0，
  与 node-B 逐字相同；21 个关键包里只有 `flash-attn` 一处差异（node-A 有、node-B 无）。
  等价性由 SMOKE 实测背书：**`teacher_probe` 两机逐位相同 0.044563621282577515**，
  step-0 74.76（node-A）对 74.38（node-B），差 0.38 pp 在 vLLM 调度噪声内
  （node-B 自身三次验证就在 74.19–74.57 波动）。
- **SMOKE 两机都 PASS**（2 步 / batch 8，均为 answer-only init）：
  node-B step-0 74.38、node-A step-0 74.76，两边
  `teacher_image_swap_fraction=1.0`、`policy_fallback_fraction=0`、
  `teacher_probe` 跨步且跨机恒定、`vopd_loss` 在动（0.0108→0.0232 / 0.0109→0.0247）。
  node-B 的 among_4view 92.34 与探针逐位一致。
- 两条 arm 同配置：156 步 / batch 64 / TEST_FREQ=SAVE_FREQ=10（16 个评测点）、
  rollout_n=1、frozen teacher、JUDGE=0、greedy、**WANDB offline**（两条一致，跑完再 sync）。
  各自的 `checkpoints/` 与 `logs/val/` 用实验名同名目录。
- 读数纪律：以 `val-aux/mindcube/nviews/{3,4}/acc` 与 family 分解为主；
  `nviews/2` 全是 pair 行（探针证明无特权信号），overall 被它稀释 26%。
- **arm A 训练内曲线（满视图 tinybench，17 点）**：step 0 **74.38** → 最佳 **step 30 = 75.52**（+1.14）；
  among 80.00 / nviews-3 76.52 同在 step 30 最高；之后 overall 滑到末步 68.76。
- **FSDP（2026-09-07 23:50，node-B）**：按用户要求只留
  `checkpoints/20260907_mindcube_mvopsd_k1_armA/global_step_30`（54G），
  其余 15 个 step 已删；`latest_checkpointed_iteration=30`。val dump / 训练日志未动。
- 下一步：盯 arm B 的 step-0 是否打中 **70.57**（CoT 的 vLLM 口径）；
  两条 arm 各 16 个评测点跑完后做对读。
- 最后更新：2026-09-07 23:50 UTC+8

### `20260906_qwen35_mindcube_sft`（**进行中**，node-A 8 卡：基座锚点评测；23:20 起）

- Registry：[`experiments/registry/20260906_qwen35_mindcube_sft.yaml`](registry/20260906_qwen35_mindcube_sft.yaml)
- 窗口：`cursor-window-mindcube`
- 目标：Qwen3.5-4B 在 MindCube 上 SFT，**两条 arm 唯一变量是监督目标**——
  arm A answer-only（`<answer>C. Curtain</answer>`，中位 9 token）vs
  arm B 文本 CoT（3DThinker 的推理正文 + 同一 answer 行，中位约 381 token）。
  步数口径沿用 3DThinker Stage1（batch 64、156 步/epoch），先跑 **2 epoch = 312 步、lr 1e-5**。
- 数据：`data/mindcube/mindcube_train_{answeronly,cot}.json`，各 **9999 条**逐行同题同序。
  全部自洽门禁通过（题面模板 1050/1050 常量、题面逐字 10000/10000、
  gt 自洽 10000/10000、CoT 结构 10000/10000）。丢弃 idx=4050（CoT 内答案与 gt 打架）。
  **join 键坑**：`idx.jsonl` 无 id，`idx` 索引的是 3DThinker 内部序，按行号对
  `MindCube_train` 只能对上 1447/10000；改用 (question, 图片路径元组) 复合键才 100% 命中
  ——`question` 单独 join 不可用（只有 8104/10000 唯一，同 LESSON-028）。
- **本轮训练侧关掉 VGGT 图像预处理**（用户决策）。原因：VGGT 的 `crop` 模式会居中裁切
  竖图高度，实测 480×640 竖图上下共丢 **24.5%**，而 MindCube **93.5% 的图会被裁**。
  该行为在 SPAR 那种横图上从不触发，所以照搬不是复刻历史而是引入新的视野损失。
  绕过后训练侧 **300 token/图**，与评测侧逐项相同（已实测）。
  **代价：本轮图像口径不可与历史 SFT run 并列**，曲线内部仍完全可比。
- 代码改动（默认值全部不变，历史复现不受影响）：`prepare_image_inputs` 加
  `use_vggt_preprocess` 开关、`DataArguments` 加 `use_vggt_image_preprocess`、
  `train.sh` 加 `EPOCHS/SAVE_STEPS/MAX_PIXELS/MIN_PIXELS/USE_VGGT_IMAGE_PREPROCESS`。
- 评测：`scripts/mindcube/run_mindcube_eval.sh`（本地 tinybench 1050 条、8 卡分片、
  greedy 2048、判分复用 `lmms_eval` 的 `mindcube_process_results`）。
  **不走 lmms-eval 的 `mindcube_tiny`**——它指向未缓存的 HF 数据集（LESSON-003）。
- 当前：基座锚点评测中，driver PID **3179821**，产物
  `logs/eval/20260906_qwen35_mindcube_sft/baseline/`。两条 arm 共用这一个 step 0 锚点。
- 下一步：锚点出分 → 两条 arm 各冒烟 2 步（**硬门禁**：训练侧每图视觉 token 必须是 300、
  CoT arm 无嵌套 `<think>`）→ node-A/node-B 并行开训。
- 最后更新：2026-09-06 23:20 UTC+8

### `20260906_sparbench_vllm_lastline`（**已完成**，两机 8 卡已释放；21:02）

- Registry：[`experiments/registry/20260906_sparbench_vllm_lastline.yaml`](registry/20260906_sparbench_vllm_lastline.yaml)
- 协议：lastline / 无 `\boxed{}` / vLLM TP=8 / greedy **2048**
- lastline：**39.72 / 42.16（+2.44）**；截断 5.3% / 2.5%
- 对照表 SPAR 节已改为只报 vLLM
- HTML：`experiments/viewers/sparbench_lastline_vllm_base_vs_spar3/index.html`
- 产物：`logs/eval/20260906_sparbench_vllm_lastline/`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-06 21:05 UTC+8

### `20260906_vsibench_vllm_tok2048`（**已完成**，node-B 8 卡已释放；06:08）

- Registry：[`experiments/registry/20260906_vsibench_vllm_tok2048.yaml`](registry/20260906_vsibench_vllm_tok2048.yaml)
- 8 臂 vLLM `max_tokens=2048` 全部 rc=0（02:56–06:08）
- greedy：original **52.81 / 46.73（−6.08）**；boxed **47.31 / 49.93（+2.62）**
- sample：original 48.32 / 47.45；boxed 47.51 / 46.90。不可与 greedy 并列，也不可与 1024/4096 旧分并列
- 对照表 VSI 已改写
- 产物：`logs/eval/20260906_vsibench_vllm_tok2048/`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-06 18:50 UTC+8

### `20260905_cvbench_blink_sample`（**部分完成**，node-B 8 卡已释放；03:21 停在 BLINK）

- Registry：[`experiments/registry/20260905_cvbench_blink_sample.yaml`](registry/20260905_cvbench_blink_sample.yaml)
- CV boxed last-line @1024 sample 已写入对照表：基座 **87.36** / SPAR3 **85.68**（Δ −1.68）；不可与 greedy 88.07 / 85.05 并列
- BLINK 未出分：`Dataset.copy`（ISSUE-002）；CV original @16 sample 也未跑
- 驱动 PID **3605620** 已退出；产物 `logs/eval/20260905_cvbench_blink_sample/`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-05 22:35 UTC+8

### `20260904_vsibench_vllm_base_spar3_step55`（**已完成**，node-B 8 卡已释放；04:34）

- Registry：[`experiments/registry/20260904_vsibench_vllm_base_spar3_step55.yaml`](registry/20260904_vsibench_vllm_base_spar3_step55.yaml)
- 7 臂串行 vLLM 全部 rc=0（01:04–04:34）。跳过基座 original greedy（**52.47**）
- greedy：original SPAR3 **47.82**（vs 基座 −4.65）；boxed 基座 **48.42** / SPAR3 **49.61**（+1.19）
- sample（t=1.0 / top_p=0.8 / seed=20260904）：original 基座 **48.95** / SPAR3 **46.87**；boxed 基座 **47.84** / SPAR3 **47.31**。不可与 greedy 并列
- 驱动：`scripts/opsd/run_vsibench_vllm_base_spar3_chain.sh`（wrapper PID **3411747**）
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-04 04:40 UTC+8

### `20260903_qwen35base_vsibench_boxed_lastline_tok1024`（**已完成**，node-B 8 卡已释放；23:34）

- Registry：[`experiments/registry/20260903_qwen35base_vsibench_boxed_lastline_tok1024.yaml`](registry/20260903_qwen35base_vsibench_boxed_lastline_tok1024.yaml)
- 模型：`models/Qwen3.5-4B`
- 协议：boxed last-line / `spatialstack` · boxed-primary · greedy **1024** · 32 帧
- overall **46.20** / 作答 **93.82%**
- 相对同题面 greedy **4096 = 48.06** 为 **−1.86**（预算）；相对 original greedy 1024 = 52.75 不可直接减（题面+解析）
- 产物：`logs/eval/20260903_qwen35base_vsibench_boxed_lastline_tok1024/`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-04 00:40 UTC+8

### `20260903_qwen35base_vsibench_plain_vllm`（**已完成**，node-B 8 卡已释放；18:47）

- Registry：[`experiments/registry/20260903_qwen35base_vsibench_plain_vllm.yaml`](registry/20260903_qwen35base_vsibench_plain_vllm.yaml)
- 模型：`models/Qwen3.5-4B`
- 协议：original / `spatialstack_plain` · vLLM greedy 1024 · 32 帧
- overall **52.47** / 作答 **98.50%**（17.4 min，4.92 q/s）
- 相对同协议 HF **52.75** 为 **−0.28**；相对历史 vLLM 52.58 为 −0.11
- **不可与** boxed last-line 48.06 并列
- 第一次失败：系统 python 无 `decord`（ISSUE-001）；sr_opsd 重跑成功
- 产物：`logs/eval/20260903_qwen35base_vsibench_plain_vllm/summary.json`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-03 21:10 UTC+8

### `20260903_spar3_k1_step55_vsibench_plain`（**已完成**，node-B 8 卡已释放；13:48）

- Registry：[`experiments/registry/20260903_spar3_k1_step55_vsibench_plain.yaml`](registry/20260903_spar3_k1_step55_vsibench_plain.yaml)
- 模型：`output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf`
- 协议：与基座相同 `spatialstack_plain` · greedy 1024
- overall **47.87** / 作答 **95.58%**（生成约 1h55）
- 相对同协议基座 **52.75** 为 **−4.88**；不可与 boxed last-line SPAR3 **49.84** 并列
- 产物：`logs/eval/20260903_spar3_k1_step55_vsibench_plain/`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-03 13:55 UTC+8

### `20260903_qwen35base_vsibench_plain_rescore_boxed`（**已完成**，不占卡；11:44）

- Registry：[`experiments/registry/20260903_qwen35base_vsibench_plain_rescore_boxed.yaml`](registry/20260903_qwen35base_vsibench_plain_rescore_boxed.yaml)
- 源：20260902 基座 original prompt dump（**52.75**）
- 规则：boxed last-line 解析（闭合框优先，否则末行）；**不重新生成**
- overall **52.75** / 作答 98.13%；闭合框 **0/5130**；0 行改判；box-only **0.00**
- **不能**当成 48.06
- 产物：`logs/eval/20260902_qwen35base_vsibench_plain/rescore_boxed_lastline/`
- 最后更新：2026-09-03 11:45 UTC+8

### `20260902_qwen35base_vsibench_plain`（**已完成**，node-B 8 卡已释放；23:32）

- Registry：[`experiments/registry/20260902_qwen35base_vsibench_plain.yaml`](registry/20260902_qwen35base_vsibench_plain.yaml)
- 模型：`models/Qwen3.5-4B`
- 协议：`spatialstack_plain`（原始无 boxed 题面）· greedy 1024 · `until=[]`
- 5130 题；题面无 `\boxed{}` 后缀
- overall **52.75** / 作答 **98.13%**（生成约 1h18）
- 与 2026-08-20 dump 的现行 `answer_tail` 重打分 **逐题型一致**；相对源落盘 53.40 为 −0.65
- **不可与** boxed last-line 48.06 并列
- 样本 HTML：[`experiments/viewers/vsibench_qwen35base_plain/index.html`](viewers/vsibench_qwen35base_plain/index.html)（完整 original prompt + 回复 + 解析）
- 产物：`logs/eval/20260902_qwen35base_vsibench_plain/vsibench/models__Qwen3.5-4B/20260902_221421_results.json`
- 窗口：cursor-window-mvopsd
- 最后更新：2026-09-02 23:33 UTC+8

### `20260902_3dthinker_s1_s2`（**Stage2 checkpoint 已清空**，待重训；Stage1 权重保留）

- **2026-09-06 23:09**：按用户要求清空 Stage2 RL 保存权重（约 15G）：删除 `checkpoint-100`…`checkpoint-600`、`gen8_smoke`、`gen8_1gpu`。**Stage1 基座未动**：`models/3DThinker-S1-Qwen2.5-VL-3B_mlp6_lr1e-4_latent12`（27G）与 `..._vllm`（2.4G）仍在。
- 旧训练日志 / wandb / `logs/.../eval_s2/` 评测产物仍保留作对照；仅 checkpoint 树已空。
- 根因见 registry **ISSUE-019**（`sim_reward` reward hacking）；重训前需先修 reward 或确认策略。
- Registry：[`experiments/registry/20260902_3dthinker_s1_s2.yaml`](registry/20260902_3dthinker_s1_s2.yaml)
- 入口：`3DThinker/run_stage2.sh`（`3DThinker-stage2-vllm`）；单卡论文口径：`NPROC=1 PER_DEV_BS=8 GRAD_ACCUM=1 CUDA_VISIBLE_DEVICES=0`
- 窗口：cursor-window-3dthinker
- 最后更新：2026-09-06 23:09 UTC+8

node-B 的环境是从 node-A 直接 rsync 过去的：训练环境为系统 python3.12 +
`~/.local`（11G），评测环境为 conda `sr_opsd`（18G），`verl` 的 editable 安装
靠两机项目路径一致而原地生效。node-B 未传 `checkpoints/`（各自冷启动，
互不 resume）。

### `20260902_qwen35base_vsibench_rescore_boxed`（**已完成**，不占卡；22:03）

- Registry：[`experiments/registry/20260902_qwen35base_vsibench_rescore_boxed.yaml`](registry/20260902_qwen35base_vsibench_rescore_boxed.yaml)
- 源：同一份 2026-08-20 基座无 boxed dump
- 规则：`boxed_primary=1`（有闭合框只认框内，否则末行 answer_tail）
- overall **52.75**，与 `rescore_answer_tail_nobox` 相同；闭合框 **0/5130**，box-only **0.00**
- **不能**当成 48.06：那是重新生成，不是换解析
- 产物：`logs/eval/20260820_qwen35base_vsi_anchor/rescore_boxed_primary/`
- 最后更新：2026-09-02 22:04 UTC+8

### `20260902_qwen35base_vsibench_rescore_plain`（**已完成**，不占卡；21:55）

- Registry：[`experiments/registry/20260902_qwen35base_vsibench_rescore_plain.yaml`](registry/20260902_qwen35base_vsibench_rescore_plain.yaml)
- 源：2026-08-20 基座无 boxed dump（当时 **53.40**）
- 现行 `answer_tail`、`boxed_primary=0`，不重新生成
- 重打分 overall **52.75**（作答 98.13%，118 行改判，Δ **−0.65**）
- 跌分集中在 `obj_appearance_order`（末行抽取收紧）；**不可与 boxed last-line 48.06 并列**
- 未改 `vsibench_doc_to_text`，未覆盖 `logs/eval/20260901_qwen35base_vsibench_boxed_lastline/`
- 产物：`logs/eval/20260820_qwen35base_vsi_anchor/rescore_answer_tail_nobox/`
- 最后更新：2026-09-02 21:56 UTC+8

### `20260902_sparbench_tok2048`（**已完成**，两机 8 卡已释放；19:03）

- Registry：[`experiments/registry/20260902_sparbench_tok2048.yaml`](registry/20260902_sparbench_tok2048.yaml)
- 协议：SPAR-Bench greedy · **max_new_tokens=2048** · `until=[]` · 原版 MCA/NA/VCI 判分
- 基座 **39.88**；SPAR3 K=1 step 55 **13.87**（−26.01）
- 不可与 20260901 的 6.63（100 tok + `until=\n\n`）并列
- 最后更新：2026-09-02 21:09 UTC+8

### `20260902_cvbench_boxed_lastline_16tok`（**已完成**，8 卡已释放；14:48）

- Registry：[`experiments/registry/20260902_cvbench_boxed_lastline_16tok.yaml`](registry/20260902_cvbench_boxed_lastline_16tok.yaml)
- 协议：boxed last-line 题面+解析，**预算 16**
- 基座 combined **22.27**（2D 11.37 / 3D 33.17 / 作答 26.35%）
- SPAR3 step 55 combined **0.35**（作答 0.57%）
- 对照：同一协议 1024 → 88.07 / 85.05；lmms_legacy 16 → 62.85 / 5.55
- 最后更新：2026-09-02 14:49 UTC+8

### `20260902_cvbench_lmms_legacy`（**已完成**，8 卡已释放；12:57）

- Registry：[`experiments/registry/20260902_cvbench_lmms_legacy.yaml`](registry/20260902_cvbench_lmms_legacy.yaml)
- 协议：`CVBENCH_PROTOCOL=lmms_legacy`（上游 16 tok + 首 [A-F]）
- 基座 combined **62.85**（2D 67.28 / 3D 58.42 / 作答 75.06%）
- SPAR3 K=1 step 55 combined **5.55**（2D 11.10 / 3D 0.00 / 作答 23.81%）
- **不可与 boxed last-line 88.07 / 85.05 并列**；SPAR3 低分是 16-token 截断 CoT
- 最后更新：2026-09-02 12:58 UTC+8

### `20260902_qwen35base_blink_spatial`（**已完成**，8 卡已释放；12:42）

- Registry：[`experiments/registry/20260902_qwen35base_blink_spatial.yaml`](registry/20260902_qwen35base_blink_spatial.yaml)
- 模型：`models/Qwen3.5-4B`
- **original 宏平均 75.30**（55.64 / 81.45 / 88.81）
- **spatialstack 宏平均 65.99**（36.09 / 81.45 / 80.42）
- 对照 SPAR3 K=1 step 55：original 13.05（格式崩）、spatialstack **68.92（+2.93）**
- 两协议不可并列
- 最后更新：2026-09-02 12:42 UTC+8

### `20260902_spar3_k1_step55_blink_spatialstack`（**已完成**，8 卡已释放；12:28）

- Registry：[`experiments/registry/20260902_spar3_k1_step55_blink_spatialstack.yaml`](registry/20260902_spar3_k1_step55_blink_spatialstack.yaml)
- 模型：同上 step 55 HF
- 协议：`BLINK_PROTOCOL=spatialstack`（VSI MCA boxed last-line）；greedy 1024；`until=[]`；`disable_thinking`
- **宏平均 68.92**：multi-view **45.11**、relative depth **79.84**、spatial relation **81.82**
- **不可与 original 13.05 并列**；dump 目录分开
- 最后更新：2026-09-02 12:28 UTC+8

### `20260902_spar3_k1_step55_blink_spatial`（**已完成**，original 协议保留；11:48）

- Registry：[`experiments/registry/20260902_spar3_k1_step55_blink_spatial.yaml`](registry/20260902_spar3_k1_step55_blink_spatial.yaml)
- 模型：`output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf`
- 协议：lmms-eval `blink_spatial` val 宏平均；greedy 1024；`disable_thinking`；**句首字母**；无 boxed
- **宏平均 13.05**：multi-view **0.00**（133）、relative depth **0.00**（124）、spatial relation **39.16**（143）
- 读数警告：0 分来自句首 `To determine...` 被抽成 T；39% 来自句首 `Based...` 被抽成 B。不是空间能力。
- 样本 HTML：[`experiments/viewers/blink_spar3_k1_step55/index.html`](viewers/blink_spar3_k1_step55/index.html)（完整 user prompt + chat 组装 + 回复）
- 最后更新：2026-09-02 12:02 UTC+8

### `20260901_mvopsd_spar3_k1_200steps`（**已停止**：用户 11:24 要求停训）

- Registry：[`experiments/registry/20260901_mvopsd_spar3_k1_200steps.yaml`](registry/20260901_mvopsd_spar3_k1_200steps.yaml)
- 00:33 评测结束并自动开训；**05:20 在 63/200 因 `total_epochs=1` 提前退出**（ISSUE-001 / LESSON-032）
- 第一段 val（boxed_lastline+sample）：step 0 **47.80%** → 10 47.38 → 20 47.53 → 30 47.48 → 40 46.89 → 50 47.72 → 60 **47.59%**
- 续跑：`resume_mode=auto` 从 `global_step_60`，`trainer.total_epochs=4`，`val_before_train=False`
- **停训**：2026-09-02 11:24 对 PGID **2681194** 发 SIGTERM；当时正在 **step 150 验证中途**。8 卡已空闲（显存 0）。
- 落盘 ckpt：**已删** `checkpoints/20260901_mvopsd_spar3_k1_200steps/`（用户 11:29）。train/val/rollouts 日志仍在。
- wandb 第一段：https://wandb.ai/slcheng/MV-OPSD/runs/hisj4h8j
- 日志：`logs/train/20260901_mvopsd_spar3_k1_200steps/train_20260902_052432.log`
- 同批清理：`20260831` FSDP 只留 `global_step_55`（训练内最佳 49.23%）。
- 最后更新：2026-09-02 11:31 UTC+8

### `20260901_vsibench_boxed_lastline`（**已完成**）

- Registry：[`experiments/registry/20260901_vsibench_boxed_lastline.yaml`](registry/20260901_vsibench_boxed_lastline.yaml)
- 协议：lmms-eval `spatialstack` = 训练内 boxed last-line 题面 + boxed_primary；greedy 4096；`disable_thinking`
- 臂：
  - 基座：**48.06** / 作答 95.75%
  - SPAR3 K=1 step55：**49.84**（+1.78）/ 作答 98.25%
  - SPAR3 K=2 step15：**49.23**（+1.17）/ 作答 95.69%
- 不可与训练内 sample ~46.5%、旧无后缀 dump、vLLM 52.58 并列
- 最后更新：2026-09-02 05:22 UTC+8

### `20260901_qwen35base_cvbench_boxed_lastline`（**已完成**，8 卡已释放；15:19）

- Registry：[`experiments/registry/20260901_qwen35base_cvbench_boxed_lastline.yaml`](registry/20260901_qwen35base_cvbench_boxed_lastline.yaml)
- 模型：`models/Qwen3.5-4B` 发布权重，零样本
- 协议：spatialstack / boxed_lastline / greedy 1024
- **combined 88.07**（2D 83.55，3D 92.58，作答 99.62%）——**替代旧对外数字 85.95**
- 同口径：SPAR3 K=1 85.05（−3.02）、K=2 86.80（−1.27），均低于基座
- 最后更新：2026-09-01 16:10 UTC+8

## 最近收口的实验

### `20260820_teacher_reliability`（**已完成**，8 卡已释放）

- **报告**：[`experiments/reports/汇报_20260821_评估Teacher监督可靠性与视角收益.md`](reports/汇报_20260821_评估Teacher监督可靠性与视角收益.md)
- **结论**：teacher 在训练池上整体 **41.4%**（124,306 条，作答率 97.6%），
  题型间可靠性差 40 倍：llava_hound 描述题 57.7%、spar_3view 距离题 57.0%，
  而 spar_32view 物-物空间想象题只有 **1.3%**。
- **视角效应（受控扫描，同题只改张数）：收益呈明显边际递减。**
  vlm3r：1→5 视角 33.0→42.1（+9.1），5/6/7/8 在 41.2-42.3 之间无趋势抖动。
  spar32：1→32 视角 11.9→22.0（+10.0），约 16 张后趋缓（16/24/32 为 20.7/20.5/22.0）。
  8→32 视角的额外收益有限，与新增计算成本不成比例；且 32 视角下仍仅为 22%，
  表明高预算区间的主要瓶颈不在信息量。
- **空间想象题的近零分是「答得不全」不是「答得不对」**（ISSUE-T05）：
  方向题按「金标每条轴都要对」全匹配计分，而 teacher 只挑一条轴回答。
  `spatial_imagination_oo_video` 答全三轴的比例只有 2.9%，
  但**在它说出口的轴上准确率 68.2%**。这批 5,730 条现在不适合直接蒸馏——
  是格式问题，改 prompt 强制三轴齐出有很大空间。
- 三条下游建议：① 采用“8 视角基础预算 + 显式帧引用按需补充”
  （spar32 扫描仅覆盖帧无关题，不能把所有 32 视角样本无条件截断为 8）；
  ② `spatial_imagination_oo_*` 改 prompt 后复测，否则从蒸馏目标摘掉；
  ③ 按题型给蒸馏样本加权，最可靠三档共 44,229 条（55-58%）优先。
- 判分可信度：规则层独立判掉 62.9%；反向抽查分歧 as_trained 3.11% / vlm3r 0.51% /
  spar32 3.28%，**人工逐条复核 as_trained 全部 12 条分歧后确认规则层对、judge 错**
  （judge 在抽取模式下会自己答题或补全没说的轴），故该数字是上界而非规则层错误率。
  语义 judge 校准：金标当回答喂回 100 条零误判，无关 caption 100 条仅 1 条误接受。
- 最后更新：2026-08-21 23:55 UTC+8（正式汇报按实验目的—设置—结果—分析重构）

<details>
<summary>过程记录（设计与踩坑，已归档）</summary>

- Registry：`experiments/registry/20260820_teacher_reliability.yaml`
- 动机：自蒸馏把 teacher 的 N 视角后验拷给 student，但训练回路从未检查过这个后验对不对
  —— `vopd_loss` 度量的是「student 与 teacher 一致」，不是「teacher 答对」（LESSON-012）。
  用 rollout8 训练时 teacher 的真实输入让冻结基座直接作答，按 teacher 视角数分组统计正确率。
- **视角数取值由数据决定，不是自选的**：逐行读 `main_train.parquet` 的
  `extra_info.n_views_teacher`，teacher 实际只见过 7 个取值：
  3（33,970，全是 spar_3view）/ 4（351）/ 5（1,531）/ 6（3,805）/ 7（1,429）/
  8（62,695）/ 32（20,525，全是 spar_32view），合计 130.2 万张视角实例。
- **N 与数据源几乎一一绑定**，所以「准确率 vs N」表读到的是数据源差异，
  不得表述为视角数效应。分离靠两条受控扫描（同题同相册，只改张数，均匀间隔取视角）：
  `vlm3r`（N=8，预算 1~8，2,500 题 × 8 = 20,000 行，选择题可规则判分）与
  `spar32`（N=32，预算 1/2/3/4/6/8/12/16/24/32，四个 frame-agnostic 题型各 500 题 = 20,000 行）。
  扫描池只用 `required_views` 为空的样本，否则截断相册会留悬空 `Frame-N`。
- **协议复刻的是训练而不是 benchmark**：无 system 轮（teacher 从来没有）、
  `enable_thinking=False`、图像按缓存原尺寸不设 min/max pixels、
  greedy、`max_tokens=1024`、`max_model_len=12288`。
- **冒烟已得的形态结论（600 条，非准确率结论）**：teacher **不直接作答**，
  中位输出 215 token；连 vlm3r 那种 prompt 明写「直接给选项字母」的选择题也先写长分析。
  同一基座在 VSI-Bench 上（带 system 轮）中位输出只有 4 token。这与 LESSON-020 里
  student 的退化形态同向，可能是形态漂移的源头，待全量数据坐实。
- 判分（用户 2026-08-20 决策）：SPAR 按题型写解析器 + gpt-oss-120b judge 只做**抽取**，
  llava_hound 描述题用语义 verdict 并附反向抽查（LESSON-019）。
  现有 `_mvopsd_score` 只覆盖 30.1%（37,471/124,306），llava_hound 覆盖 0。
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：node-A 8 卡（node-B 空闲）
- 活跃进程或作业 ID：driver PID 1623422，进度看 `logs/eval/teacher_reliability/status.txt`
- 三段串行：`as_trained`（124,306 条，约 4~5h）→ `sweep_vlm3r` → `sweep_spar32`，
  各自可断点续跑（按 `row_index` 去重，**不能**按 `sample_id`——有 661 条重名）。
- 判分链路已写完并冒烟通过（35 条测试）：`spar_scoring.py`（SPAR 金标覆盖 99.99%）、
  `teacher_scoring.py`（路由 + 聚合）、`tools/score_teacher_dump.py`（规则层，纯 CPU）、
  `tools/judge_teacher_dump.py`（judge 层，抽取模式 + 反向抽查 + 语义层校准探针）。
  judge 由 `run_teacher_judging.sh` 驱动，已用 `tools/chain_teacher_judging.sh`
  挂在生成之后自动接上（PID 1651228，盯 status.txt 的 ALL DONE，不与生成抢卡）。
- 启动命令：
  ```
  setsid nohup bash scripts/opsd/run_teacher_reliability.sh \
    > logs/eval/teacher_reliability/driver.log 2>&1
  setsid nohup bash scripts/opsd/tools/chain_teacher_judging.sh \
    > logs/eval/teacher_reliability/judging.log 2>&1
  ```
- 已修的坑：`sample_id` 在 plan 里**不唯一**（674 个 id 各指两道不同的题），
  续跑去重与扫描分组都不能用它——分别改为 `row_index` 与 `sample_id#plan_position`。
  判分侧修掉四个解析 bug（T02 句末数字、T03 金标读成物体编号、
  **T06 `Frame-4` 被读成负数 -4**、T07 金标侧另有一份宽松正则），
  四个都只会低估 teacher。见 LESSON-021 / LESSON-022 / LESSON-023。
- 实际用时：as_trained 3.06h（11.3 行/秒）、sweep_vlm3r 8.1min、sweep_spar32 16.7min、
  judge 层 20min。修 T06 后重跑了一遍完整判分链路。

</details>

## 下一轮训练标准（用户 2026-08-21 与老师讨论后决策，配置已改，尚未启动）

三项改动已落到默认配置里，**没有启动任何作业**，两机 8 卡仍空闲。
`experiment_id` 待定，启动前需在本文件与新 registry 里登记。

### 1. `rollout.n: 4 → 1`

- 改在 `mvopsd.yaml`（`actor_rollout_ref.rollout.n`）与 `run_mvopsd.sh` 的 `ROLLOUT_N` 默认值，
  SMOKE 分支也从 2 改成 1。
- 每步序列数 64×4=256 → 64×1=64，仍是「一步一个 mini-batch」，
  `ppo_mini_batch_size=64 = train_batch_size`，严格 on-policy 不变
  （`should_reuse_rollout_log_probs_as_old_log_probs` 仍成立）。
- **为什么安全**：`vopd_loss` 是逐样本把 student 与 teacher 在 student 自己的 token 上对齐，
  组内不做任何比较；`adv_estimator: grpo` 只用于跳过 critic，
  组大小为 1 时 `compute_grpo_outcome_advantage` 走 `mean=0/std=1` 分支返回原始分数，
  而 vopd 分支根本不读 advantage（LESSON-012）。不会 NaN，也不改变损失。
- **读数约束**：历史 run（v1 n=4、rollout8 n=8、view_ratio n=4）的曲线是在不同
  每步样本量下测的，**绝对值与收敛速度都不可与本轮并列**。

### 2. 训练内评测加报 validation rollout 的 response 长度均值（VSI-Bench）

- 此前训练内只有 `resp_chars`（解码字符数）。字符数与训练侧的
  `response_length/mean`（token 数）**不同单位、不能并排看**，而长度是重复退化最早的信号
  （LESSON-020）。
- 新增：`_validate()` 从生成结果的 attention mask 直接数 token，
  写进 `reward_extra_infos_dict["resp_tokens"]`，随 val dump 一起落盘（多一列）。
  指标：`val-core/vsibench/resp_tokens/mean`（主看）、
  `val-aux/vsibench/resp_tokens/{median,p95,max}`。
- **基座锚点上实测：mean 97.1 token 而 median 只有 3 token（差 32 倍）。**
  即长尾在 step 0 就存在（2% 撞 1024 上限），所以均值与中位数必须同时看：
  均值不动可能掩盖中位数翻倍，中位数不动可能掩盖长尾变粗。
- 改动文件：`ray_trainer.py`（`_response_token_lengths` + `_validate`）、
  `vsibench_metrics.py`（新增 `resp_tokens` 参数与四个指标）。
  回归测试 `test_vsibench_in_training.py` 已加断言，overall 仍为 52.57（对齐离线 52.58）。

### 3. teacher 开 thinking，student 关，评测也关

- 新增配置项 `actor_rollout_ref.actor.self_distillation.teacher_enable_thinking`
  （`null` = 继承 student，保持旧行为；`mvopsd.yaml` 里设 `True`）。
  `data.apply_chat_template_kwargs.enable_thinking` 保持 `false`——它同时管
  student rollout 与**全部训练内评测**（两者都走 `RLHFDataset` / agent loop），
  所以「student 关、测试关」不需要任何额外改动。
- Qwen3.5 模板的实际差别在生成提示词的结尾（已逐字符验证）：
  关 thinking → `<|im_start|>assistant\n<think>\n\n</think>\n\n`（推理块预先闭合）；
  开 thinking → `<|im_start|>assistant\n<think>\n`（推理块**保持打开**）。
  teacher prompt 因此比 student 短 2 个 token，相对 12,288 的上限可忽略。
- **必须记住的语义变化**：teacher 在本方法里**从不采样**，只对 student 已生成的 token
  做 teacher forcing。开 thinking 后 teacher 的上下文停在一个打开的 `<think>` 里，
  于是 student 被对齐的目标从「teacher 的作答态分布」变成
  「**teacher 的推理态分布**」。这是这个开关的作用本身，不是副作用。
  预期 `self_distillation/*` 的 JSD 平台会整体移位（前四条 arm 是 0.018~0.022），
  **移位本身不能当作方法有效或无效的证据**。
- 改动文件：`ray_trainer.py`（新增 `_teacher_apply_chat_template_kwargs()`，
  `_build_teacher_prompt_inputs` 与 answer_hint 路径都改用它）、
  `workers/config/actor.py` + `config/actor/actor.yaml`（新字段，默认 `null`）、
  `mvopsd.yaml`。
- 新回归测试 `scripts/opsd/tests/test_thinking_split.py`：校验三侧取值、
  `null` 确实继承 student、以及两侧渲染出的 generation prompt 只在推理块处不同。

### 一级指标诊断（2026-08-21，用户要求「查验一级指标计算是否正确」）

新增 `scripts/opsd/tests/test_val_metric_invariance.py`。它不校验「数看起来合理」，
而是拿 pipecheck 第二次（已修 ISSUE-203）的**真实训练内 val dump 5,130 行**
重放，检查**不变性**：

| 检查 | 结果 |
| --- | --- |
| 重放 dump 复现 trainer 自己打的 `val-core/vsibench/overall/acc` | `0.5362459737332027` **逐比特相同** |
| 塞一列极端 `resp_tokens`（含 1e9）后 overall / 每题型 / `answered/frac` | **逐比特不变** |
| `val-core` 键集合变化 | 只多 `resp_tokens/mean`，**没少任何键** |
| 手算「方向题合成 1 票、8 题型不加权平均」 | 与实现 `1e-12` 内一致，且 ≠ micro（55.23） |
| `_response_token_lengths` vs verl 自己的 `compute_response_mask` | 相同（含空回答与撞满预算两种边界） |
| 当前配置渲染的 student/评测 prompt vs dump 里被判分的那条 | **逐字节相同**，system 轮在、`<think>` 预闭合 |

**顺带确认了一个此前没记的事实：pipecheck 修好后的 step 0 = 53.62**
（`train_20260820_173529.log:6276`，四步依次 53.62 / 53.47 / 53.05 / 53.40，
`rule_only` 全程等于 `overall`，judge 因待判行数 1~3 < 16 从未唤醒）。
ACTIVE.md 此前写的预期是「对齐离线 52.58」，**实际高 1.04**，
更靠近离线 HF/lmms_eval 锚点 53.40。这是引擎/图像管线差异，不是判分差异
（判分侧本次已逐行对齐，见下）。**下一轮的基线锚点应记 53.62，不是 52.58。**

### 诊断查出的真实缺陷：MRA 阈值与上游差最后几位（ISSUE-301）

- 症状：同一批 5,130 行回答，训练内判分给 53.62，lmms_eval 给 53.64。
- 证据：逐行比对定位到 **8 行**分歧，全部是数值题且预测值恰好压在阈值上
  （1.1 vs 1.0、9 vs 10、1.7 vs 2.0、18 vs 20），且**分歧方向永远是我们更低**。
- 根因：上游 `src/lmms_eval/tasks/vsibench/utils.py:180-184` 是
  `np.linspace(0.5, 0.95, int((0.95-0.5)/0.05 + 2))`——`int()` 把 10.999999999999998
  截成 10，所以确实是 10 个阈值，但 numpy 生成的第 8 个阈值是 `0.8999999999999999`，
  `1 - 阈值 = 0.10000000000000009`；而我们写的 `0.5 + 0.05*i` 给 `0.9`，
  `1 - 阈值 = 0.09999999999999998`。相对误差「恰好 0.1」实际算出
  `0.10000000000000009`，于是**上游算命中、我们算未命中**。
- 解决：`mean_relative_accuracy` 收成全仓一份，放在
  `vsibench_scoring.py`，按上游同样的 linspace 构造阈值；
  `mvopsd_reward.py` 与 `spar_scoring.py` 都改为引用（LESSON-023 第 3 条：
  同一语义只允许一份定义）。修后 5,130 行**逐行 0 分歧**，overall 53.6391 双方相同；
  离线 vLLM 归档那条 parity 也从 52.57 vs 52.58 变成 **52.58 vs 52.58**。
- 影响面（已量化，不需要重跑任何已发布数字）：
  一位小数网格上仅 0.042% 的 (预测, 金标) 组合会变；
  teacher 可靠性 `as_trained` 124,306 行里 127 行变（0.10%），
  numeric family 均值 39.94 → 39.98，**总体 ≤0.01pp，41.4% 仍是 41.4%**；
  `sweep_spar32` 23/20,000 行变，同量级。**报告数字不改。**
- 状态：已修，已加断言（`test_val_metric_invariance.py` 用 6 个压阈值样本
  直接对比上游 `mean_relative_accuracy`，不再只看 5,130 行的平均）。
- 与本轮三项改动无关：dump 产生于 2026-08-20，缺陷比改动更早。

### 启动前仍需做的事

1. 定 `experiment_id` 与训练池（上一轮 `parquet_k_half` / `parquet_k_quarter` 是否沿用，
   以及 teacher 可靠性报告的三条建议是否采纳——按题型加权、
   `spatial_imagination_oo_*` 改 prompt 或摘除、8 视角基础预算）。
2. 用 `check_prompt_lengths.py --enable-thinking` 复核 teacher 列长度
   （该工具此前硬编码 `enable_thinking=False`，现已加开关）。
3. 跑一次 `SMOKE=1` 冒烟，确认 n=1 下 teacher forward 与新指标都正常落盘。

### 顺手修掉的两个已坏测试（与本次改动无关，属既有回归）

- `test_cvbench_val_parity.py`：`compute_score` 2026-08-20 起多返回 `resp_chars`，
  而测试用整字典相等断言，于是 4 项因「多了一个诊断列」而失败。改为只比
  `score/acc/answered`。
- `test_validation_judge.py`：stub 用 `SimpleNamespace`，缺 2026-08-20 新增的
  `_judge_vsibench_rows`，全套抛 `AttributeError`。改为把真实方法绑上去
  （CV-Bench 行会让 VSI 层直接返回，但它在调用链上，不该被 stub 掉）。

- 最后更新：2026-08-21 15:20 UTC+8（一级指标诊断完成，附带修掉 ISSUE-301）

### 解码开关改造（2026-08-23，工具链变更，未启动任何作业）

- 起因：用户要求参考上游 [OPSD](https://github.com/siyan-zhao/OPSD) 设置 `do_sample`。
  逐字核对上游两处：`eval/run_eval.sh` 只传 `--temperature 1.0 --val_n 12`，
  `eval/evaluate_math.py:648-660` 据此把 `top_p` 自动设为 0.95（thinking）/ 0.8
  （non-thinking），`top_k=-1`、`min_p=0`、`presence_penalty=0`，并在
  `temperature==0` 时**打印警告**：贪心在 thinking 模式下会造成
  "performance degradation and endless repetitions"。
  训练侧 `scripts/run_opsd_1b.sh` / `run_opsd_4b_nonthink.sh` 用
  `--temperature 1.1 --top_p 0.95 --top_k 20`。
- 我们的现状（改动前）：
  - 训练 rollout 已在采样（verl 默认 `temperature=1.0 / top_p=1 / top_k=-1 /
    do_sample=True`），但 `top_p`/`top_k` 比 OPSD 宽。
  - **训练内评测与全部离线评测都是贪心**（`mvopsd.yaml` 的
    `val_kwargs.do_sample=False`、`vsibench_eval_core.py` 的
    `temperature=0.0`）。
- 与 LESSON-020 的关系：我们记录的退化形态正是「重复枚举、刷满预算不给答案」
  （step 100 @4096 截断 90.62%），而**它是在贪心下测到的**。贪心是这种循环的
  已知成因之一，所以「退化」目前**无法与「贪心解码伪影」分离**。这是一个待验证
  假设，不是已确认结论。
- 改动（全部**默认不变、opt-in**，已发布数字一律不动）：
  - `vsibench_eval_core.py`：`EvalConfig` 新增
    `do_sample/temperature/top_p/top_k/min_p/presence_penalty/seed`，
    新函数 `resolve_decoding()` 统一解析；`summary.json` 新增 `decoding` 块，
    `format_report` 打印一行解码口径。
    **`do_sample=False` 时传任何采样旋钮直接抛错**，不静默忽略——
    避免跑出「半套协议」（同 `VSIBENCH_PROTOCOL` 的设计意图）。
  - `tools/eval_vsibench_vllm.py`：`--do-sample --temperature --top-p --top-k
    --min-p --presence-penalty --seed`。
  - `run_training_vsibench_eval.sh`：`DO_SAMPLE=1` +
    `TEMPERATURE/TOP_P/TOP_K/MIN_P/PRESENCE_PENALTY/SEED` 环境变量透传。
  - `run_mvopsd.sh`：`VAL_DO_SAMPLE=1` +
    `VAL_TEMPERATURE/VAL_TOP_P/VAL_TOP_K/VAL_N`，覆盖 `val_kwargs`。
    未开启时打印 `val decoding=greedy`。
  - 回归测试：`test_boxed_extraction.py` 新增第 7 组（默认贪心、
    `--do-sample` 单独给出即 t=1.0/top_p=0.8、显式旋钮优先、缺 `do_sample`
    抛错），全绿。
- **`--do-sample` 单独给出时的默认值**是 `temperature=1.0 / top_p=0.8`，
  即 Qwen3 **non-thinking** 指引——本仓库的 student 与全部评测都跑
  `enable_thinking=False`，故取 0.8 而非 thinking 的 0.95。
- **读数约束（必须遵守）**：采样是**另一条序列**，不是贪心曲线的延续。
  单次采样有逐题方差而贪心没有，OPSD 用的是 **Avg@12**；
  我们尚未实现 Avg@n（`val_kwargs.n` 可设，但离线侧仍只取 `outputs[0]`）。
  所以一次采样跑只能作为「退化形态是否为贪心伪影」的证据，
  **不能与 52.58 / 53.62 / 23.00 等贪心数字并列上报**。
- **用户 2026-08-23 决策**：① 立刻跑 step 100 的贪心 vs 采样对照（见下节）；
  ② 训练内评测**保持贪心默认**，需要时才用 `VAL_DO_SAMPLE=1`——曲线可比性优先。
- 仍待定：是否实现 Avg@n；训练 rollout 的 `top_p/top_k` 是否收紧到 OPSD 的 0.95/20。
- 顺带加固：`run_training_vsibench_eval.sh` 的 `OUT_DIR` 现在可被环境变量覆盖。
  它此前从 checkpoint 路径推导，**同一 step 跑第二次会就地覆盖第一次的
  `samples.jsonl` / `summary.json`**——改协议的复跑会毁掉本该用来对照的那份归档。
- 最后更新：2026-08-23 21:20 UTC+8

### 判分主口径改为严判 judge-extract（2026-08-23 用户决策）

- 决策依据：用户在双判分查看器里逐题核对分歧后判定 **judge 对**——
  规则层那些「满分」大多是蹭分，模型确实没答。
  典型 id=168：规则从 `Image N:` 枚举里抽到帧号 178 记满分。
- **注意：这与两条既有归档结论相反，必须并存不得覆盖**（LESSON 的保留原则）：
  ① teacher 可靠性那轮**人工复核 12 条分歧后判定规则层对、judge 错**
  （judge 在抽取模式下会自己答题或补全没说的轴）；
  ② 基座 VSI-Bench 上级联 judge 只 +0.08，当时结论是「规则解析器无需兜底」。
  两者与本次判定的差别在**对象不同**：那两条测的是**基座/teacher 的短回答**，
  本次测的是**退化后的长回答**。规则层在长重复文本上假阳性率高得多
  （id=168 那类帧号碰瓷只在长枚举里出现），所以「谁对」依赖于回答形态，
  不是一个全局常量。**跨形态套用任一结论都是错的。**
- 落地（离线路径，已完成）：
  - `run_training_vsibench_eval.sh` 新增 `JUDGE_MODE`，**默认 `extract`**
    （全行抽取，主口径）；`cascade` 保留旧行为，用于复现 2026-08-23 之前
    发布的一切数字。
  - `extract` 模式跑完会把主口径**盖章进根 `summary.json`**：新增
    `scoring_primary` / `primary`（指向 `judge_extract/`）与 `rule_diagnostic`
    （原 `rule_only` 原地保留，不删——它是已知上界，要留在主口径旁边可见）。
  - 新增通用驱动 `scripts/opsd/run_judge_extract.sh <samples.jsonl> [out_dir]`；
    原先路径写死的 `run_judge_extract_step100.sh` 改为薄封装，
    保证 ACTIVE.md 里记录的命令仍能复现，且实现只有一份。
- 落地（训练内路径，已完成；用户 2026-08-23 决策接受额外墙钟）：
  - 新增配置 `trainer.validation_judge.mode`，**默认 `extract`**；
    `cascade` 复现 2026-08-23 之前的曲线。
  - `ray_trainer.py`：`_judge_vsibench_rows` 按 mode 分流，新增
    `_judge_vsibench_strict()`——**全部** VSI 行送判，judge 的抽取结果
    **直接替换**规则分（**可以往下改**，这是这一层存在的理由；级联那层只能加分，
    修不了假阳性）。`val-core/vsibench/rule_only/acc` 仍然照报，作为诊断。
  - `mvopsd_judge.py`：新增 `ValidationJudge.extract_answers()`，
    **复用离线那份严判提示词与解析器**
    （`tools/judge_vsibench_tri_insurance.py` 的 `build_mca_prompt` /
    `build_na_prompt` / `parse_mca_reply` / `parse_na_reply`），
    懒加载，级联跑不付这个代价。不另写第三份提示词（LESSON-023）。
  - 新指标：`val-aux/vsibench/judge/{strict,graded,none,errors,raised,lowered,seconds}`。
    **`raised` / `lowered` 必看**——judge 大面积改写某一步的分数时要能看见，
    不能静默发生。
  - **judge 起不来或判分抛异常时会退回规则分**，此时会打
    `val-aux/vsibench/judge/fellback_to_rules=1`。该点位**与严判点位不可比**，
    读曲线时必须先看这个标志，否则会把一次判官故障读成模型跳变。
  - `min_rows` 只对级联生效：严判必须判满全部行，否则报出来的就不是严判口径。
- **代价**：每行一次 judge 调用。离线实测 5,130 行 12.4 / 13.0 min，
  叠在 ~18 min 生成之上，按 `TEST_FREQ=25` 算单次评测从 ~18 min 变成 ~31 min。
- 回归测试：`test_validation_judge.py` 新增两组严判用例（全行送判、
  假阳性被降分、raised/lowered 计数、唤不醒时打退回标志），
  并把 stub 补上 `_judge_vsibench_strict` 与 `_validation_judge_mode`。
  既有 9 组级联用例显式传 `mode="cascade"`，全套通过。
  `test_vsibench_in_training.py` 与 `test_val_metric_invariance.py` 复跑全绿。
- **写测试时查出的一个真实陷阱（已避开）**：MCA 金标格式两边不同——
  CV-Bench 是 `(B)`，**VSI-Bench 是裸字母 `B`**
  （`data/eval/vsibench_verl/vsibench_val.parquet` 实测）。
  严判里的比较写成 `letter == gt.strip().upper()`，与规则层
  （`mvopsd_reward.py:153`）**同一种归一化**。若照 CV-Bench 格式比，
  全部 MCA 行会静默记 0 分而看起来像模型崩了。
- 最后更新：2026-08-23 23:55 UTC+8

### `20260823_decoding_ab_step100`（**已完成**，8 卡已释放；21:20 起，56.9 min）

- 目的：把「训练导致的退化」与「贪心解码伪影」分开。同一 checkpoint、同 4096
  预算、同 32 帧、同判分链，**只改解码方式**。
- 对照组（贪心）**不重跑**，直接用已归档的
  `logs/vsi_train_eval/20260821_mvopsd_single_llava_hound_main/global_step_100/`：
  rule-only 23.00（修 `Image N:` 后重评 22.90）、作答率 97.45%、
  **截断率 90.62%**、中位输出 4096。
- 实验组（采样）：`DO_SAMPLE=1 SEED=20260823`，即 `temperature=1.0 / top_p=0.8 /
  top_k=-1 / min_p=0 / presence_penalty=0`。`JUDGE=0`（只比规则层，
  与 22.90/23.00 同口径；贪心侧 judge 只加 +0.07，不影响判读）。
- 窗口：`cursor-window-mvopsd`；产物 `logs/eval/decoding_ab/step100_sample/`
  （驱动日志 `driver.log`，生成日志 `eval.log`）
- 启动命令：
  ```
  OUT_DIR=logs/eval/decoding_ab/step100_sample \
  DO_SAMPLE=1 SEED=20260823 \
  FRAMES=32 MAX_TOKENS=4096 MAX_MODEL_LEN=16384 SCENES_PER_BATCH=24 JUDGE=0 \
  setsid nohup bash scripts/opsd/run_training_vsibench_eval.sh \
    checkpoints/20260821_mvopsd_single_llava_hound_main/global_step_100
  ```
- 预计墙钟：贪心那次 54.2 min；采样若真能收住长度会更快。
- **主看指标不是 overall，是形态**：截断率（90.62% → ?）、中位输出 token
  （4096 → ?）、作答率、以及 `Image N:` 枚举型回答是否消失。
  overall 只是附带——单次采样有逐题方差，贪心没有，两个数**不构成同一估计量**
  （OPSD 用 Avg@12）。
- **结果（rule-only，同口径）**：

  | 指标 | 贪心 | 采样 | Δ |
  | --- | --- | --- | --- |
  | overall | 23.00 | **35.97** | **+12.97** |
  | 作答率 % | 97.45 | 99.75 | +2.30 |
  | 截断率 % | 90.62 | **75.09** | **−15.54** |
  | 中位输出 token | 4096 | 4096 | 0 |
  | `>=10` 个 `Image N:` 的行 | 32.0% | 31.8% | −0.2 |

  题型层面几乎全线上移，最大 `object_size_estimation` +24.89、
  `obj_appearance_order` +22.33、`object_counting` +19.45；
  唯一下降是 `object_rel_direction_hard` −4.29。

- **关键拆解（这条决定怎么解读）**：按是否截断分组后，

  | 组 | 贪心行数 / 均分 | 采样行数 / 均分 |
  | --- | --- | --- |
  | 正常结束 | 481 / **64.49** | 1,278 / **62.69** |
  | 撞满 4096 | 4,649 / 18.88 | 3,852 / 28.09 |

  **正常结束的行两侧同分（64.49 vs 62.69）——采样没让模型变聪明。**
  +12.97 全部来自「能正常收尾的行从 9.4% 变成 24.9%」，加上截断行里
  答案句更常出现在被截断之前（18.88 → 28.09）。

- **结论（收紧后的表述）**：
  1. 长度爆炸**没有被修好**：中位输出仍撞满 4096，`Image N:` 枚举比例
     几乎一字不变（32.0% → 31.8%）。退化是模型本身的，不是解码伪影。
  2. 但**贪心显著放大了它的观测后果**：同一权重、同一判分链，只换解码方式
     就差 12.97 pp。此前所有「退化后分数」都是在这个放大镜下读的。
  3. 因此 `23.00` 应表述为「贪心口径下的分数」，不是这个 checkpoint 的能力上界。
- **必须补的对照**：我们**没有基座的采样分**。
  「基座 52.58（贪心）vs step 100 35.97（采样）」**不是合法比较**，
  跌幅 −16.6 里混着解码差异。要重述 LESSON-020 的「崩塌约 19 分」，
  必须先在 `./models/Qwen3.5-4B` 上用同样的 `DO_SAMPLE=1 SEED=20260823` 跑一次。
- **方差未测**：单次采样、单 seed，无误差棒。题型层面 ±4 的差异不足以判读
  （`object_rel_direction_hard` 的 −4.29 尤其不可当结论）；overall 的 +12.97
  远超任何合理噪声，这一条是稳的。
- 产物：`logs/eval/decoding_ab/step100_sample/{summary.json,samples.jsonl,eval.log}`
  （`summary.json` 的 `decoding` 块记录了实际生效的解码参数）
- 例证 id=168（sofa 金标 173cm）：贪心撞满 4096、通篇 `Image N: Kitchen.`、
  解析器兜底抽到帧号 178 记满分（假阳性）；采样只用 868 token 正常收尾，
  `<think>` 后给出 `195`，MRA 0.8。**同一题从「假阳性满分」变成「真部分分」。**

#### 补判：采样侧的严判分（2026-08-23 22:54–23:08，13.0 min，judge 用 4 卡）

用户 22:35 决定**判分主口径改为严判 judge-extract**，故给采样那一跑补判，
使两侧在同一口径下可比。产物
`logs/eval/decoding_ab/step100_sample/judge_extract/`，API 错误 0、unreadable 0。

| 解码 | **judge（主口径）** | judge 作答率 | rule（诊断） | 截断率 |
| --- | --- | --- | --- | --- |
| 贪心 | 7.37 | 12.73% | 23.00 | 90.62% |
| 采样 | **21.53** | 37.00% | 35.97 | 75.09% |
| Δ | **+14.15** | +24.27 | +12.97 | −15.54 |

**两套判分口径给出同向且同量级的结论**（judge +14.15 / rule +12.97），
所以「贪心低估了这个 checkpoint」不是解析器方言造成的假象。
judge 口径下作答率从 12.73% 涨到 37.00%，即采样让**三倍**的行真正给出了答案。

题型层面（judge 口径）最大 `object_size_estimation` +26.13、
`obj_appearance_order` +21.52、`object_counting` +19.26、`object_rel_distance` +19.15；
`object_rel_direction_hard`（+0.54）与 `room_size_estimation`（+2.08）几乎不动
——这两个题型在两套口径下都最不受解码影响。

## 进行中的实验

### `20260831_mvopsd_spar3_cvbench_boxed_lastline`（**已完成**；基座锚点已替换）

- 协议：`CVBENCH_PROTOCOL=spatialstack`，`CVBENCH_PARSER=boxed_lastline`；MCA 句与 `\boxed{}` 后缀同一行；greedy 1024 tok。
- **基座（发布权重，同协议）combined 88.07**，替代旧离线默认 85.95（word_boundary）。见 [`20260901_qwen35base_cvbench_boxed_lastline.yaml`](registry/20260901_qwen35base_cvbench_boxed_lastline.yaml)。
- SPAR3 vs 基座（同口径）：K=1 **85.05（−3.02）**，K=2 **86.80（−1.27）**，均低于基座。
- 旧 dump 84.82 / 84.84 与基座 85.95 一律不作对外数字。
- 样本 HTML：[`experiments/viewers/cvbench_spar3_boxed_lastline/index.html`](viewers/cvbench_spar3_boxed_lastline/index.html)
- 最后更新：2026-09-01 16:10 UTC+8

### `20260831_mvopsd_spar3_best_generalization_eval`（**VSI 已中止**；cvbench/blink/sparbench 旧协议产物保留）

- Registry：[`experiments/registry/20260831_mvopsd_spar3_student_k.yaml`](registry/20260831_mvopsd_spar3_student_k.yaml)（`generalization_eval` 段）
- 窗口：`cursor-window-mvopsd`
- 目标：对 SPAR3 K=1/K=2 训练最佳 checkpoint 跑 lmms-eval 离线泛化（cvbench、blink_spatial、sparbench、vsibench）。
- 臂与 checkpoint：
  - **k1 best** node-A：`global_step_55`（训练 val 49.23%）→ merge → `output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf`
  - **k2 best** node-B：`global_step_15`（训练 val 48.02%）→ merge → `output/20260831_mvopsd_spar3_k2_epoch1_nothink_global_step_15_hf`
- 输出：`logs/eval/20260831_mvopsd_spar3_k{1,2}_best_generalization/`
- 协议：`MODEL_IMPL=qwen3_5`、`disable_thinking=true`、`max_num_frames=32`；无基座对照。
- 下一步：merge → smoke LIMIT=10 → 全量 4 benchmark。
- 最后更新：2026-09-01 11:20 UTC+8

### `20260831_mvopsd_spar3_student_k`（**训练已完成**，8 卡已释放；8/31 15:27 – 9/1 00:28，约 9 h）

- Registry：[`experiments/registry/20260831_mvopsd_spar3_student_k.yaml`](registry/20260831_mvopsd_spar3_student_k.yaml)
- 窗口：`cursor-window-mvopsd`
- 老师 2026-08-31 建议：只用 SPAR 3-view，teacher=3，student 分别为 1 / 2 视角。
- **同题嵌套**：K=1 合法的只有 `required_views=1` 的 4,380 条（另 30,090 条点名两帧，K=1 会丢被点名的帧）。两臂都用这 4,380 条，K=1 ⊂ K=2，同一 train/holdout。
- 数据：train **4,064** / holdout 316；1 epoch = **63** 步。
- 配置：与 SPAR 全量相同（nothink、boxed_lastline、val sampling、JUDGE=0）；`TEST_FREQ=SAVE_FREQ=5`。
- 臂：
  - **k1** node-A：student 恒 1 张，N/K=3.0，`parquet_spar3_k1`
  - **k2** node-B：student 恒 2 张，N/K=1.5，`parquet_spar3_k2`
- 训练内最佳（boxed_lastline+sample，JUDGE=0）：
  - k1 step 0 **46.27%** → best step 55 **49.23%**（+2.96 pp）→ step 63 **48.46%**
  - k2 step 0 **47.65%** → best step 15 **48.02%**（+0.37 pp）→ step 63 **46.08%**
- wandb：k1 https://wandb.ai/slcheng/MV-OPSD/runs/uqawyz9q ；k2 https://wandb.ai/slcheng/MV-OPSD/runs/e4u79rpi
- 日志：`logs/train/20260831_mvopsd_spar3_k1_epoch1_nothink/train_20260831_152721.log`；node-B `train_20260831_152720.log`
- 下一步：见 `20260831_mvopsd_spar3_best_generalization_eval`。
- FSDP（2026-09-02）：node-A 只留 `checkpoints/20260831_mvopsd_spar3_k1_epoch1_nothink/global_step_55`；HF merge 仍在 `output/..._global_step_55_hf`。
- **FSDP（2026-09-07，node-B）**：按用户要求整树删除 `checkpoints/20260831_mvopsd_spar3_k2_epoch1_nothink/`（691G，13 个 step）。HF merge `output/..._k2..._global_step_15_hf`（9.7G）未动。
- 最后更新：2026-09-07 23:43 UTC+8

### `20260829_mvopsd_spar_all_epoch1_nothink`（**已完成**，851/851，8 卡已释放；8/29 22:11 – 8/31 02:07，27.2 h）

- Registry：[`experiments/registry/20260829_mvopsd_spar_all_epoch1_nothink.yaml`](registry/20260829_mvopsd_spar_all_epoch1_nothink.yaml)
- wandb：https://wandb.ai/slcheng/MV-OPSD/runs/pf62nv5d
- 日志：`logs/train/20260829_mvopsd_spar_all_epoch1_nothink/train_20260829_221153.log`
- val dump：`logs/val/20260829_mvopsd_spar_all_epoch1_nothink/`（0, 25, …, 850, 851 共 36 份）
- 保留 ckpt：`checkpoints/20260829_mvopsd_spar_all_epoch1_nothink/global_step_200`（其余已按盘预算清掉；`latest` 记 851）
- 同口径读数（boxed_lastline + sample，JUDGE=0）：step 0 **46.71%** → 末次验证 **46.33%**。完整逐步曲线在 wandb，待写入 registry。
- 最后更新：2026-08-31 15:25 UTC+8

### `20260829_mvopsd_spar32_epoch1_nothink`（**已取消**，改跑 SPAR 全量；约 step 100 停止）

- Registry：[`experiments/registry/20260829_mvopsd_spar32_epoch1_nothink.yaml`](registry/20260829_mvopsd_spar32_epoch1_nothink.yaml)
- 仅 32-view 且摘了 oo_*；ckpt 留 `global_step_{25,50,75,100}`。
- 最后更新：2026-08-29 22:10 UTC+8

### `20260829_mvopsd_llava_hound_epoch1_nothink`（**已停止**，用户 2026-08-29 17:33 要求停跑；8 卡已释放）

- Registry：[`experiments/registry/20260829_mvopsd_llava_hound_epoch1_nothink.yaml`](registry/20260829_mvopsd_llava_hound_epoch1_nothink.yaml)
- 窗口：`cursor-window-mvopsd`；driver PID 3435888（已退出）
- 日志：`logs/train/20260829_mvopsd_llava_hound_epoch1_nothink/train_20260829_013729.log`
- 停止点：step **449** 训练完成，step **450** 验证中途被 SIGTERM；无 step 450 dump/ckpt。
- 权重：**只保留** `checkpoints/20260829_mvopsd_llava_hound_epoch1_nothink/global_step_200`（54G）；其余 16 个 ckpt 已删（约 850G）。
- val dump 仍在 `logs/val/.../`（含 0–425）；训练日志未动。
- 训练内 overall（boxed_lastline + sample，JUDGE=0）：step 0 **47.29%** → step 200 **47.77%** → step 425 **46.50%**。
- 最后更新：2026-08-29 17:43 UTC+8

### `20260828_thinking_150_250_lastline_eval`（**已完成**，8 卡已释放；16:32–21:22，4.83 h）

- Registry：[`experiments/registry/20260828_thinking_150_250_lastline_eval.yaml`](registry/20260828_thinking_150_250_lastline_eval.yaml)
- 协议：boxed_lastline + boxed_primary + **sample@4096**（t=1.0/top_p=0.8）+ JUDGE=0；5/5 步协议核对通过
- 锚点（同口径基座）：**46.51%**（作答 96.57%，截断 ~1.7%，boxed ~94%）

| step | overall | 作答率 | boxed 合规 | 截断率 |
| --- | --- | --- | --- | --- |
| 150 | 19.83 | 48.32 | 27.78 | 72.0 |
| 175 | 20.36 | 48.01 | 28.11 | 70.9 |
| 200 | 19.95 | 47.78 | 24.54 | 73.8 |
| 225 | 20.03 | 48.64 | 26.96 | 71.9 |
| 250 | **21.32** | **51.60** | 30.43 | 67.9 |

- **结论**：150–250 曲线平坦（19.8–21.3%），相对基座崩塌约 **25–27 pp**；作答率 ~48%、截断 ~70%、boxed 合规 ~27%。无训练正收益信号，退化形态贯穿全程。250 略高但仍在单 seed 噪声内。
- 产物：`logs/vsi_train_eval/20260827_mvopsd_vlm3r_epoch1_thinking/global_step_<N>_tok4096_sample_boxed_lastline_rule/`
- 最后更新：2026-08-28 22:50 UTC+8

### `20260828_step0_tok4096_lastline_rule`（**生成已完成**；**新规则重打分已完成**，不占卡）

- Registry：`experiments/registry/20260828_step0_tok4096_lastline_rule.yaml`
- 模型：`models/Qwen3.5-4B`；boxed_lastline + boxed_primary + sample t=1.0/top_p=0.8 + 4096 + JUDGE=0
- 生成口径 **47.05%**（作答 97.88%）。**当前规则层重打分 46.51%**（作答 96.57%，69 行变了，Δ −0.54）。框内-only 仍 44.88%。
- 重打分产物：`logs/vsi_train_eval/Qwen3.5-4B/step0_tok4096_sample_boxed_lastline_rule/rescore_parser_v2/`
- Viewer 已换成重打分 `acc`：http://localhost:8765/experiments/viewers/step0_tok4096_sample_boxed_lastline_rule/
- 未重新生成。与 in-loop 49.07 不可并列。
- 最后更新：2026-08-28 15:40

### val prompt：boxed + 最后一行（2026-08-28，**parquet 已建成**，未改默认 VAL_FILE）

- 规则层已改为只信「answer is … / would be … / I choose … / so C」或**看起来像作答的最后一行**（单独 `C` / `C. left` / 唯一选项正文 / `(C)`）。不再从 `a guess`、`T, L, B, S` 里抠裸字母。数值题裸数字跳过 `frame 3, 4, 5…` 枚举末行；`* Image 178: Kitchen` 本来就不抽 178。
- Viewer 里 step0 的 `acc` 已换成 `rescore_parser_v2`（官方 46.51%）。
- 已写出（不覆盖 `vsibench_val_boxed.parquet`，帧全部 reuse）：
  `data/eval/vsibench_verl/vsibench_val_boxed_lastline.parquet`（5130 行）
- 后缀：` The final answer MUST BE put in \boxed{} on the last line of your response.`
- 下一轮训练内评测显式设
  `VAL_FILE=$PWD/data/eval/vsibench_verl/vsibench_val_boxed_lastline.parquet`。
  与 boxed-only 的 47.19 / 49.07 **不可并列**（多了一句指令）。
- 构建：`python3 scripts/opsd/build_vsibench_val_parquet.py --prompt-suffix ' The final answer MUST BE put in \boxed{} on the last line of your response.' --tag boxed_lastline`

### `20260828_val_resp4096_smoke`（**已完成**，8 卡已释放）

- Registry：`experiments/registry/20260828_val_resp4096_smoke.yaml`
- 结论：**通过**。启动日志 `val response cap 4096, engine max_model_len 15360`；
  Hydra `val_kwargs.response_length=4096`；val dump step2 出现 `resp_tokens=4096`（1/64），
  step0 有 1 条 `1104>1024`，证明上限不再是 1024。训练 rollout 仍为 1024。
- 日志：`logs/train/20260828_val_resp4096_smoke/`，val：`logs/val/20260828_val_resp4096_smoke/`
- 最后更新：2026-08-28

### `20260827_mvopsd_vlm3r_epoch1_thinking`（**已停止**，用户 2026-08-28 要求停跑）

- Registry：`experiments/registry/20260827_mvopsd_vlm3r_epoch1_thinking.yaml`
- 停止原因：in-loop val 仍用 1024 cap（`run_mvopsd.sh` 覆盖 yaml 4096）；已跑到 ~step 375+。
- 日志：`logs/train/20260827_mvopsd_vlm3r_epoch1_thinking/`
- Checkpoint：`checkpoints/20260827_mvopsd_vlm3r_epoch1_thinking/global_step_250`（其余 step 已删，2026-08-29）
- 最后更新：2026-08-28

### `20260827_step225_tok2048_sample_judge_ab`（**已完成**，node-A）

- Registry：`experiments/registry/20260827_step225_tok2048_sample_judge_ab.yaml`
- 协议：sample@2048 + trust_boxed/terse/mca_tail
- 结果：overall **44.75%**（official judge_extract），clip **2.51%**，送 Judge **136** 条（0.55% 规则无法解析=28 条）
- A/B：无法解析→Judge vs 判错，**差 0.00 pp**（Judge 0/28 救回）
- OUT：`logs/vsi_train_eval/.../global_step_225_tok2048_sample_trust_v3/`


- Registry：`experiments/registry/20260827_step225_tok4096_trust_terse.yaml`
- Checkpoint：`checkpoints/20260826_mvopsd_vlm3r_epoch1_boxed_sample/global_step_225`
- 协议：boxed prompt + `trust_boxed` + **`trust_terse`（无截断门控）** + judge extract + **max_tokens=4096**
- 两臂：`greedy` 与 `sample(t=1.0,top_p=0.8)` 离线重评
- 日志：`logs/eval/20260827_step225_tok4096_{greedy,sample}_trust_terse.log`
- 最后更新：2026-08-27

### `20260826_vlm3r_epoch1_boxed_sample`（**已完成**，477/477 步）

- Registry：`experiments/registry/20260826_vlm3r_epoch1_boxed_sample.yaml`
- 窗口：`cursor-window-mvopsd`
- 用户 2026-08-26 的四项决定：teacher non-thinking + 验证采样 + vlm3r 完整 1 epoch
  （30,567 行 / batch 64 = **477 步**）+ 验证 prompt 加 boxed + judge 严判 extract。
- **这不是单变量 arm**，同时改了四项，刻意如此：因果归因已由 `20260824_teacher_thinking_off`
  单独完成，本轮要的是一个规模化读数。若结果为负需退回单变量 arm 才能定位。

**为 boxed 落地的三处代码改动**（前三项决定都是已有开关，只有 boxed 需要新代码）：

| 改动 | 原因 | 默认 |
| --- | --- | --- |
| `build_vsibench_val_parquet.py` 加 `--prompt-suffix` / `--tag` | 训练内评测的 prompt 是预先烤进 val parquet 的，训练回路里没有 suffix 通路 | 空后缀，plain 那份逐字节不变 |
| `mvopsd_reward.py` 加 `vsibench_boxed_primary` | 有闭合框只读框内，无框退回原解析器 | `False` |
| `ray_trainer.py` 加 `validation_judge.trust_boxed` | 已从框内结算的行不再送 judge，否则严判会用框外推理覆写框内答案 | `False` |

**step 0 预期带 42~47**，不是 53.6。用归档 boxed 探针回放训练内判分链推算：

| 归档 arm | boxed 优先 OFF | ON | 合规率 |
| --- | --- | --- | --- |
| plain @1024 | 53.13 | 53.13 | 0.00%（delta 恰好 +0.00，证明开关不污染历史曲线） |
| boxed @1024 | 45.98 | **47.39** | 91.68% |
| boxed @8192 | 46.42 | 47.92 | 95.79% |

即 boxed prompt 本身值约 **−5.7 pp**（协议代价，不是模型退化），boxed 优先判分值 **+1.4 pp**
（方向是加分，说明去掉的是尾部扫描的假阳性），采样再按 2x2 实测的 −2~−5 估。
**若 step 0 落在 53 附近，说明 boxed 后缀没生效，先查 `VAL_FILE` 而不是解读曲线。**

- 已核验：plain 那份 val parquet md5 前后一致；5,130 行逐行 boxed == plain + 后缀，差异 0 行；
  frames 9,216 张全部复用未重抽；三个新开关 hydra 解析为 bool。
- 测试：新增 `test_boxed_primary_scoring.py` 全绿；`test_thinking_split.py` 因 teacher thinking
  翻转改为机制/配置分离断言后全绿；另 5 个相关测试全部复跑通过。
- 预计墙钟约 **16.7 h**（训练 4.4 h + 20 个评测点 × 约 37 min）。评测占约 74%。
- 冒烟日志：`logs/train/_smoke/boxed_sample_smoke.log`
- 最后更新：2026-08-26 17:21 UTC+8

### `20260825_token_budget_judge_extract`（**进行中**，node-A 8 卡，01:40 起）

- Registry：`experiments/registry/20260825_token_budget_judge_extract.yaml`
- 窗口：`cursor-window-mvopsd`
- 目的：同 vLLM + judge-extract 严判，仅改 `max_tokens`（1024 vs 4096）；基座 + llava_hound step 100。
- 4096 两臂复用归档：`decoding_ab/base_greedy`（53.06）、`global_step_100/judge_extract`（7.37）。
- 本轮新跑：`step100_tok1024`（01:36 起，进行中）→ 排队 `base_tok1024`（chain_after PID 2227096）。
- 驱动：`scripts/opsd/run_token_budget_judge_extract.sh`
- 日志：`logs/eval/token_budget_judge/{driver.log,status.txt}`
- 启动命令：
  ```
  setsid nohup bash scripts/opsd/run_token_budget_judge_extract.sh \
    > logs/eval/token_budget_judge/driver.log 2>&1
  ```
- 预计墙钟：base ~30 min + step100 ~60 min，串行合计约 1.5 h。
- 最后更新：2026-08-25 01:40 UTC+8

### `20260824_teacher_thinking_off`（**已完成**，node-A 8 卡，02:35–04:18，1.72 h）

- Registry：`experiments/registry/20260824_teacher_thinking_off.yaml`
- 窗口：`cursor-window-mvopsd`；实验名 `20260824_mvopsd_llava_hound_nothink_main`
- 目的：补上 `20260821_mvopsd_single_source` 缺的**单变量对照**。那一轮把
  `teacher_enable_thinking=True` 列为长度爆炸的最高优先级机制假设，但同一轮
  同时改了数据源、`rollout.n` 与 thinking 三项，所以那条 arm 的 ISSUE 只能标
  diagnosed。本 arm **只改 thinking 一项**（True → False，两侧同为 non-thinking，
  即 Vision-OPD 的原始设定），其余与 llava_hound arm 逐项相同。
- 规模：25 步（0.043 epoch）、`TEST_FREQ=SAVE_FREQ=25`。选 25 是因为对照组的失效
  **完整发生在前 25 步内**（train mean 225.7 → 821.3、clip 0 → 0.5），
  训到 175 步没给出新信息（21.31 vs 21.92）。
- **对照组精确读数**（`train_20260821_163501.log`）：
  step 0 = 53.38（rule 与 overall 相同）；step 25 **rule_only 20.85** /
  overall 21.92 / micro 22.22，验证 `resp_tokens` mean 1014.97 median 1024，
  训练 `response_length/mean` 821.28 clip 0.5，vopd_loss 0.01385。
- **跨 arm 只能比 `rule_only/acc`**：对照跑在级联判官时期（只能加分，21.92 =
  20.85 + 1.07），本 arm 跑在严判 extract 默认之后（全行送判、**可以降分**），
  两侧 `overall` 是两个不同的估计量。`rule_only` 定义一字未改，是唯一合法对比列。
- 验证解码两侧都是**贪心**（未设 `VAL_DO_SAMPLE`）。曲线可比性优先于
  「采样能抬 13 pp」那条结论。
- 第一道正确性闸门：**step 0 必须复现 53.6 带**。不复现就先查环境，
  此时任何长度结论都不可信。
- 排队方式：新增 `scripts/opsd/tools/chain_after.sh <pid> <min_free_mib> -- <cmd>`，
  等评测驱动 PID 退出**且**每卡空闲显存 ≥ 80 GB 才 exec 训练。
  两个条件都要，因为被杀掉的 vLLM 会留下孤儿 worker 占着 85 GB/卡 且利用率 0%
  （ISSUE-403），只看 PID 会在卡还被占着时启动。
  **第一版按 pgrep 模式等待是错的**——脚本自己的命令行含该模式，
  启动它的 shell 也含，于是永远匹配得到、死等；已改为按 PID。
- **01:21 的首次自动启动失败了（ISSUE-501，已修）**：排队本身工作正常，但发起排队的
  窗口里 conda `sr_opsd` 是激活的，它导出的 `PYTHONNOUSERSITE=1` 被原样继承，
  于是系统 python3 跳过 `~/.local` 看不见 vllm，judge 起不来、训练在 0 步退出。
  **与 ISSUE-202 同一根因，换了个脚本**。归档命令里的 `PATH=/usr/bin:$PATH` 只在
  非 conda shell 里成立，不够。
  `run_mvopsd.sh` 现在有一道 0.4 s 的前置闸门（`find_spec('vllm')`，不真 import），
  查不到就报出 python3 实际路径并点名 `PYTHONNOUSERSITE`。
  02:35 用 `env -u PYTHONNOUSERSITE -u CONDA_*` + 剔除 miniconda 的 PATH 重启成功。
- 日志：`logs/train/_queue/nothink_arm_retry.log`（驱动）、
  `logs/train/20260824_mvopsd_llava_hound_nothink_main/train_20260824_023543.log`
- 已核验生效：resolved config 里 `teacher_enable_thinking: False` +
  `data.apply_chat_template_kwargs.enable_thinking: False`（两侧同为 non-thinking）；
  judge 185 s 起来并已睡下。
- 实际墙钟 1.72 h，与预估 1.7 h 相符（评测占约 80%）。
- **step 0 闸门通过**：`rule_only` 53.3809 vs 对照组 53.3832，差 0.002 pp。
  环境与验证集没变，所以下面的长度结论可信。
- **结果：假设成立，且是单变量证据。**

  | 指标 @step 25 | thinking **ON**（对照） | thinking **OFF**（本 arm） |
  | --- | --- | --- |
  | VSI `rule_only` | 20.85 | **52.57** |
  | 训练 `response_length/mean` | 821.28 | **221.31** |
  | 训练 `clip_ratio` | 0.50 | **0.0156** |
  | 验证 `resp_tokens` 中位 | 1024 | **4** |
  | vopd_loss | 0.01385 | 0.01490 |

  训练侧长度 25 步全程在 **187.8~233.8** 之间无趋势（对照同期 225.7 → 821.3），
  `clip_ratio` 25 步里 23 步为 0。**长度爆炸完全没有发生。**
- **机制定论**：20260821 那轮的 response-length explosion 主因就是
  teacher 在打开的 `<think>` 里给 student 的每个 token 提供**推理态** soft target、
  而 student 自己是 non-thinking。两侧同态后机制消失。
- **同时必须记的三条否定性结论**：
  1. **不等于方法有效**。25 步 = 0.043 epoch，`rule_only` 53.38 → 52.57（−0.81 pp），
     是「不再破坏」而非「有提升」。本 arm 没有任何正收益证据。
  2. **一条小尾巴正在形成**：验证 `resp_tokens` 的 **p95 从 333 涨到 1024**
     （≥5% 的行撞满上限），`answered/frac` 98.23% → 95.05%，判官判 NONE 从 91 → 254。
     中位数与训练侧均值都没动，所以是**长尾变粗**而不是整体漂移——
     正是 LESSON-020 「均值不动可能掩盖长尾变粗」的那一面。25 步只够看到它开始。
  3. **VOPD loss 平台两条 arm 几乎相同（0.0149 vs 0.0139），下游差 31.72 pp。**
     LESSON-012 的又一次独立复现：不能用 JSD 曲线判断方法是否在起作用。
- frozen teacher 已验证：`teacher_probe` 全程 0.04456961154937744（与前七轮同值），
  `policy_fallback_fraction` 恒 0。
- 跨 arm 只比了 `rule_only`（对照是级联判官、本 arm 是严判 extract，`overall` 不同口径）。
- 下一步：① 把判定**追加**到 `20260821_mvopsd_single_source` 的
  `root_cause_assessment`（不要覆盖那条 diagnosed 记录）；
  ② 决定是否用 thinking=False 重跑完整 1 epoch（586 步），若要盯上面那条尾巴的走势，
  `TEST_FREQ` 应降到 5 或 10；③ `vsi_appr_order` / `spar_32view` 现在具备启动条件，
  但先定 ② 的步数与评测密度。
- 产物：`checkpoints/20260824_mvopsd_llava_hound_nothink_main/global_step_25`、
  `logs/val/20260824_mvopsd_llava_hound_nothink_main/`（step 0 与 25 两份 dump）
- 沉淀：LESSON-029（thinking 同态）、LESSON-030（跨环境启动与排队）
- 最后更新：2026-08-24 04:25 UTC+8

### 基座贪心@4096 + 严判评测（**已完成**，01:00–01:21，20.5 min，卡已释放）

- 目的：补齐解码 × 判分的 2×2 对照最后一格。此前只有基座**采样**@4096 严判
  （48.19）与 step 100 的两种解码，缺基座**贪心**@4096 严判，
  所以「step 100 相对基座跌多少」一直没有同口径的分母。
- 产物：`logs/eval/decoding_ab/base_greedy/`（judge API 错误 0、unreadable 0）
- 命令：
  ```
  OUT_DIR=logs/eval/decoding_ab/base_greedy \
  FRAMES=32 MAX_TOKENS=4096 MAX_MODEL_LEN=16384 SCENES_PER_BATCH=24 \
  JUDGE=1 JUDGE_MODE=extract \
  setsid nohup bash scripts/opsd/run_training_vsibench_eval.sh --hf models/Qwen3.5-4B
  ```
- **补齐后的 2×2（全部 5,130 题 / 32 帧 / 4096 预算，judge 为严判 extract 主口径）**：

  | 模型 | 解码 | **judge（主）** | rule（诊断） | 截断 % |
  | --- | --- | --- | --- | --- |
  | 基座 | 贪心 | **53.06** | 53.29 | 0.97 |
  | 基座 | 采样 | 48.19 | 48.22 | 0.64 |
  | step 100 | 贪心 | 7.37 | 23.00 | 90.62 |
  | step 100 | 采样 | 21.53 | 35.97 | 75.09 |

- **采样对基座是有害的（−4.87 judge / −5.07 rule），对 step 100 是有益的
  （+14.15 / +12.97）。** 这条推翻了此前「采样口径更宽容」的默认读法：
  采样不会让模型变聪明，它只是打断重复循环。基座本来就不循环、中位输出 2~5 token，
  采样只是往已经答对的短答里注入噪声。**「用哪种解码」不是一个可以全局选定的口径，
  它对健康模型与退化模型的作用方向相反。**
- **真实退化幅度（同口径相减）**：贪心 + 严判 **53.06 → 7.37 = −45.69 pp**；
  采样 + 严判 48.19 → 21.53 = −26.67 pp。两个都是合法数字，但它们**回答的是不同问题**
  （「贪心下这个 checkpoint 表现如何」vs「采样下」），差 19 pp。
  对外只报一个数字时必须写明解码方式。
- 基座贪心 4096 的 rule 53.29 vs 历史贪心 1024 的 52.58：**+0.71 来自放宽预算**
  （截断 2.18% → 0.97%），与判分无关。训练内锚点 53.62 仍是另一条引擎口径。
- 基座上 rule 与 judge 几乎不差（贪心 −0.23、采样 −0.02），而 step 100 上差
  −15.63 / −14.44。**判官的「严」只在退化后的长文本上体现**，
  这与 2026-08-23 那条「谁对取决于回答形态」的记录一致，不是判官整体偏严。

### `20260822_boxed_format_probe`（**已完成**，node-A 8 卡，约 1.4 h）

- Registry：`experiments/registry/20260822_boxed_format_probe.yaml`
- 结果：`logs/eval/boxed_probe/`（`max_tokens=1024`，冻结基座 `./models/Qwen3.5-4B`）
- 结论摘要：`\boxed{}` 合规率 81–92%，但触发 register shift（中位 token 4→303 on VSI）；
  `Answer:` 更短更稳；boxed 的价值是诚实报告「没答」而非控长。
- 窗口：`cursor-window-boxed-probe`；驱动 `scripts/opsd/run_boxed_probe.sh`
- 起因：老师 2026-08-22 建议「让模型把答案放进 `\boxed{}` 以方便评测」，并给了
  verl 的 `math_dapo.py` 作评分参考。
- **核对上游的结论（回答「训练和测试都用了 boxed 吗」）**：老师给的两样东西来自
  verl 的两条不同线。那句 `The final answer MUST BE put in \boxed{}.` 逐字出自
  `examples/data_preprocess/geo3k.py`，由同一个 `make_map_fn` 施加到 train 与 test
  两个 split，**两边逐字相同**；而 `math_dapo.py` 里根本没有 prompt，且
  `reward_score/__init__.py:59-62` 调它时不传 `strict_box_verify`，默认走
  `is_correct_minerva` 的 `(?i)Answer\s*:\s*([^\n]+)`——**boxed 在那个文件的默认
  路径下只是兜底**。
- **最关键的一条**：verl 这两条参考线各自都有一个让格式站住的机制——geo3k 的
  `compute_score = 0.9×acc + 0.1×format`（格式合规是付钱买的），DAPO 的
  overlong reward shaping（8192 预算、最后 4096 线性罚到 −1.0）。
  **我们两样都没有**（`pg_loss = vopd_loss`，回路里没有 reward 项，LESSON-012）。
  所以能不能靠一句 prompt 撑住格式，取决于 teacher 分布会不会照做，必须先测。
- 设计：三条 prompt 臂（`plain` 空后缀 / `boxed` / `answer`）× 两个题集
  （训练池 3,000 行配对子集 + VSI-Bench 全量 5,130）。`plain` 臂兼作闸门——
  prompt 与归档运行逐字相同，必须能复现 `as_trained` 的回答，否则臂间差异不可归因。
- 判分口径**一行未动**：boxed 是并行诊断列（`boxed_present` / `boxed_score`），
  上报数字仍由既有解析器与既有指标产出；`prompt_suffix` 默认空串（LESSON-013）。
- **顺带修掉一个先于 boxed 就存在的判分 bug（ISSUE-402）**：`\text{m}^2` 里的
  LaTeX 指数被 `_NUMBER_RE` 当成数量，而数值兜底取最后一个数字，于是面积答案
  被读成 2。模型本来就在自发写这个，与 boxed 无关。57.7 万条归档回答回放，
  518 条改判，方向全部向上：teacher 41.4% → 约 41.45%，两条视角扫描上移
  0.1~0.4 pp，**无结论翻转**。作答率变化恒为 0——旧解析器读出了数字只是读错了，
  又是 LESSON-018 那个「读错数与答得差同形」的形态。
- 待跑完后回答：合规率（总体 / 按数据源 / 按 VSI 题型）、长度与撞满预算比例、
  分数与作答率、以及 `plain` 臂的自发 boxed 率。
  **本轮不能回答「boxed 能否治住 length explosion」**——探针测的是冻结基座，
  而 explosion 发生在训练过程中。

### `20260823_llava_hound_step100_vsi4096`（**已完成**，8 卡已释放）

- Registry：`20260821_mvopsd_single_source.yaml` 的 `results.offline_eval_step100_tok4096`
  （同一 arm 的评测与训练放同一份 registry）
- 窗口：`cursor-window-mvopsd`；驱动 `scripts/opsd/run_llava_hound_step100_vsi4096.sh`
  （状态 `logs/eval/llava_hound_step100_tok4096/status.txt`）
- 动机：用户要求对 llava_hound 单源 arm 的 **step 100** checkpoint 做离线分析，
  用 vLLM 推理、`max_tokens=4096`，并计算 VSI-Bench 准确率。
- 模型：`checkpoints/.../global_step_100` → merge 为
  `output/20260821_mvopsd_single_llava_hound_main_global_step_100_hf`
- 协议：`spatialstack`、32 帧、greedy、`MAX_MODEL_LEN=16384`、规则层 + judge
- 产物：`logs/vsi_train_eval/20260821_mvopsd_single_llava_hound_main/global_step_100/`
- 时间：14:26 起（等 boxed_probe 让卡），15:23 结束，**54.2 min**
- **结果：rule-only 23.00 / judge 23.07，作答率 97.45%，截断率仍 90.62%，中位输出 4096。**
  最高 `object_rel_direction_easy` 47.47，最低 `object_abs_distance` 8.91。
- **读数约束**：训练内 1024 口径的 micro 是 21.17%，与这里的官方 8 题型非加权 23.00%
  **口径与预算都不同，不能相减当作「放宽预算的收益」**。能确定的是预算翻到 4096 后
  仍有 90.62% 撞满上限——瓶颈是重复枚举形态，不是预算不足（同向 ISSUE-305）。
- **判分注意**：满分行里有「幸运误解析」。id=168（sofa 金标 173cm）通篇是
  `Image N: Kitchen.` 枚举、没有答案句，解析器兜底抽到**帧号 178**，
  相对误差 2.89% < 5%，MRA 十档全过记满分。故 23.00% 是解析口径下的**上界**，
  不等于「模型理解正确」的比例。详见 registry 的 `scoring_caveat`。
- **重评（2026-08-23 17:21）**：修复 `Image N:` 帧号误解析后全量 5130 题重评
  （`scripts/opsd/tools/rescore_vsibench_samples.py` → `.../global_step_100/rescore/`）。
  rule-only **22.90**（-0.10），judge **22.97**（-0.10）；325 行 rule 分数变化，
  5 行从满分降为低分、3 行升为满分。id=168：178→150，1.0→0.8，`truncation_error`。
- **顺带修掉查看器缺陷 ISSUE-306**：`build_vsibench_val_viewer.py` 把 parquet 的
  `id` 与 `index` 用 `str()` 压进同一张映射表，**5,130 行里 2,638 行（51.4%）**
  的题型/场景/32 帧图属于另一道题（题面、金标、回复、分数一直是对的）。
  已改为按 `extra_info.id` 单键 join + 元数据以评测记录为准 + 冲突标红，
  重建后错位 0/5130，failure 分布逐项不变。见 LESSON-028。
- 查看器：`experiments/viewers/vsibench_20260821_mvopsd_single_llava_hound_main_step100_tok4096/`
  （`cd` 仓库根目录跑 `python3 -m http.server 8765` 后打开）
- **双判分查看器**：`experiments/viewers/vsibench_20260821_mvopsd_single_llava_hound_main_step100_tok4096_dual/`
  （同端口；规则 23.16% vs Judge 7.58%，可筛「规则满分·Judge 零分」等）

### `20260823_judge_extract_step100`（**已完成**，GPU 4–7 已释放）

- 驱动：`scripts/opsd/run_judge_extract_step100.sh`
- 产物：`global_step_100/judge_extract/{samples.jsonl,summary.json}`
- 时间：18:08 起，18:22 结束（judge API ~12.4 min）
- **结果：judge-extract overall 7.37%，作答率 12.73%（653/5130）**
- vs 规则分 23.16%：**-15.58 pp**；API 错误 0；全文送入
- id=168：NONE / 0 分（规则层曾 1.0）

### `20260823_boxed_probe_tok8192_dual`（**进行中**，11:51 重启，node-A 8 卡）

- Registry：`experiments/registry/20260823_boxed_probe_tok8192_dual.yaml`
- 窗口：`cursor-window-boxed-probe-8k`；驱动 `scripts/opsd/run_boxed_probe_tok8192_dual.sh`
  （`setsid nohup`，状态见 `logs/eval/boxed_probe_tok8192/status.txt`）
- 动机：1024 token 下 boxed 臂大量撞满预算（pool 17.5%、VSI 7.9%），中位长度暴涨；
  用户要求把推理上限提到 **8192**，并对**基座**与**后训练**模型都测三条 prompt 臂。
- 模型：
  - 基座：`./models/Qwen3.5-4B` → `logs/eval/boxed_probe_tok8192/base/`
  - 后训练：`./output/20260821_mvopsd_single_llava_hound_main_global_step_175_hf`
    （llava_hound 单源，step 175）→ `logs/eval/boxed_probe_tok8192/post_step175/`
- 生成：`max_tokens=8192`，`max_model_len=20480`（11264 prompt cap + 8192 response）
- 设计：仅 **boxed** 臂 × **VSI-Bench 全量 5,130**（`SKIP_POOL=1`，不跑训练池子集）。
- 回复落盘：`vsi_boxed/samples.jsonl`（每行含完整 `response` + boxed 诊断列 + 判分结果）。
- 01:31 首次尝试在生成前就因显存不足退出。根因是 01:26 那次训练池评测被强制停止后，
  vLLM 引擎与 8 个 worker 变成孤儿进程，占着 85 GB/卡 达 10 小时且利用率为 0%
  （ISSUE-403，已解决）。同时修掉 ~/.local 的 pydantic 版本冲突（ISSUE-404）。
- 11:51 重启成功：显存闸门通过，依赖解析与归档两轮逐行一致，8 卡满载生成中。
  驱动已加两道防护：启动前检查最小空闲显存；退出时清理 vLLM 引擎与 worker。
- 跑完后：`PROBE_ROOT=logs/eval/boxed_probe_tok8192/base python3 scripts/opsd/tools/summarize_boxed_probe.py`
  与 `.../post_step175` 各跑一次；对比 1024 探针看 cap% 与分数变化。

### `20260821_mvopsd_single_source`（**已取消**；两条 Round-1 arm 提前终止）

- Registry：`experiments/registry/20260821_mvopsd_single_source.yaml`
- 动机：teacher 可靠性全量评测把混合池拆开后发现同一个 teacher 在 llava_hound 上 57.7%、
  在 spar_32view 上 19.9%，题型层面差 40 倍。混合池上的每条曲线都是这些信号的加权和，
  而权重是数据量决定的、没人选过。用户与老师 2026-08-21 决策：一次只训一个数据集。
- 配置（四条 arm 共享）：lr 2e-6、batch 64、`rollout.n=1`、teacher 视角上限 **8**、
  student `K ~ U{2,3,4,5}` 逐样本抽、teacher 开 thinking / student 与评测关、
  每 25 步全量 VSI-Bench（5,130 题）+ 存档。
- **步数按「过一遍数据集」定，不套用固定值**（用户 2026-08-21 决策）：
  步数 = `floor(训练行数 / 64)`，即恰好 1 个 epoch。统一 300 步会让 llava_hound
  只过 0.51 个 epoch、vsi_appr_order 过 13 个 epoch，同一横轴上并列的曲线
  其实在测完全不同的东西。**代价**：横轴不再对齐，读图要按 `step/total` 的比例对齐，
  不能按绝对步数。
- **基座锚点是 53.62**（训练内口径，2026-08-21 诊断）。离线 vLLM 52.58 / 离线 HF 53.40
  是不同引擎口径，不要拿来比训练内曲线。

| arm | 节点 | 数据源 | teacher 可靠性 | 训练行数 | 1 epoch 步数 | 评测点 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llava_hound | node-A | `llava_hound_64k` | 57.7% | 37,509 | **586** | 24（每 25 步） | **step 199 提前终止；最新 val/ckpt=175** |
| vlm3r | node-B | `vlm3r_scannet` | 44.0% | 30,567 | **477** | 20（每 25 步） | **step 116 提前终止；最新 val/ckpt=100** |
| vsi_appr_order | 待定 | `vsi_appr_order` | 55.3% | 1,735 | **27** | 9（每 3 步） | **未启动，随实验取消** |
| spar_32view | — | `spar_32view` | 19.9% | 见下 | 328 / 140 / 236 | — | **未启动，随实验取消** |

两条大 arm 用同一个 `TEST_FREQ=25` 而不是各自缩放：跨 arm 比较是本轮的全部目的，
评测点落在同一步数网格上才能直接对齐。`vsi_appr_order` 只有 27 步必须单独用 3。

- **判读规则**：只读「四条 arm 的排序是否与 teacher 可靠性一致」。池大小、题型结构、
  金标形式、epoch 数都不同，**不能**用某条 arm 绝对值更高来说该数据集更好。
  `vsi_appr_order` 额外还与 VSI-Bench 的 `obj_appearance_order` 题型同源，
  它的涨幅更接近 in-domain 微调而非空间能力迁移。
- **spar_32view 为什么卡住**：teacher 相册要从 32 压到 8，但 12,051/21,025 条（57%）的
  题面直接点名 `Frame-N`（那些帧上画着红/蓝标记点）。机械层面已解决（钉住被点名的帧、
  补到 8 张、题面重编号，找不到映射时抛异常）；未解决的是证据层面——受控扫描只覆盖
  4 个**不点名帧**的题型（8 张 18.4% vs 32 张 22.0%），而点名帧的 4 个是距离题，
  需要跨视角三角化，砍到 8 张的代价没有数据。三个选项的可靠性加权已算好：
  只留不点名帧 8,974 条 → **19.4%**；全截到 8 张 21,025 条 → 19.9%；
  全截并摘掉 `spatial_imagination_oo_*` 15,155 条 → **27.1%**。
  「只保留证据覆盖到的部分」反而给出最差的池，因为近零可靠性题型集中在不点名帧那一半。
  用户 2026-08-21 决定先不测，等前三条有结论再定。三个方案对应的 1 epoch 步数分别是
  328 / 140 / 236。
- 冒烟已过（2 步 / batch 8）：`grpo_fallback_count=0`、`vopd_loss=0.0619`、
  `resp_tokens/mean` 99.1→101.4 正常落盘、`teacher_image_swap_fraction=1.0`、
  `policy_fallback_fraction=0`、`actor/lr` 走 10 步 warmup。
  JSD 0.0619 高于前四条 arm 的 0.018~0.022 平台，与「teacher 改成推理态分布」的预测同向。
- **正式结果与汇报**：
  [`experiments/reports/汇报_20260821_单数据集训练验证Teacher可靠性影响.md`](reports/汇报_20260821_单数据集训练验证Teacher可靠性影响.md)。
  两条 arm 都在 step 25 前发生 response-length explosion：
  - llava_hound：VSI 53.38% → 21.92%，验证截断率 97.4%；step 175 仍仅 21.31%；
  - vlm3r：VSI 53.36% → 18.98%，验证截断率 100%；step 100 仍仅 20.28%。
  - 两侧验证 median 都从 4 token 变为 1,024 token；VOPD loss 同时持续下降。
  结论是共享的生成形态退化压倒了数据源可靠性差异，当前配置无法检验原排序假设。
  VOPD 代码只蒸馏 Student response，不蒸馏 prompt/image；但 response 内没有
  thinking/answer 分段。Teacher thinking 开、Student 关是最高优先级机制假设，
  尚缺单变量对照，不能记成已证实唯一根因。
- **FSDP（2026-09-07，node-B）**：按用户要求删除 `checkpoints/20260821_mvopsd_single_vlm3r_main/`（213G）。
- 启动命令（两机同构，注意 `PATH` 前置系统 python——见 registry ISSUE-304）：
  ```
  PATH=/usr/bin:$PATH \
    EXPERIMENT_NAME=20260821_mvopsd_single_llava_hound_main \
    TRAIN_FILE=$PWD/data/mvopsd/parquet_single_llava_hound_64k/main_train.parquet \
    TOTAL_STEPS=586 TEST_FREQ=25 \
    setsid nohup bash scripts/opsd/run_mvopsd.sh main > logs/train_launch_llava_hound.log 2>&1
  ```
- **16:09 那次启动已作废**（当时设的 300 步）。停止时两条都还在 `val_before_train`、
  未走到第 1 个训练步、无 checkpoint，只损失约 20 分钟首次评测。
  作废的日志是 `train_20260821_160914.log` / `train_20260821_160951.log`，
  里面 `total_training_steps=300`，不要误当作本轮配置。
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：MV-OPSD 两机进程均已清零；node-A 8 卡空闲，node-B 有非本项目作业占卡
- 活跃进程或作业 ID：无
- 最后更新：2026-08-21 23:40 UTC+8（用户提前终止并完成结果诊断）

### `20260820_mvopsd_view_ratio`（两机各一条 arm）

- Registry：`experiments/registry/20260820_mvopsd_view_ratio.yaml`
- 动机：LR 与 batch 已被证明只改变到达平台的轨迹（CV-Bench 渐近线一律低于自身基座
  3~6.5 分）。用户提出新方向：**student 视角数太少**。
  **2026-08-21 订正**：本行原写「四条 arm 的 JSD 平台一律 0.0157」，与实测不符。
  0.0157 是 v1 的值；四条 arm 实测为 lr5e6_b64 0.0184 / lr5e6_b128 0.0183 /
  lr1e5_b64 0.0217 / lr1e5_b128 0.0207（末 20 步均值），**随 LR 单调抬高**。
  结论方向不变，但应表述为「平台不降反升」而非「平台不动」。
  详见 registry `20260818_mvopsd_lr_batch_sweep.yaml` 的 `results`。
  本轮把 student 预算从「逐样本随机 K∈{1,2,4}」改成「相册的固定比例」。
- 两条 arm（其余参数与 `lr5e6_b64` 完全一致：lr 5.0e-6、batch 64、n=4、300 步、
  frozen teacher、token-mean、seed 20260816）：

  | arm | 机器 | student 预算 | K（N=8 / N=32） | 平均 K | 落差 N/K | 预计墙钟 |
  | --- | --- | --- | --- | --- | --- | --- |
  | `..._khalf_main` | node-A | N/2 | 4 / 16 | 6.96 | 2.00 | 约 14.5h（实测 123.0 s/步 + 20 次评测） |
  | `..._kquarter_main` | node-B | N/4 | 2 / 8 | 3.48 | 4.00 | 约 13.3h（实测 110.2 s/步 + 20 次评测） |

  实验名前缀 `20260820_qwen35base_mvopsd`。teacher 两臂完全相同（看全部 N，平均 13.93 张）。
- 数据：**新池** 83,123 训练 + 1,674 留出（源 plan 筛 `n_views >= 8`，84,797 条）。
  `spar_3view` 整批排除（34,470 条），因为它 N=3、只能产生 K∈{1,2}，会让「视角少」
  与「来自该源」成为同一个变量。构成：llava_hound 36.7% / vlm3r_scannet 36.5% /
  spar_32view 24.7% / vsi_appr_order 2.1%。
  两臂视角**嵌套**（N/4 ⊂ N/2）且行序一致，故同一 step 看到同一批样本。
  不重跑抽帧——视角缓存与 K 无关。
- 当前阶段：**训练已按用户要求提前终止**（2026-08-20 11:17 UTC+8，node-A ~250/300、node-B ~267/300）；
  VSI-Bench @32 帧评测**已收口**（结论见 registry `evaluation.results`：khalf step 120 相对基座崩塌约 19 分，
  成因是重复退化而非预算截断，CV-Bench 护栏完全没报警）。
  当前在搭**下一轮的训练内评测链**，见下节。
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：两机 8 卡均空闲
- 活跃进程或作业 ID：无

### 训练内评测链改造（2026-08-20，用户决策）

CV-Bench 护栏已被本轮证伪（LESSON-020），训练过程评测改为 **VSI-Bench 全量 @32 帧 +
LLM judge（只补规则没接受的行）**。

**2026-08-20 17:00 用户决策改为「像 CV-Bench 那样跑在训练循环里、单机」**，
即走 verl 自己的 `_validate()`，复用训练已有的 8 个 rollout engine，不再另起进程、
不再需要第二台机器。此前搭的分离式评测节点方案（`run_training_vsibench_eval.sh` /
`watch_training_vsibench_eval.sh`）**保留**，降级为「离线复评已存 checkpoint」的工具。

改造内容（四处，均已完成）：

1. `scripts/opsd/build_vsibench_val_parquet.py`（新）：288 场景 × 32 帧抽帧落盘为 PNG
   （9,216 张 / 2.0 GB，按场景共享，不是 5,130×32），写 5,130 行 val parquet。
   抽帧函数与离线评测同一个（`vsibench_eval_core.sample_frames`），存 PNG 不存 JPEG，
   使两条路只差引擎、不差像素。
2. `scripts/opsd/mvopsd_reward.py`：新增 `vsibench/` 前缀分支，MCA exact / NA MRA，
   解析器复用 `vsibench_scoring.py`。同时给所有行加 `resp_chars`（响应长度是重复退化
   最早的信号，LESSON-020）。
3. `verl_pkg/verl/trainer/ppo/vsibench_metrics.py`（新）：按 VSI-Bench 官方定义聚合
   —— direction 三档先合并成一票，再对 8 个题型**非加权**平均。行数最多与最少的题型
   差 5 倍，micro 平均是另一个数（本轮实测 53.12 vs 51.61），并列会算错。
4. `ray_trainer.py::_judge_vsibench_rows`（新）：VSI 的第二级判分。

- **判官口径（已统一两侧）**：只送**规则读不出答案**的行（`answered == 0`），
  不送「读出了但答错」的行——后者是答错，不是没读出来。选择题走 Yes/No 级联，
  数值题走数字抽取后按 MRA 打分（Yes/No 会把连续的 MRA 压成二元准确率还顶着 MRA
  的名字上报，LESSON-019）。离线的 `vsibench_eval_core.build_judge_requests` 也已
  改为同一默认口径（`mca_scope="unreadable"`），Vision-OPD 的宽口径保留为
  `mca_scope="rule_wrong"` 供对比。
- 判官只可能加分不可能减分，所以 `val-core/vsibench/rule_only/acc` 与判分后的
  `val-core/vsibench/overall/acc` **同时上报**，趋势一律看 `rule_only`。
- **两处长度闸门**（只有一处会报错，另一处静默出错）：
  - `rollout.max_model_len` 5120 → **12288**：引擎容量，训练与评测共用，太小 vLLM 直接拒绝。
  - `rollout.val_kwargs.prompt_length` → **11264**：仅评测生效。太小的话
    `agent_loop.py:338-346` **静默截断** prompt，丢掉大部分帧却什么都不报，
    分数会变成对一段模型没看过的视频打分。
  - `data.max_prompt_length` **保持 4096 不动**：它只用于过滤、不做 padding，
    且 `rollout.prompt_length` 由它插值而来并继续管着训练侧。训练路径零改动。
- 观测指标：`val-core/vsibench/{overall/acc, rule_only/acc, answered/frac, resp_chars/median}`、
  `val-aux/vsibench/type/<题型>/acc`、`val-aux/vsibench/judge/{pending,pending_mca,pending_na,woken,recovered}`、
  `val-aux/vsibench/resp_chars/{mean,p95,max}`。
- **打分链已做逐位对齐验证**：`scripts/opsd/tests/test_vsibench_in_training.py`（新）
  回放归档的 5,130 条基座真实响应，训练内路径（reward + metrics）算出 **52.57**，
  离线公布值 52.58，同时与 lmms_eval 现算值一致。
- 单次评测成本：全量 5,130 题约 **18 分钟**（8 DP × TP=1，与训练同引擎）。
- 分离式离线工具（保留）：单 checkpoint
  `bash scripts/opsd/run_training_vsibench_eval.sh checkpoints/<exp>/global_step_N`，
  轮询 `watch_training_vsibench_eval.sh`；产物 `logs/vsi_train_eval/<exp>/<step>/`
  —— `summary.json` / `samples.jsonl` / `failures/`（按 `truncation_error` /
  `parse_error` / `partial_error` / `factual_error` / `judge_recovered` 分桶）。
  联调记录：基座 4 场景 35 题跑通完整两阶段（16:06），产物在
  `logs/vsi_train_eval/_wiring_check/`（**不是可用分数**）；
  修掉 ISSUE-201（汇总缺 `question_type` 导致静默崩溃）、
  ISSUE-202（`conda activate` 的 `PYTHONNOUSERSITE=1` 让系统 python 找不到 vllm）。

**3 步端到端串通（`20260820_qwen35base_mvopsd_khalf_pipecheck`，进行中）**

- 目的：验证训练与评测能否正常跑完，不是要分数。`TOTAL_STEPS=3 TEST_FREQ=1 SAVE_FREQ=1`，
  `val_before_train: True` 故会先出一个 step 0 基座分作为正确性校验。
- 第一次跑（16:52，已弃）step 0 得 **51.61**，与离线同模型的 52.58 差 0.97。
  查明原因见 ISSUE-203：**val parquet 漏了 system 轮**。已修并重建，
  重建后训练内渲染出的 prompt 与离线协议逐字节一致（含 32 张图、关闭思考）。
- 第二次跑（17:34 启动）用修正后的 val 集，结果待记。
- 训练终止原因：用户判断继续训练无必要；CV-Bench 已稳定在基座下方约 6–7pp
- **[实验设计 · 澄清] 两条 arm 的训练超参本就不同，这是有意为之**
  （用户 2026-08-20 12:57 澄清：本轮目的就是对比两组不同训练超参的性能差异）。
  - 更正记录：12:56 我曾把它当成混淆因素、并判定"全局最佳 checkpoint"的选法失效。
    **该判断是错的**，跨臂比较正是本轮想要的比较。保留此处以存更正过程。
  - 但表述必须收紧：结论只能说"配置 A 整体优于配置 B"，**不能归因到单一变量 K**，
    因为两臂之间不止 K 一项不同。registry 里"view ratio / N/2 vs N/4"的命名
    容易被读成单变量消融，需在报告中明确改写。
  - 仍然独立成立的读数约束：khalf 120 = 83.79 与 kquarter 75 = 83.26 只差 **0.53pp**，
    CV-Bench 是护栏指标、单随机种子、无误差棒。这个差距**不足以给两套配置排名**，
    只能说"两者相当"。选 khalf 120 去做 VSI 是合理的取点，但不等于它已被证明更好。
  - 待补（写报告前必须补）：两臂逐项超参 diff。
  - 评测侧无差异：硬件（H20 + 570.133.20）、环境（torch 2.10.0/tf 5.3.0/acc 1.13.0/
    decord 0.6.0）、评测代码 5 个文件 md5、数据集 test.jsonl md5 与 512 个视频字节数，
    2026-08-20 12:55 逐项核对，两机完全一致。
- **VSI 评测选点**：用户 11:30 决定只测一个 checkpoint。
  两臂全部 33 个评测点里 CV-Bench 最高的是 **khalf step 120 = 83.79%**
  （次高 kquarter step 75 = 83.26%）。基座 `Qwen3.5-4B` 锚点同时在 node-B 跑，
  否则单个数字无从判读——基座在本项目从未测过 VSI。
- **评测协议已改（`VSIBENCH_PROTOCOL`，默认 `spatialstack`）**：
  1. 生成上限 **16 → 1024 token**，并把 `until` 置空（否则 TaskConfig 会用空行当停止符，
     把推理在第一段就截断）。
  2. 解析器换成 `scripts/opsd/vsibench_scoring.py`：选择题按题目**实际提供的选项字母**
     取答案（优先 "answer is X"，其次 `(X)`，再次末行裸字母，最后匹配选项正文），
     数值题取显式答案句或末行数字。上游的 `pred.split(' ')[0]` 在长推理下只会读到
     "Based"，选择题记 0、数值题记最差 MRA。
  3. **指标完全不变**：MCA 用 exact match、NA 用 `MRA:.5:.95:.05`、题型间非加权平均，
     全部沿用 lmms_eval 原实现。回归测试回放 5 个历史短答 run，新旧解析器打分逐个一致
     （如 `novggt_aligned` 两侧都是 64.24），证明只改了"去哪里找答案"。
  4. 旧协议仍可复现：`VSIBENCH_PROTOCOL=lmms_legacy`（16 token + 首词解析）。
  5. 新增 `vsibench_answered` 指标，报告有多少行能读出答案，与分数分开看。
- 相关文件：`scripts/opsd/vsibench_scoring.py`（新）、
  `src/lmms_eval/tasks/vsibench/{utils.py,vsibench.yaml,vsibench_random200.yaml}`、
  回归测试 `scripts/opsd/tests/test_lmms_vsibench_protocol.py`（新，全绿）。
  两机 md5 一致。
- 评测入口：`scripts/opsd/run_view_ratio_vsi_eval.sh`（node-A）/
  `eval_frame_budget.sh`（node-B 基座）；日志
  `logs/eval/_launch/khalf_step120_vsi1024.log`、`logs/eval/_launch/base_anchor_vsi1024.log`
- 阻塞项：无。已修：`~/.local/bin/accelerate` 遮蔽 conda 环境；早前 `pkill -f lmms_eval`
  会匹配到自身 shell 导致清理失效，改用 `[l]mms_eval` 写法。
- **读数约束**：1024-token 协议下的绝对值**不可与历史 64.24 / 67.65 等并列**——那些是
  16-token 协议下测的 SFT 模型。只能与本轮同协议的基座锚点比。
- 基座锚点（node-B，已完成）：HF/lmms_eval **53.40**，answered 99.98%，截断 2.05%；
  vLLM harness **52.58**，截断 2.18%。两套 harness 长度分布一致但绝对值差 0.82，
  **各自内部可比、不可混表**。
- LLM judge（gpt-oss-120b，`scripts/opsd/tools/judge_vsibench.py`）：已接入并完成校验。
  按 benchmark 判据（judge 只抽取选项、评分仍用 lmms_eval exact match）基座
  53.40 → **53.47（+0.08）**，翻转 11 对 7 → **规则解析器无需兜底**。
  照搬 Vision-OPD 的 Yes/No 级联会得到 +1.49，但反向抽查证明那是判据错位加单向
  施加的假象（LESSON-019）。judge 不进默认计分链路，仅作定期校验工具。
- **结果（已完成，写入 registry `evaluation.results`）**：同 harness（vLLM）对比
  基座 **52.58** → khalf step 120 **33.38**，**崩塌约 19 分**。
  已排除评测预算因素：预算 1024 → 4096 后 overall 未回升（34.82 → 33.38），
  answered 仍 93.51%、解析失败仍 333 条、中位长度仍 205 token、
  截断率仅 21.07% → 20.47%。解析失败的形态是**重复退化**（逐帧枚举、连续数数），
  刷满任何预算也不给答案，judge 在此无解——响应里没有答案可抽。
- **护栏失效**：该 checkpoint 是两臂 33 个训练内评测点里 CV-Bench 最高的（83.79%）。
  单图四选一护栏对 32 帧视频上的重复退化完全不敏感，训练全程无任何预警。
- 退化起点（已查）：同一 60 场景子集、同一 vLLM harness、同为 1024 预算下
  基座 54.15 → step 45 **43.18** → step 120 35.61，单调下滑无回头。
  **step 45 就已经掉了 11 分**，退化不是训练末期事件。
  中位输出长度从基座的 4 token 跳到 168~196 token，即训练把「直接作答」改成了「写长推理」，
  其中一部分收不住并滑入重复循环。子集分数与全量分数不可并列（基座全量 52.58 vs 子集 54.15）。
- 下一步：
  1. 用新的训练内评测链回溯更早的 checkpoint（step 15 / 30），确认是否从第一个评测点就已发生；
  2. 补两臂逐项超参 diff（launch 命令 + 生效 config），报告前必须完成。
  （原第 2 条「是否把 VSI 小样本纳入训练内护栏」已有结论：直接换成 VSI 全量，见上一节。）
- 最后更新：2026-08-20 16:10 UTC+8（训练内评测链改造完成并联调通过）
- 启动命令（`max_prompt_length` 按用户 2026-08-20 决策保持 4096，未上调）：
  ```
  # node-A
  EXPERIMENT_NAME=20260820_qwen35base_mvopsd_khalf_main \
  TRAIN_FILE=$PWD/data/mvopsd/parquet_k_half/main_train.parquet \
  TOTAL_STEPS=300 TEST_FREQ=15 SAVE_FREQ=15 \
  setsid nohup bash scripts/opsd/run_mvopsd.sh main actor_rollout_ref.actor.optim.lr=5.0e-6
  # node-B：仅 EXPERIMENT_NAME 换 kquarter、TRAIN_FILE 换 parquet_k_quarter
  ```
- 监控入口：`logs/train/<实验名>/train_20260820_015637.log`（两机同名时间戳）；
  wandb `slcheng/MV-OPSD`，run 名即实验名。
  必看：`val-core/cvbench/combined/acc`、`self_distillation/*` 的 JSD、
  `actor/policy_fallback_fraction`（应恒 0）、`self_distillation/teacher_probe`（应恒定）。
- 已完成的校验（细节见 registry `data.verification`）：
  student prompt 峰值 3,340（half）/ 1,738（quarter），上限 4,096——**这一项是必查的**，
  因为配置是 `filter_overlong_prompts: False` + `truncation: error`，一条超限就会在
  它被抽到的那一步直接抛异常终止训练；teacher 峰值 6,540 < 12,288；
  嵌套/行序/K<N/悬空 `Frame-N`/占位符五项违例均为 0；两机 6 个关键文件 md5 一致。
- 训练内评测：**本轮仅 CV-Bench**（用户 2026-08-20 决策），2,638 条，`TEST_FREQ=SAVE_FREQ=15`。
  该护栏已被本轮证伪（LESSON-020），**下一轮起改为 VSI-Bench 全量**，见上面的改造小节。
- step 0 基座分（本轮池，两机同 val 同权重）：**khalf 86.21 / kquarter 86.23**，
  仅差 0.02pp——本轮跨机基线偏移可忽略（上一轮是 +0.25），两臂 CV-Bench 可直接对比。
- 启动后健康检查（node-A step 5 / node-B step 7，2026-08-20 02:22）：
  `policy_fallback_fraction` 恒 0、`teacher_probe` 两机同为 0.04456961154937744（冻结生效）、
  prompt `clip_ratio` 0（峰值 3311 < 4096，与离线预测 3340 相符）、`aborted_ratio` 0、
  K 构成与设计一致。早期 JSD 已分化：**khalf 0.0170 vs kquarter 0.0287**，方向与假设一致，
  但第 5/7 步不能当结论，要看 200 步后的平台。
  **待盯**：node-A `grad_norm` 1.73 已超 `clip_grad=1.0`（此时 lr 仍在 warmup 的 2e-6），
  node-B 0.98；若稳态都顶在 1.0，两臂有效步长不等，会成为解释差异时的干扰项。
- **读数约束**：
  1. 本轮池不含 `spar_3view`，**绝对值不可与历史 86.96 / 83.57 并列**，两臂只能互相比、
     各自用 step 0 归一化。本轮两机基线实测仅差 0.02pp，故 ±0.25 的旧偏移不再套用。
  2. CV-Bench 是护栏；渐近线回到基座不构成方法有效的证据。
  3. 主指标 VSI-Bench @{1,2,4,8} 帧必须在训练后补测，且**需要先测基座锚点**（目前没有）。
     加测 8 帧是因为本轮 student 平均能拿到 7 张 / 3.5 张：若只有 8 帧改善而 1 帧不动，
     结论是「学到了但不向低帧迁移」。
  4. **本轮无法分离「student 视角太少」与「落差太大」**——N 固定，两者绑死。需第三条
     arm（K=N/4 + teacher=2K）才能分开，视结果决定是否追加。
- checkpoint 保留规则：CV-Bench 最佳 **与最后一步都留**。上一轮四条 arm 的最优步散落在
  225/300/180/150，说明后期曲线接近平坦、argmax 可能落在噪声里；两个权重的 VSI 若接近
  则该选法站得住，若差很多则说明选法本身有问题。
- 已知并接受的风险：用户决定跳过「启动前用基座跑 VSI 冒烟」。可接受，因为本轮评测是训练后
  手动执行、不挂无人值守链条，且权重全部保留，评测链报错只损失修复时间。
- 阻塞项：无
- 下一步：见本节开头「下一步」（回溯 step 15/30 + 补超参 diff）
- 最后更新：2026-08-20 16:10 UTC+8

### `20260818_mvopsd_lr_batch_sweep`（两机并行，各串行 2 条 arm）

- Registry：`experiments/registry/20260818_mvopsd_lr_batch_sweep.yaml`
- 动机：v1（lr 2.0e-6 / batch 64）与 rollout8 都收敛到低于自身基座约 4 分的
  CV-Bench 渐近线（拟合值 83.24 / 82.27），JSD 平台都停在 0.0157（ln2 的 2.3%）。
  用户提出提高 LR 与增大 batch。本扫描测的是：**这两个旋钮能否移动渐近线与 JSD
  平台，还是只改变到达平台的轨迹。**
- 四条 arm（其余参数与 v1 完全一致：frozen teacher、n=4、300 步、同数据同 seed）：

  | arm | 机器 | LR | batch | 每步序列 | 预计墙钟 |
  | --- | --- | --- | --- | --- | --- |
  | `..._lr5e6_b64_main` | node-A | 5.0e-6 | 64 | 256 | 9.0 h |
  | `..._lr5e6_b128_main` | node-A | 5.0e-6 | 128 | 512 | 16.3 h |
  | `..._lr1e5_b64_main` | node-B | 1.0e-5 | 64 | 256 | 9.0 h |
  | `..._lr1e5_b128_main` | node-B | 1.0e-5 | 128 | 512 | 16.3 h |

  实验名前缀均为 `20260818_qwen35base_mvopsd`。每机 b64 跑完自动接 b128，
  合计每机约 25.3 h。
- 当前阶段：**已完成**（node-A lr5e6 链 2026-08-19 23:16 sweep_end；node-B lr1e5_b128
  于 step 274/300 按用户要求提前终止，其余 arm 均自然结束）
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：两机 **8 卡均已释放**（无 `run_mvopsd` / `main_ppo` 残留）
- 活跃进程或作业 ID：无
- 启动命令（driver 负责串行与守护）：
  ```
  # node-A
  LR=5.0e-6 TAG=lr5e6 setsid nohup bash scripts/opsd/run_lr_batch_sweep.sh \
    > logs/train/_sweep/lr5e6_driver.log 2>&1
  # node-B
  LR=1.0e-5 TAG=lr1e5 setsid nohup bash scripts/opsd/run_lr_batch_sweep.sh \
    > logs/train/_sweep/lr1e5_driver.log 2>&1
  ```
- 监控入口：`logs/train/_sweep/{lr5e6,lr1e5}_status.txt`（每条 arm 的
  START/END/rc/elapsed）与同目录的 `*_driver.log`；wandb `slcheng/MV-OPSD`，
  run 名即实验名。
- driver 的三道闸（`scripts/opsd/run_lr_batch_sweep.sh`，本次新增）：
  arm 在 1800s 内非零退出则中止整链（LESSON-007）；`checkpoints/<name>` 非空则跳过
  该 arm（ISSUE-101）；可用磁盘低于 1300GB 则跳过该 arm。
- 启动前核验：两机 GPU 全空、无残留进程、各 3.7T 空闲、`checkpoints/` 为空、
  wandb 凭据就绪、`models/{Qwen3.5-4B,gpt-oss-120b}` 齐备、端口 8100 空闲，
  且 `run_mvopsd.sh` / `mvopsd.yaml` / `cvbench_val.parquet` / `main_train.parquet`
  四个文件两机 md5 一致。
- **读数约束**（写结论前必看）：
  1. 本扫描用 lmms_eval 题面，v1/rollout8 用 native 题面，**绝对值不可与 86.96 /
     83.57 并列**，必须各自用 step 0 归一化。
  2. 跨机基线偏移 +0.25 分，node-A 与 node-B 差异小于约 0.3 分时不得归因于配置。
  3. `clip_grad=1.0` 而 v1 稳态 `grad_norm` 约 0.5；LR 提到 5e-6 / 1e-5 后若
     `grad_norm` 顶到 1.0，裁剪会吃掉部分增益，实际步长不是 2.5x / 5x。必须监控。
  4. CV-Bench 是护栏指标，**渐近线回到基座也不构成方法有效的证据**。主指标
     VSI-Bench 低帧曲线必须用本轮保留的 checkpoint 补测——v1/rollout8 的权重已全删，
     那两轮至今没有任何主指标数字。
- checkpoint 预算：`save_freq=15` × 300 步 = 每条 20 个 × 约 55GB ≈ 1.1T。
  **清理完成（2026-08-20）**：四条 arm 各只留训练内 CV-Bench 最佳 checkpoint（lmms_eval
  题面，已存步）：lr5e6_b64→step 225（83.89%）、lr5e6_b128→step 300（82.77%）、
  lr1e5_b64→step 180（84.20%）、lr1e5_b128→step 150（82.50%，274 步终止前已存步最高）。
  各 ~54G。v1 / rollout8 checkpoint 此前已删。
- 已解决问题：ISSUE-104（`mvopsd_reward.py` 缺 sys.path 自定位，两机首次启动均在
  130s 内失败，被 fast-failure guard 拦下；已修并复刻 verl 加载链验证后重启）
- 阻塞项：无
- 下一步：按 registry `decision_criteria` 做饱和拟合与斜率检验；用保留的 4 个
  checkpoint 补 VSI-Bench 低帧曲线（主指标仍缺数）。
- 最后更新：2026-08-20 00:10 UTC+8

### `20260813_qwen35_geo_baseline`

- Registry：`experiments/registry/20260813_qwen35_geo_baseline.yaml`
- 当前阶段：训练已完成，评测待运行
- 当前任务：运行 VSI-Bench、CV-Bench、BLINK Spatial、SPAR-Bench
- 运行窗口/负责人：未分配
- GPU/节点：未分配
- 活跃进程或作业 ID：无
- 输出目录：`logs/eval/spatialstack_qwen35_4b`
- 阻塞项：无
- 下一步：启动评测前检查 checkpoint 中存在 `config.json`
- 最后更新：2026-08-13 11:24 UTC+8

### `20260817_qwen35base_mvopsd_v1`

- Registry：`experiments/registry/20260817_qwen35base_mvopsd_v1.yaml`
- 关系：接续 v0（`20260816_qwen35base_mvopsd_v0`，产物与 registry 已于 2026-08-18 全部删除；
  结论见 `LESSON-009`/`011`/`012` 与 [`reports/MV-OPSD_实验报告.md`](../reports/MV-OPSD_实验报告.md)）。
  不复用 v0 实验名（ISSUE-101）。
- 相对 v0 的三处改动（详见 registry 的 `changes_vs_v0`）：
  1. 评测进训练循环，首轮 `TEST_FREQ=5`（计划值 10，加密以先确认机制稳定）。
  2. teacher 由 EMA(0.05) 改为**冻结**（新增 `teacher_regularization: frozen`），
     全程停在基座权重。验收看 `self_distillation/teacher_probe` 是否逐步恒定。
  3. 每步样本溯源：rollout JSONL 增加 `sample_id/source/scene_id/k_views_student/
     n_views_teacher/privilege_bucket/uid` 等列，wandb 增加 `data/step/*` 批次构成指标。
- 当前阶段：**训练已完成 300/300**（2026-08-17 22:00 启动，约 10h48m，2026-08-18 上午结束）
- 当前任务：无训练任务；结论待写入 registry 与报告
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：node-A 8 卡**已释放**（无 `run_mvopsd` / `main_ppo` / vLLM 残留进程，显存 0 MiB）
- 活跃进程或作业 ID：无（已结束）
- 保留的 checkpoint：**已全部删除**（2026-08-18，释放约 320G）
- smoke 验收（2026-08-17 21:50，2 步 / `TEST_FREQ=1` / 64 条 val，已通过）：
  - step 0 基座 combined 94.00，step 1 → 93.997，step 2 → 93.75，`answered/frac` 全程 1.0
  - `judge/pending` 0，judge 一次都没被唤醒（规则层读完了全部行，符合预期）
  - `teacher_probe` 在 step 1 与 step 2 完全相同（0.04456961154937744）→ teacher 确实冻结
  - `data/step/*` 构成指标与 rollout JSONL 的溯源字段均正常落盘
  - `policy_fallback_fraction` 恒为 0 → 走的是纯 `pg_loss = vopd_loss` 分支
- 输出目录：`checkpoints/20260817_qwen35base_mvopsd_v1_main`、
  `logs/train/20260817_qwen35base_mvopsd_v1_main`、
  `logs/val/20260817_qwen35base_mvopsd_v1_main`、
  `rollouts/20260817_qwen35base_mvopsd_v1_main`
- 观测：wandb `slcheng/MV-OPSD`，run 名同实验名。
  CV-Bench 每 5 步一次（`test_freq=5`，step 0 基座对照 + step 5..300，共 61 个点）。
  注意 `save_freq=50`，与 `test_freq` 不对齐：61 个评分点里只有 step 50/100/150/
  200/250/300 这 6 个有 checkpoint，其余无法离线重判。
  必看告警：`val-core/cvbench/answered/frac`（格式崩溃）、
  `val-aux/cvbench/judge/woken`（模型开始跑题）、
  `actor/policy_fallback_fraction`（应恒为 0）。
- step 0 基座分（全量 2,638 条，本项目第一次拿到）：**combined 86.96**，
  2D 82.43（ADE20K 76.78 / COCO 88.07）、3D 91.50，`answered/frac` 0.9992，
  judge 仅 2 行待判（低于 min_rows 16，未唤醒）。
  对照 v0 离线重判的 step 50 = 76.50 / step 300 = 72.19，v0 确实是在损伤基座。
  此前 64 条子集上的 76.09 用的是旧 val prompt，已作废。
- 训练结束时的 CV-Bench（训练内协议，官方 combined）：step 0 = 86.96（基座对照）、
  step 25 = 78.83（早期下探）、step 300 = **83.57**，相对基座 -3.4pp。
  frozen teacher 生效（不再出现 v0 的 72.19 崩塌），但基座仍有损伤。
- 阻塞项：无
- 下一步：把 300 步结论写入 registry 的 `metrics`/`conclusion` 与
  [`reports/MV-OPSD_实验报告.md`](reports/MV-OPSD_实验报告.md)。
- 最后更新：2026-08-18 15:15 UTC+8（修正此前误标的「运行中」；实际已完成并释放 GPU）

### `20260818_qwen35base_mvopsd_v1_rollout8`（node-B，与 v1 并行）

- Registry：`experiments/registry/20260818_qwen35base_mvopsd_v1_rollout8.yaml`
- 关系：接续 v1，定位为**整体对齐 Vision-OPD 官方配置**（2026-08-18 用户决策，
  非 rollout.n 单变量消融）。相对 v1 有**两处**改动：
  1. `rollout.n: 4 → 8`
  2. `loss_agg_mode: seq-mean-token-mean → token-mean`（已改在 `mvopsd.yaml`，
     无需 CLI 覆盖；理由见 LESSON-016——原偏离依据用错了长度口径）
  其余与 v1 相同：frozen teacher、batch=64、300 步、同一数据与 seed。
  **评测/存档**：`TEST_FREQ=15`（v1 首轮加密用 5，后续改为 15 以减评测墙钟）。
- 结论表述约束：只能说「对齐官方配置后 CV-Bench 如何变化」，**不能**归因到
  单独某一项；若有明确改善，再用 `rollout.n=4 + token-mean` 一条实验做拆分。
- 当前阶段：**训练已完成 300/300**（2026-08-18 03:12 在 node-B 启动，冷启动，与 v1 并行；
  2026-08-18 19:46 训练进程正常退出，vLLM 已 shutdown）
- 当前任务：无训练任务；结论待写入 registry 与报告
- 运行窗口/负责人：cursor-window-mvopsd（从 node-A 通过 `ssh opsd2` 操作 node-B）
- GPU/节点：node-B 8 卡**已释放**（无 `run_mvopsd` / `main_ppo` 残留进程，显存 0 MiB）；
  node-A 8 卡同为空闲
- 活跃进程或作业 ID：无（已结束；原 node-B PID 565189）
- 保留的 checkpoint：**已全部删除**（2026-08-18，释放约 1.1T）
- 训练日志：`logs/train/20260818_qwen35base_mvopsd_v1_rollout8_main/`（node-B）
- smoke 验收（2026-08-18 02:50，**未用 `SMOKE=1`**，见 ISSUE-102）：
  2 步 / batch 64（真实值）/ n=8 / TEST_FREQ=1 / 64 条 val，已通过
  - `max_memory_allocated` 56.0 → **59.6 GB**（reserved 67.9），95 GB 卡余量充足。
    v1 在 n=4 时约 60.4 GB → **n=8 几乎不增显存**，因为
    `ppo_micro_batch_size_per_gpu=1`，峰值由微批而非 n 决定。
    registry 里「若 OOM 退到 ROLLOUT_N=6」的备选无需启用。
  - 稳态 172.6 s/步（`timing_s/gen` 32.7 s）；n=4 折算约 124 s/步，
    故 n=8 代价约 1.4 倍而非 2 倍。300 步预计 16-18 h。
  - `teacher_probe` 在 step 1 与 step 2 完全相同（0.04456961154937744）→ teacher 冻结
  - `answered/frac` 全程 1.0，`judge/pending` 与 `woken` 均为 0
  - `pg_loss = vopd_loss = 0.042`，`policy_fallback_fraction` 恒 0 → 纯 vopd 分支
  - `response_length/mean` ≈ 280，`clip_ratio` 3.1%（正式跑需对照 v1 验证
    hypothesis 里「n=8 使 clip_ratio 略升」的预期）
- 对照假设：两处偏离都可能助长散文漂移——n=8 让短答路径更容易被采到，
  token-mean 则不再把 124-token 的 spar_32view 与 342-token 的 spar_3view 等权。
  改回官方设置后 CV-Bench 2D 下降预期减缓；同时 `response_length/clip_ratio`
  可能略升，需在 wandb 对照 v1。
- 启动顺序（全部在 node-B 上执行）：
  1. 确认数据集与 `models/{Qwen3.5-4B,gpt-oss-120b}` 已传完、8 卡空闲、无残留
     `run_mvopsd` / judge 进程
  2. **Smoke**（验证 n=8 不 OOM）：
     `SMOKE=1 ROLLOUT_N=8 EXPERIMENT_NAME=20260818_qwen35base_mvopsd_v1_rollout8_main scripts/opsd/run_mvopsd.sh main`
  3. **正式**（默认 `TEST_FREQ=15`，checkpoint 每 15 步存一次）：
     `ROLLOUT_N=8 EXPERIMENT_NAME=20260818_qwen35base_mvopsd_v1_rollout8_main scripts/opsd/run_mvopsd.sh main`
- 输出目录：`checkpoints/20260818_qwen35base_mvopsd_v1_rollout8_main`、
  `rollouts/`、`logs/val/`、`logs/train/` 同名子目录
- wandb：`slcheng/MV-OPSD`，run 名 `20260818_qwen35base_mvopsd_v1_rollout8_main`
- step 0 基座分（全量 2,638 条）：**combined 87.21**，`answered/frac` 1.0，judge 未唤醒。
  **跨机基线偏移**：v1 在 node-A 测得 86.96，本实验在 node-B 测得 87.21，差 +0.25 分。
  逐子任务净差 6 行 / 2,638（ADE20K/Count 完全相同，Relation 与 3D 升、COCO/Count 降），
  方向正负都有，属贪心解码在不同 kernel 与批次调度下的抖动，不是搬迁缺陷。
  **因此两实验 CV-Bench 差异小于约 0.3 分时不得归因于配置改动**，应看趋势与斜率，
  或各自用 step 0 归一化后再比。详见 registry 的 `cross_node_baseline_offset`。
- 使用的验证集：`cvbench_val.parquet` 的 **native `(A) 3` 排版**版本，即现已改名保留的
  `cvbench_val_native.parquet`。该文件在本实验结束后（2026-08-18 20:11）被重建为
  lmms_eval 排版，故**复现本实验曲线必须显式指定 `VAL_FILE` 指向 native 版**。
- 阻塞项：无
- 下一步：把 300 步结论写入 registry 的 `metrics`/`conclusion` 与报告，
  并对照 v1 同 step 的 `val-core/cvbench/combined/acc`（注意上面 +0.25 的跨机基线偏移）；
  另按 ISSUE-102 修 `run_mvopsd.sh:58`（v1 与本实验均已结束，可安全改动并同步两机）
- 最后更新：2026-08-18 20:15 UTC+8（订正为已完成 300/300 并释放 GPU；记录所用 val 排版）

### CV-Bench 训练内 vs lmms_eval parity 测量（工具链验证，非独立实验）

- 动机：训练内评测已按 lmms_eval 的 task 定义改造（prompt 用 `cvbench_doc_to_text`、
  官方 combined 聚合、min/max pixels 对齐 `qwen3_5`），但从未量化过「我们的数字
  与 lmms_eval 会公布的数字差多少」。用户 2026-08-18 决策：**先不改训练循环**，
  拿已有 checkpoint 跑一次离线 lmms_eval 做 diff。
- 方法：把差异拆成两项，分别测量。
  1. **解析器差异**（零 GPU 成本）：`logs/val/<exp>/<step>.jsonl` 存了全部 2,638 条
     训练内生成，用两套 parser 重打同一批文本。工具
     `scripts/opsd/tools/cvbench_parity_report.py`（新）。
     val dump 不含 `data_source`，按索引从 val parquet 恢复；
     `data.validation_shuffle: False` 是前提，且对 2,638 条 gt 做了对齐断言（零错位）。
  2. **推理差异**（需 GPU）：merge step 300 → HF，跑原版 lmms_eval（16 token、
     它自己的 parser），再把同一批短生成用我们的 parser 重打，隔离生成长度效应。
- **2026-08-18 结论（推理差异项已补齐，见 LESSON-017）：主因不是解析器，是生成长度上限。**
  官方 `cvbench.yaml` 给 16 token，训练内给 1024。step 300 的模型已改成
  「先写长推理、末尾单独一行 `(X)`」，16 token 直接切断，离线那批生成里一个选项字母都没有。
  | 口径（prompt / 生成上限 / 解析器） | 基座 Qwen3.5-4B | v1 step 300 |
  | --- | --- | --- |
  | 官方 / 16 / legacy（官方协议本身） | 62.85 | 5.00（全靠把 "Based" 读成 B） |
  | 官方 / 1024 / legacy | 73.83 | 44.20 |
  | 官方 / 1024 / word_boundary | 85.67 | 79.86 |
  | 我们 / 1024 / word_boundary（**已作废**的旧离线默认） | ~~85.95~~ | — |
  | **我们 / 1024 / boxed last-line（当前离线默认，2026-09-01 重测）** | **88.07** | — |
  | 训练内（native prompt + vLLM） | 86.96 | 83.57 |
  基座 62.85 → 86.96 的拆解（2026-08-18 word_boundary/训练内口径，**不含** boxed last-line）：
  放开长度 +11.0、换解析器 +11.8、prompt 措辞 +0.3、引擎与 native 排版 +1.0。
  **长度与解析器两项各占一半，缺任一项都会得出错误结论。**
  **对外基座 CV-Bench 现以 boxed last-line 的 88.07 为准**，85.95 不得再引用。见 LESSON-031。
  基座→step 300 的跌幅完全取决于口径：官方协议 -57.9、官方+1024 -29.6、
  换我们的解析器后 -5.8、训练内 -3.4。即模型主要丢的是「简洁作答」的能力，
  而 legacy 解析器会额外重罚散文，两者叠加才放大成 -58。
- **工具链改动（2026-08-18，已完成并测试）**：`src/lmms_eval/tasks/cvbench` 增加
  `CVBENCH_PROTOCOL` 开关——`spatialstack`（默认，对齐训练内）/ `lmms_legacy`（复现官方数字）。
  prompt、`max_new_tokens`、`until`、解析器由同一开关一起决定，避免跑出半套协议；
  新增 `cvbench_answered` 指标。回归测试 `scripts/opsd/tests/test_lmms_cvbench_protocol.py`。
  `CVBENCH_MAX_NEW_TOKENS` 可下调默认的 1024 以省墙钟。
- **题面排版差已消除（2026-08-18 20:11，node-A）**：v1 与 rollout8 均已训满 300 步、
  无进程占用该文件后，用 `python3 scripts/opsd/build_cvbench_val_parquet.py`
  （默认 `--prompt-style lmms_eval`）重建了训练内验证集，两侧题面现已逐字一致。
  - 产物：仅保留 `data/eval/cvbench_verl/cvbench_val.parquet`（2,638 条，`prompt_style
    = lmms_eval`）。此前的 native / 64 条 / pre_promptstyle 备份已于 2026-08-18 删除；
    复现 v1/rollout8 的 native 题面需 `build_cvbench_val_parquet.py --prompt-style native`。
  - 回归测试：`test_cvbench_val_parity.py` 的第 6 项与
    `test_lmms_cvbench_protocol.py` 的第 3 项此前因排版不符只作软提示，现均已转为
    真断言并通过（2638/2638 逐条匹配）；`test_validation_judge.py`、
    `test_sample_provenance.py`、`test_mvopsd_data.py` 复跑全绿。
  - 影响范围：`mvopsd.yaml` 的 `val_files` 与 `run_mvopsd.sh` 的 `VAL_FILE` 默认值
    均指向 `cvbench_val.parquet`，因此**下一个实验默认即为对齐口径，无需改配置**。
    反过来，v1/rollout8 若被重启续训会静默换成新题面（ISSUE-103 的同类风险），
    两者都已训满 300 步，不应再续训。
- 已得结果（v1，解析器项）：**差距随训练增长，不是固定偏移**
  - step 0：ours 86.96 vs 同生成 legacy 72.15（差 14.8pp）
  - step 300：ours 83.57 vs 同生成 legacy 47.11（差 **36.5pp**）
  - 机制同 ISSUE-003 / LESSON-011：长 CoT 里 legacy 取首个 `[A-F]`，
    "**B**ased on the provided images" 被读成投给 B。训练后模型更爱写散文，
    误判率随之上升，故不能用固定 delta 校正。
  - 工具自检：两个 step 的 recompute 都与 trainer 落盘的 combined 完全一致。
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：node-A 8 卡（v1 结束后空闲；不影响 node-B 的 rollout8）
- 产物：`output/20260817_qwen35base_mvopsd_v1_main_global_step_300_hf`（10.35 GB，已合并）、
  离线评测输出 `logs/eval/mvopsd_v1_step300_parity/`
- 阻塞项：无
- 下一步：把上表与 LESSON-017 写入 registry 与
  [`reports/MV-OPSD_实验报告.md`](reports/MV-OPSD_实验报告.md)；
  对外汇报 CV-Bench 时明确标注用的是哪个协议。
  排版差已消除，但**引擎差异（vLLM vs transformers）仍未量化**，
  协议对齐后离线与训练内仍差约 1 分，下一次离线评测应把这一项单独测出来。
- 最后更新：2026-08-18 20:15 UTC+8（重建 val parquet，消除题面排版差；回归测试转为真断言）

### 训练内评测改造（工具链变更，非独立实验）

- 动机：v0 训完 20 小时后才看到 benchmark 崩了，而 step 50 的 CV-Bench 3.62%
  当时就足以叫停。详见 LESSON-009。
- 改动：CV-Bench 进 verl `_validate()`，每 10 步一次，wandb 观测。
  - `scripts/opsd/build_cvbench_val_parquet.py`（新）：CV-Bench → verl val parquet
  - `scripts/opsd/tests/test_cvbench_val_parity.py`（新）：与离线 lmms_eval 对齐验证
  - `verl_pkg/verl/trainer/ppo/cvbench_metrics.py`（新）：官方 2D/3D combined 聚合
  - `scripts/opsd/mvopsd_reward.py`：按 `cvbench/` 前缀路由打分
  - `mvopsd.yaml`：`test_freq: 10`、`val_before_train: True`、`logger: [console, wandb]`
  - `run_mvopsd.sh`：`TEST_FREQ` / `VAL_FILE` / `WANDB_MODE` 开关 + wandb 登录预检
- 对齐验证：回放 4 个 checkpoint 的离线生成结果，combined 分数逐一精确复现
  （36.53 / 3.90 / 6.77 / 3.62），2638 条 prompt 与离线字节一致。
- wandb：已登录，账号 `slcheng`，凭据在 `~/.netrc`，run 落个人 entity
  `slcheng/MV-OPSD`（不设 `WANDB_ENTITY`）。已验证能真正同步到云端。
  换机器或换人时脚本会预检，未登录直接报错退出，不会等 vLLM 起完才失败。
  注意 `wandb login` 在非交互 shell 下不读管道，key 要作为参数传。
- 2026-08-17 21:30 增补：训练内评测接上 gpt-oss-120b judge，与离线协议同一套分级判分。
  - 资源：judge 不单独占卡。`run_mvopsd.sh` 在训练前把它以 vLLM sleep 模式停在同样 8 卡上，
    训练期间不占显存。让卡的窗口是 verl 自己开的——`generate_sequences()` 收尾就把 rollout
    engine 睡下、下次开头再唤醒——所以验证生成结束时显存本就是空的，
    judge 直接醒来判分、判完睡回去，训练侧不需要任何交接动作（见 ISSUE-011）。
  - 触发：只有规则层读不出选项字母的行才送 judge，少于 `min_rows`（默认 16）不唤醒。
    健康的 run 里规则层能读完全部行，judge 一次都不会醒——它恰好在模型开始跑题时才启动。
  - 新增指标：`val-aux/cvbench/judge/{pending,woken,graded,unparsed,seconds}`；
    打开 `audit_all_rows` 会把全部行送判并报 `rule_disagree_frac`，用判官反查规则层，
    但分数仍以规则层为准（审计不能变成打分）。
  - 失败一律降级为规则分：judge 起不来、唤不醒、判分抛异常、回复读不出，
    都只丢指标不丢训练；judge 即使判分失败也保证被睡回去，否则下一次 rollout 会 OOM。
  - `sleep_level` 必须是 1，不能是 2：见 LESSON-014 / ISSUE-010。
  - 相关文件：`scripts/opsd/mvopsd_judge.py`（新）、
    `scripts/opsd/tools/check_judge_handshake.py`（新，握手自检）、
    `scripts/opsd/tests/test_validation_judge.py`（新，失败路径回归）、
    `ray_trainer.py::_maybe_judge_validation`、`mvopsd.yaml: trainer.validation_judge`、
    `run_mvopsd.sh` 的 `JUDGE=` 开关（默认开，`JUDGE=0` 退回纯规则）。
- 状态：改造完成并冒烟通过，**正式训练未启动**，8 卡空闲，由用户决定启动时机。
- 最后更新：2026-08-17 21:30 UTC+8

## 窗口接管规则

1. 启动任务前填写负责人、GPU/节点、进程或作业 ID。
2. 若已有活跃负责人或进程，先确认任务状态，不重复启动。
3. 接管任务前阅读对应 registry 的 `handoff`、`issues` 和 `activity_log`。
4. 停止、失败或完成后立即更新本文件，并把细节写回 registry。
5. 本文件中的状态与 registry 冲突时，以实际进程和产物为准，随后修正两处记录。

