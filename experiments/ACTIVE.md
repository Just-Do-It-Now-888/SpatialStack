# 当前实验与窗口协作状态

所有窗口在启动、停止或接管实验任务时更新此文件。这里只记录当前状态；详细参数、结果和问题保存在对应 registry YAML。

## 进行中的实验

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

### `20260816_qwen35_mvopsd_v0`

- Registry：`experiments/registry/20260816_qwen35_mvopsd_v0.yaml`
- 当前阶段：**main arm 训练中**（300 步），训练退出后自动接评测；
  noprivilege 对照组已取消（用户决定，2026-08-16 04:08）
- 当前任务：`scripts/opsd/run_mvopsd.sh main`，代码版本 `10f3440`
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：单机 8 卡整机独占（0-7 全部占用，勿启动其他 GPU 作业）
- 活跃进程或作业 ID：PID 3671731（`python3 -m verl.trainer.main_ppo`），2026-08-16 03:54 启动
- 输出目录：`checkpoints/20260816_qwen35_mvopsd_v0_main`，日志
  `logs/train/20260816_qwen35_mvopsd_v0_main/train_20260816_035443.log`，
  rollout `rollouts/20260816_qwen35_mvopsd_v0_main`
- 已排队的后续作业：`scripts/opsd/after_train_eval.sh`（PID 3716202），
  守着训练进程退出后自动 merge `global_step_300` 并依次跑
  「训练模型」与「SFT 基线」两条 VSI-Bench 帧预算曲线（1/2/4/8/16/32 帧），
  日志 `logs/train/20260816_qwen35_mvopsd_v0_main/after_train_eval.log`
- 阻塞项：无
- 下一步：训练约 6.5 小时后结束（预计 10:40 前后），评测自动开始；
  评测期间同样整机独占，勿插入其他 GPU 作业
- 最后更新：2026-08-16 04:10 UTC+8

## 窗口接管规则

1. 启动任务前填写负责人、GPU/节点、进程或作业 ID。
2. 若已有活跃负责人或进程，先确认任务状态，不重复启动。
3. 接管任务前阅读对应 registry 的 `handoff`、`issues` 和 `activity_log`。
4. 停止、失败或完成后立即更新本文件，并把细节写回 registry。
5. 本文件中的状态与 registry 冲突时，以实际进程和产物为准，随后修正两处记录。

