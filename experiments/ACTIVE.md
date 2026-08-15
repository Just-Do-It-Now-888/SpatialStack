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
- 当前阶段：数据与训练流水线已就绪，P0-5 smoke 已通过，正式训练未启动
- 当前任务：起 `scripts/opsd/run_mvopsd.sh main` 的 300 步训练
- 运行窗口/负责人：cursor-window-mvopsd
- GPU/节点：单机 8 卡，训练时整机独占（当前空闲）
- 活跃进程或作业 ID：无
- 输出目录：`checkpoints/20260816_qwen35_mvopsd_v0_{main,noprivilege}`
- 阻塞项：无
- 下一步：提交代码并把 `git_commit` 写回 registry，再启动 main arm
- 最后更新：2026-08-16 03:40 UTC+8

## 窗口接管规则

1. 启动任务前填写负责人、GPU/节点、进程或作业 ID。
2. 若已有活跃负责人或进程，先确认任务状态，不重复启动。
3. 接管任务前阅读对应 registry 的 `handoff`、`issues` 和 `activity_log`。
4. 停止、失败或完成后立即更新本文件，并把细节写回 registry。
5. 本文件中的状态与 registry 冲突时，以实际进程和产物为准，随后修正两处记录。

