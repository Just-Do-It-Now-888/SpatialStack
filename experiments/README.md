# SpatialStack 实验管理

此目录只保存可复现所需的轻量元数据，不保存模型权重、数据集、缓存或完整日志。

## 目录约定

```text
experiments/
├── README.md
├── ACTIVE.md                   # 当前任务、负责人、GPU 与作业 ID
├── LESSONS.md                  # 跨实验复用的问题与预防措施
├── registry/                 # 每种实验设置一个文件
│   └── <experiment_id>.yaml
└── templates/
    └── run.yaml              # 新实验记录模板

output/<experiment_id>/       # SFT checkpoint 与训练器状态（不进入版本控制）
checkpoints/<experiment_id>/  # OPSD/RL checkpoint（不进入版本控制）
logs/train/<experiment_id>/   # 编排和训练日志（不进入版本控制）
logs/eval/<experiment_id>/    # 评估结果与样本日志（不进入版本控制）
```

当前已有产物不移动，避免破坏脚本和 checkpoint。旧路径通过 registry 中的 `artifacts` 字段建立映射；新实验开始采用统一命名。

## 实验命名

`experiment_id` 使用：

```text
YYYYMMDD_<model>_<variant>
```

例如：

- `20260813_qwen35_geo_baseline`
- `20260814_qwen35_no_geometry`
- `20260815_qwen35_opsd_reward_v1`

同一种模型设置的训练参数、训练结果、评测参数和评测结果全部写在同一个文件中。相同设置的重复运行可在 ID 后追加 `_r2`、`_r3`；设置不同则使用新的 ID。

## 新实验流程

1. 从 `templates/run.yaml` 复制一份到 `registry/<experiment_id>.yaml`。
2. 实验开始前填写目的、模型、数据集和 `training` 参数。
3. 用同一个 `experiment_id` 设置训练与评测输出路径，例如：

   ```bash
   EXPERIMENT_ID=20260813_qwen35_geo_baseline
   OUTPUT_DIR="./output/${EXPERIMENT_ID}" bash scripts/train/train.sh
   MODEL_PATH="./output/${EXPERIMENT_ID}" \
     OUTPUT_ROOT="./logs/eval/${EXPERIMENT_ID}" \
     bash scripts/evaluation/eval.sh
   ```

4. 训练结束后更新 `training`，评测结束后更新 `evaluation`，最后填写整体结论。
5. 若项目恢复 Git 管理，运行 `git rev-parse HEAD` 并填写 `git_commit`；有未提交修改时同时填写 `git_dirty: true`。

## 多窗口协作

1. 每个窗口开始实验工作前先读取 `ACTIVE.md`、`LESSONS.md` 和对应 registry。
2. 启动作业前在 `ACTIVE.md` 登记负责人、GPU、命令、作业 ID 和输出目录。
3. 出现错误时先写入 registry 的 `issues`，再进行重试。
4. 窗口交接或任务结束前更新 `handoff` 与 `activity_log`。
5. 可复用的问题预防措施同步整理到 `LESSONS.md`。

项目规则 `.cursor/rules/experiment-workflow.mdc` 会在新窗口中持续提供上述要求。

## 记录规则

- 参数以实际运行值为准，不只记录脚本默认值。
- 同一实验设置的训练和全部评测结果必须保存在同一个 registry 文件中。
- 只有模型结构、训练数据或关键超参数发生变化时才新建实验文件。
- 已解决的问题不得删除，应保留根因、解决方法和预防措施。
- `ACTIVE.md` 只维护实时状态，registry 保存完整历史。
- 不把绝对机器路径、访问令牌或密码写入 registry。
- `output/`、`checkpoints/`、`logs/`、`models/`、`data/` 和 `cache/` 均视为大文件目录。
- 失败实验也保留记录，写明错误摘要和下一步，避免重复踩坑。

## 当前实验索引

| Experiment ID | 训练 | 评测 | 主要结果 |
| --- | --- | --- | --- |
| `20260813_qwen35_geo_baseline` | 已完成 | 待运行 | 3308 steps，train loss 0.5245 |

