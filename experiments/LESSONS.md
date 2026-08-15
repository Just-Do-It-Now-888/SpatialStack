# 跨实验问题与经验

此文件记录会影响多个实验的通用问题。单次实验特有的问题写入对应 registry YAML 的 `issues`。

## LESSON-001：代码版本无法追溯

- 状态：未解决
- 现象：当前项目目录没有 `.git` 元数据，无法记录实验对应的 commit。
- 影响：不同窗口修改代码后，无法准确判断某个结果使用了哪一版实现。
- 临时措施：在 registry 的 `activity_log` 中记录改动文件和时间。
- 预防措施：恢复 Git 管理后，每次实验记录 `git_commit` 和 `git_dirty`。

## LESSON-002：最终模型与中间 checkpoint 的选择

- 状态：已规避
- 现象：评测入口要求模型目录包含 `config.json`；训练未完成时，最终输出目录可能尚不可用。
- 处理方式：优先检查最终输出目录，否则选择数字最大的 `checkpoint-*`。
- 预防措施：评测启动前验证所选目录存在 `config.json`，并把实际路径写入 registry。

## LESSON-003：离线训练环境会影响评测

- 状态：已规避
- 现象：训练使用 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；若评测继续继承，缺失的评测资源可能无法下载。
- 处理方式：评测前取消这两个环境变量。
- 预防措施：通过 `scripts/train_eval.sh` 或 `scripts/eval_only.sh` 启动，并记录实际环境。

## LESSON-004：实验 ID 与历史输出目录不一致

- 状态：历史兼容
- 现象：当前实验 ID 为 `20260813_qwen35_geo_baseline`，历史产物仍位于 `output/spatialstack_qwen35_train`。
- 影响：仅根据实验 ID 搜索目录可能遗漏产物。
- 处理方式：以 registry 的 `artifacts` 路径为准，不移动现有 checkpoint。
- 预防措施：新实验的训练和评测目录统一使用 `experiment_id`。

