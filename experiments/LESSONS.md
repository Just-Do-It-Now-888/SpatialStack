# 跨实验问题与经验

此文件记录会影响多个实验的通用问题。单次实验特有的问题写入对应 registry YAML 的 `issues`。

## LESSON-001：代码版本无法追溯

- 状态：已解决（2026-08-16）
- 现象：当前项目目录没有 `.git` 元数据，无法记录实验对应的 commit。
- 影响：不同窗口修改代码后，无法准确判断某个结果使用了哪一版实现。
- 处理方式：已在项目根目录执行 `git init` 并提交基线快照 `73e2fb3`。
- 预防措施：每次实验在 registry 中记录 `git_commit` 和 `git_dirty`；启动训练前先提交。

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

## LESSON-005：本镜像的 flash-attn 无法跑非因果注意力

- 状态：已规避
- 现象：`flash_attn 2.7.4.post1+25.11` 的 `varlen_fwd` 在 `causal=False` 且 `out=None` 时抛
  `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`；
  传入预分配的 `out` 或 `causal=True` 都正常。语言塔是因果注意力所以不受影响，
  视觉塔是双向注意力，一旦用 `attn_implementation=flash_attention_2` 就必然崩。
- 影响：SFT 加载 Qwen3.5 时不传 `attn_implementation`，所以从未触发；verl 默认对整个模型强制
  `flash_attention_2`，训练在第一次 actor 前向就失败。
- 处理方式：`verl/workers/fsdp_workers.py` 新增 `override_config.vision_attn_implementation`，
  用 transformers 5 的映射写法 `{"": flash_attention_2, "model.visual": sdpa}` 只把视觉塔换成 SDPA。
  语言塔必须保留 flash_attention_2，因为 `use_remove_padding` 打包序列依赖 varlen 注意力。
- 预防措施：换镜像或升级 flash-attn 后，用
  `python -c "...flash_attn_varlen_func(..., causal=False)"` 复测一次再决定是否回退该配置。

## LESSON-006：verl 的 `resume_mode=auto` 会静默续跑

- 状态：已规避
- 现象：smoke 复跑时从上一次遗留的 `global_step_2` 恢复，日志直接从 step 3 开始，
  两个 arm 的对照结果会被上一次的权重污染。
- 处理方式：`scripts/opsd/run_mvopsd.sh` 在 `SMOKE=1` 时追加 `trainer.resume_mode=disable`；
  同时把 `_smoke` 后缀移到目录名计算之前，避免 smoke 产物写进正式实验目录。
- 预防措施：正式训练保留 `auto`（断点续跑需要），但启动前确认 `checkpoints/<experiment_id>` 为空或确实要续跑。

