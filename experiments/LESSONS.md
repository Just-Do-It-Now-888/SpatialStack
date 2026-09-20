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

- 状态：已规避；2026-09-01 补了一条反向教训
- 现象：训练使用 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；若评测继续继承，缺失的评测资源可能无法下载。
- 处理方式：评测前取消这两个环境变量。
- 预防措施：通过 `scripts/train_eval.sh` 或 `scripts/eval_only.sh` 启动，并记录实际环境。
- **反向（2026-09-01）**：缓存已齐时仍 `unset HF_HUB_OFFLINE`，lmms-eval `download()` 会对 Hub 做 `dataset_info` SSL 握手。镜像卡住时 rank 0 会挂数小时，其余 rank 在 barrier 空转（GPU 利用率高、显存约 1GB、无 `Model Responding`）。此时应 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 重跑，不要继续等握手。见 `20260901_vsibench_boxed_lastline` ISSUE-001。

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

## LESSON-007：训练环境跑不了评测，两者的 python 不是同一个

- 状态：已规避
- 现象：训练结束后自动接续的评测链 12 次调用全部秒退，`/usr/bin/python: No module named lmms_eval`。
- 原因：`lmms_eval` 以 editable 方式装在 miniconda3 的 `sr_opsd` 环境（指向 `src/lmms_eval`），
  而训练跑在系统 python3.12 + `~/.local` user site 上。`scripts/evaluation/eval.sh` 里的
  `QWEN35_ENV_ROOT=$HOME/.conda/envs/spatialstack-qwen35` 在这台机器上不存在，那段 PATH 补丁是空操作。
  以往评测都是人工在已激活 `sr_opsd` 的窗口里跑的，掩盖了这个依赖。
- 处理方式：`scripts/opsd/eval_frame_budget.sh` 自己 `source ~/miniconda3/etc/profile.d/conda.sh`
  并 `conda activate sr_opsd`，导入失败就立即退出；同时固定 `HF_HOME=~/.cache/huggingface`
  （基线数字就是在这个缓存上测的）并 `unset PYTHONPATH`（verl 的路径会遮蔽环境内的包）。
- 预防措施：无人值守的链式脚本不要假设继承来的环境可用，先 `python -c "import <关键包>"` 探一次，
  失败就报错退出，而不是让后面十几个作业逐个空转。
- **订正（2026-09-10，LESSON-041）**：评测仍常在 `sr_opsd`；**此后所有训练也必须在 `sr_opsd`**。
  原先「训练不要在 sr_opsd 里起、走系统 python + ~/.local」是为了避开与评测混用 ray/torch。
  用户已改为统一训练环境，以便对齐历史 SpatialStack / 67.65 侧的第三方库栈。
  不要再启动 `spatialstack-qwen35`（geo SFT 20260909）或 `vision-opd`（几何 MV-OPSD 20260909 k2）做新训练。

## LESSON-009：训完再评测等于盲飞，评测要进训练循环

- 状态：已解决（2026-08-17）
- 现象：MV-OPSD v0 训练 7h55m，`vopd_loss` 一路收敛到 0.003，看起来完全正常；
  评测要等 merge FSDP checkpoint、切 `sr_opsd` 环境、再占满 8 卡跑 lmms_eval，
  加上 ISSUE-001 的链条没启动，实际是训练结束约 20 小时后才看到
  VSI-Bench 1.29%。事后补测中间 checkpoint 才发现 CV-Bench 在
  step 50 / 100 / 150 分别只有 3.62% / 3.90% / 6.77%——**第 50 步就已经能判死刑，
  却白烧了后面 7 个多小时**。
- 原因：`mvopsd.yaml` 里 `test_freq: -1`、`val_files: []`、`logger: ["console"]`，
  训练期间唯一可见的量是蒸馏 loss。而 JSD 收敛与 benchmark 正确率根本不是一回事：
  student 完全可以稳定地匹配上一个已经被带偏的 teacher 分布。
  离线评测链之所以贵，是因为它假设必须有一份 HF 权重——但 verl 的 `_validate()`
  直接用训练进程里已经在跑的 vLLM engine 生成，不需要 merge、不需要子进程、不占额外的卡。
- 处理方式：把 CV-Bench 转成 verl 的 val parquet
  （`scripts/opsd/build_cvbench_val_parquet.py`），
  打开 `test_freq: 10` + `val_before_train: True` + `logger: [console, wandb]`。
  `val_before_train` 顺带给出 step 0 的基座分，这正是 v0 缺的对照。
- 预防措施：任何跑得比一小时长的训练，启动前先回答「第一个能证伪的数字在第几分钟出现」。
  没有答案就不要启动。评测子集可以小，但必须在循环里。

## LESSON-010：复刻 benchmark prompt 要以运行时日志为准，不能只读 yaml

- 状态：已规避（2026-08-17）
- 现象：给 CV-Bench 构造 verl val parquet 时，按 `cvbench.yaml` 的
  `lmms_eval_specific_kwargs.default.pre_prompt: ""` 写成了空 prompt 前缀，
  与离线实际评测用的 prompt 不一致。
- 原因：`cvbench_doc_to_text` 里是
  `pre_prompt = kwargs.get("pre_prompt", "") or "These are frames of a video."`。
  `""` 是 falsy，`or` 让它回落到默认值。yaml 显式写的空串**从来没有生效过**，
  真实 prompt 一直带着 "These are frames of a video." 这一行
  （对单图的 CV-Bench 来说本身就是句错话，但基线数字就是在它上面测出来的）。
- 处理方式：从 `logs/eval/.../\*_samples_cvbench.jsonl` 的 `input` 字段反读真实 prompt，
  并写 `scripts/opsd/tests/test_cvbench_val_parity.py` 逐条比对到字节一致。
- 预防措施：复刻任何评测协议时，用旧日志里落盘的样本做 parity 测试，
  不要只读配置文件——`or` / `get(k, default)` / 后处理都可能让配置值失效。
  同一个测试还应回放旧的生成结果，验证新打分器能复现旧分数。

## LESSON-013

移植外部评测协议时，默认值必须等于上游，本项目的改动一律做成显式开关。

MV-OPSD 把 Vision-OPD 的 eval 移植过来后，为绕开本地部署限制和 v0 的字符串假象，
在判分层逐项加固：词边界字母匹配、送判回答截断、verdict 归一化、gpt-oss 参数。
每一项单独看都有理由，但它们都被设成了默认值，于是「本项目口径」悄悄取代了
「可对照口径」，而分数差异真实存在（step 300：72.19 vs 69.85）。

其中最值得记的是：截断送判回答这一项，根因其实是我们自己把 judge 的
`max-model-len` 设成了 8192，而上游根本不启动 judge，窗口是部署方的选择。
**把自己的部署限制当成协议差异，是最容易固化的错误。**

做法：默认对齐上游；加固作为显式开关；每次运行都打印「若切换到另一模式会有多少行改判」，
让偏离可见但不改变上报分数。详见 ISSUE-009。

## LESSON-011：先看回答形态，再看分数；分数能被字符串规则复现就不是分数

- 状态：已解决（2026-08-17）
- 现象：MV-OPSD v0 的 CV-Bench 曲线 3.62 → 3.90 → 6.77 → 11.67 → 26.95 → 36.53，
  单调上升，看着像模型在恢复。实际上这条曲线**零残差**等于一条与图像无关的字符串规则：
  「回答以单词 Based 开头就预测 B，否则算错」。6 个 checkpoint 逐一复现，误差 0.00。
- 原因：两个独立缺陷叠加，任何一个单独存在都不会这么隐蔽。
  1. **截断**：评测用 `max_new_tokens=16`，而训练后的模型回答平均 170 词，
     约 90% 的回答在给出答案之前就被砍断，落盘的只有前言。
  2. **解析器无词边界**：lmms_eval 的 `extract_characters_regex` 用
     `re.search(r"[ABCDEF]", s)` 取全文第一个 A-F 大写字母，不检查它是不是独立的选项标记。
     散文里唯一能高频命中的词就是句首的 "**B**ased"，于是 "Based on the provided images..."
     被判成选了 B；而 "To determine..." 不含 A-F，判为空。
  金标里 B 占 41.8%，所以这个假象的天花板是常量 B 预测器的 42.57 分。
  训练过程中前言模板从 "To determine..." 漂移到 "Based on..."，分数就"涨"上去了。
- 影响：如果不做这次诊断，会把 36.53 当成「MV-OPSD 正在恢复、再训几百步就能追上基座 76」，
  实际上再训只会趋近 42.57 并永远停在那儿。
- 处理方式：诊断脚本 `scripts/opsd/tools/analyze_cvbench_curve.py`（回答形态、截断率、
  解析失败率、误抓率）与 `verify_based_artifact.py`（零假设复现）。
- 预防措施：
  1. 拿到任何 benchmark 分数前，先打印**回答形态分布**：多少条给出了可判定答案、
     平均长度、截断率。可判定回答为 0 时，分数无论多少都是噪声。
  2. 评测的 `max_new_tokens` 必须由**当前模型的实际回答长度**决定，不能沿用基座的取值——
     训练会改变回答长度，基座能用 16 token 不代表训练后也能。
  3. 每条曲线都跑一次零假设：能否用与输入无关的常量或字符串规则复现？能复现就作废。
  4. 复用上游 benchmark 的解析器前读一遍它的正则，尤其注意缺词边界的 `re.search`。

## LESSON-012：损失的构成要读实际代码路径，不能从配置项名字推断

- 状态：已解决（2026-08-17 20:10 首次记录，2026-08-17 21:45 更正定性）
- 现象：MV-OPSD v0 训练 300 步 `vopd_loss` 漂亮地收敛到 0.003。事后查
  `rollouts/*.jsonl` 才发现每步 256 条 rollout 的 `score` **全程恒为 0**，
  得分 >0 的比例 0.0%。
- 首次归因（**已证伪，保留以记录判断过程**）：当时把配置读成
  `Loss = 0.5·JSD + 0.5·policy_loss`，据此判断 policy 项 300 步没有梯度信号，
  训练退化为对 EMA teacher 的无约束风格自蒸馏，并把「先修 reward」列为重训阻塞项。
- 更正后的事实：v0 的配置下 reward **根本不进梯度**，恒零不影响训练信号。
  1. `actor.self_distillation.alpha` 不是 JSD 与 policy loss 的混合权重，而是
     `core_algos.py::compute_self_distillation_loss` 里广义 JSD 的插值系数
     （`torch.lerp(kl_student, kl_teacher, alpha)`）。
  2. `dp_actor.py` 只在 `policy_fallback_mask`（样本缺 teacher 图像）非空时才加 GRPO 项；
     `teacher_always_on: True` + `fallback_to_policy_loss_on_missing_teacher: False`
     下该 mask 恒为空，走的是 `pg_loss = vopd_loss` 这一支。v0 日志里
     `actor/policy_fallback_fraction` 全程 0.0 就是这条路径的直接证据。
  3. `scripts/opsd/mvopsd_reward.py` 的文件头本来就写明「v0 is reward-free」，
     该分数只作诊断，Phase 1 的 GRPO 消融 arm 才会把它当真实 reward 用。
- 实际影响：reward 恒零让「训练集正确率」这个诊断量失效（看不出模型答得对不对），
  但没有制造 LESSON-009 里那种「JSD 收敛与正确率无关」的机制——
  那个机制在纯自蒸馏下本来就成立，不需要 reward 缺失来解释。
- 预防措施：
  1. 断言「某一项没有梯度」之前，先找到那一项进入 `loss` 的代码行；配置项名字
     （`alpha`、`policy_loss`）和实际语义可以完全无关。
  2. 冒烟阶段仍把 `reward 均值` 和 `reward>0 比例` 列为必看指标，但判定标准是
     「这个分数是不是应该非零」——reward-free 的 arm 恒零是预期行为，
     把它当阻塞项会白等一天。
  3. 写进 registry 的 `root_cause` 若标了「待核对」，在它成为阻塞项之前必须先核对完。

## LESSON-014：模型「活着」不等于权重还在，睡眠唤醒必须用结果验收

- 状态：已解决（2026-08-17）
- 现象：训练内 judge 让 gpt-oss-120b 与训练分时共享 8 卡，用 vLLM sleep level 2 让它
  在验证之外不占显存。唤醒后 `/health` 200、`/is_sleeping` false、显存回到
  24.8 GiB/卡，一切指标正常，但模型对任何输入都回 `!!!!!!!!`，
  整轮验证会被一个「看起来完全健康」的判官判成全错。
- 原因：level 2 睡眠丢弃权重、唤醒时重建，而这份 MXFP4 权重经 Marlin 重打包后
  无法被该路径正确还原。level 1 把权重整块搬到主机内存再原样搬回，不重建，因此正常。
- 处理方式：`sleep_level` 默认 1，代价是常驻 61 GB 主机内存（本机 1.6 TB）；
  实测睡 9s、醒 1s，睡下残留 5.4 GiB/卡。
- 预防措施：
  1. 任何 sleep/wake、权重卸载重载、offload/reload 链路，验收标准必须是**模型输出**，
     不能是进程存活、健康检查或显存曲线——这三样在权重损坏时全部正常。
  2. 用一对「一定判对 + 一定判错」的样本在睡前睡后各跑一次，两次都对才算通过；
     只跑睡后会把「判官一直是坏的」和「睡眠弄坏了判官」混为一谈。
  3. 见 `scripts/opsd/tools/check_judge_handshake.py`；换模型、换 vLLM 版本、
     改 sleep level 后都要重跑。

## LESSON-008：`set -u` 与 conda activate 钩子不兼容

- 状态：已规避（2026-08-17）
- 现象：脚本里自动 `conda activate sr_opsd` 时秒退，日志只有一行
  `~cuda-nvcc_activate.sh: line 41: NVCC_PREPEND_FLAGS: unbound variable`。
- 原因：`sr_opsd` 的 `etc/conda/activate.d/~cuda-nvcc_activate.sh` 先用
  `${NVCC_PREPEND_FLAGS+x}` 判断是否需要备份，随后却无条件展开 `${NVCC_PREPEND_FLAGS}`。
  在 `set -u` 下这一行直接终止调用方脚本。人工在已激活的窗口里跑不会重跑钩子，所以长期未暴露。
- 处理方式：激活前后临时 `set +u` / `set -u`，只在 source 与 activate 这两行放宽。
- 预防措施：脚本里 source 任何第三方环境脚本（conda、CUDA、module）前都先 `set +u`；
  这类脚本不受我们控制，不能假设它们对 `-u` 安全。改完用一次 `LIMIT=8` 的冒烟验证链路，
  不要直接提交数小时的正式作业。

## LESSON-015：checkpoint 保存步长应与训练内评测对齐

- 状态：已采纳（2026-08-18）
- 现象：v1 用 `TEST_FREQ=5` 每 5 步打一次 CV-Bench，但 `SAVE_FREQ=50` 只在
  50/100/150… 落盘；想离线复现 step 15/25 的曲线时，没有对应 checkpoint。
- 处理方式：`scripts/opsd/run_mvopsd.sh` 在未显式设置 `SAVE_FREQ` 时默认
  `SAVE_FREQ=TEST_FREQ`；`TEST_FREQ=-1`（无训练内评测）时仍回退到 50。
  `mvopsd.yaml` 的 `save_freq` 与 `test_freq` 同为 15。
- 预防措施：registry 里同时记录 `test_freq` 与 `save_freq`，二者应相等；
  需要更稀疏的存档时显式设 `SAVE_FREQ`（须为 `TEST_FREQ` 的整数倍以免错位）。


## LESSON-016：数据配比失衡要按 rollout 长度估，不能按标注答案长度

- 状态：已修正（2026-08-18）
- 现象：`MV_OPSD_Experiment_Plan.md` §6.5.1 断言默认 `token-mean` 下 SPAR 3-view 会以
  29.4% 的样本拿走 **57.2%** 梯度、`vlm3r_scannet` 只剩 **0.9%**，并据此把
  `loss_agg_mode` 强制改成 `seq-mean-token-mean`，称其为「全源配比下唯一不能留默认值的参数」。
- 原因：该测算用的是**数据集标注答案**的 token 数（`vlm3r_scannet` 的 GT 只有 1 个 token）。
  但 on-policy 自蒸馏的 loss 只覆盖 **student 自己生成的 response token**，与 GT 长度无关。
  基座 Qwen3.5-4B 对所有源都写散文，`vlm3r_scannet` 的实际平均生成长度是 258 token，
  不是 1。前提错了，结论随之失效。
- 证据：v1 前 92 步共 23,552 条 rollout，用 Qwen3.5-4B tokenizer 实测（样本占比 → token 占比）：
  `spar_3view` 27.8% → 38.6%，`llava_hound_64k` 30.3% → 25.8%，
  `vlm3r_scannet` 24.4% → **25.5%**，`spar_32view` 16.5% → 8.3%，`vsi_appr_order` 1.1% → 1.8%。
  最大偏移 +10.8pp，而非计划预计的 +27.8pp；预计被饿死的 `vlm3r_scannet` 实际几乎不受影响。
- 处理方式：`mvopsd.yaml` 改回 verl 默认、也是 Vision-OPD 官方配置所用的 `token-mean`。
- 预防措施：凡是涉及「哪个源占多少梯度」的估算，一律基于**实际 rollout 的生成长度**统计，
  而不是数据集标注长度；在没有 rollout 之前只能作为待核对假设，不能写成必选配置的理由。
  相关口径：这条同样适用于按源统计 loss、截断率、response length 的所有分析。


## LESSON-017：离线 benchmark 与训练内曲线要按「整套协议」对齐，不能只对齐解析器

- 状态：已修正（2026-08-18）
- 现象：v1 step 300 的 CV-Bench，训练内报 **83.57**，同一 checkpoint 跑离线 lmms_eval
  只有 **5.00**。此前把两者的差异归因于解析器（训练内用 word_boundary，
  官方用首个 `[A-F]` 字节），并据此以为是个可估算的固定偏移。
- 原因：主因不是解析器而是**生成长度上限**。`cvbench.yaml` 的
  `generation_kwargs.max_new_tokens: 16`，而训练内是 1024。训到 300 步的模型已经
  改成「先写 500 字推理、最后单独一行 `(X)`」，16 token 一到就被切断，
  离线那批生成里**一个选项字母都没出现**（用我们的解析器重打 answered = 0.0%，得分 0.04）。
  官方解析器给出的 5.00 完全来自把 "**B**ased on the provided images" 读成投 B，
  2D 9.26 / 3D 0.75 就是「永远蒙 B」的形状。
  次要差异另有两项：官方 `pre_prompt: ""` 被 `or` 兜到视频默认句，
  每题实际都带 "These are frames of a video."；选项排版 `A. 3` vs 我们的 `(A) 3`。
- 证据（Qwen3.5-4B 基座，同一台机、同一 checkpoint，**2026-08-18 当时的离线默认**）：
  官方协议（16 token + 视频前缀 + legacy 解析）**62.85**；
  改为整套对齐后（1024 token + 无前缀 + word_boundary）**85.95**，answered 100%；
  训练内同一基座 **86.96**。即当时协议对齐后离线与训练内只差约 1 分，
  剩余部分是 prompt 排版（native vs lmms_eval）与 vLLM/transformers 的核差异。
  v1 step 300 用 1024 token 离线重跑并用我们的解析器打分为 **79.86**（answered 97.8%）。
- **订正（2026-09-01）**：离线默认已改为与 VSI 相同的 boxed last-line（MCA 句后接 `\boxed{}` 后缀，只认闭合 `\boxed{}` 否则末行选项）。该协议下基座 **combined 88.07**（2D 83.55，3D 92.58，作答 99.62%）。**85.95 不得再作为基座 CV-Bench 对外数字**；它只描述旧 word_boundary、无 boxed 后缀的那一次评测。见 LESSON-031。
- 处理方式：`src/lmms_eval/tasks/cvbench` 增加 `CVBENCH_PROTOCOL` 开关，
  `spatialstack`（默认，对齐训练内）与 `lmms_legacy`（复现官方数字）二选一。
  prompt、生成上限、`until`、解析器**由同一个开关一起决定**，不允许各自设定；
  新增 `cvbench_answered` 指标。回归测试
  `scripts/opsd/tests/test_lmms_cvbench_protocol.py`。
- 预防措施：跨协议对比一律先核「生成上限 / prompt 逐字 / 解析器 / 停止条件」四项，
  再看分数；任何一项不同都不能用固定 delta 折算。新增评测开关时把这几项绑在一个
  开关上，避免半套协议的组合能被跑出来。
  另注：`TaskConfig` 会给缺失的 `until` 填 `["\n\n"]`（第一个空行即停），
  在 16 token 下看不出来，放开长度后会把 CoT 砍在第一段。

## LESSON-018：VSI-Bench 的首词解析在数值题上会伪装成「答错」而不是「没解析出来」

- 状态：已修正（2026-08-20）
- 现象：LESSON-017 只修了 CV-Bench，VSI-Bench 仍是上游默认的 16 token +
  `fuzzy_matching(pred) = pred.split(' ')[0]`。对会先写推理的 checkpoint，
  这个组合同样失效。
- 原因：两半各自出错，且第二半**不留痕迹**。选择题拿到 "Based" 判错，尚可从
  分数异常低看出来；数值题走的是 `to_float(fuzzy_matching(pred))`，解析失败返回
  `None`，被 `except TypeError` 捕获后写成 `WORST_CASE_FOR_METRICS`，
  即 MRA = 0。也就是说**「读不出数字」和「数字答得极差」在结果里完全同形**，
  日志里看不到任何解析失败的迹象。VSI-Bench 十个题型里有四个是数值题，
  占 2,640 / 5,130 条。
- 处理方式：新增 `VSIBENCH_PROTOCOL` 开关（`spatialstack` 默认 / `lmms_legacy`），
  预算与解析器绑在一起；`spatialstack` 后来对齐训练内 val：4096 token、
  boxed last-line 后缀、`boxed_primary`（先打闭合 `\boxed{}`）。
  新增 `vsibench_answered` 指标，让解析失败与答错分开可见。
  **指标本身一行未改**：MCA 仍 exact match，NA 仍 `MRA:.5:.95:.05`，
  仍是题型间非加权平均。
- **订正（2026-09-01）**：离线 lmms-eval `spatialstack` 已与
  `vsibench_val_boxed_lastline.parquet` 对齐。此前无 boxed 后缀、只走
  `answer_tail` 的 dump 不可与新跑并列。`VSIBENCH_PROTOCOL=lmms_legacy`
  仍复现上游 16 token。vLLM harness（`eval_vsibench_vllm.py`）的
  `prompt_suffix` / `boxed_primary` 仍默认关闭，以免移动已发布的 52.58 锚点。
- 证据：回归测试 `scripts/opsd/tests/test_lmms_vsibench_protocol.py` 回放 5 个
  历史短答 run，新旧解析器打分逐个完全一致（如 `novggt_aligned` 两侧均 64.24），
  证明改动只移动了「去哪里找答案」，没有移动任何已发布的数字。
- 预防措施：
  1. 写解析器时**只在题目实际提供的选项字母里取答案**。VSI-Bench 的选项数是 2~4，
     照搬 CV-Bench 的 A-F 扫描会把正文里的 "C" 当成投票。
  2. 任何「解析失败」路径都要有独立计数，不能悄悄折叠进最差分数——
     否则分数下降时无法区分是能力退化还是格式崩坏。
  3. 新加 benchmark 协议时，先用历史短答样本回放，确认新解析器在旧数据上与旧解析器
     同分；不同分说明改的不只是解析位置。

## LESSON-020：训练内护栏必须与主指标同分布，否则它只会在没事时报平安

- 状态：已确认（2026-08-20）
- 背景：MV-OPSD 训练全程用 CV-Bench（2,638 条单图四选一）每 15 步打一次分作为护栏，
  主指标 VSI-Bench（5,130 条 32 帧视频）只在训练结束后离线测。
- 现象：`khalf` step 120 是两臂 33 个评测点里 CV-Bench 最高的（83.79%），
  但它在 VSI-Bench 上相对基座**崩塌约 19 分**（同 vLLM harness：52.58 → 33.38）。
  训练全程没有任何指标提示这场崩溃。
- 原因：两个 benchmark 压根不考同一件事。CV-Bench 单图、四选一、
  按提示词几个 token 就能答完；VSI-Bench 是 32 帧长视频推理。
  训练把模型从「直接给选项字母」（中位 4 token）改成了「写长推理」（中位 168~205 token），
  其中一部分收不住、滑进重复循环（逐帧枚举、连续数数），刷满任何预算也不给答案。
  这种失效只在长生成路径上暴露，单图 MCQ 的短生成路径完全碰不到。
- 证据：把预算从 1024 放宽到 4096 重测，分数没回升（34.82 → 33.38），
  answered 仍 93.51%、解析失败仍 333 条——原来撞 1024 的那批改撞 4096，
  说明是真退化而不是预算不够。分题型看 object_counting 截断 57~58%、
  object_rel_distance 约 35%，是整条长视频推理路径崩坏。
  回溯到 step 45，同一 60 场景子集上已经掉了 11 分（54.15 → 43.18），
  退化从早期就开始，不是训练末期事件。
- 处理方式：训练内评测改为 **VSI-Bench 全量 + vLLM + 只补规则未接受行的 judge**，
  跑在独立评测节点上（`scripts/opsd/watch_training_vsibench_eval.sh`），
  CV-Bench 训练内评测同时关闭。每个 checkpoint 除总分外落一份诊断：
  分题型分数、answered 率、截断率、输出长度分位数，以及按
  截断 / 解析失败 / 半对 / 事实错误分桶的原始样本。
- 预防措施：
  1. 护栏与主指标的**生成长度量级**要对齐。短生成护栏对长生成失效模式是盲的，
     而「刷满预算也不给答案」这类失效恰恰只在长生成里出现。
  2. 护栏不能只报一个分数。**答得出来的比例、截断率、输出长度中位数**
     这三项能在总分掉下来之前就看出模型开始跑偏；只看总分时，
     「答错了」和「没答」长得一模一样。
  3. 换掉护栏的成本没有想象中高：全量 VSI 在 8 卡 vLLM 上约 17 分钟，
     单 checkpoint 连 judge 约 25~30 分钟，放在独立节点上不占训练时间。

## LESSON-019：LLM judge 判「意思对不对」，benchmark 判「选项选没选对」，两者不能互换

- 状态：已确认（2026-08-20）
- 背景：为与 Vision-OPD 保持一致，给 VSI-Bench 接入 gpt-oss-120b judge
  （`scripts/opsd/tools/judge_vsibench.py`）。Vision-OPD 的级联是
  **规则层只判对、不判错**，规则没接受的行全部交给 Yes/No judge。
- 现象：在基座 anchor 上照搬该级联，overall 从 53.40 涨到 54.89（+1.49），
  且 **107 行由错判对、0 行由对判错**。分收益集中在方向题（+6~+9）。
  第一反应是「解析器漏判了很多」，但这个结论是错的。
- 原因：两层叠加。
  1. **级联在构造上只能加分**。规则层 accept-only，judge 只看被拒绝的行，
     `correct → wrong` 这条路径根本不存在。judge 只要偏宽松，分数就单向虚高，
     而且从结果里完全看不出来。
  2. **judge 回答的不是 benchmark 的问题**。Vision-OPD 的 prompt 问
     "response 与 answer 是否同义"，判的是整段话的语义；VSI-Bench 的 MCA 指标
     判的是「选了哪个选项」。模型常出现字母对、正文自相矛盾的情况
     （GT=B(left)，正文 "the whiteboard is to the **right**. B"），
     benchmark 记对，judge 记错。这不是谁出 bug，是两个不同的判据。
- 证据：加反向抽查（`--audit-accepted`，把规则**已接受**的行也送 judge）后，
  judge 在 300 条已接受行上推翻 29~32 条（约 10%），与 judge 在被拒绝行上
  10.8% 的翻转率几乎相同——说明 +1.49 无法与「单向施加的 judge 噪声」区分开。
  随后改为**抽取模式**（judge 只回答「选了哪个选项」，评分仍由 lmms_eval 的
  exact match 完成，且对全部 2,490 条 MCA 生效）：overall 53.40 → 53.47，
  仅 **+0.08**，翻转 11 对 7 近似对称。即规则解析器本身没有问题，
  那 +1.5 全部来自测量方式。
- 处理方式：`--mca-mode extract`（默认）让 judge 只做抽取，指标定义不动；
  `--mca-mode verdict` 保留 Vision-OPD 原级联供对照。数值题从一开始就只用抽取——
  MRA 是 10 个阈值上的连续均值，Yes/No 会把它悄悄压成二元准确率，
  却仍顶着 `MRA:.5:.95:.05` 的名字上报。
- 预防措施：
  1. **接入任何 judge，先做反向抽查**。只测「judge 能救回多少」而不测
     「judge 会误伤多少」，得到的必然是上界，且看起来像净收益。
  2. judge 的职责应是**抽取**，评分交回 benchmark 自己的指标函数。
     让 judge 直接产出对错，等于用 judge 的判据替换了 benchmark 的定义，
     此时报出的数字已不能与该 benchmark 的公开结果比较。
  3. 判据不同导致的分歧**不会随 prompt 调优消失**。本次先怀疑是
     「GT 只传了裸字母 C、没给选项文本」，补上选项文本后分歧率
     9.7% → 10.7%，没有下降——说明问题不在 prompt 而在判据。

## LESSON-021：金标解析器要过「自洽测试」——把金标当回答喂回去必须满分

- 状态：已确认（2026-08-21）
- 背景：为测 MV-OPSD teacher 的可靠性，要给 SPAR 的模板化答案写解析器
  （`scripts/opsd/spar_scoring.py`）。金标是模板生成的、看起来「一眼就能正则」，
  回答侧才是难的，所以最初只在回答侧下功夫。
- 现象：加了一条测试——把训练池里每条金标当作模型回答喂回打分函数，要求得 1.0 分——
  49,557 条里报出 3,007 条不匹配。金标侧本身在错。
- 原因：三个独立的 bug，全部只有自洽测试能抓到：
  1. 数值题金标取的是字符串里**第一个**数字，而 SPAR 模板把物体 id 写在测量值旁边：
     "At a depth of about 6.7 meters, Object169's center..." 被读成 169。
  2. `*_imagination_*_mv` 的金标同时描述观察者移动**前后**两个状态，题面只问移动后。
     金标侧做了前后消歧，回答侧没做，于是一条正确回答被判成「读不出」。
  3. `vsibench_scoring._NUMBER_RE` 的尾部前瞻 `(?![\d.])` 本意是挡 "1.2.3"，
     却把任何**句末**数字一并挡掉了（"...the count of pipes is 0." → None）。
     它只在兜底路径生效，所以在 VSI-Bench 上潜伏了很久没被发现。
- 为什么单靠人工抽样看不出来：第 1 条 bug 在改动前**两侧犯同一个错**
  （金标和回答都取第一个数字），测试因此假通过；只有把回答侧换成更好的读法之后，
  两侧才不再对称，错误才暴露出来。也就是说，这类 bug 会在「解析器变好」的那一刻
  才开始扣分，而那时人往往以为改动是安全的。
- 预防措施：
  1. 任何有结构化金标的判分器，都配一条**金标自洽测试**：金标即回答，必须满分，
     且逐题型报出通过数。这条测试的成本是几十行代码，抓到的是判分口径的系统性偏差。
  2. 金标侧与回答侧要用**同一个读法**。两侧各写一套正则，迟早会在某个模板上分叉。
  3. 正则的前瞻若同时承担「拒绝某种格式」和「界定 token 边界」两件事，拆开写，
     并各配一条测试；合在一起时，第二个职责的副作用不会有人注意到。

## LESSON-022：全匹配指标掉到随机水平以下时，先拆成「覆盖率 × 单项准确率」再下结论

来源：20260820_teacher_reliability（ISSUE-T05）。

SPAR 的方向题金标会同时给出多条轴（左右 / 上下 / 远近），判分规则是**金标提到的每条轴
都要对**才算 1 分。中期统计里 `spatial_imagination_oo_video` 的准确率是 0.9%——
三条轴各二选一，闭着眼睛猜也该有 12.5%。低于随机水平通常意味着判分器坏了，
但这次不是：teacher 的习惯是只挑一条轴回答（"the chair is above the radiator"），
而金标要三条，于是绝大多数行因为**没说全**而不是**说错了**拿零分。

拆开看就清楚了：它答全三条轴的比例只有 2.8%，但**在它主动说出口的那些轴上准确率有 67%**。
方向感是有的，缺的是回答的完整性——这两件事对下游的处理方式完全不同：
前者要换 prompt 或换蒸馏目标，后者才需要怀疑模型能力。

做法：
1. 任何 all-of-K 形式的指标（多轴、多字段、多条件同时成立），除了主分数之外，
   固定输出两个分解列：**覆盖率**（答全 K 项的比例）与**单项准确率**（在已作答项上的微平均）。
2. 主指标不要因为难看就改口径。分解列只用来解释，不替代主指标。
3. 判断「是不是判分器坏了」有个便宜的判据：**把随机基线算出来**。
   低于随机基线就必须解释清楚原因，不能直接当成模型能力差写进结论。

## LESSON-023：判分正则必须拿 prompt 自己引入的干扰串做测试

来源：20260820_teacher_reliability（ISSUE-T06 / T07）。

多视角 prompt 把每张图标成 `Frame-0 ... Frame-N`，teacher 讲解时会频繁点名帧号
（"In Frame-4, there is another pile of clothes"）。数值题的兜底解析取
「最后一行的最后一个数字」，而 `_NUMBER_RE` 的后顾只挡数字和点、不挡字母，
于是 `Frame-4` 里的连字符被当成负号，读成 **-4**。

后果：spar32 扫描 **10.1%** 的数值题（967/9,580）被判 0 分，as_trained 是 1.25%。
32 图的回答几乎必然逐帧点名，所以中招率是 8 图场景的 20 倍——
**干扰强度随视角数增长**，小规模冒烟基本抓不到。

发现路径也值得记：不是靠读代码，是靠**反向抽查的分组分歧率**。
spar32 分歧 8.6% 而 as_trained 只有 1.8%，这个**跨 dump 的落差**才是线索；
单看 8.6% 很容易归因成「judge 和规则口径不同」而放过。

做法：
1. 判分正则的测试集必须包含**prompt 自己引入的字符串**：帧标签 `Frame-N`、
   物体标识 `Object12`、场景号 `scene0555_00`、点号 `point3`。
   这些不是脏数据，是我们自己写进 prompt 的，模型必然会复述。
2. 数字解析的后顾要挡住「数字粘在词后面」和「连字符粘在词后面」两种情况：
   `(?<![\w.])(?<!\w-)`。前者挡标识符，后者挡带连字符的标签。
3. 同一语义的解析器**全仓只允许一份定义**。T06 只修了回答侧，
   金标侧还留着一份自己写的宽松版本，自洽测试立刻报出 `2 rj45 outlet` 读成 45。
   修一个解析 bug 的过程中当场复现了同一类错误，说明「两侧各写一套」这个坑很深。
4. 反向抽查要**按 dump 和按 family 分别报**。汇总成一个数字就看不到落差，
   而落差才是定位系统性 bug 的信号。

另附一条对 judge 的认识：反向抽查的分歧**不等于**规则层的错误。
人工逐条复核 as_trained 的 12 条分歧，规则层全对、judge 全错——
MCQ 上回答末尾明确写着 `C`、规则层读出 `C`，judge 报 `B`（它在自己答题而非抽取）；
方向题上 judge 会把回答根本没说的轴补全。
**分歧率只是错误率的上界，必须人工看过样本才能下结论。**

## LESSON-024：复现上游指标要连它的浮点构造一起复现，不能照着指标名字重写

来源：20260821 一级指标诊断（ISSUE-301）。

VSI-Bench 的数值题指标叫 `MRA:.5:.95:.05`，照名字理解就是「0.5 到 0.95 步长 0.05
共 10 个阈值」，于是我们写了 `[0.5 + 0.05*i for i in range(10)]`。个数确实对，
上游 `np.linspace(0.5, 0.95, int((0.95-0.5)/0.05 + 2))` 的 `int()` 恰好把
10.999999999999998 截成 10。**错的是阈值的最后几位**：numpy 的第 8 个阈值是
`0.8999999999999999`，`1-阈值 = 0.10000000000000009`；十进制写法给 `0.9`，
`1-阈值 = 0.09999999999999998`。而「相对误差恰好 0.1」在浮点里是
`0.10000000000000009`——正好落在两者之间，上游判命中、我们判未命中。

这不是能忽略的舍入噪声，因为**压在阈值上的都是常见答案**：
预测 1.1 对金标 1.0、9 对 10、1.7 对 2.0、18 对 20。5,130 行 VSI-Bench 验证集里
中招 8 行、overall 差 0.02 分，**且方向恒定为我们更低**。恒定方向是关键特征：
真随机的判分噪声会双向抖动，单向偏差意味着比较运算的边界系统性地偏了一侧。

做法：
1. 要和上游对齐的指标，**照抄它的算式**（连 `np.linspace`、连那个看着像 bug 的
   `+2` 一起抄），不要照抄它的名字或文档描述再自己实现。抄的时候把上游文件与行号
   写进注释，说明为什么长这样。
2. 判分口径的 parity 测试不能只比总分。总分差 0.02 很容易被解释成
   「引擎/采样差异」而放过——本次就差点如此。必须**逐行比对**并把分歧数打出来
   （目标是 0，不是「小」），再对**压在边界上的构造样例**单独断言。
3. 同一指标全仓一份定义（同 LESSON-023 第 3 条）。这次 `mvopsd_reward` 与
   `spar_scoring` 各写了一份相同的错法，改一处不会自动修好另一处。
4. 发现这类缺陷的路径是**双实现互查**，不是读代码。让训练内判分和上游 lmms_eval
   对同一批回答各算一遍，是唯一能把「差 0.02」变成「第 315/329/822… 行差」的手段。

## LESSON-026：环境变量能决定「用哪个版本的包」，不只是「能不能 import」

来源：`20260822_boxed_format_probe`（ISSUE-401）。

驱动首次启动秒退在 `ModuleNotFoundError: No module named 'transformers'`。
根因是调用方 shell 带着 `PYTHONNOUSERSITE=1`，禁掉了 user site，而训练环境正是
「系统 python3.12 + `~/.local`」（LESSON-007），于是整个环境不可见。这一层容易查。

**难查的是第二层**：同一个变量还决定 conda `sr_opsd` 里**用哪个** transformers。
设置时解析到环境自带的 5.3.0，不设置时被 `~/.local` 的 5.5.0 盖住。两种情况都能
`import transformers` 成功，只探「能不能 import」完全看不出来。而所有归档数字都是
在普通人工窗口（未设置该变量）下跑的，即图像预处理实际用的是 5.5.0——
**此前「用哪个 transformers」一直取决于谁启动了作业**。

做法：
1. 环境相关的量要么固定要么打印，不能继承。脚本顶部显式 `unset` / `export`
   自己依赖的环境变量，而不是假设调用方是干净的。
2. import 探针要连**版本与路径**一起打印进日志（`module.__version__`、
   `module.__file__`），只断言 import 成功抓不到版本被 user site 遮蔽这一类。
3. 一台机器上同时存在「conda 环境」与「`~/.local` user site」两套包时，
   两者的优先级由 `PYTHONNOUSERSITE` 决定；跨环境复现数字前先确认这一位。

## LESSON-027：新增答案格式会引入新的判分干扰串，上线前先给格式本身写测试

来源：`20260822_boxed_format_probe`（ISSUE-402）。

为验证「把答案放进 `\boxed{}`」这条建议，给 boxed 路径补了金标自洽测试
（LESSON-021），立刻报出 `\boxed{30.9\,\text{m}^2}` 判 0 分。原因是
`_NUMBER_RE` 把 LaTeX 指数里的 `2` 当成了一个数量，而数值兜底取「最后一个数字」，
于是面积答案 30.9 被读成 2。`\frac{1}{2}` 读成 2，`12 cm^3` 读成 3。

这和 `Frame-4` 读成 −4（LESSON-023）是同一类错误，但多了一层值得记的东西：
**这次的干扰串是我们自己即将要求的格式引入的**。要求 `\boxed{}` 就是在邀请 LaTeX，
而四个数值题型里有两个的单位是面积和体积，`^2` / `^3` 因此不是罕见输入而是常态。

而且它**先于 boxed 就存在**——模型本来就在自发写 `\text{m}^2`。57.7 万条归档回答
回放出 518 条改判，方向全部向上（旧读指数、新读真值），**作答率变化恒为 0**：
旧解析器确实读出了一个数字，只是读错了，于是「读错数」与「答得差」再次同形
（LESSON-018）。影响 teacher 41.4% → 约 41.45%、两条视角扫描上移 0.1~0.4 pp，
无结论翻转。

做法：
1. 引入任何新的答案格式之前，先把**该格式会带来的字符串**写成测试用例：
   LaTeX 指数 `^2` / `^{2}`、单位包装 `\text{}` / `\mathrm{}`、间距宏 `\,`、
   定界符 `$`、分数 `\frac{}{}`、加粗 `\textbf{}`。
2. 装饰清理只允许作用在**已经界定为答案的片段**上（框内），不能作用于自由散文。
   框内每个字符都属于答案，去装饰无风险；散文里去装饰会改变语义。
3. 不要照抄上游的替换表。`math_dapo.normalize_final_answer` 的
   `REMOVED_EXPRESSIONS` 是按 MATH-500 见过的字面量手工维护的（列着 `"\\text{}^2"`、
   `"\\text{}^3"`、`"cm"`、`"meters"`、`"square"`），覆盖 `^2` / `^3` 却不覆盖 `^4`，
   且依赖前置替换先把 `\text{m}` 改写掉。按**结构**匹配不需要知道 benchmark 用哪些单位。
4. 金标自洽测试要跑**全量**而不是抽样：这次 5,130 条全过才敢动判分器，
   而抽样很可能漏掉只在面积题上发作的那一类。

## LESSON-025：Prompt token 不进蒸馏 loss，不等于 Prompt 状态不会改变蒸馏目标

来源：`20260821_mvopsd_single_source`（ISSUE-305）。

本轮把 Teacher 设为 thinking、Student 与评测设为 non-thinking。代码审计确认 VOPD
没有误把 prompt/image token 算进 loss：`loss_mask=response_mask`，只覆盖 Student
生成的 response。但 Teacher 会在自己的 prompt 后 teacher-force 同一条 Student response，
所以 Teacher prompt 停在开放 `<think>` 还是预闭合 `</think>`，会改变**每个 response
位置的 next-token soft target**。

两条独立单源 arm 都在 10~25 步内发生 response-length explosion：

- llava_hound：训练 step 25 mean=821 token、clip=50%；VSI 验证 97.4% 撞 1024；
- vlm3r：训练 step 15 mean=1022、clip=98.4%；VSI 验证 step 25 全部撞 1024；
- 两侧 VSI 从约 53.4% 降到约 19~22%，而 VOPD loss 持续下降。

这说明要区分三件事：

1. **Loss 覆盖哪些 token**：本实现只覆盖 response；
2. **分布由什么上下文条件化**：prompt/image/thinking 状态虽不直接计 loss，却决定 soft target；
3. **Response 内是否再分段**：当前没有 answer-only mask，逐帧分析、重复文本与最终答案等权。

预防措施：

1. 改 Teacher/Student chat-template mode 时，先做 25-step 单变量诊断，不直接开全 epoch；
2. 启动验收除 loss 外，固定看 step 10/15/20 的 response mean、median 与 clip ratio；
3. 验证的 `answered/frac` 在长文中会被帧号/中间数字虚高，必须同时报达到 cap 的比例；
4. “Teacher thinking 是本轮唯一根因”目前仍是**待验证假设**。本轮同时改变了数据源、
   rollout.n 与 thinking，只有 thinking on/off 对照才能建立因果归因；
5. 若要只蒸馏最终答案，需要设计 answer-aware mask；这是方法扩展，不是修复
   Vision-OPD 范围实现 bug。

## LESSON-028：一张映射表只能装一个命名空间，join 之后必须交叉校验

来源：`20260821_mvopsd_single_source`（ISSUE-306），HTML 结果查看器。

离线评测的 `samples.jsonl` 只带 `id`，要靠它去 val parquet 取 32 帧图。查看器的
`parquet_by_id` 同时写了两个 key：

```python
mapping[str(extra.get("id", ""))] = row
mapping[str(extra.get("index", i))] = row
```

parquet 里 `extra.id` 是**字符串**、`extra.index` 是**整数**，原生永远撞不上；
是这里的 `str()` 归一化把两个命名空间压成一个，**人为制造**了碰撞。10,260 次插入
只剩 5,155 个 key，跨行覆盖 5,103 次，最终 **5,130 行里 2,638 行（51.4%）**
join 到了另一道题。

放大它的是字段优先级：`build_sample` 用 `extra.get(k, dump.get(k, ...))`，
即 parquet 优先于评测记录。于是页面出现最难识别的一种错误形态——
**题面、金标、回复、分数全对，题型、场景、index、id 和 32 帧图全错**。
用户看到「sofa 金标 173」配着另一个房间的帧，无法判断模型到底看了什么。

三条可复用的规则：

1. **一张 dict 只装一种 ID。** 需要多种查法就分表或加前缀（`f"id:{x}"` /
   `f"idx:{x}"`），并在查找处写明用的是哪一种。绝不要靠 `str()` 把不同语义的
   编号塞进同一个 key 空间——类型差异原本正在帮你挡住这个 bug。
2. **join 的 key 要先证明唯一。** 本例 `extra.id` 5,130 个全局唯一、samples 侧
   100% 命中，所以按 id 单键 join 可以做到零错配；重复 key 应当直接报错退出，
   而不是静默覆盖。
3. **join 完要交叉校验，并把不一致写进产物。** 两边都有的字段
   （`question_type` / `dataset` / `scene_name`）必须比一遍，冲突则在页面标红、
   在构建日志报数量。修复后加了 `meta_ok` / `meta_conflicts` 两列与一个筛选项，
   同类问题下次会自己浮出来。

顺带两条读数纪律：

- **展示层的错不等于指标的错，但要分清边界。** 本例分数、`summary.json`、
  23.00% 全程正确，因为判分在 `vsibench_eval_core` 内完成、不经过这个 join。
  验收方式是重建后比对 failure 分布（3978/937/124/91 逐项一致），
  用「不该变的东西没变」证明只动了展示层。
- **按行序对齐的路径不要改成按 key 查。** 训练内 dump（`logs/val/<step>.jsonl`）
  与 parquet 是同序同长的，`df.iloc[i]` 配 `dumps[i]` 加一条长度断言即可，
  引入 id 查找只会把一个本来不可能错的地方变成可能错。

## LESSON-029：Teacher 与 Student 的 thinking 模式必须同态，否则蒸馏目标换了一个分布

2026-08-24 的单变量诊断（`20260824_teacher_thinking_off`）证实了
`20260821_mvopsd_single_source` 只能标为假设的那条根因。只把
`actor.self_distillation.teacher_enable_thinking` 从 `True` 改成 `False`、
其余 25 步配置逐项不变：

| 指标 @step 25 | teacher thinking ON | OFF |
| --- | --- | --- |
| VSI `rule_only` | 20.85 | **52.57** |
| 训练 `response_length/mean` | 821.28 | **221.31** |
| 训练 `clip_ratio` | 0.50 | **0.0156** |
| 验证 `resp_tokens` 中位 | 1024 | **4** |

机制不是「开了 thinking 所以输出变长」这么直接。Teacher 在本方法里**从不采样**，
它只对 Student 已经生成的 token 做 teacher forcing。Qwen3 模板下
`enable_thinking=True` 让 Teacher 的上下文停在一个**打开的** `<think>` 里，
于是 Student 每个 token 被对齐的目标从「Teacher 的作答态分布」变成
「Teacher 的推理态分布」——推理态本来就该继续写分析、不该收尾，
Student 被逐 token 推着学会了不停止。两侧同态后这个压力消失。

三条可复用的规则：

1. **凡是「只改渲染参数」的开关，先问它改了哪一侧的哪个分布。** 这个开关在
   prompt 里只差 2 个 token（`<think>\n` vs `<think>\n\n</think>\n\n`），
   长度上完全可以忽略，但它把蒸馏目标整体换掉了。**改动大小与后果大小无关。**
2. **loss 平台不能用来判断这类改动。** 两条 arm 的 VOPD/JSD 平台几乎相同
   （0.0149 vs 0.0139），下游差 31.72 pp。这是 LESSON-012 的独立复现：
   `vopd_loss` 度量「Student 与 Teacher 一致」，而 Teacher 本身可以是错的目标。
3. **「消除了破坏」不是「产生了收益」。** thinking 关掉后 `rule_only`
   仍从 53.38 掉到 52.57，且验证 `resp_tokens` 的 p95 从 333 涨到 1024、
   `answered/frac` 98.23% → 95.05%——中位数不动而长尾变粗（LESSON-020 的另一面）。
   写结论时这两句必须同时出现，只写第一句会把「移除混淆因素」说成「方法可用」。

## LESSON-030：从做过评测的窗口启动训练，环境不干净是默认状态

`run_mvopsd.sh` 的训练环境是**系统 python3.12 + `~/.local`**（vllm 装在
`~/.local`）。`conda activate` 会导出 `PYTHONNOUSERSITE=1`，该变量让**任何**
python 跳过 user site，于是 `/usr/bin/python3` 看不见 `~/.local` 里的 vllm。
2026-08-24 从一个跑离线评测（conda `sr_opsd`）的窗口排队启动训练，`setsid`
原样继承了这个变量，训练在 0 步退出，唯一线索是 judge 日志里一句
`No module named 'vllm'`——它指向的方向（「judge 坏了」）是错的。
这与 ISSUE-202 同一根因，只是换了个脚本，即**同一个坑第二次踩**。

- 归档的启动命令写 `PATH=/usr/bin:$PATH` 是不够的：它只在**非 conda** shell 里成立。
  跨环境启动要连 conda 的变量一起清：
  `env -u PYTHONNOUSERSITE -u CONDA_PREFIX -u CONDA_DEFAULT_ENV ...`，
  并把 miniconda 从 `PATH` 里剔除。
- **依赖可见性要在花钱之前检查，且报错要指向真正的原因。**
  `run_mvopsd.sh` 现在有一道闸门，用 `importlib.util.find_spec('vllm')`
  （不真 import，省掉约 10 s）在 0.4 s 内失败，打出 `python3` 的实际路径，
  并在 `PYTHONNOUSERSITE` 有值时直接点名它、给出改法。
- 排队脚本要同时等 **PID 退出**和**显存真正释放**两个条件：被杀掉的 vLLM 会留下
  孤儿 worker 占着 85 GB/卡 且利用率 0%（ISSUE-403），只看 PID 会在卡还被占着时启动。
- **`CUDA_VISIBLE_DEVICES` 同样会被继承。** 2026-09-05 从本窗口启 Stage2 时它是 `0`，
  torchrun `--nproc=8` 的 LOCAL_RANK 1–7 在 `qwen_module` import 时 `.to(cuda:N)` 直接
  invalid device ordinal（ISSUE-012）。启动命令要 `env -u CUDA_VISIBLE_DEVICES`，
  8 卡脚本在可见卡 <8 时自己 unset。
  另外**不要按 pgrep 模式等待**——脚本自己的命令行含那个模式，启动它的 shell 也含，
  于是永远匹配得到、死等（第一版就是这么写的）。按 PID 等。

## LESSON-031：CV-Bench 默认协议改过之后，旧基座分必须作废而不是继续引用

- 状态：已修正（2026-09-01）
- 现象：对外材料里基座 CV-Bench 长期写 **85.95**。那是 2026-08-18 的
  spatialstack + `word_boundary`、题面无 `\boxed{}` 后缀。2026-09-01 起离线默认
  已与 VSI val 对齐为 boxed last-line，同一发布权重重测为 **88.07**。
  继续引用 85.95 会把 SPAR3 K=1 85.05 / K=2 86.80 读成「接近或超过基座」，
  同口径实际是 **都低于基座**（−3.02 / −1.27）。
- 原因：prompt 与解析器绑在一起。旧协议不要求 `\boxed{}`，解析扫全文独立字母；
  新协议要求末行 `\boxed{}`，只认闭合 boxed 否则末行选项。分数差来自整套协议，
  不是同一批生成的重打分，不能用固定 delta 把 85.95 折成 88.07。
- 证据：`logs/eval/20260901_qwen35base_cvbench_boxed_lastline/`，
  combined 88.07 / 2D 83.55 / 3D 92.58 / answered 99.62%。
  SPAR3 同协议：`logs/eval/20260831_mvopsd_spar3_k{1,2}_cvbench_boxed_lastline/`。
- 处理方式：凡写「基座 CV-Bench」一律用 **88.07** 并标注 boxed last-line。
  85.95 与 SPAR3 旧 dump 84.82/84.84 仅作为 word_boundary 历史点保留。
- 预防措施：协议开关改默认值后，**基座必须重跑**，不能把旧默认协议的分数
  改个标签继续用。跨实验对比先核 prompt 逐字、解析器名字、生成上限。

## LESSON-032：`total_training_steps` 不能单独超过 1 个 epoch

- 状态：已规避（2026-09-02）
- 现象：`20260901_mvopsd_spar3_k1_200steps` 明确设了 `TOTAL_STEPS=200`，进度条也是
  `63/200`，但 1 epoch 扫完（4,064/64=63 步）训练就干净退出，wandb 同步后 8 卡空闲。
- 原因：`ray_trainer.py` 外层是 `for epoch in range(..., total_epochs)`，
  `mvopsd.yaml` 默认 `total_epochs: 1`。`total_training_steps` 只决定进度条和
  `is_last_step`，**不延长 epoch 循环**。此前 llava/vlm3r/SPAR 全量都是「1 epoch
  的步数 ≈ TOTAL_STEPS」，这个问题没暴露。
- 处理方式：从 `global_step_60` 续跑，并显式传 `trainer.total_epochs=4`
  （4×63=252>200，仍由 `is_last_step` 在 200 停）。
- 预防措施：步数超过 `ceil(n_train/batch)` 时，必须同时把 `total_epochs` 提到
  `ceil(TOTAL_STEPS / steps_per_epoch)`；启动后确认日志里的 `total_epochs`，
  不要只看进度条分母。

## LESSON-033：BLINK 的句首字母协议与 VSI boxed last-line 必须并存，不能互相覆盖

- 状态：已规避（2026-09-02）
- 现象：`blink_spatial` 默认用 letter-only `pre_prompt` + `_extract_answer_letter`
  （只看回复开头）。SPAR3 K=1 step 55 宏平均 **13.05**（0 / 0 / 39.16），
  因为 `To determine...` 被抽成 T、`Based on...` 被抽成 B（LESSON-011 同类）。
- 原因：这是打分协议问题，不是换一套解析就能在同一份 dump 上得到可比能力分。
  VSI-Bench 的 MCA 协议换了题面（boxed last-line 后缀）和停止条件（`until=[]`），
  必须重新生成。
- 处理方式：`BLINK_PROTOCOL` 二选一，默认 **`original`**，保留 13.05 可复现。
  `spatialstack` 走 VSI 同款 MCA 后缀 + `apply_boxed_primary` +
  `extract_vsibench_option`。两套 dump 分目录，禁止混报。
- 预防措施：改 BLINK 打分时用 if-else 加协议，不要改掉 `_extract_answer_letter`。
  新协议必须换 `OUTPUT_ROOT`。

## LESSON-038：`tie_word_embeddings=True` 时不要合并含 `lm_head` 的 LoRA

- 状态：已规避（2026-09-06，ISSUE-019）
- 现象：想把 Stage2 的 LoRA 合并成完整权重再评测，两条独立路径都会**静默**产出错误模型。
- 原因一（丢张量）：合并用的环境是原版 transformers，其模型类不认识基座里的
  自定义张量（3DThinker 的 28 个 `projector_model.*`），加载时按 unexpected key 丢弃，
  只打印一条 info 级提示，导出的权重就少了整个分支。
- 原因二（污染 embedding）：基座 `tie_word_embeddings=True` 且没有独立 `lm_head`，
  它与 `embed_tokens` 是同一个张量；而 adapter 把 `lm_head` 列为 LoRA target。
  一旦 merge，输出投影的 delta 会被写进共享的输入 embedding。
  PEFT 自己会警告 `tied_target_modules=['lm_head']`，但那只是 UserWarning，不会阻断。
- 处理方式：**不合并**。评测时直接挂载 adapter：读 `adapter_config.json` 里的
  `base_model_name_or_path` 加载基座，再 `PeftModel.from_pretrained(..., is_trainable=False)`。
  LoRA 只在前向生效，语义与训练逐位一致。评测环境用
  `pip install --no-deps peft==<训练同版本>`，不动已验证的 transformers。
- 预防措施：合并前先问两个问题——目标环境的模型类认得全部张量吗？
  `lm_head` 是不是 LoRA target 且与 embedding 绑定？任一为真就改成挂载 adapter。
  另外「模型加载时的 unexpected key 提示」要当成错误看，它不是噪音。

## LESSON-037：镜像全局导出的 `TRITON_*_PATH` 会在换 CUDA 小版本后突然变成致命项

- 状态：已规避（2026-09-05，ISSUE-015）
- 现象：新建的 `3DThinker-stage2-vllm`（torch 2.6.0+cu124 + triton 3.2.0）里 vLLM 引擎
  初始化直接死掉，栈顶是
  `RuntimeError: Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0`，
  外层报 `Engine core initialization failed`。pip 装完没有任何报错，`import vllm` 也正常。
- 原因：镜像在 shell 里全局导出了 `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
  （以及 `TRITON_CUDACRT_PATH` / `TRITON_CUDART_PATH` / `TRITON_CUPTI_*` 等），
  指向系统 CUDA **13.0**。triton 3.2 的 `ptx_get_version` 只映射 major 10/11/12，
  遇到 13 直接抛错。环境自带的 `triton/backends/nvidia/bin/ptxas` 是 12.4.99，本来是对的，
  但被这个环境变量抢走了优先级。旧环境用 torch 2.13+cu130，triton 认得 CUDA 13，
  所以这些变量一直存在却从未出过事。
- 处理方式：启动脚本里 `unset` 全部 `TRITON_*_PATH`，让 triton 用自带工具链；
  同时 `export PYTHONNOUSERSITE=1`（与 LESSON-030 / ISSUE-202 / ISSUE-501 同类）。
- 预防措施：换 torch 或 CUDA 小版本后，**必须跑一次真正拉起引擎的冒烟**
  （如 `3DThinker/envs/smoke_stage2_vllm.py`）。「pip 装完」与「import 通过」
  都不能证明 kernel 编译链可用——它在第一次 JIT 时才炸。

## LESSON-036：GenerationConfig 里「等于默认值」的字段会被模型侧静默换掉

- 状态：已规避（2026-09-05，ISSUE-014）
- 现象：3DThinker Stage2 单步 883 s（5000 步约 51 天），同时 `reward_std` 恒 0.0、
  `grad_norm` 恒 0.0——即又慢又没在学。trainer 明确写了
  `do_sample=True, temperature=1`，也没关 `use_cache`。
- 原因：`transformers` 的 `_prepare_generation_config`（generation/utils.py:1596-1606）
  把自定义 `GenerationConfig` 里**仍等于全局默认值**的字段整体回落到
  `model.generation_config`。`temperature=1` 就是默认值 1.0、`use_cache` 未设即默认 True，
  两者都命中该条件，于是被换成 ckpt 里的 `temperature=1e-6` 与 `use_cache=False`
  （Qwen2.5-VL 发布的 `generation_config.json` 带 1e-6，Stage1 存权重时继承下来）。
  两个后果各自独立：`use_cache=False` 让 336 步解码每步重算整段 ~2100 token（O(n²)），
  ZeRO3 下 generate 还开 `synced_gpus`、8 rank 按最长 completion 一起付费；
  `temperature=1e-6` 等于贪心，一组 8 条 rollout 几乎相同，
  `advantages=(r-mean)/(std+1e-4)` 全零。
- 处理方式：把这类字段作为 **kwargs 传给 `generate()`**，
  kwargs 在回落之后才应用（utils.py:1613），不会被覆盖。
  改 GenerationConfig 前先看 generate 打的第一条 warning
  `default values have been modified to match model-specific defaults: {...}`,
  那个字典就是被换掉的全部字段。
- 预防措施：GRPO 一类组内比较的算法，启动后必须确认 **`reward_std > 0` 且
  `grad_norm > 0`**，两个都要。**只看 loss 会被骗**——首步 loss 本来就≈0
  （ratio=1 且组内 advantage 和为 0），零梯度和正常首步在 loss 上长得一模一样。

## LESSON-035：HF generate + `use_cache=False` 会让 Qwen2.5-VL 的 mask 比 input_ids 长 1

- 状态：已规避（2026-09-04，ISSUE-007）
- 现象：3DThinker Stage2 GRPO 第一次 `generate` 在 `get_rope_index` 崩掉
  `IndexError: mask shape [2083] vs indexed tensor [2082]`，进度停在 0/5000。
- 原因：gradient checkpointing 把 `model.config.use_cache = False`。
  `generate` 每步仍给 `attention_mask` 追加一列，但 `use_cache=False` 时不更新
  `cache_position`，下一次 forward 用旧的 `cache_position` 切片 `input_ids`，
  于是 ids 长度仍是 P、mask 已是 P+1。ZeRO3 monkey patch 每步都重算 rope
  （`past_key_values.get_seq_length()==0`），于是踩中这个 off-by-one。
- 处理方式：在 `get_rope_index` 与 monkey-patched / 原始 `forward` 里，
  把 `attention_mask` 裁/补到与 `input_ids` 同长。不要为了对齐而改成
  wrap-original-forward：ZeRO3 要求所有 rank 都进 `visual()` 同一个 collective。
- 预防措施：Stage2 启动后必须看到 `step >= 1` 的 loss 才算真正跑起来，
  引擎起来或 wandb 出 run 都不算。

## LESSON-034：3DThinker 推理走 vLLM 时必须先剥 projector，且数字与 HF 不可混报

- 状态：已规避（2026-09-04）
- 现象：vLLM 0.18 加载 Stage1 ckpt 直接失败
  `There is no module or parameter named 'projector_model'`。
  剥掉后全量 MindCube-Tiny 60.67%，同协议 HF 62.19%，差 −1.52pp。
- 原因：
  1. `projector_model` 只用于 Stage1 VGGT 对齐损失，推理不用。transformers 丢弃未知键，
     vLLM 拒绝。
  2. 引擎差是真的：1050 条里文本逐字相同 0.3%、答案相同 59.4%、中位共同前缀 44 字符。
     把 transformers 5.3 的 fast image processor 关掉，200 条探针也没把前缀拉到 HF。
  3. 裸调 `sr_opsd/bin/python` 会让 `~/.local` 的 triton 3.3.0 盖掉环境里的 3.6.0，
     然后报 `CUDA version: 13.0`——必须 `conda activate sr_opsd`（ISSUE-202 / 501 同类）。
- 处理方式：默认评测仍走 transformers（`run_stage1_mindcube_eval_fixed_sharded.sh`，62.19%）。
  vLLM 是可选加速：先 `eval/make_vllm_ready_ckpt.py`，再 `eval/run_vllm_mindcube_eval.sh`。
  两套数字不可并列；对外口径是 HF 62.19，vLLM 60.67 只作对照。
- 预防措施：给 vLLM 的 3DThinker 权重先查 index 有无 `projector_model.`；
  评测脚本 `auto` prompt 在目录名无 `begin`/`end` 时硬失败，不要再静默落到 mid（ISSUE-010）。

## LESSON-039：环境修复的正当范围只到「按原版本补回丢失的包」，跨机移植 site-packages 是改基线

- 状态：已规避（2026-09-07，ISSUE-702）
- 现象：node-A 的 DSW 镜像被重建（NGC nv25.11 / CUDA 13.0）后，`~/.local` 里 OPSD 训练用的
  transformers 5.5.0 消失。补回它是修回归；但接着 vLLM 报 `libcudart.so.12` 缺失，
  处置被升级成「把 node-B 整个 `~/.local/lib/python3.12/site-packages` rsync 覆盖过来」。
  之后接连出现 pydantic-core 版本配对失败、scipy `All ufuncs must have type numpy.ufunc`，
  最终 `pd.read_parquet` 报 `Cannot convert numpy.ndarray to numpy.ndarray`。
  该机从「缺几个包」变成「连读 parquet 都不行」，且 rsync 是覆盖式的、无备份、不可回滚。
- 原因：
  1. **user site 只能遮蔽 dist-packages，不能替换它。** 同步 `~/.local` 之后，
     dist-packages 仍是本机的，于是得到一个按两套 numpy ABI（1.26 与 2.1）混编的第三种环境，
     既不是 node-A 也不是 node-B。要复现另一台机器的环境只能用不继承系统层的独立 venv。
  2. **目标错位。** 那次改动的目的是让两条 arm 的测量可比，而可比性要求同引擎同库，
     正确解法本就是把两个 checkpoint 搬到同一台机器；被修的那台机器根本不需要跑推理。
     是「让当前这条命令跑通」的局部目标一层层牵着走，没有退回去问这一步的目的。
  3. 讽刺之处在于手段与目的相悖：为消除跨机差异而改环境，本身就引入了一个
     无法复现、无法写进注册表的新差异，正是实验协议要排除的东西。
- 处理方式：停手，改为两条 arm 同机跑，把数据与权重搬过去（plan/parquet/views/val parquet
  + 9.7 GB checkpoint，`--exclude 'global_step*'`，内网 600 MB/s 下 20 秒）。
  产物在目标机上重跑测试确认可信，而不是沿用源机结论。
- 预防措施：
  1. 一旦修复动作扩大到「覆盖整个 site-packages / 跨机移植 / 改动并未丢失的包」，
     停下来问，不要把它当作调试的下一步顺手做掉。
  2. 覆盖式 rsync 进 site-packages 之前先 `mv` 留备份。
  3. 给报错驱动的修复设深度上限：同一目标上第二次冒出新的 ABI/版本冲突，
     就说明在拼装不自洽的环境，该退回审视目标，而不是再补一个包。
  4. 需要跨机可比的测量，搬被测对象优于修环境——前者是数据搬运，后者是不可复现的改造。

## LESSON-040：带 VGGT 的几何类不能塞进 vLLM；全量评测走 HF/lmms-eval

- 状态：已规避（2026-09-10）
- 现象：SpatialStack 权重 `architectures=Qwen3_5ForConditionalGenerationWithGeometry`，
  含 VGGT 与 `language_feature_fusion`。vLLM 0.18.0 只认标准 Qwen3.5，没有
  `geometry_encoder_inputs` 通路。当成普通 Qwen3.5 加载会丢掉几何，测的是另一个模型。
- 影响：训练 rollout 必须 `ROLLOUT_NAME=hf`。HF generate 吞吐远低于 vLLM，
  全量 5130 行验证约 9.4h/次，所以训练内只能用分层子集（sub480），全量留给离线。
- 处理方式：几何 checkpoint 离线评测走与 geo SFT 相同的 HF/lmms-eval
  （`MODEL_IMPL=qwen3_5`）。20260909 geo 评测当时用 conda `spatialstack-qwen35` + `PYTHONPATH=src`。
  **此后训练一律 `sr_opsd`（LESSON-041）**。不要为了速度把几何权重塞进 vLLM。
- 预防措施：几何类评测默认 lmms-eval；vLLM 只用于无几何的 Qwen 基座 / SPAR3。
  训练内子集分与全量 greedy 不可并列；sample 与 greedy 也不可并列。
  见 `20260909_spatialstack_mvopsd_spar3_student_k` ISSUE-009。

## LESSON-041：此后所有训练在 conda `sr_opsd` 上跑

- 状态：生效（2026-09-10，用户决定）
- 背景：67.65 对应的 VGGT 几何权重在 `sr_opsd` 上训/评。20260909 geo SFT 却走了
  `spatialstack-qwen35`（torch/transformers 主版本接近，但 conda 树、flash_attn、
  lmms_eval 加载方式不同），几何 MV-OPSD 又曾用 `vision-opd`。分数不能当同环境复现。
- 处理方式：SFT（`scripts/train/train.sh`）与 MV-OPSD（`scripts/opsd/run_mvopsd.sh`、
  `launch_spatialstack_spar3_mvopsd.sh`）启动前检查 python 路径含 `/envs/sr_opsd/`。
  `scripts/train_eval.sh` / `train_novggt.sh` / `train_vggt_retrain.sh` 本就 activate `sr_opsd`。
- 例外：仅 `ALLOW_NON_SROPSD=1`，并在对应 registry 写 issues。评测环境不强制本条，
  但仍应在 registry 记录实际 conda。
- 预防措施：不要为了「geo 上次用过」就再 activate `spatialstack-qwen35` 做训练。

## LESSON-042：node-C 以 root SSH 时 wandb 凭证不在用户 netrc

- 状态：生效（2026-09-15）
- 现象：`ssh nodec` 为 root。项目树在 `/home/c30084464/...`，但 C 的家目录不是 A 的 live NFS；
  没有 `.netrc`。即便拷贝过来，`netrc.netrc()` 也会因文件属主不是 root 而拒绝。
- 处理方式：凭证放到 `/root/.netrc`（600）。`launch_spatialstack_vlm3r_t8_s4.sh` 在
  `WANDB_API_KEY` 为空时从 `/root/.netrc` 读取 `api.wandb.ai` 并 export。启动命令使用
  `env HOME=/home/c30084464` 以命中 conda `sr_opsd`，不要落到 `/root/miniconda3`。
- 预防措施：node-C 正式训前用非打印检查确认 wandb host 可读；不要把 key 写进 `experiments/`。

## LESSON-043：BLINK 走项目 `cache/datasets/`，不是 HF hub

- 状态：生效（2026-09-19）
- 现象：清盘后只 rsync `~/.cache/huggingface` 时，CV/SPAR lastline 能 offline 出分，
  `blink_spatial` 在 `load_from_disk("cache/datasets/BLINK-Benchmark__BLINK/Spatial_Relation")`
  上 FileNotFound；`eval.sh` 仍可能 rc=0。
- 处理方式：从有缓存的节点同步
  `Spatial_Relation` / `Relative_Depth` / `Multi-view_Reasoning` 三个目录后再评。
- 预防措施：启动 `blink_spatial` 前检查项目相对路径这三份 disk dataset，不能只查 HF hub。
  见 `20260918_spatialstack_mvopsd_spar_t32_s16` ISSUE-001。
