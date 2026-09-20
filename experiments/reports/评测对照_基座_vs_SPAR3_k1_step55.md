# 评测对照：基座 vs SPAR3 K=1 step 55

- 最后更新：2026-09-08
- 基座：`models/Qwen3.5-4B`
- SPAR3：`output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf`（训练内最佳）
- 解码：除标明 sample 的表外均为 greedy，`disable_thinking=true`
- sample 一律 t=1.0 / top_p=0.8 / seed=20260904，单次采样（非 Avg@12），**不可与 greedy 并列上报**
- 本表只含这两套权重。K=2、训练内 sample val 不列入。
- **BLINK 全量 14 只报 vLLM greedy @1024**：original / lastline（无 boxed）与 boxed 同一张表。
- **SPAR-Bench 只报 vLLM**（`eval_sparbench_vllm.py`，lastline 与 boxed / TP=8）。HF/lmms-eval 的 SPAR 数字不列入。
- **CV-Bench 只报 vLLM boxed**（`eval_cvbench_vllm.py`，TP=8 greedy，max tokens=1024）。其它 CV 口径不列入。

### 评测方法差在哪

**现行对比口径（2026-09-08 起）**：original 与 lastline **共用末行解析**（MCA `extract_vsibench_option` / NA 末行抽数；**不认** `\boxed{}`）。两者只差题面是否追加与 SPAR-Bench 相同的一句（前导空格、接在最后一行指令后）：

```text
 The final answer MUST BE put on the last line of your response.
```

不再使用 `\boxed{}`（**VSI / CV / SPAR boxed 行除外**，见下）。旧 dump 若按 `\boxed{}` 题面生成，不得当作 lastline 题面的分数。lastline 题面 dump：`logs/eval/20260908_lastline_prompt/`。SPAR boxed dump：`logs/eval/20260908_sparbench_vllm_boxed/`。


| 方法       | 题面             | 解析      |
| -------- | -------------- | ------- |
| original | 无 lastline 后缀  | 末行      |
| lastline | 上句，与 SPAR 逐字相同 | 末行（同一套） |


Δ = SPAR3 − 同题面、同解析、同引擎、同 token、同解码的基座。sample 不可与 greedy 并列。

BLINK 生成时 original 题面仍带「只回一个字母」；**上报 original 分时一律末行解析**，不用句首字母。句首 4.10 只作反面教材。

**已测完的格子**

- SPAR lastline **39.72 / 42.16** 与 boxed **40.19 / 42.65**。
- BLINK original / lastline / boxed：见一览。
- **VSI-Bench** original **52.81 / 46.73** 与 boxed **47.31 / 49.93**。
- **CV-Bench 只报 vLLM boxed：87.62 / 85.23。**

**协议（各任务）**

- **original**：无 lastline 后缀；解析末行。BLINK 生成仍带 letter-only 前缀，但分数用末行。
- **lastline**：追加 SPAR 那句；解析末行。BLINK @1024、SPAR @2048。
- **SPAR-Bench**：lastline（无 boxed）与 boxed；vLLM @2048。
- **VSI-Bench**：original（无 boxed）与 boxed；vLLM @2048。
- **BLINK**：original / lastline（无 boxed）与 boxed；vLLM @1024。
- **CV-Bench**：boxed；vLLM @1024。
- **MindCube tinybench**：官方 `input_prompt`，不走 original/lastline。transformers 8 分片 greedy 2048。不可与 SFT / 训练内 vLLM 并列。

## 各数据集题面

共用外壳：system `You are a helpful assistant.`；`enable_thinking=False`。`MCA` = `Answer with the option's letter from the given choices directly.`；`NA` = `Please answer the question using a single word or phrase.`；`LASTLINE` =  `The final answer MUST BE put on the last line of your response.`

### VSI-Bench

yaml 里 `pre_prompt: ""` 为假值，实际落到 `These are frames of a video.`。

**original（无 boxed）MCA**：

```text
These are frames of a video.
{question}
Options:
A. ...
B. ...
Answer with the option's letter from the given choices directly.
```

NA 最后一行是 `Please answer the question using a single word or phrase.`。解析末行，不认框。

**boxed**：上述最后一行指令后接  `The final answer MUST BE put in \boxed{} on the last line of your response.`。解析：有闭合 `\boxed{}` 只认框内，否则末行。

### BLINK（14 任务同一拼装）

`doc.prompt` 已含题干和 `(A) ...`。例：`Which image is taken from a viewpoint closer to the object?\n(A) Image A\n(B) Image B`

**original 题面**（生成）：

```text
Return exactly one uppercase option letter from the given choices (A, B, ...). Do not output any explanation, punctuation, or extra text.
{doc.prompt}
```

**lastline 题面**（`BLINK_PROTOCOL=lastline`）：去掉 letter-only 前缀，题干后接 `MCA` + `LASTLINE`：

```text
{doc.prompt}
Answer with the option's letter from the given choices directly. The final answer MUST BE put on the last line of your response.
```

**boxed**：同上，但最后一句换成 `...directly. The final answer MUST BE put in \boxed{} on the last line of your response.`。解析 boxed-primary。

### CV-Bench

本表只报 boxed（已测，vLLM）：

```text
{question}
Options:
A. {choice0}
B. {choice1}
...
Answer with the option's letter from the given choices directly. The final answer MUST BE put in \boxed{} on the last line of your response.
```

解析：有闭合 `\boxed{}` 只认框内选项，否则末行。不加快传 `These are frames of a video.`。

### SPAR-Bench

`pre_prompt` 为空。对照报 lastline 与 boxed（同一张表）。

**NA**（depth/distance prediction，含 `_mv`）：

```text
{question}
Please answer the question using a single word or phrase. The final answer MUST BE put on the last line of your response.
```

**MCA**（空间关系、空间想象、中心距离推断等）：

```text
{question}

Answer with the option's letter from the given choices directly. The final answer MUST BE put on the last line of your response.
```

**MCA + bbox**（仅 `position_matching`、`camera_motion_infer`）：question 与 MCA 之间多一行
`The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image.`

**VCI**（`view_change_infer`）：`{question}` + `LASTLINE`，无 MCA/NA 句。

**boxed**：把上述 `LASTLINE` 换成 ` The final answer MUST BE put in \boxed{} on the last line of your response.`。解析 boxed-primary。

official 题面即上面去掉 `LASTLINE`；本表不报 official 分。

### MindCube tinybench

官方 `input_prompt`（1050 题共用 [Task]/[Answer Instruction]，只换 [Question]）：

```text
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints.
[Answer Instruction]
You only need to provide *ONE* correct answer selecting from the options listed below. For example, if you think the correct answer is 'A. Above' from 'A. Above B. Under C. Front D. Behind', your response should **only** be '<answer>A. Above</answer>'.

[Question]
{question including A. B. C. ... options}
```

多图走 chat 的 image 块，不写进这段文字。解析官方 `extract_answer`，不是末行协议。

## 一览


| 任务                 | 题面                | 解析            | max tokens | 基座        | SPAR3     | Δ         |
| ------------------ | ----------------- | ------------- | ---------- | --------- | --------- | --------- |
| SPAR-Bench         | lastline（无 boxed） | 末行            | 2048       | **39.72** | **42.16** | **+2.44** |
| SPAR-Bench         | boxed             | boxed-primary | 2048       | **40.19** | **42.65** | **+2.46** |
| VSI-Bench          | original（无 boxed） | 末行            | 2048       | **52.81** | **46.73** | **−6.08** |
| VSI-Bench          | boxed             | boxed-primary | 2048       | **47.31** | **49.93** | **+2.62** |
| BLINK 全量 14        | original（无 boxed） | 末行            | 1024       | **64.98** | **61.98** | **−3.00** |
| BLINK 全量 14        | lastline（无 boxed） | 末行            | 1024       | **63.43** | **64.94** | **+1.51** |
| BLINK 全量 14        | boxed             | boxed-primary | 1024       | **62.67** | **63.66** | **+0.99** |
| CV-Bench           | boxed             | boxed-primary | 1024       | **87.62** | **85.23** | **−2.39** |
| MindCube tinybench | official          | official      | 2048       | **46.86** | —         | —         |


BLINK-Spatial HF（三任务，不可与全量 14 并列）见文末。MindCube SPAR3 未测。

## BLINK 全量（14 任务，vLLM）

val 1901 题。overall = 14 项 `blink_acc` **宏平均**。引擎 **vLLM TP=8** greedy。Dump：original / boxed `logs/eval/20260908_blink_full_vllm/`；lastline `logs/eval/20260908_lastline_prompt/blink_*_lastline/`。不可与 HF `blink_spatial` 并列。


| 模型    | 题面                | 解析            | max tokens | overall   | spatial-3 | Δ         |
| ----- | ----------------- | ------------- | ---------- | --------- | --------- | --------- |
| 基座    | original（无 boxed） | 末行            | 1024       | **64.98** | 75.80     | —         |
| SPAR3 | original（无 boxed） | 末行            | 1024       | **61.98** | 68.76     | **−3.00** |
| 基座    | lastline（无 boxed） | 末行            | 1024       | **63.43** | 67.62     | —         |
| SPAR3 | lastline（无 boxed） | 末行            | 1024       | **64.94** | 67.55     | **+1.51** |
| 基座    | boxed             | boxed-primary | 1024       | **62.67** | 64.02     | —         |
| SPAR3 | boxed             | boxed-primary | 1024       | **63.66** | 69.37     | **+0.99** |


句首字母（旧协议，不当能力差）：基座 64.82 / SPAR3 **4.10**。

## SPAR-Bench

test 7211 题。overall = 各 task 等权均值 ×100（官方聚合）。引擎 **vLLM TP=8** greedy。Dump：lastline `logs/eval/20260906_sparbench_vllm_lastline/`；boxed `logs/eval/20260908_sparbench_vllm_boxed/`。

| 模型    | 题面                | 解析            | max tokens | overall   | 截断     | 框合规   | 中位 token | Δ         |
| ----- | ----------------- | ------------- | ---------- | --------- | ------ | ----- | --------- | --------- |
| 基座    | lastline（无 boxed） | 末行            | 2048       | **39.72** | 5.3%   | —     | —         | —         |
| SPAR3 | lastline（无 boxed） | 末行            | 2048       | **42.16** | 2.5%   | —     | —         | **+2.44** |
| 基座    | boxed             | boxed-primary | 2048       | **40.19** | 17.57% | 82.5% | 411       | —         |
| SPAR3 | boxed             | boxed-primary | 2048       | **42.65** | 3.36%  | 95.0% | 306       | **+2.46** |

lastline MCA 微平均：基座 **39.63** / SPAR3 **45.07**；NA：**42.62** / **40.89**；VCI：**18.68** / **21.53**。

boxed MCA 微平均：基座 **41.87** / SPAR3 **45.20**；NA：**40.85** / **41.93**；VCI：**19.49** / **21.34**。

## VSI-Bench（vLLM greedy）

test 5130 题，32 帧。overall = 8 类等权均值 ×100（方向三档先合并）。引擎 **vLLM TP=8**。Dump：`logs/eval/20260906_vsibench_vllm_tok2048/{base,spar3}_{plain,boxed}_greedy/`。


| 模型    | 题面                | 解析            | max tokens | overall   | 作答     | 截断    | Δ         |
| ----- | ----------------- | ------------- | ---------- | --------- | ------ | ----- | --------- |
| 基座    | original（无 boxed） | 末行            | 2048       | **52.81** | 98.60% | 1.54% | —         |
| SPAR3 | original（无 boxed） | 末行            | 2048       | **46.73** | 96.35% | 1.81% | **−6.08** |
| 基座    | boxed             | boxed-primary | 2048       | **47.31** | 94.74% | 4.17% | —         |
| SPAR3 | boxed             | boxed-primary | 2048       | **49.93** | 97.86% | 2.22% | **+2.62** |


boxed 题型（百分制；方向三档先等权合成一票；max tokens=2048）：


| 题型                     | 基座    | SPAR3 | Δ         |
| ---------------------- | ----- | ----- | --------- |
| object_counting        | 44.57 | 50.96 | +6.39     |
| object_abs_distance    | 28.98 | 30.92 | +1.94     |
| object_size_estimation | 56.60 | 64.43 | +7.83     |
| room_size_estimation   | 46.46 | 47.67 | +1.21     |
| object_rel_distance    | 59.58 | 58.31 | −1.27     |
| object_rel_direction   | 49.56 | 49.90 | +0.34     |
| route_planning         | 34.02 | 33.51 | −0.51     |
| obj_appearance_order   | 58.74 | 63.75 | **+5.01** |


## CV-Bench（vLLM boxed）

val 2638 题。引擎 **vLLM TP=8** greedy。题面追加 `\boxed{}` 后缀，解析 boxed-primary。Dump：`logs/eval/20260908_cvbench_vllm_boxed/`。


| 模型    | 题面    | 解析            | max tokens | overall   | 2D    | 3D    | 作答     | 截断    | Δ         |
| ----- | ----- | ------------- | ---------- | --------- | ----- | ----- | ------ | ----- | --------- |
| 基座    | boxed | boxed-primary | 1024       | **87.62** | 83.15 | 92.08 | 99.39% | 0.61% | —         |
| SPAR3 | boxed | boxed-primary | 1024       | **85.23** | 80.04 | 90.42 | 99.81% | 0.08% | **−2.39** |


## MindCube tinybench（基座 zero-shot）

1050 题（among 600 / around 250 / rotation 200）。引擎 **transformers 5.3.0** 八卡分片 greedy @2048，`disable_thinking`，官方 `input_prompt`。2026-09-08 重跑与 2026-09-06 锚点**逐项相同**。


| 模型  | 题面       | max tokens | overall   | among | around | rotation | 不可读   | 中位 token | 撞 cap |
| --- | -------- | ---------- | --------- | ----- | ------ | -------- | ----- | -------- | ----- |
| 基座  | official | 2048       | **46.86** | 42.50 | 58.80  | 45.00    | 0.00% | 288      | 2.00% |


SPAR3 未测。Dump：`logs/eval/20260908_qwen35base_mindcube_zeroshot/`（20260906 同协议 dump 仍在 `logs/eval/20260906_qwen35_mindcube_sft/baseline/`）。
**不可与** SFT ckpt-314（answer-only 74.48 / CoT 69.81）或训练内 vLLM 锚点（74.29 / 70.57）并列。

## 归档

**BLINK-Spatial HF**（仅 3 任务，不可与全量 14 并列）：original 75.30 / 13.05（句首）；boxed 65.99 / 68.92。HF boxed 全量不存在；全量 boxed 已写入 BLINK 主表（vLLM **62.67 / 63.66**）。