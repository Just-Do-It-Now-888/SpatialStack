# Vision-OPD 论文知识库

此目录是所有 Cursor 窗口共享的论文阅读记忆。聊天用于临时分析，经过核对的结论必须写入这里。

## 目录结构

```text
literature/
├── README.md
├── INDEX.md                 # 已读、在读和待读论文索引
├── SYNTHESIS.md             # 跨论文综合结论
├── OPEN_QUESTIONS.md        # 尚未解决的问题与研究假设
├── templates/
│   └── paper.md             # 单篇论文笔记模板
└── papers/
    └── <method-name>.md     # 每篇论文一份笔记
```

## 文件命名

单篇论文笔记使用论文方法名命名：

- `VOLD`：`vold.md`
- `Vision-OPD`：`vision-opd.md`
- `VA-OPD`：`va-opd.md`

文件名使用小写 kebab-case。同一篇论文只保留一个文件；arXiv/DOI 等稳定标识保存在“基本信息”与 `INDEX.md` 中，用于查重。

单篇论文笔记的一级标题使用论文方法名（例如 `# VOLD`），完整论文标题保存在“基本信息 / 论文标题”字段。

## 阅读流程

1. 阅读前检查 `INDEX.md` 和 `papers/`，确认是否已有笔记。
2. 从 `templates/paper.md` 复制到 `papers/<method-name>.md`。
3. 阅读过程中记录方法、公式、实验、局限和可复用内容。
4. 重要结论必须附章节、页码、公式、图或表等证据定位。
5. 阅读结束后更新 `INDEX.md`。
6. 将跨论文的新认识更新到 `SYNTHESIS.md`。
7. 将暂时无法回答的问题写入 `OPEN_QUESTIONS.md`。

## 记录原则

- 区分论文原文结论和自己的推断。
- 没有证据定位的内容标记为“待核对”。
- 不直接复制大段原文，只保存必要引用和定位。
- 不因新论文观点不同而覆盖旧结论，应同时记录分歧及适用条件。
- 新窗口开始论文相关工作时，先读取索引、综合结论和相关论文笔记。

