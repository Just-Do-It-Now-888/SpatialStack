#!/usr/bin/env python3
"""Build a local HTML browser for in-training VSI-Bench validation dumps.

Joins ``logs/val/<experiment>/<step>.jsonl`` with
``data/eval/vsibench_verl/vsibench_val.parquet`` (same row order) so each page
shows the 32 input frames, question, model response, ground truth, and scores.

Usage::

    python3 scripts/opsd/tools/build_vsibench_val_viewer.py \\
      --dump logs/val/20260821_mvopsd_single_llava_hound_main/175.jsonl

    # then serve the repo root (images load via relative paths):
    cd /path/to/SpatialStack_OPSD && python3 -m http.server 8765
    # open http://localhost:8765/experiments/viewers/vsibench_20260821_mvopsd_single_llava_hound_main_step175/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "opsd"))

from mvopsd_reward import VSIBENCH_MCA_TYPES, VSIBENCH_NA_TYPES  # noqa: E402
from vsibench_eval_core import classify_failure  # noqa: E402

FAILURE_LABELS = {
    "correct": "正确（满分）",
    "truncation_error": "截断错误",
    "parse_error": "解析失败",
    "partial_error": "数值部分正确",
    "factual_error": "事实性错误",
    "judge_recovered": "Judge 救回",
}

DEFAULT_PARQUET = REPO_ROOT / "data/eval/vsibench_verl/vsibench_val.parquet"
VIEWER_ROOT = REPO_ROOT / "experiments/viewers"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VSI-Bench 验证输出浏览器 — {title}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1a1d21;
      --muted: #5c6570;
      --border: #d8dee6;
      --accent: #2563eb;
      --ok: #15803d;
      --bad: #b91c1c;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }}
    h1 {{
      font-size: 16px;
      margin: 0;
      font-weight: 600;
    }}
    .sub {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 2px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    button, select, input[type="number"] {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    button {{
      cursor: pointer;
    }}
    button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 16px 20px 40px;
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 16px;
    }}
    @media (max-width: 1100px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 600;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .pill {{
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #f8fafc;
    }}
    .pill.ok {{ color: var(--ok); border-color: #86efac; background: #f0fdf4; }}
    .pill.bad {{ color: var(--bad); border-color: #fecaca; background: #fef2f2; }}
    .pill.warn {{ color: var(--warn); border-color: #fde68a; background: #fffbeb; }}
    .question {{
      white-space: pre-wrap;
      font-size: 15px;
      margin: 0 0 10px;
    }}
    .options {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
      gap: 8px;
      max-height: 70vh;
      overflow: auto;
    }}
    .frame {{
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #eef2f7;
    }}
    .frame img {{
      width: 100%;
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      cursor: zoom-in;
    }}
    .frame .cap {{
      font-size: 11px;
      text-align: center;
      padding: 4px;
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      max-height: 420px;
      overflow: auto;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }}
    .verdict {{
      font-size: 13px;
      line-height: 1.6;
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      margin: 0;
    }}
    .verdict.ok {{ border-color: #86efac; background: #f0fdf4; }}
    .verdict.bad {{ border-color: #fecaca; background: #fef2f2; }}
    .verdict.warn {{ border-color: #fde68a; background: #fffbeb; }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .score-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 8px;
    }}
    @media (max-width: 900px) {{
      .score-grid {{ grid-template-columns: 1fr; }}
    }}
    .score-col h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 600;
    }}
    .score-col.rule h3 {{ color: #1d4ed8; }}
    .score-col.judge h3 {{ color: #7c3aed; }}
    .summary-banner {{
      background: #eff6ff;
      border-bottom: 1px solid #bfdbfe;
      color: #1e3a8a;
      padding: 10px 20px;
      font-size: 13px;
      line-height: 1.6;
    }}
    .summary-banner.empty {{ display: none; }}
    .label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.82);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox img {{
      max-width: min(96vw, 1200px);
      max-height: 90vh;
      border-radius: 8px;
    }}
    footer.note {{
      max-width: 1400px;
      margin: 0 auto 24px;
      padding: 0 20px;
      font-size: 12px;
      color: var(--muted);
    }}
    .progress-banner {{
      background: #fff7ed;
      border-bottom: 1px solid #fdba74;
      color: #9a3412;
      padding: 10px 20px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .progress-banner.empty {{
      display: none;
    }}
    .progress-banner strong {{
      color: #7c2d12;
    }}
    .meta-warning {{
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: var(--bad);
      border-radius: 8px;
      padding: 8px 12px;
      margin-bottom: 12px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .meta-warning.empty {{
      display: none;
    }}
  </style>
</head>
<body>
  <div id="progressBanner" class="progress-banner"></div>
  <div id="summaryBanner" class="summary-banner"></div>
  <header>
    <div>
      <h1>VSI-Bench 验证输出浏览器</h1>
      <div class="sub">{title}</div>
    </div>
    <div class="toolbar">
      <button id="prevBtn" title="← 上一题">← 上一题</button>
      <button id="nextBtn" class="primary" title="下一题 →">下一题 →</button>
      <label title="val parquet 行号，0 起">parquet index <input id="indexInput" type="number" min="0" value="0" style="width:72px" /></label>
      <span id="posLabel" class="pill">1 / {n}</span>
      <select id="filterSelect">
        <option value="all">全部</option>
        <option value="correct">正确（满分）</option>
        <option value="wrong">错误</option>
        <option value="truncated">撞 {max_tokens} token 上限</option>
        <option value="truncation_error">截断错误</option>
        <option value="partial_error">数值部分正确</option>
        <option value="factual_error">事实性错误</option>
        <option value="parse_error">解析失败</option>
        <option value="meta_mismatch">元数据错位（图不属于本题）</option>
        <option value="disagree_rule_full">规则满分 · Judge 零分（663）</option>
        <option value="disagree_partial_zero">规则部分分 · Judge 零分（361）</option>
        <option value="disagree_judge_full">Judge 满分 · 规则零分（12）</option>
        <option value="disagree_judge_partial">Judge 部分分 · 规则零分（34）</option>
        <option value="disagree_any">任一路分数不一致（1078）</option>
        <option value="rule_ans_judge_none">规则有答案 · Judge NONE（4346）</option>
        <option value="judge_correct">Judge 满分</option>
        <option value="judge_parse_error">Judge 未抽到答案</option>
        <option value="score_mismatch">两路分数不一致</option>
      </select>
      <input id="searchBox" placeholder="搜索 id / parquet index / scene / 题型" style="width:220px" />
    </div>
  </header>

  <main>
    <section class="card">
      <h2>输入图像（32 帧）</h2>
      <div class="meta" id="sceneMeta"></div>
      <div class="meta-warning empty" id="metaWarning"></div>
      <div class="frames" id="frames"></div>
    </section>

    <section class="stack">
      <div class="card">
        <h2>题目</h2>
        <p class="question" id="question"></p>
        <ul class="options" id="options"></ul>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>判分对比（规则 vs Judge 抽取）</h2>
        <div class="meta" id="scoreMeta"></div>
        <div class="score-grid">
          <div class="score-col rule">
            <h3>规则评判</h3>
            <div class="meta" id="ruleMeta"></div>
            <p class="verdict" id="ruleVerdictBox"></p>
          </div>
          <div class="score-col judge">
            <h3>Judge 抽取评判</h3>
            <div class="meta" id="judgeMeta"></div>
            <p class="verdict" id="judgeVerdictBox"></p>
            <div class="label" style="margin-top:8px">Judge 原始回复</div>
            <pre id="judgeReplyBox" style="max-height:120px"></pre>
          </div>
        </div>
        <div class="row" style="margin-top:12px">
          <div>
            <div class="label">评测记录摘要</div>
            <pre id="recordBox"></pre>
          </div>
          <div>
            <div class="label">Ground Truth</div>
            <pre id="gtBox"></pre>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>模型完整回答</h2>
        <pre id="outputBox"></pre>
      </div>
    </section>
  </main>

  <footer class="note">
    提示：请从仓库根目录启动 HTTP 服务后打开本页，否则浏览器可能无法加载本地 PNG。
    命令：<code>cd {repo_root} && python3 -m http.server 8765</code>，
    然后访问当前目录对应的 URL。
    快捷键：<code>←</code>/<code>→</code> 或 <code>J</code>/<code>K</code> 翻页。
  </footer>

  <div class="lightbox" id="lightbox"><img alt="frame" id="lightboxImg" /></div>

  <script>
    const SAMPLES = {samples_json};
    const PROGRESS = {progress_json};
    const SUMMARY = {summary_json};
    let filtered = SAMPLES.map((_, i) => i);
    let cursor = 0;

    (function renderSummary() {{
      const box = document.getElementById("summaryBanner");
      if (!SUMMARY || !SUMMARY.has_judge) {{
        box.classList.add("empty");
        return;
      }}
      box.innerHTML =
        `<strong>全量对比</strong>：规则 overall <strong>${{SUMMARY.rule_overall.toFixed(2)}}%</strong>`
        + `（作答率 ${{SUMMARY.rule_answered_pct.toFixed(1)}}%）`
        + ` vs Judge overall <strong>${{SUMMARY.judge_overall.toFixed(2)}}%</strong>`
        + `（作答率 ${{SUMMARY.judge_answered_pct.toFixed(1)}}%）`
        + `，Δ <strong>${{SUMMARY.delta_pp >= 0 ? "+" : ""}}${{SUMMARY.delta_pp.toFixed(2)}}</strong> pp。`
        + ` 规则满分·Judge 零分：<strong>${{SUMMARY.disagree_rule_full}}</strong> 题；`
        + ` Judge 满分·规则零分：<strong>${{SUMMARY.disagree_judge_full}}</strong> 题。`;
    }})();

    (function renderProgress() {{
      const box = document.getElementById("progressBanner");
      if (!PROGRESS || !PROGRESS.lines || !PROGRESS.lines.length) {{
        box.classList.add("empty");
        return;
      }}
      box.innerHTML = PROGRESS.lines.map((line) => `<div>${{line}}</div>`).join("");
    }})();

    const els = {{
      frames: document.getElementById("frames"),
      question: document.getElementById("question"),
      options: document.getElementById("options"),
      recordBox: document.getElementById("recordBox"),
      gtBox: document.getElementById("gtBox"),
      outputBox: document.getElementById("outputBox"),
      ruleVerdictBox: document.getElementById("ruleVerdictBox"),
      judgeVerdictBox: document.getElementById("judgeVerdictBox"),
      judgeReplyBox: document.getElementById("judgeReplyBox"),
      sceneMeta: document.getElementById("sceneMeta"),
      metaWarning: document.getElementById("metaWarning"),
      scoreMeta: document.getElementById("scoreMeta"),
      ruleMeta: document.getElementById("ruleMeta"),
      judgeMeta: document.getElementById("judgeMeta"),
      posLabel: document.getElementById("posLabel"),
      indexInput: document.getElementById("indexInput"),
      filterSelect: document.getElementById("filterSelect"),
      searchBox: document.getElementById("searchBox"),
      lightbox: document.getElementById("lightbox"),
      lightboxImg: document.getElementById("lightboxImg"),
    }};

    function isFull(score) {{ return Number(score) >= 0.999; }}

    function applyFilter() {{
      const mode = els.filterSelect.value;
      const q = els.searchBox.value.trim().toLowerCase();
      filtered = SAMPLES.map((s, i) => i).filter((i) => {{
        const s = SAMPLES[i];
        const ruleScore = Number(s.rule_score ?? s.acc ?? s.score ?? 0);
        const judgeScore = Number(s.judge_score ?? NaN);
        const hasJudge = s.has_judge === true;
        if (mode === "correct" && s.failure !== "correct") return false;
        if (mode === "wrong" && s.failure === "correct") return false;
        if (mode === "truncated" && !s.truncated) return false;
        if (mode === "meta_mismatch" && s.meta_ok !== false) return false;
        if (mode === "disagree_rule_full" && !(hasJudge && isFull(ruleScore) && judgeScore < 0.001)) return false;
        if (mode === "disagree_partial_zero" && s.disagree_bucket !== "partial_to_zero") return false;
        if (mode === "disagree_judge_full" && !(hasJudge && isFull(judgeScore) && ruleScore < 0.001)) return false;
        if (mode === "disagree_judge_partial" && s.disagree_bucket !== "zero_to_partial") return false;
        if (mode === "disagree_any" && s.disagree_bucket === "agree") return false;
        if (mode === "rule_ans_judge_none" && !(hasJudge && s.rule_answered && !s.judge_answered)) return false;
        if (mode === "judge_correct" && !(hasJudge && isFull(judgeScore))) return false;
        if (mode === "judge_parse_error" && !(hasJudge && !s.judge_answered)) return false;
        if (mode === "score_mismatch" && !(hasJudge && Math.abs(ruleScore - judgeScore) > 0.001)) return false;
        const failureModes = ["all", "correct", "wrong", "truncated", "meta_mismatch",
          "disagree_rule_full", "disagree_partial_zero", "disagree_judge_full", "disagree_judge_partial",
          "disagree_any", "rule_ans_judge_none", "judge_correct", "judge_parse_error", "score_mismatch"];
        if (!failureModes.includes(mode) && s.failure !== mode) return false;
        if (q) {{
          const hay = `${{s.id}} ${{s.index}} ${{s.scene_name}} ${{s.question_type}} ${{s.dataset}}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }}
        return true;
      }});
      cursor = 0;
      render();
    }}

    function render() {{
      if (!filtered.length) {{
        els.frames.innerHTML = "<p>当前筛选无样本。</p>";
        return;
      }}
      const s = SAMPLES[filtered[cursor]];
      els.posLabel.textContent = `${{cursor + 1}} / ${{filtered.length}}（全集 ${{SAMPLES.length}}）`;
      els.indexInput.value = s.index;

      const metaBroken = s.meta_ok === false;
      const metaPills = [
        pill(`id ${{s.id}}`),
        pill(`parquet index ${{s.index}}`),
        pill(s.question_type),
        pill(`${{s.dataset}} / ${{s.scene_name}}`),
        pill(`step ${{s.step}}`),
      ];
      if (metaBroken) metaPills.push(pill("元数据错位", "bad"));
      els.sceneMeta.innerHTML = metaPills.join("");

      els.metaWarning.classList.toggle("empty", !metaBroken);
      els.metaWarning.innerHTML = metaBroken
        ? "<strong>下面这些帧不属于本题</strong>：评测记录与 parquet 行的元数据不一致，"
          + "说明按 id 的 join 没对上，图片来自另一道题的场景，不能用来核对模型推理。"
          + "题面、金标、回复与分数仍来自评测记录，可信。<br>"
          + escapeHtml((s.meta_conflicts || []).join("；"))
        : "";

      const acc = Number(s.acc ?? s.score ?? 0);
      const ruleScore = Number(s.rule_score ?? acc);
      const judgeScore = Number(s.judge_score ?? NaN);
      const hasJudge = s.has_judge === true;
      const scorePill = acc >= 0.999 ? "ok" : (s.truncated ? "bad" : "warn");
      const delta = hasJudge ? (judgeScore - ruleScore) : 0;
      const deltaPill = !hasJudge ? "" : (Math.abs(delta) < 0.001 ? "ok" : "warn");
      els.scoreMeta.innerHTML = [
        pill(`规则 ${{ruleScore.toFixed(3)}}`, ruleScore >= 0.999 ? "ok" : (s.truncated ? "bad" : "warn")),
        hasJudge ? pill(`Judge ${{judgeScore.toFixed(3)}}`, judgeScore >= 0.999 ? "ok" : "bad") : pill("无 Judge 分", "warn"),
        hasJudge ? pill(`Δ ${{delta >= 0 ? "+" : ""}}${{delta.toFixed(3)}}`, deltaPill) : "",
        pill(`tokens ${{s.resp_tokens}}`, s.truncated ? "bad" : ""),
      ].filter(Boolean).join("");

      els.ruleMeta.innerHTML = [
        pill(`score ${{ruleScore.toFixed(3)}}`, ruleScore >= 0.999 ? "ok" : "bad"),
        pill(s.rule_failure_label || s.failure_label || s.failure, s.failure === "correct" ? "ok" : "bad"),
        pill(s.rule_answered ? "answered=1" : "answered=0"),
        pill(`parsed: ${{s.rule_parsed || "(空)"}}`),
      ].join("");

      if (hasJudge) {{
        els.judgeMeta.innerHTML = [
          pill(`score ${{judgeScore.toFixed(3)}}`, judgeScore >= 0.999 ? "ok" : "bad"),
          pill(s.judge_failure_label || "—", s.judge_failure === "correct" ? "ok" : "bad"),
          pill(s.judge_answered ? "answered=1" : "answered=0"),
          pill(`parsed: ${{s.judge_parsed || "NONE"}}`),
        ].join("");
        els.judgeVerdictBox.textContent = s.judge_verdict || "（无 Judge 说明）";
        els.judgeVerdictBox.className = "verdict " + (judgeScore >= 0.999 ? "ok" : (s.truncated ? "bad" : "warn"));
        els.judgeReplyBox.textContent = s.judge_reply || "（无）";
      }} else {{
        els.judgeMeta.innerHTML = pill("未加载 Judge 判分", "warn");
        els.judgeVerdictBox.textContent = "请用 --judge-samples-jsonl 构建 viewer。";
        els.judgeVerdictBox.className = "verdict warn";
        els.judgeReplyBox.textContent = "";
      }}

      els.question.textContent = s.question;
      els.options.innerHTML = (s.options || []).map((o) => `<li>${{escapeHtml(o)}}</li>`).join("");
      els.recordBox.textContent =
        `id: ${{s.id}}\\n` +
        `parquet_index: ${{s.index}}\\n` +
        `rule_score: ${{ruleScore.toFixed(6)}}\\n` +
        `rule_parsed: ${{s.rule_parsed || ""}}\\n` +
        (hasJudge ? `judge_score: ${{judgeScore.toFixed(6)}}\\njudge_parsed: ${{s.judge_parsed || "NONE"}}\\n` : "") +
        `answered(rule): ${{s.rule_answered ?? (s.answered ? 1 : 0)}}\\n` +
        `resp_tokens: ${{s.resp_tokens}}\\n` +
        `resp_chars: ${{s.resp_chars}}\\n` +
        `step: ${{s.step}}`;
      els.gtBox.textContent = s.gt;
      els.outputBox.textContent = s.output;
      els.ruleVerdictBox.textContent = s.rule_verdict || s.verdict || "";
      els.ruleVerdictBox.className = "verdict " + (ruleScore >= 0.999 ? "ok" : (s.truncated ? "bad" : "warn"));

      els.frames.innerHTML = s.images.map((src, i) => `
        <figure class="frame">
          <img src="${{src}}" alt="frame ${{i}}" loading="lazy" data-full="${{src}}" />
          <figcaption class="cap">frame ${{String(i).padStart(2, "0")}}</figcaption>
        </figure>`).join("");
      els.frames.querySelectorAll("img").forEach((img) => {{
        img.addEventListener("click", () => {{
          els.lightboxImg.src = img.dataset.full;
          els.lightbox.classList.add("open");
        }});
      }});
    }}

    function pill(text, tone="") {{
      return `<span class="pill ${{tone}}">${{escapeHtml(text)}}</span>`;
    }}
    function escapeHtml(s) {{
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }}

    function go(delta) {{
      if (!filtered.length) return;
      cursor = Math.max(0, Math.min(filtered.length - 1, cursor + delta));
      render();
    }}

    document.getElementById("prevBtn").addEventListener("click", () => go(-1));
    document.getElementById("nextBtn").addEventListener("click", () => go(1));
    els.filterSelect.addEventListener("change", applyFilter);
    els.searchBox.addEventListener("input", applyFilter);
    const urlParams = new URLSearchParams(window.location.search);
    const bucket = urlParams.get("bucket");
    if (bucket) {{
      const opt = Array.from(els.filterSelect.options).find((o) => o.value === bucket);
      if (opt) {{ els.filterSelect.value = bucket; applyFilter(); }}
    }}
    const jumpId = urlParams.get("id");
    if (jumpId) {{
      els.searchBox.value = jumpId;
      applyFilter();
    }}
    els.indexInput.addEventListener("change", () => {{
      const want = Number(els.indexInput.value);
      const pos = filtered.findIndex((i) => SAMPLES[i].index === want);
      if (pos >= 0) {{ cursor = pos; render(); }}
    }});
    els.lightbox.addEventListener("click", () => els.lightbox.classList.remove("open"));
    window.addEventListener("keydown", (e) => {{
      if (e.target.tagName === "INPUT") return;
      if (e.key === "ArrowLeft" || e.key === "j" || e.key === "J") go(-1);
      if (e.key === "ArrowRight" || e.key === "k" || e.key === "K") go(1);
      if (e.key === "Escape") els.lightbox.classList.remove("open");
    }});

    render();
  </script>
</body>
</html>
"""


def parse_question_from_input(input_text: str) -> tuple[str, list[str]]:
    user = input_text
    if "user\n" in user:
        user = user.split("user\n", 1)[1]
    if "\nassistant" in user:
        user = user.split("\nassistant", 1)[0]

    prefix = "These are frames of a video.\n"
    if user.startswith(prefix):
        user = user[len(prefix) :]

    options: list[str] = []
    if "\nOptions:\n" in user:
        question, opts = user.split("\nOptions:\n", 1)
        if "\nPlease answer" in opts:
            opts = opts.split("\nPlease answer", 1)[0]
        elif "\nAnswer with" in opts:
            opts = opts.split("\nAnswer with", 1)[0]
        options = [line.strip() for line in opts.splitlines() if line.strip()]
        question = question.strip()
    elif "\nPlease answer" in user:
        question = user.split("\nPlease answer", 1)[0].strip()
    elif "\nAnswer with" in user:
        question = user.split("\nAnswer with", 1)[0].strip()
    else:
        question = user.strip()
    return question, options


def explain_grading(
    *,
    question_type: str,
    gt: str,
    acc: float,
    score: float,
    answered: bool,
    truncated: bool,
    failure: str,
    max_tokens: int = 1024,
    source_label: str = "训练验证时写入的 logs/val/*.jsonl",
) -> dict:
    """Explain recorded dump fields only — no re-parsing or re-scoring."""
    lines = [
        f"【数据口径】acc / score / answered 均来自{source_label}，"
        "本页不重新跑解析器，也不重算分数。",
    ]

    if truncated:
        lines.append(
            f"1. 输出长度：resp_tokens≥{max_tokens - 1}，"
            f"生成在 {max_tokens:,} token 处被截断。"
        )
    else:
        lines.append(f"1. 输出长度：未撞满 {max_tokens:,} token 上限。")

    if question_type in VSIBENCH_MCA_TYPES:
        metric = "exact_match"
        lines.append(f"2. 题型：选择题（{question_type}）。")
        lines.append("   训练时规则：从回答中提取选项字母，与金标精确匹配；acc 只能是 0 或 1。")
        lines.append(f"3. 金标：{gt.strip().upper()}")
        if acc >= 1.0 - 1e-9:
            lines.append(f"4. 训练内记录：acc={acc:.3f} → 判对。")
        elif answered:
            lines.append(
                f"4. 训练内记录：acc={acc:.3f}，answered=1 → 读出了可评分字符串，但与金标不一致。"
            )
        else:
            lines.append(f"4. 训练内记录：acc={acc:.3f}，answered=0 → 未能读出有效选项。")
    elif question_type in VSIBENCH_NA_TYPES:
        metric = "mra"
        lines.append(f"2. 题型：数值题（{question_type}）。")
        lines.append("   训练时规则：提取数值后按 MRA(.5:.95:.05) 计分，acc∈[0,1]。")
        lines.append(f"3. 金标：{gt}")
        if acc >= 1.0 - 1e-9:
            lines.append(f"4. 训练内记录：acc={acc:.3f} → MRA 满分，判对。")
        elif acc > 0.0:
            lines.append(f"4. 训练内记录：acc={acc:.3f} → 数值接近金标但未满分（数值部分正确）。")
        elif answered:
            lines.append(f"4. 训练内记录：acc=0，answered=1 → 读出了数值，但与金标偏差过大。")
        else:
            lines.append(f"4. 训练内记录：acc=0，answered=0 → 未能读出数值。")
    else:
        metric = "unknown"
        lines.append(f"2. 未知题型 {question_type}。")

    lines.append(
        f"5. 综合标签：{FAILURE_LABELS.get(failure, failure)}。"
        " 由 dump 中的 acc、answered 与是否截断按固定优先级归类，不是本页重新判分。"
    )
    if truncated and failure == "truncation_error" and acc < 1.0 - 1e-9:
        lines.append("   说明：截断样本优先归入「截断错误」，表示最终答案往往未完整写出。")
    elif truncated and failure == "truncation_error" and acc >= 1.0 - 1e-9:
        lines.append(
            "   说明：虽 acc=1.0，但因撞满 token 上限，标签仍记为截断错误（生成形态异常）。"
        )

    if abs(score - acc) > 1e-9:
        lines.append(f"6. 备注：dump 中 score={score:.6f}，与 acc={acc:.6f} 略有不同（训练内两者通常相同）。")

    return {
        "metric": metric,
        "verdict": "\n".join(lines),
    }


def explain_judge_grading(
    *,
    question_type: str,
    gt: str,
    score: float,
    answered: bool,
    parsed: str,
    truncated: bool,
    failure: str,
    max_tokens: int = 1024,
) -> dict:
    lines = [
        "【Judge 抽取口径】gpt-oss-120b 读全文，只报告模型收敛到的最终答案（或 NONE），"
        "再由官方 EM/MRA 公式算分；Judge 不直接判对错。",
    ]
    if truncated:
        lines.append(f"1. 输出在 {max_tokens:,} token 处截断。")
    else:
        lines.append(f"1. 输出未撞满 {max_tokens:,} token。")

    if question_type in VSIBENCH_MCA_TYPES:
        metric = "exact_match"
        lines.append(f"2. 选择题：Judge 抽出字母后与金标 {gt.strip().upper()} 比。")
    elif question_type in VSIBENCH_NA_TYPES:
        metric = "mra"
        lines.append(f"2. 数值题：Judge 抽出数字后与金标 {gt} 算 MRA。")
    else:
        metric = "unknown"

    lines.append(f"3. Judge 解析：{parsed or 'NONE'}")
    if not answered:
        lines.append("4. Judge 认为未给出可接受的最终答案 → score=0。")
    elif score >= 1.0 - 1e-9:
        lines.append(f"4. score={score:.3f} → 判对。")
    elif score > 0.0:
        lines.append(f"4. score={score:.3f} → 部分正确。")
    else:
        lines.append(f"4. score=0 → 有答案但与金标不符。")
    lines.append(f"5. 标签：{FAILURE_LABELS.get(failure, failure)}。")
    return {"metric": metric, "verdict": "\n".join(lines)}
    rels = []
    for item in images:
        path = Path(item["path"])
        rel = os.path.relpath(path, viewer_dir)
        rels.append(rel.replace(os.sep, "/"))
    return rels


META_FIELDS = ("question_type", "dataset", "scene_name")


def _prefer_dump(dump: dict, extra: dict, key: str, default: str = "") -> str:
    """The eval dump's own copy of a field, falling back to the parquet row.

    The dump is authoritative because it was written next to the response being
    displayed. The parquet is only needed for the frames and for the in-training
    dumps, which carry no metadata of their own.
    """
    for source in (dump, extra):
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def build_sample(
    row: dict,
    dump: dict,
    viewer_dir: Path,
    step: int,
    *,
    max_tokens: int = 1024,
    source_label: str = "训练验证时写入的 logs/val/*.jsonl",
    judge_dump: dict | None = None,
) -> dict:
    extra = row["extra_info"]
    # Both sides describe the same question only if the join was right. Frames
    # come from the parquet row, so a conflict means the images on screen belong
    # to a different scene than the response and must not be read as evidence.
    meta_conflicts = [
        f"{key}: dump={dump[key]} / parquet={extra.get(key)}"
        for key in META_FIELDS
        if dump.get(key) not in (None, "")
        and extra.get(key) not in (None, "")
        and str(dump[key]) != str(extra.get(key))
    ]
    score = float(dump.get("acc", dump.get("final_score", dump.get("rule_score", dump.get("score", 0.0)))))
    rule_score = float(
        dump.get("rule_score", judge_dump.get("rule_score_prev", score) if judge_dump else score)
    )
    rule_parsed = str(dump.get("rule_parsed", dump.get("final_parsed", "")))
    rule_answered = bool(dump.get("rule_answered", dump.get("final_answered", dump.get("answered", 0))))
    resp_tokens = float(dump.get("resp_tokens", dump.get("output_tokens", 0.0)))
    answered = bool(dump.get("answered", dump.get("final_answered", rule_answered)))
    truncated = bool(dump.get("truncated", resp_tokens >= max_tokens - 1))
    failure = classify_failure(
        truncated=truncated,
        final_answered=rule_answered,
        score=rule_score,
        judge_recovered=bool(dump.get("judge_used", 0)) and dump.get("failure_reason") == "judge_recovered",
    )
    if dump.get("failure_reason"):
        failure = str(dump["failure_reason"])
    question, options = parse_question_from_input(dump.get("input", ""))
    if not question and dump.get("question"):
        question = str(dump["question"])
    if not options and dump.get("options"):
        options = [str(x) for x in dump["options"]]
    if not options and extra.get("options_json"):
        try:
            parsed = json.loads(extra["options_json"])
            if parsed:
                options = [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

    gt = str(dump.get("gts", dump.get("ground_truth", extra.get("answer", ""))))
    question_type = _prefer_dump(
        dump, extra, "question_type", row["data_source"].split("/", 1)[-1]
    )
    grading = explain_grading(
        question_type=question_type,
        gt=gt,
        acc=rule_score,
        score=float(dump.get("score", rule_score)),
        answered=rule_answered,
        truncated=truncated,
        failure=failure,
        max_tokens=max_tokens,
        source_label=source_label,
    )

    judge_rec = judge_dump or (dump if "judge_score" in dump else None)
    has_judge = judge_rec is not None and "judge_score" in judge_rec
    judge_score = float(judge_rec["judge_score"]) if has_judge else None
    judge_parsed = str(judge_rec.get("judge_parsed", "") or "") if has_judge else ""
    judge_answered = bool(judge_rec.get("judge_answered", 0)) if has_judge else False
    judge_failure = (
        str(judge_rec.get("judge_failure_reason", ""))
        if has_judge and judge_rec.get("judge_failure_reason")
        else (
            classify_failure(
                truncated=truncated,
                final_answered=judge_answered,
                score=judge_score or 0.0,
                judge_recovered=False,
            )
            if has_judge
            else ""
        )
    )
    judge_grading = (
        explain_judge_grading(
            question_type=question_type,
            gt=gt,
            score=judge_score or 0.0,
            answered=judge_answered,
            parsed=judge_parsed,
            truncated=truncated,
            failure=judge_failure,
            max_tokens=max_tokens,
        )
        if has_judge
        else None
    )

    output = dump.get("output", dump.get("response", ""))

    sample = {
        "index": int(extra.get("index", dump.get("index", 0))),
        "id": _prefer_dump(dump, extra, "id"),
        "question_type": question_type,
        "dataset": _prefer_dump(dump, extra, "dataset"),
        "scene_name": _prefer_dump(dump, extra, "scene_name"),
        "meta_ok": not meta_conflicts,
        "meta_conflicts": meta_conflicts,
        "question": question,
        "options": options,
        "images": rel_image_paths(row["images"], viewer_dir),
        "gt": gt,
        "output": output,
        "acc": rule_score,
        "score": rule_score,
        "rule_score": rule_score,
        "rule_parsed": rule_parsed,
        "rule_answered": rule_answered,
        "rule_failure": failure,
        "rule_failure_label": FAILURE_LABELS.get(failure, failure),
        "rule_verdict": grading["verdict"],
        "resp_tokens": resp_tokens,
        "resp_chars": float(dump.get("resp_chars", len(output))),
        "answered": rule_answered,
        "truncated": truncated,
        "failure": failure,
        "failure_label": FAILURE_LABELS.get(failure, failure),
        "metric": grading["metric"],
        "verdict": grading["verdict"],
        "step": int(dump.get("step", step)),
        "max_tokens": max_tokens,
        "has_judge": has_judge,
    }
    if has_judge:
        sample.update(
            {
                "judge_score": judge_score,
                "judge_parsed": judge_parsed or "NONE",
                "judge_answered": judge_answered,
                "judge_failure": judge_failure,
                "judge_failure_label": FAILURE_LABELS.get(judge_failure, judge_failure),
                "judge_verdict": judge_grading["verdict"] if judge_grading else "",
                "judge_reply": str(judge_rec.get("judge_extract_reply", "")),
                "disagree_bucket": score_bucket(rule_score, judge_score),
            }
        )
    else:
        sample["disagree_bucket"] = "no_judge"
    return sample


def rel_image_paths(images: list[dict], viewer_dir: Path) -> list[str]:
    rels = []
    for item in images:
        path = Path(item["path"])
        rel = os.path.relpath(path, viewer_dir)
        rels.append(rel.replace(os.sep, "/"))
    return rels


def parse_eval_progress(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"\[(\d+)/(\d+)\]\s+(\d+)/(\d+)\s+questions\s+\|\s+([\d.]+)\s+q/s\s+\|\s+elapsed\s+([\d.]+)m",
        text,
    )
    if not matches:
        return {"lines": []}
    batch_done, batch_total, done, total, rate, elapsed = matches[-1]
    done_n, total_n = int(done), int(total)
    pct = 100 * done_n / max(total_n, 1)
    lines = [
        f"<strong>4096-token 离线 VSI-Bench 评测进行中</strong>："
        f"batch {batch_done}/{batch_total}，已完成 <strong>{done_n}/{total_n}</strong> 题（{pct:.1f}%），"
        f"速度 {rate} q/s，已用时 {elapsed} 分钟。",
        "samples.jsonl 尚未写出；下方为<strong>训练内 1024-token 验证 dump</strong>，供先浏览形态与判分。",
        "评测结束后请重新运行本构建脚本（或刷新 manifest）以加载 4096 结果。",
    ]
    if "VSI_EVAL_DONE" in text:
        lines = [
            "<strong>4096-token 离线 VSI-Bench 评测已完成</strong>。",
            "若下方仍为 1024 结果，请用 <code>--samples-jsonl</code> 重新构建 viewer。",
        ]
    return {
        "lines": lines,
        "batch_done": int(batch_done),
        "batch_total": int(batch_total),
        "done": done_n,
        "total": total_n,
    }


def load_eval_samples(samples_path: Path) -> list[dict]:
    rows = []
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parquet_by_id(df: pd.DataFrame) -> dict[str, dict]:
    """Rows keyed by ``extra_info.id`` alone, which is what samples.jsonl carries.

    ``id`` and ``index`` must not share a key space. The parquet stores id as a
    string and index as an int, so they never collide natively, but casting both
    with ``str()`` into one dict let ``index=168`` overwrite ``id="168"``: 2,638
    of 5,130 rows then joined to an unrelated question, and the page showed that
    row's question type, scene and 32 frames next to the right response and
    score. Duplicate ids would reintroduce the same silent mixing, so they abort.
    """
    mapping: dict[str, dict] = {}
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        key = str(row["extra_info"].get("id", ""))
        if key in mapping:
            raise SystemExit(f"extra_info.id {key!r} appears twice in the parquet; cannot join by id")
        mapping[key] = row
    return mapping


def score_bucket(rule_score: float, judge_score: float | None) -> str:
    if judge_score is None:
        return "no_judge"
    if abs(rule_score - judge_score) < 0.001:
        return "agree"
    rs, js = rule_score, judge_score
    full = lambda s: s >= 0.999
    zero = lambda s: s < 0.001
    partial = lambda s: not zero(s) and not full(s)
    if full(rs) and zero(js):
        return "full_to_zero"
    if partial(rs) and zero(js):
        return "partial_to_zero"
    if zero(rs) and full(js):
        return "zero_to_full"
    if zero(rs) and partial(js):
        return "zero_to_partial"
    if full(rs) and partial(js):
        return "full_to_partial"
    if partial(rs) and full(js):
        return "partial_to_full"
    if partial(rs) and partial(js):
        return "partial_to_partial"
    return "other_disagree"


def build_dual_summary(samples: list[dict]) -> dict:
    if not samples or not samples[0].get("has_judge"):
        return {"has_judge": False}
    n = len(samples)
    rule_sum = sum(float(s.get("rule_score", s.get("acc", 0))) for s in samples)
    judge_sum = sum(float(s["judge_score"]) for s in samples)
    rule_ans = sum(1 for s in samples if s.get("rule_answered"))
    judge_ans = sum(1 for s in samples if s.get("judge_answered"))
    disagree_rule_full = sum(
        1
        for s in samples
        if float(s.get("rule_score", 0)) >= 0.999 and float(s.get("judge_score", 0)) < 0.001
    )
    disagree_judge_full = sum(
        1
        for s in samples
        if float(s.get("judge_score", 0)) >= 0.999 and float(s.get("rule_score", 0)) < 0.001
    )
    return {
        "has_judge": True,
        "rule_overall": 100 * rule_sum / n,
        "judge_overall": 100 * judge_sum / n,
        "delta_pp": 100 * (judge_sum - rule_sum) / n,
        "rule_answered_pct": 100 * rule_ans / n,
        "judge_answered_pct": 100 * judge_ans / n,
        "disagree_rule_full": disagree_rule_full,
        "disagree_judge_full": disagree_judge_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", default="", help="logs/val/<experiment>/<step>.jsonl")
    parser.add_argument(
        "--samples-jsonl",
        default="",
        help="offline eval samples.jsonl (vLLM harness output)",
    )
    parser.add_argument(
        "--judge-samples-jsonl",
        default="",
        help="judge-extract samples.jsonl (merged by id for dual scoring)",
    )
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET), help="VSI-Bench val parquet")
    parser.add_argument("--max-tokens", type=int, default=1024, help="generation budget for truncation label")
    parser.add_argument("--progress-log", default="", help="eval.log path for in-progress banner")
    parser.add_argument(
        "--out-dir",
        default="",
        help="output directory under experiments/viewers/ (default: derived from dump path)",
    )
    args = parser.parse_args()

    if not args.dump and not args.samples_jsonl:
        raise SystemExit("provide --dump and/or --samples-jsonl")

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = REPO_ROOT / parquet_path
    if not parquet_path.exists():
        raise SystemExit(f"parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    id_map = parquet_by_id(df)

    samples_jsonl = Path(args.samples_jsonl) if args.samples_jsonl else None
    if samples_jsonl and not samples_jsonl.is_absolute():
        samples_jsonl = REPO_ROOT / samples_jsonl

    judge_jsonl = Path(args.judge_samples_jsonl) if args.judge_samples_jsonl else None
    if judge_jsonl and not judge_jsonl.is_absolute():
        judge_jsonl = REPO_ROOT / judge_jsonl

    dump_path = Path(args.dump) if args.dump else None
    if dump_path and not dump_path.is_absolute():
        dump_path = REPO_ROOT / dump_path

    max_tokens = int(args.max_tokens)
    source_label = "离线 vLLM 评测 samples.jsonl"
    step = 175
    exp_name = "vsibench_eval"
    title_suffix = ""

    if samples_jsonl and samples_jsonl.exists():
        eval_rows = load_eval_samples(samples_jsonl)
        step_match = re.search(r"global_step_(\d+)", str(samples_jsonl))
        if step_match:
            step = int(step_match.group(1))
        exp_name = samples_jsonl.parent.parent.name
        viewer_suffix = "_dual" if judge_jsonl else ""
        viewer_name = f"vsibench_{exp_name}_step{step}_tok{max_tokens}{viewer_suffix}"
        viewer_dir = Path(args.out_dir) if args.out_dir else VIEWER_ROOT / viewer_name
        if not viewer_dir.is_absolute():
            viewer_dir = REPO_ROOT / viewer_dir
        viewer_dir.mkdir(parents=True, exist_ok=True)

        judge_by_id: dict[str, dict] = {}
        if judge_jsonl and judge_jsonl.exists():
            for row in load_eval_samples(judge_jsonl):
                judge_by_id[str(row.get("id", ""))] = row
        elif "judge_score" in (eval_rows[0] if eval_rows else {}):
            for row in eval_rows:
                judge_by_id[str(row.get("id", ""))] = row

        samples = []
        for rec in eval_rows:
            key = str(rec.get("id", ""))
            row = id_map.get(key)
            if row is None:
                raise SystemExit(
                    f"samples.jsonl id={key!r} has no matching extra_info.id in {parquet_path}"
                )
            rec = dict(rec)
            judge_rec = judge_by_id.get(key)
            samples.append(
                build_sample(
                    row,
                    rec,
                    viewer_dir,
                    step,
                    max_tokens=max_tokens,
                    source_label=source_label,
                    judge_dump=judge_rec,
                )
            )
        if len(samples) != len(eval_rows):
            raise SystemExit(
                f"viewer built {len(samples)} samples from {len(eval_rows)} eval rows"
            )
        title_suffix = f"离线评测 · max_tokens={max_tokens}"
        if judge_by_id:
            title_suffix += " · 规则+Judge 双判分"
    elif dump_path and dump_path.exists():
        step = int(dump_path.stem)
        exp_name = dump_path.parent.name
        viewer_name = f"vsibench_{exp_name}_step{step}_tok{max_tokens}"
        viewer_dir = Path(args.out_dir) if args.out_dir else VIEWER_ROOT / viewer_name
        if not viewer_dir.is_absolute():
            viewer_dir = REPO_ROOT / viewer_dir
        viewer_dir.mkdir(parents=True, exist_ok=True)

        with dump_path.open(encoding="utf-8") as handle:
            dumps = [json.loads(line) for line in handle if line.strip()]

        if len(df) != len(dumps):
            raise SystemExit(f"row count mismatch: parquet={len(df)} dump={len(dumps)}")

        samples = [
            build_sample(
                df.iloc[i].to_dict(),
                dumps[i],
                viewer_dir,
                step,
                max_tokens=1024,
                source_label="训练验证时写入的 logs/val/*.jsonl",
            )
            for i in range(len(dumps))
        ]
        title_suffix = "训练内 dump · max_tokens=1024（参考）"
    else:
        raise SystemExit("no usable --dump or --samples-jsonl input found")

    progress_log = Path(args.progress_log) if args.progress_log else None
    if progress_log and not progress_log.is_absolute():
        progress_log = REPO_ROOT / progress_log
    progress = parse_eval_progress(progress_log) if progress_log else {"lines": []}

    manifest_path = viewer_dir / "manifest.json"
    manifest_path.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")

    dual_summary = build_dual_summary(samples)
    html_path = viewer_dir / "index.html"
    title = f"{exp_name} · step {step} · {len(samples)} 题 · {title_suffix}"
    html = HTML_TEMPLATE.format(
        title=title,
        n=len(samples),
        max_tokens=max_tokens,
        repo_root=str(REPO_ROOT),
        samples_json=json.dumps(samples, ensure_ascii=False),
        progress_json=json.dumps(progress, ensure_ascii=False),
        summary_json=json.dumps(dual_summary, ensure_ascii=False),
    )
    html_path.write_text(html, encoding="utf-8")

    counts = {}
    for s in samples:
        counts[s["failure"]] = counts.get(s["failure"], 0) + 1
    mismatched = [s for s in samples if not s["meta_ok"]]

    print(f"wrote {html_path}")
    print(f"wrote {manifest_path}")
    print(f"samples: {len(samples)}")
    print("failure counts:", counts)
    print(f"metadata mismatches (frames do not belong to the question): {len(mismatched)}")
    for s in mismatched[:5]:
        print(f"  id={s['id']} {'; '.join(s['meta_conflicts'])}")
    print()
    print("启动本地服务后打开：")
    print(f"  cd {REPO_ROOT}")
    print("  python3 -m http.server 8765")
    if html_path.is_relative_to(REPO_ROOT):
        print(f"  http://localhost:8765/{html_path.relative_to(REPO_ROOT).as_posix()}")
    else:
        print(f"  (--out-dir is outside the repo; serve {html_path.parent} yourself)")


if __name__ == "__main__":
    main()
