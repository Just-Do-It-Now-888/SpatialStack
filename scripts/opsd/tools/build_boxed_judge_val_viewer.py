#!/usr/bin/env python3
"""Build an HTML viewer for in-training VSI-Bench validation with boxed vs fallback tiers.

Shows which rows were scored from a closed ``\\boxed{}`` (rule reads box only) versus
rows with no closed box (rule falls back to full-text parsing; optionally sent to judge
when ``JUDGE=1`` and ``trust_boxed=True``).

Usage::

    python3 scripts/opsd/tools/build_boxed_judge_val_viewer.py \\
        --val-dir logs/val/20260826_mvopsd_vlm3r_epoch1_boxed_sample

    cd /path/to/SpatialStack_OPSD && python3 -m http.server 8765
    # open http://localhost:8765/experiments/viewers/boxed_judge_vlm3r_epoch1_boxed_sample/
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
VIEWER_ROOT = REPO_ROOT / "experiments/viewers"
DEFAULT_PARQUET = REPO_ROOT / "data/eval/vsibench_verl/vsibench_val_boxed.parquet"
DEFAULT_VAL_DIR = REPO_ROOT / "logs/val/20260826_mvopsd_vlm3r_epoch1_boxed_sample"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "opsd"))
from vsibench_scoring import extract_boxed  # noqa: E402

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #fff;
      --text: #1a1d21;
      --muted: #5c6570;
      --border: #d8dee6;
      --accent: #2563eb;
      --boxed: #15803d;
      --boxed-bg: #f0fdf4;
      --boxed-border: #86efac;
      --judge: #c2410c;
      --judge-bg: #fff7ed;
      --judge-border: #fdba74;
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
      z-index: 20;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
    }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.55; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }}
    button, select, input {{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    button {{ cursor: pointer; }}
    button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 16px 20px 48px;
      display: grid;
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .card h2 {{ margin: 0 0 10px; font-size: 14px; font-weight: 600; }}
    .split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    @media (max-width: 900px) {{ .split {{ grid-template-columns: 1fr; }} }}
    .tier-card {{
      border-radius: 10px;
      padding: 14px 16px;
      border: 2px solid var(--border);
    }}
    .tier-card.boxed {{
      background: var(--boxed-bg);
      border-color: var(--boxed-border);
    }}
    .tier-card.no-box {{
      background: var(--judge-bg);
      border-color: var(--judge-border);
    }}
    .tier-card h3 {{
      margin: 0 0 6px;
      font-size: 15px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .tier-card.boxed h3 {{ color: var(--boxed); }}
    .tier-card.no-box h3 {{ color: var(--judge); }}
    .tier-stat {{ font-size: 28px; font-weight: 700; margin: 4px 0; }}
    .tier-meta {{ font-size: 13px; color: var(--muted); }}
    .stack-bar {{
      display: flex;
      height: 28px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      margin-top: 10px;
    }}
    .stack-bar .boxed {{ background: #22c55e; }}
    .stack-bar .no-box {{ background: #f97316; }}
    .stack-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }}
    .curve-wrap {{ height: 160px; position: relative; }}
    canvas {{ width: 100%; height: 100%; }}
    .pill {{
      display: inline-block;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #f8fafc;
    }}
    .pill.ok {{ color: var(--ok); border-color: #86efac; background: #f0fdf4; }}
    .pill.bad {{ color: var(--bad); border-color: #fecaca; background: #fef2f2; }}
    .pill.warn {{ color: var(--warn); border-color: #fde68a; background: #fffbeb; }}
    .pill.boxed {{ color: var(--boxed); border-color: var(--boxed-border); background: var(--boxed-bg); }}
    .pill.no-box {{ color: var(--judge); border-color: var(--judge-border); background: var(--judge-bg); }}
    .route-banner {{
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 13px;
      margin-bottom: 12px;
      border: 1px solid;
    }}
    .route-banner.boxed {{
      background: var(--boxed-bg);
      border-color: var(--boxed-border);
      color: #14532d;
    }}
    .route-banner.no-box {{
      background: var(--judge-bg);
      border-color: var(--judge-border);
      color: #7c2d12;
    }}
    .question {{ white-space: pre-wrap; font-size: 15px; margin: 0 0 8px; }}
    .options {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
    .response {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 12px;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 8px;
      max-height: 420px;
      overflow: auto;
    }}
    mark.boxed-hl {{
      background: #fef08a;
      color: #713f12;
      padding: 0 2px;
      border-radius: 3px;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 12px;
      max-height: 72vh;
      overflow: auto;
      padding: 4px;
    }}
    .frame {{
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: #eef2f7;
      min-width: 180px;
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
      background: #f8fafc;
    }}
    .lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.85);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 24px;
      cursor: zoom-out;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox img {{
      max-width: min(96vw, 1400px);
      max-height: 92vh;
      border-radius: 8px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub" id="protocolSub">加载协议说明…</div>
    <div class="toolbar">
      <label>step <select id="stepSelect"></select></label>
      <label>判分路径
        <select id="tierFilter">
          <option value="all">全部</option>
          <option value="boxed" id="tierOptBoxed">有 \\boxed{{}}</option>
          <option value="no_box" id="tierOptNoBox">无闭合框</option>
        </select>
      </label>
      <label>结果
        <select id="resultFilter">
          <option value="all">全部</option>
          <option value="correct">满分（acc=1）</option>
          <option value="partial">部分分（0&lt;acc&lt;1）</option>
          <option value="zero">零分（已作答 acc=0）</option>
          <option value="unanswered">未读出答案</option>
        </select>
      </label>
      <label><input type="checkbox" id="truncOnly" /> 仅截断</label>
      <button type="button" id="prevBtn">← 上一条</button>
      <button type="button" id="nextBtn" class="primary">下一条 →</button>
      <label>index <input id="indexInput" type="number" min="0" value="0" style="width:72px" /></label>
      <input id="searchBox" placeholder="搜索 index / scene / 题型" style="width:200px" />
      <span id="posLabel" style="font-size:12px;color:var(--muted)"></span>
    </div>
  </header>

  <main>
    <section class="card">
      <h2>判分路径分布 · step <span id="stepLabel">—</span></h2>
      <div class="split">
        <div class="tier-card boxed">
          <h3 id="boxedTierTitle">有 \\boxed{{}} · 规则读框</h3>
          <div class="tier-stat" id="boxedAcc">—</div>
          <div class="tier-meta" id="boxedMeta">—</div>
        </div>
        <div class="tier-card no-box">
          <h3 id="noBoxTierTitle">无闭合框</h3>
          <div class="tier-stat" id="noBoxAcc">—</div>
          <div class="tier-meta" id="noBoxMeta">—</div>
        </div>
      </div>
      <div class="stack-bar" id="stackBar"></div>
      <div class="stack-labels">
        <span id="stackLeft">—</span>
        <span id="stackRight">—</span>
      </div>
      <p class="tier-meta" style="margin-top:10px" id="overallMeta">—</p>
    </section>

    <section class="card">
      <h2>训练曲线（各 step 汇总）</h2>
      <div class="curve-wrap"><canvas id="curveCanvas"></canvas></div>
      <p class="tier-meta" id="curveLegend">绿=有 box 占比 · 蓝=overall acc · 橙=无 box 路径 acc</p>
    </section>

    <section class="card" id="sampleCard">
      <div id="routeBanner" class="route-banner">加载中…</div>
      <div id="metaPills" class="toolbar" style="margin:0 0 10px;padding:0"></div>
      <p class="question" id="question"></p>
      <ul class="options" id="options"></ul>
      <p style="font-size:13px;margin:8px 0"><strong>金标：</strong><span id="gt"></span></p>
      <p style="font-size:13px;margin:8px 0" id="scoreLine"></p>
      <p style="font-size:13px;margin:8px 0" id="boxedExtract"></p>
      <pre class="response" id="response"></pre>
      <div class="frames" id="frames"></div>
    </section>
  </main>

  <div class="lightbox" id="lightbox"><img alt="frame" id="lightboxImg" /></div>

  <script>
    const META = {meta_json};
    const SUMMARY = {summary_json};
    const FRAMES = {frames_json};
    const JUDGE_ENABLED = Boolean(META.judge_enabled);

    const LABELS = JUDGE_ENABLED ? {{
      sub: "协议：boxed prompt + <code>vsibench_boxed_primary</code> + <code>trust_boxed</code> + <strong>Judge 已启用</strong>。"
        + " 有闭合 <code>\\\\boxed{{}}</code> → 规则只读框内、不送 Judge；"
        + " 无闭合框 → 送 Judge（extract），dump 中 <code>acc</code> 可能被 Judge 改写。"
        + " 选择题：对=1 / 错=0。数值题：MRA（10 档相对误差均值，可部分分），不是「等于金标才算对」。",
      tierBoxed: "有 \\\\boxed{{}} · 规则读框 · 不送 Judge",
      tierNoBox: "无闭合框 · 送 Judge",
      tierOptBoxed: "有 \\\\boxed{{}}（不送 Judge）",
      tierOptNoBox: "无闭合框（送 Judge）",
      stackNoBox: "no box",
      pillNoBox: "送 Judge",
      bannerBoxed: "<strong>判分路径 A：有闭合 \\\\boxed{{}}</strong> — 规则层从框内读取答案；"
        + " 因 <code>trust_boxed=True</code>，<strong>不送入 Judge</strong>。",
      bannerNoBox: "<strong>判分路径 B：无闭合 \\\\boxed{{}}</strong> — 无框可信任，"
        + " <strong>送入 Judge（extract）</strong> 重读全文。dump 中 acc 为 Judge 可能改写后的最终分。",
      curveNoBox: "橙=无 box 路径 acc（Judge）",
    }} : {{
      sub: "协议：boxed prompt + <code>vsibench_boxed_primary</code> + <strong>JUDGE=0（纯规则层）</strong>。"
        + " 有闭合 <code>\\\\boxed{{}}</code> → 规则只读框内；"
        + " 无闭合框 → 规则回退<strong>全文解析</strong>（<strong>未调用 Judge</strong>）。"
        + " dump 中 <code>acc</code> 均为规则分。"
        + " 选择题：对=1 / 错=0。数值题：MRA（10 档相对误差均值，可部分分），不是「等于金标才算对」。",
      tierBoxed: "有 \\\\boxed{{}} · 规则读框",
      tierNoBox: "无闭合框 · 全文规则解析",
      tierOptBoxed: "有 \\\\boxed{{}}（读框内）",
      tierOptNoBox: "无闭合框（全文规则解析）",
      stackNoBox: "无 box",
      pillNoBox: "全文规则",
      bannerBoxed: "<strong>判分路径 A：有闭合 \\\\boxed{{}}</strong> — 规则层从框内读取答案并计分。"
        + " 本轮 <code>JUDGE=0</code>，<strong>未调用 Judge</strong>。",
      bannerNoBox: "<strong>判分路径 B：无闭合 \\\\boxed{{}}</strong> — 无闭合框，规则层对<strong>全文</strong>做解析打分。"
        + " 本轮 <code>JUDGE=0</code>，<strong>未调用 Judge</strong>（不是送判）。",
      curveNoBox: "橙=无 box 路径 acc（全文规则）",
    }};

    document.getElementById("protocolSub").innerHTML = LABELS.sub;
    document.getElementById("boxedTierTitle").textContent = LABELS.tierBoxed;
    document.getElementById("noBoxTierTitle").textContent = LABELS.tierNoBox;
    document.getElementById("tierOptBoxed").textContent = LABELS.tierOptBoxed;
    document.getElementById("tierOptNoBox").textContent = LABELS.tierOptNoBox;
    document.getElementById("curveLegend").textContent =
      "绿=有 box 占比 · 蓝=overall acc · " + LABELS.curveNoBox;

    let currentStep = META.steps[0];
    let samples = [];
    let filtered = [];
    let cursor = 0;

    const els = {{
      stepSelect: document.getElementById("stepSelect"),
      tierFilter: document.getElementById("tierFilter"),
      resultFilter: document.getElementById("resultFilter"),
      truncOnly: document.getElementById("truncOnly"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      indexInput: document.getElementById("indexInput"),
      searchBox: document.getElementById("searchBox"),
      posLabel: document.getElementById("posLabel"),
      stepLabel: document.getElementById("stepLabel"),
      boxedAcc: document.getElementById("boxedAcc"),
      boxedMeta: document.getElementById("boxedMeta"),
      noBoxAcc: document.getElementById("noBoxAcc"),
      noBoxMeta: document.getElementById("noBoxMeta"),
      stackBar: document.getElementById("stackBar"),
      stackLeft: document.getElementById("stackLeft"),
      stackRight: document.getElementById("stackRight"),
      overallMeta: document.getElementById("overallMeta"),
      routeBanner: document.getElementById("routeBanner"),
      metaPills: document.getElementById("metaPills"),
      question: document.getElementById("question"),
      options: document.getElementById("options"),
      gt: document.getElementById("gt"),
      scoreLine: document.getElementById("scoreLine"),
      boxedExtract: document.getElementById("boxedExtract"),
      response: document.getElementById("response"),
      frames: document.getElementById("frames"),
      lightbox: document.getElementById("lightbox"),
      lightboxImg: document.getElementById("lightboxImg"),
      curveCanvas: document.getElementById("curveCanvas"),
    }};

    function escapeHtml(s) {{
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }}

    function pill(text, cls) {{
      return `<span class="pill ${{cls || ""}}">${{escapeHtml(text)}}</span>`;
    }}

    function highlightBoxed(text) {{
      const escaped = escapeHtml(text);
      return escaped.replace(/\\\\boxed\\{{([^}}]*)\\}}/g, '<mark class="boxed-hl">\\\\boxed{{$1}}</mark>');
    }}

    function pct(x) {{ return (100 * x).toFixed(2) + "%"; }}

    const NA_TYPES = new Set([
      "object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance",
    ]);

    function scoreBucket(s) {{
      if (!s.answered) return "unanswered";
      if (s.acc >= 0.999) return "full";
      if (s.acc > 1e-9) return "partial";
      return "zero";
    }}

    function scoreLabel(s) {{
      const b = scoreBucket(s);
      if (b === "unanswered") return ["未读出", "warn"];
      if (b === "full") return ["满分", "ok"];
      if (b === "partial") return ["部分分", "warn"];
      return ["零分", "bad"];
    }}

    function applyFilters() {{
      const tier = els.tierFilter.value;
      const result = els.resultFilter.value;
      const trunc = els.truncOnly.checked;
      const q = els.searchBox.value.trim().toLowerCase();
      filtered = samples
        .map((s, i) => i)
        .filter((i) => {{
          const s = samples[i];
          if (tier !== "all" && s.scoring_tier !== tier) return false;
          if (trunc && !s.truncated) return false;
          const bucket = scoreBucket(s);
          if (result === "correct" && bucket !== "full") return false;
          if (result === "partial" && bucket !== "partial") return false;
          if (result === "zero" && bucket !== "zero") return false;
          if (result === "unanswered" && bucket !== "unanswered") return false;
          if (q) {{
            const blob = `${{s.index}} ${{s.scene_name}} ${{s.question_type}} ${{s.dataset}}`.toLowerCase();
            if (!blob.includes(q)) return false;
          }}
          return true;
        }});
      if (cursor >= filtered.length) cursor = Math.max(0, filtered.length - 1);
    }}

    function updateStepSummary(step) {{
      const s = SUMMARY.by_step[String(step)];
      if (!s) return;
      els.stepLabel.textContent = step;
      els.boxedAcc.textContent = pct(s.boxed_acc);
      els.boxedMeta.textContent =
        `${{s.boxed_count.toLocaleString()}} 题 (${{pct(s.boxed_frac)}}) · 截断 ${{pct(s.boxed_clip_ratio)}}`;
      els.noBoxAcc.textContent = pct(s.no_box_acc);
      els.noBoxMeta.textContent =
        `${{s.no_box_count.toLocaleString()}} 题 (${{pct(s.no_box_frac)}}) · 截断 ${{pct(s.no_box_clip_ratio)}}`;
      const bPct = 100 * s.boxed_frac;
      const nPct = 100 * s.no_box_frac;
      els.stackBar.innerHTML =
        `<div class="boxed" style="width:${{bPct}}%"></div><div class="no-box" style="width:${{nPct}}%"></div>`;
      els.stackLeft.textContent = `有 box ${{bPct.toFixed(1)}}%`;
      els.stackRight.textContent = `${{LABELS.stackNoBox}} ${{nPct.toFixed(1)}}%`;
      els.overallMeta.textContent =
        `overall acc=${{pct(s.overall_acc)}} · clip=${{pct(s.clip_ratio)}} · answered=${{pct(s.answered_frac)}} · n=${{s.n}}`;
    }}

    function renderSample() {{
      if (!filtered.length) {{
        els.routeBanner.textContent = "当前筛选无样本";
        els.routeBanner.className = "route-banner";
        els.metaPills.innerHTML = "";
        els.question.textContent = "";
        els.options.innerHTML = "";
        els.gt.textContent = "";
        els.scoreLine.textContent = "";
        els.boxedExtract.textContent = "";
        els.response.textContent = "";
        els.frames.innerHTML = "";
        els.posLabel.textContent = "0 / 0";
        return;
      }}
      const s = samples[filtered[cursor]];
      const isBoxed = s.scoring_tier === "boxed";
      els.routeBanner.className = "route-banner " + s.scoring_tier;
      if (isBoxed) {{
        els.routeBanner.innerHTML = LABELS.bannerBoxed;
      }} else {{
        els.routeBanner.innerHTML = LABELS.bannerNoBox;
      }}

      const [verdict, accCls] = scoreLabel(s);
      els.metaPills.innerHTML = [
        pill(`index ${{s.index}}`),
        pill(s.question_type),
        pill(s.scene_name),
        pill(isBoxed ? "读框内" : LABELS.pillNoBox, isBoxed ? "boxed" : "no-box"),
        pill(verdict, accCls),
        pill(`acc=${{Number(s.acc).toFixed(2)}}`, accCls),
        pill(s.truncated ? `截断 ${{s.resp_tokens}}tok` : `${{s.resp_tokens}} tok`),
        s.boxed_content ? pill(`框内: ${{s.boxed_content}}`, "boxed") : "",
      ].filter(Boolean).join(" ");

      els.question.textContent = s.question;
      els.options.innerHTML = (s.options || []).map((o) => `<li>${{escapeHtml(o)}}</li>`).join("");
      els.gt.textContent = s.ground_truth;
      const isNa = NA_TYPES.has(s.question_type);
      const accPct = (100 * s.acc).toFixed(0);
      if (!s.answered) {{
        els.scoreLine.innerHTML = "<strong>规则分：</strong>未读出答案 → acc=0";
      }} else if (isNa) {{
        els.scoreLine.innerHTML =
          "<strong>规则分：</strong>数值题 MRA = <code>" + Number(s.acc).toFixed(2)
          + "</code>（" + accPct + " 分制下的 " + accPct
          + "；10 档相对误差，不是对/错二元）。";
      }} else {{
        els.scoreLine.innerHTML =
          "<strong>规则分：</strong>选择题 exact match → acc=<code>"
          + Number(s.acc).toFixed(2) + "</code>"
          + (s.acc >= 0.999 ? "（对）" : "（错）");
      }}
      if (s.boxed_content) {{
        els.boxedExtract.innerHTML =
          "<strong>extract_boxed：</strong><code>" + escapeHtml(s.boxed_content) + "</code>";
      }} else {{
        els.boxedExtract.innerHTML =
          "<strong>extract_boxed：</strong><span style='color:var(--bad)'>无闭合框</span>"
          + (s.truncated ? "（可能因截断导致框未闭合）" : "");
      }}
      els.response.innerHTML = highlightBoxed(s.output);

      const frameUrls = FRAMES[s.index] || [];
      els.frames.innerHTML = frameUrls.slice(0, 32).map((url, fi) =>
        `<figure class="frame">
          <img src="${{url}}" alt="frame ${{fi}}" loading="lazy" data-full="${{url}}" />
          <figcaption class="cap">frame ${{String(fi).padStart(2, "0")}}</figcaption>
        </figure>`
      ).join("");
      els.frames.querySelectorAll("img").forEach((img) => {{
        img.addEventListener("click", () => {{
          els.lightboxImg.src = img.dataset.full || img.src;
          els.lightbox.classList.add("open");
        }});
      }});

      els.posLabel.textContent = `${{cursor + 1}} / ${{filtered.length}}（总 ${{samples.length}}）`;
      els.indexInput.value = s.index;
    }}

    function render() {{
      applyFilters();
      updateStepSummary(currentStep);
      renderSample();
      drawCurve();
    }}

    async function loadStep(step) {{
      const resp = await fetch(`step_${{step}}.json`);
      if (!resp.ok) throw new Error(`failed to load step ${{step}}`);
      const data = await resp.json();
      samples = data.samples;
      cursor = 0;
      currentStep = step;
      render();
    }}

    function go(delta) {{
      if (!filtered.length) return;
      cursor = (cursor + delta + filtered.length) % filtered.length;
      renderSample();
    }}

    function drawCurve() {{
      const canvas = els.curveCanvas;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const steps = SUMMARY.steps;
      if (steps.length < 2) {{
        ctx.fillStyle = "#5c6570";
        ctx.font = "13px sans-serif";
        ctx.fillText("仅一个 step，无曲线", 12, 24);
        return;
      }}

      const pad = {{ l: 36, r: 12, t: 12, b: 24 }};
      const plotW = w - pad.l - pad.r;
      const plotH = h - pad.t - pad.b;
      const xs = steps.map((_, i) => pad.l + (i / (steps.length - 1)) * plotW);

      function line(values, color) {{
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        values.forEach((v, i) => {{
          const x = xs[i];
          const y = pad.t + plotH * (1 - v);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }}

      const boxedFrac = steps.map((st) => SUMMARY.by_step[String(st)].boxed_frac);
      const overall = steps.map((st) => SUMMARY.by_step[String(st)].overall_acc);
      const noBoxAcc = steps.map((st) => SUMMARY.by_step[String(st)].no_box_acc);

      ctx.strokeStyle = "#d8dee6";
      ctx.lineWidth = 1;
      for (let t = 0; t <= 1; t += 0.25) {{
        const y = pad.t + plotH * (1 - t);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + plotW, y);
        ctx.stroke();
        ctx.fillStyle = "#5c6570";
        ctx.font = "10px sans-serif";
        ctx.fillText(Math.round(t * 100) + "%", 4, y + 3);
      }}

      line(boxedFrac, "#22c55e");
      line(overall, "#2563eb");
      line(noBoxAcc, "#f97316");

      const hi = steps.indexOf(currentStep);
      if (hi >= 0) {{
        ctx.strokeStyle = "#1a1d21";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(xs[hi], pad.t);
        ctx.lineTo(xs[hi], pad.t + plotH);
        ctx.stroke();
        ctx.setLineDash([]);
      }}
    }}

    META.steps.forEach((st) => {{
      const opt = document.createElement("option");
      opt.value = st;
      opt.textContent = st;
      els.stepSelect.appendChild(opt);
    }});
    els.stepSelect.value = String(currentStep);

    els.stepSelect.addEventListener("change", () => loadStep(Number(els.stepSelect.value)));
    els.tierFilter.addEventListener("change", render);
    els.resultFilter.addEventListener("change", render);
    els.truncOnly.addEventListener("change", render);
    els.searchBox.addEventListener("input", render);
    els.prevBtn.addEventListener("click", () => go(-1));
    els.nextBtn.addEventListener("click", () => go(1));
    els.indexInput.addEventListener("change", () => {{
      const want = Number(els.indexInput.value);
      const pos = filtered.findIndex((i) => samples[i].index === want);
      if (pos >= 0) {{ cursor = pos; renderSample(); }}
    }});
    els.lightbox.addEventListener("click", () => els.lightbox.classList.remove("open"));
    window.addEventListener("resize", drawCurve);
    window.addEventListener("keydown", (e) => {{
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowLeft" || e.key === "j") go(-1);
      if (e.key === "ArrowRight" || e.key === "k") go(1);
      if (e.key === "Escape") els.lightbox.classList.remove("open");
    }});

    loadStep(currentStep).catch((err) => {{
      els.routeBanner.textContent = "加载失败: " + err.message;
    }});
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


def frame_urls(images, viewer_dir: Path) -> list[str]:
    if images is None:
        return []
    if hasattr(images, "tolist"):
        images = images.tolist()
    urls = []
    for img in images:
        if isinstance(img, dict):
            path = Path(str(img.get("path", "")))
        else:
            path = Path(str(img))
        if not path.is_absolute():
            path = REPO_ROOT / path
        rel = os.path.relpath(path, viewer_dir)
        urls.append(rel.replace(os.sep, "/"))
    return urls


def tier_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "overall_acc": 0.0,
            "boxed_count": 0,
            "no_box_count": 0,
            "boxed_frac": 0.0,
            "no_box_frac": 0.0,
            "boxed_acc": 0.0,
            "no_box_acc": 0.0,
            "clip_ratio": 0.0,
            "boxed_clip_ratio": 0.0,
            "no_box_clip_ratio": 0.0,
            "answered_frac": 0.0,
        }

    def mean_acc(subset: list[dict]) -> float:
        if not subset:
            return 0.0
        return sum(float(r["acc"]) for r in subset) / len(subset)

    def clip_ratio(subset: list[dict]) -> float:
        if not subset:
            return 0.0
        return sum(1 for r in subset if r["truncated"]) / len(subset)

    boxed = [r for r in rows if r["scoring_tier"] == "boxed"]
    no_box = [r for r in rows if r["scoring_tier"] == "no_box"]
    return {
        "n": n,
        "overall_acc": mean_acc(rows),
        "boxed_count": len(boxed),
        "no_box_count": len(no_box),
        "boxed_frac": len(boxed) / n,
        "no_box_frac": len(no_box) / n,
        "boxed_acc": mean_acc(boxed),
        "no_box_acc": mean_acc(no_box),
        "clip_ratio": clip_ratio(rows),
        "boxed_clip_ratio": clip_ratio(boxed),
        "no_box_clip_ratio": clip_ratio(no_box),
        "answered_frac": sum(1 for r in rows if r["answered"]) / n,
    }


def build_sample_row(
    index: int,
    parquet_row: dict,
    dump: dict,
    *,
    max_tokens: int,
    judge_enabled: bool,
) -> dict:
    extra = parquet_row["extra_info"]
    question, options = parse_question_from_input(str(dump.get("input", "")))
    output = str(dump.get("output", ""))
    boxed_present = float(dump.get("boxed_present", 0.0)) >= 0.5
    resp_tokens = int(float(dump.get("resp_tokens", 0)))
    truncated = resp_tokens >= max_tokens - 1
    acc = float(dump.get("acc", dump.get("score", 0.0)))
    answered = bool(dump.get("answered", 0))
    boxed_content = extract_boxed(output)

    return {
        "index": index,
        "id": str(extra.get("id", index)),
        "dataset": str(extra.get("dataset", "")),
        "scene_name": str(extra.get("scene_name", "")),
        "question_type": str(extra.get("question_type", "")),
        "question": question,
        "options": options,
        "ground_truth": str(dump.get("gts", extra.get("answer", ""))),
        "output": output,
        "acc": acc,
        "answered": answered,
        "boxed_present": boxed_present,
        "boxed_content": boxed_content,
        "scoring_tier": "boxed" if boxed_present else "no_box",
        "sent_to_judge": judge_enabled and not boxed_present,
        "resp_tokens": resp_tokens,
        "resp_chars": int(float(dump.get("resp_chars", len(output)))),
        "truncated": truncated,
    }


def discover_steps(val_dir: Path) -> list[int]:
    steps = []
    for path in val_dir.glob("*.jsonl"):
        if path.stem.isdigit():
            steps.append(int(path.stem))
    return sorted(steps)


def detect_judge_enabled(val_dir: Path, explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    exp_name = val_dir.name
    train_dirs = [
        REPO_ROOT / "logs/train" / exp_name,
        val_dir.parent.parent / "train" / exp_name,
    ]
    for train_dir in train_dirs:
        if not train_dir.is_dir():
            continue
        for pattern in ("train_*.log", "driver.log", "nohup.out"):
            for log_path in sorted(train_dir.glob(pattern)):
                text = log_path.read_text(encoding="utf-8", errors="replace")[:200_000]
                if "judge=disabled" in text or "judge=0" in text.lower():
                    return False
                if "judge=gpt-oss" in text or "validation_judge.enable=True" in text:
                    return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-dir", default=str(DEFAULT_VAL_DIR), help="logs/val/<experiment>/")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--steps", default="", help="comma-separated step list; default=all in val-dir")
    parser.add_argument("--out-dir", default="", help="output under experiments/viewers/")
    parser.add_argument("--experiment-label", default="", help="title label override")
    parser.add_argument(
        "--judge-enabled",
        choices=["auto", "yes", "no"],
        default="auto",
        help="whether validation judge ran (auto reads train log for judge=disabled)",
    )
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.is_absolute():
        val_dir = REPO_ROOT / val_dir
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = REPO_ROOT / parquet_path

    if not val_dir.exists():
        raise SystemExit(f"val dir not found: {val_dir}")
    if not parquet_path.exists():
        raise SystemExit(f"parquet not found: {parquet_path}")

    if args.steps.strip():
        steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    else:
        steps = discover_steps(val_dir)
    if not steps:
        raise SystemExit(f"no step jsonl files in {val_dir}")

    judge_explicit: bool | None = None
    if args.judge_enabled == "yes":
        judge_explicit = True
    elif args.judge_enabled == "no":
        judge_explicit = False
    judge_enabled = detect_judge_enabled(val_dir, judge_explicit)
    print(f"judge_enabled={judge_enabled} ({'JUDGE=1' if judge_enabled else 'JUDGE=0 rule-only'})")

    exp_name = val_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else VIEWER_ROOT / f"boxed_judge_{exp_name}"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    max_tokens = int(args.max_tokens)

    frames_by_index = [frame_urls(df.iloc[i].to_dict().get("images"), out_dir) for i in range(len(df))]
    (out_dir / "frames.json").write_text(
        json.dumps(frames_by_index, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_by_step: dict[str, dict] = {}
    for step in steps:
        dump_path = val_dir / f"{step}.jsonl"
        if not dump_path.exists():
            raise SystemExit(f"missing dump: {dump_path}")
        with dump_path.open(encoding="utf-8") as handle:
            dumps = [json.loads(line) for line in handle if line.strip()]
        if len(dumps) != len(df):
            raise SystemExit(f"step {step}: row count mismatch parquet={len(df)} dump={len(dumps)}")

        samples = [
            build_sample_row(
                i, df.iloc[i].to_dict(), dumps[i], max_tokens=max_tokens, judge_enabled=judge_enabled
            )
            for i in range(len(dumps))
        ]
        stats = tier_stats(samples)
        stats["step"] = step
        summary_by_step[str(step)] = stats

        step_payload = {"step": step, "samples": samples}
        (out_dir / f"step_{step}.json").write_text(
            json.dumps(step_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"step {step:>4}: overall={stats['overall_acc']*100:.2f}% "
            f"boxed={stats['boxed_frac']*100:.1f}% ({stats['boxed_acc']*100:.2f}%) "
            f"no_box={stats['no_box_frac']*100:.1f}% ({stats['no_box_acc']*100:.2f}%)"
        )

    summary = {"steps": steps, "by_step": summary_by_step}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    label = args.experiment_label or exp_name
    judge_tag = "Judge ON" if judge_enabled else "rule-only"
    title = f"{label} · boxed vs 无框路径 · {judge_tag} · {len(steps)} steps · n={len(df)}"
    meta = {
        "experiment": exp_name,
        "val_dir": str(val_dir.relative_to(REPO_ROOT)),
        "parquet": str(parquet_path.relative_to(REPO_ROOT)),
        "max_tokens": max_tokens,
        "judge_enabled": judge_enabled,
        "steps": steps,
        "n": len(df),
        "protocol": {
            "boxed_primary": True,
            "trust_boxed": True,
            "judge_mode": "extract" if judge_enabled else None,
            "judge_enabled": judge_enabled,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    html = HTML_TEMPLATE.format(
        title=title,
        meta_json=json.dumps(meta, ensure_ascii=False),
        summary_json=json.dumps(summary, ensure_ascii=False),
        frames_json=json.dumps(frames_by_index, ensure_ascii=False),
    )
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")

    print(f"\nwrote {index_path}")
    print(f"open: cd {REPO_ROOT} && python3 -m http.server 8765")
    if index_path.is_relative_to(REPO_ROOT):
        print(f"      http://localhost:8765/{index_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
