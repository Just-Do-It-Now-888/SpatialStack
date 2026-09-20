#!/usr/bin/env python3
"""Build an HTML viewer for offline VSI-Bench eval (judge_extract samples).

Supports multiple decoding arms (e.g. greedy vs sample) and shows the three
scoring routes: boxed trust, terse trust, and judge extract.

Usage::

    python3 scripts/opsd/tools/build_offline_eval_viewer.py \\
        --arm greedy=logs/vsi_train_eval/.../global_step_225_tok4096_greedy_trust_terse \\
        --arm sample=logs/vsi_train_eval/.../global_step_225_tok4096_sample_trust_terse

    cd /path/to/SpatialStack_OPSD && python3 -m http.server 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = REPO_ROOT / "experiments/viewers"
DEFAULT_PARQUET = REPO_ROOT / "data/eval/vsibench_verl/vsibench_val_boxed.parquet"

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
      --bg: #f4f6f8; --panel: #fff; --text: #1a1d21; --muted: #5c6570; --border: #d8dee6;
      --accent: #2563eb; --boxed: #15803d; --boxed-bg: #f0fdf4; --boxed-border: #86efac;
      --terse: #7c3aed; --terse-bg: #f5f3ff; --terse-border: #c4b5fd;
      --mca-tail: #0891b2; --mca-tail-bg: #ecfeff; --mca-tail-border: #67e8f9;
      --judge: #c2410c; --judge-bg: #fff7ed; --judge-border: #fdba74;
      --ok: #15803d; --bad: #b91c1c; --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
    header {{ position: sticky; top: 0; z-index: 20; background: var(--panel); border-bottom: 1px solid var(--border); padding: 12px 20px; }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.55; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }}
    button, select, input {{ border: 1px solid var(--border); background: var(--panel); border-radius: 6px; padding: 6px 10px; font-size: 13px; }}
    button {{ cursor: pointer; }}
    button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 16px 20px 48px; display: grid; gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
    .card h2 {{ margin: 0 0 10px; font-size: 14px; font-weight: 600; }}
    .split {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    @media (max-width: 1200px) {{ .split {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 700px) {{ .split {{ grid-template-columns: 1fr; }} }}
    .tier-card {{ border-radius: 10px; padding: 12px 14px; border: 2px solid var(--border); }}
    .tier-card.boxed {{ background: var(--boxed-bg); border-color: var(--boxed-border); }}
    .tier-card.terse {{ background: var(--terse-bg); border-color: var(--terse-border); }}
    .tier-card.mca_tail {{ background: var(--mca-tail-bg); border-color: var(--mca-tail-border); }}
    .tier-card.judge {{ background: var(--judge-bg); border-color: var(--judge-border); }}
    .tier-card h3 {{ margin: 0 0 4px; font-size: 13px; }}
    .tier-stat {{ font-size: 22px; font-weight: 700; }}
    .tier-meta {{ font-size: 12px; color: var(--muted); }}
    .compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }}
    .compare .arm {{ border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; }}
    .pill {{ display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); background: #f8fafc; }}
    .pill.ok {{ color: var(--ok); border-color: #86efac; background: #f0fdf4; }}
    .pill.bad {{ color: var(--bad); border-color: #fecaca; background: #fef2f2; }}
    .pill.warn {{ color: var(--warn); border-color: #fde68a; background: #fffbeb; }}
    .pill.boxed {{ color: var(--boxed); border-color: var(--boxed-border); background: var(--boxed-bg); }}
    .pill.terse {{ color: var(--terse); border-color: var(--terse-border); background: var(--terse-bg); }}
    .pill.mca_tail {{ color: var(--mca-tail); border-color: var(--mca-tail-border); background: var(--mca-tail-bg); }}
    .pill.judge {{ color: var(--judge); border-color: var(--judge-border); background: var(--judge-bg); }}
    .route-banner {{ padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 12px; border: 1px solid; }}
    .route-banner.boxed {{ background: var(--boxed-bg); border-color: var(--boxed-border); color: #14532d; }}
    .route-banner.terse {{ background: var(--terse-bg); border-color: var(--terse-border); color: #4c1d95; }}
    .route-banner.mca_tail {{ background: var(--mca-tail-bg); border-color: var(--mca-tail-border); color: #155e75; }}
    .route-banner.judge {{ background: var(--judge-bg); border-color: var(--judge-border); color: #7c2d12; }}
    .question {{ white-space: pre-wrap; font-size: 15px; margin: 0 0 8px; }}
    .options {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
    .scores {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }}
    .score-box {{ border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; background: #f8fafc; }}
    .response {{ margin: 0; padding: 12px; white-space: pre-wrap; word-break: break-word;
      font-family: ui-monospace, monospace; font-size: 12px; background: #0f172a; color: #e2e8f0;
      border-radius: 8px; max-height: 420px; overflow: auto; }}
    mark.boxed-hl {{ background: #fef08a; color: #713f12; padding: 0 2px; border-radius: 3px; }}
    .frames {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px;
      margin-top: 12px; max-height: 72vh; overflow: auto; }}
    .frame {{ border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #eef2f7; min-width: 180px; }}
    .frame img {{ width: 100%; display: block; aspect-ratio: 4/3; object-fit: cover; cursor: zoom-in; }}
    .frame .cap {{ font-size: 11px; text-align: center; padding: 4px; color: var(--muted); background: #f8fafc; }}
    .lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 100;
      align-items: center; justify-content: center; padding: 24px; cursor: zoom-out; }}
    .lightbox.open {{ display: flex; }}
    .lightbox img {{ max-width: min(96vw, 1400px); max-height: 92vh; border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub">{subtitle}</div>
    <div class="toolbar">
      <label>解码臂 <select id="armSelect"></select></label>
      <label>判分路径
        <select id="tierFilter">
          <option value="all">全部</option>
          <option value="boxed">有 \\boxed{{}}（免 Judge）</option>
          <option value="terse">整串/尾行数字字母（免 Judge）</option>
          <option value="mca_tail">MCA 尾行选项（免 Judge）</option>
          <option value="judge">送 Judge</option>
        </select>
      </label>
      <label>结果
        <select id="resultFilter">
          <option value="all">全部</option>
          <option value="correct">判对</option>
          <option value="wrong">判错</option>
          <option value="unanswered">未读出</option>
        </select>
      </label>
      <label><input type="checkbox" id="truncOnly" /> 仅截断</label>
      <label><input type="checkbox" id="disagreeOnly" /> 规则≠最终分</label>
      <button id="prevBtn">← 上一条</button>
      <button id="nextBtn" class="primary">下一条 →</button>
      <label>index <input id="indexInput" type="number" min="0" value="0" style="width:72px" /></label>
      <input id="searchBox" placeholder="搜索 index / scene / 题型" style="width:200px" />
      <span id="posLabel" style="font-size:12px;color:var(--muted)"></span>
    </div>
  </header>
  <main>
    <section class="card">
      <h2>判分路径分布 · <span id="armLabel">—</span></h2>
      <div class="split" id="tierCards"></div>
      <p class="tier-meta" id="overallMeta">—</p>
      <p class="tier-meta" id="abCompare" style="margin-top:8px"></p>
      <div class="compare" id="armCompare"></div>
    </section>
    <section class="card">
      <div id="routeBanner" class="route-banner">加载中…</div>
      <div id="metaPills" class="toolbar" style="margin:0 0 10px;padding:0"></div>
      <p class="question" id="question"></p>
      <ul class="options" id="options"></ul>
      <p style="font-size:13px"><strong>金标：</strong><span id="gt"></span></p>
      <div class="scores" id="scoreBoxes"></div>
      <p style="font-size:13px" id="boxedExtract"></p>
      <pre class="response" id="response"></pre>
      <div class="frames" id="frames"></div>
    </section>
  </main>
  <div class="lightbox" id="lightbox"><img alt="frame" id="lightboxImg" /></div>
  <script>
    const META = {meta_json};
    const SUMMARY = {summary_json};
    const FRAMES = {frames_json};

    let currentArm = META.arms[0];
    let samples = [];
    let filtered = [];
    let cursor = 0;

    const els = {{
      armSelect: document.getElementById("armSelect"),
      tierFilter: document.getElementById("tierFilter"),
      resultFilter: document.getElementById("resultFilter"),
      truncOnly: document.getElementById("truncOnly"),
      disagreeOnly: document.getElementById("disagreeOnly"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      indexInput: document.getElementById("indexInput"),
      searchBox: document.getElementById("searchBox"),
      posLabel: document.getElementById("posLabel"),
      armLabel: document.getElementById("armLabel"),
      tierCards: document.getElementById("tierCards"),
      overallMeta: document.getElementById("overallMeta"),
      abCompare: document.getElementById("abCompare"),
      armCompare: document.getElementById("armCompare"),
      routeBanner: document.getElementById("routeBanner"),
      metaPills: document.getElementById("metaPills"),
      question: document.getElementById("question"),
      options: document.getElementById("options"),
      gt: document.getElementById("gt"),
      scoreBoxes: document.getElementById("scoreBoxes"),
      boxedExtract: document.getElementById("boxedExtract"),
      response: document.getElementById("response"),
      frames: document.getElementById("frames"),
      lightbox: document.getElementById("lightbox"),
      lightboxImg: document.getElementById("lightboxImg"),
    }};

    function escapeHtml(s) {{
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }}
    function pill(t, c) {{ return `<span class="pill ${{c||""}}">${{escapeHtml(t)}}</span>`; }}
    function pct(x) {{ return (100*x).toFixed(2)+"%"; }}
    function highlightBoxed(t) {{
      return escapeHtml(t).replace(/\\\\boxed\\{{([^}}]*)\\}}/g,'<mark class="boxed-hl">\\\\boxed{{$1}}</mark>');
    }}

    async function loadArm(name) {{
      const resp = await fetch(`arm_${{name}}.json`);
      if (!resp.ok) throw new Error(`failed to load arm ${{name}}`);
      const data = await resp.json();
      currentArm = name;
      samples = data;
      cursor = 0;
      render();
    }}

    function tierLabel(t) {{
      if (t === "boxed") return "有 \\\\boxed{{}}";
      if (t === "terse") return "整串/尾行 token";
      if (t === "mca_tail") return "MCA 尾行选项";
      return "送 Judge";
    }}

    function updateSummary() {{
      const s = SUMMARY[currentArm];
      if (!s) return;
      els.armLabel.textContent = currentArm;
      const tiers = ["boxed","terse","mca_tail","judge"];
      const colors = {{boxed:"boxed", terse:"terse", mca_tail:"mca_tail", judge:"judge"}};
      const titles = {{
        boxed: "有 \\\\boxed{{}} · 免 Judge",
        terse: "整串/尾行 token · 免 Judge",
        mca_tail: "MCA 尾行选项 · 免 Judge",
        judge: "送 Judge extract",
      }};
      els.tierCards.innerHTML = tiers.map((t) => {{
        const st = s.by_tier[t];
        return `<div class="tier-card ${{colors[t]}}">
          <h3>${{titles[t]}}</h3>
          <div class="tier-stat">${{pct(st.acc)}}</div>
          <div class="tier-meta">${{st.count}} 题 (${{pct(st.frac)}}) · 截断 ${{pct(st.clip_ratio)}}</div>
        </div>`;
      }}).join("");
      els.overallMeta.textContent =
        `overall=${{pct(s.overall_acc)}} · answered=${{pct(s.answered_frac)}} · clip=${{pct(s.clip_ratio)}} · n=${{s.n}} · max_tokens=${{META.max_tokens}}`;
      if (META.judge_ab) {{
        const ab = META.judge_ab;
        els.abCompare.innerHTML =
          `<strong>规则无法匹配 A/B</strong>（${{ab.unparsed_rule.count}} 题，${{pct(ab.unparsed_rule.frac)}}）：` +
          `送 Judge <strong>${{ab.judge_fallback.overall.toFixed(2)}}%</strong> vs 判错 <strong>${{ab.rule_strict.overall.toFixed(2)}}%</strong> ` +
          `（Δ ${{ab.delta_pp >= 0 ? "+" : ""}}${{ab.delta_pp.toFixed(2)}} pp；Judge 救回 ${{ab.unparsed_rule.judge_correct}}/${{ab.unparsed_rule.count}}）`;
      }} else {{
        els.abCompare.textContent = "";
      }}
      els.armCompare.innerHTML = META.arms.map((a) => {{
        const x = SUMMARY[a];
        return `<div class="arm"><strong>${{a}}</strong><br>overall ${{pct(x.overall_acc)}} · answered ${{pct(x.answered_frac)}} · clip ${{pct(x.clip_ratio)}}<br>trust: boxed ${{x.trust.boxed}} terse ${{x.trust.terse}} mca_tail ${{x.trust.mca_tail||0}} judge ${{x.trust.pending}}</div>`;
      }}).join("");
    }}

    function applyFilters() {{
      const tier = els.tierFilter.value;
      const result = els.resultFilter.value;
      const trunc = els.truncOnly.checked;
      const disagree = els.disagreeOnly.checked;
      const q = els.searchBox.value.trim().toLowerCase();
      filtered = samples.map((_,i)=>i).filter((i) => {{
        const s = samples[i];
        if (tier !== "all" && s.scoring_tier !== tier) return false;
        if (trunc && !s.truncated) return false;
        if (disagree && Math.abs(s.rule_score - s.final_score) < 1e-6) return false;
        if (result === "correct" && s.final_score < 0.999) return false;
        if (result === "wrong" && (s.final_score >= 0.999 || !s.final_answered)) return false;
        if (result === "unanswered" && s.final_answered) return false;
        if (q) {{
          const blob = `${{s.index}} ${{s.scene_name}} ${{s.question_type}}`.toLowerCase();
          if (!blob.includes(q)) return false;
        }}
        return true;
      }});
      if (cursor >= filtered.length) cursor = Math.max(0, filtered.length-1);
    }}

    function renderSample() {{
      if (!filtered.length) {{
        els.routeBanner.textContent = "当前筛选无样本";
        return;
      }}
      const s = samples[filtered[cursor]];
      const t = s.scoring_tier;
      els.routeBanner.className = "route-banner " + t;
      const banners = {{
        boxed: "<strong>路径 A：有闭合 \\\\boxed{{}}</strong> — 规则读框内计分，<strong>不送 Judge</strong>。",
        terse: "<strong>路径 B：整串/尾行 token</strong> — 规则直接读唯一数字或字母，<strong>不送 Judge</strong>（trust_terse）。",
        mca_tail: "<strong>路径 C：MCA 尾行选项</strong> — 最后一行是 C / C. option / 唯一选项正文，<strong>不送 Judge</strong>（trust_mca_tail）。",
        judge: "<strong>路径 D：送 Judge</strong> — 未命中上述信任路径；Judge extract 结果为最终分。",
      }};
      els.routeBanner.innerHTML = banners[t] || "";

      const ok = s.final_score >= 0.999;
      const accCls = ok ? "ok" : s.final_answered ? "bad" : "warn";
      els.metaPills.innerHTML = [
        pill(`index ${{s.index}}`), pill(s.question_type), pill(s.scene_name),
        pill(tierLabel(t), t), pill(ok?"判对":s.final_answered?"判错":"未读出", accCls),
        pill(s.truncated?`截断 ${{s.output_tokens}}tok`:`${{s.output_tokens}} tok`),
        s.boxed_content ? pill(`框内: ${{s.boxed_content}}`, "boxed") : "",
      ].filter(Boolean).join(" ");

      els.question.textContent = s.question;
      els.options.innerHTML = (s.options||[]).map((o)=>`<li>${{escapeHtml(o)}}</li>`).join("");
      els.gt.textContent = s.ground_truth;

      const ruleCls = s.rule_score >= 0.999 ? "ok" : s.rule_answered ? "bad" : "warn";
      const finCls = ok ? "ok" : s.final_answered ? "bad" : "warn";
      els.scoreBoxes.innerHTML = `
        <div class="score-box">规则层（boxed-primary）<br>
          <strong class="${{ruleCls}}">${{(100*s.rule_score).toFixed(1)}}%</strong>
          · answered=${{s.rule_answered}} · parsed=${{escapeHtml(s.rule_parsed||"—")}}</div>
        <div class="score-box">最终分（judge_extract）<br>
          <strong class="${{finCls}}">${{(100*s.final_score).toFixed(1)}}%</strong>
          · answered=${{s.final_answered}} · parsed=${{escapeHtml(s.judge_parsed||"—")}}</div>`;

      if (s.boxed_content) {{
        els.boxedExtract.innerHTML = "<strong>extract_boxed：</strong><code>"+escapeHtml(s.boxed_content)+"</code>";
      }} else {{
        els.boxedExtract.innerHTML = "<strong>extract_boxed：</strong><span style='color:var(--bad)'>无闭合框</span>";
      }}
      els.response.innerHTML = highlightBoxed(s.response);
      els.frames.innerHTML = (FRAMES[s.index]||[]).slice(0,32).map((url,i)=>
        `<figure class="frame"><img src="${{url}}" alt="f${{i}}" loading="lazy" data-full="${{url}}"/><figcaption class="cap">frame ${{String(i).padStart(2,"0")}}</figcaption></figure>`
      ).join("");
      els.frames.querySelectorAll("img").forEach((img)=>{{
        img.addEventListener("click",()=>{{ els.lightboxImg.src=img.dataset.full; els.lightbox.classList.add("open"); }});
      }});
      els.posLabel.textContent = `${{cursor+1}} / ${{filtered.length}}（总 ${{samples.length}}）`;
      els.indexInput.value = s.index;

      const cmp = META.arms.filter((a)=>a!==currentArm).map(async (a)=>{{
        try {{
          const resp = await fetch(`arm_${{a}}.json`);
          const arr = await resp.json();
          const other = arr.find((r)=>r.index===s.index);
          if (!other) return "";
          return `${{a}}: ${{(100*other.final_score).toFixed(1)}}% (${{other.scoring_tier}})`;
        }} catch {{ return ""; }}
      }});
      Promise.all(cmp).then((parts)=>{{
        const text = parts.filter(Boolean).join(" · ");
        if (text) els.posLabel.textContent += " | 对照 " + text;
      }});
    }}

    function render() {{ applyFilters(); updateSummary(); renderSample(); }}
    function go(d) {{ if(!filtered.length) return; cursor=(cursor+d+filtered.length)%filtered.length; renderSample(); }}

    META.arms.forEach((a)=>{{
      const o=document.createElement("option"); o.value=a; o.textContent=a; els.armSelect.appendChild(o);
    }});
    els.armSelect.addEventListener("change", ()=>loadArm(els.armSelect.value));
    ["tierFilter","resultFilter"].forEach((id)=>document.getElementById(id).addEventListener("change",render));
    ["truncOnly","disagreeOnly"].forEach((id)=>document.getElementById(id).addEventListener("change",render));
    els.searchBox.addEventListener("input", render);
    els.prevBtn.addEventListener("click", ()=>go(-1));
    els.nextBtn.addEventListener("click", ()=>go(1));
    els.indexInput.addEventListener("change", ()=>{{
      const want=Number(els.indexInput.value);
      const pos=filtered.findIndex((i)=>samples[i].index===want);
      if(pos>=0){{cursor=pos; renderSample();}}
    }});
    els.lightbox.addEventListener("click", ()=>els.lightbox.classList.remove("open"));
    window.addEventListener("keydown",(e)=>{{
      if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT") return;
      if(e.key==="ArrowLeft"||e.key==="j") go(-1);
      if(e.key==="ArrowRight"||e.key==="k") go(1);
      if(e.key==="Escape") els.lightbox.classList.remove("open");
    }});
    loadArm(META.arms[0]).catch((err) => {{
      els.routeBanner.textContent = "加载失败: " + err.message;
    }});
  </script>
</body>
</html>
"""


def frame_urls(images, viewer_dir: Path) -> list[str]:
    if images is None:
        return []
    if hasattr(images, "tolist"):
        images = images.tolist()
    urls = []
    for img in images:
        path = Path(str(img.get("path", img) if isinstance(img, dict) else img))
        if not path.is_absolute():
            path = REPO_ROOT / path
        rel = os.path.relpath(path, viewer_dir)
        urls.append(rel.replace(os.sep, "/"))
    return urls


def rule_primary_score(row: dict) -> tuple[float, bool, str]:
    boxed = row.get("boxed_content")
    if boxed:
        return (
            float(row.get("boxed_score", 0)),
            bool(row.get("boxed_answered", 0)),
            str(boxed),
        )
    return (
        float(row.get("rule_score", 0)),
        bool(row.get("rule_answered", 0)),
        str(row.get("rule_parsed", "")),
    )


def build_sample(index: int, row: dict, parquet_row: dict) -> dict:
    extra = parquet_row.get("extra_info", {})
    trust = str(row.get("judge_trust") or "").strip()
    if trust == "boxed":
        tier = "boxed"
    elif trust == "terse":
        tier = "terse"
    elif trust == "mca_tail":
        tier = "mca_tail"
    else:
        tier = "judge"
    boxed = row.get("boxed_content")
    if boxed is None and row.get("response"):
        boxed = extract_boxed(str(row.get("response", "")))
    rule_score, rule_answered, rule_parsed = rule_primary_score({**row, "boxed_content": boxed})
    final_score = float(row.get("judge_score", row.get("final_score", 0)))
    return {
        "index": index,
        "id": str(row.get("id", extra.get("id", index))),
        "dataset": str(row.get("dataset", extra.get("dataset", ""))),
        "scene_name": str(row.get("scene_name", extra.get("scene_name", ""))),
        "question_type": str(row.get("question_type", extra.get("question_type", ""))),
        "question": str(row.get("question", "")),
        "options": row.get("options") or [],
        "ground_truth": str(row.get("ground_truth", "")),
        "response": str(row.get("response", "")),
        "output_tokens": int(row.get("output_tokens", 0)),
        "truncated": bool(row.get("truncated", False)),
        "boxed_content": boxed,
        "boxed_present": boxed is not None,
        "scoring_tier": tier,
        "sent_to_judge": tier == "judge",
        "rule_score": rule_score,
        "rule_answered": rule_answered,
        "rule_parsed": rule_parsed,
        "final_score": final_score,
        "final_answered": bool(row.get("judge_answered", row.get("final_answered", 0))),
        "judge_parsed": str(row.get("judge_parsed", "")),
        "judge_reply": str(row.get("judge_extract_reply", ""))[:200],
    }


def tier_stats(samples: list[dict]) -> dict:
    n = len(samples)
    if not n:
        return {"n": 0, "overall_acc": 0, "answered_frac": 0, "clip_ratio": 0, "by_tier": {}}

    def mean_acc(subset):
        return sum(s["final_score"] for s in subset) / len(subset) if subset else 0.0

    by_tier = {}
    for tier in ("boxed", "terse", "mca_tail", "judge"):
        sub = [s for s in samples if s["scoring_tier"] == tier]
        by_tier[tier] = {
            "count": len(sub),
            "frac": len(sub) / n,
            "acc": mean_acc(sub),
            "clip_ratio": sum(1 for s in sub if s["truncated"]) / max(len(sub), 1),
        }
    trust = {
        "boxed": by_tier["boxed"]["count"],
        "terse": by_tier["terse"]["count"],
        "mca_tail": by_tier["mca_tail"]["count"],
        "pending": by_tier["judge"]["count"],
    }
    return {
        "n": n,
        "overall_acc": sum(s["final_score"] for s in samples) / n,
        "answered_frac": sum(1 for s in samples if s["final_answered"]) / n,
        "clip_ratio": sum(1 for s in samples if s["truncated"]) / n,
        "trust": trust,
        "by_tier": by_tier,
    }


def parse_arm_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"arm must be name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return name.strip(), p


def load_arm_samples(arm_dir: Path) -> list[dict]:
    samples_path = arm_dir / "judge_extract" / "samples.jsonl"
    if not samples_path.exists():
        samples_path = arm_dir / "samples.jsonl"
    if not samples_path.exists():
        raise SystemExit(f"no samples.jsonl under {arm_dir}")
    rows = []
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="name=path to eval dir (uses judge_extract/samples.jsonl)",
    )
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--out-dir", default=str(VIEWER_ROOT / "step225_tok4096_trust_terse"))
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    if not args.arm:
        base = REPO_ROOT / "logs/vsi_train_eval/20260826_mvopsd_vlm3r_epoch1_boxed_sample"
        args.arm = [
            f"greedy={base}/global_step_225_tok4096_greedy_trust_terse",
            f"sample={base}/global_step_225_tok4096_sample_trust_terse",
        ]

    arms: dict[str, Path] = {}
    for spec in args.arm:
        name, path = parse_arm_arg(spec)
        arms[name] = path

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = REPO_ROOT / parquet_path
    df = pd.read_parquet(parquet_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_by_index = [frame_urls(df.iloc[i].to_dict().get("images"), out_dir) for i in range(len(df))]
    (out_dir / "frames.json").write_text(json.dumps(frames_by_index, ensure_ascii=False), encoding="utf-8")

    arm_data: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    max_tokens = 4096

    for name, arm_dir in arms.items():
        rows = load_arm_samples(arm_dir)
        if len(rows) != len(df):
            raise SystemExit(f"{name}: row count {len(rows)} != parquet {len(df)}")
        samples = [build_sample(i, rows[i], df.iloc[i].to_dict()) for i in range(len(rows))]
        arm_data[name] = samples
        summary[name] = tier_stats(samples)
        summary_path = arm_dir / "judge_extract" / "summary.json"
        if summary_path.exists():
            jsum = json.loads(summary_path.read_text(encoding="utf-8"))
            max_tokens = int(jsum.get("max_tokens", max_tokens))
        (out_dir / f"arm_{name}.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: overall={summary[name]['overall_acc']*100:.2f}% trust={summary[name]['trust']}")

    meta = {
        "arms": list(arms.keys()),
        "max_tokens": max_tokens,
        "protocol": "offline eval · boxed prompt · trust_boxed + trust_terse + trust_mca_tail · judge_extract v2",
        "arm_dirs": {k: str(v.relative_to(REPO_ROOT)) for k, v in arms.items()},
    }
    ab_path = next(iter(arms.values())) / "judge_ab_comparison.json"
    if ab_path.exists():
        meta["judge_ab"] = json.loads(ab_path.read_text(encoding="utf-8"))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    title = args.title or "step225 · tok4096 · trust_boxed+trust_terse · greedy vs sample"
    subtitle = (
        f"离线评测 · max_tokens={max_tokens} · 四路径：boxed / terse / mca_tail / judge · "
        f"双臂对照：{', '.join(arms.keys())}"
    )
    html = HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        meta_json=json.dumps(meta, ensure_ascii=False),
        summary_json=json.dumps(summary, ensure_ascii=False),
        frames_json=json.dumps(frames_by_index, ensure_ascii=False),
    )
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"\nwrote {index_path}")
    print(f"open: cd {REPO_ROOT} && python3 -m http.server 8765")
    print(f"      http://localhost:8765/{index_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
