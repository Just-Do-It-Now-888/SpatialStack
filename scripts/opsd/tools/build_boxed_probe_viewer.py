#!/usr/bin/env python3
"""Build a side-by-side HTML viewer for the boxed format probe (plain vs boxed).

Usage::

    python3 scripts/opsd/tools/build_boxed_probe_viewer.py

    cd /path/to/SpatialStack_OPSD && python3 -m http.server 8765
    # open http://localhost:8765/experiments/viewers/boxed_probe_qwen35_base_tok1024/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARQUET = REPO_ROOT / "data/eval/vsibench_verl/vsibench_val.parquet"
VIEWER_ROOT = REPO_ROOT / "experiments/viewers"
DEFAULT_PLAIN = REPO_ROOT / "logs/eval/boxed_probe/vsi_plain/samples.jsonl"
DEFAULT_BOXED = REPO_ROOT / "logs/eval/boxed_probe/vsi_boxed/samples.jsonl"

BOXED_SUFFIX = r" The final answer MUST BE put in \boxed{}."

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qwen3.5-4B · VSI-Bench · plain vs boxed</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #fff;
      --text: #1a1d21;
      --muted: #5c6570;
      --border: #d8dee6;
      --accent: #2563eb;
      --plain: #1d4ed8;
      --boxed: #7c3aed;
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
    }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
    .summary {{
      background: #eff6ff;
      border-bottom: 1px solid #bfdbfe;
      color: #1e3a8a;
      padding: 10px 20px;
      font-size: 13px;
      line-height: 1.6;
    }}
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
      max-width: 1500px;
      margin: 0 auto;
      padding: 16px 20px 40px;
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
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
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
    .pill.plain {{ color: var(--plain); border-color: #93c5fd; background: #eff6ff; }}
    .pill.boxed {{ color: var(--boxed); border-color: #c4b5fd; background: #f5f3ff; }}
    .question {{ white-space: pre-wrap; font-size: 15px; margin: 0 0 8px; }}
    .options {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
    .compare {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    @media (max-width: 1000px) {{ .compare {{ grid-template-columns: 1fr; }} }}
    .arm {{
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 280px;
    }}
    .arm.plain {{ border-top: 3px solid var(--plain); }}
    .arm.boxed {{ border-top: 3px solid var(--boxed); }}
    .arm-head {{
      padding: 8px 12px;
      background: #f8fafc;
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }}
    .arm-head strong {{ font-size: 13px; }}
    .response {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      flex: 1;
      overflow: auto;
      max-height: 520px;
      background: #fff;
    }}
    .response .box {{
      background: #fef9c3;
      border: 1px solid #fde047;
      border-radius: 4px;
      padding: 0 2px;
      font-weight: 600;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
      gap: 6px;
      max-height: 240px;
      overflow: auto;
    }}
    .frame {{
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      background: #eef2f7;
    }}
    .frame img {{
      width: 100%;
      display: block;
      aspect-ratio: 4/3;
      object-fit: cover;
      cursor: zoom-in;
    }}
    .frame .cap {{ font-size: 10px; text-align: center; padding: 2px; color: var(--muted); }}
    footer.note {{
      max-width: 1500px;
      margin: 0 auto 24px;
      padding: 0 20px;
      font-size: 12px;
      color: var(--muted);
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
    .lightbox img {{ max-width: min(96vw, 1200px); max-height: 90vh; border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>Qwen3.5-4B · VSI-Bench · plain vs boxed</h1>
    <div class="sub">{title}</div>
    <div class="toolbar">
      <button id="prevBtn">← 上一题</button>
      <button id="nextBtn" class="primary">下一题 →</button>
      <label>id <input id="idInput" type="number" min="0" style="width:72px" /></label>
      <span id="posLabel" class="pill">1 / {n}</span>
      <select id="filterSelect">
        <option value="all">全部</option>
        <option value="boxed_ok">boxed 合规（有闭合 \\boxed{{}}）</option>
        <option value="boxed_missing">boxed 不合规（无框）</option>
        <option value="plain_shorter">plain 更短（中位对比）</option>
        <option value="boxed_longer">boxed 更长</option>
        <option value="score_plain_better">plain 分数更高</option>
        <option value="score_boxed_better">boxed 分数更高</option>
        <option value="both_correct">两侧都满分</option>
        <option value="truncated_boxed">boxed 撞满 {max_tokens} token</option>
      </select>
      <input id="searchBox" placeholder="搜索 id / scene / 题型 / 问题" style="width:240px" />
    </div>
  </header>
  <div class="summary" id="summaryBanner"></div>

  <main>
    <section class="card">
      <h2>题目</h2>
      <div class="meta" id="questionMeta"></div>
      <p class="question" id="question"></p>
      <ul class="options" id="options"></ul>
      <details style="margin-top:12px">
        <summary style="cursor:pointer;font-size:13px;color:var(--muted)">展开 32 帧输入图</summary>
        <div class="frames" id="frames" style="margin-top:8px"></div>
      </details>
    </section>

    <section class="card">
      <h2>模型回答对比</h2>
      <div class="compare">
        <div class="arm plain">
          <div class="arm-head">
            <strong class="plain">plain</strong>
            <span>无格式后缀</span>
            <span id="plainMeta"></span>
          </div>
          <pre class="response" id="plainResponse"></pre>
        </div>
        <div class="arm boxed">
          <div class="arm-head">
            <strong class="boxed">boxed</strong>
            <span>后缀：{boxed_suffix_escaped}</span>
            <span id="boxedMeta"></span>
          </div>
          <pre class="response" id="boxedResponse"></pre>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>判分摘要</h2>
      <div class="meta" id="scoreMeta"></div>
    </section>
  </main>

  <footer class="note">
    从仓库根目录启动：<code>cd {repo_root} && python3 -m http.server 8765</code>，
    然后打开本页 URL。快捷键 <code>←</code>/<code>→</code> 或 <code>J</code>/<code>K</code> 翻页。
  </footer>

  <div class="lightbox" id="lightbox"><img alt="frame" id="lightboxImg" /></div>

  <script>
    const SAMPLES = {samples_json};
    const SUMMARY = {summary_json};
    let filtered = SAMPLES.map((_, i) => i);
    let cursor = 0;

    function esc(s) {{
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }}

    function highlightBoxed(text) {{
      const safe = esc(text);
      return safe.replace(/\\\\boxed\\{{([\\s\\S]*?)\\}}/g, '<span class="box">\\\\boxed{{$1}}</span>');
    }}

    function pill(text, cls) {{
      return `<span class="pill ${{cls || ""}}">${{esc(text)}}</span>`;
    }}

    function scoreLabel(score) {{
      const s = Number(score);
      if (s >= 0.999) return ["满分", "ok"];
      if (s <= 0.001) return ["零分", "bad"];
      return [`部分分 ${{s.toFixed(2)}}`, "warn"];
    }}

    function renderSummary() {{
      const s = SUMMARY;
      document.getElementById("summaryBanner").innerHTML =
        `<strong>全量 ${{s.n}}</strong> 题 · `
        + `plain overall <strong>${{s.plain_overall.toFixed(2)}}%</strong>（中位 ${{s.plain_median_tokens}} tok）`
        + ` vs boxed overall <strong>${{s.boxed_overall.toFixed(2)}}%</strong>（中位 ${{s.boxed_median_tokens}} tok，合规 ${{s.boxed_compliance_pct.toFixed(1)}}%）`
        + ` · Δ <strong>${{s.delta_pp >= 0 ? "+" : ""}}${{s.delta_pp.toFixed(2)}}</strong> pp`;
    }}

    function applyFilter() {{
      const mode = document.getElementById("filterSelect").value;
      const q = document.getElementById("searchBox").value.trim().toLowerCase();
      filtered = SAMPLES.map((s, i) => {{ s._i = i; return s; }}).filter((s) => {{
        if (mode === "boxed_ok" && !s.boxed_present) return false;
        if (mode === "boxed_missing" && s.boxed_present) return false;
        if (mode === "plain_shorter" && s.plain_tokens >= s.boxed_tokens) return false;
        if (mode === "boxed_longer" && s.boxed_tokens <= s.plain_tokens) return false;
        if (mode === "score_plain_better" && s.plain_score <= s.boxed_score) return false;
        if (mode === "score_boxed_better" && s.boxed_score <= s.plain_score) return false;
        if (mode === "both_correct" && (s.plain_score < 0.999 || s.boxed_score < 0.999)) return false;
        if (mode === "truncated_boxed" && !s.boxed_truncated) return false;
        if (q) {{
          const hay = [
            s.id, s.scene_name, s.question_type, s.question, s.ground_truth,
            s.plain_response, s.boxed_response,
          ].join(" ").toLowerCase();
          if (!hay.includes(q)) return false;
        }}
        return true;
      }}).map((s) => s._i);
      cursor = 0;
      render();
    }}

    function render() {{
      if (!filtered.length) {{
        document.getElementById("posLabel").textContent = "0 / 0";
        return;
      }}
      const idx = filtered[cursor];
      const s = SAMPLES[idx];
      document.getElementById("posLabel").textContent = `${{cursor + 1}} / ${{filtered.length}}`;
      document.getElementById("idInput").value = s.id;

      document.getElementById("questionMeta").innerHTML = [
        pill(`id=${{s.id}}`, ""),
        pill(s.question_type, ""),
        pill(s.scene_name, ""),
        pill(s.dataset, ""),
      ].join("");
      document.getElementById("question").textContent = s.question;
      const optEl = document.getElementById("options");
      if (s.options && s.options.length) {{
        optEl.innerHTML = s.options.map((o) => `<li>${{esc(o)}}</li>`).join("");
        optEl.style.display = "";
      }} else {{
        optEl.innerHTML = "";
        optEl.style.display = "none";
      }}

      const framesEl = document.getElementById("frames");
      framesEl.innerHTML = (s.frames || []).map((f, i) =>
        `<div class="frame"><img src="${{f}}" alt="frame ${{i}}" loading="lazy" /><div class="cap">#${{i}}</div></div>`
      ).join("");
      framesEl.querySelectorAll("img").forEach((img) => {{
        img.addEventListener("click", () => {{
          document.getElementById("lightboxImg").src = img.src;
          document.getElementById("lightbox").classList.add("open");
        }});
      }});

      const [pl, pc] = scoreLabel(s.plain_score);
      const [bl, bc] = scoreLabel(s.boxed_score);
      document.getElementById("plainMeta").innerHTML = [
        pill(`${{s.plain_tokens}} tok`, "plain"),
        pill(pl, pc),
        s.plain_truncated ? pill("截断", "bad") : "",
      ].join("");
      document.getElementById("boxedMeta").innerHTML = [
        pill(`${{s.boxed_tokens}} tok`, "boxed"),
        pill(bl, bc),
        s.boxed_present ? pill("有 box", "ok") : pill("无 box", "bad"),
        s.boxed_truncated ? pill("截断", "bad") : "",
      ].join("");

      document.getElementById("plainResponse").innerHTML = esc(s.plain_response);
      document.getElementById("boxedResponse").innerHTML = highlightBoxed(s.boxed_response);

      document.getElementById("scoreMeta").innerHTML = [
        pill(`GT: ${{s.ground_truth}}`, ""),
        pill(`plain 解析: ${{s.plain_parsed || "—"}}`, "plain"),
        pill(`boxed 解析: ${{s.boxed_parsed || "—"}}`, "boxed"),
        s.boxed_content ? pill(`boxed 内容: ${{s.boxed_content}}`, "boxed") : "",
        pill(`Δ score: ${{(100 * (s.boxed_score - s.plain_score)).toFixed(1)}} pp`, s.boxed_score > s.plain_score ? "ok" : s.boxed_score < s.plain_score ? "bad" : ""),
        pill(`Δ tokens: ${{s.boxed_tokens - s.plain_tokens >= 0 ? "+" : ""}}${{s.boxed_tokens - s.plain_tokens}}`, s.boxed_tokens > s.plain_tokens ? "warn" : ""),
      ].join("");
    }}

    document.getElementById("prevBtn").onclick = () => {{
      if (!filtered.length) return;
      cursor = (cursor - 1 + filtered.length) % filtered.length;
      render();
    }};
    document.getElementById("nextBtn").onclick = () => {{
      if (!filtered.length) return;
      cursor = (cursor + 1) % filtered.length;
      render();
    }};
    document.getElementById("idInput").onchange = (e) => {{
      const want = Number(e.target.value);
      const pos = filtered.findIndex((i) => SAMPLES[i].id === want);
      if (pos >= 0) {{ cursor = pos; render(); }}
    }};
    document.getElementById("filterSelect").onchange = applyFilter;
    document.getElementById("searchBox").oninput = applyFilter;
    document.getElementById("lightbox").onclick = () => {{
      document.getElementById("lightbox").classList.remove("open");
    }};
    document.addEventListener("keydown", (e) => {{
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      if (e.key === "ArrowLeft" || e.key === "k" || e.key === "K") document.getElementById("prevBtn").click();
      if (e.key === "ArrowRight" || e.key === "j" || e.key === "J") document.getElementById("nextBtn").click();
    }});

    renderSummary();
    applyFilter();
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parquet_by_id(df: pd.DataFrame) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        key = str(row["extra_info"].get("id", ""))
        if key in mapping:
            raise SystemExit(f"duplicate extra_info.id {key!r} in parquet")
        mapping[key] = row
    return mapping


def frame_urls(row: dict, viewer_dir: Path) -> list[str]:
    images = row.get("images")
    if images is None:
        return []
    if hasattr(images, "tolist"):
        images = images.tolist()
    urls = []
    for img in images:
        path = Path(str(img))
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = path
        urls.append("/" + str(rel).replace("\\", "/"))
    return urls


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def build_summary(samples: list[dict]) -> dict:
    n = len(samples)
    plain_scores = [float(s["plain_score"]) for s in samples]
    boxed_scores = [float(s["boxed_score"]) for s in samples]
    plain_tokens = [float(s["plain_tokens"]) for s in samples]
    boxed_tokens = [float(s["boxed_tokens"]) for s in samples]
    boxed_ok = sum(1 for s in samples if s["boxed_present"])
    return {
        "n": n,
        "plain_overall": 100 * sum(plain_scores) / n,
        "boxed_overall": 100 * sum(boxed_scores) / n,
        "delta_pp": 100 * (sum(boxed_scores) - sum(plain_scores)) / n,
        "plain_median_tokens": int(median(plain_tokens)),
        "boxed_median_tokens": int(median(boxed_tokens)),
        "boxed_compliance_pct": 100 * boxed_ok / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plain", default=str(DEFAULT_PLAIN), help="plain arm samples.jsonl")
    parser.add_argument("--boxed", default=str(DEFAULT_BOXED), help="boxed arm samples.jsonl")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model-label", default="Qwen3.5-4B")
    parser.add_argument(
        "--out-dir",
        default=str(VIEWER_ROOT / "boxed_probe_qwen35_base_tok1024"),
        help="output directory under experiments/viewers/",
    )
    args = parser.parse_args()

    plain_path = Path(args.plain)
    boxed_path = Path(args.boxed)
    parquet_path = Path(args.parquet)
    out_dir = Path(args.out_dir)
    if not plain_path.is_absolute():
        plain_path = REPO_ROOT / plain_path
    if not boxed_path.is_absolute():
        boxed_path = REPO_ROOT / boxed_path
    if not parquet_path.is_absolute():
        parquet_path = REPO_ROOT / parquet_path
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    for path in (plain_path, boxed_path, parquet_path):
        if not path.exists():
            raise SystemExit(f"missing: {path}")

    plain_rows = {str(r["id"]): r for r in load_jsonl(plain_path)}
    boxed_rows = {str(r["id"]): r for r in load_jsonl(boxed_path)}
    if plain_rows.keys() != boxed_rows.keys():
        only_plain = set(plain_rows) - set(boxed_rows)
        only_boxed = set(boxed_rows) - set(plain_rows)
        raise SystemExit(f"id mismatch: only_plain={len(only_plain)} only_boxed={len(only_boxed)}")

    df = pd.read_parquet(parquet_path)
    id_map = parquet_by_id(df)
    max_tokens = int(args.max_tokens)

    samples: list[dict] = []
    for key in sorted(plain_rows, key=lambda x: int(x)):
        plain = plain_rows[key]
        boxed = boxed_rows[key]
        row = id_map.get(key)
        if row is None:
            raise SystemExit(f"id={key!r} not in parquet")

        options = plain.get("options")
        if options is not None:
            options = [str(x) for x in options]

        samples.append(
            {
                "id": int(key),
                "dataset": str(plain.get("dataset", "")),
                "scene_name": str(plain.get("scene_name", "")),
                "question_type": str(plain.get("question_type", "")),
                "question": str(plain.get("question", "")),
                "options": options,
                "ground_truth": str(plain.get("ground_truth", "")),
                "frames": frame_urls(row, out_dir),
                "plain_response": str(plain.get("response", "")),
                "boxed_response": str(boxed.get("response", "")),
                "plain_tokens": int(plain.get("output_tokens", 0)),
                "boxed_tokens": int(boxed.get("output_tokens", 0)),
                "plain_score": float(plain.get("rule_score", plain.get("final_score", 0))),
                "boxed_score": float(boxed.get("rule_score", boxed.get("final_score", 0))),
                "plain_parsed": str(plain.get("rule_parsed", plain.get("final_parsed", ""))),
                "boxed_parsed": str(boxed.get("rule_parsed", boxed.get("final_parsed", ""))),
                "plain_truncated": bool(plain.get("truncated", int(plain.get("output_tokens", 0)) >= max_tokens - 1)),
                "boxed_truncated": bool(boxed.get("truncated", int(boxed.get("output_tokens", 0)) >= max_tokens - 1)),
                "boxed_present": bool(boxed.get("boxed_present", 0)),
                "boxed_content": boxed.get("boxed_content"),
            }
        )

    summary = build_summary(samples)
    out_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"{args.model_label} · VSI-Bench 全量 {len(samples)} 题 · "
        f"max_tokens={max_tokens} · greedy · 冻结基座 boxed 格式探针"
    )
    html = HTML_TEMPLATE.format(
        title=title,
        n=len(samples),
        max_tokens=max_tokens,
        boxed_suffix_escaped=BOXED_SUFFIX.replace("\\", "\\\\"),
        repo_root=REPO_ROOT,
        samples_json=json.dumps(samples, ensure_ascii=False),
        summary_json=json.dumps(summary, ensure_ascii=False),
    )
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")

    manifest = {
        "title": title,
        "plain_samples": str(plain_path.relative_to(REPO_ROOT)),
        "boxed_samples": str(boxed_path.relative_to(REPO_ROOT)),
        "summary": summary,
        "viewer_url": f"/experiments/viewers/{out_dir.name}/",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {index_path}")
    print(f"summary: plain={summary['plain_overall']:.2f}% boxed={summary['boxed_overall']:.2f}% "
          f"compliance={summary['boxed_compliance_pct']:.1f}%")
    print(f"open: cd {REPO_ROOT} && python3 -m http.server 8765")
    print(f"      http://localhost:8765/experiments/viewers/{out_dir.name}/")


if __name__ == "__main__":
    main()
