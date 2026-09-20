#!/usr/bin/env python3
"""HTML browser for the Qwen3.5-4B VSI-Bench original-prompt dump.

    python3 scripts/opsd/tools/build_vsibench_plain_sample_viewer.py
    # open experiments/viewers/vsibench_qwen35base_plain/index.html
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DUMP = (
    REPO
    / "logs/eval/20260902_qwen35base_vsibench_plain/vsibench/models__Qwen3.5-4B"
    / "20260902_221421_samples_vsibench.jsonl"
)
DEFAULT_OUT = REPO / "experiments/viewers/vsibench_qwen35base_plain"

MCA = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VSI-Bench · 基座 original prompt</title>
<style>
:root { --bg:#f4f6f8; --panel:#fff; --text:#1a1d21; --muted:#5c6570; --border:#d8dee6;
  --ok:#15803d; --bad:#b91c1c; --warn:#b45309; --accent:#2563eb; --part:#0369a1; }
* { box-sizing: border-box; }
body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
header { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); padding:12px 18px; }
h1 { margin:0; font-size:16px; }
.sub { font-size:12px; color:var(--muted); margin-top:4px; line-height:1.55; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
select, input, button { border:1px solid var(--border); background:#fff; border-radius:6px; padding:6px 10px; font-size:13px; }
button { cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1200px; margin:0 auto; padding:16px 18px 48px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
@media (max-width:900px){ .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin-bottom:10px; }
.label { font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); text-transform:uppercase; margin:10px 0 4px; }
.pill { display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid var(--border); margin-right:4px; }
.pill.ok { color:var(--ok); background:#f0fdf4; border-color:#86efac; }
.pill.bad { color:var(--bad); background:#fef2f2; border-color:#fecaca; }
.pill.warn { color:var(--warn); background:#fffbeb; border-color:#fde68a; }
.pill.part { color:var(--part); background:#e0f2fe; border-color:#7dd3fc; }
.meta { font-size:12px; color:var(--muted); }
.q { font-weight:600; margin:6px 0; }
.prompt, .resp { white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,Consolas,monospace; font-size:12.5px; background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; }
.resp { max-height:280px; overflow:auto; background:#0f172a; color:#e2e8f0; }
.nav { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--border); }
.options { font-size:13px; color:var(--muted); margin:0; padding-left:18px; }
</style>
</head>
<body>
<header>
  <h1>VSI-Bench 样本 · 基座 original prompt</h1>
  <div class="sub">
    模型 <code>models/Qwen3.5-4B</code> · 协议 <code>spatialstack_plain</code>（无 <code>\boxed{}</code> 后缀）·
    greedy · 1024 token · 现行末行 <code>answer_tail</code> · overall <strong>52.75</strong> / 作答 98.13%。
    不可与 boxed last-line 48.06 并列。视频帧未嵌入，这里只看题面和回复。
  </div>
  <div class="toolbar">
    <select id="status">
      <option value="all">全部结果</option>
      <option value="ok">满分</option>
      <option value="part">数值部分分</option>
      <option value="wrong">零分（已作答）</option>
      <option value="unans">未作答</option>
    </select>
    <select id="kind">
      <option value="">全部题型</option>
    </select>
    <select id="style">
      <option value="">全部回复形态</option>
      <option value="short">短答（≤3 词）</option>
      <option value="long">多行 / 长答</option>
      <option value="letter">像选项字母</option>
      <option value="number">像数字</option>
    </select>
    <input id="q" placeholder="搜索题目 / 回复 / 解析 / 金标" style="min-width:240px"/>
    <button class="primary" id="apply">筛选</button>
    <span id="count" class="meta"></span>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card" id="sum-overall"></div>
    <div class="card" id="sum-type"></div>
  </div>
  <div class="nav">
    <button id="prev">上一页</button>
    <span id="pageinfo"></span>
    <button id="next">下一页</button>
  </div>
  <div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const SUM = __SUM__;
const PAGE = 25;
let filtered = DATA.slice();
let page = 0;

function esc(s){
  return String(s??'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
function pill(row){
  if(!row.answered) return '<span class="pill warn">未作答</span>';
  if(row.full) return '<span class="pill ok">满分</span>';
  if(row.partial) return '<span class="pill part">部分分 '+row.score.toFixed(2)+'</span>';
  return '<span class="pill bad">零分</span>';
}
function card(r){
  const opts = (r.options||[]).length
    ? '<ul class="options">'+r.options.map(o=>'<li>'+esc(o)+'</li>').join('')+'</ul>'
    : '';
  return `<div class="card">
    <div>${pill(r)}
      <span class="pill">${esc(r.qtype)}</span>
      <span class="meta">${esc(r.dataset)} / ${esc(r.scene)} · id ${esc(r.id)}</span>
    </div>
    <div class="q">${esc(r.question)}</div>
    ${opts}
    <div class="label">完整 user prompt（original，无 boxed 后缀）</div>
    <div class="prompt">${esc(r.input)}</div>
    <div class="label">模型回复</div>
    <div class="resp">${esc(r.resp) || '<span class="meta">（空）</span>'}</div>
    <div class="meta" style="margin-top:8px">
      解析 <code>${esc(r.parsed)||'（空）'}</code>
      · 金标 <code>${esc(r.gold)}</code>
      · 分数 ${r.score.toFixed(3)}
      · 回复 ${r.n_chars} 字符 / ${r.n_lines} 行 / ${r.n_words} 词
    </div>
  </div>`;
}
function renderSum(){
  document.getElementById('sum-overall').innerHTML =
    '<h3 style="margin:0 0 8px;font-size:14px">总体</h3>' +
    `<div>题数 ${SUM.n} · 作答 ${SUM.answered_pct}% · overall <strong>${SUM.overall}</strong></div>` +
    `<div class="meta">满分 ${SUM.full} · 部分分 ${SUM.partial} · 零分 ${SUM.wrong} · 未作答 ${SUM.unanswered}</div>` +
    `<div class="meta">回复中位 ${SUM.median_chars} 字符；≤3 词短答 ${SUM.short_pct}%</div>`;
  const rows = Object.entries(SUM.by_type).map(([k,v]) =>
    `<tr><td>${esc(k)}</td><td>${v.n}</td><td>${v.score}</td><td>${v.answered_pct}%</td></tr>`).join('');
  document.getElementById('sum-type').innerHTML =
    '<h3 style="margin:0 0 8px;font-size:14px">题型</h3>' +
    '<table><thead><tr><th>type</th><th>n</th><th>score</th><th>作答</th></tr></thead><tbody>'+rows+'</tbody></table>';
  const sel = document.getElementById('kind');
  [...new Set(DATA.map(r=>r.qtype))].sort().forEach(t => {
    const o = document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o);
  });
}
function apply(){
  const st = document.getElementById('status').value;
  const kind = document.getElementById('kind').value;
  const style = document.getElementById('style').value;
  const q = document.getElementById('q').value.trim().toLowerCase();
  filtered = DATA.filter(r => {
    if(st==='ok' && !r.full) return false;
    if(st==='part' && !r.partial) return false;
    if(st==='wrong' && !r.zero) return false;
    if(st==='unans' && r.answered) return false;
    if(kind && r.qtype!==kind) return false;
    if(style==='short' && r.n_words>3) return false;
    if(style==='long' && r.n_lines<2 && r.n_words<=12) return false;
    if(style==='letter' && !r.like_letter) return false;
    if(style==='number' && !r.like_number) return false;
    if(q){
      const blob = [r.question,r.input,r.resp,r.parsed,r.gold,r.qtype].join(' ').toLowerCase();
      if(!blob.includes(q)) return false;
    }
    return true;
  });
  page = 0;
  draw();
}
function draw(){
  const n = filtered.length;
  const pages = Math.max(1, Math.ceil(n/PAGE));
  page = Math.min(page, pages-1);
  document.getElementById('count').textContent = n+' 条';
  document.getElementById('pageinfo').textContent = (page+1)+' / '+pages;
  document.getElementById('list').innerHTML = filtered.slice(page*PAGE, page*PAGE+PAGE).map(card).join('');
}
document.getElementById('apply').onclick = apply;
document.getElementById('prev').onclick = ()=>{ page=Math.max(0,page-1); draw(); };
document.getElementById('next').onclick = ()=>{ page+=1; draw(); };
['status','kind','style'].forEach(id=>document.getElementById(id).onchange=apply);
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter') apply(); });
renderSum();
apply();
</script>
</body>
</html>
"""


def _resp(row: dict) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        first = filtered[0]
        return str(first[0] if isinstance(first, list) else first)
    resps = row.get("resps")
    if isinstance(resps, list) and resps:
        first = resps[0]
        return str(first[0] if isinstance(first, list) else first)
    doc = row.get("doc") or {}
    return str(doc.get("prediction") or "")


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rec = json.loads(line)
            doc = rec.get("vsibench_score") or rec["doc"]
            qtype = str(doc["question_type"])
            resp = _resp(rec)
            words = resp.split()
            lines = resp.splitlines() or [""]
            score = float(doc.get("accuracy") if qtype in MCA else doc.get("MRA:.5:.95:.05") or 0.0)
            answered = int(doc.get("answered") or 0) == 1
            full = answered and abs(score - 1.0) < 1e-9
            partial = answered and 0.0 < score < 1.0 - 1e-9
            zero = answered and score <= 1e-9
            stripped = resp.strip()
            rows.append(
                {
                    "id": doc.get("id", rec.get("doc_id")),
                    "dataset": str(doc.get("dataset", "")),
                    "scene": str(doc.get("scene_name", "")),
                    "qtype": qtype,
                    "question": str(doc.get("question", "")),
                    "options": doc.get("options") or [],
                    "input": rec.get("input") or "",
                    "resp": resp,
                    "parsed": "" if doc.get("parsed_answer") is None else str(doc.get("parsed_answer")),
                    "gold": str(doc.get("ground_truth", rec.get("target", ""))),
                    "answered": answered,
                    "score": score,
                    "full": full,
                    "partial": partial,
                    "zero": zero,
                    "n_chars": len(resp),
                    "n_words": len(words),
                    "n_lines": len(lines),
                    "like_letter": bool(stripped) and stripped[:1].upper() in "ABCDEF" and len(stripped) <= 4,
                    "like_number": bool(stripped) and stripped[:1].isdigit(),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    answered = sum(1 for r in rows if r["answered"])
    chars = sorted(r["n_chars"] for r in rows)
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["qtype"], []).append(row)

    type_out = {}
    type_scores = []
    direction = []
    for name, group in sorted(by_type.items()):
        mean = 100.0 * sum(r["score"] for r in group) / max(len(group), 1)
        type_out[name] = {
            "n": len(group),
            "score": round(mean, 2),
            "answered_pct": round(100.0 * sum(1 for r in group if r["answered"]) / len(group), 2),
        }
        if name.startswith("object_rel_direction"):
            direction.append(mean)
        else:
            type_scores.append(mean)
    if direction:
        type_scores.append(sum(direction) / len(direction))
    overall = round(sum(type_scores) / max(len(type_scores), 1), 2)

    return {
        "n": n,
        "overall": overall,
        "answered_pct": round(100.0 * answered / max(n, 1), 2),
        "full": sum(1 for r in rows if r["full"]),
        "partial": sum(1 for r in rows if r["partial"]),
        "wrong": sum(1 for r in rows if r["zero"]),
        "unanswered": sum(1 for r in rows if not r["answered"]),
        "median_chars": chars[len(chars) // 2] if chars else 0,
        "short_pct": round(100.0 * sum(1 for r in rows if r["n_words"] <= 3) / max(n, 1), 1),
        "by_type": type_out,
        "style_counts": dict(Counter("short" if r["n_words"] <= 3 else "long" for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=str(DEFAULT_DUMP))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    samples = Path(args.samples)
    rows = load_rows(samples)
    if not rows:
        raise SystemExit(f"no samples in {samples}")
    summary = summarize(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False, separators=(",", ":"))).replace(
        "__SUM__", json.dumps(summary, ensure_ascii=False)
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_dir / 'index.html'} ({(out_dir / 'index.html').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
