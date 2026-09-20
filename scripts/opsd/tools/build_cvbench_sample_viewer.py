#!/usr/bin/env python3
"""Build a self-contained HTML viewer for lmms-eval CV-Bench sample dumps."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "opsd"))
from cvbench_scoring import extract_cvbench_option  # noqa: E402
from vsibench_scoring import extract_boxed  # noqa: E402

LEGACY_FIRST_AF = re.compile(r"[ABCDEF]")
STANDALONE_AF = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")
PAREN_AF = re.compile(r"\(([A-F])\)")


def last_line_letter(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return extract_cvbench_option(lines[-1])


def standalone_letters(text: str) -> list[str]:
    return STANDALONE_AF.findall(text or "")


def load_arm(path: Path, arm: str) -> list[dict]:
    rows = []
    for line in path.open():
        rec = json.loads(line)
        doc = rec["doc"]
        resp = (rec.get("filtered_resps") or rec.get("resps") or [""])
        if isinstance(resp, list):
            if resp and isinstance(resp[0], list):
                text = resp[0][0] if resp[0] else ""
            else:
                text = resp[0] if resp else ""
        else:
            text = str(resp)
        gold = (doc.get("answer") or "")[1] if doc.get("answer") else ""
        pred = doc.get("pred_answer") or ""
        choices = doc.get("choices")
        parsed = extract_cvbench_option(text, choices=choices)
        boxed = extract_boxed(text)
        legacy = ""
        m = LEGACY_FIRST_AF.search(text or "")
        if m:
            legacy = m.group(0)
        last = last_line_letter(text)
        letters = standalone_letters(text)
        flags = []
        if not pred:
            flags.append("unanswered")
        if boxed is None:
            flags.append("no_box")
        if pred != parsed:
            flags.append("pred_ne_reparse")
        if last and pred and last != pred:
            flags.append("last_line_differs")
        if legacy and pred and legacy != pred:
            flags.append("legacy_differs")
        if letters.count(pred) > 1 if pred else False:
            flags.append("letter_repeats")
        if len(letters) >= 3:
            flags.append("many_letters")
        if not doc.get("result"):
            flags.append("wrong")
        rows.append(
            {
                "arm": arm,
                "idx": doc.get("idx"),
                "type": doc.get("type"),
                "task": doc.get("task"),
                "source": doc.get("source"),
                "q": doc.get("question"),
                "choices": doc.get("choices"),
                "gold": gold,
                "pred": pred,
                "ok": int(doc.get("result") or 0),
                "answered": int(doc.get("answered") or 0),
                "parsed": parsed,
                "boxed": boxed or "",
                "legacy": legacy,
                "last": last,
                "letters": letters[:12],
                "flags": flags,
                "resp": text,
                "words": len(text.split()),
                "input": rec.get("input") or "",
                "img": f"img/{int(doc['idx']):05d}.png" if doc.get("idx") is not None else "",
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    by_src = Counter()
    n_src = Counter()
    by_task = Counter()
    n_task = Counter()
    flag_c = Counter()
    for r in rows:
        n_src[r["source"]] += 1
        by_src[r["source"]] += r["ok"]
        n_task[r["task"]] += 1
        by_task[r["task"]] += r["ok"]
        for f in r["flags"]:
            flag_c[f] += 1
    return {
        "n": n,
        "ok": ok,
        "acc": round(100.0 * ok / n, 2) if n else 0,
        "unanswered": sum(1 for r in rows if not r["answered"]),
        "wrong": sum(1 for r in rows if not r["ok"]),
        "by_source": {k: {"n": n_src[k], "ok": by_src[k], "acc": round(100 * by_src[k] / n_src[k], 2)} for k in n_src},
        "by_task": {k: {"n": n_task[k], "ok": by_task[k], "acc": round(100 * by_task[k] / n_task[k], 2)} for k in n_task},
        "flags": dict(flag_c),
        "words_p50": sorted(r["words"] for r in rows)[n // 2] if n else 0,
        "words_max": max((r["words"] for r in rows), default=0),
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CV-Bench SPAR3 boxed last-line 样本检查</title>
<style>
:root { --bg:#f4f6f8; --panel:#fff; --text:#1a1d21; --muted:#5c6570; --border:#d8dee6;
  --ok:#15803d; --bad:#b91c1c; --warn:#b45309; --accent:#2563eb; }
* { box-sizing: border-box; }
body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
header { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); padding:12px 18px; }
h1 { margin:0; font-size:16px; }
.sub { font-size:12px; color:var(--muted); margin-top:4px; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
select, input, button { border:1px solid var(--border); background:#fff; border-radius:6px; padding:6px 10px; font-size:13px; }
button { cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1480px; margin:0 auto; padding:16px 18px 48px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
@media (max-width:1100px){ .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.sample { display:grid; grid-template-columns:minmax(220px,320px) 1fr; gap:14px; align-items:start; }
@media (max-width:800px){ .sample { grid-template-columns:1fr; } }
.thumb { width:100%; max-height:360px; object-fit:contain; background:#0f172a; border-radius:8px; border:1px solid var(--border); }
.missing { font-size:12px; color:var(--muted); padding:24px; text-align:center; background:#f8fafc; border-radius:8px; }
.label { font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); text-transform:uppercase; margin:10px 0 4px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--border); }
.pill { display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid var(--border); margin-right:4px; }
.pill.ok { color:var(--ok); background:#f0fdf4; border-color:#86efac; }
.pill.bad { color:var(--bad); background:#fef2f2; border-color:#fecaca; }
.pill.warn { color:var(--warn); background:#fffbeb; border-color:#fde68a; }
.meta { font-size:12px; color:var(--muted); }
.q { font-weight:600; margin:6px 0; }
.choices { font-size:13px; margin:4px 0 8px; }
.prompt, .resp { white-space:pre-wrap; font-family:ui-monospace,monospace; font-size:12.5px; background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; overflow:auto; }
.prompt { max-height:220px; }
.resp { max-height:280px; }
.resp mark { background:#fde68a; }
.nav { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>CV-Bench 样本检查 · SPAR3 boxed last-line</h1>
  <div class="sub">协议：lmms-eval spatialstack · greedy · 1024 tok · boxed_lastline。每条展示原图、实际发给模型的 prompt，以及模型回复。请在本目录打开页面（需要旁边的 <code>img/</code>）。官方 combined 不是本页微平均。默认列出错题。</div>
  <div class="toolbar">
    <select id="arm">
      <option value="k1">K=1 best (step 55)</option>
      <option value="k2">K=2 best (step 15)</option>
      <option value="both">两臂对照（同 idx）</option>
    </select>
    <select id="status">
      <option value="wrong">仅错题</option>
      <option value="unanswered">仅未作答</option>
      <option value="no_box">无闭合 \\boxed{}</option>
      <option value="last_line_differs">末行字母 ≠ 抽出字母</option>
      <option value="legacy_differs">官方首字母 ≠ 当前解析</option>
      <option value="many_letters">独立 A-F ≥ 3 次</option>
      <option value="all">全部</option>
      <option value="ok">仅对题</option>
    </select>
    <select id="source">
      <option value="">全部 source</option>
      <option>ADE20K</option><option>COCO</option><option>Omni3D</option>
    </select>
    <select id="task">
      <option value="">全部 task</option>
      <option>Count</option><option>Relation</option><option>Distance</option><option>Depth</option>
    </select>
    <input id="q" placeholder="搜索题面 / prompt / 回答" style="min-width:240px"/>
    <button class="primary" id="apply">筛选</button>
    <span id="count" class="meta"></span>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card" id="sum-k1"></div>
    <div class="card" id="sum-k2"></div>
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
const PAGE = 12;
let filtered = [];
let page = 0;

function esc(s){
  return String(s??"").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function highlight(text, letter){
  if(!letter) return esc(text);
  const re = new RegExp('(?<![A-Za-z])('+letter+')(?![A-Za-z])|\\\\('+letter+'\\\\)');
  return esc(text).replace(new RegExp('(?<![A-Za-z])('+letter+')(?![A-Za-z])','g'), '<mark>$1</mark>')
                  .replace(new RegExp('\\('+letter+'\\)','g'), '<mark>('+letter+')</mark>');
}
function pill(r){
  return r.ok ? '<span class="pill ok">正确 '+r.pred+'</span>'
              : (r.answered ? '<span class="pill bad">错误 pred='+esc(r.pred)+' gold='+esc(r.gold)+'</span>'
                            : '<span class="pill warn">未作答 gold='+esc(r.gold)+'</span>');
}
function flags(r){
  return (r.flags||[]).filter(f=>f!=='wrong').map(f=>'<span class="pill warn">'+esc(f)+'</span>').join('');
}
function renderSum(id, arm, s){
  const src = Object.entries(s.by_source).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v.acc}%</td><td>${v.ok}/${v.n}</td></tr>`).join('');
  const task = Object.entries(s.by_task).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v.acc}%</td><td>${v.ok}/${v.n}</td></tr>`).join('');
  document.getElementById(id).innerHTML = `
    <h3 style="margin:0 0 8px">${arm}</h3>
    <div>微平均 ${s.acc}%（${s.ok}/${s.n}）· 错 ${s.wrong} · 未作答 ${s.unanswered} · 回答词数 p50=${s.words_p50} max=${s.words_max}</div>
    <div class="meta">注意：这里是样本微平均，官方 combined 是 (ADE+COCO)/2 再与 Omni3D 等权。</div>
    <table><thead><tr><th>source</th><th>acc</th><th>n</th></tr></thead><tbody>${src}</tbody></table>
    <table><thead><tr><th>task</th><th>acc</th><th>n</th></tr></thead><tbody>${task}</tbody></table>
    <div class="meta" style="margin-top:6px">flags: ${esc(JSON.stringify(s.flags))}</div>`;
}
function card(r){
  const ch = (r.choices||[]).map((c,i)=>String.fromCharCode(65+i)+'. '+esc(c)).join(' · ');
  const img = r.img
    ? `<img class="thumb" src="${esc(r.img)}" alt="idx ${r.idx}" loading="lazy" onerror="this.outerHTML='<div class=&quot;missing&quot;>图像未加载，请在 viewers/cvbench_spar3_boxed_lastline 目录打开页面</div>'"/>`
    : `<div class="missing">无图像路径</div>`;
  return `<div class="card" style="margin-top:10px">
    <div class="sample">
      <div>${img}</div>
      <div>
        <div class="meta">idx ${r.idx} · ${esc(r.arm)} · ${esc(r.type)}/${esc(r.task)}/${esc(r.source)} · ${r.words} words
          · parsed=${esc(r.parsed)} boxed=${esc(r.boxed)} last_line=${esc(r.last)} legacy_first=${esc(r.legacy)}</div>
        ${pill(r)} ${flags(r)}
        <div class="q">${esc(r.q)}</div>
        <div class="choices">${ch}</div>
        <div class="label">Prompt（实际发给模型）</div>
        <div class="prompt">${esc(r.input || '(dump 无 input 字段)')}</div>
        <div class="label">模型回复</div>
        <div class="resp">${highlight(r.resp, r.pred)}</div>
      </div>
    </div>
  </div>`;
}
function pairCard(a,b){
  return `<div class="grid" style="margin-top:10px">${card(a)}${b?card(b):'<div class="card">K=2 缺该 idx</div>'}</div>`;
}
function apply(){
  const arm = document.getElementById('arm').value;
  const st = document.getElementById('status').value;
  const src = document.getElementById('source').value;
  const task = document.getElementById('task').value;
  const q = document.getElementById('q').value.trim().toLowerCase();
  const k1 = DATA.filter(r=>r.arm==='k1');
  const byIdx = {};
  DATA.filter(r=>r.arm==='k2').forEach(r=>byIdx[r.idx]=r);
  let base = arm==='both' ? k1 : DATA.filter(r=>r.arm===arm);
  function hit(r){
    if(src && r.source!==src) return false;
    if(task && r.task!==task) return false;
    if(st==='wrong' && r.ok) return false;
    if(st==='ok' && !r.ok) return false;
    if(st==='unanswered' && r.answered) return false;
    if(st==='no_box' && !(r.flags||[]).includes('no_box')) return false;
    if(st==='last_line_differs' && !(r.flags||[]).includes('last_line_differs')) return false;
    if(st==='legacy_differs' && !(r.flags||[]).includes('legacy_differs')) return false;
    if(st==='many_letters' && !(r.flags||[]).includes('many_letters')) return false;
    if(q && !(String(r.q).toLowerCase().includes(q) || String(r.input||'').toLowerCase().includes(q) || String(r.resp).toLowerCase().includes(q))) return false;
    return true;
  }
  filtered = base.filter(hit).map(r => arm==='both' ? [r, byIdx[r.idx]] : r);
  page = 0;
  draw();
}
function draw(){
  const n = filtered.length;
  const pages = Math.max(1, Math.ceil(n/PAGE));
  page = Math.min(page, pages-1);
  document.getElementById('count').textContent = n+' 条';
  document.getElementById('pageinfo').textContent = (page+1)+' / '+pages;
  const slice = filtered.slice(page*PAGE, page*PAGE+PAGE);
  const arm = document.getElementById('arm').value;
  document.getElementById('list').innerHTML = slice.map(x => arm==='both' ? pairCard(x[0], x[1]) : card(x)).join('');
}
document.getElementById('apply').onclick = apply;
document.getElementById('prev').onclick = ()=>{ page=Math.max(0,page-1); draw(); };
document.getElementById('next').onclick = ()=>{ page+=1; draw(); };
['arm','status','source','task'].forEach(id=>document.getElementById(id).onchange=apply);
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter') apply(); });
renderSum('sum-k1','K=1 best step 55', SUM.k1);
renderSum('sum-k2','K=2 best step 15', SUM.k2);
apply();
</script>
</body>
</html>
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k1",
        default=str(
            REPO
            / "logs/eval/20260831_mvopsd_spar3_k1_cvbench_boxed_lastline/20260901/cvbench/output__20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf/20260901_134607_samples_cvbench.jsonl"
        ),
    )
    parser.add_argument(
        "--k2",
        default=str(
            REPO
            / "logs/eval/20260831_mvopsd_spar3_k2_cvbench_boxed_lastline/20260901/cvbench/output__20260831_mvopsd_spar3_k2_epoch1_nothink_global_step_15_hf/20260901_134644_samples_cvbench.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "experiments/viewers/cvbench_spar3_boxed_lastline"),
    )
    args = parser.parse_args()
    k1_path, k2_path = Path(args.k1), Path(args.k2)
    if not k1_path.exists():
        raise SystemExit(f"missing k1 samples: {k1_path}")
    if not k2_path.exists():
        raise SystemExit(f"missing k2 samples: {k2_path} (rsync from node-B if needed)")
    rows = load_arm(k1_path, "k1") + load_arm(k2_path, "k2")
    summary = {
        "k1": summarize([r for r in rows if r["arm"] == "k1"]),
        "k2": summarize([r for r in rows if r["arm"] == "k2"]),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_src = REPO / "data/eval/cvbench_verl/images"
    img_link = out_dir / "img"
    if img_src.is_dir() and not img_link.exists():
        img_link.symlink_to(img_src.resolve(), target_is_directory=True)
    elif not img_link.exists():
        print(f"warning: no image dir at {img_src}; thumbs will 404")
    html = HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace(
        "__SUM__", json.dumps(summary, ensure_ascii=False)
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'} ({len(html)/1e6:.1f} MB, {len(rows)} rows)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
