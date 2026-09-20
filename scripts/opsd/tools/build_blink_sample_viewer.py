#!/usr/bin/env python3
"""Build an HTML viewer for the SPAR3 K=1 step 55 BLINK-Spatial dump.

Usage::

    python3 scripts/opsd/tools/build_blink_sample_viewer.py
    python3 -m http.server 8765
    # open experiments/viewers/blink_spar3_k1_step55/index.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DUMP = (
    REPO
    / "logs/eval/20260902_spar3_k1_step55_blink_spatial/20260902/blink_spatial"
    / "output__20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf"
)
DEFAULT_DATA = REPO / "cache/datasets/BLINK-Benchmark__BLINK"
DEFAULT_OUT = REPO / "experiments/viewers/blink_spar3_k1_step55"
TASK_DIRS = {
    "Multi-view Reasoning": "Multi-view_Reasoning",
    "Relative Depth": "Relative_Depth",
    "Spatial Relation": "Spatial_Relation",
}
START_LETTER = re.compile(r"[\(\s]*([A-Z])[\)\.\s]*", re.I)
LAST_AF = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")
SYSTEM = "You are a helpful assistant."


def last_af(text: str) -> str:
    matches = list(LAST_AF.finditer(text or ""))
    return matches[-1].group(1) if matches else ""


def start_letter(text: str) -> str:
    match = START_LETTER.match((text or "").strip())
    return match.group(1).upper() if match else ""


def resp_text(rec: dict) -> str:
    acc = rec.get("blink_acc") or {}
    if acc.get("pred"):
        return acc["pred"]
    resp = rec.get("filtered_resps") or rec.get("resps") or [""]
    if isinstance(resp, list):
        if resp and isinstance(resp[0], list):
            return resp[0][0] if resp[0] else ""
        return resp[0] if resp else ""
    return str(resp)


def load_rows(dump_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(dump_dir.glob("*samples_blink_*.jsonl")):
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                doc = rec["doc"]
                acc = rec.get("blink_acc") or {}
                text = resp_text(rec)
                parsed = acc.get("pred_parsed") or start_letter(text)
                gold = acc.get("gt_content") or str(doc.get("answer") or "").strip("()")
                ok = bool(acc.get("is_correct"))
                lead = text.lstrip()[:24]
                flags = []
                if ok:
                    flags.append("ok")
                else:
                    flags.append("wrong")
                if lead.startswith("To determine"):
                    flags.append("lead_to")
                if lead.startswith("Based"):
                    flags.append("lead_based")
                if parsed == "T" and not ok:
                    flags.append("parsed_T")
                tail = last_af(text)
                if tail and tail == gold and not ok:
                    flags.append("tail_would_match")
                user_text = rec.get("input") or ""
                n_choices = len(doc.get("choices") or [])
                letters = ", ".join(chr(65 + i) for i in range(n_choices)) if n_choices else ""
                chat = (
                    f"<|system|>\n{SYSTEM}\n"
                    f"<|user|>\n[image] × {{nimg}}\n"
                    f"{user_text}\n"
                    f"<|assistant|>  (enable_thinking=false)"
                )
                rows.append(
                    {
                        "id": acc.get("id") or doc.get("idx") or "",
                        "task": acc.get("sub_task") or doc.get("sub_task") or "",
                        "question": doc.get("question") or "",
                        "dataset_prompt": doc.get("prompt") or "",
                        "input": user_text,
                        "chat": chat,
                        "choices": doc.get("choices") or [],
                        "choice_letters": letters,
                        "gold": gold,
                        "parsed": parsed,
                        "tail": tail,
                        "ok": int(ok),
                        "flags": flags,
                        "resp": text,
                        "words": len(text.split()),
                        "nimg": 0,
                        "imgs": [],
                    }
                )
    rows.sort(key=lambda r: (r["task"], r["id"]))
    return rows


def summarize(rows: list[dict], official: dict | None) -> dict:
    by_task: dict[str, list[int]] = {}
    flags = Counter()
    for row in rows:
        by_task.setdefault(row["task"], []).append(row["ok"])
        flags.update(row["flags"])
    task_acc = {
        task: round(100.0 * sum(vals) / len(vals), 2) if vals else 0.0
        for task, vals in by_task.items()
    }
    macro = round(sum(task_acc.values()) / len(task_acc), 2) if task_acc else 0.0
    return {
        "n": len(rows),
        "ok": sum(r["ok"] for r in rows),
        "micro": round(100.0 * sum(r["ok"] for r in rows) / len(rows), 2) if rows else 0,
        "macro": official.get("blink_spatial") if official else macro,
        "by_task": {
            task: {
                "n": len(by_task[task]),
                "ok": sum(by_task[task]),
                "acc": task_acc[task],
            }
            for task in sorted(by_task)
        },
        "flags": dict(flags),
        "words_p50": sorted(r["words"] for r in rows)[len(rows) // 2] if rows else 0,
        "words_max": max((r["words"] for r in rows), default=0),
    }


def load_image_index(data_root: Path) -> dict[str, list]:
    from datasets import load_from_disk

    index: dict[str, list] = {}
    for task, folder in TASK_DIRS.items():
        ds = load_from_disk(str(data_root / folder))
        for i in range(len(ds)):
            row = ds[i]
            images = []
            for key, value in row.items():
                if re.match(r"^image_\d+$", key) and value is not None:
                    images.append(value)
            index[row["idx"]] = images
        print(f"loaded {task}: {len(ds)}", file=sys.stderr)
    return index


def save_thumbs(rows: list[dict], image_index: dict[str, list], img_dir: Path, max_side: int, quality: int) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        images = image_index.get(row["id"]) or []
        rels = []
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["id"])
        for k, image in enumerate(images):
            rel = f"img/{stem}_{k}.jpg"
            dest = img_dir / f"{stem}_{k}.jpg"
            if not dest.exists():
                im = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
                width, height = im.size
                scale = max_side / max(width, height)
                if scale < 1:
                    im = im.resize(
                        (max(1, int(width * scale)), max(1, int(height * scale))),
                        Image.Resampling.BILINEAR,
                    )
                im.save(dest, format="JPEG", quality=quality, optimize=True)
            rels.append(rel)
        row["imgs"] = rels
        row["nimg"] = len(rels)
        row["chat"] = row["chat"].replace("{nimg}", str(len(rels)))
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"thumbs {i + 1}/{len(rows)}", file=sys.stderr)


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BLINK-Spatial · SPAR3 K=1 step 55</title>
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
.grid { display:grid; grid-template-columns:1.1fr 1fr; gap:10px; }
@media (max-width:1100px){ .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.sample { display:grid; grid-template-columns:minmax(240px,380px) 1fr; gap:14px; align-items:start; }
@media (max-width:800px){ .sample { grid-template-columns:1fr; } }
.thumbs { display:flex; flex-wrap:wrap; gap:6px; }
.thumbs img { max-width:100%; max-height:240px; object-fit:contain; background:#0f172a; border-radius:8px; border:1px solid var(--border); }
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
.prompt, .resp, .chat { white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,Consolas,monospace; font-size:12.5px; background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; }
.resp { max-height:320px; overflow:auto; }
.nav { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
.bar { height:8px; background:#e5e7eb; border-radius:99px; overflow:hidden; min-width:80px; }
.bar > span { display:block; height:100%; background:var(--accent); }
.choices { font-size:13px; margin:4px 0 8px; }
</style>
</head>
<body>
<header>
  <h1>BLINK-Spatial 样本检查 · SPAR3 K=1 step 55</h1>
  <div class="sub">
    模型 <code>output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf</code> ·
    lmms-eval <code>blink_spatial</code> val · greedy 1024 · <code>disable_thinking</code> ·
    判分只读回复<strong>句首字母</strong>。下面每条都给出完整 user prompt（<code>input</code>）
    以及 chat 组装说明。请在本目录用本地 HTTP 打开，否则缩略图可能加载失败。
  </div>
  <div class="toolbar">
    <select id="status">
      <option value="wrong">仅错题</option>
      <option value="ok">仅对题</option>
      <option value="all">全部</option>
      <option value="lead_to">句首 To determine</option>
      <option value="lead_based">句首 Based</option>
      <option value="parsed_T">解析成 T</option>
      <option value="tail_would_match">末尾 A-F 能对上金标</option>
    </select>
    <select id="task"><option value="">全部 task</option></select>
    <input id="q" placeholder="搜索 id / prompt / 回复 / 金标" style="min-width:260px"/>
    <button class="primary" id="apply">筛选</button>
    <span id="count" class="meta"></span>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card" id="sum-overall"></div>
    <div class="card" id="sum-task"></div>
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
const PAGE = 4;
let filtered = [];
let page = 0;

function esc(s){
  return String(s??"").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function pill(r){
  return r.ok
    ? '<span class="pill ok">对</span>'
    : '<span class="pill bad">错</span>';
}
function renderSum(){
  const tasks = Object.entries(SUM.by_task).map(([k,v]) => {
    const w = Math.max(0, Math.min(100, v.acc));
    return `<tr><td>${esc(k)}</td><td>${v.acc}%</td><td>${v.ok}/${v.n}</td>
      <td><div class="bar"><span style="width:${w}%"></span></div></td></tr>`;
  }).join('');
  document.getElementById('sum-overall').innerHTML = `
    <h3 style="margin:0 0 8px">汇总</h3>
    <div>协议宏平均 <b>${SUM.macro}%</b> · 微平均 ${SUM.micro}%</div>
    <div class="meta">${SUM.n} 题 · 对 ${SUM.ok} · 错 ${SUM.n - SUM.ok}
      · 回答词数 p50=${SUM.words_p50} max=${SUM.words_max}</div>
    <div class="meta" style="margin-top:6px">
      句首 To determine ${SUM.flags.lead_to||0} · Based ${SUM.flags.lead_based||0} ·
      解析成 T ${SUM.flags.parsed_T||0} · 末尾字母本可对上 ${SUM.flags.tail_would_match||0}
    </div>`;
  document.getElementById('sum-task').innerHTML = `
    <h3 style="margin:0 0 8px">按 task</h3>
    <table><thead><tr><th>task</th><th>acc</th><th>ok/n</th><th></th></tr></thead><tbody>${tasks}</tbody></table>`;
  const sel = document.getElementById('task');
  [...new Set(DATA.map(r=>r.task))].sort().forEach(t => {
    const o = document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o);
  });
}
function thumbs(r){
  if(!r.imgs || !r.imgs.length) return '<div class="missing">无缩略图</div>';
  return '<div class="thumbs">'+r.imgs.map((src,i)=>
    `<img src="${esc(src)}" alt="${esc(r.id)} image ${i+1}" loading="lazy"
      onerror="this.outerHTML='<div class=&quot;missing&quot;>图 ${i+1} 未加载，请在 viewers 目录用 HTTP 打开</div>'"/>`
  ).join('')+'</div>';
}
function card(r){
  const choiceLine = (r.choices||[]).map((c,i)=>`${String.fromCharCode(65+i)}. ${esc(c)}`).join(' · ');
  const extraFlags = (r.flags||[]).filter(f=>f!=='ok' && f!=='wrong')
    .map(f=>'<span class="pill warn">'+esc(f)+'</span>').join('');
  return `<div class="card" style="margin-top:10px">
    <div class="sample">
      <div>${thumbs(r)}</div>
      <div>
        <div class="meta">${esc(r.id)} · ${esc(r.task)} · ${r.nimg||r.imgs.length} 图 · ${r.words} words</div>
        ${pill(r)} ${extraFlags}
        <div class="q">${esc(r.question)}</div>
        <div class="choices">${choiceLine}</div>
        <div class="meta">gold = <b>${esc(r.gold)}</b> · 句首解析 = <b>${esc(r.parsed)}</b> · 文末独立 A-F = <b>${esc(r.tail)}</b></div>
        <div class="label">发给模型的完整 user prompt（lmms-eval input）</div>
        <div class="prompt">${esc(r.input)}</div>
        <div class="label">chat 组装（qwen3_5：system + N 张图 + 上面这段文字）</div>
        <div class="chat">${esc(r.chat)}</div>
        <div class="label">数据集原始 prompt 字段</div>
        <div class="prompt">${esc(r.dataset_prompt)}</div>
        <div class="label">模型回复</div>
        <div class="resp">${esc(r.resp)}</div>
      </div>
    </div>
  </div>`;
}
function apply(){
  const st = document.getElementById('status').value;
  const task = document.getElementById('task').value;
  const q = document.getElementById('q').value.trim().toLowerCase();
  filtered = DATA.filter(r => {
    if(st==='wrong' && r.ok) return false;
    if(st==='ok' && !r.ok) return false;
    if(['lead_to','lead_based','parsed_T','tail_would_match'].includes(st) && !(r.flags||[]).includes(st)) return false;
    if(task && r.task!==task) return false;
    if(q){
      const blob = [r.id,r.question,r.input,r.dataset_prompt,r.resp,r.gold,r.parsed].join(' ').toLowerCase();
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
['status','task'].forEach(id=>document.getElementById(id).onchange=apply);
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter') apply(); });
renderSum();
apply();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", default=str(DEFAULT_DUMP))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    rows = load_rows(dump_dir)
    if not rows:
        raise SystemExit(f"no BLINK samples in {dump_dir}")

    official = None
    results_files = list(dump_dir.glob("*results.json"))
    if results_files:
        blob = json.loads(results_files[0].read_text())
        official = {
            "blink_spatial": round(100 * float(blob["results"]["blink_spatial"]["blink_acc,none"]), 2),
            "blink_multi_view_reasoning": round(
                100 * float(blob["results"]["blink_multi_view_reasoning"]["blink_acc,none"]), 2
            ),
            "blink_relative_depth": round(100 * float(blob["results"]["blink_relative_depth"]["blink_acc,none"]), 2),
            "blink_spatial_relation": round(
                100 * float(blob["results"]["blink_spatial_relation"]["blink_acc,none"]), 2
            ),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_images:
        image_index = load_image_index(Path(args.data_root))
        save_thumbs(rows, image_index, out_dir / "img", args.max_side, args.jpeg_quality)
    else:
        for row in rows:
            row["chat"] = row["chat"].replace("{nimg}", "?")

    summary = summarize(rows, official)
    html = HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace(
        "__SUM__", json.dumps(summary, ensure_ascii=False)
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'} ({len(html) / 1e6:.1f} MB, {len(rows)} rows)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
