#!/usr/bin/env python3
"""Side-by-side HTML viewer: SPAR-Bench greedy @2048, base vs SPAR3 step 55.

Reuses JPEG thumbs from the older SPAR-Bench viewer when present.

Usage::

    python3 scripts/opsd/tools/build_sparbench_tok2048_dual_viewer.py
    python3 -m http.server 8765
    # open experiments/viewers/sparbench_tok2048_base_vs_spar3/index.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/opsd/tools"))
from build_sparbench_sample_viewer import (  # noqa: E402
    MCA_SET,
    NA_SET,
    NUMBER_RE,
    family_of,
    kind_of,
    parsed_pred,
    score_of,
)

LETTER_RE = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")
CLIP = 3600

DEFAULT_BASE = (
    REPO
    / "logs/eval/20260902_qwen35base_sparbench_tok2048/sparbench/models__Qwen3.5-4B"
    / "20260902_155714_samples_sparbench.jsonl"
)
DEFAULT_SPAR3 = (
    REPO
    / "logs/eval/20260902_spar3_k1_step55_sparbench_tok2048/sparbench"
    / "output__20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf"
    / "20260902_155713_samples_sparbench.jsonl"
)
DEFAULT_OUT = REPO / "experiments/viewers/sparbench_tok2048_base_vs_spar3"
DEFAULT_THUMBS = REPO / "experiments/viewers/sparbench_spar3_k1_step55/img"
DEFAULT_PROCESSOR = REPO / "models/Qwen3.5-4B"
SYSTEM = "You are a helpful assistant."


class ChatRenderer:
    """Reproduce qwen3_5 eval chat text (images as vision placeholders)."""

    def __init__(self, model_dir: Path):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self._cache: dict[tuple[int, str], str] = {}

    def render(self, nimg: int, user_text: str) -> str:
        key = (max(nimg, 0), user_text)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        content = [{"type": "image"} for _ in range(max(nimg, 0))]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self._cache[key] = text
        return text


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def last_letter(text: str) -> str:
    line = last_line(text)
    if len(line) == 1 and line.isalpha():
        return line.upper()
    matches = list(LETTER_RE.finditer(line))
    return matches[-1].group(1).upper() if matches else ""


def clip_text(text: str, limit: int = CLIP) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head, tail = limit * 3 // 5, limit * 2 // 5
    return text[:head] + "\n\n… [clipped for viewer] …\n\n" + text[-tail:], True


def resp_text(rec: dict) -> str:
    doc = rec.get("sparbench_score") or {}
    if doc.get("prediction") is not None:
        return str(doc["prediction"])
    resp = rec.get("filtered_resps") or rec.get("resps") or [""]
    if isinstance(resp, list):
        if resp and isinstance(resp[0], list):
            return resp[0][0] if resp[0] else ""
        return resp[0] if resp else ""
    return str(resp)


def load_arm(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            doc = rec.get("sparbench_score") or rec["doc"]
            text = resp_text(rec)
            task = doc.get("task") or ""
            score, metric = score_of(doc)
            sid = int(doc["id"])
            shown, clipped = clip_text(text)
            out[sid] = {
                "score": round(float(score), 4),
                "metric": metric,
                "pred": parsed_pred(task, text),
                "resp": shown,
                "full_chars": len(text),
                "words": len(text.split()),
                "clipped": clipped,
                "last": last_line(text),
                "letter": last_letter(text),
                "nnum": len(NUMBER_RE.findall(text)),
                "prompt": rec.get("input") or "",
                "q": doc.get("question") or rec.get("input") or "",
                "gold": str(doc.get("answer") or rec.get("target") or ""),
                "task": task,
                "source": doc.get("source") or "",
                "img_type": doc.get("img_type") or "",
                "fmt": doc.get("format_type") or "",
            }
    return out


def bucket(score: float) -> str:
    if score >= 0.999:
        return "full"
    if score <= 0:
        return "zero"
    return "partial"


def join_rows(
    base: dict[int, dict],
    spar3: dict[int, dict],
    thumbs: Path,
    chat: ChatRenderer | None,
) -> list[dict]:
    rows = []
    for sid in sorted(set(base) & set(spar3)):
        b, s = base[sid], spar3[sid]
        task = s["task"]
        kind = kind_of(task)
        gold = s["gold"]
        flags = []
        if bucket(s["score"]) == "zero":
            flags.append("spar3_zero")
        if bucket(b["score"]) == "full":
            flags.append("base_full")
        if kind == "MCA" and s["score"] <= 0 and s["letter"] and s["letter"] == gold.strip().upper():
            flags.append("format_fail")
        if s["words"] >= 1400 or s["full_chars"] >= 7500:
            flags.append("likely_trunc")
        if s["clipped"]:
            flags.append("clipped")
        imgs = []
        if thumbs.is_dir():
            imgs = sorted(
                f"img/{p.name}"
                for p in thumbs.glob(f"{sid:05d}_*.jpg")
            )
        user = s["prompt"] or b["prompt"]
        nimg = len(imgs)
        if chat is not None:
            chat_text = chat.render(nimg, user)
        else:
            vision = "".join("[image]" for _ in range(nimg))
            chat_text = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{vision}\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        rows.append(
            {
                "id": sid,
                "task": task,
                "kind": kind,
                "family": family_of(task),
                "source": s["source"],
                "img_type": s["img_type"],
                "fmt": s["fmt"],
                "q": s["q"],
                "user": user,
                "chat": chat_text,
                "gold": gold,
                "imgs": imgs,
                "nimg": len(imgs),
                "flags": flags,
                "base": {
                    "score": b["score"],
                    "pred": b["pred"],
                    "resp": b["resp"],
                    "words": b["words"],
                    "chars": b["full_chars"],
                    "last": b["last"],
                    "letter": b["letter"],
                    "nnum": b["nnum"],
                    "clipped": b["clipped"],
                    "bucket": bucket(b["score"]),
                },
                "spar3": {
                    "score": s["score"],
                    "pred": s["pred"],
                    "resp": s["resp"],
                    "words": s["words"],
                    "chars": s["full_chars"],
                    "last": s["last"],
                    "letter": s["letter"],
                    "nnum": s["nnum"],
                    "clipped": s["clipped"],
                    "bucket": bucket(s["score"]),
                },
            }
        )
    return rows


def summarize(rows: list[dict], official_base: float | None, official_spar3: float | None) -> dict:
    flags = Counter()
    by_task_b: dict[str, list[float]] = defaultdict(list)
    by_task_s: dict[str, list[float]] = defaultdict(list)
    by_kind = defaultdict(lambda: {"n": 0, "base": 0.0, "spar3": 0.0, "format_fail": 0})
    for r in rows:
        by_task_b[r["task"]].append(r["base"]["score"])
        by_task_s[r["task"]].append(r["spar3"]["score"])
        k = by_kind[r["kind"]]
        k["n"] += 1
        k["base"] += r["base"]["score"]
        k["spar3"] += r["spar3"]["score"]
        if "format_fail" in r["flags"]:
            k["format_fail"] += 1
        for f in r["flags"]:
            flags[f] += 1

    def task_mean(d: dict[str, list[float]]) -> dict[str, float]:
        return {t: 100 * sum(v) / len(v) for t, v in d.items()}

    tb, ts = task_mean(by_task_b), task_mean(by_task_s)
    return {
        "n": len(rows),
        "official_base": official_base,
        "official_spar3": official_spar3,
        "format_fail": flags.get("format_fail", 0),
        "likely_trunc": flags.get("likely_trunc", 0),
        "by_kind": {
            k: {
                "n": v["n"],
                "base": round(100 * v["base"] / v["n"], 2),
                "spar3": round(100 * v["spar3"] / v["n"], 2),
                "format_fail": v["format_fail"],
            }
            for k, v in by_kind.items()
        },
        "by_task": {
            t: {
                "n": len(by_task_b[t]),
                "base": round(tb[t], 2),
                "spar3": round(ts[t], 2),
            }
            for t in sorted(tb, key=lambda x: ts[x] - tb[x])
        },
        "flags": dict(flags),
    }


def official_of(path: Path) -> float | None:
    results = path.with_name(path.name.replace("_samples_sparbench.jsonl", "_results.json"))
    if not results.exists():
        return None
    blob = json.loads(results.read_text())
    val = blob.get("results", {}).get("sparbench", {}).get("sparbench_score,none")
    return round(float(val), 4) if val is not None else None


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SPAR-Bench · 基座 vs SPAR3 · greedy 2048</title>
<style>
:root { --bg:#f4f6f8; --panel:#fff; --text:#1a1d21; --muted:#5c6570; --border:#d8dee6;
  --ok:#15803d; --bad:#b91c1c; --warn:#b45309; --accent:#2563eb; }
* { box-sizing: border-box; }
body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
header { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); padding:12px 18px; }
h1 { margin:0; font-size:16px; }
.sub { font-size:12px; color:var(--muted); margin-top:4px; line-height:1.55; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
select, input, button { border:1px solid var(--border); background:#fff; border-radius:6px; padding:6px 10px; font-size:13px; }
button { cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1560px; margin:0 auto; padding:16px 18px 48px; }
.grid { display:grid; grid-template-columns:1fr 1.1fr; gap:10px; }
@media (max-width:1100px){ .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.sample { display:grid; grid-template-columns:minmax(220px,340px) 1fr; gap:14px; align-items:start; }
@media (max-width:900px){ .sample { grid-template-columns:1fr; } }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
@media (max-width:900px){ .pair { grid-template-columns:1fr; } }
.thumbs { display:flex; flex-wrap:wrap; gap:6px; }
.thumbs img { max-width:100%; max-height:200px; object-fit:contain; background:#0f172a; border-radius:8px; border:1px solid var(--border); }
.missing { font-size:12px; color:var(--muted); padding:20px; text-align:center; background:#f8fafc; border-radius:8px; }
.label { font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); text-transform:uppercase; margin:8px 0 4px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--border); }
.pill { display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid var(--border); margin:0 4px 4px 0; }
.pill.ok { color:var(--ok); background:#f0fdf4; border-color:#86efac; }
.pill.bad { color:var(--bad); background:#fef2f2; border-color:#fecaca; }
.pill.warn { color:var(--warn); background:#fffbeb; border-color:#fde68a; }
.meta { font-size:12px; color:var(--muted); }
.q { font-weight:600; margin:6px 0; }
.chatbox, .userbox, .resp { white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,Consolas,monospace; font-size:12.5px; border:1px solid var(--border); border-radius:8px; padding:10px; overflow:auto; }
.chatbox { max-height:420px; background:#0f172a; color:#e2e8f0; }
.userbox { max-height:280px; background:#f8fafc; }
.resp { max-height:320px; background:#0f172a; color:#e2e8f0; }
.nav { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
.bar { height:8px; background:#e5e7eb; border-radius:99px; overflow:hidden; min-width:70px; display:inline-block; vertical-align:middle; }
.bar > span { display:block; height:100%; background:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>SPAR-Bench 回答对照 · 基座 vs SPAR3 K=1 step 55</h1>
  <div class="sub">
    greedy · <code>max_new_tokens=2048</code> · <code>until=[]</code> · HF / lmms-eval 官方判分
    （MCA 整段 exact match，NA 抽数字算 MRA，VCI 解析动作串）。
    官方 overall 基座 <b>__OFF_B__</b> / SPAR3 <b>__OFF_S__</b>。
    每条样本先展示 <b>完整 chat prompt</b>（system + 图像占位 + user 文本 + assistant 生成前缀，
    <code>enable_thinking=false</code>），再展示 lmms-eval 的 user 文本，最后才是两套回复。
    「格式失败」= MCA 官方零分，但 SPAR3 最后一行字母等于金标。
    长回复在页面里截到约 3600 字符（头+尾）；末行始终完整。
  </div>
  <div class="toolbar">
    <select id="view">
      <option value="format_fail">MCA 格式失败（末行对、官方零）</option>
      <option value="spar3_zero">SPAR3 官方零分</option>
      <option value="likely_trunc">SPAR3 可能截断</option>
      <option value="base_ok">基座满分</option>
      <option value="all">全部</option>
    </select>
    <select id="kind">
      <option value="">全部 kind</option>
      <option>MCA</option><option>NA</option><option>VCI</option>
    </select>
    <select id="family">
      <option value="">全部 family</option>
      <option>Low</option><option>Middle</option><option>High</option>
    </select>
    <select id="imgtype">
      <option value="">全部视角</option>
      <option value="single_view">single_view</option>
      <option value="multi_view">multi_view</option>
    </select>
    <select id="task"><option value="">全部 task</option></select>
    <input id="q" placeholder="搜索题面 / 回复 / 金标 / id" style="min-width:240px"/>
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
function pillScore(arm, name){
  const b = arm.bucket;
  const cls = b==='full'?'ok':(b==='zero'?'bad':'warn');
  const lab = b==='full'?'满分':(b==='zero'?'零分':'部分分');
  return `<span class="pill ${cls}">${esc(name)} ${lab} ${arm.score}</span>`;
}
function renderSum(){
  const kind = Object.entries(SUM.by_kind).map(([k,v])=>
    `<tr><td>${esc(k)}</td><td>${v.base}%</td><td>${v.spar3}%</td><td>${v.format_fail}</td><td>${v.n}</td></tr>`).join('');
  document.getElementById('sum-overall').innerHTML = `
    <h3 style="margin:0 0 8px">汇总</h3>
    <div>官方 overall 基座 <b>${SUM.official_base}</b> · SPAR3 <b>${SUM.official_spar3}</b></div>
    <div class="meta">${SUM.n} 题对齐 · MCA 格式失败 ${SUM.format_fail} · 可能截断 ${SUM.likely_trunc}</div>
    <table><thead><tr><th>kind</th><th>基座微平均</th><th>SPAR3 微平均</th><th>格式失败</th><th>n</th></tr></thead>
    <tbody>${kind}</tbody></table>
    <div class="meta" style="margin-top:6px">MCA 官方分接近 0 是因为整段匹配；末行字母对上仍记零分。</div>`;
  const tasks = Object.entries(SUM.by_task).map(([k,v]) => {
    const w = Math.max(0, Math.min(100, v.spar3));
    return `<tr><td>${esc(k)}</td><td>${v.base}%</td><td>${v.spar3}%</td><td>${v.n}</td>
      <td><div class="bar"><span style="width:${w}%"></span></div></td></tr>`;
  }).join('');
  document.getElementById('sum-task').innerHTML = `
    <h3 style="margin:0 0 8px">按 task（官方）</h3>
    <table><thead><tr><th>task</th><th>基座</th><th>SPAR3</th><th>n</th><th></th></tr></thead>
    <tbody>${tasks}</tbody></table>`;
  const sel = document.getElementById('task');
  [...new Set(DATA.map(r=>r.task))].sort().forEach(t => {
    const o = document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o);
  });
}
function thumbs(r){
  if(!r.imgs || !r.imgs.length) return '<div class="missing">无缩略图</div>';
  return '<div class="thumbs">'+r.imgs.map((src,i)=>
    `<img src="${esc(src)}" alt="id ${r.id} view ${i}" loading="lazy"
      onerror="this.outerHTML='<div class=&quot;missing&quot;>图 ${i} 未加载</div>'"/>`
  ).join('')+'</div>';
}
function armBox(arm, title){
  return `<div>
    <div class="label">${esc(title)}</div>
    ${pillScore(arm, title)}
    <div class="meta">解析 pred=<b>${esc(arm.pred)}</b> · 末行=<b>${esc(arm.last)}</b>
      · 末行字母=${esc(arm.letter||'—')} · ${arm.words} words · ${arm.chars} chars
      · 数字个数 ${arm.nnum}${arm.clipped?' · 页面已截断':''}</div>
    <div class="resp">${esc(arm.resp)}</div>
  </div>`;
}
function card(r){
  const extra = (r.flags||[]).filter(f=>!['spar3_zero','base_full'].includes(f))
    .map(f=>'<span class="pill warn">'+esc(f)+'</span>').join('');
  return `<div class="card" style="margin-top:10px">
    <div class="sample">
      <div>${thumbs(r)}</div>
      <div>
        <div class="meta">id ${r.id} · ${esc(r.task)} · ${esc(r.kind)}/${esc(r.family)}
          · ${esc(r.source)} · ${esc(r.img_type)} · ${esc(r.fmt)} · ${r.nimg} 图</div>
        ${pillScore(r.base,'基座')}${pillScore(r.spar3,'SPAR3')}${extra}
        <div class="meta" style="margin-top:6px">数据集 question 字段（不含 post prompt）</div>
        <div class="q">${esc(r.q)}</div>
        <div class="meta">gold = <b>${esc(r.gold)}</b></div>
      </div>
    </div>
    <div class="label">发给模型的完整 chat prompt（qwen3_5 apply_chat_template · disable_thinking）</div>
    <div class="chatbox">${esc(r.chat || '')}</div>
    <div class="label">lmms-eval user 文本（doc_to_text / samples.jsonl 的 input）</div>
    <div class="userbox">${esc(r.user || '')}</div>
    <div class="pair" style="margin-top:10px">
      ${armBox(r.base, '基座')}
      ${armBox(r.spar3, 'SPAR3')}
    </div>
  </div>`;
}
function apply(){
  const view = document.getElementById('view').value;
  const kind = document.getElementById('kind').value;
  const fam = document.getElementById('family').value;
  const img = document.getElementById('imgtype').value;
  const task = document.getElementById('task').value;
  const q = document.getElementById('q').value.trim().toLowerCase();
  filtered = DATA.filter(r => {
    if(view==='format_fail' && !(r.flags||[]).includes('format_fail')) return false;
    if(view==='spar3_zero' && r.spar3.bucket!=='zero') return false;
    if(view==='likely_trunc' && !(r.flags||[]).includes('likely_trunc')) return false;
    if(view==='base_ok' && r.base.bucket!=='full') return false;
    if(kind && r.kind!==kind) return false;
    if(fam && r.family!==fam) return false;
    if(img && r.img_type!==img) return false;
    if(task && r.task!==task) return false;
    if(q){
      const blob = [r.id, r.q, r.gold, r.user, r.chat, r.base.resp, r.spar3.resp, r.spar3.last].join(' ').toLowerCase();
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
['view','kind','family','imgtype','task'].forEach(id=>document.getElementById(id).onchange=apply);
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter') apply(); });
renderSum();
apply();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-samples", default=str(DEFAULT_BASE))
    parser.add_argument("--spar3-samples", default=str(DEFAULT_SPAR3))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--thumbs", default=str(DEFAULT_THUMBS))
    parser.add_argument("--processor", default=str(DEFAULT_PROCESSOR))
    args = parser.parse_args()

    base_path = Path(args.base_samples)
    spar3_path = Path(args.spar3_samples)
    if not base_path.exists():
        raise SystemExit(f"missing base samples: {base_path}")
    if not spar3_path.exists():
        raise SystemExit(f"missing SPAR3 samples: {spar3_path}")

    print("loading base…", file=sys.stderr)
    base = load_arm(base_path)
    print(f"  {len(base)}", file=sys.stderr)
    print("loading SPAR3…", file=sys.stderr)
    spar3 = load_arm(spar3_path)
    print(f"  {len(spar3)}", file=sys.stderr)

    thumbs = Path(args.thumbs)
    renderer = ChatRenderer(Path(args.processor)) if Path(args.processor).exists() else None
    if renderer is None:
        print("processor missing; using placeholder chat text", file=sys.stderr)
    rows = join_rows(base, spar3, thumbs, renderer)
    summary = summarize(rows, official_of(base_path), official_of(spar3_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / "img"
    if thumbs.is_dir() and not link.exists():
        link.symlink_to(thumbs.resolve(), target_is_directory=True)

    html = (
        HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False))
        .replace("__SUM__", json.dumps(summary, ensure_ascii=False))
        .replace("__OFF_B__", str(summary["official_base"]))
        .replace("__OFF_S__", str(summary["official_spar3"]))
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'index.html'} ({len(html) / 1e6:.1f} MB, {len(rows)} rows)")
    print(json.dumps({k: summary[k] for k in ("n", "official_base", "official_spar3", "format_fail", "likely_trunc", "by_kind")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
