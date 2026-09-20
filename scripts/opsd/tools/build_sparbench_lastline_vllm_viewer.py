#!/usr/bin/env python3
"""HTML viewer: SPAR-Bench vLLM lastline, base vs SPAR3 (full + middle body).

    python3 scripts/opsd/tools/build_sparbench_lastline_vllm_viewer.py
    python3 -m http.server 8765
    # open experiments/viewers/sparbench_lastline_vllm_base_vs_spar3/index.html
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
from build_sparbench_sample_viewer import family_of, kind_of  # noqa: E402

LETTER_RE = re.compile(r"(?<![A-Za-z])([A-F])(?![A-Za-z])")
CLIP = 8000
DEFAULT_BASE = REPO / "logs/eval/20260906_sparbench_vllm_lastline/base/samples.jsonl"
DEFAULT_SPAR3 = REPO / "logs/eval/20260906_sparbench_vllm_lastline/spar3_step55/samples.jsonl"
DEFAULT_OUT = REPO / "experiments/viewers/sparbench_lastline_vllm_base_vs_spar3"
DEFAULT_THUMBS = REPO / "experiments/viewers/sparbench_spar3_k1_step55/img"
DEFAULT_PROCESSOR = REPO / "models/Qwen3.5-4B"
SYSTEM = "You are a helpful assistant."


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def body_text(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if len(lines) == 1:
        return ""
    return "\n".join(lines[:-1]).strip()


def last_letter(text: str) -> str:
    line = last_line(text)
    if len(line) == 1 and line.isalpha():
        return line.upper()
    matches = list(LETTER_RE.finditer(line))
    return matches[-1].group(1).upper() if matches else ""


def clip_text(text: str, limit: int = CLIP) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head, tail = limit * 55 // 100, limit * 35 // 100
    mid = limit - head - tail - 40
    body = body_text(text)
    last = last_line(text)
    if body and len(body) > head + mid:
        mid_chunk = body[len(body) // 2 - mid // 2 : len(body) // 2 + mid // 2]
        shown = (
            body[:head]
            + f"\n\n… [middle clipped; showing ~{mid} chars from center of body] …\n\n"
            + mid_chunk
            + f"\n\n… [end of body] …\n\n"
            + last
        )
        return shown, True
    return text[:head] + "\n\n… [clipped] …\n\n" + text[-tail:], True


def bucket(score: float) -> str:
    if score >= 0.999:
        return "full"
    if score <= 0:
        return "zero"
    return "partial"


def load_arm(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            sid = int(rec["id"])
            resp = rec.get("response") or ""
            shown, clipped = clip_text(resp)
            body = body_text(resp)
            body_shown, body_clipped = clip_text(body, limit=6000) if body else ("", False)
            out[sid] = {
                "score": round(float(rec.get("score") or 0), 4),
                "parsed": rec.get("parsed") or "",
                "gold": str(rec.get("answer") or ""),
                "resp": shown,
                "body": body_shown,
                "last": last_line(resp),
                "letter": last_letter(resp),
                "words": len(resp.split()),
                "chars": len(resp),
                "body_chars": len(body),
                "truncated": bool(rec.get("truncated")),
                "finish": rec.get("finish_reason") or "",
                "out_tok": rec.get("output_tokens") or 0,
                "clipped": clipped,
                "body_clipped": body_clipped,
                "prompt": rec.get("prompt") or "",
                "q": rec.get("question") or "",
                "task": rec.get("task") or "",
                "nimg": int(rec.get("nimg") or 0),
                "bucket": bucket(float(rec.get("score") or 0)),
            }
    return out


class ChatRenderer:
    def __init__(self, model_dir: Path):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self._cache: dict[tuple[int, str], str] = {}

    def render(self, nimg: int, user_text: str) -> str:
        key = (max(nimg, 0), user_text)
        if key in self._cache:
            return self._cache[key]
        content = [{"type": "image"} for _ in range(max(nimg, 0))]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        self._cache[key] = text
        return text


def join_rows(base: dict, spar3: dict, thumbs: Path, chat: ChatRenderer | None) -> list[dict]:
    rows = []
    for sid in sorted(set(base) & set(spar3)):
        b, s = base[sid], spar3[sid]
        task = s["task"]
        kind = kind_of(task)
        gold = s["gold"]
        flags = []
        if s["bucket"] == "zero" and b["bucket"] != "zero":
            flags.append("spar3_only_zero")
        if b["bucket"] == "zero" and s["bucket"] != "zero":
            flags.append("base_only_zero")
        if s["score"] > b["score"] + 0.01:
            flags.append("spar3_higher")
        elif b["score"] > s["score"] + 0.01:
            flags.append("base_higher")
        if kind == "MCA" and s["bucket"] == "zero" and s["letter"] == gold.strip().upper():
            flags.append("spar3_lastline_ok_zero")
        if s["body_chars"] >= 200:
            flags.append("spar3_has_body")
        if b["body_chars"] >= 200:
            flags.append("base_has_body")
        if s["truncated"]:
            flags.append("spar3_trunc")
        if b["truncated"]:
            flags.append("base_trunc")
        imgs = []
        if thumbs.is_dir():
            imgs = sorted(f"img/{p.name}" for p in thumbs.glob(f"{sid:05d}_*.jpg"))
        user = s["prompt"] or b["prompt"]
        chat_text = chat.render(len(imgs), user) if chat else ""
        rows.append(
            {
                "id": sid,
                "task": task,
                "kind": kind,
                "family": family_of(task),
                "gold": gold,
                "user": user,
                "chat": chat_text,
                "imgs": imgs,
                "nimg": len(imgs),
                "flags": flags,
                "base": b,
                "spar3": s,
            }
        )
    return rows


def summarize(rows: list[dict], off_b: float, off_s: float) -> dict:
    flags = Counter()
    for r in rows:
        for f in r["flags"]:
            flags[f] += 1
    by_kind = defaultdict(lambda: {"n": 0, "base": 0.0, "spar3": 0.0})
    for r in rows:
        k = by_kind[r["kind"]]
        k["n"] += 1
        k["base"] += r["base"]["score"]
        k["spar3"] += r["spar3"]["score"]
    return {
        "n": len(rows),
        "official_base": off_b,
        "official_spar3": off_s,
        "spar3_has_body": flags.get("spar3_has_body", 0),
        "base_has_body": flags.get("base_has_body", 0),
        "spar3_lastline_ok_zero": flags.get("spar3_lastline_ok_zero", 0),
        "by_kind": {
            k: {
                "n": v["n"],
                "base": round(100 * v["base"] / v["n"], 2),
                "spar3": round(100 * v["spar3"] / v["n"], 2),
            }
            for k, v in by_kind.items()
        },
        "flags": dict(flags),
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SPAR-Bench · lastline vLLM · 基座 vs SPAR3</title>
<style>
:root { --bg:#f4f6f8; --panel:#fff; --text:#1a1d21; --muted:#5c6570; --border:#d8dee6;
  --ok:#15803d; --bad:#b91c1c; --warn:#b45309; --accent:#2563eb; --mid:#7c3aed; }
* { box-sizing:border-box; }
body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
header { position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); padding:12px 18px; }
h1 { margin:0; font-size:16px; }
.sub { font-size:12px; color:var(--muted); margin-top:4px; line-height:1.55; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
select, input, button { border:1px solid var(--border); background:#fff; border-radius:6px; padding:6px 10px; font-size:13px; }
button { cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1600px; margin:0 auto; padding:16px 18px 48px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin-top:10px; }
.sample { display:grid; grid-template-columns:minmax(220px,320px) 1fr; gap:14px; }
@media (max-width:900px){ .sample { grid-template-columns:1fr; } }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
@media (max-width:900px){ .pair { grid-template-columns:1fr; } }
.thumbs { display:flex; flex-wrap:wrap; gap:6px; }
.thumbs img { max-height:180px; max-width:100%; object-fit:contain; background:#0f172a; border-radius:8px; }
.missing { font-size:12px; color:var(--muted); padding:16px; text-align:center; background:#f8fafc; border-radius:8px; }
.label { font-size:11px; font-weight:700; letter-spacing:.04em; color:var(--muted); text-transform:uppercase; margin:8px 0 4px; }
.pill { display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px; border:1px solid var(--border); margin:0 4px 4px 0; }
.pill.ok { color:var(--ok); background:#f0fdf4; }
.pill.bad { color:var(--bad); background:#fef2f2; }
.pill.warn { color:var(--warn); background:#fffbeb; }
.pill.mid { color:var(--mid); background:#f5f3ff; border-color:#ddd6fe; }
.meta { font-size:12px; color:var(--muted); }
.q { font-weight:600; margin:6px 0; }
.chatbox, .userbox, .resp, .bodybox, .lastbox {
  white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,Consolas,monospace; font-size:12.5px;
  border:1px solid var(--border); border-radius:8px; padding:10px; overflow:auto; }
.chatbox { max-height:360px; background:#0f172a; color:#e2e8f0; }
.userbox { max-height:200px; background:#f8fafc; }
.bodybox { max-height:360px; background:#faf5ff; color:#1e1b4b; }
.lastbox { max-height:80px; background:#ecfdf5; color:#064e3b; font-weight:600; }
.resp { max-height:280px; background:#0f172a; color:#e2e8f0; }
.nav { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--border); }
</style>
</head>
<body>
<header>
  <h1>SPAR-Bench lastline vLLM · 基座 vs SPAR3 · 回复检视</h1>
  <div class="sub">
    greedy @2048 · 末行解析 · 无 <code>\boxed{}</code> · overall 基座 <b>__OFF_B__</b> / SPAR3 <b>__OFF_S__</b>。
    重点展示<strong>中间推理段</strong>（body，末行之上）与<strong>末行答案</strong>分列。
    长文页面截断时保留 body 中段 + 完整末行。
  </div>
  <div class="toolbar">
    <select id="view">
      <option value="spar3_has_body">SPAR3 有长推理（body≥200字）</option>
      <option value="both_body">双方都有长推理</option>
      <option value="spar3_higher">SPAR3 分更高</option>
      <option value="base_higher">基座分更高</option>
      <option value="disagree">双方 bucket 不同</option>
      <option value="spar3_trunc">SPAR3 截断</option>
      <option value="all">全部</option>
    </select>
    <select id="kind"><option value="">全部 kind</option><option>MCA</option><option>NA</option><option>VCI</option></select>
    <select id="family"><option value="">全部 family</option><option>Low</option><option>Middle</option><option>High</option></select>
    <select id="task"><option value="">全部 task</option></select>
    <input id="q" placeholder="搜索 id / 题面 / 回复 / 金标" style="min-width:260px"/>
    <button class="primary" id="apply">筛选</button>
    <span id="count" class="meta"></span>
  </div>
</header>
<main>
  <div class="card" id="sum"></div>
  <div class="nav"><button id="prev">上一页</button><span id="pageinfo"></span><button id="next">下一页</button></div>
  <div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const SUM = __SUM__;
const PAGE = 3;
let filtered = [], page = 0;
function esc(s){ return String(s??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function pill(arm,name){
  const c=arm.bucket==='full'?'ok':(arm.bucket==='zero'?'bad':'warn');
  const lab=arm.bucket==='full'?'满分':(arm.bucket==='zero'?'零分':'部分');
  return `<span class="pill ${c}">${esc(name)} ${lab} ${arm.score}</span>`;
}
function renderSum(){
  const kind=Object.entries(SUM.by_kind).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v.base}%</td><td>${v.spar3}%</td><td>${v.n}</td></tr>`).join('');
  document.getElementById('sum').innerHTML=`
    <h3 style="margin:0 0 8px">汇总</h3>
    <div>overall 基座 <b>${SUM.official_base}</b> · SPAR3 <b>${SUM.official_spar3}</b></div>
    <div class="meta">${SUM.n} 题 · SPAR3 长 body ${SUM.spar3_has_body} · 基座长 body ${SUM.base_has_body}</div>
    <table><thead><tr><th>kind</th><th>基座</th><th>SPAR3</th><th>n</th></tr></thead><tbody>${kind}</tbody></table>`;
  const sel=document.getElementById('task');
  [...new Set(DATA.map(r=>r.task))].sort().forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;sel.appendChild(o);});
}
function thumbs(r){
  if(!r.imgs.length) return '<div class="missing">无缩略图</div>';
  return '<div class="thumbs">'+r.imgs.map((s,i)=>`<img src="${esc(s)}" loading="lazy"/>`).join('')+'</div>';
}
function armPanel(arm, title){
  const extra = [];
  if(arm.truncated) extra.push('<span class="pill bad">截断</span>');
  if(arm.body_clipped||arm.clipped) extra.push('<span class="pill warn">页面截断</span>');
  if(arm.body_chars>=200) extra.push('<span class="pill mid">长推理</span>');
  return `<div>
    <div class="label">${esc(title)}</div>
    ${pill(arm,title)} ${extra.join('')}
    <div class="meta">parsed=<b>${esc(arm.parsed)}</b> · 末行=<b>${esc(arm.last)}</b> · letter=${esc(arm.letter||'—')}
      · body ${arm.body_chars} chars · total ${arm.chars} chars · ${arm.out_tok} tok</div>
    <div class="label">中间推理（body，末行之上）</div>
    <div class="bodybox">${arm.body ? esc(arm.body) : '<span class="meta">（无中间段，仅末行或空）</span>'}</div>
    <div class="label">末行（判分读取位置）</div>
    <div class="lastbox">${esc(arm.last)||'—'}</div>
    <div class="label">完整回复（可能截断）</div>
    <div class="resp">${esc(arm.resp)}</div>
  </div>`;
}
function card(r){
  const flags=(r.flags||[]).map(f=>'<span class="pill warn">'+esc(f)+'</span>').join('');
  return `<div class="card">
    <div class="sample">
      <div>${thumbs(r)}</div>
      <div>
        <div class="meta">id ${r.id} · ${esc(r.task)} · ${esc(r.kind)}/${esc(r.family)} · gold <b>${esc(r.gold)}</b></div>
        ${flags}
        <div class="q">${esc(r.q)}</div>
      </div>
    </div>
    <div class="label">chat prompt（vLLM）</div>
    <div class="chatbox">${esc(r.chat)}</div>
    <div class="label">user 文本（含 lastline 后缀）</div>
    <div class="userbox">${esc(r.user)}</div>
    <div class="pair">${armPanel(r.base,'基座')}${armPanel(r.spar3,'SPAR3')}</div>
  </div>`;
}
function apply(){
  const view=document.getElementById('view').value;
  const kind=document.getElementById('kind').value;
  const fam=document.getElementById('family').value;
  const task=document.getElementById('task').value;
  const q=document.getElementById('q').value.trim().toLowerCase();
  filtered=DATA.filter(r=>{
    if(view==='spar3_has_body' && !(r.flags||[]).includes('spar3_has_body')) return false;
    if(view==='both_body' && (!r.flags.includes('spar3_has_body')||!r.flags.includes('base_has_body'))) return false;
    if(view==='spar3_higher' && !(r.flags||[]).includes('spar3_higher')) return false;
    if(view==='base_higher' && !(r.flags||[]).includes('base_higher')) return false;
    if(view==='disagree' && r.base.bucket===r.spar3.bucket) return false;
    if(view==='spar3_trunc' && !r.spar3.truncated) return false;
    if(kind && r.kind!==kind) return false;
    if(fam && r.family!==fam) return false;
    if(task && r.task!==task) return false;
    if(q){
      const blob=[r.id,r.q,r.gold,r.user,r.base.body,r.base.resp,r.spar3.body,r.spar3.resp].join(' ').toLowerCase();
      if(!blob.includes(q)) return false;
    }
    return true;
  });
  page=0; draw();
}
function draw(){
  const n=filtered.length, pages=Math.max(1,Math.ceil(n/PAGE));
  page=Math.min(page,pages-1);
  document.getElementById('count').textContent=n+' 条';
  document.getElementById('pageinfo').textContent=(page+1)+' / '+pages;
  document.getElementById('list').innerHTML=filtered.slice(page*PAGE,page*PAGE+PAGE).map(card).join('');
}
document.getElementById('apply').onclick=apply;
document.getElementById('prev').onclick=()=>{page=Math.max(0,page-1);draw();};
document.getElementById('next').onclick=()=>{page++;draw();};
['view','kind','family','task'].forEach(id=>document.getElementById(id).onchange=apply);
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')apply();});
renderSum(); apply();
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
        raise SystemExit(f"missing base samples: {base_path} (rsync from node-B)")
    if not spar3_path.exists():
        raise SystemExit(f"missing SPAR3 samples: {spar3_path}")

    off_b = off_s = None
    for p, attr in ((base_path.parent / "summary.json", "off_b"), (spar3_path.parent / "summary.json", "off_s")):
        if p.exists():
            blob = json.loads(p.read_text())
            if attr == "off_b":
                off_b = blob.get("overall")
            else:
                off_s = blob.get("overall")

    chat = ChatRenderer(Path(args.processor)) if Path(args.processor).exists() else None
    base = load_arm(base_path)
    spar3 = load_arm(spar3_path)
    rows = join_rows(base, spar3, Path(args.thumbs), chat)
    summary = summarize(rows, off_b, off_s)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "img"
    thumbs = Path(args.thumbs)
    if thumbs.is_dir() and not link.exists():
        link.symlink_to(thumbs.resolve(), target_is_directory=True)

    html = (
        HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False))
        .replace("__SUM__", json.dumps(summary, ensure_ascii=False))
        .replace("__OFF_B__", str(off_b))
        .replace("__OFF_S__", str(off_s))
    )
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out / 'index.html'} ({len(html)/1e6:.1f} MB, {len(rows)} rows)")


if __name__ == "__main__":
    main()
