#!/usr/bin/env python3
"""Build an HTML viewer for an lmms-eval SPAR-Bench sample dump.

Images are not stored in the jsonl; they are pulled from the local HF arrow
cache and written as JPEG thumbs next to the page (or via --img-dir).

Usage::

    python3 scripts/opsd/tools/build_sparbench_sample_viewer.py

    cd /path/to/SpatialStack_OPSD && python3 -m http.server 8765
    # then open experiments/viewers/sparbench_spar3_k1_step55/index.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[3]

MCA_TASKS = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]
NA_TASKS = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]
SPECIAL_TASKS = ["view_change_infer"]
LOW = set(NA_TASKS)
MIDDLE = {"view_change_infer", "position_matching", "camera_motion_infer"}
HIGH = set(MCA_TASKS) - MIDDLE
MCA_SET, NA_SET, SPECIAL_SET = set(MCA_TASKS), set(NA_TASKS), set(SPECIAL_TASKS)
NUMBER_RE = re.compile(r"(?<!\^)\d+\.\d+|(?<!\^)\d+")

DEFAULT_SAMPLES = (
    REPO
    / "logs/eval/20260831_mvopsd_spar3_k1_best_generalization/20260901/sparbench"
    / "output__20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf"
    / "20260901_115731_samples_sparbench.jsonl"
)
DEFAULT_RESULTS = DEFAULT_SAMPLES.with_name("20260901_115731_results.json")
DEFAULT_ARROW = (
    REPO
    / "sparbench_cache/jasonzhango___spar-bench/default/0.0.0"
    / "ee122877c25c8bb08539b07e06d872152c9968f1"
)
DEFAULT_OUT = REPO / "experiments/viewers/sparbench_spar3_k1_step55"


def family_of(task: str) -> str:
    if task in LOW:
        return "Low"
    if task in MIDDLE:
        return "Middle"
    if task in HIGH:
        return "High"
    return "Other"


def kind_of(task: str) -> str:
    if task in MCA_SET:
        return "MCA"
    if task in NA_SET:
        return "NA"
    if task in SPECIAL_SET:
        return "VCI"
    return "Other"


def score_of(doc: dict) -> tuple[float, str]:
    task = doc.get("task") or ""
    if task in MCA_SET:
        return float(doc.get("accuracy") or 0.0), "accuracy"
    if task in NA_SET:
        return float(doc.get("MRA:.5:.95:.05") or 0.0), "MRA"
    if task == "view_change_infer":
        return float(doc.get("vci_metric") or 0.0), "vci"
    return 0.0, "unknown"


def parsed_pred(task: str, text: str) -> str:
    if task in NA_SET:
        nums = NUMBER_RE.findall(text or "")
        if not nums:
            return ""
        if task.endswith("_mv"):
            return nums[-1]
        return nums[0]
    token = (text or "").split(" ")[0].rstrip(".").strip()
    return token


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            doc = rec.get("sparbench_score") or rec["doc"]
            resp = rec.get("filtered_resps") or rec.get("resps") or [""]
            if isinstance(resp, list):
                text = resp[0][0] if resp and isinstance(resp[0], list) else (resp[0] if resp else "")
            else:
                text = str(resp)
            task = doc.get("task") or ""
            score, metric = score_of(doc)
            flags = []
            if score >= 0.999:
                flags.append("full")
            elif score <= 0:
                flags.append("zero")
            else:
                flags.append("partial")
            parsed = parsed_pred(task, text)
            if task in NA_SET and not parsed:
                flags.append("no_number")
            if task in MCA_SET and not re.search(r"(?<![A-Za-z])[A-F](?![A-Za-z])", text or ""):
                flags.append("no_letter")
            sid = int(doc["id"])
            rows.append(
                {
                    "id": sid,
                    "task": task,
                    "kind": kind_of(task),
                    "family": family_of(task),
                    "source": doc.get("source") or "",
                    "img_type": doc.get("img_type") or "",
                    "fmt": doc.get("format_type") or "",
                    "q": doc.get("question") or rec.get("input") or "",
                    "gold": str(doc.get("answer") or rec.get("target") or ""),
                    "pred": parsed,
                    "score": round(score, 4),
                    "metric": metric,
                    "flags": flags,
                    "resp": text,
                    "words": len(str(text).split()),
                    "nimg": 0,
                    "imgs": [],
                }
            )
    return rows


def summarize(rows: list[dict], official: float | None) -> dict:
    by_task: dict[str, list[float]] = defaultdict(list)
    by_fam: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    flags = Counter()
    for r in rows:
        by_task[r["task"]].append(r["score"])
        by_fam[r["family"]].append(r["score"])
        by_kind[r["kind"]].append(r["score"])
        for f in r["flags"]:
            flags[f] += 1
    task_mean = {t: sum(v) / len(v) for t, v in by_task.items()}
    overall = (sum(task_mean.values()) / len(task_mean)) if task_mean else 0.0
    fam_task_means: dict[str, list[float]] = defaultdict(list)
    for t, m in task_mean.items():
        fam_task_means[family_of(t)].append(m)

    def pack(means: dict[str, float], counts: dict[str, list]) -> dict:
        return {
            k: {
                "n": len(counts[k]),
                "mean": round(100 * means[k], 2),
            }
            for k in sorted(means)
        }

    return {
        "n": len(rows),
        "official": official,
        "overall_unweighted_task_mean": round(100 * overall, 4),
        "micro": round(100 * (sum(r["score"] for r in rows) / len(rows) if rows else 0), 2),
        "full": flags.get("full", 0),
        "partial": flags.get("partial", 0),
        "zero": flags.get("zero", 0),
        "words_p50": sorted(r["words"] for r in rows)[len(rows) // 2] if rows else 0,
        "words_max": max((r["words"] for r in rows), default=0),
        "by_task": {
            t: {"n": len(by_task[t]), "mean": round(100 * task_mean[t], 2)}
            for t in sorted(task_mean, key=lambda x: task_mean[x])
        },
        "by_family": {
            fam: {
                "n": len(by_fam[fam]),
                "micro": round(100 * sum(by_fam[fam]) / len(by_fam[fam]), 2),
                "task_mean": round(100 * (sum(fam_task_means[fam]) / len(fam_task_means[fam])), 2)
                if fam_task_means[fam]
                else 0,
            }
            for fam in ("Low", "Middle", "High")
            if fam in by_fam
        },
        "by_kind": pack({k: sum(v) / len(v) for k, v in by_kind.items()}, by_kind),
        "flags": dict(flags),
    }


def load_sparbench_ds(arrow_dir: Path):
    from datasets import Dataset, concatenate_datasets

    files = sorted(arrow_dir.glob("spar-bench-test-*.arrow"))
    if not files:
        raise FileNotFoundError(f"no SPAR-Bench arrows in {arrow_dir}")
    return concatenate_datasets([Dataset.from_file(str(f)) for f in files])


def save_thumbs(ds, img_dir: Path, max_side: int, quality: int) -> dict[int, list[str]]:
    img_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, list[str]] = {}
    n = len(ds)
    nimg_col = ds.data.column("image")
    id_col = ds["id"]
    for i in range(n):
        sid = int(id_col[i])
        nimg = len(nimg_col[i])
        rels = [f"img/{sid:05d}_{k}.jpg" for k in range(nimg)]
        missing = [k for k in range(nimg) if not (img_dir / f"{sid:05d}_{k}.jpg").exists()]
        if missing:
            images = ds[i]["image"]
            if not isinstance(images, list):
                images = [images]
            for k in missing:
                im = images[k]
                if not isinstance(im, Image.Image):
                    im = Image.open(im).convert("RGB")
                else:
                    im = im.convert("RGB")
                w, h = im.size
                scale = max_side / max(w, h)
                if scale < 1:
                    im = im.resize(
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        Image.Resampling.BILINEAR,
                    )
                im.save(
                    img_dir / f"{sid:05d}_{k}.jpg",
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                )
        mapping[sid] = rels
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"thumbs {i + 1}/{n}", file=sys.stderr)
    return mapping


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SPAR-Bench · SPAR3 K=1 step 55</title>
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
.grid { display:grid; grid-template-columns:1.2fr 1fr; gap:10px; }
@media (max-width:1100px){ .grid { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.sample { display:grid; grid-template-columns:minmax(240px,380px) 1fr; gap:14px; align-items:start; }
@media (max-width:800px){ .sample { grid-template-columns:1fr; } }
.thumbs { display:flex; flex-wrap:wrap; gap:6px; }
.thumbs img { max-width:100%; max-height:220px; object-fit:contain; background:#0f172a; border-radius:8px; border:1px solid var(--border); }
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
.resp { white-space:pre-wrap; font-family:ui-monospace,monospace; font-size:12.5px; background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; max-height:280px; overflow:auto; }
.nav { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
.bar { height:8px; background:#e5e7eb; border-radius:99px; overflow:hidden; min-width:80px; }
.bar > span { display:block; height:100%; background:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>SPAR-Bench 样本检查 · SPAR3 K=1 step 55</h1>
  <div class="sub">
    模型 <code>output/20260831_mvopsd_spar3_k1_epoch1_nothink_global_step_55_hf</code> ·
    lmms-eval 默认协议：greedy · <code>max_new_tokens=100</code> · <code>until=\n\n</code> ·
    MCA 用整段回复做 exact match（长推理几乎全零）· NA 用回复里抽出的数字算 MRA。
    官方 overall 是各 task 均值再 ×100，不是微平均。
  </div>
  <div class="toolbar">
    <select id="status">
      <option value="zero">仅零分</option>
      <option value="partial">仅部分分</option>
      <option value="full">仅满分</option>
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
    <input id="q" placeholder="搜索题面 / 回答 / 金标" style="min-width:240px"/>
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
const PAGE = 8;
let filtered = [];
let page = 0;

function esc(s){
  return String(s??"").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function pill(r){
  if(r.score >= 0.999) return '<span class="pill ok">满分 '+r.score+'</span>';
  if(r.score <= 0) return '<span class="pill bad">零分</span>';
  return '<span class="pill warn">部分分 '+r.score+'</span>';
}
function renderSum(){
  const fam = Object.entries(SUM.by_family).map(([k,v])=>
    `<tr><td>${esc(k)}</td><td>${v.task_mean}%</td><td>${v.micro}%</td><td>${v.n}</td></tr>`).join('');
  const kind = Object.entries(SUM.by_kind).map(([k,v])=>
    `<tr><td>${esc(k)}</td><td>${v.mean}%</td><td>${v.n}</td></tr>`).join('');
  document.getElementById('sum-overall').innerHTML = `
    <h3 style="margin:0 0 8px">汇总</h3>
    <div>官方 overall <b>${SUM.official ?? SUM.overall_unweighted_task_mean}%</b>
      （task 等权均值 ${SUM.overall_unweighted_task_mean}%）· 微平均 ${SUM.micro}%</div>
    <div class="meta">${SUM.n} 题 · 满分 ${SUM.full} · 部分分 ${SUM.partial} · 零分 ${SUM.zero}
      · 回答词数 p50=${SUM.words_p50} max=${SUM.words_max}</div>
    <table><thead><tr><th>family</th><th>task 均值</th><th>微平均</th><th>n</th></tr></thead><tbody>${fam}</tbody></table>
    <table><thead><tr><th>kind</th><th>微平均</th><th>n</th></tr></thead><tbody>${kind}</tbody></table>
    <div class="meta" style="margin-top:6px">MCA 几乎全零是协议现象：模型写了约 80 词推理，判分要求整段等于金标字母/短语。</div>`;
  const tasks = Object.entries(SUM.by_task).map(([k,v]) => {
    const w = Math.max(0, Math.min(100, v.mean));
    return `<tr><td>${esc(k)}</td><td>${v.mean}%</td><td>${v.n}</td>
      <td><div class="bar"><span style="width:${w}%"></span></div></td></tr>`;
  }).join('');
  document.getElementById('sum-task').innerHTML = `
    <h3 style="margin:0 0 8px">按 task</h3>
    <table><thead><tr><th>task</th><th>mean</th><th>n</th><th></th></tr></thead><tbody>${tasks}</tbody></table>`;
  const sel = document.getElementById('task');
  const seen = new Set(DATA.map(r=>r.task));
  [...seen].sort().forEach(t => {
    const o = document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o);
  });
}
function thumbs(r){
  if(!r.imgs || !r.imgs.length) return '<div class="missing">无缩略图（构建时未导出图像）</div>';
  return '<div class="thumbs">'+r.imgs.map((src,i)=>
    `<img src="${esc(src)}" alt="id ${r.id} view ${i}" loading="lazy"
      onerror="this.outerHTML='<div class=&quot;missing&quot;>图 ${i} 未加载</div>'"/>`
  ).join('')+'</div>';
}
function card(r){
  return `<div class="card" style="margin-top:10px">
    <div class="sample">
      <div>${thumbs(r)}</div>
      <div>
        <div class="meta">id ${r.id} · ${esc(r.task)} · ${esc(r.kind)}/${esc(r.family)} · ${esc(r.source)}
          · ${esc(r.img_type)} · ${esc(r.fmt)} · ${r.nimg||r.imgs.length} 图 · ${r.words} words · ${esc(r.metric)}</div>
        ${pill(r)}
        ${(r.flags||[]).filter(f=>!['full','zero','partial'].includes(f)).map(f=>'<span class="pill warn">'+esc(f)+'</span>').join('')}
        <div class="q">${esc(r.q)}</div>
        <div class="meta">gold = <b>${esc(r.gold)}</b> · dump 解析 pred = <b>${esc(r.pred)}</b></div>
        <div class="label">模型回复</div>
        <div class="resp">${esc(r.resp)}</div>
      </div>
    </div>
  </div>`;
}
function apply(){
  const st = document.getElementById('status').value;
  const kind = document.getElementById('kind').value;
  const fam = document.getElementById('family').value;
  const img = document.getElementById('imgtype').value;
  const task = document.getElementById('task').value;
  const q = document.getElementById('q').value.trim().toLowerCase();
  filtered = DATA.filter(r => {
    if(st==='zero' && r.score>0) return false;
    if(st==='full' && r.score<0.999) return false;
    if(st==='partial' && (r.score<=0 || r.score>=0.999)) return false;
    if(kind && r.kind!==kind) return false;
    if(fam && r.family!==fam) return false;
    if(img && r.img_type!==img) return false;
    if(task && r.task!==task) return false;
    if(q && !(String(r.q).toLowerCase().includes(q) || String(r.resp).toLowerCase().includes(q) || String(r.gold).toLowerCase().includes(q))) return false;
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
  const slice = filtered.slice(page*PAGE, page*PAGE+PAGE);
  document.getElementById('list').innerHTML = slice.map(card).join('');
}
document.getElementById('apply').onclick = apply;
document.getElementById('prev').onclick = ()=>{ page=Math.max(0,page-1); draw(); };
document.getElementById('next').onclick = ()=>{ page+=1; draw(); };
['status','kind','family','imgtype','task'].forEach(id=>document.getElementById(id).onchange=apply);
document.getElementById('q').addEventListener('keydown', e=>{ if(e.key==='Enter') apply(); });
renderSum();
apply();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--arrow-dir", default=str(DEFAULT_ARROW))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--img-dir", default="")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    args = parser.parse_args()

    samples = Path(args.samples)
    if not samples.exists():
        raise SystemExit(f"missing samples: {samples}")
    rows = load_rows(samples)
    official = None
    results_path = Path(args.results)
    if results_path.exists():
        blob = json.loads(results_path.read_text())
        official = blob.get("results", {}).get("sparbench", {}).get("sparbench_score,none")
        if official is not None:
            official = round(float(official), 4)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir) if args.img_dir else out_dir / "img"
    link = out_dir / "img"
    if args.img_dir:
        img_dir.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(img_dir.resolve(), target_is_directory=True)

    if not args.skip_images:
        ds = load_sparbench_ds(Path(args.arrow_dir))
        mapping = save_thumbs(ds, img_dir if not args.img_dir else img_dir, args.max_side, args.jpeg_quality)
        for r in rows:
            rels = mapping.get(r["id"], [])
            r["imgs"] = rels
            r["nimg"] = len(rels)

    summary = summarize(rows, official)
    html = HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace(
        "__SUM__", json.dumps(summary, ensure_ascii=False)
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'} ({len(html)/1e6:.1f} MB, {len(rows)} rows)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
