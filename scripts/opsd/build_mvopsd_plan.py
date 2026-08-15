#!/usr/bin/env python3
"""Stage 1 of the MV-OPSD data pipeline: decide what every training sample looks like.

Draws each source at the SFT sampling rate, throws out the samples that cannot
carry a legal N -> K view gap, picks the student's view budget and view subset,
and rewrites the prompt text. No pixels are touched here; the output is a plan
that stage 2 (`materialize_mvopsd_views.py`) and stage 3
(`write_mvopsd_parquet.py`) consume.

    python scripts/opsd/build_mvopsd_plan.py --out data/mvopsd/plan

Every exclusion is counted and reported by (source, type, reason) so the
"why is your SPAR only 45% of SFT's" question has a table-shaped answer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402
from mvopsd.markers import MarkerError, marked_view_indices, parse_spar_info  # noqa: E402

# SPAR question types whose ground truth is defined in terms of the 32/3-view
# numbering itself, so any subsetting or renumbering changes the correct answer.
EXCLUDED_SPAR_TYPES = {
    "obj_frame_locate": "answer is a list of frame indices",
    "appearance_order": "answer cites absolute frame indices (\"spotted at 10\")",
    "distance_infer_center_oc_mv": "answer selects one of the input views",
    "view_change_infer": "question is about the relation between two given views",
    "camera_motion_infer": "question is about camera motion between two given views",
    "position_matching": "question asks to match a location across two given views",
}

ANSWER_FRAME_RE = re.compile(r"Frame-\d+")
ANSWER_ORDINAL_VIEW_RE = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+(image|frame|view|picture)\b", re.IGNORECASE
)

VIEW_SENSITIVE_TYPES = {"appearance_order"}


class Excluded(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def spar_scene_id(image_rel: str) -> tuple[str, str]:
    """`spar/scannet/images/scene0656_03/video_color/frame0_0.jpg` -> (scannet, scene0656_03)."""
    parts = image_rel.split("/")
    if len(parts) < 4 or parts[0] != "spar":
        return "unknown", "unknown"
    return parts[1], parts[3]


def check_answer_is_view_independent(answer: str) -> None:
    if ANSWER_FRAME_RE.search(answer):
        raise Excluded("answer_references_frame_index")
    if ANSWER_ORDINAL_VIEW_RE.search(answer):
        raise Excluded("answer_references_view_ordinal")


def build_spar_record(ann: dict, rng: random.Random, k_menu) -> dict:
    images = ann.get("images") or []
    n_views = len(images)
    if n_views < 2:
        raise Excluded("single_view_no_gap")

    info = parse_spar_info(ann)
    qtype = info.get("type") if info else None
    if qtype in EXCLUDED_SPAR_TYPES:
        raise Excluded(f"excluded_type:{qtype}")

    question, answer = ann["conversations"][0]["value"], ann["conversations"][1]["value"]
    check_answer_is_view_independent(answer)

    header, body = vw.split_image_header(question)
    style = "frame_labels" if vw.has_frame_labels(header) else "bare"

    marked = marked_view_indices(info, n_views) if info else set()
    referenced = vw.referenced_views(body)
    if any(view >= n_views for view in referenced):
        raise Excluded("frame_reference_out_of_range")

    required = set(marked) | referenced
    if vw.needs_anchor_view(body):
        required.add(0)

    view_indices = vw.choose_views(rng, n_views, required, k_menu)
    if view_indices is None:
        if len(required) >= n_views:
            raise Excluded("zero_privilege_all_views_required")
        raise Excluded("required_views_exceed_k_budget")

    renumber = {old: new for new, old in enumerate(view_indices)}
    student_body = vw.renumber_frame_refs(body, renumber) if referenced else body

    subset, scene = spar_scene_id(images[0])
    frames = []
    for index, rel in enumerate(images):
        if not rel.startswith("spar/"):
            raise Excluded("unexpected_image_path")
        is_marked = index in marked
        cache = f"marked/spar/{ann['id']}/v{index:02d}.jpg" if is_marked else f"spar/{rel[len('spar/'):]}"
        frames.append(
            {
                "src": src.SOURCES["spar_234k"].media_path(rel),
                "cache": cache,
                "marked": is_marked,
                "frame_index": None,
            }
        )

    return {
        "sample_id": f"spar{n_views}/{ann['id']}",
        "dataset": "spar_234k",
        "source": f"spar_{n_views}view",
        "subset": subset,
        "scene_id": scene,
        "question_type": qtype,
        "n_views": n_views,
        "k_views": len(view_indices),
        "view_indices": view_indices,
        "required_views": sorted(required),
        "marked_views": sorted(marked),
        "view_selection": "oracle_guided" if required else "pure_random",
        "privilege_bucket": "strong" if n_views >= 8 else "weak",
        "answer_view_sensitive": qtype in VIEW_SENSITIVE_TYPES,
        "header_style": style,
        "body": body,
        "student_body": student_body,
        "answer": answer,
        "spar_info": json.dumps(parse_spar_info(ann)) if marked else None,
        "frames": frames,
    }


def build_video_record(ann: dict, source: src.Source, tag: str, rng: random.Random, k_menu, frame_info) -> dict:
    total_frames, fps, frame_paths, video_path = frame_info
    indices = src.sft_frame_indices(total_frames, fps)
    n_views = len(indices)
    if n_views < 2:
        raise Excluded("too_few_frames")

    question, answer = ann["conversations"][0]["value"], ann["conversations"][1]["value"]
    check_answer_is_view_independent(answer)

    body = question.replace(vw.VIDEO_TOKEN, "").replace(vw.IMAGE_TOKEN, "")
    view_indices = vw.choose_views(rng, n_views, (), k_menu)
    if view_indices is None:
        raise Excluded("required_views_exceed_k_budget")

    video_id = ann["video"].rsplit("/", 1)[-1].removesuffix(".mp4")
    frames = []
    for frame_index in indices:
        if frame_paths is not None:
            frames.append(
                {
                    "src": frame_paths[frame_index],
                    "cache": f"video/{tag}/{video_id}/f{frame_index:06d}.jpg",
                    "marked": False,
                    "frame_index": None,
                }
            )
        else:
            frames.append(
                {
                    "src": video_path,
                    "cache": f"video/{tag}/{video_id}/f{frame_index:06d}.jpg",
                    "marked": False,
                    "frame_index": int(frame_index),
                }
            )

    return {
        "sample_id": f"{source.name}/{ann['id']}",
        "dataset": source.name,
        "source": source.name,
        "subset": tag,
        "scene_id": ann.get("scene_name", video_id),
        "question_type": ann.get("question_type"),
        "n_views": n_views,
        "k_views": len(view_indices),
        "view_indices": view_indices,
        "required_views": [],
        "marked_views": [],
        "view_selection": "pure_random",
        "privilege_bucket": "strong" if n_views >= 8 else "weak",
        "answer_view_sensitive": source.name == "vsi_appr_order",
        "header_style": "concat",
        "body": body,
        "student_body": body,
        "answer": answer,
        "spar_info": None,
        "frames": frames,
    }


def probe_media(annotations: list, source: src.Source, workers: int):
    """Resolve each referenced video: mp4 -> (frames, fps), frame directory -> file list."""
    unique = sorted({ann["video"] for ann in annotations})

    def probe(video_rel: str):
        path = source.media_path(video_rel)
        if os.path.isdir(path):
            frame_paths = src.list_frame_dir(path)
            if not frame_paths:
                return video_rel, None
            return video_rel, (len(frame_paths), 1.0, frame_paths, None)
        if not os.path.exists(path):
            return video_rel, None
        try:
            total, fps = src.probe_video(path)
        except Exception:
            return video_rel, None
        return video_rel, (total, fps, None, path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(probe, unique))
    return dict(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mvopsd/plan")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--k-menu", default="1,2,4")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap samples per source")
    args = parser.parse_args()

    k_menu = tuple(int(value) for value in args.k_menu.split(","))
    out_dir = os.path.join(src.REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    kept: list[dict] = []
    excluded: dict[str, Counter] = defaultdict(Counter)
    sampled_counts: dict[str, int] = {}
    sampled_ids: dict[str, list[str]] = {}

    for index, (name, source) in enumerate(src.SOURCES.items()):
        annotations = json.load(open(source.annotation_path()))
        drawn = src.sample_like_sft(annotations, source.sampling_rate, args.seed + index)
        if args.limit:
            drawn = drawn[: args.limit]
        sampled_counts[name] = len(drawn)
        sampled_ids[name] = [ann["id"] for ann in drawn]
        print(f"[{name}] {len(annotations)} -> {len(drawn)} sampled at rate {source.sampling_rate}", flush=True)

        if source.sft_expected and not args.limit and len(drawn) != source.sft_expected:
            raise SystemExit(
                f"{name}: drew {len(drawn)} samples but SFT trained on {source.sft_expected}; "
                "the sampling rate no longer matches the SFT run"
            )

        frame_info = {}
        tag = "llava_hound" if name == "llava_hound_64k" else "scannet"
        if name != "spar_234k":
            frame_info = probe_media(drawn, source, args.workers)
            print(f"[{name}] probed {len(frame_info)} media items", flush=True)

        rng = random.Random(args.seed + 1000 + index)
        for ann in drawn:
            try:
                if name == "spar_234k":
                    record = build_spar_record(ann, rng, k_menu)
                else:
                    info = frame_info.get(ann["video"])
                    if info is None:
                        raise Excluded("missing_media")
                    record = build_video_record(ann, source, tag, rng, k_menu, info)
                if record["scene_id"] in src.LEAKED_SCENES:
                    raise Excluded("vsibench_scene_leak")
            except Excluded as exc:
                excluded[name][exc.reason] += 1
                continue
            except MarkerError as exc:
                excluded[name][f"marker_error:{str(exc).split(':')[0]}"] += 1
                continue
            kept.append(record)

    random.Random(args.seed + 7).shuffle(kept)

    plan_path = os.path.join(out_dir, "plan.jsonl")
    with open(plan_path, "w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = summarize(kept, sampled_counts, excluded, k_menu, args.seed)
    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "sampled_ids.json"), "w") as handle:
        json.dump(sampled_ids, handle)

    print_report(stats)
    print(f"\nplan written to {plan_path} ({len(kept)} samples)")


def summarize(kept, sampled_counts, excluded, k_menu, seed) -> dict:
    by_source = Counter(record["source"] for record in kept)
    by_k = Counter(record["k_views"] for record in kept)
    by_selection = Counter(record["view_selection"] for record in kept)
    by_privilege = Counter(record["privilege_bucket"] for record in kept)
    by_type = Counter(f"{record['source']}/{record['question_type']}" for record in kept)
    unique_views = len({frame["cache"] for record in kept for frame in record["frames"]})
    total_view_refs = sum(len(record["frames"]) for record in kept)
    marked_views = sum(len(record["marked_views"]) for record in kept)
    return {
        "seed": seed,
        "k_menu": list(k_menu),
        "total_samples": len(kept),
        "sampled_after_sft_rate": sampled_counts,
        "kept_by_source": dict(by_source.most_common()),
        "kept_by_k": {str(key): value for key, value in sorted(by_k.items())},
        "kept_by_view_selection": dict(by_selection),
        "kept_by_privilege_bucket": dict(by_privilege),
        "kept_by_question_type": dict(by_type.most_common()),
        "excluded": {source: dict(counter.most_common()) for source, counter in excluded.items()},
        "views_expanded": total_view_refs,
        "views_unique_files": unique_views,
        "views_marked_per_sample_files": marked_views,
        "teacher_views_by_source": {
            source: dict(Counter(r["n_views"] for r in kept if r["source"] == source).most_common())
            for source in by_source
        },
    }


def print_report(stats: dict) -> None:
    print("\n=== kept ===")
    total = stats["total_samples"]
    for source, count in stats["kept_by_source"].items():
        print(f"  {source:<18} {count:>7}  {100 * count / max(total, 1):5.1f}%")
    print(f"  {'TOTAL':<18} {total:>7}")
    print("\n=== student budget K ===")
    for key, count in stats["kept_by_k"].items():
        print(f"  K={key:<3} {count:>7}  {100 * count / max(total, 1):5.1f}%")
    print("\n=== view selection ===")
    for key, count in stats["kept_by_view_selection"].items():
        print(f"  {key:<16} {count:>7}  {100 * count / max(total, 1):5.1f}%")
    print("\n=== excluded ===")
    for source, reasons in stats["excluded"].items():
        print(f"  [{source}]")
        for reason, count in reasons.items():
            print(f"    {reason:<44} {count:>7}")
    print("\n=== view cache ===")
    print(f"  expanded view references : {stats['views_expanded']:,}")
    print(f"  unique files to produce  : {stats['views_unique_files']:,}")


if __name__ == "__main__":
    main()
