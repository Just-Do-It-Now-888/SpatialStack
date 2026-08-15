#!/usr/bin/env python3
"""Stage 2 of the MV-OPSD data pipeline: write every view the plan references.

Three kinds of work, all ending in the same place -- a JPEG whose width and
height are multiples of 32, so verl's student path (``smart_resize``) and its
teacher path (bare ``Image.open``) produce identical pixels:

1. plain images (SPAR frames, pre-extracted llava_hound frames) -> resize;
2. mp4 frames (ScanNet) -> decode the SFT-selected indices, then resize;
3. marked SPAR views -> draw the markers at native resolution first, exactly as
   the SFT dataloader does, then resize. These are sample-specific and cannot be
   shared between samples.

    python scripts/opsd/materialize_mvopsd_views.py --plan data/mvopsd/plan

Re-running skips views that already exist, so the job is restartable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import sources as src  # noqa: E402
from mvopsd.geometry import resize_to_sft_geometry  # noqa: E402
from mvopsd.markers import LazyImageList, MarkerError, draw_markers  # noqa: E402

JPEG_QUALITY = 95
CACHE_ROOT = "data/mvopsd/views"


def _save(image: Image.Image, path: str) -> tuple[int, int]:
    resized = resize_to_sft_geometry(image)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    resized.save(tmp, format="JPEG", quality=JPEG_QUALITY)
    os.replace(tmp, path)
    return resized.size


def run_image_job(job) -> tuple[str, int, str]:
    source_path, cache_path = job
    try:
        with Image.open(source_path) as image:
            size = _save(image, cache_path)
        return "ok", 1, f"{size[0]}x{size[1]}"
    except Exception as exc:
        return "error", 1, f"{source_path}: {type(exc).__name__}: {exc}"


def run_video_job(job) -> tuple[str, int, str]:
    video_path, wanted = job  # wanted: {frame_index: cache_path}
    import av

    try:
        remaining = dict(wanted)
        with av.open(video_path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                cache_path = remaining.pop(index, None)
                if cache_path is not None:
                    _save(frame.to_image(), cache_path)
                if not remaining:
                    break
        if remaining:
            return "error", len(wanted), f"{video_path}: missing frames {sorted(remaining)}"
        return "ok", len(wanted), ""
    except Exception as exc:
        return "error", len(wanted), f"{video_path}: {type(exc).__name__}: {exc}"


def run_marker_job(job) -> tuple[str, int, str]:
    sample_id, info_json, source_paths, targets = job  # targets: {view_index: cache_path}
    try:
        info = json.loads(info_json)
        images = LazyImageList(source_paths, lambda path: Image.open(path).convert("RGB"))
        touched = draw_markers(images, info)
        expected = {int(key) for key in targets}
        if touched != expected:
            raise MarkerError(f"marker views changed between planning ({sorted(expected)}) and drawing ({sorted(touched)})")
        for view_index, cache_path in targets.items():
            _save(images[int(view_index)], cache_path)
        return "ok", len(targets), ""
    except Exception as exc:
        return "error", len(targets), f"{sample_id}: {type(exc).__name__}: {exc}"


def collect_jobs(records, cache_root: str, overwrite: bool):
    image_jobs: dict[str, str] = {}
    video_jobs: dict[str, dict[int, str]] = defaultdict(dict)
    marker_jobs: list = []

    for record in records:
        marker_targets: dict[int, str] = {}
        for view_index, frame in enumerate(record["frames"]):
            cache_path = os.path.join(cache_root, frame["cache"])
            exists = os.path.exists(cache_path)
            if frame["marked"]:
                if overwrite or not exists:
                    marker_targets[view_index] = cache_path
                continue
            if exists and not overwrite:
                continue
            if frame["frame_index"] is None:
                image_jobs[cache_path] = frame["src"]
            else:
                video_jobs[frame["src"]][int(frame["frame_index"])] = cache_path
        if marker_targets:
            marker_jobs.append(
                (
                    record["sample_id"],
                    record["spar_info"],
                    [frame["src"] for frame in record["frames"]],
                    marker_targets,
                )
            )

    return (
        [(source, cache) for cache, source in image_jobs.items()],
        [(video, wanted) for video, wanted in video_jobs.items()],
        marker_jobs,
    )


def execute(runner, jobs, workers: int, label: str, errors: list) -> int:
    if not jobs:
        print(f"[{label}] nothing to do")
        return 0
    done_views = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(runner, job) for job in jobs]
        for finished, future in enumerate(as_completed(futures), start=1):
            status, count, detail = future.result()
            if status == "error":
                errors.append(f"[{label}] {detail}")
            else:
                done_views += count
            if finished % max(1, len(jobs) // 20) == 0 or finished == len(jobs):
                rate = finished / max(time.time() - started, 1e-6)
                print(
                    f"[{label}] {finished}/{len(jobs)} jobs, {done_views} views, "
                    f"{rate:.1f} jobs/s, {len(errors)} errors",
                    flush=True,
                )
    return done_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/mvopsd/plan")
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug: only process the first N plan records")
    args = parser.parse_args()

    plan_path = os.path.join(src.REPO_ROOT, args.plan, "plan.jsonl")
    cache_root = os.path.join(src.REPO_ROOT, args.cache_root)
    records = [json.loads(line) for line in open(plan_path)]
    if args.limit:
        records = records[: args.limit]
    print(f"plan: {len(records)} samples from {plan_path}")

    image_jobs, video_jobs, marker_jobs = collect_jobs(records, cache_root, args.overwrite)
    print(
        f"pending: {len(image_jobs)} images, {len(video_jobs)} videos "
        f"({sum(len(w) for _, w in video_jobs)} frames), {len(marker_jobs)} marker samples "
        f"({sum(len(job[3]) for job in marker_jobs)} views)"
    )

    errors: list[str] = []
    total = 0
    total += execute(run_video_job, video_jobs, args.workers, "video", errors)
    total += execute(run_marker_job, marker_jobs, args.workers, "marker", errors)
    total += execute(run_image_job, image_jobs, args.workers, "image", errors)

    print(f"\nwrote {total} views, {len(errors)} errors")
    if errors:
        error_path = os.path.join(src.REPO_ROOT, args.plan, "materialize_errors.log")
        with open(error_path, "w") as handle:
            handle.write("\n".join(errors))
        print(f"first errors:\n  " + "\n  ".join(errors[:10]))
        print(f"full list: {error_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
