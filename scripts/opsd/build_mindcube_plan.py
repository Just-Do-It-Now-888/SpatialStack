#!/usr/bin/env python3
"""MindCube MV-OPSD pool: teacher sees all N views, student sees exactly one.

    python3 scripts/opsd/build_mindcube_plan.py

All 9,999 training rows are kept, with no filtering (user decision 2026-09-07).
Two consequences worth stating up front, because they bound what the run can
show:

* 1,302 rows (13%) are the pair-displacement template -- "in which direction did
  I move from the first view to the second view?" -- whose subject *is* the
  relation between the two views. There is no single-view form, so per the chosen
  protocol they keep their two-view wording while receiving one image. Their
  prompt therefore describes views the student cannot see and their label is not
  recoverable from its input: they can only transmit the teacher's letter prior.
  They get their own ``data_source`` (``mindcube_pair_2view``) so that
  contribution stays separable after the fact.
* 564 ``among`` rows anchor on an object rather than an image ("if I was
  positioned where the light purple sofa is"), so the kept view may not contain
  that object. Those are well-formed prompts but may be evidence-free; the
  pre-launch recoverability probe is what quantifies how often.

Pixels are untouched here. Feed the output to materialize_mvopsd_views.py and
then write_mvopsd_parquet.py --arms main --holdout-per-source 0.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvopsd import mindcube as mc  # noqa: E402
from mvopsd import sources as src  # noqa: E402
from mvopsd import views as vw  # noqa: E402

# The rotation family's premise sentence is deleted rather than rewritten, so
# these rows reach the student with no stated frame of reference; among/around
# keep a viewpoint label. Separate buckets because those are different tasks at
# K=1 and a pooled average would hide which one moved.
FAMILY_BY_ID_PREFIX = {"among": "among", "aroundnew": "around", "rotation": "rotation"}

# `mindcube_process_results` routes on the same prefix, so the training buckets
# and the tinybench families are named by one rule.
PAIR_SOURCE = "mindcube_pair_2view"


def family_of(sample_id: str) -> str:
    prefix = sample_id.split("_", 1)[0]
    if prefix not in FAMILY_BY_ID_PREFIX:
        raise ValueError(f"unknown MindCube id prefix {prefix!r} in {sample_id!r}")
    return FAMILY_BY_ID_PREFIX[prefix]


def scene_id_of(image_rel: str) -> str:
    """``other_all_image/among/shoe_173/front_301.jpg`` -> ``among/shoe_173``.

    The family directory stays in the key: `around/group71` and `rotation/group75`
    both exist, and group numbers are only unique within a family.
    """
    parts = image_rel.split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot derive a scene from {image_rel!r}")
    return "/".join(parts[-3:-1])


def choose_student_view(question: mc.MindCubeQuestion, rng: random.Random) -> list[int]:
    """The one view the student keeps: the named anchor, else a uniform draw.

    Drawing rather than always taking view 0 keeps the anchor-free rows from
    becoming a systematic "front view only" subset, which would confound the
    view budget with viewpoint.
    """
    if question.anchor is not None:
        return [question.anchor]
    return [rng.randrange(question.n_views)]


def build_mindcube_record(ann: dict, rng: random.Random, k_views: int = 1) -> dict:
    images = ann["images"]
    n_views = len(images)
    family = family_of(ann["id"])

    header, body = vw.split_image_header(ann["conversations"][0]["value"])
    if vw.has_frame_labels(header):
        raise ValueError(f"{ann['id']}: unexpected Frame-N header in MindCube")
    question = mc.parse_question(body, n_views)

    if question.template == "pair":
        # No rewrite exists, so the student gets K images under the original
        # two-view wording. Keep the first view: an arbitrary draw would add
        # variance to rows that carry no recoverable signal either way.
        view_indices = list(range(k_views))
        source = PAIR_SOURCE
    else:
        view_indices = choose_student_view(question, rng)
        if k_views != 1:
            raise ValueError(
                f"{ann['id']}: choose_student_view only builds K=1; got k_views={k_views}"
            )
        source = f"mindcube_{family}_{n_views}view"

    student_body = mc.rewrite_for_views(question, view_indices)

    frames = [
        {
            # Views are shared: 9,999 rows reference 2,785 distinct files, so the
            # cache key is the source path and stage 2 writes each file once.
            "src": src.MINDCUBE_SOURCE.media_path(rel),
            "cache": f"mindcube/{os.path.splitext(rel)[0]}.jpg",
            "marked": False,
            "frame_index": None,
            # Stage 2 reads this: MindCube must not go through the VGGT
            # centre-crop chain (see mvopsd/geometry.py).
            "geometry": "pixel_budget",
        }
        for rel in images
    ]

    return {
        "sample_id": f"mindcube/{ann['id']}",
        "dataset": "mindcube",
        "source": source,
        "subset": family,
        "scene_id": scene_id_of(images[0]),
        "question_type": question.template,
        "n_views": n_views,
        "k_views": len(view_indices),
        "view_indices": view_indices,
        # The anchor is the only view the question cannot lose. Anchor-free rows
        # have none, which is what makes their kept view a free draw.
        "required_views": [question.anchor] if question.anchor is not None else [],
        "marked_views": [],
        "view_selection": (
            "pair_first_view"
            if question.template == "pair"
            else "anchored"
            if question.anchor is not None
            else "pure_random"
        ),
        "privilege_bucket": "strong" if n_views >= 3 else "weak",
        "answer_view_sensitive": question.template == "pair",
        "header_style": "bare",
        "body": body,
        "student_body": student_body,
        "answer": mc.answer_letter(ann["conversations"][1]["value"]),
        "spar_info": None,
        "frames": frames,
        # Carried so the parquet and the reading of it can separate the rows whose
        # prompt knowingly contradicts its own input.
        "prompt_matches_view_count": question.template != "pair",
    }


def summarize(records: list[dict]) -> dict:
    by_source = Counter(record["source"] for record in records)
    ratios = [record["n_views"] / record["k_views"] for record in records]
    return {
        "total_samples": len(records),
        "kept_by_source": dict(sorted(by_source.items())),
        "kept_by_family": dict(sorted(Counter(r["subset"] for r in records).items())),
        "kept_by_template": dict(sorted(Counter(r["question_type"] for r in records).items())),
        "kept_by_k": {str(k): v for k, v in sorted(Counter(r["k_views"] for r in records).items())},
        "kept_by_n": {str(k): v for k, v in sorted(Counter(r["n_views"] for r in records).items())},
        "kept_by_view_selection": dict(sorted(Counter(r["view_selection"] for r in records).items())),
        "privilege_ratio": {
            str(k): v for k, v in sorted(Counter(round(r, 1) for r in ratios).items())
        },
        "mean_teacher_views": sum(r["n_views"] for r in records) / len(records),
        "mean_ratio": sum(ratios) / len(records),
        "rows_whose_prompt_matches_their_view_count": sum(
            1 for r in records if r["prompt_matches_view_count"]
        ),
        "answer_letters": dict(sorted(Counter(r["answer"] for r in records).items())),
        "unique_scenes": len({r["scene_id"] for r in records}),
        "unique_sample_ids": len({r["sample_id"] for r in records}),
        "unique_view_files": len({f["cache"] for r in records for f in r["frames"]}),
        "expanded_view_references": sum(len(r["frames"]) for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="data/mvopsd/plan_mindcube_k1")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--k-views",
        type=int,
        default=1,
        help="student budget. Only 1 is supported; the rotation premise cannot be "
        "rewritten for a surviving pair, so K=2 raises rather than lie about the "
        "camera motion.",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: first N annotations")
    args = parser.parse_args()

    source = src.MINDCUBE_SOURCE
    annotations = json.load(open(source.annotation_path()))
    print(f"[mindcube] {len(annotations)} annotations from {source.annotation}")
    if source.sft_expected and not args.limit and len(annotations) != source.sft_expected:
        raise SystemExit(
            f"expected {source.sft_expected} rows (the set the 20260906 SFT round trained on), "
            f"found {len(annotations)}"
        )
    if args.limit:
        annotations = annotations[: args.limit]

    rng = random.Random(args.seed)
    records = [build_mindcube_record(ann, rng, args.k_views) for ann in annotations]
    if len(records) != len(annotations):
        raise SystemExit("the pool is built with no exclusions; every annotation must yield a row")

    random.Random(args.seed + 7).shuffle(records)

    out_dir = os.path.join(src.REPO_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    plan_path = os.path.join(out_dir, "plan.jsonl")
    with open(plan_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = summarize(records)
    stats.update({"seed": args.seed, "k_views": args.k_views, "annotation": source.annotation})
    with open(os.path.join(out_dir, "stats.json"), "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(f"\nplan written to {plan_path} ({len(records)} samples)")
    print("\n=== data_source buckets ===")
    for name, count in stats["kept_by_source"].items():
        print(f"  {name:<26} {count:>6}  {100 * count / len(records):5.1f}%")
    print("\n=== teacher views N ===")
    for name, count in stats["kept_by_n"].items():
        print(f"  N={name:<4} {count:>6}")
    print("\n=== privilege ratio N/K ===")
    for name, count in stats["privilege_ratio"].items():
        print(f"  {name:<6} {count:>6}")
    print(f"\nmean teacher views {stats['mean_teacher_views']:.2f}, mean N/K {stats['mean_ratio']:.2f}")
    evidence_free = len(records) - stats["rows_whose_prompt_matches_their_view_count"]
    print(
        f"rows whose prompt knowingly contradicts its own input: {evidence_free} "
        f"({100 * evidence_free / len(records):.1f}%) -- all in {PAIR_SOURCE}"
    )
    print(f"unique view files to materialise: {stats['unique_view_files']:,} "
          f"(from {stats['expanded_view_references']:,} references)")
    print(f"unique scenes {stats['unique_scenes']}, answer letters {stats['answer_letters']}")


if __name__ == "__main__":
    main()
