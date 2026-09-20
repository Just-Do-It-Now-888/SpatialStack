"""Unit tests for the MindCube MV-OPSD pool (K=1, teacher sees all N).

    python3 -m pytest scripts/opsd/tests/test_mindcube_pool.py -q

Three groups of claims, in rising order of how expensive they are to get wrong:

* the rewriter's contract -- a stale "image 2" or count word above a single image
  is a silent semantic corruption, exactly like a dangling ``Frame-N``;
* one deliberate exception to it -- the 1,302 pair-displacement rows keep their
  two-view wording by protocol decision, asserted here so nobody repairs it;
* the built plan's self-consistency, including that the gold letter still names
  the same option text it named in the annotation (LESSON-021).
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mvopsd import mindcube as mc
from mvopsd import views as vw
from mvopsd.geometry import ALIGN_FACTOR, pixel_budget_size, resize_to_pixel_budget, visual_tokens

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
ANNOTATIONS = os.path.join(REPO_ROOT, "data", "mindcube", "mindcube_train_answeronly.json")
PLAN_PATH = os.path.join(REPO_ROOT, "data", "mvopsd", "plan_mindcube_k1", "plan.jsonl")

# Every row of the 9,999-row training file, by template. Asserted rather than
# derived so a swapped annotation file fails here instead of halfway through a run.
TEMPLATE_COUNTS = {
    "among4": 7055,
    "around3": 778,
    "pair": 1302,
    "rotation_turn": 828,
    "rotation_opposing": 36,
}
SOURCE_COUNTS = {
    "mindcube_among_4view": 7055,
    "mindcube_around_3view": 778,
    "mindcube_pair_2view": 1302,
    "mindcube_rotation_2view": 366,
    "mindcube_rotation_3view": 138,
    "mindcube_rotation_4view": 360,
}
POOL_SIZE = 9999

# The real [Task] / [Answer Instruction] block, byte-identical across all 9,999
# training rows and all 1,050 tinybench rows.
HEAD = f"\n[Task]\n{mc.TASK_PREAMBLE_PLURAL}\n[Answer Instruction]\nOnly one answer.\n\n[Question]\n"
AMONG4 = HEAD + (
    "Based on these four images (image 1, 2, 3, and 4) showing the black sneaker from different "
    "viewpoints (front, left, back, and right), with each camera aligned with room walls and "
    "partially capturing the surroundings: From the viewpoint presented in image 3, what is to "
    "the left of the black sneaker? A. Curtain B. Window"
)
ROT4 = HEAD + (
    "These four images (image 1, 2, 3, and 4) show the same scene from different viewpoints. The "
    "camera basically stood in one spot and turned 90 degrees each time to get all sides of the "
    "area. Based on these four images: If I am standing at the same spot and facing the same "
    "direction as shown in image 4 and turn 180 degrees around, what is to my left? A. Metal bin "
    "B. Pathway"
)
PAIR = HEAD + (
    "Based on these two views showing the same scene: in which direction did I move from the "
    "first view to the second view? A. Directly left B. Directly right"
)


# --- prose helpers ------------------------------------------------------------

def test_enumeration_and_label_lists_follow_mindcubes_own_style():
    assert mc.enumerate_images(1) == "image 1"
    assert mc.enumerate_images(2) == "image 1 and 2"
    assert mc.enumerate_images(3) == "image 1, 2, and 3"
    assert mc.enumerate_images(4) == "image 1, 2, 3, and 4"
    assert mc.join_labels(["front"]) == "front"
    assert mc.join_labels(["front", "back"]) == "front and back"
    assert mc.join_labels(["front", "left", "right"]) == "front, left, and right"


def test_option_parsing_ignores_letters_inside_option_text():
    stem, options = mc.parse_options("what is left? A. Two photographs B. Leather C. sofa")
    assert stem == "what is left?"
    assert options == {"A": "Two photographs", "B": "Leather", "C": "sofa"}
    # "D." never opens an option because C came after A, B -- and a bare run that
    # does not start at A is not an option list at all.
    _, none = mc.parse_options("no options here, just a D. shaped remark")
    assert none == {}


def test_answer_letter_and_text_reject_unknown_formats():
    assert mc.answer_letter("<answer>C. Curtain</answer>") == "C"
    assert mc.answer_text("<answer>C. Curtain</answer>") == "Curtain"
    with pytest.raises(mc.MindCubeRewriteError):
        mc.answer_letter("C")


# --- rewriter contract --------------------------------------------------------

def test_among4_rewrite_moves_count_enumeration_label_and_anchor_together():
    question = mc.parse_question(AMONG4, 4)
    assert question.template == "among4"
    assert question.anchor == 2  # "image 3", 0-based
    assert question.labels == ("front", "left", "back", "right")

    out = mc.rewrite_for_views(question, [2])
    assert "Based on this image (image 1) showing the black sneaker from one viewpoint (back)" in out
    assert "From the viewpoint presented in image 1," in out
    assert "four" not in out and "image 3" not in out
    # The option text survives untouched; a global count-word substitution would
    # have rewritten 262 answers across the pool.
    assert out.endswith("A. Curtain B. Window")


def test_task_preamble_stops_promising_a_plural_album_at_k1():
    out = mc.rewrite_for_views(mc.parse_question(AMONG4, 4), [2])
    assert mc.TASK_PREAMBLE_SINGULAR in out
    assert mc.TASK_PREAMBLE_PLURAL not in out


def test_rotation_premise_is_deleted_not_rewritten():
    question = mc.parse_question(ROT4, 4)
    assert question.template == "rotation_turn"
    assert question.anchor == 3
    out = mc.rewrite_for_views(question, [3])
    assert "This image (image 1) shows the scene from one viewpoint. Based on this image:" in out
    assert "turned 90 degrees each time" not in out
    # What is left is still a mental-rotation question, just anchored on one view.
    assert "as shown in image 1 and turn 180 degrees around" in out


def test_rotation_refuses_k_greater_than_one_instead_of_lying_about_the_camera():
    question = mc.parse_question(ROT4, 4)
    with pytest.raises(mc.MindCubeRewriteError):
        mc.rewrite_for_views(question, [0, 2])


def test_pair_template_keeps_its_stale_wording():
    """Protocol decision 2026-09-07, not a bug: see build_mindcube_plan.py."""
    question = mc.parse_question(PAIR, 2)
    assert question.template == "pair"
    assert not question.rewritable
    out = mc.rewrite_for_views(question, [0])
    assert "Based on these two views showing the same scene:" in out
    assert "in which direction did I move from the first view to the second view?" in out
    # The plural [Task] block stays too: the whole row is passed through verbatim.
    assert "which show the scene from different viewpoints" in out
    assert out == PAIR


def test_rewriting_refuses_to_drop_the_view_the_question_names():
    question = mc.parse_question(AMONG4, 4)
    with pytest.raises(mc.MindCubeRewriteError):
        mc.rewrite_for_views(question, [0])  # the question names image 3


def test_renumber_image_refs_matches_renumber_frame_refs_contract():
    assert vw.renumber_image_refs("as shown in image 3", {2: 0}) == "as shown in image 1"
    assert vw.renumber_image_refs("Image 2 was taken", {1: 1}) == "Image 2 was taken"
    with pytest.raises(KeyError):
        vw.renumber_image_refs("as shown in image 3", {0: 0})


def test_unknown_template_raises_rather_than_passing_through():
    with pytest.raises(mc.MindCubeRewriteError):
        mc.parse_question("\n[Task]\nx\n\n[Question]\nSome brand new wording: what is left?", 4)


# --- geometry -----------------------------------------------------------------

MINDCUBE_MIN_PIXELS = 200704
MINDCUBE_MAX_PIXELS = 1605632


def test_pixel_budget_reproduces_the_sft_geometry_without_cropping():
    # 480x640 is 79% of MindCube's views and already inside the budget, so it is
    # untouched and yields the 300 tokens/image the SFT round gated on.
    assert pixel_budget_size(480, 640, MINDCUBE_MIN_PIXELS, MINDCUBE_MAX_PIXELS) == (480, 640)
    assert visual_tokens((480, 640)) == 300
    # The oversized sources scale down whole rather than losing their edges.
    width, height = pixel_budget_size(2016, 1512, MINDCUBE_MIN_PIXELS, MINDCUBE_MAX_PIXELS)
    assert width * height <= MINDCUBE_MAX_PIXELS
    assert abs((width / height) - (2016 / 1512)) < 0.02
    # And the undersized ones scale up rather than being padded.
    width, height = pixel_budget_size(256, 192, MINDCUBE_MIN_PIXELS, MINDCUBE_MAX_PIXELS)
    assert width * height >= MINDCUBE_MIN_PIXELS


@pytest.mark.parametrize("size", [(480, 640), (2016, 1512), (4032, 3024), (256, 192), (480, 270)])
def test_student_and_teacher_image_paths_agree_on_a_budgeted_view(size):
    """verl's two image paths must be identity transforms on the cached file.

    The student runs ``fetch_image`` with the parquet's caps at
    ``data.image_patch_size=16``; the teacher runs a bare ``Image.open`` and the
    processor's own factor of 32. Both must leave the file alone, or resolution
    becomes a second privileged axis alongside view count.
    """
    from qwen_vl_utils.vision_process import smart_resize

    cached = resize_to_pixel_budget(Image.new("RGB", size, "gray"), MINDCUBE_MIN_PIXELS, MINDCUBE_MAX_PIXELS)
    width, height = cached.size
    assert width % ALIGN_FACTOR == 0 and height % ALIGN_FACTOR == 0

    student = smart_resize(
        height, width, factor=16, min_pixels=MINDCUBE_MIN_PIXELS, max_pixels=MINDCUBE_MAX_PIXELS
    )
    teacher = smart_resize(height, width, factor=ALIGN_FACTOR, min_pixels=65536, max_pixels=16777216)
    assert student == (height, width)
    assert teacher == (height, width)


# --- the whole annotation file ------------------------------------------------

@pytest.fixture(scope="module")
def annotations():
    if not os.path.exists(ANNOTATIONS):
        pytest.skip("MindCube annotations not present")
    return json.load(open(ANNOTATIONS))


def test_mindcube_templates_cover_the_whole_pool(annotations):
    from collections import Counter

    counts = Counter()
    for ann in annotations:
        _, body = vw.split_image_header(ann["conversations"][0]["value"])
        counts[mc.parse_question(body, len(ann["images"])).template] += 1
    assert dict(counts) == TEMPLATE_COUNTS
    assert sum(counts.values()) == POOL_SIZE


def test_rewriting_at_k_equals_n_reproduces_the_source_text(annotations):
    """The one available check that the prefix builders are faithful.

    With every view kept the rebuilt prose must be byte-identical to MindCube's
    own, which pins the count word, the enumeration and the label list at once.
    rotation_turn is exempt because its premise is only ever deleted, never
    regenerated.
    """
    checked = 0
    for ann in annotations:
        _, body = vw.split_image_header(ann["conversations"][0]["value"])
        question = mc.parse_question(body, len(ann["images"]))
        if question.template == "rotation_turn":
            continue
        assert mc.rewrite_for_views(question, list(range(question.n_views))) == body, ann["id"]
        checked += 1
    assert checked == POOL_SIZE - TEMPLATE_COUNTS["rotation_turn"]


def test_every_non_pair_row_names_at_most_one_view(annotations):
    """K=1 is only well defined when the question pins a single view.

    Counted *after* the premise is stripped: on the raw text the rotation premise
    cites two image numbers, which would make 504 legal rows look ineligible.
    """
    anchor_free = 0
    for ann in annotations:
        _, body = vw.split_image_header(ann["conversations"][0]["value"])
        question = mc.parse_question(body, len(ann["images"]))
        if question.template == "pair":
            assert question.anchor is None
            continue
        if question.anchor is None:
            anchor_free += 1
    # The anchor-free rows are the `among` questions that anchor on an object
    # instead ("if I was positioned where the light purple sofa is").
    assert anchor_free == 564


# --- the built plan -----------------------------------------------------------

@pytest.fixture(scope="module")
def plan():
    if not os.path.exists(PLAN_PATH):
        pytest.skip("MindCube plan not built yet")
    return [json.loads(line) for line in open(PLAN_PATH)]


def test_plan_keeps_every_row_and_buckets_them_as_designed(plan):
    from collections import Counter

    assert len(plan) == POOL_SIZE
    assert len({record["sample_id"] for record in plan}) == POOL_SIZE
    assert dict(sorted(Counter(r["source"] for r in plan).items())) == SOURCE_COUNTS
    assert dict(sorted(Counter(r["n_views"] for r in plan).items())) == {2: 1668, 3: 916, 4: 7415}


def test_plan_records_are_self_consistent(plan):
    for record in plan:
        student = vw.build_prompt(record["header_style"], record["k_views"], record["student_body"])
        teacher = vw.build_prompt(record["header_style"], record["n_views"], record["body"])

        assert record["k_views"] == 1, record["sample_id"]
        assert record["k_views"] < record["n_views"], record["sample_id"]
        assert student.count(vw.IMAGE_TOKEN) == record["k_views"] == len(record["view_indices"])
        assert teacher.count(vw.IMAGE_TOKEN) == record["n_views"] == len(record["frames"])
        assert set(record["required_views"]) <= set(record["view_indices"])
        assert vw.referenced_images(record["student_body"]) <= {0}
        assert vw.referenced_images(record["body"]) <= set(range(record["n_views"]))
        assert record["answer"] in "ABCDE" and len(record["answer"]) == 1


def test_plan_student_prompts_carry_no_stale_count_words(plan):
    """Outside the allowlisted pair bucket, nothing may still claim a plural album."""
    stale = (
        "these four images",
        "these three images",
        "these two images",
        "these two views",
        "different viewpoints",
        "which show the scene",
    )
    offenders = []
    for record in plan:
        if record["source"] == "mindcube_pair_2view":
            continue
        body = record["student_body"]
        prefix = body.split("[Question]", 1)[1]
        if any(phrase in body for phrase in stale) or "turning the camera" in prefix:
            offenders.append(record["sample_id"])
    assert offenders == []


def test_plan_pair_bucket_keeps_its_wording_verbatim(plan):
    pairs = [record for record in plan if record["source"] == "mindcube_pair_2view"]
    assert len(pairs) == 1302
    for record in pairs:
        assert record["student_body"] == record["body"], record["sample_id"]
        assert "these two views showing the same scene" in record["student_body"]
        assert record["prompt_matches_view_count"] is False


def test_plan_gold_letter_still_names_the_same_option(plan, annotations):
    """LESSON-021: the label must survive the pool build, not just the parse."""
    gold_by_id = {
        f"mindcube/{ann['id']}": ann["conversations"][1]["value"] for ann in annotations
    }
    for record in plan:
        gold = gold_by_id[record["sample_id"]]
        letter = record["answer"]
        assert letter == mc.answer_letter(gold)
        _, options = mc.parse_options(record["student_body"])
        assert letter in options, record["sample_id"]
        assert options[letter].rstrip(".") == mc.answer_text(gold).rstrip("."), record["sample_id"]


def test_plan_views_are_shared_between_rows(plan):
    """9,999 rows reference 2,785 distinct files; stage 2 must not write each row's copy."""
    caches = {frame["cache"] for record in plan for frame in record["frames"]}
    assert len(caches) == 2785
    assert sum(len(record["frames"]) for record in plan) == 35744
    for record in plan:
        for frame in record["frames"]:
            assert frame["geometry"] == "pixel_budget"
            assert frame["marked"] is False and frame["frame_index"] is None


def test_anchor_free_rows_draw_their_view_rather_than_always_taking_the_front(plan):
    from collections import Counter

    free = [record for record in plan if record["view_selection"] == "pure_random"]
    # The `among` rows that anchor on an object instead of an image. The 1,302
    # pair rows are labelled pair_first_view: they pin view 0, since a draw would
    # only add variance to rows that carry no recoverable signal either way.
    assert len(free) == 564
    assert Counter(r["view_selection"] for r in plan)["pair_first_view"] == 1302
    picked = Counter(record["view_indices"][0] for record in free)
    assert set(picked) == {0, 1, 2, 3}


def test_build_record_is_deterministic_under_a_seed(annotations):
    sample = annotations[:200]
    first = [mc_record["view_indices"] for mc_record in _build(sample, 7)]
    assert first == [mc_record["view_indices"] for mc_record in _build(sample, 7)]
    assert first != [mc_record["view_indices"] for mc_record in _build(sample, 8)]


def _build(annotations, seed):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from build_mindcube_plan import build_mindcube_record

    rng = random.Random(seed)
    return [build_mindcube_record(ann, rng) for ann in annotations]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
