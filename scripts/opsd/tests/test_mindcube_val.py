"""Unit tests for MindCube in-loop validation.

    python3 -m pytest scripts/opsd/tests/test_mindcube_val.py -q

The thing worth protecting here is that the in-training curve and the offline
anchors (46.86 base / 74.48 answer-only SFT / 69.81 CoT SFT) are the *same*
measurement. Three ways that quietly stops being true, one test group each:

* the scorer drifts from `lmms_eval.tasks.mindcube.utils` -- so the parser is
  cross-checked against `mindcube_process_results` on all 1,050 real rows;
* the prompt drifts from the one that produced the anchors -- so the full-view
  parquet is compared byte for byte against `input_prompt` and the system turn;
* the roll-up stops reproducing MindCube's own aggregate, or starts mixing the
  full-view and single-view budgets -- so the metrics are recomputed against the
  upstream aggregate functions and against a deliberately mixed batch.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "verl_pkg"))

import mindcube_scoring as mcs
import mvopsd_reward
from mvopsd import mindcube as mc
from mvopsd import views as vw
from verl.trainer.ppo.mindcube_metrics import compute_mindcube_metrics

import build_mindcube_val_parquet as builder

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
TINYBENCH_ROWS = 1050
# Asserted rather than derived: a swapped rows file should fail here, not produce
# a curve that silently is not the anchors' curve.
DATA_SOURCE_COUNTS = {
    "mindcube/among/2view": 169,
    "mindcube/among/4view": 431,
    "mindcube/around/2view": 105,
    "mindcube/around/3view": 145,
    "mindcube/rotation/3view": 200,
}


@pytest.fixture(scope="module")
def rows():
    if not os.path.exists(builder.DEFAULT_ROWS):
        pytest.skip(f"tinybench rows not present at {builder.DEFAULT_ROWS}")
    with open(builder.DEFAULT_ROWS) as handle:
        return [json.loads(line) for line in handle]


# --- one parser, shared with the offline anchor script ------------------------

@pytest.mark.parametrize(
    "response,expected",
    [
        ("<answer>C. Curtain</answer>", "C"),
        # The CoT arm reasons first and answers last; the letter-and-period rule
        # has highest priority and takes the final one, not the first mention.
        ("Looking at image 1, A. Curtain seems wrong.\n<answer>D. Window</answer>", "D"),
        ("The answer is B", "B"),
        ("", None),
        ("I have no idea", None),
    ],
)
def test_extract_answer_is_the_upstream_definition(response, expected):
    assert mcs.extract_answer(response) == expected


def test_scorer_agrees_with_mindcube_process_results_on_every_row(rows):
    """One definition of correct, in-loop and offline (LESSON-023 §3, LESSON-025).

    Three response shapes rather than one, so a change that only breaks the CoT
    arm's phrasing or only breaks unreadable responses still fails here.
    """
    for response in ("<answer>B. Left</answer>", "A. Above", "no letters here at all"):
        for row in rows:
            upstream = mcs.mindcube_process_results(row, [response])["overall_accuracy"]["score"]
            local, _ = mcs.score_mindcube_row(response, row["gt_answer"])
            assert local == upstream, (row["id"], response)


def test_family_routing_matches_upstream(rows):
    for row in rows:
        upstream = mcs.mindcube_process_results(row, ["A."])["around_accuracy"]["type"]
        assert mcs.family_of(row["id"]) == upstream


def test_reward_routes_all_three_mindcube_prefixes_through_the_same_scorer():
    """Validation and training rows must not disagree about what is correct.

    The generic training branch reads the first ``\\b[A-D]\\b`` in the response,
    which scores ``Looking at image 1, A. ...`` as an A. Routing the training
    data_sources here too is what stops the step-0 diagnostic from contradicting
    the step-0 validation score.
    """
    response = "Looking at image 1, A. Curtain seems wrong.\n<answer>D. Window</answer>"
    for data_source in ("mindcube/among/4view", "mindcube1v/rotation/3view", "mindcube_among_4view"):
        result = mvopsd_reward.compute_score(
            data_source=data_source, solution_str=response, ground_truth="D"
        )
        assert result["acc"] == 1.0, data_source
        assert result["answered"] == 1.0
        assert set(result) == {"score", "acc", "answered", "resp_chars", "boxed_present"}

    unreadable = mvopsd_reward.compute_score(
        data_source="mindcube/among/4view", solution_str="no letters", ground_truth="D"
    )
    # A wrong answer and an unreadable one both score 0; only `answered`
    # separates them, and that distinction cost a full run once (LESSON-011).
    assert (unreadable["acc"], unreadable["answered"]) == (0.0, 0.0)


# --- the full-view prompt is the anchors' prompt ------------------------------

def test_fullview_rows_carry_the_input_prompt_verbatim(rows):
    import random

    rng = random.Random(0)
    for row in rows:
        built = builder.build_row(row, builder.DEFAULT_IMAGE_ROOT, 0, rng)
        content = built["prompt"][1]["content"]
        n_images = len(row["images"])
        assert content == vw.IMAGE_TOKEN * n_images + row["input_prompt"]
        assert built["prompt"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert built["reward_model"]["ground_truth"] == row["gt_answer"]
        assert len(built["images"]) == n_images
        for image in built["images"]:
            # verl has no min_pixels knob; the parquet is the only place the
            # visual budget is set, and it must equal the anchor script's.
            assert image["min_pixels"] == 200704
            assert image["max_pixels"] == 1605632


def test_fullview_data_sources_match_the_expected_composition(rows):
    import random
    from collections import Counter

    rng = random.Random(0)
    built = [builder.build_row(row, builder.DEFAULT_IMAGE_ROOT, 0, rng) for row in rows]
    assert len(built) == TINYBENCH_ROWS
    assert dict(Counter(row["data_source"] for row in built)) == DATA_SOURCE_COUNTS


def test_pixel_caps_are_the_offline_scripts_own_literals():
    """The anchors were measured at these caps; a drift here silently rebases them."""
    assert builder.MIN_PIXELS == 256 * 28 * 28 == 200704
    assert builder.MAX_PIXELS == 1605632


# --- the single-view prompt is the trained condition -------------------------

def test_singleview_rows_show_one_image_and_say_so(rows):
    import random

    rng = random.Random(builder.DEFAULT_SEED)
    pair_rows = 0
    for row in rows:
        built = builder.build_row(row, builder.DEFAULT_IMAGE_ROOT, 1, rng)
        content = built["prompt"][1]["content"]
        assert content.count(vw.IMAGE_TOKEN) == 1 == len(built["images"])
        assert built["data_source"].startswith("mindcube1v/")
        # The available view count stays in the key even though one view is shown:
        # it is what makes the privilege gap readable, and rotation is evaluated
        # only on three-view rows.
        assert built["extra_info"]["n_views_available"] == len(row["images"])
        assert built["extra_info"]["k_views_shown"] == 1

        if built["extra_info"]["prompt_matches_view_count"]:
            body = content[len(vw.IMAGE_TOKEN) :]
            assert mc.TASK_PREAMBLE_SINGULAR in body
            assert mc.TASK_PREAMBLE_PLURAL not in body
            # No reference may survive to a view the model was not given.
            assert vw.referenced_images(body) <= {0}
        else:
            pair_rows += 1

    # The pair template keeps its two-view wording above one image by protocol
    # decision, and it is 26% of tinybench against 13% of the training pool -- so
    # the single-view number is diluted by unanswerable rows twice as hard as the
    # training signal is. Asserted so that ratio cannot drift unnoticed.
    assert pair_rows == 274


def test_singleview_keeps_the_view_the_question_names(rows):
    import random

    rng = random.Random(builder.DEFAULT_SEED)
    for row in rows:
        n_views = len(row["images"])
        question = mc.parse_question(row["input_prompt"], n_views)
        if not question.rewritable:
            continue
        kept = builder.choose_views(question, 1, rng)
        assert len(kept) == 1
        if question.anchor is not None:
            assert kept == [question.anchor]


def test_builder_refuses_a_view_budget_it_cannot_honestly_rewrite():
    """K=2 would need a rotation premise rewritten for the surviving pair.

    The rewriter raises rather than invent one, and the builder rejects the
    argument up front so the failure is a message rather than a stack trace 800
    rows in.
    """
    head = f"\n[Task]\n{mc.TASK_PREAMBLE_PLURAL}\n[Answer Instruction]\nOnly one.\n\n[Question]\n"
    rot = head + (
        "These four images (image 1, 2, 3, and 4) show the same scene from different viewpoints. "
        "The camera turned 90 degrees each time. Based on these four images: facing as in image 4, "
        "what is to my left? A. Bin B. Path"
    )
    question = mc.parse_question(rot, 4)
    with pytest.raises(mc.MindCubeRewriteError):
        mc.rewrite_for_views(question, [0, 3])


# --- the roll-up reproduces MindCube's own aggregate -------------------------

def test_overall_acc_equals_the_upstream_aggregate(rows):
    """MindCube's published score is a plain mean, so this must be exact."""
    import random

    rng = random.Random(1)
    accuracies = [float(rng.random() < 0.7) for _ in rows]
    data_sources = [f"mindcube/{mcs.family_of(row['id'])}/{len(row['images'])}view" for row in rows]

    metrics = compute_mindcube_metrics(data_sources, accuracies)
    expected_overall = mcs.mindcube_aggregate_results([{"score": a} for a in accuracies])
    assert metrics["val-core/mindcube/overall/acc"] == pytest.approx(expected_overall)
    assert metrics["val-aux/mindcube/num_samples"] == TINYBENCH_ROWS

    typed = [
        {"score": a, "type": mcs.family_of(row["id"])} for a, row in zip(accuracies, rows)
    ]
    for family, aggregate in (
        ("among", mcs.mindcube_aggregate_among_results),
        ("around", mcs.mindcube_aggregate_around_results),
        ("rotation", mcs.mindcube_aggregate_rotation_results),
    ):
        assert metrics[f"val-aux/mindcube/family/{family}/acc"] == pytest.approx(aggregate(typed))


def test_view_budgets_never_pool_and_training_rows_never_leak_in():
    """`mindcube1v/` must not land in `mindcube/`, and `mindcube_...` in neither.

    The train pool's data_sources use an underscore where the val ones use a
    slash, which is what keeps a mixed batch safe. If that convention is ever
    broken, a training row's accuracy would be averaged into the validation
    headline and the anchor comparison would quietly stop holding.
    """
    data_sources = [
        "mindcube/among/4view",
        "mindcube/among/4view",
        "mindcube1v/among/4view",
        "mindcube_among_4view",
        "mindcube_pair_2view",
        "cvbench/COCO/Count",
        "vsibench/object_counting",
    ]
    accuracies = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    metrics = compute_mindcube_metrics(data_sources, accuracies)

    assert metrics["val-core/mindcube/overall/acc"] == 1.0
    assert metrics["val-aux/mindcube/num_samples"] == 2
    assert metrics["val-core/mindcube1v/overall/acc"] == 0.0
    assert metrics["val-aux/mindcube1v/num_samples"] == 1


def test_no_mindcube_rows_reports_nothing_rather_than_zero():
    """A zero would read as "the model got everything wrong" on a VSI-only run."""
    assert compute_mindcube_metrics(["vsibench/route_planning"], [1.0]) == {}
    assert compute_mindcube_metrics([], []) == {}


def test_length_series_reports_median_and_p95_not_just_mean():
    """A rising p95 over a flat median is the early degeneration signal (LESSON-020).

    The answer-only arm starts near 11 tokens, so it has the most room to blow up
    and the mean alone would lag behind the tail.
    """
    data_sources = ["mindcube/among/4view"] * 20
    accuracies = [1.0] * 20
    tokens = [11.0] * 19 + [2048.0]
    metrics = compute_mindcube_metrics(
        data_sources, accuracies, answered=[1.0] * 20, resp_tokens=tokens
    )
    assert metrics["val-core/mindcube/resp_tokens/median"] == 11.0
    assert metrics["val-core/mindcube/resp_tokens/p95"] == 2048.0
    assert metrics["val-core/mindcube/resp_tokens/mean"] > 100.0


def test_malformed_data_source_is_ignored_rather_than_crashing_validation():
    """Validation must not be the thing that kills a training run."""
    assert compute_mindcube_metrics(["mindcube/among"], [1.0]) == {}
    assert compute_mindcube_metrics(["mindcube/among/manyview"], [1.0]) == {}
