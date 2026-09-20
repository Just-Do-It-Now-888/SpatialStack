import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "opsd"))

from lmms_eval.tasks.sparbench.utils import LASTLINE_SUFFIX, score_sparbench_prediction, sparbench_doc_to_text
from vsibench_scoring import BOXED_LASTLINE_SUFFIX


def test_mca_cot_last_letter():
    pred = "To determine the relation we look at the boxes.\n\nB"
    official = score_sparbench_prediction("obj_spatial_relation_oo", pred, "B", protocol="official")
    lastline = score_sparbench_prediction("obj_spatial_relation_oo", pred, "B", protocol="lastline")
    assert official[0] == 0.0
    assert lastline[0] == 1.0
    assert lastline[1] == "B"


def test_na_uses_last_line_number():
    pred = "The red point is at 1.0 meters. The blue towel is closer.\n\n0.8"
    lastline = score_sparbench_prediction("depth_prediction_oc", pred, "0.9", protocol="lastline")
    assert lastline[1] == "0.8"
    official = score_sparbench_prediction("depth_prediction_oc", pred, "0.9", protocol="official")
    assert official[1] == "1.0"


def test_boxed_prompt_uses_boxed_suffix():
    doc = {"question": "Where is the cup?", "task": "obj_spatial_relation_oo"}
    kwargs = {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a single word or phrase.",
    }
    boxed = sparbench_doc_to_text(doc, kwargs, protocol="boxed")
    lastline = sparbench_doc_to_text(doc, kwargs, protocol="lastline")
    assert boxed.endswith(BOXED_LASTLINE_SUFFIX)
    assert lastline.endswith(LASTLINE_SUFFIX)
    assert BOXED_LASTLINE_SUFFIX not in lastline
    assert LASTLINE_SUFFIX not in boxed


def test_boxed_primary_beats_trailing_prose():
    pred = "I choose A.\n\\boxed{B}\nThe answer is C"
    boxed = score_sparbench_prediction("obj_spatial_relation_oo", pred, "B", protocol="boxed")
    lastline = score_sparbench_prediction("obj_spatial_relation_oo", pred, "B", protocol="lastline")
    assert boxed[0] == 1.0
    assert boxed[1] == "B"
    assert lastline[1] == "C"


def test_boxed_falls_back_to_last_line_without_closed_box():
    pred = "To determine the relation we look at the boxes.\n\nB"
    boxed = score_sparbench_prediction("obj_spatial_relation_oo", pred, "B", protocol="boxed")
    assert boxed[0] == 1.0
    assert boxed[1] == "B"


if __name__ == "__main__":
    test_mca_cot_last_letter()
    test_na_uses_last_line_number()
    test_boxed_prompt_uses_boxed_suffix()
    test_boxed_primary_beats_trailing_prose()
    test_boxed_falls_back_to_last_line_without_closed_box()
    print("ok")
