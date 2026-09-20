"""One definition of MindCube answer extraction, shared by in-loop and offline.

`scripts/mindcube/eval_mindcube_qwen35.py` scores the tinybench anchors
(46.86 base / 74.48 answer-only SFT / 69.81 CoT SFT) with
`lmms_eval.tasks.mindcube.utils.extract_answer` and `mindcube_process_results`.
The in-training curve is only readable against those anchors if it uses the same
parser, so this module reaches into that file rather than restating it -- the
CV-Bench round is the cautionary case, where `mvopsd_reward` and `spar_scoring`
each grew their own copy of the same mistake and fixing one did not fix the other
(LESSON-023 §3, LESSON-025).

It is loaded **by file path**, not as `lmms_eval.tasks.mindcube.utils`. lmms_eval
is not installed in the training environment (LESSON-007) and importing the
package would drag in its whole dependency tree; the task file itself imports
nothing but `re` and `typing`, so executing it directly is safe and keeps the
parser byte-identical to the offline one.

Two properties of that parser are worth stating, because they shape how the
metrics read:

* `extract_answer` ends in a `\\b([A-E])\\b` scan over the response's lines, so
  `answered` only goes to zero for a response with no standalone capital A-E in
  it at all. It is a real alarm -- "I have no idea" reads as unanswered -- but a
  blunt one, and it cannot see a model that keeps emitting letters while its
  reasoning rots. Response length is the sensitive format alarm here
  (LESSON-020/029), not `answered`.
* Its first and highest-priority rule is `([A-E])\\.`, which matches the
  `<answer>C. Curtain</answer>` shape both SFT arms were trained to emit. A
  degraded arm that drops the letter-and-period form loses accuracy for a
  formatting reason, which is exactly the confusion LESSON-011/018 warn about.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LMMS_MINDCUBE_UTILS = os.path.join(_REPO_ROOT, "src", "lmms_eval", "tasks", "mindcube", "utils.py")

# tinybench routes families off the id prefix; `mindcube_process_results` owns
# that mapping and this is the same table, kept here only so the val parquet
# builder can name a data_source without holding a fake `doc`.
FAMILY_BY_ID_PREFIX = {
    "among": "among",
    "rotation": "rotation",
    "around": "around",
    "aroundnew": "around",
}
FAMILIES = ("among", "around", "rotation")


def _load_lmms_mindcube_utils():
    if "lmms_eval.tasks.mindcube.utils" in sys.modules:
        return sys.modules["lmms_eval.tasks.mindcube.utils"]
    spec = importlib.util.spec_from_file_location("_lmms_mindcube_utils", LMMS_MINDCUBE_UTILS)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the MindCube task utils from {LMMS_MINDCUBE_UTILS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_lmms_mindcube_utils()

extract_answer = _utils.extract_answer
mindcube_process_results = _utils.mindcube_process_results
mindcube_aggregate_results = _utils.mindcube_aggregate_results
mindcube_aggregate_among_results = _utils.mindcube_aggregate_among_results
mindcube_aggregate_around_results = _utils.mindcube_aggregate_around_results
mindcube_aggregate_rotation_results = _utils.mindcube_aggregate_rotation_results


def family_of(row_id: str) -> str:
    """``among_group693_q1_5_2`` -> ``among``, matching `mindcube_process_results`."""
    prefix = str(row_id).split("_", 1)[0]
    if prefix not in FAMILY_BY_ID_PREFIX:
        raise ValueError(f"unknown MindCube id prefix {prefix!r} in {row_id!r}")
    return FAMILY_BY_ID_PREFIX[prefix]


def score_mindcube_row(prediction: str, target: str) -> tuple[float, float]:
    """Return ``(accuracy, answered)`` for one tinybench row.

    Equivalent to `mindcube_process_results(doc, [prediction])["overall_accuracy"]`
    without needing a `doc`: that function's only use of the document is
    `doc["gt_answer"]`, which is the bare gold letter and is already `target`.
    """
    parsed = extract_answer((prediction or "").strip())
    if parsed is None:
        return 0.0, 0.0
    return float(parsed == (target or "").strip()), 1.0
