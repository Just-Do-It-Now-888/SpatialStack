import importlib.util
import os
import re
from typing import Any, Dict, List, Optional

from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml

config = load_default_template_yaml(__file__)

# BLINK_PROTOCOL selects a whole protocol. Prompt suffix, stop strings and
# parser move together (same reason as VSIBENCH_PROTOCOL / CVBENCH_PROTOCOL).
#
# * "original": this repo's previous BLINK protocol. Letter-only pre_prompt and
#   start-of-string ``_extract_answer_letter``. Reproduces the 13.05 dump.
# * "spatialstack": VSI-Bench MCA style. Same boxed last-line suffix and
#   boxed-primary + last-line option extract as vsibench_scoring.
# * "lastline": SPAR-Bench wording (no \\boxed{}); last-line option extract only.
#
# Default is "original" so existing numbers stay one env-var away, not overwritten.
BLINK_PROTOCOLS = ("original", "spatialstack", "lastline")
DEFAULT_PROTOCOL = "original"

_SHARED_SCORING = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts", "opsd", "vsibench_scoring.py")
)
_shared_module = None

MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
LETTER_ONLY_PRE_PROMPT = (
    "Return exactly one uppercase option letter from the given choices ({}). "
    "Do not output any explanation, punctuation, or extra text.\n"
)


def _resolve_protocol() -> str:
    value = (os.environ.get("BLINK_PROTOCOL") or DEFAULT_PROTOCOL).strip()
    if value not in BLINK_PROTOCOLS:
        raise ValueError(f"unknown BLINK_PROTOCOL {value!r}; expected one of {BLINK_PROTOCOLS}")
    return value


BLINK_PROTOCOL = _resolve_protocol()


def blink_protocol() -> str:
    return BLINK_PROTOCOL


def _shared():
    global _shared_module
    if _shared_module is None:
        spec = importlib.util.spec_from_file_location("vsibench_scoring", _SHARED_SCORING)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load vsibench_scoring from {_SHARED_SCORING}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _shared_module = module
    return _shared_module


_MAX_NEW_TOKENS_OVERRIDE = os.environ.get("BLINK_MAX_NEW_TOKENS")
BLINK_GENERATION_KWARGS = {
    "max_new_tokens": int(_MAX_NEW_TOKENS_OVERRIDE) if _MAX_NEW_TOKENS_OVERRIDE else 1024,
    "temperature": 0,
    "top_p": 1.0,
    "num_beams": 1,
    "do_sample": False,
}
# Opt-in sample. Same defaults as VSI `--do-sample` (t=1.0 / top_p=0.8).
if (os.environ.get("BLINK_DO_SAMPLE") or "").strip() == "1":
    BLINK_GENERATION_KWARGS["do_sample"] = True
    BLINK_GENERATION_KWARGS["temperature"] = float(os.environ.get("BLINK_TEMPERATURE") or "1.0")
    BLINK_GENERATION_KWARGS["top_p"] = float(os.environ.get("BLINK_TOP_P") or "0.8")
    seed = (os.environ.get("BLINK_SEED") or "").strip()
    if seed:
        BLINK_GENERATION_KWARGS["seed"] = int(seed)
if BLINK_PROTOCOL in ("spatialstack", "lastline"):
    # TaskConfig would otherwise stop at the fewshot delimiter (a blank line).
    BLINK_GENERATION_KWARGS["until"] = []


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    text = text.strip()
    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def _choice_options(doc: dict[str, Any]) -> list[str]:
    choices = doc.get("choices") or []
    return [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]


def blink_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    if BLINK_PROTOCOL == "original":
        num_choices = len(doc["choices"])
        choice_letters = ", ".join([chr(65 + i) for i in range(num_choices)])
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", LETTER_ONLY_PRE_PROMPT)
        return pre_prompt.format(choice_letters) + doc["prompt"]

    mca_post = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or MCA_POST_PROMPT
    text = (doc.get("prompt") or "").rstrip()
    if mca_post and mca_post not in text:
        text = text + "\n" + mca_post
    if BLINK_PROTOCOL == "lastline":
        return text + _shared().LASTLINE_SUFFIX
    return text + _shared().BOXED_LASTLINE_SUFFIX


def blink_doc_to_visual(doc: dict) -> list:
    keys = doc.keys()
    image_keys = [item for item in keys if re.match(r"^image_\d+$", item)]
    image_list = []
    for image_key in image_keys:
        image = doc[image_key]
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "blink_acc"
    grounded_output = doc["answer"].strip("()")
    response = result[0]
    boxed_present = 0

    if BLINK_PROTOCOL == "original":
        pred_letter = _extract_answer_letter(response)
    elif BLINK_PROTOCOL == "lastline":
        pred_letter = _shared().extract_vsibench_option(response, _choice_options(doc))
    else:
        scored_text, boxed_hit = _shared().apply_boxed_primary(response)
        boxed_present = 1 if boxed_hit else 0
        pred_letter = _shared().extract_vsibench_option(scored_text, _choice_options(doc))

    flag = pred_letter == grounded_output
    omnispatial_submission = {
        "id": doc["idx"],
        "gt_content": grounded_output,
        "pred_parsed": pred_letter,
        "pred": response,
        "sub_task": doc["sub_task"],
        "is_correct": flag,
        "protocol": BLINK_PROTOCOL,
        "boxed_present": boxed_present,
        "answered": 1 if pred_letter else 0,
    }
    return {key_name: omnispatial_submission}


def blink_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy
