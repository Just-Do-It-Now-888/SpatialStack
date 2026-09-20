
import importlib.util
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import re
import datasets
import math

MCA_QUESTION_TYPES = [
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
    "distance_infer_center_oo_mv"
]
NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",

    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]


SPECIAL_QUESTION_TYPES = [
    "view_change_infer",
]

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

Low = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]

Middle = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

High = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv"
]

# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)
# base_cache_dir = "/cache/data/sparbench_tiny_image"
# base_cache_dir = "/cache/data/sparbench_image"



with open(Path(__file__).parent / "sparbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

_DEFAULT_MAX_NEW_TOKENS = 100
_MAX_NEW_TOKENS_OVERRIDE = os.environ.get("SPARBENCH_MAX_NEW_TOKENS")
SPARBENCH_MAX_NEW_TOKENS = (
    int(_MAX_NEW_TOKENS_OVERRIDE) if _MAX_NEW_TOKENS_OVERRIDE else _DEFAULT_MAX_NEW_TOKENS
)

# official = upstream full-string MCA / full-text NA-VCI (39.88 / 13.87 @2048).
# lastline = VSI answer_tail extract, no \\boxed{}; prompt asks for last-line answer.
# boxed = lastline parsers after boxed-primary; prompt asks for \\boxed{} on last line.
SPARBENCH_PROTOCOLS = ("official", "lastline", "boxed")
DEFAULT_PROTOCOL = "official"
LASTLINE_SUFFIX = (
    " The final answer MUST BE put on the last line of your response."
)
_SHARED_SCORING = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "scripts",
        "opsd",
        "vsibench_scoring.py",
    )
)
_shared_module = None
_LETTER_GOLD_RE = re.compile(r"^[A-F]$", re.IGNORECASE)


def _resolve_protocol():
    value = (os.environ.get("SPARBENCH_PROTOCOL") or DEFAULT_PROTOCOL).strip()
    if value not in SPARBENCH_PROTOCOLS:
        raise ValueError(
            f"unknown SPARBENCH_PROTOCOL {value!r}; expected one of {SPARBENCH_PROTOCOLS}"
        )
    return value


SPARBENCH_PROTOCOL = _resolve_protocol()


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


def lastline_enabled(protocol: str | None = None) -> bool:
    return (protocol or SPARBENCH_PROTOCOL) == "lastline"


def boxed_enabled(protocol: str | None = None) -> bool:
    return (protocol or SPARBENCH_PROTOCOL) == "boxed"


def lastline_parse_enabled(protocol: str | None = None) -> bool:
    return (protocol or SPARBENCH_PROTOCOL) in ("lastline", "boxed")


def boxed_suffix() -> str:
    return _shared().BOXED_LASTLINE_SUFFIX


def _last_parse_line(text: str) -> str:
    scoring = _shared()
    cleaned = scoring._strip_thinking(text or "").strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return scoring._line_for_answer_parse(lines)

# Referenced from sparbench.yaml as `!function utils.SPARBENCH_GENERATION_KWARGS`.
SPARBENCH_GENERATION_KWARGS = {
    "max_new_tokens": SPARBENCH_MAX_NEW_TOKENS,
    "temperature": 0,
    "top_p": 1.0,
    "num_beams": 1,
    "do_sample": False,
}
if SPARBENCH_MAX_NEW_TOKENS > _DEFAULT_MAX_NEW_TOKENS:
    # TaskConfig fills a missing until with the fewshot delimiter ("\n\n").
    # The 100-token dump already stopped there; raising the cap without this
    # would still cut CoT at the first blank line.
    SPARBENCH_GENERATION_KWARGS["until"] = []

eval_logger.info(
    f"[sparbench] protocol={SPARBENCH_PROTOCOL} max_new_tokens={SPARBENCH_MAX_NEW_TOKENS} "
    f"until={SPARBENCH_GENERATION_KWARGS.get('until', '<TaskConfig default>')}"
)

from PIL import Image
def sparbench_doc_to_visual(doc):
    # cache_dir = os.path.join(base_cache_dir)
    # image_path = [os.path.join(cache_dir, i) for i in doc['image']]
    # image = []
    # for i in image_path:
    #     image.append(Image.open(i))
    image = doc['image']
    return image


def sparbench_doc_to_text(doc, lmms_eval_specific_kwargs=None, protocol=None):
    question = doc["question"]
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")

    if doc['task'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        text = pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['task'] in MCA_QUESTION_TYPES:
        post_prompt = ""
        if doc['task'] in ['position_matching', "camera_motion_infer"]:
            post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
        post_prompt2 = lmms_eval_specific_kwargs.get(
            "mca_post_prompt", ""
        ) or "Answer with the option's letter from the given choices directly."
        text = pre_prompt + "\n" + question + "\n" + post_prompt + "\n" + post_prompt2
    elif doc['task'] in SPECIAL_QUESTION_TYPES:
        text = pre_prompt + "\n" + question + "\n"
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")
    if boxed_enabled(protocol):
        text = text + boxed_suffix()
    elif lastline_enabled(protocol):
        text = text + LASTLINE_SUFFIX
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def process_na(pred, task):
    numbers = re.findall(r'(?<!\^)\d+\.\d+|(?<!\^)\d+', pred)

    # Convert the matched numbers to float or int
    extracted_numbers = [float(num) if '.' in num else int(num) for num in numbers]
    if task in ["depth_prediction_oc_mv", 
                "depth_prediction_oo_mv",
                "distance_prediction_oc_mv",
                "distance_prediction_oo_mv",
                ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]


def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}

def compute_vci_metric(pred, answer):

    acion_list = ["move_right", "move_left", 
                  "move_forward", "move_backward", 
                  "move_up", "move_down", 
                  "rotate_right", "rotate_left",
                  "rotate_up", "rotate_down"]
    action_order = ["move_right_left",
                    "move_up_down",
                    "move_forward_backward",
                    "rotate_right_left",
                    "rotate_up_down"]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []

    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))
    
    mra_list = []
    for gt, answer in zip(gt_list, answer_list):
        mra = mean_relative_accuracy(gt, answer, start=.5, end=.95, interval=.05)
        mra_list.append(mra)

    return np.mean(mra_list)

def compute_dic_metric(pred, answer):
    answer = pred
    answer_gt = answer
    if answer == answer_gt:
        return 1
    elif answer_gt in answer:  # TODO: This is a hacky way to handle the case where the answer is a subset of the predicted answer
        return 1
    
    return 0

def parse_cmi(text):
    pattern = r"\([0-9\.]+,[0-9\.]+\)|[0-9\.]+"

    matches = re.findall(pattern, text)

    if len(matches) < 2:
        if len(matches) == 1 and "(" in matches[0]:
            matches.append("0.0")
        elif len(matches) == 1 and "." in matches[0]:
            matches.insert(0, "(0.0,0.0)")

    result = []
    for match in matches:
        if "(" in match and ")" in match:
            num1, num2 = match.strip("()").split(",")
            result.extend([float(num1), float(num2)])
        else:
            result.append(float(match))

    return result

def compute_cmi_metric(pred, answer):

    pred_process = parse_cmi(pred)
    ans_process = parse_cmi(answer)
    dist = math.sqrt(
        (pred_process[0]/1000 - ans_process[0]/1000) ** 2 + 
        (pred_process[1]/1000 - ans_process[1]/1000) ** 2 + 
        (pred_process[2] - ans_process[2]) ** 2
    )
    return dist


def exact_match(pred, target):
    pred = "" if pred is None else str(pred)
    target = "" if target is None else str(target)
    pred_l = pred.lower()
    target_l = target.lower()
    if not pred_l:
        return 0.
    if pred_l == target_l:
        return 1.
    elif pred_l in target_l:
        return 1.
    elif pred_l[0] == target_l:
        return 1.
    else:
        return 0


def score_sparbench_prediction(task: str, pred: str, answer, protocol: str | None = None):
    """Return ``(metric, parsed, kind)``. kind is MCA / NA / VCI."""
    proto = protocol or SPARBENCH_PROTOCOL
    pred = pred or ""
    scored = pred
    if boxed_enabled(proto):
        scored, _ = _shared().apply_boxed_primary(pred, True)
    if task in MCA_QUESTION_TYPES:
        kind = "MCA"
        gold = str(answer).strip()
        if lastline_parse_enabled(proto):
            if _LETTER_GOLD_RE.match(gold):
                parsed = _shared().extract_vsibench_option(scored) or ""
            else:
                parsed = _last_parse_line(scored)
            return exact_match(parsed, gold), parsed, kind
        return exact_match(pred, gold), pred, kind
    if task in NA_QUESTION_TYPES:
        kind = "NA"
        try:
            if lastline_parse_enabled(proto):
                parsed_num = _shared().extract_vsibench_number(scored)
            else:
                parsed_num = to_float(process_na(pred, task))
            metric = mean_relative_accuracy(
                parsed_num, to_float(answer), start=.5, end=.95, interval=.05
            )
            parsed = "" if parsed_num is None else str(parsed_num)
            return float(metric), parsed, kind
        except Exception:
            return WORST_CASE_FOR_METRICS["MRA:.5:.95:.05"], "", kind
    if task in SPECIAL_QUESTION_TYPES:
        kind = "VCI"
        try:
            vci_text = _last_parse_line(scored) if lastline_parse_enabled(proto) else pred
            metric = compute_vci_metric(vci_text, answer)
            return float(metric), vci_text, kind
        except Exception:
            return 0.0, "", kind
    raise ValueError(f"Unknown question type: {task}")

def abs_dist_norm(pred, target):
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def sparbench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    metric, parsed, kind = score_sparbench_prediction(
        doc['task'], doc['prediction'], doc['answer']
    )
    doc['parsed_answer'] = parsed
    doc['protocol'] = SPARBENCH_PROTOCOL
    if kind == "MCA":
        doc['accuracy'] = metric
    elif kind == "NA":
        doc['MRA:.5:.95:.05'] = metric
    elif kind == "VCI":
        doc['vci_metric'] = metric
    else:
        raise ValueError(f"Unknown question type: {doc.get('question_type')}")

    return {"sparbench_score": doc}


def sparbench_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}
    for question_type, question_type_indexes in results.groupby('task').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in SPECIAL_QUESTION_TYPES:
            if question_type == "view_change_infer":
                output[f"{question_type}_vci_metric"] = per_question_type["vci_metric"].mean()
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    # eval_logger.info(f"Evaluation results: {output}")
    low_list = []
    middle_list = []
    high_list = []
    for task in output:
        task_name = "_".join(task.split("_")[:-1])
        if task_name in Low:
            low_list.append(output[task])
        elif task_name in Middle:
            middle_list.append(output[task])
        elif task_name in High:
            high_list.append(output[task])

    output['Low'] = np.mean(low_list)
    output['Middle'] = np.mean(middle_list)
    output['High'] = np.mean(high_list)

    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
