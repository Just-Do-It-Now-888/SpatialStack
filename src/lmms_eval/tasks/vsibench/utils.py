
import importlib.util
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]
if os.path.isdir(dataset_path):
    cache_dir = dataset_path
else:
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)

# --- evaluation protocol ---------------------------------------------------
# VSIBENCH_PROTOCOL selects a whole protocol, not a single knob:
#
# * "spatialstack" (default): the in-training val protocol. 4096-token budget,
#   the boxed last-line suffix from vsibench_val_boxed_lastline.parquet, and
#   boxed-primary scoring (closed \\boxed{} first, otherwise answer_tail).
# * "lastline": SPAR-Bench wording (no \\boxed{}). Same 4096-token budget and
#   answer_tail extract as SpatialStack offline, boxed-primary off.
# * "spatialstack_plain": the 2026-08-20 base-anchor prompt — same answer_tail
#   parser, no boxed suffix, no boxed-primary. Budget 1024 unless overridden.
# * "lmms_legacy": upstream verbatim -- 16 new tokens and ``pred.split(' ')[0]``.
#
# Budget, prompt suffix and parser move together on purpose. Half of each is
# what made one CV-Bench checkpoint read 83.57 in training and 5.00 offline:
# at 16 tokens a model that reasons first is cut off before the answer, and a
# first-token parser then scores whatever word it was mid-way through
# (LESSON-017). Old spatialstack dumps without the boxed suffix are not
# comparable to a new run. Default remains boxed last-line.
VSIBENCH_PROTOCOLS = ("spatialstack", "lastline", "spatialstack_plain", "lmms_legacy")
DEFAULT_PROTOCOL = "spatialstack"

_SHARED_SCORING = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts", "opsd", "vsibench_scoring.py")
)
_shared_module = None


def _resolve_protocol():
    value = (os.environ.get("VSIBENCH_PROTOCOL") or DEFAULT_PROTOCOL).strip()
    if value not in VSIBENCH_PROTOCOLS:
        raise ValueError(f"unknown VSIBENCH_PROTOCOL {value!r}; expected one of {VSIBENCH_PROTOCOLS}")
    return value


# Resolved once, at import, together with the generation budget below: re-reading
# the environment per call would let a late change hand the parser one protocol
# and the answer budget the other.
VSIBENCH_PROTOCOL = _resolve_protocol()


def vsibench_protocol():
    return VSIBENCH_PROTOCOL


def _shared():
    global _shared_module
    if _shared_module is None:
        spec = importlib.util.spec_from_file_location("vsibench_scoring", _SHARED_SCORING)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the shared VSI-Bench scorer from {_SHARED_SCORING}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _shared_module = module
    return _shared_module


_SPATIALSTACK_MAX_NEW_TOKENS = 4096  # offline eval + in-loop val default response budget
_PLAIN_MAX_NEW_TOKENS = 1024  # 2026-08-20 unsuffixed base-anchor budget
_LEGACY_MAX_NEW_TOKENS = 16
# VSIBENCH_MAX_NEW_TOKENS overrides either protocol's budget. It exists for the
# one comparison the protocols cannot express on their own: upstream's parser at
# our answer length, which separates "the model answers differently" from "the
# parser reads long answers differently".
_MAX_NEW_TOKENS_OVERRIDE = os.environ.get("VSIBENCH_MAX_NEW_TOKENS")

# Referenced from the yamls as `!function utils.VSIBENCH_GENERATION_KWARGS`. The
# loader returns whatever the dotted name resolves to without checking that it is
# callable, so a dict keeps the budget tied to the protocol instead of sitting in
# the yaml as a separate, forgettable knob.
def _protocol_max_new_tokens() -> int:
    if VSIBENCH_PROTOCOL == "lmms_legacy":
        return _LEGACY_MAX_NEW_TOKENS
    if VSIBENCH_PROTOCOL == "spatialstack_plain":
        return _PLAIN_MAX_NEW_TOKENS
    # spatialstack and lastline share the SpatialStack offline budget.
    return _SPATIALSTACK_MAX_NEW_TOKENS


VSIBENCH_GENERATION_KWARGS = {
    "max_new_tokens": int(_MAX_NEW_TOKENS_OVERRIDE)
    if _MAX_NEW_TOKENS_OVERRIDE
    else _protocol_max_new_tokens(),
    "temperature": 0,
    "top_p": 1.0,
    "num_beams": 1,
    "do_sample": False,
}
if VSIBENCH_PROTOCOL != "lmms_legacy":
    # TaskConfig fills a missing `until` with the fewshot delimiter, i.e. stop at
    # the first blank line. Invisible at 16 tokens, but it would cut a chain of
    # thought off after one paragraph, before it reaches the answer.
    VSIBENCH_GENERATION_KWARGS["until"] = []


def vsibench_doc_to_visual(doc):
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text_plain(doc, lmms_eval_specific_kwargs=None):
    """Question text without the boxed last-line suffix.

    The val parquet builder concatenates ``prompt_suffix`` itself so a rebuild
    of the plain file stays byte-identical (LESSON-013). Offline spatialstack
    adds the suffix in ``vsibench_doc_to_text``.
    """
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    if doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    raise ValueError(f"Unknown question type: {doc['question_type']}")


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    text = vsibench_doc_to_text_plain(doc, lmms_eval_specific_kwargs)
    if VSIBENCH_PROTOCOL in ("lmms_legacy", "spatialstack_plain"):
        return text
    if VSIBENCH_PROTOCOL == "lastline":
        return text + _shared().LASTLINE_SUFFIX
    return text + _shared().BOXED_LASTLINE_SUFFIX


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def process_docs_random200(dataset: datasets.Dataset) -> datasets.Dataset:
    sample_size = min(200, len(dataset))
    eval_logger.info(
        f"Using a fixed random VSI-Bench subset of {sample_size} samples (seed=42)."
    )
    return dataset.shuffle(seed=42).select(range(sample_size))

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

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

def vsibench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    legacy = VSIBENCH_PROTOCOL == "lmms_legacy"
    scored_text = doc["prediction"]
    if VSIBENCH_PROTOCOL == "spatialstack":
        scored_text, boxed_present = _shared().apply_boxed_primary(doc["prediction"])
        doc["boxed_present"] = 1 if boxed_present else 0
    if doc['question_type'] in MCA_QUESTION_TYPES:
        if legacy:
            parsed = fuzzy_matching(doc['prediction'])
        else:
            parsed = _shared().extract_vsibench_option(scored_text, doc.get('options'))
        doc['parsed_answer'] = parsed
        doc['answered'] = 1 if parsed else 0
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(parsed, doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        if legacy:
            parsed = to_float(fuzzy_matching(doc['prediction']))
        else:
            parsed = _shared().extract_vsibench_number(scored_text)
        doc['parsed_answer'] = "" if parsed is None else str(parsed)
        doc['answered'] = 0 if parsed is None else 1
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(parsed, to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc, "vsibench_answered": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                if metric in per_question_type:
                    metric_value = per_question_type[metric].dropna().mean()
                    if pd.notna(metric_value):
                        output[f"{question_type}_{metric}"] = metric_value
                    else:
                        eval_logger.warning(f"Metric {metric} empty for question type {question_type}, skipping.")
                else:
                    eval_logger.warning(f"Metric {metric} missing for question type {question_type}, skipping.")
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric in per_question_type:
                    metric_value = per_question_type[metric].dropna().mean()
                    if pd.notna(metric_value):
                        output[f"{question_type}_{metric}"] = metric_value
                    else:
                        eval_logger.warning(f"Metric {metric} empty for question type {question_type}, skipping.")
                else:
                    eval_logger.warning(f"Metric {metric} missing for question type {question_type}, skipping.")

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    direction_keys = [
        'object_rel_direction_easy_accuracy',
        'object_rel_direction_medium_accuracy',
        'object_rel_direction_hard_accuracy',
    ]
    direction_values = [output[key] for key in direction_keys if key in output]
    missing_direction_keys = [key for key in direction_keys if key not in output]

    if direction_values:
        output['object_rel_direction_accuracy'] = sum(direction_values) / len(direction_values)
        for key in direction_keys:
            output.pop(key, None)
        if missing_direction_keys:
            eval_logger.warning(f"Missing object-relative direction metrics: {missing_direction_keys}. Aggregated using available values.")
    else:
        eval_logger.warning("No object-relative direction accuracies found to aggregate.")
    
    if not output:
        eval_logger.warning("No metrics produced; returning 0.")
        return 0.
    
    output['overall'] = sum(output.values()) / len(output)
    # Logged next to the score but deliberately added after `overall` is
    # computed: these describe the run, they are not question-type scores and
    # must not enter the unweighted mean that defines the VSI-Bench number.
    eval_logger.info(
        f"Evaluation results: {output} "
        f"[protocol={VSIBENCH_PROTOCOL}, max_new_tokens={VSIBENCH_GENERATION_KWARGS['max_new_tokens']}"
        + (f", answered={results['answered'].mean():.4f}" if 'answered' in results else "")
        + "]"
    )
    return output['overall'] * 100.


def vsibench_aggregate_answered(results):
    """Fraction of rows an answer could be read from, as a percentage.

    Reported beside the score rather than folded into it: a score that fell
    because the model stopped answering in a readable format is a different
    failure from one that fell because the answers became wrong, and the score
    alone looks identical in both cases.
    """
    df = pd.DataFrame(results)
    if 'answered' not in df:
        return 0.
    return df['answered'].mean() * 100.
