import importlib.util
import re
import os
import pandas as pd
from pathlib import Path
import yaml
import string
from PIL import Image
from loguru import logger as eval_logger

with open(Path(__file__).parent / "cvbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

# --- evaluation protocol ---------------------------------------------------
# CVBENCH_PROTOCOL selects one of two whole protocols:
#
# * "spatialstack" (default) scores the boxed last-line protocol shared with
#   VSI-Bench: no video preamble, a 1024-token answer budget, and
#   boxed_lastline from scripts/opsd/cvbench_scoring.py. Prompt suffix and
#   parser both changed after the SPAR3 84.82/84.84 dumps; those numbers are
#   not comparable to a new run. CVBENCH_PARSER=word_boundary still reproduces
#   the old extractor on archived generations.
# * "lastline": SPAR-Bench wording (no \\boxed{}); last-line option extract.
#   Same 1024-token budget as spatialstack. Not comparable to boxed dumps.
# * "lmms_legacy" is this task as upstream ships it: 16 new tokens and the
#   first-[A-F] parser, kept so previously published numbers stay reproducible.
#
# The two differ by far more than the parser, and taking half of each is what
# made one checkpoint read 83.57 in training and 5.00 here: at 16 tokens a model
# that reasons before answering is cut off mid-sentence, and the legacy parser
# then scores the "B" of "Based on the provided images". So the switch also owns
# the generation budget below instead of leaving it in the yaml, where the two
# halves could be set independently.
CVBENCH_PROTOCOLS = ("spatialstack", "lastline", "lmms_legacy")
DEFAULT_PROTOCOL = "spatialstack"

# Prompt wording and parser have their own knobs inside the shared module
# (CVBENCH_PROMPT_STYLE, CVBENCH_PARSER); both are read there so the trainer and
# this task cannot end up on different defaults.
_SHARED_SCORING = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts", "opsd", "cvbench_scoring.py")
)
_shared_module = None


def _resolve_protocol():
    value = (os.environ.get("CVBENCH_PROTOCOL") or DEFAULT_PROTOCOL).strip()
    if value not in CVBENCH_PROTOCOLS:
        raise ValueError(f"unknown CVBENCH_PROTOCOL {value!r}; expected one of {CVBENCH_PROTOCOLS}")
    return value


# Resolved once, at import. The generation budget below is fixed at the same
# moment, so re-reading the environment per call would let a late change hand the
# parser one protocol and the answer budget the other.
CVBENCH_PROTOCOL = _resolve_protocol()


def cvbench_protocol():
    return CVBENCH_PROTOCOL


def _shared():
    """Load scripts/opsd/cvbench_scoring.py, the module the trainer scores with.

    Loaded lazily and by path. That module resolves the legacy parser from this
    file, so importing it at module scope would re-enter a half-initialised copy
    of this module, fail, and leave it silently using its own approximation of
    the legacy rule.
    """
    global _shared_module
    if _shared_module is None:
        spec = importlib.util.spec_from_file_location("cvbench_scoring", _SHARED_SCORING)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load the shared CV-Bench scorer from {_SHARED_SCORING}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _shared_module = module
    return _shared_module


_SPATIALSTACK_MAX_NEW_TOKENS = 1024  # verl's data.max_response_length
_LEGACY_MAX_NEW_TOKENS = 16
# CVBENCH_MAX_NEW_TOKENS overrides either protocol's budget. It exists for the
# one comparison the protocols cannot express on their own: upstream's prompt and
# parser at our answer length, which is what separates "the model answers
# differently now" from "the parser reads long answers differently". Setting it
# is deliberate and shows up in the aggregate log, unlike a --gen_kwargs override
# that this file would never see and would misreport.
_MAX_NEW_TOKENS_OVERRIDE = os.environ.get("CVBENCH_MAX_NEW_TOKENS")

# Referenced from cvbench.yaml as `!function utils.CVBENCH_GENERATION_KWARGS`.
# The yaml loader returns whatever the dotted name resolves to and never checks
# that it is callable, so a dict works here and keeps the answer budget tied to
# the protocol rather than sitting in the yaml as a separate, forgettable knob.
CVBENCH_GENERATION_KWARGS = {
    "max_new_tokens": int(_MAX_NEW_TOKENS_OVERRIDE)
    if _MAX_NEW_TOKENS_OVERRIDE
    else (_LEGACY_MAX_NEW_TOKENS if CVBENCH_PROTOCOL == "lmms_legacy" else _SPATIALSTACK_MAX_NEW_TOKENS),
    "temperature": 0,
    "top_p": 1.0,
    "num_beams": 1,
    "do_sample": False,
}
# Opt-in sample. Qwen3.5 generate() keys off temperature>0, so temperature
# must move with do_sample. Defaults match VSI `--do-sample` (t=1.0 / top_p=0.8).
if (os.environ.get("CVBENCH_DO_SAMPLE") or "").strip() == "1":
    CVBENCH_GENERATION_KWARGS["do_sample"] = True
    CVBENCH_GENERATION_KWARGS["temperature"] = float(os.environ.get("CVBENCH_TEMPERATURE") or "1.0")
    CVBENCH_GENERATION_KWARGS["top_p"] = float(os.environ.get("CVBENCH_TOP_P") or "0.8")
    seed = (os.environ.get("CVBENCH_SEED") or "").strip()
    if seed:
        CVBENCH_GENERATION_KWARGS["seed"] = int(seed)
if CVBENCH_PROTOCOL != "lmms_legacy":
    # TaskConfig fills a missing `until` with the fewshot delimiter, i.e. stop at
    # the first blank line. That is invisible at 16 tokens and defeats the whole
    # budget here: the models that honour stop strings would cut a chain of
    # thought off after one paragraph, before it reaches the option letter.
    # Left unset under lmms_legacy so that protocol still inherits it verbatim.
    CVBENCH_GENERATION_KWARGS["until"] = []


def cvbench_doc_to_visual(doc):
    # img_path = os.path.join(cache_dir, doc["filename"])
    # return [Image.open(img_path).convert("RGB")]
    image = doc["image"]
    return [image.convert("RGB")]

def cvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if CVBENCH_PROTOCOL != "lmms_legacy":
        return _shared().build_cvbench_prompt(doc, protocol=CVBENCH_PROTOCOL)

    question = doc["question"]
    # cvbench.yaml sets pre_prompt to "" on purpose -- CV-Bench is single images,
    # not video -- but an empty string is falsy, so every question really does
    # get the video sentence. Preserved here because reproducing the published
    # number means reproducing this too; the default protocol drops it.
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    chars = string.ascii_uppercase
    options = "Options:\n" + "\n".join([f"{chars[i]}. {c}" for i, c in enumerate(doc["choices"])])
    return "\n".join([pre_prompt, question, options, post_prompt])


def extract_characters_regex(s):
    # the choices include ABCDEF
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search(r"[ABCDEF]", s):
        return ""

    matches = re.search(r"[ABCDEF]", s)
    if matches is None:
        return ""
    return matches[0]

def cvbench_process_results(doc, results):
    if CVBENCH_PROTOCOL == "lmms_legacy":
        doc["pred_answer"] = extract_characters_regex(results[0])
    elif CVBENCH_PROTOCOL == "lastline":
        doc["pred_answer"] = _shared().extract_cvbench_option(
            results[0], parser="lastline", choices=doc.get("choices")
        )
    else:
        doc["pred_answer"] = _shared().extract_cvbench_option(results[0], choices=doc.get("choices"))
    doc["result"] = 1 if doc["pred_answer"] == doc["answer"][1] else 0
    # Whether an option letter could be read at all. Accuracy on its own cannot
    # tell a wrong answer apart from a model that stopped answering in this
    # format, and those two need different responses from whoever reads the run.
    doc["answered"] = 1 if doc["pred_answer"] else 0
    return {"cvbench_score": doc, "cvbench_answered": doc}


def cvbench_aggregate_results(results):
    df = pd.DataFrame(results)
    
        # Define a function to calculate accuracy for a given source
    def calculate_accuracy(df, source):
        source_df = df[df['source'] == source]
        accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
        return accuracy

    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    output = {
        "protocol": CVBENCH_PROTOCOL,
        "max_new_tokens": CVBENCH_GENERATION_KWARGS["max_new_tokens"],
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "combined_accuracy": combined_accuracy,
    }
    if "answered" in df:
        output["answered"] = df["answered"].mean()

    for task in ["Count", "Relation", "Distance", "Depth"]:
        output[task] = df[df["task"] == task]["result"].mean()

    eval_logger.info(f"Evaluation results: {output}")
    return output["combined_accuracy"] * 100


def cvbench_aggregate_answered(results):
    """Fraction of rows where an option letter could be read, as a percentage.

    Reported next to the score rather than folded into it: a score that fell
    because the model stopped selecting options is a different failure from one
    that fell because it selected the wrong option, and the score alone looks
    identical in both cases.
    """
    df = pd.DataFrame(results)
    return df["answered"].mean() * 100
