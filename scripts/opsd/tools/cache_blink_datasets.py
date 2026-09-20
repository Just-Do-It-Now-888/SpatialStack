#!/usr/bin/env python3
"""Materialize BLINK configs under cache/datasets/BLINK-Benchmark__BLINK.

Prefers ModelScope ``evalscope/BLINK`` (huggingface.co parquet CDN is often
unreachable here). Falls back to Hugging Face via hf-mirror.

    python3 scripts/opsd/tools/cache_blink_datasets.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO_ROOT / "cache" / "datasets" / "BLINK-Benchmark__BLINK"

CONFIGS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]


def has_val(path: Path) -> bool:
    if not path.exists():
        return False
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def from_modelscope(name: str, dest: Path) -> None:
    from datasets import DatasetDict, load_dataset
    from modelscope.hub.snapshot_download import snapshot_download

    root = Path(snapshot_download("evalscope/BLINK", repo_type="dataset"))
    val = root / name / "val-00000-of-00001.parquet"
    test = root / name / "test-00000-of-00001.parquet"
    if not val.exists():
        raise FileNotFoundError(f"missing {val}")
    data_files = {"val": str(val)}
    if test.exists():
        data_files["test"] = str(test)
    ds = load_dataset("parquet", data_files=data_files)
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"val": ds})
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dest))
    print(f"  modelscope {name} val_n={len(ds['val'])}", flush=True)


def from_hf(name: str, dest: Path) -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from datasets import load_dataset

    ds = load_dataset("BLINK-Benchmark/BLINK", name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dest))
    n_val = len(ds["val"]) if "val" in ds else None
    print(f"  hf {name} val_n={n_val}", flush=True)


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    missing = [name for name in CONFIGS if not has_val(CACHE_ROOT / name)]
    print(f"cache root {CACHE_ROOT}", flush=True)
    print(f"already present {len(CONFIGS) - len(missing)}/{len(CONFIGS)}", flush=True)
    for name in missing:
        dest = CACHE_ROOT / name
        print(f"materializing {name} -> {dest}", flush=True)
        try:
            from_modelscope(name, dest)
        except Exception as exc:
            print(f"  modelscope failed ({exc!r}); trying HF", flush=True)
            from_hf(name, dest)
    print("done", flush=True)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    sys.exit(main())
