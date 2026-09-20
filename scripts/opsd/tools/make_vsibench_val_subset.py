"""Carve a fixed, type-stratified VSI-Bench validation slice.

The HF generate rollout needs about 53 s per batch of 8, so the full 5,130-row
set costs roughly 9.4 h per validation -- unusable as an in-training signal at
test_freq=7. This slice is the in-training curve; score saved checkpoints on the
full set offline. Both arms must read the same file, so the sample is seeded and
stratified by extra_info.question_type.

  python scripts/opsd/tools/make_vsibench_val_subset.py 480
"""

from __future__ import annotations

import sys

import pandas as pd

SRC = "data/eval/vsibench_verl/vsibench_val_boxed_lastline.parquet"
SEED = 20260909


def main() -> int:
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 480
    out = f"data/eval/vsibench_verl/vsibench_val_boxed_lastline_sub{n_target}.parquet"

    df = pd.read_parquet(SRC)
    qtype = df["extra_info"].apply(lambda e: e["question_type"])

    # Proportional allocation, then top up the largest types so the total is
    # exact -- the file size has to stay divisible by the GPU counts in use.
    share = (qtype.value_counts() / len(df) * n_target).round().astype(int)
    parts = []
    for name, group in df.groupby(qtype, sort=False):
        k = min(int(share.get(name, 0)), len(group))
        if k:
            parts.append(group.sample(n=k, random_state=SEED))
    sub = pd.concat(parts)

    if len(sub) > n_target:
        sub = sub.sample(n=n_target, random_state=SEED)
    elif len(sub) < n_target:
        rest = df.drop(index=sub.index)
        sub = pd.concat([sub, rest.sample(n=n_target - len(sub), random_state=SEED)])

    sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sub.to_parquet(out, index=False)

    counts = sub["extra_info"].apply(lambda e: e["question_type"]).value_counts()
    print(f"{out}: {len(sub)} rows")
    for name, count in counts.items():
        print(f"  {name}: {count}  (full set {int((qtype == name).sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
