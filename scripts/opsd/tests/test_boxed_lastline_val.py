#!/usr/bin/env python3
"""Guard the boxed+last-line val parquet against the plain/boxed files.

    python3 scripts/opsd/tests/test_boxed_lastline_val.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "opsd"))

from build_vsibench_val_parquet import BOXED_LASTLINE_SUFFIX  # noqa: E402

VAL_DIR = os.path.join(REPO_ROOT, "data/eval/vsibench_verl")
PLAIN = os.path.join(VAL_DIR, "vsibench_val.parquet")
BOXED = os.path.join(VAL_DIR, "vsibench_val_boxed.parquet")
LASTLINE = os.path.join(VAL_DIR, "vsibench_val_boxed_lastline.parquet")

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def user_text(row) -> str:
    return row["prompt"][1]["content"]


def main() -> None:
    print("\n[1] suffix wording")
    check("\\boxed{}" in BOXED_LASTLINE_SUFFIX, "asks for \\boxed{}")
    check("last line" in BOXED_LASTLINE_SUFFIX.lower(), "asks for the last line")

    if not os.path.isfile(LASTLINE):
        print("  skip (vsibench_val_boxed_lastline.parquet not built yet)")
        print()
        print("FAILED" if failures else "OK")
        raise SystemExit(1 if failures else 0)

    import pandas as pd

    last = pd.read_parquet(LASTLINE)
    print(f"\n[2] boxed_lastline parquet ({len(last)} rows)")
    check(len(last) == 5130, f"row count {len(last)}")
    extras = last["extra_info"].tolist()
    check(
        all(e.get("prompt_suffix") == BOXED_LASTLINE_SUFFIX for e in extras),
        "every row records the last-line suffix",
    )
    sample = user_text(last.iloc[0].to_dict())
    check(BOXED_LASTLINE_SUFFIX in sample, f"user turn contains suffix (...{sample[-80:]!r})")

    if os.path.isfile(PLAIN) and os.path.isfile(BOXED):
        print("\n[3] differs from plain/boxed only by suffix")
        plain = pd.read_parquet(PLAIN)
        boxed = pd.read_parquet(BOXED)
        n_from_plain = 0
        n_diff_boxed = 0
        for i in range(len(last)):
            lu = user_text(last.iloc[i].to_dict())
            pu = user_text(plain.iloc[i].to_dict())
            bu = user_text(boxed.iloc[i].to_dict())
            if lu == pu + BOXED_LASTLINE_SUFFIX:
                n_from_plain += 1
            if lu != bu:
                n_diff_boxed += 1
        check(n_from_plain == len(last), f"plain + suffix == lastline for {n_from_plain}/{len(last)}")
        check(n_diff_boxed == len(last), f"every row differs from boxed-only ({n_diff_boxed})")

    print()
    if failures:
        print(f"FAILED ({len(failures)})")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
