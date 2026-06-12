"""
Seal holdout-era slices for the cross-pair divergence-fade survey (2026-06-12).

For each survey instrument (new downloads + GC), copy rows with timestamp
>= 2026-03-01 into data/sealed_holdout/{root}_1min_holdout_20260301_plus.csv,
then apply protect_holdout.py --init (chmod 444 + ACCESS_LOG entry).

Must run BEFORE the Gate 0 survey per
_bmad-output/precommit_pair_divergence_survey_2026-06.md. Existing MNQ/ES
sealed files are not touched (they are already 444; --init leaves them as-is).

Usage:
  .venv/bin/python seal_survey_holdout_slices.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path("data/processed/dollar_bars/1_minute")
HOLDOUT_DIR = Path("data/sealed_holdout")
CUTOFF = "2026-03-01"

ROOTS = ["si", "hg", "pl", "rty", "ym", "gc"]


def main():
    for root in ROOTS:
        src = SRC_DIR / f"{root}_1min_2025_2026.csv"
        dst = HOLDOUT_DIR / f"{root}_1min_holdout_20260301_plus.csv"
        if not src.exists():
            sys.exit(f"FAIL — missing source {src}")
        if dst.exists():
            print(f"{dst.name}: already sealed, skipping")
            continue
        df = pd.read_csv(src, parse_dates=["timestamp"])
        cut = df[df["timestamp"] >= pd.Timestamp(CUTOFF, tz="UTC")]
        cut.to_csv(dst, index=False)
        print(f"{dst.name}: {len(cut):,} rows "
              f"({cut['timestamp'].min()} → {cut['timestamp'].max()})")

    for flag in ("--init", "--verify"):
        r = subprocess.run([sys.executable, "protect_holdout.py", flag])
        if r.returncode != 0:
            sys.exit(f"protect_holdout.py {flag} failed")


if __name__ == "__main__":
    main()
