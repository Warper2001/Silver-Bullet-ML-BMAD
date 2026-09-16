"""Are the other instruments' 1-min files the front month before the March 2026 roll?

The MNQ defect: `mnq_1min_2026_ytd.csv` (and its sealed-holdout copy) is the deferred
MNQM26 until the 2026-03-12 roll. The same question applies to es/rty/ym/gc/si/hg/pl.

**No sealed-holdout file is opened.** Each instrument has a non-sealed twin under
data/processed/dollar_bars/1_minute/ covering the same dates (the dual presence the
ACCESS_LOG documents for MNQ), and those are what this reads.

Test: a deferred contract is THIN. Compare median volume per minute in the pre-roll window
(2026-03-01 → 03-11) with the window just after the roll. MNQ is the positive control: its
known-defective file trades 1-6 contracts a minute before the roll and ~600 after.

A per-contract price reference (data/term_structure/raw_contract_bars.csv) was tried first
and discarded: its daily closes are not comparable to these bars (the nearest-contract match
flips between months and the residuals exceed the inter-month spreads), so it cannot identify
a contract here.

Run: .venv/bin/python _bmad-output/diagnostics_march_roll_exposure_20260916/check_instrument_contracts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = Path("/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute")
ROLL = "2026-03-12"          # CME equity-index roll; metals roll near it
PRE0 = "2026-03-01"
FILES = {"mnq_control": "mnq_1min_2026_ytd.csv", "es": "es_1min_2025_2026.csv",
         "rty": "rty_1min_2025_2026.csv", "ym": "ym_1min_2025_2026.csv",
         "gc": "gc_1min_2025_2026.csv", "si": "si_1min_2025_2026.csv",
         "hg": "hg_1min_2025_2026.csv", "pl": "pl_1min_2025_2026.csv"}
THIN_RATIO = 8.0             # post/pre median volume at or above this = deferred pre-roll


def main() -> int:
    out = {}
    for name, f in FILES.items():
        p = BASE / f
        if not p.exists():
            out[name] = {"error": "file missing"}
            continue
        d = pd.read_csv(p, usecols=["timestamp", "volume"])
        d["ts"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601")

        def med(lo: str, hi: str):
            s = d[(d["ts"] >= lo) & (d["ts"] < hi)]["volume"]
            return float(s.median()) if len(s) else None

        pre, post = med(PRE0, ROLL), med(ROLL, "2026-04-01")
        rec = {"jan": med("2026-01-01", "2026-02-01"), "feb": med("2026-02-01", "2026-03-01"),
               "pre_roll": pre, "post_roll": post, "apr": med("2026-04-01", "2026-05-01")}
        if pre and post:
            rec["post_over_pre"] = round(post / pre, 1)
            rec["verdict"] = ("DEFERRED pre-roll" if post / pre >= THIN_RATIO
                              else "front-month throughout" if post / pre < 3 else "ambiguous")
        else:
            rec["verdict"] = "insufficient data"
        out[name] = rec
        print(f"{name:12} jan={rec['jan']!s:>6} feb={rec['feb']!s:>6} pre={rec['pre_roll']!s:>6} "
              f"post={rec['post_roll']!s:>6} ratio={rec.get('post_over_pre')!s:>6}  {rec['verdict']}")
    (HERE / "instrument_contract_check.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
