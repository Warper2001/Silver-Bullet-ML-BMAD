"""Which saved MNQ backtest results span the March 2026 roll, and how exposed are they?

The defect (established in diagnostics_gap_fade_gate0_rescore_20260916): in
`mnq_1min_2026_ytd.csv` and its sealed-holdout copy, the bars up to the 2026-03-12 CME roll
are the DEFERRED MNQM26 while MNQH26 was still the front month. Any MNQ backtest whose
window covers 2026-03-01 → 03-11 priced those days off the wrong contract, and any gap or
level taken across the switch also folds in the calendar spread. In GAP-1's Gate-0 that
window supplied +$1,888 of $9,878 — a fifth of the net — and vanished on corrected bars.

This scans saved trade lists, keeps MNQ strategy results, and ranks them by how much of
their net sits in that window. Read-only: it opens result CSVs only — no bars, no holdout
data files, no replays.

Run: .venv/bin/python _bmad-output/diagnostics_march_roll_exposure_20260916/scan_pre_roll_exposure.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
PRE0, PRE1 = "2026-03-01", "2026-03-12"          # deferred-contract window
DATE_COLS = ("date", "day", "entry_time", "entry_t", "timestamp", "open_time")
PNL_COLS = ("pnl_usd", "pnl", "pnl_pts", "profit", "net")
# not MNQ, or not a strategy result: the defect cannot apply
SKIP = re.compile(r"btc|xbt|eth|kraken|crypto|carry|funding|macro/|term_structure|"
                  r"sealed_holdout/|dollar_bars/|/bars/|equity_curve|econ_calendar|policy_shock|"
                  r"sessions\.csv|MVRV|M2SL|DTWEXBGS", re.I)


def main() -> int:
    seen, rows = set(), []
    pats = ["data/reports/**/*.csv", "_bmad-output/**/*.csv", "data/ml_training/**/*.csv",
            "data/*/**/*.csv"]
    for pat in pats:
        for p in sorted(glob.glob(str(MAIN / pat), recursive=True)):
            rel = os.path.relpath(p, MAIN)
            real = os.path.realpath(p)
            if real in seen or SKIP.search(rel) or os.path.getsize(p) > 40_000_000:
                continue
            seen.add(real)
            try:
                with open(p, newline="") as f:
                    rd = csv.DictReader(f)
                    if not rd.fieldnames:
                        continue
                    dcol = next((c for c in rd.fieldnames if c.lower() in DATE_COLS), None)
                    pcol = next((c for c in rd.fieldnames if c.lower() in PNL_COLS), None)
                    if not dcol or not pcol:
                        continue
                    recs = list(rd)
            except Exception:
                continue
            if not recs or len(recs) > 200_000:
                continue
            days, pnls = [], []
            for r in recs:
                m = re.match(r"(\d{4}-\d{2}-\d{2})", str(r.get(dcol) or ""))
                try:
                    v = float(r.get(pcol))
                except (TypeError, ValueError):
                    continue
                if m:
                    days.append(m.group(1))
                    pnls.append(v)
            if not days or max(days) < PRE0:
                continue
            pre = [(d, v) for d, v in zip(days, pnls) if PRE0 <= d < PRE1]
            if not pre:
                continue
            total = sum(pnls)
            pre_pnl = sum(v for _, v in pre)
            rows.append({"file": rel, "N": len(pnls), "pre_roll_N": len(pre),
                         "total": round(total, 2), "pre_roll_pnl": round(pre_pnl, 2),
                         "share_of_net": (round(pre_pnl / total, 3) if total else None),
                         "span": [min(days), max(days)]})
    rows.sort(key=lambda r: -abs(r["share_of_net"] or 0))
    (HERE / "pre_roll_exposure.json").write_text(json.dumps(rows, indent=2))
    print(f"{'file':66} {'N':>5} {'pre':>4} {'total':>9} {'pre P&L':>9} {'share':>7}  span")
    for r in rows:
        s = "n/a" if r["share_of_net"] is None else f"{r['share_of_net']:+.1%}"
        print(f"{r['file'][-66:]:66} {r['N']:>5} {r['pre_roll_N']:>4} {r['total']:>9,.0f} "
              f"{r['pre_roll_pnl']:>9,.0f} {s:>7}  {r['span'][0]}..{r['span'][1]}")
    print(f"\n{len(rows)} MNQ result files span the pre-roll window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
