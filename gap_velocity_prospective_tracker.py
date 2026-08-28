"""
GAP-V Prospective Phase — accrual tracker (OBSERVATION ONLY)

Pre-registration: _bmad-output/preregistration_gap_velocity_conditioned.md
  seal e5b40f6 | A1 (VELOCITY_SPLIT) 5f08c5d | A2 (INSUFFICIENT_SAMPLE) 1baf5cf
  Prospective phase authorised by Amendment 3.
  Decision rule fixed by GAP-V2: _bmad-output/preregistration_gap_v2_successor.md

GAP-V2 replaced the split-point design with a Spearman rank correlation over ALL
accrued trades -- no threshold, no subgroups, no floor. Two looks: INTERIM at N=24
(may only PASS; a null there is INCONCLUSIVE, see GAP-V2 §4) and FINAL at N=47.
Stopping date 2027-12-31.

WHY THIS SCRIPT REPORTS ALMOST NOTHING
--------------------------------------
Amendment 2 (§A2.2) established that GAP-V's retrospective test failed because a
threshold derived on one era (2025, a tariff-volatility regime) landed at the 83rd
percentile of the test era rather than near its middle. An absolute percentage
carried across regimes inherits the derivation era's volatility.

Applying the frozen 0.9421% split to forward data would repeat that defect exactly.
Seal §8.1 also forbids adjusting VELOCITY_SPLIT now that outcomes have been seen.

The resolution is NOT to pick a split here. This tracker is a pure observation
instrument: it accrues the raw facts (date, gap_pct, outcome, P&L) and reports ONLY
the sample count and progress to target. It deliberately reports NO subgroup means,
NO profit factor and NO verdict, because watching those accrue is how a decision
rule gets chosen to fit the data it will be tested on.

The classification rule and decision rule MUST be fixed in a successor seal, written
BEFORE N reaches target. Per §A2.2 that rule should be distribution-relative
(e.g. the median of the prospective window's own qualifying gaps), not an absolute
percentage carried forward.

Idempotent: natural key = entry timestamp, matching the trades.db fix (da78f5a).
Reads data/trades.db read-only. Places no orders. Modifies no GAP-1 state.

Usage:
    .venv/bin/python gap_velocity_prospective_tracker.py [--db data/trades.db]
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

# Prospective window opens at the Amendment 2 commit: the moment ALL retrospective
# outcomes had been seen. Nothing before this instant may enter the ledger.
SEAL_COMMIT = "1baf5cf"
WINDOW_OPEN = "2026-08-27T22:05:19+00:00"

N_INTERIM = 24           # GAP-V2 §4 interim look -- PASS only, never a stop
N_FINAL = 47             # GAP-V2 §4 final look (80% power for rho=0.40)
STOP_DATE = "2027-12-31" # GAP-V2 §7
LEDGER = Path("data/gap_velocity/prospective_trades.csv")
FIELDS = ["entry_ts", "date_et", "gap_pct", "gap_abs_pts", "entry_price",
          "exit_price", "exit_reason", "pnl_usd"]


def load_ledger() -> dict:
    if not LEDGER.exists():
        return {}
    with LEDGER.open() as f:
        return {r["entry_ts"]: r for r in csv.DictReader(f)}


def write_ledger(rows: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


def fetch_candidates(db: Path) -> list[dict]:
    """GAP-1 LONG trades (= gap-down fades) entered on/after the window opens."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "select timestamp, entry_price, exit_price, exit_reason, pnl, metadata "
            "from trades where trader_id='trader-gap-fade' and direction='L' "
            "and timestamp >= ? order by timestamp",
            (WINDOW_OPEN,),
        )
        out = []
        for ts, ep, xp, er, pnl, md in cur:
            m = json.loads(md) if md else {}
            out.append({
                "entry_ts": ts,
                "date_et": ts[:10],
                "gap_pct": m.get("gap_pct", ""),
                "gap_abs_pts": m.get("gap_abs_pts", ""),
                "entry_price": ep,
                "exit_price": xp,
                "exit_reason": er,
                "pnl_usd": pnl,
            })
        return out
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/trades.db")
    args = ap.parse_args()

    ledger = load_ledger()
    before = len(ledger)
    added = 0
    for row in fetch_candidates(Path(args.db)):
        if row["entry_ts"] not in ledger:      # natural-key idempotency
            ledger[row["entry_ts"]] = row
            added += 1
    write_ledger(ledger)
    n = len(ledger)

    opened = datetime.fromisoformat(WINDOW_OPEN)
    days = (datetime.now(timezone.utc) - opened).days

    print("=" * 78)
    print("GAP-V PROSPECTIVE ACCRUAL — observation only, no verdict")
    print("=" * 78)
    print(f"  seal            : {SEAL_COMMIT} (Amendment 2)")
    print(f"  window opened   : {WINDOW_OPEN}  ({days} days ago)")
    print(f"  ledger          : {LEDGER}")
    print(f"  qualifying longs: {n}   (+{added} this run, was {before})")
    rate = n / days if days > 0 else 0
    def eta(target):
        return f"~{(target - n) / rate:.0f} more days" if rate > 0 else "rate not yet estimable"
    print(f"  interim look    : N={N_INTERIM}   final look: N={N_FINAL}   stop: {STOP_DATE}")
    if n < N_INTERIM:
        print(f"  progress        : {n}/{N_INTERIM} to interim  ({eta(N_INTERIM)})")
    elif n < N_FINAL:
        print(f"  progress        : {n}  INTERIM LOOK REACHED (N>={N_INTERIM})")
        print(f"                    {n}/{N_FINAL} to final  ({eta(N_FINAL)})")
        print()
        print("  >>> INTERIM LOOK IS DUE. Run it per GAP-V2 §4/§5. <<<")
        print("  It may only PASS. A non-significant result is INCONCLUSIVE, not a stop,")
        print("  and accrual CONTINUES to the final look at N=47.")
    else:
        print(f"  progress        : {n}/{N_FINAL}  FINAL LOOK REACHED")
        print()
        print("  >>> FINAL LOOK IS DUE. Run it per GAP-V2 §4/§5. <<<")
        print("  Primary test: one-sided Spearman rho(gap_pct, pnl_usd), alpha=0.025.")
    print()
    print("  No subgroup statistics are reported by design (see module docstring).")
    print("  GAP-1 is not modified by this tracker. trades.db opened read-only.")


if __name__ == "__main__":
    main()
