#!/usr/bin/env python3
"""Cat-stop 250-vs-500 shadow ledger — sealed mechanics recorder.

Implements §3 of `preregistration_mim_nb_catstop_shadow_ledger.md` (SEALED
2026-07-07, Design B fixed-N=10). Run offline after any session where MIM-NB's
live 250pt cat-stop fired:

    .venv/bin/python tools/mim_catstop_shadow_ledger.py --day 2026-07-08

For the given day it:
  1. Reads the live trade from data/mim_nb/trades.csv (reason must be CAT_STOP*).
  2. Broker-verifies the event (ProjectX Trade/search: the bot's type-4 stop order
     must have actually filled — the 07-06 external-flatten lesson). Unverifiable
     or externally-closed events are recorded with recovered=NA, approximate=true
     and DO NOT count toward N (sealed authenticity rule).
  3. Walks data/mim_nb/bars_raw.csv from the live entry to compute the 500-arm
     counterfactual: exit at first bar crossing entry∓500 (fill AT level, −$1,000),
     else EOD close at the 16:00 ET bar (fallback: last bar ≤240min before,
     approximate=true). Bar coverage <90% of entry→EOD window → approximate=true.
  4. Appends one hash-chained row to data/mim_nb/shadow_catstop.csv:
     date,entry_px,side,arm250_pnl,arm500_pnl,recovered,approximate,chain
     (rolling SHA-256, GENESIS seed — same scheme as the bot's ChainedCsv).

Events before the seal commit are retrospective context and must NOT be appended.
Decision rule (do not evaluate before N=10 non-approximate events — no interim
looks): revert-to-500 requires BOTH paired delta Σ(arm500−arm250) > 0 AND the
winsorized MC re-run margin ≥ 5pts; anything else confirms 250. Knowledge only.
"""
import argparse
import asyncio
import csv
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADES_CSV = ROOT / "data/mim_nb/trades.csv"
BARS_CSV = ROOT / "data/mim_nb/bars_raw.csv"
ORDERS_CSV = ROOT / "data/mim_nb/orders.csv"
LEDGER = ROOT / "data/mim_nb/shadow_catstop.csv"
FIELDS = ["date", "entry_px", "side", "arm250_pnl", "arm500_pnl", "recovered", "approximate"]
PT_VAL = 2.0
ET = "America/New_York"


def chain_head():
    head = "GENESIS"
    if LEDGER.exists():
        with open(LEDGER) as f:
            for row in csv.DictReader(f):
                head = row.get("chain", head)
    return head


def append_row(row: dict):
    head = chain_head()
    payload = "|".join(str(row.get(k, "")) for k in FIELDS)
    head = hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["chain"])
        if new:
            w.writeheader()
        w.writerow({**row, "chain": head})
    print(f"appended: {row} chain={head}")


async def broker_verified_stop(day: str) -> bool:
    """True iff a type-4 (stop) order of ours filled on `day` (ProjectX truth)."""
    from src.research.projectx_auth import ProjectXAuth
    import httpx
    auth = ProjectXAuth.from_file(str(ROOT / ".projectx_api_key"))
    tok = await auth.authenticate()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.topstepx.com/api/Order/search",
                         headers={"Authorization": f"Bearer {tok}",
                                  "Content-Type": "application/json"},
                         json={"accountId": 23884932,
                               "startTimestamp": f"{day}T00:00:00Z",
                               "endTimestamp": f"{day}T23:59:59Z"})
        for o in r.json().get("orders", []):
            if o.get("type") == 4 and o.get("fillVolume"):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="session date, e.g. 2026-07-08")
    ap.add_argument("--skip-broker-check", action="store_true",
                    help="offline mode: marks the event approximate instead of verifying")
    args = ap.parse_args()
    day = args.day

    trades = pd.read_csv(TRADES_CSV, dtype=str)
    t = trades[(trades["day"] == day) & (trades["reason"].str.startswith("CAT_STOP"))]
    if len(t) != 1:
        sys.exit(f"expected exactly one CAT_STOP row for {day}, found {len(t)}")
    t = t.iloc[0]
    side = int(t["dir"])
    entry_px = float(t["entry_px"])
    arm250 = float(t["pnl_usd"])

    if LEDGER.exists():
        with open(LEDGER) as f:
            if any(r["date"] == day for r in csv.DictReader(f)):
                sys.exit(f"{day} already in shadow ledger — refusing duplicate")

    approx = False
    if args.skip_broker_check:
        approx = True
    elif not asyncio.run(broker_verified_stop(day)):
        print("WARNING: no filled stop order at broker — external close / unverified. "
              "Recording recovered=NA, approximate=true (does NOT count toward N).")
        append_row({"date": day, "entry_px": entry_px, "side": side,
                    "arm250_pnl": arm250, "arm500_pnl": "", "recovered": "NA",
                    "approximate": True})
        return

    # 500-arm walk on recorded live bars
    bars = pd.read_csv(BARS_CSV)
    bars["ts"] = pd.to_datetime(bars["ts_utc"], utc=True, format="ISO8601")
    bars["et"] = bars["ts"].dt.tz_convert(ET)
    d0 = pd.Timestamp(day).tz_localize(ET)
    entry_et = d0 + pd.Timedelta(hours=int(t["entry_t"][:2]), minutes=int(t["entry_t"][3:5]))
    eod_et = d0 + pd.Timedelta(hours=16)
    seg = bars[(bars["et"] > entry_et) & (bars["et"] <= eod_et)].sort_values("et")
    expected_bars = int((eod_et - entry_et).total_seconds() // 60)
    if expected_bars and len(seg) / expected_bars < 0.90:
        approx = True

    stop500 = entry_px - 500.0 if side == 1 else entry_px + 500.0
    arm500 = None
    for _, b in seg.iterrows():
        if side == 1 and b["low"] <= stop500:
            arm500 = -500.0 * PT_VAL
            break
        if side == -1 and b["high"] >= stop500:
            arm500 = -500.0 * PT_VAL
            break
    if arm500 is None:
        eod_seg = seg[seg["et"] >= eod_et - pd.Timedelta(minutes=240)]
        if eod_seg.empty:
            sys.exit("no bars near EOD — cannot compute 500-arm; rerun when bars_raw is complete")
        close_px = float(eod_seg.iloc[-1]["close"])
        if eod_seg.iloc[-1]["et"] < eod_et - pd.Timedelta(minutes=5):
            approx = True
        arm500 = (close_px - entry_px) * PT_VAL * (1 if side == 1 else -1)

    recovered = arm500 > arm250
    append_row({"date": day, "entry_px": entry_px, "side": side,
                "arm250_pnl": arm250, "arm500_pnl": round(arm500, 2),
                "recovered": recovered, "approximate": approx})

    with open(LEDGER) as f:
        rows = [r for r in csv.DictReader(f) if r["recovered"] in ("True", "False")
                and r["approximate"] == "False"]
    print(f"non-approximate events: {len(rows)}/10 (no interim evaluation before N=10)")


if __name__ == "__main__":
    main()
