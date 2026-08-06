#!/usr/bin/env python3
"""Independently corroborate MIM-NB's live sigma state against broker history.

Why this exists
---------------
On 2026-08-06 the live hash-chained bar record (`data/mim_nb/bars_raw.csv`) was found
holed: 2026-08-01 and 2026-08-05 missing outright, four further sessions truncated, and
one chain break in `decisions.csv`. Cause was operational, not the bot's — the files are
git-tracked AND live-appended, so branch switches reverted them.

The response is deliberately NOT to backfill and re-chain the file. Rewriting an audit
record to look clean is the opposite of an audit record, and it would be the second false
statement written into this program's sealed history in a week.

Instead: prove the LIVE state is sound, using a source that has nothing to do with the
damaged file. `state.json` holds the sigma window the bot is actually trading on. This
tool rebuilds that window from the broker's own history plus the frozen warmup CSV, and
diffs. Agreement re-establishes the §0 provenance guarantee by corroboration; disagreement
localises the damage to specific sessions.

What it does NOT do: touch, repair, or rewrite any file. Read-only by construction.

Usage:
    .venv/bin/python tools/mim_sigma_corroborate.py [--tol 1e-9]

Exit: 0 corroborated within tolerance · 1 discrepancies found · 2 could not run.
"""
import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.auth_v3 import TradeStationAuthV3   # noqa: E402

ET = pytz.timezone("America/New_York")
BASE = Path(__file__).resolve().parent.parent
STATE = BASE / "data" / "mim_nb" / "state.json"
WARMUP = BASE / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2026_ytd.csv"
RTH_FIRST, RTH_LAST, LOOKBACK = "09:31", "16:00", 14
SYMBOL = "MNQU26"
FULL_SESSION_BARS = 390          # 09:31..16:00 inclusive


async def broker_session(http, headers, day, symbol=SYMBOL):
    """RTH bars for one ET session, straight from the broker: {hm: (open, close)}."""
    first = ET.localize(datetime(day.year, day.month, day.day, 9, 30)).astimezone(timezone.utc)
    last = ET.localize(datetime(day.year, day.month, day.day, 16, 1)).astimezone(timezone.utc)
    url = (f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
           f"?interval=1&unit=Minute"
           f"&firstdate={first.strftime('%Y-%m-%dT%H:%M:%SZ')}"
           f"&lastdate={last.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    r = await http.get(url, headers=headers)
    if r.status_code != 200:
        return {}
    out = {}
    for b in r.json().get("Bars", []):
        ts = datetime.fromisoformat(b["TimeStamp"].replace("Z", "+00:00"))
        et = ts.astimezone(ET)
        if et.date() != day:
            continue
        hm = et.strftime("%H:%M")
        if RTH_FIRST <= hm <= RTH_LAST:
            out[hm] = (float(b["Open"]), float(b["Close"]))
    return out


def warmup_session(day):
    """Same shape, from the frozen warmup CSV — used for days the broker no longer serves."""
    out = {}
    if not WARMUP.exists():
        return out
    with open(WARMUP, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                et = ts.astimezone(ET)
                if et.date() != day:
                    continue
                hm = et.strftime("%H:%M")
                if RTH_FIRST <= hm <= RTH_LAST:
                    out[hm] = (float(row["open"]), float(row["close"]))
            except (ValueError, KeyError, TypeError):
                continue
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="G1 tolerance; sigma is either deterministic or it is not")
    args = ap.parse_args()

    if not STATE.exists():
        print("state.json missing — nothing to corroborate")
        return 2
    st = json.loads(STATE.read_text())
    live_hist, live_days = st.get("sigma_hist"), st.get("sigma_days")
    if not live_hist or not live_days:
        print("state.json carries no sigma_hist/sigma_days — nothing to corroborate")
        return 2

    print("== MIM-NB sigma corroboration (independent of bars_raw.csv) ==")
    print(f"  live window: {live_days[0]}..{live_days[-1]} ({len(live_days)} sessions, "
          f"{len(live_hist)} labels)")
    print(f"  sources: broker history ({SYMBOL}) + frozen warmup CSV")
    print(f"  tolerance: {args.tol:g}\n")

    auth = TradeStationAuthV3.from_file(str(BASE / ".access_token"))
    token = await auth.authenticate()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    rebuilt, per_day_bars, sources = {}, {}, {}
    async with httpx.AsyncClient(timeout=60) as http:
        for d_str in live_days:                       # oldest -> newest, deque order
            day = datetime.strptime(d_str, "%Y-%m-%d").date()
            sess = await broker_session(http, headers, day)
            src = "broker"
            if len(sess) < FULL_SESSION_BARS:
                w = warmup_session(day)
                if len(w) > len(sess):
                    sess, src = w, "warmup"
            per_day_bars[d_str], sources[d_str] = len(sess), src
            if RTH_FIRST not in sess:
                print(f"  {d_str}: NO {RTH_FIRST} bar from {src} — cannot rebuild this session")
                continue
            o = sess[RTH_FIRST][0]
            for hm, (_o, c) in sess.items():
                rebuilt.setdefault(hm, []).append(abs(c / o - 1.0))
                rebuilt[hm] = rebuilt[hm][-LOOKBACK:]

    print("  session coverage (independent sources):")
    for d in live_days:
        n, src = per_day_bars.get(d, 0), sources.get(d, "-")
        flag = "" if n >= FULL_SESSION_BARS else "   <-- INCOMPLETE"
        print(f"    {d}  {n:3d}/{FULL_SESSION_BARS} bars via {src}{flag}")

    # --- diff -------------------------------------------------------------------
    # Each label's list is ordered oldest->newest over live_days, so index i attributes a
    # value to session live_days[i]. That turns "81% mismatch" into "these sessions".
    checked = exact = 0
    worst = 0.0
    worst_label = None
    depth_mismatch = []
    per_day = {d: {"n": 0, "exact": 0, "worst": 0.0} for d in live_days}
    for hm, live_vals in live_hist.items():
        reb = rebuilt.get(hm)
        if reb is None or len(reb) != len(live_vals):
            depth_mismatch.append((hm, len(live_vals), 0 if reb is None else len(reb)))
            continue
        for i, (a, b) in enumerate(zip(live_vals, reb)):
            checked += 1
            d = abs(float(a) - float(b))
            if d < args.tol:
                exact += 1
            if d > worst:
                worst, worst_label = d, hm
            if i < len(live_days):
                s = per_day[live_days[i]]
                s["n"] += 1
                s["exact"] += d < args.tol
                s["worst"] = max(s["worst"], d)

    print(f"\n  values compared : {checked}")
    print(f"  within {args.tol:g}     : {exact}  ({100.0 * exact / checked:.2f}%)" if checked
          else "  values compared : 0")
    print(f"  worst |diff|    : {worst:.3e}" + (f"  at label {worst_label}" if worst_label else ""))
    if depth_mismatch:
        print(f"  labels whose depth could not be rebuilt: {len(depth_mismatch)}"
              f" (e.g. {depth_mismatch[:3]})")

    print("\n  per contributing session:")
    for d in live_days:
        s = per_day[d]
        if not s["n"]:
            continue
        pct = 100.0 * s["exact"] / s["n"]
        mark = "ok" if pct == 100.0 else ("PARTIAL" if pct > 0 else "NO MATCH")
        print(f"    {d}  {s['exact']:3d}/{s['n']:3d} exact ({pct:5.1f}%)  "
              f"worst {s['worst']:.3e}   {mark}")

    ok = checked > 0 and exact == checked and not depth_mismatch
    print(f"\n  VERDICT: {'CORROBORATED' if ok else 'DISCREPANCIES FOUND'}")
    if not ok:
        print("  The live sigma state does not reproduce exactly from independent sources.")
        print("  Expected contributors: sessions whose open anchor came from a live fetch")
        print("  before prereg Amendment 2 (2026-07-29, 2026-07-31), and any session the")
        print("  broker no longer serves at full depth.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
