"""Derive HG/SI front-month roll dates from daily volume, instead of assuming a convention.

Why this exists: copper and silver do NOT roll on the equity-index schedule. A probe on
2026-03-02 found HGH26 trading 2 contracts a minute while HGK26 traded 19, and SIH26 1
against SIK26's 32 — COMEX metals enter their notice period at the end of the month BEFORE
delivery, so liquidity leaves the delivery month well before the equity-index roll date.
Assuming the mid-March roll would have rebuilt the same defect the refetch is meant to fix.

Method: one Daily barchart request per candidate contract, then for each date the front month
is whichever contract has the highest volume. A roll date is the first date the successor
leads and keeps leading (a 3-session confirmation avoids single-day flickers).

Writes the calendar to metals_roll_calendar.json. Reads nothing local; makes ~16 API calls.

Usage: .venv/bin/python build_metals_roll_calendar.py
"""
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

OUT = Path("metals_roll_calendar.json")
START, END = "2025-04-01", "2026-06-12"
CONFIRM = 3          # sessions the successor must lead before the roll is called
CYCLE = ["H25", "K25", "N25", "U25", "Z25", "H26", "K26", "N26"]
ROOTS = {"hg": "HG", "si": "SI"}


def utc(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)


async def daily_volumes(auth, client, symbol: str) -> dict:
    token = await auth.authenticate()
    url = (f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
           f"?interval=1&unit=Daily"
           f"&firstdate={utc(START):%Y-%m-%dT%H:%M:%SZ}&lastdate={utc(END):%Y-%m-%dT%H:%M:%SZ}")
    r = await client.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60.0)
    if r.status_code != 200:
        print(f"  {symbol}: HTTP {r.status_code}")
        return {}
    out = {}
    for b in r.json().get("Bars", []):
        try:
            out[b["TimeStamp"][:10]] = int(b.get("TotalVolume", 0))
        except Exception:
            continue
    print(f"  {symbol}: {len(out)} daily bars, peak volume {max(out.values(), default=0):,}")
    return out


def front_month_series(vols: dict) -> dict:
    by_date = defaultdict(dict)
    for sym, series in vols.items():
        for d, v in series.items():
            by_date[d][sym] = v
    return {d: max(m.items(), key=lambda kv: kv[1])[0] for d, m in sorted(by_date.items()) if m}


def segments_from(front: dict) -> list[dict]:
    dates = sorted(front)
    segs, cur, start = [], None, None
    for i, d in enumerate(dates):
        sym = front[d]
        if sym == cur:
            continue
        window = [front[x] for x in dates[i:i + CONFIRM]]
        if len(window) == CONFIRM and len(set(window)) > 1:
            continue                       # flicker, not a roll
        if cur is not None:
            segs.append({"symbol": cur, "start": start, "end": d})
        cur, start = sym, d
    if cur is not None:
        segs.append({"symbol": cur, "start": start, "end": END})
    return segs


async def main() -> int:
    auth = TradeStationAuthV3.from_file(".access_token")
    calendar = {}
    async with httpx.AsyncClient() as client:
        for key, root in ROOTS.items():
            print(f"\n{key}: fetching daily volume per contract")
            vols = {}
            for code in CYCLE:
                vols[f"{root}{code}"] = await daily_volumes(auth, client, f"{root}{code}")
                await asyncio.sleep(1.0)
            vols = {k: v for k, v in vols.items() if v}
            front = front_month_series(vols)
            segs = segments_from(front)
            calendar[key] = {"segments": segs,
                             "first_date": min(front) if front else None,
                             "last_date": max(front) if front else None}
            print(f"  front-month segments for {key}:")
            for s in segs:
                print(f"    {s['symbol']}  {s['start']} → {s['end']}")
    OUT.write_text(json.dumps(calendar, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
