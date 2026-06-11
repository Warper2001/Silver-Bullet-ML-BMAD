"""
Download 1-min MNQ bars for Sep–Nov 2023 and Sep–Nov 2024 from TradeStation.

Uses December quarterly contract for each year (front-month for Oct/Nov):
  MNQZ23 → Sep–Nov 2023
  MNQZ24 → Sep–Nov 2024

Saves to:
  data/processed/dollar_bars/1_minute/mnq_1min_2023_sepnov.csv
  data/processed/dollar_bars/1_minute/mnq_1min_2024_sepnov.csv

Usage:
  .venv/bin/python download_mnq_octnov.py
"""
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3

MNQ_POINT_VALUE = 20.0
MAX_DAYS_PER_CHUNK = 35
OUT_DIR = Path("data/processed/dollar_bars/1_minute")

DOWNLOADS = [
    {
        "symbol": "MNQZ23",
        "start":  datetime(2023, 9, 1, tzinfo=timezone.utc),
        "end":    datetime(2023, 11, 30, 23, 59, 59, tzinfo=timezone.utc),
        "out":    OUT_DIR / "mnq_1min_2023_sepnov.csv",
    },
    {
        "symbol": "MNQZ24",
        "start":  datetime(2024, 9, 1, tzinfo=timezone.utc),
        "end":    datetime(2024, 11, 30, 23, 59, 59, tzinfo=timezone.utc),
        "out":    OUT_DIR / "mnq_1min_2024_sepnov.csv",
    },
]


def date_chunks(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


async def download_one(auth, client, cfg: dict) -> int:
    symbol = cfg["symbol"]
    start, end = cfg["start"], cfg["end"]
    out = cfg["out"]
    base_url = (
        f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
        f"?interval=1&unit=Minute"
    )

    chunks = list(date_chunks(start, end, MAX_DAYS_PER_CHUNK))
    print(f"\n{symbol}  {start.date()} → {end.date()}  ({len(chunks)} chunk(s))")
    all_bars: list[dict] = []

    for i, (cs, ce) in enumerate(chunks, 1):
        token = await auth.authenticate()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = (
            f"{base_url}"
            f"&firstdate={cs.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&lastdate={ce.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        print(f"  Chunk {i}/{len(chunks)}: {cs.date()} → {ce.date()} …", end=" ", flush=True)
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} — skipping")
            continue
        bars = resp.json().get("Bars", [])
        print(f"{len(bars):,} bars")
        for b in bars:
            try:
                high = float(b["High"])
                low  = float(b["Low"])
                vol  = int(b.get("TotalVolume", 0))
                ts   = b["TimeStamp"].replace("Z", "+00:00")
                all_bars.append({
                    "timestamp": ts,
                    "open":  float(b["Open"]),
                    "high":  high,
                    "low":   low,
                    "close": float(b["Close"]),
                    "volume": vol,
                    "notional": max(((high + low) / 2) * vol * MNQ_POINT_VALUE, 0.01),
                })
            except Exception as e:
                print(f"    parse error: {e}")

    if not all_bars:
        print(f"  No bars downloaded for {symbol}")
        return 0

    seen = set()
    unique = []
    for b in sorted(all_bars, key=lambda x: x["timestamp"]):
        if b["timestamp"] not in seen:
            seen.add(b["timestamp"])
            unique.append(b)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","open","high","low","close","volume","notional"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"  Saved {len(unique):,} bars → {out}  ({out.stat().st_size/1024:.0f} KB)")
    print(f"  Range: {unique[0]['timestamp']} → {unique[-1]['timestamp']}")
    return len(unique)


async def main():
    auth = TradeStationAuthV3.from_file(".access_token")
    async with httpx.AsyncClient(timeout=60) as client:
        for cfg in DOWNLOADS:
            await download_one(auth, client, cfg)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
