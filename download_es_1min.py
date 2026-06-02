"""
Download 1-min ES (E-mini S&P 500) bars for the S29 divergence analysis window.

Covers 2025-05-01 → 2026-05-19 across 5 front-month contracts, stitched at
standard roll dates (~8 business days before expiry, i.e., the Tuesday of
expiry week):

  ESM25  May  1 → Jun  9, 2025
  ESU25  Jun  9 → Sep  8, 2025
  ESZ25  Sep  8 → Dec  8, 2025
  ESH26  Dec  8, 2025 → Mar  9, 2026
  ESM26  Mar  9 → May 19, 2026

Output:
  data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv

Usage:
  .venv/bin/python download_es_1min.py
"""
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3

ES_POINT_VALUE = 50.0   # $50 per point for E-mini S&P 500
MAX_DAYS_PER_CHUNK = 30
OUT_PATH = Path("data/processed/dollar_bars/1_minute/es_1min_2025_2026.csv")

# Each entry: symbol, start (inclusive), end (exclusive) — UTC midnight
SEGMENTS = [
    {
        "symbol": "ESM25",
        "start":  datetime(2025,  5,  1, tzinfo=timezone.utc),
        "end":    datetime(2025,  6,  9, tzinfo=timezone.utc),
    },
    {
        "symbol": "ESU25",
        "start":  datetime(2025,  6,  9, tzinfo=timezone.utc),
        "end":    datetime(2025,  9,  8, tzinfo=timezone.utc),
    },
    {
        "symbol": "ESZ25",
        "start":  datetime(2025,  9,  8, tzinfo=timezone.utc),
        "end":    datetime(2025, 12,  8, tzinfo=timezone.utc),
    },
    {
        "symbol": "ESH26",
        "start":  datetime(2025, 12,  8, tzinfo=timezone.utc),
        "end":    datetime(2026,  3,  9, tzinfo=timezone.utc),
    },
    {
        "symbol": "ESM26",
        "start":  datetime(2026,  3,  9, tzinfo=timezone.utc),
        "end":    datetime(2026,  5, 20, tzinfo=timezone.utc),
    },
]


def date_chunks(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


async def download_segment(auth, client: httpx.AsyncClient, seg: dict) -> list[dict]:
    symbol = seg["symbol"]
    start, end = seg["start"], seg["end"]
    chunks = list(date_chunks(start, end, MAX_DAYS_PER_CHUNK))
    print(f"\n{symbol}  {start.date()} → {end.date()}  ({len(chunks)} chunk(s))")
    all_bars: list[dict] = []

    base_url = (
        f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
        f"?interval=1&unit=Minute"
    )

    for i, (cs, ce) in enumerate(chunks, 1):
        token = await auth.authenticate()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = (
            f"{base_url}"
            f"&firstdate={cs.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&lastdate={ce.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        print(f"  Chunk {i}/{len(chunks)}: {cs.date()} → {ce.date()} …", end=" ", flush=True)
        resp = await client.get(url, headers=headers, timeout=60.0)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} — skipping  ({resp.text[:120]})")
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
                    "open":     float(b["Open"]),
                    "high":     high,
                    "low":      low,
                    "close":    float(b["Close"]),
                    "volume":   vol,
                    "notional": max(((high + low) / 2) * vol * ES_POINT_VALUE, 0.01),
                })
            except Exception as e:
                print(f"    parse error: {e}")

    return all_bars


async def main():
    auth = TradeStationAuthV3.from_file(".access_token")
    async with httpx.AsyncClient() as client:
        all_bars: list[dict] = []
        for seg in SEGMENTS:
            bars = await download_segment(auth, client, seg)
            all_bars.extend(bars)

    if not all_bars:
        print("\nNo bars downloaded. Check token or network.")
        return

    # Deduplicate and sort
    seen: set[str] = set()
    unique: list[dict] = []
    for b in sorted(all_bars, key=lambda x: x["timestamp"]):
        if b["timestamp"] not in seen:
            seen.add(b["timestamp"])
            unique.append(b)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume", "notional"]
        )
        writer.writeheader()
        writer.writerows(unique)

    print(f"\nSaved {len(unique):,} bars → {OUT_PATH}  ({OUT_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"Range: {unique[0]['timestamp']} → {unique[-1]['timestamp']}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
