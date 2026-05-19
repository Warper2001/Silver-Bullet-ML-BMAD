"""
Download 1-min MNQM26 bars for Jan 1, 2026 → May 19, 2026 from TradeStation.

Saves to: data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv
Same column format as mnq_1min_2025.csv: timestamp, open, high, low, close, volume, notional

Usage:
  .venv/bin/python download_mnq_2026_ytd.py
"""
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3

SYMBOL = "MNQM26"
MNQ_POINT_VALUE = 20.0
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 5, 19, 23, 59, 59, tzinfo=timezone.utc)
MAX_DAYS_PER_CHUNK = 35  # API caps at 57,600 bars ≈ 40 calendar days; use 35 to stay safe
OUT_CSV = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

BASE_URL = (
    f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
    f"?interval=1&unit=Minute"
)


def date_chunks(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


async def main():
    auth = TradeStationAuthV3.from_file(".access_token")
    all_bars: list[dict] = []

    chunks = list(date_chunks(START, END, MAX_DAYS_PER_CHUNK))
    print(f"Downloading {SYMBOL} 1-min bars {START.date()} → {END.date()} in {len(chunks)} chunk(s)…")

    async with httpx.AsyncClient(timeout=60) as client:
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            token = await auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            url = (
                f"{BASE_URL}"
                f"&firstdate={chunk_start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                f"&lastdate={chunk_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )

            print(f"  Chunk {i}/{len(chunks)}: {chunk_start.date()} → {chunk_end.date()} …", end=" ", flush=True)
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
        print("No bars downloaded — check token and symbol.")
        return

    # Deduplicate and sort
    seen = set()
    unique = []
    for b in sorted(all_bars, key=lambda x: x["timestamp"]):
        if b["timestamp"] not in seen:
            seen.add(b["timestamp"])
            unique.append(b)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume", "notional"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"\nSaved {len(unique):,} bars → {OUT_CSV}  ({OUT_CSV.stat().st_size / 1024:.0f} KB)")
    print(f"Date range: {unique[0]['timestamp']} → {unique[-1]['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
