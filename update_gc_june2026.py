"""
Incremental update: append GCM26 bars from 2026-05-20 → 2026-06-12 to gc_1min_2025_2026.csv.
Covers the June 11 CPI event + 2h buffer. Deduplicates on merge.
"""
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "root/Silver-Bullet-ML-BMAD"))
import importlib.util, os
os.chdir("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")

from src.data.auth_v3 import TradeStationAuthV3
import httpx

OUT_PATH = Path("data/processed/dollar_bars/1_minute/gc_1min_2025_2026.csv")
GC_POINT_VALUE   = 100.0
MAX_DAYS_PER_CHUNK = 15

SEGMENT = {
    "symbol": "GCM26",
    "start":  datetime(2026, 5, 20, tzinfo=timezone.utc),
    "end":    datetime(2026, 6, 12, tzinfo=timezone.utc),
}


def date_chunks(start, end, days):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


async def main():
    auth = TradeStationAuthV3.from_file(".access_token")
    symbol = SEGMENT["symbol"]
    start, end = SEGMENT["start"], SEGMENT["end"]
    chunks = list(date_chunks(start, end, MAX_DAYS_PER_CHUNK))
    print(f"{symbol}  {start.date()} → {end.date()}  ({len(chunks)} chunk(s))")

    new_bars = []
    base_url = (
        f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
        f"?interval=1&unit=Minute"
    )

    async with httpx.AsyncClient() as client:
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
                print(f"HTTP {resp.status_code} — skipping")
                continue
            bars = resp.json().get("Bars", [])
            print(f"{len(bars):,} bars")
            for b in bars:
                try:
                    high = float(b["High"]); low = float(b["Low"])
                    vol  = int(b.get("TotalVolume", 0))
                    ts   = b["TimeStamp"].replace("Z", "+00:00")
                    new_bars.append({
                        "timestamp": ts,
                        "open": float(b["Open"]), "high": high,
                        "low": low, "close": float(b["Close"]),
                        "volume": vol,
                        "notional": max(((high+low)/2) * vol * GC_POINT_VALUE, 0.01),
                    })
                except Exception as e:
                    print(f"    parse error: {e}")

    if not new_bars:
        print("No new bars downloaded.")
        return

    print(f"\nFetched {len(new_bars):,} new bars. Merging with existing file…")

    # Load existing timestamps for dedup
    existing_ts = set()
    existing_rows = []
    with open(OUT_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ts.add(row["timestamp"])
            existing_rows.append(row)

    # Append only genuinely new rows
    added = 0
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume", "notional"]
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for b in sorted(new_bars, key=lambda x: x["timestamp"]):
            if b["timestamp"] not in existing_ts:
                writer.writerow(b)
                existing_ts.add(b["timestamp"])
                added += 1

    print(f"Added {added:,} new bars → {OUT_PATH}")
    print(f"Total rows now: {len(existing_rows) + added:,}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
