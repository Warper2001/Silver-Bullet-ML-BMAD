"""
Download 1-min GC (COMEX gold futures) bars for the GC post-catalyst study.

Covers 2025-05-01 → 2026-05-20 across 7 front-month contracts, stitched at
standard roll dates (~8 business days before contract expiry):

  GCM25  May  1 → Jun  9, 2025   (Jun 2025 delivery)
  GCQ25  Jun  9 → Aug 11, 2025   (Aug 2025 delivery)
  GCV25  Aug 11 → Oct 13, 2025   (Oct 2025 delivery)
  GCZ25  Oct 13 → Dec  8, 2025   (Dec 2025 delivery)
  GCG26  Dec  8 → Feb  9, 2026   (Feb 2026 delivery)
  GCJ26  Feb  9 → Apr  7, 2026   (Apr 2026 delivery)
  GCM26  Apr  7 → May 20, 2026   (Jun 2026 delivery)

Gold futures month codes: G=Feb J=Apr M=Jun Q=Aug V=Oct Z=Dec
Expiry: 3rd-to-last business day of delivery month; standard roll ≈8 biz days prior.

Output:
  data/processed/dollar_bars/1_minute/gc_1min_2025_2026.csv

ECONOMICS NOTE (for the eventual GC post-catalyst study):
  This script downloads full GC (100 troy oz/contract, $100/point) but the
  study will use MGC economics (Micro Gold, 10 oz/contract, $10/point).
  GC bars are the signal source; MGC sizing is applied at simulation time.

Usage:
  nohup .venv/bin/python download_gc_1min.py > /tmp/gc_fetch.log 2>&1 &
  tail -f /tmp/gc_fetch.log
"""
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3

GC_POINT_VALUE   = 100.0    # $100 per point for 1 GC contract (100 troy oz)
MAX_DAYS_PER_CHUNK = 30
OUT_PATH = Path("data/processed/dollar_bars/1_minute/gc_1min_2025_2026.csv")

# Each entry: symbol, start (inclusive), end (exclusive) — UTC midnight
# Roll dates: ~8 business days before contract expiry
# GC expiry: 3rd-to-last business day of delivery month
# Approximate roll:
#   GCM25 expires Jun 25 → roll Jun 9
#   GCQ25 expires Aug 27 → roll Aug 11
#   GCV25 expires Oct 28 → roll Oct 13
#   GCZ25 expires Dec 29 → roll Dec 8
#   GCG26 expires Feb 25 → roll Feb 9
#   GCJ26 expires Apr 28 → roll Apr 7
SEGMENTS = [
    {
        "symbol": "GCM25",
        "start":  datetime(2025,  5,  1, tzinfo=timezone.utc),
        "end":    datetime(2025,  6,  9, tzinfo=timezone.utc),
    },
    {
        "symbol": "GCQ25",
        "start":  datetime(2025,  6,  9, tzinfo=timezone.utc),
        "end":    datetime(2025,  8, 11, tzinfo=timezone.utc),
    },
    {
        "symbol": "GCV25",
        "start":  datetime(2025,  8, 11, tzinfo=timezone.utc),
        "end":    datetime(2025, 10, 13, tzinfo=timezone.utc),
    },
    {
        "symbol": "GCZ25",
        "start":  datetime(2025, 10, 13, tzinfo=timezone.utc),
        "end":    datetime(2025, 12,  8, tzinfo=timezone.utc),
    },
    {
        "symbol": "GCG26",
        "start":  datetime(2025, 12,  8, tzinfo=timezone.utc),
        "end":    datetime(2026,  2,  9, tzinfo=timezone.utc),
    },
    {
        "symbol": "GCJ26",
        "start":  datetime(2026,  2,  9, tzinfo=timezone.utc),
        "end":    datetime(2026,  4,  7, tzinfo=timezone.utc),
    },
    {
        "symbol": "GCM26",
        "start":  datetime(2026,  4,  7, tzinfo=timezone.utc),
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
                    # notional uses GC point value (100 oz × $1/oz = $100/point)
                    "notional": max(((high + low) / 2) * vol * GC_POINT_VALUE, 0.01),
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
    print(f"\nMGC economics note: This is GC data (100 oz/contract, $100/pt).")
    print(f"  When simulating, use MGC_PV=10.0 (10 oz/contract, $10/pt) for combine sizing.")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
