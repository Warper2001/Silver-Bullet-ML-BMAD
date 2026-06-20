#!/usr/bin/env python3
"""
Download Kraken PF_ETHUSD historical funding rate proxy.

Since Kraken's /historicalfundingrates endpoint is unavailable publicly,
we derive the funding rate from the mark/spot price basis — which is the
direct mathematical input to Kraken's funding rate calculation:

  funding_rate_per_8h ≈ mean(mark_close / spot_close - 1)  over each 8h window
  annualized           ≈ funding_rate_per_8h × 3 × 365

This is a high-fidelity proxy because:
- Kraken computes funding = f(mark_premium, interest_rate)
- mark_premium = (mark - spot) / spot is the dominant term
- The proxy correlates directly with actual paid/received funding

Output: data/kraken/PF_ETHUSD_funding_rate.csv
Schema: timestamp (UTC ISO-8601, 8h cadence), funding_rate (per-8h decimal)

Usage: .venv/bin/python download_kraken_btc_funding_rates.py
"""
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

SYMBOL      = "PF_ETHUSD"
CHARTS_BASE = "https://futures.kraken.com/api/charts/v1"
OUTPUT      = Path("data/kraken/PF_ETHUSD_funding_rate.csv")

# Funding rate is published every 8 hours at 00:00, 08:00, 16:00 UTC
FUNDING_INTERVAL_H = 8
# Fetch 1h bars; aggregate into 8h buckets
BAR_RESOLUTION = "1h"

# Date range matching the backtest window + some buffer
START_UTC = datetime(2024, 11, 1, tzinfo=timezone.utc)
END_UTC   = datetime.now(timezone.utc) + timedelta(days=1)


async def fetch_chart(client: httpx.AsyncClient, chart_type: str,
                      from_ts: int, to_ts: int) -> list[dict]:
    """Fetch OHLCV candles for a given chart type and time window."""
    url = f"{CHARTS_BASE}/{chart_type}/{SYMBOL}/{BAR_RESOLUTION}"
    params = {"from": str(from_ts), "to": str(to_ts)}
    resp = await client.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json().get("candles", [])


async def fetch_all_candles(client: httpx.AsyncClient, chart_type: str) -> list[dict]:
    """Page through the full date range in 60-day chunks."""
    all_candles: list[dict] = []
    cursor = START_UTC
    chunk = timedelta(days=60)

    while cursor < END_UTC:
        end_chunk = min(cursor + chunk, END_UTC)
        from_ts = int(cursor.timestamp())
        to_ts   = int(end_chunk.timestamp())

        try:
            candles = await fetch_chart(client, chart_type, from_ts, to_ts)
            all_candles.extend(candles)
            print(f"  {chart_type}: {cursor.date()} → {end_chunk.date()}: "
                  f"+{len(candles)} candles (total {len(all_candles)})")
        except Exception as e:
            print(f"  WARNING: {chart_type} fetch failed {cursor.date()}→{end_chunk.date()}: {e}")

        cursor = end_chunk

    # Deduplicate by timestamp, sort
    seen: set[int] = set()
    deduped = []
    for c in all_candles:
        t = int(c["time"])
        if t not in seen:
            seen.add(t)
            deduped.append(c)
    deduped.sort(key=lambda c: c["time"])
    return deduped


def compute_funding_rates(mark_candles: list[dict], spot_candles: list[dict]) -> list[tuple[str, float]]:
    """
    Compute 8h average basis from 1h mark/spot candles.

    For each 8h funding window (00:00, 08:00, 16:00 UTC), average the
    hourly basis values: (mark_close / spot_close - 1).
    """
    mark_by_time = {c["time"]: float(c["close"]) for c in mark_candles if c.get("close")}
    spot_by_time = {c["time"]: float(c["close"]) for c in spot_candles if c.get("close")}

    common_times = sorted(set(mark_by_time.keys()) & set(spot_by_time.keys()))
    if not common_times:
        print("ERROR: no common timestamps between mark and spot data")
        return []

    # Group into 8h buckets
    rows: list[tuple[str, float]] = []
    i = 0
    while i < len(common_times):
        # Find the 8h window start: snap to nearest 00:00/08:00/16:00 UTC
        t_ms = common_times[i]
        dt = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
        # Round down to 8h boundary
        hour_bucket = (dt.hour // 8) * 8
        window_start = dt.replace(hour=hour_bucket, minute=0, second=0, microsecond=0)
        window_end   = window_start + timedelta(hours=8)
        window_start_ms = int(window_start.timestamp() * 1000)
        window_end_ms   = int(window_end.timestamp() * 1000)

        # Collect all 1h bars in this window
        bases = []
        j = i
        while j < len(common_times) and common_times[j] < window_end_ms:
            t = common_times[j]
            if t >= window_start_ms:
                mark_p = mark_by_time[t]
                spot_p = spot_by_time[t]
                if spot_p > 0:
                    bases.append((mark_p - spot_p) / spot_p)
            j += 1

        if bases:
            avg_basis = sum(bases) / len(bases)
            rows.append((window_start.isoformat(), avg_basis))

        # Advance to next 8h window
        next_window_ms = window_end_ms
        i = j
        # Skip to bars in the next window
        while i < len(common_times) and common_times[i] < next_window_ms:
            i += 1

    return rows


async def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching 1h mark and spot prices for {SYMBOL}...")
    print(f"Date range: {START_UTC.date()} → {END_UTC.date()}\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Fetching mark price...")
        mark_candles = await fetch_all_candles(client, "mark")
        print(f"\nFetching spot price...")
        spot_candles = await fetch_all_candles(client, "spot")

    print(f"\nMark candles: {len(mark_candles)}")
    print(f"Spot candles: {len(spot_candles)}")

    print("\nComputing 8h basis as funding rate proxy...")
    funding_rows = compute_funding_rates(mark_candles, spot_candles)

    if not funding_rows:
        print("ERROR: Could not compute funding rates. Check mark/spot data.")
        sys.exit(1)

    with OUTPUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "funding_rate"])
        w.writerows(funding_rows)

    print(f"\nWrote {len(funding_rows)} 8h rows to {OUTPUT}")

    # Stats
    rates = [r for _, r in funding_rows]
    print(f"Date range:  {funding_rows[0][0][:10]} → {funding_rows[-1][0][:10]}")
    print(f"Rate range:  {min(rates):.6f} to {max(rates):.6f} per 8h")
    print(f"Mean rate:   {sum(rates)/len(rates):.6f} per 8h  "
          f"({sum(rates)/len(rates)*3*365*100:.2f}% annualized)")

    # Show distribution relative to thresholds
    short_thresh = 0.0003   # > +0.03% per 8h → SHORT bias
    long_thresh  = -0.0002  # < -0.02% per 8h → LONG bias
    n_short = sum(1 for r in rates if r > short_thresh)
    n_long  = sum(1 for r in rates if r < long_thresh)
    n_neutral = len(rates) - n_short - n_long
    print(f"\nAt thresholds short>{short_thresh*100:.3f}% / long<{long_thresh*100:.3f}%:")
    print(f"  SHORT bias:   {n_short:>4} rows ({100*n_short/len(rates):.1f}%)")
    print(f"  LONG  bias:   {n_long:>4} rows ({100*n_long/len(rates):.1f}%)")
    print(f"  NEUTRAL:      {n_neutral:>4} rows ({100*n_neutral/len(rates):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
