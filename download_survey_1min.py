"""
Download 1-min bars for the cross-pair divergence-fade survey (2026-06).

One script, five instrument roots (pass --root). Clone of the proven
download_gc_1min.py logic: TradeStation v3 barcharts REST, 30-day chunks,
front-month contracts stitched at standard roll dates (~8 business days
before contract expiry).

Roots and contract calendars:

  SI   COMEX silver   (5,000 oz, $5,000/pt)  months H K N U Z
  HG   COMEX copper   (25,000 lb, $25,000/pt) months H K N U Z
  PL   NYMEX platinum (50 oz, $50/pt)         months F J N V
  RTY  CME e-mini Russell 2000 ($50/pt)       quarterly H M U Z
  YM   CBOT e-mini Dow ($5/pt)                quarterly H M U Z

Metals expiry: 3rd-to-last business day of delivery month; roll ≈8 biz days
prior (same convention as download_gc_1min.py). Index quarterlies use the
exact ES roll dates from download_es_1min.py, extended with the M26→U26 roll
on 2026-06-08.

ECONOMICS NOTE: these are full-size contracts (signal source). Simulation
applies MICRO economics at sim time (SIL $1,000/pt, MHG $2,500/pt, M2K $5/pt,
MYM $0.50/pt; PL has no micro). Same pattern as GC bars + MGC sizing.

Output: data/processed/dollar_bars/1_minute/{root}_1min_2025_2026.csv

Usage:
  .venv/bin/python download_survey_1min.py --root SI --smoke   # 2-day probe per segment
  nohup .venv/bin/python download_survey_1min.py --root SI > logs/si_fetch.log 2>&1 &
"""
import argparse
import asyncio
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3

MAX_DAYS_PER_CHUNK = 30
OUT_DIR = Path("data/processed/dollar_bars/1_minute")

END = datetime(2026, 6, 12, tzinfo=timezone.utc)


def _seg(symbol, y1, m1, d1, y2, m2, d2):
    return {
        "symbol": symbol,
        "start": datetime(y1, m1, d1, tzinfo=timezone.utc),
        "end": datetime(y2, m2, d2, tzinfo=timezone.utc),
    }


# Metals roll dates ≈8 biz days before 3rd-to-last business day of delivery month.
# SI/HG share the H K N U Z calendar. K25 (May '25) was past first notice by
# 2025-05-01 (smoke test: ~40 bars/2d), so N25 is the front month from day one:
#   N25 exp ~Jul 29 → roll Jul 17       U25 exp ~Sep 26 → roll Sep 16
#   Z25 exp ~Dec 29 → roll Dec 16       H26 exp ~Mar 27 '26 → roll Mar 17
#   K26 exp ~May 27 → roll May 14
SI_HG_ROLLS = [
    ("N25", (2025, 5, 1), (2025, 7, 17)),
    ("U25", (2025, 7, 17), (2025, 9, 16)),
    ("Z25", (2025, 9, 16), (2025, 12, 16)),
    ("H26", (2025, 12, 16), (2026, 3, 17)),
    ("K26", (2026, 3, 17), (2026, 5, 14)),
    ("N26", (2026, 5, 14), (2026, 6, 12)),
]

# PL calendar F J N V:
#   N25 exp ~Jul 29 '25 → roll Jul 17   V25 exp ~Oct 29 → roll Oct 17
#   F26 exp ~Jan 28 '26 → roll Jan 15   J26 exp ~Apr 28 → roll Apr 16
PL_ROLLS = [
    ("N25", (2025, 5, 1), (2025, 7, 17)),
    ("V25", (2025, 7, 17), (2025, 10, 17)),
    ("F26", (2025, 10, 17), (2026, 1, 15)),
    ("J26", (2026, 1, 15), (2026, 4, 16)),
    ("N26", (2026, 4, 16), (2026, 6, 12)),
]

# Index quarterlies: exact ES roll dates (download_es_1min.py) + M26→U26 on 2026-06-08
INDEX_ROLLS = [
    ("M25", (2025, 5, 1), (2025, 6, 9)),
    ("U25", (2025, 6, 9), (2025, 9, 8)),
    ("Z25", (2025, 9, 8), (2025, 12, 8)),
    ("H26", (2025, 12, 8), (2026, 3, 9)),
    ("M26", (2026, 3, 9), (2026, 6, 8)),
    ("U26", (2026, 6, 8), (2026, 6, 12)),
]


def _build(root, rolls):
    return [_seg(f"{root}{code}", *s, *e) for code, s, e in rolls]


ROOTS = {
    "SI":  {"point_value": 5000.0,  "segments": _build("SI", SI_HG_ROLLS)},
    "HG":  {"point_value": 25000.0, "segments": _build("HG", SI_HG_ROLLS)},
    "PL":  {"point_value": 50.0,    "segments": _build("PL", PL_ROLLS)},
    "RTY": {"point_value": 50.0,    "segments": _build("RTY", INDEX_ROLLS)},
    "YM":  {"point_value": 5.0,     "segments": _build("YM", INDEX_ROLLS)},
}


def date_chunks(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


async def fetch_chunk(auth, client, symbol, cs, ce):
    token = await auth.authenticate()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    url = (
        f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
        f"?interval=1&unit=Minute"
        f"&firstdate={cs.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&lastdate={ce.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    resp = await client.get(url, headers=headers, timeout=60.0)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code} — skipping  ({resp.text[:120]})")
        return []
    return resp.json().get("Bars", [])


async def download_segment(auth, client, seg, point_value):
    symbol = seg["symbol"]
    start, end = seg["start"], seg["end"]
    chunks = list(date_chunks(start, end, MAX_DAYS_PER_CHUNK))
    print(f"\n{symbol}  {start.date()} → {end.date()}  ({len(chunks)} chunk(s))")
    all_bars = []
    for i, (cs, ce) in enumerate(chunks, 1):
        print(f"  Chunk {i}/{len(chunks)}: {cs.date()} → {ce.date()} …", end=" ", flush=True)
        bars = await fetch_chunk(auth, client, symbol, cs, ce)
        print(f"{len(bars):,} bars")
        for b in bars:
            try:
                high = float(b["High"])
                low = float(b["Low"])
                vol = int(b.get("TotalVolume", 0))
                ts = b["TimeStamp"].replace("Z", "+00:00")
                all_bars.append({
                    "timestamp": ts,
                    "open": float(b["Open"]),
                    "high": high,
                    "low": low,
                    "close": float(b["Close"]),
                    "volume": vol,
                    "notional": max(((high + low) / 2) * vol * point_value, 0.01),
                })
            except Exception as e:
                print(f"    parse error: {e}")
    return all_bars


async def smoke(auth, client, root_cfg):
    """Probe 2 days at the start of each segment; no file output."""
    ok = True
    for seg in root_cfg["segments"]:
        cs = seg["start"]
        ce = min(cs + timedelta(days=2), seg["end"])
        bars = await fetch_chunk(auth, client, seg["symbol"], cs, ce)
        status = f"{len(bars):,} bars" if bars else "NO BARS ⚠️"
        if not bars:
            ok = False
        print(f"  {seg['symbol']}  {cs.date()} +2d: {status}")
    return ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, choices=sorted(ROOTS))
    ap.add_argument("--smoke", action="store_true", help="2-day probe per segment, no output file")
    args = ap.parse_args()

    cfg = ROOTS[args.root]
    out_path = OUT_DIR / f"{args.root.lower()}_1min_2025_2026.csv"
    auth = TradeStationAuthV3.from_file(".access_token")

    async with httpx.AsyncClient() as client:
        if args.smoke:
            print(f"SMOKE TEST — {args.root}")
            ok = await smoke(auth, client, cfg)
            sys.exit(0 if ok else 1)

        all_bars = []
        for seg in cfg["segments"]:
            all_bars.extend(await download_segment(auth, client, seg, cfg["point_value"]))

    if not all_bars:
        print("\nNo bars downloaded. Check token or network.")
        sys.exit(1)

    seen = set()
    unique = []
    for b in sorted(all_bars, key=lambda x: x["timestamp"]):
        if b["timestamp"] not in seen:
            seen.add(b["timestamp"])
            unique.append(b)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume", "notional"]
        )
        writer.writeheader()
        writer.writerows(unique)

    print(f"\nSaved {len(unique):,} bars → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"Range: {unique[0]['timestamp']} → {unique[-1]['timestamp']}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
