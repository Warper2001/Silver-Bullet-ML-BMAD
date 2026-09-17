"""
Re-download 1-min HG (copper) and SI (silver) bars, front-month, stitched per contract.

Why: the existing `hg_1min_2025_2026.csv` and `si_1min_2025_2026.csv` carry the WRONG
contract for 2026-03-01 → 03-11 — median volume 2 and 1 contracts a minute, against 16 and
18 just after (`_bmad-output/diagnostics_march_roll_exposure_20260916/`). A probe settled
which way round it is: on 2026-03-02 the March contracts HGH26/SIH26 traded 2 and 1
contracts a minute while May's HGK26/SIK26 traded 19 and 32. So those files sat on the
EXPIRING contract in its notice period, not on a deferred one. This refetches them the way
`download_es_1min.py` and `download_gc_1min.py` already do: one request per contract with
explicit date ranges.

Roll dates are DERIVED from daily volume by `build_metals_roll_calendar.py`, not assumed.
COMEX metals leave the delivery month around first notice — the end of the month BEFORE
delivery — so copper rolled H26 → K26 on 2026-02-24 and silver on 2026-02-25, about three
weeks before the equity-index roll. Applying the equity-index convention here would have
rebuilt the very defect this fixes.

Output (NEW files — the defective originals are left in place as evidence):
  data/processed/dollar_bars/1_minute/hg_1min_2025_2026_frontmonth.csv
  data/processed/dollar_bars/1_minute/si_1min_2025_2026_frontmonth.csv

Usage:
  .venv/bin/python download_hg_si_1min.py --probe          # one 2-day window, no writes
  nohup .venv/bin/python download_hg_si_1min.py > /tmp/hg_si_fetch.log 2>&1 &
"""
import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

MAX_DAYS_PER_CHUNK = 30
CHUNK_PAUSE_S = 1.5          # be polite: the live traders share this API and token
OUT_DIR = Path("data/processed/dollar_bars/1_minute")


def utc(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)


# Roll dates are DERIVED, not assumed: build_metals_roll_calendar.py fetches daily volume for
# every candidate contract and rolls on the volume crossover (3-session confirmation). That
# matters — COMEX metals leave the delivery month at its notice period, ~3 weeks before the
# equity-index roll. Assuming the mid-March date would have rebuilt the very defect this fixes:
# on 2026-03-02, HGH26 traded 2 contracts/min against HGK26's 19, and SIH26 1 against SIK26's 32.
CALENDAR = Path("metals_roll_calendar.json")
_DERIVED = {
    "hg": [("HGK25", "2025-05-01", "2025-06-24"), ("HGN25", "2025-06-24", "2025-08-25"),
           ("HGU25", "2025-08-25", "2025-11-24"), ("HGZ25", "2025-11-24", "2026-02-24"),
           ("HGH26", "2026-02-24", "2026-02-24"), ("HGK26", "2026-02-24", "2026-04-27"),
           ("HGN26", "2026-04-27", "2026-06-12")],
    "si": [("SIK25", "2025-05-01", "2025-06-26"), ("SIN25", "2025-06-26", "2025-08-27"),
           ("SIU25", "2025-08-27", "2025-11-25"), ("SIZ25", "2025-11-25", "2026-02-25"),
           ("SIH26", "2026-02-25", "2026-02-25"), ("SIK26", "2026-02-25", "2026-04-28"),
           ("SIN26", "2026-04-28", "2026-06-12")],
}

INSTRUMENTS = {
    "hg": {"root": "HG", "point_value": 25_000.0,   # 25,000 lb; $25,000 per $1.00/lb
           "out": OUT_DIR / "hg_1min_2025_2026_frontmonth.csv"},
    "si": {"root": "SI", "point_value": 5_000.0,    # 5,000 troy oz; $5,000 per $1.00/oz
           "out": OUT_DIR / "si_1min_2025_2026_frontmonth.csv"},
}


def segments(key: str) -> list[dict]:
    """Front-month segments for `key`, preferring the derived calendar file when present."""
    rows = _DERIVED[key]
    if CALENDAR.exists():
        cal = json.loads(CALENDAR.read_text()).get(key, {}).get("segments")
        if cal:
            rows = [(s["symbol"], s["start"], s["end"]) for s in cal]
    out = []
    for sym, s, e in rows:
        if utc(e) <= utc(s):        # a contract that never led (zero-length segment)
            continue
        start = max(utc(s), utc("2025-05-01"))
        end = min(utc(e), utc("2026-06-12"))
        if end > start:
            out.append({"symbol": sym, "start": start, "end": end})
    return out


def date_chunks(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


async def fetch_window(auth, client: httpx.AsyncClient, symbol: str, cs: datetime, ce: datetime,
                       point_value: float) -> list[dict]:
    token = await auth.authenticate()
    url = (f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
           f"?interval=1&unit=Minute"
           f"&firstdate={cs.strftime('%Y-%m-%dT%H:%M:%SZ')}"
           f"&lastdate={ce.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    resp = await client.get(url, headers={"Authorization": f"Bearer {token}",
                                          "Accept": "application/json"}, timeout=60.0)
    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code} — {resp.text[:140]}")
        return []
    out = []
    for b in resp.json().get("Bars", []):
        try:
            high, low = float(b["High"]), float(b["Low"])
            vol = int(b.get("TotalVolume", 0))
            out.append({"timestamp": b["TimeStamp"].replace("Z", "+00:00"),
                        "open": float(b["Open"]), "high": high, "low": low,
                        "close": float(b["Close"]), "volume": vol,
                        "notional": max(((high + low) / 2) * vol * point_value, 0.01)})
        except Exception as e:  # noqa: BLE001 — a malformed bar must not kill the fetch
            print(f"    parse error: {e}")
    return out


async def download_instrument(auth, client, key: str) -> list[dict]:
    spec = INSTRUMENTS[key]
    bars: list[dict] = []
    for seg in segments(key):
        chunks = list(date_chunks(seg["start"], seg["end"], MAX_DAYS_PER_CHUNK))
        print(f"\n{seg['symbol']}  {seg['start'].date()} → {seg['end'].date()}  ({len(chunks)} chunk(s))")
        for i, (cs, ce) in enumerate(chunks, 1):
            print(f"  chunk {i}/{len(chunks)}: {cs.date()} → {ce.date()} …", end=" ", flush=True)
            got = await fetch_window(auth, client, seg["symbol"], cs, ce, spec["point_value"])
            print(f"{len(got):,} bars")
            bars.extend(got)
            await asyncio.sleep(CHUNK_PAUSE_S)
    return bars


def write_csv(path: Path, bars: list[dict]) -> int:
    seen, unique = set(), []
    for b in sorted(bars, key=lambda x: x["timestamp"]):
        if b["timestamp"] not in seen:
            seen.add(b["timestamp"])
            unique.append(b)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close",
                                          "volume", "notional"])
        w.writeheader()
        w.writerows(unique)
    return len(unique)


async def probe(auth, client) -> int:
    """One small window per instrument over the defective days. Writes nothing.

    Both contracts are fetched so the volumes speak for themselves: on 2026-03-02 the March
    contract is in its notice period (thin) and May is the liquid front month.
    """
    ok = True
    for key, spec in INSTRUMENTS.items():
        sym_expiring = f"{spec['root']}H26"   # March: in notice, thin
        sym_front = f"{spec['root']}K26"      # May: the actual front month on these dates
        for sym in (sym_expiring, sym_front):
            bars = await fetch_window(auth, client, sym, utc("2026-03-02"), utc("2026-03-04"),
                                      spec["point_value"])
            vols = sorted(b["volume"] for b in bars)
            med = vols[len(vols) // 2] if vols else 0
            print(f"  {sym}: {len(bars):,} bars, median volume/min {med}")
            ok = ok and (bars or sym != sym_front)
            await asyncio.sleep(CHUNK_PAUSE_S)
    print("\nprobe OK" if ok else "\nprobe FAILED: front-month symbol returned nothing")
    return 0 if ok else 1


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true", help="validate access on one window, write nothing")
    ap.add_argument("--only", choices=list(INSTRUMENTS), help="fetch a single instrument")
    args = ap.parse_args()

    auth = TradeStationAuthV3.from_file(".access_token")
    async with httpx.AsyncClient() as client:
        if args.probe:
            return await probe(auth, client)
        for key in ([args.only] if args.only else list(INSTRUMENTS)):
            bars = await download_instrument(auth, client, key)
            if not bars:
                print(f"\n{key}: no bars downloaded — leaving the existing file alone")
                continue
            n = write_csv(INSTRUMENTS[key]["out"], bars)
            print(f"\n{key}: wrote {n:,} unique bars → {INSTRUMENTS[key]['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
