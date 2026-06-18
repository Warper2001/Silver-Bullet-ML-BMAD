#!/usr/bin/env python3
"""Bar-parity probe — ProjectX/TopstepX vs TradeStation 1-min bars for MNQU26.

Read-only decision tool for "can the YANK data feed move to ProjectX?". It fetches
ProjectX history bars (POST /api/History/retrieveBars) and diffs them, minute-by-minute,
against the TradeStation bars the live bots already store (data/mim_nb/bars_raw.csv —
genuine MNQU26 1-min TradeStation REST bars). It NEVER touches the live trader, places
no orders, and only calls the History read endpoint.

Why this matters: YANK's sealed config + meta-model were calibrated on TradeStation
bars. Moving the live feed is only safe if ProjectX bars match — OHLC drives the price
signal (FVG/CHoCH), volume feeds the ML filter. This quantifies the gap before any
cutover (run it now, in shadow, weeks before touching the feed).

Usage:
    .venv/bin/python tools/bar_parity_probe.py [--symbol MNQU26] [--hours 6]
        [--ts-csv data/mim_nb/bars_raw.csv] [--live] [--show 12]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.research.projectx_auth import ProjectXAuth  # noqa: E402
from src.research.projectx_client import _to_contract_id  # noqa: E402

PX_HISTORY_URL = "https://api.topstepx.com/api/History/retrieveBars"
TICK = 0.25  # MNQ tick — OHLC parity tolerance for the verdict


def _floor_min(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert("UTC").floor("min")


def load_ts_bars(csv: Path, start: datetime, end: datetime) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True).dt.floor("min")
    df = df[(df["ts"] >= start) & (df["ts"] <= end)]
    return df[["ts", "open", "high", "low", "close", "volume"]].set_index("ts").sort_index()


async def fetch_px_bars(symbol: str, start: datetime, end: datetime, live: bool) -> pd.DataFrame:
    auth = ProjectXAuth.from_file(str(BASE / ".projectx_api_key"))
    token = await auth.authenticate()
    payload = {
        "contractId": _to_contract_id(symbol),
        "live": live,
        "startTime": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "endTime": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "unit": 2,          # 1=Sec 2=Min 3=Hour 4=Day
        "unitNumber": 1,
        "limit": 20000,
        "includePartialBar": False,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
               "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(PX_HISTORY_URL, json=payload, headers=headers)
    try:
        await auth.cleanup()
    except Exception:
        pass

    if resp.status_code != 200:
        raise SystemExit(f"ProjectX History HTTP {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    if not data.get("success", False):
        raise SystemExit(f"ProjectX History rejected: errorCode={data.get('errorCode')} "
                         f"msg={data.get('errorMessage')!r}")
    bars = data.get("bars") or []
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    # Defensive: ProjectX bar keys are t/o/h/l/c/v; print one raw bar if that ever changes.
    sample = bars[0]
    if not all(k in sample for k in ("t", "o", "h", "l", "c", "v")):
        raise SystemExit(f"Unexpected ProjectX bar schema — first bar: {sample!r}")
    rows = [{"ts": _floor_min(pd.Timestamp(b["t"])), "open": b["o"], "high": b["h"],
             "low": b["l"], "close": b["c"], "volume": b["v"]} for b in bars]
    return pd.DataFrame(rows).set_index("ts").sort_index()


def best_offset(ts_df: pd.DataFrame, px_df: pd.DataFrame, span: int = 3):
    """Feeds may label a 1-min bar by open vs close time → a whole-minute shift.
    Scan offsets and return (best_offset_minutes, mean_close_absdiff) minimizing the
    close mismatch, so the parity report compares like-for-like instead of adjacent bars."""
    best = (0, float("inf"))
    for off in range(-span, span + 1):
        p = px_df.copy()
        p.index = p.index + pd.Timedelta(minutes=off)
        common = ts_df.index.intersection(p.index)
        if len(common) < 10:
            continue
        m = (ts_df.loc[common, "close"].astype(float) - p.loc[common, "close"].astype(float)).abs().mean()
        if m < best[1]:
            best = (off, m)
    return best


def report(ts_df: pd.DataFrame, px_df: pd.DataFrame, show: int, symbol: str = "the PX contract") -> None:
    ts_idx, px_idx = set(ts_df.index), set(px_df.index)
    common = sorted(ts_idx & px_idx)
    ts_only = sorted(ts_idx - px_idx)
    px_only = sorted(px_idx - ts_idx)

    print("=" * 70)
    print("Bar-parity: ProjectX vs TradeStation (MNQU26, 1-min)")
    print("=" * 70)
    print(f"TradeStation bars : {len(ts_df):5d}")
    print(f"ProjectX bars     : {len(px_df):5d}")
    print(f"Common minutes    : {len(common):5d}")
    print(f"TS-only (PX gap)  : {len(ts_only):5d}" + (f"  e.g. {ts_only[:3]}" if ts_only else ""))
    print(f"PX-only (TS gap)  : {len(px_only):5d}" + (f"  e.g. {px_only[:3]}" if px_only else ""))
    if not common:
        print("\nNo overlapping minutes — widen --hours or check --live / contract/window.")
        return

    a = ts_df.loc[common]
    b = px_df.loc[common]
    print("\nPer-field abs diff over common minutes (TS vs PX):")
    print(f"  {'field':6s} {'exact':>7s} {'max':>10s} {'mean':>10s}")
    worst_close = None
    for f in ("open", "high", "low", "close", "volume"):
        d = (a[f].astype(float) - b[f].astype(float)).abs()
        exact = int((d == 0).sum())
        print(f"  {f:6s} {exact:5d}/{len(d):<5d} {d.max():10.4f} {d.mean():10.4f}")
        if f == "close":
            worst_close = d.sort_values(ascending=False)

    # Distribution-aware OHLC summary (max alone overstates it — a few boundary bars
    # dominate while the median is 0). Pool all OHLC abs-diffs.
    ohlc_diffs = pd.concat([(a[f].astype(float) - b[f].astype(float)).abs()
                            for f in ("open", "high", "low", "close")])
    ohlc_max = ohlc_diffs.max()
    ohlc_median = ohlc_diffs.median()
    within_tick = float((ohlc_diffs <= TICK).mean())
    vol_mean = (a["volume"].astype(float) - b["volume"].astype(float)).abs().mean()

    if show and worst_close is not None and worst_close.iloc[0] > 0:
        print(f"\nWorst {show} close mismatches (TS close vs PX close):")
        for t in worst_close.head(show).index:
            print(f"  {t}  TS {float(a.loc[t,'close']):>10.2f}  PX {float(b.loc[t,'close']):>10.2f}"
                  f"  Δ {float(a.loc[t,'close'])-float(b.loc[t,'close']):+.2f}")

    # Contract-mismatch guard: a large, ~constant signed close offset is a calendar
    # spread (TS reference on a different front month than the PX contract — e.g. across
    # a roll boundary), NOT a feed divergence. bars_raw.csv spans rolls, so this is a
    # real trap. Detect: big median AND a roughly constant signed gap.
    signed = (a["close"].astype(float) - b["close"].astype(float))
    if ohlc_median > 20 and abs(signed.mean()) > 20 and signed.std() < abs(signed.mean()) * 0.5:
        print("\n" + "-" * 70)
        print("VERDICT")
        print(f"  ⛔ LIKELY DIFFERENT CONTRACT — close gap ≈ {signed.mean():+.1f}pt, near-constant "
              f"(std {signed.std():.1f}).")
        print("     That's a calendar spread, not a feed divergence. The TS reference for this")
        print("     window is probably a different front month than PX (bars_raw spans rolls).")
        print(f"     Pick a window where bars_raw is the same contract as {symbol}.")
        return

    print("\n" + "-" * 70)
    print("VERDICT")
    print(f"  OHLC abs-diff: median {ohlc_median:.4f} | {within_tick*100:.1f}% within 1 tick | max {ohlc_max:.4f}")
    if ohlc_median <= TICK and within_tick >= 0.95 and not ts_only and not px_only:
        print("  -> Feeds effectively identical for the price signal (FVG/CHoCH).")
        if ohlc_max > TICK:
            print(f"     Residual: a few boundary bars differ up to {ohlc_max:.2f}pt (last-tick timing), median 0.")
    else:
        if ohlc_median > TICK or within_tick < 0.95:
            print(f"  ⚠ OHLC diverges materially ({within_tick*100:.1f}% within 1 tick) — price signal could differ.")
        if ts_only or px_only:
            print(f"  ⚠ Coverage gaps (TS-only {len(ts_only)}, PX-only {len(px_only)}) — feeds miss bars.")
    print(f"  Volume mean |Δ| = {vol_mean:.1f} contracts — volume feeds the ML filter; small is fine,")
    print("    a large gap means the meta-model would see different inputs.")
    print("  NOTE: one window is a snapshot. Run across several sessions before trusting it,")
    print("        and shadow-log live before any cutover (post-S25).")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="MNQU26")
    ap.add_argument("--ts-csv", type=Path, default=BASE / "data" / "mim_nb" / "bars_raw.csv")
    ap.add_argument("--hours", type=float, default=6.0, help="window length in hours")
    ap.add_argument("--end", default=None,
                    help="window end (ISO UTC, e.g. 2026-06-17T20:00Z); default = last TS bar")
    ap.add_argument("--live", action="store_true", help="request ProjectX 'live' market data")
    ap.add_argument("--offset", type=int, default=None,
                    help="pin the PX bar-label offset (minutes); default = auto-detect")
    ap.add_argument("--show", type=int, default=12, help="how many worst close mismatches to list")
    args = ap.parse_args()

    if not args.ts_csv.exists():
        raise SystemExit(f"TradeStation reference not found: {args.ts_csv}")

    # Window = [end - hours, end]; end defaults to the last available TS bar.
    full = pd.read_csv(args.ts_csv)
    last_ts = pd.to_datetime(full["ts_utc"], utc=True).max().to_pydatetime()
    if args.end:
        end = pd.Timestamp(args.end.replace("Z", "+00:00")).tz_convert("UTC").to_pydatetime()
    else:
        end = last_ts
    start = end - timedelta(hours=args.hours)
    print(f"Window: {start.isoformat()} → {end.isoformat()}  (last {args.hours}h of {args.ts_csv.name})\n")

    ts_df = load_ts_bars(args.ts_csv, start, end)
    px_df = await fetch_px_bars(args.symbol, start, end, live=args.live)
    if px_df.empty and not args.live:
        print("ProjectX returned 0 bars with live=false — retrying with live=true …\n")
        px_df = await fetch_px_bars(args.symbol, start, end, live=True)

    # Auto-detect & correct the bar-labeling offset (open- vs close-time labeling)
    # unless the user pins it. Without this, an off-by-one-minute convention looks
    # like a massive price divergence (it isn't — same bars, shifted timestamps).
    if not px_df.empty and args.offset is None:
        off, _ = best_offset(ts_df, px_df)
        if off != 0:
            print(f"⚠ Detected bar-labeling offset: ProjectX is {off:+d} min vs TradeStation "
                  f"(TS labels by bar-close, PX by bar-open or vice-versa).")
            print(f"  Aligning PX by {off:+d} min for a like-for-like compare. "
                  f"A live feed swap MUST apply this shift.\n")
            px_df = px_df.copy()
            px_df.index = px_df.index + pd.Timedelta(minutes=off)
    elif args.offset:
        px_df = px_df.copy()
        px_df.index = px_df.index + pd.Timedelta(minutes=args.offset)
        print(f"Applied user --offset {args.offset:+d} min to ProjectX bars.\n")

    report(ts_df, px_df, args.show, symbol=args.symbol)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
