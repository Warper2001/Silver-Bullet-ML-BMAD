"""
Replay today's MNQM26 1-min bars through Tier2StreamingTrader and reconcile
signal/trade outcomes against the live trader's ground-truth records.

Ground truth:
  - logs/tier2_filter_log.csv  — ML filter decisions (live trader wrote these)
  - Live trades: TP +$921 (18:17 UTC), SL -$954 (19:03 UTC), net -$33

Usage:
  .venv/bin/python backtest_tier2_today.py
"""
import asyncio
import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pytz

# Configure root logging BEFORE importing tier2 module so its basicConfig call is a no-op.
logging.basicConfig(level=logging.WARNING, format="%(message)s")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.research.tier2_streaming_working import (  # noqa: E402
    Tier2StreamingTrader,
    BARS_BASE_URL,
)
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

CUTOVER_UTC = datetime(2026, 5, 18, 17, 7, 0, tzinfo=timezone.utc)
TODAY = "2026-05-18"
ET_TZ = pytz.timezone("US/Eastern")

# Known live P&L for today's session (from logs/tier2_streaming_working.log)
LIVE_TRADE_COUNT = 2
LIVE_NET_PNL = -33.0  # +921 (TP 18:17) + -954 (SL 19:03)


async def main() -> None:
    # ── 1. Capture ground truth BEFORE the replay touches filter_log.csv ──────
    filter_log_path = Path("logs/tier2_filter_log.csv")
    ground_truth: list[dict] = []
    if filter_log_path.exists():
        with open(filter_log_path) as f:
            for row in csv.DictReader(f):
                ts = row["timestamp"]
                if TODAY in ts and ts > f"{TODAY} 17:":
                    ground_truth.append(row)

    # ── 2. Fetch bars from TradeStation ───────────────────────────────────────
    auth = TradeStationAuthV3.from_file(".access_token")
    token = await auth.authenticate()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    since = datetime.now(timezone.utc) - timedelta(hours=48)
    url = f"{BARS_BASE_URL}&firstdate={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers)
    bars_data = resp.json().get("Bars", [])
    print(f"Fetched {len(bars_data)} bars  ({since.strftime('%Y-%m-%d %H:%M')} UTC → now)")

    # ── 3. Instantiate trader; patch out all side-effects ─────────────────────
    trader = Tier2StreamingTrader()
    trader.auth = auth

    # Capture ML filter decisions in memory instead of appending to filter_log.csv
    captured: list[dict] = []

    def _capture_decision(timestamp, proba: float, decision: str) -> None:
        captured.append({
            "timestamp": str(timestamp),
            "filter_decision": decision,
            "probability": round(proba, 4),
        })

    trader.ml_filter._log_decision = _capture_decision

    # No-op SIM broker calls (local P&L sim in _close_active_trade is untouched)
    async def _no_bracket(*_, **__):  return (None, None, None)
    async def _no_cancel(*_, **__):   return True
    async def _no_close(*_, **__):    return None

    trader._submit_bracket_order = _no_bracket
    trader._cancel_sim_order     = _no_cancel
    trader._submit_close_order   = _no_close

    # ── 4. Replay bars ─────────────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    for raw in bars_data:
        bar = trader._parse_bar(raw)
        if not bar or bar.timestamp > now_utc:
            continue

        trader.dollar_bars.append(bar)
        trader._last_processed_timestamp = bar.timestamp

        # Replicate the session-state update that _poll_and_process does inline
        bar_et = bar.timestamp.astimezone(ET_TZ)
        if trader._current_day != bar_et.date():
            if trader._current_day is not None:
                trader._daily_ranges.append(trader._session_high - trader._session_low)
                if len(trader._daily_ranges) > 20:
                    trader._daily_ranges.pop(0)
            trader._current_day   = bar_et.date()
            trader._session_open_price = np.nan
            trader._session_high, trader._session_low = bar.high, bar.low
        else:
            trader._session_high = max(trader._session_high, bar.high)
            trader._session_low  = min(trader._session_low,  bar.low)
        if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
            trader._session_open_price = bar.open

        trader._update_h1_structure()
        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(
            bar,
            is_backfill=(bar.timestamp < CUTOVER_UTC),
        )

    # ── 5. Trade results ───────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"BACKTEST TRADES  (limit-entry, live phase ≥ {CUTOVER_UTC.strftime('%H:%M')} UTC)")
    print("=" * 65)
    for t in trader.completed_trades:
        print(
            f"  {t.exit_type.upper():<4}  {t.direction}  "
            f"entry={t.entry_price:.2f}  exit={t.exit_price:.2f}  "
            f"P&L=${t.pnl:+.2f}  "
            f"{t.entry_time.strftime('%H:%M')}–{t.exit_time.strftime('%H:%M')} UTC"
        )
    bt_count = len(trader.completed_trades)
    bt_pnl   = sum(t.pnl for t in trader.completed_trades)
    print(f"\n  Trades filled: {bt_count}   Net P&L: ${bt_pnl:+.2f}")
    print(f"  Signals seen:  {len(live_sigs := [s for s in captured if s['timestamp'] >= f'{TODAY} 17:'])}")
    misses = len(live_sigs) - bt_count - sum(1 for s in live_sigs if s['filter_decision'] == 'FILTERED')
    print(f"  Limit misses:  {max(0, misses)}  (signal fired but price never reached FVG midpoint)")
    print(f"\n  MARKET-order baseline (prev run): Trades=2  Net P&L=$-33.00")

    # ── 6. Signal reconciliation ───────────────────────────────────────────────
    live_sigs = [s for s in captured if s["timestamp"] >= f"{TODAY} 17:"]
    print()
    print("=" * 65)
    print("SIGNAL RECONCILIATION  (post-restart signals vs filter_log.csv)")
    print("=" * 65)
    print(f"  {'Timestamp':<35}  {'Decision':<8}  {'P (bt)':>8}  {'P (live)':>8}  Match")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}  -----")
    max_rows = max(len(live_sigs), len(ground_truth))
    for i in range(max_rows):
        bt  = live_sigs[i]    if i < len(live_sigs)    else None
        gt  = ground_truth[i] if i < len(ground_truth) else None
        ts       = (bt or gt)["timestamp"][:22]
        dec_bt   = bt["filter_decision"] if bt else "MISSING"
        p_bt     = bt["probability"]     if bt else float("nan")
        p_gt     = float(gt["probability"]) if gt else float("nan")
        dec_gt   = gt["filter_decision"]    if gt else "MISSING"
        ts_ok    = bt and gt and bt["timestamp"][:16] == gt["timestamp"][:16]
        p_ok     = bt and gt and abs(p_bt - p_gt) < 0.001
        dec_ok   = bt and gt and dec_bt == dec_gt
        ok       = ts_ok and p_ok and dec_ok
        print(
            f"  {ts:<35}  {dec_bt:<8}  {p_bt:>8.4f}  {p_gt:>8.4f}  {'✅' if ok else '❌'}"
        )
    if not live_sigs and not ground_truth:
        print("  (no post-restart signals found in either source)")


if __name__ == "__main__":
    asyncio.run(main())
