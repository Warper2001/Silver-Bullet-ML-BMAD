"""
Gap Ratio Sweep — S26v2 Frequency Analysis
===========================================
Runs the 1-year backtest (May 2025–May 2026) at four min_gap_atr_ratio values
and reports signal counts + PF with the S26v2 DFC filter applied.

Pre-registration: S26v2 SHA 0ff1818 (sealed before this sweep was run)

Usage:
  cpulimit -l 10 -- .venv/bin/python gap_ratio_sweep.py
"""

import asyncio
import csv
import dataclasses
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytz
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.CRITICAL)

import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

ET_TZ = pytz.timezone("US/Eastern")

# ── S26v2 DFC filter (sealed SHA 0ff1818) ─────────────────────────────────────
WINDOW_A = {9, 10}
WINDOW_B = {13, 14}
DFC_HOURS = WINDOW_A | WINDOW_B
BLOCKED_DOW = {1}  # Tuesday only

def s26v2_window(ts_utc):
    et = ts_utc.astimezone(ET_TZ)
    if et.weekday() in BLOCKED_DOW:
        return "excl"
    if et.hour in WINDOW_B:
        return "B"
    if et.hour in WINDOW_A:
        return "A"
    return "excl"

# ── Config ─────────────────────────────────────────────────────────────────────
RATIOS     = [0.10, 0.15, 0.20, 0.25]
CSV_2025   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
REPORTS    = Path("data/reports")
START_DATE = datetime(2025, 5, 19, tzinfo=timezone.utc)
END_DATE   = datetime(2026, 5, 19, 23, 59, 59, tzinfo=timezone.utc)
DAYS       = (END_DATE - START_DATE).days


# ── Stats ──────────────────────────────────────────────────────────────────────

def pf(pnls):
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    return gp / gl if gl > 0 else float("inf")

def wr(pnls):
    return sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0.0

def sharpe(pnls):
    if len(pnls) < 2:
        return float("nan")
    a = np.array(pnls, dtype=float)
    return float(a.mean() / a.std() * np.sqrt(252)) if a.std() > 0 else float("nan")


# ── Bar loader ─────────────────────────────────────────────────────────────────

def load_bars() -> list[DollarBar]:
    bars = []
    for path in [CSV_2025, CSV_2026]:
        with open(path) as f:
            for row in csv.DictReader(f):
                ts = datetime.fromisoformat(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < START_DATE or ts > END_DATE:
                    continue
                bars.append(DollarBar.model_construct(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(float(row["volume"])),
                    notional_value=float(row["notional"]),
                    is_forward_filled=False,
                ))
    bars.sort(key=lambda b: b.timestamp)
    return bars


# ── Single backtest run ────────────────────────────────────────────────────────

async def run_one(bars: list[DollarBar], ratio: float) -> list[dict]:
    trader = Tier2StreamingTrader()

    # Patch gap ratio — only this changes between runs
    trader._strategy_config = dataclasses.replace(
        trader._strategy_config,
        min_gap_atr_ratio=ratio,
    )

    mock_client = MagicMock()
    mock_client.submit_bracket_order     = AsyncMock(return_value=(None, None, None))
    mock_client.cancel_order             = AsyncMock(return_value=True)
    mock_client.close_position_at_market = AsyncMock(return_value=None)
    mock_client.reconcile_state          = AsyncMock(return_value=None)
    trader._ts_client = mock_client
    trader.ml_filter._log_decision = lambda *a, **kw: None
    tier2_mod.StatePersistence.save_state = staticmethod(lambda *a, **kw: None)

    last_h1_ts = None

    for bar in bars:
        trader.dollar_bars.append(bar)
        trader._last_processed_timestamp = bar.timestamp

        bar_et = bar.timestamp.astimezone(ET_TZ)
        if trader._current_day != bar_et.date():
            if trader._current_day is not None:
                trader._daily_ranges.append(trader._session_high - trader._session_low)
                if len(trader._daily_ranges) > 20:
                    trader._daily_ranges.pop(0)
            trader._current_day  = bar_et.date()
            trader._session_open_price = np.nan
            trader._session_high, trader._session_low = bar.high, bar.low
        else:
            trader._session_high = max(trader._session_high, bar.high)
            trader._session_low  = min(trader._session_low,  bar.low)
        if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
            trader._session_open_price = bar.open

        h1_ts = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        if h1_ts != last_h1_ts:
            trader._update_h1_structure()
            last_h1_ts = h1_ts

        trader._update_m15_choch()
        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(bar, is_backfill=False)

    trades = []
    for t in trader.completed_trades:
        entry_et = t.entry_time.astimezone(ET_TZ)
        win = s26v2_window(t.entry_time)
        trades.append({
            "entry_time": t.entry_time.isoformat(),
            "exit_type":  t.exit_type,
            "pnl":        round(t.pnl, 2),
            "hour_et":    entry_et.hour,
            "dow":        entry_et.weekday(),
            "window":     win,
        })
    return trades


# ── Report ─────────────────────────────────────────────────────────────────────

def summarise(ratio: float, trades: list[dict]) -> dict:
    all_pnl = [t["pnl"] for t in trades]
    win_b   = [t for t in trades if t["window"] == "B"]
    win_a   = [t for t in trades if t["window"] == "A"]
    excl    = [t for t in trades if t["window"] == "excl"]

    pnl_b = [t["pnl"] for t in win_b]
    pnl_a = [t["pnl"] for t in win_a]

    n_b = len(win_b)
    freq_b = n_b / DAYS

    return {
        "ratio":    ratio,
        "N_all":    len(trades),
        "PF_all":   round(pf(all_pnl), 4) if all_pnl else 0,
        "N_A":      len(win_a),
        "PF_A":     round(pf(pnl_a), 4) if pnl_a else 0,
        "WR_A":     round(wr(pnl_a), 1),
        "N_B":      n_b,
        "PF_B":     round(pf(pnl_b), 4) if pnl_b else 0,
        "WR_B":     round(wr(pnl_b), 1),
        "net_B":    round(sum(pnl_b), 0),
        "freq_B":   round(freq_b, 4),
        "days_N20": round(20 / freq_b) if freq_b > 0 else 9999,
        "N_excl":   len(excl),
        "PF_excl":  round(pf([t["pnl"] for t in excl]), 4) if excl else 0,
    }


def print_report(results: list[dict]):
    SEP = "=" * 110
    print(SEP)
    print("GAP RATIO SWEEP — S26v2 Directional Flow Continuity  (May 2025–May 2026)")
    print("  Pre-registration: S26v2 SHA 0ff1818")
    print("  Ratios tested: 0.10, 0.15, 0.20, 0.25")
    print(SEP)
    print()

    # Overall table
    print("── OVERALL SIGNAL COUNT & QUALITY ──────────────────────────────────────────────────────────")
    hdr = f"  {'Ratio':>6}  {'N_all':>6}  {'PF_all':>7}  {'N_WinA':>7}  {'PF_A':>7}  {'N_WinB':>7}  {'PF_B':>8}  {'WR_B':>6}  {'Net_B':>8}  {'days→N20':>9}"
    print(hdr)
    print("  " + "─" * 106)
    for r in results:
        pf_b_str = f"{r['PF_B']:8.4f}" if r["PF_B"] != float("inf") else "     inf"
        days_str = f"{r['days_N20']:>9}" if r["days_N20"] < 9000 else "    never"
        print(f"  {r['ratio']:>6.2f}  {r['N_all']:>6}  {r['PF_all']:>7.4f}  "
              f"{r['N_A']:>7}  {r['PF_A']:>7.4f}  {r['N_B']:>7}  {pf_b_str}  "
              f"{r['WR_B']:>5.1f}%  {r['net_B']:>+8.0f}  {days_str}")

    print()
    print("── WINDOW B FREQUENCY LIFT vs BASELINE (ratio=0.25) ────────────────────────────────────────")
    baseline = next(r for r in results if abs(r["ratio"] - 0.25) < 0.001)
    for r in results:
        lift_n   = r["N_B"] - baseline["N_B"]
        lift_pct = (r["N_B"] / baseline["N_B"] - 1) * 100 if baseline["N_B"] > 0 else 0
        freq_note = f"{r['freq_B']:.3f}/day → N=20 in ~{r['days_N20']} days"
        print(f"  ratio={r['ratio']:.2f}: N_B={r['N_B']:>3}  ({lift_pct:>+5.0f}% vs baseline)  {freq_note}")

    print()
    print("── EXCLUDED BUCKET (non-DFC hours) ──────────────────────────────────────────────────────────")
    for r in results:
        print(f"  ratio={r['ratio']:.2f}: N_excl={r['N_excl']:>3}  PF_excl={r['PF_excl']:.4f}")

    print()
    print("── INTERPRETATION ───────────────────────────────────────────────────────────────────────────")
    best = max(results, key=lambda r: r["N_B"])
    print(f"  Maximum Window B signals at ratio={best['ratio']:.2f}: N={best['N_B']} ({best['freq_B']:.3f}/day)")
    print(f"  Baseline (0.25): N_B={baseline['N_B']}, PF_B={baseline['PF_B']:.4f}")
    viable = [r for r in results if r["days_N20"] <= 365]
    if viable:
        print(f"  Viable (<365 days to N=20): {[r['ratio'] for r in viable]}")
    else:
        print("  No ratio achieves N=20 in <365 days in Window B alone")

    print(SEP)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars...", flush=True)
    bars = load_bars()
    print(f"  {len(bars):,} bars loaded", flush=True)

    results = []
    for i, ratio in enumerate(RATIOS, 1):
        print(f"\n[{i}/{len(RATIOS)}] Running ratio={ratio:.2f}...", flush=True)
        trades = asyncio.run(run_one(bars, ratio))
        s = summarise(ratio, trades)
        results.append(s)
        print(f"  Done — N_all={s['N_all']}  N_B={s['N_B']}  PF_B={s['PF_B']:.4f}  "
              f"freq_B={s['freq_B']:.3f}/day", flush=True)

    print("\n", flush=True)
    print_report(results)

    # Save CSV
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS / f"gap_ratio_sweep_{stamp}.csv"
    keys = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
