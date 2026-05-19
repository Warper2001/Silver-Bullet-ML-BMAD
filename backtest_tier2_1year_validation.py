"""
1-Year Validation Backtest: May 19, 2025 → May 19, 2026
Fixed config: MAX_PENDING_BARS=240, MAX_HOLD_BARS=60 (grid-search optima on 2025 full year)

Loads:
  data/processed/dollar_bars/1_minute/mnq_1min_2025.csv  (filter ≥ 2025-05-19)
  data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv

Produces:
  Console: per-month table + full-year summary + ASCII equity sparkline
  data/reports/backtest_1year_YYYYMMDD.csv    — one row per trade
  data/reports/backtest_1year_YYYYMMDD.txt    — full console output
  data/reports/equity_curve_1year_YYYYMMDD.csv — daily cumPnL + drawdown

Usage:
  .venv/bin/python backtest_tier2_1year_validation.py
"""
import asyncio
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytz

sys.path.insert(0, str(Path(__file__).parent))
import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

# Suppress all logging during replay
logging.disable(logging.CRITICAL)

ET_TZ = pytz.timezone("US/Eastern")
START_DATE = datetime(2025, 5, 19, tzinfo=timezone.utc)
END_DATE   = datetime(2026, 5, 19, 23, 59, 59, tzinfo=timezone.utc)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
REPORTS_DIR = Path("data/reports")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_bars(csv_path: Path, start: datetime | None = None, end: datetime | None = None) -> list[DollarBar]:
    bars = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start and ts < start:
                continue
            if end and ts > end:
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
    return bars


def profit_factor(pnl: list[float]) -> float:
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    return gp / gl if gl else float("inf")


def per_trade_sharpe(pnl: list[float]) -> float:
    arr = np.array(pnl)
    return float((arr.mean() / arr.std()) * np.sqrt(252)) if len(arr) > 1 and arr.std() > 0 else 0.0


def sortino(pnl: list[float]) -> float:
    arr = np.array(pnl)
    downside = arr[arr < 0]
    ds_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    return float((arr.mean() / ds_std) * np.sqrt(252)) if ds_std > 0 else 0.0


def max_drawdown(pnl: list[float]) -> float:
    cum, peak, dd = 0.0, 0.0, 0.0
    for p in pnl:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    step = len(values) / width
    chars = []
    for i in range(width):
        idx = min(int(i * step), len(values) - 1)
        v = (values[idx] - mn) / rng
        chars.append(blocks[int(v * (len(blocks) - 1))])
    return "".join(chars)


# ── Core backtest run ──────────────────────────────────────────────────────────

async def run_backtest(bars: list[DollarBar]) -> list:
    trader = Tier2StreamingTrader()

    async def _no_bracket(*a, **kw): return (None, None, None)
    async def _no_cancel(*a, **kw): return True
    async def _no_close(*a, **kw): return None

    trader._submit_bracket_order = _no_bracket
    trader._cancel_sim_order     = _no_cancel
    trader._submit_close_order   = _no_close
    trader.ml_filter._log_decision = lambda *a, **kw: None

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
            trader._current_day = bar_et.date()
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

        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(bar, is_backfill=False)

    return trader.completed_trades


# ── Reporting ──────────────────────────────────────────────────────────────────

def build_report(trades, start: datetime, end: datetime) -> tuple[str, list[dict], list[dict]]:
    if not trades:
        return "No trades generated.", [], []

    # Per-trade rows for CSV
    trade_rows = [{
        "entry_time":  t.entry_time.isoformat(),
        "exit_time":   t.exit_time.isoformat(),
        "direction":   t.direction,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "exit_type":   t.exit_type,
        "bars_held":   t.bars_held,
        "pnl":         round(t.pnl, 2),
    } for t in trades]

    all_pnl = [t.pnl for t in trades]

    # Group by month (YYYY-MM)
    by_month: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        key = t.entry_time.astimezone(ET_TZ).strftime("%Y-%m")
        by_month[key].append(t.pnl)

    # Daily equity curve
    by_day: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.entry_time.astimezone(ET_TZ).strftime("%Y-%m-%d")
        by_day[day] += t.pnl

    sorted_days = sorted(by_day)
    daily_pnls = [by_day[d] for d in sorted_days]
    cum_pnl = list(np.cumsum(daily_pnls))
    running_peak = [max(cum_pnl[:i+1]) for i in range(len(cum_pnl))]
    daily_dd = [running_peak[i] - cum_pnl[i] for i in range(len(cum_pnl))]

    equity_rows = [{
        "date":           sorted_days[i],
        "daily_pnl":      round(daily_pnls[i], 2),
        "cumulative_pnl": round(cum_pnl[i], 2),
        "drawdown":       round(daily_dd[i], 2),
    } for i in range(len(sorted_days))]

    # ── Text report ────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 75)
    lines.append("TIER 2 MNQ 1-YEAR VALIDATION BACKTEST")
    lines.append(f"  Period : {start.date()} → {end.date()}")
    lines.append(f"  Config : MAX_PENDING_BARS={tier2_mod.MAX_PENDING_BARS}  MAX_HOLD_BARS={tier2_mod.MAX_HOLD_BARS}")
    lines.append(f"  Data   : {CSV_2025.name} (≥ May 19 2025) + {CSV_2026.name}")
    lines.append("=" * 75)

    # Full-year summary
    wins = [p for p in all_pnl if p > 0]
    losses = [p for p in all_pnl if p < 0]
    wr = len(wins) / len(all_pnl)
    pf = profit_factor(all_pnl)
    net = sum(all_pnl)
    dd  = max_drawdown(all_pnl)
    sh  = per_trade_sharpe(all_pnl)
    so  = sortino(all_pnl)
    avg_w = np.mean(wins) if wins else 0.0
    avg_l = np.mean(losses) if losses else 0.0

    # Exit type breakdown
    exit_counts: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_counts[t.exit_type] += 1

    lines.append("")
    lines.append("FULL-YEAR SUMMARY")
    lines.append("-" * 50)
    lines.append(f"  Total trades : {len(all_pnl)}")
    lines.append(f"  Win rate     : {wr:.1%}  ({len(wins)}W / {len(losses)}L)")
    lines.append(f"  Profit factor: {pf:.3f}")
    lines.append(f"  Net P&L      : ${net:+,.2f}")
    lines.append(f"  Max drawdown : ${dd:,.2f}")
    lines.append(f"  Sharpe       : {sh:.3f}  (per-trade, √252 annualized)")
    lines.append(f"  Sortino      : {so:.3f}")
    lines.append(f"  Avg win      : ${avg_w:+,.2f}   Avg loss: ${avg_l:+,.2f}")
    if avg_l:
        lines.append(f"  Win/loss $   : {abs(avg_w / avg_l):.2f}x")
    exit_str = "  Exits        : " + "  ".join(f"{k}={v}" for k, v in sorted(exit_counts.items()))
    lines.append(exit_str)

    # Monthly breakdown
    lines.append("")
    lines.append("MONTHLY BREAKDOWN")
    hdr = f"  {'MONTH':>8}  {'TRADES':>6}  {'WIN%':>5}  {'PF':>6}  {'NET_PNL':>10}  {'MAX_DD':>9}  {'SHARPE':>7}"
    lines.append(hdr)
    lines.append("  " + "-" * 65)

    best_month = max(by_month, key=lambda m: sum(by_month[m]))
    worst_month = min(by_month, key=lambda m: sum(by_month[m]))

    for month in sorted(by_month):
        mp = by_month[month]
        mw = sum(1 for p in mp if p > 0)
        mwr = mw / len(mp) if mp else 0.0
        mpf_str = f"{profit_factor(mp):6.3f}" if len([p for p in mp if p < 0]) > 0 else "   inf"
        mnet = sum(mp)
        mdd = max_drawdown(mp)
        msh = per_trade_sharpe(mp)
        marker = " ← best" if month == best_month else (" ← worst" if month == worst_month else "")
        lines.append(f"  {month:>8}  {len(mp):>6}  {mwr:>5.1%}  {mpf_str}  ${mnet:>+9,.0f}  ${mdd:>8,.0f}  {msh:>7.3f}{marker}")

    # Equity sparkline
    if cum_pnl:
        lines.append("")
        lines.append(f"EQUITY CURVE  (${cum_pnl[0]:+,.0f} → ${cum_pnl[-1]:+,.0f})")
        lines.append("  " + sparkline(cum_pnl))
        lines.append(f"  {'':62} ${cum_pnl[-1]:+,.0f}")

    lines.append("")
    lines.append("=" * 75)

    return "\n".join(lines), trade_rows, equity_rows


async def main():
    print(f"Loading bars {START_DATE.date()} → {END_DATE.date()} …", flush=True)

    bars: list[DollarBar] = []
    bars += load_bars(CSV_2025, start=START_DATE)
    print(f"  2025 CSV : {len(bars):,} bars  ({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})")

    if CSV_2026.exists():
        bars_2026 = load_bars(CSV_2026, end=END_DATE)
        if bars_2026:
            bars += bars_2026
            print(f"  2026 CSV : {len(bars_2026):,} bars  ({bars_2026[0].timestamp.date()} → {bars_2026[-1].timestamp.date()})")
    else:
        print(f"  ⚠  {CSV_2026} not found — run download_mnq_2026_ytd.py first")

    bars.sort(key=lambda b: b.timestamp)
    print(f"  Combined : {len(bars):,} bars  ({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})\n")

    print("Running backtest …", flush=True)
    trades = await run_backtest(bars)
    print(f"Completed: {len(trades)} trades\n")

    report, trade_rows, equity_rows = build_report(trades, START_DATE, END_DATE)
    print(report)

    # Save outputs
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = REPORTS_DIR / f"backtest_1year_{stamp}.txt"
    txt_path.write_text(report)
    print(f"Report  → {txt_path}")

    if trade_rows:
        csv_path = REPORTS_DIR / f"backtest_1year_{stamp}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trade_rows[0].keys())
            w.writeheader()
            w.writerows(trade_rows)
        print(f"Trades  → {csv_path}")

    if equity_rows:
        eq_path = REPORTS_DIR / f"equity_curve_1year_{stamp}.csv"
        with open(eq_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=equity_rows[0].keys())
            w.writeheader()
            w.writerows(equity_rows)
        print(f"Equity  → {eq_path}")


if __name__ == "__main__":
    asyncio.run(main())
