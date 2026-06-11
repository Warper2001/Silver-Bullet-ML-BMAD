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
  .venv/bin/python backtest_tier2_1year_validation.py [--preregistration <git-sha>]

  --preregistration is REQUIRED when the date range includes bars on or after
  2026-03-01 (the sealed holdout cutoff). Supply the SHA of a git commit that
  contains a pre-registration document in _bmad-output/preregistration*.md or
  data/sealed_holdout/PREREGISTRATION*.md. See data/sealed_holdout/ACCESS_LOG.md
  for the full access protocol.
"""
import argparse
import asyncio
import csv
import fnmatch
import logging
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytz
from unittest.mock import AsyncMock, MagicMock

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

# Sealed holdout gate (Program C Phase 0.5)
HOLDOUT_CUTOFF  = datetime(2026, 3, 1, tzinfo=timezone.utc)
ACCESS_LOG_PATH = Path("data/sealed_holdout/ACCESS_LOG.md")

_PREREG_PATTERNS = (
    "_bmad-output/preregistration*.md",
    "data/sealed_holdout/PREREGISTRATION*.md",
)


def _exit_access_denied(msg: str) -> None:
    print(f"\n{'='*70}", file=sys.stderr)
    print("SEALED HOLDOUT ACCESS DENIED", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(msg, file=sys.stderr)
    print(f"\nSee {ACCESS_LOG_PATH} for the full access protocol.", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    sys.exit(1)


def verify_preregistration(sha: str) -> None:
    """Confirm sha is a real commit and contains a pre-registration document."""
    # Guard against non-hex input being passed to git subprocesses
    if not re.fullmatch(r"[0-9a-f]{4,40}", sha, re.IGNORECASE):
        _exit_access_denied(
            f"'{sha}' is not a valid git SHA (expected 4–40 hex characters).\n"
            f"Commit your pre-registration document first, then supply that commit's SHA."
        )

    try:
        # 1. Confirm the SHA resolves to a commit object in the local repo
        result = subprocess.run(
            ["git", "cat-file", "-t", sha],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        _exit_access_denied("'git' is not installed or not on PATH — cannot verify pre-registration.")

    if result.returncode != 0 or result.stdout.strip() != "commit":
        _exit_access_denied(
            f"SHA '{sha}' was not found in the local git repo (or is not a commit).\n"
            f"Commit your pre-registration document first, then supply that commit's SHA."
        )

    # 2. Confirm the commit contains a pre-registration document
    files_result = subprocess.run(
        ["git", "show", sha, "--name-only", "--format="],
        capture_output=True, text=True, check=False,
    )
    file_lines = [ln.strip() for ln in files_result.stdout.splitlines() if ln.strip()]
    found = any(
        fnmatch.fnmatch(f, pattern)
        for f in file_lines
        for pattern in _PREREG_PATTERNS
    )
    if not found:
        _exit_access_denied(
            f"Commit {sha[:8]} contains no pre-registration document.\n"
            f"Expected a file matching one of:\n"
            + "\n".join(f"  {p}" for p in _PREREG_PATTERNS)
            + f"\n\nFiles in that commit:\n"
            + ("\n".join(f"  {f}" for f in file_lines[:20]) or "  (none)")
        )


def append_access_log(sha: str, argv: list[str]) -> None:
    """Insert a timestamped access record into the ACCESS_LOG table."""
    if not ACCESS_LOG_PATH.exists():
        _exit_access_denied(
            f"{ACCESS_LOG_PATH} does not exist.\n"
            "Run Program C Phase 0.4 to establish the sealed holdout before accessing it."
        )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    argv_str = " ".join(argv[1:]).replace("|", "\\|")  # escape pipe chars; protect table
    log_row  = f"| {date_str} | {sha} | backtest script | `{argv_str}` | pending |\n"

    # Insert after the last markdown table row so the entry stays inside the table
    lines = ACCESS_LOG_PATH.read_text().splitlines(keepends=True)
    last_table_idx = max((i for i, ln in enumerate(lines) if ln.startswith("|")), default=None)
    if last_table_idx is not None:
        lines.insert(last_table_idx + 1, log_row)
    else:
        lines.append(log_row)
    ACCESS_LOG_PATH.write_text("".join(lines))
    print(f"Access recorded → {ACCESS_LOG_PATH}  (SHA {sha[:12]})")
    print(f"  ACTION REQUIRED: commit {ACCESS_LOG_PATH} to git to make this access record permanent.")


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

    # Mock out all broker I/O and state persistence for backtest replay
    mock_client = MagicMock()
    mock_client.submit_bracket_order = AsyncMock(return_value=(None, None, None))
    mock_client.cancel_order = AsyncMock(return_value=True)
    mock_client.close_position_at_market = AsyncMock(return_value=None)
    mock_client.reconcile_state = AsyncMock(return_value=None)
    trader._ts_client = mock_client
    trader.ml_filter._log_decision = lambda *a, **kw: None

    # Suppress state persistence writes during replay
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

        trader._update_m15_choch()
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
    from src.research.strategy_core import StrategyConfig as _SC
    _sc = _SC()
    lines.append(f"  Config : MAX_PENDING_BARS={_sc.max_pending_bars}  MAX_HOLD_BARS={_sc.max_hold_bars}")
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
        lines.append(f"  {month:>8}  {len(mp):>6}  {mwr:>5.1%}  {mpf_str}  ${mnet:>9,.0f}  ${mdd:>8,.0f}  {msh:>7.3f}{marker}")

    # Add WEEKLY BREAKDOWN
    lines.append("\nWEEKLY BREAKDOWN")
    hdr_week = f"  {'WEEK':>8}  {'TRADES':>6}  {'WIN%':>5}  {'PF':>6}  {'NET_PNL':>10}  {'MAX_DD':>9}  {'SHARPE':>7}"
    lines.append(hdr_week)
    lines.append("  " + "-" * 65)
    
    by_week: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        # Group by ISO year and week number (e.g., 2026-W01)
        key = t.entry_time.astimezone(ET_TZ).strftime("%G-W%V")
        by_week[key].append(t.pnl)
        
    for w in sorted(by_week):
        wp = by_week[w]
        ww = sum(1 for p in wp if p > 0)
        wwr = ww / len(wp) if wp else 0.0
        wpf_str = f"{profit_factor(wp):6.3f}" if len([p for p in wp if p < 0]) > 0 else "   inf"
        wnet = sum(wp)
        wdd = max_drawdown(wp)
        wsh = per_trade_sharpe(wp)
        lines.append(f"  {w:>8}  {len(wp):>6}  {wwr:>5.1%}  {wpf_str}  ${wnet:>9,.0f}  ${wdd:>8,.0f}  {wsh:>7.3f}")

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
    parser = argparse.ArgumentParser(description="Tier 2 MNQ 1-year validation backtest")
    parser.add_argument(
        "--preregistration",
        metavar="GIT_SHA",
        default=None,
        help=(
            "SHA of a git commit containing a pre-registration document. "
            "Required when the date range includes data on or after "
            f"{HOLDOUT_CUTOFF.date()} (sealed holdout cutoff)."
        ),
    )
    args = parser.parse_args()

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

    sys.stdout.flush()  # ensure bar-count lines print before any gate error on stderr

    # Sealed holdout gate — must run after bars are sorted
    if any(b.timestamp >= HOLDOUT_CUTOFF for b in bars):
        if args.preregistration is None:
            _exit_access_denied(
                f"This run includes bars on or after {HOLDOUT_CUTOFF.date()} (sealed holdout).\n\n"
                "To access holdout data you must:\n"
                "  1. Commit a pre-registration document to git specifying your hypothesis,\n"
                "     decision rule, and frozen parameters.\n"
                "  2. Re-run with:  --preregistration <commit-sha>\n\n"
                "Pre-registration file must match one of:\n"
                + "\n".join(f"  {p}" for p in _PREREG_PATTERNS)
            )
        verify_preregistration(args.preregistration)
        append_access_log(args.preregistration, sys.argv)

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
