"""
S26 Subgroup Analysis on Full 1-Year Backtest
==============================================
Runs the 1-year backtest with:
  - bearish_only = False   (both directions)
  - tuesday_exclusion = False  (Tue allowed)
  - enable_kill_zone_filter = False  (all hours captured)

Then applies the S26 kill-zone subgroup split:
  KZ_HOURS = {10, 11, 14}  (10:00-12:00 ET + 14:00-15:00 ET)
  BLOCKED_DOW = {0}         (Monday)

Usage:
  .venv/bin/python backtest_s26_subgroup_analysis.py

Writes:
  data/reports/s26_subgroup_YYYYMMDD_HHMMSS.txt
  data/reports/s26_subgroup_YYYYMMDD_HHMMSS_trades.csv
"""

import asyncio
import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytz
import yaml
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

# ── Config: write temp YAML before importing trader (it reads YAML at import time)
_BASE_YAML = Path("strategy_config.yaml")
_TMP_CFG   = Path("/root/.claude/jobs/28ad3e8c/s26_analysis_config.yaml")

with open(_BASE_YAML) as _f:
    _cfg = yaml.safe_load(_f)

_cfg["bearish_only"]            = False
_cfg["tuesday_exclusion"]       = False
_cfg["enable_kill_zone_filter"] = False

with open(_TMP_CFG, "w") as _f:
    yaml.dump(_cfg, _f)

os.environ["STRATEGY_CONFIG_PATH"] = str(_TMP_CFG)

# Suppress all logging during backtest replay
logging.disable(logging.CRITICAL)

import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

ET_TZ = pytz.timezone("US/Eastern")

# S26 filter definition (pre-committed, exact — SHA a97b21c)
KZ_HOURS    = {10, 11, 14}   # 10:00-12:00 ET  +  14:00-15:00 ET
BLOCKED_DOW = {0}             # Monday (0=Mon)

CSV_2025    = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026    = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
REPORTS_DIR = Path("data/reports")

START_DATE = datetime(2025, 5, 19, tzinfo=timezone.utc)
END_DATE   = datetime(2026, 5, 19, 23, 59, 59, tzinfo=timezone.utc)


# ── Stats helpers ──────────────────────────────────────────────────────────────

def profit_factor(pnl: list[float]) -> float:
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    return gp / gl if gl > 0 else float("inf")

def per_trade_sharpe(pnl: list[float]) -> float:
    if len(pnl) < 2:
        return float("nan")
    a = np.array(pnl, dtype=float)
    return float(a.mean() / a.std() * np.sqrt(252)) if a.std() > 0 else float("nan")

def max_drawdown(pnl: list[float]) -> float:
    cum, peak, dd = 0.0, 0.0, 0.0
    for p in pnl:
        cum += p
        peak = max(peak, cum)
        dd   = max(dd, peak - cum)
    return dd

def stats_row(label: str, trades_subset: list, total_n: int) -> str:
    n = len(trades_subset)
    if n == 0:
        return f"  {label:<44s}  N=  0  (  0%)  --\n"
    pnl  = [t["pnl"] for t in trades_subset]
    pf   = profit_factor(pnl)
    wr   = sum(1 for p in pnl if p > 0) / n * 100
    net  = sum(pnl)
    sh   = per_trade_sharpe(pnl)
    dd   = max_drawdown(pnl)
    pct  = n / total_n * 100 if total_n else 0
    exits = defaultdict(int)
    for t in trades_subset:
        exits[t["exit_type"]] += 1
    exits_str = " ".join(f"{k}={v}" for k, v in sorted(exits.items()))
    return (
        f"  {label:<44s}  N={n:>3d} ({pct:>4.0f}%)  "
        f"PF={pf:>6.4f}  WR={wr:>5.1f}%  Net=${net:>+8.0f}  "
        f"Sharpe={sh:>6.3f}  MaxDD=${dd:>7.0f}  [{exits_str}]\n"
    )


# ── Data loader ────────────────────────────────────────────────────────────────

def load_bars() -> list[DollarBar]:
    bars: list[DollarBar] = []
    for path in [CSV_2025, CSV_2026]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
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


# ── Backtest runner ────────────────────────────────────────────────────────────

async def run_backtest(bars: list[DollarBar]) -> list[dict]:
    trader = Tier2StreamingTrader()

    mock_client = MagicMock()
    mock_client.submit_bracket_order      = AsyncMock(return_value=(None, None, None))
    mock_client.cancel_order              = AsyncMock(return_value=True)
    mock_client.close_position_at_market  = AsyncMock(return_value=None)
    mock_client.reconcile_state           = AsyncMock(return_value=None)
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
        trades.append({
            "entry_time": t.entry_time.isoformat(),
            "exit_time":  t.exit_time.isoformat(),
            "direction":  t.direction,
            "exit_type":  t.exit_type,
            "bars_held":  t.bars_held,
            "pnl":        round(t.pnl, 2),
            "hour_et":    entry_et.hour,
            "dow_et":     entry_et.weekday(),   # 0=Mon … 6=Sun
            "month":      entry_et.strftime("%Y-%m"),
            "kill_zone":  entry_et.hour in KZ_HOURS,
        })
    return trades


# ── S26 subgroup breakdown ─────────────────────────────────────────────────────

def is_s26_eligible(t: dict) -> bool:
    return t["hour_et"] in KZ_HOURS and t["dow_et"] not in BLOCKED_DOW


def build_report(trades: list[dict], ts_label: str) -> str:
    N = len(trades)
    if N == 0:
        return "No trades generated.\n"

    SEP  = "=" * 105
    SEP2 = "─" * 105
    lines = [
        SEP,
        "S26 SUBGROUP ANALYSIS — 1-Year Backtest (bearish_only=False, tuesday_exclusion=False, no KZ gate)",
        f"  Generated : {ts_label}",
        f"  Period    : {START_DATE.date()} → {END_DATE.date()}",
        f"  Total trades : {N}",
        SEP,
    ]

    s26_yes = [t for t in trades if is_s26_eligible(t)]
    s26_no  = [t for t in trades if not is_s26_eligible(t)]
    bearish = [t for t in trades if t["direction"] == "bearish"]
    bullish = [t for t in trades if t["direction"] == "bullish"]

    lines += ["", "── OVERALL (all trades) " + "─" * 81]
    lines.append(stats_row("All trades", trades, N))
    lines.append(stats_row("  Bearish", bearish, N))
    lines.append(stats_row("  Bullish", bullish, N))

    lines += ["", "── S26 FILTER SPLIT ─────────────────────────────────────────────────────────────────────────────"]
    lines.append("   KZ_HOURS={10,11,14}  →  10:00-12:00 ET + 14:00-15:00 ET   |   BLOCKED_DOW={0} (Mon)")
    lines.append(stats_row("S26 ELIGIBLE  (KZ hours + not Mon)", s26_yes, N))
    lines.append(stats_row("S26 EXCLUDED  (off-hours or Mon)",   s26_no,  N))

    lines += ["", "── S26 ELIGIBLE — DIRECTION SPLIT ───────────────────────────────────────────────────────────────"]
    lines.append(stats_row("  S26 eligible · Bearish", [t for t in s26_yes if t["direction"] == "bearish"], N))
    lines.append(stats_row("  S26 eligible · Bullish", [t for t in s26_yes if t["direction"] == "bullish"], N))

    lines += ["", "── KZ HOUR BREAKDOWN (not Mon, any direction) ───────────────────────────────────────────────────"]
    for h, label in [(10, "10:00-11:00 ET (AM open)"),
                     (11, "11:00-12:00 ET (AM mid)"),
                     (14, "14:00-15:00 ET (PM open)")]:
        sub = [t for t in trades if t["hour_et"] == h and t["dow_et"] not in BLOCKED_DOW]
        lines.append(stats_row(f"  {label}", sub, N))

    lines += ["", "── S26 EXCLUDED — HOUR BREAKDOWN (non-KZ hours) ─────────────────────────────────────────────────"]
    non_kz_hours = sorted({t["hour_et"] for t in s26_no})
    for h in non_kz_hours:
        sub = [t for t in s26_no if t["hour_et"] == h]
        lines.append(stats_row(f"  Hour {h:02d}:00-{h+1:02d}:00 ET", sub, N))

    lines += ["", "── S26 ELIGIBLE — DAY-OF-WEEK ───────────────────────────────────────────────────────────────────"]
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    for dow in range(5):
        sub = [t for t in s26_yes if t["dow_et"] == dow]
        if sub or dow == 0:
            lines.append(stats_row(f"  {dow_names[dow]}", sub, N))

    lines += ["", "── S26 ELIGIBLE — MONTHLY BREAKDOWN ─────────────────────────────────────────────────────────────"]
    by_month: dict[str, list] = defaultdict(list)
    for t in s26_yes:
        by_month[t["month"]].append(t)
    for month in sorted(by_month):
        lines.append(stats_row(f"  {month}", by_month[month], N))

    # Summary comparison
    pf_all  = profit_factor([t["pnl"] for t in trades])
    pf_s26  = profit_factor([t["pnl"] for t in s26_yes])
    sh_all  = per_trade_sharpe([t["pnl"] for t in trades])
    sh_s26  = per_trade_sharpe([t["pnl"] for t in s26_yes])
    net_all = sum(t["pnl"] for t in trades)
    net_s26 = sum(t["pnl"] for t in s26_yes)
    n_s26   = len(s26_yes)
    pct_s26 = n_s26 / N * 100 if N else 0

    days = (END_DATE - START_DATE).days
    freq = n_s26 / days

    lines += [
        "",
        "── SUMMARY COMPARISON ───────────────────────────────────────────────────────────────────────────",
        f"   Unfiltered base  : N={N:>3d}       PF={pf_all:.4f}  Sharpe={sh_all:.3f}  Net=${net_all:>+8.0f}",
        f"   S26 subgroup     : N={n_s26:>3d} ({pct_s26:.0f}%)  PF={pf_s26:.4f}  Sharpe={sh_s26:.3f}  Net=${net_s26:>+8.0f}",
        f"   PF lift          : {pf_s26 - pf_all:+.4f}",
        f"   Sharpe lift      : {sh_s26 - sh_all:+.3f}",
        "",
        f"   Expected live freq (S26 filtered): {freq:.3f} trades/day",
        f"     → N=20 in ~{round(20/freq) if freq > 0 else 'inf'} days at backtest rate",
        f"     → N=20 in ~{round(20/(freq*0.6)) if freq > 0 else 'inf'} days at conservative (60%) rate",
        SEP,
    ]

    return "\n".join(lines) + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars...", flush=True)
    bars = load_bars()
    print(f"  {len(bars):,} bars ({START_DATE.date()} – {END_DATE.date()})", flush=True)

    print("Running backtest (bearish_only=False, tuesday_exclusion=False)...", flush=True)
    trades = asyncio.run(run_backtest(bars))
    print(f"  Done — {len(trades)} trades", flush=True)

    now_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    report    = build_report(trades, now_label)
    print("\n" + report, flush=True)

    REPORTS_DIR.mkdir(exist_ok=True)
    txt_path = REPORTS_DIR / f"s26_subgroup_{now_label}.txt"
    csv_path = REPORTS_DIR / f"s26_subgroup_{now_label}_trades.csv"

    txt_path.write_text(report)
    if trades:
        keys = list(trades[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(trades)

    print(f"Saved: {txt_path}", flush=True)
    print(f"Saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
