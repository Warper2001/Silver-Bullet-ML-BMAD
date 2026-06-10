"""
S27 Hold-Period Exit Management — In-Sample Exploratory Backtest
Pre-registration NOT yet filed (blocked until S25 N>=20 live trades).
Exploratory in-sample only. Same data window as S25+CHoCH backtest.

Parity gate: with all S27 flags False, must reproduce the S25+Epic5 baseline.
Baseline is N=78, PF=1.0728 with git-committed YAML (SL=5.0, TP=6.0, no KZ, ml=0.0).
Epic 5 code changed this from the pre-Epic-5 reference (N=49, PF=1.5197 at 401f938).
"""
from __future__ import annotations

import asyncio
import csv
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytz
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

# Suppress all logging during replay
logging.disable(logging.CRITICAL)

import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader, _build_strategy_config
from src.research.strategy_core import StrategyConfig
from src.data.models import DollarBar

ET_TZ = pytz.timezone("US/Eastern")

# ── Data window ────────────────────────────────────────────────────────────────
# In-sample window matching backtest_tier2_1year_validation.py START_DATE.
# Sealed holdout starts 2026-03-01 — do NOT access data after 2026-02-28.
IS_START = datetime(2025, 5, 19, tzinfo=timezone.utc)
IS_END   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

CSV_2025     = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026_YTD = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

# ── Parity gate spec ───────────────────────────────────────────────────────────
# Reference: S25+CHoCH strategy with git-committed strategy_config.yaml (SL=5.0, TP=6.0,
# enable_kill_zone_filter=false, ml_threshold=0.0) run through current HEAD code.
#
# NOTE: The original pre-Epic-5 reference (commit 401f938) produced N=49, PF=1.5197.
# Epic 5 features (consistency rule, trailing DD) changed circuit-breaker firing
# frequency; the consistency rule reduces contracts on big-win days, making per-trade
# losses smaller and allowing more entries on previously-halted days. The current
# deployed code baseline is N=78, PF=1.0728 for the IS window.
# Tolerance ±0.020 covers floating-point and exact date-slice rounding differences.
EXPECTED_BASELINE_N  = 78
EXPECTED_BASELINE_PF = 1.0728
PARITY_TOLERANCE_PF  = 0.020


# ── Bar loading ────────────────────────────────────────────────────────────────

def load_bars(csv_path: Path, start: datetime, end: datetime) -> list[DollarBar]:
    bars = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts_str = row.get("timestamp") or row.get("Datetime") or row.get("datetime")
            if ts_str is None:
                raise ValueError(f"No timestamp column found in {csv_path}")
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < start or ts > end:
                continue
            bars.append(DollarBar.model_construct(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(float(row["volume"])),
                notional_value=float(row.get("notional", 0)),
                is_forward_filled=False,
            ))
    return bars


# ── Profit factor helper ───────────────────────────────────────────────────────

def profit_factor(pnl: list[float]) -> float:
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    return round(gp / gl, 4) if gl else float("inf")


# ── Backtest runner ────────────────────────────────────────────────────────────

async def run_backtest(bars: list[DollarBar], config: StrategyConfig) -> list:
    trader = Tier2StreamingTrader()
    # Override the strategy config for this run
    trader._strategy_config = config

    # Mock all broker I/O
    mock_client = MagicMock()
    mock_client.submit_bracket_order = AsyncMock(return_value=(None, None, None))
    mock_client.place_exit_orders = AsyncMock(return_value=(None, None))
    mock_client.cancel_order = AsyncMock(return_value=True)
    mock_client.close_position_at_market = AsyncMock(return_value=None)
    mock_client.reconcile_state = AsyncMock(return_value=None)
    trader._ts_client = mock_client
    trader.ml_filter._log_decision = lambda *a, **kw: None

    # Suppress state persistence
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


def run_sync(bars: list[DollarBar], config: StrategyConfig) -> list:
    return asyncio.run(run_backtest(bars, config))


# ── Results analysis ───────────────────────────────────────────────────────────

class RunResult(NamedTuple):
    label: str
    n: int
    win_rate: float
    pf: float
    avg_pnl: float
    exit_counts: dict
    is_primary: bool = False


def analyse(trades: list, label: str, is_primary: bool = False) -> RunResult:
    if not trades:
        return RunResult(label=label, n=0, win_rate=0.0, pf=0.0, avg_pnl=0.0,
                         exit_counts={}, is_primary=is_primary)
    pnl_list = [t.pnl for t in trades]
    wins = [p for p in pnl_list if p > 0]
    pf = profit_factor(pnl_list)
    wr = len(wins) / len(pnl_list) * 100
    avg = sum(pnl_list) / len(pnl_list)
    exit_counts: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_counts[t.exit_type.upper()] += 1
    return RunResult(
        label=label, n=len(trades), win_rate=wr, pf=pf, avg_pnl=avg,
        exit_counts=dict(exit_counts), is_primary=is_primary,
    )


# ── Config factory ─────────────────────────────────────────────────────────────

def make_config(**overrides) -> StrategyConfig:
    """Build a StrategyConfig from the git-committed strategy_config.yaml (not working tree).

    Uses git-committed version so the study always runs against the same frozen
    S25 reference config regardless of uncommitted working-tree modifications.
    """
    from src.research.config_loader import load_strategy_config
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:strategy_config.yaml"],
            text=True, stderr=subprocess.DEVNULL,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(committed)
            tmp = Path(f.name)
        try:
            base = load_strategy_config(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:
        base = _build_strategy_config()
    return replace(base, **overrides)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 75)
    print("S27 HOLD-PERIOD EXIT MANAGEMENT — IN-SAMPLE EXPLORATORY BACKTEST")
    print(f"  In-sample window : {IS_START.date()} to {IS_END.date()}")
    print(f"  (Sealed holdout 2026-03-01+ NOT accessed)")
    print("=" * 75)
    print()

    # Load data
    print("Loading data...")
    bars: list[DollarBar] = []
    bars.extend(load_bars(CSV_2025, IS_START, IS_END))
    if CSV_2026_YTD.exists():
        bars.extend(load_bars(CSV_2026_YTD, IS_START, IS_END))
    bars.sort(key=lambda b: b.timestamp)
    print(f"  Loaded {len(bars):,} 1-min bars from {bars[0].timestamp.date()} to {bars[-1].timestamp.date()}")
    print()

    # ── Parity gate ──────────────────────────────────────────────────────────
    print("Running PARITY GATE (all S27 flags=False)...")
    baseline_config = make_config()  # all defaults: enable_breakeven_stop=False, enable_trailing_stop=False
    baseline_trades = run_sync(bars, baseline_config)
    baseline_result = analyse(baseline_trades, "BASELINE (flags=False)")
    baseline_pf = baseline_result.pf
    baseline_n  = baseline_result.n

    print(f"  Actual baseline : N={baseline_n}, PF={baseline_pf:.4f}")
    print(f"  Expected        : N={EXPECTED_BASELINE_N}, PF={EXPECTED_BASELINE_PF:.4f} (±{PARITY_TOLERANCE_PF})")

    pf_delta = abs(baseline_pf - EXPECTED_BASELINE_PF)
    if pf_delta > PARITY_TOLERANCE_PF:
        print()
        print(f"  PARITY GATE FAIL — PF delta {pf_delta:.4f} exceeds tolerance {PARITY_TOLERANCE_PF}")
        print("  Halting: investigate data/config mismatch before running variant grid.")
        print()
        sys.exit(1)

    print()
    print("  PARITY GATE PASS")
    print()

    # ── Variant grid ─────────────────────────────────────────────────────────
    grid: list[tuple[StrategyConfig, str, bool]] = [
        # (config, label, is_primary)
        (make_config(), "Baseline (all-off)", False),

        # Breakeven only
        (make_config(enable_breakeven_stop=True, breakeven_trigger_r=1.5), "BE only  r=1.5", False),
        (make_config(enable_breakeven_stop=True, breakeven_trigger_r=2.0), "BE only  r=2.0  [PRIMARY]", True),
        (make_config(enable_breakeven_stop=True, breakeven_trigger_r=3.0), "BE only  r=3.0", False),

        # Trailing only
        (make_config(enable_trailing_stop=True, trailing_stop_mult=1.5), "Trail only m=1.5", False),
        (make_config(enable_trailing_stop=True, trailing_stop_mult=2.0), "Trail only m=2.0", False),
        (make_config(enable_trailing_stop=True, trailing_stop_mult=2.5), "Trail only m=2.5", False),

        # Combined
        (make_config(enable_breakeven_stop=True, breakeven_trigger_r=2.0,
                     enable_trailing_stop=True, trailing_stop_mult=1.5),
         "BE r=2.0 + Trail m=1.5", False),
    ]

    print(f"Running {len(grid)} variants...")
    results: list[RunResult] = []
    for cfg, label, is_primary in grid:
        trades = run_sync(bars, cfg)
        r = analyse(trades, label, is_primary=is_primary)
        results.append(r)
        marker = " *** PRIMARY ***" if is_primary else ""
        print(f"  {label:36s}  N={r.n:3d}  PF={r.pf:.4f}  WR={r.win_rate:.1f}%{marker}")

    print()

    # ── Results table ─────────────────────────────────────────────────────────
    EXIT_KEYS = ["SL", "TP", "TIME_STOP", "BREAKEVEN_SL", "TRAILING_SL"]

    print("=" * 110)
    print("S27 VARIANT GRID — RESULTS TABLE")
    print("=" * 110)
    hdr = (f"{'Config':<36}  {'N':>3}  {'WR%':>5}  {'PF':>7}  {'AvgPnL':>8}  "
           f"{'dPF':>7}  {'SL':>4}  {'TP':>4}  {'TIME':>4}  {'BE_SL':>5}  {'TR_SL':>5}")
    print(hdr)
    print("-" * 110)

    for r in results:
        dpf = r.pf - baseline_pf
        ec = r.exit_counts
        row = (
            f"{'>>> ' + r.label + ' <<<' if r.is_primary else r.label:<36}  "
            f"{r.n:3d}  "
            f"{r.win_rate:5.1f}  "
            f"{r.pf:7.4f}  "
            f"{r.avg_pnl:8.2f}  "
            f"{dpf:+7.4f}  "
            f"{ec.get('SL', 0):4d}  "
            f"{ec.get('TP', 0):4d}  "
            f"{ec.get('TIME_STOP', 0) + ec.get('TIME', 0):4d}  "
            f"{ec.get('BREAKEVEN_SL', 0):5d}  "
            f"{ec.get('TRAILING_SL', 0):5d}"
        )
        print(row)

    print("=" * 110)
    print()

    # ── Primary cell summary ─────────────────────────────────────────────────
    primary = next((r for r in results if r.is_primary), None)
    if primary:
        print("PRIMARY SPEC CELL: BE only r=2.0 (breakeven_trigger_r=2.0, trailing_stop=off)")
        print(f"  N={primary.n}  WR={primary.win_rate:.1f}%  PF={primary.pf:.4f}  "
              f"AvgPnL=${primary.avg_pnl:.2f}  dPF={primary.pf - baseline_pf:+.4f}")
        ec = primary.exit_counts
        print(f"  Exits: SL={ec.get('SL',0)}  TP={ec.get('TP',0)}  "
              f"TIME_STOP={ec.get('TIME_STOP',0) + ec.get('TIME',0)}  "
              f"BREAKEVEN_SL={ec.get('BREAKEVEN_SL',0)}  "
              f"TRAILING_SL={ec.get('TRAILING_SL',0)}")
        print()

    # ── Baseline exit breakdown ───────────────────────────────────────────────
    baseline = results[0]
    ec = baseline.exit_counts
    print("BASELINE EXIT BREAKDOWN:")
    print(f"  SL={ec.get('SL',0)}  TP={ec.get('TP',0)}  "
          f"TIME_STOP={ec.get('TIME_STOP',0) + ec.get('TIME',0)}  "
          f"BREAKEVEN_SL={ec.get('BREAKEVEN_SL',0)}  "
          f"TRAILING_SL={ec.get('TRAILING_SL',0)}")
    print()
    print("NOTE: This is an exploratory in-sample study. No pre-registration has been filed.")
    print("      Do not act on these results until S25 reaches N>=20 live trades.")
    print()


if __name__ == "__main__":
    main()
