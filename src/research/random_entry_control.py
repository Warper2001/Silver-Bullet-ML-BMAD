"""S12 Random-Entry Control Test — Program C Phase 1 Falsification.

Establishes the null distribution of Profit Factor for a direction-matched
random-entry strategy using the same gates, exits, and data as the real
Silver-Bullet strategy. Compares actual strategy PF=0.937 (2025 training window)
against this distribution to produce the pivot-vs-survive verdict.

Pre-registration: _bmad-output/preregistration_s12_random_entry.md
Sealed commit: 7ffb3e0b712f4265478b21ae0e583e57f1249f4e

Usage:
    .venv/bin/python src/research/random_entry_control.py
    .venv/bin/python src/research/random_entry_control.py --n 10 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from math import sqrt
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Strategy-core imports — read-only, no modifications to these modules
from src.research.backtest_engine import BacktestEngine, _H1_BUFFER_BARS
from src.research.strategy_core import (
    POINT_VALUE_USD,
    Direction,
    EntryDecision,
    ExitReason,
    StrategyConfig,
    calc_atr,
    calc_profit_factor,
    calc_sharpe,
    check_exit,
    detect_liquidity_sweep,
    resample_to_h1,
    volatility_regime_filter,
)

# ── Constants ─────────────────────────────────────────────────────────────────

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
N_SIMULATIONS = 100
SEED_RANGE = range(N_SIMULATIONS)  # seeds 0–99
# Baseline PF from Epic 1 BacktestEngine full-year 2025 run
STRATEGY_PF_BASELINE = 0.937
OUTPUT_DIR = Path("_bmad-output")

# ── Data types ────────────────────────────────────────────────────────────────


class SimTrade(NamedTuple):
    """One completed trade from a random simulation."""

    pnl_usd: float
    exit_reason: str  # "TP" / "SL" / "TIME_STOP"
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    trade_date: date


class SimResult(NamedTuple):
    """Aggregate metrics for one simulation run."""

    seed: int
    pf: float
    wr: float
    sharpe: float
    trade_count: int


# ── Bar loading (mirrors BacktestEngine._load_bars) ──────────────────────────


def _load_bars(path: str) -> pd.DataFrame:
    """Load 1-min bars with same schema as BacktestEngine (AR9, AR19)."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    df = df.drop(columns=["notional"], errors="ignore")
    df = df.sort_index()
    dup = df.index.duplicated(keep="first")
    if dup.any():
        df = df[~dup]
    return df


# ── Core simulator ────────────────────────────────────────────────────────────


class RandomEntrySimulator:
    """Replicates the Tier-2 simulation loop with random entries instead of FVG.

    Shares all entry gates with the real strategy:
      - Tuesday filter
      - Daily circuit-breaker (max_daily_loss)
      - Volatility regime filter (cached per H1)
      - H1 sweep gate (cached per H1)
      - Bearish-only gate

    FVG detection is replaced by a coin flip with calibrated probability p_enter.
    Exits use strategy_core.check_exit() unchanged.
    Entries fill immediately at bar.close (no limit-order queue).
    """

    def __init__(self, bars: pd.DataFrame, config: StrategyConfig) -> None:
        self.bars = bars
        self.config = config
        self.full_h1 = resample_to_h1(bars)
        self._n = len(bars)

    def count_candidate_bars(self) -> int:
        """Count bars that pass all gates except FVG detection.

        Used to calibrate p_enter = real_trades / candidate_bars.
        """
        count = 0
        bars = self.bars
        config = self.config
        full_h1 = self.full_h1
        n = self._n

        last_h1_ts: pd.Timestamp | None = None
        vol_ok_cached = True
        sweep_cached = None
        active = False
        bars_held = 0
        daily_pnl = 0.0
        daily_halted = False
        last_date: date | None = None

        for i in range(n):
            bar_ts: pd.Timestamp = bars.index[i]

            h1_boundary = bar_ts.replace(minute=0, second=0, microsecond=0)
            if h1_boundary != last_h1_ts:
                last_h1_ts = h1_boundary
                h1_idx = int(full_h1.index.searchsorted(h1_boundary))
                h1_start = max(0, h1_idx - _H1_BUFFER_BARS // 60)
                h1_slice = full_h1.iloc[h1_start:h1_idx]
                h1_bars = h1_slice if len(h1_slice) > 0 else None
                if h1_bars is not None and len(h1_bars) >= 20:
                    try:
                        vol_ok_cached = volatility_regime_filter(h1_bars, config)
                    except ValueError:
                        vol_ok_cached = True
                else:
                    vol_ok_cached = True
                min_rows = config.h1_sweep_lookback + 5
                if h1_bars is not None and len(h1_bars) >= min_rows:
                    try:
                        sweep_cached = detect_liquidity_sweep(h1_bars, config)
                    except ValueError:
                        sweep_cached = None
                else:
                    sweep_cached = None

            if active:
                bars_held += 1
                if bars_held >= config.max_hold_bars:
                    active = False
                    bars_held = 0
                continue

            if i < 20:
                continue
            if bar_ts.weekday() == 1:
                continue

            bar_date = bar_ts.date()
            if last_date != bar_date:
                daily_pnl = 0.0
                daily_halted = False
                last_date = bar_date
            if daily_halted:
                continue
            if daily_pnl <= config.max_daily_loss:
                daily_halted = True
                continue

            if not vol_ok_cached:
                continue
            if sweep_cached is None:
                continue
            if config.bearish_only and sweep_cached.direction != Direction.BEARISH:
                continue

            count += 1

        return count

    def run(self, seed: int, p_enter: float) -> list[SimTrade]:
        """Run one simulation with the given random seed.

        Parameters
        ----------
        seed:
            numpy Generator seed (0–99 in the pre-registered range).
        p_enter:
            Per-candidate-bar entry probability calibrated to match the real
            strategy's observed trade count (real_trades / candidate_bars).

        Returns
        -------
        list[SimTrade]
            Completed trades in chronological order.
        """
        rng = np.random.default_rng(seed)
        bars = self.bars
        config = self.config
        full_h1 = self.full_h1
        n = self._n
        trades: list[SimTrade] = []

        last_h1_ts: pd.Timestamp | None = None
        vol_ok_cached = True
        sweep_cached = None
        h1_bars: pd.DataFrame | None = None
        h1_atr: float = 0.0

        active: EntryDecision | None = None
        active_ts: pd.Timestamp | None = None
        bars_held = 0

        daily_pnl: float = 0.0
        daily_halted = False
        last_date: date | None = None

        for i in range(n):
            bar_ts: pd.Timestamp = bars.index[i]

            # ── H1 boundary: refresh cached state ─────────────────────────
            h1_boundary = bar_ts.replace(minute=0, second=0, microsecond=0)
            if h1_boundary != last_h1_ts:
                last_h1_ts = h1_boundary
                h1_idx = int(full_h1.index.searchsorted(h1_boundary))
                h1_start = max(0, h1_idx - _H1_BUFFER_BARS // 60)
                h1_slice = full_h1.iloc[h1_start:h1_idx]
                h1_bars = h1_slice if len(h1_slice) > 0 else None
                h1_atr = _compute_h1_atr(h1_bars)
                if h1_bars is not None and len(h1_bars) >= 20:
                    try:
                        vol_ok_cached = volatility_regime_filter(h1_bars, config)
                    except ValueError:
                        vol_ok_cached = True
                else:
                    vol_ok_cached = True
                min_rows = config.h1_sweep_lookback + 5
                if h1_bars is not None and len(h1_bars) >= min_rows:
                    try:
                        sweep_cached = detect_liquidity_sweep(h1_bars, config)
                    except ValueError:
                        sweep_cached = None
                else:
                    sweep_cached = None

            bar = bars.iloc[i]

            # ── Advance active trade ───────────────────────────────────────
            if active is not None:
                bars_held += 1
                exit_dec = check_exit(bar, active, bars_held, config)
                if exit_dec is not None:
                    ep = exit_dec.exit_price
                    points = active.entry_price - ep  # bearish only
                    pnl = (
                        points * POINT_VALUE_USD * active.contracts
                        - config.commission_per_roundtrip
                    )
                    trades.append(
                        SimTrade(
                            pnl_usd=round(pnl, 2),
                            exit_reason=exit_dec.reason.value,
                            entry_ts=active_ts,  # type: ignore[arg-type]
                            exit_ts=bar_ts,
                            trade_date=active_ts.date(),  # type: ignore[union-attr]
                        )
                    )
                    daily_pnl += pnl
                    active = None
                    bars_held = 0
                continue  # don't enter new trade on same bar as exit

            # ── Entry detection ────────────────────────────────────────────
            if i < 20:
                continue
            if bar_ts.weekday() == 1:
                continue

            bar_date = bar_ts.date()
            if last_date != bar_date:
                daily_pnl = 0.0
                daily_halted = False
                last_date = bar_date
            if daily_halted:
                continue
            if daily_pnl <= config.max_daily_loss:
                daily_halted = True
                continue

            if not vol_ok_cached:
                continue
            if sweep_cached is None:
                continue
            if config.bearish_only and sweep_cached.direction != Direction.BEARISH:
                continue

            # Coin flip replaces detect_fvg + make_entry_decision
            if rng.random() >= p_enter:
                continue

            # Construct random bearish entry at bar.close
            entry_price = float(bar["close"])
            m1_buf = bars.iloc[max(0, i - 19) : i + 1]
            m1_atr = calc_atr(m1_buf)
            # Use minimum gate value as gap_size proxy (conservative)
            gap_size = config.atr_threshold * m1_atr
            if gap_size <= 0:
                continue

            sl_price = entry_price + config.sl_multiplier * gap_size
            tp_price = entry_price - config.tp_multiplier * gap_size

            active = EntryDecision(
                direction=Direction.BEARISH,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                contracts=config.contracts_per_trade,
            )
            active_ts = bar_ts
            bars_held = 0

        return trades


# ── H1 ATR helper (mirrors BacktestEngine._compute_h1_atr) ───────────────────


def _compute_h1_atr(h1_bars: pd.DataFrame | None) -> float:
    if h1_bars is None or len(h1_bars) < 2:
        return 0.0
    return calc_atr(h1_bars)


# ── Metrics ───────────────────────────────────────────────────────────────────


def _compute_metrics(trades: list[SimTrade]) -> tuple[float, float, float]:
    """Return (PF, WR, Sharpe) for a list of SimTrades."""
    if not trades:
        return 0.0, 0.0, 0.0

    pnls = [t.pnl_usd for t in trades]
    pf = calc_profit_factor(pnls)

    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls)

    # Aggregate to daily PnL for Sharpe (calc_sharpe takes daily returns)
    daily: dict[date, float] = {}
    for t in trades:
        daily[t.trade_date] = daily.get(t.trade_date, 0.0) + t.pnl_usd
    sharpe = calc_sharpe(list(daily.values()))

    return pf, wr, sharpe


# ── Distribution stats ────────────────────────────────────────────────────────


def _percentile_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _percentile_rank(values: list[float], target: float) -> float:
    """What fraction of values is target strictly greater than."""
    return sum(1 for v in values if target > v) / len(values)


# ── Verdict ───────────────────────────────────────────────────────────────────


def _verdict(strategy_pf: float, pf_stats: dict[str, float]) -> str:
    median = pf_stats["median"]
    p90 = pf_stats["p90"]
    if strategy_pf < median:
        return "PIVOT"
    if strategy_pf > p90:
        return "PATTERNS SURVIVE"
    return "AMBIGUOUS = TREATED AS FAIL = PIVOT"


# ── Report writer ─────────────────────────────────────────────────────────────


def _write_verdict_report(
    results: list[SimResult],
    strategy_pf: float,
    pf_rank: float,
    verdict: str,
    pf_stats: dict[str, float],
    wr_stats: dict[str, float],
    sharpe_stats: dict[str, float],
    output_dir: Path,
) -> Path:
    today = pd.Timestamp.now().strftime("%Y%m%d")
    out_path = output_dir / f"s12_verdict_{today}.md"

    lines = [
        f"# S12 Random-Entry Control Verdict — {today}",
        "",
        "## Pre-Registration",
        "Sealed at git SHA: `7ffb3e0b712f4265478b21ae0e583e57f1249f4e`",
        "",
        "## Simulation Parameters",
        f"- N = {len(results)} simulations, seeds 0–{len(results) - 1}",
        "- Data: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`",
        "- Entry gates: H1 sweep, vol regime, daily circuit-breaker, bearish-only (shared with real strategy)",
        "- Entry decision: uniform random coin flip (calibrated to match strategy trade frequency)",
        "- Entry price: bar.close (immediate fill proxy)",
        "- Exit logic: `strategy_core.check_exit()` unchanged (SL → TP → TIME_STOP)",
        "",
        "## Null Distribution (N={n} random simulations)".format(n=len(results)),
        "",
        "| Metric | Min | P25 | Median | P75 | P90 | Max |",
        "|---|---|---|---|---|---|---|",
        "| PF | {min:.3f} | {p25:.3f} | {median:.3f} | {p75:.3f} | {p90:.3f} | {max:.3f} |".format(**pf_stats),
        "| WR | {min:.3f} | {p25:.3f} | {median:.3f} | {p75:.3f} | {p90:.3f} | {max:.3f} |".format(**wr_stats),
        "| Sharpe | {min:.3f} | {p25:.3f} | {median:.3f} | {p75:.3f} | {p90:.3f} | {max:.3f} |".format(**sharpe_stats),
        "",
        "## Strategy Result",
        f"- PF: {strategy_pf} (BacktestEngine full-year 2025 run, Epic 1)",
        f"- PF Percentile Rank: {pf_rank:.1%} (strategy PF exceeds {pf_rank:.1%} of random simulations)",
        "",
        "## Decision Rule",
        "Per `_bmad-output/problem-solution-2026-05-20.md`:",
        "",
        "| Condition | Verdict |",
        "|---|---|",
        "| Strategy PF < median of random PFs | PIVOT |",
        "| Strategy PF > p90 of random PFs | PATTERNS SURVIVE |",
        "| Strategy PF in p50–p90 | AMBIGUOUS = TREATED AS FAIL = PIVOT |",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        f"Strategy PF={strategy_pf} is at the {pf_rank:.1%} percentile of the random-entry null distribution.",
        f"Null distribution median PF = {pf_stats['median']:.3f}, p90 = {pf_stats['p90']:.3f}.",
        "",
        "---",
        "_This report was produced by `src/research/random_entry_control.py` (Story 6.1)._",
        "_Sealed pre-registration: `_bmad-output/preregistration_s12_random_entry.md`_",
    ]

    out_path.write_text("\n".join(lines))
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main(n: int = N_SIMULATIONS, csv_path: str = CSV_PATH, dry_run: bool = False) -> None:
    print("=== S12 Random-Entry Control Test ===")
    print(f"Pre-registration SHA: 7ffb3e0b712f4265478b21ae0e583e57f1249f4e")
    print(f"N = {n} simulations | Data: {csv_path}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading 1-min bars...")
    bars = _load_bars(csv_path)
    print(f"  Loaded {len(bars):,} bars ({bars.index[0].date()} → {bars.index[-1].date()})")

    config = StrategyConfig()
    simulator = RandomEntrySimulator(bars, config)

    # ── Baseline run ───────────────────────────────────────────────────────
    print("\nRunning BacktestEngine baseline...")
    engine = BacktestEngine(csv_path, config=config)
    real_trades = engine.run()
    n_real = len(real_trades)
    print(f"  Real strategy: {n_real} trades, PF={STRATEGY_PF_BASELINE}")

    # ── Calibrate p_enter ──────────────────────────────────────────────────
    print("\nCounting candidate bars (for p_enter calibration)...")
    candidate_bars = simulator.count_candidate_bars()
    p_enter = n_real / candidate_bars if candidate_bars > 0 else 0.01
    print(f"  Candidate bars: {candidate_bars:,}")
    print(f"  p_enter = {n_real} / {candidate_bars:,} = {p_enter:.5f}")
    print()

    # ── Run simulations ────────────────────────────────────────────────────
    results: list[SimResult] = []
    seeds = list(SEED_RANGE)[:n]

    for seed in seeds:
        trades = simulator.run(seed, p_enter)
        pf, wr, sharpe = _compute_metrics(trades)
        results.append(SimResult(seed=seed, pf=pf, wr=wr, sharpe=sharpe, trade_count=len(trades)))
        print(
            f"  Simulation {seed + 1:3d}/{n}: "
            f"trades={len(trades):3d}  PF={pf:.3f}  WR={wr:.3f}  Sharpe={sharpe:.3f}"
        )

    print()

    # ── Distribution stats ─────────────────────────────────────────────────
    pf_vals = [r.pf for r in results]
    wr_vals = [r.wr for r in results]
    sharpe_vals = [r.sharpe for r in results]

    pf_stats = _percentile_stats(pf_vals)
    wr_stats = _percentile_stats(wr_vals)
    sharpe_stats = _percentile_stats(sharpe_vals)

    pf_rank = _percentile_rank(pf_vals, STRATEGY_PF_BASELINE)
    verdict = _verdict(STRATEGY_PF_BASELINE, pf_stats)

    # ── Print summary ──────────────────────────────────────────────────────
    print("=== NULL DISTRIBUTION ===")
    print(f"{'Metric':<10} {'Min':>7} {'P25':>7} {'Median':>7} {'P75':>7} {'P90':>7} {'Max':>7}")
    print("-" * 55)
    for label, stats in [("PF", pf_stats), ("WR", wr_stats), ("Sharpe", sharpe_stats)]:
        print(
            f"{label:<10} "
            f"{stats['min']:>7.3f} {stats['p25']:>7.3f} {stats['median']:>7.3f} "
            f"{stats['p75']:>7.3f} {stats['p90']:>7.3f} {stats['max']:>7.3f}"
        )
    print()
    print(f"Strategy PF = {STRATEGY_PF_BASELINE} → percentile rank = {pf_rank:.1%}")
    print(f"Median PF = {pf_stats['median']:.3f} | P90 PF = {pf_stats['p90']:.3f}")
    print()
    print(f"VERDICT: {verdict}")
    print()

    # ── Write report ───────────────────────────────────────────────────────
    if not dry_run:
        report_path = _write_verdict_report(
            results,
            STRATEGY_PF_BASELINE,
            pf_rank,
            verdict,
            pf_stats,
            wr_stats,
            sharpe_stats,
            OUTPUT_DIR,
        )
        print(f"Verdict report written: {report_path}")
    else:
        print("[dry-run] Verdict report not written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S12 Random-Entry Control Test")
    parser.add_argument("--n", type=int, default=N_SIMULATIONS, help="Number of simulations")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to 1-min CSV")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 simulation, no report written")
    args = parser.parse_args()

    if args.dry_run:
        main(n=1, csv_path=args.csv, dry_run=True)
    else:
        main(n=args.n, csv_path=args.csv, dry_run=False)
