"""
YANK Compressed-Cascade Phase 1 — Random-Null Falsification Test
Pre-registration: _bmad-output/preregistration_yank_compressed_cascade.md (Amendment 1)

Tests whether compressing the structure cascade one rung finer (H1 sweep -> M15,
M15 CHoCH -> M5, M1 FVG unchanged) survives a random-entry null, generalizing
s12_random_entry_control.py's method to the compressed candidate.

Does NOT modify strategy_core.py or backtest_engine.py. resample_to_timeframe()
below is a new, isolated function -- resample_to_h1/resample_to_m15 (shared with
the still-live Tier2 architecture) are never touched.

Usage:
    .venv/bin/python yank_compressed_cascade_phase1.py --data <path-to-2025-1min-csv>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import (
    Direction,
    EntryDecision,
    FVGSignal,
    StrategyConfig,
    SweepSignal,
    calc_atr,
    calc_profit_factor,
    check_exit,
    detect_fvg,
    detect_liquidity_sweep,
    detect_m15_choch,
    detect_m15_choch_bullish,
    kill_zone_filter,
    make_entry_decision,
    volatility_regime_filter,
)

# ── Amendment 1: clean Program-C baseline, NOT a live-YANK config replica ────
# (see preregistration_yank_compressed_cascade.md, Amendment 1, for why)
BASELINE_CONFIG = StrategyConfig(
    sl_multiplier=5.0,
    tp_multiplier=6.0,
    entry_pct=0.5,
    atr_threshold=0.5,
    max_gap_dollars=60.0,
    max_gap_atr_ratio=0.426,  # post gap-ceiling-fix (PR #45), disambiguated in Amendment 1
    max_hold_bars=60,
    max_pending_bars=240,
    contracts_per_trade=5,
    max_daily_loss=-750.0,
    vol_regime_lookback=120,
    vol_regime_threshold=0.75,
    min_gap_atr_ratio=0.25,
    ml_threshold=0.0,  # disabled -- out of scope, see Amendment 1
    bearish_only=True,
    h1_sweep_lookback=6,
    commission_per_roundtrip=4.0,
    enable_kill_zone_filter=False,
    m15_confirmation=True,  # S25 architecture requires CHoCH confirmation
    tuesday_exclusion=True,
)

_STRUCTURE_BUFFER_BARS = 150  # bars-of-resolution buffer; covers vol_regime_lookback(120)+margin


# ---------------------------------------------------------------------------
# New, isolated resample function (does not touch strategy_core.py)
# ---------------------------------------------------------------------------


def resample_to_timeframe(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to an arbitrary pandas offset ``rule``.

    Generalizes strategy_core.resample_to_h1 / resample_to_m15 (identical
    aggregation, parameterized frequency) without modifying either -- both are
    imported live by Tier2 and must stay untouched (sealed prereg §1).
    """
    df = bars.copy()
    if not (isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp"):
        df = df.set_index("timestamp")
    out = (
        df[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    out.index.name = "timestamp"
    return out


def _load_bars(csv_path: str) -> pd.DataFrame:
    return BacktestEngine(csv_path)._load_bars(csv_path)


# ---------------------------------------------------------------------------
# Cascade backtest (generalizes BacktestEngine.run() with configurable
# sweep/CHoCH resolution; entry can be a real FVG scan or a calibrated coin
# flip, sharing every structure gate -- mirrors S12's "entry gates shared with
# the real strategy" design).
# ---------------------------------------------------------------------------


@dataclass
class CascadeResult:
    pnls: list[float]
    n_trades: int
    n_candidate_bars: int  # bars where a real FVG scan or coin flip was attempted
    trades: list[dict] | None = None  # entry_ts/exit_ts/direction/pnl/exit_reason, real runs only


@dataclass
class GateState:
    """Per-bar structure gates (sweep/CHoCH/vol regime). Identical across the
    real run and every random-null seed -- computing once and sharing across
    all N passes is what makes N=100 tractable (each seed only differs in the
    entry coin flip, never in structure detection)."""

    vol_ok: np.ndarray  # bool
    sweep_dir: list  # Direction | None per bar
    sweep_atr: np.ndarray  # float
    choch_active: np.ndarray  # bool


def _precompute_gates(bars: pd.DataFrame, config: StrategyConfig, sweep_rule: str, choch_rule: str) -> GateState:
    n = len(bars)
    full_sweep_tf = resample_to_timeframe(bars, sweep_rule)
    full_choch_tf = resample_to_timeframe(bars, choch_rule)
    tz = bars.index.tz

    vol_ok = np.ones(n, dtype=bool)
    sweep_dir: list = [None] * n
    sweep_atr = np.zeros(n, dtype=float)
    choch_active_arr = np.zeros(n, dtype=bool)

    sweep_tf_bars: pd.DataFrame | None = None
    cur_sweep_atr = 0.0
    cur_vol_ok = True
    sweep_cached: SweepSignal | None = None
    last_sweep_boundary: pd.Timestamp | None = None

    choch_active = False
    choch_last_bar_ts = pd.Timestamp("1900-01-01", tz=tz)
    prev_sweep_dir: str | None = None

    min_sweep_rows = config.h1_sweep_lookback + 5
    bar_index = bars.index

    for i in range(n):
        bar_ts = bar_index[i]

        sweep_boundary = bar_ts.floor(sweep_rule)
        if sweep_boundary != last_sweep_boundary:
            last_sweep_boundary = sweep_boundary
            sweep_idx = int(full_sweep_tf.index.searchsorted(sweep_boundary))
            sweep_start = max(0, sweep_idx - _STRUCTURE_BUFFER_BARS)
            sweep_slice = full_sweep_tf.iloc[sweep_start:sweep_idx]
            sweep_tf_bars = sweep_slice if len(sweep_slice) > 0 else None
            cur_sweep_atr = calc_atr(sweep_tf_bars) if sweep_tf_bars is not None and len(sweep_tf_bars) >= 2 else 0.0

            if sweep_tf_bars is not None and len(sweep_tf_bars) >= 20:
                try:
                    cur_vol_ok = volatility_regime_filter(sweep_tf_bars, config)
                except ValueError:
                    cur_vol_ok = True
            else:
                cur_vol_ok = True

            if sweep_tf_bars is not None and len(sweep_tf_bars) >= min_sweep_rows:
                try:
                    sweep_cached = detect_liquidity_sweep(sweep_tf_bars, config)
                except ValueError:
                    sweep_cached = None
            else:
                sweep_cached = None

            new_dir = sweep_cached.direction.value if sweep_cached is not None else None
            if new_dir != prev_sweep_dir:
                choch_active = False
                choch_last_bar_ts = pd.Timestamp("1900-01-01", tz=tz)
            prev_sweep_dir = new_dir

        if config.m15_confirmation and sweep_cached is not None and not choch_active:
            choch_idx = int(full_choch_tf.index.searchsorted(bar_ts))
            choch_completed = full_choch_tf.iloc[: max(0, choch_idx - 1)]
            if len(choch_completed) >= 1:
                last_choch_ts = choch_completed.index[-1]
                if last_choch_ts > choch_last_bar_ts:
                    choch_last_bar_ts = last_choch_ts
                    choch_fn = (
                        detect_m15_choch
                        if sweep_cached.direction == Direction.BEARISH
                        else detect_m15_choch_bullish
                    )
                    if choch_fn(choch_completed):
                        choch_active = True

        vol_ok[i] = cur_vol_ok
        sweep_dir[i] = sweep_cached.direction if sweep_cached is not None else None
        sweep_atr[i] = cur_sweep_atr
        choch_active_arr[i] = choch_active

    return GateState(vol_ok=vol_ok, sweep_dir=sweep_dir, sweep_atr=sweep_atr, choch_active=choch_active_arr)


def _run_cascade(
    bars: pd.DataFrame,
    config: StrategyConfig,
    gates: GateState,
    *,
    random_seed: int | None = None,
    p_enter: float = 0.0,
) -> CascadeResult:
    """One backtest pass over precomputed structure gates. If random_seed is
    None, runs the real FVG-based strategy. Otherwise runs a direction-matched
    random-entry control at the candidate's own structure gates (S12
    methodology, generalized)."""
    n = len(bars)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    active: EntryDecision | None = None
    active_entry_ts: pd.Timestamp | None = None
    pending = False
    pending_bars = 0
    bars_held = 0
    daily_pnl = 0.0
    daily_halted = False
    last_date = None
    record = rng is None  # only the real run needs per-trade records

    pnls: list[float] = []
    trades: list[dict] = []
    n_trades = 0
    n_candidate_bars = 0

    for i in range(n):
        bar_ts = bars.index[i]
        bar = bars.iloc[i]

        # ── Advance active/pending trade ─────────────────────────────────
        if active is not None:
            if pending:
                pending_bars += 1
                if active.direction == Direction.BEARISH:
                    filled = float(bar["high"]) >= active.entry_price
                else:
                    filled = float(bar["low"]) <= active.entry_price
                if filled:
                    pending = False
                    bars_held = 0
                    exit_dec = check_exit(bar, active, 0, config)
                    if exit_dec is not None:
                        pnl = _trade_pnl(active, exit_dec, config)
                        pnls.append(pnl)
                        n_trades += 1
                        daily_pnl += pnl
                        if record:
                            trades.append(_trade_record(active, active_entry_ts, bar_ts, exit_dec, pnl))
                        active = None
                    continue
                elif pending_bars >= config.max_pending_bars:
                    active = None
                    pending = False
                    pending_bars = 0
                else:
                    continue

            if active is not None and not pending:
                bars_held += 1
                exit_dec = check_exit(bar, active, bars_held, config)
                if exit_dec is not None:
                    pnl = _trade_pnl(active, exit_dec, config)
                    pnls.append(pnl)
                    n_trades += 1
                    daily_pnl += pnl
                    if record:
                        trades.append(_trade_record(active, active_entry_ts, bar_ts, exit_dec, pnl))
                    active = None
                    bars_held = 0
                else:
                    continue

        if active is not None:
            continue

        if i < 20:
            continue
        if config.tuesday_exclusion and bar_ts.weekday() == 1:
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

        if not gates.vol_ok[i]:
            continue
        sweep_direction = gates.sweep_dir[i]
        if sweep_direction is None:
            continue
        if config.bearish_only and sweep_direction != Direction.BEARISH:
            continue
        if config.m15_confirmation and not gates.choch_active[i]:
            continue
        sweep = SweepSignal(direction=sweep_direction, bars_ago=0, sweep_price=0.0)

        n_candidate_bars += 1

        if rng is not None:
            # Random-entry control: same gates, coin-flip entry, ATR-sized gap.
            if rng.random() >= p_enter:
                continue
            m1_buf = bars.iloc[max(0, i - 19) : i + 1]
            gap = config.atr_threshold * calc_atr(m1_buf)
            if gap <= 0:
                continue
            entry_price = float(bar["close"])
            fvg = FVGSignal(direction=sweep_direction, gap_size=gap, entry_price=entry_price, high=entry_price, low=entry_price)
        else:
            if i < 2:
                continue
            m1_buf = bars.iloc[max(0, i - 19) : i + 1]
            try:
                fvg = detect_fvg(m1_buf, config, gates.sweep_atr[i])
            except ValueError:
                continue
            if fvg is None:
                continue

        kz = kill_zone_filter(bar_ts, config)
        if config.enable_kill_zone_filter and not kz:
            continue

        entry = make_entry_decision(sweep, fvg, config, vol_ok=True, m15_conf=True)
        if entry is None:
            continue

        active = entry
        active_entry_ts = bar_ts
        pending = True
        pending_bars = 0

    return CascadeResult(pnls=pnls, n_trades=n_trades, n_candidate_bars=n_candidate_bars, trades=trades if record else None)


def _trade_pnl(active: EntryDecision, exit_dec, config: StrategyConfig) -> float:
    from src.research.strategy_core import POINT_VALUE_USD

    if active.direction == Direction.BEARISH:
        points = active.entry_price - exit_dec.exit_price
    else:
        points = exit_dec.exit_price - active.entry_price
    return round(points * POINT_VALUE_USD * active.contracts - config.commission_per_roundtrip, 2)


def _trade_record(active: EntryDecision, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, exit_dec, pnl: float) -> dict:
    return {
        "entry_ts": entry_ts.isoformat(),
        "exit_ts": exit_ts.isoformat(),
        "direction": active.direction.value,
        "entry_price": active.entry_price,
        "exit_price": exit_dec.exit_price,
        "exit_reason": exit_dec.reason.value,
        "pnl": pnl,
    }


# ---------------------------------------------------------------------------
# Phase 1 driver
# ---------------------------------------------------------------------------


def run_phase1(csv_path: str, sweep_rule: str = "15min", choch_rule: str = "5min", n_seeds: int = 100) -> dict:
    bars = _load_bars(csv_path)
    config = BASELINE_CONFIG

    # Computed once, shared by the real run and every null seed (§ perf note:
    # structure gates don't depend on entry choice, so recomputing them per
    # seed was 100x redundant work).
    gates = _precompute_gates(bars, config, sweep_rule, choch_rule)

    real = _run_cascade(bars, config, gates)
    if real.n_trades == 0:
        return {"error": "candidate produced zero trades on training data -- cannot form a PF", "real": real}
    real_pf = calc_profit_factor(real.pnls)

    p_enter = real.n_trades / real.n_candidate_bars if real.n_candidate_bars > 0 else 0.0

    null_pfs = []
    for seed in range(n_seeds):
        res = _run_cascade(bars, config, gates, random_seed=seed, p_enter=p_enter)
        if res.n_trades > 0:
            null_pfs.append(calc_profit_factor(res.pnls))

    null_arr = np.array([pf for pf in null_pfs if np.isfinite(pf)])
    median = float(np.median(null_arr)) if len(null_arr) else float("nan")
    p90 = float(np.percentile(null_arr, 90)) if len(null_arr) else float("nan")
    pct_rank = float((null_arr < real_pf).mean() * 100) if len(null_arr) else float("nan")

    if real_pf < median:
        verdict = "PIVOT"
    elif real_pf > p90:
        verdict = "PROCEED to Phase 2"
    else:
        verdict = "AMBIGUOUS = TREATED AS FAIL = PIVOT"

    return {
        "sweep_rule": sweep_rule,
        "choch_rule": choch_rule,
        "real_pf": real_pf,
        "real_n_trades": real.n_trades,
        "real_n_candidate_bars": real.n_candidate_bars,
        "p_enter": p_enter,
        "n_null_sims_with_trades": len(null_arr),
        "null_median": median,
        "null_p90": p90,
        "pct_rank": pct_rank,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to 1-min OHLCV CSV (training window only)")
    parser.add_argument("--sweep-rule", default="15min")
    parser.add_argument("--choch-rule", default="5min")
    parser.add_argument("--n-seeds", type=int, default=100)
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: {args.data} not found", file=sys.stderr)
        sys.exit(1)

    result = run_phase1(args.data, args.sweep_rule, args.choch_rule, args.n_seeds)
    print(result)


if __name__ == "__main__":
    main()
