"""
LRC (Linear-Regression-Channel) strategy — new strategy line, not a YANK amendment.

Base structure carried over from the compressed-cascade candidate (Phase 1
verdict: PF 1.397 in-sample, but PF 0.78 on the 2026 holdout screening read --
see yank_compressed_cascade_phase1.py / _bmad-output/preregistration_yank_compressed_cascade.md):
M15 sweep -> M5 CHoCH -> M1 FVG. Everything else (SL/TP, gap ratio, and the new
regression-channel regime gate) is swept fresh here, from scratch, per Alex's
2026-08-19 direction: "this should be considered a new strategy... we will
have to start from scratch for optimization." Dev-phase discipline, also per
Alex: no OOS gate for now, 2026 data only, aggressive/exploratory.

New, standalone module -- does not modify strategy_core.py, backtest_engine.py,
or yank_compressed_cascade_phase1.py (whose frozen Phase 1/Phase 2 numbers
this deliberately does not touch or re-verify against).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.regression_channel import compute_regression_channel
from src.research.strategy_core import (
    Direction,
    EntryDecision,
    FVGSignal,
    StrategyConfig,
    SweepSignal,
    calc_atr,
    check_exit,
    detect_fvg,
    detect_liquidity_sweep,
    detect_m15_choch,
    detect_m15_choch_bullish,
    kill_zone_filter,
    make_entry_decision,
    volatility_regime_filter,
)
from yank_compressed_cascade_phase1 import _load_bars, resample_to_timeframe  # reused utilities only

SWEEP_RULE = "15min"
CHOCH_RULE = "5min"
_STRUCTURE_BUFFER_BARS = 150


@dataclass(frozen=True)
class LRCConfig:
    regression_lookback: int
    band_k: float
    regression_timeframe: str  # "15min" or "5min"
    gate_mode: str  # "slope" | "position" | "combined"
    sl_multiplier: float
    tp_multiplier: float
    min_gap_atr_ratio: float


@dataclass
class BaseGates:
    """Structure gates shared across every LRC grid cell that reuses the same
    (sweep_rule, choch_rule) -- always M15/M5 here, so computed exactly once
    per dataset, not once per grid cell (972-cell grid, 1x vs 972x)."""

    vol_ok: np.ndarray
    sweep_dir: list
    sweep_atr: np.ndarray
    choch_active: np.ndarray


def precompute_base_gates(bars: pd.DataFrame, base_config: StrategyConfig) -> BaseGates:
    n = len(bars)
    full_sweep_tf = resample_to_timeframe(bars, SWEEP_RULE)
    full_choch_tf = resample_to_timeframe(bars, CHOCH_RULE)
    tz = bars.index.tz

    vol_ok = np.ones(n, dtype=bool)
    sweep_dir: list = [None] * n
    sweep_atr = np.zeros(n, dtype=float)
    choch_active_arr = np.zeros(n, dtype=bool)

    sweep_tf_bars = None
    cur_sweep_atr = 0.0
    cur_vol_ok = True
    sweep_cached: SweepSignal | None = None
    last_sweep_boundary = None
    choch_active = False
    choch_last_bar_ts = pd.Timestamp("1900-01-01", tz=tz)
    prev_sweep_dir = None
    min_sweep_rows = base_config.h1_sweep_lookback + 5
    bar_index = bars.index

    for i in range(n):
        bar_ts = bar_index[i]
        sweep_boundary = bar_ts.floor(SWEEP_RULE)
        if sweep_boundary != last_sweep_boundary:
            last_sweep_boundary = sweep_boundary
            sweep_idx = int(full_sweep_tf.index.searchsorted(sweep_boundary))
            sweep_start = max(0, sweep_idx - _STRUCTURE_BUFFER_BARS)
            sweep_slice = full_sweep_tf.iloc[sweep_start:sweep_idx]
            sweep_tf_bars = sweep_slice if len(sweep_slice) > 0 else None
            cur_sweep_atr = calc_atr(sweep_tf_bars) if sweep_tf_bars is not None and len(sweep_tf_bars) >= 2 else 0.0

            if sweep_tf_bars is not None and len(sweep_tf_bars) >= 20:
                try:
                    cur_vol_ok = volatility_regime_filter(sweep_tf_bars, base_config)
                except ValueError:
                    cur_vol_ok = True
            else:
                cur_vol_ok = True

            if sweep_tf_bars is not None and len(sweep_tf_bars) >= min_sweep_rows:
                try:
                    sweep_cached = detect_liquidity_sweep(sweep_tf_bars, base_config)
                except ValueError:
                    sweep_cached = None
            else:
                sweep_cached = None

            new_dir = sweep_cached.direction.value if sweep_cached is not None else None
            if new_dir != prev_sweep_dir:
                choch_active = False
                choch_last_bar_ts = pd.Timestamp("1900-01-01", tz=tz)
            prev_sweep_dir = new_dir

        if base_config.m15_confirmation and sweep_cached is not None and not choch_active:
            choch_idx = int(full_choch_tf.index.searchsorted(bar_ts))
            choch_completed = full_choch_tf.iloc[: max(0, choch_idx - 1)]
            if len(choch_completed) >= 1:
                last_choch_ts = choch_completed.index[-1]
                if last_choch_ts > choch_last_bar_ts:
                    choch_last_bar_ts = last_choch_ts
                    choch_fn = detect_m15_choch if sweep_cached.direction == Direction.BEARISH else detect_m15_choch_bullish
                    if choch_fn(choch_completed):
                        choch_active = True

        vol_ok[i] = cur_vol_ok
        sweep_dir[i] = sweep_cached.direction if sweep_cached is not None else None
        sweep_atr[i] = cur_sweep_atr
        choch_active_arr[i] = choch_active

    return BaseGates(vol_ok=vol_ok, sweep_dir=sweep_dir, sweep_atr=sweep_atr, choch_active=choch_active_arr)


@dataclass
class RegressionFeatures:
    """Regression-channel readings for one (lookback, timeframe) pair,
    aligned onto the M1 index using only completed higher-timeframe bars."""

    slope: np.ndarray
    position_z: np.ndarray


def precompute_regression_features(bars: pd.DataFrame, lookback: int, timeframe: str) -> RegressionFeatures:
    n = len(bars)
    reg_tf_bars = resample_to_timeframe(bars, timeframe)
    channel = compute_regression_channel(reg_tf_bars["close"], lookback)

    slope = np.full(n, np.nan)
    position_z = np.full(n, np.nan)

    last_boundary = None
    cur_slope = np.nan
    cur_pos = np.nan
    bar_index = bars.index

    for i in range(n):
        bar_ts = bar_index[i]
        boundary = bar_ts.floor(timeframe)
        if boundary != last_boundary:
            last_boundary = boundary
            idx = int(channel.index.searchsorted(boundary))
            completed_idx = idx - 1  # exclude the still-forming higher-tf bar
            if completed_idx >= 0:
                cur_slope = channel["slope"].iloc[completed_idx]
                cur_pos = channel["position_z"].iloc[completed_idx]
            else:
                cur_slope = np.nan
                cur_pos = np.nan
        slope[i] = cur_slope
        position_z[i] = cur_pos

    return RegressionFeatures(slope=slope, position_z=position_z)


def _regime_ok(i: int, reg: RegressionFeatures, lrc: LRCConfig) -> bool:
    s = reg.slope[i]
    p = reg.position_z[i]
    if np.isnan(s) or np.isnan(p):
        return False
    slope_ok = s <= 0.0  # not in a strong uptrend -- bearish_only baseline
    position_ok = p >= lrc.band_k  # price extended above the regression channel's upper band
    if lrc.gate_mode == "slope":
        return bool(slope_ok)
    if lrc.gate_mode == "position":
        return bool(position_ok)
    if lrc.gate_mode == "combined":
        return bool(slope_ok and position_ok)
    raise ValueError(f"unknown gate_mode {lrc.gate_mode!r}")


@dataclass
class LRCResult:
    pnls: list[float]
    n_trades: int
    n_candidate_bars: int


def run_lrc_cascade(
    bars: pd.DataFrame,
    base_config: StrategyConfig,
    gates: BaseGates,
    reg: RegressionFeatures,
    lrc: LRCConfig,
    *,
    random_seed: int | None = None,
    p_enter: float = 0.0,
) -> LRCResult:
    """base_config carries bearish_only/tuesday_exclusion/vol_regime/etc (fixed
    across the grid); lrc carries the swept knobs (SL/TP/gap-ratio/regression).

    If random_seed is None, runs the real FVG-based strategy. Otherwise runs a
    direction-matched random-entry control sharing every structure/regime gate
    (S12 methodology, same design as yank_compressed_cascade_phase1.py's
    Phase-1 null test)."""
    n = len(bars)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    config = StrategyConfig(
        sl_multiplier=lrc.sl_multiplier,
        tp_multiplier=lrc.tp_multiplier,
        entry_pct=base_config.entry_pct,
        atr_threshold=base_config.atr_threshold,
        max_gap_dollars=base_config.max_gap_dollars,
        max_gap_atr_ratio=base_config.max_gap_atr_ratio,
        max_hold_bars=base_config.max_hold_bars,
        max_pending_bars=base_config.max_pending_bars,
        contracts_per_trade=base_config.contracts_per_trade,
        max_daily_loss=base_config.max_daily_loss,
        vol_regime_lookback=base_config.vol_regime_lookback,
        vol_regime_threshold=base_config.vol_regime_threshold,
        min_gap_atr_ratio=lrc.min_gap_atr_ratio,
        ml_threshold=0.0,
        bearish_only=base_config.bearish_only,
        h1_sweep_lookback=base_config.h1_sweep_lookback,
        commission_per_roundtrip=base_config.commission_per_roundtrip,
        enable_kill_zone_filter=base_config.enable_kill_zone_filter,
        m15_confirmation=base_config.m15_confirmation,
        tuesday_exclusion=base_config.tuesday_exclusion,
    )

    active: EntryDecision | None = None
    pending = False
    pending_bars = 0
    bars_held = 0
    daily_pnl = 0.0
    daily_halted = False
    last_date = None
    pnls: list[float] = []
    n_trades = 0
    n_candidate_bars = 0

    for i in range(n):
        bar_ts = bars.index[i]
        bar = bars.iloc[i]

        if active is not None:
            if pending:
                pending_bars += 1
                filled = float(bar["high"]) >= active.entry_price if active.direction == Direction.BEARISH else float(bar["low"]) <= active.entry_price
                if filled:
                    pending = False
                    bars_held = 0
                    exit_dec = check_exit(bar, active, 0, config)
                    if exit_dec is not None:
                        pnl = _pnl(active, exit_dec, config)
                        pnls.append(pnl)
                        n_trades += 1
                        daily_pnl += pnl
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
                    pnl = _pnl(active, exit_dec, config)
                    pnls.append(pnl)
                    n_trades += 1
                    daily_pnl += pnl
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
        if not _regime_ok(i, reg, lrc):
            continue
        sweep = SweepSignal(direction=sweep_direction, bars_ago=0, sweep_price=0.0)

        n_candidate_bars += 1

        if i < 2:
            continue
        m1_buf = bars.iloc[max(0, i - 19) : i + 1]

        if rng is not None:
            if rng.random() >= p_enter:
                continue
            gap = config.atr_threshold * calc_atr(m1_buf)
            if gap <= 0:
                continue
            entry_price = float(bar["close"])
            fvg = FVGSignal(direction=sweep_direction, gap_size=gap, entry_price=entry_price, high=entry_price, low=entry_price)
        else:
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
        pending = True
        pending_bars = 0

    return LRCResult(pnls=pnls, n_trades=n_trades, n_candidate_bars=n_candidate_bars)


def _pnl(active: EntryDecision, exit_dec, config: StrategyConfig) -> float:
    from src.research.strategy_core import POINT_VALUE_USD

    points = active.entry_price - exit_dec.exit_price if active.direction == Direction.BEARISH else exit_dec.exit_price - active.entry_price
    return round(points * POINT_VALUE_USD * active.contracts - config.commission_per_roundtrip, 2)
