"""Pure, stateless, deterministic strategy logic for the Tier 2 system.

This module is the single source of truth for all detection, filter, exit, and
metric logic shared by the live trader (``tier2_streaming_working.py``) and the
backtest engine (``backtest_engine.py``). It imports neither of them, and
nothing from the ``src`` package.

Purity contract (AR1) — forbidden here, permanently: I/O, logging, wall-clock
(``datetime.now()``), randomness, module-level mutable state, ``src.*`` imports,
and mutation of input arguments. Same inputs always produce the same outputs.

Story 1.1 establishes the type contracts: ``StrategyConfig``, the ``Direction``
and ``ExitReason`` enums, the unit-conversion constants, and the frozen
detection/decision dataclasses.

Story 1.2 adds the pure detection functions: ``resample_to_h1``,
``detect_liquidity_sweep``, ``detect_fvg``, and ``volatility_regime_filter``.
Entry/exit/metric functions are added by Story 1.3.

Canonical bar schema (AR9): tz-aware ``DatetimeIndex`` named ``timestamp``
(America/New_York), columns ``open``/``high``/``low``/``close`` (float64),
``volume`` (int64), sorted ascending. All public functions accept and return
this schema; callers must not pass naive datetimes.
"""

from __future__ import annotations

import zoneinfo
from dataclasses import dataclass
from datetime import time, timedelta
from enum import Enum
from math import sqrt

import numpy as np
import pandas as pd

# Unit-conversion constants (AR13) — the only sanctioned point<->USD / tick
# conversions. No inline 2.0 or 0.25 magic numbers anywhere else.
POINT_VALUE_USD: float = 2.0  # USD per MNQ index point (P&L scaling)
TICK_SIZE: float = 0.25  # MNQ minimum price increment
# E-mini NQ full-contract notional multiplier ($20/pt). Used for dollar-bar
# threshold sizing only — NOT for MNQ P&L. Use POINT_VALUE_USD for P&L.
MNQ_NOTIONAL_MULTIPLIER: float = 20.0


class Direction(Enum):
    """Trade / signal direction.

    ``.value`` is the string written to CSV logs (AR10); internally the enum
    member itself is always used.
    """

    BEARISH = "BEARISH"
    BULLISH = "BULLISH"


class ExitReason(Enum):
    """Reason an active trade was closed.

    ``.value`` is the string written to CSV logs (AR11); internally the enum
    member itself is always used.
    """

    TP = "TP"
    SL = "SL"
    TIME_STOP = "TIME_STOP"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable single source of truth for every strategy parameter.

    Defaults are ported verbatim from the constants at the top of
    ``src/research/tier2_streaming_working.py`` so the backtest engine and the
    live trader provably share one parameter set. ``frozen=True`` guards
    against accidental in-process mutation; the real integrity control is the
    pre-registration source/commit hash (Epic 3).
    """

    sl_multiplier: float = 5.0
    tp_multiplier: float = 6.0
    entry_pct: float = 0.5
    atr_threshold: float = 0.5
    max_gap_dollars: float = 60.0  # dollar ceiling for FVG gap size (added Story 1.2)
    max_hold_bars: int = 60
    max_pending_bars: int = 240
    contracts_per_trade: int = 5
    max_daily_loss: float = -750.0
    vol_regime_lookback: int = 120
    vol_regime_threshold: float = 0.75
    min_gap_atr_ratio: float = 0.25
    ml_threshold: float = 0.0
    bearish_only: bool = True
    h1_sweep_lookback: int = 6
    kill_zone_start_et: time = time(9, 30)
    kill_zone_end_et: time = time(11, 0)
    commission_per_roundtrip: float = 4.0  # $0.40/contract × 5 contracts × 2 sides (AR13)
    enable_kill_zone_filter: bool = False  # if True, blocks entries outside kill zone
    m15_confirmation: bool = False  # if True, blocks entries where prior M15 bar misaligns with H1 sweep
    tuesday_exclusion: bool = True  # if True, skips Tuesday entry candidates (default preserves existing behavior)


@dataclass(frozen=True)
class FVGSignal:
    """A detected fair value gap. Returned by ``detect_fvg`` (Story 1.2)."""

    direction: Direction
    gap_size: float  # index points
    entry_price: float  # gap midpoint
    high: float
    low: float


@dataclass(frozen=True)
class SweepSignal:
    """A detected H1 liquidity sweep. Returned by ``detect_liquidity_sweep`` (Story 1.2)."""

    direction: Direction
    bars_ago: int
    sweep_price: float


@dataclass(frozen=True)
class EntryDecision:
    """A resolved trade entry. Returned by ``make_entry_decision`` (Story 1.3)."""

    direction: Direction
    entry_price: float
    sl_price: float
    tp_price: float
    contracts: int


@dataclass(frozen=True)
class ExitDecision:
    """A resolved trade exit. Returned by ``check_exit`` (Story 1.3)."""

    reason: ExitReason
    exit_price: float


@dataclass(frozen=True)
class M15Confirmation:
    """M15 bar confirmation result. Returned by ``check_m15_confirmation`` (Story 2.3)."""

    confirmed: bool
    direction: Direction | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_bars(bars: pd.DataFrame, *, min_rows: int = 1) -> None:
    """Raise ``ValueError`` on empty, NaN-containing, or missing-column input."""
    if len(bars) == 0:
        raise ValueError("bars DataFrame is empty")
    if len(bars) < min_rows:
        raise ValueError(f"bars requires at least {min_rows} rows; got {len(bars)}")
    has_ts_index = (
        isinstance(bars.index, pd.DatetimeIndex) and bars.index.name == "timestamp"
    )
    if not has_ts_index and "timestamp" not in bars.columns:
        raise ValueError(
            "bars must have a DatetimeIndex named 'timestamp' or a 'timestamp' column"
        )
    for col in ("open", "high", "low", "close", "volume"):
        if col not in bars.columns:
            raise ValueError(f"bars is missing required column '{col}'")
        if bars[col].isna().any():
            raise ValueError(f"bars column '{col}' contains NaN values")


def calc_atr(bars: pd.DataFrame) -> float:
    """20-bar mean True Range (index points). Returns 10.0 as fallback if < 20 bars.

    Ported verbatim from ``Tier2StreamingTrader._calculate_atr``.
    Uses 1-min bars (M1); for H1 ATR pass H1 bars.
    """
    if len(bars) < 20:
        return 10.0
    sliced = bars.iloc[-20:]
    highs = sliced["high"].values
    lows = sliced["low"].values
    closes = sliced["close"].values
    trs = [
        max(
            float(highs[i]) - float(lows[i]),
            abs(float(highs[i]) - float(closes[i - 1])),
            abs(float(lows[i]) - float(closes[i - 1])),
        )
        for i in range(1, len(sliced))
    ]
    return sum(trs) / len(trs) if trs else 10.0


# ---------------------------------------------------------------------------
# Public detection functions
# ---------------------------------------------------------------------------


def resample_to_h1(bars: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars to H1 candles.

    No timezone conversion is performed (AR19); bars must arrive already
    tz-aware in America/New_York.

    Parameters
    ----------
    bars:
        Canonical 1-min bars with tz-aware DatetimeIndex named ``timestamp``
        or a ``timestamp`` column (AR9).  ``open``/``high``/``low``/``close``
        float64, ``volume`` int64.

    Returns
    -------
    pd.DataFrame
        H1 OHLCV with tz-aware DatetimeIndex named ``timestamp``.
        Aggregation: open=first, high=max, low=min, close=last, volume=sum.
        Periods with no data are dropped.

    Raises
    ------
    ValueError
        On empty input, NaN in OHLCV/volume, or missing columns.
    """
    _validate_bars(bars, min_rows=1)
    df = bars.copy()
    if not (isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp"):
        df = df.set_index("timestamp")
    h1 = (
        df[["open", "high", "low", "close", "volume"]]
        .resample("1h")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    h1.index.name = "timestamp"
    return h1


def resample_to_m15(bars: pd.DataFrame) -> pd.DataFrame:
    """Resample bars to 15-minute OHLCV candles.

    Same aggregation as ``resample_to_h1`` but at 15-minute frequency.
    No timezone conversion performed (AR19); bars must arrive tz-aware.

    Parameters
    ----------
    bars:
        Canonical bars with tz-aware DatetimeIndex named ``timestamp`` (AR9).

    Returns
    -------
    pd.DataFrame
        15-min OHLCV with tz-aware DatetimeIndex named ``timestamp``.
        Aggregation: open=first, high=max, low=min, close=last, volume=sum.
        Periods with no data are dropped.

    Raises
    ------
    ValueError
        On empty input, NaN in OHLCV/volume, or missing columns.
    """
    _validate_bars(bars, min_rows=1)
    df = bars.copy()
    if not (isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp"):
        df = df.set_index("timestamp")
    m15 = (
        df[["open", "high", "low", "close", "volume"]]
        .resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    m15.index.name = "timestamp"
    return m15


def detect_fvg(
    bars: pd.DataFrame,
    config: StrategyConfig,
    atr: float,
) -> FVGSignal | None:
    """Detect a 3-bar Fair Value Gap in the last three bars of *bars*.

    Ported verbatim from ``Tier2StreamingTrader._detect_fvg``.

    Bearish FVG: ``bars.iloc[-3]['low'] > bars.iloc[-1]['high']`` with a
    bearish middle candle (close < open).

    Bullish FVG: ``bars.iloc[-1]['low'] > bars.iloc[-3]['high']`` with a
    bullish middle candle (close > open).

    Three filters (matching the live system exactly):

    1. ``gap_size >= config.atr_threshold * _atr(bars)`` (1-min ATR gate).
    2. ``gap_size * POINT_VALUE_USD <= config.max_gap_dollars`` (dollar ceiling).
    3. ``gap_size >= config.min_gap_atr_ratio * atr`` when ``atr > 0``
       (H1 ATR ratio gate).

    Parameters
    ----------
    bars:
        Canonical 1-min bars (AR9).  Must have ≥ 3 rows.
    config:
        Active strategy parameters.
    atr:
        Current H1 ATR in index points (computed from H1 bars by the caller).
        Pass ``0.0`` to skip the H1 ATR ratio gate.

    Returns
    -------
    FVGSignal | None

    Raises
    ------
    ValueError
        On empty input, NaN in OHLCV, or fewer than 3 rows.
    """
    _validate_bars(bars, min_rows=3)
    c1 = bars.iloc[-3]
    c2 = bars.iloc[-2]
    c3 = bars.iloc[-1]

    bullish = bool(c3["low"] > c1["high"] and c2["close"] > c2["open"])
    bearish = bool(c1["low"] > c3["high"] and c2["close"] < c2["open"])

    if not (bullish or bearish):
        return None

    if bullish:
        top = float(c3["low"])
        bot = float(c1["high"])
        direction = Direction.BULLISH
    else:
        top = float(c1["low"])
        bot = float(c3["high"])
        direction = Direction.BEARISH

    if top <= bot:
        return None

    gap_pts = top - bot
    if gap_pts < config.atr_threshold * calc_atr(bars):
        return None
    if gap_pts * POINT_VALUE_USD > config.max_gap_dollars:
        return None
    if atr > 0 and gap_pts < config.min_gap_atr_ratio * atr:
        return None

    return FVGSignal(
        direction=direction,
        gap_size=gap_pts,
        entry_price=(top + bot) / 2.0,
        high=top,
        low=bot,
    )


def detect_liquidity_sweep(
    h1_bars: pd.DataFrame,
    config: StrategyConfig,
) -> SweepSignal | None:
    """Scan the last ``config.h1_sweep_lookback`` completed H1 bars for a
    liquidity sweep and reversal against a confirmed 5-bar swing pivot.

    Replaces the stateful sweep-flag expiry in ``Tier2StreamingTrader`` with a
    pure scan: a sweep is "active" if it occurred within the last
    ``h1_sweep_lookback`` H1 bars.  The most recent sweep is returned.

    Bearish sweep: a bar whose ``high`` exceeded a prior swing high and whose
    ``close`` reversed back below it (selling above resistance).

    Bullish sweep: a bar whose ``low`` undercut a prior swing low and whose
    ``close`` reversed back above it (buying below support).

    Swing confirmation: the swing pivot's timestamp must be at least 2 H1 bars
    (2 hours) before the sweep bar (ported from the live code's 2-hour check).

    Parameters
    ----------
    h1_bars:
        Completed H1 bars with canonical AR9 schema (DatetimeIndex or
        timestamp column).  Must have at least
        ``config.h1_sweep_lookback + 5`` rows to allow pivot detection.
    config:
        Active strategy parameters.

    Returns
    -------
    SweepSignal | None
        Most recent sweep found, or ``None``.  ``bars_ago=0`` means the most
        recent completed bar; ``bars_ago=1`` is the bar before that, etc.

    Raises
    ------
    ValueError
        On empty input, NaN in OHLCV, or insufficient rows.
    """
    min_rows = config.h1_sweep_lookback + 5
    _validate_bars(h1_bars, min_rows=min_rows)

    # Normalise to column-based access for consistency with live code
    if (
        isinstance(h1_bars.index, pd.DatetimeIndex)
        and h1_bars.index.name == "timestamp"
    ):
        df = h1_bars.reset_index()
    else:
        df = h1_bars.copy()

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df["timestamp"]
    n = len(df)

    # 5-bar pivot swing detection across all available bars (same as live code)
    sh: list[tuple[object, float]] = []  # (timestamp, value)
    sl: list[tuple[object, float]] = []
    for i in range(2, n - 2):
        h_i = float(highs[i])
        if (
            h_i > float(highs[i - 1])
            and h_i > float(highs[i - 2])
            and h_i > float(highs[i + 1])
            and h_i > float(highs[i + 2])
        ):
            sh.append((timestamps.iloc[i], h_i))
        l_i = float(lows[i])
        if (
            l_i < float(lows[i - 1])
            and l_i < float(lows[i - 2])
            and l_i < float(lows[i + 1])
            and l_i < float(lows[i + 2])
        ):
            sl.append((timestamps.iloc[i], l_i))

    if not sh and not sl:
        return None

    # Scan last h1_sweep_lookback bars for a sweep (most recent first)
    for bars_ago in range(config.h1_sweep_lookback):
        idx = n - 1 - bars_ago
        bar_ts = timestamps.iloc[idx]
        bar_high = float(highs[idx])
        bar_low = float(lows[idx])
        bar_close = float(closes[idx])
        cutoff = bar_ts - timedelta(hours=2)

        for t, val in sh:
            if t < cutoff and bar_high > val and bar_close < val:
                return SweepSignal(
                    direction=Direction.BEARISH,
                    bars_ago=bars_ago,
                    sweep_price=val,
                )

        for t, val in sl:
            if t < cutoff and bar_low < val and bar_close > val:
                return SweepSignal(
                    direction=Direction.BULLISH,
                    bars_ago=bars_ago,
                    sweep_price=val,
                )

    return None


def volatility_regime_filter(
    h1_bars: pd.DataFrame,
    config: StrategyConfig,
) -> bool:
    """Return ``True`` if volatility is acceptable for entry; ``False`` if blocked.

    Blocks entry when the current H1 ATR percentile rank exceeds
    ``config.vol_regime_threshold`` within the trailing
    ``config.vol_regime_lookback`` bar window.

    Ported verbatim from ``Tier2StreamingTrader._update_h1_structure`` (ATR
    history and percentile-rank block logic).

    Parameters
    ----------
    h1_bars:
        Completed H1 bars (AR9).  Should contain at least
        ``config.vol_regime_lookback`` rows for a meaningful percentile rank;
        with fewer rows the function returns ``True`` (allow) conservatively.
    config:
        Active strategy parameters.

    Returns
    -------
    bool
        ``True``  → entry allowed (ATR in normal range).
        ``False`` → entry blocked (ATR in elevated regime).

    Raises
    ------
    ValueError
        On empty input or NaN in OHLCV.
    """
    _validate_bars(h1_bars, min_rows=1)

    # Vectorised True Range (matches live code's numpy expression)
    h = h1_bars["high"].to_numpy(dtype=float)
    lo = h1_bars["low"].to_numpy(dtype=float)
    c = h1_bars["close"].to_numpy(dtype=float)
    prev_c = np.roll(c, 1).astype(float)
    prev_c[0] = np.nan  # first bar has no previous close

    tr = np.where(
        np.isnan(prev_c),
        h - lo,
        np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c))),
    )

    # Rolling 20-bar mean ATR (min 5 periods) — same as live code
    atr_series = pd.Series(tr).rolling(20, min_periods=5).mean()

    # Build history: last vol_regime_lookback positive ATR values (one per bar),
    # matching the live code's "if h1_atr > 0: h1_atr_history.append(...)" logic
    atr_history = [v for v in atr_series.dropna() if v > 0]
    lookback = config.vol_regime_lookback
    atr_history = atr_history[-lookback:]

    if len(atr_history) < 20:
        return True  # insufficient history — conservatively allow entry

    current_atr = atr_history[-1]
    pct_rank = sum(1 for v in atr_history if v < current_atr) / len(atr_history)

    return pct_rank <= config.vol_regime_threshold


# ---------------------------------------------------------------------------
# Entry / exit decision functions (Story 1.3)
# ---------------------------------------------------------------------------

_NY_TZ = zoneinfo.ZoneInfo("America/New_York")


def make_entry_decision(
    sweep: SweepSignal,
    fvg: FVGSignal,
    config: StrategyConfig,
    **filter_results: bool,
) -> EntryDecision | None:
    """Resolve a trade entry from sweep/FVG confluence and filter verdicts.

    Ported from ``Tier2StreamingTrader._enter_trade`` (lines 976+).

    Returns ``None`` if:
    - ``fvg.direction != sweep.direction`` (misalignment), or
    - any value in ``filter_results`` is falsy.

    SL/TP sign convention:
    - BEARISH (short): SL is *above* entry, TP is *below* entry.
    - BULLISH (long): SL is *below* entry, TP is *above* entry.
    """
    if fvg.direction != sweep.direction:
        return None
    if not all(filter_results.values()):
        return None

    entry = fvg.entry_price
    gap = fvg.gap_size
    if fvg.direction == Direction.BEARISH:
        sl_price = entry + config.sl_multiplier * gap
        tp_price = entry - config.tp_multiplier * gap
    else:
        sl_price = entry - config.sl_multiplier * gap
        tp_price = entry + config.tp_multiplier * gap

    return EntryDecision(
        direction=fvg.direction,
        entry_price=entry,
        sl_price=sl_price,
        tp_price=tp_price,
        contracts=config.contracts_per_trade,
    )


def check_exit(
    bar: pd.Series,
    trade: EntryDecision,
    bars_held: int,
    config: StrategyConfig,
) -> ExitDecision | None:
    """Check triple-barrier exit conditions for an active trade.

    Ported verbatim from ``Tier2StreamingTrader._advance_active_trade``
    (lines 641-660).

    Resolution order (matches reference): SL first, then TP (only if SL
    did not trigger), then time stop (only if neither SL nor TP triggered).
    Returning ``None`` means the trade remains open.

    Bearish/SHORT: stop is above entry (``bar.high >= sl_price``),
    TP is below entry (``bar.low <= tp_price``).
    Bullish/LONG: stop is below entry (``bar.low <= sl_price``),
    TP is above entry (``bar.high >= tp_price``).
    """
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])
    bar_close = float(bar["close"])

    if trade.direction == Direction.BULLISH:
        if bar_low <= trade.sl_price:
            return ExitDecision(reason=ExitReason.SL, exit_price=trade.sl_price)
        if bar_high >= trade.tp_price:
            return ExitDecision(reason=ExitReason.TP, exit_price=trade.tp_price)
    else:  # BEARISH / SHORT
        if bar_high >= trade.sl_price:
            return ExitDecision(reason=ExitReason.SL, exit_price=trade.sl_price)
        if bar_low <= trade.tp_price:
            return ExitDecision(reason=ExitReason.TP, exit_price=trade.tp_price)

    if bars_held >= config.max_hold_bars:
        return ExitDecision(reason=ExitReason.TIME_STOP, exit_price=bar_close)

    return None


def kill_zone_filter(
    bar_timestamp: pd.Timestamp,
    config: StrategyConfig,
) -> bool:
    """Return ``True`` iff ``bar_timestamp`` falls in the configured kill zone.

    Kill zone: ``[config.kill_zone_start_et, config.kill_zone_end_et)``
    (start inclusive, end exclusive), evaluated in America/New_York local time
    so DST transitions are handled automatically.

    Never compare against a fixed UTC offset — ``bar_timestamp`` must be
    tz-aware, and this function converts to NY local time before comparing.
    See NFR19 and AR21.
    """
    bar_ny = bar_timestamp.astimezone(_NY_TZ)
    bar_time = bar_ny.time()
    return config.kill_zone_start_et <= bar_time < config.kill_zone_end_et


def check_m15_confirmation(
    h1_sweep: SweepSignal,
    m15_bars: pd.DataFrame,
) -> M15Confirmation:
    """Check if the last completed M15 bar closes in the H1 sweep direction.

    For a bearish sweep: confirmed when last M15 bar close < open (closes bearish).
    For a bullish sweep: confirmed when last M15 bar close > open (closes bullish).
    A doji (close == open) is NOT confirmed for either direction.

    Returns ``M15Confirmation(confirmed=False)`` when ``m15_bars`` is empty —
    the caller must decide whether to block or allow entry.

    Parameters
    ----------
    h1_sweep:
        The active H1 liquidity sweep (from ``detect_liquidity_sweep``).
    m15_bars:
        Completed M15 bars up to (but not including) the current bar.
        Must have canonical AR9 schema (timestamp index, OHLCV columns).

    Returns
    -------
    M15Confirmation
        ``confirmed=True`` + aligned ``direction`` when last bar aligns;
        ``confirmed=False, direction=None`` otherwise.
    """
    if len(m15_bars) == 0:
        return M15Confirmation(confirmed=False)

    last = m15_bars.iloc[-1]
    close = float(last["close"])
    open_ = float(last["open"])

    if h1_sweep.direction == Direction.BEARISH:
        confirmed = close < open_
    else:
        confirmed = close > open_

    return M15Confirmation(
        confirmed=confirmed,
        direction=h1_sweep.direction if confirmed else None,
    )


# ---------------------------------------------------------------------------
# Metric functions (Story 1.3)
# ---------------------------------------------------------------------------


def calc_profit_factor(pnls: list[float]) -> float:
    """Gross profit divided by gross loss.

    Returns ``float('inf')`` when there are no losing trades.
    Returns ``0.0`` when there are no winning trades (but losses exist).
    Edge cases follow the reference ``profit_factor`` in
    ``backtest_tier2_1year_validation.py:170``.
    """
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf")
    if gross_win == 0:
        return 0.0
    return gross_win / gross_loss


def calc_sharpe(daily_returns: list[float]) -> float:
    """Annualised Sharpe ratio: ``sqrt(252) * mean / std``.

    Input is a list of *daily* return values (not per-trade PnL).
    Returns ``0.0`` if fewer than 2 samples or if standard deviation is zero.
    Uses population std (``ddof=0``) consistent with the reference
    ``per_trade_sharpe`` in ``backtest_tier2_1year_validation.py:176``.
    """
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns, dtype=float)
    std = float(arr.std())
    if std == 0.0:
        return 0.0
    return float(sqrt(252) * float(arr.mean()) / std)


def calc_max_drawdown(equity: list[float]) -> float:
    """Maximum peak-to-trough decline of an equity curve in absolute units.

    ``equity`` is a list of cumulative P&L values in the same units as trades
    (USD for MNQ dollar P&L curves).  Returns the largest ``peak - trough``
    dollar value seen; always >= 0.  Returns ``0.0`` for curves shorter than
    2 points.

    For a normalised fraction (0.073 = 7.3% drawdown), use
    ``calc_max_drawdown_pct`` instead.

    Note: the reference ``max_drawdown`` in
    ``backtest_tier2_1year_validation.py:188`` takes *per-trade* PnL and
    accumulates internally; this function accepts an already-cumulated equity
    curve.  Both produce the same result given equivalent input.
    """
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calc_max_drawdown_pct(equity: list[float]) -> float:
    """Maximum peak-to-trough decline as a fraction of peak (e.g. 0.073 = 7.3%).

    ``equity`` is a list of cumulative P&L values.  Returns the largest
    ``(peak - trough) / peak`` seen; always in [0, 1].  Returns ``0.0`` when
    the equity curve has fewer than 2 points or the peak is <= 0 (prevents
    division by zero on loss-only curves).

    For an absolute dollar drawdown use ``calc_max_drawdown``.
    """
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd_pct = 0.0
    for val in equity:
        if val > peak:
            peak = val
        if peak > 0.0:
            dd_pct = (peak - val) / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    return max_dd_pct
