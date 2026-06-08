"""sorm_core.py — Pure, stateless logic for the SORM Combine Strategy.

SORM: Session Open Range Mean Reversion
Instrument: MNQ (Micro E-mini Nasdaq-100 futures)
Pre-registration: 2026-06-08

Strategy summary:
  • Build Opening Range (ORB) from 09:30–09:45 ET.
  • Detect extension: 1-min close ≥ 0.5 × orb_size beyond ORB boundary,
    between 09:45–10:45 ET.
  • Enter on first close back inside the ORB, RSI(14) 30–70 pointing toward mid.
  • Stop at extension extreme wick ± 1 tick; skip if > $200/contract.
  • TP1 (60%) at orb_mid; TP2 (40%) at opposite ORB boundary.
  • Hard close at 11:30 ET; one trade per session.

Purity contract: no I/O, no logging, no wall-clock, no mutable module state.
Same inputs always produce the same outputs.
"""

from __future__ import annotations

import csv
import zoneinfo
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
POINT_VALUE_USD: float = 2.0   # USD per MNQ index point (P&L scaling)
TICK_SIZE: float = 0.25        # MNQ minimum price increment
ET = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")


# ── Enums ────────────────────────────────────────────────────────────────────

class Direction(Enum):
    BEARISH = "BEARISH"   # Extension above ORB → fade short back to mid
    BULLISH = "BULLISH"   # Extension below ORB → fade long back to mid


class ExitReason(Enum):
    TP1 = "TP1"           # 60% closed at orb_mid
    TP2 = "TP2"           # 40% closed at opposite boundary
    SL = "SL"             # Stop hit
    TIME_STOP = "TIME_STOP"  # Hard close at 11:30 ET


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SORMConfig:
    """Immutable parameter set — frozen before any backtest run."""
    # Timing (ET)
    orb_start_et: time = time(9, 30)
    orb_end_et: time = time(9, 45)    # exclusive: ORB includes bars up to 09:44
    extension_start_et: time = time(9, 45)
    extension_end_et: time = time(10, 45)
    hard_close_et: time = time(11, 30)
    # Entry conditions
    extension_threshold: float = 0.5   # fraction of orb_size beyond boundary
    rsi_period: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    rsi_direction_lookback: int = 3
    # Risk
    stop_skip_threshold_usd: float = 200.0
    stop_small_threshold_usd: float = 100.0
    contracts_small_stop: int = 2
    contracts_large_stop: int = 1
    daily_loss_limit_usd: float = -300.0
    # Exit
    tp1_fraction: float = 0.60
    max_trades_per_session: int = 1
    # Quality filter
    orb_min_size_points: float = 2.0
    # Market
    point_value_usd: float = 2.0
    tick_size: float = 0.25
    commission_per_contract_rt: float = 0.40


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OpeningRange:
    """The 09:30–09:44 ET session opening range."""
    high: float
    low: float
    mid: float
    size: float   # high - low in index points
    date_et: date


@dataclass(frozen=True)
class ExtensionEvent:
    """A detected ORB extension."""
    direction: Direction
    extreme_price: float      # wick extreme used for stop calculation
    detection_bar_ts: datetime  # ET timestamp of the first confirming bar's close
    extension_close: float    # close price of the confirming bar


@dataclass(frozen=True)
class EntrySignal:
    """A complete entry setup ready for execution."""
    direction: Direction
    entry_price: float        # close of the reversion bar (market entry)
    stop_price: float         # stop level
    tp1_price: float          # 60% target = orb_mid
    tp2_price: float          # 40% target = opposite boundary
    contracts: int
    stop_dist_usd: float      # stop distance in USD (for position sizing reference)
    entry_bar_ts: datetime    # ET timestamp of entry bar


@dataclass
class TradeResult:
    """One completed trade."""
    date_et: date
    direction: str
    entry_price: float
    exit_price: float
    contracts: int
    pnl_usd: float
    exit_reason: str
    entry_ts: datetime
    exit_ts: datetime


# ── Core Detection Functions ─────────────────────────────────────────────────

def build_opening_range(session_df: pd.DataFrame, cfg: SORMConfig) -> Optional[OpeningRange]:
    """Build the ORB from bars in the 09:30–09:44 ET window.

    Args:
        session_df: DataFrame with ET DatetimeIndex, columns open/high/low/close.
                    Must be pre-filtered to a single session date.
        cfg: frozen SORM config.

    Returns:
        OpeningRange or None if insufficient bars or range too small.
    """
    orb_bars = session_df.between_time(
        cfg.orb_start_et.strftime("%H:%M"),
        cfg.orb_end_et.strftime("%H:%M"),
        inclusive="left",   # [09:30, 09:45) — bars starting before 09:45
    )
    if len(orb_bars) < 3:
        return None

    orb_high = float(orb_bars["high"].max())
    orb_low = float(orb_bars["low"].min())
    orb_size = orb_high - orb_low

    if orb_size < cfg.orb_min_size_points:
        return None

    session_date = orb_bars.index[0].date()
    return OpeningRange(
        high=orb_high,
        low=orb_low,
        mid=(orb_high + orb_low) / 2.0,
        size=orb_size,
        date_et=session_date,
    )


def detect_extension(
    session_df: pd.DataFrame,
    orb: OpeningRange,
    cfg: SORMConfig,
) -> Optional[ExtensionEvent]:
    """Find the first ORB extension in the 09:45–10:44 ET window.

    An extension is a 1-min bar close that penetrates ≥ cfg.extension_threshold
    × orb_size beyond the ORB boundary.

    Returns the FIRST qualifying extension (bearish or bullish), with the
    extreme wick price from all bars UP TO AND INCLUDING the detection bar.
    """
    ext_bars = session_df.between_time(
        cfg.extension_start_et.strftime("%H:%M"),
        cfg.extension_end_et.strftime("%H:%M"),
        inclusive="left",   # [09:45, 10:45)
    )
    if ext_bars.empty:
        return None

    threshold_pts = cfg.extension_threshold * orb.size

    # Track rolling extreme as we walk through bars
    running_high = orb.high
    running_low = orb.low

    for ts, row in ext_bars.iterrows():
        running_high = max(running_high, float(row["high"]))
        running_low = min(running_low, float(row["low"]))
        close = float(row["close"])

        # Bearish extension: close above ORB_high + threshold
        if close > orb.high + threshold_pts:
            return ExtensionEvent(
                direction=Direction.BEARISH,
                extreme_price=running_high,
                detection_bar_ts=ts,
                extension_close=close,
            )

        # Bullish extension: close below ORB_low - threshold
        if close < orb.low - threshold_pts:
            return ExtensionEvent(
                direction=Direction.BULLISH,
                extreme_price=running_low,
                detection_bar_ts=ts,
                extension_close=close,
            )

    return None


def check_reversion_to_mid(
    post_ext_df: pd.DataFrame,
    orb: OpeningRange,
    extension: ExtensionEvent,
) -> bool:
    """Check whether price touches orb_mid before the end of post_ext_df.

    Used by the reversion-rate study (no entry filters applied).

    Args:
        post_ext_df: bars AFTER the extension detection bar, up to 11:30 ET.
        orb: the session opening range.
        extension: the detected extension event.

    Returns:
        True if any bar's price range includes orb_mid.
    """
    if post_ext_df.empty:
        return False

    mid = orb.mid
    for _, row in post_ext_df.iterrows():
        bar_low = float(row["low"])
        bar_high = float(row["high"])
        if bar_low <= mid <= bar_high:
            return True
        # Also count a close at or past mid (some candles gap through)
        close = float(row["close"])
        if extension.direction == Direction.BEARISH and close <= mid:
            return True
        if extension.direction == Direction.BULLISH and close >= mid:
            return True

    return False


def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Simple RSI calculation (Wilder exponential smoothing via rolling mean).

    Args:
        closes: Series of close prices.
        period: RSI lookback.

    Returns:
        Series of RSI values (0–100).
    """
    delta = closes.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    rs = avg_gains / avg_losses.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _stop_distance_usd(stop_price: float, entry_price: float, direction: Direction) -> float:
    """Stop distance in USD per contract."""
    if direction == Direction.BEARISH:
        dist_pts = stop_price - entry_price   # stop is above entry
    else:
        dist_pts = entry_price - stop_price   # stop is below entry
    return max(dist_pts, 0.0) * POINT_VALUE_USD


def detect_reversion_entry(
    post_ext_df: pd.DataFrame,
    orb: OpeningRange,
    extension: ExtensionEvent,
    rsi_series: pd.Series,
    cfg: SORMConfig,
) -> Optional[EntrySignal]:
    """Find the first valid reversion entry after an extension.

    Applies all entry filters:
    - First 1-min close back inside the ORB
    - RSI(14) in [rsi_low, rsi_high] and pointing toward mid
    - Stop distance ≤ stop_skip_threshold_usd

    Args:
        post_ext_df: bars after the extension detection bar, up to hard_close.
        orb: session opening range.
        extension: the detected extension event.
        rsi_series: full-session RSI series (aligned to post_ext_df index).
        cfg: SORM config.

    Returns:
        EntrySignal or None.
    """
    if post_ext_df.empty:
        return None

    for ts, row in post_ext_df.iterrows():
        close = float(row["close"])

        # ── Re-entry condition: close back inside ORB ────────────────
        if extension.direction == Direction.BEARISH:
            if close >= orb.high:
                continue   # still above ORB, no re-entry yet
        else:
            if close <= orb.low:
                continue   # still below ORB, no re-entry yet

        # ── RSI filter ───────────────────────────────────────────────
        if ts not in rsi_series.index:
            continue
        rsi_now = rsi_series.loc[ts]
        if pd.isna(rsi_now):
            continue
        if not (cfg.rsi_low <= rsi_now <= cfg.rsi_high):
            continue

        # RSI must be pointing toward mid (directional filter)
        lookback_idx = post_ext_df.index.get_loc(ts)
        lb = cfg.rsi_direction_lookback
        if lookback_idx >= lb:
            ts_prev = post_ext_df.index[lookback_idx - lb]
            if ts_prev in rsi_series.index:
                rsi_prev = rsi_series.loc[ts_prev]
                if not pd.isna(rsi_prev):
                    if extension.direction == Direction.BEARISH and rsi_now >= rsi_prev:
                        continue   # RSI rising — momentum still up, skip
                    if extension.direction == Direction.BULLISH and rsi_now <= rsi_prev:
                        continue   # RSI falling — momentum still down, skip

        # ── Stop calculation ─────────────────────────────────────────
        if extension.direction == Direction.BEARISH:
            stop_price = extension.extreme_price + cfg.tick_size
            tp1_price = orb.mid
            tp2_price = orb.low
        else:
            stop_price = extension.extreme_price - cfg.tick_size
            tp1_price = orb.mid
            tp2_price = orb.high

        stop_dist_usd = _stop_distance_usd(stop_price, close, extension.direction)

        # ── Stop-size filter ─────────────────────────────────────────
        if stop_dist_usd > cfg.stop_skip_threshold_usd:
            continue   # risk too wide — skip

        # ── Contracts ────────────────────────────────────────────────
        if stop_dist_usd < cfg.stop_small_threshold_usd:
            contracts = cfg.contracts_small_stop
        else:
            contracts = cfg.contracts_large_stop

        return EntrySignal(
            direction=extension.direction,
            entry_price=close,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            contracts=contracts,
            stop_dist_usd=stop_dist_usd,
            entry_bar_ts=ts,
        )

    return None


# ── Metric Helpers ───────────────────────────────────────────────────────────

def calc_profit_factor(trades: list[TradeResult]) -> float:
    """Gross profit / gross loss. Returns 0.0 if no losing trades."""
    gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    gross_loss = sum(-t.pnl_usd for t in trades if t.pnl_usd < 0)
    return gross_profit / gross_loss if gross_loss > 0 else 0.0


def calc_win_rate(trades: list[TradeResult]) -> float:
    """Fraction of trades with pnl > 0."""
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl_usd > 0) / len(trades)


def calc_max_drawdown(trades: list[TradeResult]) -> float:
    """Maximum peak-to-trough drawdown in USD on the cumulative P&L curve."""
    if not trades:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.pnl_usd
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calc_trailing_drawdown_path(trades: list[TradeResult]) -> list[float]:
    """Return the running trailing-DD value after each trade.

    The Topstep trailing DD rises permanently with the equity peak:
    trailing_dd_level = equity_peak - topstep_dd_amount
    We return the DD_consumed list (how much of the $2k we've used).
    """
    TOPSTEP_DD = 2000.0
    equity = 50000.0
    peak = 50000.0
    consumed = []
    for t in trades:
        equity += t.pnl_usd
        if equity > peak:
            peak = equity
        dd_consumed = peak - equity
        consumed.append(dd_consumed)
    return consumed


def calc_per_trade_sharpe(trades: list[TradeResult]) -> float:
    """Per-trade Sharpe (mean P&L / std P&L). Returns 0.0 if < 2 trades."""
    pnls = [t.pnl_usd for t in trades]
    if len(pnls) < 2:
        return 0.0
    mean = np.mean(pnls)
    std = np.std(pnls, ddof=1)
    return float(mean / std) if std > 0 else 0.0


def calc_consistency_ratio(trades: list[TradeResult]) -> float:
    """Best single-day P&L as fraction of total cumulative P&L.

    Topstep cap: no day > 50% of accumulated profit. Returns the peak ratio.
    """
    by_date: dict[date, float] = {}
    for t in trades:
        d = t.date_et
        by_date[d] = by_date.get(d, 0.0) + t.pnl_usd

    cumulative = 0.0
    peak_ratio = 0.0
    for d in sorted(by_date):
        daily = by_date[d]
        cumulative += daily
        if cumulative > 0 and daily > 0:
            ratio = daily / cumulative
            if ratio > peak_ratio:
                peak_ratio = ratio
    return peak_ratio


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_bars_et(
    csv_paths: list[Path],
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Load 1-min bars from one or more CSVs, filter by UTC range, return ET DataFrame.

    Columns in output: open, high, low, close, volume.
    Index: DatetimeIndex (America/New_York, tz-aware), named 'timestamp'.

    Args:
        csv_paths: ordered list of CSV file paths (e.g. [2025.csv, 2026ytd.csv]).
        start_utc: inclusive start (tz-aware UTC).
        end_utc: inclusive end (tz-aware UTC).
    """
    rows = []
    for path in csv_paths:
        if not path.exists():
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row["timestamp"]
                # Handle both '+00:00' and 'Z' suffixes
                ts_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_utc < start_utc or ts_utc > end_utc:
                    continue
                ts_et = ts_utc.astimezone(ET)
                rows.append({
                    "timestamp": ts_et,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(float(row.get("volume", 0))),
                })

    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).rename_axis("timestamp")

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df
