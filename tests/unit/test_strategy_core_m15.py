"""Unit tests for strategy_core M15 functions (Story 2.3).

Covers: resample_to_m15 schema/aggregation, check_m15_confirmation
(bearish confirmed, bearish rejected, bullish confirmed, empty bars, doji edge case).
All synthetic DataFrames use the canonical AR9 schema.
"""

from __future__ import annotations

import zoneinfo

import pandas as pd
import pytest

from src.research.strategy_core import (
    Direction,
    M15Confirmation,
    SweepSignal,
    check_m15_confirmation,
    resample_to_m15,
)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_1min_bars(
    n: int,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    volume: int = 1000,
    start: str = "2025-06-02 09:00:00",
) -> pd.DataFrame:
    """Canonical AR9 1-min bars: tz-aware DatetimeIndex named 'timestamp'."""
    idx = pd.date_range(start, periods=n, freq="1min", tz=NY_TZ)
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def make_sweep(direction: Direction) -> SweepSignal:
    return SweepSignal(direction=direction, bars_ago=1, sweep_price=100.0)


def make_m15_bar(
    open_: float,
    close: float,
    ts: str = "2025-06-02 09:00:00",
    high_extra: float = 0.5,
    low_extra: float = 0.5,
) -> pd.DataFrame:
    """Single M15 bar with canonical AR9 schema."""
    high = max(open_, close) + high_extra
    low = min(open_, close) - low_extra
    idx = pd.DatetimeIndex([pd.Timestamp(ts, tz=NY_TZ)], name="timestamp")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000},
        index=idx,
    )


# ---------------------------------------------------------------------------
# resample_to_m15 tests
# ---------------------------------------------------------------------------


class TestResampleToM15:
    def test_schema_output(self):
        """Output has correct DatetimeIndex name, OHLCV columns, 15m periods."""
        bars = make_1min_bars(60, start="2025-06-02 09:00:00")
        m15 = resample_to_m15(bars)

        assert isinstance(m15.index, pd.DatetimeIndex)
        assert m15.index.name == "timestamp"
        for col in ("open", "high", "low", "close", "volume"):
            assert col in m15.columns, f"Missing column: {col}"
        # 60 1-min bars → 4 completed 15m bars (09:00–09:14, 09:15–09:29, ...)
        assert len(m15) == 4

    def test_aggregation(self):
        """open=first, high=max, low=min, close=last, volume=sum over 15 bars."""
        # 15 bars with varying values so we can check aggregation
        bars_list = []
        base = pd.Timestamp("2025-06-02 09:00:00", tz=NY_TZ)
        for i in range(15):
            ts = base + pd.Timedelta(minutes=i)
            bars_list.append(
                {
                    "timestamp": ts,
                    "open": 100.0 + i,  # first = 100.0
                    "high": 105.0 + i,  # max = 119.0
                    "low": 98.0 - i,   # min = 84.0
                    "close": 101.0 + i,  # last = 115.0
                    "volume": 100,        # sum = 1500
                }
            )
        df = pd.DataFrame(bars_list).set_index("timestamp")
        df.index.name = "timestamp"

        m15 = resample_to_m15(df)
        assert len(m15) == 1
        row = m15.iloc[0]
        assert row["open"] == pytest.approx(100.0)   # first
        assert row["high"] == pytest.approx(119.0)   # max (105 + 14)
        assert row["low"] == pytest.approx(84.0)     # min (98 - 14)
        assert row["close"] == pytest.approx(115.0)  # last (101 + 14)
        assert row["volume"] == pytest.approx(1500)   # sum

    def test_empty_input_raises(self):
        """Empty DataFrame raises ValueError."""
        idx = pd.DatetimeIndex([], name="timestamp", tz=NY_TZ)
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"], index=idx
        )
        with pytest.raises(ValueError, match="empty"):
            resample_to_m15(empty)

    def test_15m_input_returns_same_bars(self):
        """15m input bars are returned unchanged (identity resample)."""
        # Build a 15m bar manually
        idx = pd.date_range("2025-06-02 09:00:00", periods=3, freq="15min", tz=NY_TZ)
        idx.name = "timestamp"
        df = pd.DataFrame(
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 5000},
            index=idx,
        )
        m15 = resample_to_m15(df)
        assert len(m15) == 3
        assert list(m15.index.freq.freqstr if hasattr(m15.index, "freq") else ["15T"] * 3)


# ---------------------------------------------------------------------------
# check_m15_confirmation tests
# ---------------------------------------------------------------------------


class TestCheckM15Confirmation:
    def test_bearish_confirmed(self):
        """Bearish sweep + last bar close < open → confirmed=True, direction=BEARISH."""
        sweep = make_sweep(Direction.BEARISH)
        m15 = make_m15_bar(open_=100.0, close=99.0)  # bearish candle
        result = check_m15_confirmation(sweep, m15)
        assert result.confirmed is True
        assert result.direction == Direction.BEARISH

    def test_bearish_rejected_bullish_bar(self):
        """Bearish sweep + last bar close > open → confirmed=False."""
        sweep = make_sweep(Direction.BEARISH)
        m15 = make_m15_bar(open_=100.0, close=101.0)  # bullish candle
        result = check_m15_confirmation(sweep, m15)
        assert result.confirmed is False
        assert result.direction is None

    def test_bullish_confirmed(self):
        """Bullish sweep + last bar close > open → confirmed=True, direction=BULLISH."""
        sweep = make_sweep(Direction.BULLISH)
        m15 = make_m15_bar(open_=100.0, close=101.0)  # bullish candle
        result = check_m15_confirmation(sweep, m15)
        assert result.confirmed is True
        assert result.direction == Direction.BULLISH

    def test_bullish_rejected_bearish_bar(self):
        """Bullish sweep + last bar close < open → confirmed=False."""
        sweep = make_sweep(Direction.BULLISH)
        m15 = make_m15_bar(open_=100.0, close=99.0)  # bearish candle
        result = check_m15_confirmation(sweep, m15)
        assert result.confirmed is False
        assert result.direction is None

    def test_doji_not_confirmed_bearish(self):
        """Dojo (close == open) with bearish sweep → NOT confirmed (close not strictly < open)."""
        sweep = make_sweep(Direction.BEARISH)
        m15 = make_m15_bar(open_=100.0, close=100.0)  # doji
        result = check_m15_confirmation(sweep, m15)
        assert result.confirmed is False
        assert result.direction is None

    def test_doji_not_confirmed_bullish(self):
        """Dojo (close == open) with bullish sweep → NOT confirmed."""
        sweep = make_sweep(Direction.BULLISH)
        m15 = make_m15_bar(open_=100.0, close=100.0)  # doji
        result = check_m15_confirmation(sweep, m15)
        assert result.confirmed is False
        assert result.direction is None

    def test_empty_bars_returns_not_confirmed(self):
        """Empty m15_bars → M15Confirmation(confirmed=False, direction=None)."""
        sweep = make_sweep(Direction.BEARISH)
        idx = pd.DatetimeIndex([], name="timestamp", tz=NY_TZ)
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"], index=idx
        )
        result = check_m15_confirmation(sweep, empty)
        assert result.confirmed is False
        assert result.direction is None

    def test_uses_last_bar_only(self):
        """With multiple M15 bars, only the last bar determines confirmation."""
        sweep = make_sweep(Direction.BEARISH)
        # Build two bars: first is bullish, last is bearish
        ts1 = pd.Timestamp("2025-06-02 09:00:00", tz=NY_TZ)
        ts2 = pd.Timestamp("2025-06-02 09:15:00", tz=NY_TZ)
        idx = pd.DatetimeIndex([ts1, ts2], name="timestamp")
        m15 = pd.DataFrame(
            {
                "open":  [100.0, 101.0],
                "high":  [102.0, 102.0],
                "low":   [99.0,  99.0],
                "close": [101.0, 99.5],   # bar1=bullish, bar2=bearish
                "volume": [1000, 1000],
            },
            index=idx,
        )
        result = check_m15_confirmation(sweep, m15)
        # Last bar is bearish → should be confirmed
        assert result.confirmed is True
        assert result.direction == Direction.BEARISH
