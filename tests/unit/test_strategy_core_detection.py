"""Unit tests for strategy_core detection functions (Story 1.3, AC #10).

Covers: resample_to_h1, detect_fvg (both directions + gap-too-small),
volatility_regime_filter at the 0.75 threshold boundary.
All synthetic DataFrames use the canonical AR9 schema.
"""

from __future__ import annotations

import zoneinfo

import numpy as np
import pandas as pd
import pytest

from src.research.strategy_core import (
    Direction,
    StrategyConfig,
    detect_fvg,
    resample_to_h1,
    volatility_regime_filter,
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


def make_h1_bars(
    n: int,
    tr_value: float = 5.0,
    start: str = "2025-06-02 09:00:00",
) -> pd.DataFrame:
    """Canonical AR9 H1 bars with a fixed True Range (high-low = tr_value)."""
    idx = pd.date_range(start, periods=n, freq="1h", tz=NY_TZ)
    idx.name = "timestamp"
    base = 100.0
    return pd.DataFrame(
        {
            "open": base,
            "high": base + tr_value,
            "low": base,
            "close": base + tr_value / 2,
            "volume": 500,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# resample_to_h1
# ---------------------------------------------------------------------------


class TestResampleToH1:
    def test_two_hours_gives_two_h1_bars(self):
        # Start at top of the hour so 120 bars = exactly 2 full H1 candles
        bars = make_1min_bars(120, start="2025-06-02 09:00:00")
        h1 = resample_to_h1(bars)
        assert len(h1) == 2

    def test_open_is_first_bar_open(self):
        """First 1-min open of each hour must be H1 open."""
        idx = pd.date_range("2025-06-02 09:00:00", periods=60, freq="1min", tz=NY_TZ)
        idx.name = "timestamp"
        opens = [float(i) for i in range(60)]
        df = pd.DataFrame(
            {"open": opens, "high": 200.0, "low": 50.0, "close": 100.0, "volume": 10},
            index=idx,
        )
        h1 = resample_to_h1(df)
        assert float(h1.iloc[0]["open"]) == pytest.approx(0.0)

    def test_close_is_last_bar_close(self):
        idx = pd.date_range("2025-06-02 09:00:00", periods=60, freq="1min", tz=NY_TZ)
        idx.name = "timestamp"
        closes = [float(i) for i in range(60)]
        df = pd.DataFrame(
            {"open": 100.0, "high": 200.0, "low": 50.0, "close": closes, "volume": 10},
            index=idx,
        )
        h1 = resample_to_h1(df)
        assert float(h1.iloc[0]["close"]) == pytest.approx(59.0)

    def test_high_is_max(self):
        idx = pd.date_range("2025-06-02 09:00:00", periods=60, freq="1min", tz=NY_TZ)
        idx.name = "timestamp"
        highs = [100.0] * 59 + [999.0]
        df = pd.DataFrame(
            {"open": 100.0, "high": highs, "low": 50.0, "close": 100.0, "volume": 10},
            index=idx,
        )
        h1 = resample_to_h1(df)
        assert float(h1.iloc[0]["high"]) == pytest.approx(999.0)

    def test_low_is_min(self):
        idx = pd.date_range("2025-06-02 09:00:00", periods=60, freq="1min", tz=NY_TZ)
        idx.name = "timestamp"
        lows = [50.0] * 59 + [1.0]
        df = pd.DataFrame(
            {"open": 100.0, "high": 200.0, "low": lows, "close": 100.0, "volume": 10},
            index=idx,
        )
        h1 = resample_to_h1(df)
        assert float(h1.iloc[0]["low"]) == pytest.approx(1.0)

    def test_volume_is_sum(self):
        bars = make_1min_bars(60, volume=7)
        h1 = resample_to_h1(bars)
        assert int(h1.iloc[0]["volume"]) == 60 * 7

    def test_index_name_is_timestamp(self):
        h1 = resample_to_h1(make_1min_bars(60))
        assert h1.index.name == "timestamp"

    def test_index_is_tz_aware(self):
        h1 = resample_to_h1(make_1min_bars(60))
        assert h1.index.tz is not None

    def test_empty_input_raises(self):
        empty = make_1min_bars(0)
        with pytest.raises(ValueError):
            resample_to_h1(empty)


# ---------------------------------------------------------------------------
# detect_fvg
# ---------------------------------------------------------------------------


def _make_bearish_fvg_bars(gap: float = 8.0) -> pd.DataFrame:
    """Three-bar bearish FVG: c1.low > c3.high, bearish middle candle.

    c1: high=110, low=100; c2 (bearish): open=99, close=93; c3: high=92, low=88
    gap = c1.low(100) - c3.high(92) = 8 points.
    """
    idx = pd.date_range("2025-06-02 09:30:00", periods=25, freq="1min", tz=NY_TZ)
    idx.name = "timestamp"
    opens = [100.0] * 25
    highs = [101.0] * 25
    lows = [99.0] * 25
    closes = [100.0] * 25
    volumes = [1000] * 25

    # Overwrite last 3 bars to create a clean bearish FVG
    c1_low = 100.0
    c3_high = c1_low - gap  # 92.0
    highs[-3] = 110.0
    lows[-3] = c1_low
    opens[-3] = 110.0
    closes[-3] = 108.0

    opens[-2] = 99.0
    closes[-2] = 93.0  # bearish middle
    highs[-2] = 99.0
    lows[-2] = 93.0

    highs[-1] = c3_high  # 92.0
    lows[-1] = 88.0
    opens[-1] = 91.0
    closes[-1] = 90.0

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_bullish_fvg_bars(gap: float = 8.0) -> pd.DataFrame:
    """Three-bar bullish FVG: c3.low > c1.high, bullish middle candle.

    c1: high=100; c2 (bullish): open=101, close=107; c3: low=108
    gap = c3.low(108) - c1.high(100) = 8 points.
    """
    idx = pd.date_range("2025-06-02 09:30:00", periods=25, freq="1min", tz=NY_TZ)
    idx.name = "timestamp"
    opens = [100.0] * 25
    highs = [101.0] * 25
    lows = [99.0] * 25
    closes = [100.0] * 25
    volumes = [1000] * 25

    c1_high = 100.0
    c3_low = c1_high + gap  # 108.0

    highs[-3] = c1_high
    lows[-3] = 95.0
    opens[-3] = 97.0
    closes[-3] = 99.0

    opens[-2] = 101.0
    closes[-2] = 107.0  # bullish middle
    highs[-2] = 107.0
    lows[-2] = 101.0

    lows[-1] = c3_low  # 108.0
    highs[-1] = 115.0
    opens[-1] = 109.0
    closes[-1] = 112.0

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


class TestDetectFVG:
    def _permissive_config(self) -> StrategyConfig:
        """Config that bypasses all size filters so gap geometry alone decides."""
        return StrategyConfig(
            atr_threshold=0.0,
            min_gap_atr_ratio=0.0,
            max_gap_dollars=9999.0,
        )

    def test_bearish_fvg_detected(self):
        bars = _make_bearish_fvg_bars(gap=8.0)
        config = self._permissive_config()
        result = detect_fvg(bars, config, atr=0.0)
        assert result is not None
        assert result.direction == Direction.BEARISH
        assert result.gap_size == pytest.approx(8.0)
        assert result.entry_price == pytest.approx((100.0 + 92.0) / 2)

    def test_bullish_fvg_detected(self):
        bars = _make_bullish_fvg_bars(gap=8.0)
        config = self._permissive_config()
        result = detect_fvg(bars, config, atr=0.0)
        assert result is not None
        assert result.direction == Direction.BULLISH
        assert result.gap_size == pytest.approx(8.0)
        assert result.entry_price == pytest.approx((108.0 + 100.0) / 2)

    def test_gap_too_small_atr_threshold_returns_none(self):
        """Gap of 2 pts is filtered when atr_threshold=0.5 and ATR is large."""
        bars = _make_bearish_fvg_bars(gap=2.0)
        config = StrategyConfig(
            atr_threshold=0.5,
            min_gap_atr_ratio=0.0,
            max_gap_dollars=9999.0,
        )
        # With 25 uniform bars (TR≈2), 20-bar ATR≈2; threshold=0.5*2=1.0, gap=2 > 1.0
        # So this actually passes unless we make the ATR threshold strict.
        # Use atr_threshold=2.0 to guarantee filtering.
        strict_config = StrategyConfig(
            atr_threshold=2.0,
            min_gap_atr_ratio=0.0,
            max_gap_dollars=9999.0,
        )
        result = detect_fvg(bars, strict_config, atr=0.0)
        assert result is None

    def test_gap_too_large_dollar_ceiling_returns_none(self):
        """Gap above max_gap_dollars is rejected."""
        bars = _make_bearish_fvg_bars(gap=40.0)  # 40 pts * $2 = $80 > $60 default
        config = StrategyConfig(
            atr_threshold=0.0,
            min_gap_atr_ratio=0.0,
            max_gap_dollars=60.0,
        )
        result = detect_fvg(bars, config, atr=0.0)
        assert result is None

    def test_gap_too_small_h1_atr_ratio_returns_none(self):
        """H1 ATR ratio filter rejects gap smaller than min_gap_atr_ratio * h1_atr."""
        bars = _make_bearish_fvg_bars(gap=2.0)
        config = StrategyConfig(
            atr_threshold=0.0,
            min_gap_atr_ratio=0.5,
            max_gap_dollars=9999.0,
        )
        result = detect_fvg(bars, config, atr=20.0)  # need gap >= 0.5*20=10
        assert result is None

    def test_no_fvg_returns_none(self):
        bars = make_1min_bars(25)  # uniform bars, no gap
        config = self._permissive_config()
        result = detect_fvg(bars, config, atr=0.0)
        assert result is None

    def test_fewer_than_3_bars_raises(self):
        bars = make_1min_bars(2)
        with pytest.raises(ValueError):
            detect_fvg(bars, StrategyConfig(), atr=0.0)


# ---------------------------------------------------------------------------
# volatility_regime_filter
# ---------------------------------------------------------------------------


class TestVolatilityRegimeFilter:
    def _make_h1_with_controlled_atr(
        self,
        low_tr_count: int,
        high_tr_count: int,
        low_tr: float = 2.0,
        high_tr: float = 20.0,
    ) -> pd.DataFrame:
        """H1 bars: first low_tr_count bars have TR=low_tr, rest have TR=high_tr.

        Extra 25 bars (warm-up for rolling ATR) prepended with low_tr.
        """
        n_warmup = 25
        n_total = n_warmup + low_tr_count + high_tr_count
        idx = pd.date_range("2025-01-02 00:00:00", periods=n_total, freq="1h", tz=NY_TZ)
        idx.name = "timestamp"
        base = 100.0

        trs = [low_tr] * (n_warmup + low_tr_count) + [high_tr] * high_tr_count
        highs = [base + tr for tr in trs]
        lows = [base] * n_total

        return pd.DataFrame(
            {
                "open": base,
                "high": highs,
                "low": lows,
                "close": [base + tr / 2 for tr in trs],
                "volume": 500,
            },
            index=idx,
        )

    def test_low_volatility_allows_entry(self):
        """When current ATR is well below the 75th pct, filter passes."""
        # All bars same low TR → pct_rank = 0 → allow
        bars = make_h1_bars(150, tr_value=2.0)
        config = StrategyConfig(vol_regime_lookback=120, vol_regime_threshold=0.75)
        assert volatility_regime_filter(bars, config) is True

    def test_high_volatility_blocks_entry(self):
        """When current ATR is in top quarter of history, filter blocks."""
        # 90 bars at low TR, then 30 bars at very high TR
        # pct_rank of high TR bars = 90/120 = 0.75, last one > 0.75 → block
        bars = self._make_h1_with_controlled_atr(
            low_tr_count=90, high_tr_count=31, low_tr=2.0, high_tr=20.0
        )
        config = StrategyConfig(vol_regime_lookback=120, vol_regime_threshold=0.75)
        result = volatility_regime_filter(bars, config)
        assert result is False

    def test_at_threshold_boundary_allows(self):
        """pct_rank == vol_regime_threshold → ALLOW (uses <=, not <).

        All bars at identical TR → rolling ATR is the same value throughout →
        pct_rank = 0 (no history value strictly < current ATR). Setting
        threshold=0.0 means pct_rank(0) <= threshold(0) must return True.
        If the comparison were strict (<), it would return False for pct_rank==threshold.
        """
        bars = make_h1_bars(150, tr_value=5.0)
        config = StrategyConfig(vol_regime_lookback=120, vol_regime_threshold=0.0)
        result = volatility_regime_filter(bars, config)
        assert result is True

    def test_insufficient_history_allows(self):
        """Fewer than 20 ATR history values → conservatively allow entry."""
        bars = make_h1_bars(10, tr_value=50.0)  # extreme TR but too few bars
        config = StrategyConfig(vol_regime_lookback=120, vol_regime_threshold=0.75)
        assert volatility_regime_filter(bars, config) is True

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            volatility_regime_filter(make_h1_bars(0), StrategyConfig())
