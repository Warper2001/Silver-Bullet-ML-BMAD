"""Unit tests for IFVG fallback entry infrastructure (S27).

Tests cover:
- strategy_core.check_ifvg_trigger() pure function
- IFVGCandidate storage on pending expiry in Tier2StreamingTrader
- IFVG candidate cleared when H1 sweep expires
- IFVG candidate consumed on trigger
- Feature flag correctly gates all behaviour
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.research.strategy_core import (
    Direction,
    IFVGCandidate,
    StrategyConfig,
    check_ifvg_trigger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ifvg_candidate(
    direction: Direction = Direction.BEARISH,
    gap_high: float = 20100.0,
    gap_low: float = 20090.0,
) -> IFVGCandidate:
    gap_size = gap_high - gap_low
    return IFVGCandidate(
        direction=direction,
        gap_high=gap_high,
        gap_low=gap_low,
        gap_size=gap_size,
        formed_at=datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc),
    )


def _bar(close: float, high: float | None = None, low: float | None = None) -> pd.Series:
    return pd.Series({
        "high": high if high is not None else close + 1.0,
        "low": low if low is not None else close - 1.0,
        "close": close,
    })


def _make_trader():
    """Return a Tier2StreamingTrader with mocked external dependencies."""
    from src.research.tier2_streaming_working import Tier2StreamingTrader
    with (
        patch("src.research.tier2_streaming_working.MetaLabelingFilter"),
        patch("src.research.tier2_streaming_working.LRRegimeFilter"),
    ):
        trader = Tier2StreamingTrader.__new__(Tier2StreamingTrader)
        trader._ts_client = MagicMock()
        trader._ts_client.cancel_order = AsyncMock(return_value=True)
    return trader


# ---------------------------------------------------------------------------
# strategy_core.check_ifvg_trigger — pure function tests
# ---------------------------------------------------------------------------

class TestCheckIfvgTrigger:
    def test_bearish_fires_on_close_above_gap_high(self):
        candidate = _make_ifvg_candidate(Direction.BEARISH, gap_high=20100.0, gap_low=20090.0)
        config = StrategyConfig(entry_pct=0.5)
        signal = check_ifvg_trigger(_bar(close=20101.0), candidate, config)
        assert signal is not None
        assert signal.direction == Direction.BEARISH

    def test_bearish_no_fire_at_or_below_gap_high(self):
        candidate = _make_ifvg_candidate(Direction.BEARISH, gap_high=20100.0, gap_low=20090.0)
        config = StrategyConfig(entry_pct=0.5)
        assert check_ifvg_trigger(_bar(close=20100.0), candidate, config) is None
        assert check_ifvg_trigger(_bar(close=20095.0), candidate, config) is None

    def test_bearish_entry_price_is_entry_pct_from_gap_high(self):
        candidate = _make_ifvg_candidate(Direction.BEARISH, gap_high=20100.0, gap_low=20090.0)
        config = StrategyConfig(entry_pct=0.5)
        signal = check_ifvg_trigger(_bar(close=20101.0), candidate, config)
        assert signal is not None
        # entry = gap_high - gap_size * entry_pct = 20100 - 10 * 0.5 = 20095
        assert abs(signal.entry_price - 20095.0) < 1e-9

    def test_bullish_fires_on_close_below_gap_low(self):
        candidate = _make_ifvg_candidate(Direction.BULLISH, gap_high=20100.0, gap_low=20090.0)
        config = StrategyConfig(entry_pct=0.5)
        signal = check_ifvg_trigger(_bar(close=20089.0), candidate, config)
        assert signal is not None
        assert signal.direction == Direction.BULLISH

    def test_bullish_no_fire_at_or_above_gap_low(self):
        candidate = _make_ifvg_candidate(Direction.BULLISH, gap_high=20100.0, gap_low=20090.0)
        config = StrategyConfig(entry_pct=0.5)
        assert check_ifvg_trigger(_bar(close=20090.0), candidate, config) is None

    def test_returned_signal_preserves_gap_bounds(self):
        candidate = _make_ifvg_candidate(Direction.BEARISH, gap_high=20100.0, gap_low=20090.0)
        config = StrategyConfig(entry_pct=0.5)
        signal = check_ifvg_trigger(_bar(close=20101.0), candidate, config)
        assert signal is not None
        assert signal.high == 20100.0
        assert signal.low == 20090.0
        assert signal.gap_size == 10.0


# ---------------------------------------------------------------------------
# Tier2StreamingTrader — IFVG candidate lifecycle
# ---------------------------------------------------------------------------

class TestIfvgCandidateLifecycle:
    @pytest.mark.asyncio
    async def test_candidate_stored_on_pending_expiry_when_flag_enabled(self):
        from src.research.tier2_streaming_working import ActiveTrade, Tier2StreamingTrader
        from src.research.strategy_core import StrategyConfig

        trader = _make_trader()
        trader._strategy_config = StrategyConfig(
            enable_ifvg_fallback=True, max_pending_bars=5
        )
        trader._ifvg_candidate = None
        trader._active_entry_decision = None
        trader.active_trade = ActiveTrade(
            bar_index=0,
            entry_time=datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc),
            direction="SHORT",
            entry_price=20095.0,
            tp_price=20000.0,
            sl_price=20150.0,
            bars_held=5,
            pending_entry=True,
            gap_size=10.0,
        )

        from src.research.tier2_streaming_working import DollarBar
        bar = MagicMock(spec=DollarBar)
        bar.high = 20080.0
        bar.low = 20070.0
        bar.close = 20075.0
        bar.timestamp = datetime(2026, 1, 6, 10, 5, tzinfo=timezone.utc)

        await trader._advance_active_trade(bar)

        assert trader._ifvg_candidate is not None
        assert trader._ifvg_candidate.direction == Direction.BEARISH
        assert trader._ifvg_candidate.gap_size == 10.0

    @pytest.mark.asyncio
    async def test_candidate_not_stored_when_flag_disabled(self):
        from src.research.tier2_streaming_working import ActiveTrade, DollarBar
        from src.research.strategy_core import StrategyConfig

        trader = _make_trader()
        trader._strategy_config = StrategyConfig(
            enable_ifvg_fallback=False, max_pending_bars=5
        )
        trader._ifvg_candidate = None
        trader._active_entry_decision = None
        trader.active_trade = ActiveTrade(
            bar_index=0,
            entry_time=datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc),
            direction="SHORT",
            entry_price=20095.0,
            tp_price=20000.0,
            sl_price=20150.0,
            bars_held=5,
            pending_entry=True,
            gap_size=10.0,
        )

        bar = MagicMock(spec=DollarBar)
        bar.high = 20080.0
        bar.low = 20070.0
        bar.close = 20075.0
        bar.timestamp = datetime(2026, 1, 6, 10, 5, tzinfo=timezone.utc)

        await trader._advance_active_trade(bar)

        assert trader._ifvg_candidate is None

    def test_candidate_cleared_when_bearish_sweep_expires(self):
        from src.research.tier2_streaming_working import Tier2StreamingTrader
        import numpy as np
        import pandas as pd

        trader = _make_trader()
        trader._strategy_config = StrategyConfig(enable_ifvg_fallback=True)
        trader._ifvg_candidate = _make_ifvg_candidate(Direction.BEARISH)
        trader.h1_bearish_sweep_active = True   # was active
        trader.h1_bullish_sweep_active = False
        trader._m15_choch_active = True
        trader._m15_last_bar_ts = datetime.min.replace(tzinfo=timezone.utc)
        trader._cached_sweep = None             # sweep now gone → will expire
        trader._vol_regime_high = False
        trader._h1_atr = 0.0
        trader._h1_atr_history = []
        trader._last_vol_regime_pct = 0.0
        trader._h1_slope = 0.0
        trader.dollar_bars = [MagicMock()] * 60  # satisfy len guard

        # Build a minimal H1 bar dataframe that won't find a sweep
        idx = pd.date_range("2026-01-06 09:00", periods=3, freq="h", tz="America/New_York")
        h1 = pd.DataFrame(
            {"open": [20000.0]*3, "high": [20010.0]*3, "low": [19990.0]*3, "close": [20005.0]*3, "volume": [1000]*3},
            index=idx,
        )
        h1.index.name = "timestamp"

        with patch("src.research.tier2_streaming_working.resample_to_h1", return_value=h1), \
             patch("src.research.tier2_streaming_working.detect_liquidity_sweep", return_value=None), \
             patch("src.research.tier2_streaming_working.volatility_regime_filter", return_value=True), \
             patch("src.research.tier2_streaming_working._dollar_bars_to_df", return_value=pd.DataFrame()):
            trader._update_h1_structure()

        assert trader._ifvg_candidate is None
