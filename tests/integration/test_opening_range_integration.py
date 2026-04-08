"""Integration tests for Opening Range Breakout strategy."""

import pytest
from datetime import datetime, time, timedelta

from src.data.models import DollarBar
from src.detection.opening_range_strategy import OpeningRangeSignal, OpeningRangeStrategy


class TestOpeningRangeIntegration:
    """Integration tests for end-to-end Opening Range Breakout strategy."""

    @pytest.fixture
    def full_day_bars_with_bullish_breakout(self) -> list[DollarBar]:
        """Create full day bars with bullish breakout."""
        base_time = datetime(2026, 3, 31, 9, 30, 0)
        bars = []

        # Opening range (9:30-10:30) - trending up slightly
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i * 0.3,
                high=11805.0 + i * 0.3,
                low=11799.0 + i * 0.3,
                close=11803.0 + i * 0.3,
                volume=1000 + i * 5,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Post-OR bars (10:30+)
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=60 + i),
                open=11818.0 + i * 0.1,
                high=11823.0 + i * 0.1,
                low=11817.0 + i * 0.1,
                close=11821.0 + i * 0.1,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Add bullish breakout bar (above ORH with high volume)
        orh_approx = 11805 + 59 * 0.3  # Approx ORH
        breakout_bar = DollarBar(
            timestamp=bars[-1].timestamp + timedelta(minutes=1),
            open=orh_approx,
            high=orh_approx + 3.0,  # Breaks above ORH
            low=orh_approx - 1.0,
            close=orh_approx + 2.0,  # Confirms above ORH
            volume=2000,  # 2x baseline
            notional_value=100_000_000,
        )
        bars.append(breakout_bar)

        return bars

    @pytest.fixture
    def full_day_bars_with_bearish_breakout(self) -> list[DollarBar]:
        """Create full day bars with bearish breakout."""
        base_time = datetime(2026, 3, 31, 9, 30, 0)
        bars = []

        # Opening range (9:30-10:30) - ranging
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + (i % 5),  # Oscillates
                high=11805.0 + (i % 5),
                low=11799.0 + (i % 5),
                close=11802.0 + (i % 5),
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Post-OR bars (10:30+)
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=60 + i),
                open=11802.0,
                high=11807.0,
                low=11801.0,
                close=11805.0,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Add bearish breakout bar (below ORL with high volume)
        orl_approx = 11799  # Approx ORL
        breakout_bar = DollarBar(
            timestamp=bars[-1].timestamp + timedelta(minutes=1),
            open=orl_approx,
            high=orl_approx + 1.0,
            low=orl_approx - 3.0,  # Breaks below ORL
            close=orl_approx - 2.0,  # Confirms below ORL
            volume=2000,  # 2x baseline
            notional_value=100_000_000,
        )
        bars.append(breakout_bar)

        return bars

    @pytest.fixture
    def no_breakout_bars(self) -> list[DollarBar]:
        """Create bars with no breakout."""
        base_time = datetime(2026, 3, 31, 9, 30, 0)
        bars = []

        # Opening range
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i * 0.2,
                high=11805.0 + i * 0.2,
                low=11799.0 + i * 0.2,
                close=11803.0 + i * 0.2,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Post-OR bars - stay within range
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=60 + i),
                open=11812.0,
                high=11820.0,  # Below ORH
                low=11810.0,  # Above ORL
                close=11815.0,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        return bars

    def test_end_to_end_bullish_breakout(self, full_day_bars_with_bullish_breakout):
        """Test end-to-end bullish breakout signal generation."""
        strategy = OpeningRangeStrategy(config={})

        signals = []
        for bar in full_day_bars_with_bullish_breakout:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Should generate at least one signal (the breakout)
        assert len(signals) >= 1

        # Last signal should be the bullish breakout
        last_signal = signals[-1]
        assert last_signal.direction == "long"
        assert last_signal.confidence >= 0.60
        assert last_signal.take_profit > last_signal.entry_price
        assert last_signal.stop_loss < last_signal.entry_price

    def test_end_to_end_bearish_breakout(self, full_day_bars_with_bearish_breakout):
        """Test end-to-end bearish breakout signal generation."""
        strategy = OpeningRangeStrategy(config={})

        signals = []
        for bar in full_day_bars_with_bearish_breakout:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Should generate at least one signal (the breakout)
        assert len(signals) >= 1

        # Last signal should be the bearish breakout
        last_signal = signals[-1]
        assert last_signal.direction == "short"
        assert last_signal.confidence >= 0.60
        assert last_signal.take_profit < last_signal.entry_price
        assert last_signal.stop_loss > last_signal.entry_price

    def test_no_breakout_no_signals(self, no_breakout_bars):
        """Test that no signals are generated without breakout."""
        strategy = OpeningRangeStrategy(config={})

        signals = []
        for bar in no_breakout_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Should not generate signals without breakout
        assert len(signals) == 0

    def test_opening_range_completes_correctly(self):
        """Test that opening range is properly tracked and completes."""
        strategy = OpeningRangeStrategy(config={})

        base_time = datetime(2026, 3, 31, 9, 30, 0)

        # Process opening range bars
        or_bars = []
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i * 0.2,
                high=11805.0 + i * 0.2,
                low=11799.0 + i * 0.2,
                close=11803.0 + i * 0.2,
                volume=1000,
                notional_value=50_000_000,
            )
            or_bars.append(bar)
            strategy.process_bar(bar)

        # Opening range should be tracked but not yet finalized
        or_data = strategy._or_detector.get_opening_range()
        # Should be None or not complete until we process a bar past 10:30

        # Process one bar after OR end
        post_or_bar = DollarBar(
            timestamp=base_time + timedelta(minutes=61),
            open=11812.0,
            high=11817.0,
            low=11811.0,
            close=11815.0,
            volume=1000,
            notional_value=50_000_000,
        )
        strategy.process_bar(post_or_bar)

        # Now opening range should be complete
        or_data = strategy._or_detector.get_opening_range()
        assert or_data is not None
        assert or_data.is_complete
        assert or_data.high > or_data.low

    def test_stop_loss_at_opposite_boundary(self, full_day_bars_with_bullish_breakout):
        """Test that stop loss is set at opposite OR boundary."""
        strategy = OpeningRangeStrategy(config={})

        for bar in full_day_bars_with_bullish_breakout:
            signal = strategy.process_bar(bar)
            if signal and signal.direction == "long":
                # For long, stop loss should be at ORL
                or_data = strategy._or_detector.get_opening_range()
                assert signal.stop_loss == pytest.approx(or_data.low, abs=0.5)
                break

    def test_risk_reward_ratio_maintained(self, full_day_bars_with_bullish_breakout):
        """Test that all signals maintain 2:1 reward-risk ratio."""
        strategy = OpeningRangeStrategy(config={})

        for bar in full_day_bars_with_bullish_breakout:
            signal = strategy.process_bar(bar)
            if signal:
                entry = signal.entry_price
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

                if signal.direction == "long":
                    risk = entry - stop_loss
                    reward = take_profit - entry
                else:  # short
                    risk = stop_loss - entry
                    reward = entry - take_profit

                if risk > 0:
                    ratio = reward / risk
                    assert ratio >= 1.9, f"Risk-reward ratio {ratio:.2f} below 2:1"

    def test_signal_frequency_target(self, no_breakout_bars):
        """Test signal frequency is within target range (1-3 trades/day)."""
        strategy = OpeningRangeStrategy(config={})

        signals = []
        for bar in no_breakout_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # With no breakout, should have 0 signals
        # With breakout, should have 1-2 signals (one per breakout direction)
        # Target: 1-3 per day, so 0-2 is within range for these synthetic bars
        assert len(signals) <= 3

    def test_multiple_breakouts_same_day(self):
        """Test handling of multiple breakouts in same day."""
        strategy = OpeningRangeStrategy(config={})

        base_time = datetime(2026, 3, 31, 9, 30, 0)
        bars = []

        # Opening range
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0,
                high=11805.0,
                low=11799.0,
                close=11802.0,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Add bar after OR to finalize
        bars.append(
            DollarBar(
                timestamp=base_time + timedelta(minutes=61),
                open=11805.0,
                high=11810.0,
                low=11804.0,
                close=11808.0,
                volume=1000,
                notional_value=50_000_000,
            )
        )

        # First breakout (bullish)
        bars.append(
            DollarBar(
                timestamp=base_time + timedelta(minutes=62),
                open=11806.0,
                high=11809.0,  # Above ORH
                low=11805.0,
                close=11808.0,  # Confirms
                volume=2000,
                notional_value=100_000_000,
            )
        )

        # Later, second breakout attempt (but should only count once)
        # The strategy should generate signal on first breakout
        signals = []
        for bar in bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Should generate at least 1 signal
        assert len(signals) >= 1

    def test_confidence_scoring(self, full_day_bars_with_bullish_breakout):
        """Test that confidence scores are properly calculated."""
        strategy = OpeningRangeStrategy(config={})

        for bar in full_day_bars_with_bullish_breakout:
            signal = strategy.process_bar(bar)
            if signal:
                # Check confidence is in valid range
                assert 0.60 <= signal.confidence <= 1.0

                # Check contributing factors
                assert "orh" in signal.contributing_factors
                assert "orl" in signal.contributing_factors
                assert "breakout_price" in signal.contributing_factors
                assert "volume_ratio" in signal.contributing_factors
                assert "or_size" in signal.contributing_factors

                # Higher volume should give higher confidence
                volume_ratio = signal.contributing_factors["volume_ratio"]
                if volume_ratio > 2.0:
                    assert signal.confidence >= 0.65

    def test_performance_characteristics(self, full_day_bars_with_bullish_breakout):
        """Test performance characteristics (latency, memory)."""
        import time

        strategy = OpeningRangeStrategy(config={})

        # Measure processing time
        start_time = time.time()
        for bar in full_day_bars_with_bullish_breakout:
            strategy.process_bar(bar)
        end_time = time.time()

        # Should process bars quickly
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Processing too slow: {processing_time:.2f}s"

        # Memory usage should be reasonable
        # Strategy only keeps opening range bars
        assert len(strategy._or_detector._current_or_bars) <= 60  # One hour max
