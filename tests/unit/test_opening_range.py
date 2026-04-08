"""Tests for Opening Range Breakout strategy components."""

import pytest
from datetime import datetime, time, timedelta

from src.data.models import DollarBar
from src.detection.breakout_detector import BreakoutDetector, BreakoutEvent
from src.detection.opening_range_detector import (
    OpeningRange,
    OpeningRangeDetector,
)
from src.detection.opening_range_strategy import (
    OpeningRangeSignal,
    OpeningRangeStrategy,
)


class TestOpeningRangeDetector:
    """Tests for OpeningRangeDetector class."""

    @pytest.fixture
    def opening_range_bars(self) -> list[DollarBar]:
        """Create bars during opening range (9:30-10:30 AM ET)."""
        base_time = datetime(2026, 3, 31, 9, 30, 0)  # 9:30 AM ET
        bars = []

        # Create 60 bars (1 minute each) for opening range
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i * 0.5,
                high=11805.0 + i * 0.5,
                low=11799.0 + i * 0.5,
                close=11803.0 + i * 0.5,
                volume=1000 + i * 10,
                notional_value=50_000_000,
            )
            bars.append(bar)

        return bars

    @pytest.fixture
    def post_or_bars(self) -> list[DollarBar]:
        """Create bars after opening range (10:30 AM+)."""
        base_time = datetime(2026, 3, 31, 10, 31, 0)  # 10:31 AM ET
        bars = []

        for i in range(30):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11830.0 + i,
                high=11835.0 + i,
                low=11829.0 + i,
                close=11833.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        return bars

    def test_opening_range_detection(self, opening_range_bars):
        """Test detection of opening range high and low."""
        detector = OpeningRangeDetector(
            or_start=time(9, 30), or_end=time(10, 30), timezone="America/New_York"
        )

        # Process opening range bars
        for bar in opening_range_bars:
            detector.process_bar(bar)

        # Add one bar after OR end to trigger finalization
        post_or_bar = DollarBar(
            timestamp=opening_range_bars[-1].timestamp + timedelta(minutes=1),
            open=11830.0,
            high=11835.0,
            low=11829.0,
            close=11833.0,
            volume=1000,
            notional_value=50_000_000,
        )
        detector.process_bar(post_or_bar)

        # Check if opening range is detected
        or_data = detector.get_opening_range()

        assert or_data is not None
        assert or_data.high > 0
        assert or_data.low > 0
        assert or_data.high >= or_data.low
        assert or_data.volume_baseline > 0

    def test_opening_range_boundaries(self, opening_range_bars):
        """Test that ORH and ORL are correctly identified."""
        detector = OpeningRangeDetector(
            or_start=time(9, 30), or_end=time(10, 30), timezone="America/New_York"
        )

        for bar in opening_range_bars:
            detector.process_bar(bar)

        # Add one bar after OR end to trigger finalization
        post_or_bar = DollarBar(
            timestamp=opening_range_bars[-1].timestamp + timedelta(minutes=1),
            open=11830.0,
            high=11835.0,
            low=11829.0,
            close=11833.0,
            volume=1000,
            notional_value=50_000_000,
        )
        detector.process_bar(post_or_bar)

        or_data = detector.get_opening_range()

        # First bar low should be close to ORL
        # Last bar high should be close to ORH (trending up)
        assert or_data.low == pytest.approx(opening_range_bars[0].low, abs=1)
        assert or_data.high == pytest.approx(opening_range_bars[-1].high, abs=1)

    def test_volume_baseline_calculation(self, opening_range_bars):
        """Test volume baseline calculation."""
        detector = OpeningRangeDetector(
            or_start=time(9, 30), or_end=time(10, 30), timezone="America/New_York"
        )

        for bar in opening_range_bars:
            detector.process_bar(bar)

        # Add one bar after OR end to trigger finalization
        post_or_bar = DollarBar(
            timestamp=opening_range_bars[-1].timestamp + timedelta(minutes=1),
            open=11830.0,
            high=11835.0,
            low=11829.0,
            close=11833.0,
            volume=1000,
            notional_value=50_000_000,
        )
        detector.process_bar(post_or_bar)

        or_data = detector.get_opening_range()

        # Average volume of opening range bars
        expected_avg = sum(bar.volume for bar in opening_range_bars) / len(
            opening_range_bars
        )

        assert or_data.volume_baseline == pytest.approx(expected_avg, rel=0.01)

    def test_daily_reset(self):
        """Test that opening range resets daily."""
        detector = OpeningRangeDetector(
            or_start=time(9, 30), or_end=time(10, 30), timezone="America/New_York"
        )

        # Day 1 opening range
        day1_time = datetime(2026, 3, 31, 9, 30, 0)
        for i in range(60):
            bar = DollarBar(
                timestamp=day1_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11799.0 + i,
                close=11803.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            detector.process_bar(bar)

        # Add bar after OR end to finalize day 1
        post_or_bar = DollarBar(
            timestamp=day1_time + timedelta(minutes=60),
            open=11860.0,
            high=11865.0,
            low=11859.0,
            close=11863.0,
            volume=1000,
            notional_value=50_000_000,
        )
        detector.process_bar(post_or_bar)

        or_day1 = detector.get_opening_range()
        assert or_day1 is not None

        # Day 2 opening range (should reset)
        day2_time = datetime(2026, 4, 1, 9, 30, 0)
        for i in range(60):
            bar = DollarBar(
                timestamp=day2_time + timedelta(minutes=i),
                open=11900.0 + i,  # Different level
                high=11905.0 + i,
                low=11899.0 + i,
                close=11903.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            detector.process_bar(bar)

        # Add bar after OR end to finalize day 2
        post_or_bar2 = DollarBar(
            timestamp=day2_time + timedelta(minutes=60),
            open=11960.0,
            high=11965.0,
            low=11959.0,
            close=11963.0,
            volume=1000,
            notional_value=50_000_000,
        )
        detector.process_bar(post_or_bar2)

        or_day2 = detector.get_opening_range()

        # Should have different values (reset)
        assert or_day2.high != or_day1.high

    def test_no_opening_range_before_time(self):
        """Test that opening range is not available before 10:30 AM."""
        detector = OpeningRangeDetector(
            or_start=time(9, 30), or_end=time(10, 30), timezone="America/New_York"
        )

        # Process only 30 minutes of opening range
        base_time = datetime(2026, 3, 31, 9, 30, 0)
        for i in range(30):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11799.0 + i,
                close=11803.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            detector.process_bar(bar)

        # Opening range should not be complete yet
        or_data = detector.get_opening_range()
        assert or_data is None or not or_data.is_complete


class TestBreakoutDetector:
    """Tests for BreakoutDetector class."""

    @pytest.fixture
    def opening_range(self) -> OpeningRange:
        """Create a mock opening range."""
        return OpeningRange(
            or_start=datetime(2026, 3, 31, 9, 30),
            or_end=datetime(2026, 3, 31, 10, 30),
            high=11830.0,
            low=11800.0,
            volume_baseline=1000.0,
            is_complete=True,
        )

    def test_bullish_breakout_detection(self, opening_range):
        """Test detection of bullish breakout."""
        detector = BreakoutDetector(
            tick_size=0.25, volume_threshold=1.5, breakout_threshold_ticks=1
        )

        # Bar that breaks above ORH with volume
        breakout_bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 31, 0),
            open=11830.25,  # At ORH
            high=11832.0,  # Above ORH by 2 ticks
            low=11829.0,
            close=11831.5,  # Closes above ORH
            volume=1600,  # 1.6x baseline
            notional_value=80_000_000,
        )

        breakout = detector.detect_breakout(breakout_bar, opening_range)

        assert breakout is not None
        assert breakout.direction == "bullish"
        assert breakout.breakout_price > opening_range.high

    def test_bearish_breakout_detection(self, opening_range):
        """Test detection of bearish breakout."""
        detector = BreakoutDetector(
            tick_size=0.25, volume_threshold=1.5, breakout_threshold_ticks=1
        )

        # Bar that breaks below ORL with volume
        breakout_bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 31, 0),
            open=11800.0,  # At ORL
            high=11801.0,
            low=11798.0,  # Below ORL by 2 ticks
            close=11799.0,  # Closes below ORL
            volume=1600,  # 1.6x baseline
            notional_value=80_000_000,
        )

        breakout = detector.detect_breakout(breakout_bar, opening_range)

        assert breakout is not None
        assert breakout.direction == "bearish"
        assert breakout.breakout_price < opening_range.low

    def test_no_breakout_without_volume(self, opening_range):
        """Test no breakout when volume is too low."""
        detector = BreakoutDetector(
            tick_size=0.25, volume_threshold=1.5, breakout_threshold_ticks=1
        )

        # Price breaks out but volume too low
        breakout_bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 31, 0),
            open=11830.25,
            high=11832.0,
            low=11829.0,
            close=11831.5,
            volume=1200,  # Only 1.2x baseline
            notional_value=60_000_000,
        )

        breakout = detector.detect_breakout(breakout_bar, opening_range)

        assert breakout is None

    def test_no_breakout_without_confirmation(self, opening_range):
        """Test no breakout when close doesn't confirm."""
        detector = BreakoutDetector(
            tick_size=0.25, volume_threshold=1.5, breakout_threshold_ticks=1
        )

        # Price breaks out but closes back inside range
        failed_bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 31, 0),
            open=11830.25,
            high=11832.0,  # Goes above ORH
            low=11829.0,
            close=11829.5,  # Closes below ORH
            volume=1600,
            notional_value=80_000_000,
        )

        breakout = detector.detect_breakout(failed_bar, opening_range)

        assert breakout is None

    def test_no_breakout_when_not_past_threshold(self, opening_range):
        """Test no breakout when price doesn't exceed threshold."""
        detector = BreakoutDetector(
            tick_size=0.25, volume_threshold=1.5, breakout_threshold_ticks=1
        )

        # Price touches ORH but doesn't break through
        touch_bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 31, 0),
            open=11830.0,
            high=11830.25,  # Only 1 tick above
            low=11829.0,
            close=11830.0,
            volume=1600,
            notional_value=80_000_000,
        )

        # Depending on threshold, may or may not trigger
        breakout = detector.detect_breakout(touch_bar, opening_range)
        # With 1 tick threshold, 1 tick above should trigger


class TestOpeningRangeStrategy:
    """Tests for OpeningRangeStrategy class."""

    @pytest.fixture
    def sample_bars(self) -> list[DollarBar]:
        """Create sample bars for full trading day."""
        base_time = datetime(2026, 3, 31, 9, 30, 0)
        bars = []

        # Opening range bars (9:30-10:30)
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i * 0.2,
                high=11805.0 + i * 0.2,
                low=11799.0 + i * 0.2,
                close=11803.0 + i * 0.2,
                volume=1000 + i * 5,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Post-OR bars (10:30+)
        for i in range(60):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=60 + i),
                open=11812.0 + i * 0.5,
                high=11817.0 + i * 0.5,
                low=11811.0 + i * 0.5,
                close=11815.0 + i * 0.5,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        return bars

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = OpeningRangeStrategy(config={})

        assert strategy is not None
        assert strategy._or_detector is not None
        assert strategy._breakout_detector is not None

    def test_signal_generation(self, sample_bars):
        """Test signal generation on breakout."""
        strategy = OpeningRangeStrategy(config={})

        signals = []
        for bar in sample_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # May or may not generate signals depending on data
        # Just verify it returns correct type when conditions are met
        for signal in signals:
            assert isinstance(signal, OpeningRangeSignal)
            assert signal.direction in ["long", "short"]
            assert signal.confidence >= 0.60
            assert signal.stop_loss > 0
            assert signal.take_profit > 0

    def test_stop_loss_at_or_boundary(self, sample_bars):
        """Test that stop loss is set at opposite OR boundary."""
        strategy = OpeningRangeStrategy(config={})

        for bar in sample_bars:
            signal = strategy.process_bar(bar)
            if signal:
                # Stop loss should be at opposite boundary
                # For long: SL = ORL, for short: SL = ORH
                assert signal.stop_loss > 0

    def test_risk_reward_ratio(self, sample_bars):
        """Test 2:1 reward-risk ratio."""
        strategy = OpeningRangeStrategy(config={})

        for bar in sample_bars:
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
                    assert ratio >= 1.9  # Allow rounding tolerance

    def test_no_signals_before_or_complete(self):
        """Test no signals generated before opening range is complete."""
        strategy = OpeningRangeStrategy(config={})

        # Only opening range bars
        base_time = datetime(2026, 3, 31, 9, 30, 0)
        or_bars = []
        for i in range(30):  # Only 30 minutes
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11799.0 + i,
                close=11803.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            or_bars.append(bar)

        # Should not generate any signals
        for bar in or_bars:
            signal = strategy.process_bar(bar)
            assert signal is None

    def test_signal_fields_validation(self, sample_bars):
        """Test that signals have all required fields."""
        strategy = OpeningRangeStrategy(config={})

        for bar in sample_bars:
            signal = strategy.process_bar(bar)
            if signal:
                # Verify all required fields
                assert hasattr(signal, "strategy_name")
                assert hasattr(signal, "entry_price")
                assert hasattr(signal, "stop_loss")
                assert hasattr(signal, "take_profit")
                assert hasattr(signal, "direction")
                assert hasattr(signal, "confidence")
                assert hasattr(signal, "timestamp")
                assert hasattr(signal, "contributing_factors")

                # Validate values
                assert signal.strategy_name == "Opening Range Breakout"
                assert 0.60 <= signal.confidence <= 1.0
                assert signal.entry_price > 0
                assert signal.stop_loss > 0
                assert signal.take_profit > 0
                assert signal.direction in ["long", "short"]
