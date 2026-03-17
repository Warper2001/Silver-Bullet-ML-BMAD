"""Unit tests for Signal Filter.

Tests signal filtering by probability threshold, statistics tracking,
win rate monitoring, and model drift detection.
"""

from datetime import datetime

import pytest

from src.data.models import (
    FVGEvent,
    GapRange,
    MSSEvent,
    SilverBulletSetup,
    SwingPoint,
)
from src.ml.signal_filter import SignalFilter


class TestSignalFilterInit:
    """Test SignalFilter initialization and configuration."""

    def test_init_with_default_threshold(self):
        """Verify SignalFilter initializes with default 0.65 threshold."""
        signal_filter = SignalFilter()
        assert signal_filter is not None
        assert signal_filter._threshold == 0.65

    def test_init_with_custom_threshold(self):
        """Verify SignalFilter initializes with custom threshold."""
        signal_filter = SignalFilter(threshold=0.70)
        assert signal_filter._threshold == 0.70

    def test_init_with_invalid_threshold_raises_error(self):
        """Verify ValueError raised for threshold outside [0, 1] range."""
        with pytest.raises(ValueError, match="Threshold must be in \\[0, 1\\]"):
            SignalFilter(threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be in \\[0, 1\\]"):
            SignalFilter(threshold=-0.1)

    def test_statistics_initialized_on_creation(self):
        """Verify statistics dictionary is initialized."""
        signal_filter = SignalFilter()
        assert hasattr(signal_filter, "_stats")
        assert "total_signals" in signal_filter._stats
        assert "filtered_signals" in signal_filter._stats
        assert "allowed_signals" in signal_filter._stats
        assert "filter_rate" in signal_filter._stats

    def test_allowed_signals_list_initialized(self):
        """Verify allowed signals list is initialized."""
        signal_filter = SignalFilter()
        assert hasattr(signal_filter, "_allowed_signals")
        assert isinstance(signal_filter._allowed_signals, list)


class TestFilterSignal:
    """Test filter_signal() method."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample Silver Bullet setup."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11820.0, bottom=11790.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    def test_filter_signal_with_probability_below_threshold(self, sample_signal):
        """Verify signal is filtered when probability <= threshold."""
        signal_filter = SignalFilter(threshold=0.65)
        result = signal_filter.filter_signal(sample_signal, 0.60)

        assert result["allowed"] is False
        assert result["reason"] == "filtered_by_ml_threshold"
        assert result["probability"] == 0.60
        assert result["threshold"] == 0.65
        assert "latency_ms" in result

    def test_filter_signal_with_probability_above_threshold(self, sample_signal):
        """Verify signal is allowed when probability > threshold."""
        signal_filter = SignalFilter(threshold=0.65)
        result = signal_filter.filter_signal(sample_signal, 0.70)

        assert result["allowed"] is True
        assert result["reason"] == "allowed_by_ml_threshold"
        assert result["probability"] == 0.70
        assert result["threshold"] == 0.65
        assert "latency_ms" in result

    def test_filter_signal_with_probability_at_threshold(self, sample_signal):
        """Verify signal is filtered when probability == threshold (<=)."""
        signal_filter = SignalFilter(threshold=0.65)
        result = signal_filter.filter_signal(sample_signal, 0.65)

        assert result["allowed"] is False
        assert result["reason"] == "filtered_by_ml_threshold"
        assert result["probability"] == 0.65

    def test_filter_signal_with_invalid_probability_raises_error(self, sample_signal):
        """Verify ValueError raised for probability outside [0, 1] range."""
        signal_filter = SignalFilter()

        with pytest.raises(ValueError, match="Probability must be in \\[0, 1\\]"):
            signal_filter.filter_signal(sample_signal, 1.5)

        with pytest.raises(ValueError, match="Probability must be in \\[0, 1\\]"):
            signal_filter.filter_signal(sample_signal, -0.1)

    def test_filter_signal_updates_statistics(self, sample_signal):
        """Verify statistics are updated after filtering."""
        signal_filter = SignalFilter(threshold=0.65)

        # Filter two signals
        signal_filter.filter_signal(sample_signal, 0.60)
        signal_filter.filter_signal(sample_signal, 0.70)

        stats = signal_filter.get_statistics()
        assert stats["total_signals"] == 2
        assert stats["filtered_signals"] == 1
        assert stats["allowed_signals"] == 1
        assert stats["filter_rate"] == 0.5


class TestShouldAllow:
    """Test should_allow() helper method."""

    def test_should_allow_returns_false_below_threshold(self):
        """Verify should_allow() returns False for probability <= threshold."""
        signal_filter = SignalFilter(threshold=0.65)
        assert signal_filter.should_allow(0.60) is False
        assert signal_filter.should_allow(0.65) is False

    def test_should_allow_returns_true_above_threshold(self):
        """Verify should_allow() returns True for probability > threshold."""
        signal_filter = SignalFilter(threshold=0.65)
        assert signal_filter.should_allow(0.70) is True
        assert signal_filter.should_allow(0.66) is True


class TestStatisticsTracking:
    """Test statistics tracking functionality."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample Silver Bullet setup."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11820.0, bottom=11790.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    def test_get_statistics_returns_current_stats(self, sample_signal):
        """Verify get_statistics() returns current statistics."""
        signal_filter = SignalFilter(threshold=0.65)

        # Process some signals
        signal_filter.filter_signal(sample_signal, 0.50)
        signal_filter.filter_signal(sample_signal, 0.70)
        signal_filter.filter_signal(sample_signal, 0.80)

        stats = signal_filter.get_statistics()
        assert stats["total_signals"] == 3
        assert stats["filtered_signals"] == 1
        assert stats["allowed_signals"] == 2
        assert stats["filter_rate"] == pytest.approx(0.333, rel=0.01)
        assert "session_start" in stats
        assert "last_update" in stats

    def test_allowed_signals_tracked_for_win_rate(self, sample_signal):
        """Verify allowed signals are tracked with timestamps."""
        signal_filter = SignalFilter(threshold=0.50)

        # Allow two signals
        signal_filter.filter_signal(sample_signal, 0.70)
        signal_filter.filter_signal(sample_signal, 0.80)

        assert len(signal_filter._allowed_signals) == 2
        assert signal_filter._allowed_signals[0]["probability"] == 0.70
        assert signal_filter._allowed_signals[1]["probability"] == 0.80
        assert "timestamp" in signal_filter._allowed_signals[0]

    def test_allowed_signals_limited_to_1000(self, sample_signal):
        """Verify only last 1000 allowed signals are kept."""
        signal_filter = SignalFilter(threshold=0.0)  # Allow all signals

        # Simulate 1005 signals
        for i in range(1005):
            signal_filter.filter_signal(sample_signal, 0.70)

        # Should keep only last 1000
        assert len(signal_filter._allowed_signals) == 1000


class TestWinRateTracking:
    """Test win rate tracking functionality."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample Silver Bullet setup."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11820.0, bottom=11790.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    def test_record_trade_outcome_stores_result(self, sample_signal):
        """Verify trade outcomes are recorded."""
        signal_filter = SignalFilter(threshold=0.50)

        # Allow a signal
        signal_filter.filter_signal(sample_signal, 0.70)

        # Record outcome
        signal_filter.record_trade_outcome(sample_signal, success=True)

        # Check outcome was stored
        assert "outcome" in signal_filter._allowed_signals[0]
        assert signal_filter._allowed_signals[0]["outcome"] is True

    def test_calculate_win_rate_with_no_trades(self):
        """Verify win rate is 0.0 when no trades completed."""
        signal_filter = SignalFilter()
        win_rate = signal_filter.calculate_win_rate()
        assert win_rate == 0.0

    def test_calculate_win_rate_with_mixed_outcomes(self, sample_signal):
        """Verify win rate calculation with mixed outcomes."""
        signal_filter = SignalFilter(threshold=0.50)

        # Create 10 distinct signals and allow them
        signals = []
        for i in range(10):
            base_time = datetime(2026, 3, 16, 10, 0, i)
            sample_signal.timestamp = base_time
            signal_filter.filter_signal(sample_signal, 0.70)
            signals.append(base_time)

        # Record outcomes: 7 wins, 3 losses
        for i in range(7):
            sample_signal.timestamp = signals[i]
            signal_filter.record_trade_outcome(sample_signal, success=True)
        for i in range(7, 10):
            sample_signal.timestamp = signals[i]
            signal_filter.record_trade_outcome(sample_signal, success=False)

        # Calculate win rate
        win_rate = signal_filter.calculate_win_rate()
        assert win_rate == pytest.approx(0.7, rel=0.01)

    def test_calculate_win_rate_uses_last_50_trades(self, sample_signal):
        """Verify win rate uses only last 50 completed trades."""
        signal_filter = SignalFilter(threshold=0.50)

        # Create 100 distinct signals with alternating outcomes
        signals = []
        for i in range(100):
            minutes = i // 60
            seconds = i % 60
            base_time = datetime(2026, 3, 16, 10, minutes, seconds)
            sample_signal.timestamp = base_time
            signal_filter.filter_signal(sample_signal, 0.70)
            signals.append(base_time)

        # Record outcomes for all signals (alternating)
        for i, sig_time in enumerate(signals):
            sample_signal.timestamp = sig_time
            signal_filter.record_trade_outcome(sample_signal, success=(i % 2 == 0))

        # Win rate should be based on last 50 (25 wins, 25 losses = 50%)
        win_rate = signal_filter.calculate_win_rate()
        assert win_rate == pytest.approx(0.5, rel=0.01)


class TestModelDriftDetection:
    """Test model drift detection functionality."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample Silver Bullet setup."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11820.0, bottom=11790.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    def test_check_drift_with_no_drift(self, sample_signal):
        """Verify no drift detected when performance is good."""
        signal_filter = SignalFilter(threshold=0.50)

        # Simulate 50 trades with 65% win rate (expected)
        for i in range(50):
            base_time = datetime(2026, 3, 16, 10, 0, i)
            sample_signal.timestamp = base_time
            signal_filter.filter_signal(sample_signal, 0.70)
            success = i < 32  # 32 wins out of 50 = 64%
            signal_filter.record_trade_outcome(sample_signal, success=success)

        drift_detected = signal_filter.check_drift_detection()
        assert drift_detected is False

    def test_check_drift_with_drift_detected(self, sample_signal):
        """Verify drift detected when performance drops >10% below expected."""
        signal_filter = SignalFilter(threshold=0.50)

        # Simulate 50 trades with 50% win rate (15% below expected 65%)
        for i in range(50):
            base_time = datetime(2026, 3, 16, 10, 0, i)
            sample_signal.timestamp = base_time
            signal_filter.filter_signal(sample_signal, 0.70)
            success = i < 25  # 25 wins out of 50 = 50%
            signal_filter.record_trade_outcome(sample_signal, success=success)

        drift_detected = signal_filter.check_drift_detection()
        assert drift_detected is True

    def test_check_drift_with_insufficient_data(self, sample_signal):
        """Verify no drift detection with < 50 completed trades."""
        signal_filter = SignalFilter(threshold=0.50)

        # Only 30 trades
        for i in range(30):
            base_time = datetime(2026, 3, 16, 10, 0, i)
            sample_signal.timestamp = base_time
            signal_filter.filter_signal(sample_signal, 0.70)
            signal_filter.record_trade_outcome(sample_signal, success=True)

        drift_detected = signal_filter.check_drift_detection()
        assert drift_detected is False


class TestPerformanceRequirements:
    """Test performance requirements for filtering."""

    @pytest.fixture
    def sample_signal(self):
        """Create sample Silver Bullet setup."""
        base_time = datetime(2026, 3, 16, 10, 0, 0)

        swing = SwingPoint(
            timestamp=base_time,
            price=11800.0,
            swing_type="swing_low",
            bar_index=100,
            confirmed=True,
        )

        mss = MSSEvent(
            timestamp=base_time,
            direction="bullish",
            breakout_price=11810.0,
            swing_point=swing,
            volume_ratio=1.5,
            bar_index=100,
        )

        gap_range = GapRange(top=11820.0, bottom=11790.0)

        fvg = FVGEvent(
            timestamp=base_time,
            direction="bullish",
            gap_range=gap_range,
            gap_size_ticks=30.0,
            gap_size_dollars=150.0,
            bar_index=100,
        )

        return SilverBulletSetup(
            timestamp=base_time,
            direction="bullish",
            mss_event=mss,
            fvg_event=fvg,
            entry_zone_top=11820.0,
            entry_zone_bottom=11790.0,
            invalidation_point=11800.0,
            confluence_count=2,
            priority="medium",
            bar_index=100,
            confidence=3,
        )

    def test_filtering_latency_under_5ms(self, sample_signal):
        """Verify single filter completes in under 5ms."""
        import time

        signal_filter = SignalFilter(threshold=0.65)

        start_time = time.perf_counter()
        signal_filter.filter_signal(sample_signal, 0.70)
        latency_ms = (time.perf_counter() - start_time) * 1000

        assert latency_ms < 5, f"Filtering took {latency_ms:.2f}ms, exceeds 5ms limit"

    def test_batch_filtering_performance(self, sample_signal):
        """Verify filtering 100 signals stays under 5ms average."""
        import time

        signal_filter = SignalFilter(threshold=0.65)

        start_time = time.perf_counter()
        for i in range(100):
            signal_filter.filter_signal(sample_signal, 0.70)
        total_time_ms = (time.perf_counter() - start_time) * 1000
        avg_latency_ms = total_time_ms / 100

        assert avg_latency_ms < 5, f"Average filtering took {avg_latency_ms:.2f}ms"
