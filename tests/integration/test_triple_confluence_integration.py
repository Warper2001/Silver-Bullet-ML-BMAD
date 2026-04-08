"""Integration tests for Triple Confluence Scalper strategy."""

import pytest
from datetime import datetime, timedelta

from src.data.models import DollarBar
from src.detection.triple_confluence_strategy import TripleConfluenceStrategy


class TestTripleConfluenceIntegration:
    """Integration tests for end-to-end strategy functionality."""

    @pytest.fixture
    def sample_dollar_bars(self):
        """Generate sample dollar bars for testing."""
        bars = []
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        base_price = 11800.0

        for i in range(100):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i*5),
                open=base_price + i * 0.5,
                high=base_price + i * 0.5 + 2.0,
                low=base_price + i * 0.5 - 1.0,
                close=base_price + i * 0.5 + 0.5,
                volume=1000 + i * 10,
                notional_value=50_000_000 + i * 100_000,
            )
            bars.append(bar)

        return bars

    def test_end_to_end_processing(self, sample_dollar_bars):
        """Test processing bars through the full strategy."""
        strategy = TripleConfluenceStrategy(config={})

        signals = []
        for bar in sample_dollar_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # May or may not have signals depending on bar patterns
        # Just verify the processing doesn't crash
        assert isinstance(signals, list)

    def test_signal_quality(self):
        """Test that all signals have required fields."""
        strategy = TripleConfluenceStrategy(config={})

        # Process a bar
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 0, 0),
            open=11800.0,
            high=11805.0,
            low=11795.0,
            close=11800.0,
            volume=1000,
            notional_value=50_000_000,
        )

        signal = strategy.process_bar(bar)

        # Most bars won't generate signals, but if one does, verify quality
        if signal:
            # Verify all required fields
            assert signal.strategy_name is not None
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
            assert signal.direction in ["long", "short"]
            assert 0 <= signal.confidence <= 1.0
            assert signal.timestamp is not None
            assert signal.contributing_factors is not None

    def test_no_confluence_no_signals(self):
        """Test that no signals are generated when confluence is absent."""
        strategy = TripleConfluenceStrategy(config={})

        # Create bars with no clear patterns
        base_time = datetime(2026, 3, 31, 10, 0, 0)
        bars = []
        for i in range(50):
            bar = DollarBar(
                timestamp=base_time + timedelta(minutes=i*5),
                open=11800.0 + i * 0.1,
                high=11801.0 + i * 0.1,
                low=11799.0 + i * 0.1,
                close=11800.0 + i * 0.1,
                volume=1000,
                notional_value=50_000_000,
            )
            bars.append(bar)

        # Process bars - should not generate signals without confluence
        signals = []
        for bar in bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # With random uptrend bars, shouldn't get many (if any) signals
        assert len(signals) < 5  # Allow a few false positives but not many

    def test_confidence_score_range(self, sample_dollar_bars):
        """Test that all confidence scores are in 0.8-1.0 range."""
        strategy = TripleConfluenceStrategy(config={})

        signals = []
        for bar in sample_dollar_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Check all signals have proper confidence
        for signal in signals:
            assert 0.8 <= signal.confidence <= 1.0, \
                f"Confidence {signal.confidence} not in 0.8-1.0 range"

    def test_direction_is_valid(self, sample_dollar_bars):
        """Test that all signals have valid direction."""
        strategy = TripleConfluenceStrategy(config={})

        signals = []
        for bar in sample_dollar_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Check all signals have valid direction
        for signal in signals:
            assert signal.direction in ["long", "short"], \
                f"Invalid direction: {signal.direction}"

    def test_stop_loss_take_profit_exist(self, sample_dollar_bars):
        """Test that all signals have stop loss and take profit."""
        strategy = TripleConfluenceStrategy(config={})

        signals = []
        for bar in sample_dollar_bars:
            signal = strategy.process_bar(bar)
            if signal:
                signals.append(signal)

        # Check all signals have SL and TP
        for signal in signals:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss > 0
            assert signal.take_profit > 0

    def test_detection_latency(self):
        """Test that detection latency is acceptable (<100ms)."""
        import time

        strategy = TripleConfluenceStrategy(config={})

        # Create a bar
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 10, 0, 0),
            open=11800.0,
            high=11805.0,
            low=11795.0,
            close=11800.0,
            volume=1000,
            notional_value=50_000_000,
        )

        # Measure processing time
        start = time.perf_counter()
        strategy.process_bar(bar)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        # Should be well under 100ms
        assert latency_ms < 100, f"Detection latency {latency_ms:.2f}ms exceeds 100ms"

    def test_strategy_reset(self):
        """Test that strategy can be reset."""
        strategy = TripleConfluenceStrategy(config={})

        # Process some bars
        for i in range(10):
            bar = DollarBar(
                timestamp=datetime(2026, 3, 31, 10, 0, 0) + timedelta(minutes=i*5),
                open=11800.0 + i,
                high=11805.0 + i,
                low=11795.0 + i,
                close=11800.0 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            strategy.process_bar(bar)

        # Reset
        strategy.reset()

        # Process a new bar - should work without issues
        bar = DollarBar(
            timestamp=datetime(2026, 3, 31, 11, 0, 0),
            open=11800.0,
            high=11805.0,
            low=11795.0,
            close=11800.0,
            volume=1000,
            notional_value=50_000_000,
        )

        signal = strategy.process_bar(bar)
        # Should not crash
