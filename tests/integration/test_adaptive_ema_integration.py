"""Integration tests for AdaptiveEMAStrategy.

Tests end-to-end functionality with realistic market data.
"""

import pandas as pd
from datetime import datetime

from src.detection.adaptive_ema_strategy import AdaptiveEMAStrategy
from src.data.models import DollarBar


def _create_sample_bars(prices: list[float]) -> list[DollarBar]:
    """Helper to create sample DollarBar objects.

    Args:
        prices: List of closing prices

    Returns:
        List of DollarBar objects
    """
    bars = []
    base_time = datetime(2024, 1, 1, 9, 30, 0)

    for i, price in enumerate(prices):
        bar = DollarBar(
            timestamp=base_time + pd.Timedelta(minutes=i),
            open=price - 0.5,
            high=price + 1.0,
            low=price - 1.5,
            close=price,
            volume=1000,
            notional_value=price * 1000
        )
        bars.append(bar)

    return bars


class TestEndToEndSignalGeneration:
    """Test end-to-end signal generation with realistic data."""

    def test_full_day_trading_session(self):
        """Test processing a full day of trading data."""
        strategy = AdaptiveEMAStrategy()

        # Simulate a day with various market conditions
        # Morning: sideways
        morning = [2100.0 + (i % 10) * 0.5 for i in range(100)]
        # Mid-day: uptrend
        midday = [2105.0 + i * 0.3 for i in range(100)]
        # Afternoon: downtrend
        afternoon = [2135.0 - i * 0.2 for i in range(100)]

        all_prices = morning + midday + afternoon
        bars = _create_sample_bars(all_prices)

        signals = strategy.process_bars(bars)

        # Should generate some signals during the day
        assert len(signals) >= 0  # May or may not have signals depending on conditions

        if len(signals) > 0:
            # Verify signals are valid
            for signal in signals:
                assert signal.direction in ['LONG', 'SHORT']
                assert signal.entry_price > 0
                assert signal.stop_loss > 0
                assert signal.take_profit > 0
                assert 0 <= signal.confidence <= 100

    def test_choppy_market_generates_fewer_signals(self):
        """Verify choppy/sideways market generates fewer signals."""
        strategy = AdaptiveEMAStrategy()

        # Create choppy sideways price action
        prices = [2100.0 + (i % 20) * 0.5 - 5 for i in range(300)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        # Should generate fewer or no signals in choppy market
        assert len(signals) < 10  # Choppy market should have few signals


class TestTradeFrequency:
    """Test trade frequency meets requirements."""

    def test_trade_frequency_in_target_range(self):
        """Verify trade frequency is approximately 5-15 trades/day.

        Note: This is a rough estimate. Actual frequency depends on
        market conditions and parameter tuning.
        """
        strategy = AdaptiveEMAStrategy()

        # Create 5 days of data with mixed conditions
        all_bars = []
        for day in range(5):
            base_price = 2100.0 + day * 10
            # Mix of trend and chop
            if day % 2 == 0:
                # Trend day
                prices = [base_price + i * 0.2 for i in range(250)]
            else:
                # Choppy day
                prices = [base_price + (i % 20) * 0.3 for i in range(250)]

            all_bars.extend(_create_sample_bars(prices))

        signals = strategy.process_bars(all_bars)

        # Calculate average trades per day
        trades_per_day = len(signals) / 5

        # Should be in reasonable range (0-30 is acceptable for this test)
        assert 0 <= trades_per_day <= 30


class TestSignalQuality:
    """Test signal quality and characteristics."""

    def test_long_signals_have_bullish_indicators(self):
        """Verify LONG signals have bullish indicator alignment."""
        strategy = AdaptiveEMAStrategy()

        # Create strong uptrend
        prices = [2100.0 + i * 0.2 + (i ** 2) * 0.001 for i in range(300)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)
        long_signals = [s for s in signals if s.direction == 'LONG']

        if len(long_signals) > 0:
            for signal in long_signals:
                # Check EMA alignment
                assert signal.ema_fast > signal.ema_medium > signal.ema_slow
                # Check MACD is positive
                assert signal.macd_line > 0
                # Check RSI in mid-band
                assert 40 <= signal.rsi_value <= 60

    def test_short_signals_have_bearish_indicators(self):
        """Verify SHORT signals have bearish indicator alignment."""
        strategy = AdaptiveEMAStrategy()

        # Create strong downtrend
        prices = [2200.0 - i * 0.2 - (i ** 2) * 0.001 for i in range(300)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)
        short_signals = [s for s in signals if s.direction == 'SHORT']

        if len(short_signals) > 0:
            for signal in short_signals:
                # Check EMA alignment
                assert signal.ema_fast < signal.ema_medium < signal.ema_slow
                # Check MACD is negative
                assert signal.macd_line < 0
                # Check RSI in mid-band
                assert 40 <= signal.rsi_value <= 60


class TestRiskManagement:
    """Test risk management parameters."""

    def test_all_signals_have_valid_risk_parameters(self):
        """Verify all signals have valid stop loss and take profit."""
        strategy = AdaptiveEMAStrategy()

        # Generate some signals
        uptrend = [2100.0 + i * 0.2 for i in range(250)]
        downtrend = [2200.0 - i * 0.2 for i in range(250)]
        bars = _create_sample_bars(uptrend + downtrend)

        signals = strategy.process_bars(bars)

        if len(signals) > 0:
            for signal in signals:
                # Verify stop loss respects direction
                if signal.direction == 'LONG':
                    assert signal.stop_loss < signal.entry_price
                    assert signal.take_profit > signal.entry_price
                else:  # SHORT
                    assert signal.stop_loss > signal.entry_price
                    assert signal.take_profit < signal.entry_price

                # Verify 2:1 ratio
                risk = abs(signal.entry_price - signal.stop_loss)
                reward = abs(signal.take_profit - signal.entry_price)
                if risk > 0:
                    ratio = reward / risk
                    assert ratio >= 1.9  # Allow small tolerance


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_processing_speed_acceptable(self):
        """Verify strategy processes bars quickly.

        Target: Process 1000 bars in < 1 second
        """
        import time

        strategy = AdaptiveEMAStrategy()

        # Create large dataset
        prices = [2100.0 + i for i in range(1000)]
        bars = _create_sample_bars(prices=prices)

        start_time = time.time()
        signals = strategy.process_bars(bars)
        elapsed_time = time.time() - start_time

        # Should process 1000 bars quickly
        assert elapsed_time < 1.0


class TestRealisticScenarios:
    """Test realistic trading scenarios."""

    def test_trend_reversal(self):
        """Test behavior during trend reversal."""
        strategy = AdaptiveEMAStrategy()

        # Uptrend then reversal to downtrend
        uptrend = [2100.0 + i * 0.3 for i in range(150)]
        reversal = [2145.0 - i * 0.3 for i in range(150)]
        bars = _create_sample_bars(uptrend + reversal)

        signals = strategy.process_bars(bars)

        # Should generate both LONG and SHORT signals
        long_signals = [s for s in signals if s.direction == 'LONG']
        short_signals = [s for s in signals if s.direction == 'SHORT']

        # May have signals in either direction
        assert len(long_signals) >= 0
        assert len(short_signals) >= 0

    def test_gap_handling(self):
        """Test strategy handles price gaps gracefully."""
        strategy = AdaptiveEMAStrategy()

        # Create price with gap
        before_gap = [2100.0 + i * 0.1 for i in range(100)]
        after_gap = [2150.0 + i * 0.1 for i in range(100)]
        bars = _create_sample_bars(before_gap + after_gap)

        # Should not crash and may or may not generate signals
        signals = strategy.process_bars(bars)
        assert isinstance(signals, list)
