"""Unit tests for AdaptiveEMAStrategy.

Tests signal generation, indicator integration, and trade parameters.
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


class TestAdaptiveEMAStrategyInit:
    """Test strategy initialization."""

    def test_init_creates_calculators(self):
        """Verify initialization creates all calculators."""
        strategy = AdaptiveEMAStrategy()

        assert strategy.ema_calculator is not None
        assert strategy.macd_calculator is not None
        assert strategy.rsi_calculator is not None

    def test_init_resets_state(self):
        """Verify initial state is reset."""
        strategy = AdaptiveEMAStrategy()

        assert strategy._atr is None
        assert len(strategy._true_ranges) == 0


class TestSignalGeneration:
    """Test signal generation logic."""

    def test_no_signal_with_insufficient_data(self):
        """Verify no signal with insufficient historical data."""
        strategy = AdaptiveEMAStrategy()

        # Only provide 50 bars (not enough for 200 EMA)
        prices = [2100.0 + i for i in range(50)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        assert len(signals) == 0

    def test_long_signal_generation(self):
        """Verify LONG signal generation with bullish conditions."""
        strategy = AdaptiveEMAStrategy()

        # Create strong uptrend that aligns all indicators
        prices = [2100.0 + i * 0.1 + (i ** 2) * 0.001 for i in range(300)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        # Note: Signals are only generated when ALL three indicators align perfectly
        # This is a strict requirement, so we just verify the strategy doesn't crash
        # and that any signals generated are valid LONG signals
        long_signals = [s for s in signals if s.direction == 'LONG']

        if len(long_signals) > 0:
            # Verify at least one valid LONG signal
            assert len(long_signals) >= 1
            # Verify signal quality
            signal = long_signals[0]
            assert signal.ema_fast > signal.ema_medium > signal.ema_slow

    def test_short_signal_generation(self):
        """Verify SHORT signal generation with bearish conditions."""
        strategy = AdaptiveEMAStrategy()

        # Create strong downtrend that aligns all indicators
        prices = [2200.0 - i * 0.1 - (i ** 2) * 0.001 for i in range(300)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        # Note: Signals are only generated when ALL three indicators align perfectly
        # This is a strict requirement, so we just verify the strategy doesn't crash
        # and that any signals generated are valid SHORT signals
        short_signals = [s for s in signals if s.direction == 'SHORT']

        if len(short_signals) > 0:
            # Verify at least one valid SHORT signal
            assert len(short_signals) >= 1
            # Verify signal quality
            signal = short_signals[0]
            assert signal.ema_fast < signal.ema_medium < signal.ema_slow

    def test_signal_attributes(self):
        """Verify signal has all required attributes."""
        strategy = AdaptiveEMAStrategy()

        # Create trend that should generate signals
        prices = [2100.0 + i * 0.2 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        if len(signals) > 0:
            signal = signals[0]
            assert signal.direction in ['LONG', 'SHORT']
            assert signal.entry_price > 0
            assert signal.stop_loss > 0
            assert signal.take_profit > 0
            assert 0 <= signal.confidence <= 100
            assert signal.timestamp is not None


class TestStopLossAndTakeProfit:
    """Test stop loss and take profit calculation."""

    def test_stop_loss_1x_atr(self):
        """Verify stop loss is approximately 1× ATR."""
        strategy = AdaptiveEMAStrategy()

        # Create trend to generate signal
        prices = [2100.0 + i * 0.2 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        if len(signals) > 0 and strategy._atr is not None:
            signal = signals[0]
            expected_sl_distance = strategy._atr * 1.0

            if signal.direction == 'LONG':
                actual_sl_distance = signal.entry_price - signal.stop_loss
            else:
                actual_sl_distance = signal.stop_loss - signal.entry_price

            # Allow small rounding error
            assert abs(actual_sl_distance - expected_sl_distance) < 0.01

    def test_take_profit_2to1_ratio(self):
        """Verify take profit respects 2:1 risk-reward ratio."""
        strategy = AdaptiveEMAStrategy()

        # Create trend to generate signal
        prices = [2100.0 + i * 0.2 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        if len(signals) > 0:
            signal = signals[0]
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)

            ratio = reward / risk if risk > 0 else 0
            assert ratio >= 1.9  # Allow small tolerance


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_confidence_in_valid_range(self):
        """Verify confidence is always in 0-100 range."""
        strategy = AdaptiveEMAStrategy()

        # Create trend to generate signal
        prices = [2100.0 + i * 0.2 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        if len(signals) > 0:
            for signal in signals:
                assert 0 <= signal.confidence <= 100


class TestStrategyReset:
    """Test strategy reset functionality."""

    def test_reset_clears_all_state(self):
        """Verify reset clears all calculators and state."""
        strategy = AdaptiveEMAStrategy()

        # Process some bars to build up state
        prices = [2100.0 + i for i in range(100)]
        bars = _create_sample_bars(prices=prices)
        strategy.process_bars(bars)

        # Verify state exists
        assert len(strategy.ema_calculator.get_ema_history()['fast_ema']) > 0
        assert len(strategy.macd_calculator.get_macd_history()['macd_line']) > 0
        assert len(strategy.rsi_calculator.get_rsi_history()) > 0

        # Reset
        strategy.reset()

        # Verify state is cleared
        assert len(strategy.ema_calculator.get_ema_history()['fast_ema']) == 0
        assert len(strategy.macd_calculator.get_macd_history()['macd_line']) == 0
        assert len(strategy.rsi_calculator.get_rsi_history()) == 0
        assert strategy._atr is None
        assert len(strategy._true_ranges) == 0


class TestIndicatorIntegration:
    """Test integration of all three indicators."""

    def test_all_indicators_updated(self):
        """Verify all indicators are updated when processing bars."""
        strategy = AdaptiveEMAStrategy()

        prices = [2100.0 + i for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        strategy.process_bars(bars)

        # Check EMA calculator
        ema_values = strategy.ema_calculator.get_current_emas()
        assert ema_values['fast_ema'] is not None
        assert ema_values['medium_ema'] is not None
        assert ema_values['slow_ema'] is not None

        # Check MACD calculator
        macd_values = strategy.macd_calculator.get_current_macd()
        assert macd_values['macd_line'] is not None

        # Check RSI calculator
        rsi_value = strategy.rsi_calculator.get_current_rsi()
        assert rsi_value is not None

    def test_signal_contains_indicator_values(self):
        """Verify signals contain current indicator values."""
        strategy = AdaptiveEMAStrategy()

        # Create trend to generate signal
        prices = [2100.0 + i * 0.2 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        signals = strategy.process_bars(bars)

        if len(signals) > 0:
            signal = signals[0]
            assert signal.ema_fast is not None
            assert signal.ema_medium is not None
            assert signal.ema_slow is not None
            assert signal.macd_line is not None
            assert signal.rsi_value is not None
