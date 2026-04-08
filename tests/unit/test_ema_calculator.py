"""Unit tests for EMACalculator.

Tests exponential moving average calculations for 9, 55, and 200 periods.
"""

import pandas as pd
from datetime import datetime

from src.detection.ema_calculator import EMACalculator
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
            high=price + 0.5,
            low=price - 1.0,
            close=price,
            volume=1000,
            notional_value=price * 1000
        )
        bars.append(bar)

    return bars


class TestEMACalculatorInit:
    """Test EMACalculator initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        calculator = EMACalculator()

        assert calculator._fast_period == 9
        assert calculator._medium_period == 55
        assert calculator._slow_period == 200

    def test_init_with_custom_periods(self):
        """Verify initialization with custom periods."""
        calculator = EMACalculator(fast_period=5, medium_period=20, slow_period=50)

        assert calculator._fast_period == 5
        assert calculator._medium_period == 20
        assert calculator._slow_period == 50


class TestEMACalculation:
    """Test EMA calculation."""

    def test_calculate_fast_ema(self):
        """Verify 9-period EMA calculation."""
        calculator = EMACalculator()

        # Create sample bars with closing prices
        bars = _create_sample_bars(
            prices=[2100.0, 2101.0, 2102.0, 2103.0, 2104.0,
                    2105.0, 2106.0, 2107.0, 2108.0, 2109.0]
        )

        ema_values = calculator.calculate_emas(bars)

        assert 'fast_ema' in ema_values
        assert ema_values['fast_ema'] is not None
        assert ema_values['fast_ema'] > 2100.0  # EMA should be above initial prices

    def test_calculate_medium_ema(self):
        """Verify 55-period EMA calculation."""
        calculator = EMACalculator()

        # Create 55 sample bars
        prices = [2100.0 + i for i in range(60)]
        bars = _create_sample_bars(prices=prices)

        ema_values = calculator.calculate_emas(bars)

        assert 'medium_ema' in ema_values
        assert ema_values['medium_ema'] is not None
        assert ema_values['medium_ema'] > 2100.0

    def test_calculate_slow_ema(self):
        """Verify 200-period EMA calculation."""
        calculator = EMACalculator()

        # Create 200 sample bars
        prices = [2100.0 + i for i in range(205)]
        bars = _create_sample_bars(prices=prices)

        ema_values = calculator.calculate_emas(bars)

        assert 'slow_ema' in ema_values
        assert ema_values['slow_ema'] is not None
        assert ema_values['slow_ema'] > 2100.0

    def test_insufficient_data_for_slow_ema(self):
        """Verify handling of insufficient data for slow EMA."""
        calculator = EMACalculator()

        # Only provide 10 bars (insufficient for 200-period EMA)
        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(10)]
        )

        ema_values = calculator.calculate_emas(bars)

        # Fast EMA should be available
        assert ema_values['fast_ema'] is not None
        # Slow EMA should be None
        assert ema_values['slow_ema'] is None

    def test_ema_history_storage(self):
        """Verify EMA history is stored for trend analysis."""
        calculator = EMACalculator()

        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(100)]
        )

        # Calculate EMAs multiple times
        for bar in bars:
            calculator.calculate_emas([bar])

        history = calculator.get_ema_history()

        assert 'fast_ema' in history
        assert 'medium_ema' in history
        assert 'slow_ema' in history
        assert len(history['fast_ema']) > 0


class TestTrendDirection:
    """Test trend direction analysis."""

    def test_bullish_trend_fast_above_medium_above_slow(self):
        """Verify bullish trend when fast > medium > slow."""
        calculator = EMACalculator()

        # Create rising price trend
        prices = [2100.0 + i * 0.5 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_emas([bar])

        trend = calculator.get_trend_direction()

        assert trend == 'bullish'

    def test_bearish_trend_fast_below_medium_below_slow(self):
        """Verify bearish trend when fast < medium < slow."""
        calculator = EMACalculator()

        # Create falling price trend
        prices = [2200.0 - i * 0.5 for i in range(250)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_emas([bar])

        trend = calculator.get_trend_direction()

        assert trend == 'bearish'

    def test_neutral_trend_when_emas_close(self):
        """Verify neutral trend when EMAs are close together."""
        calculator = EMACalculator()

        # Create flat price trend
        prices = [2100.0] * 250
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_emas([bar])

        trend = calculator.get_trend_direction()

        assert trend == 'neutral'

    def test_reset_clears_ema_history(self):
        """Verify reset clears EMA history."""
        calculator = EMACalculator()

        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(100)]
        )

        for bar in bars:
            calculator.calculate_emas([bar])

        # Verify history exists
        history_before = calculator.get_ema_history()
        assert len(history_before['fast_ema']) > 0

        # Reset
        calculator.reset()

        # Verify history is cleared
        history_after = calculator.get_ema_history()
        assert len(history_after['fast_ema']) == 0
