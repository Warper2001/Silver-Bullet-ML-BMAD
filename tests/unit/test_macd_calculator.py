"""Unit tests for MACDCalculator.

Tests MACD (Moving Average Convergence Divergence) calculations including
MACD line, signal line, and histogram.
"""

import pandas as pd
from datetime import datetime

from src.detection.macd_calculator import MACDCalculator
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


class TestMACDCalculatorInit:
    """Test MACDCalculator initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        calculator = MACDCalculator()

        assert calculator._fast_period == 12
        assert calculator._slow_period == 26
        assert calculator._signal_period == 9

    def test_init_with_custom_periods(self):
        """Verify initialization with custom periods."""
        calculator = MACDCalculator(fast_period=5, slow_period=10, signal_period=4)

        assert calculator._fast_period == 5
        assert calculator._slow_period == 10
        assert calculator._signal_period == 4


class TestMACDCalculation:
    """Test MACD calculation."""

    def test_calculate_macd_line(self):
        """Verify MACD line calculation (12 EMA - 26 EMA)."""
        calculator = MACDCalculator()

        # Create 30 sample bars (enough for MACD calculation)
        prices = [2100.0 + i for i in range(30)]
        bars = _create_sample_bars(prices=prices)

        macd_values = calculator.calculate_macd(bars)

        assert 'macd_line' in macd_values
        assert macd_values['macd_line'] is not None
        # MACD should be positive with rising prices
        assert macd_values['macd_line'] > 0

    def test_calculate_signal_line(self):
        """Verify signal line calculation (9 EMA of MACD)."""
        calculator = MACDCalculator()

        # Create 40 sample bars (enough for signal line)
        prices = [2100.0 + i for i in range(40)]
        bars = _create_sample_bars(prices=prices)

        macd_values = calculator.calculate_macd(bars)

        assert 'signal_line' in macd_values
        assert macd_values['signal_line'] is not None

    def test_calculate_histogram(self):
        """Verify histogram calculation (MACD - signal)."""
        calculator = MACDCalculator()

        # Create 40 sample bars
        prices = [2100.0 + i for i in range(40)]
        bars = _create_sample_bars(prices=prices)

        macd_values = calculator.calculate_macd(bars)

        assert 'histogram' in macd_values
        assert macd_values['histogram'] is not None
        # Histogram = MACD - signal
        expected_histogram = macd_values['macd_line'] - macd_values['signal_line']
        assert abs(macd_values['histogram'] - expected_histogram) < 0.01

    def test_insufficient_data_for_signal_line(self):
        """Verify handling of insufficient data for signal line."""
        calculator = MACDCalculator()

        # Only provide 30 bars (enough for MACD, not enough for signal)
        prices = [2100.0 + i for i in range(30)]
        bars = _create_sample_bars(prices=prices)

        macd_values = calculator.calculate_macd(bars)

        # MACD line should be available
        assert macd_values['macd_line'] is not None
        # Signal line should not be available yet
        assert macd_values['signal_line'] is None
        # Histogram should not be available
        assert macd_values['histogram'] is None

    def test_negative_macd_with_falling_prices(self):
        """Verify negative MACD with falling prices."""
        calculator = MACDCalculator()

        # Create falling price trend
        prices = [2200.0 - i for i in range(40)]
        bars = _create_sample_bars(prices=prices)

        macd_values = calculator.calculate_macd(bars)

        # MACD should be negative with falling prices
        assert macd_values['macd_line'] < 0


class TestMACDHistory:
    """Test MACD history storage."""

    def test_macd_history_storage(self):
        """Verify MACD history is stored for momentum analysis."""
        calculator = MACDCalculator()

        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(100)]
        )

        # Calculate MACD for each bar
        for bar in bars:
            calculator.calculate_macd([bar])

        history = calculator.get_macd_history()

        assert 'macd_line' in history
        assert 'signal_line' in history
        assert 'histogram' in history
        assert len(history['macd_line']) > 0

    def test_histogram_increasing_momentum(self):
        """Verify histogram shows increasing momentum."""
        calculator = MACDCalculator()

        # Create accelerating price trend
        prices = [2100.0 + i * 0.5 for i in range(50)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_macd([bar])

        history = calculator.get_macd_history()

        # Should have histogram history
        assert len(history['histogram']) > 0

    def test_reset_clears_macd_history(self):
        """Verify reset clears MACD history."""
        calculator = MACDCalculator()

        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(100)]
        )

        for bar in bars:
            calculator.calculate_macd([bar])

        # Verify history exists
        history_before = calculator.get_macd_history()
        assert len(history_before['macd_line']) > 0

        # Reset
        calculator.reset()

        # Verify history is cleared
        history_after = calculator.get_macd_history()
        assert len(history_after['macd_line']) == 0


class TestMomentumStrength:
    """Test momentum strength and direction."""

    def test_positive_histogram_bullish_momentum(self):
        """Verify positive histogram indicates bullish momentum."""
        calculator = MACDCalculator()

        # Rising prices with exponential trend
        prices = [2100.0 * (1.001 ** i) for i in range(80)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_macd([bar])

        macd_values = calculator.get_current_macd()

        # Should have positive histogram (bullish momentum)
        assert macd_values['histogram'] is not None
        assert macd_values['histogram'] > 0

    def test_negative_histogram_bearish_momentum(self):
        """Verify negative histogram indicates bearish momentum."""
        calculator = MACDCalculator()

        # Accelerating decline - momentum getting worse
        prices = [2200.0 - i * 0.5 - (i ** 2) * 0.01 for i in range(80)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_macd([bar])

        macd_values = calculator.get_current_macd()

        # Should have negative histogram (bearish momentum getting worse)
        assert macd_values['histogram'] is not None
        assert macd_values['histogram'] < 0

    def test_is_increasing_momentum(self):
        """Verify detection of increasing momentum."""
        calculator = MACDCalculator()

        # Create accelerating uptrend
        prices = [2100.0 + i * 0.1 * (i / 10) for i in range(60)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_macd([bar])

        # Get current state
        is_increasing = calculator.is_momentum_increasing()

        # Should detect increasing momentum
        assert is_increasing is True

    def test_is_decreasing_momentum(self):
        """Verify detection of decreasing momentum."""
        calculator = MACDCalculator()

        # Create accelerating downtrend
        prices = [2200.0 - i * 0.1 * (i / 10) for i in range(60)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_macd([bar])

        # Get current state
        is_decreasing = calculator.is_momentum_decreasing()

        # Should detect decreasing momentum
        assert is_decreasing is True
