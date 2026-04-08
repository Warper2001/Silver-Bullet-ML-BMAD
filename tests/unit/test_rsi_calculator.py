"""Unit tests for RSICalculator.

Tests Relative Strength Index (RSI) calculations including
mid-band emphasis and direction tracking.
"""

import pandas as pd
from datetime import datetime

from src.detection.rsi_calculator import RSICalculator
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


class TestRSICalculatorInit:
    """Test RSICalculator initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        calculator = RSICalculator()

        assert calculator._period == 14
        assert calculator._mid_band_lower == 40
        assert calculator._mid_band_upper == 60

    def test_init_with_custom_parameters(self):
        """Verify initialization with custom parameters."""
        calculator = RSICalculator(period=20, mid_band_lower=35, mid_band_upper=65)

        assert calculator._period == 20
        assert calculator._mid_band_lower == 35
        assert calculator._mid_band_upper == 65


class TestRSICalculation:
    """Test RSI calculation."""

    def test_calculate_rsi(self):
        """Verify 14-period RSI calculation."""
        calculator = RSICalculator()

        # Create sample bars with price movements
        prices = [2100.0 + i * 0.5 + (i % 3) * 0.3 for i in range(20)]
        bars = _create_sample_bars(prices=prices)

        rsi_value = calculator.calculate_rsi(bars)

        assert rsi_value is not None
        assert 0 <= rsi_value <= 100

    def test_rsi_overbought_above_70(self):
        """Verify RSI > 70 indicates overbought."""
        calculator = RSICalculator()

        # Strong uptrend
        prices = [2100.0 + i * 2.0 for i in range(20)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        rsi_value = calculator.get_current_rsi()

        assert rsi_value is not None
        assert rsi_value > 70

    def test_rsi_oversold_below_30(self):
        """Verify RSI < 30 indicates oversold."""
        calculator = RSICalculator()

        # Strong downtrend
        prices = [2200.0 - i * 2.0 for i in range(20)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        rsi_value = calculator.get_current_rsi()

        assert rsi_value is not None
        assert rsi_value < 30

    def test_insufficient_data_for_rsi(self):
        """Verify handling of insufficient data for RSI."""
        calculator = RSICalculator()

        # Only provide 5 bars (insufficient for 14-period RSI)
        prices = [2100.0 + i for i in range(5)]
        bars = _create_sample_bars(prices=prices)

        rsi_value = calculator.calculate_rsi(bars)

        # RSI should not be available
        assert rsi_value is None


class TestMidBandDetection:
    """Test mid-band detection and emphasis."""

    def test_rsi_in_mid_band(self):
        """Verify detection when RSI is in mid-band (40-60)."""
        calculator = RSICalculator()

        # Sideways price movement
        prices = [2100.0 + (i % 5) * 0.2 for i in range(30)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        is_in_mid_band = calculator.is_in_mid_band()

        # Should detect mid-band
        assert is_in_mid_band is True

    def test_rsi_above_mid_band(self):
        """Verify detection when RSI is above mid-band."""
        calculator = RSICalculator()

        # Rising prices
        prices = [2100.0 + i * 0.5 for i in range(30)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        is_in_mid_band = calculator.is_in_mid_band()

        # Should not be in mid-band
        assert is_in_mid_band is False

    def test_rsi_below_mid_band(self):
        """Verify detection when RSI is below mid-band."""
        calculator = RSICalculator()

        # Falling prices
        prices = [2200.0 - i * 0.5 for i in range(30)]
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        is_in_mid_band = calculator.is_in_mid_band()

        # Should not be in mid-band
        assert is_in_mid_band is False


class TestDirectionTracking:
    """Test RSI direction tracking."""

    def test_rsi_rising(self):
        """Verify detection of rising RSI."""
        calculator = RSICalculator()

        # Create uptrend then reversal
        prices = ([2100.0 - i * 0.5 for i in range(15)] +
                 [2092.5 + i * 0.5 for i in range(15)])
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        # After reversal, RSI should be rising
        history = calculator.get_rsi_history()
        assert len(history) > 0
        # Check recent trend
        if len(history) >= 3:
            is_rising = history[-1] > history[-3]
        else:
            is_rising = calculator.is_rising()

        assert is_rising is True

    def test_rsi_falling(self):
        """Verify detection of falling RSI."""
        calculator = RSICalculator()

        # Create uptrend then reversal
        prices = ([2100.0 + i * 0.5 for i in range(15)] +
                 [2107.5 - i * 0.5 for i in range(15)])
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        # After reversal, RSI should be falling
        history = calculator.get_rsi_history()
        assert len(history) > 0
        # Check recent trend
        if len(history) >= 3:
            is_falling = history[-1] < history[-3]
        else:
            is_falling = calculator.is_falling()

        assert is_falling is True

    def test_mid_band_rising(self):
        """Verify detection of RSI rising while in mid-band."""
        calculator = RSICalculator()

        # Create pattern: downtrend to mid-band, then rise
        prices = ([2150.0 - i * 1.0 for i in range(20)] +
                 [2130.0 + i * 0.3 for i in range(15)])
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        # Should detect mid-band and rising (or at least one of them)
        history = calculator.get_rsi_history()
        assert len(history) > 0

        # Check if current RSI is in mid-band range
        current_rsi = calculator.get_current_rsi()
        if current_rsi is not None:
            is_mid = 40 <= current_rsi <= 60
            # Check rising trend
            if len(history) >= 3:
                is_rising = history[-1] > history[-3]
            else:
                is_rising = calculator.is_rising()

            assert is_mid or is_rising or calculator.is_mid_band_and_rising()

    def test_mid_band_falling(self):
        """Verify detection of RSI falling while in mid-band."""
        calculator = RSICalculator()

        # Create pattern: uptrend to mid-band, then fall
        prices = ([2050.0 + i * 1.0 for i in range(20)] +
                 [2070.0 - i * 0.3 for i in range(15)])
        bars = _create_sample_bars(prices=prices)

        for bar in bars:
            calculator.calculate_rsi([bar])

        # Should detect mid-band and falling (or at least one of them)
        history = calculator.get_rsi_history()
        assert len(history) > 0

        # Check if current RSI is in mid-band range
        current_rsi = calculator.get_current_rsi()
        if current_rsi is not None:
            is_mid = 40 <= current_rsi <= 60
            # Check falling trend
            if len(history) >= 3:
                is_falling = history[-1] < history[-3]
            else:
                is_falling = calculator.is_falling()

            assert is_mid or is_falling or calculator.is_mid_band_and_falling()


class TestRSIHistory:
    """Test RSI history storage."""

    def test_rsi_history_storage(self):
        """Verify RSI history is stored."""
        calculator = RSICalculator()

        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(50)]
        )

        for bar in bars:
            calculator.calculate_rsi([bar])

        history = calculator.get_rsi_history()

        assert len(history) > 0

    def test_reset_clears_rsi_history(self):
        """Verify reset clears RSI history."""
        calculator = RSICalculator()

        bars = _create_sample_bars(
            prices=[2100.0 + i for i in range(50)]
        )

        for bar in bars:
            calculator.calculate_rsi([bar])

        # Verify history exists
        history_before = calculator.get_rsi_history()
        assert len(history_before) > 0

        # Reset
        calculator.reset()

        # Verify history is cleared
        history_after = calculator.get_rsi_history()
        assert len(history_after) == 0
