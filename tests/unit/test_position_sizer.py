"""Unit tests for PositionSizer.

Tests position size calculation based on risk parameters,
ATR-based stop loss calculation, validation, and performance.
"""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from src.data.models import SilverBulletSetup
from src.risk.position_sizer import PositionSizer, PositionSizeResult


class TestPositionSizerInit:
    """Test PositionSizer initialization and configuration."""

    def test_init_with_default_parameters(self):
        """Verify PositionSizer initializes with default parameters."""
        sizer = PositionSizer()
        assert sizer is not None
        assert sizer._account_equity == 10000.0
        assert sizer._risk_per_trade == 0.02
        assert sizer._max_position_size == 5
        assert sizer._atr_multiplier == 1.2

    def test_init_with_custom_parameters(self):
        """Verify PositionSizer initializes with custom parameters."""
        sizer = PositionSizer(
            account_equity=25000.0,
            risk_per_trade=0.03,
            max_position_size=10,
            atr_multiplier=1.5
        )
        assert sizer._account_equity == 25000.0
        assert sizer._risk_per_trade == 0.03
        assert sizer._max_position_size == 10
        assert sizer._atr_multiplier == 1.5

    def test_init_raises_error_for_invalid_equity(self):
        """Verify ValueError raised for negative account equity."""
        with pytest.raises(ValueError, match="Account equity must be positive"):
            PositionSizer(account_equity=-1000.0)

    def test_init_raises_error_for_zero_equity(self):
        """Verify ValueError raised for zero account equity."""
        with pytest.raises(ValueError, match="Account equity must be positive"):
            PositionSizer(account_equity=0.0)

    def test_init_raises_error_for_invalid_risk_per_trade(self):
        """Verify ValueError raised for risk_per_trade outside [0, 1]."""
        with pytest.raises(ValueError, match="Risk per trade must be in \\[0, 1\\]"):
            PositionSizer(risk_per_trade=-0.01)

        with pytest.raises(ValueError, match="Risk per trade must be in \\[0, 1\\]"):
            PositionSizer(risk_per_trade=1.5)

    def test_init_raises_error_for_invalid_max_position(self):
        """Verify ValueError raised for non-positive max_position_size."""
        with pytest.raises(ValueError, match="Max position size must be positive"):
            PositionSizer(max_position_size=0)

        with pytest.raises(ValueError, match="Max position size must be positive"):
            PositionSizer(max_position_size=-5)


class TestDollarRiskCalculation:
    """Test dollar risk amount calculation."""

    def test_calculate_dollar_risk_with_default_parameters(self):
        """Verify dollar risk calculation with default 2% risk."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        dollar_risk = sizer.calculate_dollar_risk()
        assert dollar_risk == 200.0  # $10,000 × 2%

    def test_calculate_dollar_risk_with_custom_parameters(self):
        """Verify dollar risk calculation with custom parameters."""
        sizer = PositionSizer(account_equity=50000.0, risk_per_trade=0.03)
        dollar_risk = sizer.calculate_dollar_risk()
        assert dollar_risk == 1500.0  # $50,000 × 3%

    def test_calculate_dollar_risk_with_one_percent_risk(self):
        """Verify dollar risk calculation with 1% risk."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        dollar_risk = sizer.calculate_dollar_risk()
        assert dollar_risk == 100.0  # $10,000 × 1%


class TestStopLossDistanceCalculation:
    """Test stop loss distance calculation based on ATR."""

    def test_calculate_stop_loss_distance_with_default_multiplier(self):
        """Verify stop loss distance calculation with default 1.2x multiplier."""
        sizer = PositionSizer(atr_multiplier=1.2)
        stop_distance = sizer.calculate_stop_loss_distance(atr=50.0)
        assert stop_distance == 60.0  # 1.2 × $50

    def test_calculate_stop_loss_distance_with_custom_multiplier(self):
        """Verify stop loss distance calculation with custom multiplier."""
        sizer = PositionSizer(atr_multiplier=1.5)
        stop_distance = sizer.calculate_stop_loss_distance(atr=50.0)
        assert stop_distance == 75.0  # 1.5 × $50

    def test_calculate_stop_loss_distance_handles_zero_atr(self):
        """Verify zero ATR uses default fallback value."""
        sizer = PositionSizer()
        stop_distance = sizer.calculate_stop_loss_distance(atr=0.0)
        assert stop_distance == 60.0  # Uses default $50 × 1.2

    def test_calculate_stop_loss_distance_handles_negative_atr(self):
        """Verify negative ATR uses default fallback value."""
        sizer = PositionSizer()
        stop_distance = sizer.calculate_stop_loss_distance(atr=-10.0)
        assert stop_distance == 60.0  # Uses default $50 × 1.2


class TestPositionSizeCalculation:
    """Test position size calculation."""

    def test_calculate_position_size_rounds_to_nearest_contract(self):
        """Verify position size rounds to nearest whole contract."""
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(
            entry_price=11800.0,
            stop_distance=60.0,
            dollar_risk=200.0
        )
        assert position_size == 3  # $200 / $60 = 3.33 → 3

    def test_calculate_position_size_rounds_up(self):
        """Verify position size rounds up when >= 0.5."""
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(
            entry_price=11800.0,
            stop_distance=40.0,
            dollar_risk=200.0
        )
        assert position_size == 5  # $200 / $40 = 5.0 → 5

    def test_calculate_position_size_ensures_minimum_one_contract(self):
        """Verify minimum position size is 1 contract."""
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(
            entry_price=11800.0,
            stop_distance=1000.0,
            dollar_risk=10.0
        )
        assert position_size == 1  # $10 / $1000 = 0.01 → 1 (minimum)

    def test_calculate_position_size_handles_zero_stop_distance(self):
        """Verify zero stop distance returns minimum 1 contract."""
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(
            entry_price=11800.0,
            stop_distance=0.0,
            dollar_risk=200.0
        )
        assert position_size == 1  # Fallback to minimum


class TestPositionSizeValidation:
    """Test position size validation."""

    def test_validate_position_size_within_limit(self):
        """Verify position size within limit is valid."""
        sizer = PositionSizer(max_position_size=5)
        valid, reason = sizer._validate_position_size(3)
        assert valid is True
        assert reason is None

    def test_validate_position_size_exceeds_limit(self):
        """Verify position size exceeding limit is invalid."""
        sizer = PositionSizer(max_position_size=5)
        valid, reason = sizer._validate_position_size(10)
        assert valid is False
        assert "exceeds maximum" in reason
        assert "5" in reason


class TestMainCalculateMethod:
    """Test main calculate_position() method."""

    @pytest.fixture
    def mock_signal(self):
        """Create mock Silver Bullet signal."""
        signal = Mock(spec=SilverBulletSetup)
        signal.entry_price = 11800.0
        signal.direction = "bullish"
        signal.atr = 50.0
        return signal

    def test_calculate_position_returns_valid_result(self, mock_signal):
        """Verify calculate_position() returns valid PositionSizeResult."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_position(mock_signal, atr=50.0)

        assert isinstance(result, PositionSizeResult)
        assert result.entry_price == 11800.0
        assert result.position_size == 3
        assert result.valid is True

    def test_calculate_position_calculates_bullish_stop_loss(self, mock_signal):
        """Verify stop loss calculated below entry for bullish signals."""
        mock_signal.direction = "bullish"
        sizer = PositionSizer()
        result = sizer.calculate_position(mock_signal, atr=50.0)

        assert result.stop_loss < result.entry_price
        assert result.stop_loss == pytest.approx(11740.0, abs=0.1)  # 11800 - 60

    def test_calculate_position_calculates_bearish_stop_loss(self, mock_signal):
        """Verify stop loss calculated above entry for bearish signals."""
        mock_signal.direction = "bearish"
        sizer = mock_signal
        mock_signal.entry_price = 11800.0
        sizer = PositionSizer()
        result = sizer.calculate_position(mock_signal, atr=50.0)

        assert result.stop_loss > result.entry_price
        assert result.stop_loss == pytest.approx(11860.0, abs=0.1)  # 11800 + 60

    def test_calculate_position_includes_dollar_risk(self, mock_signal):
        """Verify result includes dollar risk amount."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_position(mock_signal, atr=50.0)

        assert result.dollar_risk == 200.0

    def test_calculate_position_includes_calculation_time(self, mock_signal):
        """Verify result includes calculation time."""
        sizer = PositionSizer()
        result = sizer.calculate_position(mock_signal, atr=50.0)

        assert result.calculation_time_ms >= 0

    def test_calculate_position_validates_max_limit(self, mock_signal):
        """Verify position size validation against max limit."""
        # Set max position to 2
        sizer = PositionSizer(
            account_equity=10000.0,
            risk_per_trade=0.02,
            max_position_size=2
        )

        result = sizer.calculate_position(mock_signal, atr=50.0)

        # Should be invalid (3 > 2)
        assert result.valid is False
        assert "exceeds maximum" in result.validation_reason


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_calculation_completes_under_5ms(self):
        """Verify position size calculation completes in < 5ms."""
        import time

        sizer = PositionSizer()

        # Create mock signal
        signal = Mock(spec=SilverBulletSetup)
        signal.entry_price = 11800.0
        signal.direction = "bullish"

        start_time = time.perf_counter()
        result = sizer.calculate_position(signal, atr=50.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.calculation_time_ms < 5.0, f"Calculation took {elapsed_ms:.2f}ms, exceeds 5ms limit"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_calculate_position_without_atr_uses_signal_atr(self):
        """Verify calculate_position() uses signal.atr when atr not provided."""
        signal = Mock(spec=SilverBulletSetup)
        signal.entry_price = 11800.0
        signal.direction = "bullish"
        signal.atr = 75.0

        sizer = PositionSizer()
        result = sizer.calculate_position(signal)  # No atr parameter

        # Should use signal.atr
        assert result.stop_distance == pytest.approx(90.0, abs=0.1)  # 1.2 × 75

    def test_calculate_position_with_very_small_atr(self):
        """Verify calculation handles very small ATR values."""
        signal = Mock(spec=SilverBulletSetup)
        signal.entry_price = 11800.0
        signal.direction = "bullish"
        signal.atr = 1.0  # Very small ATR

        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_position(signal, atr=1.0)

        # Should still work with minimum position size
        assert result.position_size >= 1

    def test_calculate_position_with_very_large_atr(self):
        """Verify calculation handles very large ATR values."""
        signal = Mock(spec=SilverBulletSetup)
        signal.entry_price = 11800.0
        signal.direction = "bullish"
        signal.atr = 500.0  # Very large ATR

        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_position(signal, atr=500.0)

        # Should calculate small position size due to large stop distance
        assert result.position_size >= 1
