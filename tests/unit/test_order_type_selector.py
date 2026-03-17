"""Unit tests for OrderTypeSelector.

Tests order type selection based on position size,
limit price calculation for bullish/bearish signals,
validation, and performance.
"""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from src.execution.order_type_selector import (
    OrderTypeDecision,
    OrderTypeSelector,
)


class TestOrderTypeSelectorInit:
    """Test OrderTypeSelector initialization and configuration."""

    def test_init_with_default_parameters(self):
        """Verify OrderTypeSelector initializes with default parameters."""
        selector = OrderTypeSelector()
        assert selector is not None
        assert selector._market_order_threshold == 3
        assert selector._tick_size == 0.25
        assert selector._tick_offset == 2

    def test_init_with_custom_parameters(self):
        """Verify OrderTypeSelector initializes with custom parameters."""
        selector = OrderTypeSelector(
            market_order_threshold=5,
            tick_size=0.50,
            tick_offset=3
        )
        assert selector._market_order_threshold == 5
        assert selector._tick_size == 0.50
        assert selector._tick_offset == 3

    def test_init_raises_error_for_invalid_threshold(self):
        """Verify ValueError raised for non-positive threshold."""
        with pytest.raises(ValueError, match="Market order threshold must be positive"):
            OrderTypeSelector(market_order_threshold=0)

        with pytest.raises(ValueError, match="Market order threshold must be positive"):
            OrderTypeSelector(market_order_threshold=-3)

    def test_init_raises_error_for_invalid_tick_size(self):
        """Verify ValueError raised for non-positive tick size."""
        with pytest.raises(ValueError, match="Tick size must be positive"):
            OrderTypeSelector(tick_size=0.0)

        with pytest.raises(ValueError, match="Tick size must be positive"):
            OrderTypeSelector(tick_size=-0.25)

    def test_init_raises_error_for_invalid_tick_offset(self):
        """Verify ValueError raised for non-positive tick offset."""
        with pytest.raises(ValueError, match="Tick offset must be positive"):
            OrderTypeSelector(tick_offset=0)

        with pytest.raises(ValueError, match="Tick offset must be positive"):
            OrderTypeSelector(tick_offset=-2)


class TestOrderTypeSelection:
    """Test order type selection logic."""

    def test_select_market_order_for_small_position(self):
        """Verify MARKET order selected for position < threshold."""
        selector = OrderTypeSelector(market_order_threshold=3)
        order_type, rationale = selector.select_order_type(position_size=2)

        assert order_type == "MARKET"
        assert "2 < 3" in rationale
        assert "market order" in rationale.lower()
        assert "immediate execution" in rationale.lower()

    def test_select_market_order_for_threshold_minus_one(self):
        """Verify MARKET order selected for position = threshold - 1."""
        selector = OrderTypeSelector(market_order_threshold=3)
        order_type, rationale = selector.select_order_type(position_size=2)

        assert order_type == "MARKET"

    def test_select_limit_order_for_large_position(self):
        """Verify LIMIT order selected for position >= threshold."""
        selector = OrderTypeSelector(market_order_threshold=3)
        order_type, rationale = selector.select_order_type(position_size=3)

        assert order_type == "LIMIT"
        assert "3 >= 3" in rationale
        assert "limit order" in rationale.lower()
        assert "2-tick offset" in rationale

    def test_select_limit_order_for_very_large_position(self):
        """Verify LIMIT order selected for very large positions."""
        selector = OrderTypeSelector(market_order_threshold=3)
        order_type, rationale = selector.select_order_type(position_size=10)

        assert order_type == "LIMIT"
        assert "10 >= 3" in rationale

    def test_select_limit_order_for_exactly_threshold(self):
        """Verify LIMIT order selected for position = threshold."""
        selector = OrderTypeSelector(market_order_threshold=5)
        order_type, rationale = selector.select_order_type(position_size=5)

        assert order_type == "LIMIT"


class TestLimitPriceCalculation:
    """Test limit price calculation based on direction."""

    def test_calculate_limit_price_for_bullish_signal(self):
        """Verify limit price calculated above current for bullish."""
        selector = OrderTypeSelector(tick_size=0.25, tick_offset=2)
        limit_price = selector.calculate_limit_price(
            current_price=11800.0,
            direction="bullish"
        )

        # 11800 + (2 × 0.25) = 11800.50
        assert limit_price == 11800.50

    def test_calculate_limit_price_for_bearish_signal(self):
        """Verify limit price calculated below current for bearish."""
        selector = OrderTypeSelector(tick_size=0.25, tick_offset=2)
        limit_price = selector.calculate_limit_price(
            current_price=11800.0,
            direction="bearish"
        )

        # 11800 - (2 × 0.25) = 11799.50
        assert limit_price == 11799.50

    def test_calculate_limit_price_with_custom_tick_offset(self):
        """Verify limit price calculation with custom tick offset."""
        selector = OrderTypeSelector(tick_size=0.25, tick_offset=3)
        limit_price = selector.calculate_limit_price(
            current_price=11800.0,
            direction="bullish"
        )

        # 11800 + (3 × 0.25) = 11800.75
        assert limit_price == 11800.75

    def test_calculate_limit_price_with_custom_tick_size(self):
        """Verify limit price calculation with custom tick size."""
        selector = OrderTypeSelector(tick_size=0.50, tick_offset=2)
        limit_price = selector.calculate_limit_price(
            current_price=11800.0,
            direction="bullish"
        )

        # 11800 + (2 × 0.50) = 11801.00
        assert limit_price == 11801.00

    def test_calculate_limit_price_raises_error_for_zero_price(self):
        """Verify ValueError raised for zero current price."""
        selector = OrderTypeSelector()

        with pytest.raises(ValueError, match="Current price must be positive"):
            selector.calculate_limit_price(current_price=0.0, direction="bullish")

    def test_calculate_limit_price_raises_error_for_negative_price(self):
        """Verify ValueError raised for negative current price."""
        selector = OrderTypeSelector()

        with pytest.raises(ValueError, match="Current price must be positive"):
            selector.calculate_limit_price(current_price=-100.0, direction="bullish")

    def test_calculate_limit_price_raises_error_for_invalid_direction(self):
        """Verify ValueError raised for invalid direction."""
        selector = OrderTypeSelector()

        with pytest.raises(ValueError, match="Invalid direction"):
            selector.calculate_limit_price(current_price=11800.0, direction="invalid")


class TestMainDecideMethod:
    """Test main decide_order_type() method."""

    @pytest.fixture
    def default_selector(self):
        """Create selector with default parameters."""
        return OrderTypeSelector()

    def test_decide_market_order_for_small_position(self, default_selector):
        """Verify decide_order_type() returns MARKET decision for small positions."""
        result = default_selector.decide_order_type(
            position_size=2,
            direction="bullish",
            current_price=11800.0
        )

        assert isinstance(result, OrderTypeDecision)
        assert result.order_type == "MARKET"
        assert result.limit_price is None
        assert result.selection_time_ms >= 0
        assert "market order" in result.rationale.lower()

    def test_decide_limit_order_for_large_position(self, default_selector):
        """Verify decide_order_type() returns LIMIT decision for large positions."""
        result = default_selector.decide_order_type(
            position_size=5,
            direction="bullish",
            current_price=11800.0
        )

        assert isinstance(result, OrderTypeDecision)
        assert result.order_type == "LIMIT"
        assert result.limit_price == 11800.50
        assert result.selection_time_ms >= 0
        assert "limit order" in result.rationale.lower()

    def test_decide_includes_limit_price_for_bullish(self, default_selector):
        """Verify LIMIT decision includes correct limit price for bullish."""
        result = default_selector.decide_order_type(
            position_size=3,
            direction="bullish",
            current_price=11800.0
        )

        assert result.order_type == "LIMIT"
        assert result.limit_price == pytest.approx(11800.50, abs=0.01)

    def test_decide_includes_limit_price_for_bearish(self, default_selector):
        """Verify LIMIT decision includes correct limit price for bearish."""
        result = default_selector.decide_order_type(
            position_size=3,
            direction="bearish",
            current_price=11800.0
        )

        assert result.order_type == "LIMIT"
        assert result.limit_price == pytest.approx(11799.50, abs=0.01)

    def test_decide_tracks_performance(self, default_selector):
        """Verify decision tracks selection time."""
        result = default_selector.decide_order_type(
            position_size=2,
            direction="bullish",
            current_price=11800.0
        )

        assert result.selection_time_ms >= 0
        assert isinstance(result.selection_time_ms, float)

    def test_decide_raises_error_for_invalid_position_size(self, default_selector):
        """Verify ValueError raised for invalid position size."""
        with pytest.raises(ValueError, match="Position size must be positive"):
            default_selector.decide_order_type(
                position_size=0,
                direction="bullish",
                current_price=11800.0
            )

        with pytest.raises(ValueError, match="Position size must be positive"):
            default_selector.decide_order_type(
                position_size=-5,
                direction="bullish",
                current_price=11800.0
            )


class TestPerformanceRequirements:
    """Test performance requirements."""

    def test_selection_completes_under_1ms(self):
        """Verify order type selection completes in < 1ms."""
        import time

        selector = OrderTypeSelector()

        start_time = time.perf_counter()
        result = selector.decide_order_type(
            position_size=3,
            direction="bullish",
            current_price=11800.0
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.selection_time_ms < 1.0, (
            f"Selection took {elapsed_ms:.3f}ms, exceeds 1ms limit"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_decide_with_very_small_position(self):
        """Verify decision handles minimum position size."""
        selector = OrderTypeSelector()
        result = selector.decide_order_type(
            position_size=1,
            direction="bullish",
            current_price=11800.0
        )

        # Should use market order for small position
        assert result.order_type == "MARKET"

    def test_decide_with_very_large_position(self):
        """Verify decision handles very large positions."""
        selector = OrderTypeSelector()
        result = selector.decide_order_type(
            position_size=100,
            direction="bullish",
            current_price=11800.0
        )

        # Should use limit order for large position
        assert result.order_type == "LIMIT"
        assert result.limit_price is not None

    def test_decide_with_fractional_current_price(self):
        """Verify decision handles fractional current prices."""
        selector = OrderTypeSelector()
        result = selector.decide_order_type(
            position_size=3,
            direction="bullish",
            current_price=11800.75
        )

        assert result.order_type == "LIMIT"
        # 11800.75 + 0.50 = 11801.25
        assert result.limit_price == pytest.approx(11801.25, abs=0.01)
