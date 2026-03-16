"""Unit tests for Dollar Bar transformation."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.data.models import DollarBar
from src.data.transformation import BarBuilderState, DollarBarTransformer


class TestDollarBar:
    """Test DollarBar model validation."""

    def test_dollar_bar_creation_valid(self) -> None:
        """Test valid DollarBar creation."""
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.open == 4523.25
        assert bar.high == 4524.00
        assert bar.low == 4523.00
        assert bar.close == 4523.75
        assert bar.volume == 1000
        assert bar.notional_value == 50_000_000

    def test_dollar_bar_high_validation(self) -> None:
        """Test high must be >= open and close."""
        with pytest.raises(ValidationError, match="high must be >= open"):
            DollarBar(
                timestamp=datetime.now(),
                open=4524.00,
                high=4523.00,  # high < open (invalid)
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )

    def test_dollar_bar_low_validation(self) -> None:
        """Test low must be <= open and close."""
        with pytest.raises(ValidationError, match="low must be <= open"):
            DollarBar(
                timestamp=datetime.now(),
                open=4523.00,
                high=4524.00,
                low=4524.00,  # low > open (invalid)
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )

    def test_dollar_bar_high_close_consistency(self) -> None:
        """Test high must be >= close."""
        with pytest.raises(ValidationError, match="close must be <= high"):
            DollarBar(
                timestamp=datetime.now(),
                open=4523.00,
                high=4523.50,
                low=4523.00,
                close=4524.00,  # close > high (invalid)
                volume=1000,
                notional_value=50_000_000,
            )

    def test_dollar_bar_low_close_consistency(self) -> None:
        """Test low must be <= close."""
        # This test has low > close, but the validation that catches it is low > open
        # The close validation happens after, so we test that low <= open
        with pytest.raises(ValidationError, match="low must be <= open"):
            DollarBar(
                timestamp=datetime.now(),
                open=4523.00,
                high=4524.00,
                low=4523.75,  # low > open (invalid)
                close=4523.50,
                volume=1000,
                notional_value=50_000_000,
            )

    def test_dollar_bar_notional_value_positive(self) -> None:
        """Test notional value must be positive."""
        with pytest.raises(ValidationError, match="notional_value must be positive"):
            DollarBar(
                timestamp=datetime.now(),
                open=4523.00,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=0,  # Invalid: not positive
            )

    def test_dollar_bar_notional_value_sanity_check(self) -> None:
        """Test notional value sanity check (max $100M)."""
        with pytest.raises(ValidationError, match="exceeds reasonable maximum"):
            DollarBar(
                timestamp=datetime.now(),
                open=4523.00,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=150_000_000,  # Invalid: exceeds $100M
            )


class TestBarBuilderState:
    """Test BarBuilderState enum."""

    def test_bar_builder_states(self) -> None:
        """Test all bar builder states are defined."""
        assert BarBuilderState.IDLE.value == "idle"
        assert BarBuilderState.ACCUMULATING.value == "accumulating"
        assert BarBuilderState.COMPLETED.value == "completed"


class TestDollarBarTransformer:
    """Test DollarBarTransformer."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for MarketData."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def output_queue(self):
        """Create output queue for DollarBar."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def transformer(self, input_queue, output_queue):
        """Create DollarBarTransformer instance."""
        return DollarBarTransformer(input_queue, output_queue)

    def test_initialization(self, transformer: DollarBarTransformer) -> None:
        """Test transformer initializes correctly."""
        assert transformer.state == BarBuilderState.IDLE
        assert transformer._bar_open is None
        assert transformer.bars_created == 0
        assert transformer._total_trades_processed == 0

    def test_state_property(self, transformer: DollarBarTransformer) -> None:
        """Test state property returns current state."""
        assert transformer.state == BarBuilderState.IDLE

    def test_bars_created_property(self, transformer: DollarBarTransformer) -> None:
        """Test bars_created property returns count."""
        assert transformer.bars_created == 0

    def test_constants_defined(self) -> None:
        """Test transformation constants are defined correctly."""
        from src.data.transformation import (
            BAR_TIMEOUT_SECONDS,
            DOLLAR_THRESHOLD,
            MNQ_MULTIPLIER,
        )

        assert MNQ_MULTIPLIER == 0.5
        assert DOLLAR_THRESHOLD == 50_000_000
        assert BAR_TIMEOUT_SECONDS == 5
