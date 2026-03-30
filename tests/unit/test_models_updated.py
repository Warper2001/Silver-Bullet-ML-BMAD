"""Unit tests for updated DollarBar model with ge=0 validation."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.data.models import DollarBar


class TestDollarBarZeroValues:
    """Test DollarBar model allows zero values for closed markets."""

    def test_dollar_bar_with_all_zeros(self) -> None:
        """Test DollarBar accepts all zero values."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
        )

        assert bar.open == 0.0
        assert bar.high == 0.0
        assert bar.low == 0.0
        assert bar.close == 0.0
        assert bar.volume == 0
        assert bar.notional_value == 0

    def test_dollar_bar_with_zero_open(self) -> None:
        """Test DollarBar accepts zero open price (closed market)."""
        # When open is 0, other prices must also be 0 for closed market
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
        )

        assert bar.open == 0.0
        assert bar.high == 0.0
        assert bar.close == 0.0

    def test_dollar_bar_with_zero_high(self) -> None:
        """Test DollarBar accepts zero high price (closed market)."""
        # When high is 0, open must also be 0 for closed market
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
        )

        assert bar.open == 0.0
        assert bar.high == 0.0

    def test_dollar_bar_with_zero_low(self) -> None:
        """Test DollarBar accepts zero low price."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=0.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.open == 11800.0
        assert bar.low == 0.0
        assert bar.close == 11820.0

    def test_dollar_bar_with_zero_close(self) -> None:
        """Test DollarBar accepts zero close price (closed market)."""
        # When close is 0, high/low must also be 0 for closed market
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
        )

        assert bar.close == 0.0

    def test_dollar_bar_with_mixed_zeros(self) -> None:
        """Test DollarBar accepts mixed zero and non-zero values."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=11850.0,
            low=0.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.open == 0.0
        assert bar.high == 11850.0
        assert bar.low == 0.0
        assert bar.close == 11820.0

    def test_dollar_bar_with_positive_prices(self) -> None:
        """Test DollarBar still accepts positive prices."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.open == 11800.0
        assert bar.high == 11850.0
        assert bar.low == 11790.0
        assert bar.close == 11820.0

    def test_dollar_bar_negative_prices_rejected(self) -> None:
        """Test DollarBar rejects negative prices."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=-100.0,  # Negative
                high=11850.0,
                low=11790.0,
                close=11820.0,
                volume=1000,
                notional_value=50_000_000,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_dollar_bar_negative_high_rejected(self) -> None:
        """Test DollarBar rejects negative high price."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=-50.0,  # Negative
                low=11790.0,
                close=11820.0,
                volume=1000,
                notional_value=50_000_000,
            )

        assert "greater than or equal to 0" in str(exc_info.value)


class TestDollarBarValidators:
    """Test existing validators still work with ge=0 constraint."""

    def test_high_greater_than_equal_open_validator(self) -> None:
        """Test high >= open validator works with zero values."""
        # Valid: high >= open
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.high == 11850.0

    def test_high_less_than_open_rejected(self) -> None:
        """Test high < open is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11790.0,  # High < Open
                low=11790.0,
                close=11820.0,
                volume=1000,
                notional_value=50_000_000,
            )

        assert "high must be >= open" in str(exc_info.value)

    def test_low_less_than_equal_open_validator(self) -> None:
        """Test low <= open validator works with zero values."""
        # Valid: low <= open
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.low == 11790.0

    def test_low_greater_than_open_rejected(self) -> None:
        """Test low > open is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11850.0,
                low=11810.0,  # Low > Open
                close=11820.0,
                volume=1000,
                notional_value=50_000_000,
            )

        assert "low must be <= open" in str(exc_info.value)

    def test_close_within_high_low_validator(self) -> None:
        """Test close within high-low validator works."""
        # Valid: low <= close <= high
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
        )

        assert bar.close == 11820.0

    def test_close_above_high_rejected(self) -> None:
        """Test close > high is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11860.0,  # Close > High
                volume=1000,
                notional_value=50_000_000,
            )

        assert "close must be <= high" in str(exc_info.value)

    def test_close_below_low_rejected(self) -> None:
        """Test close < low is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11780.0,  # Close < Low
                volume=1000,
                notional_value=50_000_000,
            )

        assert "close must be >= low" in str(exc_info.value)

    def test_notional_value_zero_with_forward_fill(self) -> None:
        """Test zero notional value allowed for forward-filled bars."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
            is_forward_filled=True,
        )

        assert bar.notional_value == 0
        assert bar.is_forward_filled is True

    def test_notional_value_positive_for_real_bars(self) -> None:
        """Test positive notional value for real bars."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11820.0,
            volume=1000,
            notional_value=50_000_000,
            is_forward_filled=False,
        )

        assert bar.notional_value == 50_000_000

    def test_notional_value_negative_rejected(self) -> None:
        """Test negative notional value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11820.0,
                volume=1000,
                notional_value=-1000.0,  # Negative
                is_forward_filled=False,
            )

        # Negative values rejected by ge=0 constraint
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_notional_value_zero_without_forward_fill_rejected(self) -> None:
        """Test zero notional value rejected without forward fill flag."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11820.0,
                volume=1000,
                notional_value=0,  # Zero without forward fill
                is_forward_filled=False,
            )

        assert "notional_value must be positive" in str(exc_info.value)

    def test_notional_value_exceeds_maximum(self) -> None:
        """Test notional value exceeding $2B maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            DollarBar(
                timestamp=datetime.now(timezone.utc),
                open=11800.0,
                high=11850.0,
                low=11790.0,
                close=11820.0,
                volume=1000,
                notional_value=3_000_000_000,  # Exceeds $2B
                is_forward_filled=False,
            )

        assert "exceeds reasonable maximum" in str(exc_info.value)


class TestDollarBarEdgeCases:
    """Test edge cases with zero values."""

    def test_zero_volume_allowed(self) -> None:
        """Test zero volume is allowed."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=11800.0,
            high=11850.0,
            low=11790.0,
            close=11820.0,
            volume=0,
            notional_value=50_000_000,
        )

        assert bar.volume == 0

    def test_all_zeros_with_forward_fill(self) -> None:
        """Test all zeros with forward fill flag."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
            is_forward_filled=True,
        )

        assert bar.is_forward_filled is True
        assert bar.open == 0.0
        assert bar.high == 0.0
        assert bar.low == 0.0
        assert bar.close == 0.0

    def test_zero_prices_with_zero_notional(self) -> None:
        """Test zero prices with zero notional value."""
        bar = DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0,
            notional_value=0,
        )

        assert bar.notional_value == 0
