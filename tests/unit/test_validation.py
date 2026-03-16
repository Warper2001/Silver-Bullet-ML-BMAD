"""Unit tests for Dollar Bar validation."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.data.models import DollarBar, ValidationResult


class TestValidationResult:
    """Test ValidationResult model validation."""

    def test_validation_result_with_errors(self) -> None:
        """Test ValidationResult with errors has severity ERROR."""
        result = ValidationResult(
            is_valid=False,
            timestamp=datetime.now(),
            errors=["Missing field: timestamp"],
            warnings=[],
            severity="ERROR",
        )

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.severity == "ERROR"

    def test_validation_result_with_warnings_only(self) -> None:
        """Test ValidationResult with warnings only can be WARNING or ERROR."""
        result = ValidationResult(
            is_valid=True,
            timestamp=datetime.now(),
            errors=[],
            warnings=["Suspicious: high < open"],
            severity="WARNING",
        )

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.severity == "WARNING"

    def test_validation_result_pass(self) -> None:
        """Test ValidationResult with no issues has severity PASS."""
        result = ValidationResult(
            is_valid=True,
            timestamp=datetime.now(),
            errors=[],
            warnings=[],
            severity="PASS",
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.severity == "PASS"

    def test_validation_result_invalid_severity_with_errors(self) -> None:
        """Test ValidationResult with errors must have severity ERROR."""
        with pytest.raises(ValidationError, match="severity must be ERROR"):
            ValidationResult(
                is_valid=False,
                timestamp=datetime.now(),
                errors=["Missing field"],
                warnings=[],
                severity="WARNING",  # Invalid: should be ERROR
            )

    def test_validation_result_invalid_severity_with_warnings(self) -> None:
        """Test ValidationResult with warnings cannot have severity PASS."""
        with pytest.raises(ValidationError, match="severity must be WARNING or ERROR"):
            ValidationResult(
                is_valid=True,
                timestamp=datetime.now(),
                errors=[],
                warnings=["Suspicious"],
                severity="PASS",  # Invalid: should be WARNING or ERROR
            )


class TestDataValidator:
    """Test DataValidator class."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for DollarBar."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def validated_queue(self):
        """Create validated queue for DollarBar."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def error_queue(self):
        """Create error queue for anomalies."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def validator(self, input_queue, validated_queue, error_queue):
        """Create DataValidator instance."""
        from src.data.validation import DataValidator

        return DataValidator(input_queue, validated_queue, error_queue)

    def test_initialization(self, validator) -> None:
        """Test validator initializes correctly."""
        assert validator.bars_validated == 0
        assert validator.errors_detected == 0
        assert validator.warnings_detected == 0

    def test_metrics_properties(self, validator) -> None:
        """Test metrics properties return correct values."""
        assert validator.bars_validated == 0
        assert validator.errors_detected == 0
        assert validator.warnings_detected == 0


class TestDataValidatorValidation:
    """Test DataValidator validation logic."""

    @pytest.fixture
    def input_queue(self):
        """Create input queue for DollarBar."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def validated_queue(self):
        """Create validated queue for DollarBar."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def error_queue(self):
        """Create error queue for anomalies."""
        import asyncio

        return asyncio.Queue()

    @pytest.fixture
    def validator(self, input_queue, validated_queue, error_queue):
        """Create DataValidator instance."""
        from src.data.validation import DataValidator

        return DataValidator(input_queue, validated_queue, error_queue)

    @pytest.mark.asyncio
    async def test_valid_bar_passes_validation(self, validator) -> None:
        """Test valid DollarBar passes all validation checks."""
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        result = await validator._validate_bar(bar)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.severity == "PASS"

    @pytest.mark.asyncio
    async def test_dual_output_pattern_valid_bar(self, validator) -> None:
        """Test valid bars go to validated queue."""
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        result = await validator._validate_bar(bar)
        await validator._publish_result(bar, result)

        assert result.is_valid is True
        assert validator._validated_queue.qsize() == 1
        assert validator._error_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_dual_output_pattern_with_warnings(self, validator) -> None:
        """Test bars with warnings still go to validated queue."""
        # First bar to establish baseline
        bar1 = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )
        result1 = await validator._validate_bar(bar1)
        await validator._publish_result(bar1, result1)

        # Second bar with extreme price movement (>10%)
        bar2 = DollarBar(
            timestamp=datetime.now(),
            open=5000.00,
            high=5001.00,
            low=4999.00,
            close=5000.50,
            volume=1000,
            notional_value=50_000_000,
        )

        result2 = await validator._validate_bar(bar2)
        await validator._publish_result(bar2, result2)

        # Should have warnings (extreme price movement)
        # But since it's valid (no errors), it goes to validated queue
        assert result2.is_valid is True
        assert len(result2.warnings) > 0
        assert validator._validated_queue.qsize() == 2
        assert validator._error_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_extreme_price_movement_warning(self, validator) -> None:
        """Test extreme price movement (>10%) generates warning."""
        # First bar to establish baseline
        bar1 = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )
        await validator._validate_bar(bar1)

        # Second bar with >10% price change
        bar2 = DollarBar(
            timestamp=datetime.now(),
            open=5000.00,  # ~10.5% increase
            high=5001.00,
            low=4999.00,
            close=5000.50,
            volume=1000,
            notional_value=50_000_000,
        )

        result = await validator._validate_bar(bar2)

        assert result.is_valid is True
        assert any("Extreme price movement" in w for w in result.warnings)
        assert result.severity == "WARNING"
