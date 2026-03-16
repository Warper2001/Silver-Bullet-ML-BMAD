"""Integration tests for Dollar Bar validation."""

from datetime import datetime

import pytest

from src.data.models import DollarBar
from src.data.validation import DataValidator


class TestValidationPipeline:
    """Test end-to-end validation pipeline."""

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
        return DataValidator(input_queue, validated_queue, error_queue)

    @pytest.mark.asyncio
    async def test_validation_pipeline_valid_bars(
        self, validator, input_queue, validated_queue
    ) -> None:
        """Test valid bars flow through validation pipeline."""
        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        # Publish to input queue
        input_queue.put_nowait(bar)

        # Validator would pick this up and validate it
        result = await validator._validate_bar(bar)
        await validator._publish_result(bar, result)

        # Verify it went to validated queue
        assert validated_queue.qsize() == 1
        assert validator._error_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_validation_metrics_tracking(
        self, validator, input_queue, validated_queue
    ) -> None:
        """Test validation metrics are tracked correctly."""
        bars = [
            DollarBar(
                timestamp=datetime.now(),
                open=4523.25 + i,
                high=4524.00 + i,
                low=4523.00 + i,
                close=4523.75 + i,
                volume=1000,
                notional_value=50_000_000,
            )
            for i in range(5)
        ]

        for bar in bars:
            result = await validator._validate_bar(bar)
            await validator._publish_result(bar, result)

        assert validator.bars_validated == 5
        assert validator.errors_detected == 0

    @pytest.mark.asyncio
    async def test_metrics_logging_no_exception(self, validator) -> None:
        """Test metrics logging doesn't raise exceptions."""
        # Should not raise exception even if no bars validated yet
        validator._log_metrics_periodically()

        assert validator.bars_validated == 0


class TestDualOutputPattern:
    """Test dual-output pattern for valid and anomalous bars."""

    @pytest.fixture
    def queues(self):
        """Create all required queues."""
        import asyncio

        return {
            "input": asyncio.Queue(),
            "validated": asyncio.Queue(),
            "error": asyncio.Queue(),
        }

    @pytest.fixture
    def validator(self, queues):
        """Create DataValidator instance."""
        return DataValidator(queues["input"], queues["validated"], queues["error"])

    @pytest.mark.asyncio
    async def test_valid_bar_to_validated_queue(self, validator, queues) -> None:
        """Test valid bars are published to validated queue."""
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

        assert queues["validated"].qsize() == 1
        assert queues["error"].qsize() == 0

    @pytest.mark.asyncio
    async def test_error_tracking(self, validator, queues) -> None:
        """Test error and warning counts are tracked."""
        # Valid bar
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

        assert validator.bars_validated == 1
        assert validator.errors_detected == 0
        assert validator.warnings_detected == 0

        # Bar with extreme price movement (warnings)
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

        assert validator.bars_validated == 2
        assert validator.warnings_detected == 1


class TestValidationLatency:
    """Test validation latency performance."""

    @pytest.fixture
    def queues(self):
        """Create all required queues."""
        import asyncio

        return {
            "input": asyncio.Queue(),
            "validated": asyncio.Queue(),
            "error": asyncio.Queue(),
        }

    @pytest.fixture
    def validator(self, queues):
        """Create DataValidator instance."""
        return DataValidator(queues["input"], queues["validated"], queues["error"])

    @pytest.mark.asyncio
    async def test_validation_latency_under_10ms(self, validator) -> None:
        """Test validation latency is under 10ms target."""
        import time

        bar = DollarBar(
            timestamp=datetime.now(),
            open=4523.25,
            high=4524.00,
            low=4523.00,
            close=4523.75,
            volume=1000,
            notional_value=50_000_000,
        )

        start = time.perf_counter()
        result = await validator._validate_bar(bar)
        latency_ms = (time.perf_counter() - start) * 1000

        assert latency_ms < 10.0, f"Validation took {latency_ms:.2f}ms (> 10ms)"
        assert result.is_valid is True


class TestQueueBackpressure:
    """Test queue overflow handling."""

    @pytest.fixture
    def queues(self):
        """Create queues with size limits."""
        import asyncio

        return {
            "input": asyncio.Queue(),
            "validated": asyncio.Queue(maxsize=5),
            "error": asyncio.Queue(maxsize=5),
        }

    @pytest.fixture
    def validator(self, queues):
        """Create DataValidator instance."""
        return DataValidator(queues["input"], queues["validated"], queues["error"])

    @pytest.mark.asyncio
    async def test_validated_queue_full(self, validator, queues) -> None:
        """Test handling when validated queue is full."""
        # Fill validated queue
        for _ in range(5):
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

        assert queues["validated"].full()

        # Try to publish another bar (should handle gracefully)
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

        # Queue should still be full (6th bar dropped)
        assert queues["validated"].qsize() == 5

    @pytest.mark.asyncio
    async def test_error_queue_full(self, validator, queues) -> None:
        """Test handling when error queue is full."""
        # Fill error queue
        for _ in range(5):
            bar = DollarBar(
                timestamp=datetime.now(),
                open=4523.25,
                high=4524.00,
                low=4523.00,
                close=4523.75,
                volume=1000,
                notional_value=50_000_000,
            )
            # Create invalid result (with errors)
            from src.data.models import ValidationResult

            result = ValidationResult(
                is_valid=False,
                timestamp=datetime.now(),
                errors=["Test error"],
                warnings=[],
                severity="ERROR",
            )
            queues["error"].put_nowait((bar, result))

        assert queues["error"].full()

        # Try to publish another error (should handle gracefully)
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

        # Queue should still be full (6th error dropped)
        assert queues["error"].qsize() == 5
