"""Data validation and anomaly detection for Dollar Bars."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from .models import DollarBar, ValidationResult

logger = logging.getLogger(__name__)

# Validation thresholds
MAX_REASONABLE_NOTIONAL = 2_000_000_000  # $2B (adjusted for actual dollar bar notional values)
MAX_PRICE_CHANGE_PCT = 10.0  # 10% change from previous bar
MAX_VOLUME_MULTIPLIER = 10.0  # 10× average volume


class DataValidator:
    """Validate DollarBar data quality and detect anomalies.

    Handles:
    - Field completeness validation
    - Price range validation
    - OHLC consistency validation
    - Volume validation
    - Notional value validation
    - Async consumption from DollarBar queue
    - Dual-output: valid bars → validated queue, anomalies → error queue
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[DollarBar],
        validated_queue: asyncio.Queue[DollarBar],
        error_queue: asyncio.Queue[tuple[DollarBar, ValidationResult]],
    ) -> None:
        """Initialize data validator.

        Args:
            input_queue: Queue receiving DollarBar from transformation
            validated_queue: Queue publishing validated DollarBar for gap detection
            error_queue: Queue publishing anomalous bars with validation results
        """
        self._input_queue = input_queue
        self._validated_queue = validated_queue
        self._error_queue = error_queue

        # Metrics
        self._bars_validated = 0
        self._errors_detected = 0
        self._warnings_detected = 0
        self._last_log_time = datetime.now()
        self._previous_close: Optional[float] = None

    async def consume(self) -> None:
        """Consume DollarBar stream and validate data quality.

        This runs in a background task and:
        1. Receives DollarBar from input queue
        2. Validates all data quality rules
        3. Publishes valid bars to validated queue
        4. Publishes anomalous bars to error queue
        5. Logs validation metrics
        """
        logger.info("DataValidator started")

        while True:
            try:
                # Receive DollarBar with timeout
                bar = await asyncio.wait_for(
                    self._input_queue.get(),
                    timeout=5.0,
                )

                validation_start = time.perf_counter()
                result = await self._validate_bar(bar)
                validation_latency_ms = (time.perf_counter() - validation_start) * 1000

                # Log validation latency if exceeds threshold
                if validation_latency_ms > 10:
                    logger.warning(
                        f"Validation latency exceeded 10ms: {validation_latency_ms:.2f}ms"
                    )

                # Publish to appropriate queue
                await self._publish_result(bar, result)

                # Log metrics periodically
                self._log_metrics_periodically()

            except asyncio.TimeoutError:
                # No bars received - continue waiting
                continue

            except Exception as e:
                logger.error(f"Validation error: {e}")
                # Continue processing - don't let one error stop the pipeline

    async def _validate_bar(self, bar: DollarBar) -> ValidationResult:
        """Validate DollarBar data quality.

        Args:
            bar: DollarBar to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Field completeness validation
        if bar.timestamp is None:
            errors.append("Missing required field: timestamp")
        if bar.open is None:
            errors.append("Missing required field: open")
        if bar.high is None:
            errors.append("Missing required field: high")
        if bar.low is None:
            errors.append("Missing required field: low")
        if bar.close is None:
            errors.append("Missing required field: close")
        if bar.volume is None:
            errors.append("Missing required field: volume")
        if bar.notional_value is None:
            errors.append("Missing required field: notional_value")

        # 2. Price range validation
        if bar.open is not None and bar.open <= 0:
            errors.append(f"Invalid open price: {bar.open} (must be > 0)")
        if bar.high is not None and bar.high <= 0:
            errors.append(f"Invalid high price: {bar.high} (must be > 0)")
        if bar.low is not None and bar.low <= 0:
            errors.append(f"Invalid low price: {bar.low} (must be > 0)")
        if bar.close is not None and bar.close <= 0:
            errors.append(f"Invalid close price: {bar.close} (must be > 0)")

        # 3. OHLC consistency validation
        if bar.high is not None and bar.low is not None and bar.high < bar.low:
            errors.append(f"OHLC inconsistency: high ({bar.high}) < low ({bar.low})")
        if bar.high is not None and bar.open is not None and bar.high < bar.open:
            warnings.append(f"Suspicious: high ({bar.high}) < open ({bar.open})")
        if bar.high is not None and bar.close is not None and bar.high < bar.close:
            warnings.append(f"Suspicious: high ({bar.high}) < close ({bar.close})")
        if bar.low is not None and bar.open is not None and bar.low > bar.open:
            warnings.append(f"Suspicious: low ({bar.low}) > open ({bar.open})")
        if bar.low is not None and bar.close is not None and bar.low > bar.close:
            warnings.append(f"Suspicious: low ({bar.low}) > close ({bar.close})")

        # 4. Volume validation
        if bar.volume is not None and bar.volume < 0:
            errors.append(f"Invalid volume: {bar.volume} (must be >= 0)")

        # 5. Notional value validation
        if bar.notional_value is not None and bar.notional_value <= 0:
            errors.append(f"Invalid notional_value: {bar.notional_value} (must be > 0)")
        if (
            bar.notional_value is not None
            and bar.notional_value > MAX_REASONABLE_NOTIONAL
        ):
            errors.append(
                f"Notional value exceeds maximum: "
                f"${bar.notional_value:.2f} > ${MAX_REASONABLE_NOTIONAL:.2f}"
            )

        # 6. Extreme price movement detection (WARNING)
        if self._previous_close is not None and bar.close is not None:
            price_change_pct = (
                abs(bar.close - self._previous_close) / self._previous_close * 100
            )
            if price_change_pct > MAX_PRICE_CHANGE_PCT:
                warnings.append(
                    f"Extreme price movement: {price_change_pct:.2f}% "
                    f"(from {self._previous_close:.2f} to {bar.close:.2f})"
                )

        # Update previous close for next comparison
        if bar.close is not None:
            self._previous_close = bar.close

        # Determine severity
        is_valid = len(errors) == 0
        if not is_valid or any("Invalid" in e for e in errors):
            severity = "ERROR"
        elif warnings:
            severity = "WARNING"
        else:
            severity = "PASS"

        return ValidationResult(
            is_valid=is_valid,
            timestamp=datetime.now(),
            errors=errors,
            warnings=warnings,
            severity=severity,
        )

    async def _publish_result(self, bar: DollarBar, result: ValidationResult) -> None:
        """Publish bar to appropriate queue based on validation result.

        Args:
            bar: DollarBar that was validated
            result: ValidationResult from validation
        """
        self._bars_validated += 1

        # Track errors and warnings
        self._errors_detected += len(result.errors)
        self._warnings_detected += len(result.warnings)

        if result.is_valid:
            # Publish to validated queue for Story 1.6
            try:
                self._validated_queue.put_nowait(bar)
            except asyncio.QueueFull:
                logger.error("Validated queue full, dropping bar")
        else:
            # Publish to error queue for monitoring
            try:
                self._error_queue.put_nowait((bar, result))
            except asyncio.QueueFull:
                logger.error("Error queue full, dropping anomaly")

            # Log anomaly
            logger.warning(
                f"Anomaly detected (severity={result.severity}): "
                f"errors={result.errors}, warnings={result.warnings} "
                f"bar: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} "
                f"C={bar.close:.2f} V={bar.volume}"
            )

    def _log_metrics_periodically(self) -> None:
        """Log validation metrics periodically (every 60 seconds)."""
        now = datetime.now()
        if (now - self._last_log_time).total_seconds() >= 60:
            success_rate = (
                (self._bars_validated - self._errors_detected)
                / self._bars_validated
                * 100
                if self._bars_validated > 0
                else 100
            )

            logger.info(
                f"Validation metrics: "
                f"bars_validated={self._bars_validated} "
                f"errors={self._errors_detected} "
                f"warnings={self._warnings_detected} "
                f"success_rate={success_rate:.2f}% "
                f"validated_queue_depth={self._validated_queue.qsize()} "
                f"error_queue_depth={self._error_queue.qsize()}"
            )
            self._last_log_time = now

    @property
    def bars_validated(self) -> int:
        """Get total bars validated since start."""
        return self._bars_validated

    @property
    def errors_detected(self) -> int:
        """Get total errors detected since start."""
        return self._errors_detected

    @property
    def warnings_detected(self) -> int:
        """Get total warnings detected since start."""
        return self._warnings_detected
