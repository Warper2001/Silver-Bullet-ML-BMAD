"""
Binance SDK - Utility Functions

This module provides utility functions and decorators for the Binance SDK:
- WeightBasedRateLimitTracker: Client-side rate limiting (weight-based)
- retry_decorator: Retry logic with exponential backoff
- CircuitBreaker: Circuit breaker pattern for order operations
- Logging configuration

Design Pattern:
- Utilities are domain-agnostic (can be used across auth, market_data, orders)
- Follow dependency injection pattern
- Async-safe with proper locking

API Documentation: https://binance-docs.github.io/apidocs/#limits
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.execution.binance.exceptions import NetworkError, OrderError, RateLimitError

# =============================================================================
# Logging Configuration
# =============================================================================


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ of calling module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# =============================================================================
# Weight-Based Rate Limiting (Binance-Specific)
# =============================================================================


class WeightBasedRateLimitTracker:
    """
    Client-side rate limit tracker for Binance API (weight-based).

    Binance uses weight-based rate limiting (NOT time-based like TradeStation).
    Different endpoints consume different weights:
    - GET /api/v3/klines: 1 weight
    - GET /api/v3/ticker/price: 1 weight
    - POST /api/v3/order: 1 weight
    - POST /api/v3/userDataStream: 2 weight

    Limits (production):
    - 1200 weight per minute
    - Order rate: 100 orders per 10 seconds (tracked separately)

    API Docs: https://binance-docs.github.io/apidocs/#limits

    Attributes:
        weight_limit_per_minute: Max weight per minute (default: 1200)
        current_weight: Current weight consumed in current minute
        minute_window: Deque of (timestamp, weight) tuples in current minute

    Example:
        tracker = WeightBasedRateLimitTracker(weight_limit_per_minute=1200)
        await tracker.acquire_weight(1)  # Blocks if weight limit exceeded
        # Make API request (consumes 1 weight)
        tracker.update_from_headers(response_headers)  # Update from API response
    """

    def __init__(self, weight_limit_per_minute: int = 1200) -> None:
        """
        Initialize weight-based rate limit tracker.

        Args:
            weight_limit_per_minute: Maximum weight per minute (default: 1200 for production)
        """
        self.weight_limit = weight_limit_per_minute
        self.current_weight = 0
        self.minute_window: deque[tuple[datetime, int]] = deque()
        self._lock = asyncio.Lock()
        self.logger = setup_logger(f"{__name__}.WeightBasedRateLimitTracker")

    async def acquire_weight(self, weight: int = 1) -> None:
        """
        Acquire weight for a request, blocking if weight limit would be exceeded.

        This method should be called before each API request.

        Args:
            weight: Weight cost of the request (default: 1)
        """
        async with self._lock:
            now = datetime.now()

            # Clean old weights (older than 60 seconds)
            self._clean_old_weights(now)

            # Check if adding this weight would exceed limit
            if self.current_weight + weight > self.weight_limit:
                wait_time = self._calculate_wait_time()
                self.logger.warning(
                    f"Weight limit ({self.weight_limit}) approaching, "
                    f"waiting {wait_time:.1f}s (current: {self.current_weight}, "
                    f"requesting: {weight})"
                )
                await asyncio.sleep(wait_time)
                await self.acquire_weight(weight)  # Retry after waiting
                return

            # Record this weight usage
            self.minute_window.append((now, weight))
            self.current_weight += weight

    def update_from_headers(self, headers: dict[str, str]) -> None:
        """
        Update weight tracking from API response headers.

        Binance returns rate limit info in response headers:
        - X-MBX-USED-WEIGHT: Total weight used in current minute
        - X-MBX-ORDER-COUNT-1M: Order count in current minute

        Args:
            headers: HTTP response headers from Binance API
        """
        try:
            used_weight = headers.get("X-MBX-USED-WEIGHT")
            if used_weight:
                self.current_weight = int(used_weight)
                self.logger.debug(f"Updated weight from API header: {self.current_weight}")
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse weight from headers: {e}")

    def _clean_old_weights(self, now: datetime) -> None:
        """
        Remove weights older than 60 seconds.

        Args:
            now: Current timestamp
        """
        one_minute_ago = now.timestamp() - 60
        while self.minute_window and self.minute_window[0][0].timestamp() < one_minute_ago:
            _, weight = self.minute_window.popleft()
            self.current_weight -= weight

    def _calculate_wait_time(self) -> float:
        """
        Calculate wait time until next weight slot is available.

        Returns:
            Wait time in seconds
        """
        if not self.minute_window:
            return 0

        now = datetime.now()
        oldest = self.minute_window[0][0]
        wait_until = oldest.timestamp() + 60
        return max(0, wait_until - now.timestamp())


# =============================================================================
# Retry Decorator
# =============================================================================


# Type variable for return type
T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.

    Use for data fetching operations where transient failures are expected.
    Do NOT use for order operations (use CircuitBreaker instead).

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3)
        async def fetch_quotes(symbol: str) -> List[Quote]:
            return await api.get_quotes(symbol)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        logger = setup_logger(f"{__name__}.retry")

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type((NetworkError, RateLimitError)),
            before_sleep=lambda retry_state: logger.info(
                f"Retrying {func.__name__}, attempt {retry_state.attempt_number + 1}"
            ),
        )
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Final attempt failed for {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and requests are blocked."""

    pass


class CircuitBreaker:
    """
    Circuit breaker pattern for order operations.

    Prevents cascading failures by stopping requests to a failing service.
    Opens after a threshold of failures, closes after a recovery timeout.

    State Machine:
        CLOSED → Normal operation, failures count toward threshold
        OPEN → Requests blocked, recovery timeout in progress
        HALF_OPEN → One request allowed to test recovery

    Attributes:
        failure_threshold: Number of failures to open circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that counts as failure

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        try:
            async with breaker:
                result = await api.place_order(order)
        except CircuitBreakerOpenError:
            logger.error("Circuit breaker is open, blocking request")
        except OrderError:
            # OrderError counts toward circuit breaker
            pass
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = OrderError,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state: str = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
        self.logger = setup_logger(f"{__name__}.CircuitBreaker")

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter circuit breaker context, raising error if open."""
        async with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                    self.logger.info("Circuit breaker entering half-open state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open (failures: {self.failure_count})"
                    )
        return self

    async def __aexit__(self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Exit circuit breaker context, recording success or failure."""
        async with self._lock:
            if exc_type is None:
                # Success
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    self.logger.info("Circuit breaker recovered, returning to closed state")
            elif isinstance(exc_val, self.expected_exception):
                # Failure (counts toward circuit breaker)
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    self.logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )
            else:
                # Other exception (doesn't count toward circuit breaker)
                pass

    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to attempt recovery.

        Returns:
            True if recovery timeout has elapsed
        """
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async def _reset() -> None:
            async with self._lock:
                self.state = "closed"
                self.failure_count = 0
                self.last_failure_time = None
                self.logger.info("Circuit breaker manually reset")

        asyncio.create_task(_reset())
