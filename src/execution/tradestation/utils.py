"""
TradeStation SDK - Utility Functions

This module provides utility functions and decorators for the TradeStation SDK:
- RateLimitTracker: Client-side rate limiting
- retry_decorator: Retry logic with exponential backoff
- CircuitBreaker: Circuit breaker pattern for order operations
- Logging configuration

Design Pattern:
- Utilities are domain-agnostic (can be used across auth, market_data, orders)
- Follow dependency injection pattern
- Async-safe with proper locking
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

from src.execution.tradestation.exceptions import NetworkError, OrderError, RateLimitError

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
# Rate Limiting
# =============================================================================


class RateLimitTracker:
    """
    Client-side rate limit tracker for TradeStation API.

    Prevents HTTP 429 responses by tracking request rate locally
    and enforcing rate limits proactively.

    Attributes:
        requests_per_minute: Max requests per minute
        requests_per_day: Max requests per day
        minute_requests: Deque of request timestamps in current minute
        day_requests: Deque of request timestamps in current day

    Example:
        tracker = RateLimitTracker(requests_per_minute=100, requests_per_day=10000)
        await tracker.acquire_slot()  # Blocks if rate limit exceeded
        # Make API request
    """

    def __init__(self, requests_per_minute: int, requests_per_day: int) -> None:
        """
        Initialize rate limit tracker.

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
        """
        self.rpm = requests_per_minute
        self.rpd = requests_per_day
        self.minute_requests: deque[datetime] = deque()
        self.day_requests: deque[datetime] = deque()
        self._lock = asyncio.Lock()
        self.logger = setup_logger(f"{__name__}.RateLimitTracker")

    async def acquire_slot(self) -> None:
        """
        Acquire a request slot, blocking if rate limit would be exceeded.

        This method should be called before each API request.
        """
        async with self._lock:
            now = datetime.now()

            # Clean old requests
            self._clean_old_requests(now)

            # Check minute limit
            if len(self.minute_requests) >= self.rpm:
                wait_time = self._calculate_wait_time("minute")
                self.logger.warning(f"Rate limit (minute) approaching, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                await self.acquire_slot()  # Retry after waiting
                return

            # Check day limit
            if len(self.day_requests) >= self.rpd:
                wait_time = self._calculate_wait_time("day")
                self.logger.warning(f"Rate limit (day) approaching, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                await self.acquire_slot()  # Retry after waiting
                return

            # Record this request
            self.minute_requests.append(now)
            self.day_requests.append(now)

    def _clean_old_requests(self, now: datetime) -> None:
        """
        Remove requests older than the tracking window.

        Args:
            now: Current timestamp
        """
        # Clean minute requests (older than 60 seconds)
        one_minute_ago = now.timestamp() - 60
        while self.minute_requests and self.minute_requests[0].timestamp() < one_minute_ago:
            self.minute_requests.popleft()

        # Clean day requests (older than 24 hours)
        one_day_ago = now.timestamp() - 86400
        while self.day_requests and self.day_requests[0].timestamp() < one_day_ago:
            self.day_requests.popleft()

    def _calculate_wait_time(self, window: str) -> float:
        """
        Calculate wait time until next request slot is available.

        Args:
            window: "minute" or "day"

        Returns:
            Wait time in seconds
        """
        now = datetime.now()

        if window == "minute" and self.minute_requests:
            oldest = self.minute_requests[0]
            wait_until = oldest.timestamp() + 60
            return max(0, wait_until - now.timestamp())

        if window == "day" and self.day_requests:
            oldest = self.day_requests[0]
            wait_until = oldest.timestamp() + 86400
            return max(0, wait_until - now.timestamp())

        return 0


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
