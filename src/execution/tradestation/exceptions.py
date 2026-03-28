"""
TradeStation SDK Exception Hierarchy

This module defines the exception hierarchy for the TradeStation SDK.
All exceptions inherit from TradeStationError for consistent error handling.

Exception Tree:
TradeStationError (base)
├── AuthError
│   ├── TokenExpiredError
│   ├── InvalidCredentialsError
│   └── AuthRefreshFailedError
├── APIError
│   ├── RateLimitError
│   ├── NetworkError
│   └── ValidationError
└── OrderError
    ├── OrderRejectedError
    ├── PositionLimitError
    ├── InsufficientFundsError
    └── OrderNotFoundError
"""

from typing import Any


class TradeStationError(Exception):
    """
    Base exception for all TradeStation SDK errors.

    All TradeStation SDK exceptions inherit from this class, allowing
    broad exception handling: `except TradeStationError`.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
                (e.g., API response, status_code, request_id)
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


# =============================================================================
# Authentication Errors
# =============================================================================


class AuthError(TradeStationError):
    """
    Base exception for authentication-related errors.

    AuthErrors are caused by issues with OAuth 2.0 authentication,
    token management, or credential validation.

    Note: AuthErrors do NOT trigger the order circuit breaker since
    they are not order execution failures.
    """

    pass


class TokenExpiredError(AuthError):
    """
    Access token has expired and needs refresh.

    This is typically handled automatically by the TokenManager,
    but may be raised if refresh fails.
    """

    def __init__(self, message: str = "Access token expired", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class InvalidCredentialsError(AuthError):
    """
    Invalid client credentials or refresh token.

    Raised when:
    - Client ID or client secret is incorrect
    - Refresh token is invalid or expired
    - Authorization code is malformed or expired

    Action Required: Verify credentials in .env file.
    """

    def __init__(self, message: str = "Invalid credentials", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class AuthRefreshFailedError(AuthError):
    """
    Failed to refresh access token.

    Raised when automatic token refresh fails after multiple attempts.
    May indicate network issues or API unavailability.

    Action Required: Check network connectivity and API status.
    """

    def __init__(self, message: str = "Token refresh failed", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


# =============================================================================
# API Errors
# =============================================================================


class APIError(TradeStationError):
    """
    Base exception for API call failures.

    APIErrors are caused by issues with HTTP requests to the TradeStation API,
    excluding authentication errors (which use AuthError).

    Attributes:
        message: Human-readable error message
        status_code: Optional HTTP status code from API response
        details: Optional dictionary with additional error context

    Note: APIErrors for data fetching trigger retry logic.
    APIErrors for order operations may trigger circuit breaker.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.status_code = status_code
        if status_code is not None:
            self.details["status_code"] = status_code


class RateLimitError(APIError):
    """
    Rate limit exceeded for API access.

    Raised when the TradeStation API rate limit has been exceeded.
    The SDK implements client-side rate limiting, but this may still
    occur under heavy load.

    Attributes:
        retry_after: Optional seconds to wait before retrying (from API)
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after
        if retry_after is not None:
            self.details["retry_after"] = retry_after


class NetworkError(APIError):
    """
    Network connectivity failure.

    Raised when:
    - Cannot connect to TradeStation API
    - Connection timeout
    - DNS resolution failure
    - SSL/TLS handshake failure

    Note: NetworkErrors trigger retry logic with exponential backoff.
    """

    def __init__(self, message: str = "Network error", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)


class ValidationError(APIError):
    """
    API rejected request data.

    Raised when the TradeStation API validates request parameters
    and finds them invalid (e.g., malformed order, invalid symbol).

    Action Required: Validate request parameters before retrying.
    """

    def __init__(self, message: str = "Validation error", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=400, details=details)


# =============================================================================
# Order Errors (Special Handling - Circuit Breaker Trigger)
# =============================================================================


class OrderError(TradeStationError):
    """
    Base exception for order-related failures.

    OrderErrors DO trigger the circuit breaker since they represent
    order execution failures. Multiple OrderErrors in succession
    will open the circuit breaker.

    Important: OrderErrors are tracked separately from AuthErrors and
    general APIErrors in the circuit breaker logic.

    Circuit Breaker Logic:
    - OrderError: failure_count += 1 (toward circuit breaker threshold)
    - AuthError: NO impact on circuit breaker
    - APIError (non-order): NO impact on circuit breaker
    """

    pass


class OrderRejectedError(OrderError):
    """
    Order rejected by the broker.

    Raised when TradeStation rejects an order for reasons such as:
    - Invalid order parameters
    - Market closed for trading
    - Order type not supported for symbol
    - Insufficient permissions

    Action Required: Review order parameters and market conditions.
    """

    def __init__(self, message: str = "Order rejected", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class PositionLimitError(OrderError):
    """
    Position size or margin limit exceeded.

    Raised when attempting to exceed:
    - Maximum position size (configurable, default: 5 contracts)
    - Available margin for trading
    - Overnight position limits

    Action Required: Reduce order quantity or close existing positions.
    """

    def __init__(self, message: str = "Position limit exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class InsufficientFundsError(OrderError):
    """
    Insufficient account balance for order.

    Raised when account does not have sufficient funds to cover
    the order margin requirement.

    Action Required: Reduce order quantity or deposit funds.
    """

    def __init__(self, message: str = "Insufficient funds", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class OrderNotFoundError(OrderError):
    """
    Order ID not found.

    Raised when attempting to modify, cancel, or query an order
    that does not exist or has already been filled/cancelled.

    Action Required: Verify order ID is correct and still active.
    """

    def __init__(self, message: str = "Order not found", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
