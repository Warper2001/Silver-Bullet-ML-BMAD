"""
KuCoin SDK Exception Hierarchy

This module defines the exception hierarchy for the KuCoin SDK.
All exceptions inherit from KuCoinError for consistent error handling.

API Documentation: https://docs.kucoin.com/

Exception Tree:
KuCoinError (base)
├── AuthError
│   ├── InvalidCredentialsError
│   └── SignatureGenerationError
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


class KuCoinError(Exception):
    """
    Base exception for all KuCoin SDK errors.

    All KuCoin SDK exceptions inherit from this class, allowing
    broad exception handling: `except KuCoinError`.

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


class AuthError(KuCoinError):
    """
    Base exception for authentication-related errors.

    AuthErrors are caused by issues with API key authentication,
    passphrase validation, or signature generation.

    Note: AuthErrors do NOT trigger the order circuit breaker since
    they are not order execution failures.
    """

    pass


class InvalidCredentialsError(AuthError):
    """
    Invalid API key, secret, or passphrase.

    Raised when:
    - API key is incorrect or expired
    - Secret key is incorrect
    - Passphrase is incorrect
    - API key does not have required permissions

    Action Required: Verify credentials in .env.kucoin file.
    """

    def __init__(self, message: str = "Invalid credentials", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class SignatureGenerationError(AuthError):
    """
    Failed to generate HMAC SHA256 signature.

    Raised when:
    - Secret key is malformed
    - Signature generation fails
    - Query string encoding error

    KuCoin requires HMAC SHA256 signatures for all signed endpoints.

    Action Required: Verify secret key format and encoding.
    """

    def __init__(self, message: str = "Signature generation failed", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


# =============================================================================
# API Errors
# =============================================================================


class APIError(KuCoinError):
    """
    Base exception for API call failures.

    APIErrors are caused by issues with HTTP requests to the KuCoin API,
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

    KuCoin uses weight-based rate limiting.
    Different endpoints have different weights.

    Raised when:
    - Weight limit exceeded
    - Request rate too high

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
    - Cannot connect to KuCoin API
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

    Raised when the KuCoin API validates request parameters
    and finds them invalid (e.g., malformed order, invalid symbol).

    Action Required: Validate request parameters before retrying.
    """

    def __init__(self, message: str = "Validation error", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=400, details=details)


# =============================================================================
# Order Errors (Special Handling - Circuit Breaker Trigger)
# =============================================================================


class OrderError(KuCoinError):
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
    Order rejected by the exchange.

    Raised when KuCoin rejects an order for reasons such as:
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
    Position size or account limit exceeded.

    Raised when attempting to exceed:
    - Maximum position size (configurable)
    - Available balance for trading
    - Order value exceeds limits

    Action Required: Reduce order quantity or close existing positions.
    """

    def __init__(self, message: str = "Position limit exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)


class InsufficientFundsError(OrderError):
    """
    Insufficient account balance for order.

    Raised when account does not have sufficient funds to cover
    the order value (including fees).

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
