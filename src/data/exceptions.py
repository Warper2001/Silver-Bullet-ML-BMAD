"""Custom exceptions for data pipeline."""


class AuthenticationError(Exception):
    """Base exception for authentication failures."""

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize authentication error.

        Args:
            message: Error message describing the failure
            retry_count: Number of retry attempts made
            original_error: Original exception that caused this error
        """
        self.message = message
        self.retry_count = retry_count
        self.original_error = original_error
        super().__init__(self.message)


class TokenRefreshError(AuthenticationError):
    """Exception raised when token refresh fails after all retries."""

    pass


class ConfigurationError(Exception):
    """Exception raised when configuration is missing or invalid."""

    pass
