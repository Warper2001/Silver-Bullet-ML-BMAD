"""Error Logger with full stack traces and context.

Centralized error logging system that captures all errors with
complete context, stack traces, and correlation IDs.
"""

import inspect
import logging
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class ErrorLogger:
    """Centralized error logging with full context.

    Captures all errors with stack traces, context variables,
    and correlation IDs. Provides querying and aggregation.
    """

    def __init__(
        self,
        log_directory: str = "data/logs",
        notification_manager=None,
        enable_notifications: bool = True
    ):
        """Initialize error logger.

        Args:
            log_directory: Directory for error log files
            notification_manager: Optional notification manager
            enable_notifications: Whether to send notifications
        """
        self._log_directory = log_directory
        self._notification_manager = notification_manager
        self._enable_notifications = enable_notifications
        self._logger = logging.getLogger(__name__)

        # Error aggregation
        self._error_counts = {}  # {(error_type, location): count}
        self._error_history = []  # Last 1000 errors

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        severity: str = "ERROR",
        correlation_id: Optional[str] = None
    ) -> str:
        """Log error with full context.

        Args:
            error: Exception object
            context: Additional context variables
            severity: ERROR, WARNING, CRITICAL
            correlation_id: Optional correlation ID

        Returns:
            Error ID for tracking

        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     error_logger.log_error(e, context={"var": value})
        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = self._generate_correlation_id()

        # Get stack trace
        stack_trace = traceback.format_exc()

        # Get error info
        error_type = type(error).__name__
        error_message = str(error)

        # Get frame info (caller's location)
        frame = inspect.currentframe()
        caller_frame = frame.f_back

        file_name = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        function_name = caller_frame.f_code.co_name

        # Get local variables (optional, can be large)
        local_vars = dict(caller_frame.f_locals)

        # Create error record
        error_record = {
            "timestamp": self._get_current_time().isoformat(),
            "error_id": "ERR_{}".format(correlation_id[-12:]),
            "correlation_id": correlation_id,
            "severity": severity,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "file_name": file_name,
            "line_number": line_number,
            "function_name": function_name,
            "context": context or {},
            "local_vars": self._sanitize_local_vars(local_vars)
        }

        # Write to log file
        self._write_to_log_file(error_record)

        # Aggregate error count
        self._aggregate_error(error_record)

        # Add to history
        self._error_history.append(error_record)
        if len(self._error_history) > 1000:
            self._error_history.pop(0)

        # Send notification for critical errors
        if severity == "CRITICAL" and self._enable_notifications:
            self._send_critical_notification(error_record)

        return error_record["error_id"]

    def _write_to_log_file(self, error_record: Dict) -> None:
        """Write error to dedicated log file.

        Args:
            error_record: Error record dictionary
        """
        # Create log directory
        Path(self._log_directory).mkdir(parents=True, exist_ok=True)

        # Generate log filename
        log_date = self._get_current_time().strftime("%Y-%m-%d")
        log_filename = "errors_{}.log".format(log_date)
        log_path = Path(self._log_directory) / log_filename

        # Write error record
        with open(log_path, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write("Error ID: {}\n".format(error_record["error_id"]))
            f.write("Timestamp: {}\n".format(error_record["timestamp"]))
            f.write("Severity: {}\n".format(error_record["severity"]))
            f.write("Correlation ID: {}\n".format(
                error_record["correlation_id"]
            ))
            f.write("Error Type: {}\n".format(error_record["error_type"]))
            f.write("Error Message: {}\n".format(
                error_record["error_message"]
            ))
            f.write("Location: {}:{} in {}\n".format(
                error_record["file_name"],
                error_record["line_number"],
                error_record["function_name"]
            ))
            f.write("Context: {}\n".format(error_record["context"]))
            f.write("Stack Trace:\n{}\n".format(
                error_record["stack_trace"]
            ))

            # Optionally include local variables
            if error_record["local_vars"]:
                f.write("Local Variables:\n")
                for key, value in error_record["local_vars"].items():
                    f.write("  {}: {}\n".format(key, repr(value)[:100]))

            f.write("=" * 80 + "\n\n")

        self._logger.info("Error logged: {}".format(error_record["error_id"]))

    def _aggregate_error(self, error_record: Dict) -> None:
        """Aggregate error counts by type and location.

        Args:
            error_record: Error record dictionary
        """
        key = (
            error_record["error_type"],
            error_record["file_name"],
            error_record["function_name"]
        )

        self._error_counts[key] = self._error_counts.get(key, 0) + 1

    def _send_critical_notification(self, error_record: Dict) -> None:
        """Send notification for critical errors.

        Args:
            error_record: Error record dictionary
        """
        if not self._notification_manager:
            return

        self._notification_manager.send_notification(
            severity="CRITICAL",
            title="Critical Error: {}".format(
                error_record["error_type"]
            ),
            message="{} in {}:{} - {}".format(
                error_record["error_type"],
                error_record["file_name"],
                error_record["line_number"],
                error_record["error_message"]
            ),
            notification_type="CRITICAL_ERROR"
        )

    def query_errors(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        error_type: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> List[Dict]:
        """Query errors by date range, type, or correlation ID.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            error_type: Filter by error type
            correlation_id: Filter by correlation ID

        Returns:
            List of matching error records
        """
        results = []

        # For now, search in-memory history
        # In production, would scan log files
        for error_record in self._error_history:
            # Filter by date range
            if start_date or end_date:
                record_date = error_record["timestamp"][:10]
                if start_date and record_date < start_date:
                    continue
                if end_date and record_date > end_date:
                    continue

            # Filter by error type
            if error_type and error_record["error_type"] != error_type:
                continue

            # Filter by correlation ID
            if correlation_id and error_record["correlation_id"] != correlation_id:
                continue

            results.append(error_record)

        return results

    def get_error_summary(self, days: int = 7) -> Dict:
        """Get error summary statistics.

        Args:
            days: Number of days to summarize

        Returns:
            Summary statistics dictionary
        """
        # Calculate total errors
        total_errors = sum(self._error_counts.values())

        # Get top 5 error types
        sorted_errors = sorted(
            self._error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_5_errors = [
            {
                "error_type": key[0],
                "location": "{}:{}".format(key[1], key[2]),
                "count": count
            }
            for key, count in sorted_errors[:5]
        ]

        return {
            "total_errors": total_errors,
            "top_5_errors": top_5_errors,
            "unique_error_types": len(self._error_counts)
        }

    def _sanitize_local_vars(self, local_vars: Dict) -> Dict:
        """Sanitize local variables for logging.

        Removes sensitive data and limits size.

        Args:
            local_vars: Local variables dictionary

        Returns:
            Sanitized dictionary
        """
        sanitized = {}

        for key, value in local_vars.items():
            # Skip sensitive keys
            if key in ["password", "api_key", "secret", "token"]:
                sanitized[key] = "***REDACTED***"
            else:
                # Limit representation size
                repr_value = repr(value)
                if len(repr_value) > 200:
                    repr_value = repr_value[:200] + "..."
                sanitized[key] = repr_value

        return sanitized

    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for error tracking.

        Returns:
            Correlation ID string
        """
        return "evt_{}".format(uuid.uuid4().hex)

    def _get_current_time(self) -> datetime:
        """Get current time in UTC.

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)


def log_errors(
    error_logger: ErrorLogger,
    severity: str = "ERROR",
    reraise: bool = True
):
    """Decorator to automatically log errors from functions.

    Args:
        error_logger: ErrorLogger instance
        severity: Error severity level
        reraise: Whether to re-raise after logging

    Example:
        >>> @log_errors(error_logger, severity="ERROR")
        >>> def risky_function(x, y):
        ...     # If exception occurs, it's logged automatically
        ...     return x / y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error
                error_logger.log_error(
                    e,
                    context={
                        "function": func.__name__,
                        "args": str(args)[:100],
                        "kwargs": str(kwargs)[:100]
                    },
                    severity=severity
                )

                # Re-raise if requested
                if reraise:
                    raise

        return wrapper
    return decorator


def setup_global_exception_handler(error_logger: ErrorLogger) -> None:
    """Setup global exception handler for uncaught exceptions.

    Args:
        error_logger: ErrorLogger instance

    Example:
        >>> setup_global_exception_handler(error_logger)
        >>> # Now all uncaught exceptions are logged
    """
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exception."""
        # Only handle actual exceptions, not KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Create exception from traceback
        error = exc_value.with_traceback(exc_traceback)

        # Log with CRITICAL severity
        error_logger.log_error(
            error,
            severity="CRITICAL",
            context={"uncaught": True}
        )

        # Call original handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Install handler
    sys.excepthook = handle_exception
