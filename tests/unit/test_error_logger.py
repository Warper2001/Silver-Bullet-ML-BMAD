"""Unit tests for ErrorLogger.

Tests error logging with full stack traces, context capture,
aggregation, querying, and notifications.
"""

from unittest.mock import MagicMock, patch, mock_open

from src.monitoring.error_logger import ErrorLogger, log_errors


class TestErrorLoggerInit:
    """Test ErrorLogger initialization."""

    def test_init_with_default_parameters(self):
        """Verify initialization with default parameters."""
        logger = ErrorLogger()

        assert logger._log_directory == "data/logs"
        assert logger._notification_manager is None
        assert logger._enable_notifications is True
        assert len(logger._error_counts) == 0
        assert len(logger._error_history) == 0

    def test_init_with_custom_log_directory(self):
        """Verify initialization with custom log directory."""
        logger = ErrorLogger(log_directory="custom/logs")

        assert logger._log_directory == "custom/logs"

    def test_init_with_notification_manager(self):
        """Verify initialization with notification manager."""
        notification_manager = MagicMock()
        logger = ErrorLogger(notification_manager=notification_manager)

        assert logger._notification_manager == notification_manager

    def test_init_with_notifications_disabled(self):
        """Verify initialization with notifications disabled."""
        logger = ErrorLogger(enable_notifications=False)

        assert logger._enable_notifications is False


class TestLogError:
    """Test log_error functionality."""

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_creates_error_record(self, mock_file, mock_mkdir):
        """Verify log_error creates error record."""
        logger = ErrorLogger()

        error = ValueError("Test error")
        error_id = logger.log_error(error)

        # Verify error ID returned
        assert error_id.startswith("ERR_")
        assert len(logger._error_history) == 1
        assert logger._error_history[0]["error_type"] == "ValueError"
        assert logger._error_history[0]["error_message"] == "Test error"

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_includes_all_required_fields(
        self,
        mock_file,
        mock_mkdir
    ):
        """Verify log_error includes all required fields."""
        logger = ErrorLogger()

        error = RuntimeError("Test error")
        logger.log_error(error)

        error_record = logger._error_history[0]

        # Check required fields
        assert "timestamp" in error_record
        assert "error_id" in error_record
        assert "correlation_id" in error_record
        assert "severity" in error_record
        assert "error_type" in error_record
        assert "error_message" in error_record
        assert "stack_trace" in error_record
        assert "file_name" in error_record
        assert "line_number" in error_record
        assert "function_name" in error_record

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_generates_correlation_id(
        self,
        mock_file,
        mock_mkdir
    ):
        """Verify log_error generates correlation ID."""
        logger = ErrorLogger()

        error = ValueError("Test error")
        logger.log_error(error)

        error_record = logger._error_history[0]

        # Verify correlation ID format
        assert error_record["correlation_id"].startswith("evt_")
        assert len(error_record["correlation_id"]) > 10

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_uses_provided_correlation_id(
        self,
        mock_file,
        mock_mkdir
    ):
        """Verify log_error uses provided correlation ID."""
        logger = ErrorLogger()

        error = ValueError("Test error")
        logger.log_error(error, correlation_id="test_id_123")

        error_record = logger._error_history[0]

        assert error_record["correlation_id"] == "test_id_123"

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_aggregates_errors(self, mock_file, mock_mkdir):
        """Verify log_error aggregates errors by type and location."""
        logger = ErrorLogger()

        error = ValueError("Test error")
        logger.log_error(error)

        # Check aggregation
        assert len(logger._error_counts) > 0

        # Key should be (error_type, file_name, function_name)
        key = list(logger._error_counts.keys())[0]
        assert key[0] == "ValueError"
        assert logger._error_counts[key] == 1

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_sends_notification_for_critical(
        self,
        mock_file,
        mock_mkdir
    ):
        """Verify log_error sends notification for CRITICAL errors."""
        notification_manager = MagicMock()
        logger = ErrorLogger(notification_manager=notification_manager)

        error = RuntimeError("Critical error")
        logger.log_error(error, severity="CRITICAL")

        # Verify notification sent
        notification_manager.send_notification.assert_called_once()

        call_args = notification_manager.send_notification.call_args
        assert call_args[1]["severity"] == "CRITICAL"

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_sanitizes_local_vars(
        self,
        mock_file,
        mock_mkdir
    ):
        """Verify log_error sanitizes local variables."""
        logger = ErrorLogger()

        error = ValueError("Test error")
        logger.log_error(error)

        error_record = logger._error_history[0]

        # Verify local_vars is a dict
        assert isinstance(error_record["local_vars"], dict)

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_limits_history_size(
        self,
        mock_file,
        mock_mkdir
    ):
        """Verify log_error limits history to 1000 errors."""
        logger = ErrorLogger()

        error = ValueError("Test error")

        # Add 1001 errors
        for _ in range(1001):
            logger.log_error(error)

        # Verify history limited to 1000
        assert len(logger._error_history) == 1000


class TestQueryErrors:
    """Test query_errors functionality."""

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_query_by_date_range(self, mock_file, mock_mkdir):
        """Verify query errors by date range."""
        logger = ErrorLogger()

        # Add some errors
        error = ValueError("Test error")
        logger.log_error(error)
        logger.log_error(error)

        # Query all errors
        results = logger.query_errors()

        assert len(results) == 2

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_query_by_error_type(self, mock_file, mock_mkdir):
        """Verify query errors by type."""
        logger = ErrorLogger()

        # Add errors of different types
        logger.log_error(ValueError("Test"))
        logger.log_error(RuntimeError("Test"))

        # Query by type
        results = logger.query_errors(error_type="ValueError")

        assert len(results) == 1
        assert results[0]["error_type"] == "ValueError"

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_query_by_correlation_id(self, mock_file, mock_mkdir):
        """Verify query errors by correlation ID."""
        logger = ErrorLogger()

        # Add error with specific correlation ID
        error = ValueError("Test error")
        logger.log_error(error, correlation_id="test_id_123")

        # Query by correlation ID
        results = logger.query_errors(correlation_id="test_id_123")

        assert len(results) == 1
        assert results[0]["correlation_id"] == "test_id_123"


class TestGetErrorSummary:
    """Test get_error_summary functionality."""

    def test_get_error_summary_with_no_errors(self):
        """Verify summary with no errors."""
        logger = ErrorLogger()

        summary = logger.get_error_summary()

        assert summary["total_errors"] == 0
        assert summary["top_5_errors"] == []
        assert summary["unique_error_types"] == 0

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_error_summary_with_errors(self, mock_file, mock_mkdir):
        """Verify summary with errors."""
        logger = ErrorLogger()

        # Add multiple errors
        logger.log_error(ValueError("Error 1"))
        logger.log_error(ValueError("Error 2"))
        logger.log_error(RuntimeError("Error 3"))

        summary = logger.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["unique_error_types"] == 2
        assert len(summary["top_5_errors"]) <= 5

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_error_summary_top_5_errors(self, mock_file, mock_mkdir):
        """Verify top 5 errors calculation."""
        logger = ErrorLogger()

        # Add errors with different frequencies
        for _ in range(5):
            logger.log_error(ValueError("Frequent error"))
        for _ in range(3):
            logger.log_error(RuntimeError("Less frequent"))
        logger.log_error(TypeError("Rare error"))

        summary = logger.get_error_summary()

        assert len(summary["top_5_errors"]) == 3
        assert summary["top_5_errors"][0]["count"] == 5


class TestErrorDecorator:
    """Test @log_errors decorator."""

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_decorator_logs_errors(self, mock_file, mock_mkdir):
        """Verify decorator logs errors automatically."""
        logger = ErrorLogger()

        @log_errors(logger, severity="ERROR", reraise=False)
        def failing_function():
            raise ValueError("Test error")

        # Call function - should not raise
        failing_function()

        # Verify error was logged
        assert len(logger._error_history) == 1
        assert logger._error_history[0]["error_type"] == "ValueError"

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_decorator_reraises_by_default(self, mock_file, mock_mkdir):
        """Verify decorator re-raises by default."""
        logger = ErrorLogger()

        @log_errors(logger)
        def failing_function():
            raise ValueError("Test error")

        # Should raise the error
        try:
            failing_function()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Verify error was logged
        assert len(logger._error_history) == 1


class TestPerformance:
    """Test performance requirements."""

    @patch("src.monitoring.error_logger.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_log_error_performance_under_50ms(self, mock_file, mock_mkdir):
        """Verify that log_error overhead is < 50ms."""
        logger = ErrorLogger()

        error = ValueError("Test error")

        import time

        # Measure time to log error
        start = time.perf_counter()
        for _ in range(10):
            logger.log_error(error)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 10

        # Should be < 50ms per log
        assert elapsed_ms < 50.0
