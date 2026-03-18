"""Unit tests for Circuit Breaker Detection.

Tests circuit breaker status checking, halt detection, level tracking,
CSV logging, and integration with trade execution pipeline.
"""

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.risk.circuit_breaker_detector import CircuitBreakerDetector


class TestCircuitBreakerDetectorInit:
    """Test CircuitBreakerDetector initialization."""

    def test_init_with_api_client(self):
        """Verify detector initializes with API client."""
        api_client = Mock()
        detector = CircuitBreakerDetector(api_client=api_client)

        assert detector._api_client == api_client
        assert detector._halted is False
        assert detector._halt_level is None
        assert detector._halt_reason is None
        assert detector._halt_start_time is None
        assert detector._estimated_resume_time is None


class TestCheckCircuitBreakerStatus:
    """Test circuit breaker status checking."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_status_check_returns_correct_data(self, detector):
        """Verify status check returns correct data structure."""
        # Mock exchange API response
        detector._check_exchange_status = Mock(return_value={
            "halted": False,
            "level": None,
            "reason": None,
            "start_time": None,
            "estimated_resume": None
        })

        status = detector.check_circuit_breaker_status()

        assert status["is_halted"] is False
        assert status["halt_level"] is None
        assert status["reason"] is None
        assert status["start_time"] is None
        assert status["estimated_resume"] is None

    def test_status_check_detects_halt(self, detector):
        """Verify status check detects active halt."""
        # Mock exchange API response with halt
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._check_exchange_status = Mock(return_value={
            "halted": True,
            "level": 2,
            "reason": "Level 2 circuit breaker",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time before resume time
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = datetime(
                2026, 3, 17, 14, 5, 0, tzinfo=timezone.utc)

            status = detector.check_circuit_breaker_status()

            assert status["is_halted"] is True
            assert status["halt_level"] == 2
            assert status["reason"] == "Level 2 circuit breaker"
            assert status["start_time"] is not None
            assert status["estimated_resume"] is not None

    def test_status_check_updates_internal_state(self, detector):
        """Verify status check updates internal halt state."""
        # Mock exchange API response with halt
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._check_exchange_status = Mock(return_value={
            "halted": True,
            "level": 1,
            "reason": "Level 1 circuit breaker",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time before resume time
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = datetime(
                2026, 3, 17, 14, 5, 0, tzinfo=timezone.utc)

            detector.check_circuit_breaker_status()

            assert detector.is_trading_halted() is True
            assert detector.get_halt_level() == 1


class TestIsTradingHalted:
    """Test trading halt status checking."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_not_halted_initially(self, detector):
        """Verify not halted initially."""
        assert detector.is_trading_halted() is False

    def test_halted_after_breaker_detected(self, detector):
        """Verify halted after breaker detected."""
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 2,
            "reason": "Level 2 circuit breaker",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        assert detector.is_trading_halted() is True

    def test_not_halted_after_breaker_expires(self, detector):
        """Verify not halted after breaker expires."""
        # Set halt in past
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1 circuit breaker",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time past resume time
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = datetime(
                2026, 3, 17, 14, 20, 0, tzinfo=timezone.utc)
            detector.check_circuit_breaker_status()

            # Should auto-expire
            assert detector.is_trading_halted() is False


class TestGetHaltLevel:
    """Test halt level retrieval."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_halt_level_none_initially(self, detector):
        """Verify halt level is None initially."""
        assert detector.get_halt_level() is None

    def test_halt_level_1(self, detector):
        """Verify halt level 1 detected."""
        start_time = datetime.now(timezone.utc)
        resume_time = datetime.now(timezone.utc) + timedelta(minutes=15)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        assert detector.get_halt_level() == 1

    def test_halt_level_2(self, detector):
        """Verify halt level 2 detected."""
        start_time = datetime.now(timezone.utc)
        resume_time = datetime.now(timezone.utc) + timedelta(minutes=15)

        detector._update_halt_status({
            "halted": True,
            "level": 2,
            "reason": "Level 2",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        assert detector.get_halt_level() == 2

    def test_halt_level_3(self, detector):
        """Verify halt level 3 detected."""
        start_time = datetime.now(timezone.utc)
        resume_time = datetime.now(timezone.utc) + timedelta(hours=4)

        detector._update_halt_status({
            "halted": True,
            "level": 3,
            "reason": "Level 3",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        assert detector.get_halt_level() == 3


class TestGetEstimatedResumeTime:
    """Test estimated resume time retrieval."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_resume_time_none_initially(self, detector):
        """Verify resume time is None initially."""
        assert detector.get_estimated_resume_time() is None

    def test_resume_time_after_halt(self, detector):
        """Verify resume time set after halt."""
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        assert detector.get_estimated_resume_time() == resume_time


class TestGetHaltDurationRemaining:
    """Test remaining halt duration calculation."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_remaining_time_none_when_not_halted(self, detector):
        """Verify remaining time is None when not halted."""
        assert detector.get_halt_duration_remaining() is None

    def test_remaining_time_calculated(self, detector):
        """Verify remaining time calculated correctly."""
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time 5 minutes after start
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = datetime(
                2026, 3, 17, 14, 5, 0, tzinfo=timezone.utc)

            remaining = detector.get_halt_duration_remaining()
            assert remaining == timedelta(minutes=10)

    def test_remaining_time_zero_at_expiry(self, detector):
        """Verify remaining time zero at expiry."""
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time at resume time
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = resume_time

            remaining = detector.get_halt_duration_remaining()
            assert remaining == timedelta(seconds=0)


class TestCanResumeTrading:
    """Test trading resumption check."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_can_resume_when_not_halted(self, detector):
        """Verify can resume when not halted."""
        assert detector.can_resume_trading() is True

    def test_cannot_resume_during_halt(self, detector):
        """Verify cannot resume during halt."""
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time before resume time
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = datetime(
                2026, 3, 17, 14, 5, 0, tzinfo=timezone.utc)

            assert detector.can_resume_trading() is False

    def test_can_resume_after_expiry(self, detector):
        """Verify can resume after halt expires."""
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Mock current time past resume time
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = datetime(
                2026, 3, 17, 14, 20, 0, tzinfo=timezone.utc)

            assert detector.can_resume_trading() is True


class TestDetectFromPriceMovement:
    """Test backup detection from price movement."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_no_breaker_normal_move(self, detector):
        """Verify no breaker detected for normal move."""
        result = detector._detect_from_price_movement(
            current_price=11750.0,
            previous_close=11700.0
        )

        assert result is None

    def test_detect_level_1_breaker(self, detector):
        """Verify Level 1 breaker detected at 7% decline."""
        previous_close = 11750.0
        current_price = previous_close * 0.93  # 7% decline

        result = detector._detect_from_price_movement(
            current_price=current_price,
            previous_close=previous_close
        )

        assert result is not None
        assert result["level"] == 1
        assert result["halted"] is True

    def test_detect_level_2_breaker(self, detector):
        """Verify Level 2 breaker detected at 13% decline."""
        previous_close = 11750.0
        current_price = previous_close * 0.87  # 13% decline

        result = detector._detect_from_price_movement(
            current_price=current_price,
            previous_close=previous_close
        )

        assert result is not None
        assert result["level"] == 2
        assert result["halted"] is True

    def test_detect_level_3_breaker(self, detector):
        """Verify Level 3 breaker detected at 20% decline."""
        previous_close = 11750.0
        current_price = previous_close * 0.80  # 20% decline

        result = detector._detect_from_price_movement(
            current_price=current_price,
            previous_close=previous_close
        )

        assert result is not None
        assert result["level"] == 3
        assert result["halted"] is True

    def test_level_3_takes_precedence(self, detector):
        """Verify Level 3 takes precedence (20% > 13% > 7%)."""
        previous_close = 11750.0
        current_price = previous_close * 0.75  # 25% decline

        result = detector._detect_from_price_movement(
            current_price=current_price,
            previous_close=previous_close
        )

        assert result is not None
        assert result["level"] == 3  # Should trigger Level 3


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "circuit_breaker_events.csv")

        api_client = Mock()
        return CircuitBreakerDetector(
            api_client=api_client,
            audit_trail_path=audit_path
        )

    def test_csv_file_created(self, detector):
        """Verify CSV file created on first event."""
        detector._log_audit_event("CHECK", None, False)

        assert Path(detector._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, detector):
        """Verify CSV has all required columns."""
        detector._log_audit_event("CHECK", None, False)

        import csv
        with open(detector._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "halt_level",
            "is_halted",
            "halt_reason",
            "halt_start_time",
            "estimated_resume_time",
            "time_remaining"
        ]

        assert headers == expected_headers


class TestHaltStatusUpdates:
    """Test halt status updates."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_update_sets_halted_flag(self, detector):
        """Verify update sets halted flag."""
        start_time = datetime.now(timezone.utc)
        resume_time = datetime.now(timezone.utc) + timedelta(minutes=15)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        assert detector._halted is True

    def test_update_clears_halted_flag(self, detector):
        """Verify update clears halted flag."""
        # First set halted
        start_time = datetime.now(timezone.utc)
        resume_time = datetime.now(timezone.utc) + timedelta(minutes=15)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time,
            "estimated_resume": resume_time
        })

        # Then clear
        detector._update_halt_status({
            "halted": False,
            "level": None,
            "reason": None,
            "start_time": None,
            "estimated_resume": None
        })

        assert detector._halted is False
        assert detector._halt_level is None


class TestMultipleHaltsInSameDay:
    """Test multiple halts in same day."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_multiple_halts_tracked(self, detector):
        """Verify multiple halts tracked in same day."""
        # First halt
        start_time_1 = datetime(2026, 3, 17, 10, 0, 0,   tzinfo=timezone.utc)
        resume_time_1 = datetime(2026, 3, 17, 10, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 1,
            "reason": "Level 1",
            "start_time": start_time_1,
            "estimated_resume": resume_time_1
        })

        assert detector.get_halt_level() == 1

        # Clear first halt
        detector._update_halt_status({
            "halted": False,
            "level": None,
            "reason": None,
            "start_time": None,
            "estimated_resume": None
        })

        # Second halt
        start_time_2 = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)
        resume_time_2 = datetime(2026, 3, 17, 14, 15, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 2,
            "reason": "Level 2",
            "start_time": start_time_2,
            "estimated_resume": resume_time_2
        })

        assert detector.get_halt_level() == 2


class TestLevel3RestOfDayHalt:
    """Test Level 3 rest of day halt."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_level_3_long_duration(self, detector):
        """Verify Level 3 has long duration (rest of day)."""
        end_of_day = datetime(2026, 3, 17, 20, 0, 0,   tzinfo=timezone.utc)
        start_time = datetime(2026, 3, 17, 14, 0, 0,   tzinfo=timezone.utc)

        detector._update_halt_status({
            "halted": True,
            "level": 3,
            "reason": "Level 3 circuit breaker",
            "start_time": start_time,
            "estimated_resume": end_of_day
        })

        # Mock current time at start of halt
        with patch('src.risk.circuit_breaker_detector._get_current_time') as mock_time:
            mock_time.return_value = start_time

            # Should have long duration
            remaining = detector.get_halt_duration_remaining()
            assert remaining is not None
            assert remaining.total_seconds() > 3600  # More than 1 hour


class TestGracefulAPIFailureHandling:
    """Test graceful handling of API failures."""

    @pytest.fixture
    def detector(self):
        """Create circuit breaker detector."""
        api_client = Mock()
        return CircuitBreakerDetector(api_client=api_client)

    def test_api_failure_returns_default_status(self, detector):
        """Verify API failure returns safe default status."""
        # Mock API failure
        detector._check_exchange_status = Mock(side_effect=Exception("API error"))

        # Should not raise exception, return safe default
        status = detector.check_circuit_breaker_status()

        # Should default to not halted (safe to trade)
        assert status["is_halted"] is False
        assert status["halt_level"] is None

    def test_api_failure_logs_error(self, detector):
        """Verify API failure logs error."""
        # Mock API failure
        detector._check_exchange_status = Mock(side_effect=Exception("API error"))

        # Should log error but not crash
        with patch('src.risk.circuit_breaker_detector.logger') as mock_logger:
            detector.check_circuit_breaker_status()

            # Verify error logged
            mock_logger.error.assert_called()
