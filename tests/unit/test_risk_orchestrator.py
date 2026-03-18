"""Unit tests for Risk Orchestrator.

Tests integration of all 8 risk management layers using mocks for
components that require external dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock
import pytest

from src.risk.risk_orchestrator import (
    RiskOrchestrator,
    TradingSignal
)
from src.risk.emergency_stop import EmergencyStop
from src.risk.daily_loss_tracker import DailyLossTracker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_size_tracker import PositionSizeTracker
from src.risk.news_event_filter import NewsEventFilter
from src.risk.per_trade_risk_limit import PerTradeRiskLimit
from src.risk.notification_manager import NotificationManager


class TestRiskOrchestratorInit:
    """Test RiskOrchestrator initialization."""

    @pytest.fixture
    def mock_components(self):
        """Create mock risk components."""
        return {
            'emergency_stop': Mock(spec=EmergencyStop),
            'daily_loss_tracker': Mock(spec=DailyLossTracker),
            'drawdown_tracker': Mock(spec=DrawdownTracker),
            'position_size_tracker': Mock(spec=PositionSizeTracker),
            'circuit_breaker_detector': Mock(),
            'news_event_filter': Mock(spec=NewsEventFilter),
            'per_trade_risk_limit': Mock(spec=PerTradeRiskLimit),
            'notification_manager': Mock(spec=NotificationManager)
        }

    def test_init_with_all_components(self, mock_components):
        """Verify orchestrator initializes with all components."""
        orchestrator = RiskOrchestrator(**mock_components)

        assert orchestrator._emergency_stop is not None
        assert orchestrator._daily_loss_tracker is not None
        assert orchestrator._drawdown_tracker is not None
        assert orchestrator._position_size_tracker is not None
        assert orchestrator._circuit_breaker_detector is not None
        assert orchestrator._news_event_filter is not None
        assert orchestrator._per_trade_risk_limit is not None
        assert orchestrator._notification_manager is not None

    def test_init_with_audit_trail(self, mock_components):
        """Verify orchestrator initializes with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "orchestrator.csv")

        orchestrator = RiskOrchestrator(
            **mock_components,
            audit_trail_path=audit_path
        )

        assert orchestrator._audit_trail_path == audit_path


class TestValidateTradeWithMocks:
    """Test trade validation using mocks."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with all mocks."""
        mock_stop = Mock(spec=EmergencyStop)
        mock_stop.is_trading_allowed.return_value = True

        mock_daily_loss = Mock(spec=DailyLossTracker)
        mock_daily_loss.get_daily_pnl.return_value = 0
        mock_daily_loss._daily_loss_limit = 1000

        mock_drawdown = Mock(spec=DrawdownTracker)
        mock_drawdown.get_drawdown_percentage.return_value = 0.05
        mock_drawdown._max_drawdown_percentage = 0.10

        mock_position = Mock(spec=PositionSizeTracker)
        mock_position.get_position_count.return_value = 0
        mock_position._max_position_size = 20

        mock_circuit = Mock()
        mock_circuit.get_status.return_value = {
            'is_halted': False,
            'halt_level': None
        }

        mock_news = Mock(spec=NewsEventFilter)
        mock_news.get_blackout_status.return_value = {
            'is_blackout': False,
            'event_name': None
        }

        mock_risk = Mock(spec=PerTradeRiskLimit)
        mock_risk.validate_trade.return_value = {
            'is_valid': True,
            'estimated_risk': 200
        }
        mock_risk._max_risk_dollars = 500

        mock_notify = Mock(spec=NotificationManager)
        mock_notify.is_notification_enabled.return_value = False

        return RiskOrchestrator(
            emergency_stop=mock_stop,
            daily_loss_tracker=mock_daily_loss,
            drawdown_tracker=mock_drawdown,
            position_size_tracker=mock_position,
            circuit_breaker_detector=mock_circuit,
            news_event_filter=mock_news,
            per_trade_risk_limit=mock_risk,
            notification_manager=mock_notify
        )

    @pytest.fixture
    def signal(self):
        """Create test signal."""
        return TradingSignal(
            signal_id="SIG001",
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

    def test_validate_trade_all_checks_pass(self, orchestrator, signal):
        """Verify validation passes when all checks pass."""
        result = orchestrator.validate_trade(signal)

        assert result['is_valid'] is True
        assert result['block_reason'] is None
        assert len(result['checks_failed']) == 0
        assert len(result['checks_passed']) >= 6

    def test_validate_trade_emergency_stop_blocks(self, orchestrator, signal):
        """Verify emergency stop blocks trade."""
        orchestrator._emergency_stop.is_trading_allowed.return_value = False
        orchestrator._emergency_stop.get_status.return_value = {
            'is_stopped': True,
            'stop_reason': 'Manual intervention',
            'stop_time': '2026-03-17T14:30:00Z',
            'time_stopped_seconds': 60
        }

        result = orchestrator.validate_trade(signal)

        assert result['is_valid'] is False
        assert 'emergency_stop' in result['checks_failed']
        assert 'Emergency stop' in result['block_reason']

    def test_validate_trade_returns_detailed_results(self, orchestrator, signal):
        """Verify validation returns detailed results."""
        result = orchestrator.validate_trade(signal)

        assert 'validation_details' in result
        assert 'emergency_stop' in result['validation_details']
        assert 'daily_loss' in result['validation_details']
        assert 'drawdown' in result['validation_details']

    def test_notification_sent_on_rejection(self, orchestrator, signal):
        """Verify notification sent when trade rejected."""
        # Setup emergency stop to block
        orchestrator._emergency_stop.is_trading_allowed.return_value = False
        orchestrator._emergency_stop.get_status.return_value = {
            'is_stopped': True,
            'stop_reason': 'Testing',
            'stop_time': '2026-03-17T14:30:00Z',
            'time_stopped_seconds': 60
        }
        orchestrator._notification_manager.is_notification_enabled.return_value = True
        orchestrator._notification_manager.send_notification.return_value = True

        orchestrator.validate_trade(signal)

        # Verify notification was sent
        assert orchestrator._notification_manager.send_notification.called


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "orchestrator.csv")

        # Create all mocks
        mock_stop = Mock(spec=EmergencyStop)
        mock_stop.is_trading_allowed.return_value = True

        mock_daily_loss = Mock(spec=DailyLossTracker)
        mock_daily_loss.get_daily_pnl.return_value = 0
        mock_daily_loss._daily_loss_limit = 1000

        mock_drawdown = Mock(spec=DrawdownTracker)
        mock_drawdown.get_drawdown_percentage.return_value = 0.05
        mock_drawdown._max_drawdown_percentage = 0.10

        mock_position = Mock(spec=PositionSizeTracker)
        mock_position.get_position_count.return_value = 0
        mock_position._max_position_size = 20

        mock_circuit = Mock()
        mock_circuit.get_status.return_value = {
            'is_halted': False,
            'halt_level': None
        }

        mock_news = Mock(spec=NewsEventFilter)
        mock_news.get_blackout_status.return_value = {
            'is_blackout': False,
            'event_name': None
        }

        mock_risk = Mock(spec=PerTradeRiskLimit)
        mock_risk.validate_trade.return_value = {'is_valid': True}
        mock_risk._max_risk_dollars = 500

        mock_notify = Mock(spec=NotificationManager)
        mock_notify.is_notification_enabled.return_value = False

        return RiskOrchestrator(
            emergency_stop=mock_stop,
            daily_loss_tracker=mock_daily_loss,
            drawdown_tracker=mock_drawdown,
            position_size_tracker=mock_position,
            circuit_breaker_detector=mock_circuit,
            news_event_filter=mock_news,
            per_trade_risk_limit=mock_risk,
            notification_manager=mock_notify,
            audit_trail_path=audit_path
        )

    @pytest.fixture
    def signal(self):
        """Create test signal."""
        return TradingSignal(
            signal_id="SIG001",
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

    def test_csv_file_created_on_validation(self, orchestrator, signal):
        """Verify CSV file created on validation."""
        orchestrator.validate_trade(signal)

        assert Path(orchestrator._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, orchestrator, signal):
        """Verify CSV has all required columns."""
        orchestrator.validate_trade(signal)

        import csv
        with open(orchestrator._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "signal_id",
            "is_valid",
            "block_reason",
            "checks_passed",
            "checks_failed",
            "emergency_stop_status",
            "daily_loss_status",
            "drawdown_status",
            "position_size_status",
            "circuit_breaker_status",
            "news_event_status",
            "per_trade_risk_status"
        ]

        assert headers == expected_headers

    def test_csv_logs_valid_trade(self, orchestrator, signal):
        """Verify CSV logs valid trade correctly."""
        orchestrator.validate_trade(signal)

        import csv
        with open(orchestrator._audit_trail_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["signal_id"] == "SIG001"
        assert rows[0]["is_valid"] == "True"


class TestValidationDetails:
    """Test validation details structure."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocks."""
        mock_stop = Mock(spec=EmergencyStop)
        mock_stop.is_trading_allowed.return_value = True

        mock_daily_loss = Mock(spec=DailyLossTracker)
        mock_daily_loss.get_daily_pnl.return_value = 0
        mock_daily_loss._daily_loss_limit = 1000

        mock_drawdown = Mock(spec=DrawdownTracker)
        mock_drawdown.get_drawdown_percentage.return_value = 0.05
        mock_drawdown._max_drawdown_percentage = 0.10

        mock_position = Mock(spec=PositionSizeTracker)
        mock_position.get_position_count.return_value = 0
        mock_position._max_position_size = 20

        mock_circuit = Mock()
        mock_circuit.get_status.return_value = {
            'is_halted': False,
            'halt_level': None
        }

        mock_news = Mock(spec=NewsEventFilter)
        mock_news.get_blackout_status.return_value = {
            'is_blackout': False,
            'event_name': None
        }

        mock_risk = Mock(spec=PerTradeRiskLimit)
        mock_risk.validate_trade.return_value = {
            'is_valid': True,
            'estimated_risk': 200
        }
        mock_risk._max_risk_dollars = 500

        mock_notify = Mock(spec=NotificationManager)
        mock_notify.is_notification_enabled.return_value = False

        return RiskOrchestrator(
            emergency_stop=mock_stop,
            daily_loss_tracker=mock_daily_loss,
            drawdown_tracker=mock_drawdown,
            position_size_tracker=mock_position,
            circuit_breaker_detector=mock_circuit,
            news_event_filter=mock_news,
            per_trade_risk_limit=mock_risk,
            notification_manager=mock_notify
        )

    @pytest.fixture
    def signal(self):
        """Create test signal."""
        return TradingSignal(
            signal_id="SIG001",
            entry_price=11750,
            stop_loss_price=11730,
            quantity=5
        )

    def test_validation_details_include_all_checks(self, orchestrator, signal):
        """Verify validation details include all check results."""
        result = orchestrator.validate_trade(signal)

        details = result['validation_details']

        # Should have all 7 check results
        assert len(details) == 7
        assert 'emergency_stop' in details
        assert 'daily_loss' in details
        assert 'drawdown' in details
        assert 'position_size' in details
        assert 'circuit_breaker' in details
        assert 'news_events' in details
        assert 'per_trade_risk' in details

    def test_each_detail_has_passed_and_status(self, orchestrator, signal):
        """Verify each detail has required fields."""
        result = orchestrator.validate_trade(signal)

        for check_name, detail in result['validation_details'].items():
            assert 'passed' in detail
            assert 'status' in detail
            assert isinstance(detail['passed'], bool)
