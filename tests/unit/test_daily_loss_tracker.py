"""Unit tests for Daily Loss Limit Enforcement.

Tests daily loss tracking, limit enforcement, trading halt,
daily reset, CSV logging, and integration with pipeline.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import pytest

from src.risk.daily_loss_tracker import DailyLossTracker


class TestDailyLossTrackerInit:
    """Test DailyLossTracker initialization."""

    def test_init_with_valid_parameters(self):
        """Verify tracker initializes with valid parameters."""
        tracker = DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00,
            reset_time_utc="13:00"
        )

        assert tracker._daily_loss_limit == 1000.00
        assert tracker._account_balance == 50000.00
        assert tracker._daily_pnl == 0.00
        assert tracker._is_trading_halted is False

    def test_init_with_invalid_daily_loss_limit(self):
        """Verify tracker raises error with non-positive loss limit."""
        with pytest.raises(ValueError):
            DailyLossTracker(
                daily_loss_limit=-100.00,
                account_balance=50000.00
            )

    def test_init_with_invalid_account_balance(self):
        """Verify tracker raises error with non-positive balance."""
        with pytest.raises(ValueError):
            DailyLossTracker(
                daily_loss_limit=1000.00,
                account_balance=0.00
            )


class TestRecordTrade:
    """Test trade P&L recording."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00
        )

    def test_record_profitable_trade(self, tracker):
        """Verify recording profitable trade."""
        tracker.record_trade(pnl=150.00, order_id="ORDER-123")

        assert tracker.get_daily_pnl() == 150.00
        assert tracker.is_trading_halted() is False

    def test_record_loss_trade(self, tracker):
        """Verify recording loss trade."""
        tracker.record_trade(pnl=-150.00, order_id="ORDER-456")

        assert tracker.get_daily_pnl() == -150.00
        assert tracker.is_trading_halted() is False

    def test_multiple_trades_accumulate(self, tracker):
        """Verify multiple trades accumulate correctly."""
        tracker.record_trade(pnl=-150.00, order_id="ORDER-1")
        tracker.record_trade(pnl=-300.00, order_id="ORDER-2")
        tracker.record_trade(pnl=-200.00, order_id="ORDER-3")

        assert tracker.get_daily_pnl() == -650.00

    def test_profitable_trade_reduces_loss(self, tracker):
        """Verify profitable trade reduces accumulated loss."""
        tracker.record_trade(pnl=-500.00, order_id="ORDER-1")
        tracker.record_trade(pnl=200.00, order_id="ORDER-2")

        assert tracker.get_daily_pnl() == -300.00


class TestDailyLossLimitEnforcement:
    """Test daily loss limit enforcement."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00
        )

    def test_trading_halted_on_limit_breach(self, tracker):
        """Verify trading halted when loss limit breached."""
        tracker.record_trade(pnl=-1100.00, order_id="ORDER-123")

        assert tracker.is_trading_halted() is True
        assert tracker.is_trading_allowed() is False

    def test_trading_allowed_before_limit_breach(self, tracker):
        """Verify trading allowed before limit breached."""
        tracker.record_trade(pnl=-900.00, order_id="ORDER-123")

        assert tracker.is_trading_halted() is False
        assert tracker.is_trading_allowed() is True

    def test_trading_halted_at_exact_limit(self, tracker):
        """Verify trading halted at exact limit threshold."""
        tracker.record_trade(pnl=-1000.00, order_id="ORDER-123")

        assert tracker.is_trading_halted() is True
        assert tracker.is_trading_allowed() is False

    def test_halt_reason_set_correctly(self, tracker):
        """Verify halt reason set correctly."""
        tracker.record_trade(pnl=-1100.00, order_id="ORDER-123")

        reason = tracker.get_halt_reason()
        assert "Daily loss limit breached" in reason
        assert "1100.00" in reason


class TestGetRemainingLossAllowance:
    """Test remaining loss allowance calculation."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00
        )

    def test_remaining_allowance_at_start(self, tracker):
        """Verify full allowance available at start."""
        assert tracker.get_remaining_loss_allowance() == 1000.00

    def test_remaining_allowance_after_losses(self, tracker):
        """Verify allowance decreases after losses."""
        tracker.record_trade(pnl=-300.00, order_id="ORDER-123")

        assert tracker.get_remaining_loss_allowance() == 700.00

    def test_remaining_allowance_zero_after_breach(self, tracker):
        """Verify allowance zero after limit breach."""
        tracker.record_trade(pnl=-1000.00, order_id="ORDER-123")

        assert tracker.get_remaining_loss_allowance() == 0.00

    def test_remaining_allowance_increases_after_profit(self, tracker):
        """Verify allowance increases after profitable trade."""
        tracker.record_trade(pnl=-300.00, order_id="ORDER-1")
        tracker.record_trade(pnl=200.00, order_id="ORDER-2")

        assert tracker.get_remaining_loss_allowance() == 900.00


class TestGetDailySummary:
    """Test daily summary generation."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00
        )

    def test_daily_summary_at_start(self, tracker):
        """Verify summary at start of day."""
        summary = tracker.get_daily_summary()

        assert summary['daily_pnl'] == 0.00
        assert summary['loss_limit'] == 1000.00
        assert summary['remaining_allowance'] == 1000.00
        assert summary['trade_count'] == 0
        assert summary['is_halted'] is False

    def test_daily_summary_after_trades(self, tracker):
        """Verify summary after trades."""
        tracker.record_trade(pnl=-150.00, order_id="ORDER-1")
        tracker.record_trade(pnl=-300.00, order_id="ORDER-2")

        summary = tracker.get_daily_summary()

        assert summary['daily_pnl'] == -450.00
        assert summary['trade_count'] == 2
        assert summary['remaining_allowance'] == 550.00

    def test_daily_summary_after_halt(self, tracker):
        """Verify summary shows halt status."""
        tracker.record_trade(pnl=-1100.00, order_id="ORDER-123")

        summary = tracker.get_daily_summary()

        assert summary['is_halted'] is True
        assert summary['halt_reason'] is not None


class TestDailyReset:
    """Test daily reset functionality."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00,
            reset_time_utc="13:00"  # 8:00 CT
        )

    @patch('src.risk.daily_loss_tracker._get_current_time')
    def test_reset_at_reset_time(self, mock_time, tracker):
        """Verify tracker resets at reset time."""
        # Set current time to reset time (13:00 UTC)
        mock_time.return_value = datetime(
            2026, 3, 17, 13, 0, 0, tzinfo=timezone.utc
        )

        # Record some trades
        tracker.record_trade(pnl=-800.00, order_id="ORDER-1")
        assert tracker.get_daily_pnl() == -800.00

        # Advance to next day at reset time
        mock_time.return_value = datetime(
            2026, 3, 18, 13, 0, 0, tzinfo=timezone.utc
        )

        # Record new trade - should trigger reset
        tracker.record_trade(pnl=-100.00, order_id="ORDER-2")

        # Should be reset and only have new trade
        assert tracker.get_daily_pnl() == -100.00
        assert tracker.is_trading_halted() is False

    @patch('src.risk.daily_loss_tracker._get_current_time')
    def test_no_reset_before_reset_time(self, mock_time, tracker):
        """Verify tracker doesn't reset before reset time."""
        # Set current time before reset time (12:00 UTC)
        mock_time.return_value = datetime(
            2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc
        )

        # Record trade
        tracker.record_trade(pnl=-500.00, order_id="ORDER-1")

        # Still same day before reset time
        mock_time.return_value = datetime(
            2026, 3, 17, 12, 30, 0, tzinfo=timezone.utc
        )

        tracker.record_trade(pnl=-200.00, order_id="ORDER-2")

        # Should accumulate, not reset
        assert tracker.get_daily_pnl() == -700.00


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "daily_loss_tracking.csv")

        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00,
            audit_trail_path=audit_path
        )

    def test_csv_file_created(self, tracker):
        """Verify CSV file created on first trade."""
        tracker.record_trade(pnl=-150.00, order_id="ORDER-123")

        assert Path(tracker._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, tracker):
        """Verify CSV has all required columns."""
        tracker.record_trade(pnl=-150.00, order_id="ORDER-123")

        import csv
        with open(tracker._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "daily_pnl",
            "loss_limit",
            "remaining_allowance",
            "is_halted",
            "halt_reason",
            "order_id"
        ]

        assert headers == expected_headers


class TestIntegration:
    """Test integration with trading pipeline."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00
        )

    def test_signal_rejected_when_halted(self, tracker):
        """Verify signal rejected when trading halted."""
        # Halt trading
        tracker.record_trade(pnl=-1100.00, order_id="ORDER-1")

        # Check signal would be rejected
        assert tracker.is_trading_allowed() is False

    def test_pnl_recorded_after_exit(self, tracker):
        """Verify P&L recorded after position exit."""
        # Simulate position exit
        tracker.record_trade(pnl=-250.00, order_id="POSITION-1")

        assert tracker.get_daily_pnl() == -250.00
        assert tracker.get_remaining_loss_allowance() == 750.00


class TestAlertOnBreach:
    """Test alert on limit breach."""

    @pytest.fixture
    def tracker(self):
        """Create daily loss tracker."""
        return DailyLossTracker(
            daily_loss_limit=1000.00,
            account_balance=50000.00
        )

    def test_alert_logged_on_breach(self, tracker):
        """Verify alert logged when limit breached."""
        with patch('src.risk.daily_loss_tracker.logger') as mock_logger:
            tracker.record_trade(pnl=-1100.00, order_id="ORDER-123")

            # Verify critical alert logged
            mock_logger.critical.assert_called()
            call_args = str(mock_logger.critical.call_args)
            assert "DAILY LOSS LIMIT BREACHED" in call_args
