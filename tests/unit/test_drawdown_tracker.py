"""Unit tests for Maximum Drawdown Limit Enforcement.

Tests drawdown tracking, peak value updates, limit enforcement,
recovery threshold, CSV logging, and integration with pipeline.
"""

import tempfile
from pathlib import Path
import pytest

from src.risk.drawdown_tracker import DrawdownTracker


class TestDrawdownTrackerInit:
    """Test DrawdownTracker initialization."""

    def test_init_with_valid_parameters(self):
        """Verify tracker initializes with valid parameters."""
        tracker = DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

        assert tracker._max_drawdown_percentage == 0.10
        assert tracker._recovery_threshold_percentage == 0.95
        assert tracker._peak_value == 50000.00
        assert tracker._current_value == 50000.00
        assert tracker._is_halted is False

    def test_init_with_invalid_max_drawdown(self):
        """Verify tracker raises error with invalid max drawdown."""
        with pytest.raises(ValueError):
            DrawdownTracker(
                max_drawdown_percentage=1.5,  # > 1.0 (100%)
                recovery_threshold_percentage=0.95,
                initial_value=50000.00
            )

    def test_init_with_invalid_recovery_threshold(self):
        """Verify tracker raises error with invalid recovery threshold."""
        with pytest.raises(ValueError):
            DrawdownTracker(
                max_drawdown_percentage=0.10,
                recovery_threshold_percentage=1.2,  # > 1.0
                initial_value=50000.00
            )

    def test_init_with_negative_initial_value(self):
        """Verify tracker raises error with negative initial value."""
        with pytest.raises(ValueError):
            DrawdownTracker(
                max_drawdown_percentage=0.10,
                recovery_threshold_percentage=0.95,
                initial_value=-1000.00
            )


class TestDrawdownCalculation:
    """Test drawdown percentage calculation."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker."""
        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

    def test_drawdown_zero_at_start(self, tracker):
        """Verify drawdown is zero at start."""
        assert tracker.get_drawdown_percentage() == 0.0

    def test_drawdown_after_losses(self, tracker):
        """Verify drawdown calculated correctly after losses."""
        tracker.update_value(48000.00)

        # Peak: $50,000, Current: $48,000
        # Drawdown: ($50K - $48K) / $50K = 4%
        assert tracker.get_drawdown_percentage() == 0.04

    def test_drawdown_zero_during_rally(self, tracker):
        """Verify drawdown zero when account at peak."""
        tracker.update_value(52000.00)  # New peak

        assert tracker.get_drawdown_percentage() == 0.0

    def test_drawdown_multiple_declines(self, tracker):
        """Verify drawdown accumulates over multiple updates."""
        tracker.update_value(48000.00)  # 4% drawdown
        tracker.update_value(46000.00)  # 8% drawdown

        # Peak: $50,000, Current: $46,000
        # Drawdown: ($50K - $46K) / $50K = 8%
        assert tracker.get_drawdown_percentage() == 0.08


class TestPeakValueTracking:
    """Test peak value tracking."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker."""
        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

    def test_peak_starts_at_initial_value(self, tracker):
        """Verify peak starts at initial value."""
        assert tracker.get_peak_value() == 50000.00

    def test_peak_updates_on_new_high(self, tracker):
        """Verify peak updates when account reaches new high."""
        tracker.update_value(52000.00)

        assert tracker.get_peak_value() == 52000.00

    def test_peak_does_not_update_on_decline(self, tracker):
        """Verify peak doesn't update when account declines."""
        tracker.update_value(48000.00)

        assert tracker.get_peak_value() == 50000.00

    def test_peak_updates_multiple_times(self, tracker):
        """Verify peak updates through rally."""
        tracker.update_value(51000.00)
        assert tracker.get_peak_value() == 51000.00

        tracker.update_value(53000.00)
        assert tracker.get_peak_value() == 53000.00

        tracker.update_value(52000.00)
        assert tracker.get_peak_value() == 53000.00  # Peak unchanged


class TestDrawdownLimitEnforcement:
    """Test maximum drawdown limit enforcement."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker."""
        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

    def test_trading_allowed_below_limit(self, tracker):
        """Verify trading allowed when drawdown below limit."""
        tracker.update_value(48000.00)  # 4% drawdown

        assert tracker.is_trading_allowed() is True
        assert tracker.is_trading_halted() is False

    def test_trading_halted_on_limit_breach(self, tracker):
        """Verify trading halted when drawdown exceeds limit."""
        tracker.update_value(52000.00)  # New peak
        tracker.update_value(46500.00)  # 10.58% drawdown

        assert tracker.is_trading_halted() is True
        assert tracker.is_trading_allowed() is False

    def test_trading_halted_at_exact_limit(self, tracker):
        """Verify trading halted at exact limit threshold."""
        tracker.update_value(52000.00)  # New peak
        tracker.update_value(46800.00)  # 10% drawdown exactly

        assert tracker.is_trading_halted() is True
        assert tracker.is_trading_allowed() is False

    def test_halt_reason_set_correctly(self, tracker):
        """Verify halt reason includes drawdown details."""
        tracker.update_value(52000.00)  # New peak
        tracker.update_value(46000.00)  # 11.54% drawdown

        reason = tracker.get_halt_reason()
        assert "Maximum drawdown limit breached" in reason
        assert "11.54%" in reason


class TestRecoveryThreshold:
    """Test recovery threshold (hysteresis)."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker."""
        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

    def test_recovery_value_calculation(self, tracker):
        """Verify recovery value calculated correctly."""
        tracker.update_value(52000.00)  # New peak

        # Recovery threshold: 95% of peak
        # $52,000 × 0.95 = $49,400
        assert tracker.get_recovery_value() == 49400.00

    def test_trading_resumes_at_recovery_threshold(self, tracker):
        """Verify trading resumes at recovery threshold, not peak."""
        tracker.update_value(52000.00)  # New peak
        tracker.update_value(46500.00)  # Halted (10.58% drawdown)

        assert tracker.is_trading_halted() is True

        # Recover to 95% of peak
        tracker.update_value(49400.00)

        assert tracker.is_trading_halted() is False
        assert tracker.is_trading_allowed() is True

    def test_trading_still_halted_below_recovery_threshold(self, tracker):
        """Verify trading still halted below recovery threshold."""
        tracker.update_value(52000.00)  # New peak
        tracker.update_value(46500.00)  # Halted

        # Recover to 94% of peak (below 95% threshold)
        tracker.update_value(48880.00)

        assert tracker.is_trading_halted() is True


class TestGetDrawdownSummary:
    """Test drawdown summary generation."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker."""
        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

    def test_summary_at_start(self, tracker):
        """Verify summary at start."""
        summary = tracker.get_drawdown_summary()

        assert summary['peak_value'] == 50000.00
        assert summary['current_value'] == 50000.00
        assert summary['drawdown_percentage'] == 0.0
        assert summary['is_halted'] is False

    def test_summary_after_decline(self, tracker):
        """Verify summary after account decline."""
        tracker.update_value(48000.00)

        summary = tracker.get_drawdown_summary()

        assert summary['peak_value'] == 50000.00
        assert summary['current_value'] == 48000.00
        assert summary['drawdown_percentage'] == 0.04

    def test_summary_includes_recovery_value(self, tracker):
        """Verify summary includes recovery value."""
        tracker.update_value(52000.00)  # New peak

        summary = tracker.get_drawdown_summary()

        assert 'recovery_value' in summary
        assert summary['recovery_value'] == 49400.00


class TestCSVAuditTrailLogging:
    """Test CSV audit trail logging."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker with audit trail."""
        temp_dir = tempfile.mkdtemp()
        audit_path = str(Path(temp_dir) / "drawdown_tracking.csv")

        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00,
            audit_trail_path=audit_path
        )

    def test_csv_file_created(self, tracker):
        """Verify CSV file created on first update."""
        tracker.update_value(51000.00)

        assert Path(tracker._audit_trail_path).exists()

    def test_csv_has_correct_columns(self, tracker):
        """Verify CSV has all required columns."""
        tracker.update_value(51000.00)

        import csv
        with open(tracker._audit_trail_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "event_type",
            "peak_value",
            "current_value",
            "drawdown_percentage",
            "is_halted",
            "recovery_value"
        ]

        assert headers == expected_headers


class TestIntegration:
    """Test integration with trading pipeline."""

    @pytest.fixture
    def tracker(self):
        """Create drawdown tracker."""
        return DrawdownTracker(
            max_drawdown_percentage=0.10,
            recovery_threshold_percentage=0.95,
            initial_value=50000.00
        )

    def test_signal_rejected_when_halted(self, tracker):
        """Verify signal rejected when trading halted."""
        # Halt trading
        tracker.update_value(52000.00)  # Peak
        tracker.update_value(46000.00)  # Halted

        # Check signal would be rejected
        assert tracker.is_trading_allowed() is False

    def test_account_value_updated_after_trade(self, tracker):
        """Verify account value updated after trade."""
        # Simulate profitable trade
        tracker.update_value(51000.00)

        assert tracker.get_current_value() == 51000.00
        assert tracker.get_peak_value() == 51000.00
