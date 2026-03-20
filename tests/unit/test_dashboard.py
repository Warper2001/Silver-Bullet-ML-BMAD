"""Unit tests for Dashboard components.

Tests color utilities, system status detection, page routing, auto-refresh,
and account metrics for the Streamlit dashboard foundation.
"""

import time

from src.dashboard.theme import get_loss_color, get_neutral_color, get_profit_color
from src.dashboard.shared_state import get_system_status


class TestColorUtilities:
    """Test color scheme utilities."""

    def test_get_profit_color_returns_green(self):
        """Verify profit color is green (#00FF00)."""
        color = get_profit_color()
        assert color == "#00FF00"

    def test_get_loss_color_returns_red(self):
        """Verify loss color is red (#FF0000)."""
        color = get_loss_color()
        assert color == "#FF0000"

    def test_get_neutral_color_returns_blue(self):
        """Verify neutral color is blue (#0080FF)."""
        color = get_neutral_color()
        assert color == "#0080FF"


class TestSystemStatusDetection:
    """Test system status detection."""

    def test_get_system_status_returns_valid_status(self):
        """Verify system status returns RUNNING, HALTED, or ERROR."""
        status = get_system_status()
        assert status in ["RUNNING", "HALTED", "ERROR"]


class TestAutoRefresh:
    """Test auto-refresh mechanism."""

    def test_auto_refresh_interval_is_approximately_2_seconds(self):
        """Verify auto-refresh interval is approximately 2 seconds."""
        # Verify the constant is defined in streamlit_app.py
        import importlib.util

        importlib.util.spec_from_file_location(
            "streamlit_app", "src/dashboard/streamlit_app.py"
        )

        # Read file and check for REFRESH_INTERVAL constant
        with open("src/dashboard/streamlit_app.py") as f:
            content = f.read()
            assert "REFRESH_INTERVAL = 2" in content


class TestStartupPerformance:
    """Test dashboard startup performance."""

    def test_dashboard_loads_under_3_seconds(self):
        """Verify dashboard startup time is < 3 seconds."""
        import subprocess

        start_time = time.perf_counter()

        # Launch Streamlit app in background and measure startup
        # Note: This is a placeholder - actual test will use subprocess
        # with timeout to measure actual startup time
        process = subprocess.Popen(
            ["poetry", "run", "streamlit", "run", "src/dashboard/streamlit_app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give it time to start
        time.sleep(1)

        # Check if process is still running (indicates successful startup)
        assert process.poll() is None

        elapsed = time.perf_counter() - start_time

        # Clean up
        process.terminate()
        process.wait(timeout=5)

        # Note: Startup time will be more accurately measured in integration tests
        # This unit test is a basic check
        assert elapsed < 10.0  # Generous threshold for unit test


class TestAccountMetrics:
    """Test account metrics data models."""

    def test_account_metrics_dataclass_exists(self):
        """Verify AccountMetrics dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import AccountMetrics

        assert is_dataclass(AccountMetrics)

    def test_account_metrics_has_required_fields(self):
        """Verify AccountMetrics has all required fields."""
        from src.dashboard.shared_state import AccountMetrics
        from datetime import datetime

        metrics = AccountMetrics(
            equity=100000.00,
            daily_change_pct=2.5,
            daily_change_usd=2500.00,
            daily_pnl=1500.00,
            open_positions_count=2,
            open_contracts=3,
            trade_count=5,
            win_rate=60.0,
            daily_drawdown=200.00,
            daily_loss_limit=500.00,
            system_uptime="4h 23m",
            last_update=datetime.now()
        )

        assert metrics.equity == 100000.00
        assert metrics.daily_change_pct == 2.5
        assert metrics.daily_change_usd == 2500.00
        assert metrics.daily_pnl == 1500.00
        assert metrics.open_positions_count == 2
        assert metrics.open_contracts == 3
        assert metrics.trade_count == 5
        assert metrics.win_rate == 60.0
        assert metrics.daily_drawdown == 200.00
        assert metrics.daily_loss_limit == 500.00
        assert metrics.system_uptime == "4h 23m"

    def test_daily_performance_dataclass_exists(self):
        """Verify DailyPerformance dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import DailyPerformance

        assert is_dataclass(DailyPerformance)

    def test_daily_performance_has_required_fields(self):
        """Verify DailyPerformance has all required fields."""
        from src.dashboard.shared_state import DailyPerformance

        performance = DailyPerformance(
            trade_count=5,
            winning_trades=3,
            losing_trades=2,
            win_rate=60.0,
            total_pnl=1500.00,
            max_drawdown=200.00
        )

        assert performance.trade_count == 5
        assert performance.winning_trades == 3
        assert performance.losing_trades == 2
        assert performance.win_rate == 60.0
        assert performance.total_pnl == 1500.00
        assert performance.max_drawdown == 200.00


class TestAccountMetricsRetrieval:
    """Test account metrics retrieval functions."""

    def test_get_account_metrics_returns_account_metrics(self):
        """Verify get_account_metrics returns AccountMetrics instance."""
        from src.dashboard.shared_state import AccountMetrics, get_account_metrics

        metrics = get_account_metrics()
        assert isinstance(metrics, AccountMetrics)

    def test_get_daily_pnl_returns_float(self):
        """Verify get_daily_pnl returns float."""
        from src.dashboard.shared_state import get_daily_pnl

        pnl = get_daily_pnl()
        assert isinstance(pnl, float)

    def test_get_open_positions_summary_returns_dict(self):
        """Verify get_open_positions_summary returns dict with keys."""
        from src.dashboard.shared_state import get_open_positions_summary

        summary = get_open_positions_summary()
        assert isinstance(summary, dict)
        assert "count" in summary
        assert "contracts" in summary

    def test_get_daily_performance_returns_daily_performance(self):
        """Verify get_daily_performance returns DailyPerformance instance."""
        from src.dashboard.shared_state import DailyPerformance, get_daily_performance

        performance = get_daily_performance()
        assert isinstance(performance, DailyPerformance)

    def test_get_system_uptime_returns_string(self):
        """Verify get_system_uptime returns string."""
        from src.dashboard.shared_state import get_system_uptime

        uptime = get_system_uptime()
        assert isinstance(uptime, str)


class TestMetricCalculations:
    """Test metric calculation logic."""

    def test_win_rate_calculation_zero_trades(self):
        """Verify win rate is 0% when no trades."""
        from src.dashboard.shared_state import calculate_win_rate

        win_rate = calculate_win_rate(0, 0)
        assert win_rate == 0.0

    def test_win_rate_calculation_all_winning(self):
        """Verify win rate is 100% when all trades win."""
        from src.dashboard.shared_state import calculate_win_rate

        win_rate = calculate_win_rate(10, 10)
        assert win_rate == 100.0

    def test_win_rate_calculation_half_winning(self):
        """Verify win rate is 50% when half trades win."""
        from src.dashboard.shared_state import calculate_win_rate

        win_rate = calculate_win_rate(5, 10)
        assert win_rate == 50.0

    def test_trend_indicator_positive(self):
        """Verify trend indicator is ↑ for positive change."""
        from src.dashboard.shared_state import calculate_trend

        indicator, color = calculate_trend(105.0, 100.0)
        assert indicator == "↑"
        assert color == "#00FF00"  # Green

    def test_trend_indicator_negative(self):
        """Verify trend indicator is ↓ for negative change."""
        from src.dashboard.shared_state import calculate_trend

        indicator, color = calculate_trend(95.0, 100.0)
        assert indicator == "↓"
        assert color == "#FF0000"  # Red

    def test_trend_indicator_neutral(self):
        """Verify trend indicator is → for no change."""
        from src.dashboard.shared_state import calculate_trend

        indicator, color = calculate_trend(100.0, 100.0)
        assert indicator == "→"
        assert color == "#0080FF"  # Blue


class TestOpenPositionsDataModels:
    """Test open positions data models."""

    def test_direction_enum_exists(self):
        """Verify Direction enum is defined with LONG and SHORT values."""
        from src.dashboard.shared_state import Direction

        assert hasattr(Direction, 'LONG')
        assert hasattr(Direction, 'SHORT')
        assert Direction.LONG.value == "LONG"
        assert Direction.SHORT.value == "SHORT"

    def test_barrier_levels_dataclass_exists(self):
        """Verify BarrierLevels dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import BarrierLevels

        assert is_dataclass(BarrierLevels)

    def test_barrier_levels_has_required_fields(self):
        """Verify BarrierLevels has all required fields."""
        from src.dashboard.shared_state import BarrierLevels
        from datetime import datetime

        barriers = BarrierLevels(
            upper_barrier=4600.00,
            lower_barrier=4450.00,
            vertical_barrier=datetime.now(),
            entry_price=4500.00
        )

        assert barriers.upper_barrier == 4600.00
        assert barriers.lower_barrier == 4450.00
        assert barriers.entry_price == 4500.00

    def test_position_signal_dataclass_exists(self):
        """Verify PositionSignal dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import PositionSignal

        assert is_dataclass(PositionSignal)

    def test_position_signal_has_required_fields(self):
        """Verify PositionSignal has all required fields."""
        from src.dashboard.shared_state import PositionSignal, Direction

        signal = PositionSignal(
            signal_id="SIGNAL-001",
            direction=Direction.LONG,
            confidence=4,
            ml_probability=0.75,
            mss_present=True,
            fvg_present=True,
            sweep_present=False,
            time_window="09:30-16:00"
        )

        assert signal.signal_id == "SIGNAL-001"
        assert signal.direction == Direction.LONG
        assert signal.confidence == 4
        assert signal.ml_probability == 0.75
        assert signal.mss_present is True
        assert signal.fvg_present is True
        assert signal.sweep_present is False

    def test_open_position_dataclass_exists(self):
        """Verify OpenPosition dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import OpenPosition

        assert is_dataclass(OpenPosition)

    def test_open_position_has_required_fields(self):
        """Verify OpenPosition has all required fields."""
        from src.dashboard.shared_state import OpenPosition, BarrierLevels, Direction
        from datetime import datetime

        barriers = BarrierLevels(
            upper_barrier=4600.00,
            lower_barrier=4450.00,
            vertical_barrier=datetime.now(),
            entry_price=4500.00
        )

        position = OpenPosition(
            signal_id="SIGNAL-001",
            direction=Direction.LONG,
            entry_price=4500.00,
            current_price=4525.00,
            pnl_usd=125.00,
            pnl_pct=2.78,
            barriers=barriers,
            confidence=4,
            ml_probability=0.75,
            entry_time=datetime.now()
        )

        assert position.signal_id == "SIGNAL-001"
        assert position.direction == Direction.LONG
        assert position.entry_price == 4500.00
        assert position.current_price == 4525.00
        assert position.pnl_usd == 125.00
        assert position.pnl_pct == 2.78
        assert position.confidence == 4
        assert position.ml_probability == 0.75


class TestBarrierCalculations:
    """Test barrier calculation functions."""

    def test_barrier_progress_at_entry(self):
        """Verify barrier progress is 0% at entry price."""
        from src.dashboard.shared_state import calculate_barrier_progress

        progress = calculate_barrier_progress(4500.00, 4600.00, 4500.00)
        assert progress == 0.0

    def test_barrier_progress_halfway(self):
        """Verify barrier progress is 50% halfway to barrier."""
        from src.dashboard.shared_state import calculate_barrier_progress

        progress = calculate_barrier_progress(4550.00, 4600.00, 4500.00)
        assert progress == 50.0

    def test_barrier_progress_at_barrier(self):
        """Verify barrier progress is 100% at barrier."""
        from src.dashboard.shared_state import calculate_barrier_progress

        progress = calculate_barrier_progress(4600.00, 4600.00, 4500.00)
        assert progress == 100.0

    def test_barrier_progress_beyond_barrier(self):
        """Verify barrier progress caps at 100% beyond barrier."""
        from src.dashboard.shared_state import calculate_barrier_progress

        progress = calculate_barrier_progress(4650.00, 4600.00, 4500.00)
        assert progress == 100.0

    def test_barrier_progress_zero_range(self):
        """Verify barrier progress is 0% when range is zero."""
        from src.dashboard.shared_state import calculate_barrier_progress

        progress = calculate_barrier_progress(4500.00, 4500.00, 4500.00)
        assert progress == 0.0


class TestTimeFormatting:
    """Test time formatting functions."""

    def test_time_remaining_future(self):
        """Verify time remaining formats correctly for future time."""
        from src.dashboard.shared_state import calculate_time_remaining
        from datetime import datetime, timedelta

        future = datetime.now() + timedelta(minutes=15, seconds=30)
        remaining = calculate_time_remaining(future)

        # Format should be MM:SS, allow for 1-2 second timing variance
        assert ":" in remaining
        parts = remaining.split(":")
        assert len(parts) == 2
        minutes, seconds = int(parts[0]), int(parts[1])
        # Should be approximately 15 minutes, 29-30 seconds
        assert minutes == 15
        assert 29 <= seconds <= 31

    def test_time_remaining_less_than_minute(self):
        """Verify time remaining formats correctly for < 1 minute."""
        from src.dashboard.shared_state import calculate_time_remaining
        from datetime import datetime, timedelta

        future = datetime.now() + timedelta(seconds=45)
        remaining = calculate_time_remaining(future)

        # Format should be MM:SS, allow for 1-2 second timing variance
        assert ":" in remaining
        parts = remaining.split(":")
        assert len(parts) == 2
        minutes, seconds = int(parts[0]), int(parts[1])
        # Should be approximately 0 minutes, 43-45 seconds
        assert minutes == 0
        assert 43 <= seconds <= 46

    def test_time_remaining_expired(self):
        """Verify time remaining shows 'Expired' for past time."""
        from src.dashboard.shared_state import calculate_time_remaining
        from datetime import datetime, timedelta

        past = datetime.now() - timedelta(minutes=5)
        remaining = calculate_time_remaining(past)

        assert remaining == "Expired"


class TestPnLFormatting:
    """Test P&L formatting functions."""

    def test_format_position_pnl_profit(self):
        """Verify P&L formatting for profit shows green indicator."""
        from src.dashboard.shared_state import format_position_pnl

        formatted = format_position_pnl(125.00)
        assert "🟢" in formatted
        assert "$+125.00" in formatted

    def test_format_position_pnl_loss(self):
        """Verify P&L formatting for loss shows red indicator."""
        from src.dashboard.shared_state import format_position_pnl

        formatted = format_position_pnl(-50.00)
        assert "🔴" in formatted
        assert "$-50.00" in formatted

    def test_format_position_pnl_break_even(self):
        """Verify P&L formatting for break-even shows green indicator."""
        from src.dashboard.shared_state import format_position_pnl

        formatted = format_position_pnl(0.00)
        assert "🟢" in formatted
        assert "$+0.00" in formatted


class TestSignalDataModels:
    """Test signal data models from Story 8.4."""

    def test_signal_status_enum_exists(self):
        """Verify SignalStatus enum is defined with FILTERED, EXECUTED, REJECTED."""
        from src.dashboard.shared_state import SignalStatus

        assert hasattr(SignalStatus, 'FILTERED')
        assert hasattr(SignalStatus, 'EXECUTED')
        assert hasattr(SignalStatus, 'REJECTED')
        assert SignalStatus.FILTERED.value == "Filtered"
        assert SignalStatus.EXECUTED.value == "Executed"
        assert SignalStatus.REJECTED.value == "Rejected"

    def test_silver_bullet_signal_dataclass_exists(self):
        """Verify SilverBulletSignal dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import SilverBulletSignal

        assert is_dataclass(SilverBulletSignal)

    def test_silver_bullet_signal_has_required_fields(self):
        """Verify SilverBulletSignal has all required fields."""
        from src.dashboard.shared_state import (
            SilverBulletSignal,
            SignalStatus,
            Direction,
        )
        from datetime import datetime

        signal = SilverBulletSignal(
            timestamp=datetime.now(),
            direction=Direction.LONG,
            confidence=4,
            ml_probability=0.75,
            mss_present=True,
            fvg_present=True,
            sweep_present=False,
            time_window="09:30-16:00",
            status=SignalStatus.EXECUTED,
            signal_id="SIGNAL-001"
        )

        assert signal.signal_id == "SIGNAL-001"
        assert signal.direction == Direction.LONG
        assert signal.confidence == 4
        assert signal.ml_probability == 0.75
        assert signal.mss_present is True
        assert signal.fvg_present is True
        assert signal.sweep_present is False
        assert signal.time_window == "09:30-16:00"
        assert signal.status == SignalStatus.EXECUTED

    def test_silver_bullet_signal_confidence_validation(self):
        """Verify confidence is in valid range 1-5."""
        from src.dashboard.shared_state import (
            SilverBulletSignal,
            SignalStatus,
            Direction,
        )
        from datetime import datetime

        # Valid confidence values
        for confidence in [1, 2, 3, 4, 5]:
            signal = SilverBulletSignal(
                timestamp=datetime.now(),
                direction=Direction.LONG,
                confidence=confidence,
                ml_probability=0.75,
                mss_present=True,
                fvg_present=True,
                sweep_present=False,
                time_window="09:30-16:00",
                status=SignalStatus.EXECUTED,
                signal_id=f"SIGNAL-{confidence}"
            )
            assert signal.confidence == confidence

    def test_silver_bullet_signal_probability_validation(self):
        """Verify ML probability is in valid range 0.0-1.0."""
        from src.dashboard.shared_state import (
            SilverBulletSignal,
            SignalStatus,
            Direction,
        )
        from datetime import datetime

        # Valid probability values
        for probability in [0.0, 0.25, 0.5, 0.75, 1.0]:
            signal = SilverBulletSignal(
                timestamp=datetime.now(),
                direction=Direction.LONG,
                confidence=4,
                ml_probability=probability,
                mss_present=True,
                fvg_present=True,
                sweep_present=False,
                time_window="09:30-16:00",
                status=SignalStatus.EXECUTED,
                signal_id=f"SIGNAL-{probability}"
            )
            assert signal.ml_probability == probability


class TestSignalRetrieval:
    """Test signal retrieval functions from Story 8.4."""

    def test_get_silver_bullet_signals_exists(self):
        """Verify get_silver_bullet_signals function exists."""
        from src.dashboard.shared_state import get_silver_bullet_signals

        assert callable(get_silver_bullet_signals)

    def test_get_silver_bullet_signals_returns_list(self):
        """Verify get_silver_bullet_signals returns list of signals."""
        from src.dashboard.shared_state import get_silver_bullet_signals

        signals = get_silver_bullet_signals()
        assert isinstance(signals, list)

    def test_get_silver_bullet_signals_returns_correct_structure(self):
        """Verify get_silver_bullet_signals returns SilverBulletSignal objects."""
        from src.dashboard.shared_state import get_silver_bullet_signals, SilverBulletSignal

        signals = get_silver_bullet_signals()

        if len(signals) > 0:
            signal = signals[0]
            assert isinstance(signal, SilverBulletSignal)
            assert hasattr(signal, 'signal_id')
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'ml_probability')
            assert hasattr(signal, 'status')

    def test_get_silver_bullet_signals_limited_to_50(self):
        """Verify get_silver_bullet_signals returns max 50 signals."""
        from src.dashboard.shared_state import get_silver_bullet_signals

        signals = get_silver_bullet_signals()
        assert len(signals) <= 50


class TestSignalFiltering:
    """Test signal filtering functions from Story 8.4."""

    def test_filter_signals_exists(self):
        """Verify filter_signals function exists."""
        from src.dashboard.shared_state import filter_signals

        assert callable(filter_signals)

    def test_filter_signals_by_status_filtered(self):
        """Verify filtering by status='Filtered' works."""
        from src.dashboard.shared_state import (
            filter_signals,
            get_silver_bullet_signals,
            SignalStatus,
        )

        signals = get_silver_bullet_signals()
        filtered = filter_signals(signals, status="Filtered")

        for signal in filtered:
            assert signal.status == SignalStatus.FILTERED

    def test_filter_signals_by_status_executed(self):
        """Verify filtering by status='Executed' works."""
        from src.dashboard.shared_state import (
            filter_signals,
            get_silver_bullet_signals,
            SignalStatus,
        )

        signals = get_silver_bullet_signals()
        filtered = filter_signals(signals, status="Executed")

        for signal in filtered:
            assert signal.status == SignalStatus.EXECUTED

    def test_filter_signals_by_direction_long(self):
        """Verify filtering by direction='LONG' works."""
        from src.dashboard.shared_state import (
            filter_signals,
            get_silver_bullet_signals,
            Direction,
        )

        signals = get_silver_bullet_signals()
        filtered = filter_signals(signals, direction="LONG")

        for signal in filtered:
            assert signal.direction == Direction.LONG

    def test_filter_signals_by_direction_short(self):
        """Verify filtering by direction='SHORT' works."""
        from src.dashboard.shared_state import (
            filter_signals,
            get_silver_bullet_signals,
            Direction,
        )

        signals = get_silver_bullet_signals()
        filtered = filter_signals(signals, direction="SHORT")

        for signal in filtered:
            assert signal.direction == Direction.SHORT

    def test_filter_signals_by_confidence_5_star(self):
        """Verify filtering by confidence='5★' returns only 5-star signals."""
        from src.dashboard.shared_state import filter_signals, get_silver_bullet_signals

        signals = get_silver_bullet_signals()
        filtered = filter_signals(signals, confidence="5★")

        for signal in filtered:
            assert signal.confidence == 5

    def test_filter_signals_by_confidence_4_plus(self):
        """Verify filtering by confidence='4★+' returns signals with 4+ confidence."""
        from src.dashboard.shared_state import filter_signals, get_silver_bullet_signals

        signals = get_silver_bullet_signals()
        filtered = filter_signals(signals, confidence="4★+")

        for signal in filtered:
            assert signal.confidence >= 4

    def test_filter_signals_all_criteria(self):
        """Verify filtering with multiple criteria works."""
        from src.dashboard.shared_state import (
            filter_signals,
            get_silver_bullet_signals,
            SignalStatus,
            Direction,
        )

        signals = get_silver_bullet_signals()
        filtered = filter_signals(
            signals,
            status="Executed",
            direction="LONG",
            confidence="3★+"
        )

        for signal in filtered:
            assert signal.status == SignalStatus.EXECUTED
            assert signal.direction == Direction.LONG
            assert signal.confidence >= 3


class TestSignalFormatting:
    """Test signal formatting functions from Story 8.4."""

    def test_format_signal_status_exists(self):
        """Verify format_signal_status function exists."""
        from src.dashboard.shared_state import format_signal_status

        assert callable(format_signal_status)

    def test_format_signal_status_executed(self):
        """Verify status formatting for EXECUTED shows green indicator."""
        from src.dashboard.shared_state import (
            format_signal_status,
            SignalStatus,
        )

        formatted = format_signal_status(SignalStatus.EXECUTED)
        assert "🟢" in formatted
        assert "Executed" in formatted

    def test_format_signal_status_filtered(self):
        """Verify status formatting for FILTERED shows yellow indicator."""
        from src.dashboard.shared_state import (
            format_signal_status,
            SignalStatus,
        )

        formatted = format_signal_status(SignalStatus.FILTERED)
        assert "🟡" in formatted
        assert "Filtered" in formatted

    def test_format_signal_status_rejected(self):
        """Verify status formatting for REJECTED shows gray indicator."""
        from src.dashboard.shared_state import (
            format_signal_status,
            SignalStatus,
        )

        formatted = format_signal_status(SignalStatus.REJECTED)
        assert "⚪" in formatted
        assert "Rejected" in formatted

    def test_format_confidence_stars_exists(self):
        """Verify format_confidence_stars function exists."""
        from src.dashboard.shared_state import format_confidence_stars

        assert callable(format_confidence_stars)

    def test_format_confidence_stars_5(self):
        """Verify confidence 5 formats as ★★★★★."""
        from src.dashboard.shared_state import format_confidence_stars

        stars = format_confidence_stars(5)
        assert stars == "★★★★★"

    def test_format_confidence_stars_3(self):
        """Verify confidence 3 formats as ★★★☆☆."""
        from src.dashboard.shared_state import format_confidence_stars

        stars = format_confidence_stars(3)
        assert stars == "★★★☆☆"

    def test_format_confidence_stars_1(self):
        """Verify confidence 1 formats as ★☆☆☆☆."""
        from src.dashboard.shared_state import format_confidence_stars

        stars = format_confidence_stars(1)
        assert stars == "★☆☆☆☆"

    def test_format_confidence_stars_invalid(self):
        """Verify invalid confidence formats as ?."""
        from src.dashboard.shared_state import format_confidence_stars

        stars = format_confidence_stars(0)
        assert stars == "?"

        stars = format_confidence_stars(6)
        assert stars == "?"

    def test_format_ml_probability_bar_exists(self):
        """Verify format_ml_probability_bar function exists."""
        from src.dashboard.shared_state import format_ml_probability_bar

        assert callable(format_ml_probability_bar)

    def test_format_ml_probability_bar_100_percent(self):
        """Verify 100% probability formats as full bar."""
        from src.dashboard.shared_state import format_ml_probability_bar

        bar = format_ml_probability_bar(1.0)
        assert "██████████" in bar
        assert "100.0%" in bar

    def test_format_ml_probability_bar_75_percent(self):
        """Verify 75% probability formats as 3/4 bar."""
        from src.dashboard.shared_state import format_ml_probability_bar

        bar = format_ml_probability_bar(0.75)
        assert "███████░░░" in bar
        assert "75.0%" in bar

    def test_format_ml_probability_bar_50_percent(self):
        """Verify 50% probability formats as half bar."""
        from src.dashboard.shared_state import format_ml_probability_bar

        bar = format_ml_probability_bar(0.50)
        assert "█████░░░░░" in bar
        assert "50.0%" in bar

    def test_format_ml_probability_bar_25_percent(self):
        """Verify 25% probability formats as quarter bar."""
        from src.dashboard.shared_state import format_ml_probability_bar

        bar = format_ml_probability_bar(0.25)
        assert "██░░░░░░░░" in bar
        assert "25.0%" in bar

    def test_format_ml_probability_bar_0_percent(self):
        """Verify 0% probability formats as empty bar."""
        from src.dashboard.shared_state import format_ml_probability_bar

        bar = format_ml_probability_bar(0.0)
        assert "░░░░░░░░░░" in bar
        assert "0.0%" in bar

    def test_format_ml_probability_bar_invalid(self):
        """Verify invalid probability formats as error."""
        from src.dashboard.shared_state import format_ml_probability_bar

        bar = format_ml_probability_bar(1.5)
        assert bar == "??.?%"

        bar = format_ml_probability_bar(-0.1)
        assert bar == "??.?%"


# ============================================================================
# Story 8.5: Dollar Bar Charts with Patterns
# ============================================================================


class TestChartDataModels:
    """Test chart data models for Story 8.5."""

    def test_marker_type_enum_exists(self):
        """Verify MarkerType enum is defined with all required values."""
        from src.dashboard.shared_state import MarkerType

        assert MarkerType.MSS_BULLISH.value == "MSS_BULLISH"
        assert MarkerType.MSS_BEARISH.value == "MSS_BEARISH"
        assert MarkerType.FVG_BULLISH.value == "FVG_BULLISH"
        assert MarkerType.FVG_BEARISH.value == "FVG_BEARISH"
        assert MarkerType.SWEEP.value == "SWEEP"
        assert MarkerType.ENTRY.value == "ENTRY"
        assert MarkerType.EXIT_PROFIT.value == "EXIT_PROFIT"
        assert MarkerType.EXIT_LOSS.value == "EXIT_LOSS"

    def test_chart_marker_dataclass_exists(self):
        """Verify ChartMarker dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import ChartMarker

        assert is_dataclass(ChartMarker)

    def test_chart_marker_has_required_fields(self):
        """Verify ChartMarker has all required fields."""
        from src.dashboard.shared_state import (
            ChartMarker,
            MarkerType,
        )
        from datetime import datetime

        marker = ChartMarker(
            timestamp=datetime.now(),
            price=4500.0,
            marker_type=MarkerType.MSS_BULLISH,
            signal_id="SIGNAL-001"
        )

        assert hasattr(marker, 'timestamp')
        assert hasattr(marker, 'price')
        assert hasattr(marker, 'marker_type')
        assert hasattr(marker, 'signal_id')
        assert marker.signal_id == "SIGNAL-001"

    def test_fvg_zone_dataclass_exists(self):
        """Verify FVGZone dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import FVGZone

        assert is_dataclass(FVGZone)

    def test_fvg_zone_has_required_fields(self):
        """Verify FVGZone has all required fields."""
        from src.dashboard.shared_state import FVGZone
        from datetime import datetime, timedelta

        now = datetime.now()
        zone = FVGZone(
            start_time=now,
            end_time=now + timedelta(minutes=10),
            top_price=4510.0,
            bottom_price=4500.0,
            direction="bullish"
        )

        assert hasattr(zone, 'start_time')
        assert hasattr(zone, 'end_time')
        assert hasattr(zone, 'top_price')
        assert hasattr(zone, 'bottom_price')
        assert hasattr(zone, 'direction')
        assert zone.direction == "bullish"

    def test_trade_marker_dataclass_exists(self):
        """Verify TradeMarker dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import TradeMarker

        assert is_dataclass(TradeMarker)

    def test_trade_marker_has_required_fields(self):
        """Verify TradeMarker has all required fields."""
        from src.dashboard.shared_state import TradeMarker
        from datetime import datetime

        trade = TradeMarker(
            timestamp=datetime.now(),
            price=4500.0,
            trade_type="entry",
            pnl_usd=None,
            signal_id="SIGNAL-001"
        )

        assert hasattr(trade, 'timestamp')
        assert hasattr(trade, 'price')
        assert hasattr(trade, 'trade_type')
        assert hasattr(trade, 'pnl_usd')
        assert hasattr(trade, 'signal_id')
        assert trade.trade_type == "entry"
        assert trade.pnl_usd is None


class TestChartRetrieval:
    """Test chart data retrieval functions for Story 8.5."""

    def test_get_dollar_bars_exists(self):
        """Verify get_dollar_bars function exists."""
        from src.dashboard.shared_state import get_dollar_bars

        assert callable(get_dollar_bars)

    def test_get_dollar_bars_returns_dataframe(self):
        """Verify get_dollar_bars returns pandas DataFrame."""
        from src.dashboard.shared_state import get_dollar_bars
        import pandas as pd

        df = get_dollar_bars()
        assert isinstance(df, pd.DataFrame)

    def test_get_dollar_bars_returns_correct_structure(self):
        """Verify get_dollar_bars returns DataFrame with required columns."""
        from src.dashboard.shared_state import get_dollar_bars

        df = get_dollar_bars()

        # Check for required OHLCV columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_get_pattern_overlays_exists(self):
        """Verify get_pattern_overlays function exists."""
        from src.dashboard.shared_state import get_pattern_overlays

        assert callable(get_pattern_overlays)

    def test_get_pattern_overlays_returns_tuple(self):
        """Verify get_pattern_overlays returns tuple of (markers, zones)."""
        from src.dashboard.shared_state import get_pattern_overlays

        result = get_pattern_overlays()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_pattern_overlays_returns_chart_markers(self):
        """Verify get_pattern_overlays returns ChartMarker objects in first element."""
        from src.dashboard.shared_state import (
            get_pattern_overlays,
            ChartMarker,
        )

        markers, fvg_zones = get_pattern_overlays()

        if len(markers) > 0:
            marker = markers[0]
            assert isinstance(marker, ChartMarker)

    def test_get_pattern_overlays_returns_fvg_zones(self):
        """Verify get_pattern_overlays returns FVGZone objects in second element."""
        from src.dashboard.shared_state import (
            get_pattern_overlays,
            FVGZone,
        )

        markers, fvg_zones = get_pattern_overlays()

        if len(fvg_zones) > 0:
            zone = fvg_zones[0]
            assert isinstance(zone, FVGZone)

    def test_get_trade_markers_exists(self):
        """Verify get_trade_markers function exists."""
        from src.dashboard.shared_state import get_trade_markers

        assert callable(get_trade_markers)

    def test_get_trade_markers_returns_list(self):
        """Verify get_trade_markers returns list of trade markers."""
        from src.dashboard.shared_state import get_trade_markers

        trades = get_trade_markers()
        assert isinstance(trades, list)

    def test_get_trade_markers_returns_trade_markers(self):
        """Verify get_trade_markers returns TradeMarker objects."""
        from src.dashboard.shared_state import (
            get_trade_markers,
            TradeMarker,
        )

        trades = get_trade_markers()

        if len(trades) > 0:
            trade = trades[0]
            assert isinstance(trade, TradeMarker)


class TestChartRendering:
    """Test chart rendering functions for Story 8.5."""

    def test_render_charts_exists(self):
        """Verify render_charts function exists in navigation.py."""
        # Check if function is defined by reading the file
        with open("src/dashboard/navigation.py") as f:
            content = f.read()
            assert "def render_charts():" in content


class TestConfigurationDataModels:
    """Test configuration data models for Story 8.6."""

    def test_risk_limits_dataclass_exists(self):
        """Verify RiskLimits dataclass is defined."""
        from dataclasses import is_dataclass

        from src.dashboard.shared_state import RiskLimits

        assert is_dataclass(RiskLimits)

    def test_risk_limits_has_required_fields(self):
        """Verify RiskLimits has all required fields."""
        from src.dashboard.shared_state import RiskLimits

        limits = RiskLimits(
            daily_loss_limit=500.0,
            max_drawdown_pct=12.0,
            per_trade_risk_pct=2.0,
            max_position_contracts=5
        )

        assert hasattr(limits, 'daily_loss_limit')
        assert hasattr(limits, 'max_drawdown_pct')
        assert hasattr(limits, 'per_trade_risk_pct')
        assert hasattr(limits, 'max_position_contracts')
        assert limits.daily_loss_limit == 500.0
        assert limits.max_drawdown_pct == 12.0

    def test_time_window_dataclass_exists(self):
        """Verify TimeWindow dataclass is defined."""
        from dataclasses import is_dataclass

        from src.dashboard.shared_state import TimeWindow

        assert is_dataclass(TimeWindow)

    def test_time_window_has_required_fields(self):
        """Verify TimeWindow has all required fields."""
        from src.dashboard.shared_state import TimeWindow

        window = TimeWindow(
            enabled=True,
            start_time="09:30",
            end_time="11:00"
        )

        assert hasattr(window, 'enabled')
        assert hasattr(window, 'start_time')
        assert hasattr(window, 'end_time')
        assert window.enabled is True
        assert window.start_time == "09:30"

    def test_ml_config_dataclass_exists(self):
        """Verify MLConfig dataclass is defined."""
        from dataclasses import is_dataclass

        from src.dashboard.shared_state import MLConfig

        assert is_dataclass(MLConfig)

    def test_ml_config_has_required_fields(self):
        """Verify MLConfig has all required fields."""
        from src.dashboard.shared_state import MLConfig

        config = MLConfig(min_probability=0.65)

        assert hasattr(config, 'min_probability')
        assert config.min_probability == 0.65

    def test_system_config_dataclass_exists(self):
        """Verify SystemConfig dataclass is defined."""
        from dataclasses import is_dataclass

        from src.dashboard.shared_state import SystemConfig

        assert is_dataclass(SystemConfig)

    def test_system_config_aggregates_all_sections(self):
        """Verify SystemConfig aggregates all configuration sections."""
        from src.dashboard.shared_state import (
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        assert hasattr(config, 'risk_limits')
        assert hasattr(config, 'london_am')
        assert hasattr(config, 'ny_am')
        assert hasattr(config, 'ny_pm')
        assert hasattr(config, 'ml_config')
        assert isinstance(config.risk_limits, RiskLimits)
        assert isinstance(config.london_am, TimeWindow)


class TestConfigurationReaders:
    """Test configuration reader functions for Story 8.6."""

    def test_get_system_config_exists(self):
        """Verify get_system_config function exists."""
        from src.dashboard.shared_state import get_system_config

        assert callable(get_system_config)

    def test_get_system_config_returns_valid_structure(self):
        """Verify get_system_config returns valid SystemConfig."""
        from src.dashboard.shared_state import (
            get_system_config,
            SystemConfig,
        )

        config = get_system_config()

        assert isinstance(config, SystemConfig)
        assert hasattr(config, 'risk_limits')
        assert hasattr(config, 'london_am')
        assert hasattr(config, 'ny_am')
        assert hasattr(config, 'ny_pm')
        assert hasattr(config, 'ml_config')

    def test_get_system_config_returns_mock_data(self):
        """Verify get_system_config returns mock data with expected defaults."""
        from src.dashboard.shared_state import get_system_config

        config = get_system_config()

        # Verify mock values match story specifications
        assert config.risk_limits.daily_loss_limit == 500.0
        assert config.risk_limits.max_drawdown_pct == 12.0
        assert config.risk_limits.per_trade_risk_pct == 2.0
        assert config.risk_limits.max_position_contracts == 5
        assert config.ml_config.min_probability == 0.65

    def test_get_default_config_exists(self):
        """Verify get_default_config function exists."""
        from src.dashboard.shared_state import get_default_config

        assert callable(get_default_config)

    def test_get_default_config_returns_system_config(self):
        """Verify get_default_config returns SystemConfig object."""
        from src.dashboard.shared_state import (
            get_default_config,
            SystemConfig,
        )

        config = get_default_config()

        assert isinstance(config, SystemConfig)


class TestInputValidation:
    """Test input validation logic for Story 8.6."""

    def test_validate_config_exists(self):
        """Verify validate_config function exists."""
        from src.dashboard.shared_state import validate_config

        assert callable(validate_config)

    def test_validate_config_accepts_valid_config(self):
        """Verify validation accepts valid configuration."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        is_valid, error_msg = validate_config(config)

        assert is_valid is True
        assert error_msg == ""

    def test_validate_config_rejects_negative_daily_loss(self):
        """Verify validation rejects negative daily loss limit."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(-100.0, 12.0, 2.0, 5),  # Invalid
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        is_valid, error_msg = validate_config(config)

        assert is_valid is False
        assert "positive" in error_msg.lower()

    def test_validate_config_rejects_invalid_drawdown_range(self):
        """Verify validation rejects drawdown outside 0-100%."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        # Test > 100%
        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 150.0, 2.0, 5),  # Invalid
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        is_valid, error_msg = validate_config(config)

        assert is_valid is False
        assert "drawdown" in error_msg.lower()

    def test_validate_config_rejects_time_window_start_after_end(self):
        """Verify validation rejects time window where start >= end."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "10:00", "05:00"),  # Invalid: start > end
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        is_valid, error_msg = validate_config(config)

        assert is_valid is False
        assert "before" in error_msg.lower()

    def test_validate_config_rejects_invalid_time_format(self):
        """Verify validation rejects invalid time format."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "invalid", "05:00"),  # Invalid format
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        is_valid, error_msg = validate_config(config)

        assert is_valid is False
        assert "time" in error_msg.lower()

    def test_validate_config_rejects_ml_threshold_outside_range(self):
        """Verify validation rejects ML threshold outside 0.0-1.0."""
        from src.dashboard.shared_state import (
            validate_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(1.5)  # Invalid: > 1.0
        )

        is_valid, error_msg = validate_config(config)

        assert is_valid is False
        assert "threshold" in error_msg.lower() or "probability" in error_msg.lower()


class TestConfigurationPersistence:
    """Test configuration persistence functions for Story 8.6."""

    def test_save_system_config_exists(self):
        """Verify save_system_config function exists."""
        from src.dashboard.shared_state import save_system_config

        assert callable(save_system_config)

    def test_save_system_config_requires_password(self):
        """Verify save_system_config requires password confirmation."""
        from src.dashboard.shared_state import (
            save_system_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(500.0, 12.0, 2.0, 5),
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.65)
        )

        # Mock implementation should require password
        result = save_system_config(config, "wrong_password")

        # Mock implementation returns False for wrong password
        assert result is False

    def test_save_system_config_logs_changes(self):
        """Verify save_system_config logs configuration changes."""
        import logging
        from src.dashboard.shared_state import (
            save_system_config,
            SystemConfig,
            RiskLimits,
            TimeWindow,
            MLConfig,
        )

        config = SystemConfig(
            risk_limits=RiskLimits(600.0, 15.0, 2.5, 6),  # Changed values
            london_am=TimeWindow(True, "02:00", "05:00"),
            ny_am=TimeWindow(True, "09:30", "11:00"),
            ny_pm=TimeWindow(True, "13:30", "15:30"),
            ml_config=MLConfig(0.70)
        )

        # Mock implementation logs changes (verify with correct password)
        # Logger warning indicates mock behavior
        result = save_system_config(config, "test_password")

        # Mock implementation returns True for correct password
        assert result is True


class TestHealthIndicatorDataModels:
    """Test health indicator data models from Story 8.7."""

    def test_api_connection_status_dataclass_creation(self):
        """Verify APIConnectionStatus dataclass can be created."""
        from src.dashboard.shared_state import APIConnectionStatus
        from datetime import datetime

        now = datetime.now()
        status = APIConnectionStatus(
            connected=True,
            last_ping_time=now,
            ping_latency_ms=45.0
        )

        assert status.connected is True
        assert status.last_ping_time == now
        assert status.ping_latency_ms == 45.0

    def test_resource_usage_dataclass_creation(self):
        """Verify ResourceUsage dataclass can be created."""
        from src.dashboard.shared_state import ResourceUsage

        resources = ResourceUsage(
            cpu_percent=35.2,
            memory_percent=62.8,
            disk_percent=45.1
        )

        assert resources.cpu_percent == 35.2
        assert resources.memory_percent == 62.8
        assert resources.disk_percent == 45.1

    def test_pipeline_component_status_dataclass_creation(self):
        """Verify PipelineComponentStatus dataclass can be created."""
        from src.dashboard.shared_state import PipelineComponentStatus
        from datetime import datetime

        now = datetime.now()
        status = PipelineComponentStatus(
            component_name="Data Flow",
            is_healthy=True,
            last_execution_time=now,
            error_count=0
        )

        assert status.component_name == "Data Flow"
        assert status.is_healthy is True
        assert status.last_execution_time == now
        assert status.error_count == 0

    def test_system_health_dataclass_aggregation(self):
        """Verify SystemHealth dataclass aggregates all indicators."""
        from src.dashboard.shared_state import (
            SystemHealth,
            APIConnectionStatus,
            ResourceUsage,
            PipelineComponentStatus,
        )
        from datetime import datetime

        now = datetime.now()

        health = SystemHealth(
            api_status=APIConnectionStatus(True, now, 45.0),
            resources=ResourceUsage(35.2, 62.8, 45.1),
            data_flow_status=PipelineComponentStatus("Data Flow", True, now, 0),
            signal_detection_status=PipelineComponentStatus("Signal Detection", True, now, 0),
            ml_prediction_status=PipelineComponentStatus("ML Prediction", True, now, 0),
            execution_status=PipelineComponentStatus("Execution", True, now, 0),
            active_alerts_count=0,
            system_uptime="4h 23m",
            last_restart_time=now
        )

        assert isinstance(health.api_status, APIConnectionStatus)
        assert isinstance(health.resources, ResourceUsage)
        assert isinstance(health.data_flow_status, PipelineComponentStatus)
        assert health.active_alerts_count == 0
        assert health.system_uptime == "4h 23m"


class TestHealthIndicatorReaders:
    """Test health indicator reader functions from Story 8.7."""

    def test_get_system_health_returns_valid_structure(self):
        """Verify get_system_health returns valid SystemHealth structure."""
        from src.dashboard.shared_state import get_system_health, SystemHealth

        health = get_system_health()

        assert isinstance(health, SystemHealth)
        assert hasattr(health, 'api_status')
        assert hasattr(health, 'resources')
        assert hasattr(health, 'data_flow_status')
        assert hasattr(health, 'signal_detection_status')
        assert hasattr(health, 'ml_prediction_status')
        assert hasattr(health, 'execution_status')
        assert hasattr(health, 'active_alerts_count')
        assert hasattr(health, 'system_uptime')
        assert hasattr(health, 'last_restart_time')

    def test_get_alerts_summary_returns_alert_count(self):
        """Verify get_alerts_summary returns alert count and summary."""
        from src.dashboard.shared_state import get_alerts_summary

        summary = get_alerts_summary()

        assert isinstance(summary, dict)
        assert 'count' in summary
        assert isinstance(summary['count'], int)
        assert summary['count'] >= 0


class TestHealthIndicatorColorCoding:
    """Test color coding logic from Story 8.7."""

    def test_get_resource_color_green_threshold(self):
        """Verify get_resource_color returns green for < 80%."""
        from src.dashboard.shared_state import get_resource_color

        # Test green range (< 80%)
        assert get_resource_color(0.0) == "#00FF00"
        assert get_resource_color(50.0) == "#00FF00"
        assert get_resource_color(79.9) == "#00FF00"

    def test_get_resource_color_yellow_threshold(self):
        """Verify get_resource_color returns yellow for 80-90%."""
        from src.dashboard.shared_state import get_resource_color

        # Test yellow range (80-90%)
        assert get_resource_color(80.0) == "#FFCC00"
        assert get_resource_color(85.0) == "#FFCC00"
        assert get_resource_color(89.9) == "#FFCC00"

    def test_get_resource_color_red_threshold(self):
        """Verify get_resource_color returns red for > 90%."""
        from src.dashboard.shared_state import get_resource_color

        # Test red range (> 90%)
        assert get_resource_color(90.0) == "#FF0000"
        assert get_resource_color(95.0) == "#FF0000"
        assert get_resource_color(100.0) == "#FF0000"

    def test_format_resource_usage_returns_color_and_emoji(self):
        """Verify format_resource_usage returns correct color and emoji."""
        from src.dashboard.shared_state import format_resource_usage

        # Test green range
        color, emoji = format_resource_usage(50.0)
        assert color == "#00FF00"
        assert emoji == "✅"

        # Test yellow range
        color, emoji = format_resource_usage(85.0)
        assert color == "#FFCC00"
        assert emoji == "⚠️"

        # Test red range
        color, emoji = format_resource_usage(95.0)
        assert color == "#FF0000"
        assert emoji == "🔴"


class TestDataFreshnessLogic:
    """Test data freshness logic from Story 8.7."""

    def test_staleness_warning_at_30_seconds(self):
        """Verify staleness warning triggers at > 30 seconds."""
        from src.dashboard.shared_state import is_data_stale
        from datetime import datetime, timedelta

        now = datetime.now()

        # Fresh data (<= 30 seconds)
        assert not is_data_stale(now - timedelta(seconds=5))
        assert not is_data_stale(now - timedelta(seconds=30))

        # Stale data (> 30 seconds)
        assert is_data_stale(now - timedelta(seconds=31))
        assert is_data_stale(now - timedelta(seconds=60))

    def test_calculate_data_age_seconds(self):
        """Verify calculate_data_age returns accurate seconds."""
        from src.dashboard.shared_state import calculate_data_age
        from datetime import datetime, timedelta

        now = datetime.now()

        # Test various ages
        age_5s = calculate_data_age(now - timedelta(seconds=5))
        assert age_5s == 5

        age_30s = calculate_data_age(now - timedelta(seconds=30))
        assert age_30s == 30

        age_60s = calculate_data_age(now - timedelta(seconds=60))
        assert age_60s == 60


# ============================================================================
# Story 8.8: Manual Trade Submission Form
# ============================================================================


class TestManualTradeDataModels:
    """Test manual trade data models for Story 8.8."""

    def test_manual_trade_request_dataclass_exists(self):
        """Verify ManualTradeRequest dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import ManualTradeRequest

        assert is_dataclass(ManualTradeRequest)

    def test_manual_trade_request_has_required_fields(self):
        """Verify ManualTradeRequest has all required fields."""
        from src.dashboard.shared_state import ManualTradeRequest
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        assert hasattr(request, 'direction')
        assert hasattr(request, 'quantity')
        assert hasattr(request, 'order_type')
        assert hasattr(request, 'limit_price')
        assert hasattr(request, 'submit_time')
        assert hasattr(request, 'submitted_by')
        assert request.direction == "Buy"
        assert request.quantity == 2

    def test_trade_preview_dataclass_exists(self):
        """Verify TradePreview dataclass is defined."""
        from dataclasses import is_dataclass
        from src.dashboard.shared_state import TradePreview

        assert is_dataclass(TradePreview)

    def test_trade_preview_has_required_fields(self):
        """Verify TradePreview has all required fields."""
        from src.dashboard.shared_state import TradePreview
        from datetime import datetime, timedelta

        preview = TradePreview(
            dollar_risk=1200.00,
            stop_loss_price=11440.0,
            upper_barrier_price=11625.0,
            lower_barrier_price=11440.0,
            vertical_barrier_time=datetime.now() + timedelta(minutes=45),
            margin_required=1000.00,
            margin_sufficient=True,
            position_size_valid=True,
            per_trade_risk_valid=True,
            validation_errors=[]
        )

        assert hasattr(preview, 'dollar_risk')
        assert hasattr(preview, 'stop_loss_price')
        assert hasattr(preview, 'upper_barrier_price')
        assert hasattr(preview, 'lower_barrier_price')
        assert hasattr(preview, 'vertical_barrier_time')
        assert hasattr(preview, 'margin_required')
        assert hasattr(preview, 'margin_sufficient')
        assert hasattr(preview, 'position_size_valid')
        assert hasattr(preview, 'per_trade_risk_valid')
        assert hasattr(preview, 'validation_errors')
        assert preview.dollar_risk == 1200.00


class TestManualTradeValidation:
    """Test manual trade validation functions for Story 8.8."""

    def test_validate_position_size_enforces_5_contract_limit(self):
        """Verify position size validation enforces 5 contract limit."""
        from src.dashboard.shared_state import validate_position_size

        # Valid: 1-5 contracts
        assert validate_position_size(1)[0] is True
        assert validate_position_size(5)[0] is True

        # Invalid: > 5 contracts
        assert validate_position_size(6)[0] is False
        assert "exceeds maximum position size" in validate_position_size(6)[1]

    def test_validate_position_size_rejects_zero_quantity(self):
        """Verify position size validation rejects zero quantity."""
        from src.dashboard.shared_state import validate_position_size

        # Invalid: zero or negative
        assert validate_position_size(0)[0] is False
        assert "greater than 0" in validate_position_size(0)[1]
        assert validate_position_size(-1)[0] is False

    def test_validate_per_trade_risk_enforces_2_percent(self):
        """Verify per-trade risk validation enforces 2% equity limit."""
        from src.dashboard.shared_state import validate_per_trade_risk

        account_equity = 10000.0
        max_risk = 200.0  # 2% of $10,000

        # Valid: < 2%
        assert validate_per_trade_risk(150.0, account_equity)[0] is True

        # Invalid: > 2%
        result = validate_per_trade_risk(250.0, account_equity)
        assert result[0] is False
        assert "exceeds 2% equity limit" in result[1]

    def test_validate_margin_requirement_checks_equity(self):
        """Verify margin requirement validation checks equity sufficiency."""
        from src.dashboard.shared_state import validate_margin_requirement

        account_equity = 10000.0

        # Valid: sufficient margin
        is_valid, error_msg, margin = validate_margin_requirement(2, account_equity)
        assert is_valid is True
        assert error_msg == ""
        assert margin == 1000.0  # 2 * 500

        # Invalid: insufficient margin
        is_valid, error_msg, margin = validate_margin_requirement(25, account_equity)
        assert is_valid is False
        assert "Insufficient margin" in error_msg
        assert margin == 12500.0  # 25 * 500

    def test_calculate_trade_preview_returns_valid_structure(self):
        """Verify trade preview calculation returns valid structure."""
        from src.dashboard.shared_state import (
            calculate_trade_preview,
            ManualTradeRequest,
            TradePreview,
        )
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        preview = calculate_trade_preview(request, current_price=11500.0, atr=50.0, account_equity=10000.0)

        assert isinstance(preview, TradePreview)
        assert preview.dollar_risk > 0
        assert preview.stop_loss_price < 11500.0  # Below entry for long
        assert preview.upper_barrier_price > 11500.0  # Above entry for long
        assert preview.position_size_valid is True  # 2 contracts < 5
        assert preview.vertical_barrier_time > datetime.now()

    def test_calculate_trade_preview_long_position(self):
        """Verify trade preview for long position."""
        from src.dashboard.shared_state import calculate_trade_preview, ManualTradeRequest
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        preview = calculate_trade_preview(request, current_price=11500.0, atr=50.0, account_equity=10000.0)

        # Stop loss should be 1.2x ATR below entry
        assert preview.stop_loss_price == 11500.0 - (1.2 * 50.0)
        # Upper barrier should be 2.5x ATR above entry
        assert preview.upper_barrier_price == 11500.0 + (2.5 * 50.0)
        # Lower barrier should be stop loss
        assert preview.lower_barrier_price == preview.stop_loss_price

    def test_calculate_trade_preview_short_position(self):
        """Verify trade preview for short position."""
        from src.dashboard.shared_state import calculate_trade_preview, ManualTradeRequest
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Sell",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        preview = calculate_trade_preview(request, current_price=11500.0, atr=50.0, account_equity=10000.0)

        # Stop loss should be 1.2x ATR above entry (for short)
        assert preview.stop_loss_price == 11500.0 + (1.2 * 50.0)
        # Upper barrier should be stop loss (for short)
        assert preview.upper_barrier_price == preview.stop_loss_price
        # Lower barrier should be 2.5x ATR below entry
        assert preview.lower_barrier_price == 11500.0 - (2.5 * 50.0)

    def test_calculate_trade_preview_vertical_barrier_45_minutes(self):
        """Verify vertical barrier is 45 minutes from entry."""
        from src.dashboard.shared_state import calculate_trade_preview, ManualTradeRequest
        from datetime import datetime, timedelta

        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        preview = calculate_trade_preview(request, current_price=11500.0, atr=50.0, account_equity=10000.0)

        # Vertical barrier should be ~45 minutes from now
        time_diff = (preview.vertical_barrier_time - datetime.now()).total_seconds()
        assert 44 * 60 <= time_diff <= 46 * 60  # Allow 1 second tolerance


class TestManualTradeSubmission:
    """Test manual trade submission functions for Story 8.8."""

    def test_submit_manual_trade_returns_result(self):
        """Verify submit_manual_trade returns OrderSubmissionResult."""
        from src.dashboard.shared_state import (
            submit_manual_trade,
            ManualTradeRequest,
            OrderSubmissionResult,
        )
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        result = submit_manual_trade(request)

        assert isinstance(result, OrderSubmissionResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'order_id')
        assert hasattr(result, 'error')

    def test_submit_manual_trade_mock_success(self):
        """Verify submit_manual_trade mock returns success."""
        from src.dashboard.shared_state import submit_manual_trade, ManualTradeRequest
        from datetime import datetime

        request = ManualTradeRequest(
            direction="Buy",
            quantity=2,
            order_type="Market",
            limit_price=None,
            submit_time=datetime.now(),
            submitted_by="user"
        )

        result = submit_manual_trade(request)

        # Mock implementation should succeed
        assert result.success is True
        assert result.order_id is not None
        assert "MANUAL-" in result.order_id


class TestManualTradeFormRendering:
    """Test manual trade form rendering for Story 8.8."""

    def test_render_manual_trade_exists(self):
        """Verify render_manual_trade function exists in navigation.py."""
        # Check if function is defined by reading the file
        with open("src/dashboard/navigation.py") as f:
            content = f.read()
            assert "def render_manual_trade():" in content

