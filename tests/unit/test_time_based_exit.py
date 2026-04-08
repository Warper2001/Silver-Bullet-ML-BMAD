"""Unit tests for TimeBasedExit class (Task 2)."""

import pytest
from datetime import datetime, timedelta

from src.execution.exit_logic import TimeBasedExit
from src.execution.models import PositionMonitoringState, TradeOrder
from src.detection.models import EnsembleTradeSignal


class TestTimeBasedExit:
    """Test TimeBasedExit strategy."""

    @pytest.fixture
    def ensemble_signal(self):
        """Create a sample ensemble signal."""
        return EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
            strategy_confidences={"triple_confluence_scaler": 0.80, "wolf_pack_3_edge": 0.70},
            strategy_weights={"triple_confluence_scaler": 0.20, "wolf_pack_3_edge": 0.20},
            bar_timestamp=datetime.now()
        )

    @pytest.fixture
    def position(self, ensemble_signal):
        """Create a sample position."""
        entry_time = datetime.now() - timedelta(minutes=8)
        return TradeOrder(
            trade_id="pos-time-test",
            symbol="MNQ",
            direction="long",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=3,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=3,
            remaining_quantity=3
        )

    @pytest.fixture
    def time_exit(self):
        """Create a TimeBasedExit instance."""
        return TimeBasedExit(max_hold_minutes=10.0)

    def test_initialization(self, time_exit):
        """Test TimeBasedExit initialization."""
        assert time_exit.max_hold_minutes == 10.0

    def test_initialization_custom_max_hold(self):
        """Test TimeBasedExit with custom max hold time."""
        custom_exit = TimeBasedExit(max_hold_minutes=15.0)
        assert custom_exit.max_hold_minutes == 15.0

    def test_initialization_invalid_max_hold(self):
        """Test TimeBasedExit initialization with invalid max hold time."""
        with pytest.raises(ValueError, match="max_hold_minutes must be positive"):
            TimeBasedExit(max_hold_minutes=0)

        with pytest.raises(ValueError, match="max_hold_minutes must be positive"):
            TimeBasedExit(max_hold_minutes=-5)

    def test_check_exit_not_triggered(self, position, time_exit):
        """Test exit not triggered when hold time below max."""
        # Position held for 8 minutes, below 10-minute max
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=480,  # 8 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = time_exit.check_exit(state)
        assert exit_order is None

    def test_check_exit_triggered_at_limit(self, position, time_exit):
        """Test exit triggered exactly at 10-minute limit."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=600,  # Exactly 10 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = time_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_type == "full"
        assert exit_order.quantity == 3
        assert exit_order.exit_reason == "time_stop"
        assert exit_order.position_id == "pos-time-test"

    def test_check_exit_triggered_above_limit(self, position, time_exit):
        """Test exit triggered when above 10-minute limit."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = time_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "time_stop"

    def test_calculate_hold_time(self, time_exit):
        """Test hold time calculation."""
        entry_time = datetime.now() - timedelta(minutes=5, seconds=30)
        current_time = datetime.now()

        hold_time_seconds = time_exit.calculate_hold_time(entry_time, current_time)

        # Should be approximately 330 seconds (5.5 minutes)
        assert 325 <= hold_time_seconds <= 335

    def test_get_hold_time_minutes(self, position, time_exit):
        """Test getting hold time in minutes."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=300,  # 5 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        hold_time_minutes = time_exit.get_hold_time_minutes(state)
        assert hold_time_minutes == 5.0

    def test_exit_order_pnl_calculation_long(self, ensemble_signal):
        """Test P&L calculation for long position exit."""
        entry_time = datetime.now() - timedelta(minutes=11)
        position = TradeOrder(
            trade_id="pos-pnl-long",
            symbol="MNQ",
            direction="long",
            quantity=2,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=2,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=2,
            remaining_quantity=2
        )

        time_exit = TimeBasedExit()

        # Price moved up 10 points
        state = PositionMonitoringState(
            position=position,
            current_price=11860.0,
            unrealized_pnl=0,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=10.0,
            distance_to_sl=20.0,
            rr_achieved=1.0
        )

        exit_order = time_exit.check_exit(state)

        # P&L = (11860 - 11850) * 0.50 * 2 = 10 * 0.50 * 2 = $10
        assert exit_order is not None
        assert exit_order.pnl == pytest.approx(10.0)

    def test_exit_order_pnl_calculation_short(self, ensemble_signal):
        """Test P&L calculation for short position exit."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        entry_time = datetime.now() - timedelta(minutes=11)

        position = TradeOrder(
            trade_id="pos-pnl-short",
            symbol="MNQ",
            direction="short",
            quantity=2,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=signal,
            position_size=2,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=2,
            remaining_quantity=2
        )

        time_exit = TimeBasedExit()

        # Price moved down 10 points (profit for short)
        state = PositionMonitoringState(
            position=position,
            current_price=11840.0,
            unrealized_pnl=0,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=10.0,
            distance_to_sl=20.0,
            rr_achieved=1.0
        )

        exit_order = time_exit.check_exit(state)

        # P&L = (11850 - 11840) * 0.50 * 2 = 10 * 0.50 * 2 = $10
        assert exit_order is not None
        assert exit_order.pnl == pytest.approx(10.0)

    def test_exit_order_rr_calculation(self, position, time_exit):
        """Test R:R calculation in exit order."""
        state = PositionMonitoringState(
            position=position,
            current_price=11860.0,  # 10 points profit, 1R achieved
            unrealized_pnl=0,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=10.0,
            distance_to_sl=20.0,
            rr_achieved=1.0
        )

        exit_order = time_exit.check_exit(state)

        # R:R = 10 points / 10 points risk = 1.0
        assert exit_order is not None
        assert exit_order.rr_ratio == pytest.approx(1.0)

    def test_custom_max_hold_time(self, position):
        """Test TimeBasedExit with custom max hold time."""
        custom_exit = TimeBasedExit(max_hold_minutes=5.0)

        # Position held for 6 minutes, above 5-minute custom max
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=360,  # 6 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = custom_exit.check_exit(state)
        assert exit_order is not None
        assert exit_order.exit_reason == "time_stop"

    def test_partial_position_state(self, position, time_exit):
        """Test time exit with partially closed position."""
        # Update position to be partially closed
        position.remaining_quantity = 2
        position.position_state = "partially_closed"

        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = time_exit.check_exit(state)

        # Should close remaining quantity (2), not original (3)
        assert exit_order is not None
        assert exit_order.quantity == 2
