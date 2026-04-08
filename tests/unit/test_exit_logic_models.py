"""Unit tests for exit logic models (Task 1)."""

import pytest
from datetime import datetime, timedelta
from src.execution.models import TradeOrder, ExitOrder, PositionMonitoringState
from src.detection.models import EnsembleTradeSignal


class TestTradeOrderExtensions:
    """Test TradeOrder model extensions for exit tracking."""

    @pytest.fixture
    def ensemble_signal(self):
        """Create a sample ensemble signal for testing."""
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
    def trade_order(self, ensemble_signal):
        """Create a sample trade order for testing."""
        return TradeOrder(
            trade_id="test-trade-123",
            symbol="MNQ",
            direction="long",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=datetime.now(),
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=3,
            entry_time=datetime.now(),
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

    def test_trade_order_with_exit_fields(self, trade_order):
        """Test trade order with exit tracking fields."""
        assert trade_order.position_state == "open"
        assert trade_order.original_quantity == 3
        assert trade_order.remaining_quantity == 3
        assert trade_order.exit_time is None
        assert trade_order.exit_price is None
        assert trade_order.exit_reason is None

    def test_hold_time_minutes(self, trade_order):
        """Test hold time calculation."""
        # Not exited yet
        assert trade_order.hold_time_minutes() == 0.0

        # Exited after 5 minutes
        trade_order.exit_time = trade_order.entry_time + timedelta(minutes=5)
        trade_order.hold_time_seconds = 300
        assert trade_order.hold_time_minutes() == 5.0

    def test_is_held_max_time_not_exceeded(self, trade_order):
        """Test max hold time check when not exceeded."""
        # Position just entered
        assert not trade_order.is_held_max_time(max_hold_minutes=10.0)

    def test_is_held_max_time_exceeded(self, trade_order):
        """Test max hold time check when exceeded."""
        # Position held for 11 minutes
        trade_order.exit_time = trade_order.entry_time + timedelta(minutes=11)
        trade_order.hold_time_seconds = 660
        assert trade_order.is_held_max_time(max_hold_minutes=10.0)

    def test_is_held_max_time_exactly_at_limit(self, trade_order):
        """Test max hold time check at exactly 10 minutes."""
        # Position held for exactly 10 minutes
        trade_order.exit_time = trade_order.entry_time + timedelta(minutes=10)
        trade_order.hold_time_seconds = 600
        assert trade_order.is_held_max_time(max_hold_minutes=10.0)

    def test_is_at_take_profit_long(self, trade_order):
        """Test take profit detection for long position."""
        # Not at TP yet
        assert not trade_order.is_at_take_profit(11860.0)

        # At TP
        assert trade_order.is_at_take_profit(11870.0)

        # Past TP
        assert trade_order.is_at_take_profit(11875.0)

    def test_is_at_take_profit_short(self, ensemble_signal):
        """Test take profit detection for short position."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        order = TradeOrder(
            trade_id="test-short",
            symbol="MNQ",
            direction="short",
            quantity=2,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=datetime.now(),
            status="filled",
            ensemble_signal=signal,
            position_size=2,
            entry_time=datetime.now(),
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

        # Not at TP yet
        assert not order.is_at_take_profit(11840.0)

        # At TP
        assert order.is_at_take_profit(11830.0)

        # Past TP
        assert order.is_at_take_profit(11825.0)

    def test_is_at_stop_loss_long(self, trade_order):
        """Test stop loss detection for long position."""
        # Not at SL yet
        assert not trade_order.is_at_stop_loss(11845.0)

        # At SL
        assert trade_order.is_at_stop_loss(11840.0)

        # Past SL
        assert trade_order.is_at_stop_loss(11835.0)

    def test_is_at_stop_loss_short(self, ensemble_signal):
        """Test stop loss detection for short position."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        order = TradeOrder(
            trade_id="test-short",
            symbol="MNQ",
            direction="short",
            quantity=2,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=datetime.now(),
            status="filled",
            ensemble_signal=signal,
            position_size=2,
            entry_time=datetime.now(),
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

        # Not at SL yet
        assert not order.is_at_stop_loss(11855.0)

        # At SL
        assert order.is_at_stop_loss(11860.0)

        # Past SL
        assert order.is_at_stop_loss(11865.0)

    def test_is_at_hybrid_partial_long(self, trade_order):
        """Test hybrid partial level detection for long position."""
        # Risk = 10 points, 1.5R = 15 points, target = 11850 + 15 = 11865
        # Not at partial yet
        assert not trade_order.is_at_hybrid_partial(11860.0)

        # At partial level
        assert trade_order.is_at_hybrid_partial(11865.0)

        # Past partial level
        assert trade_order.is_at_hybrid_partial(11870.0)

    def test_is_at_hybrid_partial_short(self, ensemble_signal):
        """Test hybrid partial level detection for short position."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        order = TradeOrder(
            trade_id="test-short",
            symbol="MNQ",
            direction="short",
            quantity=2,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=datetime.now(),
            status="filled",
            ensemble_signal=signal,
            position_size=2,
            entry_time=datetime.now(),
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

        # Risk = 10 points, 1.5R = 15 points, target = 11850 - 15 = 11835
        # Not at partial yet
        assert not order.is_at_hybrid_partial(11840.0)

        # At partial level
        assert order.is_at_hybrid_partial(11835.0)

        # Past partial level
        assert order.is_at_hybrid_partial(11830.0)

    def test_remaining_quantity_validation(self, trade_order, ensemble_signal):
        """Test remaining quantity cannot exceed original."""
        with pytest.raises(ValueError, match="remaining_quantity cannot exceed original_quantity"):
            TradeOrder(
                trade_id="test-trade-123",
                symbol="MNQ",
                direction="long",
                quantity=3,
                order_type="market",
                entry_price=11850.0,
                limit_price=None,
                stop_loss=11840.0,
                take_profit=11870.0,
                timestamp=datetime.now(),
                status="filled",
                ensemble_signal=ensemble_signal,
                position_size=3,
                entry_time=datetime.now(),
                exit_time=None,
                exit_price=None,
                exit_reason=None,
                hold_time_seconds=None,
                realized_pnl=None,
                rr_achieved=None,
                position_state="open",
                original_quantity=3,
                remaining_quantity=5  # Invalid: exceeds original
            )

    def test_position_state_transitions(self, trade_order):
        """Test position state transitions."""
        # Open
        assert trade_order.position_state == "open"

        # Partially closed
        trade_order.position_state = "partially_closed"
        assert trade_order.position_state == "partially_closed"

        # Closed
        trade_order.position_state = "closed"
        assert trade_order.position_state == "closed"


class TestExitOrder:
    """Test ExitOrder model."""

    def test_create_full_exit_order(self):
        """Test creating a full exit order."""
        exit_order = ExitOrder(
            position_id="pos-123",
            exit_type="full",
            quantity=3,
            exit_price=11870.0,
            exit_reason="take_profit",
            timestamp=datetime.now(),
            pnl=150.0,
            rr_ratio=2.0
        )

        assert exit_order.position_id == "pos-123"
        assert exit_order.exit_type == "full"
        assert exit_order.quantity == 3
        assert exit_order.exit_reason == "take_profit"
        assert exit_order.pnl == 150.0
        assert exit_order.rr_ratio == 2.0

    def test_create_partial_exit_order(self):
        """Test creating a partial exit order."""
        exit_order = ExitOrder(
            position_id="pos-123",
            exit_type="partial",
            quantity=2,
            exit_price=11865.0,
            exit_reason="hybrid_partial",
            timestamp=datetime.now(),
            pnl=75.0,
            rr_ratio=1.5
        )

        assert exit_order.exit_type == "partial"
        assert exit_order.exit_reason == "hybrid_partial"
        assert exit_order.quantity == 2

    def test_hybrid_partial_requires_partial_exit_type(self):
        """Test that hybrid_partial exit reason requires partial exit type."""
        with pytest.raises(ValueError, match="hybrid_partial exit_reason requires partial exit_type"):
            ExitOrder(
                position_id="pos-123",
                exit_type="full",
                quantity=3,
                exit_price=11865.0,
                exit_reason="hybrid_partial",
                timestamp=datetime.now(),
                pnl=75.0,
                rr_ratio=1.5
            )

    def test_take_profit_requires_full_exit_type(self):
        """Test that take_profit exit reason requires full exit type."""
        with pytest.raises(ValueError, match="take_profit exit_reason requires full exit_type"):
            ExitOrder(
                position_id="pos-123",
                exit_type="partial",
                quantity=1,
                exit_price=11870.0,
                exit_reason="take_profit",
                timestamp=datetime.now(),
                pnl=50.0,
                rr_ratio=2.0
            )

    def test_stop_loss_requires_full_exit_type(self):
        """Test that stop_loss exit reason requires full exit type."""
        with pytest.raises(ValueError, match="stop_loss exit_reason requires full exit_type"):
            ExitOrder(
                position_id="pos-123",
                exit_type="partial",
                quantity=1,
                exit_price=11840.0,
                exit_reason="stop_loss",
                timestamp=datetime.now(),
                pnl=-50.0,
                rr_ratio=-1.0
            )

    def test_time_stop_requires_full_exit_type(self):
        """Test that time_stop exit reason requires full exit type."""
        with pytest.raises(ValueError, match="time_stop exit_reason requires full exit_type"):
            ExitOrder(
                position_id="pos-123",
                exit_type="partial",
                quantity=1,
                exit_price=11850.0,
                exit_reason="time_stop",
                timestamp=datetime.now(),
                pnl=0.0,
                rr_ratio=0.0
            )


class TestPositionMonitoringState:
    """Test PositionMonitoringState model."""

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
        return TradeOrder(
            trade_id="pos-123",
            symbol="MNQ",
            direction="long",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=datetime.now(),
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=3,
            entry_time=datetime.now(),
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

    def test_create_monitoring_state(self, position):
        """Test creating a position monitoring state."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=180,
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        assert state.position.trade_id == "pos-123"
        assert state.current_price == 11855.0
        assert state.unrealized_pnl == 37.5
        assert state.time_since_entry_seconds == 180

    def test_hold_time_minutes(self, position):
        """Test hold time calculation in monitoring state."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=300,  # 5 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        assert state.hold_time_minutes() == 5.0

    def test_is_at_max_hold_time_not_exceeded(self, position):
        """Test max hold time check when not exceeded."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=540,  # 9 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        assert not state.is_at_max_hold_time(max_hold_minutes=10.0)

    def test_is_at_max_hold_time_exceeded(self, position):
        """Test max hold time check when exceeded."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        assert state.is_at_max_hold_time(max_hold_minutes=10.0)

    def test_is_at_max_hold_time_exactly_at_limit(self, position):
        """Test max hold time check at exactly limit."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=600,  # 10 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        assert state.is_at_max_hold_time(max_hold_minutes=10.0)
