"""Unit tests for HybridExit class (Task 4)."""

import pytest
from datetime import datetime, timedelta

from src.execution.exit_logic import HybridExit
from src.execution.models import _now, NY_TZ, PositionMonitoringState, TradeOrder
from src.detection.models import EnsembleTradeSignal


class TestHybridExit:
    """Test HybridExit strategy."""

    @pytest.fixture
    def ensemble_signal(self):
        """Create a sample ensemble signal."""
        return EnsembleTradeSignal(
            strategy_name="Ensemble-Weighted Confidence",
            timestamp=_now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
            strategy_confidences={"triple_confluence_scaler": 0.80, "wolf_pack_3_edge": 0.70},
            strategy_weights={"triple_confluence_scaler": 0.20, "wolf_pack_3_edge": 0.20},
            bar_timestamp=_now()
        )

    @pytest.fixture
    def position(self, ensemble_signal):
        """Create a sample long position."""
        entry_time = _now() - timedelta(minutes=3)
        return TradeOrder(
            trade_id="pos-hybrid-test",
            symbol="MNQ",
            direction="long",
            quantity=4,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=4,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=4,
            remaining_quantity=4
        )

    @pytest.fixture
    def hybrid_exit(self):
        """Create a HybridExit instance."""
        return HybridExit(partial_rr=1.5, partial_percent=0.50, max_hold_minutes=10.0)

    def test_initialization(self, hybrid_exit):
        """Test HybridExit initialization."""
        assert hybrid_exit.partial_rr == 1.5
        assert hybrid_exit.partial_percent == 0.50
        assert hybrid_exit.max_hold_minutes == 10.0

    def test_initialization_custom_params(self):
        """Test HybridExit with custom parameters."""
        custom = HybridExit(partial_rr=1.8, partial_percent=0.40, max_hold_minutes=15.0)
        assert custom.partial_rr == 1.8
        assert custom.partial_percent == 0.40
        assert custom.max_hold_minutes == 15.0

    def test_initialization_invalid_partial_rr(self):
        """Test HybridExit initialization with invalid partial_rr."""
        with pytest.raises(ValueError, match="partial_rr must be positive"):
            HybridExit(partial_rr=0)

        with pytest.raises(ValueError, match="partial_rr must be positive"):
            HybridExit(partial_rr=-1.0)

    def test_initialization_invalid_partial_percent(self):
        """Test HybridExit initialization with invalid partial_percent."""
        with pytest.raises(ValueError, match="partial_percent must be between 0 and 1"):
            HybridExit(partial_percent=0)

        with pytest.raises(ValueError, match="partial_percent must be between 0 and 1"):
            HybridExit(partial_percent=1.5)

    def test_initialization_invalid_max_hold(self):
        """Test HybridExit initialization with invalid max_hold_minutes."""
        with pytest.raises(ValueError, match="max_hold_minutes must be positive"):
            HybridExit(max_hold_minutes=0)

        with pytest.raises(ValueError, match="max_hold_minutes must be positive"):
            HybridExit(max_hold_minutes=-5)

    def test_check_exit_no_trigger(self, position, hybrid_exit):
        """Test no exit when price hasn't hit any level."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,  # Below 1.5R (11865) and 2R (11870)
            unrealized_pnl=50.0,
            time_since_entry_seconds=180,  # 3 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = hybrid_exit.check_exit(state)
        assert exit_order is None

    def test_check_exit_partial_triggered(self, position, hybrid_exit):
        """Test partial exit triggered at 1.5R."""
        # 1.5R = 11850 + 10 * 1.5 = 11865
        state = PositionMonitoringState(
            position=position,
            current_price=11865.0,  # Exactly at 1.5R
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        exit_order = hybrid_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_type == "partial"
        assert exit_order.exit_reason == "Hybrid partial (1.5R)"
        # 4 * 0.50 = 2 contracts
        assert exit_order.quantity == 2
        assert exit_order.rr_ratio == 1.5
        # P&L = (11865 - 11850) * 0.50 * 2 = 15 * 0.50 * 2 = $15
        assert exit_order.pnl == pytest.approx(15.0)

    def test_check_exit_partial_above_level(self, position, hybrid_exit):
        """Test partial exit triggered when above 1.5R."""
        state = PositionMonitoringState(
            position=position,
            current_price=11867.0,  # Above 1.5R
            unrealized_pnl=85.0,
            time_since_entry_seconds=180,
            distance_to_tp=3.0,
            distance_to_sl=27.0,
            rr_achieved=1.7
        )

        exit_order = hybrid_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Hybrid partial (1.5R)"

    def test_check_exit_final_take_profit(self, ensemble_signal):
        """Test final exit at 2R take profit."""
        entry_time = _now() - timedelta(minutes=5)
        position = TradeOrder(
            trade_id="pos-hybrid-final",
            symbol="MNQ",
            direction="long",
            quantity=4,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=4,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="partially_closed",  # Already partially closed
            original_quantity=4,
            remaining_quantity=2  # 2 remaining after partial
        )

        hybrid_exit = HybridExit()

        # At 2R TP
        state = PositionMonitoringState(
            position=position,
            current_price=11870.0,
            unrealized_pnl=100.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = hybrid_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_type == "full"
        assert exit_order.exit_reason == "Hybrid trail (2R)"
        assert exit_order.quantity == 2  # Remaining quantity
        # P&L = (11870 - 11850) * 0.50 * 2 = 20 * 0.50 * 2 = $20
        assert exit_order.pnl == pytest.approx(20.0)
        assert exit_order.rr_ratio == pytest.approx(2.0)

    def test_check_exit_time_stop(self, position, hybrid_exit):
        """Test final exit at 10-minute time stop."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,
            unrealized_pnl=50.0,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = hybrid_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_type == "full"
        assert exit_order.exit_reason == "Time stop (10-min max)"
        assert exit_order.quantity == 4

    def test_scale_out_partial_even_quantity(self, position, hybrid_exit):
        """Test partial exit with even quantity."""
        exit_order = hybrid_exit.scale_out_partial(position)

        # 4 * 0.50 = 2 contracts
        assert exit_order.quantity == 2
        assert exit_order.exit_type == "partial"
        assert exit_order.exit_reason == "Hybrid partial (1.5R)"

    def test_scale_out_partial_odd_quantity(self, ensemble_signal):
        """Test partial exit with odd quantity (rounds down)."""
        entry_time = _now() - timedelta(minutes=3)
        position = TradeOrder(
            trade_id="pos-odd-qty",
            symbol="MNQ",
            direction="long",
            quantity=5,  # Odd quantity
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=5,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=5,
            remaining_quantity=5
        )

        hybrid_exit = HybridExit()
        exit_order = hybrid_exit.scale_out_partial(position)

        # 5 * 0.50 = 2.5 → 2 (rounds down)
        assert exit_order.quantity == 2

    def test_scale_out_partial_small_quantity(self, ensemble_signal):
        """Test partial exit with small quantity (minimum 1)."""
        entry_time = _now() - timedelta(minutes=3)
        position = TradeOrder(
            trade_id="pos-small-qty",
            symbol="MNQ",
            direction="long",
            quantity=1,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=1,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=1,
            remaining_quantity=1
        )

        hybrid_exit = HybridExit()
        exit_order = hybrid_exit.scale_out_partial(position)

        # 1 * 0.50 = 0.5 → 1 (minimum 1)
        assert exit_order.quantity == 1

    def test_trail_stop_to_breakeven(self, position, hybrid_exit):
        """Test trailing stop loss to breakeven."""
        old_stop = position.stop_loss
        assert old_stop == 11840.0

        hybrid_exit.trail_stop_to_breakeven(position)

        assert position.stop_loss == position.entry_price
        assert position.stop_loss == 11850.0

    def test_partial_then_final_exit_flow(self, position, hybrid_exit):
        """Test complete hybrid exit flow: partial then final."""
        # Stage 1: Partial exit at 1.5R
        state_partial = PositionMonitoringState(
            position=position,
            current_price=11865.0,  # At 1.5R
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        partial_exit = hybrid_exit.check_exit(state_partial)
        assert partial_exit is not None
        assert partial_exit.exit_reason == "Hybrid partial (1.5R)"
        assert partial_exit.quantity == 2

        # Simulate partial execution (update position)
        position.position_state = "partially_closed"
        position.remaining_quantity = 2
        hybrid_exit.trail_stop_to_breakeven(position)

        # Verify stop moved to breakeven
        assert position.stop_loss == 11850.0

        # Stage 2: Final exit at 2R
        state_final = PositionMonitoringState(
            position=position,
            current_price=11870.0,  # At 2R
            unrealized_pnl=100.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=20.0,
            rr_achieved=2.0
        )

        final_exit = hybrid_exit.check_exit(state_final)
        assert final_exit is not None
        assert final_exit.exit_reason == "Hybrid trail (2R)"
        assert final_exit.quantity == 2  # Remaining quantity

    def test_short_position_partial_exit(self, ensemble_signal):
        """Test hybrid partial exit for short position."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        entry_time = _now() - timedelta(minutes=3)

        position = TradeOrder(
            trade_id="pos-short-hybrid",
            symbol="MNQ",
            direction="short",
            quantity=4,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=signal,
            position_size=4,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="open",
            original_quantity=4,
            remaining_quantity=4
        )

        hybrid_exit = HybridExit()

        # 1.5R for short: 11850 - 15 = 11835
        state = PositionMonitoringState(
            position=position,
            current_price=11835.0,  # At 1.5R
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        exit_order = hybrid_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Hybrid partial (1.5R)"
        assert exit_order.quantity == 2
        # P&L = (11850 - 11835) * 0.50 * 2 = 15 * 0.50 * 2 = $15
        assert exit_order.pnl == pytest.approx(15.0)

    def test_custom_partial_percent(self, position):
        """Test hybrid exit with custom partial percentage."""
        custom = HybridExit(partial_percent=0.40)  # 40% instead of 50%

        state = PositionMonitoringState(
            position=position,
            current_price=11865.0,  # At 1.5R
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        exit_order = custom.check_exit(state)

        # 4 * 0.40 = 1.6 → 1 (rounds down)
        assert exit_order is not None
        assert exit_order.quantity == 1

    def test_partial_exit_pnl_calculation(self, position, hybrid_exit):
        """Test P&L calculation for partial exit."""
        state = PositionMonitoringState(
            position=position,
            current_price=11865.0,
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        exit_order = hybrid_exit.check_exit(state)

        # P&L = (11865 - 11850) * 0.50 * 2 = $15
        assert exit_order.pnl == pytest.approx(15.0)

    def test_final_exit_after_partial_trail(self, position, ensemble_signal):
        """Test final exit calculation after partial with trailed stop."""
        # Create partially closed position
        entry_time = _now() - timedelta(minutes=8)

        partially_closed = TradeOrder(
            trade_id="pos-after-partial",
            symbol="MNQ",
            direction="long",
            quantity=4,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11845.0,  # Slightly above original but still valid
            take_profit=11870.0,
            timestamp=entry_time,
            status="filled",
            ensemble_signal=ensemble_signal,
            position_size=4,
            entry_time=entry_time,
            exit_time=None,
            exit_price=None,
            exit_reason=None,
            hold_time_seconds=None,
            realized_pnl=None,
            rr_achieved=None,
            position_state="partially_closed",
            original_quantity=4,
            remaining_quantity=2
        )

        hybrid_exit = HybridExit()

        # Price between 1.5R and 2R, not at time stop
        state = PositionMonitoringState(
            position=partially_closed,
            current_price=11868.0,  # Close to 2R but not there yet
            unrealized_pnl=90.0,
            time_since_entry_seconds=480,  # 8 minutes
            distance_to_tp=2.0,
            distance_to_sl=23.0,
            rr_achieved=1.8
        )

        exit_order = hybrid_exit.check_exit(state)

        # Should not exit yet (not at 2R and not at time stop)
        assert exit_order is None
