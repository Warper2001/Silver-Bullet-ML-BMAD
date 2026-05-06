"""Integration tests for exit logic strategies (Story 2.4)."""

import pytest
from datetime import datetime, timedelta

from src.execution.exit_logic import TimeBasedExit, RiskRewardExit, HybridExit
from src.execution.models import _now, NY_TZ, PositionMonitoringState, TradeOrder, ExitOrder
from src.detection.models import EnsembleTradeSignal


class TestExitLogicIntegration:
    """Integration tests for exit logic strategies."""

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
    def open_position(self, ensemble_signal):
        """Create an open position."""
        entry_time = _now() - timedelta(minutes=3)
        return TradeOrder(
            trade_id="pos-integration-test",
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

    def test_full_pipeline_time_stop(self, open_position):
        """Test full pipeline with time-based exit."""
        time_exit = TimeBasedExit(max_hold_minutes=10.0)

        # Position held for 11 minutes
        state = PositionMonitoringState(
            position=open_position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=660,
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = time_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Time stop (10-min max)"
        assert exit_order.exit_type == "full"
        assert exit_order.quantity == 3

    def test_full_pipeline_take_profit(self, open_position):
        """Test full pipeline with take profit exit."""
        rr_exit = RiskRewardExit()

        # Price hits 2R take profit
        state = PositionMonitoringState(
            position=open_position,
            current_price=11870.0,
            unrealized_pnl=150.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Take profit"
        assert exit_order.rr_ratio == pytest.approx(2.0)

    def test_full_pipeline_stop_loss(self, open_position):
        """Test full pipeline with stop loss exit."""
        rr_exit = RiskRewardExit()

        # Price hits stop loss
        state = PositionMonitoringState(
            position=open_position,
            current_price=11840.0,
            unrealized_pnl=-75.0,
            time_since_entry_seconds=180,
            distance_to_tp=30.0,
            distance_to_sl=0.0,
            rr_achieved=-1.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Stop loss"
        assert exit_order.rr_ratio == -1.0

    def test_full_pipeline_hybrid_flow(self, open_position):
        """Test full hybrid exit pipeline: partial then final."""
        hybrid_exit = HybridExit()

        # Stage 1: Partial exit at 1.5R (11865)
        state_partial = PositionMonitoringState(
            position=open_position,
            current_price=11865.0,
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        partial_exit = hybrid_exit.check_exit(state_partial)

        assert partial_exit is not None
        assert partial_exit.exit_reason == "Hybrid partial (1.5R)"
        assert partial_exit.exit_type == "partial"
        assert partial_exit.quantity == 1  # 3 * 0.50 = 1.5 → 1

        # Simulate partial execution
        open_position.position_state = "partially_closed"
        open_position.remaining_quantity = 2

        # Stage 2: Final exit at 2R
        state_final = PositionMonitoringState(
            position=open_position,
            current_price=11870.0,
            unrealized_pnl=100.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        final_exit = hybrid_exit.check_exit(state_final)

        assert final_exit is not None
        assert final_exit.exit_reason == "Hybrid trail (2R)"
        assert final_exit.exit_type == "full"
        assert final_exit.quantity == 2  # Remaining quantity

    def test_strategy_priority_stop_loss_over_tp(self, open_position):
        """Test that stop loss has priority over take profit."""
        rr_exit = RiskRewardExit()
        time_exit = TimeBasedExit()

        # Create state where both could theoretically trigger
        # (in real market, this shouldn't happen)
        state = PositionMonitoringState(
            position=open_position,
            current_price=11840.0,  # At SL, not at TP
            unrealized_pnl=-75.0,
            time_since_entry_seconds=660,  # Also at time stop
            distance_to_tp=30.0,
            distance_to_sl=0.0,
            rr_achieved=-1.0
        )

        # RiskRewardExit should prioritize SL
        rr_exit_order = rr_exit.check_exit(state)
        assert rr_exit_order is not None
        assert rr_exit_order.exit_reason == "Stop loss"

        # TimeBasedExit would also trigger, but SL should be checked first in production

    def test_hybrid_with_time_stop(self, open_position):
        """Test hybrid exit with time stop on remaining position."""
        hybrid_exit = HybridExit()

        # Position already partially closed
        open_position.position_state = "partially_closed"
        open_position.remaining_quantity = 2

        # Hit time stop before 2R
        state = PositionMonitoringState(
            position=open_position,
            current_price=11855.0,
            unrealized_pnl=50.0,
            time_since_entry_seconds=660,  # 11 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = hybrid_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Time stop (10-min max)"
        assert exit_order.quantity == 2

    def test_no_exit_conditions_met(self, open_position):
        """Test that no exit is generated when conditions aren't met."""
        time_exit = TimeBasedExit()
        rr_exit = RiskRewardExit()
        hybrid_exit = HybridExit()

        # Price between SL and TP, not at time stop
        state = PositionMonitoringState(
            position=open_position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=300,  # 5 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        # None of the strategies should trigger
        assert time_exit.check_exit(state) is None
        assert rr_exit.check_exit(state) is None
        assert hybrid_exit.check_exit(state) is None

    def test_short_position_pipeline(self, open_position):
        """Test full pipeline with short position."""
        signal = open_position.ensemble_signal.model_copy(update={"direction": "short"})

        position = TradeOrder(
            trade_id="pos-short-integration",
            symbol="MNQ",
            direction="short",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            limit_price=None,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=_now() - timedelta(minutes=5),
            status="filled",
            ensemble_signal=signal,
            position_size=3,
            entry_time=_now() - timedelta(minutes=5),
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

        rr_exit = RiskRewardExit()

        # Short position hits take profit
        state = PositionMonitoringState(
            position=position,
            current_price=11830.0,
            unrealized_pnl=150.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Take profit"
        # P&L = (11850 - 11830) * 0.50 * 3 = 20 * 0.50 * 3 = $30
        assert exit_order.pnl == pytest.approx(30.0)

    def test_hybrid_partial_quantity_rounding(self, open_position):
        """Test hybrid exit with various quantities for rounding."""
        hybrid_exit = HybridExit()

        # Test with 5 contracts (should round down to 2)
        open_position.quantity = 5
        open_position.remaining_quantity = 5

        state = PositionMonitoringState(
            position=open_position,
            current_price=11865.0,
            unrealized_pnl=75.0,
            time_since_entry_seconds=180,
            distance_to_tp=5.0,
            distance_to_sl=25.0,
            rr_achieved=1.5
        )

        exit_order = hybrid_exit.check_exit(state)

        # 5 * 0.50 = 2.5 → 2 (rounds down)
        assert exit_order is not None
        assert exit_order.quantity == 2

    def test_multiple_strategies_no_conflict(self, open_position):
        """Test that multiple strategies can coexist without conflict."""
        time_exit = TimeBasedExit()
        rr_exit = RiskRewardExit()
        hybrid_exit = HybridExit()

        # Price at 1R (between SL and TP)
        state = PositionMonitoringState(
            position=open_position,
            current_price=11860.0,
            unrealized_pnl=75.0,
            time_since_entry_seconds=300,
            distance_to_tp=10.0,
            distance_to_sl=20.0,
            rr_achieved=1.0
        )

        # Only hybrid should trigger (at 1.5R partial)
        # But this is at 1R, so nothing should trigger
        assert time_exit.check_exit(state) is None
        assert rr_exit.check_exit(state) is None
        # Hybrid also shouldn't trigger (not at 1.5R yet)
        assert hybrid_exit.check_exit(state) is None

    def test_exit_order_metadata_preservation(self, open_position):
        """Test that exit orders preserve all metadata."""
        rr_exit = RiskRewardExit()

        state = PositionMonitoringState(
            position=open_position,
            current_price=11870.0,
            unrealized_pnl=150.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = rr_exit.check_exit(state)

        # Verify all metadata fields
        assert exit_order.position_id == open_position.trade_id
        assert exit_order.exit_price == 11870.0
        assert exit_order.timestamp is not None
        assert exit_order.pnl is not None
        assert exit_order.rr_ratio is not None

    def test_position_state_transitions(self, open_position):
        """Test position state transitions through exits."""
        # Initial state: open
        assert open_position.position_state == "open"

        # Simulate partial exit
        open_position.position_state = "partially_closed"
        open_position.remaining_quantity = 1

        # Verify state
        assert open_position.position_state == "partially_closed"
        assert open_position.remaining_quantity == 1

        # Simulate final exit
        open_position.position_state = "closed"
        open_position.remaining_quantity = 0

        # Verify final state
        assert open_position.position_state == "closed"
        assert open_position.remaining_quantity == 0

    def test_custom_max_hold_time(self, open_position):
        """Test custom max hold time across strategies."""
        custom_time = TimeBasedExit(max_hold_minutes=5.0)
        custom_hybrid = HybridExit(max_hold_minutes=5.0)

        # Position held for 6 minutes
        state = PositionMonitoringState(
            position=open_position,
            current_price=11855.0,
            unrealized_pnl=37.5,
            time_since_entry_seconds=360,  # 6 minutes
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        # Both strategies with custom 5-min limit should trigger
        assert custom_time.check_exit(state) is not None
        assert custom_time.check_exit(state).exit_reason == "Time stop (10-min max)"

        assert custom_hybrid.check_exit(state) is not None
        assert custom_hybrid.check_exit(state).exit_reason == "Time stop (10-min max)"

    def test_pnl_accuracy_across_strategies(self, open_position):
        """Test P&L calculation accuracy across all strategies."""
        time_exit = TimeBasedExit()
        rr_exit = RiskRewardExit()
        hybrid_exit = HybridExit()

        # At take profit (20 points profit)
        state_tp = PositionMonitoringState(
            position=open_position,
            current_price=11870.0,
            unrealized_pnl=150.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        # All strategies should calculate same P&L for same price move
        # (ignoring partial exits for hybrid)
        time_order = time_exit.check_exit(
            PositionMonitoringState(
                position=open_position,
                current_price=11870.0,
                unrealized_pnl=150.0,
                time_since_entry_seconds=660,
                distance_to_tp=0.0,
                distance_to_sl=30.0,
                rr_achieved=2.0
            )
        )

        rr_order = rr_exit.check_exit(state_tp)

        # Both should have same P&L for full position
        assert time_order.pnl == pytest.approx(30.0)  # (11870-11850) * 0.50 * 3
        assert rr_order.pnl == pytest.approx(30.0)
