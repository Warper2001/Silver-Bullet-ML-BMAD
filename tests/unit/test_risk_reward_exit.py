"""Unit tests for RiskRewardExit class (Task 3)."""

import pytest
from datetime import datetime, timedelta

from src.execution.exit_logic import RiskRewardExit
from src.execution.models import _now, NY_TZ, PositionMonitoringState, TradeOrder
from src.detection.models import EnsembleTradeSignal


class TestRiskRewardExit:
    """Test RiskRewardExit strategy."""

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
        entry_time = _now() - timedelta(minutes=5)
        return TradeOrder(
            trade_id="pos-rr-test",
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
    def rr_exit(self):
        """Create a RiskRewardExit instance."""
        return RiskRewardExit(rr_ratio=2.0)

    def test_initialization(self, rr_exit):
        """Test RiskRewardExit initialization."""
        assert rr_exit.rr_ratio == 2.0

    def test_initialization_custom_rr(self):
        """Test RiskRewardExit with custom R:R ratio."""
        custom_exit = RiskRewardExit(rr_ratio=2.5)
        assert custom_exit.rr_ratio == 2.5

    def test_initialization_invalid_rr(self):
        """Test RiskRewardExit initialization with invalid R:R."""
        with pytest.raises(ValueError, match="rr_ratio must be positive"):
            RiskRewardExit(rr_ratio=0)

        with pytest.raises(ValueError, match="rr_ratio must be positive"):
            RiskRewardExit(rr_ratio=-1.0)

    def test_calculate_take_profit_long(self, rr_exit):
        """Test take profit calculation for long position."""
        entry = 11850.0
        stop_loss = 11840.0
        risk = 10

        # TP = entry + risk * 2 = 11850 + 20 = 11870
        tp = rr_exit.calculate_take_profit(entry, stop_loss, "long")
        assert tp == pytest.approx(11870.0)

    def test_calculate_take_profit_short(self, rr_exit):
        """Test take profit calculation for short position."""
        entry = 11850.0
        stop_loss = 11860.0
        risk = 10

        # TP = entry - risk * 2 = 11850 - 20 = 11830
        tp = rr_exit.calculate_take_profit(entry, stop_loss, "short")
        assert tp == pytest.approx(11830.0)

    def test_check_take_profit_hit_long(self, position, rr_exit):
        """Test take profit hit detection for long position."""
        # Not at TP
        assert not rr_exit.check_take_profit_hit(11860.0, 11870.0, "long")

        # At TP
        assert rr_exit.check_take_profit_hit(11870.0, 11870.0, "long")

        # Past TP
        assert rr_exit.check_take_profit_hit(11875.0, 11870.0, "long")

    def test_check_take_profit_hit_short(self, rr_exit):
        """Test take profit hit detection for short position."""
        # Not at TP
        assert not rr_exit.check_take_profit_hit(11840.0, 11830.0, "short")

        # At TP
        assert rr_exit.check_take_profit_hit(11830.0, 11830.0, "short")

        # Past TP
        assert rr_exit.check_take_profit_hit(11825.0, 11830.0, "short")

    def test_check_stop_loss_hit_long(self, position, rr_exit):
        """Test stop loss hit detection for long position."""
        # Not at SL
        assert not rr_exit.check_stop_loss_hit(11845.0, 11840.0, "long")

        # At SL
        assert rr_exit.check_stop_loss_hit(11840.0, 11840.0, "long")

        # Past SL
        assert rr_exit.check_stop_loss_hit(11835.0, 11840.0, "long")

    def test_check_stop_loss_hit_short(self, rr_exit):
        """Test stop loss hit detection for short position."""
        # Not at SL
        assert not rr_exit.check_stop_loss_hit(11855.0, 11860.0, "short")

        # At SL
        assert rr_exit.check_stop_loss_hit(11860.0, 11860.0, "short")

        # Past SL
        assert rr_exit.check_stop_loss_hit(11865.0, 11860.0, "short")

    def test_check_exit_no_trigger(self, position, rr_exit):
        """Test no exit triggered when price is between SL and TP."""
        state = PositionMonitoringState(
            position=position,
            current_price=11855.0,  # Between SL (11840) and TP (11870)
            unrealized_pnl=37.5,
            time_since_entry_seconds=300,
            distance_to_tp=15.0,
            distance_to_sl=15.0,
            rr_achieved=0.5
        )

        exit_order = rr_exit.check_exit(state)
        assert exit_order is None

    def test_check_exit_take_profit_triggered(self, position, rr_exit):
        """Test take profit exit triggered."""
        state = PositionMonitoringState(
            position=position,
            current_price=11870.0,  # Exactly at TP
            unrealized_pnl=150.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_type == "full"
        assert exit_order.exit_reason == "Take profit"
        assert exit_order.quantity == 3
        assert exit_order.exit_price == 11870.0
        # P&L = (11870 - 11850) * 0.50 * 3 = 20 * 0.50 * 3 = $30
        assert exit_order.pnl == pytest.approx(30.0)
        assert exit_order.rr_ratio == pytest.approx(2.0)

    def test_check_exit_stop_loss_triggered(self, position, rr_exit):
        """Test stop loss exit triggered."""
        state = PositionMonitoringState(
            position=position,
            current_price=11840.0,  # Exactly at SL
            unrealized_pnl=-75.0,
            time_since_entry_seconds=300,
            distance_to_tp=30.0,
            distance_to_sl=0.0,
            rr_achieved=-1.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_type == "full"
        assert exit_order.exit_reason == "Stop loss"
        assert exit_order.exit_price == 11840.0
        # P&L = (11840 - 11850) * 0.50 * 3 = -10 * 0.50 * 3 = -$15
        assert exit_order.pnl == pytest.approx(-15.0)
        assert exit_order.rr_ratio == -1.0

    def test_stop_loss_priority_over_take_profit(self, position, rr_exit):
        """Test stop loss has higher priority than take profit."""
        # Create position where both SL and TP could be hit
        # In real market, this shouldn't happen, but we test priority
        state = PositionMonitoringState(
            position=position,
            current_price=11840.0,  # At SL, not at TP
            unrealized_pnl=-75.0,
            time_since_entry_seconds=300,
            distance_to_tp=30.0,
            distance_to_sl=0.0,
            rr_achieved=-1.0
        )

        exit_order = rr_exit.check_exit(state)

        # Should trigger stop loss, not take profit
        assert exit_order is not None
        assert exit_order.exit_reason == "Stop loss"

    def test_calculate_rr_achieved_profit(self, rr_exit):
        """Test R:R calculation for profitable exit."""
        entry = 11850.0
        exit_price = 11870.0
        stop_loss = 11840.0

        # (11870 - 11850) / (11850 - 11840) = 20 / 10 = 2.0
        rr = rr_exit.calculate_rr_achieved(entry, exit_price, stop_loss, "long")
        assert rr == pytest.approx(2.0)

    def test_calculate_rr_achieved_loss(self, rr_exit):
        """Test R:R calculation for loss exit."""
        entry = 11850.0
        exit_price = 11840.0
        stop_loss = 11840.0

        # Stopped at SL = -1R
        rr = rr_exit.calculate_rr_achieved(entry, exit_price, stop_loss, "long")
        assert rr == -1.0

    def test_calculate_rr_achieved_partial_profit(self, rr_exit):
        """Test R:R calculation for partial profit (1.5R)."""
        entry = 11850.0
        exit_price = 11865.0
        stop_loss = 11840.0

        # (11865 - 11850) / (11850 - 11840) = 15 / 10 = 1.5
        rr = rr_exit.calculate_rr_achieved(entry, exit_price, stop_loss, "long")
        assert rr == pytest.approx(1.5)

    def test_custom_rr_ratio(self, ensemble_signal):
        """Test custom R:R ratio calculation."""
        entry = 11850.0
        stop_loss = 11840.0
        risk = 10

        # Custom 2.5:1 ratio
        custom_exit = RiskRewardExit(rr_ratio=2.5)
        tp = custom_exit.calculate_take_profit(entry, stop_loss, "long")

        # TP = 11850 + 10 * 2.5 = 11875
        assert tp == pytest.approx(11875.0)

    def test_short_position_take_profit(self, ensemble_signal):
        """Test take profit for short position."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        entry_time = _now() - timedelta(minutes=5)

        position = TradeOrder(
            trade_id="pos-short-tp",
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

        rr_exit = RiskRewardExit()

        # At TP (11830)
        state = PositionMonitoringState(
            position=position,
            current_price=11830.0,
            unrealized_pnl=100.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Take profit"
        # P&L = (11850 - 11830) * 0.50 * 2 = 20 * 0.50 * 2 = $20
        assert exit_order.pnl == pytest.approx(20.0)

    def test_short_position_stop_loss(self, ensemble_signal):
        """Test stop loss for short position."""
        signal = ensemble_signal.model_copy(update={"direction": "short"})
        entry_time = _now() - timedelta(minutes=5)

        position = TradeOrder(
            trade_id="pos-short-sl",
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

        rr_exit = RiskRewardExit()

        # At SL (11860)
        state = PositionMonitoringState(
            position=position,
            current_price=11860.0,
            unrealized_pnl=-50.0,
            time_since_entry_seconds=300,
            distance_to_tp=30.0,
            distance_to_sl=0.0,
            rr_achieved=-1.0
        )

        exit_order = rr_exit.check_exit(state)

        assert exit_order is not None
        assert exit_order.exit_reason == "Stop loss"
        # P&L = (11850 - 11860) * 0.50 * 2 = -10 * 0.50 * 2 = -$10
        assert exit_order.pnl == pytest.approx(-10.0)
        assert exit_order.rr_ratio == -1.0

    def test_partial_position_rr_exit(self, position, rr_exit):
        """Test R:R exit with partially closed position."""
        # Update position to be partially closed
        position.remaining_quantity = 2
        position.position_state = "partially_closed"

        state = PositionMonitoringState(
            position=position,
            current_price=11870.0,  # At TP
            unrealized_pnl=100.0,
            time_since_entry_seconds=300,
            distance_to_tp=0.0,
            distance_to_sl=30.0,
            rr_achieved=2.0
        )

        exit_order = rr_exit.check_exit(state)

        # Should close remaining quantity (2), not original (3)
        assert exit_order is not None
        assert exit_order.quantity == 2
        # P&L = (11870 - 11850) * 0.50 * 2 = 20 * 0.50 * 2 = $20
        assert exit_order.pnl == pytest.approx(20.0)
