"""Unit tests for entry logic models and position sizing."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.execution.models import TradeOrder, EntryDecision
from src.detection.models import EnsembleTradeSignal


class TestTradeOrderModel:
    """Test TradeOrder Pydantic model."""

    def test_create_trade_order_long(self):
        """Test creating a long trade order."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
            strategy_confidences={"triple_confluence_scaler": 0.80, "wolf_pack_3_edge": 0.70},
            strategy_weights={"triple_confluence_scaler": 0.50, "wolf_pack_3_edge": 0.50},
            bar_timestamp=datetime.now()
        )

        order = TradeOrder(
            trade_id="test-trade-001",
            symbol="MNQ",
            direction="long",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=datetime.now(),
            ensemble_signal=signal,
            position_size=3
        )

        assert order.trade_id == "test-trade-001"
        assert order.symbol == "MNQ"
        assert order.direction == "long"
        assert order.quantity == 3
        assert order.status == "pending"

    def test_create_trade_order_short(self):
        """Test creating a short trade order."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="short",
            entry_price=11850.0,
            stop_loss=11860.0,
            take_profit=11830.0,
            composite_confidence=0.65,
            contributing_strategies=["adaptive_ema_momentum"],
            strategy_confidences={"adaptive_ema_momentum": 0.65},
            strategy_weights={"adaptive_ema_momentum": 1.0},
            bar_timestamp=datetime.now()
        )

        order = TradeOrder(
            trade_id="test-trade-002",
            symbol="MNQ",
            direction="short",
            quantity=2,
            order_type="limit",
            entry_price=11850.0,
            limit_price=11850.0,
            stop_loss=11860.0,
            take_profit=11830.0,
            timestamp=datetime.now(),
            ensemble_signal=signal,
            position_size=2
        )

        assert order.direction == "short"
        assert order.limit_price == 11850.0

    def test_quantity_must_be_1_to_5(self):
        """Test quantity validation (1-5 contracts)."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler"],
            strategy_confidences={"triple_confluence_scaler": 0.75},
            strategy_weights={"triple_confluence_scaler": 1.0},
            bar_timestamp=datetime.now()
        )

        # Quantity too low
        with pytest.raises(ValidationError):
            TradeOrder(
                trade_id="test",
                symbol="MNQ",
                direction="long",
                quantity=0,
                order_type="market",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                timestamp=datetime.now(),
                ensemble_signal=signal,
                position_size=0
            )

        # Quantity too high
        with pytest.raises(ValidationError):
            TradeOrder(
                trade_id="test",
                symbol="MNQ",
                direction="long",
                quantity=6,
                order_type="market",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                timestamp=datetime.now(),
                ensemble_signal=signal,
                position_size=6
            )

    def test_notional_value_calculation(self):
        """Test notional value calculation helper method."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler"],
            strategy_confidences={"triple_confluence_scaler": 0.75},
            strategy_weights={"triple_confluence_scaler": 1.0},
            bar_timestamp=datetime.now()
        )

        order = TradeOrder(
            trade_id="test",
            symbol="MNQ",
            direction="long",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=datetime.now(),
            ensemble_signal=signal,
            position_size=3
        )

        # MNQ multiplier is $0.50 per point
        # 11850 * $0.50 * 3 = $17,775
        assert order.notional_value() == 17775.0

    def test_risk_per_contract_calculation(self):
        """Test risk per contract calculation helper method."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler"],
            strategy_confidences={"triple_confluence_scaler": 0.75},
            strategy_weights={"triple_confluence_scaler": 1.0},
            bar_timestamp=datetime.now()
        )

        order = TradeOrder(
            trade_id="test",
            symbol="MNQ",
            direction="long",
            quantity=3,
            order_type="market",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            timestamp=datetime.now(),
            ensemble_signal=signal,
            position_size=3
        )

        # Risk = entry - stop_loss = 11850 - 11840 = 10 points
        assert order.risk_per_contract() == 10.0


class TestEntryDecisionModel:
    """Test EntryDecision Pydantic model."""

    def test_create_accepted_entry_decision(self):
        """Test creating an accepted entry decision."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler"],
            strategy_confidences={"triple_confluence_scaler": 0.75},
            strategy_weights={"triple_confluence_scaler": 1.0},
            bar_timestamp=datetime.now()
        )

        decision = EntryDecision(
            signal=signal,
            position_size=3,
            risk_checks_passed=True,
            risk_check_details={
                "daily_pnl": {"current": -200.0, "limit": -1000.0, "passed": True},
                "drawdown": {"current": 0.05, "limit": 0.12, "passed": True},
                "open_positions": {"current": 2, "limit": 5, "passed": True},
                "stop_loss_defined": {"defined": True, "passed": True}
            },
            decision="ACCEPT",
            rejection_reason=None,
            timestamp=datetime.now()
        )

        assert decision.decision == "ACCEPT"
        assert decision.position_size == 3
        assert decision.risk_checks_passed is True
        assert decision.rejection_reason is None

    def test_create_rejected_entry_decision(self):
        """Test creating a rejected entry decision."""
        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,
            contributing_strategies=["triple_confluence_scaler"],
            strategy_confidences={"triple_confluence_scaler": 0.75},
            strategy_weights={"triple_confluence_scaler": 1.0},
            bar_timestamp=datetime.now()
        )

        decision = EntryDecision(
            signal=signal,
            position_size=0,
            risk_checks_passed=False,
            risk_check_details={
                "daily_pnl": {"current": -1200.0, "limit": -1000.0, "passed": False},
                "drawdown": {"current": 0.05, "limit": 0.12, "passed": True},
                "open_positions": {"current": 2, "limit": 5, "passed": True},
                "stop_loss_defined": {"defined": True, "passed": True}
            },
            decision="REJECT",
            rejection_reason="Daily loss limit exceeded",
            timestamp=datetime.now()
        )

        assert decision.decision == "REJECT"
        assert decision.position_size == 0
        assert decision.risk_checks_passed is False
        assert decision.rejection_reason == "Daily loss limit exceeded"
