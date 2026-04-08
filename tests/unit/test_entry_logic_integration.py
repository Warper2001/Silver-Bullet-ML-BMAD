"""Unit tests for risk validation and entry logic integration."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.execution.entry_logic import PositionSizer, RiskValidator, EntryLogic
from src.execution.models import EntryDecision, TradeOrder
from src.detection.models import EnsembleTradeSignal


class TestRiskValidator:
    """Test RiskValidator integration."""

    def test_validate_all_checks_passed(self):
        """Test validation when all risk checks pass."""
        # Create mock risk orchestrator
        mock_orchestrator = Mock()
        validator = RiskValidator(mock_orchestrator)

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

        result = validator.validate_entry(
            signal=signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        assert result["risk_checks_passed"] is True
        assert result["risk_check_details"]["daily_pnl"]["passed"] is True
        assert result["risk_check_details"]["drawdown"]["passed"] is True
        assert result["risk_check_details"]["open_positions"]["passed"] is True
        assert result["risk_check_details"]["stop_loss_defined"]["passed"] is True

    def test_validate_daily_loss_limit_failed(self):
        """Test validation when daily loss limit is exceeded."""
        mock_orchestrator = Mock()
        validator = RiskValidator(mock_orchestrator)

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

        result = validator.validate_entry(
            signal=signal,
            current_pnl=-1200.0,  # Exceeds $1000 limit
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        assert result["risk_checks_passed"] is False
        assert result["risk_check_details"]["daily_pnl"]["passed"] is False
        assert result["risk_check_details"]["daily_pnl"]["current"] == -1200.0

    def test_validate_drawdown_limit_failed(self):
        """Test validation when drawdown limit is exceeded."""
        mock_orchestrator = Mock()
        validator = RiskValidator(mock_orchestrator)

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

        # 15% drawdown exceeds 12% limit
        result = validator.validate_entry(
            signal=signal,
            current_pnl=0.0,
            current_equity=42500.0,  # 15% below peak
            peak_equity=50000.0,
            open_positions=2,
        )

        assert result["risk_checks_passed"] is False
        assert result["risk_check_details"]["drawdown"]["passed"] is False
        assert result["risk_check_details"]["drawdown"]["current"] == pytest.approx(0.15, rel=0.01)

    def test_validate_max_positions_failed(self):
        """Test validation when max positions limit is reached."""
        mock_orchestrator = Mock()
        validator = RiskValidator(mock_orchestrator)

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

        result = validator.validate_entry(
            signal=signal,
            current_pnl=0.0,
            current_equity=50000.0,
            peak_equity=50000.0,
            open_positions=5,  # At limit
        )

        assert result["risk_checks_passed"] is False
        assert result["risk_check_details"]["open_positions"]["passed"] is False

    def test_validate_custom_limits(self):
        """Test validation with custom limits."""
        mock_orchestrator = Mock()
        validator = RiskValidator(mock_orchestrator)

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

        result = validator.validate_entry(
            signal=signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
            daily_loss_limit=500.0,  # Custom $500 limit
            max_drawdown=0.10,  # Custom 10% limit
            max_positions=3,  # Custom 3 position limit
        )

        assert result["risk_checks_passed"] is True
        assert result["risk_check_details"]["daily_pnl"]["limit"] == -500.0
        assert result["risk_check_details"]["drawdown"]["limit"] == 0.10
        assert result["risk_check_details"]["open_positions"]["limit"] == 3


class TestEntryLogic:
    """Test EntryLogic orchestrator."""

    def test_process_signal_accepted(self):
        """Test processing signal that gets accepted."""
        position_sizer = PositionSizer()
        mock_orchestrator = Mock()
        risk_validator = RiskValidator(mock_orchestrator)
        entry_logic = EntryLogic(position_sizer, risk_validator)

        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="long",
            entry_price=11850.0,
            stop_loss=11840.0,
            take_profit=11870.0,
            composite_confidence=0.75,  # Tier 3 → 3 contracts
            contributing_strategies=["triple_confluence_scaler"],
            strategy_confidences={"triple_confluence_scaler": 0.75},
            strategy_weights={"triple_confluence_scaler": 1.0},
            bar_timestamp=datetime.now()
        )

        decision = entry_logic.process_signal(
            signal=signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        assert isinstance(decision, EntryDecision)
        assert decision.decision == "ACCEPT"
        assert decision.position_size == 3  # Tier 3
        assert decision.risk_checks_passed is True
        assert decision.rejection_reason is None

    def test_process_signal_rejected(self):
        """Test processing signal that gets rejected."""
        position_sizer = PositionSizer()
        mock_orchestrator = Mock()
        risk_validator = RiskValidator(mock_orchestrator)
        entry_logic = EntryLogic(position_sizer, risk_validator)

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

        decision = entry_logic.process_signal(
            signal=signal,
            current_pnl=-1200.0,  # Exceeds daily loss limit
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        assert isinstance(decision, EntryDecision)
        assert decision.decision == "REJECT"
        assert decision.position_size == 0
        assert decision.risk_checks_passed is False
        assert "daily_pnl" in decision.rejection_reason or "Risk check failed" in decision.rejection_reason

    def test_create_trade_order_from_accepted_decision(self):
        """Test creating trade order from accepted decision."""
        position_sizer = PositionSizer()
        mock_orchestrator = Mock()
        risk_validator = RiskValidator(mock_orchestrator)
        entry_logic = EntryLogic(position_sizer, risk_validator)

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

        decision = entry_logic.process_signal(
            signal=signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        trade_order = entry_logic.create_trade_order(decision)

        assert isinstance(trade_order, TradeOrder)
        assert trade_order.symbol == "MNQ"
        assert trade_order.direction == "long"
        assert trade_order.quantity == 3
        assert trade_order.entry_price == 11850.0
        assert trade_order.stop_loss == 11840.0
        assert trade_order.take_profit == 11870.0
        assert trade_order.status == "pending"
        assert trade_order.order_type == "market"

    def test_create_trade_order_from_rejected_decision_raises_error(self):
        """Test creating order from rejected decision raises error."""
        position_sizer = PositionSizer()
        mock_orchestrator = Mock()
        risk_validator = RiskValidator(mock_orchestrator)
        entry_logic = EntryLogic(position_sizer, risk_validator)

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

        decision = entry_logic.process_signal(
            signal=signal,
            current_pnl=-1200.0,  # Reject
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        with pytest.raises(ValueError, match="Cannot create trade order for REJECT decision"):
            entry_logic.create_trade_order(decision)

    def test_prioritize_signals(self):
        """Test signal prioritization by confidence."""
        position_sizer = PositionSizer()
        mock_orchestrator = Mock()
        risk_validator = RiskValidator(mock_orchestrator)
        entry_logic = EntryLogic(position_sizer, risk_validator)

        signals = [
            EnsembleTradeSignal(
                timestamp=datetime.now(),
                direction="long",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                composite_confidence=0.65,
                contributing_strategies=["strategy_a"],
                strategy_confidences={"strategy_a": 0.65},
                strategy_weights={"strategy_a": 1.0},
                bar_timestamp=datetime.now()
            ),
            EnsembleTradeSignal(
                timestamp=datetime.now(),
                direction="short",
                entry_price=11850.0,
                stop_loss=11860.0,
                take_profit=11830.0,
                composite_confidence=0.85,
                contributing_strategies=["strategy_b"],
                strategy_confidences={"strategy_b": 0.85},
                strategy_weights={"strategy_b": 1.0},
                bar_timestamp=datetime.now()
            ),
            EnsembleTradeSignal(
                timestamp=datetime.now(),
                direction="long",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                composite_confidence=0.55,
                contributing_strategies=["strategy_c"],
                strategy_confidences={"strategy_c": 0.55},
                strategy_weights={"strategy_c": 1.0},
                bar_timestamp=datetime.now()
            ),
        ]

        prioritized = entry_logic.prioritize_signals(signals)

        assert len(prioritized) == 3
        assert prioritized[0].composite_confidence == 0.85  # Highest
        assert prioritized[1].composite_confidence == 0.65  # Middle
        assert prioritized[2].composite_confidence == 0.55  # Lowest

    def test_full_pipeline_from_signal_to_order(self):
        """Test full pipeline: signal → decision → order."""
        position_sizer = PositionSizer()
        mock_orchestrator = Mock()
        risk_validator = RiskValidator(mock_orchestrator)
        entry_logic = EntryLogic(position_sizer, risk_validator)

        signal = EnsembleTradeSignal(
            timestamp=datetime.now(),
            direction="short",
            entry_price=11850.0,
            stop_loss=11860.0,
            take_profit=11830.0,
            composite_confidence=0.82,  # Tier 4 → 4 contracts
            contributing_strategies=["adaptive_ema_momentum", "vwap_bounce"],
            strategy_confidences={"adaptive_ema_momentum": 0.85, "vwap_bounce": 0.80},
            strategy_weights={"adaptive_ema_momentum": 0.50, "vwap_bounce": 0.50},
            bar_timestamp=datetime.now()
        )

        # Process signal
        decision = entry_logic.process_signal(
            signal=signal,
            current_pnl=100.0,  # Positive P&L
            current_equity=50100.0,
            peak_equity=50000.0,
            open_positions=1,
        )

        # Verify decision
        assert decision.decision == "ACCEPT"
        assert decision.position_size == 4

        # Create order
        trade_order = entry_logic.create_trade_order(decision)

        # Verify order
        assert trade_order.symbol == "MNQ"
        assert trade_order.direction == "short"
        assert trade_order.quantity == 4
        assert trade_order.entry_price == 11850.0
        assert trade_order.stop_loss == 11860.0  # Above entry for short
        assert trade_order.take_profit == 11830.0  # Below entry for short
        assert trade_order.notional_value() == pytest.approx(11850.0 * 0.50 * 4, rel=0.01)
