"""Integration tests for entry logic with ensemble signals.

These tests verify the full pipeline from ensemble signals
through position sizing and risk validation to trade orders.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.execution.entry_logic import PositionSizer, RiskValidator, EntryLogic
from src.execution.models import EntryDecision, TradeOrder
from src.detection.models import EnsembleTradeSignal


@pytest.fixture
def position_sizer():
    """Create position sizer for testing."""
    return PositionSizer(min_contracts=1, max_contracts=5)


@pytest.fixture
def risk_validator():
    """Create risk validator for testing."""
    mock_orchestrator = Mock()
    return RiskValidator(mock_orchestrator)


@pytest.fixture
def entry_logic(position_sizer, risk_validator):
    """Create entry logic for testing."""
    return EntryLogic(position_sizer, risk_validator)


@pytest.fixture
def sample_long_signal():
    """Create sample long ensemble signal."""
    return EnsembleTradeSignal(
        timestamp=datetime.now(),
        direction="long",
        entry_price=11850.0,
        stop_loss=11840.0,
        take_profit=11870.0,
        composite_confidence=0.75,
        contributing_strategies=["triple_confluence_scaler", "wolf_pack_3_edge"],
        strategy_confidences={
            "triple_confluence_scaler": 0.80,
            "wolf_pack_3_edge": 0.70
        },
        strategy_weights={
            "triple_confluence_scaler": 0.50,
            "wolf_pack_3_edge": 0.50
        },
        bar_timestamp=datetime.now()
    )


@pytest.fixture
def sample_short_signal():
    """Create sample short ensemble signal."""
    return EnsembleTradeSignal(
        timestamp=datetime.now(),
        direction="short",
        entry_price=11850.0,
        stop_loss=11860.0,
        take_profit=11830.0,
        composite_confidence=0.82,
        contributing_strategies=["adaptive_ema_momentum", "vwap_bounce"],
        strategy_confidences={
            "adaptive_ema_momentum": 0.85,
            "vwap_bounce": 0.80
        },
        strategy_weights={
            "adaptive_ema_momentum": 0.50,
            "vwap_bounce": 0.50
        },
        bar_timestamp=datetime.now()
    )


class TestEntryLogicIntegration:
    """Integration tests for full entry logic pipeline."""

    def test_full_pipeline_long_signal_accepted(
        self, entry_logic, sample_long_signal
    ):
        """Test full pipeline for long signal that gets accepted."""
        # Process signal
        decision = entry_logic.process_signal(
            signal=sample_long_signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        # Verify decision
        assert decision.decision == "ACCEPT"
        assert decision.position_size == 3  # Tier 3 for 0.75 confidence
        assert decision.risk_checks_passed is True

        # Create order
        trade_order = entry_logic.create_trade_order(decision)

        # Verify order
        assert isinstance(trade_order, TradeOrder)
        assert trade_order.direction == "long"
        assert trade_order.quantity == 3
        assert trade_order.entry_price == 11850.0
        assert trade_order.stop_loss == 11840.0
        assert trade_order.take_profit == 11870.0
        assert trade_order.status == "pending"

    def test_full_pipeline_short_signal_accepted(
        self, entry_logic, sample_short_signal
    ):
        """Test full pipeline for short signal that gets accepted."""
        # Process signal
        decision = entry_logic.process_signal(
            signal=sample_short_signal,
            current_pnl=100.0,
            current_equity=50100.0,
            peak_equity=50000.0,
            open_positions=1,
        )

        # Verify decision
        assert decision.decision == "ACCEPT"
        assert decision.position_size == 4  # Tier 4 for 0.82 confidence
        assert decision.risk_checks_passed is True

        # Create order
        trade_order = entry_logic.create_trade_order(decision)

        # Verify order
        assert trade_order.direction == "short"
        assert trade_order.quantity == 4
        assert trade_order.stop_loss == 11860.0  # Above entry for short
        assert trade_order.take_profit == 11830.0  # Below entry for short

    def test_signal_rejected_daily_loss_limit(
        self, entry_logic, sample_long_signal
    ):
        """Test signal rejection when daily loss limit exceeded."""
        decision = entry_logic.process_signal(
            signal=sample_long_signal,
            current_pnl=-1200.0,  # Exceeds $1000 limit
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        assert decision.decision == "REJECT"
        assert decision.position_size == 0
        assert decision.risk_checks_passed is False

    def test_signal_rejected_max_positions(
        self, entry_logic, sample_long_signal
    ):
        """Test signal rejection when max positions reached."""
        decision = entry_logic.process_signal(
            signal=sample_long_signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=5,  # At limit
        )

        assert decision.decision == "REJECT"
        assert decision.position_size == 0
        assert decision.risk_checks_passed is False

    def test_position_scaling_by_confidence(self, entry_logic):
        """Test that position size scales with confidence."""
        base_time = datetime.now()

        # Test all confidence tiers
        test_cases = [
            (0.55, 1, "Tier 1"),
            (0.65, 2, "Tier 2"),
            (0.75, 3, "Tier 3"),
            (0.85, 4, "Tier 4"),
            (0.95, 5, "Tier 5"),
        ]

        for confidence, expected_size, tier in test_cases:
            signal = EnsembleTradeSignal(
                timestamp=base_time,
                direction="long",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                composite_confidence=confidence,
                contributing_strategies=["test_strategy"],
                strategy_confidences={"test_strategy": confidence},
                strategy_weights={"test_strategy": 1.0},
                bar_timestamp=base_time
            )

            decision = entry_logic.process_signal(
                signal=signal,
                current_pnl=0.0,
                current_equity=50000.0,
                peak_equity=50000.0,
                open_positions=0,
            )

            assert decision.position_size == expected_size, \
                f"{tier}: expected {expected_size}, got {decision.position_size}"

    def test_multiple_signals_prioritization(self, entry_logic):
        """Test prioritization when multiple signals arrive simultaneously."""
        base_time = datetime.now()

        # Create multiple signals with different confidences
        signals = [
            EnsembleTradeSignal(
                timestamp=base_time,
                direction="long",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                composite_confidence=0.65,
                contributing_strategies=["strategy_a"],
                strategy_confidences={"strategy_a": 0.65},
                strategy_weights={"strategy_a": 1.0},
                bar_timestamp=base_time
            ),
            EnsembleTradeSignal(
                timestamp=base_time,
                direction="short",
                entry_price=11850.0,
                stop_loss=11860.0,
                take_profit=11830.0,
                composite_confidence=0.85,
                contributing_strategies=["strategy_b"],
                strategy_confidences={"strategy_b": 0.85},
                strategy_weights={"strategy_b": 1.0},
                bar_timestamp=base_time
            ),
            EnsembleTradeSignal(
                timestamp=base_time,
                direction="long",
                entry_price=11850.0,
                stop_loss=11840.0,
                take_profit=11870.0,
                composite_confidence=0.55,
                contributing_strategies=["strategy_c"],
                strategy_confidences={"strategy_c": 0.55},
                strategy_weights={"strategy_c": 1.0},
                bar_timestamp=base_time
            ),
        ]

        # Prioritize signals
        prioritized = entry_logic.prioritize_signals(signals)

        # Verify order
        assert prioritized[0].composite_confidence == 0.85  # Highest first
        assert prioritized[1].composite_confidence == 0.65
        assert prioritized[2].composite_confidence == 0.55  # Lowest last

        # Execute highest confidence signal
        decision = entry_logic.process_signal(
            signal=prioritized[0],
            current_pnl=0.0,
            current_equity=50000.0,
            peak_equity=50000.0,
            open_positions=0,
        )

        assert decision.decision == "ACCEPT"
        assert decision.position_size == 4  # Tier 4

    def test_risk_check_details_completeness(self, entry_logic, sample_long_signal):
        """Test that all risk check details are included."""
        decision = entry_logic.process_signal(
            signal=sample_long_signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        # Verify all risk check details are present
        assert "daily_pnl" in decision.risk_check_details
        assert "drawdown" in decision.risk_check_details
        assert "open_positions" in decision.risk_check_details
        assert "stop_loss_defined" in decision.risk_check_details

        # Verify each check has required fields
        for check_name, check_details in decision.risk_check_details.items():
            assert "current" in check_details or "defined" in check_details
            assert "limit" in check_details or "defined" in check_details
            assert "passed" in check_details

    def test_order_metadata_preservation(
        self, entry_logic, sample_long_signal
    ):
        """Test that ensemble signal metadata is preserved in trade order."""
        decision = entry_logic.process_signal(
            signal=sample_long_signal,
            current_pnl=-200.0,
            current_equity=49000.0,
            peak_equity=50000.0,
            open_positions=2,
        )

        trade_order = entry_logic.create_trade_order(decision)

        # Verify ensemble signal reference is preserved
        assert trade_order.ensemble_signal == sample_long_signal
        assert trade_order.ensemble_signal.composite_confidence == 0.75
        assert len(trade_order.ensemble_signal.contributing_strategies) == 2

    def test_position_sizer_history_tracking(self, position_sizer):
        """Test position size history tracking across multiple signals."""
        # Generate multiple position sizes
        confidences = [0.55, 0.75, 0.85, 0.65, 0.95]
        expected_sizes = [1, 3, 4, 2, 5]

        for conf, expected in zip(confidences, expected_sizes):
            size = position_sizer.calculate_position_size(conf)
            assert size == expected

        # Verify history
        assert len(position_sizer.position_size_history) == 5
        assert position_sizer.position_size_history == expected_sizes

        # Verify average
        avg = position_sizer.get_average_position_size()
        assert avg == sum(expected_sizes) / len(expected_sizes)

        # Verify distribution
        distribution = position_sizer.get_position_size_distribution()
        assert distribution[1] == 1  # One 1-contract position
        assert distribution[2] == 1  # One 2-contract position
        assert distribution[3] == 1  # One 3-contract position
        assert distribution[4] == 1  # One 4-contract position
        assert distribution[5] == 1  # One 5-contract position
