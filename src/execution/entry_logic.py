"""Entry logic for position sizing and risk validation.

This module implements:
- PositionSizer: Confidence-based position sizing (1-5 contracts)
- RiskValidator: Integration with risk management system
- EntryLogic: Main orchestrator for entry decisions

Position sizing is based on ensemble confidence scores:
- Tier 1 (0.50-0.60): 1 contract
- Tier 2 (0.60-0.70): 2 contracts
- Tier 3 (0.70-0.80): 3 contracts
- Tier 4 (0.80-0.90): 4 contracts
- Tier 5 (>0.90): 5 contracts
"""

import logging
import uuid
from collections import defaultdict
from typing import Literal

from src.execution.models import EntryDecision, TradeOrder, _now
from src.risk.risk_orchestrator import RiskOrchestrator
from src.detection.models import EnsembleTradeSignal

logger = logging.getLogger(__name__)


class PositionSizer:
    """Confidence-based position sizing algorithm.

    Scales position size from 1-5 contracts based on ensemble confidence score.
    Higher confidence → larger position size.
    """

    CONFIDENCE_TIERS = {
        1: (0.50, 0.60),  # Tier 1: 0.50-0.60 → 1 contract
        2: (0.60, 0.70),  # Tier 2: 0.60-0.70 → 2 contracts
        3: (0.70, 0.80),  # Tier 3: 0.70-0.80 → 3 contracts
        4: (0.80, 0.90),  # Tier 4: 0.80-0.90 → 4 contracts
        5: (0.90, 1.01),  # Tier 5: >0.90 → 5 contracts
    }

    def __init__(self, min_contracts: int = 1, max_contracts: int = 5) -> None:
        """Initialize position sizer."""
        if min_contracts < 1 or min_contracts > 5:
            raise ValueError("min_contracts must be between 1 and 5")
        if max_contracts < 1 or max_contracts > 5:
            raise ValueError("max_contracts must be between 1 and 5")
        if min_contracts > max_contracts:
            raise ValueError("min_contracts must be <= max_contracts")

        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.position_size_history: list[int] = []

    def calculate_position_size(self, confidence: float) -> int:
        """Calculate position size based on confidence score.

        Returns 0 if confidence < 0.50 (blocking entry).
        """
        if not 0 <= confidence <= 1.05: # Allow slightly above for float errors
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        # Patch 5: Block trades below 0.50
        if confidence < 0.50:
            logger.debug(f"Confidence {confidence:.3f} below minimum threshold 0.50")
            return 0

        # Determine tier from mapping
        position_size = 1
        tier_label = "Below Tier 1"
        
        for size, (low, high) in self.CONFIDENCE_TIERS.items():
            if low <= confidence < high:
                position_size = size
                tier_label = f"Tier {size}"
                break
        else:
            if confidence >= 1.0:
                position_size = 5
                tier_label = "Tier 5"

        # Enforce min/max limits
        position_size = max(self.min_contracts, min(position_size, self.max_contracts))

        # Track history
        self.position_size_history.append(position_size)

        logger.debug(
            f"Position sizing: confidence={confidence:.3f} → {tier_label} → {position_size} contracts"
        )

        return position_size

    def get_confidence_tier(self, confidence: float) -> str:
        """Get confidence tier description."""
        # Patch 6: Correct labels for sub-threshold
        if confidence < 0.50:
            return "Below Threshold (<0.50)"
            
        for size, (low, high) in self.CONFIDENCE_TIERS.items():
            if low <= confidence < high:
                return f"Tier {size} ({low:.2f}-{high:.2f})"
        
        return "Tier 5 (>0.90)"

    def get_average_position_size(self) -> float:
        """Calculate average position size from history.

        Returns:
            Average position size, or 0 if no history
        """
        if not self.position_size_history:
            return 0.0
        return sum(self.position_size_history) / len(self.position_size_history)

    def get_position_size_distribution(self) -> dict[int, int]:
        """Get distribution of position sizes from history.

        Returns:
            Dictionary mapping position size to count
        """
        distribution = defaultdict(int)
        for size in self.position_size_history:
            distribution[size] += 1
        return dict(distribution)


class RiskValidator:
    """Risk validation for entry decisions.

    Integrates with existing risk management system to validate
    entry criteria before allowing trades.

    Attributes:
        risk_orchestrator: Risk management orchestrator
    """

    def __init__(self, risk_orchestrator: RiskOrchestrator) -> None:
        """Initialize risk validator.

        Args:
            risk_orchestrator: Risk management orchestrator instance
        """
        self.risk_orchestrator = risk_orchestrator

    def validate_entry(
        self,
        signal: EnsembleTradeSignal,
        current_pnl: float = 0.0,
        current_equity: float = 50000.0,
        peak_equity: float = 50000.0,
        open_positions: int = 0,
        daily_loss_limit: float = 1000.0,
        max_drawdown: float = 0.12,
        max_positions: int = 5,
    ) -> dict:
        """Validate entry criteria.

        Args:
            signal: Ensemble trade signal to validate
            current_pnl: Current daily P&L
            current_equity: Current account equity
            peak_equity: Peak account equity (for drawdown calculation)
            open_positions: Number of currently open positions
            daily_loss_limit: Daily loss limit in USD
            max_drawdown: Maximum drawdown as fraction (e.g., 0.12 = 12%)
            max_positions: Maximum number of open positions

        Returns:
            Dictionary with validation results
        """
        risk_check_details = {
            "daily_pnl": {
                "current": current_pnl,
                "limit": -daily_loss_limit,
                "passed": current_pnl > -daily_loss_limit,
            },
            "drawdown": {
                "current": (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0,
                "limit": max_drawdown,
                "passed": current_equity >= peak_equity * (1 - max_drawdown),
            },
            "open_positions": {
                "current": open_positions,
                "limit": max_positions,
                "passed": open_positions < max_positions,
            },
            "stop_loss_defined": {
                "defined": signal.stop_loss is not None and signal.stop_loss > 0,
                "passed": signal.stop_loss is not None and signal.stop_loss > 0,
            },
        }

        all_passed = all(check["passed"] for check in risk_check_details.values())

        logger.info(
            f"Risk validation: {'PASSED' if all_passed else 'FAILED'} - "
            f"P&L: ${current_pnl:.2f}, Drawdown: {risk_check_details['drawdown']['current']:.1%}, "
            f"Positions: {open_positions}/{max_positions}"
        )

        return {
            "risk_checks_passed": all_passed,
            "risk_check_details": risk_check_details,
        }


class EntryLogic:
    """Entry logic orchestrator.

    Coordinates position sizing and risk validation to make
    entry decisions for ensemble signals.

    Attributes:
        position_sizer: Position sizing algorithm
        risk_validator: Risk validation integration
    """

    def __init__(self, position_sizer: PositionSizer, risk_validator: RiskValidator) -> None:
        """Initialize entry logic.

        Args:
            position_sizer: Position sizing algorithm
            risk_validator: Risk validation integration
        """
        self.position_sizer = position_sizer
        self.risk_validator = risk_validator

    def process_signal(
        self,
        signal: EnsembleTradeSignal,
        current_pnl: float = 0.0,
        current_equity: float = 50000.0,
        peak_equity: float = 50000.0,
        open_positions: int = 0,
    ) -> EntryDecision:
        """Process ensemble signal and make entry decision."""
        # Validate risk
        risk_result = self.risk_validator.validate_entry(
            signal=signal,
            current_pnl=current_pnl,
            current_equity=current_equity,
            peak_equity=peak_equity,
            open_positions=open_positions,
        )

        # Make decision
        if risk_result["risk_checks_passed"]:
            # Calculate position size from confidence
            position_size = self.position_sizer.calculate_position_size(
                signal.composite_confidence
            )
            # Patch 5 reinforcement: Ensure position size is at least 1 if accepted
            if position_size == 0:
                decision = "REJECT"
                rejection_reason = "Position size 0 (Confidence below threshold)"
            else:
                decision = "ACCEPT"
                rejection_reason = None
        else:
            position_size = 0
            decision = "REJECT"
            # Find first failed check for rejection reason
            for check_name, check_result in risk_result["risk_check_details"].items():
                if not check_result["passed"]:
                    rejection_reason = f"Risk check failed: {check_name}"
                    break

        # Create entry decision
        entry_decision = EntryDecision(
            signal=signal,
            position_size=position_size,
            risk_checks_passed=risk_result["risk_checks_passed"],
            risk_check_details=risk_result["risk_check_details"],
            decision=decision,  # type: ignore
            rejection_reason=rejection_reason,
            timestamp=_now(), # Patch 11: Unified aware time
        )

        return entry_decision

    def process_signal_batch(
        self,
        signals: list[EnsembleTradeSignal],
        current_pnl: float = 0.0,
        current_equity: float = 50000.0,
        peak_equity: float = 50000.0,
        initial_open_positions: int = 0,
    ) -> list[EntryDecision]:
        """Process multiple signals with risk re-evaluation (Patch 7)."""
        prioritized = self.prioritize_signals(signals)
        decisions = []
        active_positions = initial_open_positions

        for signal in prioritized:
            decision = self.process_signal(
                signal=signal,
                current_pnl=current_pnl,
                current_equity=current_equity,
                peak_equity=peak_equity,
                open_positions=active_positions
            )
            decisions.append(decision)
            
            # If accepted, increment count for next signal re-evaluation
            if decision.decision == "ACCEPT":
                active_positions += 1
                
        return decisions

    def create_trade_order(self, decision: EntryDecision) -> TradeOrder:
        """Create trade order from accepted entry decision."""
        if decision.decision != "ACCEPT":
            raise ValueError(f"Cannot create trade order for {decision.decision} decision")

        # Patch 10: Use full UUID for safety
        trade_id = f"entry-{uuid.uuid4()}"

        # Create trade order with ALL required fields (Patch 3)
        trade_order = TradeOrder(
            trade_id=trade_id,
            symbol="MNQ",
            direction=decision.signal.direction.lower(), # Ensure lowercase
            quantity=decision.position_size,
            order_type="market",
            entry_price=decision.signal.entry_price,
            limit_price=None,
            stop_loss=decision.signal.stop_loss,
            take_profit=decision.signal.take_profit,
            timestamp=decision.timestamp,
            ensemble_signal=decision.signal,
            position_size=decision.position_size,
            # Missing fields from Patch 3:
            entry_time=decision.timestamp,
            original_quantity=decision.position_size,
            remaining_quantity=decision.position_size,
        )

        logger.info(f"Trade order created: {trade_id} - {trade_order.direction} "
                    f"{trade_order.quantity} contracts @ {trade_order.entry_price}")

        return trade_order

    def prioritize_signals(self, signals: list[EnsembleTradeSignal]) -> list[EnsembleTradeSignal]:
        """Prioritize signals by composite confidence (descending)."""
        return sorted(signals, key=lambda s: s.composite_confidence, reverse=True)
