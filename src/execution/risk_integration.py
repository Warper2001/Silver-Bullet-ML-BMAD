"""Risk integration for execution pipeline.

This module provides integration between the execution pipeline and the
risk management system, converting TradingSignal formats and providing
a clean interface for risk validation.

Features:
- Wraps RiskOrchestrator for execution pipeline integration
- Converts TradingSignal between execution and risk formats
- Detailed validation result logging
"""

import logging
from typing import Optional

from src.risk.risk_orchestrator import (
    RiskOrchestrator,
    TradingSignal as RiskTradingSignal
)

logger = logging.getLogger(__name__)


class RiskValidator:
    """Validate trading signals through risk management system.

    Wraps RiskOrchestrator to provide clean integration with the
    execution pipeline. Converts TradingSignal format and provides
    detailed validation results.

    Attributes:
        _risk_orchestrator: RiskOrchestrator instance with all 8 risk layers

    Example:
        >>> from src.risk.factory import RiskComponentFactory
        >>> orchestrator = RiskComponentFactory.create_risk_orchestrator()
        >>> validator = RiskValidator(orchestrator)
        >>> result = validator.validate_trade(signal)
        >>> if result['is_valid']:
        ...     # Submit order
        ... else:
        ...     # Handle rejection
        ...     print(f"Blocked: {result['block_reason']}")
    """

    def __init__(self, risk_orchestrator: RiskOrchestrator) -> None:
        """Initialize RiskValidator.

        Args:
            risk_orchestrator: RiskOrchestrator instance with all 8 risk layers

        Raises:
            ValueError: If risk_orchestrator is None
        """
        if risk_orchestrator is None:
            raise ValueError("RiskOrchestrator cannot be None")

        self._risk_orchestrator = risk_orchestrator
        logger.info("RiskValidator initialized with RiskOrchestrator")

    def validate_trade(self, execution_signal) -> dict:
        """Validate trading signal through all risk management layers.

        Converts execution pipeline TradingSignal to risk orchestrator format,
        then validates through all 8 risk layers.

        Risk Layers Checked:
            1. Emergency Stop
            2. Daily Loss Limit ($500 USD)
            3. Max Drawdown (12%)
            4. Max Position Size (5 contracts)
            5. Circuit Breaker
            6. News Events
            7. Per-Trade Risk Limit

        Args:
            execution_signal: TradingSignal from execution pipeline

        Returns:
            Dictionary with validation result:
            - is_valid: Whether trade passed all risk checks
            - block_reason: Reason for rejection (if invalid)
            - checks_passed: List of risk checks that passed
            - checks_failed: List of risk checks that failed
            - validation_details: Detailed results from each check

        Example:
            >>> result = validator.validate_trade(signal)
            >>> if result['is_valid']:
            ...     print("All risk checks passed")
            >>> else:
            ...     print(f"Blocked: {result['block_reason']}")
            ...     print(f"Failed checks: {result['checks_failed']}")
        """
        # Convert execution signal to risk orchestrator format
        risk_signal = self._convert_signal_format(execution_signal)

        # Validate through risk orchestrator
        validation_result = self._risk_orchestrator.validate_trade(risk_signal)

        # Log validation result
        self._log_validation_result(execution_signal, validation_result)

        return validation_result

    def _convert_signal_format(self, execution_signal) -> RiskTradingSignal:
        """Convert execution TradingSignal to risk orchestrator format.

        Args:
            execution_signal: TradingSignal from execution pipeline

        Returns:
            RiskTradingSignal in format expected by RiskOrchestrator
        """
        # Extract stop loss from signal prediction or patterns
        # Default to 2% risk from entry price if not specified
        stop_loss_price = self._extract_stop_loss(execution_signal)

        # Default quantity to 1 contracts if not specified
        quantity = self._extract_quantity(execution_signal)

        # Create risk trading signal
        risk_signal = RiskTradingSignal(
            signal_id=execution_signal.signal_id,
            entry_price=execution_signal.entry_price,
            stop_loss_price=stop_loss_price,
            quantity=quantity
        )

        return risk_signal

    def _extract_stop_loss(self, execution_signal) -> float:
        """Extract stop loss price from execution signal.

        Args:
            execution_signal: TradingSignal from execution pipeline

        Returns:
            Stop loss price (defaults to 2% below entry for long, 2% above for short)
        """
        # Check if prediction contains stop loss
        if hasattr(execution_signal, 'prediction') and execution_signal.prediction:
            if 'stop_loss' in execution_signal.prediction:
                return execution_signal.prediction['stop_loss']

        # Check if patterns contain stop loss info
        if hasattr(execution_signal, 'patterns') and execution_signal.patterns:
            # Check for stop loss in pattern metadata
            for pattern in execution_signal.patterns:
                if isinstance(pattern, dict) and 'stop_loss' in pattern:
                    return pattern['stop_loss']

        # Default: 2% risk from entry price
        if execution_signal.direction == "bullish":
            # Long position: stop loss 2% below entry
            return execution_signal.entry_price * 0.98
        else:
            # Short position: stop loss 2% above entry
            return execution_signal.entry_price * 1.02

    def _extract_quantity(self, execution_signal) -> int:
        """Extract quantity from execution signal.

        Args:
            execution_signal: TradingSignal from execution pipeline

        Returns:
            Order quantity in contracts (defaults to 1)
        """
        # Check if prediction contains quantity
        if hasattr(execution_signal, 'prediction') and execution_signal.prediction:
            if 'quantity' in execution_signal.prediction:
                return int(execution_signal.prediction['quantity'])

        # Default to 1 contract
        return 1

    def _log_validation_result(
        self,
        execution_signal,
        validation_result: dict
    ) -> None:
        """Log risk validation result.

        Args:
            execution_signal: TradingSignal that was validated
            validation_result: Validation result from RiskOrchestrator
        """
        if validation_result['is_valid']:
            logger.info(
                "Risk validation passed for signal {}: {} checks passed".format(
                    execution_signal.signal_id,
                    len(validation_result['checks_passed'])
                )
            )
        else:
            logger.warning(
                "Risk validation FAILED for signal {}: {} - Failed checks: {}".format(
                    execution_signal.signal_id,
                    validation_result['block_reason'],
                    ", ".join(validation_result['checks_failed'])
                )
            )
