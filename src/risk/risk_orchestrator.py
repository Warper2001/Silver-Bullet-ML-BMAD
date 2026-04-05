"""Risk management orchestrator.

This module coordinates all risk management components and ensures
comprehensive validation before any trade is executed. All risk layers
must allow trading for orders to proceed.

Features:
- Unified risk validation across all 8 safety layers
- Detailed validation results with pass/fail per check
- CSV audit trail logging
- Integration with notification system
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.risk.emergency_stop import EmergencyStop
from src.risk.daily_loss_tracker import DailyLossTracker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_size_tracker import PositionSizeTracker
from src.risk.circuit_breaker_detector import CircuitBreakerDetector
from src.risk.news_event_filter import NewsEventFilter
from src.risk.per_trade_risk_limit import PerTradeRiskLimit
from src.risk.notification_manager import NotificationManager

logger = logging.getLogger(__name__)


class TradingSignal:
    """Trading signal placeholder."""

    def __init__(
        self,
        signal_id: str,
        entry_price: float,
        stop_loss_price: float,
        quantity: int
    ):
        self.signal_id = signal_id
        self.entry_price = entry_price
        self.stop_loss_price = stop_loss_price
        self.quantity = quantity


class RiskOrchestrator:
    """Orchestrate all risk management checks.

    This class coordinates all risk management components and
    ensures comprehensive validation before any trade is executed.

    Attributes:
        _emergency_stop: Emergency stop button
        _daily_loss_limit: Daily loss limit enforcer
        _max_drawdown_limit: Maximum drawdown enforcer
        _max_position_size: Maximum position size enforcer
        _circuit_breaker_detector: Circuit breaker detector
        _news_event_filter: News event filter
        _per_trade_risk_limit: Per-trade risk limit
        _notification_manager: Notification manager
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> orchestrator = RiskOrchestrator(
        ...     emergency_stop=stop,
        ...     daily_loss_limit=daily_limit,
        ...     max_drawdown_limit=drawdown_limit,
        ...     max_position_size=position_size_limit,
        ...     circuit_breaker_detector=detector,
        ...     news_event_filter=news_filter,
        ...     per_trade_risk_limit=risk_limit,
        ...     notification_manager=notification_mgr
        ... )
        >>> result = orchestrator.validate_trade(signal)
        >>> if result['is_valid']:
        ...     submit_order(signal)
    """

    def __init__(
        self,
        emergency_stop: EmergencyStop,
        daily_loss_tracker: DailyLossTracker,
        drawdown_tracker: DrawdownTracker,
        position_size_tracker: PositionSizeTracker,
        circuit_breaker_detector: CircuitBreakerDetector,
        news_event_filter: NewsEventFilter,
        per_trade_risk_limit: PerTradeRiskLimit,
        notification_manager: NotificationManager,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize risk orchestrator.

        Args:
            emergency_stop: Emergency stop button
            daily_loss_tracker: Daily loss tracker
            drawdown_tracker: Drawdown tracker
            position_size_tracker: Position size tracker
            circuit_breaker_detector: Circuit breaker detector
            news_event_filter: News event filter
            per_trade_risk_limit: Per-trade risk limit
            notification_manager: Notification manager
            audit_trail_path: Path to CSV audit trail file (optional)

        Example:
            >>> orchestrator = RiskOrchestrator(...)
        """
        self._emergency_stop = emergency_stop
        self._daily_loss_tracker = daily_loss_tracker
        self._drawdown_tracker = drawdown_tracker
        self._position_size_tracker = position_size_tracker
        self._circuit_breaker_detector = circuit_breaker_detector
        self._news_event_filter = news_event_filter
        self._per_trade_risk_limit = per_trade_risk_limit
        self._notification_manager = notification_manager
        self._audit_trail_path = audit_trail_path

        logger.info("RiskOrchestrator initialized with all 8 risk layers")

    def validate_trade(self, signal: TradingSignal) -> dict:
        """Validate trade against all risk management layers.

        Args:
            signal: Trading signal to validate

        Returns:
            Dictionary with validation result:
            - is_valid: Whether trade passed all checks
            - block_reason: Reason for rejection (if invalid)
            - checks_passed: List of checks that passed
            - checks_failed: List of checks that failed
            - validation_details: Details from each check

        Example:
            >>> result = orchestrator.validate_trade(signal)
            >>> if result['is_valid']:
            ...     print("Trade approved")
            ... else:
            ...     print("Rejected: {}".format(
            ...         result['block_reason']
            ...     ))
        """
        checks_passed = []
        checks_failed = []
        validation_details = {}
        block_reason = None

        # Check 1: Emergency Stop
        result = self._check_emergency_stop(signal)
        validation_details['emergency_stop'] = result
        if result['passed']:
            checks_passed.append('emergency_stop')
        else:
            checks_failed.append('emergency_stop')
            block_reason = result['reason']
            self._send_rejection_notification(
                "Emergency Stop Active",
                result['reason']
            )

        # Check 2: Daily Loss Limit
        result = self._check_daily_loss_limit(signal)
        validation_details['daily_loss'] = result
        if result['passed']:
            checks_passed.append('daily_loss')
        else:
            checks_failed.append('daily_loss')
            if block_reason is None:
                block_reason = result['reason']
                self._send_rejection_notification(
                    "Daily Loss Limit Breached",
                    result['reason']
                )

        # Check 3: Max Drawdown
        result = self._check_max_drawdown(signal)
        validation_details['drawdown'] = result
        if result['passed']:
            checks_passed.append('drawdown')
        else:
            checks_failed.append('drawdown')
            if block_reason is None:
                block_reason = result['reason']
                self._send_rejection_notification(
                    "Max Drawdown Exceeded",
                    result['reason']
                )

        # Check 4: Max Position Size
        result = self._check_max_position_size(signal)
        validation_details['position_size'] = result
        if result['passed']:
            checks_passed.append('position_size')
        else:
            checks_failed.append('position_size')
            if block_reason is None:
                block_reason = result['reason']
                self._send_rejection_notification(
                    "Max Position Size Exceeded",
                    result['reason']
                )

        # Check 5: Circuit Breaker
        result = self._check_circuit_breaker(signal)
        validation_details['circuit_breaker'] = result
        if result['passed']:
            checks_passed.append('circuit_breaker')
        else:
            checks_failed.append('circuit_breaker')
            if block_reason is None:
                block_reason = result['reason']
                self._send_rejection_notification(
                    "Circuit Breaker Active",
                    result['reason']
                )

        # Check 6: News Events
        result = self._check_news_events(signal)
        validation_details['news_events'] = result
        if result['passed']:
            checks_passed.append('news_events')
        else:
            checks_failed.append('news_events')
            if block_reason is None:
                block_reason = result['reason']

        # Check 7: Per-Trade Risk
        result = self._check_per_trade_risk(signal)
        validation_details['per_trade_risk'] = result
        if result['passed']:
            checks_passed.append('per_trade_risk')
        else:
            checks_failed.append('per_trade_risk')
            if block_reason is None:
                block_reason = result['reason']
                self._send_rejection_notification(
                    "Per-Trade Risk Exceeded",
                    result['reason']
                )

        # Determine overall validity
        is_valid = len(checks_failed) == 0

        # Log validation result
        self._log_validation_result(
            signal,
            is_valid,
            block_reason,
            checks_passed,
            checks_failed,
            validation_details
        )

        return {
            'is_valid': is_valid,
            'block_reason': block_reason,
            'checks_passed': checks_passed,
            'checks_failed': checks_failed,
            'validation_details': validation_details
        }

    def _check_emergency_stop(self, signal: TradingSignal) -> dict:
        """Check emergency stop status.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        is_allowed = self._emergency_stop.is_trading_allowed()

        if is_allowed:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }
        else:
            status = self._emergency_stop.get_status()
            return {
                'passed': False,
                'reason': 'Emergency stop active: {}'.format(
                    status['stop_reason']
                ),
                'status': 'STOPPED'
            }

    def _check_daily_loss_limit(self, signal: TradingSignal) -> dict:
        """Check daily loss limit.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        is_limit_breached = (
            self._daily_loss_tracker.get_daily_pnl() <=
            -self._daily_loss_tracker._daily_loss_limit
        )

        if is_limit_breached:
            return {
                'passed': False,
                'reason': 'Daily loss limit breached',
                'status': 'BREACHED',
                'current_loss': abs(self._daily_loss_tracker.get_daily_pnl()),
                'max_loss': self._daily_loss_tracker._daily_loss_limit
            }
        else:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }

    def _check_max_drawdown(self, signal: TradingSignal) -> dict:
        """Check maximum drawdown.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        is_exceeded = (
            self._drawdown_tracker.get_drawdown_percentage() >=
            self._drawdown_tracker._max_drawdown_percentage
        )

        if is_exceeded:
            return {
                'passed': False,
                'reason': 'Maximum drawdown exceeded',
                'status': 'EXCEEDED',
                'current_drawdown': (
                    self._drawdown_tracker.get_drawdown_percentage()
                ),
                'max_drawdown': (
                    self._drawdown_tracker._max_drawdown_percentage
                )
            }
        else:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }

    def _check_max_position_size(self, signal: TradingSignal) -> dict:
        """Check maximum position size.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        # Get current position
        current_position = self._position_size_tracker.get_position_count()

        # Calculate new position size
        new_position = current_position + signal.quantity

        is_exceeded = new_position > self._position_size_tracker._max_position_size

        if is_exceeded:
            return {
                'passed': False,
                'reason': (
                    'Max position size exceeded: {} / {} contracts'.format(
                        new_position,
                        self._position_size_tracker._max_position_size
                    )
                ),
                'status': 'EXCEEDED',
                'current_position': current_position,
                'requested_quantity': signal.quantity,
                'max_allowed': self._position_size_tracker._max_contracts
            }
        else:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }

    def _check_circuit_breaker(self, signal: TradingSignal) -> dict:
        """Check circuit breaker status.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        is_halted = self._circuit_breaker_detector.is_trading_halted()

        if is_halted:
            halt_level = self._circuit_breaker_detector.get_halt_level()
            return {
                'passed': False,
                'reason': (
                    'Circuit breaker Level {} halted trading'.format(
                        halt_level if halt_level else 'Unknown'
                    )
                ),
                'status': 'HALTED',
                'halt_level': halt_level
            }
        else:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }

    def _check_news_events(self, signal: TradingSignal) -> dict:
        """Check news event blackout.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        current_time = datetime.now(timezone.utc)
        blackout_status = self._news_event_filter.get_blackout_status(
            current_time
        )

        if blackout_status['is_blackout']:
            return {
                'passed': False,
                'reason': (
                    'News event blackout active: {}'.format(
                        blackout_status['event_name']
                    )
                ),
                'status': 'BLACKOUT',
                'event_name': blackout_status['event_name']
            }
        else:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }

    def _check_per_trade_risk(self, signal: TradingSignal) -> dict:
        """Check per-trade risk limit.

        Args:
            signal: Trading signal

        Returns:
            Check result dictionary
        """
        result = self._per_trade_risk_limit.validate_trade(
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss_price,
            quantity=signal.quantity
        )

        if result['is_valid']:
            return {
                'passed': True,
                'reason': None,
                'status': 'OK'
            }
        else:
            return {
                'passed': False,
                'reason': (
                    'Per-trade risk exceeded: ${:.2f} / ${:.2f}'.format(
                        result['estimated_risk'],
                        self._per_trade_risk_limit._max_risk_dollars
                    )
                ),
                'status': 'EXCEEDED',
                'estimated_risk': result['estimated_risk'],
                'max_allowed': (
                    self._per_trade_risk_limit._max_risk_dollars
                )
            }

    def _send_rejection_notification(
        self,
        title: str,
        reason: str
    ) -> None:
        """Send notification for trade rejection.

        Args:
            title: Notification title
            reason: Rejection reason
        """
        if self._notification_manager.is_notification_enabled():
            self._notification_manager.send_notification(
                severity="WARNING",
                title=title,
                message=reason,
                notification_type="TRADE_REJECTED"
            )

    def _log_validation_result(
        self,
        signal: TradingSignal,
        is_valid: bool,
        block_reason: Optional[str],
        checks_passed: list,
        checks_failed: list,
        validation_details: dict
    ) -> None:
        """Log validation result to CSV audit trail.

        Args:
            signal: Trading signal
            is_valid: Overall validation result
            block_reason: Reason for rejection
            checks_passed: List of passed checks
            checks_failed: List of failed checks
            validation_details: Details from each check
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if file exists
        file_exists = (
            audit_path.exists() and
            audit_path.stat().st_size > 0
        )

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "signal_id",
                    "is_valid",
                    "block_reason",
                    "checks_passed",
                    "checks_failed",
                    "emergency_stop_status",
                    "daily_loss_status",
                    "drawdown_status",
                    "position_size_status",
                    "circuit_breaker_status",
                    "news_event_status",
                    "per_trade_risk_status"
                ])

            # Extract status from each check
            def get_status(check_name: str) -> str:
                if check_name in validation_details:
                    return validation_details[check_name].get('status', 'UNKNOWN')
                return 'NOT_CHECKED'

            # Write event
            writer.writerow([
                timestamp,
                signal.signal_id,
                is_valid,
                block_reason or "",
                ",".join(checks_passed),
                ",".join(checks_failed),
                get_status('emergency_stop'),
                get_status('daily_loss'),
                get_status('drawdown'),
                get_status('position_size'),
                get_status('circuit_breaker'),
                get_status('news_events'),
                get_status('per_trade_risk')
            ])

        logger.debug("Risk orchestration audit logged")

    def _get_current_time(self) -> datetime:
        """Get current time (UTC).

        Returns:
            Current datetime in UTC
        """
        return datetime.now(timezone.utc)
