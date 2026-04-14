"""Trade execution pipeline integration.

This module orchestrates the complete end-to-end trading execution flow,
integrating signal processing, order submission, position monitoring, and
exit execution into a unified automated system.

Features:
- End-to-end signal processing pipeline
- Integration with all order execution components
- Position monitoring and automatic exit execution
- Time window filtering
- Risk-based position sizing
- Order type selection
- Comprehensive audit trail logging
- Error handling and recovery
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal for execution.

    Attributes:
        signal_id: Unique signal ID
        symbol: Trading symbol
        direction: "bullish" or "bearish"
        confidence_score: ML confidence score (0-1)
        timestamp: Signal generation timestamp
        entry_price: Desired entry price
        patterns: List of detected patterns
        prediction: ML prediction details
        quantity: Number of contracts to trade
        stop_loss_price: Stop loss price for risk validation
        take_profit_price: Take profit price for risk validation
    """

    signal_id: str
    symbol: str
    direction: str
    confidence_score: float
    timestamp: datetime
    entry_price: float
    patterns: list[str]
    prediction: dict
    quantity: int = 5  # Default quantity
    stop_loss_price: float | None = None  # Calculated from prediction
    take_profit_price: float | None = None  # Calculated from prediction


@dataclass
class PipelineResult:
    """Result of signal processing through pipeline.

    Attributes:
        success: Whether signal was successfully processed
        order_id: Submitted order ID (if successful)
        filled: Whether order was filled
        fill_price: Fill price (if filled)
        block_reason: Reason signal was blocked (if blocked)
        error_message: Error message (if failed)
        position_id: Position ID created
    """

    success: bool
    order_id: Optional[str]
    filled: bool
    fill_price: Optional[float]
    block_reason: Optional[str]
    error_message: Optional[str]
    position_id: Optional[str]


class TradeExecutionPipeline:
    """End-to-end automated trade execution pipeline.

    Orchestrates signal processing, order submission, position monitoring,
    and exit execution. Integrates all components from Epic 4.

    Attributes:
        _position_sizer: PositionSizeCalculator (Story 4.1)
        _order_type_selector: OrderTypeSelector (Story 4.2)
        _market_order_submitter: MarketOrderSubmitter (Story 4.3)
        _limit_order_submitter: LimitOrderSubmitter (Story 4.4)
        _partial_fill_handler: PartialFillHandler (Story 4.5)
        _barrier_calculator: TripleBarrierCalculator (Story 4.6)
        _barrier_monitor: TripleBarrierMonitor (Story 4.6)
        _exit_executor: TripleBarrierExitExecutor (Story 4.6)
        _position_monitoring_service: PositionMonitoringService (Story 4.7)
        _time_window_filter: TimeWindowFilter (Story 4.8)
        _audit_trail: ImmutableAuditTrail (Story 4.9)
        _position_tracker: PositionTracker
        _api_client: TradeStation API client

    Example:
        >>> pipeline = TradeExecutionPipeline(
        ...     api_client=api_client,
        ...     position_sizer=sizer,
        ...     order_type_selector=selector,
        ...     market_order_submitter=market_submitter,
        ...     limit_order_submitter=limit_submitter,
        ...     partial_fill_handler=fill_handler,
        ...     barrier_calculator=calculator,
        ...     barrier_monitor=monitor,
        ...     exit_executor=executor,
        ...     position_monitoring_service=monitoring_service,
        ...     time_window_filter=time_filter,
        ...     audit_trail=audit_trail
        ... )
        >>> result = pipeline.process_signal(signal)
        >>> if result.success:
        ...     print(f"Order submitted: {result.order_id}")
    """

    def __init__(
        self,
        api_client,
        position_sizer,
        order_type_selector,
        market_order_submitter,
        limit_order_submitter,
        partial_fill_handler,
        barrier_calculator,
        barrier_monitor,
        exit_executor,
        position_monitoring_service,
        time_window_filter,
        audit_trail,
        risk_orchestrator=None
    ) -> None:
        """Initialize trade execution pipeline.

        Args:
            api_client: TradeStation API client
            position_sizer: PositionSizeCalculator instance
            order_type_selector: OrderTypeSelector instance
            market_order_submitter: MarketOrderSubmitter instance
            limit_order_submitter: LimitOrderSubmitter instance
            partial_fill_handler: PartialFillHandler instance
            barrier_calculator: TripleBarrierCalculator instance
            barrier_monitor: TripleBarrierMonitor instance
            exit_executor: TripleBarrierExitExecutor instance
            position_monitoring_service: PositionMonitoringService instance
            time_window_filter: TimeWindowFilter instance
            audit_trail: ImmutableAuditTrail instance
            risk_orchestrator: RiskOrchestrator instance (optional, for Story 4.2)

        Raises:
            ValueError: If required dependencies are None
        """
        # Validate required dependencies
        if api_client is None:
            raise ValueError("API client cannot be None")
        if position_sizer is None:
            raise ValueError("Position sizer cannot be None")
        if order_type_selector is None:
            raise ValueError("Order type selector cannot be None")
        if market_order_submitter is None:
            raise ValueError("Market order submitter cannot be None")
        if limit_order_submitter is None:
            raise ValueError("Limit order submitter cannot be None")
        if partial_fill_handler is None:
            raise ValueError("Partial fill handler cannot be None")
        if barrier_calculator is None:
            raise ValueError("Barrier calculator cannot be None")
        if barrier_monitor is None:
            raise ValueError("Barrier monitor cannot be None")
        if exit_executor is None:
            raise ValueError("Exit executor cannot be None")
        if position_monitoring_service is None:
            raise ValueError("Position monitoring service cannot be None")
        if time_window_filter is None:
            raise ValueError("Time window filter cannot be None")
        if audit_trail is None:
            raise ValueError("Audit trail cannot be None")

        # Store dependencies
        self._api_client = api_client
        self._position_sizer = position_sizer
        self._order_type_selector = order_type_selector
        self._market_order_submitter = market_order_submitter
        self._limit_order_submitter = limit_order_submitter
        self._partial_fill_handler = partial_fill_handler
        self._barrier_calculator = barrier_calculator
        self._barrier_monitor = barrier_monitor
        self._exit_executor = exit_executor
        self._position_monitoring_service = position_monitoring_service
        self._time_window_filter = time_window_filter
        self._audit_trail = audit_trail

        # Get position tracker from monitoring service
        self._position_tracker = position_monitoring_service._position_tracker

        # Initialize risk validator if risk orchestrator provided
        self._risk_validator = None
        if risk_orchestrator is not None:
            from src.execution.risk_integration import RiskValidator
            self._risk_validator = RiskValidator(risk_orchestrator)
            logger.info("TradeExecutionPipeline initialized with risk validation")
        else:
            logger.info("TradeExecutionPipeline initialized without risk validation")

    def process_signal(
        self,
        signal: TradingSignal
    ) -> PipelineResult:
        """Process trading signal through execution pipeline.

        Pipeline Flow:
            1. Validate signal
            2. Check time window (Story 4.8)
            3. Validate through risk management (Story 4.2)
            4. Calculate position size (Story 4.1)
            5. Select order type (Story 4.2)
            6. Submit order (Story 4.3 or 4.4)
            7. Handle partial fills (Story 4.5)
            8. Calculate barriers (Story 4.6)
            9. Start position monitoring (Story 4.7)
            10. Log all actions (Story 4.9)

        Args:
            signal: TradingSignal with signal details

        Returns:
            PipelineResult with execution status

        Example:
            >>> signal = TradingSignal(
            ...     signal_id="SIG-123",
            ...     symbol="MNQ 03-26",
            ...     direction="bullish",
            ...     confidence_score=0.85,
            ...     timestamp=datetime.now(timezone.utc),
            ...     entry_price=11800.00,
            ...     patterns=[],
            ...     prediction={}
            ... )
            >>> result = pipeline.process_signal(signal)
            >>> if result.success:
            ...     print(f"Order submitted: {result.order_id}")
        """
        try:
            # Step 1: Validate signal
            is_valid, error_message = self._validate_signal(signal)
            if not is_valid:
                logger.warning(
                    "Signal validation failed: {} - {}".format(
                        signal.signal_id, error_message
                    )
                )
                return PipelineResult(
                    success=False,
                    order_id=None,
                    filled=False,
                    fill_price=None,
                    block_reason=error_message,
                    error_message=None,
                    position_id=None
                )

            # Step 2: Check time window
            is_allowed, block_reason = self._check_time_window(signal)
            if not is_allowed:
                logger.info(
                    "Signal blocked by time window: {} - {}".format(
                        signal.signal_id, block_reason
                    )
                )
                return PipelineResult(
                    success=False,
                    order_id=None,
                    filled=False,
                    fill_price=None,
                    block_reason=block_reason,
                    error_message=None,
                    position_id=None
                )

            # Step 3: Validate through risk management (Story 4.2)
            if self._risk_validator is not None:
                risk_result = self._validate_risk(signal)
                if not risk_result['is_valid']:
                    logger.info(
                        "Signal blocked by risk validation: {} - {}".format(
                            signal.signal_id, risk_result['block_reason']
                        )
                    )
                    # Log to audit trail
                    self._audit_trail.log_event(
                        event_type="RISK_VALIDATION_FAILED",
                        details={
                            "signal_id": signal.signal_id,
                            "block_reason": risk_result['block_reason'],
                            "checks_failed": risk_result['checks_failed']
                        }
                    )
                    return PipelineResult(
                        success=False,
                        order_id=None,
                        filled=False,
                        fill_price=None,
                        block_reason=risk_result['block_reason'],
                        error_message=None,
                        position_id=None
                    )

            # Step 4: Calculate position size
            position_size = self._calculate_position_size(signal)

            # Step 5: Select order type
            order_type = self._select_order_type(position_size)

            # Step 6: Submit order
            order_result = self._submit_order(
                signal,
                position_size,
                order_type
            )

            if not order_result.success:
                logger.error(
                    "Order submission failed: {} - {}".format(
                        signal.signal_id, order_result.error_message
                    )
                )
                return PipelineResult(
                    success=False,
                    order_id=None,
                    filled=False,
                    fill_price=None,
                    block_reason=None,
                    error_message=order_result.error_message,
                    position_id=None
                )

            # Step 7: Handle partial fills (for limit orders)
            order_result = self._submit_order(
                signal,
                position_size,
                order_type
            )

            if not order_result.success:
                logger.error(
                    "Order submission failed: {} - {}".format(
                        signal.signal_id, order_result.error_message
                    )
                )
                return PipelineResult(
                    success=False,
                    order_id=None,
                    filled=False,
                    fill_price=None,
                    block_reason=None,
                    error_message=order_result.error_message,
                    position_id=None
                )

            # Step 6: Handle partial fills (for limit orders)
            if order_type == "LIMIT" and not order_result.filled:
                fill_result = self._partial_fill_handler.monitor_fills(
                    order_result.order_id,
                    signal,
                    order_type
                )
                # Update order result with fill status
                order_result.filled = fill_result.filled
                order_result.filled_price = fill_result.fill_price

            # Step 8: Initialize position monitoring
            if order_result.filled:
                self._initialize_position_monitoring(
                    order_result.order_id,
                    (
                        order_result.fill_price
                        if order_result.fill_price
                        else signal.entry_price
                    ),
                    position_size,
                    signal.direction,
                    signal.timestamp
                )

            # Log success
            logger.info(
                "Signal processed successfully: {} -> {}".format(
                    signal.signal_id, order_result.order_id
                )
            )

            return PipelineResult(
                success=True,
                order_id=order_result.order_id,
                filled=order_result.filled,
                fill_price=order_result.fill_price,
                block_reason=None,
                error_message=None,
                position_id=order_result.order_id
            )

        except Exception as e:
            # Log error to audit trail
            self._audit_trail.log_error(
                order_id=None,
                error_message=str(e),
                stack_trace=None  # Could add traceback here
            )

            logger.error(
                "Signal processing error: {} - {}".format(
                    signal.signal_id, e
                )
            )

            return PipelineResult(
                success=False,
                order_id=None,
                filled=False,
                fill_price=None,
                block_reason=None,
                error_message=str(e),
                position_id=None
            )

    def on_bar_update(
        self,
        bar
    ) -> list:
        """Handle new dollar bar update.

        Called by data pipeline when new dollar bar received.

        Args:
            bar: New dollar bar

        Returns:
            List of ExitResult for positions that exited

        Example:
            >>> exits = pipeline.on_bar_update(bar)
            >>> for exit_result in exits:
            ...     print(f"Exited: {exit_result.order_id}")
        """
        try:
            # Monitor positions for barrier hits
            exits = self._position_monitoring_service.on_price_update(
                current_price=bar.close,
                current_time=bar.timestamp
            )

            if exits:
                logger.info(
                    "Bar update processed: {} positions exited".format(
                        len(exits)
                    )
                )

            return exits

        except Exception as e:
            logger.error("Bar update error: {}".format(e))
            self._audit_trail.log_error(
                order_id=None,
                error_message="Bar update error: {}".format(str(e)),
                stack_trace=None
            )
            return []

    def _validate_signal(
        self,
        signal: TradingSignal
    ) -> tuple[bool, Optional[str]]:
        """Validate signal has required fields.

        Args:
            signal: TradingSignal to validate

        Returns:
            Tuple of (is_valid, error_message)

        Validation Checks:
            - signal_id is not None
            - symbol is not None
            - direction is "bullish" or "bearish"
            - confidence_score is between 0 and 1
            - timestamp is not None
            - entry_price is not None
        """
        if not signal.signal_id:
            return False, "Signal ID is required"

        if not signal.symbol:
            return False, "Symbol is required"

        if signal.direction not in ["bullish", "bearish"]:
            return False, "Direction must be 'bullish' or 'bearish'"

        if not (0.0 <= signal.confidence_score <= 1.0):
            return False, "Confidence score must be between 0 and 1"

        if not signal.timestamp:
            return False, "Timestamp is required"

        if not signal.entry_price or signal.entry_price <= 0:
            return False, "Valid entry price is required"

        return True, None

    def _check_time_window(
        self,
        signal: TradingSignal
    ) -> tuple[bool, Optional[str]]:
        """Check if signal timestamp is within allowed trading window.

        Args:
            signal: TradingSignal to check

        Returns:
            Tuple of (is_allowed, block_reason)

        Example:
            >>> allowed, reason = self._check_time_window(signal)
            >>> if not allowed:
            ...     return PipelineResult(
            ...         success=False,
            ...         block_reason=reason
            ...     )
        """
        time_window_result = self._time_window_filter.check_time_window(
            signal.timestamp
        )

        if time_window_result.allowed:
            return True, None
        else:
            return False, time_window_result.reason

    def _validate_risk(
        self,
        signal: TradingSignal
    ) -> dict:
        """Validate trading signal through risk management layers.

        Args:
            signal: TradingSignal to validate

        Returns:
            Dictionary with validation result from RiskValidator

        Integration:
            - Story 4.2: Risk Management Integration
            - Validates through all 8 risk layers
            - Blocks trades when risk limits exceeded

        Example:
            >>> result = self._validate_risk(signal)
            >>> if not result['is_valid']:
            ...     return PipelineResult(
            ...         success=False,
            ...         block_reason=result['block_reason']
            ...     )
        """
        return self._risk_validator.validate_trade(signal)

    def _calculate_position_size(
        self,
        signal: TradingSignal
    ) -> int:
        """Calculate position size based on risk parameters.

        Args:
            signal: TradingSignal with entry price

        Returns:
            Position size in contracts

        Integration:
            - Uses PositionSizeCalculator (Story 4.1)
            - Accounts for risk per trade ($150)
            - Accounts for MNQ tick size and offset
        """
        return self._position_sizer.calculate_position_size(
            signal.entry_price
        )

    def _select_order_type(
        self,
        position_size: int
    ) -> str:
        """Select order type based on position size.

        Args:
            position_size: Calculated position size

        Returns:
            "MARKET" or "LIMIT"

        Integration:
            - Uses OrderTypeSelector (Story 4.2)
            - Small positions (< 3 contracts): MARKET
            - Large positions (≥ 3 contracts): LIMIT
        """
        return self._order_type_selector.select_order_type(
            position_size
        )

    def _submit_order(
        self,
        signal: TradingSignal,
        position_size: int,
        order_type: str
    ):
        """Submit order to TradeStation API.

        Args:
            signal: TradingSignal with signal details
            position_size: Position size in contracts
            order_type: Order type (MARKET or LIMIT)

        Returns:
            OrderResult with submission status

        Integration:
            - If MARKET: use MarketOrderSubmitter (Story 4.3)
            - If LIMIT: use LimitOrderSubmitter (Story 4.4)
            - Log submission to audit trail (Story 4.9)

        Example:
            >>> result = self._submit_order(signal, 5, "LIMIT")
            >>> if result.success:
            ...     order_id = result.order_id
            >>> else:
            ...     error = result.error_message
        """
        if order_type == "MARKET":
            result = self._market_order_submitter.submit_market_order(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action="BUY" if signal.direction == "bullish" else "SELL",
                quantity=position_size
            )
        else:  # LIMIT
            # Calculate limit price (use entry price from signal)
            limit_price = signal.entry_price

            result = self._limit_order_submitter.submit_limit_order(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action="BUY" if signal.direction == "bullish" else "SELL",
                quantity=position_size,
                limit_price=limit_price
            )

        # Log submission to audit trail
        if result.success:
            self._audit_trail.log_order_submit(
                order_id=result.order_id,
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action="BUY" if signal.direction == "bullish" else "SELL",
                order_type=order_type,
                quantity=position_size,
                price=signal.entry_price if order_type == "LIMIT" else None
            )

        return result

    def _initialize_position_monitoring(
        self,
        order_id: str,
        entry_price: float,
        quantity: int,
        direction: str,
        entry_time: datetime
    ) -> None:
        """Initialize position monitoring with triple barriers.

        Args:
            order_id: Order ID from broker
            entry_price: Position entry price
            quantity: Position size
            direction: Position direction
            entry_time: Entry timestamp

        Integration:
            - Uses PositionMonitoringService (Story 4.7)
            - Calculates barriers (Story 4.6)
            - Starts monitoring for barrier hits
            - Logs position entry to audit trail
        """
        self._position_monitoring_service.on_position_entered(
            order_id=order_id,
            entry_price=entry_price,
            quantity=quantity,
            direction=direction,
            entry_time=entry_time
        )

        logger.info(
            "Position monitoring initialized: {} @ {}".format(
                order_id, entry_price
            )
        )
