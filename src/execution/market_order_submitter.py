"""Market order submission to TradeStation API.

This module handles market order submission for positions < 3 contracts
with immediate execution at best available price.

Features:
- Order payload construction for TradeStation API
- API submission with retry logic (max 3 attempts, exponential backoff)
- Order confirmation handling and parsing
- Active position tracking updates
- CSV audit trail logging
- Performance monitoring (<200ms submission time)
"""

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class OrderSubmissionError(Exception):
    """Exception raised when order submission fails."""

    pass


@dataclass
class OrderSubmissionResult:
    """Result of market order submission.

    Attributes:
        success: Whether order was accepted by exchange
        order_id: TradeStation order ID (None if failed)
        execution_price: Filled price in dollars (None if failed)
        submitted_quantity: Number of contracts submitted
        submission_time_ms: Time from submission to confirmation in milliseconds
        error_message: Error details if submission failed (None if successful)
    """

    success: bool
    order_id: str | None
    execution_price: float | None
    submitted_quantity: int
    submission_time_ms: float
    error_message: str | None


class MarketOrderSubmitter:
    """Submits market orders to TradeStation API.

    Handles order submission for positions < 3 contracts with immediate
    execution at best available price.

    Attributes:
        _api_client: TradeStation API client (authenticated)
        _position_tracker: Active position tracking storage
        _audit_trail_path: Path to CSV audit trail file
        _max_retries: Maximum retry attempts (default 3)
        _retry_delays: Exponential backoff delays in seconds [1s, 2s, 4s]

    Example:
        >>> api_client = TradeStationApiClient()
        >>> position_tracker = PositionTracker()
        >>> submitter = MarketOrderSubmitter(api_client, position_tracker)
        >>> result = submitter.submit_market_order(
        ...     signal, position_result, order_decision
        ... )
        >>> result.success
        True
        >>> result.order_id
        'ORDER-12345'
    """

    DEFAULT_MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff in seconds

    def __init__(
        self,
        api_client,
        position_tracker,
        audit_trail_path: str = "data/audit_trail.csv",
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize market order submitter.

        Args:
            api_client: Authenticated TradeStation API client
            position_tracker: Position tracking storage instance
            audit_trail_path: Path to CSV audit trail file
            max_retries: Maximum retry attempts (default 3)

        Raises:
            ValueError: If api_client or position_tracker is None
        """
        if api_client is None:
            raise ValueError("API client cannot be None")
        if position_tracker is None:
            raise ValueError("Position tracker cannot be None")

        self._api_client = api_client
        self._position_tracker = position_tracker
        self._audit_trail_path = audit_trail_path
        self._max_retries = max_retries

        logger.info(
            "MarketOrderSubmitter initialized: audit_trail={}, max_retries={}".format(
                audit_trail_path, max_retries
            )
        )

    def construct_order_payload(
        self,
        position_size: int,
        direction: str,
    ) -> dict:
        """Construct market order payload for TradeStation API.

        Args:
            position_size: Number of contracts to trade
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Order payload dictionary for TradeStation API

        Raises:
            ValueError: If position_size is invalid or direction not recognized

        Example:
            >>> submitter = MarketOrderSubmitter(api_client, tracker)
            >>> payload = submitter.construct_order_payload(2, "bullish")
            >>> payload['symbol']
            'MNQ'
            >>> payload['side']
            'BUY'
        """
        if position_size <= 0:
            raise ValueError(
                "Position size must be positive, got {}".format(position_size)
            )

        # Convert direction to API side
        if direction == "bullish":
            side = "BUY"
        elif direction == "bearish":
            side = "SELL"
        else:
            raise ValueError(
                "Invalid direction: {}. Expected 'bullish' or 'bearish'".format(
                    direction
                )
            )

        payload = {
            "symbol": "MNQ",
            "quantity": position_size,
            "side": side,
            "orderType": "MARKET",
            "timeInForce": "DAY",
        }

        logger.debug(
            "Order payload constructed: symbol={}, quantity={}, side={}, type={}"
            .format(
                payload["symbol"], payload["quantity"],
                payload["side"], payload["orderType"],
            )
        )
        return payload

    def submit_order_with_retry(self, payload: dict) -> dict:
        """Submit order to TradeStation API with retry logic.

        Args:
            payload: Order payload dictionary

        Returns:
            API response dictionary with order confirmation

        Raises:
            OrderSubmissionError: If all retry attempts fail or order rejected

        Performance:
            Completes in < 200ms (excluding network delays)
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                start_time = time.perf_counter()

                # Submit order
                response = self._api_client.submit_order(payload)

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Check if successful
                if response.get("success", False):
                    logger.info(
                        "Order submitted successfully: attempt={}, "
                        "order_id={}, time={:.2f}ms".format(
                            attempt + 1,
                            response.get("order_id"),
                            elapsed_ms
                        )
                    )
                    return response
                else:
                    # Order rejected, don't retry
                    error_msg = response.get("error", "Unknown error")
                    raise OrderSubmissionError(
                        "Order rejected by API: {}".format(error_msg)
                    )

            except OrderSubmissionError:
                # Don't retry rejected orders
                raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "API error on attempt {}: {}".format(attempt + 1, str(e))
                )

                # Don't sleep after last attempt
                if attempt < self._max_retries - 1:
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    logger.info("Retrying in {} seconds...".format(delay))
                    time.sleep(delay)

        # All retries failed
        raise OrderSubmissionError(
            "Failed to submit order after {} attempts: {}".format(
                self._max_retries, str(last_error)
            )
        )

    def log_to_audit_trail(
        self,
        signal_id: str,
        order_id: str,
        quantity: int,
        execution_price: float,
        order_type: str,
    ) -> None:
        """Log order submission to CSV audit trail.

        Args:
            signal_id: Signal identifier
            order_id: TradeStation order ID
            quantity: Number of contracts
            execution_price: Filled price
            order_type: Order type (MARKET or LIMIT)

        Example:
            >>> submitter = MarketOrderSubmitter(api_client, tracker)
            >>> submitter.log_to_audit_trail(
            ...     "SIG-123", "ORDER-456", 2, 11800.50, "MARKET"
            ... )
            # Logs to CSV audit trail
        """
        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if file exists and has content
        file_exists = audit_path.exists() and audit_path.stat().st_size > 0

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow(
                    ["timestamp", "signal_id", "order_id", "quantity",
                     "execution_price", "order_type"]
                )

            # Write log entry
            writer.writerow(
                [
                    timestamp,
                    signal_id,
                    order_id,
                    quantity,
                    "{:.2f}".format(execution_price),
                    order_type,
                ]
            )

        logger.debug("Audit trail updated: {}".format(audit_path))

    def submit_market_order(
        self,
        signal,
        position_size_result,
        order_type_decision,
    ) -> OrderSubmissionResult:
        """Submit market order for a signal.

        Args:
            signal: Silver Bullet setup with signal details
            position_size_result: Position sizing result from Story 4.1
            order_type_decision: Order type decision from Story 4.2

        Returns:
            OrderSubmissionResult with submission details

        Raises:
            ValueError: If order_type_decision is not MARKET
            OrderSubmissionError: If order submission fails after retries

        Performance:
            Completes in < 200ms from submission to confirmation
        """
        start_time = time.perf_counter()

        # Validate order type
        if order_type_decision.order_type != "MARKET":
            raise ValueError(
                "Expected MARKET order type, got {}".format(
                    order_type_decision.order_type
                )
            )

        try:
            # Construct payload
            payload = self.construct_order_payload(
                position_size=position_size_result.position_size,
                direction=signal.direction
            )

            # Construct payload
            payload = self.construct_order_payload(
                position_size=position_size_result.position_size,
                direction=signal.direction
            )

            # Submit with retry
            response = self.submit_order_with_retry(payload)

            # Extract confirmation details
            order_id = response.get("order_id")
            execution_price = response.get("execution_price")

            # Update position tracking
            self._position_tracker.add_position(
                order_id=order_id,
                signal_id=signal.signal_id,
                entry_price=execution_price,
                quantity=position_size_result.position_size,
                direction=signal.direction,
                timestamp=datetime.now(timezone.utc),
            )

            # Calculate performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log to audit trail
            self.log_to_audit_trail(
                signal_id=signal.signal_id,
                order_id=order_id,
                quantity=position_size_result.position_size,
                execution_price=execution_price,
                order_type="MARKET"
            )

            # Log result
            logger.info(
                "Market order submitted: order_id={}, qty={}, price=${:.2f}, "
                "time={:.2f}ms".format(
                    order_id,
                    position_size_result.position_size,
                    execution_price,
                    elapsed_ms
                )
            )

            return OrderSubmissionResult(
                success=True,
                order_id=order_id,
                execution_price=execution_price,
                submitted_quantity=position_size_result.position_size,
                submission_time_ms=elapsed_ms,
                error_message=None,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Error submitting market order: {}".format(e))

            return OrderSubmissionResult(
                success=False,
                order_id=None,
                execution_price=None,
                submitted_quantity=position_size_result.position_size,
                submission_time_ms=elapsed_ms,
                error_message=str(e),
            )
