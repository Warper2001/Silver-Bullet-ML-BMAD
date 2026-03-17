"""Order type selection based on position size.

This module determines whether to use market orders or limit orders
based on calculated position size:
- Position size < threshold (default 3): MARKET order for immediate execution
- Position size ≥ threshold (default 3): LIMIT order with tick offset

This balances execution speed and slippage reduction.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderTypeDecision:
    """Result of order type selection.

    Attributes:
        order_type: Order type ("MARKET" or "LIMIT")
        limit_price: Limit price in dollars (None for MARKET orders)
        rationale: Explanation of order type selection
        selection_time_ms: Decision time in milliseconds
    """

    order_type: str
    limit_price: float | None
    rationale: str
    selection_time_ms: float


class OrderTypeSelector:
    """Determines optimal order type based on position size.

    Uses threshold-based approach:
    - Position size < threshold (default 3): MARKET order (immediate execution)
    - Position size ≥ threshold (default 3): LIMIT order (tick offset)

    This balances execution speed and slippage reduction for different
    position sizes.

    Attributes:
        _market_order_threshold: Position size threshold for market orders
        _tick_size: MNQ tick size in dollars (default $0.25)
        _tick_offset: Number of ticks for limit order offset (default 2)

    Example:
        >>> selector = OrderTypeSelector()
        >>> result = selector.decide_order_type(
        ...     position_size=2,
        ...     direction="bullish",
        ...     current_price=11800.0
        ... )
        >>> result.order_type
        'MARKET'
        >>> result = selector.decide_order_type(
        ...     position_size=5,
        ...     direction="bullish",
        ...     current_price=11800.0
        ... )
        >>> result.order_type
        'LIMIT'
        >>> result.limit_price
        11800.50
    """

    MARKET_ORDER_THRESHOLD = 3  # contracts
    TICK_SIZE = 0.25  # MNQ tick size ($0.25)
    TICK_OFFSET = 2  # ticks offset for limit orders

    def __init__(
        self,
        market_order_threshold: int = MARKET_ORDER_THRESHOLD,
        tick_size: float = TICK_SIZE,
        tick_offset: int = TICK_OFFSET,
    ) -> None:
        """Initialize order type selector with parameters.

        Args:
            market_order_threshold: Position size threshold for market orders
                (default 3 contracts)
            tick_size: MNQ tick size in dollars (default $0.25)
            tick_offset: Number of ticks for limit order offset (default 2)

        Raises:
            ValueError: If parameters are invalid (negative values, zero threshold)
        """
        if market_order_threshold <= 0:
            raise ValueError(
                "Market order threshold must be positive, got {}".format(
                    market_order_threshold
                )
            )
        if tick_size <= 0:
            raise ValueError(
                "Tick size must be positive, got {}".format(tick_size)
            )
        if tick_offset <= 0:
            raise ValueError(
                "Tick offset must be positive, got {}".format(tick_offset)
            )

        self._market_order_threshold = market_order_threshold
        self._tick_size = tick_size
        self._tick_offset = tick_offset

        logger.info(
            "OrderTypeSelector initialized: threshold={}, "
            "tick_size=${:.2f}, tick_offset={}".format(
                market_order_threshold, tick_size, tick_offset
            )
        )

    def select_order_type(self, position_size: int) -> tuple[str, str]:
        """Select order type based on position size.

        Args:
            position_size: Calculated position size in contracts

        Returns:
            Tuple of (order_type: str, rationale: str)

        Example:
            >>> selector = OrderTypeSelector()
            >>> selector.select_order_type(position_size=2)
            ('MARKET', 'Position size 2 < 3: using market order...')
            >>> selector.select_order_type(position_size=5)
            ('LIMIT', 'Position size 5 >= 3: using limit order with 2-tick offset...')
        """
        if position_size < self._market_order_threshold:
            order_type = "MARKET"
            rationale = (
                "Position size {} < {}: "
                "using market order for immediate execution".format(
                    position_size, self._market_order_threshold
                )
            )
        else:
            order_type = "LIMIT"
            rationale = (
                "Position size {} >= {}: "
                "using limit order with {}-tick offset to reduce slippage".format(
                    position_size,
                    self._market_order_threshold,
                    self._tick_offset,
                )
            )

        logger.debug("Order type selected: {} ({})".format(order_type, rationale))
        return order_type, rationale

    def calculate_limit_price(
        self, current_price: float, direction: str
    ) -> float:
        """Calculate limit price with tick offset.

        Args:
            current_price: Current market price
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Limit price with tick offset applied

        Raises:
            ValueError: If current_price is invalid or direction is not recognized

        Example:
            >>> selector = OrderTypeSelector()
            >>> selector.calculate_limit_price(
            ...     current_price=11800.0, direction="bullish"
            ... )
            11800.50  # 11800 + (2 * $0.25)
            >>> selector.calculate_limit_price(
            ...     current_price=11800.0, direction="bearish"
            ... )
            11799.50  # 11800 - (2 * $0.25)
        """
        if current_price <= 0:
            raise ValueError(
                "Current price must be positive, got {}".format(current_price)
            )

        offset = self._tick_offset * self._tick_size

        if direction == "bullish":
            limit_price = current_price + offset
        elif direction == "bearish":
            limit_price = current_price - offset
        else:
            raise ValueError(
                "Invalid direction: {}. Expected 'bullish' or 'bearish'".format(
                    direction
                )
            )

        logger.debug(
            "Limit price calculated: ${:.2f} "
            "(current=${:.2f}, offset=${:.2f}, direction={})".format(
                limit_price, current_price, offset, direction
            )
        )
        return limit_price

    def decide_order_type(
        self,
        position_size: int,
        direction: str,
        current_price: float,
    ) -> OrderTypeDecision:
        """Decide order type and calculate limit price if needed.

        Args:
            position_size: Calculated position size in contracts
            direction: Signal direction ("bullish" or "bearish")
            current_price: Current market price

        Returns:
            OrderTypeDecision with order type, limit price, rationale, and timing

        Raises:
            ValueError: If position_size or current_price is invalid

        Performance:
            Completes in < 1ms per decision
        """
        import time

        start_time = time.perf_counter()

        try:
            # Validate inputs
            if position_size <= 0:
                raise ValueError(
                    "Position size must be positive, got {}".format(
                        position_size
                    )
                )

            # Select order type
            order_type, rationale = self.select_order_type(position_size)

            # Calculate limit price if needed
            limit_price = None
            if order_type == "LIMIT":
                limit_price = self.calculate_limit_price(current_price, direction)

            # Calculate performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log result
            logger.info(
                "Order type decision: type={}, limit_price={}, "
                "rationale={}, time={:.3f}ms".format(
                    order_type,
                    "${:.2f}".format(limit_price) if limit_price else "N/A",
                    rationale,
                    elapsed_ms,
                )
            )

            # Performance warning
            if elapsed_ms >= 1.0:
                logger.warning(
                    "Order type selection took {:.3f}ms, exceeds 1ms target".format(
                        elapsed_ms
                    )
                )

            return OrderTypeDecision(
                order_type=order_type,
                limit_price=limit_price,
                rationale=rationale,
                selection_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error("Error deciding order type: {}".format(e))
            raise
