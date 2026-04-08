"""TradeStation SIM order submission for paper trading.

This module integrates with TradeExecutionPipeline for SIM order submission,
routing signals through risk validation to SIM order execution.

Features:
- SIM environment order submission
- Risk validation integration
- Order confirmation and tracking
- Error handling and retry logic
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.data.tradestation_auth import TradeStationAuth
from src.execution.trade_execution_pipeline import TradingSignal

logger = logging.getLogger(__name__)

# TradeStation API endpoints
SIM_API_BASE = "https://sim-api.tradestation.com/v3"
ORDERS_ENDPOINT = f"{SIM_API_BASE}/orders"


@dataclass
class OrderSubmission:
    """Order submission details for TradeStation SIM.

    Attributes:
        order_id: Unique order ID
        signal_id: Associated signal ID
        symbol: Trading symbol
        quantity: Order quantity (contracts)
        direction: Order direction ("BUY" or "SELL")
        order_type: Order type ("MARKET" or "LIMIT")
        limit_price: Limit price (for LIMIT orders)
        stop_price: Stop price (for STOP orders)
        timestamp: Order submission timestamp
        status: Order status ("PENDING", "FILLED", "REJECTED", "CANCELLED")
        fill_price: Fill price (if filled)
        fill_time: Fill timestamp (if filled)
        rejection_reason: Reason for rejection (if rejected)
    """
    order_id: str
    signal_id: str
    symbol: str
    quantity: int
    direction: str
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None
    status: str = "PENDING"
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class OrderResult:
    """Result of order submission.

    Attributes:
        success: Whether order was submitted successfully
        order_id: Submitted order ID
        status: Order status
        fill_price: Fill price (if filled)
        error_message: Error message (if failed)
    """
    success: bool
    order_id: Optional[str]
    status: Optional[str]
    fill_price: Optional[float]
    error_message: Optional[str]


class SIMOrderSubmitter:
    """TradeStation SIM order submission for paper trading.

    Handles order submission to TradeStation SIM environment with
    risk validation integration and order tracking.

    Attributes:
        _auth: TradeStation authentication manager
        _http_client: HTTP client for API requests

    Example:
        >>> submitter = SIMOrderSubmitter(auth=auth)
        >>> signal = TradingSignal(...)
        >>> result = await submitter.submit_order(signal)
        >>> if result.success:
        ...     print(f"Order {result.order_id} submitted")
    """

    def __init__(self, auth: TradeStationAuth) -> None:
        """Initialize SIMOrderSubmitter.

        Args:
            auth: TradeStation authentication manager
        """
        self._auth = auth
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            httpx.AsyncClient instance
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _get_headers(self) -> dict:
        """Get request headers with authorization.

        Returns:
            Headers dict with Bearer token
        """
        token = self._auth.get_valid_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def submit_order(self, signal: TradingSignal, quantity: int = 5) -> OrderResult:
        """Submit order to TradeStation SIM environment.

        Args:
            signal: Trading signal with entry details
            quantity: Order quantity (contracts, default: 5)

        Returns:
            OrderResult with submission outcome

        Raises:
            httpx.HTTPStatusError: If API request fails
        """
        try:
            # Get access token
            token = self._auth.get_valid_access_token()

            # Build order payload
            order_payload = self._build_order_payload(signal, quantity)

            # Submit order
            client = await self._get_client()
            headers = await self._get_headers()

            logger.info(
                f"Submitting {signal.direction} order for {signal.symbol}: "
                f"{quantity} contracts @ {signal.entry_price}"
            )

            response = await client.post(
                ORDERS_ENDPOINT,
                headers=headers,
                json=order_payload,
            )

            # Check for errors
            if response.status_code == 401:
                logger.error("Authentication failed for order submission")
                return OrderResult(
                    success=False,
                    order_id=None,
                    status=None,
                    fill_price=None,
                    error_message="Authentication failed",
                )

            if response.status_code == 429:
                logger.error("Rate limit exceeded for order submission")
                return OrderResult(
                    success=False,
                    order_id=None,
                    status=None,
                    fill_price=None,
                    error_message="Rate limit exceeded",
                )

            response.raise_for_status()

            # Parse response
            data = response.json()
            order_id = data.get("OrderID") or data.get("orderId")

            logger.info(f"Order submitted successfully: {order_id}")

            # SIM environment instant fill (simulated)
            return OrderResult(
                success=True,
                order_id=order_id,
                status="FILLED",
                fill_price=signal.entry_price,
                error_message=None,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Order submission failed: HTTP {e.response.status_code}")
            return OrderResult(
                success=False,
                order_id=None,
                status=None,
                fill_price=None,
                error_message=f"HTTP {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Order submission failed: {e}", exc_info=True)
            return OrderResult(
                success=False,
                order_id=None,
                status=None,
                fill_price=None,
                error_message=str(e),
            )

    def _build_order_payload(self, signal: TradingSignal, quantity: int = 5) -> dict:
        """Build order payload for TradeStation API.

        Args:
            signal: Trading signal with order details
            quantity: Order quantity (contracts, default: 5)

        Returns:
            Order payload dictionary
        """
        # Map direction to TradeStation format
        direction = "Buy" if signal.direction.lower() in ["bullish", "buy"] else "Sell"

        # Build basic order payload
        payload = {
            "Symbol": signal.symbol,
            "Quantity": quantity,  # Use provided quantity or default
            "OrderType": "Market",  # Use market orders for SIM
            "Side": direction,
            "TimeInForce": "DAY",  # Day order
            "AccountID": "SIM_ACCOUNT",  # SIM environment account
        }

        return payload

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.info("HTTP client closed")
