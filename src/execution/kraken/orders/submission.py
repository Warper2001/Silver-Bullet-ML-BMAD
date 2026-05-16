"""Kraken Futures order placement and cancellation."""

import logging
from typing import Optional

import httpx

from src.execution.kraken.auth.api_key import KrakenFuturesAuth
from src.execution.kraken.exceptions import KrakenAPIError, KrakenAuthError, KrakenOrderError

logger = logging.getLogger(__name__)

BASE_URL = "https://demo-futures.kraken.com/derivatives/api/v3"
SENDORDER_ENDPOINT = "/derivatives/api/v3/sendorder"
CANCELORDER_ENDPOINT = "/derivatives/api/v3/cancelorder"


class KrakenOrdersClient:
    """Submits and cancels orders on Kraken Futures (demo environment for paper trading)."""

    def __init__(self, auth: KrakenFuturesAuth, http_client: httpx.AsyncClient) -> None:
        self._auth = auth
        self._client = http_client

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """Submit an order to Kraken Futures demo.

        Args:
            symbol: e.g. "PF_XBTUSD"
            side: "buy" or "sell"
            order_type: "lmt", "stp", or "mkt"
            size: Number of contracts (integer)
            limit_price: Required for "lmt" orders
            stop_price: Required for "stp" orders

        Returns:
            order_id string from Kraken

        Raises:
            KrakenAuthError: On 401 response
            KrakenOrderError: On rejected order
            KrakenAPIError: On other HTTP errors
        """
        post_data: dict = {
            "orderType": order_type,
            "symbol": symbol,
            "side": side,
            "size": size,
        }
        if limit_price is not None:
            post_data["limitPrice"] = limit_price
        if stop_price is not None:
            post_data["stopPrice"] = stop_price

        headers = self._auth.get_headers(SENDORDER_ENDPOINT, post_data)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            response = await self._client.post(
                f"{BASE_URL}/sendorder", data=post_data, headers=headers, timeout=15.0
            )
        except httpx.RequestError as exc:
            raise KrakenAPIError(0, str(exc)) from exc

        if response.status_code == 401:
            raise KrakenAuthError(f"Kraken auth failed: {response.text[:200]}")
        if response.status_code not in (200, 201):
            raise KrakenAPIError(response.status_code, response.text)

        body = response.json()
        result = body.get("result", "")
        if result != "success":
            raw_err = body.get("error", body.get("sendStatus", {}))
            raise KrakenOrderError(f"Order rejected: {raw_err}", raw=body)

        send_status = body.get("sendStatus", {})
        order_id = send_status.get("order_id", "")
        if not order_id:
            raise KrakenOrderError("No order_id in sendStatus", raw=body)

        logger.info(f"Order placed: {side} {size}x {symbol} @ {limit_price or stop_price} → id={order_id}")
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Kraken order UUID

        Returns:
            True if cancelled (or already gone), False on unexpected error
        """
        post_data = {"order_id": order_id}
        headers = self._auth.get_headers(CANCELORDER_ENDPOINT, post_data)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            response = await self._client.post(
                f"{BASE_URL}/cancelorder", data=post_data, headers=headers, timeout=10.0
            )
        except httpx.RequestError as exc:
            logger.warning(f"Cancel request error for {order_id}: {exc}")
            return False

        if response.status_code == 401:
            raise KrakenAuthError(f"Kraken auth failed on cancel: {response.text[:200]}")

        body = response.json()
        result = body.get("result", "")
        if result == "success":
            logger.info(f"Order {order_id} cancelled")
            return True

        # Order may already be filled/gone — treat as OK
        status = body.get("cancelStatus", {}).get("status", "")
        if status in ("notFound", "filled"):
            logger.debug(f"Order {order_id} cancel skipped — status: {status}")
            return True

        logger.warning(f"Unexpected cancel response for {order_id}: {body}")
        return False
