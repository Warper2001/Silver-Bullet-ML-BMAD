"""Kraken Spot REST API client — market orders and balance queries."""

import asyncio
import logging
from datetime import datetime, timezone
from types import TracebackType
from typing import Optional, Type

import httpx

from src.execution.kraken.exceptions import KrakenAPIError, KrakenAuthError, KrakenOrderError
from src.execution.kraken.spot.auth import KrakenSpotAuth
from src.execution.kraken.spot.models import SpotOrderResult

logger = logging.getLogger(__name__)

SPOT_BASE_URL   = "https://api.kraken.com"
ADD_ORDER_PATH  = "/0/private/AddOrder"
QUERY_ORDERS_PATH = "/0/private/QueryOrders"
BALANCE_PATH    = "/0/private/Balance"

BTC_PAIR        = "XXBTZUSD"
BTC_BALANCE_KEY = "XXBT"
USD_BALANCE_KEY = "ZUSD"


class KrakenSpotClient:
    """Async context manager for Kraken Spot REST API.

    Usage:
        async with KrakenSpotClient() as client:
            result = await client.place_market_order("buy", 0.1)
            usd = await client.get_usd_balance()
    """

    def __init__(self) -> None:
        self._auth = KrakenSpotAuth()
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "KrakenSpotClient":
        self._http = httpx.AsyncClient(base_url=SPOT_BASE_URL, timeout=15.0)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val:  Optional[BaseException],
        exc_tb:   Optional[TracebackType],
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_market_order(self, side: str, volume_btc: float) -> SpotOrderResult:
        """Place a BTC/USD spot market order and wait for fill confirmation.

        Args:
            side:       "buy" or "sell"
            volume_btc: BTC quantity (minimum 0.0001)

        Returns:
            SpotOrderResult with actual fill price and volume.

        Raises:
            KrakenAuthError:  Credential problem.
            KrakenOrderError: Order rejected or timed out waiting for fill.
            KrakenAPIError:   HTTP-level error.
        """
        assert self._http is not None, "Use as async context manager"

        post_data: dict = {
            "pair":      BTC_PAIR,
            "type":      side,
            "ordertype": "market",
            "volume":    str(round(volume_btc, 8)),
        }
        headers = self._auth.get_headers(ADD_ORDER_PATH, post_data)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            resp = await self._http.post(ADD_ORDER_PATH, data=post_data, headers=headers)
        except httpx.RequestError as exc:
            raise KrakenAPIError(0, str(exc)) from exc

        if resp.status_code == 403:
            raise KrakenAuthError(f"Kraken Spot auth failed: {resp.text[:200]}")
        if resp.status_code != 200:
            raise KrakenAPIError(resp.status_code, resp.text)

        body = resp.json()
        if body.get("error"):
            raise KrakenOrderError(f"AddOrder error: {body['error']}", raw=body)

        txid_list = body.get("result", {}).get("txid", [])
        if not txid_list:
            raise KrakenOrderError("No txid in AddOrder response", raw=body)

        txid = txid_list[0]
        logger.info(f"Spot {side} placed: {volume_btc} BTC → txid={txid}")
        return await self._confirm_fill(txid, volume_btc, side)

    async def _confirm_fill(
        self, txid: str, expected_volume: float, side: str
    ) -> SpotOrderResult:
        """Poll QueryOrders until the market order is closed (filled).

        Retries once after 2 seconds. Market orders on liquid pairs fill in <200ms
        under normal conditions; the retry covers momentary API lag.

        Raises:
            KrakenOrderError: If order is not closed after 2 attempts.
        """
        assert self._http is not None

        for attempt in range(2):
            if attempt > 0:
                await asyncio.sleep(2.0)

            post_data: dict = {"txid": txid, "trades": "true"}
            headers = self._auth.get_headers(QUERY_ORDERS_PATH, post_data)
            headers["Content-Type"] = "application/x-www-form-urlencoded"

            try:
                resp = await self._http.post(
                    QUERY_ORDERS_PATH, data=post_data, headers=headers
                )
            except httpx.RequestError as exc:
                raise KrakenAPIError(0, str(exc)) from exc

            if resp.status_code != 200:
                raise KrakenAPIError(resp.status_code, resp.text)

            body = resp.json()
            if body.get("error"):
                raise KrakenOrderError(f"QueryOrders error: {body['error']}", raw=body)

            order = body.get("result", {}).get(txid, {})
            status = order.get("status", "")

            if status == "closed":
                vol_exec   = float(order.get("vol_exec", expected_volume))
                fill_price = float(order.get("price", 0.0))
                logger.info(
                    f"Spot {side} confirmed: vol_exec={vol_exec} BTC "
                    f"@ ${fill_price:,.2f}  txid={txid}"
                )
                return SpotOrderResult(
                    txid=txid,
                    side=side,
                    volume_btc=expected_volume,
                    vol_exec=vol_exec,
                    fill_price=fill_price,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            logger.debug(f"Order {txid} status={status!r} (attempt {attempt + 1}/2)")

        raise KrakenOrderError(
            f"Market order {txid} did not fill within 4s (status={status!r})",
            raw={"txid": txid, "last_status": status},
        )

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------

    async def _get_balance_field(self, key: str) -> float:
        assert self._http is not None

        post_data: dict = {}
        headers = self._auth.get_headers(BALANCE_PATH, post_data)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            resp = await self._http.post(BALANCE_PATH, data=post_data, headers=headers)
        except httpx.RequestError as exc:
            raise KrakenAPIError(0, str(exc)) from exc

        if resp.status_code == 403:
            raise KrakenAuthError(f"Kraken Spot auth failed on Balance: {resp.text[:200]}")
        if resp.status_code != 200:
            raise KrakenAPIError(resp.status_code, resp.text)

        body = resp.json()
        if body.get("error"):
            raise KrakenAPIError(resp.status_code, str(body["error"]))

        return float(body.get("result", {}).get(key, 0.0))

    async def get_btc_balance(self) -> float:
        """Return available BTC balance (XXBT key)."""
        return await self._get_balance_field(BTC_BALANCE_KEY)

    async def get_usd_balance(self) -> float:
        """Return available USD balance (ZUSD key)."""
        return await self._get_balance_field(USD_BALANCE_KEY)
