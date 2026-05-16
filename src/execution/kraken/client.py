"""KrakenFuturesClient — async context manager wrapping history + orders clients."""

from types import TracebackType
from typing import Optional, Type

import httpx

from src.execution.kraken.auth.api_key import KrakenFuturesAuth
from src.execution.kraken.market_data.history import KrakenHistoryClient
from src.execution.kraken.orders.submission import KrakenOrdersClient


class KrakenFuturesClient:
    """Async context manager providing access to Kraken Futures market data and orders.

    Usage:
        async with KrakenFuturesClient() as client:
            bars = await client.history.fetch_bars("PF_XBTUSD")
            order_id = await client.orders.place_order(...)
    """

    def __init__(self) -> None:
        self.auth = KrakenFuturesAuth()
        self._http: Optional[httpx.AsyncClient] = None
        self.history: KrakenHistoryClient
        self.orders: KrakenOrdersClient

    async def __aenter__(self) -> "KrakenFuturesClient":
        self._http = httpx.AsyncClient()
        self.history = KrakenHistoryClient(self._http)
        self.orders = KrakenOrdersClient(self.auth, self._http)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
