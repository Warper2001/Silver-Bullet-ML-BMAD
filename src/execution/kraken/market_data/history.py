"""Kraken Futures historical bar fetcher (REST polling, no WebSocket)."""

import logging
import time

import httpx

from src.execution.kraken.exceptions import KrakenAPIError
from src.execution.kraken.models import KrakenBar

logger = logging.getLogger(__name__)

CHARTS_URL = "https://futures.kraken.com/derivatives/api/v3/charts"


class KrakenHistoryClient:
    """Fetches OHLCV bars from the Kraken Futures charts endpoint (public, no auth)."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    async def fetch_bars(
        self, symbol: str, interval: str = "1m", count: int = 2
    ) -> list[KrakenBar]:
        """Fetch the most recent `count` completed 1-minute bars.

        Args:
            symbol: e.g. "PF_XBTUSD"
            interval: Kraken resolution string ("1m", "5m", etc.)
            count: Number of bars to return (from most recent)

        Returns:
            List of KrakenBar, oldest-first, length ≤ count.
        """
        now_ms = int(time.time() * 1000)
        # Fetch 60 seconds × count bars worth of history
        interval_ms = 60_000  # 1m in milliseconds
        from_ms = now_ms - interval_ms * (count + 1)

        params = {
            "symbol": symbol,
            "resolution": interval,
            "from": from_ms,
            "to": now_ms,
        }

        try:
            response = await self._client.get(CHARTS_URL, params=params, timeout=15.0)
        except httpx.RequestError as exc:
            raise KrakenAPIError(0, str(exc)) from exc

        if response.status_code != 200:
            raise KrakenAPIError(response.status_code, response.text)

        data = response.json()
        candles = data.get("candles", [])
        bars = [KrakenBar.from_candle(c) for c in candles]

        # Return the `count` most-recent bars (last bar may still be forming — skip it)
        # Keep bars[:-1] (completed) then take last `count`.
        completed = bars[:-1] if len(bars) > 1 else bars
        return completed[-count:]
