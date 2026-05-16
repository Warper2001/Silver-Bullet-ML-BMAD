"""Kraken Futures historical bar fetcher (REST polling, no WebSocket)."""

import logging
import time

import httpx

from src.execution.kraken.exceptions import KrakenAPIError
from src.execution.kraken.models import KrakenBar

logger = logging.getLogger(__name__)

CHARTS_BASE = "https://futures.kraken.com/api/charts/v1/trade"


class KrakenHistoryClient:
    """Fetches OHLCV bars from the Kraken Futures charts endpoint (public, no auth)."""

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    async def fetch_bars(
        self, symbol: str, interval: str = "1m", count: int = 2
    ) -> list[KrakenBar]:
        """Fetch the most recent `count` completed 1-minute bars.

        URL format: GET /api/charts/v1/trade/{symbol}/{resolution}
        Params: from/to as Unix timestamps in seconds.

        Args:
            symbol: e.g. "PF_XBTUSD"
            interval: Kraken resolution string ("1m", "5m", etc.)
            count: Number of bars to return (from most recent)

        Returns:
            List of KrakenBar, oldest-first, length ≤ count.
        """
        now_s = int(time.time())
        from_s = now_s - 60 * (count + 2)  # small buffer to ensure we get enough bars

        url = f"{CHARTS_BASE}/{symbol}/{interval}"
        params = {"from": from_s, "to": now_s}

        try:
            response = await self._client.get(url, params=params, timeout=15.0)
        except httpx.RequestError as exc:
            raise KrakenAPIError(0, str(exc)) from exc

        if response.status_code != 200:
            raise KrakenAPIError(response.status_code, response.text)

        data = response.json()
        candles = data.get("candles", [])
        bars = []
        for c in candles:
            try:
                bars.append(KrakenBar.from_candle(c))
            except (ValueError, Exception):
                continue

        # Drop the last bar (may still be forming); return `count` most-recent completed
        completed = bars[:-1] if len(bars) > 1 else bars
        return completed[-count:]
