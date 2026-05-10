"""Pydantic models for Kraken Futures API responses."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class KrakenOrder(BaseModel):
    """Represents a single Kraken Futures order."""

    model_config = ConfigDict(populate_by_name=True)

    order_id: str = Field(alias="order_id")
    symbol: str
    side: str
    size: float
    status: str
    fill_price: Optional[float] = Field(default=None, alias="filled_price")


class KrakenBar(BaseModel):
    """One OHLCV bar from Kraken Futures charts endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    time: datetime
    open: float = Field(alias="open")
    high: float = Field(alias="high")
    low: float = Field(alias="low")
    close: float = Field(alias="close")
    volume: float = Field(alias="volume")

    @classmethod
    def from_candle(cls, candle: dict) -> "KrakenBar":
        """Parse a candle dict returned by the /charts endpoint.

        The API returns time as milliseconds since epoch.
        Raises ValueError for zero-close candles (malformed / pre-open data).
        """
        ts_ms = candle.get("time", 0)
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        close = float(candle.get("close", 0))
        if close == 0:
            raise ValueError(f"Zero close price in candle at {ts}")
        return cls(
            time=ts,
            open=float(candle.get("open", close)),
            high=float(candle.get("high", close)),
            low=float(candle.get("low", close)),
            close=close,
            volume=float(candle.get("volume", 0)),
        )
