"""Data models for Kraken Spot API responses."""

from dataclasses import dataclass


@dataclass
class SpotOrderResult:
    """Filled spot market order — returned by KrakenSpotClient.place_market_order."""
    txid:        str    # Kraken transaction ID, e.g. "OXXXXX-YYYYY-ZZZZZ"
    side:        str    # "buy" or "sell"
    volume_btc:  float  # requested volume
    vol_exec:    float  # actual filled volume
    fill_price:  float  # average fill price (USD)
    timestamp:   str    # ISO UTC string at time of confirmation
