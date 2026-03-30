"""Business logic validation layer for market data.

This module provides validation for market data beyond Pydantic model validation,
implementing trading-specific business rules to determine if data is tradable.
"""

import logging
from typing import Any

from src.data.models import DollarBar

logger = logging.getLogger(__name__)


class MarketDataValidator:
    """Business logic validation for market data.

    This class validates market data for trading suitability, checking for
    closed markets, stale data, and other trading-specific conditions.

    Example:
        validator = MarketDataValidator()
        is_tradable = validator.validate_bar_for_trading(bar)
        status = validator.validate_market_status(quote)
    """

    def validate_bar_for_trading(self, bar: DollarBar) -> bool:
        """Validate if a DollarBar has meaningful price data for trading.

        This method implements business logic to determine if a bar contains
        usable trading data, as opposed to just passing Pydantic validation.

        A bar is considered tradable if:
        - Not all price fields are zero (closed market)
        - Close price is non-zero (trading activity occurred)
        - High >= Low (price data is internally consistent)

        Args:
            bar: DollarBar to validate

        Returns:
            True if bar is tradable, False otherwise

        Example:
            >>> bar = DollarBar(
            ...     timestamp=datetime.now(),
            ...     open=11800.0, high=11850.0, low=11790.0, close=11820.0,
            ...     volume=1000, notional_value=50_000_000
            ... )
            >>> validator.validate_bar_for_trading(bar)
            True
        """
        # Check if all prices are zero (closed market or no data)
        if bar.high == 0 and bar.low == 0:
            logger.debug("Bar rejected: high and low are zero (closed market)")
            return False

        # Check if close price is zero (no trading activity)
        if bar.close == 0:
            logger.debug("Bar rejected: close price is zero (no trading activity)")
            return False

        # Check price consistency (high should be >= low)
        if bar.high < bar.low:
            logger.warning(
                f"Bar rejected: high ({bar.high}) < low ({bar.low}), "
                "invalid price data"
            )
            return False

        # Check for reasonable price data (all prices should be non-negative)
        if bar.open < 0 or bar.high < 0 or bar.low < 0 or bar.close < 0:
            logger.warning("Bar rejected: negative price detected")
            return False

        # Check for reasonable notional value
        if bar.notional_value == 0 and not bar.is_forward_filled:
            logger.debug("Bar rejected: zero notional value (no trading activity)")
            return False

        # Bar passed all validation checks
        logger.debug(
            f"Bar accepted: open={bar.open}, high={bar.high}, "
            f"low={bar.low}, close={bar.close}, volume={bar.volume}"
        )
        return True

    def validate_market_status(self, quote: dict[str, Any]) -> dict[str, Any]:
        """Validate market status from quote data.

        This method analyzes a quote from the TradeStation API to determine
        if the market is open, closed, or has no data.

        Args:
            quote: Quote data from TradeStation API with fields:
                - Last: Last trade price
                - Bid: Current bid price
                - Ask: Current ask price
                - Volume: Trading volume
                - ExpirationDate: Contract expiration date (optional)

        Returns:
            Dictionary with market status:
                - is_tradable: bool - Whether market is tradable
                - market_state: str - "open", "closed", "no_data", "expired"
                - reason: str - Human-readable explanation
                - last_price: float | None - Last price from quote
                - bid_ask_spread: float | None - Bid-ask spread if available

        Example:
            >>> quote = {"Last": 11800.0, "Bid": 11799.0, "Ask": 11801.0}
            >>> validator.validate_market_status(quote)
            {
                "is_tradable": True,
                "market_state": "open",
                "reason": "Market is open with active quotes",
                "last_price": 11800.0,
                "bid_ask_spread": 2.0
            }
        """
        last_price = quote.get("Last")
        bid_price = quote.get("Bid")
        ask_price = quote.get("Ask")
        volume = quote.get("Volume", 0)

        # Check for no data
        if last_price is None:
            return {
                "is_tradable": False,
                "market_state": "no_data",
                "reason": "No last price available (market closed or no data)",
                "last_price": None,
                "bid_ask_spread": None,
            }

        # Check for zero price (closed/expired market)
        if last_price == 0:
            if volume == 0:
                return {
                    "is_tradable": False,
                    "market_state": "closed",
                    "reason": "Zero price and volume (market closed or expired)",
                    "last_price": 0.0,
                    "bid_ask_spread": None,
                }
            else:
                return {
                    "is_tradable": False,
                    "market_state": "no_data",
                    "reason": "Zero price with volume (data anomaly)",
                    "last_price": 0.0,
                    "bid_ask_spread": None,
                }

        # Calculate bid-ask spread if both available
        bid_ask_spread = None
        if bid_price and ask_price:
            if bid_price > 0 and ask_price > 0:
                bid_ask_spread = ask_price - bid_price
            else:
                # Zero bid or ask indicates market is closed
                return {
                    "is_tradable": False,
                    "market_state": "closed",
                    "reason": "Zero bid or ask price (market closed)",
                    "last_price": float(last_price),
                    "bid_ask_spread": None,
                }

        # Market appears open and tradable
        return {
            "is_tradable": True,
            "market_state": "open",
            "reason": "Market is open with active quotes",
            "last_price": float(last_price),
            "bid_ask_spread": bid_ask_spread,
        }

    def get_trading_signals(self, bar: DollarBar) -> list[str]:
        """Analyze bar for trading signals and warnings.

        This method examines a DollarBar and returns a list of signals,
        warnings, or informational messages that may be relevant for trading.

        Args:
            bar: DollarBar to analyze

        Returns:
            List of signal descriptions (may be empty)

        Example:
            >>> bar = DollarBar(
            ...     timestamp=datetime.now(),
            ...     open=11800.0, high=11850.0, low=11790.0, close=11820.0,
            ...     volume=1000, notional_value=50_000_000
            ... )
            >>> validator.get_trading_signals(bar)
            ["Bullish close above open", "Moderate volatility"]
        """
        signals: list[str] = []

        # Price action signals
        if bar.close > bar.open:
            signals.append("Bullish close above open")
        elif bar.close < bar.open:
            signals.append("Bearish close below open")

        # Volatility signals (based on high-low range relative to close)
        if bar.high > bar.low:
            range_pct = ((bar.high - bar.low) / bar.close) * 100 if bar.close > 0 else 0
            if range_pct > 0.5:
                signals.append(f"High volatility: {range_pct:.2f}% range")
            elif range_pct > 0.2:
                signals.append(f"Moderate volatility: {range_pct:.2f}% range")
            else:
                signals.append(f"Low volatility: {range_pct:.2f}% range")

        # Volume signals
        if bar.volume == 0:
            signals.append("Warning: Zero volume (forward-filled or closed market)")

        # Forward fill warning
        if bar.is_forward_filled:
            signals.append("Warning: Bar was forward-filled due to data gap")

        # Notional value check
        if bar.notional_value > 0:
            # Dollar bar threshold is $50M
            if bar.notional_value >= 50_000_000:
                signals.append(
                    f"Dollar bar threshold met: ${bar.notional_value:,.0f}"
                )
            else:
                signals.append(
                    f"Partial bar: ${bar.notional_value:,.0f} "
                    f"({bar.notional_value / 50_000_000 * 100:.1f}% of threshold)"
                )

        return signals

    def get_trading_recommendation(
        self, bar: DollarBar, market_status: dict[str, Any]
    ) -> dict[str, Any]:
        """Get trading recommendation based on bar and market status.

        This method combines bar validation and market status to provide
        a trading recommendation.

        Args:
            bar: DollarBar to evaluate
            market_status: Market status from validate_market_status()

        Returns:
            Dictionary with recommendation:
                - should_trade: bool - Whether to consider trading
                - confidence: str - "high", "medium", "low", or "none"
                - reasons: list[str] - List of reasons for the recommendation
                - signals: list[str] - Trading signals from the bar

        Example:
            >>> market_status = validator.validate_market_status(quote)
            >>> recommendation = validator.get_trading_recommendation(bar, market_status)
        """
        reasons: list[str] = []
        should_trade = True
        confidence = "high"

        # Check market status
        if not market_status.get("is_tradable", False):
            should_trade = False
            confidence = "none"
            reasons.append(market_status.get("reason", "Market not tradable"))

        # Check bar validation
        if not self.validate_bar_for_trading(bar):
            should_trade = False
            if confidence != "none":
                confidence = "low"
            reasons.append("Bar does not meet trading criteria")

        # Check for forward fill
        if bar.is_forward_filled:
            should_trade = False
            if confidence not in ["none", "low"]:
                confidence = "medium"
            reasons.append("Forward-filled bar - data gap detected")

        # Check volume
        if bar.volume == 0:
            should_trade = False
            if confidence != "none":
                confidence = "low"
            reasons.append("Zero volume - no trading activity")

        # Get trading signals
        signals = self.get_trading_signals(bar)

        return {
            "should_trade": should_trade,
            "confidence": confidence,
            "reasons": reasons,
            "signals": signals,
        }
