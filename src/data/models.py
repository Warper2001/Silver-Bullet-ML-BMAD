"""Pydantic models for TradeStation market data."""

from datetime import datetime
from typing import Literal, Optional

import asyncio
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class MarketData(BaseModel):
    """Real-time market data from TradeStation WebSocket."""

    symbol: str = Field(..., description="Trading symbol (e.g., 'MNQ')")
    timestamp: datetime = Field(..., description="Data timestamp")
    bid: Optional[float] = Field(None, ge=0, description="Current bid price")
    ask: Optional[float] = Field(None, gt=0, description="Current ask price")
    last: Optional[float] = Field(None, gt=0, description="Last trade price")
    volume: int = Field(..., ge=0, description="Trade volume")

    @field_validator("ask")
    @classmethod
    def ask_must_be_greater_than_bid(
        cls, v: float | None, info
    ) -> float | None:  # type: ignore[no-untyped-def]
        """Validate ask price is greater than bid price."""
        if v is not None and info.data.get("bid") is not None:
            if v <= info.data["bid"]:
                raise ValueError("ask must be greater than bid")
        return v

    def has_required_fields(self) -> bool:
        """Check if all required market data fields are present."""
        return (
            self.bid is not None
            and self.ask is not None
            and self.last is not None
            and self.volume >= 0
        )


class WebSocketMessage(BaseModel):
    """Raw WebSocket message from TradeStation API."""

    symbol: str
    timestamp: str  # ISO 8601 format
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: int = 0

    def to_market_data(self) -> MarketData:
        """Convert to MarketData model with validation."""
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.fromisoformat(self.timestamp.replace("Z", "+00:00")),
            bid=self.bid,
            ask=self.ask,
            last=self.last,
            volume=self.volume,
        )


class DollarBar(BaseModel):
    """Dollar Bar with fixed $50M notional value threshold."""

    timestamp: datetime = Field(..., description="Bar completion timestamp")
    open: float = Field(..., gt=0, description="Open price (first trade)")
    high: float = Field(..., gt=0, description="High price (max trade)")
    low: float = Field(..., gt=0, description="Low price (min trade)")
    close: float = Field(..., gt=0, description="Close price (last trade)")
    volume: int = Field(..., ge=0, description="Total volume in bar")
    is_forward_filled: bool = Field(
        default=False, description="True if bar was forward-filled due to data gap"
    )
    notional_value: float = Field(..., ge=0, description="Notional value ($)")

    @field_validator("high")
    @classmethod
    def high_gte_open_and_close(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate high is >= open and close."""
        # Only validate against open (close not set yet when high is validated)
        open_val = info.data.get("open")
        if open_val is not None and v < open_val:
            raise ValueError("high must be >= open")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_and_close(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate low is <= open and close."""
        # Only validate against open (close not set yet when low is validated)
        open_val = info.data.get("open")
        if open_val is not None and v > open_val:
            raise ValueError("low must be <= open")
        return v

    @field_validator("close")
    @classmethod
    def close_within_high_low(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate close is within high and low."""
        high_val = info.data.get("high")
        low_val = info.data.get("low")
        if high_val is not None and v > high_val:
            raise ValueError("close must be <= high")
        if low_val is not None and v < low_val:
            raise ValueError("close must be >= low")
        return v

    @field_validator("notional_value")
    @classmethod
    def notional_value_sanity_check(
        cls, v: float, info
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate notional value is reasonable for Dollar Bars."""
        # Allow zero for forward-filled bars
        is_forward_filled = info.data.get("is_forward_filled", False)
        if v == 0 and is_forward_filled:
            return v

        # Notional should be positive for real bars
        if v <= 0:
            raise ValueError("notional_value must be positive")

        # Sanity check: notional should not exceed $2B (40× threshold)
        # This catches calculation errors or malformed data
        # Increased limit for historical data with high-volume periods
        max_reasonable = 2_000_000_000  # $2B
        if v > max_reasonable:
            raise ValueError(
                f"notional_value ${v:.2f} exceeds reasonable maximum ${max_reasonable:.2f}"
            )

        return v


class ValidationResult(BaseModel):
    """Validation result for DollarBar data quality check."""

    is_valid: bool = Field(..., description="Whether bar passed validation")
    timestamp: datetime = Field(..., description="Validation timestamp")
    errors: list[str] = Field(default_factory=list, description="Critical errors found")
    warnings: list[str] = Field(default_factory=list, description="Warnings found")
    severity: Literal["ERROR", "WARNING", "PASS"] = Field(
        ..., description="Validation severity level"
    )

    @field_validator("severity")
    @classmethod
    def severity_matches_validation(
        cls, v: str, info: ValidationInfo
    ) -> str:  # type: ignore[no-untyped-def]
        """Validate severity matches errors/warnings."""
        errors = info.data.get("errors", [])
        warnings = info.data.get("warnings", [])
        is_valid = info.data.get("is_valid", True)

        if not is_valid or errors:
            if v != "ERROR":
                raise ValueError("severity must be ERROR when validation fails")
        elif warnings:
            if v not in ("WARNING", "ERROR"):
                raise ValueError(
                    "severity must be WARNING or ERROR when warnings present"
                )
        elif v == "ERROR":
            raise ValueError(
                "severity must be PASS or WARNING when validation succeeds"
            )
        return v


class GapRange(BaseModel):
    """Represents the price range of a fair value gap.

    For bullish FVG: top = candle 1 high, bottom = candle 3 low
    For bearish FVG: top = candle 3 high, bottom = candle 1 low
    """

    top: float = Field(..., gt=0, description="Higher price level of the gap")
    bottom: float = Field(..., gt=0, description="Lower price level of the gap")

    @field_validator("top")
    @classmethod
    def top_must_be_greater_than_bottom(
        cls, v: float, info: ValidationInfo
    ) -> float:  # type: ignore[no-untyped-def]
        """Validate top price is greater than bottom price."""
        bottom = info.data.get("bottom")
        if bottom is not None and v <= bottom:
            raise ValueError("top must be greater than bottom")
        return v


class SwingPoint(BaseModel):
    """Represents a pivot high or low in price structure.

    Swing points are key levels in market structure that indicate
    potential support/resistance zones and trend direction.
    """

    timestamp: datetime = Field(..., description="Swing point timestamp")
    price: float = Field(..., gt=0, description="Swing point price level")
    swing_type: Literal["swing_high", "swing_low"] = Field(
        ..., description="Type of swing point"
    )
    bar_index: int = Field(..., ge=0, description="Position in Dollar Bar sequence")
    confirmed: bool = Field(
        default=True, description="Whether swing is confirmed (N bars on each side)"
    )


class MSSEvent(BaseModel):
    """Represents a Market Structure Shift event.

    MSS events occur when price breaks through a previous swing point
    with volume confirmation, indicating a potential trend change.
    """

    timestamp: datetime = Field(..., description="MSS detection timestamp")
    direction: Literal["bullish", "bearish"] = Field(
        ..., description="MSS direction (bullish breakout or bearish breakdown)"
    )
    breakout_price: float = Field(..., gt=0, description="Price that broke swing level")
    swing_point: SwingPoint = Field(..., description="The swing point that was broken")
    volume_ratio: float = Field(
        ..., ge=0, description="Breakout volume / avg volume(20 bars)"
    )
    bar_index: int = Field(..., ge=0, description="Bar index where MSS occurred")
    confidence: float = Field(
        default=0.0, ge=0, le=5, description="Confidence score (1-5), calculated later"
    )


class FVGEvent(BaseModel):
    """Represents a Fair Value Gap event.

    FVG events occur when price moves aggressively in one direction,
    leaving a gap that often acts as a magnet for price to revisit.
    """

    timestamp: datetime = Field(..., description="FVG detection timestamp")
    direction: Literal["bullish", "bearish"] = Field(
        ..., description="FVG direction (bullish gap up, bearish gap down)"
    )
    gap_range: GapRange = Field(..., description="Price range of the gap")
    gap_size_ticks: float = Field(
        ..., ge=0, description="Gap size in futures contract ticks"
    )
    gap_size_dollars: float = Field(
        ..., ge=0, description="Dollar value of the gap"
    )
    bar_index: int = Field(..., ge=0, description="Bar index where FVG detected")
    filled: bool = Field(default=False, description="Whether gap has been filled")
    fill_time: datetime | None = Field(
        default=None, description="Timestamp when gap was filled"
    )
    fill_bar_index: int | None = Field(
        default=None, ge=0, description="Bar index when gap was filled"
    )
    confidence: float = Field(
        default=0.0, ge=0, le=5, description="Confidence score (1-5), calculated later"
    )


class LiquiditySweepEvent(BaseModel):
    """Represents a liquidity sweep (stop hunt) event.

    Liquidity sweeps occur when price briefly trades beyond a swing point
    (stopping out traders) then reverses, indicating potential trap and reversal.
    """

    timestamp: datetime = Field(..., description="Sweep detection timestamp")
    direction: Literal["bullish", "bearish"] = Field(
        ..., description="Sweep direction (bullish sweep of lows, bearish sweep of highs)"
    )
    swing_point_price: float = Field(
        ..., gt=0, description="Price level of swing point that was swept"
    )
    sweep_depth_ticks: float = Field(
        ..., ge=0, description="How far price extended beyond swing point (in ticks)"
    )
    sweep_depth_dollars: float = Field(
        ..., ge=0, description="Sweep depth in dollar value"
    )
    bar_index: int = Field(..., ge=0, description="Bar index where sweep detected")
    confidence: float = Field(
        default=0.0, ge=0, le=5, description="Confidence score (1-5), calculated later"
    )


class SilverBulletSetup(BaseModel):
    """Represents a Silver Bullet setup (confluence of MSS, FVG, and liquidity sweep).

    Silver Bullet setups are high-probability trading opportunities that occur when
    multiple ICT patterns align in confluence within a short time window.
    """

    timestamp: datetime = Field(..., description="Setup detection timestamp")
    direction: Literal["bullish", "bearish"] = Field(
        ..., description="Setup direction (from MSS direction)"
    )
    mss_event: MSSEvent = Field(..., description="Market Structure Shift event")
    fvg_event: FVGEvent = Field(..., description="Fair Value Gap event")
    liquidity_sweep_event: LiquiditySweepEvent | None = Field(
        default=None, description="Optional liquidity sweep event (3-pattern confluence)"
    )
    entry_zone_top: float = Field(
        ..., gt=0, description="Top of entry zone (FVG gap top)"
    )
    entry_zone_bottom: float = Field(
        ..., gt=0, description="Bottom of entry zone (FVG gap bottom)"
    )
    invalidation_point: float = Field(
        ..., gt=0, description="Invalidation price level (opposite swing point)"
    )
    confluence_count: int = Field(
        ...,
        ge=2,
        le=3,
        description="Number of confluence patterns (2=MSS+FVG, 3=all three)",
    )
    priority: Literal["low", "medium", "high"] = Field(
        ..., description="Setup priority based on confluence count"
    )
    bar_index: int = Field(..., ge=0, description="Bar index where setup detected")
    confidence: float = Field(
        default=0.0, ge=0, le=5, description="Confidence score (1-5), calculated later"
    )


class TimeWindow(BaseModel):
    """Represents a time window for filtering Silver Bullet signals.

    Time windows define high-probability trading periods when signals
    should be generated. Outside these windows, signals are suppressed.
    """

    name: str = Field(..., description="Window name (e.g., 'London AM', 'NY AM')")
    start_hour: int = Field(..., ge=0, le=23, description="Start hour (0-23)")
    start_minute: int = Field(..., ge=0, le=59, description="Start minute (0-59)")
    end_hour: int = Field(..., ge=0, le=23, description="End hour (0-23)")
    end_minute: int = Field(..., ge=0, le=59, description="End minute (0-59)")
    timezone: str = Field(
        ..., description="Timezone (e.g., 'EST', 'UTC')"
    )  # Note: Could use Literal for validation
