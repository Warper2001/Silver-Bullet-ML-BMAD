"""Pydantic models for Triple Confluence Scalper and Wolf Pack strategies."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LevelSweepEvent(BaseModel):
    """Represents a level sweep event.

    A level sweep occurs when price trades beyond an established
    daily high or low and then reverses back through that level.
    """

    timestamp: datetime = Field(..., description="Sweep detection timestamp")
    level_type: Literal["daily_high", "daily_low"] = Field(
        ..., description="Type of level that was swept"
    )
    level_price: float = Field(..., gt=0, description="Price level that was swept")
    sweep_extreme: float = Field(..., gt=0, description="Extreme price reached during sweep")
    reversal_price: float = Field(..., gt=0, description="Price where reversal occurred")
    sweep_direction: Literal["bullish", "bearish"] = Field(
        ..., description="Sweep direction (bullish sweep of high, bearish sweep of low)"
    )
    sweep_extent_ticks: float = Field(
        ..., ge=0, description="How far price extended beyond level (in ticks)"
    )
    volume_at_sweep: float = Field(
        ..., ge=0, description="Volume at the sweep extreme"
    )


class TripleConfluenceFVGEvent(BaseModel):
    """Represents a Fair Value Gap event for Triple Confluence strategy.

    This is a simplified version of the main FVGEvent specifically
    for the Triple Confluence Scalper strategy.
    """

    timestamp: datetime = Field(..., description="FVG detection timestamp")
    fvg_type: Literal["bullish", "bearish"] = Field(
        ..., description="FVG direction (bullish gap up, bearish gap down)"
    )
    gap_size_ticks: float = Field(..., ge=0, description="Gap size in ticks")
    gap_edge_high: float = Field(..., gt=0, description="Upper edge of the gap")
    gap_edge_low: float = Field(..., gt=0, description="Lower edge of the gap")
    third_candle_close: float = Field(..., gt=0, description="Close price of third candle")

    @field_validator("gap_edge_high")
    @classmethod
    def gap_edge_high_must_be_greater_than_low(
        cls, v: float, info
    ):
        """Validate gap edge high is greater than gap edge low."""
        low = info.data.get("gap_edge_low")
        if low is not None and v <= low:
            raise ValueError("gap_edge_high must be greater than gap_edge_low")
        return v


class TripleConfluenceSignal(BaseModel):
    """Represents a Triple Confluence Scalper trading signal.

    Generated when all three conditions align:
    1. Level sweep detected
    2. Fair Value Gap present
    3. VWAP alignment confirmed
    """

    strategy_name: str = Field(
        default="Triple Confluence Scalper",
        description="Name of the strategy"
    )
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profit: float = Field(..., gt=0, description="Take profit price level")
    direction: Literal["long", "short"] = Field(
        ..., description="Trade direction"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score (0-1 scale)"
    )
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    contributing_factors: dict = Field(
        ..., description="Details of sweep, FVG, and VWAP factors"
    )
    expected_win_rate: float = Field(
        default=0.775, description="Expected win rate (75-80% target)"
    )

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_valid_range(
        cls, v: float
    ):
        """Validate confidence is in valid range (0.65-1.0).

        Allows 0.70-1.0 for dual confluence (2-of-3 factors)
        Allows 0.80-1.0 for triple confluence (3-of-3 factors)
        """
        if v < 0.65 or v > 1.0:
            raise ValueError("confidence must be between 0.65 and 1.0")
        return v

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_respect_direction(
        cls, v: float, info
    ):
        """Validate stop loss position based on direction."""
        direction = info.data.get("direction")
        entry = info.data.get("entry_price")

        if entry is not None and direction is not None:
            if direction == "long" and v >= entry:
                raise ValueError("stop_loss must be below entry_price for long trades")
            if direction == "short" and v <= entry:
                raise ValueError("stop_loss must be above entry_price for short trades")
        return v

    @field_validator("take_profit")
    @classmethod
    def take_profit_must_respect_2to1_ratio(
        cls, v: float, info
    ):
        """Validate take profit respects 2:1 reward-risk ratio."""
        entry = info.data.get("entry_price")
        stop_loss = info.data.get("stop_loss")

        if entry is not None and stop_loss is not None:
            risk = abs(entry - stop_loss)
            reward = abs(v - entry)

            # Allow small rounding errors but generally require 2:1
            if risk > 0:
                ratio = reward / risk
                if ratio < 1.9:  # Allow small tolerance
                    raise ValueError(
                        f"take_profit must respect 2:1 ratio (current: {ratio:.2f}:1)"
                    )
        return v


# Wolf Pack Strategy Models


class SwingPoint(BaseModel):
    """Represents a swing point (high or low) for Wolf Pack strategy."""

    timestamp: datetime = Field(..., description="Swing point timestamp")
    price: float = Field(..., gt=0, description="Swing point price")
    swing_type: Literal["high", "low"] = Field(
        ..., description="Type of swing point"
    )
    bar_index: int = Field(..., ge=0, description="Bar index in series")


class WolfPackSweepEvent(BaseModel):
    """Represents a liquidity sweep event for Wolf Pack strategy.

    Enhanced version of LevelSweepEvent with additional fields.
    """

    timestamp: datetime = Field(..., description="Sweep detection timestamp")
    swing_level: float = Field(..., gt=0, description="Swing level that was swept")
    swing_type: Literal["high", "low"] = Field(
        ..., description="Type of swing level"
    )
    sweep_extreme: float = Field(..., gt=0, description="Extreme price reached during sweep")
    reversal_price: float = Field(..., gt=0, description="Price where reversal occurred")
    sweep_direction: Literal["bullish", "bearish"] = Field(
        ..., description="Sweep direction (bullish = sweep of high, bearish = sweep of low)"
    )
    sweep_extent_ticks: float = Field(
        ..., ge=0, description="How far price extended beyond level (in ticks)"
    )
    reversal_volume: float = Field(
        ..., ge=0, description="Volume at the reversal confirmation"
    )


class TrappedTraderEvent(BaseModel):
    """Represents a trapped trader event (behavioral edge)."""

    timestamp: datetime = Field(..., description="Trap detection timestamp")
    trap_type: Literal["trapped_long", "trapped_short"] = Field(
        ..., description="Type of trap (who is trapped)"
    )
    severity: float = Field(
        ..., ge=1.0, description="Trap severity (volume ratio: trap_volume / avg_volume)"
    )
    entry_price: float = Field(..., gt=0, description="Price where traders likely entered")
    rejection_price: float = Field(..., gt=0, description="Price where rejection occurred")
    volume_at_trap: float = Field(
        ..., ge=0, description="Volume at the trap point"
    )


class StatisticalExtremeEvent(BaseModel):
    """Represents a statistical extreme event (statistical edge)."""

    timestamp: datetime = Field(..., description="Extreme detection timestamp")
    z_score: float = Field(..., description="Z-score (standard deviations from mean)")
    direction: Literal["high", "low"] = Field(
        ..., description="Direction of extreme (price is too high or too low)"
    )
    magnitude: float = Field(
        ..., ge=0, description="Absolute magnitude (|z-score|)"
    )
    rolling_mean: float = Field(..., gt=0, description="Rolling mean at detection")
    rolling_std: float = Field(..., ge=0, description="Rolling std at detection")
    current_price: float = Field(..., gt=0, description="Current price")


class WolfPackSignal(BaseModel):
    """Represents a Wolf Pack 3-Edge trading signal.

    Generated when all three edges align:
    1. Microstructure edge: Liquidity sweep with reversal
    2. Behavioral edge: Trapped traders on wrong side
    3. Statistical edge: Price deviation >2 SD from mean
    """

    strategy_name: str = Field(
        default="Wolf Pack 3-Edge",
        description="Name of the strategy"
    )
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profit: float = Field(..., gt=0, description="Take profit price level")
    direction: Literal["long", "short"] = Field(
        ..., description="Trade direction"
    )
    confidence: float = Field(
        ..., ge=0.8, le=1.0, description="Confidence score (0.8-1.0 for 3-edge confluence)"
    )
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    contributing_factors: dict = Field(
        ..., description="Details of sweep, trapped_trader, and statistical_extreme edges"
    )
    expected_win_rate: float = Field(
        default=0.775, description="Expected win rate (75-80% target)"
    )

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_respect_direction(
        cls, v: float, info
    ):
        """Validate stop loss position based on direction."""
        direction = info.data.get("direction")
        entry = info.data.get("entry_price")

        if entry is not None and direction is not None:
            if direction == "long" and v >= entry:
                raise ValueError("stop_loss must be below entry_price for long trades")
            if direction == "short" and v <= entry:
                raise ValueError("stop_loss must be above entry_price for short trades")
        return v

    @field_validator("take_profit")
    @classmethod
    def take_profit_must_respect_2to1_ratio(
        cls, v: float, info
    ):
        """Validate take profit respects 2:1 reward-risk ratio."""
        entry = info.data.get("entry_price")
        stop_loss = info.data.get("stop_loss")

        if entry is not None and stop_loss is not None:
            risk = abs(entry - stop_loss)
            reward = abs(v - entry)

            # Allow small rounding errors but generally require 2:1
            if risk > 0:
                ratio = reward / risk
                if ratio < 1.9:  # Allow small tolerance
                    raise ValueError(
                        f"take_profit must respect 2:1 ratio (current: {ratio:.2f}:1)"
                    )
        return v


class MomentumSignal(BaseModel):
    """Signal generated by Adaptive EMA Momentum strategy.

    Generated when all three indicators align:
    1. Triple EMA: 9 EMA > 55 EMA > 200 EMA (or reverse for SHORT)
    2. MACD: Positive and increasing (or negative and decreasing)
    3. RSI: In mid-band (40-60) and rising (or falling)

    Attributes:
        timestamp: Signal generation time
        direction: Trade direction ('LONG' or 'SHORT')
        entry_price: Recommended entry price
        stop_loss: Stop loss price (1.0× ATR from entry)
        take_profit: Take profit price (2:1 risk-reward)
        confidence: Signal confidence (0-100)
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        macd_line: MACD line value
        macd_signal: MACD signal line value
        macd_histogram: MACD histogram value
        rsi_value: RSI value
        rsi_in_mid_band: Whether RSI is in mid-band
    """

    timestamp: datetime = Field(..., description="Signal generation time")
    direction: Literal["LONG", "SHORT"] = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Recommended entry price")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    take_profit: float = Field(..., gt=0, description="Take profit price")
    confidence: float = Field(..., ge=0, le=100, description="Signal confidence (0-100)")

    # Indicator values
    ema_fast: float | None = Field(None, description="Fast EMA value")
    ema_medium: float | None = Field(None, description="Medium EMA value")
    ema_slow: float | None = Field(None, description="Slow EMA value")
    macd_line: float | None = Field(None, description="MACD line value")
    macd_signal: float | None = Field(None, description="MACD signal line value")
    macd_histogram: float | None = Field(None, description="MACD histogram value")
    rsi_value: float | None = Field(None, description="RSI value")
    rsi_in_mid_band: bool = Field(False, description="Whether RSI is in mid-band")

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
