"""Pydantic models for Triple Confluence Scalper and Wolf Pack strategies."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


# Ensemble Signal Model


class EnsembleSignal(BaseModel):
    """Normalized signal format for ensemble processing.

    This model provides a unified interface for signals from all 5 strategies:
    - Triple Confluence Scalper
    - Wolf Pack 3-Edge
    - Adaptive EMA Momentum
    - VWAP Bounce
    - Opening Range Breakout

    The normalization preserves all critical information while providing
    a consistent format for ensemble weighting and decision-making.
    """

    strategy_name: str = Field(..., description="Name of the strategy generating the signal")
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    direction: Literal["long", "short"] = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profit: float = Field(..., gt=0, description="Take profit price level")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1 scale)")
    bar_timestamp: datetime = Field(..., description="Which bar triggered the signal")
    metadata: dict = Field(default_factory=dict, description="Strategy-specific data for transparency")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_valid_range(cls, v: float) -> float:
        """Validate confidence is in valid range (0-1)."""
        if v < 0 or v > 1:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @field_validator("direction")
    @classmethod
    def direction_must_be_valid(cls, v: str) -> str:
        """Validate direction is either 'long' or 'short'."""
        if v not in ["long", "short"]:
            raise ValueError("direction must be 'long' or 'short'")
        return v

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_respect_direction(cls, v: float, info) -> float:
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
    def take_profit_must_respect_2to1_ratio(cls, v: float, info) -> float:
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

    def risk_reward_ratio(self) -> float:
        """Calculate the risk-reward ratio.

        Returns:
            Ratio of reward to risk (e.g., 2.0 for 2:1)
        """
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    def is_valid(self) -> bool:
        """Validate signal integrity.

        Returns:
            True if signal passes all validation checks
        """
        # Check confidence range
        if not (0 <= self.confidence <= 1):
            return False

        # Check stop loss direction
        if self.direction == "long" and self.stop_loss >= self.entry_price:
            return False
        if self.direction == "short" and self.stop_loss <= self.entry_price:
            return False

        # Check take profit ratio (approximately 2:1)
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        if risk > 0:
            ratio = reward / risk
            if ratio < 1.9:  # Allow small tolerance
                return False

        return True


class EnsembleTradeSignal(BaseModel):
    """Ensemble trade signal with weighted confidence scoring.

    This model represents a composite trading signal generated by the
    ensemble system when multiple strategies agree on direction and
    meet the confidence threshold.

    Attributes:
        strategy_name: Name of ensemble strategy (fixed: "Ensemble-Weighted Confidence")
        timestamp: Signal generation timestamp
        direction: Trade direction (long or short)
        entry_price: Weighted average entry price from contributing strategies
        stop_loss: Recommended stop loss level
        take_profit: Recommended take profit level (2:1 reward-risk ratio)
        composite_confidence: Weighted average confidence (0-1 scale)
        contributing_strategies: List of strategy names that contributed
        strategy_confidences: Individual strategy confidence scores (for transparency)
        strategy_weights: Weights applied to each strategy (for audit trail)
        bar_timestamp: Which bar triggered the signal
    """

    strategy_name: str = Field(default="Ensemble-Weighted Confidence", description="Ensemble strategy name")
    timestamp: datetime = Field(..., description="Signal generation timestamp")
    direction: Literal["long", "short"] = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profit: float = Field(..., gt=0, description="Take profit price level")
    composite_confidence: float = Field(..., ge=0, le=1, description="Composite confidence score (0-1 scale)")
    contributing_strategies: list[str] = Field(..., min_length=1, description="Strategies that contributed to signal")
    strategy_confidences: dict[str, float] = Field(..., description="Individual strategy confidences")
    strategy_weights: dict[str, float] = Field(..., description="Weights applied to each strategy")
    bar_timestamp: datetime = Field(..., description="Which bar triggered the signal")

    @model_validator(mode="after")
    def validate_contributing_strategies(self) -> "EnsembleTradeSignal":
        """Validate that all contributing strategies have confidence values."""
        for strategy in self.contributing_strategies:
            if strategy not in self.strategy_confidences:
                raise ValueError(
                    f"Contributing strategy '{strategy}' must have a confidence value in strategy_confidences"
                )
        return self

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_respect_direction(cls, v: float, info) -> float:
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
    def take_profit_must_respect_direction(cls, v: float, info) -> float:
        """Validate take profit position based on direction."""
        direction = info.data.get("direction")
        entry = info.data.get("entry_price")

        if entry is not None and direction is not None:
            if direction == "long" and v <= entry:
                raise ValueError("take_profit must be above entry_price for long trades")
            if direction == "short" and v >= entry:
                raise ValueError("take_profit must be below entry_price for short trades")
        return v

    def contributing_count(self) -> int:
        """Return the number of contributing strategies.

        Returns:
            Number of strategies that contributed to this signal
        """
        return len(self.contributing_strategies)

    def is_unanimous(self) -> bool:
        """Check if all 5 strategies contributed to this signal.

        Returns:
            True if all 5 strategies signaled (unanimous)
        """
        return len(self.contributing_strategies) == 5

    def get_weighted_entry(self) -> float:
        """Return the weighted entry price.

        For this model, the entry_price is already calculated as the weighted
        average from all contributing strategies. This method returns it for
        consistency and potential future enhancements.

        Returns:
            Weighted entry price
        """
        return self.entry_price


# Performance Tracking Models for Dynamic Weight Optimization


class StrategyPerformance(BaseModel):
    """Performance metrics for a single strategy.

    Tracks performance metrics over a rolling window (typically 4 weeks)
    to inform dynamic weight optimization.

    Attributes:
        strategy_name: Name of the strategy
        window_start: Start of performance tracking window
        window_end: End of performance tracking window
        total_trades: Total number of trades in window
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: Win rate (0-1 scale)
        gross_profit: Sum of all winning trade P&L
        gross_loss: Sum of all losing trade P&L (absolute value)
        profit_factor: Gross profit divided by gross loss
        performance_score: Win rate × profit factor
        data_quality: Data sufficiency flag
    """

    strategy_name: str = Field(..., description="Name of the strategy")
    window_start: datetime = Field(..., description="Start of performance window")
    window_end: datetime = Field(..., description="End of performance window")
    total_trades: int = Field(..., ge=0, description="Total trades in window")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    gross_profit: float = Field(..., ge=0, description="Sum of winning P&L")
    gross_loss: float = Field(..., ge=0, description="Sum of losing P&L (absolute)")
    profit_factor: float = Field(..., ge=0, description="Gross profit / gross loss")
    performance_score: float = Field(..., ge=0, description="Win rate × profit factor")
    data_quality: Literal["sufficient", "insufficient_4weeks", "insufficient_8weeks"] = Field(
        ..., description="Data sufficiency flag"
    )

    def calculate_performance_score(self) -> float:
        """Calculate performance score.

        Returns:
            Performance score (win_rate × profit_factor)
        """
        return self.win_rate * self.profit_factor

    def is_data_sufficient(self) -> bool:
        """Check if data is sufficient for weight optimization.

        Returns:
            True if data quality is "sufficient"
        """
        return self.data_quality == "sufficient"


class WeightUpdate(BaseModel):
    """Record of a weight update event.

    Captures all information about a weight rebalancing event for
    audit trail and analysis.

    Attributes:
        timestamp: When the update occurred
        previous_weights: Weights before update (by strategy)
        new_weights: Weights after update (by strategy)
        performance_scores: Performance scores used for calculation
        constraint_adjustments: Which strategies hit floor/ceiling
        rebalancing_reason: Why rebalancing occurred
    """

    timestamp: datetime = Field(..., description="Update timestamp")
    previous_weights: dict[str, float] = Field(..., description="Weights before update")
    new_weights: dict[str, float] = Field(..., description="Weights after update")
    performance_scores: dict[str, float] = Field(..., description="Performance scores used")
    constraint_adjustments: dict[str, str] = Field(
        default_factory=dict, description="Constraint adjustments (floor/ceiling hits)"
    )
    rebalancing_reason: str = Field(..., description="Reason for rebalancing")

    @field_validator("previous_weights", "new_weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float], info) -> dict[str, float]:
        """Validate that weights sum to approximately 1.0."""
        total = sum(v.values())
        tolerance = 0.0001
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Weights must sum to 1.0 (got {total:.4f})")
        return v

    @field_validator("previous_weights", "new_weights")
    @classmethod
    def weights_within_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that all weights are between 0 and 1."""
        for strategy, weight in v.items():
            if weight < 0 or weight > 1:
                raise ValueError(f"Weight for {strategy} must be between 0 and 1 (got {weight})")
        return v

    def get_weight_change(self, strategy: str) -> float:
        """Get weight change for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            Weight change (new - previous)
        """
        if strategy not in self.previous_weights or strategy not in self.new_weights:
            return 0.0
        return self.new_weights[strategy] - self.previous_weights[strategy]


class CompletedTrade(BaseModel):
    """Record of a completed trade for performance tracking.

    Captures essential information about completed trades for
    performance analysis and weight optimization.

    Attributes:
        trade_id: Unique trade identifier
        strategy_name: Strategy that generated the trade
        direction: Trade direction (long or short)
        entry_price: Entry price
        exit_price: Exit price
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        pnl: Profit/loss in USD
        exit_reason: Reason for exit (TP, SL, time_stop, etc.)
        bars_held: Number of bars held
    """

    trade_id: str = Field(..., description="Unique trade identifier")
    strategy_name: str = Field(..., description="Strategy that generated the trade")
    direction: Literal["long", "short"] = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Entry price")
    exit_price: float = Field(..., gt=0, description="Exit price")
    entry_time: datetime = Field(..., description="Entry timestamp")
    exit_time: datetime = Field(..., description="Exit timestamp")
    pnl: float = Field(..., description="Profit/loss in USD")
    exit_reason: Literal["take_profit", "stop_loss", "time_stop", "hybrid_partial", "hybrid_trail"] = Field(
        ..., description="Reason for exit"
    )
    bars_held: int = Field(..., ge=0, description="Number of bars held")

    def is_winner(self) -> bool:
        """Check if trade was a winner.

        Returns:
            True if pnl > 0
        """
        return self.pnl > 0

    def get_hold_time_minutes(self) -> float:
        """Calculate hold time in minutes.

        Returns:
            Hold time in minutes
        """
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 60.0

