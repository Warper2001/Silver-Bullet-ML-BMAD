"""Wolf Pack 3-Edge Strategy.

This module implements the Wolf Pack 3-Edge trading strategy, which requires
confluence of three edges:
1. Microstructure edge: Liquidity sweep with reversal
2. Behavioral edge: Trapped traders on wrong side
3. Statistical edge: Price deviation >2 SD from mean
"""

import logging
from datetime import datetime

from src.data.models import DollarBar
from src.detection.models import (
    StatisticalExtremeEvent,
    TrappedTraderEvent,
    WolfPackSignal,
    WolfPackSweepEvent,
)
from src.detection.statistical_extreme_detector import StatisticalExtremeDetector
from src.detection.trapped_trader_detector import TrappedTraderDetector
from src.detection.wolf_pack_liquidity_sweep_detector import (
    WolfPackLiquiditySweepDetector,
)

logger = logging.getLogger(__name__)


class WolfPackStrategy:
    """Wolf Pack 3-Edge Strategy.

    Generates trading signals when all three edges align with high
    confidence (0.8-1.0).

    Attributes:
        _sweep_detector: Liquidity sweep detector (microstructure edge)
        _trap_detector: Trapped trader detector (behavioral edge)
        _extreme_detector: Statistical extreme detector (statistical edge)
        _tick_size: Tick size for risk-reward calculations
    """

    DEFAULT_TICK_SIZE = 0.25  # MNQ tick size
    DEFAULT_RISK_TICKS = 20  # Risk in ticks for stop loss
    DEFAULT_MIN_CONFIDENCE = 0.8  # Minimum confidence for signal

    def __init__(
        self,
        tick_size: float = DEFAULT_TICK_SIZE,
        risk_ticks: float = DEFAULT_RISK_TICKS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        """Initialize Wolf Pack 3-Edge Strategy.

        Args:
            tick_size: Tick size for futures contract (default: 0.25 for MNQ)
            risk_ticks: Risk in ticks for stop loss
            min_confidence: Minimum confidence for signal generation
        """
        self._tick_size = tick_size
        self._risk_ticks = risk_ticks
        self._min_confidence = min_confidence

        # Initialize detectors
        self._sweep_detector = WolfPackLiquiditySweepDetector(tick_size=tick_size)
        self._trap_detector = TrappedTraderDetector()
        self._extreme_detector = StatisticalExtremeDetector()

    def process_bars(self, bars: list[DollarBar]) -> list[WolfPackSignal]:
        """Process new bars and generate signals based on 3-edge confluence.

        Args:
            bars: List of new Dollar Bars to process

        Returns:
            List of trading signals (0-1 per bar)
        """
        if not bars:
            return []

        # Detect edges
        sweep_events = self._sweep_detector.process_bars(bars)
        signals = []

        for sweep in sweep_events:
            # Check for trapped traders based on sweep
            trap_events = self._trap_detector.process_bars(bars, sweep)

            # Check for statistical extreme
            extreme_events = self._extreme_detector.process_bars(bars)

            # Check for 3-edge confluence
            for trap in trap_events:
                for extreme in extreme_events:
                    # Check if all edges agree on direction
                    signal = self._check_3_edge_confluence(
                        sweep, trap, extreme, bars[-1]
                    )
                    if signal:
                        signals.append(signal)

        return signals

    def _check_3_edge_confluence(
        self,
        sweep: WolfPackSweepEvent,
        trap: TrappedTraderEvent,
        extreme: StatisticalExtremeEvent,
        current_bar: DollarBar,
    ) -> WolfPackSignal | None:
        """Check if all three edges align for a signal.

        Args:
            sweep: Liquidity sweep event
            trap: Trapped trader event
            extreme: Statistical extreme event
            current_bar: Most recent bar for entry price

        Returns:
            WolfPackSignal if 3-edge confluence detected, None otherwise
        """
        # Determine direction from sweep
        # Bearish sweep = sweep of high = price going down = SHORT
        # Bullish sweep = sweep of low = price going up = LONG
        if sweep.sweep_direction == "bearish":
            expected_direction = "short"
            # Check: trapped longs (trapped on wrong side)
            # Check: statistical extreme high (price too high, should come down)
            if trap.trap_type != "trapped_long":
                return None
            if extreme.direction != "high":
                return None
        else:  # bullish sweep
            expected_direction = "long"
            # Check: trapped shorts (trapped on wrong side)
            # Check: statistical extreme low (price too low, should go up)
            if trap.trap_type != "trapped_short":
                return None
            if extreme.direction != "low":
                return None

        # Calculate confidence based on edge strength
        # Base confidence: 0.8
        # Add up to 0.1 for sweep extent (>10 ticks)
        # Add up to 0.1 for trap severity (>2.0)
        confidence = 0.8

        if sweep.sweep_extent_ticks > 10:
            confidence += 0.05
        if trap.severity > 2.0:
            confidence += 0.05

        # Cap at 1.0
        confidence = min(confidence, 1.0)

        # Check if confidence meets threshold
        if confidence < self._min_confidence:
            return None

        # Calculate entry, SL, TP
        entry_price = current_bar.close
        risk_amount = self._risk_ticks * self._tick_size

        if expected_direction == "long":
            stop_loss = entry_price - risk_amount
            take_profit = entry_price + (risk_amount * 2)  # 2:1 ratio
        else:  # short
            stop_loss = entry_price + risk_amount
            take_profit = entry_price - (risk_amount * 2)  # 2:1 ratio

        # Create signal
        signal = WolfPackSignal(
            strategy_name="Wolf Pack 3-Edge",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=expected_direction,
            confidence=confidence,
            timestamp=current_bar.timestamp,
            contributing_factors={
                "sweep": {
                    "swing_level": sweep.swing_level,
                    "sweep_extreme": sweep.sweep_extreme,
                    "sweep_extent_ticks": sweep.sweep_extent_ticks,
                    "sweep_direction": sweep.sweep_direction,
                },
                "trapped_trader": {
                    "trap_type": trap.trap_type,
                    "severity": trap.severity,
                    "entry_price": trap.entry_price,
                    "rejection_price": trap.rejection_price,
                },
                "statistical_extreme": {
                    "z_score": extreme.z_score,
                    "direction": extreme.direction,
                    "magnitude": extreme.magnitude,
                    "rolling_mean": extreme.rolling_mean,
                    "rolling_std": extreme.rolling_std,
                },
            },
            expected_win_rate=0.775,
        )

        logger.info(
            f"Wolf Pack 3-Edge signal generated: {expected_direction.upper()}, "
            f"entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}, "
            f"confidence={confidence:.2f}"
        )

        return signal

    def reset(self) -> None:
        """Reset all detectors."""
        self._sweep_detector.reset()
        self._trap_detector.reset()
        self._extreme_detector.reset()
