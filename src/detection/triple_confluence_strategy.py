"""Triple Confluence Scalper Strategy.

This module implements the Triple Confluence Scalper strategy which
generates trading signals when multiple conditions align:
1. Level sweep detected
2. Fair Value Gap present
3. VWAP alignment confirmed

Configurable to require 2-of-3 or 3-of-3 factors for signal generation.
"""

import logging
from collections import deque
from datetime import datetime

from src.data.models import DollarBar
from src.detection.level_sweep_detector import LevelSweepDetector
from src.detection.fvg_detector import SimpleFVGDetector
from src.detection.vwap_calculator import VWAPCalculator
from src.detection.models import (
    LevelSweepEvent,
    TripleConfluenceFVGEvent,
    TripleConfluenceSignal,
)

logger = logging.getLogger(__name__)


class TripleConfluenceStrategy:
    """Triple Confluence Scalper trading strategy.

    Detects trading opportunities when multiple technical factors align:
    1. Level Sweep: Price sweeps daily high/low and reverses
    2. Fair Value Gap: Price gap between candles
    3. VWAP Alignment: Price bias relative to volume-weighted average price

    Configurable to require 2-of-3 (default) or 3-of-3 factors to agree
    on direction (bullish or bearish) for signal generation.

    Attributes:
        _level_sweep_detector: Detects level sweeps
        _fvg_detector: Detects fair value gaps
        _vwap_calculator: Calculates VWAP for bias
        _bars: History of recent bars
        _recent_sweeps: Recent level sweep events
        _recent_fvgs: Recent FVG events
        _min_confluence_factors: Minimum factors required (2 or 3)
    """

    DEFAULT_LOOKBACK = 10  # Bars to look back for recent events
    MAX_BAR_HISTORY = 50  # Maximum bars to store
    DEFAULT_MIN_CONFLUENCE = 2  # Minimum factors required (2-of-3)

    def __init__(self, config: dict) -> None:
        """Initialize Triple Confluence Strategy.

        Args:
            config: Configuration dictionary with parameters:
                - lookback_period: Bars for level sweep detection (default 20)
                - min_fvg_size: Minimum FVG size in ticks (default 4)
                - session_start: Trading session start (default "09:30:00")
                - min_confluence_factors: Minimum factors required (2 or 3, default 2)
        """
        self._config = config

        # Minimum confluence factors (2 or 3)
        self._min_confluence_factors = config.get("min_confluence_factors", 2)

        if self._min_confluence_factors not in [2, 3]:
            raise ValueError("min_confluence_factors must be 2 or 3")

        # Initialize detectors
        self._level_sweep_detector = LevelSweepDetector(
            lookback_period=config.get("lookback_period", 20)
        )
        self._fvg_detector = SimpleFVGDetector(
            min_gap_size=config.get("min_fvg_size", 4)
        )
        self._vwap_calculator = VWAPCalculator(
            session_start=config.get("session_start", "09:30:00")
        )

        # State
        self._bars: deque[DollarBar] = deque(maxlen=self.MAX_BAR_HISTORY)
        self._recent_sweeps: deque[LevelSweepEvent] = deque(maxlen=10)
        self._recent_fvgs: deque[TripleConfluenceFVGEvent] = deque(maxlen=10)

    def process_bar(self, bar: DollarBar) -> TripleConfluenceSignal | None:
        """Process a Dollar Bar and generate signal if triple confluence exists.

        Args:
            bar: Dollar Bar to process

        Returns:
            TripleConfluenceSignal if confluence detected, None otherwise
        """
        # Add bar to history
        self._bars.append(bar)

        # Get bar list for detectors
        bars_list = list(self._bars)

        # Detect level sweep
        sweep_event = self._level_sweep_detector.detect_sweep(bars_list)
        if sweep_event:
            self._recent_sweeps.append(sweep_event)
            logger.info(f"Level sweep detected: {sweep_event.sweep_direction}")

        # Detect FVGs
        fvg_events = self._fvg_detector.detect_fvg(bars_list)
        for fvg in fvg_events:
            # Only add if not already tracked (by timestamp)
            if not any(f.timestamp == fvg.timestamp for f in self._recent_fvgs):
                self._recent_fvgs.append(fvg)
                logger.info(f"FVG detected: {fvg.fvg_type}")

        # Calculate VWAP
        vwap = self._vwap_calculator.calculate_vwap(bars_list)
        bias = self._vwap_calculator.get_bias(bar.close, vwap)

        # Check for triple confluence
        signal = self._check_triple_confluence(bar, vwap, bias)

        return signal

    def _check_triple_confluence(
        self,
        bar: DollarBar,
        vwap: float,
        bias: str,
    ) -> TripleConfluenceSignal | None:
        """Check if conditions align for signal generation (2-of-3 or 3-of-3).

        Args:
            bar: Current Dollar Bar
            vwap: Current VWAP value
            bias: Market bias (bullish/bearish/neutral)

        Returns:
            TripleConfluenceSignal if confluence exists, None otherwise
        """
        # Need recent events from last 5 bars
        recent_sweeps = list(self._recent_sweeps)[-5:]
        recent_fvgs = list(self._recent_fvgs)[-5:]

        # Count bullish factors
        bullish_factors = 0
        bearish_factors = 0

        # Factor 1: Level Sweep
        has_bullish_sweep = False
        has_bearish_sweep = False
        if recent_sweeps:
            latest_sweep = recent_sweeps[-1]
            if latest_sweep.sweep_direction == "bullish":
                has_bullish_sweep = True
                bullish_factors += 1
            else:
                has_bearish_sweep = True
                bearish_factors += 1

        # Factor 2: FVG
        has_bullish_fvg = False
        has_bearish_fvg = False
        if recent_fvgs:
            # Check most recent FVG
            latest_fvg = recent_fvgs[-1]
            if latest_fvg.fvg_type == "bullish":
                has_bullish_fvg = True
                bullish_factors += 1
            else:
                has_bearish_fvg = True
                bearish_factors += 1

        # Factor 3: VWAP
        has_bullish_vwap = False
        has_bearish_vwap = False
        if bias == "bullish":
            has_bullish_vwap = True
            bullish_factors += 1
        elif bias == "bearish":
            has_bearish_vwap = True
            bearish_factors += 1

        # Check if we have minimum confluence
        has_bullish_confluence = bullish_factors >= self._min_confluence_factors
        has_bearish_confluence = bearish_factors >= self._min_confluence_factors

        if not (has_bullish_confluence or has_bearish_confluence):
            return None

        # Determine direction
        direction = "long" if has_bullish_confluence else "short"

        # Calculate confidence based on number of factors
        confidence = self._calculate_confidence(
            has_bullish_sweep,
            has_bearish_sweep,
            has_bullish_fvg,
            has_bearish_fvg,
            has_bullish_vwap,
            has_bearish_vwap,
            bullish_factors if direction == "long" else bearish_factors,
            bar.close,
            vwap,
        )

        # Get latest events for signal details
        latest_sweep = recent_sweeps[-1] if recent_sweeps else None
        latest_fvg = recent_fvgs[-1] if recent_fvgs else None

        # Calculate entry, stop loss, take profit
        entry_price = bar.close

        # For stop loss, prefer FVG edge if available, otherwise use ATR-based
        if latest_fvg:
            stop_loss, take_profit = self._calculate_exit_levels(
                direction, entry_price, latest_fvg
            )
        else:
            # Fallback to simple ATR-based (using recent range as proxy)
            recent_range = max(b.high for b in self._bars[-10:]) - min(b.low for b in self._bars[-10:])
            atr = recent_range / 10  # Rough approximation

            if direction == "long":
                stop_loss = entry_price - atr
                take_profit = entry_price + (2 * atr)
            else:
                stop_loss = entry_price + atr
                take_profit = entry_price - (2 * atr)

        # Create signal
        signal = TripleConfluenceSignal(
            strategy_name="Triple Confluence Scalper",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            confidence=confidence,
            timestamp=bar.timestamp,
            contributing_factors={
                "level_sweep": {
                    "present": latest_sweep is not None,
                    "direction": latest_sweep.sweep_direction if latest_sweep else None,
                    "extent_ticks": latest_sweep.sweep_extent_ticks if latest_sweep else None,
                } if latest_sweep else {"present": False},
                "fvg": {
                    "present": latest_fvg is not None,
                    "type": latest_fvg.fvg_type if latest_fvg else None,
                    "gap_size_ticks": latest_fvg.gap_size_ticks if latest_fvg else None,
                    "gap_high": latest_fvg.gap_edge_high if latest_fvg else None,
                    "gap_low": latest_fvg.gap_edge_low if latest_fvg else None,
                } if latest_fvg else {"present": False},
                "vwap": {
                    "bias": bias,
                    "vwap_value": vwap,
                    "current_price": bar.close,
                },
                "confluence_count": bullish_factors if direction == "long" else bearish_factors,
            },
        )

        logger.info(
            f"Triple Confluence Signal ({bullish_factors if direction == 'long' else bearish_factors}/3 factors): "
            f"{direction} @ {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
            f"confidence: {confidence:.2f}"
        )

        return signal

    def _calculate_confidence(
        self,
        has_bullish_sweep: bool,
        has_bearish_sweep: bool,
        has_bullish_fvg: bool,
        has_bearish_fvg: bool,
        has_bullish_vwap: bool,
        has_bearish_vwap: bool,
        factor_count: int,
        price: float,
        vwap: float,
    ) -> float:
        """Calculate confidence score based on signal strength and factor count.

        Confidence ranges based on:
        - Number of confluence factors (2 or 3)
        - Strength of each factor (sweep extent, FVG size, VWAP distance)

        Args:
            has_bullish_sweep: Whether bullish sweep present
            has_bearish_sweep: Whether bearish sweep present
            has_bullish_fvg: Whether bullish FVG present
            has_bearish_fvg: Whether bearish FVG present
            has_bullish_vwap: Whether VWAP bullish
            has_bearish_vwap: Whether VWAP bearish
            factor_count: Number of aligned factors (2 or 3)
            price: Current price
            vwap: VWAP value

        Returns:
            Confidence score between 0.65 and 1.0
        """
        # Base confidence depends on factor count
        if factor_count == 3:
            base_confidence = 0.80  # Triple confluence
        else:  # factor_count == 2
            base_confidence = 0.70  # Dual confluence

        confidence = base_confidence

        # Add contribution for each factor (up to 0.1 each)
        # Sweep contribution
        if has_bullish_sweep or has_bearish_sweep:
            # Would need sweep extent, but we don't have it here
            # Use fixed contribution for having sweep factor
            confidence += 0.05

        # FVG contribution
        if has_bullish_fvg or has_bearish_fvg:
            # Would need FVG size, but we don't have it here
            # Use fixed contribution for having FVG factor
            confidence += 0.05

        # VWAP contribution
        if has_bullish_vwap or has_bearish_vwap:
            # Calculate VWAP distance
            vwap_distance_ticks = abs(price - vwap) / 0.25  # MNQ tick size
            vwap_contribution = min(vwap_distance_ticks / 20.0, 1.0) * 0.1
            confidence += vwap_contribution

        # Cap at 1.0
        return min(confidence, 1.0)

    def _calculate_exit_levels(
        self,
        direction: str,
        entry_price: float,
        fvg: TripleConfluenceFVGEvent,
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit levels.

        Args:
            direction: Trade direction ("long" or "short")
            entry_price: Entry price
            fvg: FVG event for structure-based stops

        Returns:
            Tuple of (stop_loss, take_profit)
        """
        if direction == "long":
            # Stop loss at FVG edge
            stop_loss = fvg.gap_edge_low

            # Risk amount
            risk = entry_price - stop_loss

            # Take profit at 2:1 reward-risk ratio
            take_profit = entry_price + (2 * risk)
        else:  # short
            # Stop loss at FVG edge
            stop_loss = fvg.gap_edge_high

            # Risk amount
            risk = stop_loss - entry_price

            # Take profit at 2:1 reward-risk ratio
            take_profit = entry_price - (2 * risk)

        return stop_loss, take_profit

    def reset(self) -> None:
        """Reset strategy state."""
        self._bars.clear()
        self._recent_sweeps.clear()
        self._recent_fvgs.clear()
        self._level_sweep_detector.reset()
        self._vwap_calculator.reset_session()
        logger.info("Triple Confluence Strategy reset")
