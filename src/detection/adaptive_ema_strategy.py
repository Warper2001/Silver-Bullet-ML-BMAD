"""Adaptive EMA Momentum strategy for MNQ futures trading.

This strategy combines three indicators:
1. Triple EMA (9, 55, 200 periods) for trend direction
2. MACD histogram for momentum strength and direction
3. RSI with mid-band emphasis (40-60 range) for entry timing

Target: 80.79% win rate, 5-15 trades/day, 2-5 minute holding period
"""

import logging
from collections import deque
from datetime import datetime

from src.data.models import DollarBar
from src.detection.ema_calculator import EMACalculator
from src.detection.macd_calculator import MACDCalculator
from src.detection.rsi_calculator import RSICalculator
from src.detection.models import MomentumSignal

logger = logging.getLogger(__name__)


class AdaptiveEMAStrategy:
    """Adaptive EMA Momentum trading strategy.

    Entry conditions (RELAXED for better signal generation):
    - LONG: 9 EMA > 55 EMA > 100 EMA, MACD positive, RSI 30-70
    - SHORT: 9 EMA < 55 EMA < 100 EMA, MACD negative, RSI 30-70

    Removed restrictive momentum requirements:
    - No longer require MACD histogram to be increasing
    - No longer require RSI to be rising/falling
    """

    DEFAULT_ATR_PERIOD = 14
    MAX_HISTORY = 250  # Keep enough data for 200-period EMA

    def __init__(self) -> None:
        """Initialize Adaptive EMA Momentum strategy."""
        self.ema_calculator = EMACalculator()
        self.macd_calculator = MACDCalculator()
        self.rsi_calculator = RSICalculator()

        # ATR calculation for stop loss
        self._atr_period = self.DEFAULT_ATR_PERIOD
        self._atr: float | None = None
        self._true_ranges: deque[float] = deque(maxlen=self._atr_period)

    def process_bars(self, bars: list[DollarBar]) -> list[MomentumSignal]:
        """Process new dollar bars and generate signals.

        Args:
            bars: List of new Dollar Bars

        Returns:
            List of MomentumSignal objects (empty if no signal)
        """
        if not bars:
            return []

        signals = []

        for bar in bars:
            # Update all indicators
            self.ema_calculator.calculate_emas([bar])
            self.macd_calculator.calculate_macd([bar])
            self.rsi_calculator.calculate_rsi([bar])
            self._update_atr(bar)

            # Check for signal conditions
            signal = self._check_signal_conditions(bar)
            if signal:
                signals.append(signal)
                logger.info(
                    f"Signal generated: {signal.direction} @ {signal.entry_price:.2f}, "
                    f"confidence: {signal.confidence:.1f}%"
                )

        return signals

    def _update_atr(self, bar: DollarBar) -> None:
        """Update Average True Range (ATR) for stop loss calculation.

        Args:
            bar: Current DollarBar
        """
        if len(self._true_ranges) == 0:
            # First bar, can't calculate ATR yet
            self._prev_high = bar.high
            self._prev_low = bar.low
            self._prev_close = bar.close
            return

        # Calculate True Range
        tr1 = bar.high - bar.low
        tr2 = abs(bar.high - self._prev_close)
        tr3 = abs(bar.low - self._prev_close)
        true_range = max(tr1, tr2, tr3)

        self._true_ranges.append(true_range)

        # Calculate ATR
        if len(self._true_ranges) == self._atr_period:
            self._atr = sum(self._true_ranges) / self._atr_period

        # Store previous values
        self._prev_high = bar.high
        self._prev_low = bar.low
        self._prev_close = bar.close

    def _check_signal_conditions(self, bar: DollarBar) -> MomentumSignal | None:
        """Check if signal conditions are met.

        Args:
            bar: Current DollarBar

        Returns:
            MomentumSignal if conditions met, None otherwise
        """
        # Get current indicator values
        ema_values = self.ema_calculator.get_current_emas()
        macd_values = self.macd_calculator.get_current_macd()
        rsi_value = self.rsi_calculator.get_current_rsi()

        # Check if all indicators are available
        if None in (ema_values['fast_ema'], ema_values['medium_ema'], ema_values['slow_ema']):
            return None
        if None in (macd_values['macd_line'], macd_values['histogram']):
            return None
        if rsi_value is None:
            return None

        # Check LONG conditions (RELAXED)
        long_conditions = (
            ema_values['fast_ema'] > ema_values['medium_ema'] > ema_values['slow_ema'] and
            macd_values['macd_line'] > 0 and
            30 <= rsi_value <= 70  # Expanded from 40-60
        )

        if long_conditions:
            return self._create_signal('LONG', bar, ema_values, macd_values, rsi_value)

        # Check SHORT conditions (RELAXED)
        short_conditions = (
            ema_values['fast_ema'] < ema_values['medium_ema'] < ema_values['slow_ema'] and
            macd_values['macd_line'] < 0 and
            30 <= rsi_value <= 70  # Expanded from 40-60
        )

        if short_conditions:
            return self._create_signal('SHORT', bar, ema_values, macd_values, rsi_value)

        return None

    def _create_signal(
        self,
        direction: str,
        bar: DollarBar,
        ema_values: dict,
        macd_values: dict,
        rsi_value: float
    ) -> MomentumSignal:
        """Create a MomentumSignal.

        Args:
            direction: 'LONG' or 'SHORT'
            bar: Current DollarBar
            ema_values: EMA values
            macd_values: MACD values
            rsi_value: RSI value

        Returns:
            MomentumSignal object
        """
        entry_price = bar.close

        # Calculate stop loss and take profit
        if self._atr is not None:
            atr_distance = self._atr * 1.0  # 1× ATR for stop loss
        else:
            # Fallback: use 0.25% of price
            atr_distance = entry_price * 0.0025

        if direction == 'LONG':
            stop_loss = entry_price - atr_distance
            take_profit = entry_price + (atr_distance * 2.0)  # 2:1 ratio
        else:  # SHORT
            stop_loss = entry_price + atr_distance
            take_profit = entry_price - (atr_distance * 2.0)  # 2:1 ratio

        # Calculate confidence (0-100)
        confidence = self._calculate_confidence(ema_values, macd_values, rsi_value)

        return MomentumSignal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            ema_fast=ema_values['fast_ema'],
            ema_medium=ema_values['medium_ema'],
            ema_slow=ema_values['slow_ema'],
            macd_line=macd_values['macd_line'],
            macd_signal=macd_values['signal_line'],
            macd_histogram=macd_values['histogram'],
            rsi_value=rsi_value,
            rsi_in_mid_band=self.rsi_calculator.is_in_mid_band()
        )

    def _calculate_confidence(
        self,
        ema_values: dict,
        macd_values: dict,
        rsi_value: float
    ) -> float:
        """Calculate signal confidence score.

        Args:
            ema_values: EMA values
            macd_values: MACD values
            rsi_value: RSI value

        Returns:
            Confidence score (0-100)
        """
        confidence = 50.0  # Base confidence

        # EMA alignment strength (±10 points)
        fast_medium_spread = abs(ema_values['fast_ema'] - ema_values['medium_ema'])
        medium_slow_spread = abs(ema_values['medium_ema'] - ema_values['slow_ema'])
        avg_ema = (ema_values['fast_ema'] + ema_values['medium_ema'] + ema_values['slow_ema']) / 3

        if avg_ema > 0:
            # Wider spread = stronger trend = higher confidence
            spread_strength = ((fast_medium_spread + medium_slow_spread) / avg_ema) * 10000
            confidence += min(spread_strength, 10)

        # MACD strength (±20 points)
        if macd_values['histogram'] is not None:
            macd_strength = min(abs(macd_values['histogram']) * 10, 20)
            confidence += macd_strength

        # RSI positioning (±10 points)
        # Closer to 50 = better positioning
        rsi_distance_from_mid = abs(rsi_value - 50)
        rsi_bonus = max(10 - rsi_distance_from_mid / 5, 0)
        confidence += rsi_bonus

        # Clamp to 0-100
        return max(0, min(100, confidence))

    def reset(self) -> None:
        """Reset all calculators and state."""
        self.ema_calculator.reset()
        self.macd_calculator.reset()
        self.rsi_calculator.reset()
        self._atr = None
        self._true_ranges.clear()
        logger.debug("Adaptive EMA Momentum strategy reset")
