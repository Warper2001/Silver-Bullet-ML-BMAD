"""Exit logic strategies for ensemble trading positions.

This module implements three exit strategies for managing ensemble positions:
1. TimeBasedExit - Maximum 10-minute hold time
2. RiskRewardExit - 2:1 reward-risk ratio take profit and stop loss
3. HybridExit - Partial scaling at 1.5R with trailing stop to breakeven

Each strategy independently evaluates positions and can generate exit orders.
"""

import logging
from datetime import datetime
from typing import Optional

from src.execution.models import ExitOrder, PositionMonitoringState, TradeOrder, _now, NY_TZ

logger = logging.getLogger(__name__)


class TimeBasedExit:
    """Time-based exit strategy with maximum hold time.

    Monitors positions and exits at the 10-minute maximum hold time
    as specified in NFR2. This is a hard stop to prevent over-holding
    positions regardless of P&L.

    Attributes:
        max_hold_minutes: Maximum hold time in minutes (default 10)
    """

    def __init__(self, max_hold_minutes: float = 10.0) -> None:
        """Initialize time-based exit strategy.

        Args:
            max_hold_minutes: Maximum hold time in minutes (default 10.0)

        Raises:
            ValueError: If max_hold_minutes is not positive
        """
        if max_hold_minutes <= 0:
            raise ValueError("max_hold_minutes must be positive")

        self.max_hold_minutes = max_hold_minutes
        logger.info(f"TimeBasedExit initialized with max_hold_minutes={max_hold_minutes}")

    def check_exit(self, state: PositionMonitoringState) -> Optional[ExitOrder]:
        """Check if position should be exited based on hold time."""
        if state.is_at_max_hold_time(self.max_hold_minutes):
            # Calculate P&L
            multiplier = 0.50  # MNQ contract multiplier
            price_diff = state.current_price - state.position.entry_price

            if state.position.direction == "short":
                price_diff = -price_diff

            pnl = price_diff * multiplier * state.position.remaining_quantity
            pnl_ticks = price_diff / 0.25 # Patch 8: P&L in ticks

            # Calculate R:R achieved
            risk = state.position.risk_per_contract()
            rr_achieved = (price_diff / risk) if risk > 0 else 0.0

            logger.info(
                f"Time-based exit triggered for position {state.position.trade_id}: "
                f"hold_time={state.hold_time_minutes():.1f}min, "
                f"pnl=${pnl:.2f}, "
                f"rr_achieved={rr_achieved:.2f}"
            )

            return ExitOrder(
                position_id=state.position.trade_id,
                exit_type="full",
                quantity=state.position.remaining_quantity,
                exit_price=state.current_price,
                exit_reason="Time stop (10-min max)", # Decision 2A
                timestamp=_now(), # Patch 7: aware time
                pnl=pnl,
                pnl_ticks=pnl_ticks,
                rr_ratio=rr_achieved
            )

        return None

    def calculate_hold_time(self, entry_time: datetime, current_time: datetime) -> int:
        """Calculate time since position entry in seconds.

        Args:
            entry_time: Position entry timestamp
            current_time: Current timestamp

        Returns:
            Hold time in seconds
        """
        delta = current_time - entry_time
        return int(delta.total_seconds())

    def get_hold_time_minutes(self, state: PositionMonitoringState) -> float:
        """Get hold time in minutes for a position.

        Args:
            state: Position monitoring state

        Returns:
            Hold time in minutes
        """
        return state.hold_time_minutes()


class RiskRewardExit:
    """Risk-reward based exit strategy with 2:1 ratio.

    Monitors positions and exits when either:
    - Take profit level is hit (2:1 reward-risk ratio)
    - Stop loss level is hit (as defined by entry signal)

    This is a traditional binary exit strategy that fully exits
    the position when either level is hit.

    Attributes:
        rr_ratio: Target reward-risk ratio (default 2.0)
    """

    def __init__(self, rr_ratio: float = 2.0) -> None:
        """Initialize risk-reward exit strategy.

        Args:
            rr_ratio: Target reward-risk ratio (default 2.0)

        Raises:
            ValueError: If rr_ratio is not positive
        """
        if rr_ratio <= 0:
            raise ValueError("rr_ratio must be positive")

        self.rr_ratio = rr_ratio
        logger.info(f"RiskRewardExit initialized with rr_ratio={rr_ratio}")

    def check_exit(self, state: PositionMonitoringState) -> Optional[ExitOrder]:
        """Check if position should be exited based on TP or SL hit.

        Priority:
        1. Stop loss hit (highest priority - protect capital)
        2. Take profit hit (realize gains)

        Args:
            state: Current position monitoring state

        Returns:
            ExitOrder if TP or SL hit, None otherwise
        """
        # Check stop loss first (higher priority)
        if self.check_stop_loss_hit(state.current_price, state.position.stop_loss, state.position.direction):
            return self._create_stop_loss_exit(state)

        # Then check take profit
        if self.check_take_profit_hit(state.current_price, state.position.take_profit, state.position.direction):
            return self._create_take_profit_exit(state)

        return None

    def calculate_take_profit(self, entry: float, stop_loss: float, direction: str, rr: float | None = None) -> float:
        """Calculate take profit level based on risk and R:R ratio.

        Args:
            entry: Entry price
            stop_loss: Stop loss price
            direction: Trade direction ("long" or "short")
            rr: Reward-risk ratio (uses instance rr_ratio if None)

        Returns:
            Take profit price level
        """
        if rr is None:
            rr = self.rr_ratio

        risk = abs(entry - stop_loss)
        reward = risk * rr

        if direction == "long":
            return entry + reward
        else:  # short
            return entry - reward

    def check_take_profit_hit(self, current_price: float, tp: float, direction: str) -> bool:
        """Check if take profit level has been hit.

        Args:
            current_price: Current market price
            tp: Take profit level
            direction: Trade direction

        Returns:
            True if take profit has been hit
        """
        if direction == "long":
            return current_price >= tp
        else:  # short
            return current_price <= tp

    def check_stop_loss_hit(self, current_price: float, sl: float, direction: str) -> bool:
        """Check if stop loss level has been hit.

        Args:
            current_price: Current market price
            sl: Stop loss level
            direction: Trade direction

        Returns:
            True if stop loss has been hit
        """
        if direction == "long":
            return current_price <= sl
        else:  # short
            return current_price >= sl

    def calculate_rr_achieved(self, entry: float, exit_price: float, stop_loss: float, direction: str) -> float:
        """Calculate reward-risk ratio achieved (Fix Patch 6)."""
        risk = abs(entry - stop_loss)
        if risk == 0:
            return 0.0

        if direction == "long":
            if exit_price >= entry:
                return (exit_price - entry) / risk
            return -1.0 # Simplified for SL hit
        else: # short
            if exit_price <= entry:
                return (entry - exit_price) / risk
            return -1.0

    def _create_take_profit_exit(self, state: PositionMonitoringState) -> ExitOrder:
        """Create a take profit exit order."""
        multiplier = 0.50  # MNQ contract multiplier
        price_diff = state.current_price - state.position.entry_price

        if state.position.direction == "short":
            price_diff = -price_diff

        pnl = price_diff * multiplier * state.position.remaining_quantity
        pnl_ticks = price_diff / 0.25

        rr_achieved = self.calculate_rr_achieved(
            state.position.entry_price,
            state.current_price,
            state.position.stop_loss,
            state.position.direction
        )

        logger.info(
            f"Take profit exit triggered for position {state.position.trade_id}: "
            f"exit_price={state.current_price}, "
            f"pnl=${pnl:.2f}, "
            f"rr_achieved={rr_achieved:.2f}"
        )

        return ExitOrder(
            position_id=state.position.trade_id,
            exit_type="full",
            quantity=state.position.remaining_quantity,
            exit_price=state.current_price,
            exit_reason="Take profit",
            timestamp=_now(),
            pnl=pnl,
            pnl_ticks=pnl_ticks,
            rr_ratio=rr_achieved
        )

    def _create_stop_loss_exit(self, state: PositionMonitoringState) -> ExitOrder:
        """Create a stop loss exit order."""
        multiplier = 0.50  # MNQ contract multiplier
        price_diff = state.current_price - state.position.entry_price

        if state.position.direction == "short":
            price_diff = -price_diff

        pnl = price_diff * multiplier * state.position.remaining_quantity
        pnl_ticks = price_diff / 0.25

        # Stop loss always means -1R or calculated loss
        rr_achieved = self.calculate_rr_achieved(
            state.position.entry_price,
            state.current_price,
            state.position.stop_loss,
            state.position.direction
        )

        logger.warning(
            f"Stop loss exit triggered for position {state.position.trade_id}: "
            f"exit_price={state.current_price}, "
            f"pnl=${pnl:.2f}, "
            f"rr_achieved={rr_achieved:.2f}"
        )

        return ExitOrder(
            position_id=state.position.trade_id,
            exit_type="full",
            quantity=state.position.remaining_quantity,
            exit_price=state.current_price,
            exit_reason="Stop loss",
            timestamp=_now(),
            pnl=pnl,
            pnl_ticks=pnl_ticks,
            rr_ratio=rr_achieved
        )


class HybridExit:
    """Hybrid exit strategy with partial scaling and trailing stop.

    Implements a two-stage exit strategy:
    1. Scale out 50% of position at 1.5R (partial take profit)
    2. Trail remaining 50% with stop to breakeven
    3. Close remaining at 2R take profit or 10-minute time stop

    This strategy aims to capture some profit while leaving room for
    additional upside, with risk protection via trailing stop.

    Attributes:
        partial_rr: R:R ratio for partial exit (default 1.5)
        partial_percent: Percentage to exit at partial (default 0.50)
        max_hold_minutes: Maximum hold time for remaining position (default 10.0)
    """

    def __init__(self, partial_rr: float = 1.5, partial_percent: float = 0.50, max_hold_minutes: float = 10.0) -> None:
        """Initialize hybrid exit strategy.

        Args:
            partial_rr: R:R ratio for partial exit (default 1.5)
            partial_percent: Percentage to exit at partial (default 0.50)
            max_hold_minutes: Maximum hold time in minutes (default 10.0)

        Raises:
            ValueError: If parameters are invalid
        """
        if partial_rr <= 0:
            raise ValueError("partial_rr must be positive")
        if partial_percent <= 0 or partial_percent > 1.0:
            raise ValueError("partial_percent must be between 0 and 1")
        if max_hold_minutes <= 0:
            raise ValueError("max_hold_minutes must be positive")

        self.partial_rr = partial_rr
        self.partial_percent = partial_percent
        self.max_hold_minutes = max_hold_minutes
        logger.info(
            f"HybridExit initialized with partial_rr={partial_rr}, "
            f"partial_percent={partial_percent}, max_hold_minutes={max_hold_minutes}"
        )

    def check_exit(self, state: PositionMonitoringState) -> Optional[ExitOrder]:
        """Check if position should be exited using hybrid strategy.

        Decision priority: 
        1. Stop loss hit (Critical Patch 4)
        2. Partial exit at 1.5R
        3. Final exit at 2R take profit
        4. Final exit at 10-minute time stop
        """
        # Patch 4: Check stop loss first (always highest priority)
        if self._check_stop_loss(state):
            return self._create_final_exit(state, "Stop loss")

        # Check if position is still open (no partial yet)
        if state.position.position_state == "open":
            # Check for partial exit at 1.5R
            if self._check_partial_exit_level(state):
                return self._create_partial_exit(state)

        # Check for final exits
        # Check 2R take profit
        if self._check_final_take_profit(state):
            return self._create_final_exit(state, "Hybrid trail (2R)")

        # Check time stop
        if state.is_at_max_hold_time(self.max_hold_minutes):
            return self._create_final_exit(state, "Time stop (10-min max)")

        return None

    def _check_stop_loss(self, state: PositionMonitoringState) -> bool:
        """Check if stop loss has been hit (handles trailed stop)."""
        if state.position.direction == "long":
            return state.current_price <= state.position.stop_loss
        else: # short
            return state.current_price >= state.position.stop_loss

    def scale_out_partial(self, position: TradeOrder) -> ExitOrder:
        """Create a partial scale-out exit order."""
        # Patch 9: Rounding fix for small positions
        quantity = int(position.remaining_quantity * self.partial_percent)
        
        # Ensure we don't scale out more than we have or leave an invalid state
        if quantity < 1:
            quantity = 1
        
        # Guard: If remaining is 1, a 50% scale-out is not possible as partial.
        # It must be a full exit or stay open.
        if quantity >= position.remaining_quantity:
            quantity = position.remaining_quantity

        # Calculate target based on direction
        risk = position.risk_per_contract()
        multiplier = 1.0 if position.direction == "long" else -1.0
        partial_target = position.entry_price + (risk * self.partial_rr * multiplier)

        # Calculate P&L
        multiplier_usd = 0.50
        price_diff = (partial_target - position.entry_price) * multiplier
        pnl = price_diff * multiplier_usd * quantity
        pnl_ticks = (partial_target - position.entry_price) / 0.25

        logger.info(
            f"Hybrid partial exit for position {position.trade_id}: "
            f"quantity={quantity}, "
            f"partial_price={partial_target:.2f}, "
            f"pnl=${pnl:.2f}"
        )

        return ExitOrder(
            position_id=position.trade_id,
            exit_type="partial",
            quantity=quantity,
            exit_price=partial_target,
            exit_reason="Hybrid partial (1.5R)",
            timestamp=_now(),
            pnl=pnl,
            pnl_ticks=pnl_ticks,
            rr_ratio=self.partial_rr
        )

    def trail_stop_to_breakeven(self, position: TradeOrder) -> None:
        """Trail stop loss to breakeven for remaining position.

        Args:
            position: Position to trail stop for (modified in-place)
        """
        old_stop = position.stop_loss
        position.stop_loss = position.entry_price

        logger.info(
            f"Trailed stop loss to breakeven for position {position.trade_id}: "
            f"{old_stop} → {position.entry_price}"
        )

    def _check_partial_exit_level(self, state: PositionMonitoringState) -> bool:
        """Check if price has hit partial exit level (1.5R).

        Args:
            state: Position monitoring state

        Returns:
            True if partial level has been hit
        """
        risk = state.position.risk_per_contract()
        partial_target = state.position.entry_price + (risk * self.partial_rr) if state.position.direction == "long" else state.position.entry_price - (risk * self.partial_rr)

        if state.position.direction == "long":
            return state.current_price >= partial_target
        else:  # short
            return state.current_price <= partial_target

    def _check_final_take_profit(self, state: PositionMonitoringState) -> bool:
        """Check if price has hit final take profit (2R).

        Args:
            state: Position monitoring state

        Returns:
            True if 2R take profit has been hit
        """
        return state.position.is_at_take_profit(state.current_price)

    def _create_partial_exit(self, state: PositionMonitoringState) -> ExitOrder:
        """Create a partial exit order at 1.5R.

        Args:
            state: Position monitoring state

        Returns:
            ExitOrder for partial exit
        """
        quantity = int(state.position.remaining_quantity * self.partial_percent)

        # Round down for odd quantities
        if quantity < 1:
            quantity = 1

        multiplier = 0.50
        price_diff = state.current_price - state.position.entry_price
        if state.position.direction == "short":
            price_diff = -price_diff
        pnl = price_diff * multiplier * quantity

        logger.info(
            f"Hybrid partial exit triggered for position {state.position.trade_id}: "
            f"quantity={quantity}, "
            f"exit_price={state.current_price}, "
            f"pnl=${pnl:.2f}, "
            f"rr={self.partial_rr:.1f}"
        )

    def _create_partial_exit(self, state: PositionMonitoringState) -> ExitOrder:
        """Create a partial scale-out exit order."""
        # Fix: ensure quantity rounding matches scale_out_partial
        quantity = int(state.position.remaining_quantity * self.partial_percent)
        if quantity < 1: quantity = 1
        if quantity >= state.position.remaining_quantity: quantity = state.position.remaining_quantity

        # Calculate P&L
        multiplier = 0.50
        price_diff = state.current_price - state.position.entry_price
        if state.position.direction == "short":
            price_diff = -price_diff
        
        pnl = price_diff * multiplier * quantity
        pnl_ticks = (state.current_price - state.position.entry_price) / 0.25

        return ExitOrder(
            position_id=state.position.trade_id,
            exit_type="partial",
            quantity=quantity,
            exit_price=state.current_price,
            exit_reason="Hybrid partial (1.5R)",
            timestamp=_now(),
            pnl=pnl,
            pnl_ticks=pnl_ticks,
            rr_ratio=self.partial_rr
        )

    def _create_final_exit(self, state: PositionMonitoringState, exit_reason: str) -> ExitOrder:
        """Create a final exit order for remaining position."""
        multiplier = 0.50
        price_diff = state.current_price - state.position.entry_price
        if state.position.direction == "short":
            price_diff = -price_diff
        
        pnl = price_diff * multiplier * state.position.remaining_quantity
        pnl_ticks = (state.current_price - state.position.entry_price) / 0.25

        # Calculate R:R achieved
        risk = state.position.risk_per_contract()
        rr_achieved = (price_diff / risk) if risk > 0 else 0.0

        logger.info(
            f"Hybrid final exit ({exit_reason}) for position {state.position.trade_id}: "
            f"quantity={state.position.remaining_quantity}, "
            f"pnl=${pnl:.2f}, ticks={pnl_ticks:.1f}"
        )

        return ExitOrder(
            position_id=state.position.trade_id,
            exit_type="full",
            quantity=state.position.remaining_quantity,
            exit_price=state.current_price,
            exit_reason=exit_reason,
            timestamp=_now(),
            pnl=pnl,
            pnl_ticks=pnl_ticks,
            rr_ratio=rr_achieved
        )


class ExitLogic:
    """Wrapper class for exit logic strategies in backtesting."""

    def __init__(self, config_path: str = "config-sim.yaml"):
        """Initialize exit logic with correct strategy parameters."""
        self.time_exit = TimeBasedExit(max_hold_minutes=10.0)
        # Patch 3: Fix parameter name (rr_ratio vs target_rr)
        self.rr_exit = RiskRewardExit(rr_ratio=2.0)
        self.hybrid_exit = HybridExit(partial_rr=1.5, partial_percent=0.50, max_hold_minutes=10.0)

        logger.info("ExitLogic initialized with time, risk-reward, and hybrid exit strategies")

    def evaluate_exit(self, bar, position: dict) -> Optional[object]:
        """Evaluate exit using correct priorities: SL > TP > Time."""
        from dataclasses import dataclass

        @dataclass
        class ExitDecision:
            should_exit: bool
            exit_price: Optional[float]
            exit_reason: str

        current_price = bar["close"]
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        direction = position["direction"]
        bar_time = bar["timestamp"]
        entry_time = position["entry_time"]

        # 1. Stop loss hit (Priority 1)
        if direction == "long" and current_price <= stop_loss:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="Stop loss")
        elif direction == "short" and current_price >= stop_loss:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="Stop loss")

        # 2. Take profit hit (Priority 2)
        if direction == "long" and current_price >= take_profit:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="Take profit")
        elif direction == "short" and current_price <= take_profit:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="Take profit")

        # 3. Time stop (Priority 3)
        hold_minutes = (bar_time - entry_time).total_seconds() / 60
        if hold_minutes >= 10.0:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="Time stop (10-min max)")

        # No exit triggered
        return ExitDecision(should_exit=False, exit_price=None, exit_reason="")
