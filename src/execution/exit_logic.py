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

from src.execution.models import ExitOrder, PositionMonitoringState, TradeOrder

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
        """Check if position should be exited based on hold time.

        Args:
            state: Current position monitoring state

        Returns:
            ExitOrder if max hold time reached, None otherwise
        """
        if state.is_at_max_hold_time(self.max_hold_minutes):
            # Calculate P&L
            multiplier = 0.50  # MNQ contract multiplier
            price_diff = state.current_price - state.position.entry_price

            if state.position.direction == "short":
                price_diff = -price_diff

            pnl = price_diff * multiplier * state.position.remaining_quantity

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
                exit_reason="time_stop",
                timestamp=datetime.now(),
                pnl=pnl,
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

    def calculate_rr_achieved(self, entry: float, exit_price: float, stop_loss: float) -> float:
        """Calculate reward-risk ratio achieved.

        Args:
            entry: Entry price
            exit_price: Exit price
            stop_loss: Stop loss price

        Returns:
            R:R achieved (positive for profit, negative for loss)
        """
        risk = abs(entry - stop_loss)
        if risk == 0:
            return 0.0

        if exit_price >= entry:
            # Profit: (exit - entry) / risk
            return (exit_price - entry) / risk
        else:
            # Loss: always -1.0 (stopped at defined risk)
            return -1.0

    def _create_take_profit_exit(self, state: PositionMonitoringState) -> ExitOrder:
        """Create a take profit exit order.

        Args:
            state: Position monitoring state

        Returns:
            ExitOrder for take profit
        """
        multiplier = 0.50  # MNQ contract multiplier
        price_diff = state.current_price - state.position.entry_price

        if state.position.direction == "short":
            price_diff = -price_diff

        pnl = price_diff * multiplier * state.position.remaining_quantity

        rr_achieved = self.calculate_rr_achieved(
            state.position.entry_price,
            state.current_price,
            state.position.stop_loss
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
            exit_reason="take_profit",
            timestamp=datetime.now(),
            pnl=pnl,
            rr_ratio=rr_achieved
        )

    def _create_stop_loss_exit(self, state: PositionMonitoringState) -> ExitOrder:
        """Create a stop loss exit order.

        Args:
            state: Position monitoring state

        Returns:
            ExitOrder for stop loss
        """
        multiplier = 0.50  # MNQ contract multiplier
        price_diff = state.current_price - state.position.entry_price

        if state.position.direction == "short":
            price_diff = -price_diff

        pnl = price_diff * multiplier * state.position.remaining_quantity

        # Stop loss always means -1R
        rr_achieved = -1.0

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
            exit_reason="stop_loss",
            timestamp=datetime.now(),
            pnl=pnl,
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

        Priority for open positions:
        1. Partial exit at 1.5R (if not yet executed)
        2. Final exit at 2R take profit
        3. Final exit at 10-minute time stop

        For partially closed positions:
        1. Final exit at 2R take profit
        2. Final exit at 10-minute time stop

        Args:
            state: Current position monitoring state

        Returns:
            ExitOrder if exit condition met, None otherwise
        """
        # Check if position is still open (no partial yet)
        if state.position.position_state == "open":
            # Check for partial exit at 1.5R
            if self._check_partial_exit_level(state):
                return self._create_partial_exit(state)

        # Check for final exits (regardless of partial state)
        # Priority: 2R TP > time stop

        # Check 2R take profit
        if self._check_final_take_profit(state):
            return self._create_final_exit(state, "hybrid_trail")

        # Check time stop
        if state.is_at_max_hold_time(self.max_hold_minutes):
            return self._create_final_exit(state, "time_stop")

        return None

    def scale_out_partial(self, position: TradeOrder) -> ExitOrder:
        """Create a partial scale-out exit order.

        Args:
            position: Position to partially exit

        Returns:
            ExitOrder for partial exit
        """
        quantity = int(position.remaining_quantity * self.partial_percent)

        # Round down for odd quantities (e.g., 5 * 0.5 = 2.5 → 2)
        if quantity < 1:
            quantity = 1

        # Calculate 1.5R price level
        risk = position.risk_per_contract()
        partial_target = position.entry_price + (risk * self.partial_rr) if position.direction == "long" else position.entry_price - (risk * self.partial_rr)

        # Calculate P&L
        multiplier = 0.50
        price_diff = partial_target - position.entry_price
        if position.direction == "short":
            price_diff = -price_diff
        pnl = price_diff * multiplier * quantity

        logger.info(
            f"Hybrid partial exit for position {position.trade_id}: "
            f"quantity={quantity}, "
            f"partial_price={partial_target}, "
            f"pnl=${pnl:.2f}"
        )

        return ExitOrder(
            position_id=position.trade_id,
            exit_type="partial",
            quantity=quantity,
            exit_price=partial_target,
            exit_reason="hybrid_partial",
            timestamp=datetime.now(),
            pnl=pnl,
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

        return ExitOrder(
            position_id=state.position.trade_id,
            exit_type="partial",
            quantity=quantity,
            exit_price=state.current_price,
            exit_reason="hybrid_partial",
            timestamp=datetime.now(),
            pnl=pnl,
            rr_ratio=self.partial_rr
        )

    def _create_final_exit(self, state: PositionMonitoringState, exit_reason: str) -> ExitOrder:
        """Create a final exit order for remaining position.

        Args:
            state: Position monitoring state
            exit_reason: Exit reason ("hybrid_trail" or "time_stop")

        Returns:
            ExitOrder for final exit
        """
        multiplier = 0.50
        price_diff = state.current_price - state.position.entry_price
        if state.position.direction == "short":
            price_diff = -price_diff
        pnl = price_diff * multiplier * state.position.remaining_quantity

        # Calculate R:R achieved
        risk = state.position.risk_per_contract()
        rr_achieved = (price_diff / risk) if risk > 0 else 0.0

        logger.info(
            f"Hybrid final exit ({exit_reason}) for position {state.position.trade_id}: "
            f"quantity={state.position.remaining_quantity}, "
            f"exit_price={state.current_price}, "
            f"pnl=${pnl:.2f}, "
            f"rr_achieved={rr_achieved:.2f}"
        )

        return ExitOrder(
            position_id=state.position.trade_id,
            exit_type="full",
            quantity=state.position.remaining_quantity,
            exit_price=state.current_price,
            exit_reason=exit_reason,
            timestamp=datetime.now(),
            pnl=pnl,
            rr_ratio=rr_achieved
        )


class ExitLogic:
    """Wrapper class for exit logic strategies in backtesting.

    Provides a simplified interface for ensemble backtesting by combining
    all three exit strategies (time-based, risk-reward, hybrid) into a
    single interface.
    """

    def __init__(self, config_path: str = "config-sim.yaml"):
        """Initialize exit logic with default strategies.

        Args:
            config_path: Path to configuration file
        """
        self.time_exit = TimeBasedExit(max_hold_minutes=10.0)
        self.rr_exit = RiskRewardExit(target_rr=2.0)
        self.hybrid_exit = HybridExit(partial_rr=1.5, partial_percent=0.50, max_hold_minutes=10.0)

        logger.info("ExitLogic initialized with time, risk-reward, and hybrid exit strategies")

    def evaluate_exit(self, bar, position: dict) -> Optional[object]:
        """Evaluate if position should be exited.

        For backtesting, uses simplified logic:
        1. Check time stop (10 minutes max)
        2. Check take profit (2R)
        3. Check stop loss breach

        Args:
            bar: Current dollar bar (dict or Series)
            position: Position dictionary

        Returns:
            Exit decision object with should_exit bool and exit_price
        """
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

        # Calculate hold time
        entry_time = position["entry_time"]
        if isinstance(bar, dict):
            bar_time = bar["timestamp"]
        else:
            bar_time = bar["timestamp"]

        hold_minutes = (bar_time - entry_time).total_seconds() / 60

        # Check 1: Time stop (10 minutes max)
        if hold_minutes >= 10.0:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="time_stop")

        # Check 2: Take profit hit
        if direction == "long" and current_price >= take_profit:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="take_profit")
        elif direction == "short" and current_price <= take_profit:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="take_profit")

        # Check 3: Stop loss hit
        if direction == "long" and current_price <= stop_loss:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="stop_loss")
        elif direction == "short" and current_price >= stop_loss:
            return ExitDecision(should_exit=True, exit_price=current_price, exit_reason="stop_loss")

        # No exit triggered
        return ExitDecision(should_exit=False, exit_price=None, exit_reason="")
