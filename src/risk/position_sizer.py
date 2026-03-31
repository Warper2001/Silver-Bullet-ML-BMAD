"""Position sizing based on risk parameters and market conditions.

This module calculates position sizes using a Kelly Criterion-inspired approach:
- Risk amount = account_equity × risk_per_trade (default 2%)
- Stop distance = 1.2 × ATR (volatility-adjusted)
- Position size = risk_amount / stop_distance

This ensures consistent risk exposure regardless of market volatility.
"""

import logging
from dataclasses import dataclass

from src.data.models import SilverBulletSetup

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position size calculation.

    Attributes:
        entry_price: Signal entry price
        stop_loss: Calculated stop loss price
        take_profit: Calculated take profit price
        position_size: Number of contracts (rounded)
        dollar_risk: Dollar amount at risk
        stop_distance: ATR-based stop distance
        take_profit_distance: Distance to take profit
        risk_reward_ratio: R:R ratio used
        calculation_time_ms: Calculation performance in milliseconds
        valid: Whether position size is within limits
        validation_reason: Reason if position size is invalid
    """

    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    dollar_risk: float
    stop_distance: float
    take_profit_distance: float
    risk_reward_ratio: float
    calculation_time_ms: float
    valid: bool
    validation_reason: str | None


class PositionSizer:
    """Calculates position sizes based on risk parameters and market conditions.

    Uses the Kelly Criterion-inspired approach with configurable risk-reward:
    - Risk amount = account_equity × risk_per_trade (default 2%)
    - Stop distance = ATR multiplier × ATR (volatility-adjusted)
    - Take profit = risk_reward_ratio × stop_distance
    - Position size = risk_amount / stop_distance

    This ensures consistent risk exposure and defines profit targets.

    Attributes:
        _account_equity: Total account equity in dollars
        _risk_per_trade: Fraction of equity to risk per trade
        _max_position_size: Maximum contracts per trade
        _atr_multiplier: ATR multiplier for stop loss
        _risk_reward_ratio: Risk-reward ratio (default 3.0)

    Example:
        >>> sizer = PositionSizer(risk_reward_ratio=3.0)
        >>> result = sizer.calculate_position(signal, atr=50.0)
        >>> result.take_profit
        11860.0  # Entry + (3 × stop_distance)
    """

    DEFAULT_ACCOUNT_EQUITY = 10000.0
    DEFAULT_RISK_PER_TRADE = 0.02  # 2%
    DEFAULT_MAX_POSITION_SIZE = 5
    DEFAULT_ATR_MULTIPLIER = 0.3  # Updated for 3:1 R:R (was 1.2)
    DEFAULT_ATR_VALUE = 50.0
    DEFAULT_RISK_REWARD_RATIO = 3.0  # NEW: 3:1 risk-reward

    def __init__(
        self,
        account_equity: float = DEFAULT_ACCOUNT_EQUITY,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
        max_position_size: int = DEFAULT_MAX_POSITION_SIZE,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        risk_reward_ratio: float = DEFAULT_RISK_REWARD_RATIO,
    ) -> None:
        """Initialize position sizer with risk parameters.

        Args:
            account_equity: Total account equity in dollars (default $10,000)
            risk_per_trade: Fraction of equity to risk per trade (default 0.02 for 2%)
            max_position_size: Maximum contracts per trade (default 5)
            atr_multiplier: ATR multiplier for stop loss (default 0.3)
            risk_reward_ratio: Risk-reward ratio (default 3.0 for 3:1)

        Raises:
            ValueError: If parameters are invalid (negative equity, risk outside [0,1])
        """
        if account_equity <= 0:
            raise ValueError(
                f"Account equity must be positive, got {account_equity}"
            )
        if not 0.0 <= risk_per_trade <= 1.0:
            raise ValueError(
                f"Risk per trade must be in [0, 1], got {risk_per_trade}"
            )
        if max_position_size <= 0:
            raise ValueError(
                f"Max position size must be positive, got {max_position_size}"
            )
        if atr_multiplier <= 0:
            raise ValueError(
                f"ATR multiplier must be positive, got {atr_multiplier}"
            )
        if risk_reward_ratio <= 1.0:
            raise ValueError(
                f"Risk-reward ratio must be > 1.0, got {risk_reward_ratio}"
            )

        self._account_equity = account_equity
        self._risk_per_trade = risk_per_trade
        self._max_position_size = max_position_size
        self._atr_multiplier = atr_multiplier
        self._risk_reward_ratio = risk_reward_ratio

        logger.info(
            f"PositionSizer initialized: equity=${account_equity:,.2f}, "
            f"risk={risk_per_trade:.1%}, max_position={max_position_size}, "
            f"R:R={risk_reward_ratio:.1f}:1"
        )

    def calculate_dollar_risk(self) -> float:
        """Calculate the dollar amount to risk on this trade.

        Returns:
            Dollar risk amount (account_equity × risk_per_trade)

        Example:
            >>> sizer = PositionSizer(account_equity=10000, risk_per_trade=0.02)
            >>> sizer.calculate_dollar_risk()
            200.0  # $10,000 × 2%
        """
        dollar_risk = self._account_equity * self._risk_per_trade
        logger.debug(f"Dollar risk calculated: ${dollar_risk:.2f}")
        return dollar_risk

    def calculate_stop_loss_distance(self, atr: float) -> float:
        """Calculate stop loss distance based on ATR.

        Args:
            atr: Average True Range value from market data

        Returns:
            Stop loss distance in price points (atr_multiplier × ATR)

        Example:
            >>> sizer = PositionSizer(atr_multiplier=1.2)
            >>> sizer.calculate_stop_loss_distance(atr=50.0)
            60.0  # 1.2 × $50
        """
        if atr <= 0:
            logger.warning(
                "Invalid ATR value: {}, using default ${}".format(
                    atr, self.DEFAULT_ATR_VALUE
                ),
            )
            atr = self.DEFAULT_ATR_VALUE

        stop_distance = self._atr_multiplier * atr
        logger.debug(f"Stop distance calculated: ${stop_distance:.2f} (ATR=${atr:.2f})")
        return stop_distance

    def calculate_take_profit_distance(self, stop_distance: float) -> float:
        """Calculate take profit distance based on risk-reward ratio.

        Args:
            stop_distance: Stop loss distance in price points

        Returns:
            Take profit distance in price points (R:R ratio × stop_distance)

        Example:
            >>> sizer = PositionSizer(risk_reward_ratio=3.0)
            >>> sizer.calculate_take_profit_distance(60.0)
            180.0  # 3.0 × $60
        """
        take_profit_distance = self._risk_reward_ratio * stop_distance
        logger.debug(
            f"Take profit calculated: ${take_profit_distance:.2f} "
            f"({self._risk_reward_ratio:.1f}:1 R:R, SL=${stop_distance:.2f})"
        )
        return take_profit_distance

    def calculate_position_size(
        self, entry_price: float, stop_distance: float, dollar_risk: float
    ) -> int:
        """Calculate position size in contracts.

        Args:
            entry_price: Signal entry price
            stop_distance: Stop loss distance in price points
            dollar_risk: Dollar amount at risk

        Returns:
            Number of contracts (rounded to nearest whole number)

        Example:
            >>> sizer = PositionSizer()
            >>> sizer.calculate_position_size(
            ...     entry_price=11800.0,
            ...     stop_distance=60.0,
            ...     dollar_risk=200.0
            ... )
            3  # $200 / $60 = 3.33 → 3 contracts
        """
        if stop_distance <= 0:
            logger.warning(
                "Stop distance is zero or negative, "
                "using minimum 1 contract"
            )
            return 1

        # Calculate raw position size
        position_size_raw = dollar_risk / stop_distance

        # Round to nearest whole contract
        position_size = int(round(position_size_raw))

        # Ensure minimum of 1 contract
        position_size = max(1, position_size)

        logger.debug(
            "Position size calculated: {:.2f} → {} contracts".format(
                position_size_raw, position_size
            )
        )
        return position_size

    def _validate_position_size(self, position_size: int) -> tuple[bool, str | None]:
        """Validate position size against constraints.

        Args:
            position_size: Calculated position size in contracts

        Returns:
            Tuple of (is_valid: bool, reason: str | None)
        """
        if position_size > self._max_position_size:
            return (
                False,
                "Position size {} exceeds maximum {}".format(
                    position_size, self._max_position_size
                ),
            )

        return True, None

    def calculate_position(
        self, signal: SilverBulletSetup, atr: float | None = None
    ) -> PositionSizeResult:
        """Calculate position size and take profit for a signal.

        Args:
            signal: Silver Bullet setup with entry price
            atr: ATR value (uses signal.atr if None)

        Returns:
            PositionSizeResult with entry, stop, TP, position size, and risk details

        Performance:
            Completes in < 5ms per signal
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get entry price from signal
            entry_price = signal.entry_price

            # Calculate dollar risk
            dollar_risk = self.calculate_dollar_risk()

            # Calculate stop loss distance
            atr_value = atr or getattr(signal, "atr", self.DEFAULT_ATR_VALUE)
            stop_distance = self.calculate_stop_loss_distance(atr_value)

            # Calculate take profit distance (NEW for 3:1 R:R)
            take_profit_distance = self.calculate_take_profit_distance(stop_distance)

            # Calculate stop loss price
            if signal.direction == "bullish":
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + take_profit_distance
            else:  # bearish
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - take_profit_distance

            # Calculate position size
            position_size = self.calculate_position_size(
                entry_price, stop_distance, dollar_risk
            )

            # Validate position size
            valid, validation_reason = self._validate_position_size(position_size)

            # Calculate performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log result
            logger.info(
                f"Position sizing: entry=${entry_price:.2f}, "
                f"stop=${stop_loss:.2f}, TP=${take_profit:.2f}, "
                f"size={position_size}, risk=${dollar_risk:.2f}, "
                f"R:R={self._risk_reward_ratio:.1f}:1, time={elapsed_ms:.2f}ms"
            )

            return PositionSizeResult(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,  # NEW
                position_size=position_size,
                dollar_risk=dollar_risk,
                stop_distance=stop_distance,
                take_profit_distance=take_profit_distance,  # NEW
                risk_reward_ratio=self._risk_reward_ratio,  # NEW
                calculation_time_ms=elapsed_ms,
                valid=valid,
                validation_reason=validation_reason,
            )

        except Exception as e:
            logger.error(f"Error calculating position: {e}")
            raise
