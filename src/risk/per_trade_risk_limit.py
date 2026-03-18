"""Per-trade risk limit enforcement.

This module implements a per-trade risk limit that caps the maximum
dollar risk for any single trade. This prevents any individual trade
from exposing the account to excessive loss.

Features:
- Maximum dollar risk per trade enforcement
- Trade validation against risk limit
- Risk calculation methods
- Max allowed quantity calculation
- Account balance updates
- CSV audit trail logging
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PerTradeRiskLimit:
    """Enforce maximum dollar risk per trade.

    Attributes:
        _max_risk_dollars: Maximum dollar risk per trade
        _account_balance: Current account balance
        _audit_trail_path: Path to CSV audit trail file

    Example:
        >>> limit = PerTradeRiskLimit(
        ...     max_risk_dollars=500,
        ...     account_balance=50000
        ... )
        >>> limit.validate_trade(
        ...     entry_price=11750,
        ...     stop_loss_price=11730,
        ...     quantity=5
        ... )
        {'is_valid': True, 'estimated_risk': 100}
    """

    def __init__(
        self,
        max_risk_dollars: float,
        account_balance: float,
        audit_trail_path: Optional[str] = None
    ) -> None:
        """Initialize per-trade risk limit.

        Args:
            max_risk_dollars: Maximum dollar risk per trade
            account_balance: Current account balance
            audit_trail_path: Path to CSV audit trail file (optional)

        Raises:
            ValueError: If max_risk_dollars <= 0

        Example:
            >>> limit = PerTradeRiskLimit(
            ...     max_risk_dollars=500,
            ...     account_balance=50000
            ... )
        """
        if max_risk_dollars <= 0:
            raise ValueError(
                "Max risk dollars must be positive: {}".format(
                    max_risk_dollars
                )
            )

        self._max_risk_dollars = max_risk_dollars
        self._account_balance = account_balance
        self._audit_trail_path = audit_trail_path

        logger.info(
            "PerTradeRiskLimit initialized: max_risk=${:.2f}".format(
                max_risk_dollars
            )
        )

    def validate_trade(
        self,
        entry_price: float,
        stop_loss_price: float,
        quantity: int
    ) -> dict:
        """Validate trade against per-trade risk limit.

        Args:
            entry_price: Entry price per contract
            stop_loss_price: Stop loss price per contract
            quantity: Number of contracts

        Returns:
            Dictionary with validation result:
            - is_valid: Whether trade is within limit
            - estimated_risk: Estimated dollar risk
            - risk_per_contract: Risk per contract
            - max_allowed_quantity: Max contracts allowed
            - violation_amount: Amount over limit (if invalid)

        Example:
            >>> result = limit.validate_trade(
            ...     entry_price=11750,
            ...     stop_loss_price=11730,
            ...     quantity=5
            ... )
            >>> if result['is_valid']:
            ...     print("Trade within risk limit")
        """
        # Calculate risk per contract
        risk_per_contract = self.get_risk_per_contract(
            entry_price,
            stop_loss_price
        )

        # Calculate total estimated risk
        estimated_risk = self.calculate_risk_dollars(
            entry_price,
            stop_loss_price,
            quantity
        )

        # Calculate max allowed quantity
        max_allowed_quantity = self.get_max_allowed_quantity(
            entry_price,
            stop_loss_price
        )

        # Check if within limit
        is_valid = estimated_risk <= self._max_risk_dollars

        # Calculate violation amount if invalid
        violation_amount = None
        if not is_valid:
            violation_amount = estimated_risk - self._max_risk_dollars

        # Determine block reason
        block_reason = None
        if not is_valid:
            block_reason = "Per-trade risk limit exceeded"

        # Log validation
        event_type = "REJECT" if not is_valid else "VALIDATE"
        self._log_audit_event(
            event_type,
            entry_price,
            stop_loss_price,
            quantity,
            estimated_risk,
            is_valid,
            violation_amount
        )

        return {
            "is_valid": is_valid,
            "estimated_risk": estimated_risk,
            "risk_per_contract": risk_per_contract,
            "max_allowed_quantity": max_allowed_quantity,
            "violation_amount": violation_amount,
            "block_reason": block_reason
        }

    def calculate_risk_dollars(
        self,
        entry_price: float,
        stop_loss_price: float,
        quantity: int
    ) -> float:
        """Calculate dollar risk for a trade.

        Args:
            entry_price: Entry price per contract
            stop_loss_price: Stop loss price per contract
            quantity: Number of contracts

        Returns:
            Dollar risk amount

        Example:
            >>> risk = limit.calculate_risk_dollars(
            ...     entry_price=11750,
            ...     stop_loss_price=11730,
            ...     quantity=5
            ... )
            >>> print(risk)
            200.0
        """
        # Calculate risk per contract
        risk_per_contract = self.get_risk_per_contract(
            entry_price,
            stop_loss_price
        )

        # Total risk
        return risk_per_contract * quantity

    def get_max_allowed_quantity(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> int:
        """Get maximum contracts allowed within risk limit.

        Args:
            entry_price: Entry price per contract
            stop_loss_price: Stop loss price per contract

        Returns:
            Maximum number of contracts allowed

        Example:
            >>> max_qty = limit.get_max_allowed_quantity(
            ...     entry_price=11750,
            ...     stop_loss_price=11730
            ... )
            >>> print(max_qty)
            12
        """
        # Calculate risk per contract
        risk_per_contract = self.get_risk_per_contract(
            entry_price,
            stop_loss_price
        )

        # Calculate max quantity (floor to be conservative)
        if risk_per_contract > 0:
            max_qty = int(self._max_risk_dollars / risk_per_contract)
        else:
            max_qty = 0

        return max_qty

    def get_risk_per_contract(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """Calculate risk per contract.

        Args:
            entry_price: Entry price per contract
            stop_loss_price: Stop loss price per contract

        Returns:
            Dollar risk per contract

        Example:
            >>> risk_per_contract = limit.get_risk_per_contract(
            ...     entry_price=11750,
            ...     stop_loss_price=11730
            ... )
            >>> print(risk_per_contract)
            40.0
        """
        # MNQ point value is $2 per contract
        # Risk = abs(entry_price - stop_loss_price) × $2

        price_diff = abs(entry_price - stop_loss_price)
        risk_per_contract = price_diff * 2

        return risk_per_contract

    def update_account_balance(self, new_balance: float) -> None:
        """Update account balance (recalculates limits if needed).

        Args:
            new_balance: New account balance

        Example:
            >>> limit.update_account_balance(51000)
        """
        old_balance = self._account_balance
        self._account_balance = new_balance

        logger.info(
            "Account balance updated: ${:.2f} → ${:.2f}".format(
                old_balance,
                new_balance
            )
        )

        # Log update
        self._log_audit_event(
            "UPDATE",
            entry_price=0,
            stop_loss_price=0,
            quantity=0,
            estimated_risk=0,
            is_valid=True,
            violation_amount=None
        )

    def set_max_risk_dollars(self, max_risk: float) -> None:
        """Update maximum risk limit.

        Args:
            max_risk: New maximum dollar risk per trade

        Raises:
            ValueError: If max_risk <= 0

        Example:
            >>> limit.set_max_risk_dollars(750)
        """
        if max_risk <= 0:
            raise ValueError(
                "Max risk dollars must be positive: {}".format(max_risk)
            )

        old_limit = self._max_risk_dollars
        self._max_risk_dollars = max_risk

        logger.info(
            "Max risk limit updated: ${:.2f} → ${:.2f}".format(
                old_limit,
                max_risk
            )
        )

        # Log update
        self._log_audit_event(
            "UPDATE",
            entry_price=0,
            stop_loss_price=0,
            quantity=0,
            estimated_risk=0,
            is_valid=True,
            violation_amount=None
        )

    def _log_audit_event(
        self,
        event_type: str,
        entry_price: float,
        stop_loss_price: float,
        quantity: int,
        estimated_risk: float,
        is_valid: bool,
        violation_amount: Optional[float]
    ) -> None:
        """Log event to CSV audit trail.

        Args:
            event_type: Type of event (VALIDATE, REJECT, UPDATE)
            entry_price: Entry price per contract
            stop_loss_price: Stop loss price per contract
            quantity: Number of contracts
            estimated_risk: Estimated dollar risk
            is_valid: Validation result
            violation_amount: Amount over limit
        """
        if self._audit_trail_path is None:
            return

        # Ensure audit trail directory exists
        audit_path = Path(self._audit_trail_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if file exists
        file_exists = (
            audit_path.exists() and
            audit_path.stat().st_size > 0
        )

        # Append to CSV
        with open(audit_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "entry_price",
                    "stop_loss_price",
                    "quantity",
                    "estimated_risk",
                    "max_risk_limit",
                    "is_valid",
                    "violation_amount"
                ])

            # Write event
            writer.writerow([
                timestamp,
                event_type,
                entry_price,
                stop_loss_price,
                quantity,
                estimated_risk,
                self._max_risk_dollars,
                is_valid,
                violation_amount or ""
            ])

        logger.debug("Per-trade risk audit logged: {}".format(event_type))
