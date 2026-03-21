"""Futures contract symbol generator for MNQ (Micro E-mini Nasdaq-100).

This module generates quarterly futures contract symbols going back 8 quarters
(2 years) from the current contract, handling expiration rollovers and edge cases.
"""

import datetime
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# MNQ quarterly expiration month codes
# H=March, M=June, U=September, Z=December
QUARTERLY_CODES = ["H", "M", "U", "Z"]

# Days before expiration to switch to next contract
EXPIRATION_BUFFER_DAYS = 7

# Minimum days of historical data required for a contract
MIN_HISTORY_DAYS = 30


@dataclass
class ContractInfo:
    """Information about a futures contract."""

    symbol: str
    year: int
    month_code: str
    expiration_month: int
    display_name: str


class FuturesSymbolGenerator:
    """Generate MNQ futures contract symbols for historical data download.

    MNQ contracts expire quarterly in H (March), M (June), U (September), Z (December).
    This class generates symbols going back 8 quarters (2 years) from the current
    contract, handling edge cases like near-expiration and year rollovers.
    """

    def __init__(self) -> None:
        """Initialize symbol generator."""
        self._current_date = datetime.date.today()

    def generate_mnq_symbols(
        self, months_back: int = 24
    ) -> list[str]:
        """Generate MNQ symbols for historical data download.

        Returns 8 quarterly contract symbols in chronological order (oldest first).

        Args:
            months_back: Number of months to go back (default: 24 = 8 quarters)

        Returns:
            List of MNQ symbols in chronological order (oldest first)

        Example:
            If current date is March 15, 2026:
            Returns: ['MNQM24', 'MNQU24', 'MNQZ24', 'MNQH25',
                      'MNQM25', 'MNQU25', 'MNQZ25', 'MNQH26']
        """
        # Find current active contract
        current_contract = self._find_current_contract()

        # Generate contracts going backwards
        symbols = []
        contract = current_contract

        for _ in range(8):  # 8 quarters
            symbols.append(contract.symbol)
            contract = self._get_previous_contract(contract)

        # Reverse to get chronological order (oldest first)
        symbols.reverse()

        logger.info(
            f"Generated {len(symbols)} MNQ symbols from {symbols[0]} "
            f"to {symbols[-1]}"
        )

        return symbols

    def _find_current_contract(self) -> ContractInfo:
        """Find the currently active MNQ contract.

        The active contract is determined by:
        1. Finding the nearest expiration after today
        2. If within EXPIRATION_BUFFER_DAYS of expiration, use next contract

        Returns:
            ContractInfo for the active contract
        """
        today = self._current_date
        current_year = today.year
        current_month = today.month

        # Find current quarter's contract
        month_code = self._get_month_code_for_month(current_month)
        expiration_month = self._get_expiration_month(month_code)

        # Get contract info
        contract = ContractInfo(
            symbol=f"MNQ{month_code}{current_year % 100:02d}",
            year=current_year,
            month_code=month_code,
            expiration_month=expiration_month,
            display_name=f"MNQ {month_code}{current_year}",
        )

        # Check if we're near expiration (within buffer days)
        expiration_date = self._get_contract_expiration_date(contract)
        days_until_expiration = (expiration_date - today).days

        if days_until_expiration < EXPIRATION_BUFFER_DAYS:
            # Use next quarter's contract instead
            logger.info(
                f"Within {days_until_expiration} days of expiration, "
                f"using next contract"
            )
            contract = self._get_next_contract(contract)

        return contract

    def _get_month_code_for_month(self, month: int) -> str:
        """Get quarterly month code for a given calendar month.

        Args:
            month: Calendar month (1-12)

        Returns:
            Quarter month code (H, M, U, or Z)

        Example:
            February → 'H' (March contract)
            April → 'M' (June contract)
        """
        # Find which quarterly contract is active for this month
        for i, code in enumerate(QUARTERLY_CODES):
            expiration_month = self._get_expiration_month(code)
            if month <= expiration_month:
                return code

        # December is special case - wrap to next year's H contract
        return QUARTERLY_CODES[0]  # H (March)

    def _get_expiration_month(self, month_code: str) -> int:
        """Get calendar month for a quarterly contract code.

        Args:
            month_code: Quarterly code (H, M, U, Z)

        Returns:
            Calendar month number (1-12)

        Example:
            'H' → 3 (March)
            'M' → 6 (June)
        """
        mapping = {
            "H": 3,   # March
            "M": 6,   # June
            "U": 9,   # September
            "Z": 12,  # December
        }
        return mapping[month_code]

    def _get_contract_expiration_date(self, contract: ContractInfo) -> datetime.date:
        """Get approximate expiration date for a contract.

        MNQ contracts typically expire on the third Friday of the expiration month.
        This is an approximation - actual expiration may vary.

        Args:
            contract: Contract info

        Returns:
            Approximate expiration date
        """
        year = contract.year
        month = contract.expiration_month

        # Find third Friday of the month
        first_day = datetime.date(year, month, 1)
        first_friday = first_day

        # Find first Friday
        while first_friday.weekday() != 4:  # Friday = 4
            first_friday += datetime.timedelta(days=1)

        # Third Friday is 14 days after first Friday
        third_friday = first_friday + datetime.timedelta(days=14)

        return third_friday

    def _get_next_contract(self, contract: ContractInfo) -> ContractInfo:
        """Get the next quarterly contract after the given contract.

        Args:
            contract: Current contract

        Returns:
            Next quarter's contract info
        """
        current_index = QUARTERLY_CODES.index(contract.month_code)

        # Move to next quarter
        next_index = (current_index + 1) % len(QUARTERLY_CODES)
        next_code = QUARTERLY_CODES[next_index]

        # Handle year rollover
        if next_index == 0:  # Wrapped from Z to H
            next_year = contract.year + 1
        else:
            next_year = contract.year

        return ContractInfo(
            symbol=f"MNQ{next_code}{next_year % 100:02d}",
            year=next_year,
            month_code=next_code,
            expiration_month=self._get_expiration_month(next_code),
            display_name=f"MNQ {next_code}{next_year}",
        )

    def _get_previous_contract(self, contract: ContractInfo) -> ContractInfo:
        """Get the previous quarterly contract before the given contract.

        Args:
            contract: Current contract

        Returns:
            Previous quarter's contract info
        """
        current_index = QUARTERLY_CODES.index(contract.month_code)

        # Move to previous quarter
        prev_index = (current_index - 1) % len(QUARTERLY_CODES)
        prev_code = QUARTERLY_CODES[prev_index]

        # Handle year rollover (backwards)
        if current_index == 0:  # Wrapped from H to Z (previous year)
            prev_year = contract.year - 1
        else:
            prev_year = contract.year

        return ContractInfo(
            symbol=f"MNQ{prev_code}{prev_year % 100:02d}",
            year=prev_year,
            month_code=prev_code,
            expiration_month=self._get_expiration_month(prev_code),
            display_name=f"MNQ {prev_code}{prev_year}",
        )

    def _has_sufficient_history(
        self, symbol: str, min_days: int = MIN_HISTORY_DAYS
    ) -> bool:
        """Check if a contract has sufficient historical data.

        This is a heuristic check based on contract age.
        Actual data availability should be validated via API.

        Args:
            symbol: Contract symbol
            min_days: Minimum required days of history

        Returns:
            True if contract likely has sufficient data, False otherwise
        """
        # Parse symbol to get year and month code
        if len(symbol) < 6:
            return False

        month_code = symbol[3:4]
        year_suffix = symbol[4:6]

        try:
            year = 2000 + int(year_suffix)
        except ValueError:
            return False

        # Get contract expiration date
        contract = ContractInfo(
            symbol=symbol,
            year=year,
            month_code=month_code,
            expiration_month=self._get_expiration_month(month_code),
            display_name=f"MNQ {month_code}{year}",
        )

        # Calculate contract age (time since expiration)
        expiration = self._get_contract_expiration_date(contract)
        age_days = (self._current_date - expiration).days

        # Contract needs to be expired long enough to have history
        return age_days >= min_days

    def _get_quarterly_contracts(
        self, year: int, start_month: int = 1
    ) -> list[ContractInfo]:
        """Get all quarterly contracts for a year.

        Args:
            year: Year to get contracts for
            start_month: Starting month (default: 1 = January)

        Returns:
            List of quarterly contracts for the year
        """
        contracts = []

        for code in QUARTERLY_CODES:
            expiration_month = self._get_expiration_month(code)
            if expiration_month >= start_month:
                contracts.append(
                    ContractInfo(
                        symbol=f"MNQ{code}{year % 100:02d}",
                        year=year,
                        month_code=code,
                        expiration_month=expiration_month,
                        display_name=f"MNQ {code}{year}",
                    )
                )

        return contracts
