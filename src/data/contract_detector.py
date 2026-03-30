"""Active futures contract detection for TradeStation API.

This module implements automatic detection of active futures contracts by
checking expiration dates and rolling over to the next quarterly contract.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# MNQ quarterly expiration month codes
# H=March, M=June, U=September, Z=December
QUARTERLY_CODES = ["H", "M", "U", "Z"]


class ContractDetector:
    """Detect active futures contracts from TradeStation API.

    This class checks if a given futures contract symbol is expired and
    automatically finds the next active contract by querying the TradeStation
    quote API.

    Example:
        detector = ContractDetector(access_token="...")
        active_symbol = await detector.detect_active_contract("MNQH26")
        # Returns: "MNQM26" (if MNQH26 is expired)
    """

    QUOTE_ENDPOINT = "https://api.tradestation.com/v3/data/quote"
    MAX_CONTRACT_CHECKS = 16  # Prevent infinite loops (4 quarters × 4 years)

    def __init__(self, access_token: str) -> None:
        """Initialize contract detector.

        Args:
            access_token: Valid TradeStation API access token
        """
        self._access_token = access_token
        self._client: Optional[httpx.AsyncClient] = None

        logger.debug("ContractDetector initialized")

    async def detect_active_contract(
        self, symbol: str = "MNQH26"
    ) -> str:
        """Detect the active futures contract for a given symbol.

        This method checks if the given contract is expired and automatically
        rolls forward to find the next active contract.

        Args:
            symbol: Futures contract symbol (e.g., "MNQH26")

        Returns:
            Active contract symbol (e.g., "MNQM26")

        Raises:
            ValueError: If active contract cannot be determined
            httpx.HTTPStatusError: If API request fails
        """
        current_symbol = symbol
        checks = 0

        while checks < self.MAX_CONTRACT_CHECKS:
            checks += 1
            logger.debug(f"Checking contract {current_symbol} (attempt {checks})")

            try:
                contract_info = await self.get_contract_info(current_symbol)

                # Check if contract is expired
                if self._is_contract_expired(contract_info):
                    logger.info(
                        f"Contract {current_symbol} is expired, "
                        f"rolling to next contract"
                    )
                    current_symbol = self._generate_next_contract_symbol(
                        current_symbol
                    )
                else:
                    # Found active contract
                    logger.info(
                        f"Active contract found: {current_symbol} "
                        f"(expiration: {contract_info.get('ExpirationDate', 'Unknown')})"
                    )
                    return current_symbol

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Contract not found, try next
                    logger.warning(
                        f"Contract {current_symbol} not found (404), "
                        f"rolling to next contract"
                    )
                    current_symbol = self._generate_next_contract_symbol(
                        current_symbol
                    )
                else:
                    raise

        # Max checks exceeded
        raise ValueError(
            f"Could not find active contract after {self.MAX_CONTRACT_CHECKS} checks. "
            f"Started with {symbol}, ended with {current_symbol}. "
            "Please verify symbol format and API access."
        )

    async def get_contract_info(self, symbol: str) -> dict:
        """Get contract information from TradeStation API.

        Args:
            symbol: Futures contract symbol (e.g., "MNQH26")

        Returns:
            Dictionary with contract information including:
            - Symbol: Contract symbol
            - ExpirationDate: Expiration date (YYYY-MM-DD format)
            - Last: Last price
            - Volume: Volume
            - Bid: Bid price
            - Ask: Ask price

        Raises:
            httpx.HTTPStatusError: If API request fails
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        url = f"{self.QUOTE_ENDPOINT}/{symbol}"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        response = await self._client.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        # TradeStation API returns data in a specific format
        # Extract first quote if array
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"No data returned for symbol {symbol}")
            quote = data[0]
        else:
            quote = data

        logger.debug(
            f"Quote data for {symbol}: "
            f"Last={quote.get('Last', 'N/A')}, "
            f"Expiration={quote.get('ExpirationDate', 'N/A')}"
        )

        return quote

    def _is_contract_expired(self, contract_info: dict) -> bool:
        """Check if contract is expired based on quote data.

        A contract is considered expired if:
        1. ExpirationDate is in the past, OR
        2. Last price is 0 (no trading activity), OR
        3. Volume is 0 (no liquidity)

        Args:
            contract_info: Contract info from get_contract_info()

        Returns:
            True if contract appears expired, False otherwise
        """
        # Check expiration date
        expiration_str = contract_info.get("ExpirationDate")
        if expiration_str:
            try:
                # Parse expiration date (format: "2026-03-20")
                expiration_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()

                # Add buffer: contract expires 1 day before expiration date
                if datetime.now().date() >= (expiration_date - timedelta(days=1)):
                    logger.debug(
                        f"Contract expired: {expiration_str} "
                        f"(today: {datetime.now().date()})"
                    )
                    return True
            except ValueError as e:
                logger.warning(f"Failed to parse expiration date '{expiration_str}': {e}")

        # Check for zero price (indicates expired or inactive contract)
        last_price = contract_info.get("Last", 0)
        if last_price == 0:
            logger.debug("Contract has zero last price (expired/inactive)")
            return True

        # Check for zero volume (may indicate expired contract)
        volume = contract_info.get("Volume", 0)
        if volume == 0:
            logger.debug("Contract has zero volume (likely expired)")
            return True

        # Contract appears active
        return False

    def _generate_next_contract_symbol(self, symbol: str) -> str:
        """Generate the next quarterly contract symbol.

        MNQ contracts follow the pattern: MNQ + quarter code + 2-digit year
        Quarter codes: H (March), M (June), U (September), Z (December)

        Args:
            symbol: Current contract symbol (e.g., "MNQH26")

        Returns:
            Next contract symbol (e.g., "MNQM26")

        Example:
            MNQH26 → MNQM26 → MNQU26 → MNQZ26 → MNQH27
        """
        if len(symbol) < 6:
            raise ValueError(f"Invalid symbol format: {symbol}")

        # Parse symbol
        root = symbol[:3]  # "MNQ"
        quarter_code = symbol[3:4]  # "H"
        year_suffix = symbol[4:6]  # "26"

        # Validate quarter code
        if quarter_code not in QUARTERLY_CODES:
            raise ValueError(f"Invalid quarter code: {quarter_code}")

        # Get current quarter index
        current_index = QUARTERLY_CODES.index(quarter_code)

        # Move to next quarter
        next_index = (current_index + 1) % len(QUARTERLY_CODES)
        next_code = QUARTERLY_CODES[next_index]

        # Handle year rollover (Z → H)
        if next_index == 0:
            # Rollover to next year
            try:
                year = int(year_suffix) + 1
                year_suffix = f"{year % 100:02d}"
            except ValueError:
                raise ValueError(f"Invalid year suffix: {year_suffix}")

        next_symbol = f"{root}{next_code}{year_suffix}"
        logger.debug(f"Generated next contract: {symbol} → {next_symbol}")

        return next_symbol

    async def cleanup(self) -> None:
        """Clean up resources (close HTTP client)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        logger.debug("ContractDetector resources cleaned up")
