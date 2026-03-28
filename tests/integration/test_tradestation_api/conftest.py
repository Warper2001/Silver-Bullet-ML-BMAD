"""Shared fixtures for TradeStation API integration tests."""

import os
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from src.execution.tradestation.client import TradeStationClient


def get_sim_credentials() -> tuple[str, str]:
    """
    Get TradeStation SIM credentials from environment variables.

    Returns:
        Tuple of (client_id, client_secret)

    Raises:
        pytest.skip.Exception: If credentials are not configured
    """
    client_id = os.getenv("TRADESTATION_SIM_CLIENT_ID")
    client_secret = os.getenv("TRADESTATION_SIM_CLIENT_SECRET")

    if not client_id or not client_secret:
        pytest.skip(
            "TradeStation SIM credentials not configured. "
            "Set TRADESTATION_SIM_CLIENT_ID and TRADESTATION_SIM_CLIENT_SECRET environment variables."
        )

    return client_id, client_secret


@pytest.fixture(scope="session")
def sim_client_id() -> str:
    """Get SIM client ID from environment."""
    client_id, _ = get_sim_credentials()
    return client_id


@pytest.fixture(scope="session")
def sim_client_secret() -> str:
    """Get SIM client secret from environment."""
    _, client_secret = get_sim_credentials()
    return client_secret


@pytest.fixture
async def tradestation_client(sim_client_id: str, sim_client_secret: str) -> AsyncGenerator[TradeStationClient, None]:
    """
    Create an authenticated TradeStation client for SIM environment.

    This fixture manages the client lifecycle and ensures proper cleanup.

    Yields:
        Authenticated TradeStationClient instance

    Example:
        async def test_quotes(tradestation_client):
            from src.execution.tradestation.market_data.quotes import QuotesClient
            quotes_client = QuotesClient(tradestation_client)
            quotes = await quotes_client.get_quotes(["MNQH26"])
            assert len(quotes) > 0
    """
    async with TradeStationClient(
        env="sim",
        client_id=sim_client_id,
        client_secret=sim_client_secret,
    ) as client:
        yield client


@pytest.fixture
def test_symbols() -> list[str]:
    """
    Get list of symbols to use in integration tests.

    Returns:
        List of symbol strings (e.g., ["MNQH26", "ESM26"])

    Note:
        These symbols should be actively trading futures contracts.
        Update quarterly when contracts roll.
    """
    # Current actively-traded micro futures (as of March 2026)
    return [
        "MNQH26",  # Micro E-mini Nasdaq-100 March 2026
        "MESM26",  # Micro E-mini S&P 500 June 2026
    ]


@pytest.fixture
def test_symbol() -> str:
    """
    Get a single symbol for testing.

    Returns:
        A symbol string (e.g., "MNQH26")
    """
    return "MNQH26"


@pytest.fixture
def mock_client() -> TradeStationClient:
    """
    Create a mock TradeStationClient for testing without API calls.

    Returns:
        Mocked TradeStationClient instance

    Example:
        def test_without_api(mock_client):
            from src.execution.tradestation.market_data.quotes import QuotesClient
            quotes_client = QuotesClient(mock_client)
            assert quotes_client.client is not None
    """
    client = MagicMock()
    client.api_base_url = "https://sim-api.tradestation.com/v3"
    client._ensure_authenticated = MagicMock(return_value="mock_token")
    return client


@pytest.fixture
def integration_test_config() -> dict:
    """
    Get configuration for integration tests.

    Returns:
        Dictionary with test configuration parameters

    Example:
        {
            "timeout": 30.0,
            "retry_attempts": 3,
            "test_order_quantity": 1,
            "small_order_size": 1,
            "price_buffer": 0.5,  # Buffer away from market for limit orders
        }
    """
    return {
        "timeout": 30.0,  # API request timeout in seconds
        "retry_attempts": 3,  # Number of retry attempts for transient failures
        "test_order_quantity": 1,  # Quantity for test orders
        "small_order_size": 1,  # Small size for safety
        "price_buffer": 0.5,  # Price buffer for limit orders (ticks away from market)
        "max_wait_time": 10.0,  # Maximum wait time for order fills/cancellations
    }


# Skip integration tests if running in CI without credentials
def pytest_configure(config):
    """Configure pytest to handle integration tests appropriately."""
    # Add a custom marker for integration tests
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require credentials)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip integration tests if credentials not set."""
    has_credentials = bool(
        os.getenv("TRADESTATION_SIM_CLIENT_ID") and os.getenv("TRADESTATION_SIM_CLIENT_SECRET")
    )

    if not has_credentials:
        skip_integration = pytest.mark.skip(
            reason="TradeStation SIM credentials not configured. "
            "Set TRADESTATION_SIM_CLIENT_ID and TRADESTATION_SIM_CLIENT_SECRET to run integration tests."
        )

        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
