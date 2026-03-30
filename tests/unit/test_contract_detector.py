"""Unit tests for futures contract detection."""

import unittest
from unittest.mock import AsyncMock

import pytest
import httpx

from src.data.contract_detector import ContractDetector


class TestContractDetector:
    """Test ContractDetector class."""

    def test_initialization(self) -> None:
        """Test ContractDetector initialization."""
        detector = ContractDetector(access_token="test_token")

        assert detector._access_token == "test_token"
        assert detector._client is None

    def test_generate_next_contract_symbol_same_year(self) -> None:
        """Test generating next contract symbol within same year."""
        detector = ContractDetector(access_token="test_token")

        # H -> M -> U -> Z
        assert detector._generate_next_contract_symbol("MNQH26") == "MNQM26"
        assert detector._generate_next_contract_symbol("MNQM26") == "MNQU26"
        assert detector._generate_next_contract_symbol("MNQU26") == "MNQZ26"

    def test_generate_next_contract_symbol_year_rollover(self) -> None:
        """Test generating next contract symbol with year rollover."""
        detector = ContractDetector(access_token="test_token")

        # Z -> H (next year)
        assert detector._generate_next_contract_symbol("MNQZ26") == "MNQH27"

    def test_generate_next_contract_symbol_multiple_years(self) -> None:
        """Test generating multiple contract symbols across years."""
        detector = ContractDetector(access_token="test_token")

        symbol = "MNQH26"
        symbols = [symbol]

        # Generate 8 contracts (2 years)
        for _ in range(8):
            symbol = detector._generate_next_contract_symbol(symbol)
            symbols.append(symbol)

        expected = [
            "MNQH26",
            "MNQM26",
            "MNQU26",
            "MNQZ26",
            "MNQH27",
            "MNQM27",
            "MNQU27",
            "MNQZ27",
            "MNQH28",
        ]
        assert symbols == expected

    def test_generate_next_contract_symbol_invalid_format(self) -> None:
        """Test generating next contract with invalid symbol format."""
        detector = ContractDetector(access_token="test_token")

        with pytest.raises(ValueError) as exc_info:
            detector._generate_next_contract_symbol("MNQ")  # Too short

        assert "Invalid symbol format" in str(exc_info.value)

    def test_generate_next_contract_symbol_invalid_quarter(self) -> None:
        """Test generating next contract with invalid quarter code."""
        detector = ContractDetector(access_token="test_token")

        with pytest.raises(ValueError) as exc_info:
            detector._generate_next_contract_symbol("MNQX26")  # X is invalid

        assert "Invalid quarter code" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_contract_info_success(self) -> None:
        """Test getting contract info from API."""
        detector = ContractDetector(access_token="test_token")

        mock_quote_data = {
            "Symbol": "MNQH26",
            "Last": 11800.0,
            "Bid": 11799.0,
            "Ask": 11801.0,
            "Volume": 1000,
            "ExpirationDate": "2026-03-20",
        }

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.json.return_value = mock_quote_data
            mock_response.raise_for_status = AsyncMock()
            mock_client.get.return_value = mock_response

            contract_info = await detector.get_contract_info("MNQH26")

            assert contract_info["Symbol"] == "MNQH26"
            assert contract_info["Last"] == 11800.0
            assert contract_info["ExpirationDate"] == "2026-03-20"

            # Verify API call was made correctly
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert "MNQH26" in call_args[0][0]
            assert call_args[1]["headers"]["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_get_contract_info_array_response(self) -> None:
        """Test getting contract info when API returns array."""
        detector = ContractDetector(access_token="test_token")

        mock_quote_array = [
            {
                "Symbol": "MNQH26",
                "Last": 11800.0,
                "Bid": 11799.0,
                "Ask": 11801.0,
                "Volume": 1000,
                "ExpirationDate": "2026-03-20",
            }
        ]

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.json.return_value = mock_quote_array
            mock_response.raise_for_status = AsyncMock()
            mock_client.get.return_value = mock_response

            contract_info = await detector.get_contract_info("MNQH26")

            assert contract_info["Last"] == 11800.0

    @pytest.mark.asyncio
    async def test_get_contract_info_empty_array(self) -> None:
        """Test getting contract info when API returns empty array."""
        detector = ContractDetector(access_token="test_token")

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.json.return_value = []
            mock_response.raise_for_status = AsyncMock()
            mock_client.get.return_value = mock_response

            with pytest.raises(ValueError) as exc_info:
                await detector.get_contract_info("MNQH26")

            assert "No data returned" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_is_contract_expired_by_date(self) -> None:
        """Test contract expiration detection by date."""
        detector = ContractDetector(access_token="test_token")

        # Expired contract (past expiration date)
        expired_contract = {
            "Symbol": "MNQH26",
            "Last": 11800.0,
            "Volume": 1000,
            "ExpirationDate": "2020-03-20",  # Past date
        }

        assert detector._is_contract_expired(expired_contract) is True

        # Active contract (future expiration date)
        active_contract = {
            "Symbol": "MNQH26",
            "Last": 11800.0,
            "Volume": 1000,
            "ExpirationDate": "2027-03-20",  # Future date
        }

        assert detector._is_contract_expired(active_contract) is False

    @pytest.mark.asyncio
    async def test_is_contract_expired_by_zero_price(self) -> None:
        """Test contract expiration detection by zero price."""
        detector = ContractDetector(access_token="test_token")

        # Zero price indicates expired/inactive contract
        zero_price_contract = {
            "Symbol": "MNQH26",
            "Last": 0.0,
            "Volume": 1000,
            "ExpirationDate": "2026-03-20",
        }

        assert detector._is_contract_expired(zero_price_contract) is True

    @pytest.mark.asyncio
    async def test_is_contract_expired_by_zero_volume(self) -> None:
        """Test contract expiration detection by zero volume."""
        detector = ContractDetector(access_token="test_token")

        # Zero volume with valid price may still indicate expired contract
        zero_volume_contract = {
            "Symbol": "MNQH26",
            "Last": 11800.0,
            "Volume": 0,
            "ExpirationDate": "2026-03-20",
        }

        assert detector._is_contract_expired(zero_volume_contract) is True

    @pytest.mark.asyncio
    async def test_is_contract_not_expired(self) -> None:
        """Test active contract detection."""
        detector = ContractDetector(access_token="test_token")

        # Active contract
        active_contract = {
            "Symbol": "MNQH26",
            "Last": 11800.0,
            "Volume": 1000,
            "ExpirationDate": "2027-03-20",
        }

        assert detector._is_contract_expired(active_contract) is False

    @pytest.mark.asyncio
    async def test_detect_active_contract_already_active(self) -> None:
        """Test detecting active contract when current is already active."""
        detector = ContractDetector(access_token="test_token")

        mock_quote_data = {
            "Symbol": "MNQH26",
            "Last": 11800.0,
            "Volume": 1000,
            "ExpirationDate": "2027-03-20",
        }

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.json.return_value = mock_quote_data
            mock_response.raise_for_status = AsyncMock()
            mock_client.get.return_value = mock_response

            active_symbol = await detector.detect_active_contract("MNQH26")

            assert active_symbol == "MNQH26"
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_active_contract_rolls_forward(self) -> None:
        """Test detecting active contract when current is expired."""
        detector = ContractDetector(access_token="test_token")

        # First call returns expired contract
        expired_quote = {
            "Symbol": "MNQH26",
            "Last": 0.0,
            "Volume": 0,
            "ExpirationDate": "2020-03-20",
        }

        # Second call returns active contract
        active_quote = {
            "Symbol": "MNQM26",
            "Last": 11800.0,
            "Volume": 1000,
            "ExpirationDate": "2027-06-20",
        }

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response1 = AsyncMock()
            mock_response1.json.return_value = expired_quote
            mock_response1.raise_for_status = AsyncMock()

            mock_response2 = AsyncMock()
            mock_response2.json.return_value = active_quote
            mock_response2.raise_for_status = AsyncMock()

            mock_client.get.side_effect = [mock_response1, mock_response2]

            active_symbol = await detector.detect_active_contract("MNQH26")

            assert active_symbol == "MNQM26"
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_detect_active_contract_404_error(self) -> None:
        """Test detecting active contract when API returns 404."""
        detector = ContractDetector(access_token="test_token")

        # First call returns 404 (contract not found)
        mock_error_response = AsyncMock()
        mock_error_response.status_code = 404

        mock_error = httpx.HTTPStatusError(
            "Not found", request=AsyncMock(), response=mock_error_response
        )
        mock_error_response.raise_for_status.side_effect = mock_error

        # Second call returns active contract
        active_quote = {
            "Symbol": "MNQM26",
            "Last": 11800.0,
            "Volume": 1000,
            "ExpirationDate": "2027-06-20",
        }

        mock_success_response = AsyncMock()
        mock_success_response.json.return_value = active_quote
        mock_success_response.raise_for_status = AsyncMock()

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_client.get.side_effect = [mock_error_response, mock_success_response]

            active_symbol = await detector.detect_active_contract("MNQH26")

            assert active_symbol == "MNQM26"
            assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_detect_active_contract_max_checks_exceeded(self) -> None:
        """Test detecting active contract when max checks exceeded."""
        detector = ContractDetector(access_token="test_token")

        # All calls return expired contracts
        expired_quote = {
            "Symbol": "MNQH26",
            "Last": 0.0,
            "Volume": 0,
            "ExpirationDate": "2020-03-20",
        }

        with unittest.mock.patch.object(httpx, "AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_response = AsyncMock()
            mock_response.json.return_value = expired_quote
            mock_response.raise_for_status = AsyncMock()
            mock_client.get.return_value = mock_response

            with pytest.raises(ValueError) as exc_info:
                await detector.detect_active_contract("MNQH26")

            assert "Could not find active contract" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup closes HTTP client."""
        detector = ContractDetector(access_token="test_token")

        # Create mock client
        mock_client = AsyncMock()
        detector._client = mock_client

        await detector.cleanup()

        mock_client.aclose.assert_called_once()
        assert detector._client is None
