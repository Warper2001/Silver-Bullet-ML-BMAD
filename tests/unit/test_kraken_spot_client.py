"""Unit tests for KrakenSpotClient — mocked HTTP, no network required."""

import base64
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.kraken.exceptions import KrakenAPIError, KrakenAuthError, KrakenOrderError
from src.execution.kraken.spot.client import KrakenSpotClient
from src.execution.kraken.spot.models import SpotOrderResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_KEY    = "testKey"
_TEST_SECRET = base64.b64encode(b"test_secret_32_bytes_padded_here").decode()


def _env():
    return {"KRAKEN_SPOT_API_KEY": _TEST_KEY, "KRAKEN_SPOT_API_SECRET": _TEST_SECRET}


def _add_order_response(txid: str = "OXXXXX-YYYYY-ZZZZZ") -> dict:
    return {"error": [], "result": {"descr": {}, "txid": [txid]}}


def _query_orders_response(txid: str, status: str = "closed", price: str = "70000.00", vol_exec: str = "0.10000000") -> dict:
    return {
        "error": [],
        "result": {
            txid: {
                "status":   status,
                "price":    price,
                "vol_exec": vol_exec,
                "vol":      vol_exec,
            }
        },
    }


def _balance_response(zusd: str = "15000.00", xxbt: str = "0.50000000") -> dict:
    return {"error": [], "result": {"ZUSD": zusd, "XXBT": xxbt}}


async def _make_client() -> KrakenSpotClient:
    with patch.dict(os.environ, _env()):
        return KrakenSpotClient()


# ---------------------------------------------------------------------------
# place_market_order — success paths
# ---------------------------------------------------------------------------

class TestPlaceMarketOrder:
    @pytest.mark.asyncio
    async def test_buy_returns_spot_order_result(self):
        txid = "OBUY01-AAAAA-BBBBB"
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[
            _mock_resp(_add_order_response(txid)),
            _mock_resp(_query_orders_response(txid, "closed", "70500.00", "0.14200000")),
        ])
        client._http = mock_http

        result = await client.place_market_order("buy", 0.142)

        assert isinstance(result, SpotOrderResult)
        assert result.txid   == txid
        assert result.side   == "buy"
        assert result.fill_price == 70500.0
        assert result.vol_exec   == 0.142

    @pytest.mark.asyncio
    async def test_sell_returns_spot_order_result(self):
        txid = "OSELL1-CCCCC-DDDDD"
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[
            _mock_resp(_add_order_response(txid)),
            _mock_resp(_query_orders_response(txid, "closed", "69800.00", "0.14200000")),
        ])
        client._http = mock_http

        result = await client.place_market_order("sell", 0.142)

        assert result.side == "sell"
        assert result.fill_price == 69800.0


# ---------------------------------------------------------------------------
# _confirm_fill — retry logic
# ---------------------------------------------------------------------------

class TestConfirmFill:
    @pytest.mark.asyncio
    async def test_retries_once_on_open_status(self):
        txid   = "OOPEN1-EEEEE-FFFFF"
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[
            _mock_resp(_add_order_response(txid)),
            _mock_resp(_query_orders_response(txid, "open")),        # first poll: open
            _mock_resp(_query_orders_response(txid, "closed")),      # second poll: closed
        ])
        client._http = mock_http

        with patch("src.execution.kraken.spot.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.place_market_order("buy", 0.1)

        assert result.txid == txid

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self):
        txid   = "OSTUCK-GGGGG-HHHHH"
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[
            _mock_resp(_add_order_response(txid)),
            _mock_resp(_query_orders_response(txid, "open")),   # attempt 1
            _mock_resp(_query_orders_response(txid, "open")),   # attempt 2 — still open
        ])
        client._http = mock_http

        with patch("src.execution.kraken.spot.client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(KrakenOrderError, match="did not fill"):
                await client.place_market_order("buy", 0.1)


# ---------------------------------------------------------------------------
# Balance methods
# ---------------------------------------------------------------------------

class TestBalances:
    @pytest.mark.asyncio
    async def test_get_btc_balance_parses_xxbt(self):
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_mock_resp(_balance_response(xxbt="1.50000000")))
        client._http = mock_http

        balance = await client.get_btc_balance()
        assert balance == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_get_usd_balance_parses_zusd(self):
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_mock_resp(_balance_response(zusd="25000.00")))
        client._http = mock_http

        balance = await client.get_usd_balance()
        assert balance == pytest.approx(25000.0)

    @pytest.mark.asyncio
    async def test_missing_balance_key_returns_zero(self):
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_mock_resp({"error": [], "result": {}}))
        client._http = mock_http

        assert await client.get_btc_balance() == 0.0
        assert await client.get_usd_balance() == 0.0


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_add_order_non_empty_error_raises_order_error(self):
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_mock_resp(
            {"error": ["EOrder:Insufficient funds"], "result": {}}
        ))
        client._http = mock_http

        with pytest.raises(KrakenOrderError, match="Insufficient funds"):
            await client.place_market_order("buy", 0.1)

    @pytest.mark.asyncio
    async def test_balance_403_raises_auth_error(self):
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_mock_resp({}, status_code=403))
        client._http = mock_http

        with pytest.raises(KrakenAuthError):
            await client.get_usd_balance()

    @pytest.mark.asyncio
    async def test_non_200_raises_api_error(self):
        client = await _make_client()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_mock_resp({}, status_code=500))
        client._http = mock_http

        with pytest.raises(KrakenAPIError):
            await client.place_market_order("buy", 0.1)


# ---------------------------------------------------------------------------
# Helper — mock HTTP response
# ---------------------------------------------------------------------------

def _mock_resp(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text        = json.dumps(body)
    resp.json        = MagicMock(return_value=body)
    return resp
