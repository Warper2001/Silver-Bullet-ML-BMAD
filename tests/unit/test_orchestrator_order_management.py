"""Unit tests for TradeStationClient order management and reconciliation (Story 4-1)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.research.tier2_streaming_working import (
    AccountConfig,
    ActiveTrade,
    TradeState,
    TradeStationClient,
    SIM_ORDERS_URL,
    SIM_ACCOUNT_ID,
)
from src.research.strategy_core import Direction, EntryDecision


def _make_account_config(symbol: str = "MNQM26") -> AccountConfig:
    return AccountConfig(
        account_id=SIM_ACCOUNT_ID,
        execution_mode="sim",
        symbol=symbol,
        point_value=2.0,
        tick_size=0.25,
        contracts=5,
    )


def _make_auth() -> MagicMock:
    auth = MagicMock()
    auth.authenticate = AsyncMock(return_value="test-token")
    return auth


def _make_client(auth=None, cfg=None, http=None) -> TradeStationClient:
    return TradeStationClient(
        auth=auth or _make_auth(),
        account_config=cfg or _make_account_config(),
        httpx_client=http or AsyncMock(spec=httpx.AsyncClient),
    )


def _make_entry_decision(direction=Direction.BEARISH) -> EntryDecision:
    return EntryDecision(
        direction=direction,
        entry_price=18000.0,
        sl_price=18050.0,
        tp_price=17940.0,
        contracts=5,
    )


class TestTradeStationClientReconciliation:
    @pytest.mark.asyncio
    async def test_reconcile_open_order_no_position_returns_pending(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        orders_resp = MagicMock()
        orders_resp.status_code = 200
        orders_resp.json.return_value = {
            "Orders": [{"Symbol": "MNQM26", "OrderID": "ORD123", "OrderType": "Limit"}]
        }
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {"Positions": []}
        http.get = AsyncMock(side_effect=[orders_resp, pos_resp])

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "PENDING"
        assert state.entry_order_id == "ORD123"
        assert state.position_qty == 0

    @pytest.mark.asyncio
    async def test_reconcile_filled_order_with_position_returns_active(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        orders_resp = MagicMock()
        orders_resp.status_code = 200
        orders_resp.json.return_value = {"Orders": []}
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {
            "Positions": [{"Symbol": "MNQM26", "Quantity": "5"}]
        }
        http.get = AsyncMock(side_effect=[orders_resp, pos_resp])

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "ACTIVE"
        assert state.position_qty == 5

    @pytest.mark.asyncio
    async def test_reconcile_no_order_no_position_returns_flat(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        orders_resp = MagicMock()
        orders_resp.status_code = 200
        orders_resp.json.return_value = {"Orders": []}
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {"Positions": []}
        http.get = AsyncMock(side_effect=[orders_resp, pos_resp])

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "FLAT"
        assert state.position_qty == 0

    @pytest.mark.asyncio
    async def test_reconcile_filters_other_symbol_orders(self):
        """Orders for a different symbol must not affect reconciliation result."""
        http = AsyncMock(spec=httpx.AsyncClient)
        orders_resp = MagicMock()
        orders_resp.status_code = 200
        orders_resp.json.return_value = {
            "Orders": [{"Symbol": "MESM26", "OrderID": "ORD999", "OrderType": "Limit"}]
        }
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {"Positions": []}
        http.get = AsyncMock(side_effect=[orders_resp, pos_resp])

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "FLAT"

    @pytest.mark.asyncio
    async def test_reconcile_short_position_returns_active(self):
        """Short positions have negative Quantity — must still return ACTIVE (P2 fix)."""
        http = AsyncMock(spec=httpx.AsyncClient)
        orders_resp = MagicMock()
        orders_resp.status_code = 200
        orders_resp.json.return_value = {"Orders": []}
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {
            "Positions": [{"Symbol": "MNQM26", "Quantity": "-5"}]
        }
        http.get = AsyncMock(side_effect=[orders_resp, pos_resp])

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "ACTIVE"
        assert state.position_qty == 5  # abs value stored

    @pytest.mark.asyncio
    async def test_reconcile_float_string_quantity_returns_active(self):
        """Quantity returned as float string e.g. '5.0' must not raise ValueError (P3 fix)."""
        http = AsyncMock(spec=httpx.AsyncClient)
        orders_resp = MagicMock()
        orders_resp.status_code = 200
        orders_resp.json.return_value = {"Orders": []}
        pos_resp = MagicMock()
        pos_resp.status_code = 200
        pos_resp.json.return_value = {
            "Positions": [{"Symbol": "MNQM26", "Quantity": "5.0"}]
        }
        http.get = AsyncMock(side_effect=[orders_resp, pos_resp])

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "ACTIVE"
        assert state.position_qty == 5

    @pytest.mark.asyncio
    async def test_reconcile_returns_flat_on_network_error(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        client = _make_client(http=http)
        state = await client.reconcile_state(SIM_ACCOUNT_ID)

        assert state.status == "FLAT"


class TestTradeStationClientOrderSubmission:
    @pytest.mark.asyncio
    async def test_submit_bracket_order_posts_to_sim_url(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {
            "Orders": [
                {"OrderID": "E1", "Message": "Limit Entry"},
                {"OrderID": "T1", "Message": "Limit"},
                {"OrderID": "S1", "Message": "Stop Market"},
            ]
        }
        http.post = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        decision = _make_entry_decision()
        await client.submit_bracket_order(decision, SIM_ACCOUNT_ID)

        call_args = http.post.call_args
        assert call_args[0][0] == SIM_ORDERS_URL

    @pytest.mark.asyncio
    async def test_submit_bracket_order_payload_has_oso_with_two_legs(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {"Orders": [{"OrderID": "E1"}]}
        http.post = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        decision = _make_entry_decision()
        await client.submit_bracket_order(decision, SIM_ACCOUNT_ID)

        payload = http.post.call_args[1]["json"]
        assert "OSOs" in payload
        assert len(payload["OSOs"]) == 1
        assert len(payload["OSOs"][0]["Orders"]) == 2

    @pytest.mark.asyncio
    async def test_submit_bracket_order_returns_order_ids_via_order_type(self):
        """Primary path: OrderType + TimeInForce.Duration classify entry vs TP correctly."""
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {
            "Orders": [
                {"OrderID": "E1", "OrderType": "Limit", "TimeInForce": {"Duration": "DAY"}, "Message": "Limit order submitted"},
                {"OrderID": "T1", "OrderType": "Limit", "TimeInForce": {"Duration": "GTC"}, "Message": "Limit order submitted"},
                {"OrderID": "S1", "OrderType": "StopMarket", "TimeInForce": {"Duration": "GTC"}, "Message": "Stop Market order submitted"},
            ]
        }
        http.post = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        decision = _make_entry_decision()
        e_id, tp_id, sl_id = await client.submit_bracket_order(decision, SIM_ACCOUNT_ID)

        assert e_id == "E1"
        assert tp_id == "T1"
        assert sl_id == "S1"

    @pytest.mark.asyncio
    async def test_submit_bracket_order_returns_order_ids_fallback_positional(self):
        """Fallback path: no OrderType in response → positional assignment."""
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 201
        resp.json.return_value = {
            "Orders": [
                {"OrderID": "E1", "Message": "Order accepted"},
                {"OrderID": "T1", "Message": "Order accepted"},
                {"OrderID": "S1", "Message": "Stop Market"},
            ]
        }
        http.post = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        decision = _make_entry_decision()
        e_id, tp_id, sl_id = await client.submit_bracket_order(decision, SIM_ACCOUNT_ID)

        assert e_id == "E1"
        assert tp_id == "T1"
        assert sl_id == "S1"

    @pytest.mark.asyncio
    async def test_submit_bracket_order_returns_none_on_http_500(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        http.post = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        decision = _make_entry_decision()
        e_id, tp_id, sl_id = await client.submit_bracket_order(decision, SIM_ACCOUNT_ID)

        assert e_id is None
        assert tp_id is None
        assert sl_id is None

    @pytest.mark.asyncio
    async def test_cancel_order_sends_delete_with_correct_url(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 200
        http.delete = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        result = await client.cancel_order("ORD-XYZ")

        assert result is True
        call_url = http.delete.call_args[0][0]
        assert "ORD-XYZ" in call_url

    @pytest.mark.asyncio
    async def test_cancel_order_returns_true_on_404(self):
        """404 means order already gone — treat as success."""
        http = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.status_code = 404
        http.delete = AsyncMock(return_value=resp)

        client = _make_client(http=http)
        result = await client.cancel_order("GONE-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_returns_false_on_timeout(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.delete = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        client = _make_client(http=http)
        result = await client.cancel_order("ORD-X")

        assert result is False


class TestSinglePositionEnforcement:
    @pytest.mark.asyncio
    async def test_detect_and_enter_skips_when_active_trade_present(self):
        """AC#3: _detect_and_enter() must not call submit_bracket_order when active_trade exists."""
        from src.research.tier2_streaming_working import Tier2StreamingTrader, DollarBar

        trader = Tier2StreamingTrader.__new__(Tier2StreamingTrader)
        trader.active_trade = MagicMock(spec=ActiveTrade)
        trader._ts_client = MagicMock(spec=TradeStationClient)
        trader._ts_client.submit_bracket_order = AsyncMock()

        bar = MagicMock(spec=DollarBar)
        bar.timestamp = datetime(2026, 1, 6, 15, 0, tzinfo=timezone.utc)

        await trader._detect_and_enter(bar, is_backfill=False)

        trader._ts_client.submit_bracket_order.assert_not_called()
