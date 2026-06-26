"""Order-ID-scoped ProjectXClient helpers (commingling safety for two bots on one
account, same contract). Verifies is_order_open / cancel_orders reason about ONLY
the given IDs and never touch another bot's orders."""
import json

import pytest

from src.research.projectx_client import ProjectXClient


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _Auth:
    async def authenticate(self):
        return "tok"


class _Http:
    """Routes POSTs by URL. open_ids = orders currently open on the account
    (mix of both bots). Records cancel calls."""
    def __init__(self, open_ids):
        self.open_ids = list(open_ids)
        self.cancelled = []

    async def post(self, url, json=None, headers=None):
        if url.endswith("/Order/searchOpen"):
            return _Resp(200, {"orders": [{"id": i, "contractId": "CON.F.US.MNQ.U26"} for i in self.open_ids]})
        if url.endswith("/Order/cancel"):
            self.cancelled.append(json["orderId"])
            return _Resp(200, {"success": True})
        return _Resp(404, {})


class _Cfg:
    symbol = "MNQU26"


def _client(open_ids):
    http = _Http(open_ids)
    return ProjectXClient(_Auth(), _Cfg(), http, projectx_account_id=23884932), http


# account has MIM cat-stop #111 + YANK orders #222, #333 open simultaneously
@pytest.mark.asyncio
async def test_is_order_open_sees_only_the_queried_id():
    c, _ = _client([111, 222, 333])
    assert await c.is_order_open(111) is True     # MIM's own cat-stop still live
    assert await c.is_order_open(999) is False    # MIM's filled cat-stop -> gone
    assert await c.is_order_open(None) is False


@pytest.mark.asyncio
async def test_is_order_open_none_on_error():
    c, http = _client([111])
    async def boom(*a, **k):
        raise RuntimeError("network")
    http.post = boom
    assert await c.is_order_open(111) is None     # caller treats None as "assume holding"


@pytest.mark.asyncio
async def test_cancel_orders_scoped_to_own_ids():
    c, http = _client([111, 222, 333])
    out = await c.cancel_orders([111])            # MIM cancels ONLY its own
    assert out == ["111"]
    assert http.cancelled == [111]                # YANK's 222/333 untouched


@pytest.mark.asyncio
async def test_cancel_orders_skips_none():
    c, http = _client([111])
    await c.cancel_orders([None, 111])
    assert http.cancelled == [111]
