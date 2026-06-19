"""Unit tests for the TradeStation SIM order mirror (src/research/ts_sim_mirror.py).

Focus: the mirror is a STRICT subordinate of the authoritative ProjectX combine
path. The contract under test:

  1. order-placement returns the ProjectX result verbatim (mirror result discarded)
  2. a ProjectX exception propagates; the mirror is never invoked on failure
  3. mirror failures (HTTP error / network exception) never raise into the caller
  4. read methods are NOT overridden — they stay ProjectX-authoritative
  5. ProjectX payloads translate correctly to TS SIM single-leg orders
  6. mirrored cancels follow the px->ts id map
  7. the queue is bounded and drops oldest on overflow (never blocks the caller)
"""

import asyncio

import pytest

from src.research.projectx_client import ProjectXClient
from src.research.ts_sim_mirror import (
    MirrorProjectXClient,
    SimScaler,
    TSSimMirror,
    _contract_to_ts_symbol,
    px_payload_to_ts_order,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Cfg:
    symbol = "MNQU26"


class FakeMirror:
    """Records submit/cancel calls; the mirror seam used by MirrorProjectXClient."""

    def __init__(self):
        self.submits = []
        self.cancels = []

    def submit(self, px_order_id, payload):
        self.submits.append((px_order_id, payload))

    def cancel(self, px_order_id):
        self.cancels.append(px_order_id)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeHTTP:
    """Minimal async httpx stand-in: scripted responses + call recording."""

    def __init__(self, post_response=None, delete_response=None,
                 post_exc=None, delete_exc=None):
        self.post_calls = []
        self.delete_calls = []
        self._post_response = post_response
        self._delete_response = delete_response
        self._post_exc = post_exc
        self._delete_exc = delete_exc

    async def post(self, url, headers=None, json=None):
        self.post_calls.append((url, json))
        if self._post_exc:
            raise self._post_exc
        return self._post_response

    async def delete(self, url, headers=None):
        self.delete_calls.append(url)
        if self._delete_exc:
            raise self._delete_exc
        return self._delete_response

    async def get(self, url, headers=None):
        # Equity-poll endpoint; default to "no data" so the scaler keeps test-set equity.
        return FakeResponse(404)


class FakeAuth:
    async def authenticate(self):
        return "fake-token"


def _mirror_client(monkeypatch, mirror, place_return=None, place_exc=None,
                   cancel_return=True):
    """Build a MirrorProjectXClient whose ProjectX base methods are stubbed."""

    async def fake_place(self, payload):
        if place_exc:
            raise place_exc
        return place_return

    async def fake_cancel(self, order_id):
        return cancel_return

    monkeypatch.setattr(ProjectXClient, "_place_order", fake_place)
    monkeypatch.setattr(ProjectXClient, "cancel_order", fake_cancel)
    return MirrorProjectXClient(FakeAuth(), _Cfg(), FakeHTTP(),
                                projectx_account_id=999, ts_mirror=mirror)


# --------------------------------------------------------------------------- #
# 5. Payload translation
# --------------------------------------------------------------------------- #
def test_contract_to_ts_symbol():
    assert _contract_to_ts_symbol("CON.F.US.MNQ.U26") == "MNQU26"
    assert _contract_to_ts_symbol("CON.F.US.MES.Z25") == "MESZ25"


def test_translate_limit_order():
    ts = px_payload_to_ts_order(
        {"accountId": 1, "contractId": "CON.F.US.MNQ.U26", "type": 1,
         "side": 1, "size": 2, "limitPrice": 21000.5}, "SIM1")
    assert ts["OrderType"] == "Limit"
    assert ts["TradeAction"] == "SELL"
    assert ts["Symbol"] == "MNQU26"
    assert ts["Quantity"] == "2"
    assert ts["LimitPrice"] == "21000.5"
    assert ts["AccountID"] == "SIM1"
    assert ts["TimeInForce"] == {"Duration": "GTC"}


def test_translate_market_order():
    ts = px_payload_to_ts_order(
        {"contractId": "CON.F.US.MNQ.U26", "type": 2, "side": 0, "size": 1}, "SIM1")
    assert ts["OrderType"] == "Market"
    assert ts["TradeAction"] == "BUY"
    assert "LimitPrice" not in ts and "StopPrice" not in ts
    assert ts["TimeInForce"] == {"Duration": "DAY"}


def test_translate_stop_order():
    ts = px_payload_to_ts_order(
        {"contractId": "CON.F.US.MNQ.U26", "type": 4, "side": 0,
         "size": 1, "stopPrice": 20500.0}, "SIM1")
    assert ts["OrderType"] == "StopMarket"
    assert ts["StopPrice"] == "20500.0"


def test_translate_unknown_type_returns_none():
    assert px_payload_to_ts_order(
        {"contractId": "CON.F.US.MNQ.U26", "type": 99, "side": 0, "size": 1}, "SIM1") is None


def test_translate_qty_override():
    ts = px_payload_to_ts_order(
        {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 0, "size": 2,
         "limitPrice": 100.0}, "SIM1", qty_override=7)
    assert ts["Quantity"] == "7"  # SIM-scaled size, not the combine's size=2


# --------------------------------------------------------------------------- #
# SimScaler — fractional-Kelly sizing, $5K unlock latch, symmetric, margin cap
# --------------------------------------------------------------------------- #
def _scaler(**kw):
    defaults = dict(strategy="T", base_contracts=1, unlock_profit=5000.0,
                    kelly_fraction=0.5, edge_mean=30.0, edge_std=300.0,
                    margin_per_contract=200.0, margin_buffer=0.8, max_contracts=50,
                    equity_fraction=1.0, ttl_seconds=0.0)
    defaults.update(kw)
    return SimScaler(**defaults)


def test_scaler_base_size_before_any_equity():
    assert _scaler().target_contracts() == 1


def test_scaler_base_size_before_unlock():
    s = _scaler()
    s.update_equity(10_000)            # start_equity = 10k, profit 0 < 5k
    assert s.unlocked is False
    assert s.target_contracts() == 1   # building the cushion at base size


def test_scaler_unlocks_at_5k_and_scales():
    s = _scaler()
    s.update_equity(10_000)            # start = 10k
    s.update_equity(15_000)            # +5k -> unlock
    assert s.unlocked is True
    # contracts = 0.5 * 30 * 15000 / 300^2 = 225000 / 90000 = 2.5 -> floor 2
    assert s.target_contracts() == 2


def test_scaler_unlock_latches_through_drawdown():
    s = _scaler()
    s.update_equity(10_000)
    s.update_equity(16_000)            # unlock
    s.update_equity(11_000)            # drawdown back below start+5k
    assert s.unlocked is True          # latch holds (rolling high-water)
    assert s.hwm == 16_000             # high-water mark retained


def test_scaler_symmetric_scale_down_on_drawdown():
    s = _scaler()
    s.update_equity(10_000)
    s.update_equity(40_000)
    big = s.target_contracts()         # 0.5*30*40000/90000 = 6.67 -> 6
    s.update_equity(20_000)
    small = s.target_contracts()       # 0.5*30*20000/90000 = 3.33 -> 3
    assert big == 6 and small == 3 and small < big


def test_scaler_margin_cap_binds():
    s = _scaler(max_contracts=999)
    s.update_equity(10_000)
    s.update_equity(100_000, buying_power=1_000)  # cap = 0.8*1000/200 = 4
    assert s.target_contracts() == 4


def test_scaler_max_contracts_ceiling():
    s = _scaler(max_contracts=3)
    s.update_equity(10_000)
    s.update_equity(200_000)
    assert s.target_contracts() == 3


def test_scaler_ttl_caches_within_window():
    s = _scaler(ttl_seconds=600.0)
    s.update_equity(10_000)
    s.update_equity(15_000)
    first = s.target_contracts()
    # mutate equity WITHOUT update_equity (which would bust the cache) -> stays cached
    s.equity = 40_000
    assert s.target_contracts() == first


def test_scaler_persistence_round_trip(tmp_path):
    p = tmp_path / "scaler.json"
    s1 = _scaler(state_path=p)
    s1.update_equity(10_000)
    s1.update_equity(16_000)           # unlock + hwm
    s2 = _scaler(state_path=p)
    assert s2.start_equity == 10_000
    assert s2.hwm == 16_000
    assert s2.unlocked is True


@pytest.mark.asyncio
async def test_mirror_applies_scaler_quantity():
    http = FakeHTTP(post_response=FakeResponse(200, {"Orders": [{"OrderID": "TS-9"}]}))
    s = _scaler()
    s.update_equity(10_000)
    s.update_equity(15_000)            # -> 2 contracts
    m = TSSimMirror(FakeAuth(), sim_account_id="SIM1", http=http, scaler=s)
    await m.start()
    try:
        m.submit(20, {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 1,
                      "size": 1, "limitPrice": 100.0})  # combine size 1
        await m._queue.join()
        assert http.post_calls[0][1]["Quantity"] == "2"  # SIM scaled to 2
    finally:
        await m.stop()


# --------------------------------------------------------------------------- #
# 1-3. MirrorProjectXClient subordination
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_place_returns_primary_result_and_mirrors(monkeypatch):
    mirror = FakeMirror()
    client = _mirror_client(monkeypatch, mirror, place_return=555)
    payload = {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 1,
               "size": 2, "limitPrice": 100.0}

    result = await client._place_order(payload)

    assert result == 555  # ProjectX result returned verbatim
    assert mirror.submits == [(555, payload)]


@pytest.mark.asyncio
async def test_place_failure_not_mirrored(monkeypatch):
    mirror = FakeMirror()
    client = _mirror_client(monkeypatch, mirror, place_return=None)  # ProjectX rejected
    result = await client._place_order({"type": 1})
    assert result is None
    assert mirror.submits == []  # nothing to mirror when ProjectX places nothing


@pytest.mark.asyncio
async def test_place_exception_propagates_and_not_mirrored(monkeypatch):
    mirror = FakeMirror()
    client = _mirror_client(monkeypatch, mirror, place_exc=RuntimeError("px down"))
    with pytest.raises(RuntimeError):
        await client._place_order({"type": 1})
    assert mirror.submits == []


@pytest.mark.asyncio
async def test_cancel_returns_primary_result_and_mirrors(monkeypatch):
    mirror = FakeMirror()
    client = _mirror_client(monkeypatch, mirror, cancel_return=True)
    ok = await client.cancel_order("777")
    assert ok is True
    assert mirror.cancels == ["777"]


@pytest.mark.asyncio
async def test_mirror_submit_exception_never_reaches_caller(monkeypatch):
    class ExplodingMirror(FakeMirror):
        def submit(self, *a, **k):
            raise RuntimeError("mirror queue exploded")

    client = _mirror_client(monkeypatch, ExplodingMirror(), place_return=42)
    # Even if the mirror seam raises, the authoritative result must come back clean.
    assert await client._place_order({"type": 1}) == 42


# --------------------------------------------------------------------------- #
# 4. Reads are not overridden — they stay ProjectX-authoritative
# --------------------------------------------------------------------------- #
def test_read_methods_not_overridden():
    for name in ("is_order_open", "reconcile_state", "net_position",
                 "account_balance", "cancel_orders", "submit_bracket_order",
                 "place_exit_orders", "close_position_at_market"):
        assert getattr(MirrorProjectXClient, name) is getattr(ProjectXClient, name), (
            f"{name} must be inherited unchanged, not overridden")
    # Only the two universal write seams are overridden.
    assert MirrorProjectXClient._place_order is not ProjectXClient._place_order
    assert MirrorProjectXClient.cancel_order is not ProjectXClient.cancel_order


# --------------------------------------------------------------------------- #
# 6. TSSimMirror worker: place maps id, cancel follows the map
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_worker_place_then_cancel_follows_idmap():
    http = FakeHTTP(
        post_response=FakeResponse(200, {"Orders": [{"OrderID": "TS-1"}]}),
        delete_response=FakeResponse(200),
    )
    m = TSSimMirror(FakeAuth(), sim_account_id="SIM1", http=http)
    await m.start()
    try:
        m.submit(10, {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 1,
                      "size": 1, "limitPrice": 100.0})
        await m._queue.join()
        assert m._idmap["10"] == "TS-1"
        assert http.post_calls and http.post_calls[0][1]["Symbol"] == "MNQU26"

        m.cancel(10)
        await m._queue.join()
        assert http.delete_calls == ["https://sim-api.tradestation.com/v3/orderexecution/orders/TS-1"]
        assert "10" not in m._idmap  # mapping consumed on cancel
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_worker_http_error_does_not_map_or_raise():
    http = FakeHTTP(post_response=FakeResponse(400, text="bad request"))
    m = TSSimMirror(FakeAuth(), sim_account_id="SIM1", http=http)
    await m.start()
    try:
        m.submit(11, {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 1,
                      "size": 1, "limitPrice": 100.0})
        await m._queue.join()
        assert "11" not in m._idmap  # HTTP error -> no mapping, no crash
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_worker_network_exception_is_firewalled():
    http = FakeHTTP(post_exc=ConnectionError("TS unreachable"))
    m = TSSimMirror(FakeAuth(), sim_account_id="SIM1", http=http)
    await m.start()
    try:
        m.submit(12, {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 1,
                      "size": 1, "limitPrice": 100.0})
        await m._queue.join()  # worker survives the exception
        # A subsequent good item still processes -> worker task is alive.
        http._post_exc = None
        http._post_response = FakeResponse(200, {"Orders": [{"OrderID": "TS-2"}]})
        m.submit(13, {"contractId": "CON.F.US.MNQ.U26", "type": 1, "side": 1,
                      "size": 1, "limitPrice": 100.0})
        await m._queue.join()
        assert m._idmap.get("13") == "TS-2"
    finally:
        await m.stop()


# --------------------------------------------------------------------------- #
# 7. Bounded queue drops oldest — the producer never blocks
# --------------------------------------------------------------------------- #
def test_offer_drops_oldest_on_overflow():
    # Worker not started, so the queue fills; maxsize=1 forces overflow.
    m = TSSimMirror(FakeAuth(), sim_account_id="SIM1", http=FakeHTTP(), maxsize=1)
    m.submit(1, {"type": 1})
    m.submit(2, {"type": 1})
    m.submit(3, {"type": 1})
    assert m._queue.qsize() == 1   # bounded
    assert m.dropped == 2          # two oldest dropped, producer never blocked
    # The survivor is the most-recent item.
    action, px_oid, _ = m._queue.get_nowait()
    assert (action, px_oid) == ("place", "3")
