"""Unit tests for MIM-NB auto-roll contract resolution (src/research/mim_nb_live.py).

Covers the 2026-06-16 incident fix: the bot must always trade the broker's
active front-month contract rather than a hardcoded month that silently
rejects orders once it rolls off.
"""
import pytest

from src.research.mim_nb_live import resolve_front_month, _contract_id_to_symbol
from src.research.projectx_client import _to_contract_id


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _Http:
    """Minimal async stub mimicking httpx.AsyncClient.post."""
    def __init__(self, resp=None, exc=None):
        self._resp, self._exc = resp, exc
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append((url, json))
        if self._exc:
            raise self._exc
        return self._resp


class _Auth:
    async def authenticate(self):
        return "tok"


def test_contract_id_symbol_roundtrip():
    assert _contract_id_to_symbol("CON.F.US.MNQ.U26") == "MNQU26"
    assert _to_contract_id("MNQU26") == "CON.F.US.MNQ.U26"
    # every quarterly code round-trips
    for sym in ("MNQH26", "MNQM26", "MNQU26", "MNQZ26", "MNQH27"):
        assert _contract_id_to_symbol(_to_contract_id(sym)) == sym


@pytest.mark.asyncio
async def test_picks_active_contract():
    http = _Http(_Resp(200, {"contracts": [
        {"id": "CON.F.US.MNQ.M26", "activeContract": False},
        {"id": "CON.F.US.MNQ.U26", "activeContract": True},
        {"id": "CON.F.US.MNQ.Z26", "activeContract": False},
    ]}))
    assert await resolve_front_month(http, _Auth()) == "MNQU26"


@pytest.mark.asyncio
async def test_ignores_non_root_matches():
    # a fuzzy search could surface look-alike roots; only exact MNQ counts
    http = _Http(_Resp(200, {"contracts": [
        {"id": "CON.F.US.MNQX.U26", "activeContract": True},   # wrong root
        {"id": "CON.F.US.MNQ.Z26", "activeContract": True},    # correct root
    ]}))
    assert await resolve_front_month(http, _Auth()) == "MNQZ26"


@pytest.mark.asyncio
async def test_falls_back_on_http_error(monkeypatch):
    _stub_date_fallback(monkeypatch, "MNQH27")
    http = _Http(_Resp(500, {}))
    assert await resolve_front_month(http, _Auth()) == "MNQH27"


@pytest.mark.asyncio
async def test_falls_back_on_empty(monkeypatch):
    _stub_date_fallback(monkeypatch, "MNQM26")
    http = _Http(_Resp(200, {"contracts": []}))
    assert await resolve_front_month(http, _Auth()) == "MNQM26"


@pytest.mark.asyncio
async def test_falls_back_on_exception(monkeypatch):
    _stub_date_fallback(monkeypatch, "MNQU26")
    http = _Http(exc=RuntimeError("network down"))
    assert await resolve_front_month(http, _Auth()) == "MNQU26"


def _stub_date_fallback(monkeypatch, symbol):
    import src.data.futures_symbols as fs

    class _FakeGen:
        def _find_current_contract(self):
            return type("C", (), {"symbol": symbol})()

    monkeypatch.setattr(fs, "FuturesSymbolGenerator", _FakeGen)
