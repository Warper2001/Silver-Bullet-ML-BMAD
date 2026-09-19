"""A prev_close restored from state must belong to the ACTIVE contract.

Residual left open by the 2026-09-15 roll-contract fix: that fix guards the bar path, but
a restart BETWEEN sessions across a roll never touches it. `initialize()` resolves the new
front month, `_maybe_roll()` therefore sees no change and never re-derives, and state.json
carried no contract — so the retired contract's close came back as if it were the new
one's, and the next session's bands were built across two contracts (~293 pts apart at the
U26→Z26 roll).
"""
import json

import pytest

from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive

SIGMA = {"09:31": [0.001] * 14}


def _bot(symbol="MNQZ26"):
    o = object.__new__(MimNbLive)
    o.symbol = symbol
    o.sigma_hist, o.sigma_days, o.prev_close = {}, [], None
    o.day, o.position, o.entry_px, o.entry_t = "2026-09-18", 0, None, None
    o.cat_stop_id, o.day_pnl = None, 0.0
    return o


def _state(prev_close=29254.0, symbol="MNQZ26"):
    st = {"sigma_hist": SIGMA, "sigma_days": ["2026-09-17"], "prev_close": prev_close}
    if symbol is not None:
        st["symbol"] = symbol
    return st


@pytest.mark.asyncio
async def test_same_contract_restores_prev_close():
    o = _bot("MNQZ26")
    o._load_persisted_position = lambda: _state(symbol="MNQZ26")
    await MimNbLive._backfill(o)
    assert o.prev_close == 29254.0


@pytest.mark.asyncio
async def test_retired_contract_prev_close_is_discarded(caplog):
    """The 2026-09-15 shape: state saved under U26, bot now on Z26."""
    o = _bot("MNQZ26")
    o._load_persisted_position = lambda: _state(prev_close=29151.5, symbol="MNQU26")
    with caplog.at_level("CRITICAL"):
        await MimNbLive._backfill(o)
    assert o.prev_close is None, "a retired contract's close must not become the new one's"
    assert "STATE CONTRACT MISMATCH" in caplog.text
    assert o.sigma_hist, "sigma is contract-agnostic and must still be restored"


@pytest.mark.asyncio
async def test_legacy_state_without_a_contract_still_restores():
    """States written before this change carry no symbol; behaviour is unchanged."""
    o = _bot("MNQZ26")
    o._load_persisted_position = lambda: _state(symbol=None)
    await MimNbLive._backfill(o)
    assert o.prev_close == 29254.0


@pytest.mark.asyncio
async def test_absent_prev_close_is_not_an_error():
    o = _bot("MNQZ26")
    o._load_persisted_position = lambda: _state(prev_close=None, symbol="MNQU26")
    await MimNbLive._backfill(o)
    assert o.prev_close is None


def test_save_state_records_the_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    o = _bot("MNQZ26")
    o.prev_close, o.sigma_hist, o.sigma_days = 29744.75, SIGMA, ["2026-09-17"]
    MimNbLive._save_state(o)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["symbol"] == "MNQZ26"
    assert saved["prev_close"] == 29744.75


def test_saved_then_restored_across_a_roll_fails_closed(tmp_path, monkeypatch):
    """End to end: save on U26, come back on Z26, prev_close must not survive."""
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    old = _bot("MNQU26")
    old.prev_close, old.sigma_hist, old.sigma_days = 29151.5, SIGMA, ["2026-09-14"]
    MimNbLive._save_state(old)
    saved = json.loads((tmp_path / "state.json").read_text())

    import asyncio
    new = _bot("MNQZ26")
    new._load_persisted_position = lambda: saved
    asyncio.run(MimNbLive._backfill(new))
    assert new.prev_close is None
