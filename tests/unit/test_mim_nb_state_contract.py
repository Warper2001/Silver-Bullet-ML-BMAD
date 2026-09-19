"""Price provenance is distinct from the active contract and sigma provenance."""

import json
from unittest.mock import AsyncMock
import pytest
from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive

SIGMA = {"09:31": [0.001] * 14}


def bot():
    o = object.__new__(MimNbLive)
    o.symbol = "MNQZ26"
    o.prev_close = None
    o.prev_close_symbol = None
    o.sigma_hist = {}
    o.sigma_days = []
    o.day = "2026-09-18"
    o.position = 0
    o.entry_px = None
    o.entry_t = None
    o.cat_stop_id = None
    o.day_pnl = 0.0
    o._prev_close_for_symbol = AsyncMock(return_value=29700.0)
    return o


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [None, "MNQU26", "UNKNOWN", "MNQZ26"])
async def test_restart_uses_price_provenance(provenance):
    o = bot()
    o.position = 1
    o.entry_px = 29000.0
    o.cat_stop_id = 123
    st = dict(
        symbol="MNQZ26",
        prev_close_symbol=provenance,
        prev_close=29254.0,
        sigma_hist=SIGMA,
        sigma_days=["2026-09-17"],
    )
    o._load_persisted_position = lambda: st
    await o._backfill()
    assert o.prev_close == (29254.0 if provenance == "MNQZ26" else 29700.0)
    assert o.prev_close_symbol == "MNQZ26"
    assert o.sigma_hist == SIGMA and o.sigma_days == ["2026-09-17"]
    assert (o.position, o.entry_px, o.cat_stop_id) == (1, 29000.0, 123)
    assert o._prev_close_for_symbol.await_count == (0 if provenance == "MNQZ26" else 1)


@pytest.mark.asyncio
async def test_failed_lookup_clears_price():
    o = bot()
    o._prev_close_for_symbol = AsyncMock(return_value=None)
    o._load_persisted_position = lambda: dict(prev_close=29254.0, sigma_hist=SIGMA)
    await o._backfill()
    assert o.prev_close is None and o.prev_close_symbol is None
    assert o.sigma_hist == SIGMA


@pytest.mark.asyncio
async def test_cold_seed_cannot_relabel_old_contract():
    o = bot()
    o._load_persisted_position = lambda: None

    def seed():
        o.sigma_hist = SIGMA
        o.prev_close = 27000.0
        o.prev_close_symbol = "MNQU26"

    o._seed_sigma_from_bars = seed
    await o._backfill()
    assert o.prev_close == 29700.0 and o.prev_close_symbol == "MNQZ26"


def test_save_preserves_price_symbol_independent_of_active(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    o = bot()
    o.prev_close = 27000.0
    o.prev_close_symbol = "MNQU26"
    o._save_state()
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["symbol"] == "MNQZ26" and st["prev_close_symbol"] == "MNQU26"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "invalid", None])
async def test_bad_saved_price_rederived(value):
    o = bot()
    o._load_persisted_position = lambda: dict(
        prev_close=value, prev_close_symbol=o.symbol, sigma_hist=SIGMA
    )
    await o._backfill()
    assert o.prev_close == 29700.0


@pytest.mark.asyncio
async def test_raised_lookup_preserves_open_position_and_sigma():
    o = bot()
    o.position = 1
    o._prev_close_for_symbol = AsyncMock(side_effect=ValueError("malformed bar"))
    o._load_persisted_position = lambda: dict(prev_close=1, sigma_hist=SIGMA)
    await o._backfill()
    assert o.position == 1 and o.sigma_hist == SIGMA and o.prev_close is None
