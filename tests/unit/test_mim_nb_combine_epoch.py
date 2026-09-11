"""MIM-NB must replay only the CURRENT combine account's trades.

A Topstep reset opens a brand-new account at COMBINE_START_BALANCE with a fresh trailing
floor. `_init_combine_balance` previously replayed the whole of data/mim_nb/trades.csv,
which spans resets: on 2026-09-11 it carried 17 trades worth +$681.00 from the account
retired on 2026-08-13, reporting balance $50,347.50 / hwm $50,787.50 against a true
$50,000.08 / $50,298.96.

Nothing gated on that number (floor gating removed 2026-07-29; the DLL guard is static)
and the runtime path prefers the shared floor state -- but this own-ledger path is the
fallback used whenever the floor monitor goes stale, which happened 28 times on
2026-07-29/30. And the contamination was NOT conservative by construction: it understated
the buffer only because the retired account happened to end profitable, lifting the
high-water mark. A loss-making retired epoch would push it the other way.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from src.research import mim_nb_live as M


HEADER = "day,dir,entry_t,entry_px,exit_t,exit_px,reason,pnl_pts,pnl_usd,day_pnl_usd,chain\n"


def _row(day: str, pnl: float) -> str:
    return f"{day},1,10:00,100.00,16:00,101.00,EOD,+1.00,{pnl:+.2f},{pnl:+.2f},abc\n"


@pytest.fixture()
def bot(tmp_path, monkeypatch):
    """A MimNbLive with just the attributes _init_combine_balance touches."""
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "FLOOR_STATE_FILE", tmp_path / "floor_state.json")
    b = M.MimNbLive.__new__(M.MimNbLive)      # skip __init__: no network, no broker
    b._realized_pnl = 0.0
    b._mll_eod_hwm = M.COMBINE_START_BALANCE
    return b


def _write_trades(tmp_path, rows):
    (tmp_path / "trades.csv").write_text(HEADER + "".join(rows))


def _write_floor_state(tmp_path, combine_start: str):
    (tmp_path / "floor_state.json").write_text(json.dumps({
        "account_id": "26556101", "combine_start": combine_start,
        "hwm": 50298.96, "floor": 48298.96, "balance": 50000.08, "equity": 50000.08,
    }))


def test_trades_before_the_epoch_are_excluded(bot, tmp_path):
    """The actual defect: a retired account's profit inflating this account's balance."""
    _write_floor_state(tmp_path, "2026-08-13T16:54:13+00:00")
    _write_trades(tmp_path, [
        _row("2026-07-01", +500.00),   # retired account
        _row("2026-07-15", +181.00),   # retired account
        _row("2026-08-20", -200.00),   # current account
        _row("2026-09-04", -133.50),   # current account
    ])
    bot._init_combine_balance()
    assert bot._realized_pnl == pytest.approx(-333.50), "only post-reset trades count"
    assert bot._mll_eod_hwm == pytest.approx(M.COMBINE_START_BALANCE), (
        "a losing current account never rises above its start balance"
    )


def test_epoch_is_read_from_the_authoritative_floor_state(bot, tmp_path):
    _write_floor_state(tmp_path, "2026-08-13T16:54:13+00:00")
    assert bot._combine_epoch_start() == date(2026, 8, 13)


def test_missing_floor_state_falls_back_not_replays_everything(bot, tmp_path):
    """The critical failure mode.

    The floor state being unreadable is EXACTLY when this own-ledger path is in use, so
    degrading to "replay everything" there would reintroduce the bug at the worst moment.
    """
    # no floor_state.json written
    assert bot._combine_epoch_start() == M.COMBINE_EPOCH_START_FALLBACK

    _write_trades(tmp_path, [
        _row("2026-07-01", +681.00),   # retired -- must still be excluded
        _row("2026-08-20", -333.50),
    ])
    bot._init_combine_balance()
    assert bot._realized_pnl == pytest.approx(-333.50)


def test_malformed_floor_state_falls_back(bot, tmp_path):
    (tmp_path / "floor_state.json").write_text("{not json")
    assert bot._combine_epoch_start() == M.COMBINE_EPOCH_START_FALLBACK


def test_rows_with_unparseable_day_are_skipped_not_fatal(bot, tmp_path):
    _write_floor_state(tmp_path, "2026-08-13T16:54:13+00:00")
    _write_trades(tmp_path, [
        _row("", +999.00),
        _row("not-a-date", +999.00),
        _row("2026-08-20", -100.00),
    ])
    bot._init_combine_balance()
    assert bot._realized_pnl == pytest.approx(-100.00)


def test_high_water_mark_tracks_the_running_peak_within_the_epoch(bot, tmp_path):
    _write_floor_state(tmp_path, "2026-08-13T16:54:13+00:00")
    _write_trades(tmp_path, [
        _row("2026-07-01", +5000.00),  # excluded: must not lift this account's hwm
        _row("2026-08-14", +300.00),   # peak 50300
        _row("2026-08-20", -100.00),
    ])
    bot._init_combine_balance()
    assert bot._mll_eod_hwm == pytest.approx(50_300.00)
    assert bot._realized_pnl == pytest.approx(+200.00)


def test_stays_mim_only_and_conservative(bot, tmp_path):
    """Pins the 2026-09-11 decision: exclude YANK, under-estimate the real buffer.

    Real numbers from that date -- MIM-only post-reset gives buffer $1,637.00 against the
    true combined $1,701.12. Lower is the safe direction; this must not drift above it.
    """
    _write_floor_state(tmp_path, "2026-08-13T16:54:13+00:00")
    _write_trades(tmp_path, [
        _row("2026-06-11", +681.00),
        _row("2026-08-14", +29.50),
        _row("2026-09-04", -363.00),
    ])
    bot._init_combine_balance()
    balance = M.COMBINE_START_BALANCE + bot._realized_pnl
    buffer_ = balance - (bot._mll_eod_hwm - M.MLL_DD)
    assert buffer_ == pytest.approx(1_637.00)
    assert buffer_ < 1_701.12, "MIM-only must under-estimate the true combined buffer"
