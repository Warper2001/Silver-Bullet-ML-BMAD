"""Floor-monitor regression tests — sealed prereg combine-restart-floor-hwm.

Two things are under test, both of which failed silently in production:

  1. The trailing HWM must ratchet on EQUITY (balance + open MtM), because that is
     what Topstep's MLL ratchets on. Commit 317f3ff switched it to realized balance
     and acct 23884932 breached on 2026-07-06 with the monitor reporting $1,339 of
     room. §6 of the seal commits to reproducing the real series; that is
     `test_recorded_series_*` below.

  2. Floor state must not survive an account change. A trailing floor is meaningless
     on a different account, and a combine reset that inherits one starts the new
     account with the dead one's numbers.

Nothing here touches the network or `do_halt`. Pure functions and a tmp state file.

Run:  .venv/bin/python -m pytest tests/unit/test_combine_floor_monitor_state.py -v
"""
from __future__ import annotations

import importlib
import json

import pytest

m = importlib.import_module("src.research.combine_floor_monitor")


# ----------------------------------------------------------------- real series
# Recorded ticks from data/combine_joint/monitor.csv (138,125 ticks over
# 2026-06-17..2026-08-13). These five are the load-bearing ones.
PEAK_EQUITY = 50_955.56          # 2026-06-22 16:03 UTC — open YANK short in profit
PEAK_BALANCE = 49_287.50         # realized balance at the 2026-06-26 HWM reset
TRUE_FLOOR = 48_955.56           # PEAK_EQUITY - $2,000 trail
EQ_0625 = 48_932.60              # first dip through the true floor (-$22.96)
EQ_0706 = 48_811.66              # the breach Topstep acted on (-$143.90)
BROKEN_FLOOR = 47_287.50         # what 317f3ff's realized-balance HWM produced


def _ratchet(equities):
    """Replay a series through the CORRECTED logic; return (hwm, floor)."""
    hwm, floor = m.START_EQUITY, m.FLOOR_START
    for eq in equities:
        hwm = max(hwm, eq)
        floor = m.update_floor(floor, hwm)
    return hwm, floor


# ------------------------------------------------------------------ item 1
def test_hwm_ratchets_on_equity_not_balance():
    """The core fix: an open position in profit must raise the floor."""
    hwm, floor = _ratchet([50_000.0, PEAK_EQUITY])
    assert hwm == PEAK_EQUITY
    assert floor == pytest.approx(TRUE_FLOOR)
    # Same series under the pre-fix denomination (realized balance): the peak never
    # registers, so the floor stays pinned at the account's opening MLL.
    balance_floor = m.update_floor(m.FLOOR_START, PEAK_BALANCE)
    assert balance_floor == pytest.approx(m.FLOOR_START)
    assert floor - balance_floor == pytest.approx(955.56, abs=0.01)


def test_production_floor_was_below_anything_the_code_can_produce():
    """The $47,287.50 that ran in production is unreachable by the ratchet.

    `update_floor` is clamped below by FLOOR_START ($48,000), so no sequence of
    balances yields $47,287.50 — 317f3ff reached it by hand-writing floor_state.json.
    The denominator bug cost $955.56 of floor; the accompanying manual state reset
    cost a further $712.50, putting the floor below the account's own opening MLL.
    """
    for bal in (0.0, 40_000.0, PEAK_BALANCE, 49_999.0):
        assert m.update_floor(m.FLOOR_START, bal) >= m.FLOOR_START
    assert BROKEN_FLOOR < m.FLOOR_START
    assert m.FLOOR_START - BROKEN_FLOOR == pytest.approx(712.50, abs=0.01)


def test_recorded_series_reproduces_the_true_floor():
    """Seal §6: replaying the real peak must yield floor $48,955.56."""
    _, floor = _ratchet([50_626.52, 50_428.80, PEAK_EQUITY, 50_863.08])
    assert floor == pytest.approx(TRUE_FLOOR)


def test_recorded_series_fires_on_2026_06_25():
    """The alarm 317f3ff dismissed as false was real, at -$22.96."""
    _, floor = _ratchet([PEAK_EQUITY])
    reason = m.evaluate_triggers(EQ_0625, floor, 1.5, 5)
    assert reason is not None and "DISTANCE_TO_FLOOR" in reason
    assert EQ_0625 - floor == pytest.approx(-22.96, abs=0.01)


def test_recorded_series_breaches_on_2026_07_06():
    """Topstep independently reports the MLL breach on 7/6/26."""
    _, floor = _ratchet([PEAK_EQUITY])
    assert EQ_0706 < floor
    assert EQ_0706 - floor == pytest.approx(-143.90, abs=0.01)
    assert "DISTANCE_TO_FLOOR" in m.evaluate_triggers(EQ_0706, floor, 1.5, 21)


def test_pre_fix_logic_was_blind_to_both_dates():
    """Characterize the bug: on the broken floor neither date registers at all."""
    assert m.evaluate_triggers(EQ_0625, BROKEN_FLOOR, 1.5, 5) is None
    assert m.evaluate_triggers(EQ_0706, BROKEN_FLOOR, 1.5, 21) is None
    assert EQ_0706 - BROKEN_FLOOR == pytest.approx(1_524.16, abs=0.01)


def test_floor_never_exceeds_account_start():
    """Topstep's ratchet stops at the starting balance; unchanged, pinned here."""
    _, floor = _ratchet([60_000.0])
    assert floor == m.START_EQUITY


def test_floor_never_ratchets_down():
    _, floor = _ratchet([PEAK_EQUITY, 48_000.0, 47_000.0])
    assert floor == pytest.approx(TRUE_FLOOR)


# ------------------------------------------------------------------ items 2-4
@pytest.fixture
def state_file(tmp_path, monkeypatch):
    f = tmp_path / "floor_state.json"
    monkeypatch.setattr(m, "STATE_FILE", f)
    return f


def test_genesis_when_no_state(state_file, monkeypatch):
    monkeypatch.setattr(m, "ACCOUNT_ID", "111")
    st = m.load_state()
    assert st["account_id"] == "111"
    assert st["hwm"] == m.START_EQUITY and st["floor"] == m.FLOOR_START
    assert st["chain"] == "GENESIS"
    assert st["combine_start"]


def test_matching_account_loads_unchanged(state_file, monkeypatch):
    monkeypatch.setattr(m, "ACCOUNT_ID", "111")
    state_file.write_text(json.dumps(
        {"account_id": "111", "combine_start": "2026-08-17T00:00:00+00:00",
         "hwm": 50_500.0, "floor": 48_500.0, "chain": "abc"}))
    st = m.load_state()
    assert st["hwm"] == 50_500.0 and st["floor"] == 48_500.0 and st["chain"] == "abc"


def test_account_change_regenesises_and_archives(state_file, monkeypatch):
    """The restart path: a new combine must not inherit the dead one's floor."""
    monkeypatch.setattr(m, "ACCOUNT_ID", "999")
    state_file.write_text(json.dumps(
        {"account_id": "23884932", "combine_start": "2026-06-17T00:00:00+00:00",
         "hwm": 50_217.86, "floor": 48_217.86, "chain": "deadbeef"}))
    st = m.load_state()
    assert st["account_id"] == "999"
    assert st["hwm"] == m.START_EQUITY and st["floor"] == m.FLOOR_START
    assert st["combine_start"] != "2026-06-17T00:00:00+00:00"
    archive = state_file.with_name(f"{state_file.name}.acct-23884932")
    assert archive.exists(), "prior state must be preserved, not discarded"
    assert json.loads(archive.read_text())["floor"] == 48_217.86


def test_legacy_unbound_state_is_adopted_not_reset(state_file, monkeypatch):
    """Back-compat: the running monitor's readout must not move on deploy."""
    monkeypatch.setattr(m, "ACCOUNT_ID", "23884932")
    state_file.write_text(json.dumps(
        {"hwm": 50_217.86, "floor": 48_217.86, "chain": "deadbeef"}))
    st = m.load_state()
    assert st["account_id"] == "23884932"
    assert st["hwm"] == 50_217.86 and st["floor"] == 48_217.86
    assert st["combine_start"] == m.COMBINE_START


def test_unreadable_state_genesises(state_file, monkeypatch):
    monkeypatch.setattr(m, "ACCOUNT_ID", "111")
    state_file.write_text("{not json")
    st = m.load_state()
    assert st["account_id"] == "111" and st["floor"] == m.FLOOR_START
