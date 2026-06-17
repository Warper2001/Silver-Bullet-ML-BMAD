"""Combine floor monitor — pure trigger logic + combined-PF DB reader."""
import sqlite3

import pytest

from src.research.combine_floor_monitor import (
    update_floor, evaluate_triggers, combined_pf_and_count,
    FLOOR_START, START_EQUITY, HALT_DISTANCE, PF_THRESHOLD, COMBINE_START,
)


def test_floor_ratchets_up_then_caps_and_never_drops():
    assert update_floor(48000, 49000) == 48000          # hwm-2000 below start floor
    assert update_floor(48000, 51000) == 49000          # ratchets up to hwm-2000
    assert update_floor(49000, 53000) == 50000          # caps at start equity
    assert update_floor(49000, 49500) == 49000          # never decreases


def test_distance_to_floor_trigger():
    floor = 49000.0
    assert evaluate_triggers(floor + 600, floor, 1.5, 40) is None         # comfortable
    assert evaluate_triggers(floor + 500, floor, 1.5, 40) is not None     # exactly at boundary -> halt
    assert "DISTANCE_TO_FLOOR" in evaluate_triggers(floor + 100, floor, 1.5, 40)
    assert evaluate_triggers(floor + 501, floor, 1.5, 40) is None


def test_combined_pf_trigger_needs_min_trades():
    floor = 49000.0
    eq = floor + 1000                                                     # distance ok
    assert evaluate_triggers(eq, floor, 0.69, 29) is None                 # below 30 trades -> no fire
    assert "COMBINED_PF" in evaluate_triggers(eq, floor, 0.69, 30)        # 30 trades, pf<0.70 -> fire
    assert evaluate_triggers(eq, floor, 0.71, 30) is None                 # pf above threshold
    assert evaluate_triggers(eq, floor, None, 40) is None                 # no losses yet -> pf None


def test_distance_takes_priority_over_pf():
    floor = 49000.0
    r = evaluate_triggers(floor + 100, floor, 0.50, 40)                   # both fire
    assert "DISTANCE_TO_FLOOR" in r


def _make_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE trades (trader_id TEXT, timestamp TEXT, pnl REAL)")
    con.executemany("INSERT INTO trades (trader_id, timestamp, pnl) VALUES (?,?,?)", rows)
    con.commit(); con.close()


def test_combined_pf_counts_both_bots_excludes_old(tmp_path):
    db = tmp_path / "t.db"
    _make_db(db, [
        ("trader-mim-nb", "2026-06-18T15:00:00+00:00", 740.0),   # in window
        ("trader-yank",   "2026-06-18T16:00:00+00:00", -200.0),  # in window
        ("trader-yank",   "2026-06-18T17:00:00+00:00", 100.0),   # in window
        ("trader-yank",   "2025-01-01T00:00:00+00:00", -9999.0), # OLD backtest row -> excluded
        ("trader-other",  "2026-06-18T18:00:00+00:00", -5000.0), # different bot -> excluded
    ])
    pf, n = combined_pf_and_count(db, COMBINE_START)
    assert n == 3                                                # both bots, in-window only
    assert pf == pytest.approx((740.0 + 100.0) / 200.0)          # gross win / gross loss


def test_combined_pf_none_when_no_losses(tmp_path):
    db = tmp_path / "t.db"
    _make_db(db, [("trader-mim-nb", "2026-06-18T15:00:00+00:00", 740.0)])
    pf, n = combined_pf_and_count(db, COMBINE_START)
    assert n == 1 and pf is None
