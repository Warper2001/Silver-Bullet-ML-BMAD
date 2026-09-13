#!/usr/bin/env python3
"""Tests for tools/gap_fade_census.py — the GAP-1 parity and learning-arm harness.

The sealed engine is read from the tracked backtest_gap_fade.py; every market input is
synthetic. Pinned properties: the sealed constants survive extraction unchanged; the
session rules match the seal (prior session >= 300 RTH bars, Fridays skipped, 0.5% gap);
the sealed engine resolves a same-bar target/stop touch TARGET-first (and the audit sees
it); a recording gap produces a stale prior close, which is why Q1 mismatches must be
inspected before they are called live defects.

Run:   .venv/bin/python -m pytest tools/test_gap_fade_census.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "tools"))
gf = importlib.import_module("gap_fade_census")

SEALED = gf.load_sealed(BASE / "backtest_gap_fade.py")


def census(seed=0):
    return gf.Census(SEALED, np.random.default_rng(seed))


def rth_day(date: str, open_px: float, close_px: float, n=390, drift=0.0):
    """One RTH session of n one-minute bars from 09:30 ET; opens at open_px, last close = close_px."""
    idx = pd.date_range(f"{date} 09:30", periods=n, freq="1min", tz="US/Eastern")
    px = np.linspace(open_px, close_px, n) + drift
    df = pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px}, index=idx)
    df.iloc[0, df.columns.get_loc("open")] = open_px
    df.iloc[-1, df.columns.get_loc("close")] = close_px
    return df


def test_sealed_constants_survive_extraction():
    assert (SEALED["GAP_MIN_PCT"], SEALED["STOP_MULT"], SEALED["TIME_STOP_HOUR"], SEALED["MIN_RTH_BARS"]) == \
        (0.005, 2.0, 13, 300)
    assert callable(SEALED["is_rth"]) and callable(SEALED["simulate_day"])


def test_extraction_fails_loudly_when_the_engine_lacks_a_function(tmp_path):
    fake = tmp_path / "engine.py"
    fake.write_text("GAP_MIN_PCT = 0.005\n")
    with pytest.raises(RuntimeError, match="missing"):
        gf.load_sealed(fake)


def test_sessions_follow_the_seal():
    c = census()
    bars = pd.concat([rth_day("2026-07-06", 20000, 20000),          # Mon
                      rth_day("2026-07-07", 20200, 20100),          # Tue: +1.0% gap -> ENTERED
                      rth_day("2026-07-08", 20150, 20000),          # Wed: +0.25% -> NO_SETUP
                      rth_day("2026-07-09", 20000, 20000, n=200),   # Thu: short session
                      rth_day("2026-07-10", 21000, 21000)])         # Fri: prior < 300 bars -> no session
    s = c.sessions(c.rth_by_day(bars))
    assert s["2026-07-07"]["action"] == "ENTERED" and s["2026-07-07"]["prior_close"] == 20000
    assert s["2026-07-08"]["action"] == "NO_SETUP"
    assert "2026-07-10" not in s


def test_friday_is_skipped_even_with_a_big_gap():
    c = census()
    bars = pd.concat([rth_day("2026-07-09", 20000, 20000), rth_day("2026-07-10", 20500, 20500)])
    assert c.sessions(c.rth_by_day(bars))["2026-07-10"]["action"] == "SKIPPED_FRIDAY"


def test_recording_gap_makes_a_stale_prior_close():
    """Missing days in the recording -> the harness uses the last recorded session. That is
    exactly the 2026-09-07 Labor Day artifact: inspect Q1 mismatches before blaming the bot."""
    c = census()
    bars = pd.concat([rth_day("2026-09-01", 20000, 19000), rth_day("2026-09-08", 19150, 19150)])  # 09-02..04 missing
    s = c.sessions(c.rth_by_day(bars))
    assert s["2026-09-08"]["prior_close"] == 19000        # the stale 09-01 close, not the true prior session


def test_decision_parity_holds_then_breaks_on_one_flip():
    sess = {"2026-07-07": {"action": "ENTERED", "prior_close": 20000.0, "rth_open": 20200.0}}
    dec = pd.DataFrame([{"date_et": "2026-07-07", "action": "ENTERED", "prior_close": 20000.0, "rth_open": 20200.0}])
    _, ok = gf.decision_parity(dec, sess, None)
    assert ok["verdict"] == "PARITY HOLDS" and ok["action_agreement"] == "1/1"
    dec.loc[0, "action"] = "NO_SETUP"
    _, bad = gf.decision_parity(dec, sess, None)
    assert bad["verdict"] == "PARITY BROKEN" and bad["action_mismatches"][0]["date"] == "2026-07-07"


def test_decision_parity_lists_sessions_without_recorded_bars():
    dec = pd.DataFrame([{"date_et": "2026-09-03", "action": "NO_SETUP", "prior_close": 1.0, "rth_open": 1.0}])
    _, r = gf.decision_parity(dec, {}, None)
    assert r["missing_recorded"] == ["2026-09-03"] and r["sessions_with_recorded_bars"] == 0


def test_sealed_engine_resolves_same_bar_touch_target_first_and_audit_sees_it():
    c = census()
    idx = pd.date_range("2026-07-07 09:31", periods=3, freq="1min", tz="US/Eastern")
    day = pd.DataFrame({"open": [100.0] * 3, "high": [100.5, 125.0, 100.5], "low": [99.5, 85.0, 99.5],
                        "close": [100.0] * 3}, index=idx)
    # short: entry 100, target 90, stop 120 — bar 2 touches both
    assert c.simulate_day(day, -1, 100.0, 90.0, 120.0)[0] == "fill"     # sealed: target first
    assert c.stop_first(day, -1, 100.0, 90.0, 120.0)[0] == "stop"
    assert c.both_touched(day, -1, 90.0, 120.0) is True


def test_arms_for_era_scores_the_sealed_fade_and_its_inverse():
    c = census(seed=1)
    bars = pd.concat([rth_day("2026-07-06", 20000, 20000),
                      rth_day("2026-07-07", 20200, 20200)])   # +1% gap, flat day -> time stop at 13:00
    days = c.rth_by_day(bars)
    df = c.arms_for_era(days, c.sessions(days), "2026-07-01", "2026-07-31")
    assert len(df) == 1 and df.B_outcome.iloc[0] == "time"
    ga = 200.0     # +1% gap, short fade filled at 20199.75, flat path exits at the 13:00 open 20200
    assert df.R_B.iloc[0] == pytest.approx((-0.25 - gf.COST) / (2 * ga))
    assert df.R_I.iloc[0] == pytest.approx((-0.25 - gf.COST) / (2 * ga))   # follow-the-gap long at 20200.25
    df2 = census(seed=1).arms_for_era(days, c.sessions(days), "2026-07-01", "2026-07-31")
    assert df.R_A.iloc[0] == df2.R_A.iloc[0]                # seed-reproducible random arm


def test_recorder_files_load_in_eastern_time_and_can_drop_backfilled_rows(tmp_path):
    hdr = "bar_ts,open,high,low,close,volume,fetched_at,lag_s,live,chain\n"
    a = tmp_path / "MNQZ26.csv"
    a.write_text(hdr + "2026-09-14T13:31:00Z,10,11,9,10.5,5,x,5,1,c1\n"
                       "2026-09-14T13:32:00Z,10.5,12,10,11,5,x,900,0,c2\n")
    b = tmp_path / "other.csv"
    b.write_text(hdr + "2026-09-14T13:31:00Z,99,99,99,99,5,x,5,1,c9\n")
    bars = gf.Census.bars_from_recorder([a, b])
    assert str(bars.index[0]) == "2026-09-14 09:31:00-04:00"
    assert bars.iloc[0]["close"] == 10.5                       # first file listed wins an overlap
    assert list(bars.columns) == ["open", "high", "low", "close"]
    assert len(gf.Census.bars_from_recorder([a], live_only=True)) == 1
