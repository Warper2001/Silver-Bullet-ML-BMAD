#!/usr/bin/env python3
"""Tests for the two 2026-08-07 silent-failure detectors.

Per the party-mode sidebar that found YANK 17 days silent with every existing check
green: a detector nobody has seen fire is not a detector. Both of these exist ONLY to
catch a specific failure that already happened once, so each is tested on the shape of
that failure AND on the healthy case (a monitor that always fires gets muted, which is
the same as not having it).

Covered:
  - combine_ops_healthcheck.structural_silence: the confirmed-structure/zero-FVG funnel
  - fvg_feasibility.atr_bound / feasibility: the gap-window arithmetic

Run:   .venv/bin/python -m pytest tools/test_silence_and_feasibility.py -v
       .venv/bin/python tools/test_silence_and_feasibility.py
"""
from __future__ import annotations

import dataclasses
import importlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

hc = importlib.import_module("combine_ops_healthcheck")
fz = importlib.import_module("fvg_feasibility")

HEADER = ("bar_timestamp,h1_sweep_active,kill_zone_active,vol_regime_blocked,"
          "m15_confirmed,fvg_detected,action\n")


def write_funnel(path: Path, rows) -> None:
    """rows = [(minutes_ago, h1_sweep, m15_confirmed, fvg_detected)]"""
    now = datetime.now(hc.ET)
    with path.open("w") as fh:
        fh.write(HEADER)
        for ago, h1, m15, fvg in rows:
            ts = (now - timedelta(minutes=ago)).isoformat()
            fh.write(f"{ts},{h1},False,False,{m15},{fvg},SKIP\n")


@pytest.fixture()
def funnel(tmp_path, monkeypatch):
    """Point the checker's BASE at a tmp dir and hand back the funnel-log path."""
    monkeypatch.setattr(hc, "BASE", tmp_path)
    (tmp_path / "logs").mkdir()
    return tmp_path / "logs" / "decisions.csv"


REL = "logs/decisions.csv"


# --- structural_silence -------------------------------------------------------------

def test_missing_log_returns_none(funnel):
    assert hc.structural_silence(REL) is None


def test_empty_log_returns_none(funnel):
    funnel.write_text(HEADER)
    assert hc.structural_silence(REL) is None


def test_silent_funnel_is_detected(funnel):
    """The 2026-08-07 shape: structure confirms all week, nothing ever fires."""
    write_funnel(funnel, [(i, True, True, False) for i in range(1, 401)])
    confirmed, fvg, sweep = hc.structural_silence(REL)
    assert confirmed == 400 and fvg == 0 and sweep == 400
    assert confirmed >= hc.SILENCE_MIN_CONFIRMED, "fixture must clear the judging threshold"


def test_productive_funnel_is_not_flagged(funnel):
    """One FVG in the window is enough to prove the bot CAN fire — no alarm."""
    rows = [(i, True, True, False) for i in range(1, 401)]
    rows.append((5, True, True, True))
    write_funnel(funnel, rows)
    confirmed, fvg, _ = hc.structural_silence(REL)
    assert fvg == 1 and confirmed == 401


def test_sweep_without_choch_does_not_count_as_confirmed(funnel):
    """Silence downstream of an UNCONFIRMED sweep is normal — must not raise the count."""
    write_funnel(funnel, [(i, True, False, False) for i in range(1, 401)])
    confirmed, fvg, sweep = hc.structural_silence(REL)
    assert sweep == 400 and confirmed == 0 and fvg == 0


def test_thin_sample_is_reported_but_not_alarming(funnel):
    """Below SILENCE_MIN_CONFIRMED the caller must not treat silence as a finding."""
    write_funnel(funnel, [(i, True, True, False) for i in range(1, 11)])
    confirmed, fvg, _ = hc.structural_silence(REL)
    assert fvg == 0
    assert confirmed < hc.SILENCE_MIN_CONFIRMED


def test_rows_older_than_the_window_are_ignored(funnel):
    old = int(timedelta(days=hc.SILENCE_WINDOW_DAYS + 3).total_seconds() // 60)
    write_funnel(funnel, [(old + i, True, True, False) for i in range(400)])
    assert hc.structural_silence(REL) is None, "no in-window rows => no verdict, not a false alarm"


def test_naive_timestamps_are_skipped_not_crashed(funnel):
    with funnel.open("w") as fh:
        fh.write(HEADER)
        fh.write("2026-08-07T12:00:00,True,False,False,True,False,SKIP\n")
    assert hc.structural_silence(REL) is None


def test_short_rows_are_skipped(funnel):
    ts = datetime.now(hc.ET).isoformat()
    with funnel.open("w") as fh:
        fh.write(HEADER)
        fh.write(f"{ts},True\n")
        fh.write(f"{ts},True,False,False,True,False,SKIP\n")
    confirmed, fvg, _ = hc.structural_silence(REL)
    assert confirmed == 1 and fvg == 0


def test_bounded_read_drops_only_the_leading_fragment(funnel, monkeypatch):
    """The tail read is bounded; it must lose at most the first (partial) line."""
    monkeypatch.setattr(hc, "SILENCE_TAIL_BYTES", 4096)
    write_funnel(funnel, [(i, True, True, False) for i in range(1, 401)])
    confirmed, _, _ = hc.structural_silence(REL)
    assert 0 < confirmed < 400, "bounded read should see a suffix, not the whole file"


# --- feasibility arithmetic ---------------------------------------------------------

def cfg(ratio: float = 0.25, dollars: float = 60.0):
    # StrategyConfig is a frozen dataclass (sealed config must not be mutable in place).
    return dataclasses.replace(
        fz.StrategyConfig(), min_gap_atr_ratio=ratio, max_gap_dollars=dollars
    )


def test_atr_bound_matches_the_live_config():
    """$60 / ($2 x 0.25) = 120 index points — the number behind the drought."""
    assert fz.atr_bound(cfg()) == pytest.approx(120.0)


def test_s25_tightening_halved_the_bound():
    """0.15 -> 0.25 on 2026-05-24 moved the bound from 200 pts to 120 pts."""
    assert fz.atr_bound(cfg(ratio=0.15)) == pytest.approx(200.0)
    assert fz.atr_bound(cfg(ratio=0.25)) == pytest.approx(120.0)


def test_zero_ratio_disables_the_floor():
    assert fz.atr_bound(cfg(ratio=0.0)) == float("inf")


def test_window_empty_exactly_above_the_bound():
    idx = pd.date_range("2026-08-01", periods=3, freq="h", tz="UTC")
    atr = pd.Series([119.0, 120.0, 121.0], index=idx)
    f = fz.feasibility(atr, cfg())
    assert list(f["empty"]) == [False, False, True], "the bound itself is still feasible"
    assert f["width_pts"].iloc[2] == 0.0, "empty windows report zero width, never negative"


def test_width_is_cap_minus_floor():
    idx = pd.date_range("2026-08-01", periods=1, freq="h", tz="UTC")
    f = fz.feasibility(pd.Series([100.0], index=idx), cfg())
    assert f["floor_pts"].iloc[0] == pytest.approx(25.0)
    assert f["cap_pts"].iloc[0] == pytest.approx(30.0)
    assert f["width_pts"].iloc[0] == pytest.approx(5.0)


def test_h1_atr_series_uses_the_live_calc_atr():
    """20 H1 bars in, exactly one ATR out — the live 20-bar window, not a pandas roll."""
    idx = pd.date_range("2026-08-01", periods=20 * 60, freq="min", tz="UTC")
    px = pd.Series(range(len(idx)), dtype="float64") * 0.25 + 20000.0
    bars = pd.DataFrame(
        {"open": px.values, "high": px.values + 2, "low": px.values - 2,
         "close": px.values, "volume": 1},
        index=idx,
    )
    bars.index.name = "timestamp"
    s = fz.h1_atr_series(bars)
    assert len(s) == 1 and s.iloc[0] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
