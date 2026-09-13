#!/usr/bin/env python3
"""Tests for tools/tier2_entry_arms.py — the Tier2/YANK entry-arm simulator.

Synthetic bars only; no repository data. The properties pinned here are the ones the
2026-09-13 pre-registered run depended on: a touch fills the engine replica (B0) but not
the trade-through reference (B1); unfilled signals score 0; stops resolve before targets
inside a bar; market and stop entries pay one tick of adverse slippage; running out of
bars is INCOMPLETE, never a silent zero.

Run:   .venv/bin/python -m pytest tools/test_tier2_entry_arms.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))
ea = importlib.import_module("tier2_entry_arms")
from src.research.strategy_core import Direction  # noqa: E402

GAP = 10.0            # stop = 2*GAP = 20 pts, target = 8*GAP = 80 pts


def make_bars(n=400, price=1000.0, overrides=None) -> "ea.Bars":
    """Flat bars at ``price`` with a 1-point range; overrides = {index: (o, h, l, c)}."""
    opens = np.full(n, price)
    highs = np.full(n, price + 0.5)
    lows = np.full(n, price - 0.5)
    closes = np.full(n, price)
    for i, (o, h, lo, c) in (overrides or {}).items():
        opens[i], highs[i], lows[i], closes[i] = o, h, lo, c
    ts = pd.Series(pd.date_range("2025-06-02 14:00", periods=n, freq="1min", tz="UTC"))
    return ea.Bars(opens, highs, lows, closes, ts)


def short_levels(entry=1005.0):
    return entry, entry + 2 * GAP, entry - 8 * GAP   # entry, sl, tp


def test_touch_fills_engine_replica_but_not_trade_through_reference():
    entry, sl, tp = short_levels()
    b = make_bars(overrides={3: (1000.0, entry, 999.5, 1000.0)})   # high == entry exactly
    b0 = ea.arm_B(b, 0, entry, sl, tp, GAP, touch=True)
    b1 = ea.arm_B(b, 0, entry, sl, tp, GAP, touch=False)
    assert b0["fill_i"] == 3
    assert b1 == {"R": 0.0, "fill_i": None, "exit": None}     # never traded through -> unfilled scores 0


def test_trade_through_fills_and_time_stop_exits_after_60_bars():
    entry, sl, tp = short_levels()
    b = make_bars(overrides={2: (1000.0, entry + ea.TICK, 999.5, 1004.0)})
    r = ea.arm_B(b, 0, entry, sl, tp, GAP, touch=False)
    assert r["fill_i"] == 2
    exit_px, kind, held = r["exit"]
    assert kind == "time" and held == ea.HOLD and exit_px == 1000.0
    assert r["R"] == pytest.approx(((entry - 1000.0) - ea.COST_PTS) / (2 * GAP))


def test_stop_is_checked_before_target_inside_one_bar():
    entry, sl, tp = short_levels()
    b = make_bars(overrides={2: (1000.0, entry + 1, 999.5, 1004.0),
                             5: (1000.0, sl + 1, tp - 1, 1000.0)})      # bar 5 touches both
    r = ea.arm_B(b, 0, entry, sl, tp, GAP, touch=False)
    assert r["exit"][1] == "sl" and r["exit"][0] == sl


def test_running_out_of_bars_is_incomplete_not_zero():
    entry, sl, tp = short_levels()
    b = make_bars(n=50, overrides={2: (1000.0, entry + 1, 999.5, 1004.0)})
    assert ea.arm_B(b, 0, entry, sl, tp, GAP, touch=False) == ea.INCOMPLETE
    assert ea.arm_B(make_bars(n=50), 0, entry, sl, tp, GAP, touch=False) == ea.INCOMPLETE  # pending window cut short


def test_market_entry_pays_one_tick_adverse_in_either_direction():
    b = make_bars(overrides={1: (1000.0, 1000.5, 999.5, 1000.0)})
    short = ea.arm_market(b, 0, GAP, Direction.BEARISH)
    long = ea.arm_market(b, 0, GAP, Direction.BULLISH)
    assert short["fill_i"] == 1 and long["fill_i"] == 1
    # flat path -> time stop at 1000.0; short sold at 999.75, long bought at 1000.25
    assert short["R"] == pytest.approx(((999.75 - 1000.0) - ea.COST_PTS) / (2 * GAP))
    assert long["R"] == pytest.approx(((1000.0 - 1000.25) - ea.COST_PTS) / (2 * GAP))


def test_bracket_is_anchored_at_the_fill_and_snapped_to_ticks():
    assert ea.bracket(Direction.BEARISH, 1000.1, GAP) == (1020.0, 920.0)
    assert ea.bracket(Direction.BULLISH, 1000.0, 10.125) == (979.75, 1081.0)


def test_sell_stop_triggers_one_tick_below_signal_low_with_slippage():
    b = make_bars(overrides={0: (1000.0, 1000.5, 999.0, 999.5), 4: (999.0, 999.0, 998.0, 998.5)})
    r = ea.arm_S(b, 0, GAP)
    assert r["fill_i"] == 4
    assert r["R"] == pytest.approx((((998.75 - 0.25) - 1000.0) - ea.COST_PTS) / (2 * GAP))


def test_sell_stop_never_triggered_scores_zero():
    b = make_bars(overrides={0: (1000.0, 1000.5, 990.0, 999.5)})   # trigger far below the flat path
    assert ea.arm_S(b, 0, GAP) == {"R": 0.0, "fill_i": None, "exit": None}


def test_random_arm_is_seed_reproducible_and_incomplete_near_the_end():
    b = make_bars(n=700)
    r1 = ea.arm_A(b, 0, GAP, np.random.default_rng(42), nrand=20)
    r2 = ea.arm_A(b, 0, GAP, np.random.default_rng(42), nrand=20)
    assert r1 == r2
    assert ea.arm_A(b, 700 - ea.PEND - 1, GAP, np.random.default_rng(0), nrand=5) == ea.INCOMPLETE


def test_sequential_skips_signals_while_a_position_is_busy():
    sig = pd.DataFrame({"s": [0, 10, 100]})
    results = [{"R": 1.0, "fill_i": 5, "exit": (0, "time", 60)},   # busy until bar 65
               {"R": 2.0, "fill_i": 12, "exit": (0, "time", 60)},  # skipped
               {"R": 3.0, "fill_i": None, "exit": None}]           # unfilled, taken
    assert ea.sequential(sig, results) == [1.0, 3.0]
