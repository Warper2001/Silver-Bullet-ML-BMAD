#!/usr/bin/env python3
"""Tests for tools/tier2_census.py — the instrumented Tier2/YANK signal census.

No replay is run here (that takes minutes and needs the model and bars). These pin the
guard rails: the sealed-holdout refusal, pin parsing, the live-file guard, and that the
in-process hook records a signal only when the engine actually creates a NEW pending order.

Run:   .venv/bin/python -m pytest tools/test_tier2_census.py -v
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "tools"))
tc = importlib.import_module("tier2_census")

CUTOFF = datetime(2026, 3, 1, tzinfo=timezone.utc)


def test_window_before_cutoff_is_accepted_in_utc():
    s, e = tc.check_window("2025-05-19", "2026-02-28", CUTOFF)
    assert s.tzinfo is timezone.utc and (e.hour, e.minute) == (23, 59)


@pytest.mark.parametrize("end", ["2026-03-01", "2026-05-19"])
def test_window_reaching_the_holdout_is_refused(end):
    with pytest.raises(SystemExit, match="holdout"):
        tc.check_window("2025-05-19", end, CUTOFF)


def test_start_after_end_is_refused():
    with pytest.raises(SystemExit):
        tc.check_window("2026-02-01", "2026-01-01", CUTOFF)


def test_parse_pins_types():
    assert tc.parse_pins(["max_daily_loss=-750", "ml_threshold=0.5", "bearish_only=true", "name=x"]) == \
        {"max_daily_loss": -750, "ml_threshold": 0.5, "bearish_only": True, "name": "x"}
    assert tc.parse_pins([]) == {}


@pytest.mark.parametrize("bad", ["max_daily_loss", "=5"])
def test_parse_pins_rejects_malformed(bad):
    with pytest.raises(ValueError):
        tc.parse_pins([bad])


def test_guard_snapshot_reports_missing_files_as_none(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/trades.db").write_bytes(b"x")
    snap = tc.guard_snapshot(tmp_path)
    assert snap["data/trades.db"][0] == 1 and snap["logs/tier2_trade_log.csv"] is None


def _fake_module():
    class MetaLabelingFilter:
        def predict_proba(self, features):
            return 0.61

    class Tier2StreamingTrader:
        def __init__(self):
            self.active_trade = None
            self.ml_filter = MetaLabelingFilter()

        async def _enter_trade(self, fvg, bar, idx, is_backfill):
            self.ml_filter.predict_proba({})
            if not is_backfill:
                self.active_trade = SimpleNamespace(
                    entry_time=datetime(2025, 6, 2, 14, 0, tzinfo=timezone.utc), direction="SHORT",
                    entry_price=100.0, sl_price=120.0, tp_price=20.0, gap_size=10.0, contracts=5)

    return SimpleNamespace(MetaLabelingFilter=MetaLabelingFilter, Tier2StreamingTrader=Tier2StreamingTrader)


def test_hook_records_only_newly_created_pending_orders():
    mod = _fake_module()
    signals: list = []
    tc.install_hooks(mod, signals, {"p": None})
    trader = mod.Tier2StreamingTrader()
    fvg = SimpleNamespace(high=105.0, low=95.0)

    asyncio.run(trader._enter_trade(fvg, None, 0, True))      # backfill: no order -> nothing recorded
    assert signals == []
    asyncio.run(trader._enter_trade(fvg, None, 1, False))     # new order -> recorded with its proba
    assert len(signals) == 1
    assert signals[0]["signal_ts"] == "2025-06-02T14:00:00+00:00"
    assert signals[0]["ml_proba"] == 0.61 and signals[0]["fvg_top"] == 105.0

    before = len(signals)
    asyncio.run(trader._enter_trade(fvg, None, 2, True))       # engine leaves the SAME pending order -> no new row
    assert len(signals) == before
