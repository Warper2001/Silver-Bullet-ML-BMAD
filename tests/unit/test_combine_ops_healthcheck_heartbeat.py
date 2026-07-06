"""Fixture tests for the yank-heartbeat check in tools/combine_ops_healthcheck.py.

Amelia's ship gate: every alarm branch must be exercised offline (pure evaluate_heartbeat
on dict fixtures, no live bot, no systemctl) before the check judges the real-money bot.
Weekday anchors (2026): Mon 07-06, Tue 07-07, Fri 07-10, Sat 07-11, Sun 07-12.
"""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")
_SPEC = importlib.util.spec_from_file_location(
    "combine_ops_healthcheck",
    Path(__file__).resolve().parent.parent.parent / "tools" / "combine_ops_healthcheck.py")
hc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hc)


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _utc(et_dt):
    return et_dt.astimezone(timezone.utc)


def _hb(now_utc, *, ts_age=30, bar_age=60, market_open=True, backfill=False,
        fails=0, detect_total=0, **extra):
    hb = {
        "ts": (now_utc - timedelta(seconds=ts_age)).isoformat(),
        "loop_seq": 100,
        "market_open": market_open,
        "is_backfill": backfill,
        "data_source": "tradestation",
        "contract": "MNQU26",
        "last_bar_ts": (now_utc - timedelta(seconds=bar_age)).isoformat(),
        "bars_evaluated_total": 500,
        "consec_poll_failures": fails,
        "poll_error": None,
        "detect_errors_total": detect_total,
        "last_exception": None,
    }
    hb.update(extra)
    return hb


def _levels(results):
    return [lvl for lvl, _ in results]


# ---------------------------------------------------------------- globex window

@pytest.mark.parametrize("dt,expected", [
    (_et(2026, 7, 11, 12, 0), False),   # Saturday noon
    (_et(2026, 7, 12, 17, 59), False),  # Sunday 17:59 — pre-open
    (_et(2026, 7, 12, 18, 1), True),    # Sunday 18:01 — open
    (_et(2026, 7, 10, 16, 59), True),   # Friday 16:59 — still open
    (_et(2026, 7, 10, 17, 1), False),   # Friday 17:01 — closed for weekend
    (_et(2026, 7, 7, 17, 30), False),   # Tuesday daily break
    (_et(2026, 7, 7, 14, 0), True),     # Tuesday RTH
    (_et(2026, 7, 7, 18, 5), True),     # Tuesday evening reopen
])
def test_in_globex_window_boundaries(dt, expected):
    assert hc.in_globex_window(dt) is expected


def test_secs_since_globex_reopen():
    # Monday 10:00 ET -> most recent open is SUNDAY 18:00 (16h prior), not a same-day open
    assert hc.secs_since_globex_reopen(_et(2026, 7, 6, 10, 0)) == pytest.approx(16 * 3600)
    # Tuesday 18:05 -> 300s into the evening session
    assert hc.secs_since_globex_reopen(_et(2026, 7, 7, 18, 5)) == pytest.approx(300)
    # Saturday -> outside window
    assert hc.secs_since_globex_reopen(_et(2026, 7, 11, 12, 0)) is None


# ---------------------------------------------------------------- missing / corrupt

def test_missing_file_within_grace_is_ok():
    res = hc.evaluate_heartbeat(None, datetime.now(timezone.utc), uptime_secs=30)
    assert _levels(res) == [hc.OK]


def test_missing_file_past_grace_is_critical():
    res = hc.evaluate_heartbeat(None, datetime.now(timezone.utc), uptime_secs=700)
    assert _levels(res) == [hc.CRIT]


def test_missing_file_unknown_uptime_is_critical():
    res = hc.evaluate_heartbeat(None, datetime.now(timezone.utc), uptime_secs=None)
    assert _levels(res) == [hc.CRIT]


def test_corrupt_json_treated_as_missing_not_crash(tmp_path):
    p = tmp_path / "hb.json"
    p.write_text("{ torn write")
    assert hc.read_heartbeat(p) is None


# ---------------------------------------------------------------- staleness

def test_stale_ts_critical_past_threshold():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, ts_age=301), now, uptime_secs=9999)
    assert _levels(res) == [hc.CRIT]
    assert "stale" in res[0][1]


def test_fresh_ts_at_299s_is_ok():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, ts_age=299), now, uptime_secs=9999)
    assert _levels(res) == [hc.OK]


# ---------------------------------------------------------------- poll failures

def test_consec_poll_failures_critical_at_3():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, fails=3, poll_error="HTTP 401"), now, 9999)
    assert hc.CRIT in _levels(res)
    assert any("401" in msg for _, msg in res)


def test_consec_poll_failures_2_not_critical():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, fails=2), now, 9999)
    assert _levels(res) == [hc.OK]


# ---------------------------------------------------------------- detect errors

def test_detect_errors_critical_on_single_occurrence():
    """Amelia's call: a detect exception silently skipped an entry decision and is
    deterministic per code path — sticky CRITICAL until restart, no 3-strike grace."""
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(
        _hb(now, detect_total=1, last_exception="_detect_and_enter: ValueError('x')"), now, 9999)
    assert hc.CRIT in _levels(res)
    assert any("_detect_and_enter" in msg for _, msg in res)


# ---------------------------------------------------------------- bar lag gating

def test_bar_lag_critical_in_globex_hours():
    now = _utc(_et(2026, 7, 7, 14, 0))  # Tuesday RTH, session opened Monday 18:00
    res = hc.evaluate_heartbeat(_hb(now, bar_age=400), now, 9999)
    assert _levels(res) == [hc.CRIT]
    assert "data gap" in res[0][1] or "bar" in res[0][1]


def test_bar_lag_ok_under_threshold():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, bar_age=200), now, 9999)
    assert _levels(res) == [hc.OK]


def test_bar_lag_suppressed_when_market_closed_flag():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, bar_age=90000, market_open=False), now, 9999)
    assert _levels(res) == [hc.OK]


def test_bar_lag_suppressed_during_backfill():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, bar_age=90000, backfill=True), now, 9999)
    assert _levels(res) == [hc.OK]


def test_bar_lag_suppressed_on_saturday():
    now = _utc(_et(2026, 7, 11, 12, 0))
    res = hc.evaluate_heartbeat(_hb(now, bar_age=90000), now, 9999)
    assert _levels(res) == [hc.OK]


def test_bar_lag_grace_just_after_reopen():
    # Tuesday 18:03 ET: 180s into the session < 300s grace — stale weekend bar OK
    now = _utc(_et(2026, 7, 7, 18, 3))
    res = hc.evaluate_heartbeat(_hb(now, bar_age=4000), now, 9999)
    assert _levels(res) == [hc.OK]


def test_bar_lag_no_bar_yet_after_grace_is_critical():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now, last_bar_ts=None), now, 9999)
    assert hc.CRIT in _levels(res)


# ---------------------------------------------------------------- healthy path

def test_healthy_heartbeat_single_ok_line():
    now = _utc(_et(2026, 7, 7, 14, 0))
    res = hc.evaluate_heartbeat(_hb(now), now, 9999)
    assert _levels(res) == [hc.OK]
    assert "loop_seq" in res[0][1]
