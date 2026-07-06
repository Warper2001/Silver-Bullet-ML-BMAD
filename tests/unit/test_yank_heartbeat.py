"""Tests for the YANK evaluation heartbeat (yank_heartbeat.py + yank_streaming_working.py).

Contract under test (spec: _bmad-output/spec_yank_evaluation_heartbeat.md, Amelia review
W1-W4): the heartbeat proves the strategy loop is EVALUATING bars, never alters trading
control flow, and is structurally unable to take the trader down.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.research.yank_heartbeat import HeartbeatWriter
from src.research.yank_streaming_working import Tier2StreamingTrader


# ---------------------------------------------------------------- HeartbeatWriter

def test_heartbeat_write_atomic_valid_json(tmp_path):
    path = tmp_path / "hb.json"
    ok = HeartbeatWriter(path).write({"ts": "2026-07-06T00:00:00+00:00", "loop_seq": 7})
    assert ok is True
    data = json.loads(path.read_text())
    assert data["loop_seq"] == 7
    assert not (tmp_path / "hb.json.tmp").exists()  # tmp renamed away, no litter


def test_heartbeat_write_firewall_never_raises(tmp_path, monkeypatch):
    path = tmp_path / "hb.json"
    writer = HeartbeatWriter(path)
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    assert writer.write({"x": 1}) is False  # swallowed, reported via return value only


def test_heartbeat_write_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "hb.json"
    assert HeartbeatWriter(path).write({"x": 1}) is True
    assert json.loads(path.read_text()) == {"x": 1}


def test_heartbeat_write_serializes_datetimes(tmp_path):
    # default=str: a stray datetime in the payload must not kill the write
    path = tmp_path / "hb.json"
    assert HeartbeatWriter(path).write({"ts": datetime(2026, 7, 6, tzinfo=timezone.utc)}) is True


# ---------------------------------------------------------------- trader plumbing

def _bare_trader(tmp_path) -> Tier2StreamingTrader:
    """Trader skeleton without __init__ (avoids model/YAML/broker setup): only the
    attributes the heartbeat + poll paths touch."""
    t = Tier2StreamingTrader.__new__(Tier2StreamingTrader)
    t._init_heartbeat_state()
    t._heartbeat = HeartbeatWriter(tmp_path / "yank_heartbeat.json")
    t._symbol = "MNQU26"
    t._data_source = "tradestation"
    t._data_shadow = False
    t._is_backfill = False
    t._last_processed_timestamp = None
    t.dollar_bars = []
    t._bar_processing_times = []
    t._current_day = None
    t._daily_ranges = []
    t._session_open_price = np.nan
    t._session_high, t._session_low = float("-inf"), float("inf")
    t._bars_base_url = "https://example.invalid/barcharts/MNQU26?interval=1&unit=Minute"
    # neutralize strategy/trade methods — heartbeat tests must not touch trading logic
    t._update_h1_structure = lambda: None
    t._update_m15_choch = lambda: None
    t._check_stale = lambda bar: False
    async def _noop_advance(bar): pass
    t._advance_active_trade = _noop_advance
    return t


def _bar(ts: datetime) -> SimpleNamespace:
    return SimpleNamespace(timestamp=ts, open=100.0, high=101.0, low=99.0, close=100.5)


class _Resp:
    def __init__(self, status, bars):
        self.status_code = status
        self._bars = bars
    def json(self):
        return {"Bars": self._bars}


def _client(responses):
    """Async stub: pops one _Resp per get(); raises the item instead if it's an exception."""
    async def get(url, headers=None):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    return SimpleNamespace(get=get)


_AUTH = SimpleNamespace(authenticate=None)
async def _tok():
    return "tok"
_AUTH.authenticate = _tok


def test_write_heartbeat_payload_fields(tmp_path):
    t = _bare_trader(tmp_path)
    t._last_processed_timestamp = datetime(2026, 7, 6, 14, 30, tzinfo=timezone.utc)
    t._hb_loop_seq = 42
    t._write_heartbeat(market_open=True)
    hb = json.loads((tmp_path / "yank_heartbeat.json").read_text())
    for field in ("ts", "loop_seq", "pid", "started_at", "market_open", "is_backfill",
                  "data_source", "contract", "last_bar_ts", "bars_new_this_cycle",
                  "bars_evaluated_total", "consec_poll_failures", "poll_http_status",
                  "poll_error", "detect_errors_this_cycle", "detect_errors_total",
                  "last_exception", "last_exception_ts"):
        assert field in hb, f"missing heartbeat field: {field}"
    assert hb["loop_seq"] == 42
    assert hb["market_open"] is True
    assert hb["contract"] == "MNQU26"
    assert hb["last_bar_ts"] == "2026-07-06T14:30:00+00:00"
    assert hb["pid"] == os.getpid()


def test_heartbeat_written_when_market_closed(tmp_path, monkeypatch):
    """W2: the closed branch must still tick the heartbeat (market_open=false) so the
    healthcheck's 24/7 ts-staleness check doesn't false-page all weekend."""
    t = _bare_trader(tmp_path)
    monkeypatch.setattr(Tier2StreamingTrader, "_is_market_open", staticmethod(lambda: False))
    async def _fail_poll():
        raise AssertionError("_poll_and_process must not run while market closed")
    t._poll_and_process = _fail_poll
    async def _stop_after_first_sleep(secs):
        t.running = False
    monkeypatch.setattr(asyncio, "sleep", _stop_after_first_sleep)
    async def _stop(): pass
    t.stop = _stop
    asyncio.run(t.start_streaming())
    hb = json.loads((tmp_path / "yank_heartbeat.json").read_text())
    assert hb["market_open"] is False
    assert hb["loop_seq"] == 1


def test_detect_exception_recorded_and_swallowed(tmp_path):
    """W1: a _detect_and_enter exception is annotated for the heartbeat, then re-raised
    into _poll_and_process's EXISTING catch-all — it never escapes (trader survives)."""
    t = _bare_trader(tmp_path)
    now = datetime.now(timezone.utc)
    t._parse_bar = lambda d: _bar(now)
    t.auth, t.client = _AUTH, _client([_Resp(200, [{"raw": 1}])])
    async def _boom(bar, is_backfill):
        raise ValueError("injected detect fault")
    t._detect_and_enter = _boom
    asyncio.run(t._poll_and_process())  # must NOT raise
    assert t._hb_detect_errors_this_cycle == 1
    assert t._hb_detect_errors_total == 1
    assert t._hb_last_exception.startswith("_detect_and_enter:")
    assert "injected detect fault" in t._hb_last_exception


def test_consec_poll_failures_increment_and_reset(tmp_path):
    t = _bare_trader(tmp_path)
    now = datetime.now(timezone.utc)
    t._parse_bar = lambda d: _bar(now)
    async def _ok_detect(bar, is_backfill): pass
    t._detect_and_enter = _ok_detect
    t.auth = _AUTH
    t.client = _client([_Resp(401, []), _Resp(401, []), _Resp(200, [{"raw": 1}])])
    asyncio.run(t._poll_and_process())
    assert (t._hb_consec_poll_failures, t._hb_poll_error) == (1, "HTTP 401")
    asyncio.run(t._poll_and_process())
    assert t._hb_consec_poll_failures == 2
    asyncio.run(t._poll_and_process())  # recovery: 200 with a bar
    assert (t._hb_consec_poll_failures, t._hb_poll_error) == (0, None)
    assert t._hb_poll_http_status == 200
    assert t._hb_bars_evaluated_total == 1
    assert t._last_processed_timestamp == now


def test_timeout_counts_as_poll_failure(tmp_path):
    import httpx
    t = _bare_trader(tmp_path)
    t.auth = _AUTH
    t.client = _client([httpx.TimeoutException("slow")])
    asyncio.run(t._poll_and_process())  # must NOT raise
    assert t._hb_consec_poll_failures == 1
    assert t._hb_poll_error == "timeout"


def test_backfill_flag_in_payload(tmp_path):
    t = _bare_trader(tmp_path)
    t._is_backfill = True
    t._write_heartbeat(market_open=True)
    assert json.loads((tmp_path / "yank_heartbeat.json").read_text())["is_backfill"] is True
