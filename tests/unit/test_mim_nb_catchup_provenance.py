"""Catch-up open anchor — prereg sigma-provenance Amendment 2 (B-G1..B-G4).

`_catch_up_today()` rebuilt today's state from a live TradeStation fetch. `open_d` anchors
every entry in `today_moves`, and `today_moves` is what the 16:00 fold appends to sigma
history — so a mid-session restart made the day's sigma contribution depend on what the
API happened to return at restart time. Measured live: the bot used open_d 27914.75 on
07-29 and 28576.25 on 07-31 while its own bars_raw.csv held 27915.25 and 28576.00; the one
session with no restart matched exactly.

These tests pin that catch-up reads the bot's own record and makes no network call.
"""
from datetime import datetime, timezone

import pytest
import pytz

from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive, RTH_FIRST

ET = pytz.timezone("America/New_York")
DAY = "2026-07-31"


def _fixed_now(hm="14:00"):
    """A datetime subclass pinned to `hm` ET on DAY, so catch-up believes it is mid-session.
    Subclassed (not mocked) so fromisoformat still works inside the reader."""
    h, m = (int(x) for x in hm.split(":"))
    pinned = ET.localize(datetime(2026, 7, 31, h, m))

    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return pinned.astimezone(tz) if tz else pinned.replace(tzinfo=None)

    return _FakeDT


def _write_bars(path, marks, day=DAY):
    """bars_raw-shaped CSV. marks = [(hm, open, close, volume), ...]. ET -> UTC is +4h."""
    lines = ["ts_utc,open,high,low,close,volume,received_at,chain"]
    for hm, o, c, v in marks:
        h, m = hm.split(":")
        ts = f"{day}T{int(h) + 4:02d}:{m}:00Z"
        lines.append(f"{ts},{o},{max(o, c) + 1},{min(o, c) - 1},{c},{v},{ts},abc")
    path.write_text("\n".join(lines) + "\n")


def _bot():
    o = object.__new__(MimNbLive)
    o.day = None
    o.open_d = None
    o.today_saw_close = False
    o.cum_pv = 0.0
    o.cum_v = 0.0
    o.today_moves = {}
    o.day_pnl = 0.0
    o.day_deactivated = False
    o.last_bar_ts = None

    async def _boom(*a, **k):                      # any network call fails the test
        raise AssertionError("_catch_up_today made a network call")
    o._ts_get_bars = _boom
    return o


MARKS = [(RTH_FIRST, 28576.00, 28598.75, 10),
         ("10:00", 28590.0, 28610.0, 20),
         ("14:00", 28600.0, 28620.0, 30)]


class TestNoNetworkAndRecordAnchor:
    """B-G1 — open_d comes from the record, exactly, with no fetch."""

    @pytest.mark.asyncio
    async def test_open_d_matches_recorded_0931_open(self, tmp_path, monkeypatch):
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, MARKS)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "datetime", _fixed_now("14:00"))

        bot = _bot()
        await bot._catch_up_today()

        assert bot.open_d == 28576.00, "open anchor must equal the recorded 09:31 open"
        assert bot.today_moves[RTH_FIRST] == abs(28598.75 / 28576.00 - 1.0)
        assert len(bot.today_moves) == len(MARKS)

    @pytest.mark.asyncio
    async def test_vwap_accumulators_rebuilt_from_record(self, tmp_path, monkeypatch):
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, MARKS)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "datetime", _fixed_now("14:00"))

        bot = _bot()
        await bot._catch_up_today()

        assert bot.cum_v == sum(v for _hm, _o, _c, v in MARKS)
        assert bot.cum_pv == pytest.approx(sum(c * v for _hm, _o, c, v in MARKS))

    @pytest.mark.asyncio
    async def test_last_bar_ts_is_last_recorded_bar(self, tmp_path, monkeypatch):
        """B-G4 — the poll loop must resume exactly where the record ends."""
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, MARKS)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "datetime", _fixed_now("14:00"))

        bot = _bot()
        await bot._catch_up_today()

        assert bot.last_bar_ts == f"{DAY}T18:00:00Z"   # 14:00 ET


class TestStandDownPaths:
    """B-G2 / B-G3 — refuse the session rather than reconstruct it from the network."""

    @pytest.mark.asyncio
    async def test_no_record_stands_down(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.setattr(M, "BARS_RAW_CSV", tmp_path / "absent.csv")
        monkeypatch.setattr(M, "datetime", _fixed_now("14:00"))

        bot = _bot()
        with caplog.at_level(logging.WARNING):
            await bot._catch_up_today()

        assert bot.open_d is None, "must not trade a session it cannot reconstruct"
        assert bot.today_moves == {}, "a non-reconstructable day must not reach sigma"
        assert "CATCHUP_NO_RECORD" in caplog.text

    @pytest.mark.asyncio
    async def test_record_missing_the_open_stands_down(self, tmp_path, monkeypatch):
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, [("10:00", 28590.0, 28610.0, 20)])   # no 09:31
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "datetime", _fixed_now("14:00"))

        bot = _bot()
        await bot._catch_up_today()
        assert bot.open_d is None and bot.today_moves == {}

    @pytest.mark.asyncio
    async def test_outside_rth_is_a_noop(self, tmp_path, monkeypatch):
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, MARKS)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "datetime", _fixed_now("17:30"))

        bot = _bot()
        await bot._catch_up_today()
        assert bot.day is None and bot.open_d is None


class TestReaderIsolation:
    """The new reader must not disturb other days or non-RTH rows."""

    def test_reader_filters_to_the_requested_day_and_rth(self, tmp_path):
        p = tmp_path / "bars_raw.csv"
        lines = ["ts_utc,open,high,low,close,volume,received_at,chain",
                 f"2026-07-30T13:31:00Z,1,2,0,1,1,x,c",      # prior day 09:31 ET
                 f"{DAY}T12:00:00Z,9,9,9,9,1,x,c",           # 08:00 ET — pre-RTH
                 f"{DAY}T13:31:00Z,10,11,9,10,5,x,c",        # 09:31 ET
                 f"{DAY}T21:00:00Z,7,7,7,7,1,x,c"]           # 17:00 ET — post-RTH
        p.write_text("\n".join(lines) + "\n")

        rows = MimNbLive._read_recorded_day(p, datetime(2026, 7, 31).date())
        assert [r[0] for r in rows] == [RTH_FIRST]
        assert rows[0][1] == 10.0 and rows[0][3] == 5.0

    def test_missing_file_returns_empty(self, tmp_path):
        rows = MimNbLive._read_recorded_day(tmp_path / "nope.csv", datetime(2026, 7, 31).date())
        assert rows == []
