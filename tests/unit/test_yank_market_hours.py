"""YANK's session calendar must follow the EXCHANGE clock, not a fixed UTC offset.

Regression for 2026-09-17: `_is_market_open` compared UTC hours written in CST, so from
March to November it had the CME maintenance halt an hour late — it polled straight
through the real 16:00-17:00 CT halt and then slept through 17:00-18:00 CT, the first
hour of the new session. Every case below is stated in CT and checked in both DST and
standard time; the two are the same wall-clock rule and must give the same answer.
"""
from datetime import datetime, timezone

import pytest

from src.research import yank_streaming_working as y


def _at(monkeypatch, iso_utc: str):
    """Pin the module's clock to an instant given in UTC."""
    fixed = datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed

    monkeypatch.setattr(y, "datetime", _DT)


# (utc instant, expected open, what it is in CT)
CASES = [
    # --- DST (CDT, UTC-5): the window that was broken ---
    ("2026-09-17T20:59:00", True,  "Thu 15:59 CT — trading"),
    ("2026-09-17T21:30:00", False, "Thu 16:30 CT — the real maintenance halt"),
    ("2026-09-17T22:30:00", True,  "Thu 17:30 CT — session reopened (bot used to sleep)"),
    ("2026-09-18T13:30:00", True,  "Fri 08:30 CT — RTH open"),
    ("2026-09-18T20:59:00", True,  "Fri 15:59 CT — just before the weekly close"),
    ("2026-09-18T21:30:00", False, "Fri 16:30 CT — closed for the week"),
    ("2026-09-19T18:00:00", False, "Sat 13:00 CT — closed all day"),
    ("2026-09-20T21:30:00", False, "Sun 16:30 CT — before the weekly reopen"),
    ("2026-09-20T22:30:00", True,  "Sun 17:30 CT — weekly reopen"),
    # --- standard time (CST, UTC-6): same rule, one hour later in UTC ---
    ("2026-01-15T21:59:00", True,  "Thu 15:59 CST — trading"),
    ("2026-01-15T22:30:00", False, "Thu 16:30 CST — maintenance halt"),
    ("2026-01-15T23:30:00", True,  "Thu 17:30 CST — session reopened"),
    ("2026-01-16T22:30:00", False, "Fri 16:30 CST — closed for the week"),
    ("2026-01-18T22:30:00", False, "Sun 16:30 CST — before the weekly reopen"),
    ("2026-01-18T23:30:00", True,  "Sun 17:30 CST — weekly reopen"),
]


@pytest.mark.parametrize("utc,expected,why", CASES)
def test_session_calendar_follows_the_exchange_clock(monkeypatch, utc, expected, why):
    _at(monkeypatch, utc)
    assert y.Tier2StreamingTrader._is_market_open() is expected, why


def test_the_halt_is_the_same_wall_clock_hour_in_both_seasons(monkeypatch):
    """The bug was a seasonal shift, so pin the symmetry itself."""
    _at(monkeypatch, "2026-09-17T21:30:00")          # Thu 16:30 CDT
    summer = y.Tier2StreamingTrader._is_market_open()
    _at(monkeypatch, "2026-01-15T22:30:00")          # Thu 16:30 CST
    winter = y.Tier2StreamingTrader._is_market_open()
    assert summer is winter is False

    _at(monkeypatch, "2026-09-17T22:30:00")          # Thu 17:30 CDT
    summer_open = y.Tier2StreamingTrader._is_market_open()
    _at(monkeypatch, "2026-01-15T23:30:00")          # Thu 17:30 CST
    winter_open = y.Tier2StreamingTrader._is_market_open()
    assert summer_open is winter_open is True
