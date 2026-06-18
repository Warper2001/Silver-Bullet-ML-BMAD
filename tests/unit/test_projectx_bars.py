"""Unit tests for the ProjectX market-data adapter (src/research/projectx_bars.py).

Verifies the adapter emits TradeStation-shaped 1-min bar dicts with the +1-min
labeling offset applied, is a drop-in for the TradeStation bar shape both live bots
parse, and fails closed (ProjectXBarFetchError) on transport / schema errors.
Mirrors the mock-http pattern of tests/unit/test_projectx_commingling.py.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.research.projectx_bars import (
    PX_LABEL_OFFSET_MIN,
    ProjectXBarFetchError,
    fetch_px_ts_shaped,
)

NOW = datetime(2026, 6, 18, 3, 0, 0, tzinfo=timezone.utc)
CID = "CON.F.US.MNQ.U26"


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class _Auth:
    async def authenticate(self):
        return "tok"


class _Http:
    """Returns a canned retrieveBars payload. `bars` is a raw ProjectX bar list
    (newest-first, like the real API). Override status/payload to test errors."""
    def __init__(self, bars=None, status_code=200, payload=None):
        self._resp = _Resp(status_code, payload if payload is not None
                           else {"success": True, "bars": bars or []})
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return self._resp


def _raw(t_iso, o, h, l, c, v):
    return {"t": t_iso, "o": o, "h": h, "l": l, "c": c, "v": v}


async def _fetch(http, **kw):
    return await fetch_px_ts_shaped(http, _Auth(), CID, now_utc=NOW, live=False, **kw)


@pytest.mark.asyncio
async def test_plus_one_minute_offset_applied():
    # ProjectX open-labels a bar at 02:13; TradeStation close-labels it 02:14.
    http = _Http([_raw("2026-06-18T02:13:00+00:00", 100.0, 101.0, 99.0, 100.5, 50)])
    out = await _fetch(http, since_utc=NOW - timedelta(minutes=30))
    assert PX_LABEL_OFFSET_MIN == 1
    assert out[0]["TimeStamp"] == "2026-06-18T02:14:00Z"


@pytest.mark.asyncio
async def test_tradestation_shaped_keys_and_types():
    http = _Http([_raw("2026-06-18T02:13:00+00:00", 100.0, 101.25, 99.5, 100.5, 50)])
    out = await _fetch(http, since_utc=NOW - timedelta(minutes=30))
    bar = out[0]
    assert set(bar) == {"TimeStamp", "Open", "High", "Low", "Close", "TotalVolume"}
    assert bar["TimeStamp"].endswith("Z")
    assert isinstance(bar["Open"], float) and isinstance(bar["TotalVolume"], int)


@pytest.mark.asyncio
async def test_dropin_equivalence_with_tradestation_bar():
    """A PX bar (open-labeled, t-1min) must yield exactly the TradeStation dict
    (close-labeled) for the same OHLCV — proving downstream parse is identical."""
    ts_bar = {"TimeStamp": "2026-06-18T02:14:00Z", "Open": 100.0, "High": 101.0,
              "Low": 99.0, "Close": 100.5, "TotalVolume": 50}
    http = _Http([_raw("2026-06-18T02:13:00+00:00", 100.0, 101.0, 99.0, 100.5, 50)])
    out = await _fetch(http, since_utc=NOW - timedelta(minutes=30))
    assert out[0] == ts_bar


@pytest.mark.asyncio
async def test_future_or_partial_bar_dropped():
    # A bar whose +1min close-time exceeds now_utc is incomplete -> dropped.
    http = _Http([
        _raw("2026-06-18T02:58:00+00:00", 1, 2, 0, 1, 10),   # -> 02:59Z  (<= now, kept)
        _raw("2026-06-18T03:00:00+00:00", 1, 2, 0, 1, 10),   # -> 03:01Z  (> now, dropped)
    ])
    out = await _fetch(http, since_utc=NOW - timedelta(minutes=30))
    stamps = [b["TimeStamp"] for b in out]
    assert "2026-06-18T02:59:00Z" in stamps
    assert "2026-06-18T03:01:00Z" not in stamps


@pytest.mark.asyncio
async def test_sorted_ascending_even_when_api_returns_newest_first():
    http = _Http([
        _raw("2026-06-18T02:16:00+00:00", 1, 2, 0, 1, 1),
        _raw("2026-06-18T02:14:00+00:00", 1, 2, 0, 1, 1),
        _raw("2026-06-18T02:15:00+00:00", 1, 2, 0, 1, 1),
    ])
    out = await _fetch(http, since_utc=NOW - timedelta(minutes=30))
    stamps = [b["TimeStamp"] for b in out]
    assert stamps == sorted(stamps)


@pytest.mark.asyncio
async def test_barsback_trims_to_n():
    bars = [_raw(f"2026-06-18T02:{m:02d}:00+00:00", 1, 2, 0, 1, 1) for m in range(0, 20)]
    out = await _fetch(_Http(bars), barsback=5)
    assert len(out) == 5
    # kept the most-recent 5
    assert out[-1]["TimeStamp"] == "2026-06-18T02:20:00Z"


@pytest.mark.asyncio
async def test_empty_bars_returns_empty_list():
    out = await _fetch(_Http([]), since_utc=NOW - timedelta(minutes=30))
    assert out == []


@pytest.mark.asyncio
async def test_success_false_raises():
    http = _Http(payload={"success": False, "errorCode": 7, "errorMessage": "nope"})
    with pytest.raises(ProjectXBarFetchError):
        await _fetch(http, since_utc=NOW - timedelta(minutes=30))


@pytest.mark.asyncio
async def test_http_500_raises():
    http = _Http(status_code=500, payload={})
    with pytest.raises(ProjectXBarFetchError):
        await _fetch(http, since_utc=NOW - timedelta(minutes=30))


@pytest.mark.asyncio
async def test_malformed_bar_schema_raises():
    http = _Http(payload={"success": True, "bars": [{"t": "2026-06-18T02:13:00+00:00", "o": 1}]})
    with pytest.raises(ProjectXBarFetchError):
        await _fetch(http, since_utc=NOW - timedelta(minutes=30))
