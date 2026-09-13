"""Tests for src/monitoring/ts_bar_recorder.py — gap-fade's independent bar witness.

No live credentials: auth is an AsyncMock and HTTP goes through httpx.MockTransport.
The properties pinned here are the ones the witness exists for:
  - first sighting wins; a later, different value becomes a revision row, never an edit
  - re-polling the same bars appends nothing (restart/overlap safety)
  - only closed bars are recorded
  - rows recovered after an outage are marked live=0, so the file never passes off the
    venue's current history as what the bot saw
  - the chain is the format tools/verify_chain.py checks, and it resumes across restarts
"""

from __future__ import annotations

import csv
import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))
rec = importlib.import_module("src.monitoring.ts_bar_recorder")
verify_chain = importlib.import_module("verify_chain")

T0 = datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc)


def bar(
    minute: int, close: float = 100.0, status: str | None = "Closed", vol: int = 10
) -> dict:
    ts = (T0 + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")
    b = {
        "TimeStamp": ts,
        "Open": str(close),
        "High": str(close + 1),
        "Low": str(close - 1),
        "Close": str(close),
        "TotalVolume": str(vol),
    }
    if status is not None:
        b["BarStatus"] = status
    return b


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def at(minute: int, seconds: int = 5) -> datetime:
    return T0 + timedelta(minutes=minute, seconds=seconds)


def test_records_closed_bars_once_and_chain_verifies(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    c = r.ingest([bar(0), bar(1), bar(2, status="Open")], at(2))
    assert c["appended"] == 2 and c["skipped_open"] == 1
    again = r.ingest([bar(0), bar(1), bar(2)], at(3))  # overlap: only minute 2 is new
    assert again["appended"] == 1 and again["revisions"] == 0
    path = tmp_path / "MNQZ26.csv"
    assert [x["bar_ts"] for x in rows(path)] == [bar(i)["TimeStamp"] for i in range(3)]
    n, bad, _, err = verify_chain.verify(path)
    assert (n, bad, err) == (3, None, None)


def test_bars_without_status_are_trusted_only_once_a_minute_old(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    c = r.ingest([bar(0, status=None), bar(2, status=None)], at(2, seconds=30))
    assert c["appended"] == 1 and c["skipped_open"] == 1


def test_later_disagreement_is_a_revision_row_not_an_edit(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    r.ingest([bar(0, close=100.0)], at(1))
    c = r.ingest([bar(0, close=100.5)], at(2))  # the venue revised minute 0
    assert c["revisions"] == 4  # open, high, low, close all moved
    assert (
        rows(tmp_path / "MNQZ26.csv")[0]["close"] == "100.0"
    )  # the witness keeps the first sighting
    revs = rows(tmp_path / "MNQZ26_revisions.csv")
    assert {x["field"] for x in revs} == {"open", "high", "low", "close"}
    assert all(x["first_seen"] != x["now_seen"] for x in revs)
    assert (
        r.ingest([bar(0, close=100.5)], at(3))["revisions"] == 0
    )  # same disagreement: not re-logged
    assert (
        r.ingest([bar(0, close=101.0)], at(4))["revisions"] == 4
    )  # a new value: logged again
    assert verify_chain.verify(tmp_path / "MNQZ26_revisions.csv")[1] is None


def test_equal_values_in_different_text_are_not_revisions(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    r.ingest([bar(0, close=100.0)], at(1))
    b = bar(0, close=100.0)
    b["Close"] = "100.00"
    assert r.ingest([b], at(2))["revisions"] == 0


def test_rows_recovered_after_an_outage_are_marked_not_live(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    r.ingest([bar(0)], at(1))
    c = r.ingest([bar(1), bar(40)], at(41))  # minute 1 fetched 40 minutes late
    assert c["backfilled"] == 1
    out = rows(tmp_path / "MNQZ26.csv")
    assert (out[1]["live"], out[2]["live"]) == ("0", "1")
    assert int(out[1]["lag_s"]) > rec.LIVE_LAG_S


def test_restart_resumes_the_chain_and_the_last_minute(tmp_path):
    r1 = rec.BarRecorder("MNQZ26", tmp_path)
    r1.ingest([bar(0), bar(1)], at(2))
    r2 = rec.BarRecorder("MNQZ26", tmp_path)  # new process, same files
    assert r2.last_ts == T0 + timedelta(minutes=1)
    assert r2.ingest([bar(0), bar(1), bar(2)], at(3))["appended"] == 1
    assert (
        r2.ingest([bar(1, close=99.0)], at(4))["revisions"] == 4
    )  # revision memory survives restart
    n, bad, _, _ = verify_chain.verify(tmp_path / "MNQZ26.csv")
    assert (n, bad) == (3, None)


def test_revision_memory_survives_restart_without_relogging(tmp_path):
    r1 = rec.BarRecorder("MNQZ26", tmp_path)
    r1.ingest([bar(0, close=100.0)], at(1))
    r1.ingest([bar(0, close=100.5)], at(2))
    r2 = rec.BarRecorder("MNQZ26", tmp_path)
    assert r2.ingest([bar(0, close=100.5)], at(3))["revisions"] == 0


def test_wrong_header_refuses_to_start(tmp_path):
    (tmp_path / "MNQZ26.csv").write_text("minute,open,chain\n")
    with pytest.raises(RuntimeError, match="header"):
        rec.BarRecorder("MNQZ26", tmp_path)


def test_barsback_covers_the_gap_within_bounds(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    assert r.barsback(at(0)) == rec.MAX_BARSBACK  # nothing recorded yet
    r.ingest([bar(0)], at(1))
    assert r.barsback(at(1)) == rec.MIN_BARSBACK
    assert r.barsback(at(60)) == 61  # 59 min since caught up
    r2 = rec.BarRecorder("MNQZ26", tmp_path)  # restart: sized from the bar
    assert r2.barsback(at(60)) == 62
    assert r.barsback(at(5000)) == rec.MAX_BARSBACK


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_fetch_uses_gap_fades_endpoint_and_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = request.url, request.headers["Authorization"]
        return httpx.Response(200, json={"Bars": [bar(0)]})

    auth = AsyncMock()
    auth.authenticate.return_value = "tok"
    async with _mock_client(handler) as http:
        out = await rec.fetch_bars(http, auth, "MNQZ26", 17)
    assert out == [bar(0)]
    assert str(seen["url"]).startswith(f"{rec.TS_BARS_BASE}/MNQZ26?")
    assert dict(seen["url"].params) == {
        "interval": "1",
        "unit": "Minute",
        "barsback": "17",
    }
    assert seen["auth"] == "Bearer tok"


@pytest.mark.asyncio
async def test_fetch_failures_return_none_instead_of_raising():
    auth = AsyncMock()
    auth.authenticate.return_value = "tok"
    async with _mock_client(lambda req: httpx.Response(503)) as http:
        assert await rec.fetch_bars(http, auth, "MNQZ26", 5) is None
    auth.authenticate.side_effect = RuntimeError("token refresh failed")
    async with _mock_client(lambda req: httpx.Response(200, json={"Bars": []})) as http:
        assert await rec.fetch_bars(http, auth, "MNQZ26", 5) is None


@pytest.mark.asyncio
async def test_poll_once_records_each_symbol_and_survives_a_bad_payload(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "MNQU26" in request.url.path:
            return httpx.Response(200, json={"Bars": [{"TimeStamp": "not-a-time"}]})
        return httpx.Response(200, json={"Bars": [bar(0), bar(1)]})

    auth = AsyncMock()
    auth.authenticate.return_value = "tok"
    recs = {s: rec.BarRecorder(s, tmp_path) for s in ("MNQU26", "MNQZ26")}
    async with _mock_client(handler) as http:
        out = await rec.poll_once(recs, http, auth, now=lambda: at(2))
    assert out["MNQU26"] == "ingest_failed"
    assert out["MNQZ26"]["appended"] == 2


def test_config_requires_symbols_and_defaults_to_the_gitignored_bars_dir():
    with pytest.raises(SystemExit, match="RECORDER_SYMBOLS"):
        rec.config_from_env({})
    symbols, out_dir = rec.config_from_env({"RECORDER_SYMBOLS": "MNQU26, MNQZ26"})
    assert symbols == ["MNQU26", "MNQZ26"]
    assert out_dir == rec.BASE_DIR / "data" / "gap_fade" / "bars"


def test_next_poll_lands_five_seconds_after_the_minute():
    assert (
        rec.seconds_to_next_poll(datetime(2026, 9, 14, 12, 0, 30, tzinfo=timezone.utc))
        == 35
    )
    assert (
        rec.seconds_to_next_poll(datetime(2026, 9, 14, 12, 0, 4, tzinfo=timezone.utc))
        == 61
    )


def test_closed_market_keeps_requests_small(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    r.ingest([bar(0), bar(1)], at(2))  # Friday's last bars
    for m in range(3, 3000):  # the weekend: nothing new
        r.ingest([bar(0), bar(1)], at(m))
    assert r.barsback(at(3000)) == rec.MIN_BARSBACK


def test_a_response_that_misses_the_held_minute_is_not_caught_up(tmp_path):
    r = rec.BarRecorder("MNQZ26", tmp_path)
    r.ingest([bar(0)], at(1))
    r.ingest([bar(50)], at(51))  # minutes 1..49 never arrived
    assert r.caught_up_at == at(1)
    assert r.barsback(at(52)) == 5  # sized from minute 50
