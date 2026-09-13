#!/usr/bin/env python3
"""Tests for the bar-witness checks in tools/combine_ops_healthcheck.py.

Pinned: the recorder is a registered service; coverage is judged on the file for the
contract gap-fade actually trades (so a roll the recorder missed is caught at once,
not at the next open); a stale or backfilled newest bar warns only inside RTH.

Run:   .venv/bin/python -m pytest tools/test_witness_healthcheck.py -v
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
hc = importlib.import_module("combine_ops_healthcheck")

NOW = datetime(2026, 9, 14, 15, 0, 5, tzinfo=timezone.utc)
HEADER = "bar_ts,open,high,low,close,volume,fetched_at,lag_s,live,chain\n"


@pytest.fixture
def bars(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "BASE", tmp_path)
    d = tmp_path / hc.WITNESS_DIR
    d.mkdir(parents=True)

    def write(symbol, *rows):
        body = "".join(f"{ts},1,2,0,1,5,x,{lag},{live},abc\n" for ts, lag, live in rows)
        (d / f"{symbol}.csv").write_text(HEADER + body)
    return write


def stamp(seconds_ago):
    return (NOW - timedelta(seconds=seconds_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_recorder_is_a_registered_service():
    critical, log, *_ = hc.SERVICES["recorder-gap-fade-bars"]
    assert (critical, log) == (False, "ts_bar_recorder.log")


def test_newest_bar_age_and_live_flag(bars):
    bars("MNQZ26", (stamp(600), 5, 1), (stamp(60), 5, 1))
    age, live = hc.newest_witness_bar("MNQZ26", NOW)
    assert age == 60 and live


def test_missing_or_header_only_file_is_none(bars, tmp_path):
    assert hc.newest_witness_bar("MNQZ26", NOW) is None
    bars("MNQZ26")
    assert hc.newest_witness_bar("MNQZ26", NOW) is None


def test_tail_read_of_a_large_file(bars):
    bars("MNQZ26", *[(stamp(60 * i), 5, 1) for i in range(500, 0, -1)])
    assert hc.newest_witness_bar("MNQZ26", NOW)[0] == 60


def test_contract_not_recorded_warns_even_outside_rth(bars):
    bars("MNQU26", (stamp(60), 5, 1))
    level, msg = hc.witness_finding("MNQZ26", in_rth=False, now=NOW)
    assert level == hc.WARN and "RECORDER_SYMBOLS" in msg


def test_stale_bar_warns_in_rth_only(bars):
    bars("MNQZ26", (stamp(3600), 5, 1))
    assert hc.witness_finding("MNQZ26", in_rth=True, now=NOW)[0] == hc.WARN
    assert hc.witness_finding("MNQZ26", in_rth=False, now=NOW)[0] == hc.OK


def test_backfilled_newest_bar_warns_in_rth(bars):
    bars("MNQZ26", (stamp(60), 900, 0))
    level, msg = hc.witness_finding("MNQZ26", in_rth=True, now=NOW)
    assert level == hc.WARN and "live=0" in msg


def test_fresh_live_bar_is_ok(bars):
    bars("MNQZ26", (stamp(65), 5, 1))
    assert hc.witness_finding("MNQZ26", in_rth=True, now=NOW)[0] == hc.OK


def test_unknown_symbol_warns():
    assert hc.witness_finding(None, in_rth=True, now=NOW)[0] == hc.WARN


def test_unit_env_reads_one_key(monkeypatch):
    out = subprocess.CompletedProcess([], 0, stdout="GAP_FADE_SYMBOL=MNQZ26 GAP_FADE_TS_SIM=1\n")
    monkeypatch.setattr(hc.subprocess, "run", lambda *a, **k: out)
    assert hc.unit_env("trader-gap-fade", "GAP_FADE_SYMBOL") == "MNQZ26"
    assert hc.unit_env("trader-gap-fade", "NOPE") is None

    def boom(*a, **k):
        raise OSError("no systemctl")
    monkeypatch.setattr(hc.subprocess, "run", boom)
    assert hc.unit_env("trader-gap-fade", "GAP_FADE_SYMBOL") is None
