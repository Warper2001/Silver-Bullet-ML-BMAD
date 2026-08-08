#!/usr/bin/env python3
"""Tests for GAP-1's double-entry guard (`_already_decided_today`).

Why this file exists: on 2026-06-25 the bot logged the same session twice — two
chained appends to trades.csv and decisions.csv across a 23:54:55 stop / 23:56:26
restart, on its first day, before any guard existed. The guard was added two days
later (7c9bc0a) and has held for 17 subsequent sessions, but it was never tested,
and it FAILED OPEN on any read error: `except (FileNotFoundError, Exception): pass`
returned False, silently re-enabling the exact defect it exists to prevent.

This guard is the only thing between a restart and a second live order for the
session. On a funded account that is a position, not a log row. So it gets tests.

Run:   .venv/bin/python -m pytest tools/test_gap_fade_double_entry_guard.py -v
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

gf = importlib.import_module("src.research.gap_fade_live")

FIELDS = ["date_et", "dow", "gap_pct", "gap_abs_pts", "prior_close",
          "rth_open", "action", "detail", "chain"]


def write_decisions(path: Path, dates) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for d in dates:
            w.writerow({"date_et": d, "dow": "Thu", "gap_pct": "2.359",
                        "gap_abs_pts": "695.25", "prior_close": "29466.25",
                        "rth_open": "30161.5", "action": "ENTERED",
                        "detail": "short", "chain": "deadbeefdeadbeef"})


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    """A bare trader whose DATA_DIR points at a tmp dir (no network, no __init__)."""
    monkeypatch.setattr(gf, "DATA_DIR", tmp_path)
    t = gf.GapFadeTrader.__new__(gf.GapFadeTrader)   # no broker, no auth
    return t, tmp_path / "decisions.csv"


def test_first_run_no_file_is_not_decided(guard):
    """The genuine first-run case — and the ONLY case that may answer 'no'."""
    t, path = guard
    assert not path.exists()
    assert t._already_decided_today("2026-06-25") is False


def test_header_only_file_is_not_decided(guard):
    t, path = guard
    write_decisions(path, [])
    assert t._already_decided_today("2026-06-25") is False


def test_todays_decision_is_detected(guard):
    """The 2026-06-25 restart: the row was on disk and must stop a second entry."""
    t, path = guard
    write_decisions(path, ["2026-06-25"])
    assert t._already_decided_today("2026-06-25") is True


def test_other_dates_do_not_block_today(guard):
    """A guard that blocks every session is as useless as one that blocks none."""
    t, path = guard
    write_decisions(path, ["2026-06-23", "2026-06-24"])
    assert t._already_decided_today("2026-06-25") is False


def test_duplicate_rows_still_report_decided(guard):
    """The corrupt historical file must not confuse the guard."""
    t, path = guard
    write_decisions(path, ["2026-06-25", "2026-06-25"])
    assert t._already_decided_today("2026-06-25") is True


def test_unreadable_file_fails_CLOSED(guard, monkeypatch):
    """The regression this file was written for.

    Old behaviour: any exception -> False -> the bot enters a second time.
    New behaviour: unreadable evidence -> True -> the bot stands down.
    """
    t, path = guard
    write_decisions(path, ["2026-06-25"])

    def boom(*a, **k):
        raise OSError("simulated I/O error")

    monkeypatch.setattr("builtins.open", boom)
    assert t._already_decided_today("2026-06-25") is True


def test_malformed_csv_fails_CLOSED(guard, monkeypatch):
    t, path = guard
    path.write_bytes(b"\xff\xfe\x00garbage\x00not,a,csv\n")

    real_reader = csv.DictReader

    class Exploding(real_reader):
        def __next__(self):
            raise csv.Error("simulated parse failure")

    monkeypatch.setattr(csv, "DictReader", Exploding)
    assert t._already_decided_today("2026-06-25") is True


def test_a_directory_where_the_file_should_be_fails_CLOSED(guard):
    """exists() is True, open() raises IsADirectoryError — must not fall through."""
    t, path = guard
    path.mkdir()
    assert t._already_decided_today("2026-06-25") is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
