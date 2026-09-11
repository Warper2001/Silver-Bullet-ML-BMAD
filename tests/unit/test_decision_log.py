"""Shared per-bar decision log (src/research/decision_log.py).

Three bots append to one file. Before this module they each carried their own copy of
the writer and the copies drifted -- only YANK had the backfill guard, so the other two
re-logged historical bars on every restart until the file reached 1.66 GB. These tests
pin the three properties that made the drift expensive: the guard, the trader stamp,
and the reason vocabulary.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import pytest

from src.research import decision_log


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Point the module at a tmp file — never touch the real shared log."""
    monkeypatch.setattr(decision_log, "LOG_PATH", tmp_path / "decisions.csv")
    monkeypatch.setattr(decision_log, "ARCHIVE_DIR", tmp_path / "archive")
    return tmp_path


def _rows():
    with decision_log.LOG_PATH.open() as f:
        return list(csv.DictReader(f))


def _ts(minute=0):
    return datetime(2026, 9, 11, 12, minute, tzinfo=timezone.utc)


def test_writes_header_and_row():
    decision_log.append_decision(
        trader_id="trader-yank", bar_timestamp=_ts(), action="SKIP",
        rejection_reason="vol_regime", vol_regime_blocked=True, vol_regime_pct=0.91,
    )
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["trader_id"] == "trader-yank"
    assert rows[0]["action"] == "SKIP"
    assert rows[0]["rejection_reason"] == "vol_regime"
    assert rows[0]["vol_regime_blocked"] == "True"
    assert rows[0]["vol_regime_pct"] == "0.91"


def test_backfill_bars_are_never_logged():
    """The guard that existed in only one of three copies, and cost 1.66 GB."""
    for _ in range(50):
        decision_log.append_decision(
            trader_id="trader-tier2", bar_timestamp=_ts(), action="SKIP",
            is_backfill=True,
        )
    assert not decision_log.LOG_PATH.exists()


def test_rows_from_different_bots_are_attributable():
    for tid in ("trader-yank", "trader-tier2", "trader-btc-combine"):
        decision_log.append_decision(
            trader_id=tid, bar_timestamp=_ts(), action="SKIP", rejection_reason="no_fvg",
        )
    assert [r["trader_id"] for r in _rows()] == [
        "trader-yank", "trader-tier2", "trader-btc-combine"
    ]


def test_absent_percentile_writes_blank_not_zero():
    """0.0 is a real percentile; a missing one must not masquerade as the lowest."""
    decision_log.append_decision(
        trader_id="t", bar_timestamp=_ts(), action="SKIP", vol_regime_pct=None,
    )
    assert _rows()[0]["vol_regime_pct"] == ""


def test_schema_change_rotates_instead_of_corrupting():
    """An old-schema file must be archived, not appended under a mismatched header.

    Appending 10 columns of data beneath a 7-column header produces a file that parses
    without error and means nothing -- the worst possible failure for an audit trail.
    """
    decision_log.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with decision_log.LOG_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar_timestamp", "h1_sweep_active", "action"])  # old, narrower
        w.writerow(["2026-09-10T00:00:00+00:00", "True", "SKIP"])

    decision_log.append_decision(
        trader_id="trader-yank", bar_timestamp=_ts(), action="ENTER",
        rejection_reason="entered",
    )

    rows = _rows()
    assert len(rows) == 1, "fresh file after rotation"
    assert rows[0]["action"] == "ENTER"
    archived = list((decision_log.ARCHIVE_DIR).glob("*_preschema.csv"))
    assert len(archived) == 1, "the old rows must survive, not be discarded"
    assert "2026-09-10" in archived[0].read_text()


def test_matching_header_is_appended_not_rotated():
    for i in range(3):
        decision_log.append_decision(
            trader_id="t", bar_timestamp=_ts(i), action="SKIP", rejection_reason="no_fvg",
        )
    assert len(_rows()) == 3
    assert not (decision_log.ARCHIVE_DIR).exists() or not list(
        decision_log.ARCHIVE_DIR.glob("*_preschema.csv")
    )


def test_reason_vocabulary_covers_the_gate_chain():
    """analyze_filter_funnel.py groups on these; drift between them is silent."""
    for required in ("vol_regime", "no_fvg", "fvg_wrong_direction", "lr_regime",
                     "ml_threshold", "no_sweep_or_choch", "tuesday", "daily_breaker"):
        assert required in decision_log.REASONS


def test_write_failure_never_raises(monkeypatch):
    """Telemetry must not be able to kill a trading loop."""
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(decision_log.Path, "open", boom)
    decision_log.append_decision(trader_id="t", bar_timestamp=_ts(), action="SKIP")
