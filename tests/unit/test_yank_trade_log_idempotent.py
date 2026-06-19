"""Regression tests for TradeLogger idempotency (yank_streaming_working.py).

Root cause (diagnostic 2026-06-19): the bot replays historical bars on every
restart (backfill) and re-closes already-logged trades, so the trade log
re-appended each trade on every restart — 1,676 of 2,827 live rows were dupes,
plus 131 entries logged with conflicting backfill-recomputed exits.

Contract under test:
  1. a completed trade (keyed on timestamp_entry + direction) is logged at most once
  2. the guard survives a process restart (seen-keys seeded from the existing file)
  3. the first logged exit for an entry wins (later conflicting exits are dropped)
  4. genuinely distinct trades are all logged
"""

import csv
from datetime import datetime, timezone

import pytest

from src.research.yank_streaming_working import TradeLogger, TradeRecord


def _record(entry_minute: int, direction: str = "SHORT",
            exit_reason: str = "SL", exit_price: float = 21166.5) -> TradeRecord:
    entry = datetime(2026, 6, 1, 14, entry_minute, tzinfo=timezone.utc)
    exit_t = datetime(2026, 6, 1, 14, entry_minute + 5, tzinfo=timezone.utc)
    return TradeRecord(
        timestamp_entry=entry, timestamp_exit=exit_t, direction=direction,
        entry_price=21121.0, exit_price=exit_price, tp_price=20984.5, sl_price=21234.5,
        gap_size=22.75, pnl_usd=-486.5, exit_reason=exit_reason,
        h1_sweep_bars_ago=0, m15_confirmed=True, kill_zone_active=False,
        vol_regime_pct=0.675, contracts=2,
    )


def _rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_duplicate_same_identity_skipped(tmp_path):
    log = tmp_path / "tier2_trade_log.csv"
    tl = TradeLogger(log_path=log)
    tl.append_trade(_record(0))
    tl.append_trade(_record(0))  # identical identity → must be skipped
    rows = _rows(log)
    assert len(rows) == 1


def test_idempotent_across_restart(tmp_path):
    log = tmp_path / "tier2_trade_log.csv"
    TradeLogger(log_path=log).append_trade(_record(0))
    # simulate a process restart: a brand-new logger instance on the same file
    restarted = TradeLogger(log_path=log)
    restarted.append_trade(_record(0))  # backfill re-close of an already-logged trade
    assert len(_rows(log)) == 1


def test_same_entry_different_exit_keeps_first(tmp_path):
    log = tmp_path / "tier2_trade_log.csv"
    tl = TradeLogger(log_path=log)
    tl.append_trade(_record(0, exit_reason="SL", exit_price=21166.5))
    # a backfill re-close that recomputed a DIFFERENT exit for the same entry
    tl.append_trade(_record(0, exit_reason="TIME_STOP", exit_price=21169.25))
    rows = _rows(log)
    assert len(rows) == 1
    assert rows[0]["exit_reason"] == "SL"  # the first (live) exit wins
    assert rows[0]["exit_price"] == "21166.5"


def test_distinct_trades_all_logged(tmp_path):
    log = tmp_path / "tier2_trade_log.csv"
    tl = TradeLogger(log_path=log)
    tl.append_trade(_record(0, direction="SHORT"))
    tl.append_trade(_record(5, direction="SHORT"))      # different entry minute
    tl.append_trade(_record(0, direction="LONG"))       # same minute, different side
    rows = _rows(log)
    assert len(rows) == 3


def test_header_written_once(tmp_path):
    log = tmp_path / "tier2_trade_log.csv"
    tl = TradeLogger(log_path=log)
    tl.append_trade(_record(0))
    tl.append_trade(_record(5))
    lines = log.read_text().splitlines()
    assert lines[0].startswith("timestamp_entry,")
    assert sum(1 for ln in lines if ln.startswith("timestamp_entry,")) == 1


def test_default_path_used_when_unset():
    # constructing with no arg must not raise and must resolve the prod path
    tl = TradeLogger()
    assert tl._log_path == TradeLogger._LOG_PATH
