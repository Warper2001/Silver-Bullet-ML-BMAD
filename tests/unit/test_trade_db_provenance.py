"""Provenance tagging on TradeDatabase.log_trade.

Why this exists: `write_mode` was added to the schema on 2026-09-02 and populated by a
one-shot migration, but the code change that lets `log_trade` WRITE it never reached the
checkout the live bots run from. Every row logged afterwards was NULL, and the NULL set
grew with each trade -- so `write_mode != 'backfilled'` silently both dropped real trades
and admitted backtest rows. No test covered the tagging, which is why a missing plumbing
change went unnoticed for nine days.

These are real DB round-trips against a tmp file, not source-inspection assertions.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.monitoring.trade_db import TradeDatabase


@pytest.fixture()
def db(tmp_path):
    return TradeDatabase(db_path=str(tmp_path / "trades.db"))


def _rows(db, *cols):
    with sqlite3.connect(db.db_path) as conn:
        return conn.execute(f"SELECT {', '.join(cols)} FROM trades").fetchall()


def test_fresh_db_has_provenance_columns(db):
    """A DB created from this code must have both columns.

    The live DB got them from a migration run directly against the file; a DB created
    from code alone did not, so the two could drift apart silently.
    """
    with sqlite3.connect(db.db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)")}
    assert "write_mode" in cols
    assert "execution_mode" in cols


def test_defaults_are_realtime_and_unknown(db):
    """An un-annotated call is what a running bot does, so it defaults to realtime.

    execution_mode deliberately defaults to 'unknown', never 'live': an honest unknown is
    recoverable; a wrong 'live' is how a paper bot's +$2,500 was read as a track record.
    """
    db.log_trade(trader_id="t", timestamp="2026-09-11T00:00:00+00:00", pnl=1.0)
    assert _rows(db, "write_mode", "execution_mode") == [("realtime", "unknown")]


def test_explicit_values_are_persisted(db):
    db.log_trade(
        trader_id="t", timestamp="2026-09-11T00:00:00+00:00", pnl=1.0,
        write_mode="backfilled", execution_mode="paper",
    )
    assert _rows(db, "write_mode", "execution_mode") == [("backfilled", "paper")]


@pytest.mark.parametrize("mode", ["live", "sim", "paper", "unknown"])
def test_each_execution_mode_round_trips(db, mode):
    db.log_trade(
        trader_id="t", timestamp=f"2026-09-11T00:00:0{len(mode)}+00:00", pnl=1.0,
        execution_mode=mode,
    )
    assert _rows(db, "execution_mode") == [(mode,)]


def test_tagging_does_not_break_idempotency(db):
    """The natural-key UNIQUE index must still collapse a replayed duplicate.

    Guards the 2026-06-19 fix (restart-replay was doubling P&L) against regression from
    the provenance columns -- they are NOT part of the identity key, so the same trade
    logged twice stays one row even if the second write claims a different provenance.
    """
    for _ in range(2):
        db.log_trade(
            trader_id="t", timestamp="2026-09-11T00:00:00+00:00", pnl=42.0,
            direction="L", entry_price=1.0, exit_price=2.0,
        )
    assert len(_rows(db, "id")) == 1

    db.log_trade(
        trader_id="t", timestamp="2026-09-11T00:00:00+00:00", pnl=42.0,
        direction="L", entry_price=1.0, exit_price=2.0,
        write_mode="backfilled", execution_mode="sim",
    )
    rows = _rows(db, "write_mode", "execution_mode")
    assert len(rows) == 1, "provenance must not widen the identity key"
    assert rows[0] == ("realtime", "unknown"), "first write wins; INSERT OR IGNORE"


def test_no_null_write_mode_is_producible_through_the_helper(db):
    """The regression this suite exists to prevent: rows landing with NULL write_mode."""
    db.log_trade(trader_id="a", timestamp="2026-09-11T00:00:00+00:00", pnl=1.0)
    db.log_trade(trader_id="b", timestamp="2026-09-11T00:01:00+00:00", pnl=-1.0,
                 execution_mode="live")
    with sqlite3.connect(db.db_path) as conn:
        n_null = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE write_mode IS NULL"
        ).fetchone()[0]
    assert n_null == 0
