#!/usr/bin/env python3
"""
trades.db provenance migration.

WHY THIS EXISTS (2026-09-02)
----------------------------
The shop's system of record mixed three kinds of row in one table with no way to
tell them apart:

  * rows written by a live bot as the trade happened,
  * rows written by a backfill/replay long after the fact,
  * rows from a bot that was never actually trading real money.

Consequence found on 2026-09-02: `trader-yank` showed +$102,151.90 in the ledger.
1,841 of its 1,846 rows were written a MEDIAN of 207 days after their own trade
timestamp -- a replay, not a track record. Meanwhile `trader-btc-carry` showed a
single row while its executor had been running in PAPER mode for three months.

Neither defect was a missing gate. Both were a missing LABEL.

WHAT IT ADDS
------------
  write_mode      how the row got here, derived MECHANICALLY from evidence:
                    'realtime'   created_at within REALTIME_LAG_H of timestamp
                    'backfilled' created_at >= REALTIME_LAG_H after timestamp
                    'unknown'    created_at missing -- never guessed

  execution_mode  what kind of money was at risk. Set per-trader ONLY where the
                  shop has a documented answer; everything else stays 'unknown'.
                  An honest 'unknown' beats a plausible fabrication -- that is the
                  whole lesson of the BTC-CARRY incident.

Idempotent: safe to re-run. Backs the database up before touching it.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_DEFAULT = Path("data/trades.db")

# A live writer logs within minutes. A replay logs days-to-months later.
# 24h is far outside normal live jitter and far inside observed backfill lag
# (smallest observed backfill on 2026-09-02 was 30.5h; largest live lag 10.0h).
REALTIME_LAG_H = 24.0

# Documented execution venue per trader. Anything absent here stays 'unknown'.
# Sources: repo records + direct verification 2026-09-02.
EXECUTION_MODE = {
    # verified 2026-09-02: unit file runs btc_carry_executor.py with NO --live
    # flag; 250,414 log lines marked [PAPER], zero marked [LIVE].
    "trader-btc-carry": "paper",
    # TradeStation SIM account, symbol-isolated (never promoted to the combine).
    "trader-gap-fade": "sim",
    # Live Topstep combine account.
    "trader-mim-nb": "live",
    "trader-s26-combine": "live",
    # trader-yank / trader-s26 / trader-s27 deliberately omitted: their rows span
    # both replayed history and live trading, so the venue is per-row, not
    # per-trader. write_mode separates those; execution_mode stays 'unknown'
    # until someone establishes it from evidence rather than memory.
}


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r[1] == col for r in conn.execute(f"PRAGMA table_info({table})"))


def migrate(db_path: Path, apply: bool) -> int:
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist", file=sys.stderr)
        return 2

    if apply:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = db_path.with_suffix(f".db.pre-provenance.{stamp}.bak")
        shutil.copy2(db_path, backup)
        print(f"backup written: {backup}")

    conn = sqlite3.connect(db_path)
    try:
        for col in ("write_mode", "execution_mode"):
            if not column_exists(conn, "trades", col):
                print(f"{'ADD' if apply else 'would add'} column trades.{col}")
                if apply:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT")
            else:
                print(f"column trades.{col} already present")

        # --- write_mode: mechanical, from the created_at/timestamp lag ---
        lag_h = "(julianday(created_at) - julianday(timestamp)) * 24.0"
        rules = [
            ("unknown", "created_at IS NULL OR timestamp IS NULL"),
            (
                "realtime",
                f"created_at IS NOT NULL AND timestamp IS NOT NULL AND {lag_h} < {REALTIME_LAG_H}",
            ),
            (
                "backfilled",
                f"created_at IS NOT NULL AND timestamp IS NOT NULL AND {lag_h} >= {REALTIME_LAG_H}",
            ),
        ]
        for mode, cond in rules:
            n = conn.execute(f"SELECT COUNT(*) FROM trades WHERE ({cond})").fetchone()[0]
            print(f"  write_mode='{mode}': {n:>5} rows")
            if apply:
                conn.execute(f"UPDATE trades SET write_mode = ? WHERE ({cond})", (mode,))

        # --- execution_mode: documented facts only; never inferred ---
        if apply:
            conn.execute(
                "UPDATE trades SET execution_mode = 'unknown' WHERE execution_mode IS NULL"
            )
        for trader, mode in EXECUTION_MODE.items():
            n = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE trader_id = ?", (trader,)
            ).fetchone()[0]
            if n:
                print(f"  execution_mode='{mode}': {n:>5} rows ({trader})")
                if apply:
                    conn.execute(
                        "UPDATE trades SET execution_mode = ? WHERE trader_id = ?",
                        (mode, trader),
                    )
        placeholders = ",".join("?" * len(EXECUTION_MODE))
        n_unknown = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE trader_id NOT IN ({placeholders})",
            tuple(EXECUTION_MODE),
        ).fetchone()[0]
        print(
            f"  execution_mode='unknown': {n_unknown:>5} rows (venue not established from evidence)"
        )

        if apply:
            conn.commit()
            print("\ncommitted.")
        else:
            print("\nDRY RUN -- nothing written. Re-run with --apply.")

        # --- audit: what the ledger now says ---
        # On a dry run against a not-yet-migrated database the columns do not
        # exist, so there is nothing to audit. Say so rather than crash.
        if not (column_exists(conn, "trades", "write_mode")
                and column_exists(conn, "trades", "execution_mode")):
            print("\n(columns not present yet -- audit runs after --apply)")
            return 0

        print("\n=== post-migration ledger, honestly grouped ===")
        q = """
            SELECT trader_id,
                   COALESCE(write_mode,'(unset)')     AS wmode,
                   COALESCE(execution_mode,'(unset)') AS emode,
                   COUNT(*) n, ROUND(SUM(pnl),2) pnl
            FROM trades GROUP BY trader_id, wmode, emode ORDER BY trader_id, n DESC
        """
        print(f"{'trader':<22}{'write':<12}{'exec':<10}{'n':>6}{'pnl':>14}")
        for tid, wm, em, n, pnl in conn.execute(q):
            print(f"{tid:<22}{wm:<12}{em:<10}{n:>6}{pnl:>14,.2f}")

        print("\n=== the number that started this: realtime rows only ===")
        for tid, n, pnl in conn.execute(
            "SELECT trader_id, COUNT(*), ROUND(SUM(pnl),2) FROM trades "
            "WHERE write_mode='realtime' GROUP BY trader_id ORDER BY 3 DESC"
        ):
            print(f"  {tid:<22} n={n:<5} pnl={pnl:>12,.2f}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", type=Path, default=DB_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    a = ap.parse_args()
    raise SystemExit(migrate(a.db, a.apply))
