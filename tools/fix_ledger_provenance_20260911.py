#!/usr/bin/env python3
"""
Two ledger-provenance corrections to data/trades.db, from the 2026-09-10 fleet audit.

(1) trader-s26-combine is PAPER, not LIVE.
    All of its rows carry execution_mode='live' (72) or NULL (9). The bot stamps
    metadata['paper']=True on every closed trade and only sends broker orders when
    S26_COMBINE_PLACE_ORDERS=1, which is unset; its systemd unit is even titled
    "S26 Combine Bot MBTM26 Paper Trader". The wrong value came from a hardcoded
    dict in tools/migrate_trades_provenance.py (corrected in the same change) that
    read the trader's NAME rather than checking it. Its +$2,500 is the largest
    figure in the ledger and read as a live track record.

(2) write_mode was never maintained, so every row written after the one-shot
    provenance migration is NULL and the set grows with each trade.
    Backfilled here using the SAME rule the original migration used, so the tagging
    stays consistent with the ~2,195 rows already tagged:
        realtime   if (created_at - timestamp) <  24h
        backfilled if (created_at - timestamp) >= 24h
        unknown    if either timestamp is NULL
    This correctly tags the 8 legacy 2025 trader-yank rows as 'backfilled' and the
    September 2026 rows as 'realtime'.

    The durable fix is in src/monitoring/trade_db.py (log_trade now takes and writes
    write_mode/execution_mode) plus the six bot call sites. This script only cleans
    up rows written while that plumbing was missing from the live checkout.

Idempotent and safe to re-run. Takes its own timestamped backup before writing.
Run with --apply to write; default is a dry run.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB = Path("/root/Silver-Bullet-ML-BMAD/data/trades.db")
REALTIME_LAG_H = 24.0  # mirrors tools/migrate_trades_provenance.py

LAG_SQL = "(julianday(created_at) - julianday(timestamp)) * 24.0"

# Mirrors EXECUTION_MODE in tools/migrate_trades_provenance.py (with the 2026-09-11
# s26-combine correction). Traders absent here stay 'unknown' by design: yank / s26 /
# s27 rows span both replayed history and live trading, so the venue is per-row, not
# per-trader, and an honest 'unknown' beats a guess.
EXECUTION_MODE = {
    "trader-btc-carry": "paper",
    "trader-gap-fade": "sim",
    "trader-mim-nb": "live",
    "trader-s26-combine": "paper",
}


def report(conn: sqlite3.Connection, label: str) -> None:
    print(f"\n--- {label} ---")
    print("  execution_mode, trader-s26-combine:")
    for mode, n in conn.execute(
        "SELECT execution_mode, COUNT(*) FROM trades "
        "WHERE trader_id='trader-s26-combine' GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"      {str(mode):<10} {n:>5}")
    n_null = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE write_mode IS NULL"
    ).fetchone()[0]
    print(f"  write_mode IS NULL: {n_null}")
    print("  write_mode totals:")
    for mode, n in conn.execute(
        "SELECT write_mode, COUNT(*) FROM trades GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"      {str(mode):<10} {n:>5}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    if not DB.exists():
        print(f"ERROR: {DB} not found", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA busy_timeout = 15000")  # 7 bots write this file concurrently
    report(conn, "BEFORE")

    # --- what would change -------------------------------------------------
    n_s26 = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE trader_id='trader-s26-combine' "
        "AND (execution_mode IS NULL OR execution_mode <> 'paper')"
    ).fetchone()[0]
    n_rt = conn.execute(
        f"SELECT COUNT(*) FROM trades WHERE write_mode IS NULL AND created_at IS NOT NULL "
        f"AND timestamp IS NOT NULL AND {LAG_SQL} < ?", (REALTIME_LAG_H,)
    ).fetchone()[0]
    n_bf = conn.execute(
        f"SELECT COUNT(*) FROM trades WHERE write_mode IS NULL AND created_at IS NOT NULL "
        f"AND timestamp IS NOT NULL AND {LAG_SQL} >= ?", (REALTIME_LAG_H,)
    ).fetchone()[0]
    n_unk = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE write_mode IS NULL "
        "AND (created_at IS NULL OR timestamp IS NULL)"
    ).fetchone()[0]

    n_exec_null = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE execution_mode IS NULL"
    ).fetchone()[0]

    print("\n=== planned changes ===")
    print(f"  (1) execution_mode -> 'paper' : {n_s26:>5} rows (trader-s26-combine)")
    print(f"  (2) write_mode -> 'realtime'  : {n_rt:>5} rows (lag < {REALTIME_LAG_H}h)")
    print(f"      write_mode -> 'backfilled': {n_bf:>5} rows (lag >= {REALTIME_LAG_H}h)")
    print(f"      write_mode -> 'unknown'   : {n_unk:>5} rows (no usable timestamps)")
    print(f"  (3) execution_mode NULL fill  : {n_exec_null:>5} rows "
          f"(per-trader dict, else 'unknown')")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = DB.with_name(f"{DB.name}.pre-fix-{stamp}.bak")
    shutil.copy2(DB, backup)
    print(f"\nbackup written: {backup}")

    with conn:
        conn.execute(
            "UPDATE trades SET execution_mode='paper' WHERE trader_id='trader-s26-combine'"
        )
        conn.execute(
            f"UPDATE trades SET write_mode='realtime' WHERE write_mode IS NULL "
            f"AND created_at IS NOT NULL AND timestamp IS NOT NULL AND {LAG_SQL} < ?",
            (REALTIME_LAG_H,),
        )
        conn.execute(
            f"UPDATE trades SET write_mode='backfilled' WHERE write_mode IS NULL "
            f"AND created_at IS NOT NULL AND timestamp IS NOT NULL AND {LAG_SQL} >= ?",
            (REALTIME_LAG_H,),
        )
        conn.execute(
            "UPDATE trades SET write_mode='unknown' WHERE write_mode IS NULL"
        )
        # (3) execution_mode: same rule the original migration used — a documented
        # venue where one exists, an explicit 'unknown' otherwise. NULL is the one
        # value that means "nobody has looked", and it should not persist.
        for trader, mode in EXECUTION_MODE.items():
            conn.execute(
                "UPDATE trades SET execution_mode=? WHERE trader_id=? "
                "AND (execution_mode IS NULL OR execution_mode <> ?)",
                (mode, trader, mode),
            )
        conn.execute(
            "UPDATE trades SET execution_mode='unknown' WHERE execution_mode IS NULL"
        )

    report(conn, "AFTER")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
