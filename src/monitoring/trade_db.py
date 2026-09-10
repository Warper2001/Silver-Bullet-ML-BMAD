"""
Centralized Trade Database for BMAD trading bots.
Replaces scattered CSV logs with a structured SQLite database.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Any, Optional, List, Dict
import logging
import os
import json

logger = logging.getLogger(__name__)

class TradeDatabase:
    """
    Manages persistence of trade results across all trading bots.
    Provides a standardized schema and easy retrieval for performance analysis.
    """
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trader_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL NOT NULL,
                    exit_reason TEXT,
                    ml_proba REAL,
                    metadata TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trader ON trades(trader_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
            # Idempotency: a natural-key UNIQUE index makes every writer (live bot,
            # backfill replay, ad-hoc sync script) safe to re-run without inserting
            # duplicate rows. Paired with INSERT OR IGNORE in log_trade(). A restart-
            # replay re-logging trades was silently doubling P&L (s26/yank, 2026-06-19).
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_trade_identity "
                "ON trades(trader_id, timestamp, direction, entry_price, exit_price, pnl)"
            )
            # Provenance columns (2026-09-02). Without these the table silently
            # mixes replayed history with live trading: trader-yank read
            # +$102,151.90 while only 5 of its 1,846 rows were written live, and
            # trader-btc-carry logged 1 row while its executor had been in PAPER
            # mode for three months. Both defects were a missing label, not a
            # missing gate. See tools/migrate_trades_provenance.py.
            for _col in ("write_mode", "execution_mode"):
                if not any(r[1] == _col for r in conn.execute("PRAGMA table_info(trades)")):
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {_col} TEXT")
            conn.commit()

    def log_trade(
        self, 
        trader_id: str, 
        timestamp: str, 
        pnl: float, 
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        ml_proba: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_mode: str = "unknown",
        write_mode: str = "realtime",
    ):
        """Record a completed trade.

        execution_mode: what was actually at risk -- 'live', 'sim', 'paper', or
            'unknown'. Callers MUST pass this. The default is deliberately
            'unknown' rather than 'live': an honest unknown is recoverable, a
            wrong 'live' is the BTC-CARRY incident (a paper bot recorded as
            trading real money for three months).
        write_mode: 'realtime' when logged as the trade happens (the default,
            since that is what a running bot does), 'backfilled' for replay or
            catch-up writes. A backfill that forgets to pass 'backfilled' is how
            +$101,892.90 of replayed P&L came to sit in the ledger looking like
            a track record.
        """
        meta_json = json.dumps(metadata) if metadata else None

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO trades
                    (trader_id, timestamp, symbol, direction, entry_price, exit_price, pnl, exit_reason, ml_proba, metadata, created_at, write_mode, execution_mode)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trader_id, timestamp, symbol, direction, entry_price,
                    exit_price, pnl, exit_reason, ml_proba, meta_json,
                    datetime.utcnow().isoformat(), write_mode, execution_mode
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log trade for {trader_id}: {e}")

    def get_trades_df(self, trader_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieve trades as a pandas DataFrame for analysis."""
        query = "SELECT * FROM trades"
        params = []
        if trader_id:
            query += " WHERE trader_id = ?"
            params.append(trader_id)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_all_trader_ids(self) -> List[str]:
        """Get a list of all bots that have logged trades."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT trader_id FROM trades")
            return [row[0] for row in cursor.fetchall()]
