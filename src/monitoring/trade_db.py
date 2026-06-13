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
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a completed trade."""
        meta_json = json.dumps(metadata) if metadata else None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades 
                    (trader_id, timestamp, symbol, direction, entry_price, exit_price, pnl, exit_reason, ml_proba, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trader_id, timestamp, symbol, direction, entry_price, 
                    exit_price, pnl, exit_reason, ml_proba, meta_json, 
                    datetime.utcnow().isoformat()
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
