"""Shared state database for dashboard and live trading system communication.

This module provides SQLite-based persistence for real-time trading data,
enabling the dashboard to display live positions, signals, and metrics.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Database location
DB_DIR = Path("data/state")
DB_PATH = DB_DIR / "shared_state.db"

# Connection pooling - single connection for writes
_write_connection: Optional[sqlite3.Connection] = None


def _validate_probability(value: float, param_name: str = "probability") -> None:
    """Validate probability is in [0, 1] range.

    Args:
        value: Probability value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If probability is invalid
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{param_name} must be a number, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{param_name} must be between 0.0 and 1.0, got {value}")


def _validate_positive(value: float, param_name: str = "value") -> None:
    """Validate value is positive.

    Args:
        value: Value to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{param_name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{param_name} must be non-negative, got {value}")


def _validate_non_empty_string(value: str, param_name: str = "value") -> None:
    """Validate string is non-empty.

    Args:
        value: String to validate
        param_name: Parameter name for error messages

    Raises:
        ValueError: If string is invalid
    """
    if not isinstance(value, str):
        raise ValueError(f"{param_name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{param_name} cannot be empty")


def init_db() -> None:
    """Initialize shared state database with required tables.

    Enables WAL mode for better concurrent read/write access and
    sets busy timeout to handle read/write contention.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable WAL mode for better concurrent access (fixes F1 - race condition)
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

    # Positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            order_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            current_price REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            unrealized_pnl_percent REAL NOT NULL,
            probability REAL NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL,
            probability REAL NOT NULL,
            confidence INTEGER NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            expires_at TEXT
        )
    """)

    # Account metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS account_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equity REAL NOT NULL,
            daily_pnl REAL NOT NULL,
            daily_drawdown REAL NOT NULL,
            daily_loss_limit REAL NOT NULL,
            open_positions_count INTEGER NOT NULL,
            open_contracts INTEGER NOT NULL,
            trade_count INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            system_uptime TEXT NOT NULL,
            last_update TEXT NOT NULL
        )
    """)

    # Indexes for performance (fixes F10 - missing index on signals status)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_account_metrics_last_update ON account_metrics(last_update)")

    conn.commit()
    conn.close()

    logger.info(f"Shared state database initialized: {DB_PATH}")


def write_position(
    order_id: str,
    signal_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    quantity: int,
    current_price: float,
    unrealized_pnl: float,
    unrealized_pnl_percent: float,
    probability: float,
    status: str = "OPEN"
) -> None:
    """Write or update position in shared state.

    Args:
        order_id: TradeStation order ID
        signal_id: Signal identifier
        symbol: Trading symbol
        direction: "bullish" or "bearish"
        entry_price: Entry price
        quantity: Number of contracts
        current_price: Current market price
        unrealized_pnl: Unrealized P&L in dollars
        unrealized_pnl_percent: Unrealized P&L percentage
        probability: ML probability score
        status: Position status (OPEN, CLOSED)

    Raises:
        ValueError: If validation fails
    """
    # Input validation (fixes F3 - no validation)
    _validate_non_empty_string(order_id, "order_id")
    _validate_non_empty_string(signal_id, "signal_id")
    _validate_non_empty_string(symbol, "symbol")
    _validate_non_empty_string(direction, "direction")
    _validate_positive(entry_price, "entry_price")
    _validate_positive(quantity, "quantity")
    _validate_positive(current_price, "current_price")
    _validate_probability(probability, "probability")
    _validate_non_empty_string(status, "status")

    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO positions (
                order_id, signal_id, symbol, direction, entry_price, quantity,
                current_price, unrealized_pnl, unrealized_pnl_percent, probability,
                status, timestamp, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_id, signal_id, symbol, direction, entry_price, quantity,
            current_price, unrealized_pnl, unrealized_pnl_percent, probability,
            status, now, now
        ))
        conn.commit()
        logger.debug(f"Position written to shared state: {order_id}")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Failed to write position {order_id}: {e}")
        raise
    finally:
        conn.close()


def write_signal(
    signal_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: Optional[float],
    probability: float,
    confidence: int,
    status: str = "ACTIVE",
    expires_at: Optional[str] = None
) -> None:
    """Write or update signal in shared state.

    Args:
        signal_id: Signal identifier
        symbol: Trading symbol
        direction: "bullish" or "bearish"
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        probability: ML probability score
        confidence: Confidence level (1-5)
        status: Signal status (ACTIVE, EXPIRED, FILLED)
        expires_at: Expiration timestamp
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    cursor.execute("""
        INSERT OR REPLACE INTO signals (
            signal_id, symbol, direction, entry_price, stop_loss, take_profit,
            probability, confidence, status, timestamp, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_id, symbol, direction, entry_price, stop_loss, take_profit,
        probability, confidence, status, now, expires_at
    ))

    conn.commit()
    conn.close()

    logger.debug(f"Signal written to shared state: {signal_id}")


def write_account_metrics(
    equity: float,
    daily_pnl: float,
    daily_drawdown: float,
    daily_loss_limit: float,
    open_positions_count: int,
    open_contracts: int,
    trade_count: int,
    win_rate: float,
    system_uptime: str
) -> None:
    """Write account metrics to shared state.

    Args:
        equity: Current account equity
        daily_pnl: Daily profit/loss
        daily_drawdown: Current daily drawdown
        daily_loss_limit: Daily loss limit
        open_positions_count: Number of open positions
        open_contracts: Total open contracts
        trade_count: Total trades today
        win_rate: Win rate percentage
        system_uptime: System uptime string

    Raises:
        ValueError: If validation fails
    """
    # Input validation (fixes F3)
    _validate_positive(equity, "equity")
    _validate_positive(daily_loss_limit, "daily_loss_limit")
    _validate_positive(open_positions_count, "open_positions_count")
    _validate_positive(open_contracts, "open_contracts")
    _validate_positive(trade_count, "trade_count")
    _validate_probability(win_rate / 100.0, "win_rate")  # Convert percentage to 0-1
    _validate_non_empty_string(system_uptime, "system_uptime")

    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    try:
        # Delete old metrics (keep only latest)
        cursor.execute("DELETE FROM account_metrics")

        cursor.execute("""
            INSERT INTO account_metrics (
                equity, daily_pnl, daily_drawdown, daily_loss_limit,
                open_positions_count, open_contracts, trade_count, win_rate,
                system_uptime, last_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            equity, daily_pnl, daily_drawdown, daily_loss_limit,
            open_positions_count, open_contracts, trade_count, win_rate,
            system_uptime, now
        ))
        conn.commit()
        logger.debug("Account metrics written to shared state")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Failed to write account metrics: {e}")
        raise
    finally:
        conn.close()


def write_trading_state(
    positions_data: List[dict],
    account_metrics_data: dict
) -> None:
    """Write positions and account metrics in a single transaction.

    This ensures atomicity - either both positions and metrics are written,
    or neither is (fixes F6 - transaction safety).

    Args:
        positions_data: List of position dictionaries to write
        account_metrics_data: Account metrics dictionary to write

    Raises:
        ValueError: If validation fails
        sqlite3.Error: If database operation fails
    """
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    try:
        # Begin transaction
        cursor.execute("BEGIN TRANSACTION")

        # Delete old positions and write new ones
        for pos in positions_data:
            _validate_non_empty_string(pos["order_id"], "order_id")
            _validate_non_empty_string(pos["signal_id"], "signal_id")
            _validate_positive(pos["entry_price"], "entry_price")
            _validate_probability(pos["probability"], "probability")

            cursor.execute("""
                INSERT OR REPLACE INTO positions (
                    order_id, signal_id, symbol, direction, entry_price, quantity,
                    current_price, unrealized_pnl, unrealized_pnl_percent, probability,
                    status, timestamp, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos["order_id"], pos["signal_id"], pos["symbol"], pos["direction"],
                pos["entry_price"], pos["quantity"], pos.get("current_price", pos["entry_price"]),
                pos.get("unrealized_pnl", 0.0), pos.get("unrealized_pnl_percent", 0.0),
                pos["probability"], pos.get("status", "OPEN"), now, now
            ))

        # Delete old metrics and write new ones
        metrics = account_metrics_data
        _validate_positive(metrics["equity"], "equity")
        _validate_positive(metrics["daily_loss_limit"], "daily_loss_limit")

        cursor.execute("DELETE FROM account_metrics")
        cursor.execute("""
            INSERT INTO account_metrics (
                equity, daily_pnl, daily_drawdown, daily_loss_limit,
                open_positions_count, open_contracts, trade_count, win_rate,
                system_uptime, last_update
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics["equity"], metrics["daily_pnl"], metrics.get("daily_drawdown", 0.0),
            metrics["daily_loss_limit"], metrics["open_positions_count"],
            metrics["open_contracts"], metrics["trade_count"], metrics["win_rate"],
            metrics["system_uptime"], now
        ))

        # Commit transaction
        conn.commit()
        logger.debug(f"Trading state written: {len(positions_data)} positions, account metrics")

    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Failed to write trading state transaction: {e}")
        raise
    finally:
        conn.close()


def read_positions(status: str = "OPEN") -> List[dict]:
    """Read positions from shared state.

    Args:
        status: Filter by status (default: OPEN)

    Returns:
        List of position dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, signal_id, symbol, direction, entry_price, quantity,
               current_price, unrealized_pnl, unrealized_pnl_percent, probability,
               status, timestamp, updated_at
        FROM positions
        WHERE status = ?
        ORDER BY timestamp DESC
    """, (status,))

    rows = cursor.fetchall()
    conn.close()

    positions = []
    for row in rows:
        positions.append({
            "order_id": row[0],
            "signal_id": row[1],
            "symbol": row[2],
            "direction": row[3],
            "entry_price": row[4],
            "quantity": row[5],
            "current_price": row[6],
            "unrealized_pnl": row[7],
            "unrealized_pnl_percent": row[8],
            "probability": row[9],
            "status": row[10],
            "timestamp": row[11],
            "updated_at": row[12]
        })

    return positions


def read_signals(status: str = "ACTIVE") -> List[dict]:
    """Read signals from shared state.

    Args:
        status: Filter by status (default: ACTIVE)

    Returns:
        List of signal dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT signal_id, symbol, direction, entry_price, stop_loss, take_profit,
               probability, confidence, status, timestamp, expires_at
        FROM signals
        WHERE status = ?
        ORDER BY timestamp DESC
        LIMIT 50
    """, (status,))

    rows = cursor.fetchall()
    conn.close()

    signals = []
    for row in rows:
        signals.append({
            "signal_id": row[0],
            "symbol": row[1],
            "direction": row[2],
            "entry_price": row[3],
            "stop_loss": row[4],
            "take_profit": row[5],
            "probability": row[6],
            "confidence": row[7],
            "status": row[8],
            "timestamp": row[9],
            "expires_at": row[10]
        })

    return signals


def read_account_metrics() -> Optional[dict]:
    """Read latest account metrics from shared state.

    Returns:
        Account metrics dictionary or None if no data
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT equity, daily_pnl, daily_drawdown, daily_loss_limit,
               open_positions_count, open_contracts, trade_count, win_rate,
               system_uptime, last_update
        FROM account_metrics
        ORDER BY last_update DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "equity": row[0],
        "daily_pnl": row[1],
        "daily_drawdown": row[2],
        "daily_loss_limit": row[3],
        "open_positions_count": row[4],
        "open_contracts": row[5],
        "trade_count": row[6],
        "win_rate": row[7],
        "system_uptime": row[8],
        "last_update": row[9]
    }


def cleanup_old_data() -> None:
    """Clean up old data from shared state (keep last 24 hours)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # Delete old closed positions
    cursor.execute("DELETE FROM positions WHERE status = 'CLOSED' AND updated_at < ?", (cutoff,))

    # Delete old expired signals
    cursor.execute("DELETE FROM signals WHERE status = 'EXPIRED' AND timestamp < ?", (cutoff,))

    # Delete old account metrics (keep latest 100)
    cursor.execute("""
        DELETE FROM account_metrics
        WHERE id NOT IN (
            SELECT id FROM account_metrics
            ORDER BY last_update DESC
            LIMIT 100
        )
    """)

    conn.commit()
    conn.close()

    logger.debug("Old data cleaned up from shared state")
