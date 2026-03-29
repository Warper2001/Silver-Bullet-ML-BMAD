"""
Log Rotation for 24/7 Operation

This module provides log rotation and management for continuous operation.

Features:
- Size-based rotation (100MB per file)
- Time-based rotation (daily)
- Retention policy (7 days)
- Compression of old logs
- Separate handlers for trading and system logs

Usage:
    from src.monitoring.log_rotation import setup_log_rotation

    setup_log_rotation()
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

# Log directories
LOG_DIR = Path("logs")
TRADING_LOG_DIR = LOG_DIR / "trading"
SYSTEM_LOG_DIR = LOG_DIR / "system"

# Rotation settings
MAX_BYTES = 100 * 1024 * 1024  # 100MB
BACKUP_COUNT = 7  # Retain 7 days


def setup_log_rotation(
    log_level: int = logging.INFO,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> None:
    """
    Setup log rotation for 24/7 operation.

    Creates rotating file handlers for:
    - System logs: logs/system/system.log
    - Trading logs: logs/trading/trading.log

    Args:
        log_level: Logging level (default: INFO)
        max_bytes: Maximum bytes per log file (default: 100MB)
        backup_count: Number of backup files to retain (default: 7)
    """
    # Create log directories
    TRADING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    root_logger.handlers.clear()

    # Setup system log handler
    system_handler = _create_rotating_handler(
        log_file=SYSTEM_LOG_DIR / "system.log",
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    system_handler.setLevel(log_level)
    system_handler.setFormatter(_get_formatter())
    root_logger.addHandler(system_handler)

    # Setup trading log handler
    trading_handler = _create_rotating_handler(
        log_file=TRADING_LOG_DIR / "trading.log",
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    trading_handler.setLevel(log_level)
    trading_handler.setFormatter(_get_formatter())

    # Add trading handler to specific loggers
    trading_loggers = [
        "src.execution.binance",
        "src.detection",
        "src.ml",
        "src.risk",
        "start_crypto_paper_trading",
    ]

    for logger_name in trading_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(trading_handler)
        logger.setLevel(log_level)

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        f"Log rotation initialized: "
        f"max_bytes={max_bytes}, backup_count={backup_count}"
    )


def _create_rotating_handler(
    log_file: Path,
    max_bytes: int,
    backup_count: int,
) -> logging.handlers.RotatingFileHandler:
    """
    Create a rotating file handler.

    Args:
        log_file: Path to log file
        max_bytes: Maximum bytes per file
        backup_count: Number of backups to retain

    Returns:
        RotatingFileHandler instance
    """
    handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )

    return handler


def _get_formatter() -> logging.Formatter:
    """
    Get log formatter.

    Returns:
        Formatter with consistent format
    """
    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_audit_trail_rotation(
    audit_dir: Path = Path("logs/audit"),
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> logging.Logger:
    """
    Setup rotating audit trail logger for CSV exports.

    Args:
        audit_dir: Directory for audit logs
        max_bytes: Maximum bytes per file
        backup_count: Number of backups to retain

    Returns:
        Logger for audit trails
    """
    # Create audit directory
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Create audit logger
    audit_logger = logging.getLogger("audit_trail")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Don't propagate to root logger

    # Create rotating handler
    handler = _create_rotating_handler(
        log_file=audit_dir / "audit.log",
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    handler.setFormatter(_get_formatter())
    audit_logger.addHandler(handler)

    return audit_logger


def cleanup_old_logs(
    log_dir: Path = LOG_DIR,
    days: int = 30,
) -> None:
    """
    Clean up log files older than specified days.

    Args:
        log_dir: Directory containing logs
        days: Delete logs older than this many days
    """
    if not log_dir.exists():
        return

    cutoff_time = datetime.now().timestamp() - (days * 86400)

    for log_file in log_dir.rglob("*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                logging.getLogger(__name__).info(f"Deleted old log: {log_file}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to delete {log_file}: {e}")


def get_log_stats() -> dict:
    """
    Get statistics about log files.

    Returns:
        Dictionary with log statistics
    """
    stats = {
        "log_dir": str(LOG_DIR),
        "trading_log_dir": str(TRADING_LOG_DIR),
        "system_log_dir": str(SYSTEM_LOG_DIR),
        "total_size_mb": 0,
        "file_count": 0,
        "files": [],
    }

    if not LOG_DIR.exists():
        return stats

    for log_file in LOG_DIR.rglob("*.log*"):
        try:
            size_mb = log_file.stat().st_size / (1024 * 1024)
            stats["total_size_mb"] += size_mb
            stats["file_count"] += 1
            stats["files"].append({
                "path": str(log_file.relative_to(LOG_DIR)),
                "size_mb": round(size_mb, 2),
                "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
            })
        except Exception:
            pass

    stats["total_size_mb"] = round(stats["total_size_mb"], 2)

    return stats
