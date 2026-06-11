"""
Performance Tracker: Reads live trading logs and updates strategy metrics automatically.

Runs on a schedule to:
1. Parse paper/live trading logs for completed trades
2. Compute win rate, profit factor, sharpe, drawdown
3. Update strategy_registry.json via promotion_engine
4. Alert if daily loss limit is approaching (80%+ consumed)
5. Trigger promotion check automatically
"""

import json
import re
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.promotion_engine import update_paper_metrics, status_report

LOGS_DIR = Path(__file__).parent.parent / "logs"
REGISTRY_PATH = Path(__file__).parent / "strategy_registry.json"

# Log files to monitor per strategy
STRATEGY_LOG_MAP = {
    "silver-bullet-tier2-hybrid": [
        "tier2_paper_trading.log",
        "tier2_paper_trader.log",
        "tier2_streaming_working.log",
    ]
}

# Regex patterns for trade events
ENTRY_RE = re.compile(r"(TIER\s*\d+)\s+ENTRY:\s+(LONG|SHORT)\s+at\s+\$?([\d.]+)", re.IGNORECASE)
TP_RE = re.compile(r"TP\s+hit.*?\+\$?([\d.]+)", re.IGNORECASE)
SL_RE = re.compile(r"SL\s+hit.*?-?\$?([\d.]+)", re.IGNORECASE)
PNL_RE = re.compile(r"PnL[:\s]+([+-]?\$?[\d.]+)", re.IGNORECASE)


def parse_trades_from_log(log_path: Path) -> list[dict]:
    """Parse completed trades from log file."""
    trades = []
    if not log_path.exists():
        return trades

    with open(log_path, errors="replace") as f:
        lines = f.readlines()

    current_trade = None
    for line in lines:
        # Detect entry
        m = ENTRY_RE.search(line)
        if m:
            current_trade = {
                "direction": m.group(2),
                "entry_price": float(m.group(3)),
                "timestamp": line[:23].strip(),
                "pnl": None,
                "outcome": None,
            }
            continue

        if current_trade:
            # Detect TP hit
            m = TP_RE.search(line)
            if m:
                current_trade["pnl"] = float(m.group(1).replace("$", ""))
                current_trade["outcome"] = "WIN"
                trades.append(current_trade)
                current_trade = None
                continue

            # Detect SL hit
            m = SL_RE.search(line)
            if m:
                current_trade["pnl"] = -float(m.group(1).replace("$", ""))
                current_trade["outcome"] = "LOSS"
                trades.append(current_trade)
                current_trade = None
                continue

    return trades


def compute_metrics(trades: list[dict]) -> dict:
    """Compute trading metrics from trade list."""
    if not trades:
        return {}

    pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
    if not pnls:
        return {}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls)

    # Sharpe (simplified daily)
    if len(pnls) > 1:
        mean = avg_pnl
        variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.001
        sharpe = (mean / std) * math.sqrt(252) if std > 0 else 0
    else:
        sharpe = 0

    # Days active (rough estimate from timestamps)
    days_active = 1
    if len(trades) > 1:
        try:
            t0 = datetime.strptime(trades[0]["timestamp"][:10], "%Y-%m-%d")
            t1 = datetime.strptime(trades[-1]["timestamp"][:10], "%Y-%m-%d")
            days_active = max(1, (t1 - t0).days + 1)
        except Exception:
            pass

    # Daily loss tracking
    daily_pnl = defaultdict(float)
    for t in trades:
        day = t["timestamp"][:10]
        if t["pnl"] is not None:
            daily_pnl[day] += t["pnl"]

    daily_loss_limit_hits = sum(1 for d, p in daily_pnl.items() if p < -500)
    max_daily_loss = min(daily_pnl.values()) if daily_pnl else 0

    return {
        "trades": len(pnls),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(sharpe, 3),
        "days_active": days_active,
        "daily_loss_limit_hits": daily_loss_limit_hits,
        "max_daily_loss": round(max_daily_loss, 2),
    }


def run_update():
    """Parse logs, compute metrics, update registry."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running performance tracker...")

    for strategy_id, log_files in STRATEGY_LOG_MAP.items():
        all_trades = []
        for log_name in log_files:
            log_path = LOGS_DIR / log_name
            trades = parse_trades_from_log(log_path)
            all_trades.extend(trades)
            if trades:
                print(f"  {log_name}: {len(trades)} trades parsed")

        if all_trades:
            metrics = compute_metrics(all_trades)
            print(f"  Metrics for {strategy_id}: {metrics}")
            update_paper_metrics(strategy_id, metrics)

            # Alert if daily loss limit approaching today
            today = datetime.now().strftime("%Y-%m-%d")
            daily_pnl = defaultdict(float)
            for t in all_trades:
                if t["timestamp"][:10] == today and t["pnl"] is not None:
                    daily_pnl[today] += t["pnl"]

            today_pnl = daily_pnl.get(today, 0)
            if today_pnl < -400:
                print(f"\n  *** ALERT: Today's P&L is ${today_pnl:.2f} - approaching $500 daily limit! ***")
            elif today_pnl < -250:
                print(f"\n  WARNING: Today's P&L is ${today_pnl:.2f} - 50% of daily limit consumed.")
        else:
            print(f"  No completed trades found for {strategy_id}")

    print("\n" + status_report())


if __name__ == "__main__":
    run_update()
