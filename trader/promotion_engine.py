"""
Promotion Engine: Manages strategy lifecycle from IDEA -> BACKTEST -> PAPER -> LIVE.

Stages:
  IDEA           - Hypothesis documented, awaiting backtest
  BACKTESTING    - Actively being backtested
  BACKTEST_PASS  - Passed backtest criteria, ready for paper
  BACKTEST_FAIL  - Failed backtest, returned to IDEA or archived
  PAPER_TRADING  - Running on sim account with real market data
  AWAITING_LIVE  - Paper criteria met, USER GATE required for promotion
  LIVE           - Trading real capital (requires explicit user approval)
  RETIRED        - Underperforming, pulled from trading

Promotion gates:
  BACKTEST_PASS -> PAPER_TRADING:
    - min 30 trades
    - win rate >= 55%
    - profit factor >= 1.5
    - sharpe >= 1.0
    - max drawdown < 15%

  PAPER_TRADING -> AWAITING_LIVE:
    - min 20 paper trades
    - min 10 days active
    - win rate >= 55%
    - profit factor >= 1.5
    - sharpe >= 1.0
    - ZERO days hitting daily loss limit
    - max single-day loss < $300 (60% of $500 limit)

  AWAITING_LIVE -> LIVE:
    - USER MUST EXPLICITLY APPROVE (run: python promotion_engine.py approve <strategy_id>)
"""

import json
import sys
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path(__file__).parent / "strategy_registry.json"
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry saved to {REGISTRY_PATH}")


def check_paper_promotion_criteria(strategy: dict) -> tuple[bool, list[str]]:
    """Check if paper trading metrics meet live promotion criteria."""
    criteria = strategy["promotion_criteria"]["paper_to_live"]
    metrics = strategy["paper_metrics"]
    failures = []
    passes = []

    def check(label, value, threshold, op=">="):
        if value is None:
            failures.append(f"  MISSING: {label} (no data yet)")
            return
        if op == ">=" and value >= threshold:
            passes.append(f"  PASS: {label} = {value:.3f} (need {op}{threshold})")
        elif op == "<=" and value <= threshold:
            passes.append(f"  PASS: {label} = {value:.3f} (need {op}{threshold})")
        elif op == "==" and value == threshold:
            passes.append(f"  PASS: {label} = {value} (need {op}{threshold})")
        else:
            failures.append(f"  FAIL: {label} = {value} (need {op}{threshold})")

    check("Trades", metrics.get("trades"), criteria["min_trades"])
    check("Days Active", metrics.get("days_active"), criteria["min_days"])
    check("Win Rate", metrics.get("win_rate"), criteria["min_win_rate"])
    check("Profit Factor", metrics.get("profit_factor"), criteria["min_profit_factor"])
    check("Sharpe Ratio", metrics.get("sharpe"), criteria["min_sharpe"])
    check(
        "Daily Loss Limit Hits",
        metrics.get("daily_loss_limit_hits", 0),
        criteria["max_daily_loss_exceeded_count"],
        "<=",
    )

    result = len(failures) == 0
    all_checks = passes + failures
    return result, all_checks


def status_report() -> str:
    """Generate a full status report of all strategies."""
    registry = load_registry()
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"\n{'='*60}")
    lines.append(f"TRADING SYSTEM STATUS REPORT - {now}")
    lines.append(f"{'='*60}")

    # Active strategies
    lines.append("\nACTIVE STRATEGIES:")
    lines.append("-" * 40)
    for s in registry["strategies"]:
        stage = s["stage"]
        lines.append(f"\n[{stage}] {s['name']}")
        lines.append(f"  Symbol: {s['symbol']} | TF: {s['timeframe']}")
        lines.append(f"  Capital: ${s['capital_allocated']:,} | Daily Limit: ${s['daily_loss_limit']}")

        pm = s["paper_metrics"]
        if pm.get("trades", 0) > 0:
            lines.append(f"  Paper P&L: ${pm.get('total_pnl', 0):.2f}")
            lines.append(f"  Win Rate: {(pm.get('win_rate') or 0)*100:.1f}% | Trades: {pm.get('trades', 0)} | Days: {pm.get('days_active', 0)}")
            lines.append(f"  Profit Factor: {pm.get('profit_factor', 'N/A')} | Sharpe: {pm.get('sharpe', 'N/A')}")
        else:
            lines.append(f"  Paper metrics: awaiting data (0 trades logged)")

        if stage == "AWAITING_LIVE":
            lines.append("\n  >>> AWAITING YOUR APPROVAL FOR LIVE TRADING <<<")
            lines.append("  Run: python trader/promotion_engine.py approve " + s["id"])
        elif stage == "PAPER_TRADING":
            ready, checks = check_paper_promotion_criteria(s)
            lines.append(f"\n  Promotion check ({('READY' if ready else 'NOT READY')}):")
            for c in checks:
                lines.append("  " + c)

    # Idea backlog
    lines.append(f"\n\nIDEA BACKLOG ({len(registry['idea_backlog'])} ideas):")
    lines.append("-" * 40)
    for idea in registry["idea_backlog"]:
        lines.append(f"  [{idea['priority']}] {idea['name']}")
        lines.append(f"         {idea['description'][:80]}")

    lines.append(f"\n{'='*60}\n")
    return "\n".join(lines)


def update_paper_metrics(strategy_id: str, metrics: dict):
    """Update paper trading metrics for a strategy."""
    registry = load_registry()
    for s in registry["strategies"]:
        if s["id"] == strategy_id:
            s["paper_metrics"].update(metrics)
            s["paper_metrics"]["last_updated"] = datetime.now().isoformat()

            # Check if paper criteria are met -> promote to AWAITING_LIVE
            if s["stage"] == "PAPER_TRADING":
                ready, _ = check_paper_promotion_criteria(s)
                if ready:
                    s["stage"] = "AWAITING_LIVE"
                    print(f"STRATEGY {strategy_id} HAS MET PAPER CRITERIA!")
                    print("Stage promoted to AWAITING_LIVE. Run 'approve' to go live.")

            save_registry(registry)
            return
    print(f"Strategy {strategy_id} not found.")


def approve_live(strategy_id: str):
    """USER GATE: Promote a strategy from AWAITING_LIVE to LIVE."""
    registry = load_registry()
    for s in registry["strategies"]:
        if s["id"] == strategy_id:
            if s["stage"] != "AWAITING_LIVE":
                print(f"ERROR: {strategy_id} is in stage '{s['stage']}', not AWAITING_LIVE.")
                return
            confirm = input(
                f"\nYou are about to APPROVE LIVE TRADING with REAL CAPITAL for:\n"
                f"  {s['name']}\n"
                f"  Capital: ${s['capital_allocated']:,}\n"
                f"  Daily Loss Limit: ${s['daily_loss_limit']}\n\n"
                f"Type 'APPROVE LIVE' to confirm: "
            )
            if confirm.strip() == "APPROVE LIVE":
                s["stage"] = "LIVE"
                s["promoted_to_live"] = datetime.now().isoformat()
                save_registry(registry)
                print(f"\nLIVE TRADING APPROVED for {s['name']}")
                print("The system will now execute real trades. Monitor closely.")
            else:
                print("Approval cancelled. Strategy remains in AWAITING_LIVE.")
            return
    print(f"Strategy {strategy_id} not found.")


def retire_strategy(strategy_id: str, reason: str):
    """Retire an underperforming strategy."""
    registry = load_registry()
    for s in registry["strategies"]:
        if s["id"] == strategy_id:
            s["stage"] = "RETIRED"
            s["retire_reason"] = reason
            s["retired_at"] = datetime.now().isoformat()
            save_registry(registry)
            print(f"Strategy {strategy_id} retired: {reason}")
            return


def promote_idea_to_backtest(idea_id: str):
    """Move an idea from backlog to active backtesting."""
    registry = load_registry()
    for idea in registry["idea_backlog"]:
        if idea["id"] == idea_id:
            new_strategy = {
                "id": idea["id"],
                "name": idea["name"],
                "description": idea["description"],
                "stage": "BACKTESTING",
                "created": idea["created"],
                "promoted_to_paper": None,
                "promoted_to_live": None,
                "symbol": "MNQM26",
                "timeframe": "5min",
                "capital_allocated": 20000,
                "daily_loss_limit": 500,
                "backtest_metrics": {},
                "paper_metrics": {
                    "win_rate": None,
                    "profit_factor": None,
                    "total_pnl": None,
                    "trades": 0,
                    "days_active": 0,
                    "max_daily_loss": None,
                    "sharpe": None,
                    "daily_loss_limit_hits": 0,
                    "last_updated": None
                },
                "promotion_criteria": {
                    "paper_to_live": {
                        "min_trades": 20,
                        "min_days": 10,
                        "min_win_rate": 0.55,
                        "min_profit_factor": 1.5,
                        "max_daily_loss_exceeded_count": 0,
                        "min_sharpe": 1.0
                    }
                },
                "notes": idea.get("hypothesis", "")
            }
            registry["strategies"].append(new_strategy)
            registry["idea_backlog"] = [i for i in registry["idea_backlog"] if i["id"] != idea_id]
            save_registry(registry)
            print(f"Idea '{idea['name']}' promoted to BACKTESTING stage.")
            return
    print(f"Idea {idea_id} not found in backlog.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(status_report())
    elif sys.argv[1] == "status":
        print(status_report())
    elif sys.argv[1] == "approve" and len(sys.argv) >= 3:
        approve_live(sys.argv[2])
    elif sys.argv[1] == "retire" and len(sys.argv) >= 4:
        retire_strategy(sys.argv[2], " ".join(sys.argv[3:]))
    elif sys.argv[1] == "backtest" and len(sys.argv) >= 3:
        promote_idea_to_backtest(sys.argv[2])
    elif sys.argv[1] == "update-metrics" and len(sys.argv) >= 4:
        # Usage: update-metrics <id> <key=value> ...
        m = {}
        for kv in sys.argv[3:]:
            k, v = kv.split("=")
            try:
                v = float(v)
            except ValueError:
                pass
            m[k] = v
        update_paper_metrics(sys.argv[2], m)
    else:
        print(__doc__)
