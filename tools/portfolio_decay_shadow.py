#!/usr/bin/env python3
"""PDM-PHASE2 shadow observability tool.

Mirrors each live strategy's own already-sealed native-metric halt trigger
into one log file. Read-only with respect to data/trades.db and
data/thursday_ts/trades.csv; writes only to logs/portfolio_decay_shadow.csv.
No halt authority, no new threshold -- see the pre-registration:

    _bmad-output/preregistration_portfolio_decay_monitor_phase2_shadow_observability.md

Run: .venv/bin/python tools/portfolio_decay_shadow.py
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRADES_DB = REPO / "data/trades.db"
THURSDAY_CSV = REPO / "data/thursday_ts/trades.csv"
OUT = REPO / "logs/portfolio_decay_shadow.csv"

FIELDS = [
    "run_at",
    "strategy",
    "n_trades",
    "metric_name",
    "metric_value",
    "gate_n",
    "gate_threshold",
    "trades_to_gate",
    "distance_to_threshold",
    "note",
]

# Frozen per the pre-registration's §3 design table -- one PF gate list per
# strategy, in ascending N. Do not add strategies, metrics, or gates here
# without a pre-registration amendment.
PF_STRATEGIES: dict[str, dict] = {
    "YANK": {"trader_id": "trader-yank", "gates": [(20, 0.90), (30, 1.00)]},
    "MIM-NB": {"trader_id": "trader-mim-nb", "gates": [(30, 0.70)]},
    "GAP-1": {"trader_id": "trader-gap-fade", "gates": [(30, 1.00)]},
}

THURSDAY_RESTART_DATE = "2026-09-17"
THURSDAY_GATE = (30, 0.80)  # (N, Sharpe threshold; PASS if Sharpe > threshold)


def profit_factor(pnl: pd.Series) -> float | None:
    """Gross wins / gross losses. None if no trades; +inf if no losses yet."""
    if len(pnl) == 0:
        return None
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else None
    return gains / losses


def next_gate(n_trades: int, gates: list[tuple[int, float]]) -> tuple[int, float]:
    """The nearest not-yet-reached gate, or the last gate once all are reached."""
    for gate_n, gate_thr in gates:
        if n_trades < gate_n:
            return gate_n, gate_thr
    return gates[-1]


def load_realtime_trades(con: sqlite3.Connection, trader_id: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT timestamp, pnl FROM trades WHERE write_mode='realtime' AND trader_id=?",
        con,
        params=(trader_id,),
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp")


def row_for_pf_strategy(name: str, cfg: dict, con: sqlite3.Connection, run_at: str) -> dict:
    df = load_realtime_trades(con, cfg["trader_id"])
    n = len(df)
    pf = profit_factor(df["pnl"]) if n else None
    gate_n, gate_thr = next_gate(n, cfg["gates"])
    trades_to_gate = max(gate_n - n, 0)
    # Distance is only sealed-meaningful once the gate's own N has been reached --
    # report it blank beforehand rather than imply the rule is live early.
    distance = (pf - gate_thr) if (pf is not None and n >= gate_n and pf != float("inf")) else None
    return {
        "run_at": run_at,
        "strategy": name,
        "n_trades": n,
        "metric_name": "PF",
        "metric_value": pf,
        "gate_n": gate_n,
        "gate_threshold": gate_thr,
        "trades_to_gate": trades_to_gate,
        "distance_to_threshold": distance,
        "note": "",
    }


def sharpe_of_weekly_pnl(weekly_pnl: pd.Series) -> float | None:
    """Annualized (x sqrt(52)) Sharpe of weekly total P&L across both legs.

    This specific formula is this tool's own choice, not lifted from a
    committed backtest script -- the sealed Thursday-short restart doc states
    the Sharpe > 0.80 gate but not a script that computes it. Cross-check
    against the original backtest methodology before trusting this number for
    a real decision; that reconciliation is exactly what the pre-registration's
    §5 manual-verification pass criterion requires before N reaches 30.
    """
    if len(weekly_pnl) < 2:
        return None
    sd = weekly_pnl.std(ddof=1)
    if sd == 0:
        return None
    return float((weekly_pnl.mean() / sd) * (52 ** 0.5))


def row_for_thursday(run_at: str) -> dict:
    gate_n, gate_thr = THURSDAY_GATE
    if not THURSDAY_CSV.exists():
        return {
            "run_at": run_at,
            "strategy": "Kraken Thursday-short",
            "n_trades": 0,
            "metric_name": "Sharpe",
            "metric_value": None,
            "gate_n": gate_n,
            "gate_threshold": gate_thr,
            "trades_to_gate": gate_n,
            "distance_to_threshold": None,
            "note": "data/thursday_ts/trades.csv not found",
        }

    with open(THURSDAY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    # Restart-void rule: only Thursdays on/after 2026-09-17 count.
    eligible = [r for r in rows if r["thursday"] >= THURSDAY_RESTART_DATE]
    n_thursdays = len({r["thursday"] for r in eligible})

    if n_thursdays == 0:
        note = (
            f"restart accrual begins {THURSDAY_RESTART_DATE}; "
            f"0 eligible Thursdays as of this run (pre-restart rows voided)"
        )
        sharpe = None
    else:
        weekly = (
            pd.DataFrame(eligible)
            .assign(pnl_usd=lambda d: d["pnl_usd"].astype(float))
            .groupby("thursday")["pnl_usd"]
            .sum()
        )
        sharpe = sharpe_of_weekly_pnl(weekly)
        note = "Sharpe formula is this tool's own choice -- verify against original methodology"

    return {
        "run_at": run_at,
        "strategy": "Kraken Thursday-short",
        "n_trades": n_thursdays,
        "metric_name": "Sharpe",
        "metric_value": sharpe,
        "gate_n": gate_n,
        "gate_threshold": gate_thr,
        "trades_to_gate": max(gate_n - n_thursdays, 0),
        "distance_to_threshold": (sharpe - gate_thr) if (sharpe is not None and n_thursdays >= gate_n) else None,
        "note": note,
    }


def main() -> None:
    run_at = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(TRADES_DB)
    try:
        rows = [row_for_pf_strategy(name, cfg, con, run_at) for name, cfg in PF_STRATEGIES.items()]
    finally:
        con.close()
    rows.append(row_for_thursday(run_at))

    write_header = not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
