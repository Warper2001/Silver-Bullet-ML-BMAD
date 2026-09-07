#!/usr/bin/env python3
"""PL Gate-1 offline net analysis.

Reads a trade list emitted by ``backtest_tier2_1year_validation.py --instrument pl``
and evaluates it against the sealed decision rule in
``_bmad-output/preregistration_pl_gate1_holdout.md``.

The backtest runs in structural mode with ``commission_per_roundtrip = 0``, so every
``pnl`` in the CSV is GROSS. Net is applied here, offline, at the measured cost:

    net_i = gross_i - COST_PER_RT      (1 full PL contract)

Binding cost is $34.00/RT (measured pooled median spread $30.00 + $4.00 commission).
$44.00/RT (worst qualifying session) is reported as a non-binding sensitivity.

Usage:
    PYTHONPATH=. python tools/pl_gate1_net_analysis.py <trades.csv> [--label NAME]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Frozen in the pre-registration — do not edit without a new seal.
COST_BINDING = 34.00
COST_SENSITIVITY = 44.00

# Sealed decision rule thresholds.
PASS_NET_PF = 1.10
N_FLOOR = 15


def _profit_factor(pnls: list[float]) -> float:
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _max_drawdown(pnls: list[float]) -> float:
    """Peak-to-trough of the cumulative trade-by-trade equity curve (negative $)."""
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def load_trades(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pnl"] = float(r["pnl"])
        r["entry_dt"] = datetime.fromisoformat(r["entry_time"])
        r["exit_dt"] = datetime.fromisoformat(r["exit_time"])
    return rows


def report(rows: list[dict], cost: float, binding: bool) -> dict:
    gross = [r["pnl"] for r in rows]
    net = [g - cost for g in gross]
    n = len(rows)

    # Group net P&L by EXIT date (a trade's realized day).
    by_day: dict = defaultdict(float)
    for r, netpnl in zip(rows, net):
        by_day[r["exit_dt"].date()] += netpnl
    top3 = sorted(by_day.values(), reverse=True)[:3]
    ex_top3 = sum(net) - sum(top3)

    by_month: dict = defaultdict(float)
    for r, netpnl in zip(rows, net):
        by_month[r["exit_dt"].strftime("%Y-%m")] += netpnl

    exits: dict = defaultdict(int)
    for r in rows:
        exits[r["exit_type"]] += 1

    wins = sum(1 for p in net if p > 0)
    res = {
        "n": n,
        "cost": cost,
        "gross_pf": _profit_factor(gross),
        "gross_total": sum(gross),
        "net_pf": _profit_factor(net),
        "net_total": sum(net),
        "net_avg": sum(net) / n if n else 0.0,
        "wr": wins / n if n else 0.0,
        "ex_top3": ex_top3,
        "max_dd": _max_drawdown(net),
        "worst_trade": min(net) if net else 0.0,
        "exits": dict(exits),
        "by_month": dict(sorted(by_month.items())),
        "top3_days": top3,
    }

    tag = "BINDING" if binding else "SENSITIVITY (non-binding)"
    print(f"\n{'='*66}")
    print(f"  Cost ${cost:.2f}/RT  —  {tag}")
    print(f"{'='*66}")
    print(f"  N                 : {n}")
    print(f"  Gross PF          : {res['gross_pf']:.4f}   (total ${res['gross_total']:+,.2f})")
    print(f"  NET PF            : {res['net_pf']:.4f}   (total ${res['net_total']:+,.2f})")
    print(f"  Net avg / trade   : ${res['net_avg']:+,.2f}")
    print(f"  Win rate          : {res['wr']*100:.1f}%")
    print(f"  Ex-top-3-days net : ${res['ex_top3']:+,.2f}   (top3 days: "
          f"{', '.join(f'${d:+,.0f}' for d in top3)})")
    print(f"  Max drawdown      : ${res['max_dd']:+,.2f}")
    print(f"  Worst single trade: ${res['worst_trade']:+,.2f}")
    print(f"  Exit mix          : {res['exits']}")
    print(f"  Monthly net       : " + "  ".join(f"{m} ${v:+,.0f}" for m, v in res["by_month"].items()))
    return res


def verdict(res: dict) -> str:
    n, net_pf, ex_top3 = res["n"], res["net_pf"], res["ex_top3"]
    if n < N_FLOOR:
        return (f"INSUFFICIENT — N={n} < {N_FLOOR}. PARK; no verdict on the edge. "
                f"(Sealed rule: N floor binds regardless of PF.)")
    if net_pf >= PASS_NET_PF and ex_top3 > 0:
        return (f"PASS — net PF {net_pf:.4f} >= {PASS_NET_PF}, N={n} >= {N_FLOOR}, "
                f"ex-top-3-days ${ex_top3:+,.2f} > $0. Authorizes drafting a DEPLOYMENT "
                f"pre-registration (TS SIM, 1 contract). Nothing trades from this result.")
    if net_pf >= PASS_NET_PF and ex_top3 <= 0:
        # NOT a permanent close: the sealed FAIL clause is net PF < 1.00. Failing only the
        # fat-day clause means "no deployment authorization", i.e. PARK — same disposition
        # as MARGINAL. Do not report this as FAIL; it does not close PL.
        return (f"NOT PASS → PARK (fat-day clause) — net PF {net_pf:.4f} clears {PASS_NET_PF} "
                f"and N={n} clears {N_FLOOR}, but ex-top-3-days is ${ex_top3:+,.2f} <= $0. "
                f"PASS requires all three clauses, so no deployment path; but the sealed FAIL "
                f"clause (net PF < 1.00) is NOT met, so PL is parked, not closed. The edge is "
                f"real but tail-carried.")
    if net_pf >= 1.00:
        return (f"MARGINAL — net PF {net_pf:.4f} in [1.00, {PASS_NET_PF}), N={n}. PARK; no "
                f"deployment path. Read at face value, not as 'almost passed'.")
    return (f"FAIL — net PF {net_pf:.4f} < 1.00 with N={n} >= {N_FLOOR}. PL closed as a net "
            f"candidate on every vehicle: 'structural edge did not survive the holdout at "
            f"measured cost.' No re-runs, no subgroup rescue.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trades_csv", type=Path)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = load_trades(args.trades_csv)
    print(f"\nPL Gate-1 net analysis — {args.label or args.trades_csv.name}")
    print(f"Trade list: {args.trades_csv}")
    if rows:
        print(f"Window    : {rows[0]['entry_dt'].date()} → {rows[-1]['exit_dt'].date()}")

    binding = report(rows, COST_BINDING, binding=True)
    report(rows, COST_SENSITIVITY, binding=False)

    print(f"\n{'='*66}")
    print("  SEALED DECISION RULE (evaluated at the BINDING $34.00/RT only)")
    print(f"{'='*66}")
    print(f"  {verdict(binding)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
