#!/usr/bin/env python3
"""combine_fit_gate.py — Topstep-style trailing-MLL combine-fit gate for any trade list.

Answers a SIZING/RISK question, not a signal question: can a given strategy, at the
size implied by its trade-list P&L, survive a Topstep-style trailing Max-Loss-Limit
account? Run this BEFORE spending a sealed holdout on a full-size / no-micro instrument
— if the smallest tradeable contract busts the trailing DD, the holdout is moot and the
one-shot is saved. (This is exactly what closed the YANK→platinum port on 2026-07-05:
PL slippage PASSED but 1 full 50oz contract busts the $2K MLL — see
_bmad-output/pl_combine_fit_verdict_20260705.md.)

Combine model (defaults = Topstep 50K; from repo MC scripts study_mim_noise_bands_gate2_mc.py
/ rerun_sigc_combined_mc.py):
    start balance   = --account         (default $50,000)
    initial floor   = account - mll     (default $48,000)
    EOD ratchet     : floor = min(account, max(floor, EOD_balance - mll))
    BUST            : balance <= floor at any closed-trade point (trailing MLL touched)
    profit target   : profit >= --target with consistency (biggest_day < 0.5*profit)
The trailing floor only ratchets UP at end of day and caps at the starting balance.

Input CSV must have columns: `exit_time` (parseable timestamp) and `pnl` ($ per the size
being tested). Optional `exit_type` enriches the per-trade SL stats. P&L is applied in
exit_time order; days are bucketed by exit date in --tz. Slippage (--slippage-rt) is
subtracted per round-trip to test the realistic net path alongside gross.

Usage:
    PYTHONPATH=. .venv/bin/python tools/combine_fit_gate.py --trades <csv> [options]

Examples:
    # Platinum (full 50oz), net of measured $34/RT slippage — reproduces the 2026-07-05 verdict
    python tools/combine_fit_gate.py \\
        --trades data/reports/backtest_1year_20260626_025416.csv --slippage-rt 34

    # A micro instrument on a Topstep 50K, no slippage adjustment
    python tools/combine_fit_gate.py --trades trades.csv

    # A different account spec (e.g. 100K: $3,000 MLL, $2,000 daily, $6,000 target)
    python tools/combine_fit_gate.py --trades trades.csv \\
        --account 100000 --mll 3000 --daily-limit 2000 --target 6000
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def run(df, cost, account, mll, daily_limit, target):
    """Return a dict of gate metrics for one cost scenario (cost = $/RT subtracted)."""
    d = df.copy()
    d["p"] = d["pnl"] - cost

    worst = d["p"].min()
    sl = d[d["exit_type"] == "sl"]["p"] if "exit_type" in d.columns else pd.Series(dtype=float)
    daily = d.groupby("day")["p"].sum()

    bal, floor = float(account), float(account) - float(mll)
    peak, max_dd, min_gap = float(account), 0.0, 1e18
    busted, bust_day = False, None
    for day, g in d.groupby("day"):
        for p in g["p"]:
            bal += p
            peak = max(peak, bal)
            max_dd = max(max_dd, peak - bal)
            min_gap = min(min_gap, bal - floor)
            if bal <= floor and not busted:
                busted, bust_day = True, day
        floor = min(float(account), max(floor, bal - float(mll)))  # EOD ratchet

    profit = bal - float(account)
    biggest_day = daily.max()
    consistency_ok = (profit > 0) and (biggest_day < 0.5 * profit)
    return {
        "net_pnl": d["p"].sum(), "worst_trade": worst,
        "sl_median": sl.median() if len(sl) else float("nan"),
        "sl_worst": sl.min() if len(sl) else float("nan"),
        "worst_day": daily.min(), "best_day": daily.max(),
        "days_le_daily": int((daily <= -abs(daily_limit)).sum()),
        "final_bal": bal, "max_dd": max_dd, "min_gap": min_gap,
        "busted": busted, "bust_day": bust_day,
        "profit": profit, "biggest_day": biggest_day,
        "target_ok": profit >= float(target), "consistency_ok": consistency_ok,
    }


def report(label, m, account, mll, daily_limit, target):
    print(f"\n===== {label} =====")
    print(f"  N-net P&L=${m['net_pnl']:,.0f}   worst single trade=${m['worst_trade']:,.0f}"
          f"   median SL=${m['sl_median']:,.0f}   worst SL=${m['sl_worst']:,.0f}")
    ws = -m["worst_trade"]
    print(f"  worst single-trade loss ${ws:,.0f}  vs ${daily_limit:,.0f} daily halt: "
          f"{'EXCEEDS in one trade ❌' if ws > daily_limit else 'within ✅'};  "
          f"vs ${mll:,.0f} MLL: {'EXCEEDS ❌' if ws > mll else 'within ✅'}")
    print(f"  worst day=${m['worst_day']:,.0f}   best day=${m['best_day']:,.0f}   "
          f"days <= -${daily_limit:,.0f}: {m['days_le_daily']}")
    print(f"  trailing-MLL sim: final=${m['final_bal']:,.0f}   max DD from HWM=${m['max_dd']:,.0f}"
          f"   closest approach to floor=${m['min_gap']:,.0f}")
    print(f"  {'❌ BUST on '+str(m['bust_day']) if m['busted'] else '✅ survived — no floor touch'}")
    print(f"  target: profit ${m['profit']:,.0f} >= ${target:,.0f}? {'✅' if m['target_ok'] else '❌'}"
          f"   consistency biggest-day ${m['biggest_day']:,.0f} < 50% profit? "
          f"{'✅' if m['consistency_ok'] else '❌'}")
    # overall gate verdict for this scenario
    passed = (not m["busted"]) and (ws <= mll) and m["target_ok"] and m["consistency_ok"]
    print(f"  → combine-fit {'🟢 PASS' if passed else '🔴 FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser(description="Topstep-style combine-fit gate for a trade list.")
    ap.add_argument("--trades", required=True, help="CSV with exit_time + pnl (+ optional exit_type)")
    ap.add_argument("--slippage-rt", type=float, default=0.0, help="$/round-trip to subtract for the net path")
    ap.add_argument("--account", type=float, default=50000.0)
    ap.add_argument("--mll", type=float, default=2000.0, help="trailing max-loss-limit buffer")
    ap.add_argument("--daily-limit", type=float, default=1000.0)
    ap.add_argument("--target", type=float, default=3000.0)
    ap.add_argument("--tz", default="America/New_York", help="timezone for EOD day bucketing")
    args = ap.parse_args()

    path = Path(args.trades)
    if not path.exists():
        sys.exit(f"trade list not found: {path}")
    df = pd.read_csv(path)
    for col in ("exit_time", "pnl"):
        if col not in df.columns:
            sys.exit(f"CSV missing required column '{col}' (has: {list(df.columns)})")
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["day"] = df["exit_time"].dt.tz_convert(args.tz).dt.date
    df = df.sort_values("exit_time").reset_index(drop=True)

    print(f"combine-fit gate: {path.name}  (N={len(df)})")
    print(f"  account ${args.account:,.0f} | trailing MLL ${args.mll:,.0f} | "
          f"daily ${args.daily_limit:,.0f} | target ${args.target:,.0f} | slippage ${args.slippage_rt:.2f}/RT")

    scenarios = [("GROSS", 0.0)]
    if args.slippage_rt:
        scenarios.append((f"NET @ ${args.slippage_rt:.0f}/RT slippage", args.slippage_rt))
    results = []
    for label, cost in scenarios:
        m = run(df, cost, args.account, args.mll, args.daily_limit, args.target)
        results.append(report(label, m, args.account, args.mll, args.daily_limit, args.target))

    # exit non-zero if the binding (net if present, else gross) scenario fails — CI-friendly
    sys.exit(0 if results[-1] else 1)


if __name__ == "__main__":
    main()
