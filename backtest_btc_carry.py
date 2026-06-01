#!/usr/bin/env python3
"""
BTC Funding-Rate Cash-and-Carry Backtest (BTC-CARRY)

Strategy: Long BTC spot + short BTC perpetual (delta-neutral).
          Collect 8h funding payments when annualized rate > hurdle.
          Exit when funding turns persistently negative.

Pre-registration: _bmad-output/preregistration_btc_carry_backtest.md
                  (committed to git before this script was created, commit 35d9e4d)

Usage:
    python backtest_btc_carry.py
    python backtest_btc_carry.py --hurdle 5.0        # lower entry threshold
    python backtest_btc_carry.py --no-threshold       # always in carry (baseline)
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import calc_max_drawdown_pct, calc_sharpe

FUNDING_CSV = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
REPORTS_DIR = Path("data/reports")

# Pre-registered parameters
HURDLE_ANNUAL_PCT  = 10.0   # minimum annualised yield to enter
COST_BPS           = 15.0   # round-trip cost per leg transition
NEG_THRESHOLD      = -0.0001  # -0.01% per 8h
NEG_STOP_PERIODS   = 3      # consecutive periods below threshold to exit


def load_funding(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: Funding data not found: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["funding_rate"] = df["funding_rate"].astype(float)
    return df


def simulate_carry(
    funding: pd.DataFrame,
    hurdle_annual_pct: float = HURDLE_ANNUAL_PCT,
    cost_bps: float = COST_BPS,
    neg_threshold: float = NEG_THRESHOLD,
    neg_stop_periods: int = NEG_STOP_PERIODS,
    always_in: bool = False,
) -> pd.DataFrame:
    """
    Simulate the cash-and-carry.  P&L is expressed as fraction of notional
    (so 0.001 = +0.1% of capital in one 8h window).
    """
    hurdle_per_8h = hurdle_annual_pct / 100.0 / (3 * 365)
    cost_frac = cost_bps / 10_000.0

    f = funding.copy()
    f["ann_rate"] = f["funding_rate"] * 3 * 365

    # Negative-period counter
    neg_count = 0
    in_carry = False
    positions = []

    for i, (ts, row) in enumerate(f.iterrows()):
        rate = row["funding_rate"]
        ann  = row["ann_rate"]

        # Negative funding tracker
        if rate < neg_threshold:
            neg_count += 1
        else:
            neg_count = 0

        # State machine
        if always_in:
            new_pos = 1
        elif not in_carry:
            new_pos = 1 if ann > hurdle_annual_pct / 100.0 else 0
        else:
            new_pos = 0 if neg_count >= neg_stop_periods else 1

        positions.append(new_pos)
        in_carry = bool(new_pos)

    f["position"] = positions
    f["pos_change"] = f["position"].diff().abs().fillna(f["position"].abs())

    # Carry P&L: collect funding when in position, pay when negative
    f["carry_pnl"] = f["position"].shift(1).fillna(0) * f["funding_rate"]
    # Cost on transitions
    f["cost"]      = f["pos_change"] * cost_frac
    f["net_pnl"]   = f["carry_pnl"] - f["cost"]

    # Equity curve
    f["equity"]    = (1 + f["net_pnl"]).cumprod()

    return f


def score(f: pd.DataFrame, label: str, n_periods_per_year: float = 3 * 365) -> dict:
    net     = f["net_pnl"].dropna()
    equity  = f["equity"].dropna().tolist()
    n       = len(net)
    if n < 10:
        return {"label": label, "error": "too few periods"}

    total_net_pnl    = net.sum()
    periods_in_carry = int(f["position"].sum())
    n_trades         = int(f["pos_change"].sum() / 2)  # entry+exit = 2 transitions
    avg_ann_rate     = float(f.loc[f["position"] == 1, "ann_rate"].mean()) if periods_in_carry else 0.0

    # Annualised return = geometric extrapolation from cumulative P&L
    ann_return = float((equity[-1] ** (n_periods_per_year / n)) - 1) if equity[-1] > 0 else -1.0

    # Sharpe on 8h net returns (annualized via √(3*365) to match daily convention)
    # We annualize by scaling to daily: 3 periods/day → annualize daily Sharpe by √252
    daily_net = net.values.reshape(-1, 3).mean(axis=1) if len(net) >= 3 else net.values
    sharpe = calc_sharpe(daily_net.tolist())

    max_dd = calc_max_drawdown_pct(equity)

    return {
        "label":           label,
        "n_8h_periods":    n,
        "n_periods_carry": periods_in_carry,
        "pct_time_carry":  periods_in_carry / n,
        "n_trades":        n_trades,
        "avg_ann_rate":    avg_ann_rate,
        "total_net_pnl":   total_net_pnl,
        "ann_return":      ann_return,
        "sharpe":          sharpe,
        "max_dd":          max_dd,
    }


def monthly_breakdown(f: pd.DataFrame) -> list[dict]:
    f = f.copy()
    f["ym"] = f.index.tz_localize(None).to_period("M")
    rows = []
    for ym, grp in f.groupby("ym"):
        rows.append({
            "month":        str(ym),
            "n_periods":    len(grp),
            "avg_rate_ann": float(grp["ann_rate"].mean() * 100),
            "pct_carry":    float(grp["position"].mean()),
            "net_pnl":      float(grp["net_pnl"].sum()),
        })
    return rows


def verdict(r: dict, hurdle: float) -> str:
    if "error" in r:
        return f"AMBIGUOUS ({r['error']})"
    y = r["ann_return"] * 100
    dd = r["max_dd"]
    if y > hurdle and dd < 0.05:
        return f"PASS — net_yield={y:.1f}%>hurdle AND maxdd={dd*100:.1f}%<5%"
    if y < 5.0 or dd > 0.10:
        return f"FAIL — net_yield={y:.1f}% or maxdd={dd*100:.1f}%>10%"
    return f"AMBIGUOUS — yield={y:.1f}%, maxdd={dd*100:.1f}%"


def format_report(
    r: dict,
    r_always: dict,
    monthly: list[dict],
    hurdle: float,
    cost_bps: float,
) -> str:
    lines = ["=" * 70]
    lines.append("BTC FUNDING-RATE CASH-AND-CARRY BACKTEST (BTC-CARRY)")
    lines.append(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 70)
    lines.append(f"  hurdle_annual_pct : {hurdle:.1f}%")
    lines.append(f"  cost_bps          : {cost_bps:.0f} (round-trip per leg)")
    lines.append(f"  neg_stop          : < {NEG_THRESHOLD*100:.3f}%/8h for {NEG_STOP_PERIODS} periods")
    lines.append("")

    def row(label, filtered, always):
        def f(v, fmt=".2f"):
            return "N/A" if v != v else f"{v:{fmt}}"
        lines.append(f"  {label:<32} {filtered:>12} {always:>14}")

    lines.append(f"  {'Metric':<32} {'Hurdle-filtered':>12} {'Always-in':>14}")
    lines.append(f"  {'-'*32} {'-'*12} {'-'*14}")
    if "error" not in r:
        row("8h periods",      f"{r['n_8h_periods']}", f"{r_always['n_8h_periods']}")
        row("Periods in carry",f"{r['n_periods_carry']}", f"{r_always['n_periods_carry']}")
        row("% time in carry", f"{r['pct_time_carry']*100:.0f}%", f"{r_always['pct_time_carry']*100:.0f}%")
        row("Avg funding (ann)",f"{r['avg_ann_rate']*100:.1f}%", f"{r_always['avg_ann_rate']*100:.1f}%")
        row("Ann. net return",  f"{r['ann_return']*100:+.1f}%", f"{r_always['ann_return']*100:+.1f}%")
        row("Sharpe",           f"{r['sharpe']:.2f}", f"{r_always['sharpe']:.2f}")
        row("Max Drawdown",     f"{r['max_dd']*100:.2f}%", f"{r_always['max_dd']*100:.2f}%")
    else:
        lines.append(f"  ERROR: {r['error']}")

    lines.append("")
    lines.append(f"VERDICT (pre-registered decision rule, hurdle={hurdle:.0f}%):")
    lines.append(f"  {verdict(r, hurdle)}")
    lines.append("")
    lines.append("MONTHLY BREAKDOWN (hurdle-filtered strategy)")
    lines.append(f"  {'Month':<10} {'Periods':>8} {'Avg_Rate%':>10} {'%Carry':>8} {'NetPnL':>10}")
    lines.append(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
    for m in monthly:
        lines.append(
            f"  {m['month']:<10} {m['n_periods']:>8} "
            f"{m['avg_rate_ann']:>9.3f}% {m['pct_carry']:>7.0%} "
            f"{m['net_pnl']:>+10.6f}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Cash-and-Carry Backtest")
    p.add_argument("--hurdle",        type=float, default=10.0,  help="Min annual yield % to enter (default: 10.0)")
    p.add_argument("--cost-bps",      type=float, default=15.0,  help="Round-trip cost bps (default: 15)")
    p.add_argument("--no-threshold",  action="store_true",       help="Always in carry (no hurdle filter)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading funding rate data...")
    funding = load_funding(FUNDING_CSV)
    print(f"Funding rows: {len(funding)} ({funding.index[0].date()} → {funding.index[-1].date()})")

    f_filtered = simulate_carry(
        funding,
        hurdle_annual_pct=args.hurdle,
        cost_bps=args.cost_bps,
        neg_threshold=NEG_THRESHOLD,
        neg_stop_periods=NEG_STOP_PERIODS,
        always_in=args.no_threshold,
    )
    f_always = simulate_carry(
        funding,
        hurdle_annual_pct=0.0,
        cost_bps=args.cost_bps,
        always_in=True,
    )

    r_filtered = score(f_filtered, "hurdle-filtered")
    r_always   = score(f_always,   "always-in")
    monthly    = monthly_breakdown(f_filtered)

    report = format_report(r_filtered, r_always, monthly, args.hurdle, args.cost_bps)
    print(report)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    txt = REPORTS_DIR / f"backtest_btc_carry_{ts}.txt"
    txt.write_text(report)
    print(f"Saved: {txt}")

    # Save per-period CSV
    out_csv = REPORTS_DIR / f"backtest_btc_carry_{ts}.csv"
    f_filtered[["funding_rate", "ann_rate", "position", "carry_pnl", "cost", "net_pnl", "equity"]].to_csv(out_csv)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
