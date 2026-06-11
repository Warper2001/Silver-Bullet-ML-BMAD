#!/usr/bin/env python3
"""
BTC Carry — v2 Exit Rules Backtest

Compares original exit rules vs. pre-registered v2 changes:
  A) Sliding-window neg exit: 3-of-5 payments below threshold (vs. 3 consecutive)
  B) Below-hurdle exit: 4 consecutive payments while rate < hurdle

Pre-registration: _bmad-output/preregistration_btc_carry_exit_rules_v2.md
                  (commit 79612bc — sealed before this script was run)
"""

import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import calc_max_drawdown_pct, calc_sharpe

FUNDING_CSV = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
REPORTS_DIR = Path("data/reports")

# Unchanged from original pre-reg
HURDLE_ANNUAL_PCT = 10.0
COST_BPS          = 15.0
NEG_THRESHOLD     = -0.0001

# Original v1 parameter
NEG_STOP_PERIODS  = 3  # consecutive

# Pre-registered v2 parameters
NEG_WINDOW_SIZE         = 5  # rolling window length
NEG_WINDOW_MIN_NEG      = 3  # exit if >= this many in window are below threshold
BELOW_HURDLE_EXIT_PERIODS = 4  # exit if rate < hurdle for this many consecutive payments


# ---------------------------------------------------------------------------
# v1 simulator (original logic — unchanged)
# ---------------------------------------------------------------------------

def simulate_v1(funding: pd.DataFrame) -> pd.DataFrame:
    hurdle = HURDLE_ANNUAL_PCT / 100.0
    cost   = COST_BPS / 10_000.0

    f = funding.copy()
    f["ann_rate"] = f["funding_rate"] * 3 * 365

    neg_count = 0
    in_carry  = False
    positions = []

    for _, row in f.iterrows():
        rate = row["funding_rate"]
        ann  = row["ann_rate"]

        if rate < NEG_THRESHOLD:
            neg_count += 1
        else:
            neg_count = 0

        if not in_carry:
            new_pos = 1 if ann > hurdle else 0
        else:
            new_pos = 0 if neg_count >= NEG_STOP_PERIODS else 1

        positions.append(new_pos)
        in_carry = bool(new_pos)

    f["position"]  = positions
    f["pos_change"] = f["position"].diff().abs().fillna(f["position"].abs())
    f["carry_pnl"] = f["position"].shift(1).fillna(0) * f["funding_rate"]
    f["cost"]      = f["pos_change"] * cost
    f["net_pnl"]   = f["carry_pnl"] - f["cost"]
    f["equity"]    = (1 + f["net_pnl"]).cumprod()
    return f


# ---------------------------------------------------------------------------
# v2 simulator — sliding window + below-hurdle exit
# ---------------------------------------------------------------------------

def simulate_v2(funding: pd.DataFrame) -> pd.DataFrame:
    hurdle = HURDLE_ANNUAL_PCT / 100.0
    cost   = COST_BPS / 10_000.0

    f = funding.copy()
    f["ann_rate"] = f["funding_rate"] * 3 * 365

    window             = deque()   # sliding window of last NEG_WINDOW_SIZE rates
    below_hurdle_count = 0
    in_carry           = False
    positions          = []
    exit_reasons       = []        # for debugging: "win" / "bhx" / "hold" / "flat"

    for _, row in f.iterrows():
        rate = row["funding_rate"]
        ann  = row["ann_rate"]

        if not in_carry:
            window.clear()
            below_hurdle_count = 0
            new_pos = 1 if ann > hurdle else 0
            exit_reasons.append("entry" if new_pos else "flat")
        else:
            # Update sliding window
            window.append(rate)
            if len(window) > NEG_WINDOW_SIZE:
                window.popleft()

            neg_in_window = sum(r < NEG_THRESHOLD for r in window)

            # Below-hurdle counter
            if ann < hurdle:
                below_hurdle_count += 1
            else:
                below_hurdle_count = 0

            if neg_in_window >= NEG_WINDOW_MIN_NEG:
                new_pos = 0
                exit_reasons.append("win")   # window exit
            elif below_hurdle_count >= BELOW_HURDLE_EXIT_PERIODS:
                new_pos = 0
                exit_reasons.append("bhx")   # below-hurdle exit
            else:
                new_pos = 1
                exit_reasons.append("hold")

        positions.append(new_pos)
        in_carry = bool(new_pos)

    f["position"]    = positions
    f["exit_reason"] = exit_reasons
    f["pos_change"]  = f["position"].diff().abs().fillna(f["position"].abs())
    f["carry_pnl"]   = f["position"].shift(1).fillna(0) * f["funding_rate"]
    f["cost"]        = f["pos_change"] * cost
    f["net_pnl"]     = f["carry_pnl"] - f["cost"]
    f["equity"]      = (1 + f["net_pnl"]).cumprod()
    return f


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(f: pd.DataFrame, label: str) -> dict:
    net    = f["net_pnl"].dropna()
    equity = f["equity"].dropna().tolist()
    n      = len(net)
    if n < 10:
        return {"label": label, "error": "too few periods"}

    n_periods_per_year = 3 * 365
    periods_carry = int(f["position"].sum())
    n_trades      = int(f["pos_change"].sum() / 2)
    avg_ann_rate  = float(f.loc[f["position"] == 1, "ann_rate"].mean()) if periods_carry else 0.0
    total_net_pnl = net.sum()
    ann_return    = float((equity[-1] ** (n_periods_per_year / n)) - 1) if equity[-1] > 0 else -1.0

    daily_net = net.values.reshape(-1, 3).mean(axis=1) if len(net) >= 3 else net.values
    sharpe    = calc_sharpe(daily_net.tolist())
    max_dd    = calc_max_drawdown_pct(equity)

    return {
        "label":           label,
        "n_8h_periods":    n,
        "n_periods_carry": periods_carry,
        "pct_time_carry":  periods_carry / n,
        "n_trades":        n_trades,
        "avg_ann_rate":    avg_ann_rate,
        "total_net_pnl":   total_net_pnl,
        "ann_return":      ann_return,
        "sharpe":          sharpe,
        "max_dd":          max_dd,
    }


def verdict(r: dict) -> str:
    if "error" in r:
        return f"AMBIGUOUS ({r['error']})"
    y  = r["ann_return"] * 100
    dd = r["max_dd"]
    if y > HURDLE_ANNUAL_PCT and dd < 0.05:
        return f"PASS  (yield={y:+.1f}% > {HURDLE_ANNUAL_PCT:.0f}%, maxdd={dd*100:.2f}% < 5%)"
    if y < 5.0 or dd > 0.10:
        return f"FAIL  (yield={y:+.1f}%, maxdd={dd*100:.2f}%)"
    return f"AMBIGUOUS  (yield={y:+.1f}%, maxdd={dd*100:.2f}%)"


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------

def monthly_breakdown(f: pd.DataFrame) -> list[dict]:
    fc = f.copy()
    fc["ym"] = fc.index.tz_localize(None).to_period("M")
    rows = []
    for ym, grp in fc.groupby("ym"):
        rows.append({
            "month":    str(ym),
            "n":        len(grp),
            "avg_ann":  float(grp["ann_rate"].mean() * 100),
            "pct_in":   float(grp["position"].mean()),
            "net_pnl":  float(grp["net_pnl"].sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(r1: dict, r2: dict, monthly1: list, monthly2: list) -> str:
    W = 72
    lines = ["=" * W]
    lines.append("BTC CARRY — v1 (ORIGINAL) vs. v2 (SLIDING WINDOW + BELOW-HURDLE EXIT)")
    lines.append(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"  v1 exit: < {NEG_THRESHOLD*100:.3f}%/8h for {NEG_STOP_PERIODS} consecutive periods")
    lines.append(f"  v2 exit: ({NEG_WINDOW_MIN_NEG}-of-{NEG_WINDOW_SIZE} window below threshold)")
    lines.append(f"           OR (rate < {HURDLE_ANNUAL_PCT:.0f}% ann. for {BELOW_HURDLE_EXIT_PERIODS} consecutive periods)")
    lines.append("=" * W)

    def row(label, v1, v2):
        lines.append(f"  {label:<30} {v1:>16} {v2:>16}")

    row("Metric", "v1 (original)", "v2 (new)")
    row("-" * 30, "-" * 16, "-" * 16)

    def fmt(r, key, pct=False, sign=False):
        if "error" in r:
            return "N/A"
        v = r[key]
        if pct:
            return f"{v*100:+.1f}%" if sign else f"{v*100:.1f}%"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    row("8h periods",       fmt(r1, "n_8h_periods"),    fmt(r2, "n_8h_periods"))
    row("Periods in carry", fmt(r1, "n_periods_carry"), fmt(r2, "n_periods_carry"))
    row("% time in carry",  fmt(r1, "pct_time_carry", pct=True), fmt(r2, "pct_time_carry", pct=True))
    row("# round-trips",    fmt(r1, "n_trades"),        fmt(r2, "n_trades"))
    row("Avg funding (ann)",f"{r1['avg_ann_rate']*100:.1f}%", f"{r2['avg_ann_rate']*100:.1f}%")
    row("Ann. net return",  fmt(r1, "ann_return", pct=True, sign=True), fmt(r2, "ann_return", pct=True, sign=True))
    row("Sharpe",           fmt(r1, "sharpe"),          fmt(r2, "sharpe"))
    row("Max drawdown",     fmt(r1, "max_dd", pct=True), fmt(r2, "max_dd", pct=True))

    lines.append("")
    lines.append(f"  VERDICT v1: {verdict(r1)}")
    lines.append(f"  VERDICT v2: {verdict(r2)}")

    # Exit reason breakdown for v2
    lines.append("")
    lines.append("MONTHLY COMPARISON  (net_pnl as fraction of notional)")
    lines.append(f"  {'Month':<10} {'AvgAnn%':>8} {'v1 in%':>7} {'v1 pnl':>9} {'v2 in%':>7} {'v2 pnl':>9}")
    lines.append(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*9} {'-'*7} {'-'*9}")
    m2_map = {m["month"]: m for m in monthly2}
    for m in monthly1:
        m2 = m2_map.get(m["month"], {})
        lines.append(
            f"  {m['month']:<10} {m['avg_ann']:>7.2f}% "
            f"{m['pct_in']:>6.0%} {m['net_pnl']:>+9.5f} "
            f"{m2.get('pct_in', 0):>6.0%} {m2.get('net_pnl', 0):>+9.5f}"
        )
    lines.append("=" * W)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not FUNDING_CSV.exists():
        sys.exit(f"ERROR: {FUNDING_CSV} not found")

    print("Loading funding rate data...")
    df = pd.read_csv(FUNDING_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["funding_rate"] = df["funding_rate"].astype(float)
    print(f"  {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    print("Running v1 (original)...")
    f1 = simulate_v1(df)
    r1 = score(f1, "v1-original")

    print("Running v2 (sliding window + below-hurdle)...")
    f2 = simulate_v2(df)
    r2 = score(f2, "v2-new")

    # Exit reason summary for v2
    if "exit_reason" in f2.columns:
        win_exits = (f2["exit_reason"] == "win").sum()
        bhx_exits = (f2["exit_reason"] == "bhx").sum()
        print(f"  v2 window exits: {win_exits}  |  below-hurdle exits: {bhx_exits}")

    monthly1 = monthly_breakdown(f1)
    monthly2 = monthly_breakdown(f2)

    report = print_report(r1, r2, monthly1, monthly2)
    print()
    print(report)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = REPORTS_DIR / f"backtest_btc_carry_v2_{ts}.txt"
    out_txt.write_text(report)
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
