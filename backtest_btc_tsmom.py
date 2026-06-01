#!/usr/bin/env python3
"""
BTC Time-Series Momentum Backtest (BTC-TSMOM)

Strategy: 28-day log-return momentum signal, long/flat only, 5-day rebalance,
          20-day realized-vol targeting, 15 bps round-trip transaction costs.

Literature basis: Han, Kang & Ryu (2024) — "Time-Series and Cross-Sectional
Momentum in the Cryptocurrency Market: A Comprehensive Analysis under Realistic
Assumptions." Best config (cost-adjusted): lookback=28d, hold=5d, Sharpe=1.51.

Pre-registration: _bmad-output/preregistration_btc_tsmom_backtest.md
                  (committed to git before this script was created)

Usage:
    python backtest_btc_tsmom.py                       # primary config
    python backtest_btc_tsmom.py --sweep               # 36-combo param grid
    python backtest_btc_tsmom.py --lookback 40         # custom lookback
    python backtest_btc_tsmom.py --cost-bps 5          # custom costs
    python backtest_btc_tsmom.py --target-vol 0.20     # custom vol target
    python backtest_btc_tsmom.py --no-oos-gate         # full-sample, no split
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import (
    calc_max_drawdown_pct,
    calc_profit_factor,
    calc_sharpe,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/kraken/PF_XBTUSD_1min.csv")
REPORTS_DIR = Path("data/reports")

# ---------------------------------------------------------------------------
# Pre-registered IS / OOS split dates
# ---------------------------------------------------------------------------
IS_START  = "2024-11-08"
IS_END    = "2025-08-31"
OOS_START = "2025-09-01"
OOS_END   = "2026-05-31"

# ---------------------------------------------------------------------------
# Load and resample
# ---------------------------------------------------------------------------

def load_daily(csv_path: Path) -> pd.DataFrame:
    """Load 1-min Kraken CSV and resample to daily OHLCV."""
    if not csv_path.exists():
        sys.exit(f"ERROR: Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    daily = df.resample("D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return daily


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

def run_strategy(
    daily: pd.DataFrame,
    lookback: int = 28,
    rebalance: int = 5,
    vol_window: int = 20,
    target_vol: float = 0.30,
    max_leverage: float = 2.0,
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """
    Simulate BTC-TSMOM on daily bars.

    Returns a DataFrame with per-day columns:
        log_ret, mom, raw_signal, signal, rvol, size, strat_ret, hodl_ret,
        strat_equity, hodl_equity
    """
    d = daily.copy()

    # Daily log returns
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))

    # 28-day momentum signal
    d["mom"] = np.log(d["close"] / d["close"].shift(lookback))
    d["raw_signal"] = (d["mom"] > 0).astype(int)

    # Rebalance: evaluate signal only at every REBALANCE-th bar (from first valid)
    first_valid = d["raw_signal"].first_valid_index()
    if first_valid is None:
        raise ValueError("No valid signal bars found")
    valid_bars = d.loc[first_valid:].index
    eval_idx = valid_bars[::rebalance]
    # Forward-fill between rebalance dates
    d["signal"] = np.nan
    d.loc[eval_idx, "signal"] = d.loc[eval_idx, "raw_signal"].values
    d["signal"] = d["signal"].ffill().fillna(0).astype(int)

    # Realized vol (annualized with sqrt(252) to match calc_sharpe convention)
    d["rvol"] = d["log_ret"].rolling(vol_window).std() * np.sqrt(252)

    # Vol-targeted size: clip to [0, max_leverage]; 0 when vol is unavailable
    cost_frac = cost_bps / 10_000.0
    size_raw = (target_vol / d["rvol"].replace(0.0, np.nan)).clip(0, max_leverage)
    d["size"] = size_raw.fillna(0.0) * d["signal"]

    # Position change triggers round-trip cost (one leg each way = cost_frac each)
    pos_change = d["signal"].diff().abs()
    d["cost"] = pos_change * cost_frac

    # Strategy return: sized position return minus transaction cost
    d["strat_ret"] = d["size"].shift(1).fillna(0) * d["log_ret"] - d["cost"]
    d["hodl_ret"]  = d["log_ret"]

    # Equity curves (cumulative sum of log returns, exponentiated)
    d["strat_equity"] = np.exp(d["strat_ret"].fillna(0).cumsum())
    d["hodl_equity"]  = np.exp(d["hodl_ret"].fillna(0).cumsum())

    return d


# ---------------------------------------------------------------------------
# Per-trade extraction (for profit factor)
# ---------------------------------------------------------------------------

def extract_trades(d: pd.DataFrame) -> list[dict]:
    """Extract contiguous long blocks as individual trades."""
    trades = []
    in_trade = False
    trade_start = None
    trade_rets: list[float] = []

    for ts, row in d.iterrows():
        sig = int(row["signal"])
        if sig == 1 and not in_trade:
            in_trade = True
            trade_start = ts
            trade_rets = [row["strat_ret"]]
        elif sig == 1 and in_trade:
            trade_rets.append(row["strat_ret"])
        elif sig == 0 and in_trade:
            pnl = sum(trade_rets)
            trades.append({
                "entry_date": trade_start,
                "exit_date": ts,
                "days_held": len(trade_rets),
                "pnl_log": pnl,
                "pnl_pct": (np.exp(pnl) - 1) * 100,
            })
            in_trade = False
            trade_rets = []

    # Close any open trade at end of period
    if in_trade and trade_rets:
        pnl = sum(trade_rets)
        trades.append({
            "entry_date": trade_start,
            "exit_date": d.index[-1],
            "days_held": len(trade_rets),
            "pnl_log": pnl,
            "pnl_pct": (np.exp(pnl) - 1) * 100,
        })

    return trades


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_period(d: pd.DataFrame, label: str) -> dict:
    """Compute strategy and HODL metrics for a sub-period DataFrame."""
    d = d.dropna(subset=["strat_ret", "hodl_ret"])
    if len(d) < 5:
        return {"label": label, "n_days": len(d), "error": "too few bars"}

    strat_daily = d["strat_ret"].tolist()
    hodl_daily  = d["hodl_ret"].tolist()

    # Cumulative equity for drawdown
    strat_equity = np.exp(np.array(strat_daily).cumsum()).tolist()
    hodl_equity  = np.exp(np.array(hodl_daily).cumsum()).tolist()

    # Trades
    trades = extract_trades(d)
    trade_pnls = [t["pnl_log"] for t in trades]

    strat_sharpe   = calc_sharpe(strat_daily)
    hodl_sharpe    = calc_sharpe(hodl_daily)
    strat_pf       = calc_profit_factor(trade_pnls) if trade_pnls else float("nan")
    strat_maxdd    = calc_max_drawdown_pct(strat_equity)
    hodl_maxdd     = calc_max_drawdown_pct(hodl_equity)
    n_trades       = len(trades)
    win_trades     = sum(1 for p in trade_pnls if p > 0)
    win_rate       = win_trades / n_trades if n_trades else float("nan")
    strat_ann_ret  = float(np.exp(np.sum(strat_daily) * 365 / len(strat_daily)) - 1)
    hodl_ann_ret   = float(np.exp(np.sum(hodl_daily) * 365 / len(hodl_daily)) - 1)

    days_long = int(d["signal"].sum())
    pct_invested = days_long / len(d) if len(d) else 0

    return {
        "label":          label,
        "n_days":         len(d),
        "n_trades":       n_trades,
        "win_rate":       win_rate,
        "strat_sharpe":   strat_sharpe,
        "hodl_sharpe":    hodl_sharpe,
        "strat_pf":       strat_pf,
        "strat_maxdd":    strat_maxdd,
        "hodl_maxdd":     hodl_maxdd,
        "strat_ann_ret":  strat_ann_ret,
        "hodl_ann_ret":   hodl_ann_ret,
        "days_long":      days_long,
        "pct_invested":   pct_invested,
        "trades":         trades,
    }


# ---------------------------------------------------------------------------
# Monthly breakdown
# ---------------------------------------------------------------------------

def monthly_breakdown(d: pd.DataFrame) -> list[dict]:
    rows = []
    d = d.dropna(subset=["strat_ret", "hodl_ret"]).copy()
    d["ym"] = d.index.tz_localize(None).to_period("M")
    for ym, grp in d.groupby("ym"):
        strat_ret_m = np.exp(grp["strat_ret"].sum()) - 1
        hodl_ret_m  = np.exp(grp["hodl_ret"].sum()) - 1
        trades_m    = extract_trades(grp)
        n_long      = int(grp["signal"].sum())
        rows.append({
            "month":      str(ym),
            "n_days":     len(grp),
            "n_trades":   len(trades_m),
            "n_long_days": n_long,
            "strat_ret":  strat_ret_m,
            "hodl_ret":   hodl_ret_m,
        })
    return rows


# ---------------------------------------------------------------------------
# Decision rule (OOS)
# ---------------------------------------------------------------------------

def verdict(oos: dict) -> str:
    if "error" in oos:
        return "AMBIGUOUS (insufficient OOS data)"
    sharpe  = oos["strat_sharpe"]
    maxdd   = oos["strat_maxdd"]
    n_tr    = oos["n_trades"]
    beats   = sharpe > oos["hodl_sharpe"]
    if sharpe > 1.0 and beats and n_tr >= 10:
        return "PASS — oos_sharpe>1.0 AND beats HODL AND N_trades>=10"
    if sharpe <= 0.5 or maxdd > 0.40:
        return "FAIL — oos_sharpe≤0.5 OR maxdd>40%"
    return "AMBIGUOUS — edge not confirmed, not disconfirmed"


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def fmt_pct(v) -> str:
    if v != v:
        return "N/A"
    return f"{v*100:+.1f}%"

def fmt_f(v, dec=2) -> str:
    if v != v:
        return "N/A"
    if v == float("inf"):
        return "inf"
    return f"{v:.{dec}f}"


def format_report(
    is_r: dict,
    oos_r: dict | None,
    monthly: list[dict],
    lookback: int,
    rebalance: int,
    vol_window: int,
    target_vol: float,
    cost_bps: float,
    full_sample_r: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("BTC TIME-SERIES MOMENTUM BACKTEST (BTC-TSMOM)")
    lines.append(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("PARAMETERS (pre-registered)")
    lines.append(f"  lookback_days  : {lookback}")
    lines.append(f"  rebalance_days : {rebalance}")
    lines.append(f"  vol_window     : {vol_window}")
    lines.append(f"  target_vol     : {target_vol:.0%}")
    lines.append(f"  max_leverage   : 2.0x")
    lines.append(f"  cost_bps       : {cost_bps:.0f} bps (round-trip)")
    lines.append(f"  direction      : LONG/FLAT only (no shorting)")
    lines.append("")

    def section(r: dict) -> list[str]:
        out = []
        out.append(f"  {'Metric':<28} {'TSMOM':>12} {'HODL':>12}")
        out.append(f"  {'-'*28} {'-'*12} {'-'*12}")
        out.append(f"  {'Days':<28} {r['n_days']:>12} {r['n_days']:>12}")
        out.append(f"  {'Trades (long blocks)':<28} {r['n_trades']:>12} {'—':>12}")
        out.append(f"  {'Win Rate':<28} {fmt_pct(r.get('win_rate','nan')):>12} {'—':>12}")
        out.append(f"  {'Ann. Return':<28} {fmt_pct(r['strat_ann_ret']):>12} {fmt_pct(r['hodl_ann_ret']):>12}")
        out.append(f"  {'Sharpe (ann.)':<28} {fmt_f(r['strat_sharpe']):>12} {fmt_f(r['hodl_sharpe']):>12}")
        out.append(f"  {'Profit Factor':<28} {fmt_f(r['strat_pf']):>12} {'—':>12}")
        out.append(f"  {'Max Drawdown':<28} {fmt_pct(r['strat_maxdd']):>12} {fmt_pct(r['hodl_maxdd']):>12}")
        out.append(f"  {'Days Invested':<28} {r['days_long']:>10}d   {'—':>12}")
        out.append(f"  {'% Time Invested':<28} {r['pct_invested']:>11.0%} {'—':>12}")
        return out

    if oos_r is not None:
        lines.append(f"IN-SAMPLE  ({IS_START} → {IS_END})")
        lines.extend(section(is_r))
        lines.append("")
        lines.append(f"OOS HOLDOUT  ({OOS_START} → {OOS_END})")
        if "error" in oos_r:
            lines.append(f"  ERROR: {oos_r['error']}")
        else:
            lines.extend(section(oos_r))
        lines.append("")
        lines.append("VERDICT (OOS, pre-registered decision rule):")
        lines.append(f"  {verdict(oos_r)}")
    else:
        lines.append(f"FULL SAMPLE  ({IS_START} → {OOS_END})  [--no-oos-gate]")
        lines.extend(section(is_r))

    lines.append("")
    lines.append("MONTHLY BREAKDOWN")
    lines.append(f"  {'Month':<10} {'Days':>5} {'Trades':>7} {'Long_d':>7} {'TSMOM':>8} {'HODL':>8}")
    lines.append(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    for m in monthly:
        lines.append(
            f"  {m['month']:<10} {m['n_days']:>5} {m['n_trades']:>7} "
            f"{m['n_long_days']:>7} {fmt_pct(m['strat_ret']):>8} {fmt_pct(m['hodl_ret']):>8}"
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

SWEEP_LOOKBACKS  = [20, 28, 40, 60]
SWEEP_COSTS      = [5.0, 15.0, 26.0]
SWEEP_TARGET_VOLS = [0.20, 0.30, 0.40]


def run_sweep(daily: pd.DataFrame, rebalance: int = 5, vol_window: int = 20) -> str:
    lines = ["=" * 80]
    lines.append("ROBUSTNESS SWEEP  (full sample — 36 combinations)")
    lines.append(f"  rebalance_days={rebalance}  vol_window={vol_window}")
    lines.append("=" * 80)
    lines.append(
        f"  {'Lookback':>8} {'Cost_bps':>9} {'TargVol':>8} "
        f"{'Sharpe':>8} {'HODL_S':>8} {'PF':>7} {'MaxDD':>7} {'N_tr':>5} {'Beat':>6}"
    )
    lines.append(f"  {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*6}")

    for lb, cost, tv in product(SWEEP_LOOKBACKS, SWEEP_COSTS, SWEEP_TARGET_VOLS):
        try:
            d = run_strategy(daily, lookback=lb, rebalance=rebalance,
                             vol_window=vol_window, target_vol=tv,
                             max_leverage=2.0, cost_bps=cost)
            r = score_period(d, "full")
            if "error" in r:
                lines.append(f"  {lb:>8} {cost:>9.0f} {tv:>8.0%}  — {r['error']}")
                continue
            beat = "YES" if r["strat_sharpe"] > r["hodl_sharpe"] else "no"
            lines.append(
                f"  {lb:>8} {cost:>9.0f} {tv:>8.0%} "
                f"{fmt_f(r['strat_sharpe']):>8} {fmt_f(r['hodl_sharpe']):>8} "
                f"{fmt_f(r['strat_pf']):>7} {fmt_pct(r['strat_maxdd']):>7} "
                f"{r['n_trades']:>5} {beat:>6}"
            )
        except Exception as e:
            lines.append(f"  {lb:>8} {cost:>9.0f} {tv:>8.0%}  ERROR: {e}")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------

def save_reports(report_text: str, trades: list[dict], ts: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = REPORTS_DIR / f"backtest_btc_tsmom_{ts}.txt"
    csv_path = REPORTS_DIR / f"backtest_btc_tsmom_{ts}.csv"
    txt_path.write_text(report_text)
    if trades:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
            writer.writeheader()
            writer.writerows(trades)
        print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Time-Series Momentum Backtest")
    p.add_argument("--lookback",    type=int,   default=28,   help="Momentum lookback in days (default: 28)")
    p.add_argument("--rebalance",   type=int,   default=5,    help="Rebalance interval in days (default: 5)")
    p.add_argument("--vol-window",  type=int,   default=20,   help="Realized vol window in days (default: 20)")
    p.add_argument("--target-vol",  type=float, default=0.30, help="Annualized vol target (default: 0.30)")
    p.add_argument("--cost-bps",    type=float, default=15.0, help="Round-trip cost in bps (default: 15)")
    p.add_argument("--no-oos-gate", action="store_true",      help="Use full sample without IS/OOS split")
    p.add_argument("--sweep",       action="store_true",      help="Run 36-combo robustness sweep instead")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading Kraken 1-min bars and resampling to daily...")
    daily = load_daily(DATA_PATH)
    print(f"Daily bars: {len(daily)} ({daily.index[0].date()} → {daily.index[-1].date()})")

    if args.sweep:
        print("Running robustness sweep (36 combinations)...")
        sweep_text = run_sweep(daily, rebalance=args.rebalance, vol_window=args.vol_window)
        print(sweep_text)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sweep_path = REPORTS_DIR / f"backtest_btc_tsmom_sweep_{ts}.txt"
        sweep_path.write_text(sweep_text)
        print(f"Saved: {sweep_path}")
        return

    # Simulate
    d = run_strategy(
        daily,
        lookback=args.lookback,
        rebalance=args.rebalance,
        vol_window=args.vol_window,
        target_vol=args.target_vol,
        max_leverage=2.0,
        cost_bps=args.cost_bps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.no_oos_gate:
        full_r = score_period(d, "full")
        monthly = monthly_breakdown(d)
        report = format_report(
            is_r=full_r,
            oos_r=None,
            monthly=monthly,
            lookback=args.lookback,
            rebalance=args.rebalance,
            vol_window=args.vol_window,
            target_vol=args.target_vol,
            cost_bps=args.cost_bps,
        )
        all_trades = full_r.get("trades", [])
    else:
        is_mask  = (d.index >= IS_START)  & (d.index <= IS_END)
        oos_mask = (d.index >= OOS_START) & (d.index <= OOS_END)
        is_r     = score_period(d[is_mask],  f"IS  ({IS_START}→{IS_END})")
        oos_r    = score_period(d[oos_mask], f"OOS ({OOS_START}→{OOS_END})")
        monthly  = monthly_breakdown(d)
        report   = format_report(
            is_r=is_r,
            oos_r=oos_r,
            monthly=monthly,
            lookback=args.lookback,
            rebalance=args.rebalance,
            vol_window=args.vol_window,
            target_vol=args.target_vol,
            cost_bps=args.cost_bps,
        )
        all_trades = is_r.get("trades", []) + oos_r.get("trades", [])

    print(report)
    save_reports(report, all_trades, ts)


if __name__ == "__main__":
    main()
