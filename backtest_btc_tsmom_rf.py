#!/usr/bin/env python3
"""
BTC-TSMOM-RF: TSMOM + 200-day SMA Regime Filter

Strategy: 28-day log-return momentum, long/flat only, 5-day rebalance,
          vol-targeted sizing — PLUS regime condition: only long when
          close price > 200-day SMA.

Pre-registration: _bmad-output/preregistration_btc_tsmom_rf_backtest.md
                  (committed at 35d9e4d, before this file was written)

All TSMOM parameters identical to base BTC-TSMOM (pre-reg 86842af).
The 200-day SMA is the only addition.

Usage:
    python backtest_btc_tsmom_rf.py               # TSMOM-RF vs base TSMOM vs HODL
    python backtest_btc_tsmom_rf.py --no-regime   # plain TSMOM (for reference)
"""

import argparse
import csv as csv_mod
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import (
    calc_max_drawdown_pct,
    calc_profit_factor,
    calc_sharpe,
)

DATA_PATH   = Path("data/kraken/PF_XBTUSD_1min.csv")
REPORTS_DIR = Path("data/reports")

IS_START  = "2024-11-08"
IS_END    = "2025-08-31"
OOS_START = "2025-09-01"
OOS_END   = "2026-05-31"

# Pre-registered parameters (identical to base TSMOM)
LOOKBACK    = 28
REBALANCE   = 5
VOL_WINDOW  = 20
TARGET_VOL  = 0.30
MAX_LEV     = 2.0
COST_BPS    = 15.0
SMA_WINDOW  = 200   # NEW: regime filter


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    daily = (
        df.set_index("timestamp")
        .sort_index()
        .resample("D")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .dropna(subset=["close"])
    )
    return daily


def run_strategy(daily: pd.DataFrame, use_regime: bool = True) -> pd.DataFrame:
    d = daily.copy()
    cost_frac = COST_BPS / 10_000.0

    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    d["mom"]     = np.log(d["close"] / d["close"].shift(LOOKBACK))
    d["sma_200"] = d["close"].rolling(SMA_WINDOW).mean()
    d["rvol"]    = d["log_ret"].rolling(VOL_WINDOW).std() * np.sqrt(252)

    if use_regime:
        d["raw_signal"] = ((d["mom"] > 0) & (d["close"] > d["sma_200"])).astype(int)
    else:
        d["raw_signal"] = (d["mom"] > 0).astype(int)

    # 5-day rebalance: evaluate at every 5th bar from first valid signal
    first_valid = d["raw_signal"].first_valid_index()
    eval_idx = d.loc[first_valid:].index[::REBALANCE]
    d["signal"] = np.nan
    d.loc[eval_idx, "signal"] = d.loc[eval_idx, "raw_signal"].values
    d["signal"] = d["signal"].ffill().fillna(0).astype(int)

    d["size"] = (TARGET_VOL / d["rvol"].replace(0.0, np.nan)).clip(0, MAX_LEV).fillna(0) * d["signal"]

    pos_change  = d["signal"].diff().abs()
    d["cost"]   = pos_change * cost_frac
    d["strat_ret"] = d["size"].shift(1).fillna(0) * d["log_ret"] - d["cost"]
    d["hodl_ret"]  = d["log_ret"]

    d["strat_equity"] = np.exp(d["strat_ret"].fillna(0).cumsum())
    d["hodl_equity"]  = np.exp(d["hodl_ret"].fillna(0).cumsum())
    return d


def extract_trades(d: pd.DataFrame) -> list[dict]:
    trades, in_trade, t_start, t_rets = [], False, None, []
    for ts, row in d.iterrows():
        sig = int(row["signal"])
        if sig == 1 and not in_trade:
            in_trade, t_start, t_rets = True, ts, [row["strat_ret"]]
        elif sig == 1:
            t_rets.append(row["strat_ret"])
        elif in_trade:
            trades.append({"entry": t_start, "exit": ts, "days": len(t_rets), "pnl": sum(t_rets)})
            in_trade, t_rets = False, []
    if in_trade and t_rets:
        trades.append({"entry": t_start, "exit": d.index[-1], "days": len(t_rets), "pnl": sum(t_rets)})
    return trades


def score_period(d: pd.DataFrame, label: str) -> dict:
    d = d.dropna(subset=["strat_ret", "hodl_ret"])
    if len(d) < 5:
        return {"label": label, "error": "too few bars"}
    strat_eq = np.exp(np.array(d["strat_ret"].tolist()).cumsum()).tolist()
    hodl_eq  = np.exp(np.array(d["hodl_ret"].tolist()).cumsum()).tolist()
    trades   = extract_trades(d)
    trade_pnls = [t["pnl"] for t in trades]
    return {
        "label":         label,
        "n_days":        len(d),
        "n_trades":      len(trades),
        "win_rate":      sum(1 for p in trade_pnls if p > 0) / len(trades) if trades else float("nan"),
        "strat_sharpe":  calc_sharpe(d["strat_ret"].tolist()),
        "hodl_sharpe":   calc_sharpe(d["hodl_ret"].tolist()),
        "strat_pf":      calc_profit_factor(trade_pnls) if trade_pnls else float("nan"),
        "strat_maxdd":   calc_max_drawdown_pct(strat_eq),
        "hodl_maxdd":    calc_max_drawdown_pct(hodl_eq),
        "strat_ann_ret": float(np.exp(sum(d["strat_ret"].tolist()) * 365 / len(d)) - 1),
        "hodl_ann_ret":  float(np.exp(sum(d["hodl_ret"].tolist()) * 365 / len(d)) - 1),
        "days_long":     int(d["signal"].sum()),
        "pct_invested":  d["signal"].mean(),
        "trades":        trades,
    }


def verdict_rf(oos_rf: dict, base_tsmom_sharpe: float = -0.73) -> str:
    if "error" in oos_rf:
        return "AMBIGUOUS (insufficient data)"
    s = oos_rf["strat_sharpe"]
    n = oos_rf["n_trades"]
    beats_base = s > base_tsmom_sharpe
    if s > 0.8 and beats_base and n >= 8:
        return f"PASS — oos_sharpe={s:.2f}>0.8 AND beats base({base_tsmom_sharpe}) AND N={n}>=8"
    if s <= 0.0:
        improvement = s - base_tsmom_sharpe
        return (f"FAIL — oos_sharpe={s:.2f}<=0.0 (though +{improvement:+.2f} vs base {base_tsmom_sharpe}); "
                f"not yet positive")
    return f"AMBIGUOUS — oos_sharpe={s:.2f} (beats base={beats_base}, N={n}); marginal"


def fmt(v, dec=2):
    if v != v: return "N/A"
    if v == float("inf"): return "inf"
    return f"{v:.{dec}f}"

def fmtp(v):
    if v != v: return "N/A"
    return f"{v*100:+.1f}%"


def format_report(rf: dict, base: dict, hodl_label: str, period: str, oos: bool) -> list[str]:
    lines = []
    lines.append(f"  {'Metric':<28} {'TSMOM-RF':>10} {'TSMOM-base':>11} {'HODL':>8}")
    lines.append(f"  {'-'*28} {'-'*10} {'-'*11} {'-'*8}")
    for lbl, rv, bv in [
        ("Days",          fmt(rf["n_days"],0),   fmt(base["n_days"],0),   fmt(rf["n_days"],0)),
        ("Trades",        fmt(rf["n_trades"],0), fmt(base["n_trades"],0), "—"),
        ("Win Rate",      fmtp(rf["win_rate"]),  fmtp(base["win_rate"]),  "—"),
        ("Ann. Return",   fmtp(rf["strat_ann_ret"]), fmtp(base["strat_ann_ret"]), fmtp(rf["hodl_ann_ret"])),
        ("Sharpe",        fmt(rf["strat_sharpe"]),   fmt(base["strat_sharpe"]),   fmt(rf["hodl_sharpe"])),
        ("Profit Factor", fmt(rf["strat_pf"]),        fmt(base["strat_pf"]),        "—"),
        ("Max Drawdown",  fmtp(rf["strat_maxdd"]),    fmtp(base["strat_maxdd"]),    fmtp(rf["hodl_maxdd"])),
        ("Days Invested", f"{rf['days_long']}d",       f"{base['days_long']}d",       "—"),
        ("% Invested",    fmtp(rf["pct_invested"]),    fmtp(base["pct_invested"]),    "—"),
    ]:
        lines.append(f"  {lbl:<28} {rv:>10} {bv:>11} {hodl_label if lbl=='Ann. Return' else ('—' if lbl not in ('Ann. Return','Sharpe','Max Drawdown') else fmt(rf['hodl_sharpe']) if lbl=='Sharpe' else fmtp(rf['hodl_maxdd'])):>8}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-regime", action="store_true", help="Disable regime filter (plain TSMOM)")
    args = p.parse_args()

    print("Loading and resampling Kraken 1-min data...")
    daily = load_daily(DATA_PATH)
    print(f"Daily bars: {len(daily)} ({daily.index[0].date()} → {daily.index[-1].date()})")

    use_regime = not args.no_regime
    d_rf   = run_strategy(daily, use_regime=True)
    d_base = run_strategy(daily, use_regime=False)

    is_mask  = (daily.index >= IS_START)  & (daily.index <= IS_END)
    oos_mask = (daily.index >= OOS_START) & (daily.index <= OOS_END)

    is_rf   = score_period(d_rf[is_mask],    "IS TSMOM-RF")
    is_base = score_period(d_base[is_mask],  "IS TSMOM-base")
    oos_rf  = score_period(d_rf[oos_mask],   "OOS TSMOM-RF")
    oos_base = score_period(d_base[oos_mask],"OOS TSMOM-base")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lines = []
    lines.append("=" * 70)
    lines.append("BTC-TSMOM-RF: TSMOM + 200-DAY SMA REGIME FILTER")
    lines.append(f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("PARAMETERS")
    lines.append(f"  lookback={LOOKBACK}d  rebalance={REBALANCE}d  vol_window={VOL_WINDOW}d")
    lines.append(f"  target_vol={TARGET_VOL:.0%}  max_lev={MAX_LEV}x  cost={COST_BPS:.0f}bps")
    lines.append(f"  regime_filter: close > SMA({SMA_WINDOW}d)  [{'ENABLED' if use_regime else 'DISABLED'}]")
    lines.append("")

    lines.append(f"IN-SAMPLE ({IS_START} → {IS_END})")
    # Side-by-side table
    lines.append(f"  {'Metric':<28} {'TSMOM-RF':>10} {'TSMOM-base':>11} {'HODL':>8}")
    lines.append(f"  {'-'*28} {'-'*10} {'-'*11} {'-'*8}")
    for lbl, rf_v, base_v, hodl_v in [
        ("Days",          fmt(is_rf["n_days"],0),   fmt(is_base["n_days"],0),   "—"),
        ("Trades",        fmt(is_rf["n_trades"],0), fmt(is_base["n_trades"],0), "—"),
        ("Win Rate",      fmtp(is_rf["win_rate"]),  fmtp(is_base["win_rate"]),  "—"),
        ("Ann. Return",   fmtp(is_rf["strat_ann_ret"]),  fmtp(is_base["strat_ann_ret"]),  fmtp(is_rf["hodl_ann_ret"])),
        ("Sharpe",        fmt(is_rf["strat_sharpe"]),    fmt(is_base["strat_sharpe"]),    fmt(is_rf["hodl_sharpe"])),
        ("Profit Factor", fmt(is_rf["strat_pf"]),        fmt(is_base["strat_pf"]),        "—"),
        ("Max Drawdown",  fmtp(is_rf["strat_maxdd"]),    fmtp(is_base["strat_maxdd"]),    fmtp(is_rf["hodl_maxdd"])),
        ("Days Invested", f"{is_rf['days_long']}d",       f"{is_base['days_long']}d",       "—"),
    ]:
        lines.append(f"  {lbl:<28} {rf_v:>10} {base_v:>11} {hodl_v:>8}")

    lines.append("")
    lines.append(f"OOS HOLDOUT ({OOS_START} → {OOS_END})")
    lines.append(f"  {'Metric':<28} {'TSMOM-RF':>10} {'TSMOM-base':>11} {'HODL':>8}")
    lines.append(f"  {'-'*28} {'-'*10} {'-'*11} {'-'*8}")
    for lbl, rf_v, base_v, hodl_v in [
        ("Days",          fmt(oos_rf["n_days"],0),   fmt(oos_base["n_days"],0),   "—"),
        ("Trades",        fmt(oos_rf["n_trades"],0), fmt(oos_base["n_trades"],0), "—"),
        ("Win Rate",      fmtp(oos_rf["win_rate"]),  fmtp(oos_base["win_rate"]),  "—"),
        ("Ann. Return",   fmtp(oos_rf["strat_ann_ret"]),  fmtp(oos_base["strat_ann_ret"]),  fmtp(oos_rf["hodl_ann_ret"])),
        ("Sharpe",        fmt(oos_rf["strat_sharpe"]),    fmt(oos_base["strat_sharpe"]),    fmt(oos_rf["hodl_sharpe"])),
        ("Profit Factor", fmt(oos_rf["strat_pf"]),        fmt(oos_base["strat_pf"]),        "—"),
        ("Max Drawdown",  fmtp(oos_rf["strat_maxdd"]),    fmtp(oos_base["strat_maxdd"]),    fmtp(oos_rf["hodl_maxdd"])),
        ("Days Invested", f"{oos_rf['days_long']}d",       f"{oos_base['days_long']}d",       "—"),
    ]:
        lines.append(f"  {lbl:<28} {rf_v:>10} {base_v:>11} {hodl_v:>8}")

    lines.append("")
    lines.append("VERDICT (pre-registered decision rule, base TSMOM OOS Sharpe = −0.73):")
    lines.append(f"  {verdict_rf(oos_rf)}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    txt = REPORTS_DIR / f"backtest_btc_tsmom_rf_{ts}.txt"
    txt.write_text(report)
    print(f"Saved: {txt}")

    # Per-trade CSV
    all_trades = oos_rf.get("trades", []) + is_rf.get("trades", [])
    if all_trades:
        c = REPORTS_DIR / f"backtest_btc_tsmom_rf_{ts}.csv"
        with open(c, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=["entry","exit","days","pnl"])
            w.writeheader(); w.writerows(all_trades)
        print(f"Saved: {c}")


if __name__ == "__main__":
    main()
