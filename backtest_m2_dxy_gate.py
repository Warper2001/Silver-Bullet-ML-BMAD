#!/usr/bin/env python3
"""Backtest: M2/DXY Macro Gate Overlay (MACRO-GATE).

Pre-registration: _bmad-output/preregistration_m2_dxy_macro_gate.md
Sealed: 2026-06-05

Tests whether filtering S26 Kraken crypto trades through a binary macro
condition (DXY weakening + M2 expanding) improves profit factor.

Requires:
  data/macro/DTWEXBGS.csv   (from download_macro_data.py)
  data/macro/M2SL.csv       (from download_macro_data.py)
  logs/s26_crypto_filter_log.csv  (existing S26 trade log)

Decision rule (from preregistration):
  PASS if PF(macro_bull_trades) >= PF(all_trades) + 0.10
        AND macro_bull_trades >= 40% of all trades
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- Frozen parameters (from preregistration) ---
DXY_LOOKBACK_WEEKS = 4
M2_LOOKBACK_WEEKS = 8
DATA_LAG_DAYS = 7
MIN_TRADE_RETENTION = 0.40
PF_IMPROVEMENT_THRESHOLD = 0.10

# --- Data paths ---
DXY_PATH = Path("data/macro/DTWEXBGS.csv")
M2_PATH = Path("data/macro/M2SL.csv")
# Most complete BTC crypto swing trade log with per-trade P&L
S26_LOG = Path("data/reports/backtest_adaptive_regime_btc_20260511_230926.csv")


def load_macro() -> pd.DataFrame:
    if not DXY_PATH.exists() or not M2_PATH.exists():
        logger.error("Macro data not found — run download_macro_data.py first")
        sys.exit(1)

    dxy = pd.read_csv(DXY_PATH, parse_dates=["date"]).rename(columns={"value": "dxy"})
    m2 = pd.read_csv(M2_PATH, parse_dates=["date"]).rename(columns={"value": "m2"})

    # Build daily index from DXY (weekly), forward-fill M2 (monthly) into it
    daily_dates = pd.date_range(
        start=min(dxy["date"].min(), m2["date"].min()),
        end=max(dxy["date"].max(), m2["date"].max()),
        freq="D",
    )
    macro = pd.DataFrame({"date": daily_dates})
    macro = macro.merge(dxy, on="date", how="left")
    macro = macro.merge(m2, on="date", how="left")
    macro["dxy"] = macro["dxy"].ffill()
    macro["m2"] = macro["m2"].ffill()
    macro = macro.dropna(subset=["dxy", "m2"]).reset_index(drop=True)

    # Rate-of-change with lag applied (publication delay)
    lag = DATA_LAG_DAYS
    dxy_lb = DXY_LOOKBACK_WEEKS * 7
    m2_lb = M2_LOOKBACK_WEEKS * 7

    macro["dxy_roc"] = macro["dxy"].shift(lag) / macro["dxy"].shift(lag + dxy_lb) - 1
    macro["m2_roc"] = macro["m2"].shift(lag) / macro["m2"].shift(lag + m2_lb) - 1
    macro["macro_bull"] = (macro["dxy_roc"] < 0) & (macro["m2_roc"] > 0)

    macro = macro.dropna(subset=["dxy_roc", "m2_roc"]).reset_index(drop=True)
    logger.info(
        f"Macro data: {len(macro)} daily rows | "
        f"{macro['date'].min().date()} → {macro['date'].max().date()}"
    )
    bull_pct = macro["macro_bull"].mean()
    logger.info(f"Macro bull regime: {bull_pct:.1%} of days")
    return macro.set_index("date")


def load_s26_trades() -> pd.DataFrame:
    if not S26_LOG.exists():
        logger.error(f"Trade log not found at {S26_LOG}")
        sys.exit(1)

    # Skip comment lines starting with '#'
    df = pd.read_csv(S26_LOG, comment="#")
    logger.info(f"Trade log columns: {list(df.columns)}")

    # Normalise expected columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "entry_time" in cl or "entry_ts" in cl:
            col_map[c] = "timestamp"
        elif "timestamp" in cl and "timestamp" not in col_map.values():
            col_map[c] = "timestamp"
        if "pnl" in cl and "pnl" not in col_map.values():
            col_map[c] = "pnl"
        elif "profit" in cl and "pnl" not in col_map.values():
            col_map[c] = "pnl"
    df = df.rename(columns=col_map)

    if "timestamp" not in df.columns:
        logger.error("Cannot find entry time column in trade log")
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)

    if "pnl" not in df.columns:
        logger.error("No P&L column found")
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.dropna(subset=["pnl"])

    logger.info(f"Trades loaded: {len(df)} | "
                f"{df['date'].min().date()} → {df['date'].max().date()} | "
                f"PF = {df.loc[df['pnl']>0,'pnl'].sum() / abs(df.loc[df['pnl']<0,'pnl'].sum()):.3f}")
    return df


def compute_pf(pnl_series: pd.Series) -> float:
    wins = pnl_series[pnl_series > 0].sum()
    losses = pnl_series[pnl_series < 0].abs().sum()
    if losses == 0:
        return float("inf")
    return wins / losses


def run_backtest(macro: pd.DataFrame, trades: pd.DataFrame) -> None:
    # Join macro signal to each trade
    trades = trades.copy()
    trades["macro_bull"] = trades["date"].map(
        lambda d: macro.loc[macro.index.normalize() == d, "macro_bull"].iloc[0]
        if d in macro.index.normalize() else None
    )

    # Drop trades where macro data is unavailable (outside macro series range)
    n_before = len(trades)
    trades = trades.dropna(subset=["macro_bull"])
    n_dropped = n_before - len(trades)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} trades outside macro data range")

    if "pnl" not in trades.columns:
        logger.warning("No 'pnl' column found in S26 log — using win/loss from 'result' column")
        if "result" in trades.columns:
            trades["pnl"] = trades["result"].map(
                lambda r: 1.0 if str(r).lower() in ("win", "tp", "profit", "1") else -1.0
            )
        else:
            logger.error("Cannot determine trade P&L — log must have 'pnl' or 'result' column")
            logger.info(f"Available columns: {list(trades.columns)}")
            return

    n_total = len(trades)
    n_bull = trades["macro_bull"].sum()
    n_bear = (~trades["macro_bull"]).sum()
    pct_bull = n_bull / n_total if n_total else 0

    pf_all = compute_pf(trades["pnl"])
    pf_bull = compute_pf(trades.loc[trades["macro_bull"], "pnl"])
    pf_bear = compute_pf(trades.loc[~trades["macro_bull"], "pnl"])
    pf_delta = pf_bull - pf_all

    logger.info("=" * 60)
    logger.info("MACRO-GATE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trades:         {n_total}")
    logger.info(f"Macro bull trades:    {n_bull} ({pct_bull:.1%})")
    logger.info(f"Macro bear trades:    {n_bear} ({1-pct_bull:.1%})")
    logger.info(f"PF all trades:        {pf_all:.3f}")
    logger.info(f"PF macro_bull trades: {pf_bull:.3f}  (Δ = {pf_delta:+.3f})")
    logger.info(f"PF macro_bear trades: {pf_bear:.3f}")

    # --- Decision rule ---
    logger.info("-" * 60)
    passes_pf = pf_delta >= PF_IMPROVEMENT_THRESHOLD
    passes_retention = pct_bull >= MIN_TRADE_RETENTION

    if passes_pf and passes_retention:
        verdict = "PASS"
        detail = "Integrate MACRO-GATE into live strategies"
    elif pf_delta < -0.05 or pct_bull < 0.20:
        verdict = "FAIL"
        detail = "Gate adds no value; discard"
    else:
        verdict = "AMBIGUOUS"
        detail = "Try single-factor gates (DXY-only or M2-only)"

    logger.info(f"Passes PF threshold (Δ≥{PF_IMPROVEMENT_THRESHOLD}): {passes_pf}")
    logger.info(f"Passes retention threshold (≥{MIN_TRADE_RETENTION:.0%}): {passes_retention}")
    logger.info(f"VERDICT: {verdict} — {detail}")
    logger.info("=" * 60)

    # Additional breakdown: DXY-only and M2-only gates
    if "dxy_roc" in macro.columns and "m2_roc" in macro.columns:
        trades["dxy_bull"] = trades["date"].map(
            lambda d: (macro.loc[macro.index.normalize() == d, "dxy_roc"].iloc[0] < 0)
            if d in macro.index.normalize() else None
        )
        trades["m2_bull"] = trades["date"].map(
            lambda d: (macro.loc[macro.index.normalize() == d, "m2_roc"].iloc[0] > 0)
            if d in macro.index.normalize() else None
        )
        trades = trades.dropna(subset=["dxy_bull", "m2_bull"])
        pf_dxy = compute_pf(trades.loc[trades["dxy_bull"], "pnl"])
        pf_m2 = compute_pf(trades.loc[trades["m2_bull"], "pnl"])
        logger.info(f"Single-factor PF | DXY-only bull: {pf_dxy:.3f} | M2-only bull: {pf_m2:.3f}")


def main() -> None:
    logger.info("Loading macro data...")
    macro = load_macro()

    logger.info("Loading S26 trade log...")
    trades = load_s26_trades()

    run_backtest(macro, trades)


if __name__ == "__main__":
    main()
