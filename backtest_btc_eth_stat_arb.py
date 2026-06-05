#!/usr/bin/env python3
"""Backtest: BTC-ETH Statistical Arbitrage (BTC-ETH-STATARB).

Pre-registration: _bmad-output/preregistration_btc_eth_stat_arb.md
Sealed: 2026-06-05

Spread construction: log(btc_close) - log(eth_close), z-scored over a rolling
60-bar (1-hour) window. Enter at ±2σ, exit at ±0.25σ, stop at ±3.5σ or 5-day hold.

Requires:
  data/kraken/PF_XBTUSD_1min.csv  (existing)
  data/kraken/PF_ETHUSD_1min.csv  (download via download_kraken_eth_1min.py)

Decision rule (from preregistration):
  PASS if Sharpe >= 1.5 AND MaxDD < 15% AND N_trades >= 30
  FAIL if Sharpe < 0.8 OR MaxDD > 25%
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
ZSCORE_WINDOW = 60          # bars (1-hour)
ENTRY_Z = 2.0
EXIT_Z = 0.25
STOP_Z = 3.5
MAX_HOLD_BARS = 120         # 5 days × 24h
POSITION_NOTIONAL = 10_000  # USD per leg
COST_BPS = 10               # per leg entry/exit
BACKTEST_START = "2024-11-01"
MIN_SPREAD_VOL = 0.005      # minimum rolling std to enter

# --- Data paths ---
BTC_PATH = Path("data/kraken/PF_XBTUSD_1min.csv")
ETH_PATH = Path("data/kraken/PF_ETHUSD_1min.csv")


def load_hourly(path: Path, symbol: str) -> pd.Series:
    if not path.exists():
        logger.error(f"{symbol} data not found at {path}")
        if symbol == "ETH":
            logger.error("Run: .venv/bin/python download_kraken_eth_1min.py")
        sys.exit(1)

    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    hourly = df.set_index("timestamp")["close"].resample("1h").last().dropna()
    hourly = hourly[hourly.index >= BACKTEST_START]
    logger.info(f"{symbol}: {len(hourly)} hourly bars | "
                f"{hourly.index.min()} → {hourly.index.max()}")
    return hourly


def build_spread(btc: pd.Series, eth: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"btc": btc, "eth": eth}).dropna()
    df["log_spread"] = np.log(df["btc"]) - np.log(df["eth"])

    roll = df["log_spread"].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW)
    df["spread_mean"] = roll.mean()
    df["spread_std"] = roll.std()
    df["zscore"] = (df["log_spread"] - df["spread_mean"]) / df["spread_std"]

    df = df.dropna(subset=["zscore"]).reset_index()
    logger.info(f"Spread built: {len(df)} bars | "
                f"z-score range [{df['zscore'].min():.2f}, {df['zscore'].max():.2f}]")
    return df


def run_backtest(df: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Simulate the stat-arb strategy. Returns (pnl_series, trade_list)."""
    pnl_series = []
    trades = []

    position = 0       # +1 = long BTC / short ETH,  -1 = short BTC / long ETH
    entry_z = None
    entry_bar = None
    entry_notional_btc = 0.0
    entry_notional_eth = 0.0
    hold_bars = 0
    cost_per_leg = COST_BPS / 10_000 * POSITION_NOTIONAL

    for i, row in df.iterrows():
        ts = row["timestamp"]
        z = row["zscore"]
        btc_px = row["btc"]
        eth_px = row["eth"]
        spread_std = row["spread_std"]
        period_pnl = 0.0

        if position == 0:
            # No open position — check for entry
            if spread_std < MIN_SPREAD_VOL:
                pnl_series.append((ts, 0.0))
                continue

            if z < -ENTRY_Z:
                # Long BTC / Short ETH
                position = 1
                entry_z = z
                entry_bar = i
                entry_notional_btc = POSITION_NOTIONAL
                entry_notional_eth = POSITION_NOTIONAL
                hold_bars = 0
                period_pnl = -2 * cost_per_leg  # entry cost both legs

            elif z > ENTRY_Z:
                # Short BTC / Long ETH
                position = -1
                entry_z = z
                entry_bar = i
                entry_notional_btc = POSITION_NOTIONAL
                entry_notional_eth = POSITION_NOTIONAL
                hold_bars = 0
                period_pnl = -2 * cost_per_leg

        else:
            hold_bars += 1

            # Per-bar P&L: spread change × notional (approximation for small moves)
            # Long BTC / Short ETH (position=+1): profit when spread rises (z reverts up)
            # Short BTC / Long ETH (position=-1): profit when spread falls (z reverts down)
            if i > 0:
                prev_log_spread = df.loc[i - 1, "log_spread"] if i > 0 else row["log_spread"]
                spread_change = row["log_spread"] - prev_log_spread
                period_pnl = position * spread_change * POSITION_NOTIONAL

            # Check exit conditions
            exit_reason = None
            if abs(z) >= STOP_Z:
                exit_reason = "stop_z"
            elif position == 1 and z >= -EXIT_Z:
                exit_reason = "target_z"
            elif position == -1 and z <= EXIT_Z:
                exit_reason = "target_z"
            elif hold_bars >= MAX_HOLD_BARS:
                exit_reason = "time_stop"

            if exit_reason:
                period_pnl -= 2 * cost_per_leg  # exit cost both legs
                trade_pnl = pnl_series[-hold_bars:]  # approx
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "exit_ts": ts,
                    "direction": "long_btc" if position == 1 else "short_btc",
                    "entry_z": entry_z,
                    "exit_z": z,
                    "hold_bars": hold_bars,
                    "exit_reason": exit_reason,
                })
                position = 0
                entry_z = None
                entry_bar = None
                hold_bars = 0

        pnl_series.append((ts, period_pnl))

    pnl = pd.Series(
        [x[1] for x in pnl_series],
        index=pd.DatetimeIndex([x[0] for x in pnl_series]),
        name="pnl",
    )
    return pnl, trades


def compute_metrics(pnl: pd.Series, trades: list[dict]) -> None:
    periods_per_year = 8760  # hourly bars
    total_return = pnl.sum() / POSITION_NOTIONAL * 100

    std = pnl.std()
    sharpe = (pnl.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    cum = pnl.cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / POSITION_NOTIONAL
    max_dd = dd.min() * 100

    n_trades = len(trades)
    if n_trades > 0:
        wins = sum(1 for t in trades if t.get("pnl", 0) >= 0)
        by_reason = pd.Series([t["exit_reason"] for t in trades]).value_counts()
    else:
        wins = 0
        by_reason = pd.Series(dtype=int)

    # Compute per-trade P&L from cumulative
    trade_pnls = []
    pnl_cum = pnl.cumsum()
    for t in trades:
        entry_ts = pnl.index[t["entry_bar"]] if t["entry_bar"] < len(pnl) else None
        exit_ts = t["exit_ts"]
        if entry_ts is not None:
            pnl_at_entry = pnl_cum[pnl_cum.index <= entry_ts].iloc[-1] if any(pnl_cum.index <= entry_ts) else 0
            pnl_at_exit = pnl_cum[pnl_cum.index <= exit_ts].iloc[-1] if any(pnl_cum.index <= exit_ts) else 0
            trade_pnls.append(pnl_at_exit - pnl_at_entry)
    if trade_pnls:
        gross_win = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
    else:
        pf = 0.0
        win_rate = 0.0

    logger.info("=" * 60)
    logger.info("BTC-ETH STAT ARB RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total return:   {total_return:.2f}%")
    logger.info(f"Sharpe (ann):   {sharpe:.3f}")
    logger.info(f"Max drawdown:   {max_dd:.2f}%")
    logger.info(f"N trades:       {n_trades}")
    logger.info(f"Win rate:       {win_rate:.1%}")
    logger.info(f"Profit factor:  {pf:.3f}")
    logger.info(f"Exit reasons:   {dict(by_reason)}")

    # --- Decision rule ---
    logger.info("-" * 60)
    passes_sharpe = sharpe >= 1.5
    passes_dd = max_dd > -15.0
    passes_n = n_trades >= 30
    fails_sharpe = sharpe < 0.8
    fails_dd = max_dd < -25.0

    if fails_sharpe or fails_dd:
        verdict = "FAIL"
        detail = "No edge; close"
    elif passes_sharpe and passes_dd and passes_n:
        verdict = "PASS"
        detail = "Viable; build live executor"
    else:
        verdict = "AMBIGUOUS"
        detail = "Investigate z-score window or entry threshold"

    logger.info(f"Sharpe ≥ 1.5:   {passes_sharpe}  ({sharpe:.3f})")
    logger.info(f"MaxDD > -15%:   {passes_dd}   ({max_dd:.2f}%)")
    logger.info(f"N ≥ 30 trades:  {passes_n}  ({n_trades})")
    logger.info(f"VERDICT: {verdict} — {detail}")
    logger.info("=" * 60)


def main() -> None:
    logger.info("Loading hourly BTC and ETH data...")
    btc = load_hourly(BTC_PATH, "BTC")
    eth = load_hourly(ETH_PATH, "ETH")

    logger.info("Building log-ratio spread and z-score...")
    df = build_spread(btc, eth)

    if len(df) < ZSCORE_WINDOW * 2:
        logger.error(f"Not enough data: {len(df)} bars < {ZSCORE_WINDOW * 2} minimum")
        sys.exit(1)

    logger.info("Running backtest simulation...")
    pnl, trades = run_backtest(df)

    compute_metrics(pnl, trades)


if __name__ == "__main__":
    main()
