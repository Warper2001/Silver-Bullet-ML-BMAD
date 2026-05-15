#!/usr/bin/env python3
"""Adaptive Regime-Based BTC Strategy Backtest.

Strategy: Cansever, A. (2025). Adaptive Regime-Based Trading on Bitcoin.
ResearchGate DOI:10.13140/RG.2.2.35154.41922

Logic:
1. Resample 1-min Kraken data to 15-min bars
2. Compute normalized price slope over rolling window
3. Classify regime (UP/DOWN/SIDEWAYS) using historical percentile bands
4. Apply daily HTF trend filter: EMA(50) vs EMA(100) on daily bars
5. Enter long (UP+BULL) or short (DOWN+BEAR) when regimes align
6. Exit via ATR trailing stop evaluated on each 15m bar close

Data: data/kraken/PF_XBTUSD_1min.csv (Nov 2024 – May 2026)
Split: train < 2025-11-08 | holdout >= 2025-11-08 (same as silver bullet)
Position: 0.1 BTC | Commission: 0.05% taker per side
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ── Strategy parameters ──────────────────────────────────────────────────────
DATA_PATH      = "data/kraken/PF_XBTUSD_1min.csv"
RESAMPLE_TF    = "15min"
SPLIT_DATE     = "2025-11-08"

SLOPE_WINDOW   = 20          # 15m bars for slope calc (20 × 15min = 5 hr)
SLOPE_HISTORY  = 100         # bars for rolling percentile calibration
PCT_UP         = 70          # slope percentile threshold for UP regime
PCT_DOWN       = 30          # slope percentile threshold for DOWN regime

ATR_PERIOD     = 14
ATR_MULTIPLIER = 2.5

HTF_FAST_MA    = 50          # daily EMA — warm-up: ~50 days from Nov 2024 = Jan 2025
HTF_SLOW_MA    = 100         # daily EMA — warm-up: ~100 days = Feb 2025

BTC_SIZE       = 0.1         # BTC per trade ($0.10 P&L per $1 price move)
COMMISSION_PCT = 0.0005      # 0.05% taker per side (Kraken futures)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_1min() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])


# ── Indicators ───────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def compute_regime(close: pd.Series) -> pd.Series:
    """Slope-based regime: UP / DOWN / SIDEWAYS using rolling percentile bands.

    Thresholds are computed from slope.shift(1) so that slope[t] is compared
    to a quantile derived only from slope[t-HISTORY..t-1] — no look-ahead.
    """
    slope = (close - close.shift(SLOPE_WINDOW)) / (close.shift(SLOPE_WINDOW) * SLOPE_WINDOW)
    # Shift by 1 before rolling: threshold at t uses only history up to t-1
    slope_hist  = slope.shift(1)
    up_thresh   = slope_hist.rolling(SLOPE_HISTORY).quantile(PCT_UP   / 100.0)
    down_thresh = slope_hist.rolling(SLOPE_HISTORY).quantile(PCT_DOWN / 100.0)
    regime = pd.Series("SIDEWAYS", index=close.index, dtype=object)
    regime[slope > up_thresh]   = "UP"
    regime[slope < down_thresh] = "DOWN"
    return regime


def compute_htf_trend(df_daily: pd.DataFrame) -> pd.Series:
    """Returns 'BULL' / 'BEAR' / 'NEUTRAL' indexed by UTC date."""
    ema_fast = df_daily["close"].ewm(span=HTF_FAST_MA, min_periods=HTF_FAST_MA).mean()
    ema_slow = df_daily["close"].ewm(span=HTF_SLOW_MA, min_periods=HTF_SLOW_MA).mean()
    trend = pd.Series("NEUTRAL", index=df_daily.index, dtype=object)
    trend[ema_fast > ema_slow] = "BULL"
    trend[ema_fast < ema_slow] = "BEAR"
    # Shift by 1 to avoid look-ahead (use prior day's trend for today's signals)
    return trend.shift(1)


# ── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(df_15m: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_15m.copy()
    df["atr"]    = compute_atr(df, ATR_PERIOD)
    df["regime"] = compute_regime(df["close"])

    daily_trend = compute_htf_trend(df_daily)
    # Map daily trend forward to each 15m bar via date
    df["date_only"] = df.index.normalize()
    df["htf"] = df["date_only"].map(daily_trend)

    # Drop warmup bars (need SLOPE_WINDOW + SLOPE_HISTORY for regime + HTF_SLOW_MA days)
    df = df.dropna(subset=["atr", "regime", "htf"])

    split_ts = pd.Timestamp(SPLIT_DATE, tz="UTC")
    trades = []
    position = None

    bars = df.itertuples()
    prev_bar = None

    for bar in tqdm(bars, total=len(df), desc="Backtesting", unit="bar"):
        if prev_bar is None:
            prev_bar = bar
            continue

        ts    = bar.Index
        close = bar.close

        # ── Manage open position ──────────────────────────────────────────
        if position is not None:
            direction  = position["direction"]
            trail_stop = position["trail_stop"]

            # Update trailing stop using previous bar's close and ATR
            if direction == "long":
                new_stop = prev_bar.close - ATR_MULTIPLIER * prev_bar.atr
                trail_stop = max(trail_stop, new_stop)
            else:
                new_stop = prev_bar.close + ATR_MULTIPLIER * prev_bar.atr
                trail_stop = min(trail_stop, new_stop)
            position["trail_stop"] = trail_stop

            # Check exit on current bar close
            exit_hit = (direction == "long"  and close <= trail_stop) or \
                       (direction == "short" and close >= trail_stop)

            if exit_hit:
                ep = position["entry_price"]
                xp = trail_stop
                comm = COMMISSION_PCT * (ep + xp) * BTC_SIZE
                pnl = ((xp - ep) if direction == "long" else (ep - xp)) * BTC_SIZE - comm

                trades.append({
                    "entry_time":  position["entry_time"],
                    "exit_time":   ts,
                    "direction":   direction,
                    "entry_price": ep,
                    "exit_price":  xp,
                    "pnl":         pnl,
                    "bars_held":   position["bars_held"],
                    "split":       "train" if position["entry_time"] < split_ts else "holdout",
                })
                position = None

        if position is not None:
            position["bars_held"] += 1

        # ── Entry signal (use previous bar's regime/htf — no look-ahead) ─
        if position is None:
            regime = prev_bar.regime
            htf    = prev_bar.htf
            atr    = prev_bar.atr

            if pd.isna(atr) or atr <= 0 or htf in (None, "NEUTRAL") or pd.isna(htf):
                prev_bar = bar
                continue

            entry_price = bar.open  # enter at next bar open after signal

            if regime == "UP" and htf == "BULL":
                init_stop = entry_price - ATR_MULTIPLIER * atr
                if entry_price - init_stop <= 0:
                    prev_bar = bar
                    continue
                position = {
                    "direction":   "long",
                    "entry_price": entry_price,
                    "entry_time":  ts,
                    "trail_stop":  init_stop,
                    "bars_held":   0,
                }

            elif regime == "DOWN" and htf == "BEAR":
                init_stop = entry_price + ATR_MULTIPLIER * atr
                if init_stop - entry_price <= 0:
                    prev_bar = bar
                    continue
                position = {
                    "direction":   "short",
                    "entry_price": entry_price,
                    "entry_time":  ts,
                    "trail_stop":  init_stop,
                    "bars_held":   0,
                }

        prev_bar = bar

    # Force-close any open position at end of data
    if position is not None and prev_bar is not None:
        ep = position["entry_price"]
        xp = prev_bar.close  # close at last available bar
        direction = position["direction"]
        comm = COMMISSION_PCT * (ep + xp) * BTC_SIZE
        pnl = ((xp - ep) if direction == "long" else (ep - xp)) * BTC_SIZE - comm
        trades.append({
            "entry_time":  position["entry_time"],
            "exit_time":   prev_bar.Index,
            "direction":   direction,
            "entry_price": ep,
            "exit_price":  xp,
            "pnl":         pnl,
            "bars_held":   position["bars_held"],
            "split":       "train" if position["entry_time"] < split_ts else "holdout",
        })

    return pd.DataFrame(trades)


# ── Metrics ──────────────────────────────────────────────────────────────────

def metrics(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"\n{label}: no trades")
        return

    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    wr  = len(wins) / len(df) * 100
    gp  = wins["pnl"].sum() if len(wins) else 0.0
    gl  = abs(losses["pnl"].sum()) if len(losses) else 1e-9
    pf  = gp / gl
    avg_w = wins["pnl"].mean() if len(wins) else 0.0
    avg_l = losses["pnl"].mean() if len(losses) else 0.0

    # Annualised Sharpe on trade-level P&L (conservative vs daily)
    if df["pnl"].std() > 0:
        # Estimate trades/year from date range
        days = (pd.to_datetime(df["exit_time"]).max() - pd.to_datetime(df["exit_time"]).min()).days
        trades_per_day = len(df) / max(days, 1)
        ann_factor = np.sqrt(252 * trades_per_day)
        sharpe = (df["pnl"].mean() / df["pnl"].std()) * ann_factor
    else:
        sharpe = 0.0

    cum   = df["pnl"].cumsum()
    mdd   = (cum - cum.cummax()).min()
    total = df["pnl"].sum()

    avg_hold_bars = df["bars_held"].mean()
    avg_hold_hr   = avg_hold_bars * 15 / 60

    longs  = len(df[df["direction"] == "long"])
    shorts = len(df[df["direction"] == "short"])

    print(f"\n{'='*56}")
    print(f"  {label}")
    print(f"{'='*56}")
    print(f"  Trades:         {len(df):>6}    (Long {longs} / Short {shorts})")
    print(f"  Win Rate:       {wr:>6.1f}%")
    print(f"  Profit Factor:  {pf:>8.3f}")
    print(f"  Sharpe (ann):   {sharpe:>8.2f}")
    print(f"  Total P&L:      ${total:>9.2f}")
    print(f"  Avg Win:        ${avg_w:>9.2f}")
    print(f"  Avg Loss:       ${avg_l:>9.2f}")
    print(f"  Max Drawdown:   ${mdd:>9.2f}")
    print(f"  Avg Hold:       {avg_hold_hr:>6.1f} hr  ({avg_hold_bars:.0f} bars × 15m)")
    print(f"{'='*56}")

    # Monthly P&L breakdown
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["exit_time"]).dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].agg(["sum", "count"])
    print(f"\n  Monthly P&L:")
    for month, row in monthly.iterrows():
        bar_str = "▓" * int(abs(row["sum"]) / 5) if abs(row["sum"]) > 0 else ""
        sign = "+" if row["sum"] >= 0 else ""
        print(f"    {month}  {sign}${row['sum']:7.2f}  ({int(row['count'])} trades) {bar_str}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading 1-min BTC data from Kraken...")
    df_1min = load_1min()
    logger.info(f"  {df_1min.index[0].date()} → {df_1min.index[-1].date()}  ({len(df_1min):,} bars)")

    logger.info("Resampling to 15-min and daily...")
    df_15m   = resample(df_1min, "15min")
    df_daily = resample(df_1min, "1D")
    logger.info(f"  15m: {len(df_15m):,} bars | Daily: {len(df_daily):,} bars")
    logger.info(f"  HTF EMA({HTF_FAST_MA}/{HTF_SLOW_MA}) warms up ~{HTF_SLOW_MA} days from data start")

    logger.info("Running adaptive regime backtest...")
    trades = run_backtest(df_15m, df_daily)

    if trades.empty:
        logger.warning("No trades generated — try relaxing regime or HTF parameters")
        return

    logger.info(f"Total trades generated: {len(trades)}")

    train   = trades[trades["split"] == "train"]
    holdout = trades[trades["split"] == "holdout"]

    metrics(train,   f"TRAIN     (< {SPLIT_DATE})")
    metrics(holdout, f"HOLDOUT   (>= {SPLIT_DATE})")
    metrics(trades,  "COMBINED  (all trades)")

    # Save trade log
    out_path = Path("data/reports")
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_path / f"backtest_adaptive_regime_btc_{ts}.csv"
    trades.to_csv(csv_path, index=False)
    logger.info(f"\nTrade log saved → {csv_path}")

    print(f"\n  Parameters used:")
    print(f"    Slope window:   {SLOPE_WINDOW} bars × 15m = {SLOPE_WINDOW*15/60:.1f} hr")
    print(f"    Percentile:     UP > {PCT_UP}th | DOWN < {PCT_DOWN}th  (history: {SLOPE_HISTORY} bars)")
    print(f"    ATR:            period={ATR_PERIOD}, multiplier={ATR_MULTIPLIER}×")
    print(f"    HTF filter:     daily EMA({HTF_FAST_MA}) vs EMA({HTF_SLOW_MA})")
    print(f"    Position:       {BTC_SIZE} BTC | Commission: {COMMISSION_PCT*100:.3f}%/side")


if __name__ == "__main__":
    main()
