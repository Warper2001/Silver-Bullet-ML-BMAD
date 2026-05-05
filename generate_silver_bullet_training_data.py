#!/usr/bin/env python3
"""Generate ML training data from corrected Silver Bullet backtest trades.

Reads the latest corrected backtest CSV (273 trades) + 1-minute bar data,
computes 8 features at each trade's fill bar, and writes:
  data/ml_training/silver_bullet_corrected_features.csv
  data/ml_training/silver_bullet_corrected_history.csv

Feature redefinitions vs. old DOE runs (column names preserved for model compat):
  fvg_fill_pct       → gap_size / atr        (gap quality in ATR units)
  sweep_window_vol   → log1p(bar_volume)      (log-scaled fill-bar volume)
  h1_trend_slope     → h1_slope * dir_sign    (direction-aligned trend strength)
  session_volume_ratio → log1p(raw_ratio)     (log-scaled to remove outlier skew)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK_ATR = 14
LOOKBACK_VOL = 20
LOOKBACK_H1  = 60


def load_bars_df(json_path: str = "mnq_historical.json") -> pd.DataFrame:
    logger.info(f"Loading bars from {json_path} ...")
    with open(json_path) as f:
        raw = json.load(f)
    rows = []
    for b in raw:
        ts = b.get("TimeStamp") or b.get("timestamp") or b.get("t")
        rows.append({
            "timestamp": ts,
            "open":   float(b.get("Open",  b.get("open",  0))),
            "high":   float(b.get("High",  b.get("high",  0))),
            "low":    float(b.get("Low",   b.get("low",   0))),
            "close":  float(b.get("Close", b.get("close", 0))),
            "volume": float(b.get("TotalVolume", b.get("Volume", b.get("volume", 0)))),
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.DateOffset(months=6)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    logger.info(f"  {len(df):,} bars  [{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}]")
    return df


def precompute(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised computation of all rolling features."""
    logger.info("Precomputing rolling features ...")
    et = df["timestamp"].dt.tz_convert("US/Eastern")
    df["et_hour"]     = et.dt.hour
    df["day_of_week"] = et.dt.dayofweek
    df["et_date"]     = et.dt.date

    # ATR(14) — proper True Range
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(LOOKBACK_ATR, min_periods=1).mean()

    # Volume ratio (bar vol / 20-bar rolling mean)
    vol_ma = df["volume"].rolling(LOOKBACK_VOL, min_periods=1).mean()
    df["volume_ratio"] = (df["volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)

    # H1 raw slope: (close - close_60_bars_ago) / 60
    df["_h1_raw"] = (df["close"] - df["close"].shift(LOOKBACK_H1).bfill()) / LOOKBACK_H1

    # Session features
    grp = df.groupby("et_date")
    df["session_open"] = grp["open"].transform("first")
    df["session_high"] = grp["high"].transform("cummax")
    df["session_low"]  = grp["low"].transform("cummin")

    # session_displacement: (close - session_open) / atr
    df["session_displacement"] = (
        (df["close"] - df["session_open"]) / df["atr"].replace(0, np.nan)
    ).fillna(0.0)

    # Raw session volume ratio (used for log transform)
    df["_svr_raw"] = (df["volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)

    logger.info("  Done.")
    return df


def compute_feature_row(df: pd.DataFrame, bar_i: int, direction: str,
                        gap_size: float, pnl: float) -> dict | None:
    if bar_i < LOOKBACK_H1 or bar_i >= len(df):
        return None

    row = df.iloc[bar_i]
    atr_val      = max(float(row["atr"]), 0.01)
    vol_ratio    = float(row["volume_ratio"])
    raw_slope    = float(row["_h1_raw"])
    session_disp = float(row["session_displacement"])
    svr_raw      = float(row["_svr_raw"])
    volume       = float(row["volume"])

    dir_sign = 1 if direction == "bullish" else -1

    # ── 8 model features (column names match FEATURE_COLS) ────────────────
    fvg_fill_pct        = gap_size / atr_val                  # gap quality
    sweep_window_vol    = float(np.log1p(volume))             # log bar volume
    volume_ratio        = vol_ratio                           # best predictor
    signal_direction    = 1 if (raw_slope * dir_sign) > 0 else 0  # slope alignment
    h1_trend_slope      = raw_slope * dir_sign                # direction-signed slope
    atr                 = atr_val
    session_displacement = session_disp
    session_volume_ratio = float(np.log1p(svr_raw))          # log-scaled

    return {
        "fvg_fill_pct":          fvg_fill_pct,
        "sweep_window_vol":      sweep_window_vol,
        "volume_ratio":          volume_ratio,
        "signal_direction":      signal_direction,
        "h1_trend_slope":        h1_trend_slope,
        "atr":                   atr,
        "session_displacement":  session_displacement,
        "session_volume_ratio":  session_volume_ratio,
        "label": 1 if pnl > 0 else 0,
    }


def main():
    df = precompute(load_bars_df())

    # Build fast timestamp → integer index
    ts_index = {ts: i for i, ts in enumerate(df["timestamp"])}

    # Load latest corrected trades CSV
    trade_files = sorted(
        Path("data/reports").glob("backtest_full_silver_bullet_ml_6months_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not trade_files:
        logger.error("No backtest CSV found in data/reports/"); sys.exit(1)

    trades = pd.read_csv(trade_files[0], parse_dates=["entry_time"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    logger.info(f"Loaded {len(trades)} trades from {trade_files[0].name}")

    feature_rows, history_rows = [], []

    for _, t in trades.iterrows():
        entry_ts  = t["entry_time"]
        direction = t["direction"]
        pnl       = float(t["pnl"])
        stop      = float(t["stop_loss"])
        entry_p   = float(t["entry_price"])
        gap_size  = max(abs(entry_p - stop), 0.25)

        bar_i = ts_index.get(entry_ts)
        if bar_i is None:
            prior = df[df["timestamp"] <= entry_ts]
            if prior.empty: continue
            bar_i = int(prior.index[-1])

        feat = compute_feature_row(df, bar_i, direction, gap_size, pnl)
        if feat is None:
            continue

        feature_rows.append(feat)
        exit_type = str(t.get("exit_reason", "time")).replace("stop_loss", "sl").replace("target", "tp")
        history_rows.append({
            "pnl": pnl, "win": feat["label"],
            "bars_held": int(t.get("bars_held", 0)),
            "exit_type": exit_type,
            "timestamp": entry_ts.isoformat(),
            "direction": direction,
        })

    feat_df = pd.DataFrame(feature_rows)
    hist_df = pd.DataFrame(history_rows)

    pos = (feat_df["label"] == 1).sum()
    logger.info(f"{len(feat_df)} samples — {pos} winners ({pos/len(feat_df)*100:.1f}%), "
                f"{len(feat_df)-pos} losers")

    # Quick per-feature AUC check
    from sklearn.metrics import roc_auc_score
    FEATURE_COLS = ['fvg_fill_pct','sweep_window_vol','volume_ratio','signal_direction',
                    'h1_trend_slope','atr','session_displacement','session_volume_ratio']
    logger.info("Per-feature AUC:")
    for col in FEATURE_COLS:
        v = feat_df[col]
        auc = roc_auc_score(feat_df["label"], v) if v.std() > 0 else 0.5
        logger.info(f"  {col:<25} AUC={auc:.4f}")

    out = Path("data/ml_training")
    out.mkdir(parents=True, exist_ok=True)
    feat_path = out / "silver_bullet_corrected_features.csv"
    hist_path = out / "silver_bullet_corrected_history.csv"
    feat_df.to_csv(feat_path, index=False)
    hist_df.to_csv(hist_path, index=False)
    logger.info(f"Saved → {feat_path}  ({len(feat_df)} rows)")
    logger.info(f"Saved → {hist_path}")


if __name__ == "__main__":
    main()
