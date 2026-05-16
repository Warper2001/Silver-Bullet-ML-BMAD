#!/usr/bin/env python3
"""Compute LR channel + HMM regime labels at each MNQ tier2 signal timestamp.

Reads:
  data/ml_training/doe_run_08_fullyear_history.csv  (signal timestamps + direction)
  data/processed/dollar_bars/1_minute/mnq_1min_2025.csv  (OHLCV bars)
  models/hmm/regime_model_1min/  (pre-trained HMM)

Writes:
  data/ml_training/doe_run_08_regime_enriched.csv
    columns: timestamp, direction, lr_regime, hmm_regime

LR channel windows on 1-min MNQ bars:
  fast_len=50  →  50 minutes  (intraday momentum)
  slow_len=200 →  200 minutes / ~3.3 hours  (session-level trend)

HMM regime integers (from metadata.json):
  0 → trending_down (bearish)
  1 → trending_up   (bullish)
  2 → trending_down (bearish)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BARS_PATH    = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
HISTORY_PATH = Path("data/ml_training/doe_run_08_fullyear_history.csv")
HMM_MODEL    = Path("models/hmm/regime_model_1min")
OUT_PATH     = Path("data/ml_training/doe_run_08_regime_enriched.csv")

LR_FAST = 390   # 1 regular MNQ session (6.5 h × 60 min) — intraday trend
LR_SLOW = 1950  # 5 sessions (1 trading week) — multi-day regime
HMM_WARMUP = 25   # minimum bars for HMM feature rolling windows


def main() -> None:
    # ── Load bar data ─────────────────────────────────────────────────────────
    if not BARS_PATH.exists():
        sys.exit(f"Error: {BARS_PATH} not found")
    print(f"Loading bar data from {BARS_PATH} …")
    bars = pd.read_csv(BARS_PATH, parse_dates=["timestamp"])
    # Strip timezone info so timestamps are naive UTC — consistent with history CSV
    if bars["timestamp"].dt.tz is not None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize(None)
    bars.sort_values("timestamp", inplace=True)
    bars.reset_index(drop=True, inplace=True)
    print(f"  {len(bars):,} bars | {bars['timestamp'].iloc[0]} → {bars['timestamp'].iloc[-1]}")

    # ── Load signal history ───────────────────────────────────────────────────
    if not HISTORY_PATH.exists():
        sys.exit(f"Error: {HISTORY_PATH} not found")
    hist = pd.read_csv(HISTORY_PATH)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    if hist["timestamp"].dt.tz is not None:
        hist["timestamp"] = hist["timestamp"].dt.tz_localize(None)
    print(f"  {len(hist)} signals | {hist['timestamp'].min()} → {hist['timestamp'].max()}")

    # ── Find bar index at-or-before each signal timestamp ────────────────────
    bar_times_ns = bars["timestamp"].values.astype("int64")
    sig_times_ns  = hist["timestamp"].values.astype("int64")
    signal_bar_idx = np.searchsorted(bar_times_ns, sig_times_ns, side="right") - 1

    missing = (signal_bar_idx < 0).sum()
    if missing:
        print(f"  [warn] {missing} signals precede all bars — will be marked SIDEWAYS/neutral")

    # ── LR channel regime ─────────────────────────────────────────────────────
    print(f"\nComputing LR channel regime (fast={LR_FAST}, slow={LR_SLOW}) over {len(bars):,} bars …")
    from src.ml.regime_detection.lr_channel_detector import LRChannelRegimeDetector
    lr_detector = LRChannelRegimeDetector(fast_len=LR_FAST, slow_len=LR_SLOW)
    all_lr = lr_detector.fit_predict(bars["close"].values)  # array of "UP"/"DOWN"/"SIDEWAYS"

    lr_at_signal = []
    for idx in signal_bar_idx:
        if idx < 0 or idx < LR_SLOW:
            lr_at_signal.append("SIDEWAYS")   # insufficient warm-up
        else:
            lr_at_signal.append(all_lr[idx])

    lr_counts = pd.Series(lr_at_signal).value_counts()
    print(f"  LR regime distribution at signals: {lr_counts.to_dict()}")

    # ── HMM regime ────────────────────────────────────────────────────────────
    hmm_at_signal: list[int] = []

    if not HMM_MODEL.exists():
        print(f"\n[warn] HMM model not found at {HMM_MODEL} — filling hmm_regime with -1 (neutral)")
        hmm_at_signal = [-1] * len(hist)
    else:
        print(f"\nLoading HMM model from {HMM_MODEL} …")
        try:
            from src.ml.regime_detection.hmm_detector import HMMRegimeDetector
            from src.ml.regime_detection.features import HMMFeatureEngineer

            hmm_detector = HMMRegimeDetector.load(HMM_MODEL)
            feat_eng = HMMFeatureEngineer()

            print(f"  Engineering HMM features for {len(bars):,} bars …")
            hmm_features = feat_eng.engineer_features(bars)   # index-aligned with bars
            hmm_preds_all = hmm_detector.predict(hmm_features)  # int array, same length

            for idx in signal_bar_idx:
                if idx < HMM_WARMUP or idx < 0:
                    hmm_at_signal.append(-1)   # warm-up → neutral
                else:
                    hmm_at_signal.append(int(hmm_preds_all[idx]))

            hmm_counts = pd.Series(hmm_at_signal).value_counts().sort_index()
            print(f"  HMM regime distribution at signals (0=down,1=up,2=down,-1=neutral): {hmm_counts.to_dict()}")

        except Exception as exc:
            print(f"  [warn] HMM prediction failed: {exc}")
            print(f"  Falling back to -1 (neutral) for all signals")
            hmm_at_signal = [-1] * len(hist)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "timestamp":  hist["timestamp"].values,
        "direction":  hist["direction"].values,
        "lr_regime":  lr_at_signal,
        "hmm_regime": hmm_at_signal,
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} ({len(out)} rows)")

    # Summary
    print("\nRegime agreement with signal direction:")
    for col, labels in [
        ("lr_regime",  {"UP": "bullish", "DOWN": "bearish", "SIDEWAYS": None}),
        ("hmm_regime", {1: "bullish", 0: "bearish", 2: "bearish", -1: None}),
    ]:
        agrees = 0
        disagrees = 0
        neutral = 0
        for _, row in out.iterrows():
            r = row[col]
            d = row["direction"]
            if labels.get(r) is None:
                neutral += 1
            elif labels.get(r) == d:
                agrees += 1
            else:
                disagrees += 1
        total = len(out)
        print(
            f"  {col}: agree={agrees} ({agrees/total:.1%}), "
            f"disagree={disagrees} ({disagrees/total:.1%}), "
            f"neutral={neutral} ({neutral/total:.1%})"
        )


if __name__ == "__main__":
    main()
