#!/usr/bin/env python3
"""2026 evaluation: frozen-2025 model vs proper daily walk-forward (retrain through 2026).

Tests the hypothesis that scoring all of 2026 with a model frozen at end-2025 unfairly
handicaps daily ML training. The fair comparison retrains each trading day on ALL prior
data (2025 + 2026-to-date), so only the current day is out-of-sample.

For each config (SL, TP) it reports 2026 metrics three ways:
  - no-ML            : take every signal (SL/TP change only)
  - frozen-2025 @thr : train all 2025, freeze, score 2026   (the old, harsh check)
  - daily-WF   @thr  : expanding daily walk-forward through 2026 (only today is OOS)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from backtest_tier2_wf_adaptive import profit_factor, win_rate, max_drawdown, per_trade_sharpe, make_model
from grid_search_sl_tp_ml import FEATURE_COLS
from grid_holdout_check_2026 import regen_2026
from grid_search_sl_tp_ml import regen_features

CACHE = Path(".grid_cache")

CONFIGS = [  # (SL, TP, threshold)
    (2.0, 8.0, 0.50),   # current live
    (5.0, 8.0, 0.50),   # grid top
    (4.0, 6.0, 0.50),
    (5.0, 5.0, 0.50),
]


def load_combined(sl, tp):
    f25, h25 = regen_features(sl, tp, CACHE)
    f26, h26 = regen_2026(sl, tp)
    def pack(f, h):
        d = f[FEATURE_COLS + ["label"]].copy()
        d["pnl"] = h["pnl"].values
        d["timestamp"] = pd.to_datetime(h["timestamp"].values)
        return d
    df = pd.concat([pack(f25, h25), pack(f26, h26)], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["year"] = df["timestamp"].dt.year
    return df


def m(pnl):
    return dict(n=len(pnl), wr=win_rate(pnl), pf=profit_factor(pnl),
                pnl=float(pnl.sum()), dd=max_drawdown(pnl))


def frozen_2025(df, thr):
    tr = (df["year"] == 2025).values
    te = (df["year"] == 2026).values
    model = make_model(); model.fit(df.loc[tr, FEATURE_COLS].values, df.loc[tr, "label"].values)
    p = model.predict_proba(df.loc[te, FEATURE_COLS].values)[:, 1]
    pnl = df.loc[te, "pnl"].values
    return m(pnl[p >= thr]), m(pnl)  # (ML filtered, no-ML)


def daily_walkforward_2026(df, thr):
    """Expanding daily WF through 2026: each 2026 date trains on all strictly-prior signals."""
    X = df[FEATURE_COLS].values
    y = df["label"].values
    test_dates = sorted(df.loc[df["year"] == 2026, "date"].unique())
    probs, pnls = [], []
    for D in test_dates:
        tr = (df["timestamp"] < pd.Timestamp(D)).values     # all prior (2025 + 2026<D)
        te = (df["date"] == D).values
        if tr.sum() < 50 or len(set(y[tr])) < 2:
            continue
        model = make_model(); model.fit(X[tr], y[tr])
        probs.append(model.predict_proba(X[te])[:, 1])
        pnls.append(df.loc[te, "pnl"].values)
    if not probs:
        return m(np.array([])), 0
    p = np.concatenate(probs); pnl = np.concatenate(pnls)
    return m(pnl[p >= thr]), len(pnl)


def main():
    print("2026 EVALUATION — frozen-2025 vs daily walk-forward (bearish-only)")
    print("=" * 96)
    print(f"{'SL':>4} {'TP':>4} {'thr':>5} | {'noML PF':>8} {'noML$':>8} | "
          f"{'frozPF':>7} {'froz$':>8} {'frozN':>6} | {'wfPF':>7} {'wf$':>8} {'wfN':>5} {'wfDD':>8}")
    print("-" * 96)
    for sl, tp, thr in CONFIGS:
        df = load_combined(sl, tp)
        fz_ml, fz_no = frozen_2025(df, thr)
        wf_ml, wf_n = daily_walkforward_2026(df, thr)
        tag = "  <- live" if (sl, tp) == (2.0, 8.0) else ""
        print(f"{sl:>4.1f} {tp:>4.1f} {thr:>5.2f} | {fz_no['pf']:>8.3f} {fz_no['pnl']:>8,.0f} | "
              f"{fz_ml['pf']:>7.3f} {fz_ml['pnl']:>8,.0f} {fz_ml['n']:>6} | "
              f"{wf_ml['pf']:>7.3f} {wf_ml['pnl']:>8,.0f} {wf_ml['n']:>5} {wf_ml['dd']:>8,.0f}{tag}")

    out = Path("data/reports/holdout_2026_walkforward.txt")
    print(f"\n(report mirrors stdout; frozen vs daily-WF on the same 2026 signals)")
    out.write_text("see stdout\n")


if __name__ == "__main__":
    main()
