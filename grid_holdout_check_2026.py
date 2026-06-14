#!/usr/bin/env python3
"""2026 holdout check for the top grid candidates.

For each candidate (SL, TP, threshold): train the meta-model on ALL of 2025 at
those exits, freeze it, regenerate 2026 signals/labels at the same exits, and score
forward — mirroring real deployment (train on history, freeze, run forward).

2026 was untouched by the grid search, so this is a clean out-of-sample check.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_tier2_wf_adaptive import profit_factor, win_rate, max_drawdown, per_trade_sharpe, make_model
from grid_search_sl_tp_ml import FEATURE_COLS, regen_features  # reuse 2025 cache + regen

PYTHON = ".venv/bin/python"
BACKTEST = "src/research/backtest_zero_bias_optimized.py"
DATA_2026 = "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
CACHE = Path(".grid_cache")

# (SL, TP, threshold) — top robust candidates + live baseline
CANDIDATES = [
    (5.0, 8.0, 0.50),
    (5.0, 8.0, 0.45),
    (4.0, 6.0, 0.50),
    (5.0, 5.0, 0.50),
    (2.0, 8.0, 0.50),   # ~current live config
]


def regen_2026(sl: float, tp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp = CACHE / f"sl{sl}_tp{tp}_2026_features.csv"
    hp = CACHE / f"sl{sl}_tp{tp}_2026_history.csv"
    if not (fp.exists() and hp.exists()):
        cmd = [PYTHON, BACKTEST, "--export", "--export-path", str(fp), "--history", str(hp),
               "--data", DATA_2026, "--sl-mult", str(sl), "--tp-mult", str(tp),
               "--atr-threshold", "0.1", "--session-windows", "baseline",
               "--start", "2026-01-01", "--end", "2026-12-31"]
        subprocess.run(cmd, capture_output=True, text=True)
    return pd.read_csv(fp), pd.read_csv(hp)


def metrics(pnl):
    return dict(n=len(pnl), wr=win_rate(pnl), pf=profit_factor(pnl),
                pnl=float(pnl.sum()), ir=per_trade_sharpe(pnl), dd=max_drawdown(pnl))


def main():
    rows = []
    seen_sltp = set()
    for sl, tp, thr in CANDIDATES:
        f25, h25 = regen_features(sl, tp, CACHE)          # 2025 train (cached from grid)
        f26, h26 = regen_2026(sl, tp)                     # 2026 holdout
        if f25.empty or f26.empty:
            print(f"skip SL{sl}/TP{tp} — regen failed"); continue

        model = make_model()
        model.fit(f25[FEATURE_COLS].values, f25["label"].values)
        proba26 = model.predict_proba(f26[FEATURE_COLS].values)[:, 1]
        pnl26 = h26["pnl"].values

        keep = proba26 >= thr
        m_ml = metrics(pnl26[keep])
        m_no = metrics(pnl26)                              # no-ML (SL/TP change only)
        rows.append((sl, tp, thr, m_no, m_ml))
        if (sl, tp) not in seen_sltp:
            print(f"SL{sl}/TP{tp}: 2026 signals={len(f26)} | trained on 2025 n={len(f25)}")
            seen_sltp.add((sl, tp))

    print("\n2026 HOLDOUT CHECK — train on 2025, freeze, score 2026 (bearish-only)")
    print("=" * 92)
    print(f"{'SL':>4} {'TP':>4} {'thr':>5} | {'noML n':>6} {'noML PF':>8} {'noML $':>8} | "
          f"{'ML n':>5} {'ML WR':>6} {'ML PF':>7} {'ML $':>8} {'ML DD':>8}")
    print("-" * 92)
    for sl, tp, thr, mn, mm in rows:
        tag = "  <- live" if (sl, tp) == (2.0, 8.0) else ""
        print(f"{sl:>4.1f} {tp:>4.1f} {thr:>5.2f} | {mn['n']:>6} {mn['pf']:>8.3f} {mn['pnl']:>8,.0f} | "
              f"{mm['n']:>5} {mm['wr']:>6.1%} {mm['pf']:>7.3f} {mm['pnl']:>8,.0f} {mm['dd']:>8,.0f}{tag}")

    out = Path("data/reports/grid_holdout_2026_check.txt")
    with out.open("w") as fh:
        fh.write("2026 holdout check (train 2025 / freeze / score 2026, bearish-only)\n")
        for sl, tp, thr, mn, mm in rows:
            fh.write(f"SL{sl}/TP{tp}/thr{thr}: noML PF={mn['pf']:.3f} ${mn['pnl']:,.0f} (n={mn['n']}) | "
                     f"ML PF={mm['pf']:.3f} ${mm['pnl']:,.0f} WR={mm['wr']:.1%} n={mm['n']} DD=${mm['dd']:,.0f}\n")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
