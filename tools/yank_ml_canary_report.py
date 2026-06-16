#!/usr/bin/env python3
"""Weekly rollup for the YANK passive ML drift canary.

Reads logs/yank_ml_canary.csv (written by Tier2StreamingTrader._log_ml_canary at each
trade close: entry meta-model P(success) paired with the realized win/loss) and reports
discrimination (AUC) + calibration (Brier) with a bootstrap confidence band.

This is a DIAGNOSTIC for the weekly human review, not a control. Per the 2026-06-16
party-mode decision the actuator stays the outcome guardrail (disable ML if live
PF < 0.90 after N>=20); the meta-model's baseline AUC is ~0.50, so read every number
against that null — a band straddling 0.50 means "no detectable discrimination," which
is the expected state, not an alarm. The signal worth a human look is DIVERGENCE: proba
quality looks fine while PF craters, or vice-versa.

Usage:
    .venv/bin/python tools/yank_ml_canary_report.py [--window N] [--csv PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "logs" / "yank_ml_canary.csv"


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC; NaN if only one class present (undefined)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc_bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for AUC. Wide by design on small N — that's the point."""
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if len(np.unique(ys)) < 2:
            continue
        stats.append(_auc(ys, ps))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="canary CSV path")
    ap.add_argument("--window", type=int, default=30,
                    help="rolling window size in trades for the trailing AUC/Brier")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"No canary log yet at {args.csv} — nothing to report.\n"
              f"(Written once an ML-filtered YANK trade closes with the model active.)")
        return 0

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["ml_proba", "win"])
    n = len(df)
    if n == 0:
        print(f"{args.csv} has no scored rows yet.")
        return 0

    y = df["win"].astype(int).to_numpy()
    p = df["ml_proba"].astype(float).to_numpy()

    auc = _auc(y, p)
    lo, hi = _auc_bootstrap_ci(y, p)
    brier = _brier(y, p)
    base_rate = float(y.mean())
    brier_base = _brier(y, np.full_like(p, base_rate))  # always-predict-base-rate reference

    print("=" * 66)
    print(f"YANK ML drift canary — {n} closed ML-filtered trades")
    print(f"source: {args.csv}")
    print("=" * 66)
    print(f"Win rate (realized)     : {base_rate:6.3f}  ({int(y.sum())}/{n})")
    print(f"Mean predicted P(succ)  : {p.mean():6.3f}   <- vs win rate = calibration eyeball")
    print(f"AUC (discrimination)    : {auc:6.3f}   95% CI [{lo:.3f}, {hi:.3f}]   null=0.500")
    if not np.isnan(lo) and lo <= 0.5 <= hi:
        print("                          -> CI straddles 0.50: NO detectable discrimination (expected).")
    elif not np.isnan(hi) and hi < 0.5:
        print("                          -> CI fully BELOW 0.50: model is anti-discriminating; flag for review.")
    elif not np.isnan(lo) and lo > 0.5:
        print("                          -> CI fully ABOVE 0.50: genuine discrimination (notable — verify).")
    print(f"Brier score             : {brier:6.3f}   (base-rate ref {brier_base:.3f}; lower=better)")

    if n >= args.window:
        yr = y[-args.window:]
        pr = p[-args.window:]
        print("-" * 66)
        print(f"Trailing {args.window} trades:")
        print(f"  win rate {yr.mean():.3f} | AUC {_auc(yr, pr):.3f} | Brier {_brier(yr, pr):.3f}")
    else:
        print("-" * 66)
        print(f"(Only {n} trades; need {args.window} for the trailing window. "
              f"AUC on <~30 binary outcomes is near-useless — wait for N.)")

    print("=" * 66)
    print("Reminder: this is observe-only. Actuator = PF<0.90-after-N>=20 guardrail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
