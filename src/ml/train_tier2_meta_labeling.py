#!/usr/bin/env python3
"""Train Tier 2 meta-labeling model: Pipeline(StandardScaler + LogisticRegression).

At 300–1,200 samples with 8 features, LR generalizes better than XGBoost due to
far fewer free parameters (~9 vs thousands) and analytically interpretable L2
regularization. No separate Platt calibration fold needed — LR predict_proba() is
already well-calibrated.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'fvg_fill_pct', 'sweep_window_vol', 'volume_ratio', 'signal_direction',
    'h1_trend_slope', 'atr', 'session_displacement', 'session_volume_ratio',
]

AUC_GATE = 0.52
PF_GATE = 1.15
ECE_GATE = 0.08


def train_model(csv_path: Path, history_path: Path) -> None:
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
    if not history_path.exists():
        print(f"Error: {history_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_history = pd.read_csv(history_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    assert len(df) == len(df_history), (
        f"CSV row count mismatch: {csv_path} has {len(df)} rows, "
        f"{history_path} has {len(df_history)} rows. Re-export both with --export --history."
    )

    X = df[FEATURE_COLS]
    y = df["label"]

    print(f"Feature columns: {list(X.columns)}")
    print(f"Label distribution: {y.value_counts(normalize=True).to_dict()}")

    # 80/20 temporal split — no calibration fold needed (LR is already well-calibrated)
    n = len(df)
    train_end = int(n * 0.80)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:], y.iloc[train_end:]
    y_pnl_oos = df_history['pnl'].iloc[train_end:].values

    # Hyperparameter selection via time-series cross-validation on training fold
    best_c, best_auc_cv = 1.0, 0.0
    print("\nC selection via TimeSeriesSplit CV:")
    for c in [0.01, 0.1, 1.0, 10.0]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=c, solver='lbfgs', max_iter=1000,
                class_weight='balanced', random_state=42,
            ))
        ])
        cv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
        auc_cv = scores.mean()
        print(f"  C={c:6.2f}: CV AUC={auc_cv:.4f} (std={scores.std():.4f})")
        if auc_cv > best_auc_cv:
            best_auc_cv, best_c = auc_cv, c

    if best_auc_cv <= 0.0:
        print(f"  WARNING: All CV AUC scores <= 0.0 — signal may be inverted in CV folds. "
              f"Using C={best_c} as least-bad fallback; verify on OOS fold.")
    print(f"\nSelected C={best_c} (CV AUC={best_auc_cv:.4f})")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=best_c, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=42,
        ))
    ])
    model.fit(X_train, y_train)

    # OOS evaluation
    y_proba_oos = model.predict_proba(X_val)[:, 1]

    if len(np.unique(y_val)) < 2:
        print("Warning: OOS set has only one class. AUC calculation skipped.")
        auc = 0.5
    else:
        auc = roc_auc_score(y_val, y_proba_oos)

    print(f"\nOOS AUC: {auc:.4f}")

    # ECE diagnostic (calibration quality check)
    # n_bins=5 is appropriate for OOS folds < 300 samples: gives ~n/5 samples per bin
    # vs n_bins=10 which gives ~n/10 per bin (too noisy below 300 samples).
    ece_bins = 'N/A'
    if len(np.unique(y_val)) >= 2:
        n_oos = len(y_val)
        # n_bins=5 for n_oos < 300: gives ~n/5 samples/bin (statistically sound)
        # n_bins=10 gives ~n/10 per bin — too noisy at small OOS sizes.
        ece_bins = 5 if n_oos < 300 else 10
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_val, y_proba_oos, n_bins=ece_bins
        )
        ece = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))
    else:
        ece = 0.0  # can't compute with one class
    print(f"OOS ECE: {ece:.4f} (n_bins={ece_bins})")

    # PF Threshold Sweep on OOS fold
    print("\nPF Threshold Sweep (OOS Fold):")
    thresholds = np.arange(0.40, 0.81, 0.01)
    best_pf, best_thresh = 0.0, 0.55
    for t in thresholds:
        mask = y_proba_oos >= t
        if mask.sum() < 10:
            continue
        gross_win = y_pnl_oos[mask & (y_pnl_oos > 0)].sum()
        gross_loss = abs(y_pnl_oos[mask & (y_pnl_oos < 0)].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 0.0
        print(f"  Threshold {t:.2f}: PF={pf:.3f} (Trades={mask.sum()})")
        if pf > best_pf:
            best_pf, best_thresh = pf, t

    print(f"\nPF-optimal threshold: {best_thresh:.2f} (PF={best_pf:.3f})")

    # Model gates
    if auc <= AUC_GATE:
        print(f"\nAUC {auc:.4f} not > {AUC_GATE}. Model NOT saved.")
        return

    if best_pf == 0.0:
        print(f"\nNo threshold produced ≥10 trades on OOS fold. Model NOT saved.")
        return

    if best_pf < PF_GATE:
        print(f"\nBest PF {best_pf:.3f} not > {PF_GATE}. Model NOT saved.")
        return

    if ece >= ECE_GATE:
        print(f"\nECE {ece:.4f} >= {ECE_GATE} — probabilities poorly calibrated. Model NOT saved.")
        return

    model_dir = Path("models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tier2_meta_labeling_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Flag if threshold changed significantly from previous 0.55
    if abs(best_thresh - 0.55) > 0.10:
        print(f"⚠️  WARNING: PF-optimal threshold {best_thresh:.2f} differs from previous 0.55 by "
              f"{abs(best_thresh - 0.55):.2f} (> 0.10). Flag to Alex before updating ML_THRESHOLD.")
    else:
        print(f"IMPORTANT: Set ML_THRESHOLD in live trader to {best_thresh:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str,
                        default="data/ml_training/doe_run_08_fullyear_features.csv",
                        help="Path to ML features CSV (DOE survivor full-year dataset)")
    parser.add_argument("--history-path", type=str,
                        default="data/ml_training/doe_run_08_fullyear_history.csv",
                        help="Path to trade history CSV (parallel to features CSV)")
    args = parser.parse_args()
    train_model(Path(args.csv_path), Path(args.history_path))
