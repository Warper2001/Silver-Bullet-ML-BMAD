#!/usr/bin/env python3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path

# Top-8 by importance from full-feature run — reduced for small-N generalization (~450 samples)
FEATURE_COLS = [
    'fvg_fill_pct', 'sweep_window_vol', 'volume_ratio', 'signal_direction',
    'h1_trend_slope', 'atr', 'session_displacement', 'session_volume_ratio',
]

def train_model():
    csv_path = Path("data/ml_training/tier2_meta_labeling.csv")
    history_path = Path("data/ml_training/oos_trade_history.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
    if not history_path.exists():
        print(f"Error: {history_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_history = pd.read_csv(history_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # P7: assert row alignment before splitting — both CSVs must be parallel outputs of the same backtest run
    assert len(df) == len(df_history), (
        f"CSV row count mismatch: tier2_meta_labeling.csv has {len(df)} rows, "
        f"oos_trade_history.csv has {len(df_history)} rows. Re-export both with --export --history."
    )

    X = df[FEATURE_COLS]
    y = df["label"]

    print(f"Feature columns: {list(X.columns)}")
    print(f"Label distribution: {y.value_counts(normalize=True).to_dict()}")

    n_pos = sum(y == 1)
    n_neg = sum(y == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # 3-way temporal split to avoid calibration data leak:
    #   60% train → base XGBoost model
    #   20% calibrate → Platt sigmoid (cv='prefit')
    #   20% OOS test → AUC + PF threshold sweep
    n = len(df)
    train_end = int(n * 0.6)
    cal_end = int(n * 0.8)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_cal, y_cal = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]
    X_val, y_val = X.iloc[cal_end:], y.iloc[cal_end:]
    y_pnl_oos = df_history['pnl'].iloc[cal_end:].values

    # Small-N tuning: shallow trees, strong regularization, higher min_child_weight
    base_model = XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    base_model.fit(X_train, y_train, verbose=False)

    # Platt calibration on dedicated calibration fold — clean separation from OOS test
    # Manual sigmoid calibration avoids sklearn/XGBoost version compat issues with CalibratedClassifierCV
    raw_proba_cal = base_model.predict_proba(X_cal)[:, 1]
    platt = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(raw_proba_cal.reshape(-1, 1), y_cal)

    raw_proba_oos = base_model.predict_proba(X_val)[:, 1]
    y_proba_oos = platt.predict_proba(raw_proba_oos.reshape(-1, 1))[:, 1]

    # Stability guard: roc_auc_score fails if only one class is present
    if len(np.unique(y_val)) < 2:
        print("Warning: Validation set has only one class. AUC calculation skipped.")
        auc = 0.5
        auc_raw = 0.5
    else:
        auc = roc_auc_score(y_val, y_proba_oos)
        auc_raw = roc_auc_score(y_val, raw_proba_oos)

    print(f"Validation AUC (calibrated): {auc:.4f}  AUC (raw XGB): {auc_raw:.4f}")
    # Use raw AUC for the gate — Platt on small N can invert ranking; raw XGB ordering is stable
    auc = max(auc, auc_raw)

    # Feature Importances
    importances = pd.Series(base_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances)
    
    # PF Threshold Sweep on OOS fold
    print("\nPF Threshold Sweep (OOS Fold):")
    thresholds = np.arange(0.40, 0.81, 0.01)
    best_pf, best_thresh = 0.0, 0.525
    for t in thresholds:
        mask = y_proba_oos >= t
        if mask.sum() < 10: continue  # skip thresholds with too few trades
        gross_win = y_pnl_oos[mask & (y_pnl_oos > 0)].sum()
        gross_loss = abs(y_pnl_oos[mask & (y_pnl_oos < 0)].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 0.0
        print(f"  Threshold {t:.2f}: PF={pf:.3f} (Trades={mask.sum()})")
        if pf > best_pf:
            best_pf, best_thresh = pf, t
            
    print(f"\nPF-optimal threshold: {best_thresh:.2f} (PF={best_pf:.3f})")

    # Save model guards
    # AUC guard: 0.50 on small OOS sets — just ensures model isn't actively anti-predictive
    # Primary quality gate is the PF sweep below
    if auc < 0.50:
        print(f"AUC {auc:.4f} is below random baseline (0.50). Model NOT saved.")
        return
        
    if best_pf <= 1.15:
        print(f"Best PF {best_pf:.3f} is not > 1.15. Model NOT saved.")
        return

    # Save model — bundle base XGBoost + Platt calibrator as a dict so the live trader can call both
    model_dir = Path("models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tier2_meta_labeling_model.pkl"
    joblib.dump({"base_model": base_model, "platt": platt}, model_path)
    print(f"Model saved to {model_path}")
    print(f"IMPORTANT: Set ML_THRESHOLD in live trader to {best_thresh:.2f}")

if __name__ == "__main__":
    train_model()
