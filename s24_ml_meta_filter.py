"""
S24: ML meta-filter trainer — Phase 2 meta-labeling
Program C Phase 2 ML

Loads the S23 labeled trade dataset and trains a binary classifier
(XGBoost) to predict whether each H1·M15·M1·g0.25 signal will be a
winner (TP hit) or loser (SL / time-stop).

Evaluation: walk-forward time-series cross-validation (4 folds,
expanding window). Reports OOS PF with and without the ML filter at
multiple probability thresholds.

Saves the best model (trained on all in-sample data) to:
  models/s24_meta_filter/xgb_model.pkl
  models/s24_meta_filter/threshold.json
  models/s24_meta_filter/feature_names.json

If OOS PF (ML-filtered) > OOS PF (unfiltered) with N_filtered >= 15:
  Prints recommendation to pre-register S25 with the selected threshold.

NO holdout access. This is in-sample model selection only.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
except ImportError as e:
    print(f"ERROR: missing library — {e}", file=sys.stderr)
    sys.exit(1)

INPUT_PATH  = Path("data/ml_training/s23_meta_labels_2025.csv")
MODEL_DIR   = Path("models/s24_meta_filter")

FEATURE_COLS = [
    "gap_atr_ratio",
    "gap_dollars",
    "gap_m1_atr_ratio",
    "h1_atr",
    "m1_atr",
    "h1_vol_pct",
    "hour_et",
    "dow_et",
    "m1_bars_since_sweep",
    "m1_bars_since_choch",
]

THRESHOLDS      = [0.45, 0.50, 0.55, 0.60, 0.65]
N_FOLDS         = 4
MIN_TRAIN_SIZE  = 30    # minimum trades needed to fit a model
MIN_FILTERED_N  = 10    # minimum OOS trades kept for PF to be meaningful


# ── Utilities ─────────────────────────────────────────────────────────────────

def profit_factor(labels: np.ndarray, pnl: np.ndarray) -> float:
    gp = pnl[labels == 1].sum() if (labels == 1).any() else 0.0
    gl = abs(pnl[labels == 0].sum()) if (labels == 0).any() else 0.0
    return float(gp / gl) if gl > 0 else float("inf")


def pf_from_mask(mask: np.ndarray, pnl: np.ndarray) -> tuple:
    """Return (PF, N) for the trades selected by mask."""
    if mask.sum() == 0:
        return 0.0, 0
    sel_pnl = pnl[mask]
    gp = sel_pnl[sel_pnl > 0].sum()
    gl = abs(sel_pnl[sel_pnl < 0].sum())
    pf = float(gp / gl) if gl > 0 else float("inf")
    return pf, int(mask.sum())


def build_xgb(scale_pos_weight: float = 1.0):
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )


def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.3, max_iter=1000, random_state=42)),
    ])


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def walk_forward_cv(X: np.ndarray, y: np.ndarray, pnl: np.ndarray, scale_pos_weight: float = 1.0):
    """
    4-fold expanding-window walk-forward CV.
    Returns aggregated OOS probabilities (XGB and LR) in chronological order.
    """
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    all_indices = []
    all_xgb_probs = []
    all_lr_probs  = []

    fold_reports = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < MIN_TRAIN_SIZE:
            print(f"  Fold {fold_idx+1}: skip — only {len(train_idx)} train trades")
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
        pnl_test = pnl[test_idx]

        # XGBoost
        xgb = build_xgb(scale_pos_weight)
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]

        # Logistic Regression
        lr = build_lr()
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]

        # Unfiltered OOS PF
        unfiltered_pf, unfiltered_n = pf_from_mask(np.ones(len(y_test), dtype=bool), pnl_test)

        fold_reports.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "unfiltered_pf": unfiltered_pf,
            "xgb_probs": xgb_probs,
            "lr_probs": lr_probs,
            "pnl_test": pnl_test,
            "y_test": y_test,
        })

        all_indices.append(test_idx)
        all_xgb_probs.append(xgb_probs)
        all_lr_probs.append(lr_probs)

        print(f"  Fold {fold_idx+1}: train={len(train_idx)}, test={len(test_idx)}, "
              f"OOS PF (unfiltered)={unfiltered_pf:.4f}")

    if not all_indices:
        return None, None, None, fold_reports

    all_idx_concat  = np.concatenate(all_indices)
    all_xgb_concat  = np.concatenate(all_xgb_probs)
    all_lr_concat   = np.concatenate(all_lr_probs)

    # Sort by original index to preserve chronological order
    sort_order      = np.argsort(all_idx_concat)
    all_xgb_sorted  = all_xgb_concat[sort_order]
    all_lr_sorted   = all_lr_concat[sort_order]
    all_idx_sorted  = all_idx_concat[sort_order]

    return all_idx_sorted, all_xgb_sorted, all_lr_sorted, fold_reports


# ── Threshold evaluation ───────────────────────────────────────────────────────

def evaluate_thresholds(oos_pnl: np.ndarray, oos_xgb: np.ndarray, oos_lr: np.ndarray):
    unfiltered_pf, unfiltered_n = pf_from_mask(np.ones(len(oos_pnl), dtype=bool), oos_pnl)

    rows = []
    for thresh in THRESHOLDS:
        xgb_mask = oos_xgb >= thresh
        lr_mask  = oos_lr  >= thresh

        xgb_pf, xgb_n = pf_from_mask(xgb_mask, oos_pnl)
        lr_pf,  lr_n  = pf_from_mask(lr_mask,   oos_pnl)

        rows.append({
            "threshold": thresh,
            "xgb_n": xgb_n, "xgb_pf": xgb_pf,
            "lr_n":  lr_n,  "lr_pf":  lr_pf,
        })

    return unfiltered_pf, unfiltered_n, rows


# ── Feature importance ─────────────────────────────────────────────────────────

def feature_importance_table(model, feature_names: list) -> str:
    importances = model.feature_importances_
    pairs = sorted(zip(importances, feature_names), reverse=True)
    lines = ["  Feature                    Importance"]
    lines.append("  " + "-" * 40)
    for imp, name in pairs:
        lines.append(f"  {name:<28s}  {imp:.4f}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("S24: ML meta-filter trainer — Phase 2")
    print("=" * 70)

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run s23_meta_label_features.py first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_ts", "exit_ts"])
    df = df.dropna(subset=["label"] + FEATURE_COLS)
    df = df.sort_values("entry_ts").reset_index(drop=True)

    # Use pnl > 0 as the prediction target — captures profitable time-stops,
    # not just TP hits. TP label is 20% positive; pnl>0 gives a more balanced class.
    df["profitable"] = (df["pnl_1x"] > 0).astype(int)

    n_total = len(df)
    n_wins  = int(df["profitable"].sum())
    n_tp    = int(df["label"].sum())
    print(f"Loaded {n_total} labeled trades")
    print(f"  TP hits:    {n_tp}  ({n_tp/n_total*100:.1f}%)")
    print(f"  pnl > 0:    {n_wins}  ({n_wins/n_total*100:.1f}%)  ← ML target")

    if n_total < 20:
        print("ERROR: too few trades for walk-forward CV (need ≥ 20).", file=sys.stderr)
        sys.exit(1)

    X   = df[FEATURE_COLS].values.astype(float)
    y   = df["profitable"].values.astype(int)   # pnl > 0 target
    pnl = df["pnl_1x"].values.astype(float)

    # Class weight for imbalanced target
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    scale_pos = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  scale_pos_weight: {scale_pos:.2f}")

    print(f"\nFeatures ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    print(f"Walk-forward CV: {N_FOLDS} folds (expanding window)\n")

    oos_idx, oos_xgb, oos_lr, fold_reports = walk_forward_cv(X, y, pnl, scale_pos_weight=scale_pos)

    if oos_idx is None:
        print("ERROR: no folds completed — dataset too small.", file=sys.stderr)
        sys.exit(1)

    oos_pnl = pnl[oos_idx]
    unfiltered_pf, unfiltered_n, threshold_rows = evaluate_thresholds(oos_pnl, oos_xgb, oos_lr)

    print(f"\nOOS pool: {len(oos_pnl)} trades across all folds")
    print(f"OOS PF (unfiltered): {unfiltered_pf:.4f}  (N={unfiltered_n})")
    print()

    # Print threshold table
    header = f"{'Thresh':>7}  {'XGB N':>6}  {'XGB PF':>8}  {'LR N':>5}  {'LR PF':>7}"
    print(header)
    print("-" * len(header))
    best_xgb_row = None
    for row in threshold_rows:
        xgb_flag = " ← best" if (
            best_xgb_row is None and
            row["xgb_n"] >= MIN_FILTERED_N and
            row["xgb_pf"] > unfiltered_pf
        ) else ""
        if xgb_flag:
            best_xgb_row = row
        print(f"  {row['threshold']:.2f}   {row['xgb_n']:>6}   {row['xgb_pf']:>8.4f}  "
              f"{row['lr_n']:>5}   {row['lr_pf']:>7.4f}{xgb_flag}")

    # Train final model on ALL in-sample data
    print("\n" + "─" * 70)
    print("Training final XGBoost model on ALL in-sample trades …")
    final_xgb = build_xgb(scale_pos_weight=scale_pos)
    final_xgb.fit(X, y)

    print("\nFeature importances (XGBoost, full-data model):")
    print(feature_importance_table(final_xgb, FEATURE_COLS))

    # Determine best threshold (first one that improves OOS PF with N >= MIN_FILTERED_N)
    best_thresh = 0.55   # fallback
    best_pf     = unfiltered_pf
    for row in threshold_rows:
        if row["xgb_n"] >= MIN_FILTERED_N and row["xgb_pf"] > best_pf:
            best_thresh = row["threshold"]
            best_pf     = row["xgb_pf"]
            break  # take lowest threshold that improves PF (most trades retained)

    # Save model + threshold
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path  = MODEL_DIR / "xgb_model.pkl"
    thresh_path = MODEL_DIR / "threshold.json"
    feat_path   = MODEL_DIR / "feature_names.json"

    joblib.dump(final_xgb, model_path)
    thresh_path.write_text(json.dumps({
        "threshold": best_thresh,
        "oos_pf_unfiltered": round(unfiltered_pf, 4),
        "oos_pf_filtered":   round(best_pf, 4),
        "n_insample_trades": n_total,
    }, indent=2))
    feat_path.write_text(json.dumps(FEATURE_COLS, indent=2))

    print(f"\nModel saved → {model_path}")
    print(f"Threshold   → {thresh_path}  (p ≥ {best_thresh:.2f})")

    # Decision and recommendation
    print("\n" + "=" * 70)
    print("S24 VERDICT")
    print("=" * 70)

    improved = best_pf > unfiltered_pf and best_pf != float("inf")
    if improved:
        print(f"OOS PF with ML filter ({best_thresh:.2f}): {best_pf:.4f}")
        print(f"OOS PF without filter:                  {unfiltered_pf:.4f}")
        print(f"Improvement: +{best_pf - unfiltered_pf:.4f} PF points")
        print()
        print("RECOMMENDATION: OOS PF improved with ML filter.")
        print(f"Pre-register S25 with threshold = {best_thresh:.2f}.")
        print("Architecture: load s24 model → filter signals → measure holdout PF.")
    else:
        print(f"OOS PF with ML filter: did not improve over unfiltered ({unfiltered_pf:.4f})")
        print()
        print("RECOMMENDATION: ML filter does not add holdout value at this sample size.")
        print(f"Pre-register S25 WITHOUT ML filter (use base H1·M15·M1·g0.25 cascade).")
        print("If you choose to add the filter anyway, pre-register with threshold = 0.55.")

    print("=" * 70)
    print(f"\nN warning: {n_total} in-sample trades is a small training set.")
    print("OOS PF estimates have high variance. Treat S24 as directional guidance,")
    print("not a reliable PF forecast. The holdout test (S25) is the final arbiter.")


if __name__ == "__main__":
    main()
