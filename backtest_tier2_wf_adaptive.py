#!/usr/bin/env python3
"""Walk-forward backtest for the Tier2 LR meta-labeling model.

Methodology:
  - 6-month minimum training window before first prediction
  - Per-month adaptive threshold: search [0.40–0.65] on the last 2 months
    of the training fold (val tail); apply the winner to the test month
  - Zero look-ahead: test month data is never seen during threshold search
  - Model: Pipeline(StandardScaler + LogisticRegression) matching production

Outputs:
  - Per-month table + aggregate summary (stdout)
  - data/reports/tier2_wf_adaptive_YYYYMMDD.txt
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "fvg_fill_pct", "sweep_window_vol", "volume_ratio", "signal_direction",
    "h1_trend_slope", "atr", "session_displacement", "session_volume_ratio",
]

FEATURES_PATH = Path("data/ml_training/doe_run_08_fullyear_features.csv")
HISTORY_PATH  = Path("data/ml_training/doe_run_08_fullyear_history.csv")
REPORT_DIR    = Path("data/reports")

THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MIN_TRAIN_MONTHS     = 6
VAL_TAIL_MONTHS      = 2
MIN_VAL_TRADES       = 10
DEFAULT_THRESHOLD    = 0.50
AUC_WARN_THRESHOLD   = 0.48
AUC_WARN_STREAK      = 3


def profit_factor(pnl: np.ndarray) -> float:
    gains  = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def per_trade_sharpe(pnl: np.ndarray) -> float:
    """Information ratio (mean/std, no annualisation — per-trade PnL not daily)."""
    if len(pnl) < 2 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std())


def max_drawdown(pnl: np.ndarray) -> float:
    cum = pd.Series(pnl).cumsum()
    return float((cum - cum.cummax()).min())


def win_rate(pnl: np.ndarray) -> float:
    return float((pnl > 0).mean()) if len(pnl) > 0 else 0.0


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.01, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=42,
        )),
    ])


def select_threshold(
    proba_val: np.ndarray,
    pnl_val: np.ndarray,
) -> tuple[float, str]:
    """Pick threshold that maximises PF on the val tail. Returns (threshold, note)."""
    best_pf: float = float("-inf")
    best_thr = DEFAULT_THRESHOLD
    note = "default"

    for thr in THRESHOLD_CANDIDATES:
        mask = proba_val >= thr
        if mask.sum() < MIN_VAL_TRADES:
            continue
        pf = profit_factor(pnl_val[mask])
        if pf > best_pf:
            best_pf, best_thr, note = pf, thr, f"PF={pf:.3f}"

    if note == "default":
        print(f"  [warn] all threshold candidates yield <{MIN_VAL_TRADES} val trades → using {DEFAULT_THRESHOLD}")

    return best_thr, note


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not FEATURES_PATH.exists():
        sys.exit(f"Error: {FEATURES_PATH} not found")
    if not HISTORY_PATH.exists():
        sys.exit(f"Error: {HISTORY_PATH} not found")

    feat = pd.read_csv(FEATURES_PATH)
    hist = pd.read_csv(HISTORY_PATH)

    if len(feat) != len(hist):
        raise ValueError(
            f"Row-count mismatch: {FEATURES_PATH} has {len(feat)} rows, "
            f"{HISTORY_PATH} has {len(hist)} rows. Re-export both files."
        )

    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist["year_month"] = hist["timestamp"].dt.to_period("M")
    feat["year_month"] = hist["year_month"].values

    print(f"Loaded {len(feat)} samples | {hist['timestamp'].min().date()} → {hist['timestamp'].max().date()}")
    return feat, hist


def run_walkforward(feat: pd.DataFrame, hist: pd.DataFrame) -> list[dict]:
    unique_months = sorted(feat["year_month"].unique())
    n_months = len(unique_months)

    # Ask First: if fewer than MIN_TRAIN_MONTHS unique months exist total, warn
    if n_months < MIN_TRAIN_MONTHS:
        ans = input(
            f"\nDataset has only {n_months} unique months; need {MIN_TRAIN_MONTHS} as seed. "
            "There will be no test months. Continue anyway? [y/n]: "
        ).strip().lower()
        if ans != "y":
            sys.exit("Aborted by user.")

    print(f"Months in dataset: {[str(m) for m in unique_months]}")
    print(f"Min training window: {MIN_TRAIN_MONTHS} months | Val tail: {VAL_TAIL_MONTHS} months\n")

    X_all   = feat[FEATURE_COLS].values
    y_all   = feat["label"].values
    pnl_all = hist["pnl"].values
    ym_all  = feat["year_month"].values

    results = []
    auc_warn_streak = 0

    for i, test_month in enumerate(unique_months):
        prior_months = unique_months[:i]

        if len(prior_months) < MIN_TRAIN_MONTHS:
            print(f"  skip {test_month} — only {len(prior_months)} training months (need {MIN_TRAIN_MONTHS})")
            continue

        train_mask = ym_all < test_month
        test_mask  = ym_all == test_month

        if test_mask.sum() == 0:
            print(f"  skip {test_month} — no test trades")
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  pnl_test = X_all[test_mask],  pnl_all[test_mask]

        # Val tail: last VAL_TAIL_MONTHS of training fold
        val_tail_months = prior_months[-VAL_TAIL_MONTHS:]
        val_mask_in_train = np.isin(ym_all[train_mask], val_tail_months)

        model = make_model()
        model.fit(X_train, y_train)

        # AUC check on val tail (advisory)
        if val_mask_in_train.sum() >= 10:
            try:
                proba_val = model.predict_proba(X_train[val_mask_in_train])[:, 1]
                auc_val = roc_auc_score(y_train[val_mask_in_train], proba_val)
                if auc_val < AUC_WARN_THRESHOLD:
                    auc_warn_streak += 1
                    print(f"  [warn] {test_month}: val-tail AUC={auc_val:.3f} < {AUC_WARN_THRESHOLD} (streak={auc_warn_streak})")
                    if auc_warn_streak >= AUC_WARN_STREAK:
                        ans = input(
                            f"\nAUC has been below {AUC_WARN_THRESHOLD} for {AUC_WARN_STREAK} consecutive months. "
                            "Continue? [y/n]: "
                        ).strip().lower()
                        if ans != "y":
                            sys.exit("Aborted by user.")
                        auc_warn_streak = 0
                else:
                    auc_warn_streak = 0
            except Exception:
                pass

        # Threshold selection on val tail
        pnl_val_tail = pnl_all[train_mask][val_mask_in_train]
        if val_mask_in_train.sum() >= MIN_VAL_TRADES:
            proba_val_tail = model.predict_proba(X_train[val_mask_in_train])[:, 1]
            threshold, thr_note = select_threshold(proba_val_tail, pnl_val_tail)
        else:
            threshold, thr_note = DEFAULT_THRESHOLD, "default (thin val tail)"

        # Apply to test month — cache per-trade arrays to avoid a second fit
        proba_test = model.predict_proba(X_test)[:, 1]
        fil_mask   = proba_test >= threshold

        n_total  = len(pnl_test)
        n_fil    = fil_mask.sum()
        pnl_fil  = pnl_test[fil_mask]

        results.append({
            "month":         str(test_month),
            "n_total":       n_total,
            "n_taken":       int(n_fil),
            "threshold":     threshold,
            "pnl_unfil":     float(pnl_test.sum()),
            "wr_unfil":      win_rate(pnl_test),
            "pf_unfil":      profit_factor(pnl_test),
            "pnl_fil":       float(pnl_fil.sum()) if n_fil > 0 else 0.0,
            "wr_fil":        win_rate(pnl_fil) if n_fil > 0 else 0.0,
            "pf_fil":        profit_factor(pnl_fil) if n_fil > 0 else 0.0,
            "sparse":        n_total < 5,
            "_pnl_test_arr": pnl_test.tolist(),   # cached for aggregate stats
            "_pnl_fil_arr":  pnl_fil.tolist(),
        })

        sparse_tag = " *sparse*" if n_total < 5 else ""
        print(
            f"  {test_month}  n={n_total:3d}  thr={threshold:.2f} ({thr_note:12s})  "
            f"unfil PF={results[-1]['pf_unfil']:.3f}  fil PF={results[-1]['pf_fil']:.3f}  "
            f"taken={n_fil}/{n_total}{sparse_tag}"
        )

    return results


def build_report(results: list[dict]) -> str:
    # Aggregate per-trade arrays from cached data — no second model fit needed
    unfil_arr = np.array([x for r in results for x in r["_pnl_test_arr"]])
    fil_arr   = np.array([x for r in results for x in r["_pnl_fil_arr"]])

    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines += [
        f"Tier2 Walk-Forward — Adaptive Threshold  ({ts})",
        f"{'='*70}",
        f"Min training window: {MIN_TRAIN_MONTHS} months | Val tail: {VAL_TAIL_MONTHS} months",
        f"Threshold candidates: {THRESHOLD_CANDIDATES}",
        f"",
        f"{'Month':<10} {'N':>4} {'Taken':>5} {'Thr':>5}  {'Unfil WR':>8} {'Unfil PF':>8} {'Unfil $':>8}  {'Fil WR':>7} {'Fil PF':>7} {'Fil $':>8}",
        f"{'-'*90}",
    ]

    for r in results:
        sparse_tag = " *sparse*" if r["sparse"] else ""
        lines.append(
            f"{r['month']:<10} {r['n_total']:>4} {r['n_taken']:>5} {r['threshold']:>5.2f}"
            f"  {r['wr_unfil']:>8.1%} {r['pf_unfil']:>8.3f} {r['pnl_unfil']:>8.0f}"
            f"  {r['wr_fil']:>7.1%} {r['pf_fil']:>7.3f} {r['pnl_fil']:>8.0f}{sparse_tag}"
        )

    n_total_agg  = len(unfil_arr)
    n_taken_agg  = len(fil_arr)
    agg_pf_unfil = profit_factor(unfil_arr)
    agg_pf_fil   = profit_factor(fil_arr)
    agg_pnl_unfil = unfil_arr.sum()
    agg_pnl_fil   = fil_arr.sum()
    wr_unfil_agg  = win_rate(unfil_arr)
    wr_fil_agg    = win_rate(fil_arr)
    pct_taken     = n_taken_agg / n_total_agg if n_total_agg > 0 else 0.0

    lines += [
        f"{'-'*90}",
        f"{'TOTAL':<10} {n_total_agg:>4} {n_taken_agg:>5} {'—':>5}"
        f"  {wr_unfil_agg:>8.1%} {agg_pf_unfil:>8.3f} {agg_pnl_unfil:>8.0f}"
        f"  {wr_fil_agg:>7.1%} {agg_pf_fil:>7.3f} {agg_pnl_fil:>8.0f}",
        f"",
        f"── Summary ─────────────────────────────────────────────────",
        f"  Unfiltered: {n_total_agg} trades | WR {wr_unfil_agg:.1%} | PF {agg_pf_unfil:.3f} | Total ${agg_pnl_unfil:,.0f}",
        f"  Filtered  : {n_taken_agg} trades ({pct_taken:.1%} taken) | WR {wr_fil_agg:.1%} | PF {agg_pf_fil:.3f} | Total ${agg_pnl_fil:,.0f}",
        f"  Filter delta: ${agg_pnl_fil - agg_pnl_unfil:+,.0f}  PF delta: {agg_pf_fil - agg_pf_unfil:+.3f}",
        f"",
        f"── Risk metrics (per-trade information ratio, not annualised) ──",
        f"  Unfiltered: IR {per_trade_sharpe(unfil_arr):.3f} | MaxDD ${max_drawdown(unfil_arr):,.0f}",
        f"  Filtered  : IR {per_trade_sharpe(fil_arr):.3f}  | MaxDD ${max_drawdown(fil_arr):,.0f}",
    ]
    return "\n".join(lines)


def main() -> None:
    feat, hist = load_data()
    print()

    results = run_walkforward(feat, hist)

    if not results:
        print("No test months produced results.")
        return

    report = build_report(results)
    print("\n" + report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = REPORT_DIR / f"tier2_wf_adaptive_{date_str}.txt"
    out_path.write_text(report + "\n")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
