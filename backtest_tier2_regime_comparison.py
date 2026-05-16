#!/usr/bin/env python3
"""Three-track walk-forward comparison: baseline vs LR channel vs HMM regime pre-filter.

Methodology (identical to backtest_tier2_wf_adaptive.py):
  - 6-month minimum training window, first test month = July 2025
  - Per-month adaptive threshold on val tail (last 2 months of training fold)
  - Zero look-ahead: regime labels are computed from bars before signal timestamp

Three tracks per test month:
  1. Baseline   — all signals, 8 ICT features + adaptive threshold
  2. LR regime  — regime pre-filter, then same model + adaptive threshold
  3. HMM regime — regime pre-filter, then same model + adaptive threshold

Regime filter logic (permissive):
  LR "UP"        → agrees with bullish signals  (bearish filtered out)
  LR "DOWN"      → agrees with bearish signals  (bullish filtered out)
  LR "SIDEWAYS"  → neutral, both directions pass
  HMM 1          → trending_up   → agrees with bullish
  HMM 0 or 2     → trending_down → agrees with bearish
  HMM -1         → neutral (warm-up or load failure), both directions pass

Inputs:
  data/ml_training/doe_run_08_fullyear_features.csv
  data/ml_training/doe_run_08_fullyear_history.csv
  data/ml_training/doe_run_08_regime_enriched.csv  (from enrich_tier2_with_regime.py)

Outputs:
  data/reports/tier2_regime_comparison_YYYYMMDD.txt
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
REGIME_PATH   = Path("data/ml_training/doe_run_08_regime_enriched.csv")
REPORT_DIR    = Path("data/reports")

THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
MIN_TRAIN_MONTHS     = 6
VAL_TAIL_MONTHS      = 2
MIN_VAL_TRADES       = 10
DEFAULT_THRESHOLD    = 0.50
AUC_WARN_THRESHOLD   = 0.48
AUC_WARN_STREAK      = 3

TRACKS = ["baseline", "lr_regime", "hmm_regime"]


# ── Helpers (identical to backtest_tier2_wf_adaptive.py) ─────────────────────

def profit_factor(pnl: np.ndarray) -> float:
    gains  = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def per_trade_sharpe(pnl: np.ndarray) -> float:
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


def select_threshold(proba_val: np.ndarray, pnl_val: np.ndarray) -> tuple[float, str]:
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
    return best_thr, note


def regime_agrees(lr_or_hmm_regime, signal_direction: str) -> bool:
    """Return True if the regime is neutral or aligns with signal direction."""
    # LR channel
    if lr_or_hmm_regime == "UP":
        return signal_direction == "bullish"
    if lr_or_hmm_regime == "DOWN":
        return signal_direction == "bearish"
    if lr_or_hmm_regime == "SIDEWAYS":
        return True   # neutral → pass through
    # HMM integer
    if lr_or_hmm_regime == 1:     # trending_up
        return signal_direction == "bullish"
    if lr_or_hmm_regime in (0, 2):  # trending_down (both states)
        return signal_direction == "bearish"
    return True  # -1 or unknown → neutral, pass through


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (FEATURES_PATH, HISTORY_PATH, REGIME_PATH):
        if not p.exists():
            sys.exit(
                f"Error: {p} not found.\n"
                + ("Run: .venv/bin/python enrich_tier2_with_regime.py" if p == REGIME_PATH else "")
            )

    feat   = pd.read_csv(FEATURES_PATH)
    hist   = pd.read_csv(HISTORY_PATH)
    regime = pd.read_csv(REGIME_PATH)

    if not (len(feat) == len(hist) == len(regime)):
        raise ValueError(
            f"Row-count mismatch: features={len(feat)}, history={len(hist)}, regime={len(regime)}"
        )

    hist["timestamp"]   = pd.to_datetime(hist["timestamp"])
    hist["year_month"]  = hist["timestamp"].dt.to_period("M")
    feat["year_month"]  = hist["year_month"].values
    regime["year_month"] = hist["year_month"].values

    print(
        f"Loaded {len(feat)} samples | "
        f"{hist['timestamp'].min().date()} → {hist['timestamp'].max().date()}"
    )
    return feat, hist, regime


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walkforward(
    feat: pd.DataFrame,
    hist: pd.DataFrame,
    regime: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Returns dict of track_name → list of monthly result dicts."""

    unique_months = sorted(feat["year_month"].unique())
    n_months = len(unique_months)

    if n_months < MIN_TRAIN_MONTHS:
        ans = input(
            f"\nDataset has only {n_months} unique months; need {MIN_TRAIN_MONTHS} as seed. "
            "Continue? [y/n]: "
        ).strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    print(f"Months: {[str(m) for m in unique_months]}")
    print(f"Min training window: {MIN_TRAIN_MONTHS} | Val tail: {VAL_TAIL_MONTHS}\n")

    X_all   = feat[FEATURE_COLS].values
    y_all   = feat["label"].values
    pnl_all = hist["pnl"].values
    dir_all = hist["direction"].values
    ym_all  = feat["year_month"].values
    lr_all  = regime["lr_regime"].values
    hmm_all = regime["hmm_regime"].values

    track_results: dict[str, list[dict]] = {t: [] for t in TRACKS}
    auc_warn_streak = 0

    for i, test_month in enumerate(unique_months):
        prior_months = unique_months[:i]

        if len(prior_months) < MIN_TRAIN_MONTHS:
            print(f"  skip {test_month} — only {len(prior_months)} training months")
            continue

        train_mask = ym_all < test_month
        test_mask  = ym_all == test_month

        if test_mask.sum() == 0:
            print(f"  skip {test_month} — no test trades")
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test,  pnl_test = X_all[test_mask], pnl_all[test_mask]
        dir_test = dir_all[test_mask]
        lr_test  = lr_all[test_mask]
        hmm_test = hmm_all[test_mask]

        val_tail_months = prior_months[-VAL_TAIL_MONTHS:]
        val_mask_in_train = np.isin(ym_all[train_mask], val_tail_months)
        pnl_val_tail = pnl_all[train_mask][val_mask_in_train]
        dir_val = dir_all[train_mask][val_mask_in_train]
        lr_val  = lr_all[train_mask][val_mask_in_train]
        hmm_val = hmm_all[train_mask][val_mask_in_train]

        model = make_model()
        model.fit(X_train, y_train)

        # AUC advisory check (baseline val tail)
        if val_mask_in_train.sum() >= 10:
            try:
                proba_v = model.predict_proba(X_train[val_mask_in_train])[:, 1]
                auc = roc_auc_score(y_train[val_mask_in_train], proba_v)
                if auc < AUC_WARN_THRESHOLD:
                    auc_warn_streak += 1
                    print(f"  [warn] {test_month}: val AUC={auc:.3f} (streak={auc_warn_streak})")
                    if auc_warn_streak >= AUC_WARN_STREAK:
                        ans = input("AUC warn streak hit — continue? [y/n]: ").strip().lower()
                        if ans != "y":
                            sys.exit("Aborted.")
                        auc_warn_streak = 0
                else:
                    auc_warn_streak = 0
            except Exception:
                pass

        # Model probabilities
        proba_val_all  = (
            model.predict_proba(X_train[val_mask_in_train])[:, 1]
            if val_mask_in_train.sum() >= MIN_VAL_TRADES else np.array([])
        )
        proba_test_all = model.predict_proba(X_test)[:, 1]

        n_total = len(pnl_test)

        for track in TRACKS:
            # Build boolean regime-pass masks for val tail and test month
            if track == "baseline":
                regime_pass_val  = np.ones(len(pnl_val_tail), dtype=bool)
                regime_pass_test = np.ones(n_total, dtype=bool)
            elif track == "lr_regime":
                regime_pass_val  = np.array([regime_agrees(r, d) for r, d in zip(lr_val, dir_val)])
                regime_pass_test = np.array([regime_agrees(r, d) for r, d in zip(lr_test, dir_test)])
            else:  # hmm_regime
                regime_pass_val  = np.array([regime_agrees(r, d) for r, d in zip(hmm_val, dir_val)])
                regime_pass_test = np.array([regime_agrees(r, d) for r, d in zip(hmm_test, dir_test)])

            # Threshold search on regime-filtered val tail
            pnl_val_filtered  = pnl_val_tail[regime_pass_val]
            proba_val_filtered = proba_val_all[regime_pass_val] if len(proba_val_all) > 0 else np.array([])

            if len(proba_val_filtered) >= MIN_VAL_TRADES:
                threshold, thr_note = select_threshold(proba_val_filtered, pnl_val_filtered)
            else:
                threshold, thr_note = DEFAULT_THRESHOLD, "default (thin)"

            # Apply to test month: regime filter first, then model threshold
            regime_test_idx   = np.where(regime_pass_test)[0]
            pnl_regime        = pnl_test[regime_pass_test]
            proba_test_regime = proba_test_all[regime_pass_test]

            model_pass = proba_test_regime >= threshold
            pnl_taken  = pnl_regime[model_pass]

            n_regime = regime_pass_test.sum()
            n_taken  = model_pass.sum()

            track_results[track].append({
                "month":          str(test_month),
                "n_total":        n_total,
                "n_regime":       int(n_regime),
                "n_taken":        int(n_taken),
                "threshold":      threshold,
                "pnl_unfil":      float(pnl_test.sum()),
                "wr_unfil":       win_rate(pnl_test),
                "pf_unfil":       profit_factor(pnl_test),
                "pnl_regime":     float(pnl_regime.sum()) if n_regime > 0 else 0.0,
                "wr_regime":      win_rate(pnl_regime) if n_regime > 0 else 0.0,
                "pf_regime":      profit_factor(pnl_regime) if n_regime > 0 else 0.0,
                "pnl_taken":      float(pnl_taken.sum()) if n_taken > 0 else 0.0,
                "wr_taken":       win_rate(pnl_taken) if n_taken > 0 else 0.0,
                "pf_taken":       profit_factor(pnl_taken) if n_taken > 0 else 0.0,
                "_pnl_unfil_arr": pnl_test.tolist(),
                "_pnl_regime_arr": pnl_regime.tolist(),
                "_pnl_taken_arr": pnl_taken.tolist(),
            })

        # Progress line (baseline track for readability)
        br = track_results["baseline"][-1]
        lr_r = track_results["lr_regime"][-1]
        hm_r = track_results["hmm_regime"][-1]
        print(
            f"  {test_month}  n={n_total}  "
            f"base PF={br['pf_taken']:.3f} ({br['n_taken']})  "
            f"LR PF={lr_r['pf_taken']:.3f} ({lr_r['n_taken']}/{lr_r['n_regime']})  "
            f"HMM PF={hm_r['pf_taken']:.3f} ({hm_r['n_taken']}/{hm_r['n_regime']})"
        )

    return track_results


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(track_results: dict[str, list[dict]]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def agg(results: list[dict], key: str) -> np.ndarray:
        return np.array([x for r in results for x in r[key]])

    sections = [
        f"MNQ Tier2 — Regime Comparison Walk-Forward  ({ts})",
        f"{'='*80}",
        f"Tracks: Baseline (no regime) | LR Channel (50/200-bar) | HMM (1-min model)",
        f"Regime filter: permissive (SIDEWAYS/neutral = pass through)",
        f"",
    ]

    # Per-track detailed table
    for track_key, label in [
        ("baseline",  "BASELINE — 8 ICT features, no regime pre-filter"),
        ("lr_regime", "LR CHANNEL REGIME — permissive pre-filter then model"),
        ("hmm_regime","HMM REGIME — permissive pre-filter then model"),
    ]:
        results = track_results[track_key]
        unfil_arr  = agg(results, "_pnl_unfil_arr")
        regime_arr = agg(results, "_pnl_regime_arr")
        taken_arr  = agg(results, "_pnl_taken_arr")

        sections.append(f"── {label} ──")
        hdr = f"{'Month':<10} {'N_sig':>5} {'N_reg':>6} {'N_tak':>6} {'Thr':>5}  {'Reg WR':>6} {'Reg PF':>7} {'Reg $':>7}  {'Tak WR':>6} {'Tak PF':>7} {'Tak $':>7}"
        sections.append(hdr)
        sections.append("-" * len(hdr))

        for r in results:
            sections.append(
                f"{r['month']:<10} {r['n_total']:>5} {r['n_regime']:>6} {r['n_taken']:>6} {r['threshold']:>5.2f}"
                f"  {r['wr_regime']:>6.1%} {r['pf_regime']:>7.3f} {r['pnl_regime']:>7.0f}"
                f"  {r['wr_taken']:>6.1%} {r['pf_taken']:>7.3f} {r['pnl_taken']:>7.0f}"
            )

        n_sig_agg   = len(unfil_arr)
        n_reg_agg   = len(regime_arr)
        n_tak_agg   = len(taken_arr)
        sections.append("-" * len(hdr))
        sections.append(
            f"{'TOTAL':<10} {n_sig_agg:>5} {n_reg_agg:>6} {n_tak_agg:>6} {'—':>5}"
            f"  {win_rate(regime_arr):>6.1%} {profit_factor(regime_arr):>7.3f} {regime_arr.sum():>7.0f}"
            f"  {win_rate(taken_arr):>6.1%} {profit_factor(taken_arr):>7.3f} {taken_arr.sum():>7.0f}"
        )
        sections.append(
            f"  IR(taken): {per_trade_sharpe(taken_arr):.3f}  "
            f"MDD(taken): ${max_drawdown(taken_arr):,.0f}  "
            f"regime-pass: {n_reg_agg}/{n_sig_agg} ({n_reg_agg/n_sig_agg:.1%})  "
            f"model-pass: {n_tak_agg}/{n_reg_agg} ({n_tak_agg/n_reg_agg:.1%} of regime)"
        )
        sections.append("")

    # ── Side-by-side summary ─────────────────────────────────────────────────
    def summary_row(track_key: str) -> dict:
        res = track_results[track_key]
        taken = agg(res, "_pnl_taken_arr")
        regime_arr = agg(res, "_pnl_regime_arr")
        unfil = agg(res, "_pnl_unfil_arr")
        return {
            "n_sig":  len(unfil),
            "n_reg":  len(regime_arr),
            "n_tak":  len(taken),
            "wr":     win_rate(taken),
            "pf":     profit_factor(taken),
            "pnl":    taken.sum(),
            "ir":     per_trade_sharpe(taken),
            "mdd":    max_drawdown(taken),
        }

    base = summary_row("baseline")
    lr   = summary_row("lr_regime")
    hmm  = summary_row("hmm_regime")

    sections += [
        "── Side-by-side Aggregate Summary ──────────────────────────────────",
        f"{'Metric':<22} {'Baseline':>12} {'LR Channel':>12} {'HMM Regime':>12}",
        f"{'-'*60}",
        f"{'Signals (test months)':<22} {base['n_sig']:>12} {lr['n_sig']:>12} {hmm['n_sig']:>12}",
        f"{'Regime-pass':<22} {base['n_reg']:>12} {lr['n_reg']:>12} {hmm['n_reg']:>12}",
        f"{'Model-taken':<22} {base['n_tak']:>12} {lr['n_tak']:>12} {hmm['n_tak']:>12}",
        f"{'Win Rate':<22} {base['wr']:>12.1%} {lr['wr']:>12.1%} {hmm['wr']:>12.1%}",
        f"{'Profit Factor':<22} {base['pf']:>12.3f} {lr['pf']:>12.3f} {hmm['pf']:>12.3f}",
        f"{'Total P&L ($)':<22} {base['pnl']:>12,.0f} {lr['pnl']:>12,.0f} {hmm['pnl']:>12,.0f}",
        f"{'Info Ratio (per-trade)':<22} {base['ir']:>12.3f} {lr['ir']:>12.3f} {hmm['ir']:>12.3f}",
        f"{'Max Drawdown ($)':<22} {base['mdd']:>12,.0f} {lr['mdd']:>12,.0f} {hmm['mdd']:>12,.0f}",
        f"{'-'*60}",
        f"{'PF delta vs baseline':<22} {'—':>12} {lr['pf']-base['pf']:>+12.3f} {hmm['pf']-base['pf']:>+12.3f}",
        f"{'P&L delta vs baseline':<22} {'—':>12} {lr['pnl']-base['pnl']:>+12,.0f} {hmm['pnl']-base['pnl']:>+12,.0f}",
    ]

    return "\n".join(sections)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    feat, hist, regime = load_data()
    print()

    track_results = run_walkforward(feat, hist, regime)

    if not any(track_results[t] for t in TRACKS):
        print("No test months produced results.")
        return

    report = build_report(track_results)
    print("\n" + report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = REPORT_DIR / f"tier2_regime_comparison_{date_str}.txt"
    out_path.write_text(report + "\n")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
