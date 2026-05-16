#!/usr/bin/env python3
"""Year-long walk-forward validation of the deployed LR counter-trend + ML threshold system.

Deployed configuration:
  - LR counter-trend filter: fast_len=390, slow_len=1950
  - Counter-trend polarity: UP+bearish=PASS, DOWN+bullish=PASS, SIDEWAYS=PASS
  - ML model: Pipeline(StandardScaler + LogisticRegression(C=0.01, class_weight="balanced"))
  - Adaptive threshold: per-month search [0.40–0.65] on regime-filtered val tail

Walk-forward parameters:
  - 3-month minimum training seed → first test month = April 2025
  - 9 test months: April–December 2025
  - Val tail: last 2 months of each training fold

Outputs:
  - stdout: per-month table + summary
  - data/reports/tier2_final_validation_YYYYMMDD.txt
  - data/reports/tier2_equity_curve_YYYYMMDD.csv
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

MIN_TRAIN_MONTHS     = 3
VAL_TAIL_MONTHS      = 2
THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
DEFAULT_THRESHOLD    = 0.50
MIN_VAL_TRADES       = 10
LR_FAST              = 390
LR_SLOW              = 1950


# ── Helpers (verbatim from backtest_tier2_wf_adaptive.py) ─────────────────────

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

    if note == "default":
        print(f"  [warn] all threshold candidates yield <{MIN_VAL_TRADES} val trades → using {DEFAULT_THRESHOLD}")

    return best_thr, note


# ── Regime filter (identical to deployed LRRegimeFilter polarity) ──────────────

def regime_counter(lr_regime: str, direction: str) -> bool:
    """Counter-trend: pass when regime opposes signal direction (Silver Bullet logic).
    SIDEWAYS always passes (permissive neutral)."""
    if lr_regime == "UP":   return direction == "bearish"
    if lr_regime == "DOWN": return direction == "bullish"
    return True  # SIDEWAYS


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (FEATURES_PATH, HISTORY_PATH, REGIME_PATH):
        if not p.exists():
            sys.exit(f"Error: {p} not found")

    feat   = pd.read_csv(FEATURES_PATH)
    hist   = pd.read_csv(HISTORY_PATH)
    regime = pd.read_csv(REGIME_PATH)

    n_feat, n_hist, n_reg = len(feat), len(hist), len(regime)
    if not (n_feat == n_hist == n_reg):
        sys.exit(
            f"Row-count mismatch: features={n_feat}, history={n_hist}, regime={n_reg}. "
            "Re-run enrich_tier2_with_regime.py."
        )

    hist["timestamp"]  = pd.to_datetime(hist["timestamp"])
    hist["year_month"] = hist["timestamp"].dt.to_period("M")
    feat["year_month"] = hist["year_month"].values

    date_min = hist["timestamp"].min().date()
    date_max = hist["timestamp"].max().date()
    print(f"Loaded {n_feat} signals | {date_min} → {date_max}")
    print(f"LR regime distribution: {pd.Series(regime['lr_regime'].values).value_counts().to_dict()}")
    return feat, hist, regime


# ── Walk-forward ───────────────────────────────────────────────────────────────

def run_walkforward(
    feat: pd.DataFrame, hist: pd.DataFrame, regime: pd.DataFrame
) -> tuple[list[dict], list[dict]]:
    unique_months = sorted(feat["year_month"].unique())
    n_months = len(unique_months)

    if n_months < MIN_TRAIN_MONTHS:
        sys.exit(
            f"Dataset has only {n_months} unique months; need {MIN_TRAIN_MONTHS} as seed."
        )

    print(f"Months in dataset: {[str(m) for m in unique_months]}")
    print(f"Min training window: {MIN_TRAIN_MONTHS} months | Val tail: {VAL_TAIL_MONTHS} months\n")

    X_all   = feat[FEATURE_COLS].values
    y_all   = feat["label"].values
    pnl_all = hist["pnl"].values
    dir_all = hist["direction"].values
    lr_all  = regime["lr_regime"].values
    ts_all  = hist["timestamp"].values
    ym_all  = feat["year_month"].values

    base_results: list[dict] = []
    dep_results:  list[dict] = []

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

        val_tail_months = prior_months[-VAL_TAIL_MONTHS:]
        val_tail_mask   = np.isin(ym_all[train_mask], val_tail_months)

        model = make_model()
        model.fit(X_all[train_mask], y_all[train_mask])

        proba_val  = model.predict_proba(X_all[train_mask][val_tail_mask])[:, 1]
        proba_test = model.predict_proba(X_all[test_mask])[:, 1]
        pnl_val    = pnl_all[train_mask][val_tail_mask]
        pnl_test   = pnl_all[test_mask]
        dir_val    = dir_all[train_mask][val_tail_mask]
        dir_test   = dir_all[test_mask]
        lr_val     = lr_all[train_mask][val_tail_mask]
        lr_test    = lr_all[test_mask]
        ts_test    = ts_all[test_mask]

        # ── Baseline track ─────────────────────────────────────────────────
        thr_base, thr_base_note = select_threshold(proba_val, pnl_val)
        fil_base = proba_test >= thr_base
        pnl_fil_base = pnl_test[fil_base]

        base_results.append({
            "month":         str(test_month),
            "n_total":       len(pnl_test),
            "n_taken":       int(fil_base.sum()),
            "threshold":     thr_base,
            "wr":            win_rate(pnl_fil_base) if fil_base.sum() > 0 else 0.0,
            "pf":            profit_factor(pnl_fil_base) if fil_base.sum() > 0 else 0.0,
            "pnl":           float(pnl_fil_base.sum()) if fil_base.sum() > 0 else 0.0,
            "_pnl_all":      pnl_test.tolist(),
            "_pnl_fil":      pnl_fil_base.tolist(),
            "_ts_fil":       ts_test[fil_base].tolist(),
            "sparse":        len(pnl_test) < 5,
        })

        # ── Deployed track (LR counter-trend) ──────────────────────────────
        regime_pass_val  = np.array([regime_counter(r, d) for r, d in zip(lr_val, dir_val)])
        regime_pass_test = np.array([regime_counter(r, d) for r, d in zip(lr_test, dir_test)])

        if regime_pass_val.sum() >= MIN_VAL_TRADES:
            thr_dep, thr_dep_note = select_threshold(
                proba_val[regime_pass_val], pnl_val[regime_pass_val]
            )
        else:
            thr_dep, thr_dep_note = DEFAULT_THRESHOLD, "default (thin regime val)"

        fil_dep = regime_pass_test & (proba_test >= thr_dep)
        pnl_fil_dep = pnl_test[fil_dep]

        dep_results.append({
            "month":         str(test_month),
            "n_total":       len(pnl_test),
            "n_regime_pass": int(regime_pass_test.sum()),
            "n_taken":       int(fil_dep.sum()),
            "threshold":     thr_dep,
            "wr":            win_rate(pnl_fil_dep) if fil_dep.sum() > 0 else 0.0,
            "pf":            profit_factor(pnl_fil_dep) if fil_dep.sum() > 0 else 0.0,
            "pnl":           float(pnl_fil_dep.sum()) if fil_dep.sum() > 0 else 0.0,
            "_pnl_all":      pnl_test.tolist(),
            "_pnl_fil":      pnl_fil_dep.tolist(),
            "_ts_fil":       ts_test[fil_dep].tolist(),
            "sparse":        len(pnl_test) < 5,
        })

        dpf = dep_results[-1]["pf"] - base_results[-1]["pf"]
        sign = "+" if dpf >= 0 else ""
        sparse_tag = " *sparse*" if len(pnl_test) < 5 else ""
        print(
            f"  {test_month}  n={len(pnl_test):3d}"
            f"  base: thr={thr_base:.2f} n={int(fil_base.sum()):3d} PF={base_results[-1]['pf']:.3f}"
            f"  dep:  thr={thr_dep:.2f} n={int(fil_dep.sum()):3d} PF={dep_results[-1]['pf']:.3f}"
            f"  ΔPF={sign}{dpf:.3f}{sparse_tag}"
        )

    return base_results, dep_results


# ── Equity curve CSV ───────────────────────────────────────────────────────────

def build_equity_curve(
    base_results: list[dict], dep_results: list[dict]
) -> pd.DataFrame:
    rows = []
    cum_base = 0.0
    cum_dep  = 0.0

    for br, dr in zip(base_results, dep_results):
        assert br["month"] == dr["month"]

        base_ts  = list(br["_ts_fil"])
        base_pnl = list(br["_pnl_fil"])
        dep_ts   = list(dr["_ts_fil"])
        dep_pnl  = list(dr["_pnl_fil"])

        all_ts = sorted(set(str(t) for t in base_ts + dep_ts))

        base_map: dict[str, float] = {}
        for t, p in zip(base_ts, base_pnl):
            base_map[str(t)] = base_map.get(str(t), 0.0) + p

        dep_map: dict[str, float] = {}
        for t, p in zip(dep_ts, dep_pnl):
            dep_map[str(t)] = dep_map.get(str(t), 0.0) + p

        for ts in all_ts:
            bp = base_map.get(ts, 0.0)
            dp = dep_map.get(ts, 0.0)
            cum_base += bp
            cum_dep  += dp
            rows.append({
                "timestamp":    ts,
                "month":        br["month"],
                "pnl_base":     bp,
                "pnl_dep":      dp,
                "cum_baseline": cum_base,
                "cum_deployed": cum_dep,
            })

    return pd.DataFrame(rows)


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(base_results: list[dict], dep_results: list[dict]) -> str:
    unfil_base = np.array([x for r in base_results for x in r["_pnl_fil"]])
    unfil_dep  = np.array([x for r in dep_results  for x in r["_pnl_fil"]])
    all_signals = np.array([x for r in base_results for x in r["_pnl_all"]])

    total_n_sig       = sum(r["n_total"] for r in base_results)
    total_n_regime    = sum(r["n_regime_pass"] for r in dep_results)
    total_n_base_tak  = sum(r["n_taken"] for r in base_results)
    total_n_dep_tak   = sum(r["n_taken"] for r in dep_results)

    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"MNQ Tier2 — Final Validation Walk-Forward (Apr–Dec 2025)  [{ts_now}]",
        f"Deployed: LR counter-trend fast={LR_FAST}, slow={LR_SLOW} | ML LogisticRegression(C=0.01)",
        f"Training seed: {MIN_TRAIN_MONTHS} months | Val tail: {VAL_TAIL_MONTHS} months | Threshold search: {THRESHOLD_CANDIDATES}",
        f"{'='*90}",
        f"",
        f"{'Month':<9} {'N_sig':>5}  {'─── Baseline ───':^28}  {'── Deployed (LR Counter) ──':^32}  {'ΔPF':>7}",
        f"{'':9} {'':5}  {'N_tak':>5} {'Thr':>5} {'WR':>7} {'PF':>7}  {'N_rg':>5} {'N_tak':>5} {'Thr':>5} {'WR':>7} {'PF':>7}",
        f"{'-'*90}",
    ]

    for br, dr in zip(base_results, dep_results):
        assert br["month"] == dr["month"]
        dpf = dr["pf"] - br["pf"]
        sign = "+" if dpf >= 0 else ""
        sparse_tag = " *" if br["sparse"] else "  "
        lines.append(
            f"{br['month']:<9}{sparse_tag}{br['n_total']:>5}"
            f"  {br['n_taken']:>5} {br['threshold']:>5.2f} {br['wr']:>7.1%} {br['pf']:>7.3f}"
            f"  {dr['n_regime_pass']:>5} {dr['n_taken']:>5} {dr['threshold']:>5.2f} {dr['wr']:>7.1%} {dr['pf']:>7.3f}"
            f"  {sign}{dpf:>6.3f}"
        )

    pf_base = profit_factor(unfil_base)
    pf_dep  = profit_factor(unfil_dep)
    wr_base = win_rate(unfil_base)
    wr_dep  = win_rate(unfil_dep)
    pnl_base = unfil_base.sum()
    pnl_dep  = unfil_dep.sum()

    lines += [
        f"{'-'*90}",
        f"{'TOTAL':<9}  {total_n_sig:>5}"
        f"  {total_n_base_tak:>5} {'—':>5} {wr_base:>7.1%} {pf_base:>7.3f}"
        f"  {total_n_regime:>5} {total_n_dep_tak:>5} {'—':>5} {wr_dep:>7.1%} {pf_dep:>7.3f}"
        f"  {'+' if pf_dep>=pf_base else ''}{pf_dep-pf_base:>6.3f}",
        f"",
        f"── Summary ─────────────────────────────────────────────────────────────────",
        f"  Baseline : {total_n_base_tak:>3} trades taken | WR {wr_base:.1%} | PF {pf_base:.3f} | "
        f"IR {per_trade_sharpe(unfil_base):.3f} | MDD ${max_drawdown(unfil_base):,.0f} | P&L ${pnl_base:+,.0f}",
        f"  Deployed : {total_n_dep_tak:>3} trades taken | WR {wr_dep:.1%} | PF {pf_dep:.3f} | "
        f"IR {per_trade_sharpe(unfil_dep):.3f} | MDD ${max_drawdown(unfil_dep):,.0f} | P&L ${pnl_dep:+,.0f}",
        f"  Delta    : PF {'+' if pf_dep>=pf_base else ''}{pf_dep-pf_base:.3f} | "
        f"IR {'+' if per_trade_sharpe(unfil_dep)>=per_trade_sharpe(unfil_base) else ''}"
        f"{per_trade_sharpe(unfil_dep)-per_trade_sharpe(unfil_base):.3f} | "
        f"MDD ${max_drawdown(unfil_dep)-max_drawdown(unfil_base):+,.0f} | P&L ${pnl_dep-pnl_base:+,.0f}",
        f"",
        f"── Regime filter impact ────────────────────────────────────────────────────",
        f"  LR counter-trend passes {total_n_regime}/{total_n_sig} raw signals ({total_n_regime/total_n_sig:.1%})",
        f"  Deployed takes  {total_n_dep_tak}/{total_n_regime} regime-passed signals ({total_n_dep_tak/total_n_regime:.1%} ML filter)",
        f"  Overall acceptance: {total_n_dep_tak}/{total_n_sig} ({total_n_dep_tak/total_n_sig:.1%})",
        f"",
        f"── Unfiltered baseline (no ML, all signals) ───────────────────────────────",
        f"  {total_n_sig} trades | WR {win_rate(all_signals):.1%} | PF {profit_factor(all_signals):.3f} | "
        f"P&L ${all_signals.sum():+,.0f}",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    feat, hist, regime = load_data()
    print()

    base_results, dep_results = run_walkforward(feat, hist, regime)

    if not base_results:
        print("No test months produced results.")
        return

    report = build_report(base_results, dep_results)
    print("\n" + report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    txt_path = REPORT_DIR / f"tier2_final_validation_{date_str}.txt"
    txt_path.write_text(report + "\n")
    print(f"\nReport saved → {txt_path}")

    eq_df = build_equity_curve(base_results, dep_results)
    csv_path = REPORT_DIR / f"tier2_equity_curve_{date_str}.csv"
    eq_df.to_csv(csv_path, index=False)
    print(f"Equity curve → {csv_path} ({len(eq_df)} rows)")


if __name__ == "__main__":
    main()
