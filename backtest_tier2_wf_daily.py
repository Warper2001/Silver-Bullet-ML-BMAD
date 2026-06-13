#!/usr/bin/env python3
"""Daily-retrain walk-forward / model-drift test for the YANK (Tier2) LR meta-labeling model.

Mirrors the stat-arb robustness test ("retrain once a day"). Where
`backtest_tier2_wf_adaptive.py` retrains the LR meta-labeling model **per month**,
this script retrains it **once per day** on a **rolling trailing window**, and
contrasts it against the frozen production model to quantify drift.

Tracks evaluated over the IDENTICAL set of test days (apples-to-apples):
  • NO-FILTER       — take every signal (context / lower bound).
  • PROD-FROZEN     — the actual deployed model (models/xgboost/tier2_meta_labeling_model.pkl)
                      with its production threshold, held fixed across all test days.
                      Directly answers "is the deployed model drifting?". (caveat: it was
                      trained on a different/smaller dataset, so its calibration is its own.)
  • WARMUP-FROZEN   — a model trained ONCE on the first rolling window and frozen. The clean
                      same-data/same-features drift baseline.
  • DAILY-RETRAINED — the model retrained every day on the trailing window, with an adaptive
                      threshold re-selected daily on the val tail.

Methodology:
  - Rolling trailing training window of TRAIN_WINDOW_DAYS calendar days.
  - Zero look-ahead: a test day D trains strictly on signals dated < D (asserted at runtime).
  - Model architecture matches production: Pipeline(StandardScaler + LogisticRegression).
  - 18-feature set, matching the deployed model's feature_names_in_.

Outputs:
  - stdout per-month rollup + aggregate summary + drift verdict
  - data/reports/tier2_wf_daily_<YYYYMMDD>.txt
  - data/reports/tier2_wf_daily_trajectory_<YYYYMMDD>.csv  (per-signal test predictions)
  - data/reports/tier2_wf_daily_<YYYYMMDD>.png             (cum PnL + rolling AUC)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the validated helpers from the monthly walk-forward template.
from backtest_tier2_wf_adaptive import (
    profit_factor,
    per_trade_sharpe,
    max_drawdown,
    win_rate,
    make_model,
    select_threshold,
    THRESHOLD_CANDIDATES,
    DEFAULT_THRESHOLD,
    MIN_VAL_TRADES,
    AUC_WARN_THRESHOLD,
    AUC_WARN_STREAK,
)

# 18-feature set = the deployed model's feature_names_in_ (everything in the
# feature CSV except the `label` column). Keeps all tracks directly comparable.
FEATURE_COLS = [
    "atr", "gap_size", "volume_ratio", "et_hour", "day_of_week", "signal_direction",
    "session_displacement", "adr_pct_used", "fvg_to_sweep_bars", "prior_setup_proximity",
    "h1_trend_slope", "sin_hour", "cos_hour", "session_volume_ratio", "fvg_fill_pct",
    "bar_body_ratio", "sweep_window_vol", "slope_direction_match",
]

FEATURES_PATH = Path("data/ml_training/yank_drift_combined_features.csv")
HISTORY_PATH  = Path("data/ml_training/yank_drift_combined_history.csv")
PROD_MODEL_PATH = Path("models/xgboost/tier2_meta_labeling_model.pkl")
PROD_THRESH_PATH = Path("models/xgboost/tier2_threshold.json")
REPORT_DIR = Path("data/reports")

TRAIN_WINDOW_DAYS = 182   # ~6 months rolling (calendar days); override via --window
VAL_TAIL_DAYS     = 61    # ~2 months, for adaptive threshold search; override via --val-tail
MIN_TRAIN_SAMPLES = 60    # need this many trades in the window to fit; override via --min-train
ROLLING_AUC_N     = 100   # trailing #signals for the rolling-AUC trajectory
EXPANDING         = False  # if True, train on ALL prior data (expanding window); set via --expanding


# ── Data ────────────────────────────────────────────────────────────────────── #

def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists() or not HISTORY_PATH.exists():
        sys.exit(f"Error: combined dataset not found ({FEATURES_PATH} / {HISTORY_PATH})")

    feat = pd.read_csv(FEATURES_PATH)
    hist = pd.read_csv(HISTORY_PATH)
    if len(feat) != len(hist):
        sys.exit(f"Row-count mismatch: features {len(feat)} vs history {len(hist)}")

    missing = [c for c in FEATURE_COLS if c not in feat.columns]
    if missing:
        sys.exit(f"Feature columns missing from dataset: {missing}")

    df = feat[FEATURE_COLS + ["label"]].copy()
    df["pnl"] = hist["pnl"].values
    df["timestamp"] = pd.to_datetime(hist["timestamp"].values)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["year_month"] = df["timestamp"].dt.to_period("M")
    print(f"Loaded {len(df)} signals | {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def load_prod_model():
    if not PROD_MODEL_PATH.exists():
        print(f"[warn] production model {PROD_MODEL_PATH} not found — PROD track skipped")
        return None, None
    import joblib
    model = joblib.load(PROD_MODEL_PATH)
    thr = 0.5
    if PROD_THRESH_PATH.exists():
        try:
            thr = float(json.loads(PROD_THRESH_PATH.read_text())["threshold"])
        except Exception as e:
            print(f"[warn] could not read prod threshold ({e}) — using {thr}")
    # Sanity: feature alignment
    expected = list(getattr(model, "feature_names_in_", FEATURE_COLS))
    if expected != FEATURE_COLS:
        print(f"[warn] prod model feature order differs from FEATURE_COLS — reindexing to model order")
    print(f"Production model loaded | threshold={thr}")
    return model, thr


# ── Walk-forward ──────────────────────────────────────────────────────────────── #

def run_walkforward(df: pd.DataFrame, prod_model, prod_thr: float) -> pd.DataFrame:
    X_all   = df[FEATURE_COLS].values
    y_all   = df["label"].values
    dates   = df["date"].values  # datetime64[ns], midnight-normalised

    start_date = df["date"].min()
    first_test = start_date + pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    test_dates = sorted(d for d in df["date"].unique() if d >= first_test)
    if not test_dates:
        sys.exit("No test days after warmup window — dataset too short.")

    mode = "EXPANDING (all prior data)" if EXPANDING else f"ROLLING {TRAIN_WINDOW_DAYS}d"
    print(f"Window mode: {mode} | warmup: {TRAIN_WINDOW_DAYS}d | val tail: {VAL_TAIL_DAYS}d")
    print(f"Warmup ends {pd.Timestamp(first_test).date()} | {len(test_dates)} test days "
          f"({pd.Timestamp(test_dates[0]).date()} → {pd.Timestamp(test_dates[-1]).date()})\n")

    # ---- WARMUP-FROZEN: train once on the first rolling window, freeze ----
    win_lo = first_test - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    warm_mask = (df["date"] >= win_lo) & (df["date"] < first_test)
    assert df.loc[warm_mask, "date"].max() < first_test, "warmup look-ahead!"
    frozen_model = make_model()
    frozen_model.fit(X_all[warm_mask.values], y_all[warm_mask.values])
    # frozen threshold from its own val tail
    vt_lo = first_test - pd.Timedelta(days=VAL_TAIL_DAYS)
    fvt_mask = (df["date"] >= vt_lo) & (df["date"] < first_test)
    if fvt_mask.sum() >= MIN_VAL_TRADES:
        p_fvt = frozen_model.predict_proba(X_all[fvt_mask.values])[:, 1]
        frozen_thr, _ = select_threshold(p_fvt, df.loc[fvt_mask, "pnl"].values)
    else:
        frozen_thr = DEFAULT_THRESHOLD
    print(f"WARMUP-FROZEN trained on {int(warm_mask.sum())} signals | frozen threshold={frozen_thr:.2f}")

    records = []
    for D in test_dates:
        D = pd.Timestamp(D)
        lo = start_date if EXPANDING else D - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        train_mask = (df["date"] >= lo) & (df["date"] < D)
        if train_mask.sum() < MIN_TRAIN_SAMPLES:
            continue
        # Look-ahead guard
        assert df.loc[train_mask, "date"].max() < D, f"look-ahead at {D.date()}"

        Xtr, ytr = X_all[train_mask.values], y_all[train_mask.values]
        daily_model = make_model()
        daily_model.fit(Xtr, ytr)

        # adaptive threshold on val tail (last VAL_TAIL_DAYS of window)
        vlo = D - pd.Timedelta(days=VAL_TAIL_DAYS)
        vmask = (df["date"] >= vlo) & (df["date"] < D)
        if vmask.sum() >= MIN_VAL_TRADES:
            p_v = daily_model.predict_proba(X_all[vmask.values])[:, 1]
            daily_thr, _ = select_threshold(p_v, df.loc[vmask, "pnl"].values)
        else:
            daily_thr = DEFAULT_THRESHOLD

        test_mask = (df["date"] == D)
        idx = np.where(test_mask.values)[0]
        Xte = X_all[idx]
        p_daily  = daily_model.predict_proba(Xte)[:, 1]
        p_frozen = frozen_model.predict_proba(Xte)[:, 1]
        if prod_model is not None:
            p_prod = prod_model.predict_proba(pd.DataFrame(Xte, columns=FEATURE_COLS))[:, 1]
        else:
            p_prod = np.full(len(idx), np.nan)

        for k, row_i in enumerate(idx):
            records.append({
                "timestamp": df.at[row_i, "timestamp"],
                "date":      D,
                "year_month": str(df.at[row_i, "year_month"]),
                "label":     int(y_all[row_i]),
                "pnl":       float(df.at[row_i, "pnl"]),
                "p_prod":    float(p_prod[k]),
                "p_frozen":  float(p_frozen[k]),
                "p_daily":   float(p_daily[k]),
                "thr_daily": float(daily_thr),
            })

    rec = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    rec.attrs["frozen_thr"] = frozen_thr
    rec.attrs["prod_thr"] = prod_thr if prod_model is not None else None
    return rec


# ── Metrics & report ──────────────────────────────────────────────────────────── #

def track_kept_mask(rec: pd.DataFrame, track: str) -> np.ndarray:
    if track == "nofilter":
        return np.ones(len(rec), dtype=bool)
    if track == "daily":
        return (rec["p_daily"] >= rec["thr_daily"]).values
    if track == "frozen":
        return (rec["p_frozen"] >= rec.attrs["frozen_thr"]).values
    if track == "prod":
        if rec.attrs["prod_thr"] is None:
            return np.zeros(len(rec), dtype=bool)
        return (rec["p_prod"] >= rec.attrs["prod_thr"]).values
    raise ValueError(track)


def track_stats(rec: pd.DataFrame, track: str) -> dict:
    m = track_kept_mask(rec, track)
    pnl = rec.loc[m, "pnl"].values
    return {
        "track": track,
        "n": int(m.sum()),
        "pct_taken": float(m.mean()),
        "wr": win_rate(pnl),
        "pf": profit_factor(pnl),
        "pnl": float(pnl.sum()),
        "ir": per_trade_sharpe(pnl),
        "maxdd": max_drawdown(pnl),
    }


def monthly_auc(rec: pd.DataFrame, proba_col: str) -> pd.Series:
    out = {}
    for ym, g in rec.groupby("year_month"):
        if g["label"].nunique() < 2 or g[proba_col].isna().all():
            out[ym] = np.nan
        else:
            try:
                out[ym] = roc_auc_score(g["label"], g[proba_col])
            except Exception:
                out[ym] = np.nan
    return pd.Series(out)


def rolling_auc(rec: pd.DataFrame, proba_col: str, n: int) -> pd.Series:
    vals = {}
    lab = rec["label"].values
    p = rec[proba_col].values
    for i in range(n - 1, len(rec)):
        ys = lab[i - n + 1:i + 1]
        ps = p[i - n + 1:i + 1]
        if len(set(ys)) < 2 or np.isnan(ps).all():
            vals[rec.at[i, "timestamp"]] = np.nan
        else:
            try:
                vals[rec.at[i, "timestamp"]] = roc_auc_score(ys, ps)
            except Exception:
                vals[rec.at[i, "timestamp"]] = np.nan
    return pd.Series(vals)


def build_report(rec: pd.DataFrame, has_prod: bool) -> str:
    tracks = ["nofilter", "prod", "frozen", "daily"] if has_prod else ["nofilter", "frozen", "daily"]
    labels = {"nofilter": "NO-FILTER", "prod": "PROD-FROZEN", "frozen": "WARMUP-FROZEN", "daily": "DAILY-RETRAINED"}
    stats = {t: track_stats(rec, t) for t in tracks}

    L = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L += [
        f"YANK (Tier2) Daily-Retrain Walk-Forward / Model-Drift Test  ({ts})",
        "=" * 78,
        f"Rolling window: {TRAIN_WINDOW_DAYS}d | val tail: {VAL_TAIL_DAYS}d | features: {len(FEATURE_COLS)}",
        f"Test signals: {len(rec)} | {rec['timestamp'].min().date()} → {rec['timestamp'].max().date()}",
        f"Frozen threshold: {rec.attrs['frozen_thr']:.2f}"
        + (f" | Prod threshold: {rec.attrs['prod_thr']:.2f}" if has_prod else ""),
        "",
        "── Aggregate over all test days ─────────────────────────────────────────────",
        f"{'Track':<16} {'N':>5} {'%Taken':>7} {'WR':>7} {'PF':>7} {'Total $':>10} {'IR':>7} {'MaxDD $':>10}",
        "-" * 78,
    ]
    for t in tracks:
        s = stats[t]
        L.append(f"{labels[t]:<16} {s['n']:>5} {s['pct_taken']:>6.1%} {s['wr']:>6.1%} "
                 f"{s['pf']:>7.3f} {s['pnl']:>10,.0f} {s['ir']:>7.3f} {s['maxdd']:>10,.0f}")

    # ---- Drift: monthly PF + AUC per model track ----
    L += ["", "── Per-month PF (filtered) ──────────────────────────────────────────────────",
          f"{'Month':<9}" + "".join(f"{labels[t][:9]:>11}" for t in tracks if t != "nofilter")
          + f"{'Unfil PF':>11}"]
    months = sorted(rec["year_month"].unique())
    proba_map = {"prod": "p_prod", "frozen": "p_frozen", "daily": "p_daily"}
    for ym in months:
        g = rec[rec["year_month"] == ym]
        line = f"{ym:<9}"
        for t in tracks:
            if t == "nofilter":
                continue
            gm = track_kept_mask(rec, t)[rec["year_month"].values == ym]
            pnl = g["pnl"].values[gm]
            line += f"{profit_factor(pnl):>11.3f}"
        line += f"{profit_factor(g['pnl'].values):>11.3f}"
        L.append(line)

    L += ["", "── Per-month AUC (discrimination; <0.50 = worse than random) ─────────────────",
          f"{'Month':<9}" + "".join(f"{labels[t][:9]:>11}" for t in tracks if t != "nofilter")]
    auc_series = {t: monthly_auc(rec, proba_map[t]) for t in tracks if t != "nofilter"}
    for ym in months:
        line = f"{ym:<9}"
        for t in tracks:
            if t == "nofilter":
                continue
            v = auc_series[t].get(ym, np.nan)
            line += f"{v:>11.3f}" if not np.isnan(v) else f"{'—':>11}"
        L.append(line)

    # ---- Drift verdict ----
    L += ["", "── Drift verdict ────────────────────────────────────────────────────────────"]
    # AUC trend: first-half vs second-half of test period, per track
    half = len(rec) // 2
    for t in tracks:
        if t == "nofilter":
            continue
        col = proba_map[t]
        a1 = _safe_auc(rec.iloc[:half], col)
        a2 = _safe_auc(rec.iloc[half:], col)
        tag = "DRIFT↓" if (not np.isnan(a1) and not np.isnan(a2) and a2 < a1 - 0.03) else "stable"
        L.append(f"  {labels[t]:<16} AUC 1st-half={_fmt(a1)}  2nd-half={_fmt(a2)}  → {tag}")

    pf_frozen = stats["frozen"]["pf"]
    pf_daily = stats["daily"]["pf"]
    pnl_gain = stats["daily"]["pnl"] - stats["frozen"]["pnl"]
    L += [
        "",
        f"  Daily-retrain vs warmup-frozen: PF {pf_frozen:.3f} → {pf_daily:.3f} "
        f"({pf_daily - pf_frozen:+.3f}) | PnL {pnl_gain:+,.0f}",
    ]
    if has_prod:
        pf_prod = stats["prod"]["pf"]
        L.append(f"  Deployed PROD model PF over test period: {pf_prod:.3f} "
                 f"(vs daily-retrain {pf_daily:.3f})")
    return "\n".join(L)


def _safe_auc(g: pd.DataFrame, col: str) -> float:
    if g["label"].nunique() < 2 or g[col].isna().all():
        return float("nan")
    try:
        return roc_auc_score(g["label"], g[col])
    except Exception:
        return float("nan")


def _fmt(v: float) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "—"


def make_plot(rec: pd.DataFrame, has_prod: bool, out_png: Path) -> None:
    tracks = ["nofilter", "prod", "frozen", "daily"] if has_prod else ["nofilter", "frozen", "daily"]
    labels = {"nofilter": "No-filter", "prod": "Prod-frozen", "frozen": "Warmup-frozen", "daily": "Daily-retrained"}
    colors = {"nofilter": "#888888", "prod": "#d62728", "frozen": "#1f77b4", "daily": "#2ca02c"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for t in tracks:
        m = track_kept_mask(rec, t)
        cum = rec["pnl"].where(m, 0.0).cumsum()
        ax1.plot(rec["timestamp"], cum, label=f"{labels[t]} (${cum.iloc[-1]:,.0f})",
                 color=colors[t], lw=1.6)
    ax1.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax1.set_ylabel("Cumulative PnL ($)")
    ax1.set_title("YANK Tier2 daily-retrain drift test — cumulative PnL by track")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    proba_map = {"prod": "p_prod", "frozen": "p_frozen", "daily": "p_daily"}
    for t in tracks:
        if t == "nofilter":
            continue
        ra = rolling_auc(rec, proba_map[t], ROLLING_AUC_N)
        if len(ra):
            ax2.plot(ra.index, ra.values, label=labels[t], color=colors[t], lw=1.4)
    ax2.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5, label="random (0.50)")
    ax2.set_ylabel(f"Rolling AUC (trailing {ROLLING_AUC_N})")
    ax2.set_xlabel("Date")
    ax2.set_title("Model discrimination over time (drift signal)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    import argparse
    global TRAIN_WINDOW_DAYS, VAL_TAIL_DAYS, MIN_TRAIN_SAMPLES, EXPANDING, FEATURES_PATH, HISTORY_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=TRAIN_WINDOW_DAYS, help="Rolling training window / warmup length (calendar days)")
    ap.add_argument("--val-tail", type=int, default=None, help="Val-tail window for threshold search (days); default = window/3")
    ap.add_argument("--min-train", type=int, default=None, help="Min signals required in window to fit")
    ap.add_argument("--expanding", action="store_true", help="Train on ALL prior data each day (expanding window) instead of rolling")
    ap.add_argument("--features", type=str, default=str(FEATURES_PATH), help="Feature CSV path")
    ap.add_argument("--history", type=str, default=str(HISTORY_PATH), help="History CSV path")
    ap.add_argument("--tag", type=str, default="", help="Extra suffix for output filenames (e.g. 'bearish')")
    args = ap.parse_args()
    FEATURES_PATH = Path(args.features)
    HISTORY_PATH = Path(args.history)
    TRAIN_WINDOW_DAYS = args.window
    VAL_TAIL_DAYS = args.val_tail if args.val_tail is not None else max(21, TRAIN_WINDOW_DAYS // 3)
    if args.min_train is not None:
        MIN_TRAIN_SAMPLES = args.min_train
    EXPANDING = args.expanding

    df = load_data()
    prod_model, prod_thr = load_prod_model()
    print()
    rec = run_walkforward(df, prod_model, prod_thr)
    if rec.empty:
        print("No test predictions produced.")
        return
    has_prod = prod_model is not None

    report = build_report(rec, has_prod)
    print("\n" + report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    tag = f"{date_str}_" + ("expanding" if EXPANDING else f"w{TRAIN_WINDOW_DAYS}")
    if args.tag:
        tag += f"_{args.tag}"
    txt = REPORT_DIR / f"tier2_wf_daily_{tag}.txt"
    csv = REPORT_DIR / f"tier2_wf_daily_trajectory_{tag}.csv"
    png = REPORT_DIR / f"tier2_wf_daily_{tag}.png"
    txt.write_text(report + "\n")
    rec.to_csv(csv, index=False)
    make_plot(rec, has_prod, png)
    print(f"\nReport     → {txt}")
    print(f"Trajectory → {csv}")
    print(f"Plot       → {png}")


if __name__ == "__main__":
    main()
