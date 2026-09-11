"""Measure how much measured model performance is manufactured by the CV scheme.

WHY THIS EXISTS
---------------
A cross-validation scheme that lets information cross the train/test boundary
reports performance that does not exist out of sample. On financial time series
the two usual channels are:

  (a) random splitting of serially-correlated rows -- a test row's temporal
      near-twin sits in the training set;
  (b) overlapping label horizons -- a training sample's outcome window extends
      into the test window (fixed by *purging*, plus an *embargo* for residual
      serial correlation).

This tool quantifies (a) and (b) by scoring the SAME data and model under
several CV schemes and comparing.

CALIBRATION FIRST -- THE POSITIVE CONTROL
-----------------------------------------
A leak detector that reports "no leak" is worthless until you show it can find
one. `--mode control` builds a dataset where the true predictability is exactly
zero by construction: a driftless random walk, causal features built only from
past bars, and labels equal to the sign of the FORWARD h-bar return. Forward
returns of a random walk are unpredictable, so an honest CV scheme must score
AUC = 0.5. Anything above 0.5 is leakage and nothing else. Labels overlap
whenever h > 1, so both channels are live.

Then `--mode real` applies the calibrated instrument to an actual dataset.

SCHEMES COMPARED
----------------
  random_split       train_test_split(shuffle=True, stratify=y)
                     -- as used at src/ml/retraining.py:997 and
                        scripts/train_premium_regime_models.py:78
  stratified_kfold   cross_val_score(cv=5) -> StratifiedKFold
                     -- as used at scripts/tune_regime_*.py
  timeseries_split   TimeSeriesSplit(5)
                     -- as used at src/ml/train_tier2_meta_labeling.py:78
  purged_walkforward TimeSeriesSplit + purge of overlapping label spans + embargo
                     -- the leak-free reference

USAGE
-----
    .venv-research/bin/python tools/validation/cv_leakage_probe.py --mode control
    .venv-research/bin/python tools/validation/cv_leakage_probe.py --mode real \
        --csv data/ml_training/s23_meta_labels_2025.csv

Runs on the research venv, NOT the live venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:  # pragma: no cover
    sys.exit(f"ERROR: missing library -- {e}. Use the research venv.")


# ---------------------------------------------------------------- models ----
def make_model(kind: str):
    if kind == "logreg":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    if kind == "forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      class_weight="balanced", random_state=0, n_jobs=-1)
    raise ValueError(kind)


def safe_auc(y_true, score) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)


def fit_score(model_kind, Xtr, ytr, Xte, yte) -> float:
    if len(np.unique(ytr)) < 2:
        return np.nan
    m = make_model(model_kind)
    m.fit(Xtr, ytr)
    return safe_auc(yte, m.predict_proba(Xte)[:, 1])


# ------------------------------------------------------------- CV schemes ---
def auc_random_split(X, y, model_kind, repeats, rng) -> float:
    out = []
    for r in range(repeats):
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=int(rng.integers(1 << 31)), stratify=y)
        except ValueError:
            continue
        out.append(fit_score(model_kind, Xtr, ytr, Xte, yte))
    return float(np.nanmean(out)) if out else np.nan


def auc_stratified_kfold(X, y, model_kind, n_splits=5) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    out = [fit_score(model_kind, X[tr], y[tr], X[te], y[te]) for tr, te in cv.split(X, y)]
    return float(np.nanmean(out))


MIN_TRAIN = 20  # shared by timeseries_split and purged_walkforward so that the
                # two average over the SAME folds; otherwise a fold dropped by
                # one and kept by the other shows up as a fake purging effect.


def auc_timeseries_split(X, y, model_kind, n_splits=5) -> float:
    cv = TimeSeriesSplit(n_splits=n_splits)
    out = [fit_score(model_kind, X[tr], y[tr], X[te], y[te])
           for tr, te in cv.split(X) if len(tr) >= MIN_TRAIN]
    return float(np.nanmean(out)) if out else np.nan


def auc_purged_walkforward(X, y, model_kind, label_end, n_splits=5, embargo=0) -> float:
    """TimeSeriesSplit, then drop training rows whose label window reaches the
    test block (purge), plus `embargo` rows before it.

    `label_end[i]` is the integer index at which sample i's outcome is known.
    Folds are skipped on the same MIN_TRAIN rule as auc_timeseries_split, on the
    UNPURGED training set, so both functions score an identical fold set and any
    difference between them is purging alone.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for tr, te in cv.split(X):
        if len(tr) < MIN_TRAIN:
            continue
        t0 = te.min()
        keep = tr[label_end[tr] < (t0 - embargo)]
        if len(keep) == 0:
            continue
        out.append(fit_score(model_kind, X[keep], y[keep], X[te], y[te]))
    return float(np.nanmean(out)) if out else np.nan


# ------------------------------------------------------- data constructors --
def make_control(n, h, n_feat_windows=(5, 10, 20, 60), seed=0):
    """Random walk; causal past-only features; label = sign of FORWARD h-bar
    return. True predictability is exactly zero.
    """
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n + h + 100) * 0.01
    price = 100 * np.exp(np.cumsum(ret))
    s = pd.Series(price)

    feats = {}
    for w in n_feat_windows:
        feats[f"mom_{w}"] = s.pct_change(w)
        feats[f"vol_{w}"] = s.pct_change().rolling(w).std()
        feats[f"z_{w}"] = (s - s.rolling(w).mean()) / s.rolling(w).std()
    F = pd.DataFrame(feats)

    fwd = s.shift(-h) / s - 1.0          # forward h-bar return
    lab = (fwd > 0).astype(int)

    df = F.copy()
    df["y"] = lab
    df = df.iloc[100:100 + n]            # drop warm-up
    df = df.dropna()
    X = df.drop(columns="y").to_numpy(float)
    y = df["y"].to_numpy(int)
    # sample i's outcome is known h bars later
    label_end = np.arange(len(y)) + h
    return X, y, label_end


def load_real(csv, feature_cols=None):
    df = pd.read_csv(csv)
    if "entry_ts" not in df or "exit_ts" not in df or "label" not in df:
        sys.exit("ERROR: real mode needs entry_ts, exit_ts, label columns.")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True)
    df = df.sort_values("entry_ts").reset_index(drop=True)

    drop = {"entry_ts", "exit_ts", "label", "exit_type", "pnl_1x",
            "entry_price", "sl", "tp"}
    cols = feature_cols or [c for c in df.columns if c not in drop]
    X = df[cols].to_numpy(float)
    y = df["label"].to_numpy(int)

    # label_end[i] = index of the last trade whose ENTRY precedes trade i's exit,
    # i.e. how far forward sample i's outcome window reaches in sample space.
    # searchsorted returns the first entry >= exits[i]; minus 1 is the last trade
    # that had already started. A trade with no overlap gives label_end == i.
    entries = df["entry_ts"].to_numpy()
    exits = df["exit_ts"].to_numpy()
    label_end = np.searchsorted(entries, exits, side="left") - 1
    idx = np.arange(len(df))
    label_end = np.maximum(label_end, idx)
    n_overlap = int((label_end > idx).sum())
    print(f"  features: {cols}")
    print(f"  n={len(df)}  positives={int(y.sum())} ({y.mean():.1%})")
    print(f"  samples whose outcome window reaches a LATER trade's entry: "
          f"{n_overlap} ({n_overlap / len(df):.1%})")
    if n_overlap == 0:
        print("  NOTE: zero overlap. This strategy holds one position at a time, so")
        print("  every trade closes before the next opens. Purging has nothing to")
        print("  remove here -- purged_walkforward reduces to TimeSeriesSplit, and any")
        print("  inflation the other schemes show is feature-side serial correlation,")
        print("  not overlapping label horizons.")
    return X, y, label_end


# --------------------------------------------------------------------- run --
def run(X, y, label_end, model_kinds, repeats, embargo, rng):
    rows = []
    for mk in model_kinds:
        rows.append({
            "model": mk,
            "random_split": auc_random_split(X, y, mk, repeats, rng),
            "stratified_kfold": auc_stratified_kfold(X, y, mk),
            "timeseries_split": auc_timeseries_split(X, y, mk),
            "purged_walkforward": auc_purged_walkforward(X, y, mk, label_end, embargo=embargo),
        })
    return pd.DataFrame(rows)


def report(tab: pd.DataFrame, truth: float | None, title: str):
    print(f"\n=== {title} ===")
    print("mean ROC AUC by CV scheme")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    ref = "purged_walkforward"
    print(f"\ninflation vs {ref} (AUC points):")
    for _, r in tab.iterrows():
        base = r[ref]
        deltas = "  ".join(
            f"{c}={r[c] - base:+.4f}" for c in
            ["random_split", "stratified_kfold", "timeseries_split"])
        print(f"  {r['model']:8s}  {deltas}")
    if truth is not None:
        print(f"\ntrue AUC by construction = {truth:.4f}; "
              f"any excess is manufactured by the CV scheme")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["control", "real"], default="control")
    ap.add_argument("--csv", default="data/ml_training/s23_meta_labels_2025.csv")
    ap.add_argument("--n", type=int, default=1500, help="control: sample count")
    ap.add_argument("--paths", type=int, default=15,
                    help="control: independent random-walk paths to average over")
    ap.add_argument("--horizon", type=int, default=20, help="control: label horizon in bars")
    ap.add_argument("--repeats", type=int, default=20, help="random_split repeats")
    ap.add_argument("--embargo", type=int, default=0)
    ap.add_argument("--models", default="logreg,forest")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model_kinds = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.mode == "control":
        print("POSITIVE CONTROL -- driftless random walk, causal features, "
              f"label = sign of forward {args.horizon}-bar return.")
        print("True predictability is ZERO by construction; honest CV must return AUC = 0.5.")
        print(f"Averaging over {args.paths} INDEPENDENT paths -- a single path has realised")
        print("drift, so AUC=0.5 holds only in expectation across paths.\n")
        tabs = []
        for p in range(args.paths):
            X, y, label_end = make_control(args.n, args.horizon, seed=args.seed + 1000 * p)
            tabs.append(run(X, y, label_end, model_kinds, args.repeats, args.embargo, rng))
            print(f"  path {p + 1}/{args.paths} done", flush=True)
        cat = pd.concat(tabs)
        tab = cat.groupby("model", as_index=False).mean(numeric_only=True)
        sem = cat.groupby("model", as_index=False).sem(numeric_only=True)
        report(tab, 0.5, f"POSITIVE CONTROL (n={args.n}, h={args.horizon}, "
                         f"{args.paths} paths)")
        print("\nstandard error across paths:")
        print(sem.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        worst = (tab["random_split"] - tab["purged_walkforward"]).max()
        print(f"\nDETECTOR CALIBRATION: max random-split inflation = {worst:+.4f} AUC.")
        print("  If this is comfortably positive, the probe can detect leakage.")
    else:
        print(f"REAL DATA -- {args.csv}")
        X, y, label_end = load_real(args.csv)
        tab = run(X, y, label_end, model_kinds, args.repeats, args.embargo, rng)
        report(tab, None, f"REAL: {Path(args.csv).name}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "mode": args.mode, "seed": args.seed,
            "table": tab.to_dict(orient="records")}, indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
