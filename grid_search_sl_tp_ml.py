#!/usr/bin/env python3
"""3-D grid search over (SL multiplier, TP multiplier, ML threshold) for YANK (Tier2).

Discovery-only: searches on 2025 with an expanding walk-forward (every signal is
scored out-of-sample by a model trained solely on prior data). All of 2026 is left
untouched as a sealed holdout for the post-pre-registration OOS gate.

The SL/TP <-> ML coupling is handled correctly: each (SL,TP) cell regenerates its
own trades + win/loss labels at those exit levels, then retrains the meta-model on
those labels — no stale calibration.

Bearish-only, MNQ, run-08 base (ATR 0.1, baseline session) — matches the live config.

Outputs:
  - data/reports/grid_sl_tp_ml_<YYYYMMDD>.csv   (one row per cell)
  - data/reports/grid_sl_tp_ml_<YYYYMMDD>.txt   (ranked table + recommendation)
  - data/reports/grid_sl_tp_ml_<YYYYMMDD>.png   (PF heatmaps)
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest_tier2_wf_adaptive import profit_factor, win_rate, max_drawdown, per_trade_sharpe, make_model

FEATURE_COLS = [
    "atr", "gap_size", "volume_ratio", "et_hour", "day_of_week", "signal_direction",
    "session_displacement", "adr_pct_used", "fvg_to_sweep_bars", "prior_setup_proximity",
    "h1_trend_slope", "sin_hour", "cos_hour", "session_volume_ratio", "fvg_fill_pct",
    "bar_body_ratio", "sweep_window_vol", "slope_direction_match",
]

PYTHON   = ".venv/bin/python"
BACKTEST = "src/research/backtest_zero_bias_optimized.py"
DATA_2025 = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
REPORT_DIR = Path("data/reports")

# Discovery window (2025); 2026 reserved as sealed holdout.
DISCOVERY_START = "2025-01-01"
DISCOVERY_END   = "2025-12-31"
WARMUP_DAYS     = 150   # train at least ~5 months before first OOS month
MIN_TRAIN       = 50    # min signals to fit a fold
MIN_OOS_TRADES  = 30    # a cell needs this many OOS trades to be recommendable

ATR_THRESHOLD = 0.1
SESSION       = "baseline"


def regen_features(sl: float, tp: float, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate bearish-only features+history at (sl, tp) over the discovery window."""
    feat_p = cache / f"sl{sl}_tp{tp}_features.csv"
    hist_p = cache / f"sl{sl}_tp{tp}_history.csv"
    if not (feat_p.exists() and hist_p.exists()):
        cmd = [
            PYTHON, BACKTEST, "--export",
            "--export-path", str(feat_p), "--history", str(hist_p),
            "--data", DATA_2025,
            "--sl-mult", str(sl), "--tp-mult", str(tp),
            "--atr-threshold", str(ATR_THRESHOLD), "--session-windows", SESSION,
            "--start", DISCOVERY_START, "--end", DISCOVERY_END,
        ]  # bearish-only is the default
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not feat_p.exists():
            print(f"  [err] regen sl{sl}/tp{tp} failed: {r.stderr[-300:]}")
            return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(feat_p), pd.read_csv(hist_p)


def oos_probas(feat: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Expanding walk-forward: pooled out-of-sample probabilities per signal."""
    df = feat[FEATURE_COLS + ["label"]].copy()
    df["pnl"] = hist["pnl"].values
    df["timestamp"] = pd.to_datetime(hist["timestamp"].values)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["year_month"] = df["timestamp"].dt.to_period("M")

    start = df["date"].min()
    first_test = start + pd.Timedelta(days=WARMUP_DAYS)
    test_months = sorted(df.loc[df["date"] >= first_test, "year_month"].unique())

    X = df[FEATURE_COLS].values
    y = df["label"].values
    rows = []
    for ym in test_months:
        cut = ym.start_time
        tr = (df["timestamp"] < cut).values          # strictly prior → no look-ahead
        te = (df["year_month"] == ym).values
        if tr.sum() < MIN_TRAIN or len(set(y[tr])) < 2 or te.sum() == 0:
            continue
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        sub = df.loc[te, ["pnl", "label"]].copy()
        sub["proba"] = p
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["pnl", "label", "proba"])


def sweep(oos: pd.DataFrame, sl: float, tp: float, thresholds: list[float]) -> list[dict]:
    out = []
    for thr in thresholds:
        mask = np.ones(len(oos), dtype=bool) if thr <= 0 else (oos["proba"].values >= thr)
        pnl = oos["pnl"].values[mask]
        out.append({
            "sl": sl, "tp": tp, "threshold": thr,
            "n_oos": int(mask.sum()),
            "wr": win_rate(pnl),
            "pf": profit_factor(pnl),
            "pnl": float(pnl.sum()),
            "ir": per_trade_sharpe(pnl),
            "maxdd": max_drawdown(pnl),
        })
    return out


def make_heatmaps(res: pd.DataFrame, out_png: Path) -> None:
    sls = sorted(res["sl"].unique())
    tps = sorted(res["tp"].unique())
    # best PF per (sl,tp) across thresholds (subject to min trades)
    best_pf = np.full((len(sls), len(tps)), np.nan)
    best_thr = np.full((len(sls), len(tps)), np.nan)
    for i, s in enumerate(sls):
        for j, t in enumerate(tps):
            cell = res[(res.sl == s) & (res.tp == t) & (res.n_oos >= MIN_OOS_TRADES)]
            if len(cell):
                r = cell.loc[cell["pf"].idxmax()]
                best_pf[i, j] = r["pf"]
                best_thr[i, j] = r["threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, fmt in [
        (ax1, best_pf, "Best OOS PF per SL×TP", "{:.2f}"),
        (ax2, best_thr, "ML threshold at best PF", "{:.2f}"),
    ]:
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto",
                       vmin=(0.8 if title.startswith("Best") else 0.0),
                       vmax=(1.6 if title.startswith("Best") else 0.65))
        ax.set_xticks(range(len(tps))); ax.set_xticklabels([f"TP {t}" for t in tps])
        ax.set_yticks(range(len(sls))); ax.set_yticklabels([f"SL {s}" for s in sls])
        ax.set_title(title)
        for i in range(len(sls)):
            for j in range(len(tps)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("YANK bearish grid search (2025 discovery, OOS walk-forward)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--tp", type=float, nargs="+", default=[4.0, 5.0, 6.0, 8.0])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.45, 0.50, 0.55, 0.60])
    args = ap.parse_args()

    cache = Path(__file__).parent / ".grid_cache"
    cache.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    n_cells = len(args.sl) * len(args.tp)
    k = 0
    for sl in args.sl:
        for tp in args.tp:
            k += 1
            print(f"[{k}/{n_cells}] SL {sl} / TP {tp} ...", flush=True)
            feat, hist = regen_features(sl, tp, cache)
            if feat.empty:
                continue
            oos = oos_probas(feat, hist)
            if oos.empty:
                print(f"   no OOS signals (n_total={len(feat)})")
                continue
            rows = sweep(oos, sl, tp, args.thresholds)
            all_rows.extend(rows)
            base = next(r for r in rows if r["threshold"] <= 0)
            print(f"   OOS n={base['n_oos']} | no-ML PF={base['pf']:.3f} ${base['pnl']:,.0f}")

    if not all_rows:
        sys.exit("No results produced.")
    res = pd.DataFrame(all_rows)
    date_str = datetime.now().strftime("%Y%m%d")
    csv = REPORT_DIR / f"grid_sl_tp_ml_{date_str}.csv"
    res.to_csv(csv, index=False)

    # Ranking: recommendable cells need >= MIN_OOS_TRADES; rank by PF then PnL.
    rec = res[res["n_oos"] >= MIN_OOS_TRADES].copy()
    rec = rec.sort_values(["pf", "pnl"], ascending=False)

    lines = [
        f"YANK bearish 3-D grid search (SL × TP × ML threshold)  {datetime.now():%Y-%m-%d %H:%M}",
        "=" * 78,
        f"Discovery: {DISCOVERY_START}..{DISCOVERY_END} (2025) | 2026 reserved as sealed holdout",
        f"Eval: expanding walk-forward, pooled OOS | warmup {WARMUP_DAYS}d | min OOS trades {MIN_OOS_TRADES}",
        f"Grid: SL {args.sl} × TP {args.tp} × thr {args.thresholds}  ({len(res)} cells)",
        "",
        "── Top 15 by OOS PF (>= min trades) ─────────────────────────────────────────",
        f"{'SL':>4} {'TP':>4} {'thr':>5} {'n':>4} {'WR':>6} {'PF':>7} {'PnL$':>9} {'IR':>7} {'MaxDD$':>9}",
        "-" * 70,
    ]
    for _, r in rec.head(15).iterrows():
        lines.append(f"{r.sl:>4.1f} {r.tp:>4.1f} {r.threshold:>5.2f} {int(r.n_oos):>4} "
                     f"{r.wr:>6.1%} {r.pf:>7.3f} {r.pnl:>9,.0f} {r.ir:>7.3f} {r.maxdd:>9,.0f}")

    # Current live config row for reference (SL2/TP8 @ ~0.50)
    cur = res[(res.sl == 2.0) & (res.tp == 8.0)]
    lines += ["", "── Current live config (SL 2.0 / TP 8.0) across thresholds ──────────────────",
              f"{'thr':>5} {'n':>4} {'WR':>6} {'PF':>7} {'PnL$':>9}"]
    for _, r in cur.sort_values("threshold").iterrows():
        lines.append(f"{r.threshold:>5.2f} {int(r.n_oos):>4} {r.wr:>6.1%} {r.pf:>7.3f} {r.pnl:>9,.0f}")

    if len(rec):
        b = rec.iloc[0]
        lines += ["", "── Candidate (discovery winner — NOT yet validated) ─────────────────────────",
                  f"  SL {b.sl} / TP {b.tp} / ML threshold {b.threshold:.2f}",
                  f"  OOS: n={int(b.n_oos)}  WR={b.wr:.1%}  PF={b.pf:.3f}  PnL=${b.pnl:,.0f}  MaxDD=${b.maxdd:,.0f}",
                  "  Multiple-comparison caveat: 80-cell grid → in-sample optimistic.",
                  "  REQUIRED before live: pre-register this set, then validate on the",
                  "  reserved 2026 holdout via oos_checkpoint.py."]

    report = "\n".join(lines)
    print("\n" + report)
    txt = REPORT_DIR / f"grid_sl_tp_ml_{date_str}.txt"
    txt.write_text(report + "\n")
    png = REPORT_DIR / f"grid_sl_tp_ml_{date_str}.png"
    make_heatmaps(res, png)
    print(f"\nCSV → {csv}\nReport → {txt}\nHeatmaps → {png}")


if __name__ == "__main__":
    main()
