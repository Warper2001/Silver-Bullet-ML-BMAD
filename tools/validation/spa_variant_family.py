"""SPA / StepM multiple-testing correction over a family of strategy variants.

WHY THIS EXISTS
---------------
This shop's documented failure mode is selection bias: run many variants, keep
the best-looking one, watch it die out-of-sample. The naive question asked of a
variant family is "is the best one significantly profitable?" -- but that
question is asked *after* looking at all of them, so its p-value is wrong.

Hansen's Superior Predictive Ability (SPA) test answers the right question:
    H0: no variant in the family beats the benchmark,
given that you searched the whole family. Romano-Wolf StepM then returns the
*set* of variants that survive at a family-wise error rate, rather than one
verdict.

Both come from `arch.bootstrap` (Kevin Sheppard). arch's StepM is the literal
Romano & Wolf (2005) stepdown -- confirmed from its source docstring citation
(Econometrica 73(4), 1237-1282).

CONVENTION
----------
arch's SPA/StepM take *losses*, and test whether any model has LOWER expected
loss than the benchmark. Daily P&L is a gain, so loss = -pnl. The benchmark is
cash (zero P&L every day), i.e. a zero loss series.

USAGE
-----
    .venv-research/bin/python tools/validation/spa_variant_family.py
    .venv-research/bin/python tools/validation/spa_variant_family.py \
        --glob "data/ml_training/doe_run_0*_history.csv" --reps 10000

Runs on the research venv (/root/Silver-Bullet-ML-BMAD/.venv-research), NOT the
live venv -- the live combine bots run on .venv and its dependency set is frozen.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from arch.bootstrap import SPA, StepM
    from statsmodels.stats.multitest import multipletests
except ImportError as e:  # pragma: no cover
    sys.exit(f"ERROR: missing library -- {e}. Use the research venv.")


DEFAULT_GLOB = "data/ml_training/doe_run_0*_history.csv"
# run_08_fullyear covers a different window (full 2025) than the rest
# (2025-08-01 onward); mixing windows would make the family incomparable.
EXCLUDE = ("fullyear",)


def variant_name(path: str) -> str:
    m = re.search(r"(doe_run_\d+[a-z_]*)_history", Path(path).name)
    return m.group(1) if m else Path(path).stem


def load_daily_pnl(paths: list[str]) -> pd.DataFrame:
    """Aligned daily P&L matrix: rows = calendar days, cols = variants.

    A day on which a variant did not trade earns 0 -- that is the correct
    return for a flat strategy, not a missing observation.
    """
    series = {}
    for p in paths:
        df = pd.read_csv(p)
        if "pnl" not in df.columns or "timestamp" not in df.columns:
            print(f"  skip {p}: needs 'pnl' and 'timestamp' columns")
            continue
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        daily = pd.Series(df["pnl"].to_numpy(), index=ts.dt.date).groupby(level=0).sum()
        series[variant_name(p)] = daily
    if not series:
        sys.exit("ERROR: no usable variant files matched.")
    mat = pd.DataFrame(series).sort_index()
    # Restrict to the window every variant actually covers, then fill
    # non-trading days with 0 inside that window.
    starts = [s.index.min() for s in series.values()]
    ends = [s.index.max() for s in series.values()]
    lo, hi = max(starts), min(ends)
    mat = mat.loc[(mat.index >= lo) & (mat.index <= hi)]
    full = pd.date_range(lo, hi, freq="B").date
    mat = mat.reindex(full).fillna(0.0)
    return mat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--block-size", type=int, default=5, help="block bootstrap length (5 = a trading week)")
    ap.add_argument("--size", type=float, default=0.05, help="StepM FWER")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    paths = sorted(p for p in globmod.glob(args.glob) if not any(x in p for x in EXCLUDE))
    print(f"Variant family: {len(paths)} files matching {args.glob!r}")
    for p in paths:
        print(f"  - {p}")
    if len(paths) < 2:
        sys.exit("ERROR: SPA needs a family (>= 2 variants).")

    mat = load_daily_pnl(paths)
    T, k = mat.shape
    print(f"\nCommon window: {mat.index[0]} -> {mat.index[-1]}  ({T} business days, {k} variants)")

    # ---- Per-variant descriptive + NAIVE (uncorrected) significance ---------
    rows = []
    for c in mat.columns:
        x = mat[c].to_numpy(dtype=float)
        t, p_two = stats.ttest_1samp(x, 0.0)
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2  # one-sided: mean > 0
        rows.append({
            "variant": c,
            "total_pnl": x.sum(),
            "mean_daily": x.mean(),
            "sd_daily": x.std(ddof=1),
            "sharpe_ann": (x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else np.nan,
            "t_stat": t,
            "p_naive_1s": p_one,
        })
    summ = pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)

    print("\n=== Per-variant, NAIVE (uncorrected, one-sided mean > 0) ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    naive_hits = summ.loc[summ["p_naive_1s"] < 0.05, "variant"].tolist()
    best = summ.iloc[0]["variant"]
    print(f"\nBest by total P&L: {best}")
    print(f"Naively 'significant' at 5% (UNCORRECTED, WRONG): {len(naive_hits)} of {k} -> {naive_hits}")

    # ---- Multiple-testing corrections on the naive p-values -----------------
    pvals = summ["p_naive_1s"].to_numpy()
    holm = multipletests(pvals, alpha=0.05, method="holm")
    bh = multipletests(pvals, alpha=0.05, method="fdr_bh")
    summ["p_holm"] = holm[1]
    summ["p_bh_fdr"] = bh[1]
    print("\n=== After multiple-testing correction (statsmodels.multipletests) ===")
    print(summ[["variant", "p_naive_1s", "p_holm", "p_bh_fdr"]].to_string(
        index=False, float_format=lambda v: f"{v:10.4f}"))
    print(f"Holm (FWER 5%) survivors:    {summ.loc[holm[0], 'variant'].tolist()}")
    print(f"BH-FDR (5%) survivors:       {summ.loc[bh[0], 'variant'].tolist()}")

    # ---- SPA: does ANY variant beat cash, given we searched all of them? ----
    # arch takes LOSSES. loss = -pnl. Benchmark = cash = zero P&L = zero loss.
    losses = (-mat).to_numpy(dtype=float)
    benchmark = np.zeros(T, dtype=float)

    rs = np.random.RandomState(args.seed)
    spa = SPA(benchmark, losses, block_size=args.block_size, reps=args.reps, seed=rs)
    spa.compute()
    print("\n=== Hansen SPA (H0: no variant beats cash, correcting for the search) ===")
    print(f"  block_size={args.block_size}  reps={args.reps}  seed={args.seed}")
    print(f"  SPA p-values -> lower: {spa.pvalues['lower']:.4f}   "
          f"consistent: {spa.pvalues['consistent']:.4f}   upper: {spa.pvalues['upper']:.4f}")
    spa_p = float(spa.pvalues["consistent"])
    print("  (use the CONSISTENT p-value; 'lower'/'upper' bound it)")
    verdict_spa = "REJECT H0 -- at least one variant genuinely beats cash" if spa_p < 0.05 \
        else "FAIL TO REJECT -- the family's best result is consistent with luck"
    print(f"  Verdict at 5%: {verdict_spa}")

    # ---- StepM: WHICH variants survive at FWER 5%? -------------------------
    rs2 = np.random.RandomState(args.seed)
    stepm = StepM(benchmark, losses, size=args.size, block_size=args.block_size,
                  reps=args.reps, seed=rs2)
    stepm.compute()
    superior_idx = stepm.superior_models
    cols = list(mat.columns)
    superior = [cols[i] if isinstance(i, (int, np.integer)) else str(i) for i in superior_idx]
    print(f"\n=== Romano-Wolf StepM (FWER {args.size:.0%}) ===")
    print(f"  Superior variants: {superior if superior else 'NONE'}")

    # ---- Bottom line -------------------------------------------------------
    print("\n" + "=" * 72)
    print("BOTTOM LINE")
    print("=" * 72)
    print(f"  Variants searched:                {k}")
    print(f"  'Significant' if you ignore that: {len(naive_hits)}")
    print(f"  Survive Holm (FWER 5%):           {int(holm[0].sum())}")
    print(f"  Survive BH-FDR (5%):              {int(bh[0].sum())}")
    print(f"  Survive Romano-Wolf StepM:        {len(superior)}")
    print(f"  Hansen SPA consistent p-value:    {spa_p:.4f}  -> {verdict_spa.split(' -- ')[0]}")

    if args.json_out:
        out = {
            "window": [str(mat.index[0]), str(mat.index[-1])],
            "n_days": int(T), "n_variants": int(k),
            "reps": args.reps, "block_size": args.block_size, "seed": args.seed,
            "per_variant": summ.to_dict(orient="records"),
            "naive_significant": naive_hits,
            "holm_survivors": summ.loc[holm[0], "variant"].tolist(),
            "bh_survivors": summ.loc[bh[0], "variant"].tolist(),
            "stepm_superior": superior,
            "spa_pvalues": {kk: float(vv) for kk, vv in spa.pvalues.items()},
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
