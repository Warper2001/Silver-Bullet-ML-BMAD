#!/usr/bin/env python3
"""ETFTM-1 A1 power gate — implements _bmad-output/preregistration_etftm1_power_gate.md (seal 3845bf6).

OUTCOME-BLIND: the timing portfolio p_t = mean_i x~_{i,t} * y~_{i,t+1} is computed ONLY under misaligned
pairings of signal and outcome (circular time shifts of the signal matrix; stationary block bootstrap
of outcome months). `pairing_stats` raises on the true alignment. Reads development data only.

Run: .venv-research/bin/python tools/etftm1_power_gate.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
PANEL = ROOT / "data/etf_daily/panel_dev.csv"
RF = ROOT / "data/etf_daily/rf_dev.csv"
MANIFEST = ROOT / "_bmad-output/etftm1_manifest_20260925.json"
OUT = ROOT / "_bmad-output/etftm1_power_verdict.json"
SEAL = "3845bf6"

COM_DAYS = 60            # MOP (2012) §2.4: centre of mass 60 days, annualisation factor 261
ANN = 261
MIN_HISTORY_DAYS = 252
BLOCK_MEAN = 12
B_DRAWS = 10_000
SEED = 20260925
MAX_TRUE_ALIGNED_FRAC = 0.10
Z_A, Z_B = 1.645, 0.842  # one-sided alpha 0.05, 80% power
ANCHORS = {"optimistic_MOP_decayed": 0.72, "central_DBMF_decade": 0.39, "pessimistic_SGCTA_decade": 0.07}
HOLDOUT_MONTHS = 60      # 2021-10 .. 2026-09 from the manifest span; the holdout file is not opened


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_inputs() -> dict:
    man = json.loads(MANIFEST.read_text())["outputs"]
    got = {str(p.relative_to(ROOT)): sha(p) for p in (PANEL, RF)}
    for k, v in got.items():
        if man.get(k) != v:
            raise SystemExit(f"input hash mismatch for {k}")
    return got


def build_matrices() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str], dict]:
    panel = pd.read_csv(PANEL, parse_dates=["date"])
    assert panel["date"].max() < pd.Timestamp("2021-10-01"), "development data only"
    tr = panel.pivot(index="date", columns="sym", values="tr_ret").sort_index()
    rf = pd.read_csv(RF, parse_dates=["date"]).set_index("date")["rf_daily"]
    rf = rf.reindex(tr.index).ffill()
    ex = tr.sub(rf, axis=0)

    m = ex.ewm(com=COM_DAYS, adjust=False, ignore_na=True).mean()
    var = ((ex - m) ** 2).ewm(com=COM_DAYS, adjust=False, ignore_na=True).mean() * ANN
    sigma = np.sqrt(var)
    nobs = ex.notna().cumsum()

    month = ex.index.to_period("M")
    month_end = ex.groupby(month).apply(lambda g: g.index.max())
    mret = (1 + ex).groupby(month).prod(min_count=1) - 1
    full = ex.notna().groupby(month).sum() == ex.groupby(month).size().values[:, None]
    mret = mret.where(full)                                   # partial months are not returns
    sig_me = sigma.loc[month_end.values].set_axis(mret.index)
    n_me = nobs.loc[month_end.values].set_axis(mret.index)

    trail12 = (1 + mret).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1
    x = np.sign(trail12)
    y = mret.shift(-1) / (sig_me / np.sqrt(12))               # row t holds the month t+1 outcome
    elig = x.notna() & y.notna() & sig_me.gt(0) & n_me.ge(MIN_HISTORY_DAYS) & x.ne(0)
    x, y = x.where(elig), y.where(elig)
    xt = x - x.mean()                                         # asset fixed effects
    yt = y - y.mean()
    keep = elig.any(axis=1)
    xt, yt = xt[keep], yt[keep]
    info = {"months": int(keep.sum()), "first_month": str(xt.index.min()), "last_month": str(xt.index.max()),
            "assets": list(xt.columns), "eligible_obs": int(elig.values.sum()),
            "assets_per_month_median": float(elig[keep].sum(axis=1).median()),
            "assets_per_month_min": int(elig[keep].sum(axis=1).min())}
    return xt.to_numpy(), yt.to_numpy(), xt.index, list(xt.columns), info


def pairing_stats(X: np.ndarray, Y: np.ndarray, aligned_rows: np.ndarray) -> tuple[float, float]:
    """mean and sd of the equal-weight timing portfolio under a pairing. Refuses the true alignment."""
    if np.all(aligned_rows):
        raise RuntimeError("firewall: true alignment requested")
    prod = X * Y
    cnt = np.sum(np.isfinite(prod), axis=1)
    p = np.where(cnt > 0, np.nansum(prod, axis=1) / np.maximum(cnt, 1), np.nan)
    p = p[np.isfinite(p)]
    return float(p.mean()), float(p.std(ddof=1))


def stationary_index(T: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(T, dtype=int)
    i = rng.integers(T)
    for t in range(T):
        idx[t] = i
        i = rng.integers(T) if rng.random() < 1.0 / BLOCK_MEAN else (i + 1) % T
    return idx


def participation_ratio(M: np.ndarray) -> float | None:
    rows = np.all(np.isfinite(M), axis=1)
    if rows.sum() < M.shape[1] + 2:
        return None
    lam = np.clip(np.linalg.eigvalsh(np.corrcoef(M[rows].T)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def main() -> int:
    hashes = check_inputs()
    X, Y, months, syms, info = build_matrices()
    T = len(months)
    ar = np.arange(T)

    shift_stats = [pairing_stats(np.roll(X, s, axis=0), Y, np.roll(ar, s) == ar) for s in range(13, T - 12)]
    rng = np.random.default_rng(SEED)
    boot_stats, discarded = [], 0
    while len(boot_stats) < B_DRAWS:
        idx = stationary_index(T, rng)
        if np.mean(idx == ar) > MAX_TRUE_ALIGNED_FRAC:
            discarded += 1
            continue
        boot_stats.append(pairing_stats(X, Y[idx], idx == ar))

    se_shift = float(np.std([m for m, _ in shift_stats], ddof=1))
    se_boot = float(np.std([m for m, _ in boot_stats], ddof=1))
    se = max(se_shift, se_boot)
    sd_p = float(np.mean([s for _, s in shift_stats + boot_stats]))
    scale = np.sqrt(12) * se / sd_p                          # IR units per 1 SE
    mde_ir = (Z_A + Z_B) * scale
    power = {k: float(norm.cdf(a / scale - Z_A)) for k, a in ANCHORS.items()}
    hold_scale = scale * np.sqrt(T / HOLDOUT_MONTHS)
    power_hold = {k: float(norm.cdf(a / hold_scale - Z_A)) for k, a in ANCHORS.items()}

    if power["central_DBMF_decade"] >= 0.80:
        verdict = "POWERED"
    elif power["optimistic_MOP_decayed"] >= 0.80:
        verdict = "MARGINALLY_POWERED"
    else:
        verdict = "UNDERPOWERED"

    res = {"seal": SEAL, "script_sha256": sha(Path(__file__)), "inputs_sha256": hashes, "sample": info,
           "T_months": T, "T_years": round(T / 12, 2),
           "n_shift_pairings": len(shift_stats), "n_boot_draws": len(boot_stats), "boot_discarded": discarded,
           "se_mean_p_shift": se_shift, "se_mean_p_boot": se_boot, "se_used": se, "sd_p": sd_p,
           "iid_se_equivalent": sd_p / np.sqrt(T), "se_inflation_vs_iid": se / (sd_p / np.sqrt(T)),
           "mde_ir_80pct": float(mde_ir), "power_dev": power,
           "holdout_mde_ir_80pct": float((Z_A + Z_B) * hold_scale), "power_holdout_alone": power_hold,
           "effective_breadth_outcomes": participation_ratio(Y), "effective_breadth_signals": participation_ratio(X),
           "years_needed_portfolio_80pct": {f"{ir:.1f}": round((Z_A + Z_B) ** 2 / ir ** 2, 1)
                                            for ir in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)},
           "anchors": ANCHORS, "verdict": verdict}
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("inputs_sha256",)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
