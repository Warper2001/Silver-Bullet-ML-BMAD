#!/usr/bin/env python3
"""ETFTM-1 Gate 0 — the one true-alignment run sealed in
_bmad-output/preregistration_etftm1_time_series_momentum.md (seal c3365a9).

Refuses to run unless: the A1 verdict allows it (POWERED or MARGINALLY_POWERED; Alex chose "Go, Gate 0
first"), the A1 script and its inputs are byte-identical to what the verdict recorded, and the seal
commit exists. Reuses build_matrices() from tools/etftm1_power_gate.py unchanged. Development data only.

Run: .venv-research/bin/python tools/etftm1_gate0.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
GATE = ROOT / "tools/etftm1_power_gate.py"
VERDICT = ROOT / "_bmad-output/etftm1_power_verdict.json"
OUT_JSON = ROOT / "_bmad-output/etftm1_gate0_results.json"
SEAL = "c3365a9"
NULL_DRAWS = 1000
NULL_SEED = 20260926
Z_CRIT = 1.645
NULL_PCT = 95.0
LOO_MIN_POSITIVE = 7          # of 8 asset classes

sys.path.insert(0, str(ROOT))
from tools.etftm1_universe import ASSET_CLASS  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preflight() -> dict:
    v = json.loads(VERDICT.read_text())
    if v["verdict"] not in ("POWERED", "MARGINALLY_POWERED"):
        raise SystemExit(f"A1 verdict {v['verdict']} does not allow Gate 0")
    if sha(GATE) != v["script_sha256"]:
        raise SystemExit("tools/etftm1_power_gate.py changed since the A1 verdict")
    for rel, h in v["inputs_sha256"].items():
        if sha(ROOT / rel) != h:
            raise SystemExit(f"input changed since A1: {rel}")
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{SEAL}^{{commit}}"]).returncode:
        raise SystemExit(f"seal commit {SEAL} not found")
    return v


def portfolio_mean(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    prod = X * Y
    cnt = np.sum(np.isfinite(prod), axis=1)
    p = np.where(cnt > 0, np.nansum(prod, axis=1) / np.maximum(cnt, 1), np.nan)
    return float(np.nanmean(p)), p


def main() -> int:
    v = preflight()
    spec = importlib.util.spec_from_file_location("gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    X, Y, months, syms, info = gate.build_matrices()
    T = len(months)
    se = v["se_used"]

    m_hat, p = portfolio_mean(X, Y)                                      # THE one true-alignment statistic
    sd_p = float(np.nanstd(p, ddof=1))
    z = m_hat / se
    cond_i = z >= Z_CRIT

    rng = np.random.default_rng(NULL_SEED)
    null = np.empty(NULL_DRAWS)
    for b in range(NULL_DRAWS):
        Xs = np.column_stack([np.roll(X[:, j], int(rng.integers(13, T - 12))) for j in range(X.shape[1])])
        null[b] = portfolio_mean(Xs, Y)[0]
    pct = float((null < m_hat).mean() * 100)
    cond_ii = pct >= NULL_PCT

    classes = sorted(set(ASSET_CLASS[s] for s in syms))
    loo = {}
    for c in classes:
        keep = [j for j, s in enumerate(syms) if ASSET_CLASS[s] != c]
        loo[c] = portfolio_mean(X[:, keep], Y[:, keep])[0]
    n_pos = sum(val > 0 for val in loo.values())
    cond_iii = n_pos >= LOO_MIN_POSITIVE

    verdict = "PASS" if (cond_i and cond_ii and cond_iii) else "FAIL"

    by_class = {c: portfolio_mean(X[:, [j for j, s in enumerate(syms) if ASSET_CLASS[s] == c]],
                                  Y[:, [j for j, s in enumerate(syms) if ASSET_CLASS[s] == c]])[0] for c in classes}
    yrs = np.array([m.year for m in months])
    sub = lambda mask: float(np.nanmean(p[mask])) if mask.any() else None  # noqa: E731
    res = {"seal": SEAL, "script_sha256": sha(Path(__file__)), "a1_verdict": v["verdict"],
           "T_months": T, "m_hat": m_hat, "sd_p": sd_p, "ir_annual": m_hat / sd_p * np.sqrt(12),
           "se_A1": se, "z": z, "p_one_sided": float(1 - norm.cdf(z)), "cond_i_significance": bool(cond_i),
           "null_percentile": pct, "null_p95": float(np.percentile(null, 95)), "null_mean": float(null.mean()),
           "cond_ii_random_null": bool(cond_ii),
           "leave_one_class_out": loo, "loo_positive": n_pos, "cond_iii_loo": bool(cond_iii),
           "VERDICT": verdict,
           "descriptive": {"by_class_m": by_class,
                           "pre_2012_m": sub(yrs < 2012), "post_2012_m": sub(yrs >= 2012),
                           "ex_2008_m": sub(yrs != 2008), "months_pre_2012": int((yrs < 2012).sum()),
                           "months_post_2012": int((yrs >= 2012).sum())}}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
