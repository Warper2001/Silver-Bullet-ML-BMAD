"""XSMOM-1 POWER GATE — runs first, and structurally cannot see the answer.

Pre-registration: _bmad-output/preregistration_xsmom1_cross_sectional_momentum.md
(sealed commit 1ff3735, before this script was written).

FIREWALL (seal §4.1). This program computes the cross-sectional IC *only under
mismatched pairings* -- block-bootstrap resampled week indices, or circular
shifts with |s| > k. It asserts on every draw that the identity pairing is never
evaluated. The aligned signal->return statistic never exists as a value here.
Everything the gate needs (sd(IC_t), SE, valid-draw count, span) is obtainable
from mismatched pairings alone.

Emits power_verdict.json carrying input + script SHA-256. The evaluation script
refuses to run unless that file exists, hashes match, and the verdict permits.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/Silver-Bullet-ML-BMAD")
PILOT = REPO / "data/commodity_curve/coverage-pilot-2025-20260906-v3"
BARS = PILOT / "contract_bars.csv"
CONTRACTS = PILOT / "contracts.csv"
OUT = Path(__file__).resolve().parents[1] / "_bmad-output" / "power_verdict.json"

# --- everything below is frozen by the seal -------------------------------
K_WEEKS = 8                    # seal §3, not swept
MIN_DAYS_TO_EXPIRY = 7         # front-contract rule, seal §2
BLOCK_MEAN_LEN = 4             # stationary bootstrap, seal §4.2
B_DRAWS = 10_000
SEED = 20260907
ALPHA = 0.05                   # one-sided, seal §3
Z_ALPHA = stats.norm.ppf(1 - ALPHA)      # 1.645
Z_POWER = stats.norm.ppf(0.80)           # 0.842
THETA_PLAUS = 0.0243           # seal §4.3 -- gross SR 0.50. MAY NOT BE REVISED.
THETA_PESSIMISTIC = 0.0121
THETA_OPTIMISTIC = 0.0364
MIN_VALID_DRAWS = 39           # seal §4.4 P2
SR_PER_IC = 20.6               # measured conversion, seal §4.3


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_weekly_panel(k: int) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Returns (signal_ranks, forward_ranks, roots, week_index).

    Both are (T_k x N) row-standardised rank matrices. They are built here but
    are NEVER paired index-to-index anywhere in this program.
    """
    bars = pd.read_csv(
        BARS, usecols=["canonical_root", "contract_code", "TimeStamp", "Close", "OpenInterest"]
    )
    meta = pd.read_csv(CONTRACTS, usecols=["Symbol", "ExpirationDate"])
    exp = dict(zip(meta["Symbol"], pd.to_datetime(meta["ExpirationDate"]).dt.tz_localize(None)))

    bars["date"] = pd.to_datetime(bars["TimeStamp"]).dt.tz_localize(None).dt.normalize()
    bars["expiry"] = bars["contract_code"].map(exp)
    bars = bars.dropna(subset=["expiry", "Close", "OpenInterest"])
    bars["dte"] = (bars["expiry"] - bars["date"]).dt.days
    bars = bars[bars["dte"] > MIN_DAYS_TO_EXPIRY]

    # front contract = highest open interest among eligible, per root per day
    bars = bars.sort_values(["canonical_root", "date", "OpenInterest"])
    front = bars.groupby(["canonical_root", "date"], as_index=False).last()

    # same-contract log returns only; roll days contribute nothing (seal §2)
    front = front.sort_values(["canonical_root", "date"])
    front["prev_close"] = front.groupby("canonical_root")["Close"].shift(1)
    front["prev_code"] = front.groupby("canonical_root")["contract_code"].shift(1)
    same = front["contract_code"] == front["prev_code"]
    front["logret"] = np.where(
        same & front["prev_close"].gt(0) & front["Close"].gt(0),
        np.log(front["Close"] / front["prev_close"]),
        0.0,
    )

    # weekly (Friday-ending) sums, roots as columns
    front["week"] = front["date"].dt.to_period("W-FRI")
    wk = front.pivot_table(index="week", columns="canonical_root", values="logret", aggfunc="sum")
    wk = wk.dropna(axis=0, how="any")           # complete cross-sections only
    wk = wk.iloc[1:-1]                          # drop partial head/tail weeks

    roots = list(wk.columns)
    R = wk.to_numpy()                            # (T x N) weekly log returns
    T = R.shape[0]

    # trailing k-week cumulative return, known at the close of week t-1
    sig = np.full_like(R, np.nan)
    for t in range(k, T):
        sig[t] = R[t - k : t].sum(axis=0)
    valid = ~np.isnan(sig).any(axis=1)
    S_raw, F_raw = sig[valid], R[valid]
    weeks = wk.index[valid]

    def row_rank_standardise(M: np.ndarray) -> np.ndarray:
        out = np.empty_like(M, dtype=float)
        for i, row in enumerate(M):
            r = stats.rankdata(row)
            r = r - r.mean()
            out[i] = r / np.linalg.norm(r)
        return out

    return row_rank_standardise(S_raw), row_rank_standardise(F_raw), roots, weeks


def ic_under_pairing(S: np.ndarray, F: np.ndarray, pairing: np.ndarray) -> np.ndarray:
    """IC_t for a MISMATCHED pairing only. Hard-refuses the identity pairing."""
    T = S.shape[0]
    identity = np.arange(T)
    if pairing.shape == identity.shape and np.array_equal(pairing, identity):
        raise AssertionError(
            "FIREWALL VIOLATION: identity pairing requested. The power gate must "
            "never evaluate the aligned signal->return statistic (seal §4.1)."
        )
    # row-wise dot product of standardised ranks == Spearman IC
    return np.einsum("ij,ij->i", S, F[pairing])


def main() -> None:
    S, F, roots, weeks = build_weekly_panel(K_WEEKS)
    T, N = S.shape
    span_years = (weeks[-1].end_time - weeks[0].start_time).days / 365.25
    print(f"XSMOM-1 POWER GATE  (k={K_WEEKS} weeks, frozen by seal)")
    print(f"  roots: {len(roots)}  {roots}")
    print(f"  evaluable weeks T = {T}   span = {span_years:.2f} years")

    rng = np.random.default_rng(SEED)

    # --- Cross-check C: circular shifts, valid range only (seal §4.2) -------
    valid_shifts = [s for s in range(K_WEEKS + 1, T - K_WEEKS)]
    shift_ics = []
    for s in valid_shifts:
        assert abs(s) > K_WEEKS, "shift <= k re-creates signal/return overlap"
        shift_ics.append(ic_under_pairing(S, F, (np.arange(T) + s) % T))
    sd_shift = float(np.std(np.concatenate(shift_ics), ddof=1)) if shift_ics else float("nan")
    se_C = sd_shift / np.sqrt(T) if shift_ics else float("nan")

    # --- Cross-check A: iid weeks. sd(IC_t) from mismatched draws ----------
    sd_ic = float(np.std(np.concatenate(shift_ics), ddof=1))
    se_A = sd_ic / np.sqrt(T)

    # --- PRIMARY: stationary block bootstrap of return cross-sections ------
    p_geom = 1.0 / BLOCK_MEAN_LEN
    boot_means = np.empty(B_DRAWS)
    for b in range(B_DRAWS):
        idx = np.empty(T, dtype=int)
        i = 0
        while i < T:
            start = rng.integers(0, T)
            blen = min(rng.geometric(p_geom), T - i)
            idx[i : i + blen] = (start + np.arange(blen)) % T
            i += blen
        if np.array_equal(idx, np.arange(T)):      # astronomically unlikely; refuse anyway
            continue
        boot_means[b] = ic_under_pairing(S, F, idx).mean()
    se_B = float(np.std(boot_means, ddof=1))

    mde80 = (Z_ALPHA + Z_POWER) * se_B
    power = lambda theta: float(stats.norm.cdf(theta / se_B - Z_ALPHA))
    pi = power(THETA_PLAUS)

    print(f"\n  sd(IC_t) [mismatched]      = {sd_ic:.4f}")
    print(f"  SE_A iid-weeks             = {se_A:.4f}")
    print(f"  SE_B block bootstrap (PRIMARY) = {se_B:.4f}")
    print(f"  SE_C circular shift        = {se_C:.4f}   ({len(valid_shifts)} valid shifts)")
    print(f"\n  MDE80 (one-sided) = {mde80:.4f} mean IC  =>  gross SR {mde80*SR_PER_IC:.2f}")
    print(f"  Theta_plaus       = {THETA_PLAUS:.4f} mean IC  =>  gross SR "
          f"{THETA_PLAUS*SR_PER_IC:.2f}")
    print(f"  achieved power pi = {pi:.1%}")

    print("\n  power across the declared range:")
    for lbl, th in (("pessimistic 0.25", THETA_PESSIMISTIC),
                    ("central     0.50", THETA_PLAUS),
                    ("optimistic  0.75", THETA_OPTIMISTIC),
                    ("SR 1.00 (beyond lit)", 1.00 / SR_PER_IC),
                    ("SR 2.00 (absurd)", 2.00 / SR_PER_IC)):
        print(f"    {lbl:<22} power = {power(th):5.1%}")

    # --- hard stops (seal §4.4) -------------------------------------------
    p2_ok = len(valid_shifts) >= MIN_VALID_DRAWS
    inflation = se_B / se_A if se_A > 0 else float("nan")
    span_floor = ((Z_ALPHA + Z_POWER) * inflation / (THETA_OPTIMISTIC * SR_PER_IC)) ** 2
    p4_ok = span_years >= span_floor

    if pi >= 0.80:
        verdict = "POWERED"
    elif pi >= 0.50:
        verdict = "MARGINALLY_POWERED"
    else:
        verdict = "UNDERPOWERED"
    if not p2_ok:
        verdict = "UNDERPOWERED"

    print(f"\n  [P2] valid null draws {len(valid_shifts)} >= {MIN_VALID_DRAWS}: "
          f"{'PASS' if p2_ok else 'FAIL'}")
    print(f"  [P4] span {span_years:.2f}y >= floor {span_floor:.1f}y: "
          f"{'PASS' if p4_ok else 'FAIL'}")
    print(f"\n  >>> VERDICT: {verdict} <<<")
    if verdict == "UNDERPOWERED":
        print("      TERMINAL. No strategy returns are computed; the 2025 panel")
        print("      remains unspent for a future adequately-powered seal.")

    # span required for 80% power at each declared effect
    req = {f"SR_{sr:.2f}": round(((Z_ALPHA + Z_POWER) * inflation / sr) ** 2, 1)
           for sr in (0.30, 0.40, 0.50, 0.60, 0.75, 1.00)}
    print(f"\n  years of data needed for 80% power: {req}")

    OUT.write_text(json.dumps({
        "seal": "XSMOM-1",
        "k_weeks": K_WEEKS, "T_weeks": T, "n_roots": N, "roots": roots,
        "span_years": round(span_years, 3),
        "sd_ic": round(sd_ic, 5),
        "se_A_iid": round(se_A, 5), "se_B_bootstrap": round(se_B, 5),
        "se_C_shift": round(se_C, 5), "n_valid_shifts": len(valid_shifts),
        "mde80_mean_ic": round(mde80, 5), "mde80_gross_sr": round(mde80 * SR_PER_IC, 3),
        "theta_plaus": THETA_PLAUS, "achieved_power": round(pi, 4),
        "P2_randomisation_ok": bool(p2_ok), "P4_span_ok": bool(p4_ok),
        "span_floor_years": round(span_floor, 2),
        "years_for_80pct_power": req,
        "verdict": verdict,
        "seed": SEED, "n_bootstrap": B_DRAWS,
        "input_sha256": sha256(BARS), "script_sha256": sha256(Path(__file__).resolve()),
        "firewall": "IC computed only under mismatched pairings; identity pairing refused",
    }, indent=2))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
