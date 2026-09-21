"""POWER GATE — one-shot confirmatory test of the sealed MIM-NB engine (S=250, live config) on MNQ front-month 2021-2024.

OUTCOME-BLIND. The sealed engine is NEVER run on the target window. This program reads the target file only to COUNT
sessions (bars at 09:31 and 16:00 present) — no signal, no band, no P&L. Effect and dispersion come from the 2025 DEV window
(in-sample by construction: it is the window the spec was gated on) and from the live ledger; both are inputs, not results
on the target. Target: data/mim_x/mnq_1min_2021_2024_frontmonth.csv (NOT data/sealed_holdout/).

WINDOW HYGIENE. Unseen = 2021-2024 minus the months the noise-bands Gate-0 script already ran as non-gating diagnostics
(study_mim_noise_bands_gate0.py: 2023 Sep-Nov, 2024 Sep-Nov), minus the 14-session sigma warm-up, minus the first session
after each quarterly roll (the prior close would come from another contract; 4/yr x 4 yrs = 16, a conservative count).

INPUTS (anchors, all in per-trade standardised units d = mean/SD of net trade return, net of the sealed 1.12 pt cost):
  ceiling : 2025 dev, S=250: +14.11 bp/trade, SD 122.8 bp  => d = 0.115   (in-sample, upward-biased: best case)
  report  : third-party noise-area report full-period +2.6 bp/trade, at the 2025 SD  => d = 0.021
  live    : MIM-NB ledger N=29: +0.31 bp/trade, SD 75.9 bp                     => d = 0.004
  trade rate per session: 0.509 (2025 dev: 114 trades / 224 sessions, a count) and 0.414 (live: 29 trades / 70 sessions)
  shape: the 2025 dev trade distribution (standardised), to keep the fat tail (13 cat-stops of -250 pts, 5 days carry the
         edge) instead of assuming normality.

PRE-COMMITTED DECISION RULE (written before any number below was produced):
  Test: one-sided one-sample t-test that mean net return per trade > 0 on the unseen window, alpha = 0.05.
  Power: simulated, 20,000 draws, trades drawn iid from the 2025 standardised shape shifted to the anchor d
         (DEFF = 1.0 = the MOST favourable setting; a DEFF of 1.5 for day/regime clustering is reported as a sensitivity).
  N_avail_low / N_avail_high = unseen sessions x (0.414 | 0.509).
  POWERED       if simulated power >= 0.80 at the CEILING d with N_avail_low
  UNDERPOWERED  if simulated power <  0.80 at the CEILING d with N_avail_high   (=> underpowered at every lower d, rate, and DEFF)
  UNDETERMINED  otherwise
  (0.80 power and alpha 0.05 are the repo's convention in the earlier gates; the leash here is the data itself, not a year count.)

Run: .venv/bin/python _bmad-output/diagnostics_mim_nb_2021_2024_power_gate_20260921/power_gate.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/root/Silver-Bullet-ML-BMAD")
TARGET = REPO / "data/mim_x/mnq_1min_2021_2024_frontmonth.csv"
DEV_TRADES = REPO / "_bmad-output/diagnostics_mim_nb_2025_rerun_20260921/trades_A_frozen_S250.csv"
DEV_BARS = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
OUT = Path(__file__).parent
assert "sealed_holdout" not in str(TARGET)

COST_PTS = 1.12
WARMUP = 14
ROLL_SESSIONS = 16
SEEN_MONTHS = [("2023-09-01", "2023-11-30"), ("2024-09-01", "2024-11-30")]
RATE_HIGH, RATE_LOW = 114 / 224, 29 / 70
ALPHA, POWER = 0.05, 0.80
Z = stats.norm.ppf(1 - ALPHA) + stats.norm.ppf(POWER)
N_SIM = 20000
rng = np.random.default_rng(20260921)


def count_sessions() -> tuple[int, int, int]:
    d = pd.read_csv(TARGET, usecols=["timestamp"])
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
    day = ts.dt.normalize().dt.tz_localize(None)
    hm = ts.dt.strftime("%H:%M")
    have = pd.DataFrame({"day": day, "open": hm == "09:31", "close": hm == "16:00"}).groupby("day").any()
    sess = have.index[have["open"] & have["close"]]
    seen = pd.Series(False, index=sess)
    for lo, hi in SEEN_MONTHS:
        seen |= (sess >= lo) & (sess <= hi)
    return len(sess), int(seen.sum()), int((~seen).sum())


def dev_shape() -> tuple[np.ndarray, float, float]:
    """2025 dev net trade returns in bp of that day's 09:31 open (same convention as the 2025 re-run)."""
    t = pd.read_csv(DEV_TRADES)
    b = pd.read_csv(DEV_BARS, usecols=["timestamp", "open"])
    ts = pd.to_datetime(b["timestamp"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    b = b[ts.dt.strftime("%H:%M") == "09:31"]
    opens = pd.Series(b["open"].to_numpy(), index=ts[b.index].dt.date.astype(str))
    opens = opens[~opens.index.duplicated()]
    bp = (t["pnl_pts"] - COST_PTS) / t["day"].astype(str).map(opens) * 1e4
    x = bp.to_numpy()
    return (x - x.mean()) / x.std(ddof=1), float(x.mean()), float(x.std(ddof=1))


def sim_power(z: np.ndarray, d: float, n: int) -> float:
    if n < 3:
        return 0.0
    draws = rng.choice(z, size=(N_SIM, n)) + d
    m, s = draws.mean(axis=1), draws.std(axis=1, ddof=1)
    t = m / (s / math.sqrt(n))
    return float(np.mean(t > stats.t.ppf(1 - ALPHA, n - 1)))


def n_for_power(z: np.ndarray, d: float, target: float = POWER, cap: int = 20000) -> int | None:
    lo, hi = 5, cap
    if sim_power(z, d, hi) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if sim_power(z, d, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    total, seen_n, unseen = count_sessions()
    usable = unseen - WARMUP - ROLL_SESSIONS
    n_low, n_high = usable * RATE_LOW, usable * RATE_HIGH
    z, dev_mean_bp, dev_sd_bp = dev_shape()
    d_ceiling = dev_mean_bp / dev_sd_bp
    anchors = {"ceiling (2025 dev, in-sample)": d_ceiling, "x0.75": 0.75 * d_ceiling, "x0.50": 0.5 * d_ceiling,
               "x0.25": 0.25 * d_ceiling, "third-party report (+2.6 bp)": 2.6 / dev_sd_bp, "live ledger (+0.31 bp)": 0.31 / 75.9}
    kurt = float(stats.kurtosis(z, fisher=False)); skew = float(stats.skew(z))

    rows = []
    for name, d in anchors.items():
        normal_n = (Z / d) ** 2
        n_sim = n_for_power(z, d)
        rows.append({"anchor": name, "d": d, "mean_bp": d * dev_sd_bp,
                     "N_req_normal_deff1": normal_n, "N_req_normal_deff1.5": 1.5 * normal_n,
                     "N_req_sim_deff1": n_sim,
                     "power_at_N_low": sim_power(z, d, int(n_low)), "power_at_N_high": sim_power(z, d, int(n_high)),
                     "years_to_accrue_sim_at_rate_high_252d": (n_sim / (RATE_HIGH * 252)) if n_sim else None})

    ceil = rows[0]
    if ceil["power_at_N_low"] >= POWER:
        verdict = "POWERED"
    elif ceil["power_at_N_high"] < POWER:
        verdict = "UNDERPOWERED"
    else:
        verdict = "UNDETERMINED"

    mde = {k: {"N": int(n), "MDE_d_normal": Z / math.sqrt(n), "MDE_bp_normal": Z / math.sqrt(n) * dev_sd_bp,
               "MDE_over_ceiling": (Z / math.sqrt(n)) / d_ceiling}
           for k, n in (("N_low", n_low), ("N_high", n_high))}

    result = {"verdict": verdict,
              "verdict_rule": "UNDERPOWERED if simulated power < 0.80 at the ceiling d with N_avail_high, DEFF 1.0",
              "window": {"sessions_with_open_and_close": total, "seen_months_sessions_excluded": seen_n,
                         "unseen_sessions": unseen, "usable_after_warmup_and_rolls": usable,
                         "N_avail_low": n_low, "N_avail_high": n_high, "rate_low": RATE_LOW, "rate_high": RATE_HIGH},
              "dev_shape": {"mean_bp": dev_mean_bp, "sd_bp": dev_sd_bp, "skew": skew, "kurtosis": kurt, "n": int(len(z))},
              "grid": rows, "mde": mde, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "power_verdict.json").write_text(json.dumps(result, indent=2, default=float))

    L = [f"VERDICT: {verdict}   (rule: {result['verdict_rule']})", "",
         f"window: {total} sessions with 09:31+16:00; {seen_n} in seen months (2023/24 Sep-Nov) excluded -> {unseen} unseen; "
         f"usable {usable} after {WARMUP}-session warm-up and {ROLL_SESSIONS} roll sessions",
         f"expected trades: N_low={n_low:.0f} (rate {RATE_LOW:.3f}) .. N_high={n_high:.0f} (rate {RATE_HIGH:.3f})",
         f"2025 dev shape: mean {dev_mean_bp:+.2f} bp, SD {dev_sd_bp:.1f} bp, skew {skew:.2f}, kurtosis {kurt:.1f}, n={len(z)}", "",
         "anchor                          d      mean_bp | N_req normal(DEFF1) (DEFF1.5) | N_req simulated | power @N_low  @N_high | years@0.509/d"]
    for r in rows:
        ns = r["N_req_sim_deff1"]
        L.append(f"{r['anchor']:<30} {r['d']:.4f} {r['mean_bp']:7.2f} | {r['N_req_normal_deff1']:>9.0f} {r['N_req_normal_deff1.5']:>9.0f} | "
                 f"{('%d' % ns) if ns else '>20000':>15} | {r['power_at_N_low']:.2f} {r['power_at_N_high']:.2f} | "
                 f"{('%.1f' % r['years_to_accrue_sim_at_rate_high_252d']) if ns else 'n/a'}")
    L += ["", "MDE at the expected N (normal approx., 80% power, one-sided 5%):"]
    for k, m in mde.items():
        L.append(f"  {k}: N={m['N']}  d={m['MDE_d_normal']:.3f} ({m['MDE_bp_normal']:.1f} bp) = {m['MDE_over_ceiling']:.2f}x the 2025 in-sample effect")
    (OUT / "power_gate_output.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
