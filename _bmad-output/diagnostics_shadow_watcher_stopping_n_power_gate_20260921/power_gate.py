"""POWER GATE — stopping N for the YANK bullish shadow watcher (preregistration_yank_bidirectional_m15_choch.md, Amendment 3).

OUTCOME-BLIND. The shadow ledger (logs/yank_shadow_bullish_trades.csv) is NEVER opened: Amendment 4 §4.7 forbids using it as evidence, and a
stopping N does not need it. Inputs come only from (a) the Amendment 4 corrected-bar derivation-window trade list
(_bmad-output/bidir_frontmonth_rerun_20260920/corrected_bidir_trades.csv; that window is spent on this hypothesis, already seen,
2025-01-01..2026-02-28), and (b) two accrual RATES: the derivation window's own bullish rate, and the shadow watcher's observed rate quoted in
the pre-registration text ("4 completed shadow trades to 2026-09-16", deployed 2026-08-19) — a count taken from the document, not from the file.
`data/sealed_holdout/` is not touched.

WHAT A STOPPING N IS FOR. The watcher accrues unseen bullish shadow trades so that the bullish leg can eventually be judged on new data (Amendment 3).
The decision it feeds is the pre-registration's own bar: G2 bullish PF > 1.3 (§5). The read-out must therefore be able to SEE the mean per trade that
a PF of 1.3 implies. That effect, not the derivation window's noisy point estimate, is the primary anchor.

PRE-COMMITTED DECISION RULE (written before any number below was produced):
  theta_bar   = mean net $/trade implied by PF = 1.3 on the observed bullish trade-outcome distribution, holding gross loss per trade fixed:
                theta_bar = (1.3 - 1) x (gross loss / N)            [derived from §5's bar and the corrected-bar trades; nothing hand-set]
  SD          = SD of the 19 corrected-bar bullish trades (held fixed; sensitivity x0.75 / x1.25)
  d_bar       = theta_bar / SD
  N_stop      = ((z_alpha + z_power) / d_bar)^2 x DEFF, alpha 0.05 one-sided, power 0.80; DEFF 1.0 headline (trades are rare and mostly on different days),
                1.5 shown as a sensitivity
  years       = N_stop / (rate per week x 52), at TWO rates:  r_low = 19 bullish trades / 424 days of the derivation window  (backtest rate)
                                                                r_high = 4 shadow trades / 28 days (2026-08-19 .. 2026-09-16)  (live-watcher rate; 4 trades, so it is noisy)
  POWERED             if years <= 2.0 at BOTH rates
  POWER_UNDETERMINED  if years <= 2.0 at r_high only  (the rate must be measured; N is then fixed in advance, the calendar is not)
  UNDERPOWERED        if years  > 2.0 at BOTH rates   (=> no stopping N is reachable within the leash; the watcher cannot answer the PF-1.3 question)
  (2.0y is the operator-leash convention of the 09-20/09-21 gates, not a derived quantity; the full grid is printed so another leash can be read off.)
  The derivation window's own point estimate (mean $27.18, PF 1.105) is reported as a SECOND anchor, never as the verdict basis: that window was mined
  for three hypotheses (Amendment 2) and N=19 cannot resolve it.

Run: .venv/bin/python _bmad-output/diagnostics_shadow_watcher_stopping_n_power_gate_20260921/power_gate.py
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
TRADES = REPO / "_bmad-output/bidir_frontmonth_rerun_20260920/corrected_bidir_trades.csv"
OUT = Path(__file__).parent
assert "sealed_holdout" not in str(TRADES) and "shadow" not in TRADES.name

Z = stats.norm.ppf(0.95) + stats.norm.ppf(0.80)
BAR_PF = 1.3
LEASH_YEARS = 2.0
WINDOW_DAYS = (pd.Timestamp("2026-02-28") - pd.Timestamp("2025-01-01")).days + 1      # 424
R_LOW = 19 / WINDOW_DAYS * 7                                                          # per week
R_HIGH = 4 / ((pd.Timestamp("2026-09-16") - pd.Timestamp("2026-08-19")).days) * 7     # per week (28 days)
N_SIM, SEED = 20000, 20260921


def n_req(d: float, deff: float = 1.0) -> float:
    return math.inf if d <= 0 else deff * (Z / d) ** 2


def sim_power(z: np.ndarray, d: float, n: int, rng) -> float:
    draws = rng.choice(z, size=(N_SIM, n)) + d
    t = draws.mean(1) / (draws.std(1, ddof=1) / math.sqrt(n))
    return float(np.mean(t > stats.t.ppf(0.95, n - 1)))


def main() -> None:
    d = pd.read_csv(TRADES)
    bull, bear = d[d.direction == "BULLISH"].pnl_usd.to_numpy(), d[d.direction == "BEARISH"].pnl_usd.to_numpy()
    # reproduce the amendment's documented numbers before using anything
    pf = bull[bull > 0].sum() / -bull[bull < 0].sum()
    assert len(bull) == 19 and abs(bull.sum() - 516.50) < 0.01 and abs(pf - 1.105) < 0.001, (len(bull), bull.sum(), pf)
    assert len(bear) == 41 and abs(bear.sum() - 4139.75) < 0.01, (len(bear), bear.sum())
    mean_obs, sd = float(bull.mean()), float(bull.std(ddof=1))
    assert abs(mean_obs - 27.18) < 0.01 and abs(sd - 707.68) < 0.01

    gross_loss_per_trade = float(-bull[bull < 0].sum() / len(bull))
    theta_bar = (BAR_PF - 1) * gross_loss_per_trade
    anchors = {"PF-1.3 bar (PRIMARY)": theta_bar, "0.5 x bar": 0.5 * theta_bar, "1.5 x bar": 1.5 * theta_bar,
               "derivation point est. (+$27.18, PF 1.105)": mean_obs}

    rows = []
    for name, th in anchors.items():
        for sdm in (0.75, 1.0, 1.25):
            dd = th / (sd * sdm)
            r = {"anchor": name, "theta_usd": th, "sd_mult": sdm, "d": dd}
            for deff in (1.0, 1.5):
                n = n_req(dd, deff)
                r[f"N_deff{deff}"] = n
                r[f"years_deff{deff}_low"] = n / (R_LOW * 52)
                r[f"years_deff{deff}_high"] = n / (R_HIGH * 52)
            rows.append(r)

    head = next(r for r in rows if r["anchor"].startswith("PF-1.3") and r["sd_mult"] == 1.0)
    y_low, y_high = head["years_deff1.0_low"], head["years_deff1.0_high"]
    verdict = "POWERED" if max(y_low, y_high) <= LEASH_YEARS else ("POWER_UNDETERMINED" if y_high <= LEASH_YEARS else "UNDERPOWERED")

    # MDE reachable inside the leash at each rate; simulated power (bullish+bearish pooled, group-standardised shape) at the headline N
    n_leash = {"low": R_LOW * 52 * LEASH_YEARS, "high": R_HIGH * 52 * LEASH_YEARS}
    mde = {k: {"N": v, "MDE_usd": Z * sd / math.sqrt(v), "MDE_over_bar": Z * sd / math.sqrt(v) / theta_bar} for k, v in n_leash.items()}
    shape = np.concatenate([(bull - bull.mean()) / bull.std(ddof=1), (bear - bear.mean()) / bear.std(ddof=1)])
    rng = np.random.default_rng(SEED)
    d_bar = head["d"]
    n_head = int(math.ceil(head["N_deff1.0"]))
    sim = {"shape_n": int(len(shape)), "size_at_d0_N_head": sim_power(shape, 0.0, n_head, rng),
           "power_at_N_head": sim_power(shape, d_bar, n_head, rng),
           "power_at_N_leash_low": sim_power(shape, d_bar, max(int(n_leash["low"]), 3), rng),
           "power_at_N_leash_high": sim_power(shape, d_bar, max(int(n_leash["high"]), 3), rng),
           "N_head": n_head}

    # rate uncertainty (sensitivity only; the verdict rule above is unchanged): exact Poisson 95% CI on 4 shadow trades in 28 days
    lo_n, hi_n = stats.chi2.ppf(0.025, 2 * 4) / 2, stats.chi2.ppf(0.975, 2 * (4 + 1)) / 2
    r_ci = {"count_ci95": [float(lo_n), float(hi_n)], "rate_per_week_ci95": [float(lo_n / 4), float(hi_n / 4)]}
    for name, th, m in (("PF-1.3 bar, SD x1.0", theta_bar, 1.0), ("PF-1.3 bar, SD x0.75", theta_bar, 0.75), ("1.5 x bar, SD x0.75", 1.5 * theta_bar, 0.75)):
        n = n_req(th / (sd * m), 1.0)
        r_ci[f"years_at_upper_rate | {name}"] = n / (hi_n / 4 * 52)
    n_leash_upper = hi_n / 4 * 52 * LEASH_YEARS
    r_ci["N_in_leash_at_upper_rate"] = n_leash_upper
    r_ci["MDE_usd_at_upper_rate"] = Z * sd / math.sqrt(n_leash_upper)

    result = {"verdict": verdict, "rate_uncertainty": r_ci,
              "verdict_rule": "POWERED if <=2.0y at both rates; POWER_UNDETERMINED if <=2.0y at r_high only; UNDERPOWERED if >2.0y at both — at the PF-1.3 anchor, DEFF 1.0",
              "inputs": {"bull_N": 19, "bull_mean": mean_obs, "bull_sd": sd, "bull_pf": float(pf), "gross_loss_per_trade": gross_loss_per_trade,
                         "theta_bar_usd": theta_bar, "d_bar": d_bar, "d_observed": mean_obs / sd,
                         "rate_low_per_week": R_LOW, "rate_high_per_week": R_HIGH, "window_days": WINDOW_DAYS},
              "grid": rows, "mde_inside_leash": mde, "simulation": sim,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "power_verdict.json").write_text(json.dumps(result, indent=2, default=float))

    L = [f"VERDICT: {verdict}   (rule: {result['verdict_rule']})", "",
         f"corrected-bar bullish trades: N=19 mean ${mean_obs:.2f} SD ${sd:.2f} PF {pf:.3f}; gross loss/trade ${gross_loss_per_trade:.2f}",
         f"PF-1.3 bar implies theta = ${theta_bar:.2f}/trade  d = {d_bar:.4f}   (observed d = {mean_obs/sd:.4f})",
         f"rates: backtest {R_LOW:.3f}/wk ({R_LOW*52:.1f}/yr) | shadow {R_HIGH:.3f}/wk ({R_HIGH*52:.1f}/yr, 4 trades in 28 days)", "",
         "anchor                                   theta$   SDx    d     | N(DEFF1)  yrs@low  yrs@high | N(DEFF1.5) yrs@low  yrs@high"]
    for r in rows:
        L.append(f"{r['anchor']:<40} {r['theta_usd']:>6.1f}  {r['sd_mult']:.2f}  {r['d']:.4f} | {r['N_deff1.0']:>8.0f} {r['years_deff1.0_low']:>8.1f} {r['years_deff1.0_high']:>8.1f} | "
                 f"{r['N_deff1.5']:>9.0f} {r['years_deff1.5_low']:>8.1f} {r['years_deff1.5_high']:>8.1f}")
    L += ["", f"Inside a {LEASH_YEARS:.0f}y leash (80% power, one-sided 5%):"]
    for k, m in mde.items():
        L.append(f"  rate {k}: N = {m['N']:.0f}   MDE = ${m['MDE_usd']:.0f}/trade = {m['MDE_over_bar']:.2f}x the PF-1.3 effect")
    L += ["", f"simulated (pooled bullish+bearish shape, n={sim['shape_n']}): size at d=0, N={sim['N_head']}: {sim['size_at_d0_N_head']:.3f}; "
          f"power at the PF-1.3 effect: N={sim['N_head']} {sim['power_at_N_head']:.2f} | N_leash_low {sim['power_at_N_leash_low']:.2f} | N_leash_high {sim['power_at_N_leash_high']:.2f}"]
    L += ["", f"rate uncertainty (exact Poisson 95% on 4 shadow trades/28d): {r_ci['rate_per_week_ci95'][0]:.2f} .. {r_ci['rate_per_week_ci95'][1]:.2f} per week"]
    L += [f"  years to N_stop at the UPPER rate: " + " | ".join(f"{k.split('| ')[1]}: {v:.1f}" for k, v in r_ci.items() if k.startswith("years_at_upper")),
          f"  inside a 2y leash at the upper rate: N = {r_ci['N_in_leash_at_upper_rate']:.0f}, MDE = ${r_ci['MDE_usd_at_upper_rate']:.0f}/trade = {r_ci['MDE_usd_at_upper_rate']/theta_bar:.2f}x the PF-1.3 effect"]
    (OUT / "power_gate_output.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
