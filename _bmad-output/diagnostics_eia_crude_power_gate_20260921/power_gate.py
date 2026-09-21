"""POWER GATE — EIA-Wednesday crude intraday momentum (Wen, Indriawan, Lien, Xu, Energy Journal 44(5), 2023).

OUTCOME-BLIND. No strategy is built, no signal is fired, no bars are read. Inputs are ONLY the paper's
own published summary statistics (accepted manuscript, Table 6, EIA-day trading rule; USO ETF, 1-min data,
2006-04-10 .. 2019-07-31, 591 Wednesday-10:30 ET EIA days; signal = 10:30-11:00 return, trade = 15:30-16:00):

    annualised mean 4.14 %  (= daily mean x 252)   t = 1.88   daily SD 0.21 %   N = 591 events
    (units decoded in the recon digest: eia-r2-1.md; this script re-checks that decode reproduces the t-stat)

PRE-COMMITTED DECISION RULE (written before any number below was produced):
  theta_ceiling = the paper's own POINT ESTIMATE per trade. It is the best case: the paper's r3 window and its
                  EIA-Wednesday subset were chosen in-sample, no out-of-sample test exists, so any honest
                  shrinkage can only LOWER it. Verdict is therefore computed at the ceiling, and it is one-sided:
                  if the test is underpowered at the ceiling, no shrink or lower rate can rescue it.
  N_req         = ((z_alpha + z_power) / d)^2, d = theta/SD (per-trade standardised effect), alpha = 0.05
                  one-sided, power = 0.80. DEFF = 1.0 headline (one non-overlapping trade per week);
                  DEFF = 1.5 shown as a sensitivity (same convention as the 2026-09-20 regime-hold gate).
  events/yr     = 591 / 13.4y = 44.1 (paper's observed rate, headline) and 52 (calendar maximum, every Wednesday).
  accrual years = N_req / events per year.
  POWERED       if accrual <= 2.0y at 52/yr AND at the ceiling
  UNDERPOWERED  if accrual  > 2.0y at 52/yr at the ceiling   (=> also underpowered at every lower rate/effect)
  UNDETERMINED  otherwise
  (2.0y is an operator-leash assumption carried over from the 2026-09-20 regime-hold gate, not a derived
   quantity; the full grid is printed so another leash can be read off without re-running.)
  COST FLAG (separate, does not change the power verdict): break-even round-trip cost = theta_ceiling x notional
   per contract. Reported over a price grid. Contract cost is NOT sourced in this run (spread/commission for
   MCL at 15:30-16:00 ET were not retrieved) -- the flag is a comparison against an ASSUMED one-tick floor.

Run: .venv/bin/python _bmad-output/diagnostics_eia_crude_power_gate_20260921/power_gate.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).parent

# ---- paper's published numbers (Table 6, EIA-day rule) -------------------------------------
MEAN_ANN_PCT = 4.14
T_STAT = 1.88
SD_DAILY_PCT = 0.21
N_EVENTS = 591
SAMPLE_YEARS = (2019 + 7 / 12 + 31 / 365) - (2006 + 3 / 12 + 10 / 365)   # 2006-04-10 .. 2019-07-31
TRADING_DAYS = 252
CALENDAR_MAX_EVENTS_PER_YEAR = 52

# ---- test design ----------------------------------------------------------------------------
Z_A, Z_P = stats.norm.ppf(0.95), stats.norm.ppf(0.80)
LEASH_YEARS = 2.0
DEFF_HEADLINE, DEFF_SENS = 1.0, 1.5

# ---- MCL contract facts (CME education page, search-snippet level; recon source [14]) -------
MCL_BBL, MCL_TICK_USD = 100, 1.00
PRICE_GRID = [50, 60, 70, 80, 90]         # $/bbl; ASSUMED grid, front-month price not sourced here


def n_required(d: float, deff: float = 1.0) -> float:
    return deff * ((Z_A + Z_P) / d) ** 2


def main() -> None:
    mean_per_trade = MEAN_ANN_PCT / TRADING_DAYS / 100          # fraction of price
    sd_per_trade = SD_DAILY_PCT / 100
    d_paper = mean_per_trade / sd_per_trade
    t_check = d_paper * math.sqrt(N_EVENTS)
    assert abs(t_check - T_STAT) < 0.05, f"unit decode does not reproduce paper t: {t_check:.3f} vs {T_STAT}"

    rate_paper = N_EVENTS / SAMPLE_YEARS
    rates = {"paper_observed": rate_paper, "calendar_max": float(CALENDAR_MAX_EVENTS_PER_YEAR)}

    grid = []
    for label, mult in [("ceiling (paper point est.)", 1.0), ("x0.75", 0.75), ("x0.50", 0.50),
                        ("x0.33", 1 / 3), ("lower 1-SE (d - 1/sqrt(N))", None)]:
        d = d_paper - 1 / math.sqrt(N_EVENTS) if mult is None else d_paper * mult
        row = {"effect": label, "d": d, "theta_bp": d * sd_per_trade * 1e4}
        for deff in (DEFF_HEADLINE, DEFF_SENS):
            n = n_required(d, deff)
            row[f"N_req_deff{deff}"] = n
            for rname, r in rates.items():
                row[f"years_deff{deff}_{rname}"] = n / r
        grid.append(row)

    ceiling = grid[0]
    accrual_max_rate = ceiling[f"years_deff{DEFF_HEADLINE}_calendar_max"]
    if accrual_max_rate <= LEASH_YEARS:
        verdict = "POWERED"
    elif accrual_max_rate > LEASH_YEARS:
        verdict = "UNDERPOWERED"

    # MDE inside the leash: smallest d detectable at 80% power with N = rate x leash
    mde = {}
    for rname, r in rates.items():
        n_leash = r * LEASH_YEARS
        mde[rname] = {"N_in_leash": n_leash, "MDE_d": (Z_A + Z_P) / math.sqrt(n_leash)}
        mde[rname]["MDE_over_paper_effect"] = mde[rname]["MDE_d"] / d_paper

    # break-even cost per round trip vs an assumed one-tick floor
    theta_frac = mean_per_trade
    cost = []
    for p in PRICE_GRID:
        notional = p * MCL_BBL
        cost.append({"price": p, "notional_usd": notional, "gross_edge_usd": theta_frac * notional,
                     "one_tick_usd": MCL_TICK_USD,
                     "gross_edge_over_one_tick": theta_frac * notional / MCL_TICK_USD})

    result = {
        "verdict": verdict,
        "verdict_rule": "UNDERPOWERED if accrual > 2.0y at the paper's own point estimate and 52 events/yr",
        "inputs": {"mean_ann_pct": MEAN_ANN_PCT, "t": T_STAT, "sd_daily_pct": SD_DAILY_PCT, "N": N_EVENTS,
                   "sample_years": SAMPLE_YEARS, "events_per_year_paper": rate_paper},
        "derived": {"mean_per_trade_bp": mean_per_trade * 1e4, "sd_per_trade_bp": sd_per_trade * 1e4,
                    "d_paper": d_paper, "t_reproduced": t_check},
        "grid": grid,
        "mde_inside_leash": mde,
        "cost_break_even": cost,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "power_verdict.json").write_text(json.dumps(result, indent=2))

    lines = [f"VERDICT: {verdict}   (rule: {result['verdict_rule']})", "",
             f"paper: mean {mean_per_trade*1e4:.2f} bp/trade, SD {sd_per_trade*1e4:.1f} bp, d={d_paper:.4f}, "
             f"t reproduced {t_check:.2f} (paper {T_STAT}); events/yr paper {rate_paper:.1f}", "",
             "effect                          d      theta_bp | N_req(DEFF1)  yrs@paper  yrs@52 | N_req(DEFF1.5)  yrs@paper  yrs@52"]
    for g in grid:
        lines.append(f"{g['effect']:<30} {g['d']:.4f}  {g['theta_bp']:6.2f}  | {g['N_req_deff1.0']:9.0f} "
                     f"{g['years_deff1.0_paper_observed']:9.1f} {g['years_deff1.0_calendar_max']:8.1f} | "
                     f"{g['N_req_deff1.5']:11.0f} {g['years_deff1.5_paper_observed']:9.1f} "
                     f"{g['years_deff1.5_calendar_max']:8.1f}")
    lines += ["", f"MDE inside a {LEASH_YEARS:.0f}y leash (80% power, one-sided 5%):"]
    for rname, m in mde.items():
        lines.append(f"  {rname}: N={m['N_in_leash']:.0f}  MDE d={m['MDE_d']:.3f}  = {m['MDE_over_paper_effect']:.1f}x the paper's effect")
    lines += ["", "COST BREAK-EVEN (gross edge per contract vs an ASSUMED one-tick $1.00 spread floor; price grid assumed):",
              "  price  notional  gross_edge_usd  edge/one_tick"]
    for c in cost:
        lines.append(f"  ${c['price']:>3}  ${c['notional_usd']:>6,.0f}  ${c['gross_edge_usd']:.2f}        {c['gross_edge_over_one_tick']:.2f}")
    (OUT / "power_gate_output.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
