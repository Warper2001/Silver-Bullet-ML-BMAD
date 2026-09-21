"""POWER GATE — overnight-intraday (CO-OC) cross-sectional reversal on index futures
(Della Corte, Kosowski, Wang, "Market Closure and Short-Term Reversal", Nov 2015 draft; the only source read in full).

OUTCOME-BLIND. The tradable version here is a 2-leg spread: 2 MNQ vs 3 MES (the basket the repo's stat-arb bot already
uses), long the contract with the LOWER overnight return, short the other, entered at the 09:30 ET open, flat at the
16:00 ET close. FIREWALL: this program never forms the signal (CO spread) and never pairs it with the open->close
return. It uses only (a) the paper's published summary statistics and (b) SIGNAL-FREE properties of our clean
front-month bars: the SD of the FIXED-DIRECTION open->close spread P&L (a sign flip does not change an SD), and the
cross-day dispersion of the two overnight returns taken separately. The aligned mean CO x OC is never computed.
Reads data/mim_x/*_frontmonth.csv only (2021-2024; entirely AFTER the paper's 1982-2014 sample, so it is unseen
data that a later confirmatory test would need) -- NOT data/sealed_holdout/.

PAPER (Nov 2015 draft, Tables 3/4/7; gross of costs; 5 CME index futures, 1982-2014; signal AND entry use the same open):
  CO-OC mean 0.252 %/day, SD 0.980 (same units), Sharpe 4.078  => per-day d = 0.252/0.980 = 0.257
  US-stock implementability (Table 7): entry 1 s after 9:30 -> 0.36 %/day; 9:31 -> 0.11; 9:45 -> 0.04
  => retention of the effect: 1 min = 0.306, 15 min = 0.111.  No futures delayed-entry or cost test exists in the draft.

PRE-COMMITTED DECISION RULE (written before any number below was produced):
  d_tradable  = d_paper x retention(1 minute)   [the earliest entry a market order can realistically get; derived from
                the paper's own stock table, applied to futures as an ASSUMPTION since no futures delay test exists]
  cost_d      = daily round-trip cost of the basket (USD) / SD of the basket's daily open->close P&L (USD)
  net_d       = d_tradable - cost_d            (SD unchanged by the sign flip; mean scales with retention)
  N_req       = DEFF x ((z_a + z_p) / net_d)^2, alpha 0.05 one-sided, power 0.80, DEFF 1.0 headline / 1.5 sensitivity
  years       = N_req / trading days per year (252)
  POWERED       if net_d > 0 and years <= 2.0
  UNDERPOWERED  if net_d <= 0 (cost-bound) or years > 2.0
  (Paper-as-printed d, 0-delay retention, is reported as a reference only: the paper itself calls it possibly not
   implementable in real time. 2.0y leash = operator assumption carried over from the 09-20/09-21 gates.)
COST MODEL (labelled ASSUMPTIONS): commission $1.04 RT/contract (repo edge-headroom memo, MNQ; applied to MES) plus
  1 tick slippage per side per contract (MNQ tick $0.50, MES tick $1.25). MNQ RT = 1.04+2x0.50 = $2.04; MES RT = 1.04+2x1.25 = $3.54.
  (The operator's quoted $2.24 MNQ RT is also run as a sensitivity.)

Run: .venv/bin/python _bmad-output/diagnostics_overnight_reversal_power_gate_20260921/power_gate.py
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
MNQ = REPO / "data/mim_x/mnq_1min_2021_2024_frontmonth.csv"
MES = REPO / "data/mim_x/mes_1min_2021_2024_frontmonth.csv"
OUT = Path(__file__).parent
for p in (MNQ, MES):
    assert "sealed_holdout" not in str(p)

# ---- paper's published numbers ----------------------------------------------------------------
PAPER_MEAN, PAPER_SD, PAPER_SHARPE = 0.252, 0.980, 4.078
RET_1MIN = 0.11 / 0.36
RET_15MIN = 0.04 / 0.36
D_PAPER = PAPER_MEAN / PAPER_SD

# ---- test design -------------------------------------------------------------------------------
Z_A, Z_P = stats.norm.ppf(0.95), stats.norm.ppf(0.80)
DAYS_PER_YEAR = 252
LEASH_YEARS = 2.0
DEFF = (1.0, 1.5)

# ---- basket / cost (ASSUMPTIONS, see docstring) ------------------------------------------------
N_MNQ, N_MES = 2, 3
MNQ_PT, MES_PT = 2.0, 5.0
MNQ_TICK, MES_TICK = 0.50, 1.25
COMM_RT = 1.04


def rt_cost(mnq_rt_override: float | None = None) -> float:
    mnq = mnq_rt_override if mnq_rt_override is not None else COMM_RT + 2 * MNQ_TICK
    mes = COMM_RT + 2 * MES_TICK
    return N_MNQ * mnq + N_MES * mes


def session_prices(path: Path) -> pd.DataFrame:
    """Close-stamped bars: the bar stamped 09:31 ET opens at 09:30; the bar stamped 16:00 ET closes at 16:00."""
    d = pd.read_csv(path, usecols=["timestamp", "open", "close"])
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
    d["date"] = ts.dt.normalize().dt.tz_localize(None)
    d["hm"] = ts.dt.strftime("%H:%M")
    o = d[d["hm"] == "09:31"].set_index("date")["open"].rename("open_0930")
    c = d[d["hm"] == "16:00"].set_index("date")["close"].rename("close_1600")
    out = pd.concat([o, c], axis=1).dropna()
    return out[~out.index.duplicated()]


def n_req(d: float, deff: float) -> float:
    return math.inf if d <= 0 else deff * ((Z_A + Z_P) / d) ** 2


def main() -> None:
    a, b = session_prices(MNQ), session_prices(MES)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    # fixed-direction basket P&L, open->close: long 2 MNQ, short 3 MES. SIGNAL-FREE (no signal exists here).
    x = N_MNQ * MNQ_PT * (a["close_1600"] - a["open_0930"]) - N_MES * MES_PT * (b["close_1600"] - b["open_0930"])
    sd_usd = float(x.std(ddof=1))
    mad_sd = float(1.4826 * (x - x.median()).abs().median())
    notional_ratio = (N_MNQ * MNQ_PT * a["open_0930"]) / (N_MES * MES_PT * b["open_0930"])

    # overnight dispersion, each leg separately (no pairing with open->close): informational, roll days contaminate
    co_a = np.log(a["open_0930"] / a["close_1600"].shift(1)).dropna()
    co_b = np.log(b["open_0930"] / b["close_1600"].shift(1)).dropna()
    both = co_a.index.intersection(co_b.index)
    dco = (co_a.loc[both] - co_b.loc[both])
    co_stats = {"n_days": int(len(both)), "sd_overnight_mnq_bp": float(co_a.std() * 1e4),
                "sd_overnight_mes_bp": float(co_b.std() * 1e4), "sd_overnight_spread_bp": float(dco.std() * 1e4),
                "robust_sd_overnight_spread_bp": float(1.4826 * (dco - dco.median()).abs().median() * 1e4)}

    days = int(len(x))
    span_years = (idx.max() - idx.min()).days / 365.25

    rows = []
    for cost_label, cost in [("cost card: comm 1.04 + 1 tick/side (MNQ RT 2.04)", rt_cost()),
                             ("operator MNQ RT 2.24, MES scaled as above", rt_cost(2.24))]:
        cost_d = cost / sd_usd
        for ret_label, ret in [("paper as printed (0 delay, reference only)", 1.0),
                               ("tradable ceiling (1-min retention 0.306)", RET_1MIN),
                               ("15-min retention 0.111", RET_15MIN)]:
            d_gross = D_PAPER * ret
            net = d_gross - cost_d
            row = {"cost": cost_label, "cost_usd_per_day": cost, "cost_d": cost_d, "entry": ret_label,
                   "d_gross": d_gross, "gross_usd_per_day": d_gross * sd_usd, "net_d": net}
            for deff in DEFF:
                n = n_req(net, deff)
                row[f"N_req_deff{deff}"] = n
                row[f"years_deff{deff}"] = n / DAYS_PER_YEAR
            row["no_cost_N_req_deff1.0"] = n_req(d_gross, 1.0)
            row["no_cost_years_deff1.0"] = n_req(d_gross, 1.0) / DAYS_PER_YEAR
            rows.append(row)

    headline = next(r for r in rows if r["entry"].startswith("tradable") and r["cost"].startswith("cost card"))
    if headline["net_d"] <= 0 or headline["years_deff1.0"] > LEASH_YEARS:
        verdict = "UNDERPOWERED"
    else:
        verdict = "POWERED"

    # MDE inside the leash and the cost ceiling implied by it
    n_leash = DAYS_PER_YEAR * LEASH_YEARS
    mde_d = (Z_A + Z_P) / math.sqrt(n_leash)

    # retention of the paper's gross effect needed for POWERED inside the leash (cost card): net_d must reach the MDE
    cost_d_card = headline["cost_d"]
    required_retention = (mde_d + cost_d_card) / D_PAPER

    result = {
        "verdict": verdict,
        "required_retention_for_powered": required_retention,
        "verdict_rule": "UNDERPOWERED if net_d<=0 or years>2.0 at tradable ceiling (1-min retention) with cost card",
        "inputs": {"paper_d": D_PAPER, "ret_1min": RET_1MIN, "ret_15min": RET_15MIN, "days": days,
                   "span_years": span_years, "basket": f"{N_MNQ} MNQ vs {N_MES} MES"},
        "signal_free_properties": {"basket_daily_sd_usd": sd_usd, "basket_daily_robust_sd_usd": mad_sd,
                                   "notional_ratio_mnq_over_mes_median": float(notional_ratio.median()),
                                   "notional_ratio_p05_p95": [float(notional_ratio.quantile(.05)),
                                                              float(notional_ratio.quantile(.95))],
                                   "overnight_dispersion": co_stats},
        "grid": rows,
        "mde_inside_leash": {"N": n_leash, "MDE_d": mde_d, "MDE_over_paper_d": mde_d / D_PAPER},
        "max_cost_usd_for_positive_net_at_tradable_ceiling": headline["gross_usd_per_day"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "power_verdict.json").write_text(json.dumps(result, indent=2))

    L = [f"VERDICT: {verdict}   (rule: {result['verdict_rule']})", "",
         f"days with both legs: {days} over {span_years:.2f}y; basket {N_MNQ} MNQ vs {N_MES} MES; paper d = {D_PAPER:.4f} "
         f"(Sharpe {PAPER_SHARPE}/sqrt(252) = {PAPER_SHARPE/math.sqrt(252):.4f})",
         f"SIGNAL-FREE: daily SD of fixed-direction basket open->close P&L = ${sd_usd:,.2f} (robust ${mad_sd:,.2f}); "
         f"notional ratio MNQ/MES median {notional_ratio.median():.2f} "
         f"(p05-p95 {notional_ratio.quantile(.05):.2f}-{notional_ratio.quantile(.95):.2f})",
         f"overnight dispersion (each leg alone): SD MNQ {co_stats['sd_overnight_mnq_bp']:.1f} bp, MES "
         f"{co_stats['sd_overnight_mes_bp']:.1f} bp, spread {co_stats['sd_overnight_spread_bp']:.1f} bp "
         f"(robust {co_stats['robust_sd_overnight_spread_bp']:.1f} bp; roll days contaminate the plain SD)", ""]
    for r in rows:
        L.append(f"[{r['cost']}] cost ${r['cost_usd_per_day']:.2f}/day = {r['cost_d']:.3f} d | {r['entry']}: "
                 f"gross d {r['d_gross']:.4f} (${r['gross_usd_per_day']:.2f}/day) net d {r['net_d']:+.4f} | "
                 f"no-cost yrs {r['no_cost_years_deff1.0']:.1f} | net yrs DEFF1 {r['years_deff1.0']:.1f} DEFF1.5 {r['years_deff1.5']:.1f}")
    L += ["", f"MDE inside {LEASH_YEARS:.0f}y leash (N={n_leash:.0f}): d = {mde_d:.3f} = {mde_d/D_PAPER:.2f}x the paper's d",
          f"Max total round-trip cost that leaves positive net at the tradable ceiling: ${headline['gross_usd_per_day']:.2f}/day "
          f"(vs cost card ${headline['cost_usd_per_day']:.2f})",
          f"Retention of the paper's gross effect needed for POWERED inside the leash (cost card): "
          f"{required_retention:.3f} (paper's own 1-min stock retention = {RET_1MIN:.3f})"]
    (OUT / "power_gate_output.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
