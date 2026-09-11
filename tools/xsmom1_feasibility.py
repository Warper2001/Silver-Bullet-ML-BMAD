"""XSMOM-1 executable-feasibility and cost check (seal §6, §7).

Pre-outcome by construction: uses only contract specs and *second moments*
(realised volatility). No signal, no pairing, no returns-to-signal alignment.
Safe to run regardless of the power verdict.

Answers two questions the seal requires before any P&L could be labelled
`executable`:
  1. What account size is needed to hold the frozen book in integer contracts?
  2. What is the estimated cost drag, and is cost or power the binding constraint?
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/root/Silver-Bullet-ML-BMAD")
PILOT = REPO / "data/commodity_curve/coverage-pilot-2025-20260906-v3"
OUT = Path(__file__).resolve().parents[1] / "_bmad-output" / "xsmom1_feasibility.json"

MIN_DTE = 7
# seal §6 -- ESTIMATES. No fill data exists for any full-size contract here.
TICKS_RT = {"CL": 1.0, "NG": 1.0, "HG": 1.0, "ZC": 1.0, "ZS": 1.0,
            "HO": 1.5, "RB": 1.5, "ZW": 1.5, "ZM": 1.5, "ZL": 1.5, "LE": 1.5, "HE": 2.0}
FEES_RT = 5.00
THETA_PLAUS_SR = 0.50
VOL_TARGET = 0.10          # 10% annualised portfolio vol
DRAG_K8 = 0.084            # measured in design under mismatched pairing


def main() -> None:
    bars = pd.read_csv(PILOT / "contract_bars.csv",
                       usecols=["canonical_root", "contract_code", "TimeStamp",
                                "Close", "OpenInterest"])
    meta = pd.read_csv(PILOT / "contracts.csv",
                       usecols=["Symbol", "ExpirationDate", "PriceFormat"])
    exp = dict(zip(meta["Symbol"], pd.to_datetime(meta["ExpirationDate"]).dt.tz_localize(None)))
    spec = {}
    for sym, pf in zip(meta["Symbol"], meta["PriceFormat"]):
        d = ast.literal_eval(pf)          # stringified python dict, not JSON
        spec[sym] = (float(d["PointValue"]), float(d["Increment"]))

    bars["date"] = pd.to_datetime(bars["TimeStamp"]).dt.tz_localize(None).dt.normalize()
    bars["expiry"] = bars["contract_code"].map(exp)
    bars = bars.dropna(subset=["expiry", "Close", "OpenInterest"])
    bars = bars[(bars["expiry"] - bars["date"]).dt.days > MIN_DTE]
    bars = bars.sort_values(["canonical_root", "date", "OpenInterest"])
    front = bars.groupby(["canonical_root", "date"], as_index=False).last()

    front = front.sort_values(["canonical_root", "date"])
    front["prev"] = front.groupby("canonical_root")["Close"].shift(1)
    front["prev_code"] = front.groupby("canonical_root")["contract_code"].shift(1)
    same = front["contract_code"] == front["prev_code"]
    front["logret"] = np.where(same & front["prev"].gt(0), np.log(front["Close"] / front["prev"]), 0.0)
    front["week"] = front["date"].dt.to_period("W-FRI")

    rows = []
    for root, g in front.groupby("canonical_root"):
        pv, inc = spec[g["contract_code"].iloc[-1]]
        tick_val = pv * inc
        wk = g.groupby("week")["logret"].sum()
        med_px = g["Close"].median()
        sigma_wk_usd = float(wk.std(ddof=1) * med_px * pv)     # $ vol per contract per week
        cost_rt = FEES_RT + TICKS_RT[root] * tick_val
        rows.append({
            "root": root, "point_value": pv, "tick_value": round(tick_val, 2),
            "median_price": round(med_px, 4),
            "notional_per_contract": round(med_px * pv, 0),
            "weekly_sigma_usd": round(sigma_wk_usd, 0),
            "cost_rt_estimate": round(cost_rt, 2),
            "cost_pct_of_weekly_sigma": round(100 * cost_rt / sigma_wk_usd, 2),
        })
    df = pd.DataFrame(rows).sort_values("cost_pct_of_weekly_sigma")

    print("XSMOM-1 FEASIBILITY  (pre-outcome: specs + second moments only)\n")
    print(df.to_string(index=False))

    # --- capital: binding root is the one with the largest $ vol per contract
    binding = df.loc[df["weekly_sigma_usd"].idxmax()]
    w_top3 = 1 / 3
    port_sigma_wk_needed = binding["weekly_sigma_usd"] / w_top3
    port_sigma_ann = port_sigma_wk_needed * np.sqrt(52)
    acct_top3 = port_sigma_ann / VOL_TARGET

    print(f"\nBinding root (largest $ vol/contract): {binding['root']} "
          f"@ ${binding['weekly_sigma_usd']:,.0f}/week")
    print(f"  top-3/bottom-3 book (|w|=1/3):")
    print(f"    portfolio weekly sigma to hold 1 contract = ${port_sigma_wk_needed:,.0f}")
    print(f"    annualised = ${port_sigma_ann:,.0f}  ->  account at {VOL_TARGET:.0%} vol target "
          f"= ${acct_top3:,.0f}")
    w_min_rank = 1 / (2 * 6)     # smallest rank weight in a 12-root demeaned book
    acct_rank = (binding["weekly_sigma_usd"] / w_min_rank) * np.sqrt(52) / VOL_TARGET
    print(f"  full rank-weighted book (smallest |w|≈{w_min_rank:.3f}) -> ${acct_rank:,.0f}")

    print(f"\nAll 12 roots are FULL-SIZE contracts -- no micros exist in this universe.")
    print(f"Cost drag at k=8 = {DRAG_K8:.3f} Sharpe pts = "
          f"{100*DRAG_K8/THETA_PLAUS_SR:.0f}% of the declared plausible effect "
          f"(doubled: {100*2*DRAG_K8/THETA_PLAUS_SR:.0f}%).")
    print("=> Cost is material but NOT the binding constraint. Power is (10.6%),")
    print("   and capital is an independent second blocker.")

    excl = df[df["cost_pct_of_weekly_sigma"] > 2.5]["root"].tolist()
    print(f"\nSeal §6 cost-exclusion rule (>2.5% of weekly sigma) would remove: {excl}")
    print("   Seal pre-commits to NOT applying it: cutting breadth 12->8 scales the")
    print("   plausible effect by sqrt(8/12)=0.816, which costs more than the drag saved.")

    OUT.write_text(json.dumps({
        "seal": "XSMOM-1", "per_root": rows,
        "binding_root": binding["root"],
        "min_account_top3_usd": round(float(acct_top3), 0),
        "min_account_rank_weighted_usd": round(float(acct_rank), 0),
        "vol_target": VOL_TARGET,
        "cost_drag_sharpe_k8": DRAG_K8,
        "cost_drag_pct_of_theta_plaus": round(100 * DRAG_K8 / THETA_PLAUS_SR, 1),
        "would_exclude_at_2p5pct": excl,
        "exclusion_applied": False,
        "execution_label": "theoretical",
        "note": "All roots full-size; no micros. Pre-outcome: no signal or returns alignment used.",
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
