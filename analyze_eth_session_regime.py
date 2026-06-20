#!/usr/bin/env python3
"""
ETH session-momentum: WHY did the edge die in H2?  (Gate-0 post-mortem, diagnostic)

The Gate-0 test (backtest_eth_session_momentum.py) showed ETH session-momentum has
a real per-trade edge POOLED (+9.5 bps net of 10bps cost) but it is entirely in the
first chronological half (H1 +19.4 / H2 -0.5 bps). This script asks the load-bearing
question:

    Is H2's failure the "post-2025 funding-negative bear" regime?

If yes, two consequences:
  1. S1 is regime-conditional, not dead - it works in risk-on/positive-funding/trending
     regimes and our 18mo back-half is simply hostile.
  2. The SAME regime curse threatens the runner-up S2 (ETH funding-extreme reversal),
     because positive-funding extremes (the BIS crash-premium mechanism) disappear in a
     funding-negative bear -> S2 would be event-starved in exactly this window too.

DIAGNOSTIC ONLY. This is NOT a new strategy and NOT a validation. We are explaining a
failure, not searching for a favourable subset to revive S1 (that would be the
restrict-to-favourable-subset trap: feedback_iteration_loop_pattern). Funding, trend
and vol are CONFOUNDED in crypto; we report all three and say which best tracks the edge.

Note: ETH funding history is not on disk; BTC funding (PF_XBTUSD_funding_rate.csv) is
used as the crypto-wide funding-regime proxy. ETH funding runs hotter but is highly
correlated in sign/regime.

Usage:
    .venv/bin/python analyze_eth_session_regime.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_eth_session_momentum import ETH_CSV, BTC_CSV, build_daily, load_bars, parse_window

FUNDING_CSV = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
COST_BPS = 10.0


def daily_close(df: pd.DataFrame) -> pd.DataFrame:
    """Last close per date -> daily trend/vol features."""
    d = df.groupby("date")["close"].last().rename("eth_close").to_frame()
    d["eth_dret"] = d["eth_close"].pct_change()
    d["eth_trend_30d"] = d["eth_close"].pct_change(30)
    d["eth_rv_20d"] = d["eth_dret"].rolling(20).std()
    return d.reset_index()


def load_daily_funding(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    f["date"] = f["timestamp"].dt.date
    f["funding_rate"] = f["funding_rate"].astype(float)
    # annualise each 8h print, then daily mean
    f["funding_ann"] = f["funding_rate"] * 3 * 365
    g = f.groupby("date")["funding_ann"].mean().rename("funding_ann").reset_index()
    return g


def cond_table(df: pd.DataFrame, label: str, mask_pos, mask_neg, name_pos, name_neg) -> str:
    out = [f"\n{label}:"]
    for nm, m in [(name_pos, mask_pos), (name_neg, mask_neg)]:
        sub = df[m]
        if len(sub) == 0:
            out.append(f"  {nm:<28} n=0")
            continue
        out.append(f"  {nm:<28} n={len(sub):4d}  "
                   f"mean_net={sub['net_bps'].mean():+7.2f} bps  "
                   f"winrate={sub['gross_pos'].mean():.3f}  "
                   f"sum_net={sub['net_bps'].sum()/100:+.1f} %ret")
    return "\n".join(out)


def main():
    eu_win = parse_window("07:00-13:00")
    us_win = parse_window("13:00-21:00")

    eth = load_bars(ETH_CSV)
    daily = build_daily(eth, eu_win, us_win)

    # momentum net pnl per day at thr=0 (amplitude doesn't help -> use all days)
    sig = np.sign(daily["eu_ret"].values)
    gross = sig * daily["us_ret"].values
    daily["net"] = gross - COST_BPS / 1e4
    daily["net_bps"] = daily["net"] * 1e4
    daily["gross_pos"] = (gross > 0).astype(float)

    # regime features
    feats = daily_close(eth)
    fund = load_daily_funding(FUNDING_CSV)
    daily = daily.merge(feats, on="date", how="left").merge(fund, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)

    n = len(daily)
    half = n // 2
    h1 = daily.iloc[:half]
    h2 = daily.iloc[half:]

    print("=" * 80)
    print("ETH SESSION-MOMENTUM H2-FAILURE POST-MORTEM  (diagnostic, not validation)")
    print("=" * 80)
    print(f"Session-days: {n}   H1: {h1['date'].iloc[0]}..{h1['date'].iloc[-1]}   "
          f"H2: {h2['date'].iloc[0]}..{h2['date'].iloc[-1]}")
    fund_cov = daily['funding_ann'].notna().mean()
    print(f"BTC-funding coverage of session-days: {fund_cov:.0%} "
          f"(proxy for crypto funding regime)")

    print("\n--- H1 vs H2 regime character ---")
    for nm, h in [("H1", h1), ("H2", h2)]:
        print(f"  {nm}: session-mom net={h['net_bps'].mean():+6.2f} bps | "
              f"funding_ann(BTC)={h['funding_ann'].mean():+.1%} | "
              f"%days funding<0={ (h['funding_ann']<0).mean():.0%} | "
              f"ETH 30d-trend={h['eth_trend_30d'].mean():+.1%} | "
              f"%days uptrend={ (h['eth_trend_30d']>0).mean():.0%} | "
              f"ETH RV20={h['eth_rv_20d'].mean():.4f}")

    # ---- condition the edge on each regime variable ----
    d = daily.dropna(subset=["funding_ann", "eth_trend_30d", "eth_rv_20d"]).copy()
    print(f"\n(conditioning on rows with full regime features: n={len(d)})")

    print(cond_table(d, "By FUNDING sign (BTC proxy)",
                     d["funding_ann"] > 0, d["funding_ann"] <= 0,
                     "funding POSITIVE", "funding <= 0"))
    # funding tertiles
    qf = d["funding_ann"].quantile([1/3, 2/3]).values
    print(cond_table(d, "By FUNDING level (low vs high tertile)",
                     d["funding_ann"] >= qf[1], d["funding_ann"] <= qf[0],
                     "funding HIGH tertile", "funding LOW tertile"))

    print(cond_table(d, "By ETH 30d TREND sign",
                     d["eth_trend_30d"] > 0, d["eth_trend_30d"] <= 0,
                     "ETH uptrend", "ETH downtrend"))

    qv = d["eth_rv_20d"].quantile([1/3, 2/3]).values
    print(cond_table(d, "By ETH 20d realized VOL (high vs low tertile)",
                     d["eth_rv_20d"] >= qv[1], d["eth_rv_20d"] <= qv[0],
                     "vol HIGH tertile", "vol LOW tertile"))

    # ---- which monthly regime variable best tracks the monthly edge? ----
    m = daily.groupby("month").agg(
        n=("net_bps", "size"),
        net_bps=("net_bps", "mean"),
        funding_ann=("funding_ann", "mean"),
        eth_ret=("eth_dret", lambda s: (1 + s.fillna(0)).prod() - 1),
        rv=("eth_rv_20d", "mean"),
    ).reset_index()
    print("\n--- Monthly breakdown ---")
    print(m.to_string(index=False,
          formatters={"net_bps": lambda x: f"{x:+.1f}", "funding_ann": lambda x: f"{x:+.1%}",
                      "eth_ret": lambda x: f"{x:+.1%}", "rv": lambda x: f"{x:.4f}"}))

    mm = m.dropna()
    def corr(a, b):
        return float(np.corrcoef(mm[a], mm[b])[0, 1]) if len(mm) > 2 else float("nan")
    c_fund = corr("net_bps", "funding_ann")
    c_trend = corr("net_bps", "eth_ret")
    c_vol = corr("net_bps", "rv")
    print("\n--- Monthly correlation of session-momentum edge with regime variables ---")
    print(f"  corr(monthly net_bps, funding_ann) = {c_fund:+.2f}")
    print(f"  corr(monthly net_bps, ETH month ret) = {c_trend:+.2f}")
    print(f"  corr(monthly net_bps, ETH realized vol) = {c_vol:+.2f}")

    # ---- verdict ----
    print("\n" + "#" * 80)
    print("DIAGNOSIS")
    print("#" * 80)
    h2_funding_neg = (h2["funding_ann"] < 0).mean()
    h1_funding_neg = (h1["funding_ann"] < 0).mean()
    funding_dropped = h2["funding_ann"].mean() < h1["funding_ann"].mean()
    drivers = sorted([("funding", abs(c_fund)), ("trend", abs(c_trend)), ("vol", abs(c_vol))],
                     key=lambda x: -x[1])
    print(f"H1 mean funding {h1['funding_ann'].mean():+.1%} -> H2 mean funding "
          f"{h2['funding_ann'].mean():+.1%}  (dropped={funding_dropped})")
    print(f"%days funding<0:  H1={h1_funding_neg:.0%}  H2={h2_funding_neg:.0%}")
    print(f"Strongest monthly driver of the edge: {drivers[0][0]} "
          f"(|corr|={drivers[0][1]:.2f}), then {drivers[1][0]} ({drivers[1][1]:.2f}).")
    print()
    if h2_funding_neg >= 0.30 and drivers[0][0] == "funding":
        print("=> H2 failure IS substantially the funding-negative bear. S1 is REGIME-")
        print("   CONDITIONAL (works risk-on/positive-funding). CRITICAL: S2 inherits the")
        print("   same curse - positive-funding extremes vanish here, so S2 is event-")
        print("   starved in this exact window. The crypto program is REGIME-BLOCKED,")
        print("   same posture as BTC-CARRY dormant. Park, don't pivot; wait for funding")
        print("   to normalize, then both S1 and S2 become testable on fresh data.")
    elif funding_dropped and drivers[0][0] != "funding":
        print(f"=> Funding DID compress into H2, but the edge tracks {drivers[0][0]} more")
        print(f"   tightly than funding. H2 failure is better explained by {drivers[0][0]}")
        print(f"   regime than by funding sign per se (the three are confounded). S2's")
        print(f"   funding mechanism is NOT automatically doomed by the same variable -")
        print(f"   evaluate S2 on its own funding-event count, not S1's trend/vol decay.")
    else:
        print("=> H2 failure is NOT primarily the funding-negative bear. The session-")
        print("   momentum edge decays on trend/vol regime, largely independent of funding")
        print("   sign. S2 must be judged separately on funding-extreme event frequency.")
    print("\n(Diagnostic only. No subset is being promoted to a strategy.)")


if __name__ == "__main__":
    main()
