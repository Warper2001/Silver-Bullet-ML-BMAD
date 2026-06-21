#!/usr/bin/env python3
"""
ETH/BTC LINEAR-vs-INVERSE funding-spread harvest — GATE 0 backtest.

Origin: bmad-party-mode round. The only borrow-free way to harvest negative funding
on Kraken (Amelia + Dr. Quinn) is to hedge a perp with ANOTHER perp on the same
underlying: LONG one of {linear PF_XBTUSD, inverse PI_XBTUSD} + SHORT the other.
Price exposure cancels (both track the same BTC index); you collect the DIVERGENCE
between the two funding rates -- a market-SEGMENTATION edge (different collateral,
different trader populations), NOT the headline funding level, and NOT a price bet.

DECISION STAT (Dr. Quinn): 25th-percentile of the annualized net harvest must clear
~10-15%/yr, else KILL. p25 (not mean) because carry must survive a bad quarter.

SIGN CONVENTION (Amelia's load-bearing flag — pinned here):
  funding f is "rate paid by LONGS" = mark/spot - 1 (perp rich => longs pay).
  Funding PnL for a leg held in direction d (+1 long, -1 short) = -d * f.
  Construction "SHORT linear / LONG inverse":  PnL = (+f_L) + (-f_I) = f_L - f_I.
  Construction "LONG linear / SHORT inverse":  PnL = f_I - f_L  (the negative).
  We harvest |f_L - f_I| by choosing the side; the static direction is just the one
  bit "which leg funds hotter on average" (NOT a fitted parameter).

CAVEATS (Mary):
  * Both funding series are BASIS PROXIES (mark/spot-1), not Kraken's published paid
    funding. This backtest screens whether the spread EXISTED & was big enough; it does
    NOT prove capturability. A prospective logger on REAL funding prints is the true
    Gate 0 before any capital. We cannot validate the proxy vs real prints w/o API keys.
  * Cap sensitivity applied (linear +/-0.5%/hr, inverse +/-0.25%/hr) since the proxy is
    uncapped and the 25th-pct tail is exactly where caps bite.

Usage: .venv/bin/python backtest_funding_spread_harvest.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

LIN = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
INV = Path("data/kraken/PI_XBTUSD_funding_rate.csv")

PERIODS_PER_YEAR = 3 * 365           # 8h funding -> 1095/yr
TAKER_BPS = 5.0                      # per leg per side
PAIR_RT_BPS = TAKER_BPS * 2 * 2      # open 2 legs + close 2 legs = 20 bps per full round-trip
LIN_CAP_8H = 0.005 * 8               # +/-0.5%/hr -> +/-4% per 8h
INV_CAP_8H = 0.0025 * 8              # +/-0.25%/hr -> +/-2% per 8h
DEADBAND_GRID_BPS = [0, 5, 10, 20, 40]   # the ONE knob for the sign-following variant


def load(path):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d["funding_rate"] = d["funding_rate"].astype(float)
    return d[["timestamp", "funding_rate"]].sort_values("timestamp")


def ann(x):
    return x * PERIODS_PER_YEAR


def pstats(net_8h, years, label):
    """net_8h: per-period net harvest (decimal). Returns dict of annualized stats."""
    a = ann(net_8h)
    return dict(
        label=label, n=len(net_8h),
        mean=round(float(np.mean(a)) * 100, 2),
        median=round(float(np.median(a)) * 100, 2),
        p25=round(float(np.percentile(a, 25)) * 100, 2),
        p75=round(float(np.percentile(a, 75)) * 100, 2),
        pos_frac=round(float(np.mean(net_8h > 0)), 3),
    )


def fmt(s):
    return (f"  {s['label']:<34} n={s['n']:>4}  mean={s['mean']:>7.2f}%  "
            f"med={s['median']:>7.2f}%  p25={s['p25']:>7.2f}%  p75={s['p75']:>7.2f}%  "
            f"pos={s['pos_frac']:.3f}")


def main():
    lin = load(LIN).rename(columns={"funding_rate": "f_L"})
    inv = load(INV).rename(columns={"funding_rate": "f_I"})
    df = pd.merge(lin, inv, on="timestamp", how="inner").reset_index(drop=True)
    years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25
    n = len(df)
    print("=" * 92)
    print("LINEAR (PF_XBTUSD) vs INVERSE (PI_XBTUSD) FUNDING-SPREAD HARVEST — GATE 0")
    print(f"{n} aligned 8h periods  {df['timestamp'].iloc[0].date()}..{df['timestamp'].iloc[-1].date()}  ({years:.2f}y)")
    print("=" * 92)
    print(f"mean linear funding   = {ann(df['f_L'].mean())*100:+.2f}%/yr")
    print(f"mean inverse funding  = {ann(df['f_I'].mean())*100:+.2f}%/yr")
    print(f"mean spread (f_L-f_I) = {ann((df['f_L']-df['f_I']).mean())*100:+.2f}%/yr "
          f"(>0 => SHORT-linear/LONG-inverse is the static-harvest direction)")

    half = n // 2
    h1 = df.iloc[:half]; h2 = df.iloc[half:]

    # ---- raw spread (gross, sign-agnostic magnitude) ----
    sp = (df["f_L"] - df["f_I"]).values            # harvest of SHORT-lin/LONG-inv construction
    # static direction = sign of overall mean (one bit, not a fit)
    static_dir = 1.0 if sp.mean() >= 0 else -1.0   # +1 => short-lin/long-inv
    dir_name = "SHORT-linear / LONG-inverse" if static_dir > 0 else "LONG-linear / SHORT-inverse"

    print(f"\n--- Gross funding-spread distribution (annualized) ---")
    print(fmt(pstats(np.abs(sp) * 0 + sp, years, "raw f_L - f_I (gross)")))

    # ---- Construction 1: STATIC pair, held whole window ----
    fee_static_8h = (PAIR_RT_BPS / 1e4) / (n)      # one open+close amortized across all periods
    net1 = static_dir * sp - fee_static_8h
    s1 = pstats(net1, years, f"STATIC {dir_name} (net)")
    s1_h1 = pstats(static_dir * (h1["f_L"]-h1["f_I"]).values - fee_static_8h, years, "  static H1")
    s1_h2 = pstats(static_dir * (h2["f_L"]-h2["f_I"]).values - fee_static_8h, years, "  static H2")
    print(f"\n--- Construction 1: STATIC (held whole window, fees amortized) ---")
    print(fmt(s1)); print(fmt(s1_h1)); print(fmt(s1_h2))

    # ---- Construction 2: SIGN-FOLLOWING with dead-band (the ONE knob) ----
    # position from PRIOR-period spread sign (no look-ahead); flat inside dead-band.
    print(f"\n--- Construction 2: SIGN-FOLLOWING, dead-band sweep (no look-ahead) ---")
    print(f"  {'band_bps':>8} {'n_trades':>9} {'flips/yr':>9} {'mean%':>8} {'p25%':>8} {'pos':>6} {'net_after_fee%':>15}")
    for band_bps in DEADBAND_GRID_BPS:
        band = band_bps / 1e4
        prev = pd.Series(sp).shift(1).values        # prior-period spread, no look-ahead
        pos = np.where(prev > band, 1.0, np.where(prev < -band, -1.0, 0.0))
        pnl = pos * sp                              # collect this period's spread in chosen dir
        active = pos != 0
        flips = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], pos]))) > 0))
        fee_total = flips * (PAIR_RT_BPS / 1e4)     # 20bps per position change
        gross_ann = ann(np.nanmean(pnl[active])) * 100 if active.any() else float("nan")
        p25_ann = ann(np.nanpercentile(pnl[active], 25)) * 100 if active.any() else float("nan")
        net_ann = (np.nansum(pnl) - fee_total) / years * 100   # total net / years
        print(f"  {band_bps:>8} {int(active.sum()):>9} {flips/years:>9.1f} "
              f"{gross_ann:>8.2f} {p25_ann:>8.2f} {np.mean(active):>6.3f} {net_ann:>15.2f}")

    # ---- Cap sensitivity (Mary): clamp each leg to its real cap, recompute static p25 ----
    fL_c = df["f_L"].clip(-LIN_CAP_8H, LIN_CAP_8H).values
    fI_c = df["f_I"].clip(-INV_CAP_8H, INV_CAP_8H).values
    net1_cap = static_dir * (fL_c - fI_c) - fee_static_8h
    s1c = pstats(net1_cap, years, "STATIC w/ funding caps applied")
    print(f"\n--- Cap sensitivity (linear +/-0.5%/hr, inverse +/-0.25%/hr) ---")
    print(fmt(s1c))
    n_lin_capped = int((df['f_L'].abs() > LIN_CAP_8H).sum())
    n_inv_capped = int((df['f_I'].abs() > INV_CAP_8H).sum())
    print(f"  periods where proxy exceeded cap: linear={n_lin_capped}, inverse={n_inv_capped}")

    # ---- Verdict ----
    print(f"\n{'#'*92}\nGATE 0 VERDICT (Dr. Quinn: 25th-pct annualized net must clear ~10-15%/yr)\n{'#'*92}")
    p25 = s1["p25"]; p25_h1 = s1_h1["p25"]; p25_h2 = s1_h2["p25"]; p25_cap = s1c["p25"]
    print(f"STATIC net p25      = {p25:+.2f}%/yr   (H1 {p25_h1:+.2f} / H2 {p25_h2:+.2f})")
    print(f"STATIC net p25 capped = {p25_cap:+.2f}%/yr")
    print(f"STATIC net mean     = {s1['mean']:+.2f}%/yr")
    passed = (p25 >= 10.0) and (p25_h1 > 0) and (p25_h2 > 0) and (p25_cap >= 10.0)
    thin_edge = (s1['mean'] > 0) and (p25 < 10.0)
    if passed:
        print("VERDICT: PASS (provisional, basis-proxy) — p25 clears the bar in both halves and")
        print("         survives caps. NEXT: prospective logger on REAL funding prints = true Gate 0.")
    elif thin_edge:
        print(f"VERDICT: FAIL on the pre-registered bar — a REAL but THIN segmentation spread exists")
        print(f"         (mean +{s1['mean']:.1f}%/yr, median +{s1['median']:.1f}%/yr), but the 25th-pct is")
        print(f"         {p25:+.1f}%/yr — it does NOT clear ~10-15%/yr and is NEGATIVE (losing quarters).")
        print(f"         Tail-fragile and regime-decaying (H1 p25 {p25_h1:+.1f} -> H2 {p25_h2:+.1f}); the")
        print(f"         sign-following variant is fee-churned at low band / fat-tail-noise (N<15) at high")
        print(f"         band. KILL negative-funding harvest as an income source. NOT deployable.")
    else:
        print("VERDICT: FAIL — no positive spread even at the mean. KILL the negative-funding harvest idea.")
    print("\n(Basis-proxy screen only. Real published funding + a prospective logger remain the true")
    print(" Gate 0 before any capital — this cannot, by construction, authorize a live position.)")


if __name__ == "__main__":
    main()
