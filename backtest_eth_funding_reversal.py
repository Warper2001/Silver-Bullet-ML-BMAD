#!/usr/bin/env python3
"""
ETH Funding-Extreme Reversal (S2) — GATE 0 test.

Hypothesis (technical-best-strategy-kraken-pro-perpetual-futures-research-2026-06-20,
runner-up S2; mechanism = BIS WP1087 "Crypto Carry": high carry predicts future price
crashes, +10% standardized carry -> ~22% of OI in sell-liquidations next month):

    When ETH perp funding is in an extreme HIGH percentile, leveraged longs are
    crowded and fragile -> SHORT ETH expecting a reversion/crash. Symmetric on
    extreme-LOW funding (crowded shorts) -> LONG ETH. The contrarian side also
    COLLECTS funding while waiting (short during +funding gets paid), so the trade
    is reversion-alpha + carry combined.

GATE 0 questions for S2 (its binding constraints differ from S1):
    (1) Does fading ETH funding extremes generate >=30 INDEPENDENT events/yr at a
        threshold loose enough to be validatable? (S2's known weakness: the edge
        concentrates in rare extremes -> dilution tension.)
    (2) Is net per-trade expectancy > 0 after 10 bps RT cost, stable across both
        time-halves?
    (3) DECOMPOSITION: is any edge genuine DIRECTIONAL reversion, or is it just the
        collected funding (i.e. carry in disguise, redundant with our BTC-CARRY)?

Discipline:
  * ONE swept knob: the funding-extreme percentile (rolling, no look-ahead).
  * Hold horizon fixed at 7d on mechanism grounds (BIS crash plays out over weeks);
    3d/14d reported as robustness, NOT tuned.
  * Non-overlapping trades entered on extreme ONSET -> honest independent-event count.
  * BTC run as a cross-asset check (mechanism is documented on BTC+ETH).
  * No OI/price-rejection confirmation gate: historical OI is not on disk (live-only).
    That gate would only REDUCE event count; if the unfiltered trade fails Gate 0,
    the filtered one cannot rescue it.

Diagnostic Gate-0 reproduction only. Not a pre-registration; places no orders.

Usage:
    .venv/bin/python backtest_eth_funding_reversal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_eth_session_momentum import ETH_CSV, BTC_CSV, load_bars

ETH_FUND = Path("data/kraken/PF_ETHUSD_funding_rate.csv")
BTC_FUND = Path("data/kraken/PF_XBTUSD_funding_rate.csv")

COST_BPS = 10.0
ROLL_WINDOW = 270          # trailing 8h periods for the rolling percentile (~90 days)
PCTL_GRID = [80, 85, 90, 95, 97.5]   # the swept knob: funding-extreme percentile
HOLD_PERIODS_PRIMARY = 21  # 7 days (8h periods); mechanism-motivated, fixed
HOLD_GRID = [9, 21, 42]    # 3d / 7d / 14d robustness


def load_funding_8h(path: Path) -> pd.DataFrame:
    f = pd.read_csv(path)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    f = f.sort_values("timestamp").reset_index(drop=True)
    f["funding_rate"] = f["funding_rate"].astype(float)   # per-8h decimal
    f["funding_ann"] = f["funding_rate"] * 3 * 365
    return f


def align_price_to_funding(bars: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Attach the ETH close at (or just before) each 8h funding timestamp."""
    bars = bars.sort_values("timestamp")
    px = bars[["timestamp", "close"]].rename(columns={"close": "px"})
    merged = pd.merge_asof(fund, px, on="timestamp", direction="backward")
    return merged.dropna(subset=["px"]).reset_index(drop=True)


def run(df: pd.DataFrame, pctl: float, hold: int, cost_bps: float):
    """Non-overlapping contrarian trades on funding-extreme onset. Returns trade df."""
    r = df["funding_rate"].values
    px = df["px"].values
    ts = df["timestamp"].values
    n = len(df)

    # Rolling percentile thresholds from STRICTLY PRIOR data (shift 1 -> no look-ahead).
    s = pd.Series(r)
    hi = s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).quantile(pctl / 100.0).values
    lo = s.shift(1).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).quantile(1 - pctl / 100.0).values

    cost = cost_bps / 1e4
    trades = []
    i = ROLL_WINDOW
    while i < n - hold:
        d = 0
        if not np.isnan(hi[i]) and r[i] >= hi[i]:
            d = -1   # crowded longs -> SHORT
        elif not np.isnan(lo[i]) and r[i] <= lo[i]:
            d = +1   # crowded shorts -> LONG
        if d == 0:
            i += 1
            continue
        # open non-overlapping trade, hold `hold` periods
        entry_px, exit_px = px[i], px[i + hold]
        dir_pnl = d * (exit_px / entry_px - 1.0)
        fund_pnl = float(np.sum(-d * r[i + 1: i + hold + 1]))   # contrarian collects funding
        net = dir_pnl + fund_pnl - cost
        trades.append(dict(
            t=ts[i], d=d, dir_bps=dir_pnl * 1e4, fund_bps=fund_pnl * 1e4,
            net_bps=net * 1e4, funding_ann=df["funding_ann"].values[i],
        ))
        i += hold   # non-overlapping
    return pd.DataFrame(trades)


def summarize(tr: pd.DataFrame, years: float) -> dict:
    if len(tr) == 0:
        return dict(n=0)
    net = tr["net_bps"].values / 1e4
    half_t = tr["t"].iloc[len(tr) // 2]
    h1 = tr[tr["t"] < half_t]["net_bps"]
    h2 = tr[tr["t"] >= half_t]["net_bps"]
    sharpe = float(np.mean(net) / np.std(net) * np.sqrt(len(tr) / years)) if np.std(net) > 0 else float("nan")
    equity = np.cumprod(1 + net)
    return dict(
        n=len(tr),
        per_yr=round(len(tr) / years, 1),
        n_short=int((tr["d"] < 0).sum()),
        n_long=int((tr["d"] > 0).sum()),
        net_bps=round(tr["net_bps"].mean(), 1),
        dir_bps=round(tr["dir_bps"].mean(), 1),
        fund_bps=round(tr["fund_bps"].mean(), 1),
        win=round(float((tr["net_bps"] > 0).mean()), 3),
        h1_bps=round(float(h1.mean()), 1) if len(h1) else float("nan"),
        h2_bps=round(float(h2.mean()), 1) if len(h2) else float("nan"),
        tot=round(float(equity[-1] - 1), 3),
        sharpe=round(sharpe, 2),
    )


def analyse(name: str, fund_path: Path, bars_path: Path):
    fund = load_funding_8h(fund_path)
    bars = load_bars(bars_path)
    df = align_price_to_funding(bars, fund)
    years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25
    print(f"\n{'='*86}\n{name}: {len(df)} 8h funding rows  "
          f"({df['timestamp'].iloc[0].date()}..{df['timestamp'].iloc[-1].date()}, {years:.2f}y)  "
          f"mean funding {df['funding_ann'].mean():+.1%} ann\n{'='*86}")

    print(f"\nPERCENTILE SWEEP @ hold={HOLD_PERIODS_PRIMARY}p (7d), cost {COST_BPS}bps RT  (PRIMARY)")
    print(f"{'pctl':>5} {'n':>4} {'/yr':>5} {'short/long':>10} {'net_bps':>8} "
          f"{'dir':>6} {'fund':>6} {'win':>5} {'h1':>7} {'h2':>7} {'tot':>7} {'shrp':>5}")
    primary_rows = {}
    for p in PCTL_GRID:
        tr = run(df, p, HOLD_PERIODS_PRIMARY, COST_BPS)
        s = summarize(tr, years)
        primary_rows[p] = s
        if s["n"] == 0:
            print(f"{p:>5} {0:>4}")
            continue
        sl = f"{s['n_short']}/{s['n_long']}"
        print(f"{p:>5} {s['n']:>4} {s['per_yr']:>5} {sl:>10} "
              f"{s['net_bps']:>8} {s['dir_bps']:>6} {s['fund_bps']:>6} {s['win']:>5} "
              f"{s['h1_bps']:>7} {s['h2_bps']:>7} {s['tot']:>7} {s['sharpe']:>5}")

    print(f"\nHOLD-HORIZON ROBUSTNESS @ pctl=90 (3d/7d/14d)")
    print(f"{'hold':>5} {'n':>4} {'/yr':>5} {'net_bps':>8} {'dir':>6} {'fund':>6} {'win':>5} {'h1':>7} {'h2':>7}")
    for h in HOLD_GRID:
        tr = run(df, 90, h, COST_BPS)
        s = summarize(tr, years)
        if s["n"] == 0:
            print(f"{h:>5} {0:>4}"); continue
        print(f"{h:>5} {s['n']:>4} {s['per_yr']:>5} {s['net_bps']:>8} {s['dir_bps']:>6} "
              f"{s['fund_bps']:>6} {s['win']:>5} {s['h1_bps']:>7} {s['h2_bps']:>7}")

    return primary_rows


def verdict(eth_rows: dict):
    print(f"\n{'#'*86}\nS2 GATE 0 VERDICT (ETH)\n{'#'*86}")
    # validatable: >=30/yr AND net>0 AND both halves>0
    ok = {p: s for p, s in eth_rows.items()
          if s.get("n", 0) and s["per_yr"] >= 30 and s["net_bps"] > 0
          and s["h1_bps"] > 0 and s["h2_bps"] > 0}
    # frequency wall: which thresholds even reach 30/yr?
    freq_ok = {p: s for p, s in eth_rows.items() if s.get("n", 0) and s["per_yr"] >= 30}
    print(f"Thresholds reaching >=30 events/yr: {sorted(freq_ok.keys())}")
    print(f"Thresholds that ALSO clear cost + both time-halves: {sorted(ok.keys())}")
    # is the edge real reversion or just funding?
    for p, s in eth_rows.items():
        if s.get("n", 0):
            tag = "DIRECTIONAL>0" if s["dir_bps"] > 0 else "dir<=0 (edge is just funding=carry)"
            print(f"  pctl {p}: net {s['net_bps']:+.1f} = dir {s['dir_bps']:+.1f} + fund {s['fund_bps']:+.1f}  [{tag}]")
    print()
    if not ok:
        if not freq_ok:
            print("VERDICT: FAIL - cannot even reach 30 events/yr at any tested percentile")
            print("         while staying loose. S2's event-count wall is binding. KILL S2.")
        else:
            print("VERDICT: FAIL - thresholds reach 30/yr but none clears cost in BOTH halves.")
            print("         The reversion edge does not survive cost net of regime. KILL S2.")
    else:
        all_dir_pos = all(eth_rows[p]["dir_bps"] > 0 for p in ok)
        if all_dir_pos:
            print("VERDICT: PASS (provisional, in-sample) - a validatable threshold clears cost in")
            print("         both halves WITH positive directional component (genuine reversion, not")
            print("         just carry). NEXT: sealed prereg + regime-aware OOS before any capital.")
        else:
            print("VERDICT: MARGINAL - clears cost but the edge is the COLLECTED FUNDING, not")
            print("         directional reversion -> it is carry in disguise, redundant with")
            print("         BTC-CARRY. Not a distinct new edge. Treat as FAIL for S2's thesis.")


def main():
    eth_rows = analyse("ETH (S2 primary)", ETH_FUND, ETH_CSV)
    analyse("BTC (cross-asset check)", BTC_FUND, BTC_CSV)
    verdict(eth_rows)
    print("\n(Gate 0 only. Provisional/in-sample; sealed prereg + regime-aware OOS required")
    print(" before any capital. OI-confirmation gate omitted - would only reduce events.)")


if __name__ == "__main__":
    main()
