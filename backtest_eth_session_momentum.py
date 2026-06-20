#!/usr/bin/env python3
"""
ETH Session-Momentum Amplitude — GATE 0 cost-wall backtest (S1).

Hypothesis (from technical-best-strategy-kraken-pro-perpetual-futures-research-2026-06-20):
    ETH runs ~1.5-2x BTC realized vol. A session-conditioned momentum pattern -
    the sign of one regional session's return predicts the *next* session's
    direction - produces a per-trade gross spread in ETH large enough to clear
    the ~10 bps round-trip taker cost, where the identical pattern in BTC does
    not. Concretely: take the sign of the EU-session return, hold an ETH-perp
    position in that direction through the following US session.

GATE 0 question (the ONLY question this script answers):
    Does the edge survive REAL cost (~10 bps round-trip taker) on OUR OWN ETH
    1-min data? If net per-trade expectancy <= 0 at 10 bps, the candidate is
    KILLED here - no tuning, no favourable-subset restriction (see
    memory: feedback_iteration_loop_pattern, feedback_derive_dont_assert_one_knob).

Methodology discipline encoded here:
  * ONE swept knob: the amplitude gate |eu_ret| >= threshold. Nothing else is tuned.
  * The sweep is reported in FULL. Picking the single best threshold is in-sample
    optimisation; the honest PASS condition is a CONTIGUOUS band of thresholds that
    clears cost AND stays positive in BOTH time-halves AND beats the BTC control.
    A knife-edge single-threshold "win" is a FAIL signal, not a pass.
  * BTC is run as a NEGATIVE CONTROL - the mechanism claim is ETH-works / BTC-doesn't.
  * No look-ahead: the EU session fully closes (<= EU_end) before the US entry open.

This is a Gate-0 reproduction only. It is NOT a pre-registration and places no
orders. If (and only if) Gate 0 passes does a sealed prereg + regime-aware OOS follow.

Usage:
    .venv/bin/python backtest_eth_session_momentum.py
    .venv/bin/python backtest_eth_session_momentum.py --eu 07:00-13:00 --us 13:00-21:00
    .venv/bin/python backtest_eth_session_momentum.py --cost-bps 10 --maker-cost-bps 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import calc_max_drawdown_pct, calc_sharpe

ETH_CSV = Path("data/kraken/PF_ETHUSD_1min.csv")
BTC_CSV = Path("data/kraken/PF_XBTUSD_1min.csv")
REPORTS_DIR = Path("data/reports")

# Amplitude-gate sweep (the single knob). Values are |EU-session return| floors, in bps.
# 0 bps = trade every day; higher = only act on large EU moves.
AMPLITUDE_GRID_BPS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500]


def parse_window(s: str) -> tuple[int, int]:
    """'07:00-13:00' -> (start_minute_of_day, end_minute_of_day), end exclusive."""
    a, b = s.split("-")

    def to_min(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    return to_min(a), to_min(b)


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df


def session_open_close(df: pd.DataFrame, win: tuple[int, int], prefix: str) -> pd.DataFrame:
    """Per-date open (first bar) and close (last bar) within [start, end)."""
    start, end = win
    m = (df["minute"] >= start) & (df["minute"] < end)
    g = df[m].groupby("date")
    out = g.agg(**{f"{prefix}_open": ("open", "first"), f"{prefix}_close": ("close", "last")})
    out[f"{prefix}_n"] = g.size()
    return out


def build_daily(df: pd.DataFrame, eu_win, us_win, min_bars_frac=0.5) -> pd.DataFrame:
    """One row per date: EU-session return (signal) and US-session return (trade)."""
    if eu_win[1] > us_win[0]:
        raise ValueError("EU session must close at or before US session opens (no look-ahead).")
    eu = session_open_close(df, eu_win, "eu")
    us = session_open_close(df, us_win, "us")
    daily = eu.join(us, how="inner").reset_index()

    # Require a session to be reasonably complete (guards thin/halted days).
    eu_full = (eu_win[1] - eu_win[0]) * min_bars_frac
    us_full = (us_win[1] - us_win[0]) * min_bars_frac
    daily = daily[(daily["eu_n"] >= eu_full) & (daily["us_n"] >= us_full)].copy()

    daily["eu_ret"] = daily["eu_close"] / daily["eu_open"] - 1.0
    daily["us_ret"] = daily["us_close"] / daily["us_open"] - 1.0
    daily = daily.dropna(subset=["eu_ret", "us_ret"]).sort_values("date").reset_index(drop=True)
    return daily


def run_sweep(daily: pd.DataFrame, cost_bps: float, direction: int = +1) -> pd.DataFrame:
    """Sweep the amplitude gate. direction=+1 momentum (signal=sign(eu_ret)),
    direction=-1 reversal (diagnostic mirror)."""
    cost = cost_bps / 1e4
    n_total = len(daily)
    span_days = (daily["date"].iloc[-1] - daily["date"].iloc[0]).days or 1
    years = span_days / 365.25
    half = n_total // 2

    rows = []
    for thr_bps in AMPLITUDE_GRID_BPS:
        thr = thr_bps / 1e4
        mask = daily["eu_ret"].abs() >= thr
        sub = daily[mask]
        n = len(sub)
        if n == 0:
            rows.append(dict(thr_bps=thr_bps, n=0))
            continue
        signal = direction * np.sign(sub["eu_ret"].values)
        gross = signal * sub["us_ret"].values
        net = gross - cost

        equity = np.cumprod(1.0 + net)
        mean_net_bps = float(np.mean(net) * 1e4)
        # Time-split robustness: expectancy in each chronological half of the FULL sample.
        idx = sub.index.values
        h1 = net[idx < half]
        h2 = net[idx >= half]
        mean_h1_bps = float(np.mean(h1) * 1e4) if len(h1) else float("nan")
        mean_h2_bps = float(np.mean(h2) * 1e4) if len(h2) else float("nan")

        rows.append(dict(
            thr_bps=thr_bps,
            n=n,
            trades_per_yr=round(n / years, 1),
            winrate=round(float(np.mean(gross > 0)), 3),
            gross_bps=round(float(np.mean(gross) * 1e4), 2),
            net_bps=round(mean_net_bps, 2),
            net_h1_bps=round(mean_h1_bps, 2),
            net_h2_bps=round(mean_h2_bps, 2),
            tot_ret=round(float(equity[-1] - 1.0), 4),
            sharpe=round(calc_sharpe(list(net)), 2),
            maxdd=round(calc_max_drawdown_pct(list(equity)), 3),
        ))
    return pd.DataFrame(rows)


def fmt_sweep(df: pd.DataFrame) -> str:
    cols = ["thr_bps", "n", "trades_per_yr", "winrate", "gross_bps",
            "net_bps", "net_h1_bps", "net_h2_bps", "tot_ret", "sharpe", "maxdd"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_string(index=False)


def verdict(eth_sweep: pd.DataFrame, btc_sweep: pd.DataFrame, min_trades_per_yr=30.0) -> list[str]:
    """Honest Gate-0 verdict: a CONTIGUOUS band of thresholds must clear cost,
    stay positive in BOTH halves, retain >=30 trades/yr, and beat BTC control."""
    out = []
    valid = eth_sweep[(eth_sweep["n"] > 0) & (eth_sweep["trades_per_yr"] >= min_trades_per_yr)]
    clears = valid[(valid["net_bps"] > 0) & (valid["net_h1_bps"] > 0) & (valid["net_h2_bps"] > 0)]

    # Headline diagnostic: thresholds whose POOLED net>0 but one time-half is <=0.
    # That is the single-regime artifact our methodology is built to reject.
    decay = valid[(valid["net_bps"] > 0) &
                  ((valid["net_h1_bps"] <= 0) | (valid["net_h2_bps"] <= 0))]
    if len(decay):
        for _, r in decay.iterrows():
            out.append(f"REGIME-DECAY FLAG @thr={int(r['thr_bps'])}bps: pooled net "
                       f"{r['net_bps']:+.1f} bps LOOKS positive but splits "
                       f"H1={r['net_h1_bps']:+.1f} / H2={r['net_h2_bps']:+.1f} bps "
                       f"- edge is half-sample only, NOT a stable edge.")
        out.append("")

    if len(clears) == 0:
        out.append("VERDICT: FAIL - no amplitude threshold clears 10 bps net while")
        out.append("         staying positive in both time-halves with >=30 trades/yr.")
        out.append("         KILL S1 per Gate 0. Do not tune. Runner-up S2 is next.")
        return out

    # contiguity over the grid
    grid = AMPLITUDE_GRID_BPS
    clearing_thrs = sorted(clears["thr_bps"].tolist())
    pos = [grid.index(t) for t in clearing_thrs]
    contiguous = all(pos[i + 1] - pos[i] == 1 for i in range(len(pos) - 1))
    band_n = len(clearing_thrs)

    # BTC control: best net_bps among >=30/yr thresholds should be clearly worse / negative
    btc_valid = btc_sweep[(btc_sweep["n"] > 0) & (btc_sweep["trades_per_yr"] >= min_trades_per_yr)]
    btc_best = float(btc_valid["net_bps"].max()) if len(btc_valid) else float("nan")
    eth_best = float(clears["net_bps"].max())

    out.append(f"ETH thresholds clearing all conditions (>=30/yr, net>0, both halves>0): "
               f"{clearing_thrs} bps  (contiguous={contiguous}, n_band={band_n})")
    out.append(f"ETH best net/trade in band: {eth_best:.2f} bps   |   "
               f"BTC control best net/trade: {btc_best:.2f} bps")

    if band_n >= 2 and contiguous and (np.isnan(btc_best) or eth_best > btc_best + 5.0):
        out.append("VERDICT: PASS (provisional) - a contiguous threshold band survives 10 bps,")
        out.append("         is stable across both time-halves, and beats the BTC control.")
        out.append("         NEXT: seal a prereg (prereg_seal.py) + regime-aware OOS before any capital.")
        out.append("         CAUTION: this is in-sample; the sweep located the band, it did not validate it.")
    elif band_n >= 1:
        out.append("VERDICT: MARGINAL / LIKELY FAIL - only a knife-edge threshold clears, or BTC")
        out.append("         control is not clearly worse. This is the fat-tail/over-fit pattern our")
        out.append("         methodology distrusts (feedback_iteration_loop_pattern). Treat as FAIL")
        out.append("         unless a mechanism reason justifies the specific threshold.")
    return out


def analyse(name: str, df: pd.DataFrame, eu_win, us_win, cost_bps, maker_cost_bps) -> dict:
    daily = build_daily(df, eu_win, us_win)
    print(f"\n{'='*78}\n{name}: {len(daily)} complete session-days  "
          f"({daily['date'].iloc[0]} -> {daily['date'].iloc[-1]})\n{'='*78}")

    sweep_taker = run_sweep(daily, cost_bps, direction=+1)
    print(f"\n[{name}] MOMENTUM, taker cost {cost_bps} bps RT  (PRIMARY hypothesis)")
    print(fmt_sweep(sweep_taker))

    sweep_maker = run_sweep(daily, maker_cost_bps, direction=+1)
    print(f"\n[{name}] MOMENTUM, maker cost {maker_cost_bps} bps RT  (if limit fills realistic)")
    print(fmt_sweep(sweep_maker))

    sweep_rev = run_sweep(daily, cost_bps, direction=-1)
    print(f"\n[{name}] REVERSAL (diagnostic mirror, taker {cost_bps} bps) - "
          f"if THIS is the winner, the momentum thesis is wrong")
    print(fmt_sweep(sweep_rev))

    return {"daily": daily, "taker": sweep_taker, "maker": sweep_maker, "reversal": sweep_rev}


def main():
    ap = argparse.ArgumentParser(description="ETH session-momentum amplitude - Gate 0 cost-wall test")
    ap.add_argument("--eu", default="07:00-13:00", help="EU signal session window UTC (default 07:00-13:00)")
    ap.add_argument("--us", default="13:00-21:00", help="US trade session window UTC (default 13:00-21:00)")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="round-trip taker cost in bps (default 10)")
    ap.add_argument("--maker-cost-bps", type=float, default=4.0, help="round-trip maker cost in bps (default 4)")
    ap.add_argument("--no-btc", action="store_true", help="skip the BTC negative control")
    args = ap.parse_args()

    eu_win = parse_window(args.eu)
    us_win = parse_window(args.us)

    print("ETH SESSION-MOMENTUM AMPLITUDE - GATE 0 COST-WALL TEST")
    print(f"EU signal window (UTC): {args.eu}   US trade window (UTC): {args.us}")
    print(f"Cost: taker {args.cost_bps} bps RT, maker {args.maker_cost_bps} bps RT")
    print("Signal = sign(EU-session return); hold ETH perp that direction through US session.")
    print("Single swept knob = amplitude gate |EU return| >= threshold.")

    eth_df = load_bars(ETH_CSV)
    eth_res = analyse("ETH", eth_df, eu_win, us_win, args.cost_bps, args.maker_cost_bps)

    btc_res = None
    if not args.no_btc:
        btc_df = load_bars(BTC_CSV)
        btc_res = analyse("BTC (negative control)", btc_df, eu_win, us_win,
                          args.cost_bps, args.maker_cost_bps)

    print(f"\n{'#'*78}\nGATE 0 VERDICT\n{'#'*78}")
    btc_taker = btc_res["taker"] if btc_res is not None else eth_res["taker"].assign(net_bps=-999, n=0, trades_per_yr=0)
    for line in verdict(eth_res["taker"], btc_taker):
        print(line)
    print(f"\n(Reminder: PASS here is PROVISIONAL/in-sample. Gate 0 only rules out the")
    print(f" cost wall. A sealed prereg + regime-aware OOS is required before any capital.)")


if __name__ == "__main__":
    main()
