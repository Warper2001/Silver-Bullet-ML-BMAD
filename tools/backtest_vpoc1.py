"""VPOC-1 backtest: volume-profile value-area fade on MNQ.

Pre-registration: _bmad-output/preregistration_vpoc1_volume_profile_fade.md

Reads existing 1-min MNQ bars (no new data fetch), builds a per-session
volume profile (POC / Value Area) from a rolling lookback of N prior RTH
sessions, classifies each session's open as "balance" or "discovery" per
classic Market-Profile Open-Type logic, and trades VA-edge fades back to
POC only in balance sessions, gated by a frozen weekly-VWAP confluence
filter. One swept knob: N in {1, 3, 5, 10}, locked on dev-window PF.
"""
from __future__ import annotations

import zoneinfo
from pathlib import Path

import numpy as np
import pandas as pd

ET = zoneinfo.ZoneInfo("America/New_York")

DEV_CSV = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
HOLD_CSV = "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"

BIN_SIZE = 5.0          # index points, fixed (not swept)
VALUE_AREA_PCT = 0.70    # fixed (not swept)
STOP_MULT = 1.0          # x (VAH-VAL), fixed (not swept)
COST_PER_RT = 2.24       # MNQ, this shop's established live cost model
POINT_VALUE = 2.00       # MNQ $/pt
N_GRID = [1, 3, 5, 10]

DEV_START, DEV_END = "2025-01-01", "2025-12-31"
HOLD_START = "2026-01-01"


def load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC") if df["timestamp"].dt.tz is not None else df["timestamp"].dt.tz_localize("UTC")
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df.sort_values("timestamp").reset_index(drop=True)


def add_weekly_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous weekly VWAP over ALL bars (not RTH-filtered), reset each Monday 00:00 UTC."""
    df = df.copy()
    iso = df["timestamp"].dt.isocalendar()
    week_key = iso["year"].astype(str) + "-W" + iso["week"].astype(str)
    pv = df["typical"] * df["volume"]
    df["week_key"] = week_key.values
    df["cum_pv"] = pv.groupby(df["week_key"]).cumsum()
    df["cum_vol"] = df["volume"].groupby(df["week_key"]).cumsum()
    df["weekly_vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)
    return df


def rth_filter(df: pd.DataFrame) -> pd.DataFrame:
    et = df["timestamp"].dt.tz_convert(ET)
    is_weekday = et.dt.weekday < 5
    tod = et.dt.hour * 60 + et.dt.minute
    in_rth = (tod >= 9 * 60 + 30) & (tod < 16 * 60)
    out = df[is_weekday & in_rth].copy()
    out["session_date"] = et[is_weekday & in_rth].dt.date
    return out.reset_index(drop=True)


def session_profile(rth_bars: pd.DataFrame) -> dict:
    """One session's volume-at-price histogram, keyed by bin index (floor(typical/BIN_SIZE))."""
    bins = np.floor(rth_bars["typical"] / BIN_SIZE).astype(int)
    return rth_bars.groupby(bins)["volume"].sum().to_dict()


def poc_and_value_area(profile: dict) -> tuple[float, float, float] | None:
    """Returns (poc_price, val, vah) for a combined bin->volume profile, or None if empty."""
    if not profile:
        return None
    total = sum(profile.values())
    if total <= 0:
        return None
    poc_bin = max(profile, key=profile.get)
    included = {poc_bin}
    cum = profile[poc_bin]
    lo, hi = poc_bin, poc_bin
    while cum < VALUE_AREA_PCT * total:
        below = profile.get(lo - 1, 0)
        above = profile.get(hi + 1, 0)
        if below == 0 and above == 0:
            break
        if below >= above:
            lo -= 1
            cum += below
        else:
            hi += 1
            cum += above
        included.add(lo)
        included.add(hi)
    poc_price = (poc_bin + 0.5) * BIN_SIZE
    val = lo * BIN_SIZE
    vah = (hi + 1) * BIN_SIZE
    return poc_price, val, vah


def build_session_profiles(rth: pd.DataFrame) -> dict:
    """session_date -> per-session bin volume dict."""
    out = {}
    for date, grp in rth.groupby("session_date"):
        out[date] = session_profile(grp)
    return out


def combine_profiles(profiles: list[dict]) -> dict:
    combined: dict = {}
    for p in profiles:
        for b, v in p.items():
            combined[b] = combined.get(b, 0) + v
    return combined


def simulate(rth: pd.DataFrame, session_dates: list, per_session_profile: dict, n_lookback: int) -> list[dict]:
    trades = []
    for i, date in enumerate(session_dates):
        if i < n_lookback:
            continue
        lookback_dates = session_dates[i - n_lookback:i]
        combined = combine_profiles([per_session_profile[d] for d in lookback_dates])
        pv = poc_and_value_area(combined)
        if pv is None:
            continue
        poc, val, vah = pv
        if vah - val < BIN_SIZE or poc <= val or poc >= vah:
            continue  # degenerate guard

        day_bars = rth[rth["session_date"] == date].reset_index(drop=True)
        if day_bars.empty:
            continue
        open_px = day_bars.iloc[0]["open"]
        if not (val <= open_px <= vah):
            continue  # "discovery" session -- no trade

        # Frozen rule (pre-reg §4d): first qualifying bar each session only -- once the
        # first triggered trade resolves (TP/SL/TIME), no further entries this session.
        in_trade = None
        for _, bar in day_bars.iterrows():
            if in_trade is None:
                wv = bar["weekly_vwap"]
                if pd.isna(wv):
                    continue
                if bar["low"] <= val and bar["close"] <= wv:
                    stop = val - STOP_MULT * (vah - val)
                    in_trade = {"dir": 1, "entry": bar["close"], "target": poc, "stop": stop,
                                "entry_ts": bar["timestamp"], "n_lookback": n_lookback}
                elif bar["high"] >= vah and bar["close"] >= wv:
                    stop = vah + STOP_MULT * (vah - val)
                    in_trade = {"dir": -1, "entry": bar["close"], "target": poc, "stop": stop,
                                "entry_ts": bar["timestamp"], "n_lookback": n_lookback}
                continue

            d = in_trade["dir"]
            hit_tp = (bar["high"] >= in_trade["target"]) if d == 1 else (bar["low"] <= in_trade["target"])
            hit_sl = (bar["low"] <= in_trade["stop"]) if d == 1 else (bar["high"] >= in_trade["stop"])
            if hit_tp and hit_sl:
                exit_px, reason = in_trade["stop"], "SL(tie)"
            elif hit_sl:
                exit_px, reason = in_trade["stop"], "SL"
            elif hit_tp:
                exit_px, reason = in_trade["target"], "TP"
            else:
                continue
            pnl_pts = (exit_px - in_trade["entry"]) * d
            pnl = pnl_pts * POINT_VALUE - COST_PER_RT
            trades.append({**in_trade, "exit": exit_px, "exit_ts": bar["timestamp"], "reason": reason, "pnl": pnl})
            in_trade = None
            break

        if in_trade is not None:
            last = day_bars.iloc[-1]
            exit_px = last["close"]
            pnl_pts = (exit_px - in_trade["entry"]) * in_trade["dir"]
            pnl = pnl_pts * POINT_VALUE - COST_PER_RT
            trades.append({**in_trade, "exit": exit_px, "exit_ts": last["timestamp"], "reason": "TIME", "pnl": pnl})
    return trades


def trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "ev": float("nan"), "pf": float("nan"), "wr": float("nan"), "be_wr": float("nan"),
                "worst_month": float("nan")}
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    ev = pnls.mean()
    gains = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    wr = (pnls > 0).mean()
    rrs = []
    for t in trades:
        risk = abs(t["entry"] - t["stop"])
        reward = abs(t["target"] - t["entry"])
        if risk > 0:
            rrs.append(reward / risk)
    avg_rr = np.mean(rrs) if rrs else float("nan")
    be_wr = 1.0 / (avg_rr + 1.0) if avg_rr and not np.isnan(avg_rr) else float("nan")
    df = pd.DataFrame(trades)
    df["month"] = pd.to_datetime(df["exit_ts"]).dt.to_period("M")
    worst_month = df.groupby("month")["pnl"].mean().min()
    return {"n": n, "ev": ev, "pf": pf, "wr": wr, "be_wr": be_wr, "worst_month": worst_month, "avg_rr": avg_rr}


def run_window(rth: pd.DataFrame, start: str, end: str | None, n_lookback: int) -> list[dict]:
    """rth must already include lookback bars before `start` for profile continuity."""
    session_dates = sorted(rth["session_date"].unique())
    per_session_profile = build_session_profiles(rth)
    all_trades = simulate(rth, session_dates, per_session_profile, n_lookback)
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date() if end else None
    out = [t for t in all_trades if t["entry_ts"].tz_convert(ET).date() >= start_d
           and (end_d is None or t["entry_ts"].tz_convert(ET).date() <= end_d)]
    return out


def main() -> None:
    dev_raw = load_bars(DEV_CSV)
    hold_raw = load_bars(HOLD_CSV)
    full = pd.concat([dev_raw, hold_raw], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    full = add_weekly_vwap(full)
    rth = rth_filter(full)

    print(f"Loaded {len(full)} total 1-min bars ({len(rth)} RTH bars, "
          f"{rth['session_date'].nunique()} sessions)")

    lines = ["# VPOC-1 Backtest Results\n",
              "Pre-registration: `_bmad-output/preregistration_vpoc1_volume_profile_fade.md`\n",
              f"Data: {len(rth)} RTH 1-min bars, {rth['session_date'].nunique()} sessions, "
              f"MNQ 2025-01-01 -> latest available.\n",
              "## Lookback sweep (dev window 2025, pre-declared grid, locked on max PF)\n",
              "| N | Trades | EV/trade | PF | WR | be_WR | Worst month avg |",
              "|---|---|---|---|---|---|---|"]

    dev_results = {}
    for n in N_GRID:
        trades = run_window(rth, DEV_START, DEV_END, n)
        stats = trade_stats(trades)
        dev_results[n] = (trades, stats)
        lines.append(f"| {n} | {stats['n']} | ${stats['ev']:.2f} | {stats['pf']:.3f} | "
                      f"{stats['wr']*100:.1f}% | {stats['be_wr']*100:.1f}% | ${stats['worst_month']:.2f} |"
                      if stats["n"] > 0 else f"| {n} | 0 | - | - | - | - | - |")

    valid = {n: s for n, (t, s) in dev_results.items() if s["n"] > 0}
    if not valid:
        lines.append("\n**No trades fired at any N. VPOC-1 FAILS Gate 0 -- signal never triggers on this universe/window.**\n")
        Path("_bmad-output/vpoc1_backtest_results.md").write_text("\n".join(lines))
        print("No trades at any N -- see results file.")
        return

    locked_n = max(valid, key=lambda n: valid[n]["pf"] if not np.isnan(valid[n]["pf"]) else -1)
    lines.append(f"\n**Locked N = {locked_n}**\n")

    dev_trades, dev_stats = dev_results[locked_n]
    lines.append("## Gate 0 (dev window 2025, locked N)\n")
    if dev_stats["n"] < 20:
        verdict0 = "INCONCLUSIVE (N<20)"
    elif dev_stats["ev"] > 0 and dev_stats["pf"] >= 1.20 and dev_stats["wr"] >= dev_stats["be_wr"] + 0.05:
        verdict0 = "PASS"
    else:
        verdict0 = "FAIL"
    lines.append(f"N={dev_stats['n']}, EV=${dev_stats['ev']:.2f}, PF={dev_stats['pf']:.3f}, "
                 f"WR={dev_stats['wr']*100:.1f}%, be_WR={dev_stats['be_wr']*100:.1f}%, "
                 f"worst_month_avg=${dev_stats['worst_month']:.2f}, avg_RR={dev_stats['avg_rr']:.2f}")
    lines.append(f"\n**Gate 0 verdict: {verdict0}**\n")

    lines.append("## Gate 1 (holdout 2026 YTD, same locked N)\n")
    if verdict0 == "PASS":
        hold_trades = run_window(rth, HOLD_START, None, locked_n)
        hold_stats = trade_stats(hold_trades)
        if hold_stats["n"] < 10:
            verdict1 = "INCONCLUSIVE (N<10)"
        elif hold_stats["pf"] >= 1.10 and hold_stats["wr"] >= hold_stats["be_wr"] + 0.03:
            verdict1 = "PASS"
        else:
            verdict1 = "FAIL"
        lines.append(f"N={hold_stats['n']}, PF={hold_stats['pf']:.3f}, WR={hold_stats['wr']*100:.1f}%, "
                     f"be_WR={hold_stats['be_wr']*100:.1f}%")
        lines.append(f"\n**Gate 1 verdict: {verdict1}**\n")
    else:
        lines.append(f"NOT EVALUATED -- Gate 0 did not PASS (verdict: {verdict0}); holdout not spent per pre-registration.\n")

    lines.append("## Diagnostic: weekly-VWAP confluence filter impact (informational only)\n")
    trades_no_filter = []
    session_dates = sorted(rth["session_date"].unique())
    per_session_profile = build_session_profiles(rth)
    for i, date in enumerate(session_dates):
        if i < locked_n:
            continue
        lookback_dates = session_dates[i - locked_n:i]
        combined = combine_profiles([per_session_profile[d] for d in lookback_dates])
        pv = poc_and_value_area(combined)
        if pv is None:
            continue
        poc, val, vah = pv
        if vah - val < BIN_SIZE or poc <= val or poc >= vah:
            continue
        day_bars = rth[rth["session_date"] == date].reset_index(drop=True)
        if day_bars.empty or not (val <= day_bars.iloc[0]["open"] <= vah):
            continue
        for _, bar in day_bars.iterrows():
            if bar["low"] <= val or bar["high"] >= vah:
                trades_no_filter.append(1)
                break
    lines.append(f"Sessions with a VA-edge touch in a balance session (locked N={locked_n}), "
                 f"before the weekly-VWAP filter: {len(trades_no_filter)}. "
                 f"After filter (actual trades taken): {dev_stats['n'] + (hold_stats['n'] if verdict0=='PASS' else 0)}.\n")

    Path("_bmad-output/vpoc1_backtest_results.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\n-> _bmad-output/vpoc1_backtest_results.md")


if __name__ == "__main__":
    main()
