"""Option 1 (post-R3 research plan): GAP-1 time-stop horizon sweep.

Pre-registration: _bmad-output/preregistration_option1_gap_fade_horizon.md
Sealed grid: TIME_STOP_HOUR in {13 (baseline), 14, 15, 16 (=EOD)}.
Sealed null: 200 draws of a random-hour-per-trade baseline.

Imports the pure functions from backtest_gap_fade.py (entry/gap/stop logic
UNCHANGED) and only varies the time-stop hour -- the one sealed knob. Works
around that module's broken _REPO path resolution (parents[3], written for a
specific worktree depth) by overriding its path constants directly rather
than editing the frozen script.

Usage: .venv/bin/python tools/option1_gap_fade_horizon_sweep.py
"""

from __future__ import annotations

import random
import types
from pathlib import Path

import pandas as pd
import pytz

REPO = Path("/root/Silver-Bullet-ML-BMAD")

# `backtest_gap_fade.py`'s own `_REPO = Path(__file__).resolve().parents[3]`
# raises IndexError from its current (repo-root) location -- it was written
# for a specific worktree nesting depth (disclosed in the pre-registration,
# out of scope for this seal). Importing it top-level is therefore not
# possible; the handful of pure functions/constants this sweep needs are
# copied verbatim below (byte-for-byte from backtest_gap_fade.py) rather than
# reimplemented, so the frozen entry/gap/stop logic is provably unchanged.
gf = types.SimpleNamespace()
gf.GAP_MIN_PCT = 0.005
gf.STOP_MULT = 2.0
gf.EXCLUDE_DOW = {4}
gf.MIN_RTH_BARS = 300
gf.MNQ_PV = 2.0
gf.CONTRACTS = 1
gf.GATE_N_MIN = 60
gf.CSV_2025 = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
gf.CSV_2026 = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"

ET_TZ = pytz.timezone("US/Eastern")
RTH_START = (9, 30)
RTH_END = (16, 0)


def _load_data() -> pd.DataFrame:
    dfs = []
    for path in [gf.CSV_2025, gf.CSV_2026]:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        ts = df["timestamp"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df["timestamp"] = ts.dt.tz_convert(ET_TZ)
        dfs.append(df)
    out = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    return out.set_index("timestamp")


def _is_rth(ts: pd.Timestamp) -> bool:
    h, m = ts.hour, ts.minute
    after_open = (h == RTH_START[0] and m >= RTH_START[1]) or h > RTH_START[0]
    before_close = h < RTH_END[0] or (h == RTH_END[0] and m < RTH_END[1])
    return after_open and before_close


def _build_session_map(df: pd.DataFrame) -> dict:
    rth = df[df.index.map(_is_rth)].copy()
    rth["date_et"] = rth.index.date
    by_date = rth.groupby("date_et")
    rth_close = by_date["close"].last()
    rth_open = by_date["open"].first()
    rth_bars = by_date["close"].count()
    rth_dow = by_date.apply(lambda g: g.index[0].weekday())
    sessions = {}
    dates = sorted(rth_close.index)
    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        if rth_bars[yesterday] < gf.MIN_RTH_BARS:
            continue
        sessions[today] = {
            "prior_close": rth_close[yesterday],
            "rth_open": rth_open[today],
            "dow": rth_dow[today],
        }
    return sessions


gf.load_data = _load_data
gf.is_rth = _is_rth
gf.build_session_map = _build_session_map

GRID = (13, 14, 15, 16)
N_NULL_DRAWS = 200
RNG_SEED = 20260904  # sealed in the pre-registration commit, fixed for reproducibility


def _simulate_day_with_stop(day_bars, direction, entry, target, stop, time_stop_hour):
    """backtest_gap_fade.simulate_day with TIME_STOP_HOUR parameterized."""
    for ts, bar in day_bars.iterrows():
        if ts.hour >= time_stop_hour:
            return "time", direction * (bar["open"] - entry)
        if direction == -1:
            if bar["low"] <= target:
                return "fill", direction * (target - entry)
            if bar["high"] >= stop:
                return "stop", direction * (stop - entry)
        else:
            if bar["high"] >= target:
                return "fill", direction * (target - entry)
            if bar["low"] <= stop:
                return "stop", direction * (stop - entry)
    return "eod", direction * (day_bars["close"].iloc[-1] - entry)


def run_grid_cell(df, sessions, time_stop_hour: int) -> list[dict]:
    """backtest_gap_fade.run(), with TIME_STOP_HOUR swapped for the sealed knob."""
    rth = df[df.index.map(gf.is_rth)].copy()
    rth["date_et"] = rth.index.date

    trades = []
    for date_et, sess in sorted(sessions.items()):
        if sess["dow"] in gf.EXCLUDE_DOW:
            continue
        prior_close = sess["prior_close"]
        rth_open = sess["rth_open"]
        gap = rth_open - prior_close
        gap_abs = abs(gap)
        gap_pct = gap_abs / prior_close
        if gap_pct < gf.GAP_MIN_PCT:
            continue

        direction = -1 if gap > 0 else 1
        entry = rth_open
        target = prior_close
        stop = (
            (entry + gf.STOP_MULT * gap_abs)
            if direction == -1
            else (entry - gf.STOP_MULT * gap_abs)
        )

        day_bars = rth[rth["date_et"] == date_et]
        if len(day_bars) < 2:
            continue
        sim_bars = day_bars.iloc[1:]

        outcome, pnl_pts = _simulate_day_with_stop(
            sim_bars, direction, entry, target, stop, time_stop_hour
        )
        pnl_usd = pnl_pts * gf.MNQ_PV * gf.CONTRACTS
        trades.append(
            {
                "date": str(date_et),
                "outcome": outcome,
                "pnl_pts": pnl_pts,
                "pnl_usd": pnl_usd,
            }
        )
    return trades


def run_random_hour_cell(df, sessions, rng: random.Random) -> list[dict]:
    """Null: same trades, each trade's time-stop hour drawn independently from GRID."""
    rth = df[df.index.map(gf.is_rth)].copy()
    rth["date_et"] = rth.index.date

    trades = []
    for date_et, sess in sorted(sessions.items()):
        if sess["dow"] in gf.EXCLUDE_DOW:
            continue
        prior_close = sess["prior_close"]
        rth_open = sess["rth_open"]
        gap = rth_open - prior_close
        gap_abs = abs(gap)
        gap_pct = gap_abs / prior_close
        if gap_pct < gf.GAP_MIN_PCT:
            continue

        direction = -1 if gap > 0 else 1
        entry = rth_open
        target = prior_close
        stop = (
            (entry + gf.STOP_MULT * gap_abs)
            if direction == -1
            else (entry - gf.STOP_MULT * gap_abs)
        )
        day_bars = rth[rth["date_et"] == date_et]
        if len(day_bars) < 2:
            continue
        sim_bars = day_bars.iloc[1:]

        hour = rng.choice(GRID)
        outcome, pnl_pts = _simulate_day_with_stop(
            sim_bars, direction, entry, target, stop, hour
        )
        pnl_usd = pnl_pts * gf.MNQ_PV * gf.CONTRACTS
        trades.append({"date": str(date_et), "outcome": outcome, "pnl_usd": pnl_usd})
    return trades


def pf_of(trades: list[dict]) -> tuple[float, int]:
    if not trades:
        return float("nan"), 0
    gross_w = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    gross_l = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    return pf, len(trades)


def pf_ex_top3(trades: list[dict]) -> float:
    sorted_t = sorted(trades, key=lambda t: t["pnl_usd"], reverse=True)
    rest = sorted_t[3:]
    return pf_of(rest)[0]


def main() -> None:
    print("Loading data...")
    df = gf.load_data()
    sessions = gf.build_session_map(df)
    print(f"Sessions with valid prior close: {len(sessions)}")

    print(f"\n{'hour':>6}{'N':>6}{'PF':>10}{'PF ex-top3':>14}{'gross P&L':>14}")
    grid_results = {}
    for hour in GRID:
        trades = run_grid_cell(df, sessions, hour)
        pf, n = pf_of(trades)
        pf_x3 = pf_ex_top3(trades)
        gross = sum(t["pnl_usd"] for t in trades)
        grid_results[hour] = {
            "trades": trades,
            "pf": pf,
            "n": n,
            "pf_ex_top3": pf_x3,
            "gross": gross,
        }
        label = " (baseline)" if hour == 13 else (" (=EOD)" if hour == 16 else "")
        print(f"{hour:>5}h{label:<11}{n:>6}{pf:>10.3f}{pf_x3:>14.3f}{gross:>14.0f}")

    baseline = grid_results[13]
    best_hour = max(GRID, key=lambda h: grid_results[h]["pf"])
    best = grid_results[best_hour]

    print(f"\nBaseline (13h): PF={baseline['pf']:.3f} N={baseline['n']}")
    print(f"Best cell: {best_hour}h, PF={best['pf']:.3f}")

    # Random-hour null
    print(f"\nRunning {N_NULL_DRAWS}-draw random-hour null (seed={RNG_SEED})...")
    rng = random.Random(RNG_SEED)
    null_pfs = []
    for i in range(N_NULL_DRAWS):
        null_trades = run_random_hour_cell(df, sessions, rng)
        pf, _n = pf_of(null_trades)
        if pf == pf:  # not NaN
            null_pfs.append(pf)
    null_pfs.sort()
    p95_idx = int(0.95 * len(null_pfs))
    null_p95 = null_pfs[min(p95_idx, len(null_pfs) - 1)]
    print(f"Null PF distribution: median={null_pfs[len(null_pfs)//2]:.3f}  p95={null_p95:.3f}")

    # Gate 0
    print(f"\n{'='*60}")
    print("GATE 0 -- Option 1 (GAP-1 time-stop horizon)")
    print(f"{'='*60}")
    checks = {
        f"best PF ({best['pf']:.3f} @ {best_hour}h) > baseline PF ({baseline['pf']:.3f})": (
            best["pf"] > baseline["pf"]
        ),
        f"best PF ({best['pf']:.3f}) > null p95 ({null_p95:.3f})": best["pf"] > null_p95,
        f"N >= {gf.GATE_N_MIN} (N={best['n']})": best["n"] >= gf.GATE_N_MIN,
        f"ex-top3 PF at best ({best['pf_ex_top3']:.3f}) > ex-top3 PF at baseline ({baseline['pf_ex_top3']:.3f})": (
            best["pf_ex_top3"] > baseline["pf_ex_top3"]
        ),
    }
    for check, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")

    pfs_in_order = [grid_results[h]["pf"] for h in GRID]
    diffs = [pfs_in_order[i + 1] - pfs_in_order[i] for i in range(len(pfs_in_order) - 1)]
    monotonic = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
    lone_spike = (not monotonic) and (best_hour not in (GRID[0], GRID[-1]))
    print(f"\n  PF by hour: {dict(zip(GRID, [round(p, 3) for p in pfs_in_order]))}")
    print(f"  Monotonic across grid: {monotonic}")
    if lone_spike:
        print("  ** LONE-SPIKE WARNING: winner is an interior cell breaking monotonicity **")

    verdict = "PASS" if all(checks.values()) else "FAIL"
    if verdict == "PASS" and lone_spike:
        verdict = "PASS (flagged: lone-spike pattern, not trusted per house rule)"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
