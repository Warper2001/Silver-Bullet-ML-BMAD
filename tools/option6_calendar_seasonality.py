"""Option 6: MNQ calendar seasonality Gate-0 screen.

Pre-registration: _bmad-output/preregistration_option6_calendar_seasonality.md
Sealed: 3 hypotheses (turn-of-month, day-of-week, pre-holiday), compared against
the UNCONDITIONAL ALWAYS-LONG baseline (not zero), Bonferroni alpha=0.05/7
(99.29th null percentile), N floor 100/cell, fat-day robustness check.

Unit: RTH open -> RTH close, 1ct, $4.00 round turn. Vectorized; no
Tier2StreamingTrader involvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytz

REPO = "/root/Silver-Bullet-ML-BMAD"
FILES = [
    f"{REPO}/data/mim_x/mnq_1min_2021_2024_frontmonth.csv",
    f"{REPO}/data/processed/dollar_bars/1_minute/mnq_1min_2025.csv",
    f"{REPO}/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv",
]
ET = pytz.timezone("US/Eastern")
RTH_START, RTH_END = (9, 30), (16, 0)
MNQ_PV = 2.0
RT_COST = 4.00
N_NULL = 2000
N_TESTS = 7
ALPHA = 0.05 / N_TESTS          # 0.00714
NULL_PCTL = 100 * (1 - ALPHA)   # 99.29
N_FLOOR = 100
RNG_SEED = 20260906

# US market holidays 2021-2026 (NYSE/CME equity-index holiday closes).
HOLIDAYS = {
    "2021-01-01","2021-01-18","2021-02-15","2021-04-02","2021-05-31","2021-07-05","2021-09-06","2021-11-25","2021-12-24",
    "2022-01-17","2022-02-21","2022-04-15","2022-05-30","2022-06-20","2022-07-04","2022-09-05","2022-11-24","2022-12-26",
    "2023-01-02","2023-01-16","2023-02-20","2023-04-07","2023-05-29","2023-06-19","2023-07-04","2023-09-04","2023-11-23","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-03-29","2024-05-27","2024-06-19","2024-07-04","2024-09-02","2024-11-28","2024-12-25",
    "2025-01-01","2025-01-20","2025-02-17","2025-04-18","2025-05-26","2025-06-19","2025-07-04","2025-09-01","2025-11-27","2025-12-25",
    "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25","2026-06-19","2026-07-03","2026-09-07",
}


def load_sessions() -> pd.DataFrame:
    """One row per trading day: RTH open, RTH close, net day-session P&L (1ct, long)."""
    frames = []
    for f in FILES:
        df = pd.read_csv(f, usecols=["timestamp", "open", "close"], parse_dates=["timestamp"])
        ts = df["timestamp"]
        ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        df["timestamp"] = ts.dt.tz_convert(ET)
        frames.append(df)
    df = pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

    h, m = df["timestamp"].dt.hour, df["timestamp"].dt.minute
    in_rth = ((h > RTH_START[0]) | ((h == RTH_START[0]) & (m >= RTH_START[1]))) & (
        (h < RTH_END[0]) | ((h == RTH_END[0]) & (m < RTH_END[1]))
    )
    rth = df[in_rth].copy()
    rth["date"] = rth["timestamp"].dt.date

    g = rth.groupby("date")
    sess = pd.DataFrame({
        "rth_open": g["open"].first(),
        "rth_close": g["close"].last(),
        "bars": g["close"].count(),
    }).reset_index()
    sess = sess[sess["bars"] >= 300].reset_index(drop=True)  # drop half-days/degenerate sessions
    sess["pnl"] = (sess["rth_close"] - sess["rth_open"]) * MNQ_PV - RT_COST
    sess["ts"] = pd.to_datetime(sess["date"])
    sess["dow"] = sess["ts"].dt.weekday
    sess["ym"] = sess["ts"].dt.to_period("M")
    return sess


def tag_cells(sess: pd.DataFrame) -> pd.DataFrame:
    # turn-of-month: last session of a month, and first three of the next
    sess = sess.copy()
    sess["is_last_of_month"] = sess["ym"] != sess["ym"].shift(-1)
    rank_in_month = sess.groupby("ym").cumcount()
    sess["is_first3_of_month"] = rank_in_month < 3
    sess["TOM"] = sess["is_last_of_month"] | sess["is_first3_of_month"]

    # pre-holiday: a holiday date falls strictly between this session and the next
    import datetime as _dt

    hol_set = {_dt.date.fromisoformat(h) for h in HOLIDAYS}
    dates = list(sess["ts"].dt.date)
    pre = []
    for i, d0 in enumerate(dates):
        if i >= len(dates) - 1:
            pre.append(False)
            continue
        d1 = dates[i + 1]
        cur = d0 + _dt.timedelta(days=1)
        hit = False
        while cur < d1:
            if cur in hol_set:
                hit = True
                break
            cur += _dt.timedelta(days=1)
        pre.append(hit)
    sess["PREHOL"] = pre
    return sess


def null_pctl(pnls: np.ndarray, cell_n: int, rng: np.random.Generator) -> float:
    draws = np.empty(N_NULL)
    for i in range(N_NULL):
        draws[i] = rng.choice(pnls, size=cell_n, replace=False).mean()
    return float(np.percentile(draws, NULL_PCTL))


def evaluate(name, mask, sess, rng, gate_eligible=True):
    pnls = sess["pnl"].to_numpy()
    cell = sess.loc[mask, "pnl"].to_numpy()
    n = len(cell)
    base_mean = pnls.mean()
    cell_mean = cell.mean() if n else float("nan")

    # fat-day robustness: drop top 5 from each
    cell_x5 = np.sort(cell)[:-5].mean() if n > 5 else float("nan")
    base_x5 = np.sort(pnls)[:-5].mean()

    thresh = null_pctl(pnls, n, rng) if n and n < len(pnls) else float("nan")

    checks = {
        f"N >= {N_FLOOR}": n >= N_FLOOR,
        "cell mean > baseline mean": cell_mean > base_mean,
        f"cell mean > null p{NULL_PCTL:.2f}": cell_mean > thresh,
        "ex-top5 cell > ex-top5 baseline": cell_x5 > base_x5,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    if not gate_eligible:
        verdict = "DESCRIPTIVE ONLY (below N floor, pre-declared)"

    print(f"\n--- {name} ---")
    print(f"  N={n}  cell mean=${cell_mean:.2f}/day  baseline mean=${base_mean:.2f}/day")
    print(f"  null p{NULL_PCTL:.2f}=${thresh:.2f}   ex-top5: cell=${cell_x5:.2f} vs base=${base_x5:.2f}")
    for c, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {c}")
    print(f"  => {verdict}")
    return verdict.startswith("PASS")


def main() -> None:
    sess = tag_cells(load_sessions())
    print(f"Trading sessions: {len(sess)}  ({sess['date'].min()} .. {sess['date'].max()})")
    print(f"Unconditional always-long: mean=${sess['pnl'].mean():.2f}/day  total=${sess['pnl'].sum():.0f}")
    print(f"Bonferroni: {N_TESTS} tests, alpha={ALPHA:.5f} -> null percentile {NULL_PCTL:.2f}")

    rng = np.random.default_rng(RNG_SEED)
    results = {}
    results["TOM"] = evaluate("Turn-of-month (offsets -1,+1,+2,+3)", sess["TOM"], sess, rng)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for d, nm in enumerate(dow_names):
        results[f"DOW-{nm}"] = evaluate(f"Day-of-week: {nm}", sess["dow"] == d, sess, rng)
    evaluate("Pre-holiday", sess["PREHOL"], sess, rng, gate_eligible=False)

    print(f"\n{'='*64}\nOPTION 6 GATE 0 VERDICT\n{'='*64}")
    passed = [k for k, v in results.items() if v]
    print(f"  gate-eligible cells passing all 4 checks: {passed if passed else 'NONE'}")
    print(f"  VERDICT: {'PASS -- ' + ', '.join(passed) if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
