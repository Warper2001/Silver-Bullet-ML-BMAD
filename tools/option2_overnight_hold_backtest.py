"""Option 2 (post-R3 research plan): MNQ overnight/Globex hold.

Pre-registration: _bmad-output/preregistration_option2_overnight_hold.md
Primary test: always-long, RTH close -> next RTH open, dev window only
(< HOLDOUT_CUTOFF 2026-03-01).

Reuses backtest_gap_fade.load_data()/build_session_map() UNCHANGED --
those already compute exactly the (prior_close, next_open) pairs this test
needs; no new session-pairing logic. MIN_RTH_BARS=300 already filters
degenerate/holiday-shortened sessions. No additional maintenance-window
exclusion is needed beyond that: this test holds close-to-open on two
POINT prices, it does not bar-scan overnight, so the ~22:00-23:00 UTC CME
maintenance gap (which shows up as thin/absent 1-min bars within the night,
confirmed separately) never enters the calculation -- unlike the ticksim
context the seal's "excluded nights" language was borrowed from, where
continuous book reconstruction made that gap matter. Noted here rather than
silently deviating from the sealed text.

Usage: .venv/bin/python tools/option2_overnight_hold_backtest.py
"""

from __future__ import annotations

import random
import types
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytz

REPO = Path("/root/Silver-Bullet-ML-BMAD")

# backtest_gap_fade.py's own _REPO = Path(__file__).resolve().parents[3] raises
# IndexError from its current (repo-root) location -- same pre-existing,
# disclosed issue as Option 1a's sweep. Copying its pure functions verbatim
# rather than importing the broken module top-level (see option1_gap_fade_
# horizon_sweep.py for the same pattern and full explanation).
gf = types.SimpleNamespace()
gf.CSV_2025 = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
gf.CSV_2026 = REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
gf.MIN_RTH_BARS = 300
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
gf.build_session_map = _build_session_map

MNQ_PV = 2.0
ROUND_TURN_COST_USD = 4.00  # standard MNQ RT friction figure used elsewhere in this project (TSC-1, etc.)
HOLDOUT_CUTOFF = datetime(2026, 3, 1, tzinfo=timezone.utc)
N_NULL = 200
RNG_SEED = 20260905
GATE_PF_FLOOR = 1.15  # project's standing weak-edge floor (TSC-1/TSMOM-1)


def main() -> None:
    print("Loading data...")
    df = gf.load_data()
    sessions = gf.build_session_map(df)
    print(f"Sessions with valid prior close: {len(sessions)}")

    # dev window only
    dev_sessions = {
        d: s for d, s in sessions.items()
        if datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc) < HOLDOUT_CUTOFF
    }
    print(f"Dev-window sessions (< {HOLDOUT_CUTOFF.date()}): {len(dev_sessions)}")

    rows = []
    for date_et, sess in sorted(dev_sessions.items()):
        gap_pts = sess["rth_open"] - sess["prior_close"]  # always-long: profit if open > prior close
        pnl_usd = gap_pts * MNQ_PV - ROUND_TURN_COST_USD
        is_friday = sess["dow"] == 4
        rows.append({"date": str(date_et), "gap_pts": gap_pts, "pnl_usd": pnl_usd, "is_friday": is_friday})

    def pf_of(rs):
        w = sum(r["pnl_usd"] for r in rs if r["pnl_usd"] > 0)
        losses = abs(sum(r["pnl_usd"] for r in rs if r["pnl_usd"] < 0))
        return w / losses if losses > 0 else float("inf")

    n = len(rows)
    pf = pf_of(rows)
    gross = sum(r["pnl_usd"] for r in rows)
    wins = sum(1 for r in rows if r["pnl_usd"] > 0)
    fri_rows = [r for r in rows if r["is_friday"]]
    non_fri_rows = [r for r in rows if not r["is_friday"]]

    sorted_desc = sorted(rows, key=lambda r: r["pnl_usd"], reverse=True)
    pf_ex_top5 = pf_of(sorted_desc[5:])

    print(f"\n{'='*60}")
    print("OPTION 2 -- MNQ overnight/Globex hold (always-long, primary test)")
    print(f"{'='*60}")
    print(f"N nights          : {n}")
    print(f"Win rate          : {wins/n*100:.1f}%")
    print(f"Net PF            : {pf:.3f}")
    print(f"Total P&L (net)   : ${gross:.0f}  (RT cost ${ROUND_TURN_COST_USD:.2f}/night)")
    print(f"Ex-top-5-nights PF: {pf_ex_top5:.3f}")
    print(f"Friday nights     : N={len(fri_rows)} PF={pf_of(fri_rows):.3f} gross=${sum(r['pnl_usd'] for r in fri_rows):.0f}")
    print(f"Non-Friday nights : N={len(non_fri_rows)} PF={pf_of(non_fri_rows):.3f} gross=${sum(r['pnl_usd'] for r in non_fri_rows):.0f}")

    # random-direction null
    rng = random.Random(RNG_SEED)
    null_pfs = []
    for _ in range(N_NULL):
        flipped = []
        for r in rows:
            sign = 1 if rng.random() < 0.5 else -1
            flipped.append({"pnl_usd": sign * (r["gap_pts"] * MNQ_PV) - ROUND_TURN_COST_USD})
        null_pfs.append(pf_of(flipped))
    null_pfs.sort()
    p95 = null_pfs[int(0.95 * len(null_pfs))]
    median = null_pfs[len(null_pfs) // 2]
    print(f"\nRandom-direction null (N={N_NULL}): median={median:.3f} p95={p95:.3f}")

    print(f"\n{'='*60}")
    print("GATE 0")
    print(f"{'='*60}")
    checks = {
        f"N floor (several hundred expected, got {n})": n >= 100,
        f"net PF ({pf:.3f}) > {GATE_PF_FLOOR}": pf > GATE_PF_FLOOR,
        f"net PF ({pf:.3f}) > null p95 ({p95:.3f})": pf > p95,
        f"ex-top5 PF ({pf_ex_top5:.3f}) > 1.0": pf_ex_top5 > 1.0,
    }
    for c, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {c}")
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
