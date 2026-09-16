"""Fix the roll-boundary gap artifact the way the live bot already behaves.

Live GAP-1 fetches `barsback` bars for ONE symbol, so its prior RTH close and today's RTH
open always come from the same contract: it never measures a gap across a contract change.
The research series does, because it is an unadjusted continuous splice, and on the first
session of a new contract the "gap" folds in the calendar spread (2025-03-03: 370.25 pts /
1.768%, carrying 9.4% of the sealed 2025 net).

Whole-series adjustment is NOT neutral for this strategy and is therefore rejected here:
  * point (Panama) back-adjustment preserves dollar P&L but shifts price LEVELS, and the
    trigger is 0.5% OF the prior close — measured: one marginal trade dropped
    (2025-03-27), one added (2025-06-02), on top of the intended boundary fix;
  * ratio adjustment preserves percentages but rescales historical point P&L.

This harness instead reproduces live behaviour: for every session, the prior RTH close is
taken from the SAME contract's own prior RTH session, read from the pinned raw extract
(both contracts trade during roll weeks, so the new contract has its own prior session).
Everything else — bars, thresholds, geometry, exits — is the front-month rebuild and the
strategy's own replay loop.

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_splice_20260916/same_contract_prior_close.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
EXTRACT = (MAIN / "_bmad-output/planning-artifacts/research"
           / "technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/imports/provenance-raw-2025.jsonl")
FRONT = HERE / "mnq_1min_2025_frontmonth.csv"

sys.path.insert(0, str(WT))
spec = importlib.util.spec_from_file_location("gfl", WT / "src/research/gap_fade_live.py")
gfl = importlib.util.module_from_spec(spec)
sys.modules["gfl"] = gfl
spec.loader.exec_module(gfl)
ms_spec = importlib.util.spec_from_file_location("ms", HERE / "measure_splice.py")
ms = importlib.util.module_from_spec(ms_spec)
sys.modules["ms"] = ms
ms_spec.loader.exec_module(ms)


def raw_rth_by_contract() -> tuple[dict, dict]:
    """{(contract, date): last RTH close}, and {date: dominant contract}, from the raw extract."""
    closes: dict = {}
    counts: dict = defaultdict(Counter)
    for line in EXTRACT.open():
        b = json.loads(line)["bar"]
        t = datetime.strptime(b["TimeStamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(gfl.ET)
        if not gfl._is_rth(t):
            continue
        d = t.date()
        key = (b["Contract"], d)
        closes[key] = float(b["Close"])          # extract is chronological: last write wins
        counts[d][b["Contract"]] += 1
    dominant = {d: c.most_common(1)[0][0] for d, c in counts.items()}
    return closes, dominant


def trades_same_contract(csv_path: Path, closes: dict, dominant: dict) -> list[dict]:
    """The strategy's replay loop, but the prior close comes from the session's own contract."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ts = df["timestamp"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df["timestamp"] = ts.dt.tz_convert(gfl.ET)
    df = df.set_index("timestamp").sort_index()
    rth = df[df.index.map(lambda t: gfl._is_rth(t))].copy()
    rth["date_et"] = rth.index.date
    by = rth.groupby("date_et")
    opens, counts = by["open"].first(), by["close"].count()
    dows = by.apply(lambda g: g.index[0].weekday(), include_groups=False)
    out, dates, skipped = [], sorted(opens.index), []
    for i in range(1, len(dates)):
        today, yest = dates[i], dates[i - 1]
        if counts[yest] < gfl.MIN_RTH_BARS or dows[today] in gfl.EXCLUDE_DOW:
            continue
        contract = dominant[today]
        pc = closes.get((contract, yest))        # SAME contract's own prior RTH close
        if pc is None:
            skipped.append({"date": str(today), "contract": contract,
                            "reason": "no prior-session close in this contract"})
            continue
        ro = opens[today]
        gap = ro - pc
        gap_abs = abs(gap)
        if gap_abs / pc < gfl.GAP_MIN_PCT:
            continue
        direction = -1 if gap > 0 else 1
        entry, target = ro, pc
        stop = (entry + gfl.STOP_MULT * gap_abs) if direction == -1 else (entry - gfl.STOP_MULT * gap_abs)
        day_bars = rth[rth["date_et"] == today].iloc[1:]
        outcome = "eod"
        exit_px = day_bars["close"].iloc[-1] if len(day_bars) else entry
        for ts_b, bar in day_bars.iterrows():
            if ts_b.hour >= gfl.TIME_STOP_HOUR:
                outcome, exit_px = "time", bar["open"]
                break
            if direction == -1:
                if bar["low"] <= target:
                    outcome, exit_px = "fill", target
                    break
                if bar["high"] >= stop:
                    outcome, exit_px = "stop", stop
                    break
            else:
                if bar["high"] >= target:
                    outcome, exit_px = "fill", target
                    break
                if bar["low"] <= stop:
                    outcome, exit_px = "stop", stop
                    break
        out.append({"date": str(today), "dir": "short" if direction == -1 else "long",
                    "gap_pct": round(100 * gap_abs / pc, 3), "outcome": outcome,
                    "pnl_usd": round(direction * (exit_px - entry) * gfl.MNQ_PV, 2), "contract": contract})
    return out, skipped


def main() -> int:
    meta = json.loads((HERE / "rebuild_meta.json").read_text())
    boundaries = set(meta["contract_switches_between_sessions"])
    closes, dominant = raw_rth_by_contract()
    fixed, skipped = trades_same_contract(FRONT, closes, dominant)
    front = ms.trades_for(FRONT)

    f_by = {t["date"]: t for t in front}
    x_by = {t["date"]: {k: v for k, v in t.items() if k != "contract"} for t in fixed}
    changed = sorted(d for d in set(f_by) & set(x_by) if f_by[d] != x_by[d])
    res = {
        "frontmonth_unadjusted": ms.summary(front),
        "same_contract_prior_close": ms.summary(fixed),
        "boundary_sessions": sorted(boundaries),
        "skipped_no_prior_in_contract": skipped,
        "only_in_unadjusted": sorted(set(f_by) - set(x_by)),
        "only_in_fixed": sorted(set(x_by) - set(f_by)),
        "changed": [{"date": d, "is_boundary": d in boundaries, "unadjusted": f_by[d], "fixed": x_by[d]}
                    for d in changed],
    }
    touched = set(changed) | (set(f_by) ^ set(x_by))
    res["only_boundary_sessions_touched"] = touched <= boundaries
    res["sessions_touched"] = sorted(touched)
    (HERE / "same_contract_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
