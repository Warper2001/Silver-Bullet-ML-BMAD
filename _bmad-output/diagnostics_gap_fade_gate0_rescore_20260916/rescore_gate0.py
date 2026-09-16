"""Re-score GAP-1's sealed Gate-0 figure on corrected bars.

Sealed prereg: _bmad-output/preregistration_gap_fade_gate0_rescore.md (commit 151f1d05),
ACCESS_LOG row appended before this ran.

Bars (prereg §2):
  2025-01-01 → 12-31   front-month rebuild (same dollar-bar writer, one contract per
                        session); gate = unfiltered rebuild reproduces the frozen CSV
  2026-01-01 → 03-11   front-month MNQH26 1-min records from the raw JSON; sessions with
                        more than one label dropped whole
  2026-03-12 → 06-11   mnq_1min_2026_ytd.csv rows unchanged (MNQM26 is front after the roll)
Prior close: always from the session's OWN contract; a session whose contract has no prior
RTH session in the data is skipped (what live does). No price adjustment (pre-excluded).

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/rescore_gate0.py
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
SPLICE = WT / "_bmad-output/diagnostics_gap_fade_splice_20260916"
RAW_JSON = Path("/root/mnq_historical.json")
EXTRACT = (MAIN / "_bmad-output/planning-artifacts/research"
           / "technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/imports/provenance-raw-2025.jsonl")
CSV_2026 = MAIN / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
SEALED = MAIN / "data/reports/gap_fade_20260625_205328.csv"
ROLL_2026 = datetime(2026, 3, 12).date()     # CME H26 -> M26 roll date (prereg §2)
END = datetime(2026, 6, 11, 23, 59, tzinfo=timezone.utc)

sys.path.insert(0, str(WT))
spec = importlib.util.spec_from_file_location("gfl", WT / "src/research/gap_fade_live.py")
gfl = importlib.util.module_from_spec(spec)
sys.modules["gfl"] = gfl
spec.loader.exec_module(gfl)


def load_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["timestamp"])
    ts = d["timestamp"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    d["timestamp"] = ts.dt.tz_convert(gfl.ET)
    return d.set_index("timestamp").sort_index()


def raw_2026_frontmonth() -> tuple[pd.DataFrame, dict, list]:
    """MNQH26 minutes for 2026-01-01 → 03-11, pure sessions only; plus per-(contract,date) RTH closes."""
    import re
    field = re.compile(r'^\s*"(High|Low|Open|Close|TimeStamp|TotalVolume|Contract)":\s*"?([^",]*)"?,?\s*$')
    rows, cur = [], {}
    with open(RAW_JSON) as f:
        for line in f:
            m = field.match(line)
            if not m:
                continue
            k, v = m.groups()
            cur[k] = v
            if k == "Contract":
                ts = cur["TimeStamp"]
                if ts >= "2026-01-01":
                    rows.append((ts, float(cur["Open"]), float(cur["High"]), float(cur["Low"]),
                                 float(cur["Close"]), int(float(cur["TotalVolume"])), v))
                cur = {}
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "contract"])
    d["ts"] = pd.to_datetime(d["ts"], format="%Y-%m-%dT%H:%M:%SZ", utc=True).dt.tz_convert(gfl.ET)
    d = d.set_index("ts").sort_index()
    # same-contract RTH closes for every (contract, date) seen in raw 2026
    closes: dict = {}
    rth = d[d.index.map(lambda t: gfl._is_rth(t))]
    for (c, day), g in rth.groupby([rth["contract"], rth.index.date]):
        closes[(c, day)] = float(g["close"].iloc[-1])
    # bars for the pre-roll window, pure sessions only
    pre = d[(d.index.date < ROLL_2026)]
    counts: dict = defaultdict(Counter)
    for t, c in zip(pre.index, pre["contract"]):
        counts[t.date()][c] += 1
    dropped = sorted(str(day) for day, c in counts.items() if len(c) > 1)
    keep_days = {day for day, c in counts.items() if len(c) == 1 and c.most_common(1)[0][0] == "MNQH26"}
    bars = pre[[day in keep_days for day in pre.index.date]].copy()
    return bars, closes, dropped


def extract_2025_closes() -> tuple[dict, dict]:
    """Per-(contract, date) RTH close, and each session's DOMINANT contract.

    Dominance is the same rule that built the front-month bars, so a session's "own
    contract" is the one its bars came from. Selecting by anything else (e.g. dict order)
    silently mis-assigns roll-week sessions.
    """
    closes: dict = {}
    counts: dict = defaultdict(Counter)
    for line in EXTRACT.open():
        b = json.loads(line)["bar"]
        t = datetime.strptime(b["TimeStamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(gfl.ET)
        if gfl._is_rth(t):
            closes[(b["Contract"], t.date())] = float(b["Close"])
            counts[t.date()][b["Contract"]] += 1
    dominant = {d: c.most_common(1)[0][0] for d, c in counts.items()}
    return closes, dominant


def replay(bars: pd.DataFrame, contract_of_day: dict, closes: dict) -> tuple[list, list]:
    rth = bars[bars.index.map(lambda t: gfl._is_rth(t))].copy()
    rth["date_et"] = rth.index.date
    by = rth.groupby("date_et")
    opens, counts = by["open"].first(), by["close"].count()
    dows = by.apply(lambda g: g.index[0].weekday(), include_groups=False)
    trades, skipped, dates = [], [], sorted(opens.index)
    for i in range(1, len(dates)):
        today, yest = dates[i], dates[i - 1]
        if counts[yest] < gfl.MIN_RTH_BARS or dows[today] in gfl.EXCLUDE_DOW:
            continue
        contract = contract_of_day.get(today)
        pc = closes.get((contract, yest))
        if pc is None:
            skipped.append({"date": str(today), "contract": contract,
                            "reason": "no prior-session RTH close in this contract"})
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
        trades.append({"date": str(today), "dir": "short" if direction == -1 else "long",
                       "gap_pct": round(100 * gap_abs / pc, 3), "outcome": outcome,
                       "pnl_usd": round(direction * (exit_px - entry) * gfl.MNQ_PV, 2)})
    return trades, skipped


def summary(tr: list[dict]) -> dict:
    p = pd.Series([t["pnl_usd"] for t in tr], dtype=float)
    loss = -p[p < 0].sum()
    return {"N": len(tr), "net_usd": round(float(p.sum())), "WR": round(float((p > 0).mean()) * 100, 1),
            "PF": round(float(p[p > 0].sum() / loss), 3) if loss else None}


def segment(day: str) -> str:
    if day >= "2026-05-20":
        return "2026-05-20..06-11 post-holdout"
    if day >= "2026-03-12":
        return "2026-03-12..05-19 holdout, front month"
    if day >= "2026-03-01":
        return "2026-03-01..03-11 pre-roll"
    if day >= "2026-01-01":
        return "2026 Jan-Feb"
    return "2025"


def main() -> int:
    front_2025 = SPLICE / "mnq_1min_2025_frontmonth.csv"
    if not front_2025.exists():
        raise SystemExit(f"missing {front_2025} — run rebuild_2025_frontmonth.py first (gate included)")
    meta_2025 = json.loads((SPLICE / "rebuild_meta.json").read_text())
    if not meta_2025.get("gate_reproduces_frozen_csv"):
        raise SystemExit("2025 rebuild gate did not pass — refusing to interpret (prereg §4)")

    b25 = load_csv(front_2025)
    closes, dominant_2025 = extract_2025_closes()
    raw26, closes26, dropped26 = raw_2026_frontmonth()
    closes.update(closes26)

    csv26 = load_csv(CSV_2026)
    post = csv26[(csv26.index.date >= ROLL_2026) & (csv26.index <= END)].copy()
    post["contract"] = "MNQM26"
    post_rth = post[post.index.map(lambda t: gfl._is_rth(t))]
    for day, g in post_rth.groupby(post_rth.index.date):
        closes[("MNQM26", day)] = float(g["close"].iloc[-1])

    # contract per session across the whole window
    contract_of_day: dict = {}
    for d in sorted({t.date() for t in b25.index}):
        contract_of_day[d] = dominant_2025.get(d)
    for d in sorted({t.date() for t in raw26.index}):
        contract_of_day[d] = "MNQH26"
    for d in sorted({t.date() for t in post.index}):
        contract_of_day[d] = "MNQM26"

    bars = pd.concat([b25[["open", "high", "low", "close"]],
                      raw26[["open", "high", "low", "close"]],
                      post[["open", "high", "low", "close"]]]).sort_index()
    trades, skipped = replay(bars, contract_of_day, closes)

    sealed = pd.read_csv(SEALED)
    sealed_tr = [{"date": str(r.date), "outcome": r.outcome, "pnl_usd": float(r.pnl_usd)}
                 for r in sealed.itertuples()]
    s_by = {t["date"]: t for t in sealed_tr}
    c_by = {t["date"]: t for t in trades}

    res = {
        "prereg": "151f1d05499da6f951d0743399c74f9caf76ee32",
        "sealed_gate0": summary(sealed_tr),
        "corrected_gate0": summary(trades),
        "window": [min(c_by) if c_by else None, max(c_by) if c_by else None],
        "bars": {"2025_frontmonth": len(b25), "2026_raw_prerolldropped_sessions": dropped26,
                 "2026_raw_pre_roll": len(raw26), "2026_csv_post_roll": len(post)},
        "skipped_no_same_contract_prior_close": skipped,
        "by_segment": {},
        "sealed_by_segment": {},
        "only_in_sealed": sorted(set(s_by) - set(c_by)),
        "only_in_corrected": sorted(set(c_by) - set(s_by)),
    }
    for name, trs, bucket in (("corrected", trades, "by_segment"), ("sealed", sealed_tr, "sealed_by_segment")):
        for t in trs:
            b = res[bucket].setdefault(segment(t["date"]), {"N": 0, "net_usd": 0.0})
            b["N"] += 1
            b["net_usd"] = round(b["net_usd"] + t["pnl_usd"], 2)
    changed = sorted(d for d in set(s_by) & set(c_by)
                     if abs(s_by[d]["pnl_usd"] - c_by[d]["pnl_usd"]) > 1e-9
                     or s_by[d]["outcome"] != c_by[d]["outcome"])
    res["changed"] = [{"date": d, "segment": segment(d), "sealed": s_by[d], "corrected": c_by[d]}
                      for d in changed]
    (HERE / "rescore_results.json").write_text(json.dumps(res, indent=2, default=str))
    pd.DataFrame(trades).to_csv(HERE / "corrected_gate0_trades.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "changed"}, indent=1, default=str)[:2600])
    print(f"\nchanged trades: {len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
