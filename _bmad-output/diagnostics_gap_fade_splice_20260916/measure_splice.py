"""Measure GAP-1's roll-splice sensitivity: frozen CSV vs the front-month rebuild.

Both CSVs come from the SAME dollar-bar writer on the SAME pinned extract; the only
difference is that the rebuild keeps one contract per session (rebuild_2025_frontmonth.py,
whose gate reproduced the frozen CSV byte-for-byte). Every session is present in both, so
the prior-close chain is intact and the comparison isolates the splices.

Trades are produced by the strategy's own replay loop (transcribed from
src/research/gap_fade_live.py._run_replay and checked against its printed summary).

Also separates roll-BOUNDARY trades: the first session of a new contract, where the gap is
measured against the previous contract's close. That artifact exists in BOTH versions and
is a different defect from the interleaving.

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_splice_20260916/measure_splice.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
FROZEN = MAIN / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
CORRECTED = HERE / "mnq_1min_2025_frontmonth.csv"
META = HERE / "rebuild_meta.json"

sys.path.insert(0, str(WT))
spec = importlib.util.spec_from_file_location("gfl", WT / "src/research/gap_fade_live.py")
gfl = importlib.util.module_from_spec(spec)
sys.modules["gfl"] = gfl
spec.loader.exec_module(gfl)


def trades_for(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ts = df["timestamp"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df["timestamp"] = ts.dt.tz_convert(gfl.ET)
    df = df.set_index("timestamp").sort_index()
    rth = df[df.index.map(lambda t: gfl._is_rth(t))].copy()
    rth["date_et"] = rth.index.date
    by = rth.groupby("date_et")
    closes, opens, counts = by["close"].last(), by["open"].first(), by["close"].count()
    dows = by.apply(lambda g: g.index[0].weekday(), include_groups=False)
    out, dates = [], sorted(closes.index)
    for i in range(1, len(dates)):
        today, yest = dates[i], dates[i - 1]
        if counts[yest] < gfl.MIN_RTH_BARS or dows[today] in gfl.EXCLUDE_DOW:
            continue
        pc, ro = closes[yest], opens[today]
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
                    "pnl_usd": round(direction * (exit_px - entry) * gfl.MNQ_PV, 2)})
    return out


def summary(tr: list[dict]) -> dict:
    p = pd.Series([t["pnl_usd"] for t in tr], dtype=float)
    loss = -p[p < 0].sum()
    return {"N": len(tr), "net_usd": round(float(p.sum())), "WR": round(float((p > 0).mean()) * 100, 1),
            "PF": round(float(p[p > 0].sum() / loss), 3) if loss else None}


def main() -> int:
    meta = json.loads(META.read_text())
    if not meta.get("gate_reproduces_frozen_csv"):
        raise SystemExit("rebuild gate did not pass — refusing to interpret")
    mixed = set(meta["mixed_session_detail"])
    boundaries = set(meta["contract_switches_between_sessions"])

    frozen, corrected = trades_for(FROZEN), trades_for(CORRECTED)
    f_by = {t["date"]: t for t in frozen}
    c_by = {t["date"]: t for t in corrected}

    def tag(date: str) -> str:
        if date in boundaries:
            return "roll_boundary"      # gap measured across a contract change (both versions)
        return "interleaved_session" if date in mixed else "clean"

    res: dict = {"frozen_csv": summary(frozen), "corrected_frontmonth": summary(corrected),
                 "sealed_baseline": dict(gfl.SEALED_PARITY_2025)}
    res["parity_frozen_matches_sealed"] = (
        summary(frozen)["N"] == gfl.SEALED_PARITY_2025["N"]
        and summary(frozen)["PF"] == gfl.SEALED_PARITY_2025["PF"])

    only_f = sorted(set(f_by) - set(c_by))
    only_c = sorted(set(c_by) - set(f_by))
    both_diff = sorted(d for d in set(f_by) & set(c_by) if f_by[d] != c_by[d])
    res["trades_only_in_frozen"] = [{**f_by[d], "tag": tag(d)} for d in only_f]
    res["trades_only_in_corrected"] = [{**c_by[d], "tag": tag(d)} for d in only_c]
    res["trades_changed"] = [{"date": d, "tag": tag(d), "frozen": f_by[d], "corrected": c_by[d]}
                             for d in both_diff]
    res["identical_trades"] = len(set(f_by) & set(c_by)) - len(both_diff)

    # attribution: how much of each version's P&L sits on each kind of session
    for name, trs in (("frozen_csv", frozen), ("corrected_frontmonth", corrected)):
        buckets: dict = {}
        for t in trs:
            b = buckets.setdefault(tag(t["date"]), {"N": 0, "net_usd": 0.0})
            b["N"] += 1
            b["net_usd"] = round(b["net_usd"] + t["pnl_usd"], 2)
        res[f"{name}_by_session_kind"] = buckets
    # what the sealed number becomes with roll-boundary trades also removed
    for name, trs in (("frozen_csv", frozen), ("corrected_frontmonth", corrected)):
        clean = [t for t in trs if tag(t["date"]) == "clean"]
        res[f"{name}_clean_sessions_only"] = summary(clean)

    (HERE / "splice_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "trades_changed"}, indent=1, default=str))
    print("\nchanged trades:")
    for c in res["trades_changed"]:
        print(" ", c["date"], c["tag"], "| frozen", c["frozen"]["outcome"], c["frozen"]["pnl_usd"],
              "-> corrected", c["corrected"]["outcome"], c["corrected"]["pnl_usd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
