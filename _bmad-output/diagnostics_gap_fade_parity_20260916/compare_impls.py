"""Why does GAP-1's --replay parity check disagree with its sealed Gate-0 number?

Sealed expectation printed by src/research/gap_fade_live.py: N=78, WR 62.8%, PF 1.760, Net $6,462
on data/processed/dollar_bars/1_minute/mnq_1min_2025.csv. The replay gives N=77, PF 2.017 — at
EVERY commit of that file back to its creation (bisect, 2026-09-16), so nothing drifted in it.

This compares, trade by trade, on the same 2025 bars:
  A. the sealed study's own functions (backtest_gap_fade.build_session_map / run)
  B. the loop inside gap_fade_live._run_replay, transcribed here and checked against its output

Reads the 2025 CSV only, never the 2026 file (which carries holdout rows). Writes nothing
outside this folder.

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_parity_20260916/compare_impls.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]                      # this worktree; the study resolves _REPO from here
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
CSV_2025 = MAIN / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(WT))
bgf = load("backtest_gap_fade", WT / "backtest_gap_fade.py")          # the sealed study
gfl = load("gap_fade_live_mod", WT / "src/research/gap_fade_live.py")  # the live bot (constants + _is_rth)


def load_2025() -> pd.DataFrame:
    df = pd.read_csv(CSV_2025, parse_dates=["timestamp"])
    ts = df["timestamp"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df["timestamp"] = ts.dt.tz_convert(bgf.ET_TZ)
    return df.set_index("timestamp").sort_index()


def replay_trades(df: pd.DataFrame) -> list[dict]:
    """The loop from gap_fade_live._run_replay, transcribed verbatim (checked against its output)."""
    rth = df[df.index.map(lambda t: gfl._is_rth(t))].copy()
    rth["date_et"] = rth.index.date
    by = rth.groupby("date_et")
    closes, opens, counts = by["close"].last(), by["open"].first(), by["close"].count()
    dows = by.apply(lambda g: g.index[0].weekday(), include_groups=False)
    out, dates = [], sorted(closes.index)
    for i in range(1, len(dates)):
        today, yest = dates[i], dates[i - 1]
        if counts[yest] < gfl.MIN_RTH_BARS:
            continue
        if dows[today] in gfl.EXCLUDE_DOW:
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
        out.append({"date": str(today), "outcome": outcome,
                    "pnl_pts": round(direction * (exit_px - entry), 2)})
    return out


def summarize(tr: list[dict]) -> dict:
    p = pd.Series([t["pnl_pts"] for t in tr]) * gfl.MNQ_PV
    loss = -p[p < 0].sum()
    return {"N": len(tr), "net_usd": round(float(p.sum())), "WR": round(float((p > 0).mean()) * 100, 1),
            "PF": round(float(p[p > 0].sum() / loss), 3) if loss else None}


def main() -> int:
    df = load_2025()
    study = bgf.run(df, bgf.build_session_map(df))
    replay = replay_trades(df)
    res = {"study_sealed_functions": summarize(study), "replay_loop": summarize(replay)}
    s_by = {t["date"]: t for t in study}
    r_by = {t["date"]: t for t in replay}
    only_study = sorted(set(s_by) - set(r_by))
    only_replay = sorted(set(r_by) - set(s_by))
    differing = sorted(d for d in set(s_by) & set(r_by)
                       if abs(s_by[d]["pnl_pts"] - r_by[d]["pnl_pts"]) > 1e-9
                       or s_by[d]["outcome"] != r_by[d]["outcome"])
    res["only_in_study"] = [{"date": d, **{k: s_by[d][k] for k in ("outcome", "pnl_pts")}} for d in only_study]
    res["only_in_replay"] = [{"date": d, **{k: r_by[d][k] for k in ("outcome", "pnl_pts")}} for d in only_replay]
    res["differing"] = [{"date": d, "study": s_by[d], "replay": r_by[d]} for d in differing]
    (HERE / "compare_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
