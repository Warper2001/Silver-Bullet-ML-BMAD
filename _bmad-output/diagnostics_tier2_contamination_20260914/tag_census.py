"""Tag YANK/Tier2 backtest signals and trades against the MNQ 1-min CSV defects (diagnostic).

Defects (see diagnostics_h2l2_contamination_20260914): 2025 roll weeks interleave two contracts
minute by minute (+/-~240-pt splices); Jan-Feb 2026 in mnq_1min_2026_ytd.csv is the deferred
MNQM26, not the front month.

Inputs, all pre-2026-03-01:
- raw /root/mnq_historical.json contract labels (via the ATR-band gate's parser)
- the 2026-09-13 census replays (reproduce June's seal replays: 68 ML0.50 / 91 no-ML trades)
- data/ml_training/doe_run_08_fullyear_history.csv (the ML filter's training trades)
- data/mim_x/mnq_1min_2021_2024_frontmonth.csv (YANK-FLOOR's OOS bars): splice spot-check

Reads trade P&L that is already part of sealed June results; no replay, no holdout.

Run: .venv/bin/python _bmad-output/diagnostics_tier2_contamination_20260914/tag_census.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
CENSUS = MAIN / "_bmad-output/diagnostics_yank_entry_mechanics_20260913"
DOE = MAIN / "data/ml_training/doe_run_08_fullyear_history.csv"
FLOOR_OOS = MAIN / "data/mim_x/mnq_1min_2021_2024_frontmonth.csv"
NY = "America/New_York"
AFTER_SESSIONS = 5          # post-roll carry-over window for H1/M15/ATR state


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


atr = load("atr_gate", HERE.parent / "diagnostics_atr_band_long_power_gate_20260914/power_gate.py")


def gkey(ts: pd.DatetimeIndex) -> np.ndarray:
    """Globex trade date: 18:00 ET opens the next day's session."""
    return np.array((ts + pd.Timedelta(hours=6)).date)


def build_map(raw: pd.DataFrame) -> dict:
    sess = gkey(raw.index)
    s = pd.Series(raw["contract"].to_numpy()).groupby(sess).nunique()
    sessions = list(s.index)
    inter = {d for d, n in s.items() if n > 1}
    last_i, since = None, {}
    for i, d in enumerate(sessions):
        if d in inter:
            last_i = i
        since[d] = (i - last_i) if last_i is not None else 10 ** 6
    con = raw["contract"].to_numpy()
    sw = raw.index[np.r_[False, con[1:] != con[:-1]]]
    return {"inter": inter, "since": since, "switch_ns": sw.asi8, "sessions": sessions}


def tag(ts_utc: pd.Series, m: dict) -> pd.DataFrame:
    t = pd.DatetimeIndex(pd.to_datetime(ts_utc, utc=True)).tz_convert(NY)
    k = gkey(t)
    ns = t.asi8
    n1h = np.searchsorted(m["switch_ns"], ns, "right") - np.searchsorted(m["switch_ns"], ns - 3_600 * 10 ** 9, "right")
    n24 = np.searchsorted(m["switch_ns"], ns, "right") - np.searchsorted(m["switch_ns"], ns - 86_400 * 10 ** 9, "right")
    since = np.array([m["since"].get(d, 10 ** 6) for d in k])
    back = np.array([d >= pd.Timestamp("2026-01-02").date() for d in k])
    cls = np.where(since == 0, "in_interleaved_session",
                   np.where(since <= AFTER_SESSIONS, f"within_{AFTER_SESSIONS}_sessions_after",
                            np.where(back, "back_month_2026", "clean")))
    return pd.DataFrame({"session": k, "switches_prev_1h": n1h, "switches_prev_24h": n24,
                         "sessions_since_interleaved": since, "class": cls})


def summarize(df: pd.DataFrame, pnl: str | None) -> dict:
    out = {}
    for c, g in df.groupby("class"):
        row = {"n": int(len(g)), "with_switch_prev_24h": int((g["switches_prev_24h"] > 0).sum())}
        if pnl:
            row.update({"pnl_sum": float(g[pnl].sum()), "pnl_mean": float(g[pnl].mean()),
                        "wins": int((g[pnl] > 0).sum())})
        out[c] = row
    if pnl:
        out["_total"] = {"n": int(len(df)), "pnl_sum": float(df[pnl].sum())}
    return out


def floor_oos_check() -> dict:
    """Largest 1-min close-to-close moves in YANK-FLOOR's OOS file: are they roll-week splices?"""
    d = pd.read_csv(FLOOR_OOS, nrows=None)
    tcol = [c for c in d.columns if "time" in c.lower()][0]
    d["ts"] = pd.to_datetime(d[tcol], utc=True, format="ISO8601")
    d = d.sort_values("ts")
    jump = d["close"].diff().abs()
    big = d.assign(jump=jump).nlargest(15, "jump")[["ts", "close", "jump"]]
    return {"columns": list(d.columns), "rows": int(len(d)),
            "contract_column": next((c for c in d.columns if "contract" in c.lower() or "symbol" in c.lower()), None),
            "n_jumps_over_150pts": int((jump > 150).sum()),
            "top_jumps": [{"ts": str(r.ts), "jump": float(r.jump)} for r in big.itertuples()]}


def main() -> int:
    raw = atr.parse_raw()
    m = build_map(raw)
    res: dict = {"interleaved_globex_sessions_in_census_window": sorted(
        str(d) for d in m["inter"] if pd.Timestamp("2025-05-19").date() <= d <= pd.Timestamp("2026-02-28").date())}
    for tag_ in ("ml050", "noml"):
        sig = pd.read_csv(CENSUS / f"signals_{tag_}.csv")
        trd = pd.read_csv(CENSUS / f"census_trades_{tag_}.csv")
        st, tt = tag(sig["signal_ts"], m), tag(trd["entry_time"], m)
        trd = pd.concat([trd, tt], axis=1)
        sig = pd.concat([sig, st], axis=1)
        res[f"census_{tag_}"] = {"signals": summarize(sig, None), "trades": summarize(trd, "pnl"),
                                 "flagged_trades": trd[trd["class"] != "clean"][
                                     ["entry_time", "exit_type", "pnl", "class", "switches_prev_24h"]].astype(str)
                                 .to_dict("records")}
        trd.to_csv(HERE / f"tagged_trades_{tag_}.csv", index=False)
        sig.to_csv(HERE / f"tagged_signals_{tag_}.csv", index=False)
    doe = pd.read_csv(DOE)
    res["doe_run_08_ml_training"] = {}
    for assume in ("UTC", NY):
        ts = pd.to_datetime(doe["timestamp"]).dt.tz_localize(assume).dt.tz_convert("UTC")
        t = tag(ts, m)
        res["doe_run_08_ml_training"][f"timestamps_as_{assume}"] = summarize(
            pd.concat([doe[["pnl"]], t], axis=1), "pnl")
    res["doe_run_08_span"] = [str(doe["timestamp"].min()), str(doe["timestamp"].max())]
    res["yank_floor_oos_file"] = floor_oos_check()
    (HERE / "tag_results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("census")}, indent=1, default=str)[:4000])
    for tag_ in ("ml050", "noml"):
        print(tag_, json.dumps(res[f"census_{tag_}"]["trades"], default=str))
        print(tag_, "signals", json.dumps(res[f"census_{tag_}"]["signals"], default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
