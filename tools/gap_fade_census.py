#!/usr/bin/env python3
"""GAP-1 (gap-fade) census: live-vs-sealed parity, fill fidelity, and learning arms.

Read-only. Nothing in data/gap_fade/ is written and src/research/gap_fade_live.py is never
executed. The sealed engine's constants and pure functions (is_rth, simulate_day) are pulled
VERBATIM from backtest_gap_fade.py with ``ast`` — importing that module fails at the repo
root (its _REPO = Path(__file__).parents[3] raises IndexError).

Q1 decision-input parity: prior RTH close + 09:30 open recomputed from recorded live bars vs
   the bot's logged decisions (decisions.csv, deduplicated by date). PARITY HOLDS iff every
   action agrees and both inputs are within 1 tick on >= 95% of sessions. A missing-data gap
   in the recording makes the harness use a stale prior session — inspect mismatches before
   calling one a live defect.
Q2 outcome parity: every trades.db trade re-simulated with the sealed simulate_day on the
   recorded bars (logged entry/target/stop). HOLDS iff outcomes all match and P&L is within
   1 tick on >= 95%.
Q3 fill fidelity: realized minus modeled P&L (fills.csv) and entry slippage.
Q4 learning arms per era: B sealed fade at 09:30, A random fade time 09:31-12:30 (200 draws),
   I follow-the-gap inverse; fade-direction path D_k; Holm across A, I vs B; same-bar
   target/stop ambiguity audit (the sealed engine checks target first).

Origin and regression check: promoted from
_bmad-output/diagnostics_gap_fade_census_20260913/gap_fade_census.py (plan sha 3df09ccd).
With default eras and --seed 20260913 it must reproduce that folder's results.json exactly.

Example:
  PYTHONPATH=. .venv/bin/python tools/gap_fade_census.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import ast
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from census_stats import boot_p_gt0_with_ci, holm, paired_iid_test, sharpe  # noqa: E402

TICK, COST = 0.25, 0.52
NRAND = 200
KS = [5, 15, 30, 60, 120, 210]
SEALED_NAMES = ("GAP_MIN_PCT", "STOP_MULT", "TIME_STOP_HOUR", "MIN_RTH_BARS", "RTH_START", "RTH_END")


def load_sealed(engine_path: Path) -> dict:
    """Extract the sealed constants and is_rth/simulate_day from backtest_gap_fade.py verbatim."""
    tree = ast.parse(engine_path.read_text())
    keep = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in SEALED_NAMES for t in n.targets):
            keep.append(n)
        if isinstance(n, ast.FunctionDef) and n.name in ("is_rth", "simulate_day"):
            keep.append(n)
    ns = {"pd": pd}
    exec(compile(ast.Module(body=keep, type_ignores=[]), f"{engine_path.name}[sealed extract]", "exec"), ns)
    missing = [k for k in (*SEALED_NAMES, "is_rth", "simulate_day") if k not in ns]
    if missing:
        raise RuntimeError(f"sealed engine extract is missing {missing}")
    return ns


class Census:
    def __init__(self, sealed: dict, rng):
        self.S = sealed
        self.rng = rng
        self.is_rth, self.simulate_day = sealed["is_rth"], sealed["simulate_day"]

    # ---------- bars and sessions ----------
    @staticmethod
    def bars_from_recorded(path: Path, prefix: str) -> pd.DataFrame:
        cols = ["minute"] + [f"{prefix}_{c}" for c in ("open", "high", "low", "close")]
        p = pd.read_csv(path, usecols=cols).dropna()
        p["timestamp"] = pd.to_datetime(p["minute"], utc=True).dt.tz_convert("US/Eastern")
        p = p.rename(columns={f"{prefix}_{c}": c for c in ("open", "high", "low", "close")})
        return p.drop_duplicates("timestamp").set_index("timestamp").sort_index()[["open", "high", "low", "close"]]

    @staticmethod
    def bars_from_processed(bars_dir: Path, before_utc: str) -> pd.DataFrame:
        dfs = []
        for f in ("mnq_1min_2025.csv", "mnq_1min_2026_ytd.csv"):
            d = pd.read_csv(bars_dir / f)
            d["timestamp"] = pd.to_datetime(d["timestamp"], format="ISO8601", utc=True)
            dfs.append(d)
        d = pd.concat(dfs)
        d = d[d.timestamp < pd.Timestamp(before_utc, tz="UTC")]
        d["timestamp"] = d["timestamp"].dt.tz_convert("US/Eastern")
        return d.drop_duplicates("timestamp").set_index("timestamp").sort_index()[["open", "high", "low", "close"]]

    def rth_by_day(self, df: pd.DataFrame) -> dict:
        rth = df[df.index.map(self.is_rth)].copy()
        rth["date_et"] = rth.index.date
        return {d: g for d, g in rth.groupby("date_et")}

    def sessions(self, days: dict) -> dict:
        """Sealed session rules: prior session needs >= MIN_RTH_BARS; Fridays skipped; gap >= GAP_MIN_PCT."""
        out, dates = {}, sorted(days)
        for i in range(1, len(dates)):
            y, t = days[dates[i - 1]], days[dates[i]]
            if len(y) < self.S["MIN_RTH_BARS"]:
                continue
            pc, ro = float(y["close"].iloc[-1]), float(t["open"].iloc[0])
            gap = ro - pc
            dow = t.index[0].weekday()
            action = "SKIPPED_FRIDAY" if dow == 4 else ("ENTERED" if abs(gap) / pc >= self.S["GAP_MIN_PCT"] else "NO_SETUP")
            out[str(dates[i])] = {"prior_close": pc, "rth_open": ro, "gap": gap, "gap_abs": abs(gap),
                                  "gap_pct": abs(gap) / pc, "dow": dow, "action": action}
        return out

    # ---------- exit audits ----------
    def stop_first(self, day_bars, direction, entry, target, stop):
        """simulate_day with the stop checked before the target (audit only)."""
        for ts, bar in day_bars.iterrows():
            if ts.hour >= self.S["TIME_STOP_HOUR"]:
                return "time", direction * (bar["open"] - entry)
            if direction == -1:
                if bar["high"] >= stop:
                    return "stop", direction * (stop - entry)
                if bar["low"] <= target:
                    return "fill", direction * (target - entry)
            else:
                if bar["low"] <= stop:
                    return "stop", direction * (stop - entry)
                if bar["high"] >= target:
                    return "fill", direction * (target - entry)
        return "eod", direction * (day_bars["close"].iloc[-1] - entry)

    def both_touched(self, day_bars, direction, target, stop) -> bool:
        """True if the first exit bar touched both target and stop."""
        for ts, bar in day_bars.iterrows():
            if ts.hour >= self.S["TIME_STOP_HOUR"]:
                return False
            hit_t = bar["low"] <= target if direction == -1 else bar["high"] >= target
            hit_s = bar["high"] >= stop if direction == -1 else bar["low"] <= stop
            if hit_t and hit_s:
                return True
            if hit_t or hit_s:
                return False
        return False

    # ---------- Q4 ----------
    def arms_for_era(self, days, sess, lo, hi) -> pd.DataFrame:
        stop_mult = self.S["STOP_MULT"]
        rows = []
        for date, s in sorted(sess.items()):
            if not (lo <= date <= hi) or s["action"] != "ENTERED":
                continue
            g = days[pd.Timestamp(date).date()]
            if len(g) < 211:
                continue
            d = -1 if s["gap"] > 0 else 1
            ga, o0, pc = s["gap_abs"], float(g["open"].iloc[0]), s["prior_close"]
            rest = g.iloc[1:]
            fill = o0 + d * TICK
            stop_lvl = (o0 + stop_mult * ga) if d == -1 else (o0 - stop_mult * ga)
            oc, pts = self.simulate_day(rest, d, fill, pc, stop_lvl)
            rB = (pts - COST) / (2 * ga)
            _, pts2 = self.stop_first(rest, d, fill, pc, stop_lvl)
            amb = self.both_touched(rest, d, pc, stop_lvl)
            cand = [i for i, ts in enumerate(g.index) if (9, 31) <= (ts.hour, ts.minute) <= (12, 30)]
            vals = []
            for u in self.rng.choice(cand, NRAND):
                f = float(g["open"].iloc[u]) + d * TICK
                _, p = self.simulate_day(g.iloc[u + 1:], d, f, f + d * ga, f - d * stop_mult * ga)
                vals.append((p - COST) / (2 * ga))
            rA = float(np.mean(vals))
            di = -d
            fi = o0 + di * TICK
            _, pi = self.simulate_day(rest, di, fi, fi + di * ga, fi - di * stop_mult * ga)
            rI = (pi - COST) / (2 * ga)
            path = {k: d * (float(g["close"].iloc[k]) - o0) / (2 * ga) for k in KS}
            rows.append({"date": date, "gap_pct": s["gap_pct"] * 100, "dir": "S" if d == -1 else "L", "B_outcome": oc,
                         "R_B": rB, "R_B_stopfirst": (pts2 - COST) / (2 * ga), "ambiguous_bar": amb,
                         "R_A": rA, "R_I": rI, **{f"D{k}": v for k, v in path.items()}})
        return pd.DataFrame(rows)

    def era_stats(self, df: pd.DataFrame) -> dict:
        if len(df) < 5:
            return {"n": len(df), "note": "too few sessions"}
        out = {"n": len(df), "mean_R": {a: float(df[f"R_{a}"].mean()) for a in "BAI"},
               "sharpe": {a: sharpe(df[f"R_{a}"]) for a in "BAI"},
               "B_outcomes": df.B_outcome.value_counts().to_dict(),
               "ambiguous_bars": int(df.ambiguous_bar.sum()),
               "stopfirst_delta_R_total": float((df.R_B_stopfirst - df.R_B).sum())}
        tests = {a: paired_iid_test(df[f"R_{a}"] - df["R_B"], self.rng) for a in "AI"}
        hg, hl = holm([tests[a]["p_gt"] for a in "AI"]), holm([tests[a]["p_lt"] for a in "AI"])
        for a, g_, l_ in zip("AI", hg, hl):
            tests[a].update({"holm_p_gt": g_, "holm_p_lt": l_, "rho": float(np.corrcoef(df[f"R_{a}"], df.R_B)[0, 1]),
                             "verdict": "SUPERIOR" if g_ < 0.05 else ("INFERIOR" if l_ < 0.05 else "INCONCLUSIVE")})
        out["tests_vs_B"] = tests
        path, ps = {}, []
        for k in KS:
            p, ci = boot_p_gt0_with_ci(df[f"D{k}"], self.rng)
            path[str(k)] = {"mean": float(df[f"D{k}"].mean()), "median": float(df[f"D{k}"].median()), "ci": ci}
            if k in (30, 60, 120):
                ps.append(p)
        h = holm(ps)
        out["path_R"] = path
        out["path_holm_p_30_60_120"] = h
        out["path_verdict"] = ("INFORMATION PRESENT"
                               if any(path[str(k)]["mean"] > 0 and p < 0.05 for k, p in zip((30, 60, 120), h))
                               else "NO INFORMATION DETECTED")
        return out


def decision_parity(dec: pd.DataFrame, sess_ts: dict, sess_px: dict | None) -> tuple[pd.DataFrame, dict]:
    rows = []
    for r in dec.itertuples():
        s = sess_ts.get(r.date_et)
        sp = sess_px.get(r.date_et) if sess_px is not None else None
        rows.append({"date": r.date_et, "logged_action": r.action, "logged_prior_close": r.prior_close,
                     "logged_rth_open": r.rth_open,
                     "rec_action": s["action"] if s else None,
                     "d_prior_close": (s["prior_close"] - r.prior_close) if s else None,
                     "d_rth_open": (s["rth_open"] - r.rth_open) if s else None,
                     "px_action": sp["action"] if sp else None,
                     "px_d_prior_close": (sp["prior_close"] - r.prior_close) if sp else None,
                     "px_d_rth_open": (sp["rth_open"] - r.rth_open) if sp else None})
    q1 = pd.DataFrame(rows)
    have = q1.dropna(subset=["rec_action"])
    act_ok = have.logged_action == have.rec_action
    pc_ok = have.d_prior_close.abs() <= TICK + 1e-9
    ro_ok = have.d_rth_open.abs() <= TICK + 1e-9
    res = {"sessions_logged": len(q1), "sessions_with_recorded_bars": len(have),
           "action_agreement": f"{int(act_ok.sum())}/{len(have)}",
           "prior_close_within_tick": float(pc_ok.mean()), "rth_open_within_tick": float(ro_ok.mean()),
           "action_mismatches": have[~act_ok][["date", "logged_action", "rec_action",
                                               "d_prior_close", "d_rth_open"]].to_dict("records"),
           "input_mismatches": have[~(pc_ok & ro_ok)][["date", "d_prior_close", "d_rth_open"]].round(2).to_dict("records"),
           "missing_recorded": q1[q1.rec_action.isna()].date.tolist()}
    res["verdict"] = "PARITY HOLDS" if (act_ok.all() and pc_ok.mean() >= 0.95 and ro_ok.mean() >= 0.95) else "PARITY BROKEN"
    if sess_px is not None:
        hp = q1.dropna(subset=["px_action"])
        res["projectx_sensitivity"] = {"action_agreement": f"{int((hp.logged_action == hp.px_action).sum())}/{len(hp)}",
                                       "prior_close_within_tick": float((hp.px_d_prior_close.abs() <= TICK + 1e-9).mean()),
                                       "rth_open_within_tick": float((hp.px_d_rth_open.abs() <= TICK + 1e-9).mean())}
    return q1, res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sealed-engine", default=str(ROOT / "backtest_gap_fade.py"))
    ap.add_argument("--recorded-bars", default=str(ROOT / "logs/yank_shadow_parity.csv"))
    ap.add_argument("--decisions", default=str(ROOT / "data/gap_fade/decisions.csv"))
    ap.add_argument("--fills", default=str(ROOT / "data/gap_fade/fills.csv"))
    ap.add_argument("--trades-db", default=str(ROOT / "data/trades.db"))
    ap.add_argument("--bars-dir", default=str(ROOT / "data/processed/dollar_bars/1_minute"))
    ap.add_argument("--is-start", default="2025-01-02")
    ap.add_argument("--is-end", default="2026-02-27")
    ap.add_argument("--is-bars-before", default="2026-02-28", help="processed bars before this UTC date only")
    ap.add_argument("--live-start", default="2026-06-26")
    ap.add_argument("--live-end", default="2026-09-11")
    ap.add_argument("--live-entered-from", default="2026-06-25", help="first date counted in rule-says-enter checks")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    c = Census(load_sealed(Path(a.sealed_engine)), np.random.default_rng(a.seed))
    res = {}
    days_ts = c.rth_by_day(c.bars_from_recorded(Path(a.recorded_bars), "ts"))
    days_px = c.rth_by_day(c.bars_from_recorded(Path(a.recorded_bars), "px"))
    sess_ts, sess_px = c.sessions(days_ts), c.sessions(days_px)

    dec = pd.read_csv(a.decisions).drop_duplicates("date_et", keep="first")
    q1, res["Q1"] = decision_parity(dec, sess_ts, sess_px)
    q1.to_csv(out / "q1_decision_parity.csv", index=False)
    rec_entered = {d for d, s in sess_ts.items() if s["action"] == "ENTERED" and a.live_entered_from <= d <= a.live_end}

    con = sqlite3.connect(f"file:{a.trades_db}?mode=ro", uri=True)
    tdb = pd.read_sql("select timestamp, direction, entry_price, exit_price, pnl, exit_reason, metadata from trades "
                      "where trader_id='trader-gap-fade' and timestamp >= '2026-06-01' order by timestamp", con)
    tdb["date"] = pd.to_datetime(tdb.timestamp, format="ISO8601", utc=True).dt.tz_convert("US/Eastern").dt.date.astype(str)
    q2 = []
    for r in tdb.itertuples():
        md = json.loads(r.metadata)
        g = days_ts.get(pd.Timestamp(r.date).date())
        d = -1 if r.direction == "S" else 1
        if g is None or len(g) < 2:
            q2.append({"date": r.date, "note": "no recorded bars"})
            continue
        oc, pts = c.simulate_day(g.iloc[1:], d, r.entry_price, md["target"], md["stop"])
        q2.append({"date": r.date, "live_outcome": r.exit_reason, "sim_outcome": oc, "live_pts": r.pnl / 2,
                   "sim_pts": pts, "d_pts": pts - r.pnl / 2, "commissioning": r.date == a.live_entered_from})
    q2 = pd.DataFrame(q2)
    q2.to_csv(out / "q2_outcome_parity.csv", index=False)
    ok = q2.dropna(subset=["sim_outcome"])
    oc_ok = ok.live_outcome == ok.sim_outcome
    pt_ok = ok.d_pts.abs() <= TICK + 1e-9
    trade_dates = set(tdb.date)
    res["Q2"] = {"live_trades": len(tdb), "compared": len(ok), "outcome_agreement": f"{int(oc_ok.sum())}/{len(ok)}",
                 "pnl_within_tick": float(pt_ok.mean()),
                 "mismatches": ok[~(oc_ok & pt_ok)][
                     ["date", "live_outcome", "sim_outcome", "live_pts", "sim_pts"]].round(2).to_dict("records"),
                 "rule_says_enter_no_trade": sorted(rec_entered - trade_dates),
                 "trade_but_rule_says_no": sorted(d for d in trade_dates if d in sess_ts and sess_ts[d]["action"] != "ENTERED"),
                 "verdict": "PARITY HOLDS" if (oc_ok.all() and pt_ok.mean() >= 0.95) else "PARITY BROKEN"}

    fills = pd.read_csv(a.fills)
    dl = fills.delta_usd.dropna().to_numpy()
    _, ci = boot_p_gt0_with_ci(dl, c.rng)
    dmap = dec.set_index("date_et")["rth_open"].to_dict()
    slip = [((dmap[r.date_et] - r.entry_exec) if r.dir == "S" else (r.entry_exec - dmap[r.date_et]))
            for r in fills.itertuples() if r.date_et in dmap]
    res["Q3"] = {"n_fills": len(dl), "delta_usd_mean": float(dl.mean()), "delta_usd_median": float(np.median(dl)),
                 "delta_usd_ci": ci, "entry_slip_pts_mean": float(np.mean(slip)),
                 "entry_slip_pts_median": float(np.median(slip)), "n_slip": len(slip)}

    days_is = c.rth_by_day(c.bars_from_processed(Path(a.bars_dir), a.is_bars_before))
    sess_is = c.sessions(days_is)
    is_df = c.arms_for_era(days_is, sess_is, a.is_start, a.is_end)
    live_df = c.arms_for_era(days_ts, sess_ts, a.live_start, a.live_end)
    is_df.to_csv(out / "q4_arms_is.csv", index=False)
    live_df.to_csv(out / "q4_arms_live.csv", index=False)
    res["Q4"] = {"IS": c.era_stats(is_df), "LIVE": c.era_stats(live_df)}

    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k].get("verdict", "") for k in ("Q1", "Q2")}))
    return res


if __name__ == "__main__":
    main()
