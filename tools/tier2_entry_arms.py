#!/usr/bin/env python3
"""Post-signal decay and paired entry-mechanics arms for Tier2/YANK census signals.

Input: the outputs of tools/tier2_census.py (signals_<tag>.csv, census_trades_<tag>.csv in
--run-dir) plus 1-minute MNQ bars. Everything is read-only; results go to --out-dir.

Gates (the run STOPS interpreting a tag if either fails):
  G0  census trades exiting on/before --g0-end match a reference replay CSV exactly
      (entry_time, entry_price, exit_type, exit_price, bars_held). Optional (--reference).
  G1  arm B0 (engine replica: midpoint limit, touch fill, strategy_core.check_exit)
      reproduces every census fill (type, price, bars held).

Step 1  D_k = (close[s] - close[s+k]) / (2*gap), k in {1,5,15,30,60,120,240}, every signal;
        INFORMATION PRESENT if mean D_k > 0 at any k in {15,30,60} with Holm p < 0.05.
Step 2  arms, net R per signal (unfilled = 0), paired vs B1 by calendar-day cluster bootstrap,
        Holm across A, M, S, I:
          B0 limit/touch, B1 limit/trade-through (reference), A random market entry in
          [s+1, s+240] (200 draws), M market at open of s+1, S sell-stop 1 tick below the
          signal-bar low, I inverse long market at open of s+1.
        Costs 0.52 pt per round trip; 1 tick adverse slippage on market/stop entries.

Origin and regression check: this is the promoted form of
_bmad-output/diagnostics_yank_entry_mechanics_20260913/analyze.py (plan sha 1b4eabab).
Run with --tags ml050 noml --seed 20260913 on that run's census files and it must reproduce
that folder's results.json exactly (rng draws are consumed in the same order).

Example:
  PYTHONPATH=. .venv/bin/python tools/tier2_entry_arms.py --run-dir <census dir> \
      --tags ml050 noml --reference ml050=data/reports/backtest_1year_20260615_181838.csv \
      --reference noml=data/reports/backtest_1year_20260615_185354.csv --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
from census_stats import (boot_mean_ci, holm, mde_dsr, p_greater_zero,  # noqa: E402
                          paired_cluster_test, sharpe, sr_diff_ci)
from src.research.strategy_core import Direction, EntryDecision, StrategyConfig, check_exit  # noqa: E402

TICK, COST_PTS = 0.25, 0.52           # $1.04 per contract round trip = 0.52 MNQ points
PEND, HOLD = 240, 60
CFG = StrategyConfig(sl_multiplier=2.0, tp_multiplier=8.0, max_hold_bars=HOLD, max_pending_bars=PEND)
NBOOT, NRAND = 10_000, 200
KS = [1, 5, 15, 30, 60, 120, 240]
INCOMPLETE = "incomplete"


@dataclass
class Bars:
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    ts: pd.Series

    @property
    def N(self) -> int:
        return len(self.C)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Bars":
        return cls(*(df[c].to_numpy(float) for c in ("open", "high", "low", "close")), df["ts"])


def load_bars(bars_dir: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    bars = pd.concat([pd.read_csv(bars_dir / f) for f in ("mnq_1min_2025.csv", "mnq_1min_2026_ytd.csv")],
                     ignore_index=True)
    bars["ts"] = pd.to_datetime(bars["timestamp"], format="ISO8601", utc=True)
    return bars[bars.ts < cutoff].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def snap(p: float) -> float:
    return round(round(p / TICK) * TICK, 10)


def run_exit(b: Bars, direction, fill_i, fill_px, sl, tp):
    """check_exit semantics from the fill bar (held=0 on the fill bar). (exit_px, type, held) or None."""
    dec = EntryDecision(direction=direction, entry_price=fill_px, sl_price=sl, tp_price=tp, contracts=1)
    held, j = 0, fill_i
    while j < b.N:
        ex = check_exit({"high": b.H[j], "low": b.L[j], "close": b.C[j]}, dec, held, CFG)
        if ex is not None:
            return ex.exit_price, {"SL": "sl", "TP": "tp"}.get(ex.reason.name, "time"), held
        j += 1
        held += 1
    return None


def bracket(direction, fill_px, gap):
    if direction == Direction.BEARISH:
        return snap(fill_px + 2 * gap), snap(fill_px - 8 * gap)
    return snap(fill_px - 2 * gap), snap(fill_px + 8 * gap)


def net_R(direction, fill_px, exit_px, gap):
    pts = (fill_px - exit_px) if direction == Direction.BEARISH else (exit_px - fill_px)
    return (pts - COST_PTS) / (2 * gap)


def arm_B(b: Bars, s, entry, sl, tp, gap, touch):
    """Midpoint limit. touch=True replicates the engine; touch=False requires trade-through."""
    held, j = 0, s + 1
    while j < b.N:
        held += 1
        hit = b.H[j] >= entry if touch else b.H[j] >= entry + TICK
        if hit:
            r = run_exit(b, Direction.BEARISH, j, entry, sl, tp)
            if r is None:
                return INCOMPLETE
            return {"R": net_R(Direction.BEARISH, entry, r[0], gap), "fill_i": j, "exit": r}
        if held >= PEND:
            return {"R": 0.0, "fill_i": None, "exit": None}
        j += 1
    return INCOMPLETE


def arm_market(b: Bars, s_bar, gap, direction, slip=TICK):
    """Market order at the open of bar s_bar+1."""
    j = s_bar + 1
    if j >= b.N:
        return INCOMPLETE
    px = snap(b.O[j] - slip) if direction == Direction.BEARISH else snap(b.O[j] + slip)
    sl, tp = bracket(direction, px, gap)
    r = run_exit(b, direction, j, px, sl, tp)
    if r is None:
        return INCOMPLETE
    return {"R": net_R(direction, px, r[0], gap), "fill_i": j, "exit": r}


def arm_S(b: Bars, s, gap, slip=TICK):
    """Sell stop one tick below the signal bar's low; pending up to PEND bars."""
    trig = snap(b.L[s] - TICK)
    held, j = 0, s + 1
    while j < b.N:
        held += 1
        if b.L[j] <= trig:
            px = snap(trig - slip)
            sl, tp = bracket(Direction.BEARISH, px, gap)
            r = run_exit(b, Direction.BEARISH, j, px, sl, tp)
            if r is None:
                return INCOMPLETE
            return {"R": net_R(Direction.BEARISH, px, r[0], gap), "fill_i": j, "exit": r}
        if held >= PEND:
            return {"R": 0.0, "fill_i": None, "exit": None}
        j += 1
    return INCOMPLETE


def arm_A(b: Bars, s, gap, rng, slip=TICK, nrand=NRAND):
    """Random market entry at the open after a bar drawn uniformly from [s+1, s+PEND]; mean R over draws."""
    if s + PEND + 1 >= b.N:
        return INCOMPLETE
    vals = []
    for u in rng.integers(s + 1, s + PEND + 1, nrand):
        r = arm_market(b, int(u), gap, Direction.BEARISH, slip)
        if r == INCOMPLETE:
            return INCOMPLETE
        vals.append(r["R"])
    return {"R": float(np.mean(vals)), "fill_i": None, "exit": None}


def sequential(sig_df, results):
    """Honor one position at a time: skip a signal while the arm's previous position is pending/active."""
    busy_until, kept = -1, []
    for k, row in enumerate(sig_df.itertuples()):
        res = results[k]
        if row.s <= busy_until:
            continue
        kept.append(res["R"])
        if res["fill_i"] is not None and res["exit"] is not None:
            busy_until = res["fill_i"] + res["exit"][2]
        elif res["fill_i"] is None and res["exit"] is None:
            busy_until = row.s + PEND
    return kept


def analyze_tag(tag, sig, cen, reference, bars_df, b: Bars, rng, g0_end, out_dir: Path) -> dict:
    out = {}
    N, C = b.N, b.C
    idx_map = {t: i for i, t in enumerate(bars_df.ts)}

    # ---------- G0 ----------
    if reference is not None:
        end_ts = pd.Timestamp(f"{g0_end} 23:59:59", tz="UTC")
        jn = reference[pd.to_datetime(reference.exit_time, format="ISO8601", utc=True) <= end_ts].reset_index(drop=True)
        cn = cen[pd.to_datetime(cen.exit_time, format="ISO8601", utc=True) <= end_ts].reset_index(drop=True)
        keys = ["entry_time", "entry_price", "exit_type", "exit_price", "bars_held"]
        g0 = len(jn) == len(cn) and all((jn[k].astype(str) == cn[k].astype(str)).all() for k in keys)
        out["G0"] = {"june_n": len(jn), "census_n": len(cn), "pass": bool(g0)}
    else:
        out["G0"] = {"pass": True, "note": "no reference supplied"}

    sig = sig.copy()
    sig["ts"] = pd.to_datetime(sig.signal_ts, format="ISO8601", utc=True)
    sig["s"] = sig.ts.map(idx_map)
    sig = sig.dropna(subset=["s"]).astype({"s": int}).reset_index(drop=True)
    sig["day"] = sig.ts.dt.tz_convert("US/Eastern").dt.date.astype(str)
    out["signals"] = {"n": len(sig), "filled": int(sig.filled.sum()), "expired": int((~sig.filled).sum())}

    # ---------- G1 ----------
    b0 = [arm_B(b, r.s, r.entry, r.sl, r.tp, r.gap, touch=True) for r in sig.itertuples()]
    cmap = {row.entry_time: row for row in cen.itertuples()}
    ok = tot = 0
    for r, res in zip(sig.itertuples(), b0):
        if r.signal_ts in cmap and res != INCOMPLETE and res["exit"] is not None:
            tot += 1
            c = cmap[r.signal_ts]
            ok += (res["exit"][1] == c.exit_type and abs(res["exit"][0] - c.exit_price) < 1e-6
                   and res["exit"][2] == c.bars_held)
    filled_census = sum(1 for r in sig.itertuples() if r.signal_ts in cmap)
    out["G1"] = {"matched": ok, "compared": tot, "census_filled": filled_census,
                 "pass": bool(tot == filled_census and ok == tot)}
    if not (out["G0"]["pass"] and out["G1"]["pass"]):
        out["STOP"] = "gate failure — no diagnostic interpreted"
        return out

    # ---------- Step 1 ----------
    days = sig.day.to_numpy()
    dec = {}
    for k in KS:
        okk = (sig.s + k) < N
        Dk = (C[sig.s[okk]] - C[sig.s[okk] + k]) / (2 * sig.gap[okk].to_numpy())
        ci, _ = boot_mean_ci(Dk, rng)
        ci_day, _ = boot_mean_ci(Dk, rng, groups=days[okk.to_numpy()])
        dec[str(k)] = {"n": int(okk.sum()), "mean": float(Dk.mean()), "median": float(np.median(Dk)),
                       "ci": ci, "ci_day": ci_day}
    pv = []
    for k in (15, 30, 60):
        okk = (sig.s + k) < N
        Dk = (C[sig.s[okk]] - C[sig.s[okk] + k]) / (2 * sig.gap[okk].to_numpy())
        pv.append(p_greater_zero(Dk, rng))
    ph = holm(pv)
    info = any(dec[str(k)]["mean"] > 0 and p < 0.05 for k, p in zip((15, 30, 60), ph))
    m5, m60 = dec["5"]["mean"], dec["60"]["mean"]
    speed = "FAST" if (m60 > 0 and m5 >= 0.5 * m60) else "SLOW"
    ok60 = (sig.s + 60) < N
    D60 = (C[sig.s[ok60]] - C[sig.s[ok60] + 60]) / (2 * sig.gap[ok60].to_numpy())
    fl = sig.filled[ok60].to_numpy()
    if (~fl).sum() >= 3 and fl.sum() >= 3:
        diffs = [D60[~fl][rng.integers(0, (~fl).sum(), (~fl).sum())].mean()
                 - D60[fl][rng.integers(0, fl.sum(), fl.sum())].mean() for _ in range(NBOOT)]
        sel = {"expired_mean": float(D60[~fl].mean()), "filled_mean": float(D60[fl].mean()),
               "diff": float(D60[~fl].mean() - D60[fl].mean()),
               "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        sel["verdict"] = "SELECTION CONFIRMED" if sel["diff"] > 0 and sel["ci"][0] > 0 else "NOT CONFIRMED"
    else:
        sel = {"verdict": "too few expired signals", "expired_n": int((~fl).sum())}
    out["step1"] = {"path_R": dec, "holm_p_15_30_60": ph,
                    "verdict": "INFORMATION PRESENT" if info else "NO INFORMATION DETECTED",
                    "decay_speed": speed, "selection": sel}

    # ---------- Step 2 ----------
    arms = {"B0": b0,
            "B1": [arm_B(b, r.s, r.entry, r.sl, r.tp, r.gap, touch=False) for r in sig.itertuples()],
            "A": [arm_A(b, r.s, r.gap, rng) for r in sig.itertuples()],
            "M": [arm_market(b, r.s, r.gap, Direction.BEARISH) for r in sig.itertuples()],
            "S": [arm_S(b, r.s, r.gap) for r in sig.itertuples()],
            "I": [arm_market(b, r.s, r.gap, Direction.BULLISH) for r in sig.itertuples()]}
    complete = np.array([all(arms[a][i] != INCOMPLETE for a in arms) for i in range(len(sig))])
    sub = sig[complete].reset_index(drop=True)
    R = {a: np.array([arms[a][i]["R"] for i in range(len(sig)) if complete[i]]) for a in arms}
    g = sub.day.to_numpy()
    n = len(sub)
    st2 = {"n_complete": int(n), "n_incomplete": int((~complete).sum()),
           "fill_rate": {a: float(np.mean([arms[a][i]["fill_i"] is not None for i in range(len(sig)) if complete[i]]))
                         for a in ("B0", "B1", "M", "S", "I")},
           "mean_R": {a: float(R[a].mean()) for a in R}, "sharpe": {a: sharpe(R[a]) for a in R},
           "mean_R_ci_day": {a: boot_mean_ci(R[a], rng, groups=g, reps=4000)[0] for a in R}}
    tests = {}
    for a in ("A", "M", "S", "I"):
        d = R[a] - R["B1"]
        t = paired_cluster_test(d, g, rng)
        t["rho_vs_B1"] = float(np.corrcoef(R[a], R["B1"])[0, 1])
        t["mde_dSR"] = mde_dsr(t["rho_vs_B1"], n)
        t["dSR"] = sharpe(R[a]) - sharpe(R["B1"])
        t["dSR_ci_day"] = sr_diff_ci(R[a], R["B1"], g, rng)
        tests[a] = t
    hg = holm([tests[a]["p_gt"] for a in ("A", "M", "S", "I")])
    hl = holm([tests[a]["p_lt"] for a in ("A", "M", "S", "I")])
    for a, pg, pl in zip(("A", "M", "S", "I"), hg, hl):
        tests[a]["holm_p_gt"], tests[a]["holm_p_lt"] = pg, pl
        tests[a]["verdict"] = "SUPERIOR" if pg < 0.05 else ("INFERIOR" if pl < 0.05 else "INCONCLUSIVE")
    st2["tests_vs_B1"] = tests
    joint = []
    if tests["A"]["verdict"] == "INCONCLUSIVE" and R["A"].mean() <= 0:
        joint.append("A~B1 and A<=0: FVG trigger adds nothing over random timing; context has no net drift "
                     "-> close trigger-replacement-inside-this-context")
    if (tests["M"]["verdict"] == "SUPERIOR" or tests["S"]["verdict"] == "SUPERIOR") and tests["A"]["verdict"] != "SUPERIOR":
        joint.append("fill mechanics are the lever")
    if R["I"].mean() > R["M"].mean():
        joint.append("I > M (descriptive): signal direction looks backwards")
    if R["M"].mean() < 0 and R["I"].mean() < 0:
        joint.append("M and I both negative: costs dominate")
    st2["joint_readings"] = joint

    sens = {}
    fullres = {a: [arms[a][i] for i in range(len(sig)) if complete[i]] for a in arms}
    for a in ("B1", "M", "S", "I"):
        kept = sequential(sub, fullres[a])
        sens[f"sequential_{a}"] = {"n": len(kept), "mean_R": float(np.mean(kept)), "sharpe": sharpe(kept)}
    for a in ("M", "S", "I"):
        if a == "S":
            r0 = [arm_S(b, r.s, r.gap, slip=0.0) for r in sub.itertuples()]
        else:
            r0 = [arm_market(b, r.s, r.gap, Direction.BEARISH if a == "M" else Direction.BULLISH, slip=0.0)
                  for r in sub.itertuples()]
        v = np.array([x["R"] for x in r0 if x != INCOMPLETE])
        sens[f"zero_slip_{a}"] = {"mean_R": float(v.mean()), "sharpe": sharpe(v)}
    for a in ("A", "M", "S", "I"):
        sens[f"trade_bootstrap_ci_{a}"] = boot_mean_ci(R[a] - R["B1"], rng, reps=4000)[0]
    st2["sensitivities"] = sens
    out["step2"] = st2
    pd.DataFrame({"signal_ts": sub.signal_ts, "filled_engine": sub.filled, **{f"R_{a}": R[a] for a in R}}).to_csv(
        out_dir / f"arms_per_signal_{tag}.csv", index=False)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="folder with signals_<tag>.csv and census_trades_<tag>.csv")
    ap.add_argument("--tags", nargs="+", required=True, help="processed in this order with ONE rng")
    ap.add_argument("--reference", action="append", default=[], help="TAG=replay.csv for gate G0 (repeatable)")
    ap.add_argument("--g0-end", default="2026-02-28", help="G0 compares trades exiting on/before this date")
    ap.add_argument("--bars-dir", default=str(ROOT / "data/processed/dollar_bars/1_minute"))
    ap.add_argument("--cutoff", default="2026-03-01", help="bars on/after this UTC date are not loaded")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    refs = dict(r.split("=", 1) for r in a.reference)
    run_dir, out_dir = Path(a.run_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars_df = load_bars(Path(a.bars_dir), pd.Timestamp(a.cutoff, tz="UTC"))
    b = Bars.from_frame(bars_df)
    rng = np.random.default_rng(a.seed)
    results = {}
    for tag in a.tags:
        sig = pd.read_csv(run_dir / f"signals_{tag}.csv")
        cen = pd.read_csv(run_dir / f"census_trades_{tag}.csv")
        ref = pd.read_csv(refs[tag]) if tag in refs else None
        results[tag] = analyze_tag(tag, sig, cen, ref, bars_df, b, rng, a.g0_end, out_dir)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({t: {k: v for k, v in r.items() if k in ("G0", "G1", "signals", "STOP")} for t, r in results.items()}))
    return results


if __name__ == "__main__":
    main()
