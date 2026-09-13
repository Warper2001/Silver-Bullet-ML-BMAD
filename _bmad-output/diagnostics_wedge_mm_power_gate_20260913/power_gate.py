"""Wedge (three-push) reversal and measured-move fade power gates (MNQ 5-min RTH).

Plan: analysis_plan.md (SHA-256 pinned below; the script refuses to run on a mismatch).
Window, bars, firewall, placebo, power formula and verdict rule are imported from the
committed H2/L2 gate (4482d37), whose SHA-256 is also pinned.

FIREWALL: real events are used only for count, timing, direction and risk. All
dispersion comes from placebo pairings shifted 5..ND-5 sessions (identity refused).

Run:  .venv/bin/python _bmad-output/diagnostics_wedge_mm_power_gate_20260913/power_gate.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
H2 = REPO / "_bmad-output/diagnostics_h2l2_power_gate_20260913"
PLAN = HERE / "analysis_plan.md"
PLAN_SHA = "b5287cb3835ea3936b207ef9b7e2954caae1d0e71772dd3481abd1d51784b601"
H2_GATE_SHA = "35bc2de44b11435464196d929fea7cd33ebac446c7d0dc7f2f08ef5bd81787ef"

sys.path.insert(0, str(H2))
import power_gate as pg  # noqa: E402

TICK = pg.TICK
STRENGTHS = [1, 2, 3]          # 1 primary; 2, 3 pre-declared sensitivity arms
PRIMARY_S = 1
DOJI_BODY = 0.30


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rtick(x: float) -> float:
    return round(x / TICK) * TICK


def pivots(v: np.ndarray, s: int, high: bool) -> list[int]:
    """Strict strength-s pivots within one session; bar p is known at close of p+s."""
    out = []
    for p in range(s, len(v) - s):
        nb = np.r_[v[p - s:p], v[p + 1:p + s + 1]]
        if (high and v[p] > nb.max()) or (not high and v[p] < nb.min()):
            out.append(p)
    return out


def detect_wedge(b: pd.DataFrame, s: int) -> pd.DataFrame:
    ev = []
    sidx = {d: i for i, d in enumerate(sorted(b["session"].unique()))}
    for day, g in b.groupby("session", sort=True):
        o, h, l, c, sl = (g[k].to_numpy() for k in ("open", "high", "low", "close", "slot"))
        n = len(g)
        for d in (-1, 1):                               # -1: wedge top (short); +1: wedge bottom (long)
            piv = pivots(h if d == -1 else l, s, high=(d == -1))
            prev, run = None, 0
            for p in piv:
                further = prev is not None and ((h[p] > h[prev]) if d == -1 else (l[p] < l[prev]))
                run = run + 1 if further else 1
                prev = p
                if run != 3:
                    continue
                q, t = p + s, p + s + 1
                if t >= n or sl[t] > pg.LAST_TRIGGER_SLOT:
                    continue
                rng, body = h[q] - l[q], abs(c[q] - o[q])
                if not (rng > 0 and body > DOJI_BODY * rng):
                    continue
                if d == -1:
                    if not (c[q] < o[q] and l[t] < l[q]):
                        continue
                    r = (h[p] + TICK) - (l[q] - TICK)
                else:
                    if not (c[q] > o[q] and h[t] > h[q]):
                        continue
                    r = (h[q] + TICK) - (l[p] - TICK)
                ev.append({"session": sidx[day], "slot": int(sl[t]), "dir": d, "r_pts": float(r)})
    return pd.DataFrame(ev, columns=["session", "slot", "dir", "r_pts"]).sort_values(["session", "slot"]).reset_index(drop=True)


def detect_mm(b: pd.DataFrame, s: int) -> tuple[pd.DataFrame, int]:
    """Measured-move target fade. Returns (events, same-bar fill/cancel ambiguities resolved as cancel)."""
    ev, ambiguous = [], 0
    sidx = {d: i for i, d in enumerate(sorted(b["session"].unique()))}
    for day, g in b.groupby("session", sort=True):
        h, l, sl = (g[k].to_numpy() for k in ("high", "low", "slot"))
        n = len(g)
        for mode in ("bull", "bear"):                   # bull MM -> short fade; bear MM -> long fade
            piv = pivots(l if mode == "bull" else h, s, high=(mode == "bear"))
            confirm = {p + s: p for p in piv}
            order = None                                # dict(D, C_ext, P, live_from)
            for t in range(n):
                if order is not None and t >= order["live_from"]:
                    if sl[t] > pg.LAST_TRIGGER_SLOT:
                        order = None
                    else:
                        if mode == "bull":
                            cancel, fill = l[t] < order["C"], h[t] >= order["D"] + TICK
                        else:
                            cancel, fill = h[t] > order["C"], l[t] <= order["D"] - TICK
                        if cancel and fill:
                            ambiguous += 1
                        if cancel:
                            order = None
                        elif fill:
                            ev.append({"session": sidx[day], "slot": int(sl[t]),
                                       "dir": -1 if mode == "bull" else 1, "r_pts": float(order["P"])})
                            order = None
                if t in confirm:
                    cidx = confirm[t]
                    prior = [p for p in piv if p < cidx]
                    if not prior:
                        continue
                    a = prior[-1]
                    if mode == "bull":
                        if not l[a] < l[cidx]:
                            continue
                        B = h[a + 1:cidx].max()
                        D, P, C = rtick(l[cidx] + (B - l[a])), B - l[cidx], l[cidx]
                        pre = h[cidx + 1:t + 1].max() >= D
                    else:
                        if not h[a] > h[cidx]:
                            continue
                        B = l[a + 1:cidx].min()
                        D, P, C = rtick(h[cidx] - (h[a] - B)), h[cidx] - B, h[cidx]
                        pre = l[cidx + 1:t + 1].min() <= D
                    if P <= 0 or pre:
                        continue
                    order = {"D": D, "C": C, "P": P, "live_from": t + 1}   # replaces any pending order
    out = pd.DataFrame(ev, columns=["session", "slot", "dir", "r_pts"]).sort_values(["session", "slot"]).reset_index(drop=True)
    return out, ambiguous


def size(ev: pd.DataFrame, H, L, C, nd: int) -> dict:
    evl = pg.thin(ev)
    r_usd = pg.PV * ev["r_pts"].to_numpy()
    rbar = float(r_usd.mean())
    out = {"events": {"N_upper": len(ev), "N_lower": len(evl), "per_session_upper": len(ev) / nd,
                      "long": int((ev["dir"] == 1).sum()), "short": int((ev["dir"] == -1).sum()),
                      "sessions_with_event": int(ev["session"].nunique())},
           "risk": {"R_usd_mean": rbar, "R_usd_median": float(np.median(r_usd)),
                    "R_usd_p10": float(np.percentile(r_usd, 10)), "R_usd_p90": float(np.percentile(r_usd, 90)),
                    "share_R_over_150": float((r_usd > 150).mean())},
           "cost": {cn: {"mean_c_over_R": float((c / r_usd).mean()), "breakeven_theta_R": c / rbar}
                    for cn, c in pg.COSTS.items()},
           "placebo": {tn: pg.shift_stats(ev, H, L, C, tv) for tn, tv in pg.TARGETS.items()},
           "grid": {}}
    for tn in pg.TARGETS:
        sig = out["placebo"][tn]["sigma_cluster_usd"]
        for cn, c in pg.COSTS.items():
            for th_n, th in pg.THETAS.items():
                mu = th * rbar - c
                row = {"mu_net_usd": mu,
                       "power_IS_upper": pg.power(mu, sig, len(ev)), "power_IS_lower": pg.power(mu, sig, len(evl)),
                       "power_HOLD_upper": pg.power(mu, sig, len(ev) / nd * pg.HOLD_SESSIONS)}
                if mu > 0:
                    n80 = ((pg.Z_A + pg.Z_B) * sig / mu) ** 2
                    row.update({"N_for_80": n80, "years_for_80_upper_rate": n80 / (len(ev) / nd) / 252})
                else:
                    row.update({"N_for_80": None, "years_for_80_upper_rate": None})
                out["grid"][f"{tn}|{cn}|{th_n}"] = row
    r = out["grid"]["1R|primary|central"]
    if r["mu_net_usd"] <= 0:
        v = "COST-BOUND"
    elif r["power_IS_lower"] >= 0.80:
        v = "POWERED"
    elif r["power_IS_upper"] >= 0.80:
        v = "POWERED-IF-DENSE"
    elif r["power_IS_upper"] >= 0.50:
        v = "MARGINAL"
    else:
        v = "UNDERPOWERED"
    out["verdict_rule_applied"] = v
    return out


def main() -> int:
    if sha(PLAN) != PLAN_SHA:
        sys.exit(f"PLAN HASH MISMATCH: {sha(PLAN)} != pinned {PLAN_SHA}")
    if sha(H2 / "power_gate.py") != H2_GATE_SHA:
        sys.exit("INHERITED H2/L2 GATE HASH MISMATCH")
    b = pg.load_5min()
    nd = b["session"].nunique()
    H, L, C = pg.grids(b, nd)
    res: dict = {"plan_sha256": PLAN_SHA, "script_sha256": sha(Path(__file__)), "inherited_gate_sha256": H2_GATE_SHA,
                 "inputs": {str(p.relative_to(REPO)): sha(p) for p in pg.INPUTS},
                 "window": {"first": str(b.index.min()), "last": str(b.index.max()), "sessions": nd, "bars": len(b)},
                 "W_wedge": {}, "M_measured_move_fade": {}}
    for s in STRENGTHS:
        arm = "primary" if s == PRIMARY_S else "sensitivity"
        res["W_wedge"][f"s{s}_{arm}"] = size(detect_wedge(b, s), H, L, C, nd)
        mm, amb = detect_mm(b, s)
        res["M_measured_move_fade"][f"s{s}_{arm}"] = {**size(mm, H, L, C, nd), "same_bar_ambiguous_cancelled": amb}
    res["VERDICT"] = {"W_wedge": res["W_wedge"][f"s{PRIMARY_S}_primary"]["verdict_rule_applied"],
                      "M_measured_move_fade": res["M_measured_move_fade"][f"s{PRIMARY_S}_primary"]["verdict_rule_applied"]}
    (HERE / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print("VERDICT", res["VERDICT"])
    for con in ("W_wedge", "M_measured_move_fade"):
        for arm, v in res[con].items():
            g = v["grid"]
            print(f"\n{con} {arm}: {v['events']}  R {({k: round(x, 2) for k, x in v['risk'].items()})}")
            print(f"  cost/R primary {v['cost']['primary']['mean_c_over_R']:.3f}  breakeven θ {v['cost']['primary']['breakeven_theta_R']:.3f}"
                  f"  σ1R ${v['placebo']['1R']['sigma_cluster_usd']:.1f}  σ2R ${v['placebo']['2R']['sigma_cluster_usd']:.1f}"
                  f"  rule→ {v['verdict_rule_applied']}" + (f"  ambiguous {v['same_bar_ambiguous_cancelled']}" if 'same_bar_ambiguous_cancelled' in v else ""))
            for k, r in g.items():
                print("   ", k, {kk: (round(x, 3) if isinstance(x, float) else x) for kk, x in r.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
