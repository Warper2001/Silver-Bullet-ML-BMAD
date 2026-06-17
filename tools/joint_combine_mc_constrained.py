"""
joint_combine_mc_constrained.py — re-run the joint combine MC + both trigger
derivations using the TOPSTEP-SESSION-CONSTRAINED YANK series (flat by 15:10 CT,
no entries 15:08-17:00 CT). Required because the original authorization used
unconstrained 24h YANK, which cannot trade as-is on a Topstep combine (§3 blocker).

Constrained YANK file: data/reports/yank_topstep_constrained.csv (built by the
transform that force-flattens at the real 1-min 15:10 CT price). Re-cost to
$2.24/ct, ET-date-of-entry grouping — identical handling to the sealed engine's
load_yank, only the input trades differ.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from tools.joint_combine_mc import load_mim, simulate, OVL_START, OVL_END

M = "/root/Silver-Bullet-ML-BMAD"
ET = "America/New_York"
N_SIM, MAX_DAYS = 20000, 90


def load_yank_constrained():
    df = pd.read_csv(f"{M}/data/reports/yank_topstep_constrained.csv")
    df["net1ct"] = (df["pnl"] + 4.0) / 5 - 2.24            # re-cost $0.80/ct -> $2.24/ct
    en = pd.to_datetime(df["entry_time"], utc=True, format="ISO8601").dt.tz_convert(ET)
    ex = pd.to_datetime(df["exit_time"], utc=True, format="ISO8601").dt.tz_convert(ET)
    df["day"] = en.dt.strftime("%Y-%m-%d")
    df["ts"] = ex.dt.tz_localize(None)
    days = {}
    for d, g in df.sort_values("ts").groupby("day"):
        days[d] = [(r.ts, r.net1ct, "Y") for r in g.itertuples()]
    return days


def primary_pool(mim_days, yank_days):
    def in_ovl(d):
        t = pd.Timestamp(d)
        return OVL_START <= t <= OVL_END
    dates = sorted({d for d in mim_days if in_ovl(d)} | {d for d in yank_days if in_ovl(d)})
    return [sorted(mim_days.get(d, []) + yank_days.get(d, []), key=lambda x: x[0]) for d in dates]


def instrumented(pool, ny, mode):
    """mode='floor' -> per-day distance-to-floor states; 'pf' -> running-PF-at-30 states."""
    rng = np.random.default_rng(42); nd = len(pool); recs = []
    pass_n = blow_n = 0
    for _ in range(N_SIM):
        bal, floor, best, out = 50000.0, 48000.0, 0.0, None
        dists = []; gp = gl = 0.0; ntr = 0; pf30 = None
        for di in rng.integers(0, nd, MAX_DAYS):
            if mode == "floor":
                dists.append(bal - floor)
            dp = md = yd = 0.0; mdead = ydead = False
            for ts, p1, strat in pool[di]:
                if strat == "M":
                    if mdead: continue
                    pnl = p1
                else:
                    if ydead: continue
                    pnl = p1 * ny
                bal += pnl; dp += pnl
                if mode == "pf":
                    ntr += 1
                    if pnl >= 0: gp += pnl
                    else: gl += -pnl
                    if pf30 is None and ntr == 30:
                        pf30 = (gp / gl) if gl > 0 else np.inf
                if strat == "M":
                    md += pnl
                    if md <= -1000: mdead = True
                else:
                    yd += pnl
                    if yd <= -1000: ydead = True
                if bal <= floor: out = "blow"; break
            if out: break
            best = max(best, dp); profit = bal - 50000
            if profit >= 3000 and best < 0.5 * profit: out = "pass"; break
            floor = min(50000.0, max(floor, bal - 2000.0))
        pass_n += out == "pass"; blow_n += out == "blow"
        blow = 1 if out == "blow" else 0
        if mode == "floor":
            recs.extend((d, blow) for d in dists)
        elif pf30 is not None:
            recs.append((pf30, blow))
    return np.array(recs), pass_n / N_SIM, blow_n / N_SIM


def crossing(arr, edges, ascending_dist):
    """find threshold where P(blow) crosses 50%."""
    star = None
    for lo, hi in zip(edges, edges[1:]):
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.sum() < 30: continue
        pb = arr[m, 1].mean()
        lab = f"{lo}-{hi if hi != np.inf else '+'}"
        print(f"    {lab:>12} n={int(m.sum()):>7} P(blow)={pb:.1%}")
        if ascending_dist and star is None and pb <= 0.5:
            star = lo
        if not ascending_dist and pb > 0.5:
            star = hi
    return star


def main():
    mim_days = load_mim()
    yk = load_yank_constrained()
    pool = primary_pool(mim_days, yk)
    print(f"constrained YANK days: {len(yk)} | primary pool: {len(pool)}\n")

    print("MC grid (primary, constrained YANK):")
    print(f"{'size':>8} {'pass%':>7} {'blow%':>7} {'run%':>6}")
    base = None
    for ny in (1, 2, 3):
        p, b, m = simulate(pool, ny)
        if ny == 2: base = (p, b)
        print(f"{'1:'+str(ny):>8} {p:>7.1%} {b:>7.1%} {1-p-b:>6.1%}")

    print("\nDistance-to-floor trigger (constrained, 1:2):")
    arr, _, _ = instrumented(pool, 2, "floor")
    fstar = crossing(arr, [0,250,500,750,1000,1250,1500,1750,2000,2500,np.inf], ascending_dist=True)
    print(f"  -> derived floor trigger d* = ${fstar}")

    print("\nCombined-PF trigger (constrained, 1:2):")
    arr, _, _ = instrumented(pool, 2, "pf")
    pstar = crossing(arr, [0,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0,np.inf], ascending_dist=False)
    print(f"  -> derived PF trigger PF* = {pstar}")


if __name__ == "__main__":
    main()
