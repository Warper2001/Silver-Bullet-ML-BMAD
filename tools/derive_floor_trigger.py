"""
derive_floor_trigger.py — DERIVE (not assert) the distance-to-floor halt trigger
for the YANK+MIM-NB joint combine at the locked 1:2 size.

Pre-stated objective (set before computing): the halt-and-review circuit breaker
should fire when the account enters the zone where it is MORE LIKELY TO BLOW THAN
TO PASS — i.e. P(eventual blow | current distance-to-floor) > 0.50. The threshold
is read off where that probability crosses 0.50; it is not chosen by hand.

One knob: distance-to-floor. Correlation is NOT swept here (it is demoted to an
observe-only diagnostic in the deployment prereg). See
project_yank_mim_correlation_portfolio / feedback_derive_dont_assert_one_knob.

Reuses the sealed joint MC's pool + rules (tools/joint_combine_mc.py).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from tools.joint_combine_mc import load_mim, load_yank, OVL_START, OVL_END

N_SIM, MAX_DAYS, NY = 20000, 90, 2  # locked 1:2 deployment size


def build_primary_pool(mim_days, yank_days):
    def in_ovl(d):
        t = pd.Timestamp(d)
        return OVL_START <= t <= OVL_END
    dates = sorted({d for d in mim_days if in_ovl(d)} | {d for d in yank_days if in_ovl(d)})
    return [sorted(mim_days.get(d, []) + yank_days.get(d, []), key=lambda x: x[0]) for d in dates]


def main():
    pool = build_primary_pool(load_mim(), load_yank())
    rng = np.random.default_rng(42)
    nd = len(pool)
    states = []  # (start-of-day distance-to-floor, eventual_blow)
    pass_n = blow_n = 0
    for _ in range(N_SIM):
        bal, floor, best, out = 50000.0, 48000.0, 0.0, None
        day_dists = []
        for di in rng.integers(0, nd, MAX_DAYS):
            day_dists.append(bal - floor)
            dp = md = yd = 0.0
            mdead = ydead = False
            for ts, p1, strat in pool[di]:
                if strat == "M":
                    if mdead:
                        continue
                    pnl = p1
                else:
                    if ydead:
                        continue
                    pnl = p1 * NY
                bal += pnl
                dp += pnl
                if strat == "M":
                    md += pnl
                    if md <= -1000:
                        mdead = True
                else:
                    yd += pnl
                    if yd <= -1000:
                        ydead = True
                if bal <= floor:
                    out = "blow"
                    break
            if out:
                break
            best = max(best, dp)
            profit = bal - 50000
            if profit >= 3000 and best < 0.5 * profit:
                out = "pass"
                break
            floor = min(50000.0, max(floor, bal - 2000.0))
        pass_n += out == "pass"
        blow_n += out == "blow"
        blow = 1 if out == "blow" else 0
        states.extend((d, blow) for d in day_dists)

    arr = np.array(states)
    print(f"sims={N_SIM} pass={pass_n/N_SIM:.1%} blow={blow_n/N_SIM:.1%} day-states={len(arr)}")
    edges = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500]
    rows = []
    dstar = None
    print(f"\n{'band ($)':>14} {'n':>9} {'P(blow)':>8}")
    for lo, hi in zip(edges, edges[1:] + [10**9]):
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        n = int(m.sum())
        if n < 50:
            continue
        pb = float(arr[m, 1].mean())
        rows.append((lo, hi, n, pb))
        print(f"{str(lo)+'-'+(str(hi) if hi < 10**9 else '+'):>14} {n:>9} {pb:>8.1%}")
        if dstar is None and pb <= 0.5:
            dstar = lo
    print(f"\nDERIVED distance-to-floor trigger d* = ${dstar} "
          f"(smallest band with P(blow) <= 50%; below this the account is more likely to blow than pass)")


if __name__ == "__main__":
    main()
