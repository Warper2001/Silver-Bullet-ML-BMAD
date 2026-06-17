"""
derive_pf_trigger.py — DERIVE (not assert) the combined-PF halt threshold for the
YANK+MIM-NB joint combine at the locked 1:2 size.

One knob: the PF threshold. N (the trade-count checkpoint) is held at the inherited
30 (prereg 7939eed); re-deriving N is a separate single-knob task.

Pre-stated objective (fixed before computing): fire the halt-and-review when the
account is MORE LIKELY TO BLOW THAN PASS — P(eventual blow | running combined PF
at the 30-trade checkpoint <= x) > 0.50. Read x off the crossing.

Reuses the sealed joint MC pool/rules (tools/joint_combine_mc.py).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from tools.joint_combine_mc import load_mim, load_yank, OVL_START, OVL_END

N_SIM, MAX_DAYS, NY, CHECK_N = 20000, 90, 2, 30


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
    recs = []  # (running combined PF at CHECK_N executed trades, eventual_blow)
    pass_n = blow_n = 0
    for _ in range(N_SIM):
        bal, floor, best, out = 50000.0, 48000.0, 0.0, None
        gp = gl = 0.0
        ntr = 0
        pf_at_check = None
        for di in rng.integers(0, nd, MAX_DAYS):
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
                ntr += 1
                if pnl >= 0:
                    gp += pnl
                else:
                    gl += -pnl
                if pf_at_check is None and ntr == CHECK_N:
                    pf_at_check = (gp / gl) if gl > 0 else np.inf
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
        if pf_at_check is not None:
            recs.append((pf_at_check, 1 if out == "blow" else 0))

    arr = np.array(recs)
    print(f"sims={N_SIM} pass={pass_n/N_SIM:.1%} blow={blow_n/N_SIM:.1%}")
    print(f"reached {CHECK_N} trades: {len(arr)} ({len(arr)/N_SIM:.0%}); blow-rate among those {arr[:,1].mean():.1%}")
    print(f"\n{'PF band':>12} {'n':>7} {'P(blow)':>8}")
    edges = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, np.inf]
    for lo, hi in zip(edges, edges[1:]):
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        n = int(m.sum())
        if n < 30:
            continue
        lab = f"{lo:.1f}-{hi:.1f}" if hi != np.inf else f">{lo:.1f}"
        print(f"{lab:>12} {n:>7} {arr[m,1].mean():>8.1%}")
    pfstar = None
    for lo, hi in zip(edges, edges[1:]):
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.sum() >= 30 and arr[m, 1].mean() > 0.5:
            pfstar = hi
    print(f"\nDERIVED PF* = {pfstar}: halt-and-review if running combined PF after {CHECK_N} trades < {pfstar} "
          f"(below this, P(eventual blow) > 50%).")


if __name__ == "__main__":
    main()
