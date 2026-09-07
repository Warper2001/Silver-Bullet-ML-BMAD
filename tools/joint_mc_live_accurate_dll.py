"""Joint combine MC re-run with LIVE-ACCURATE per-strategy daily loss limits.

Context (correction, 2026-09-05): the Option 3 review claimed the deployed 1:2
MIM:YANK sizing "depended on" the two derived halt triggers (distance-to-floor
+$500, PF<0.70) that were disabled 2026-07-29. Reading the engine shows that is
WRONG in an important way: `joint_combine_mc.simulate()` never modelled those
triggers at all. They were derived POST-HOC from instrumented runs (where does
P(eventual blow) cross 50%) as an operational alarm overlay. The sealed
pass/blow numbers were therefore computed in a no-trigger world -- i.e. exactly
today's world -- so disabling the triggers does not invalidate the sizing
decision. See the accompanying correction doc.

What the engine DOES model, and what is worth re-checking: per-strategy daily
loss limits. The sealed engine hardcodes -$1000 for BOTH strategies. Live, MIM
uses DLL_GUARD_USD = -$1000 (matches) but YANK uses max_daily_loss = -$750
(stricter than modelled). This script re-runs the constrained primary pool with
a parameterized DLL so the sealed cell can be reproduced as a validation check
and the live-accurate configuration reported alongside it.

Simulation logic is otherwise byte-identical to joint_combine_mc.simulate().
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

JOINT_MC_WORKTREE = "/root/Silver-Bullet-ML-BMAD/.claude/worktrees/joint-mc-prereg"
sys.path.insert(0, JOINT_MC_WORKTREE)

from tools.joint_combine_mc import OVL_END, OVL_START, load_mim  # noqa: E402
from tools.joint_combine_mc_constrained import (  # noqa: E402
    load_yank_constrained,
    primary_pool,
)

N_SIM, MAX_DAYS = 20000, 90


def simulate_dll(day_lists, n_yank, mim_dll: float, yank_dll: float, seed: int = 42):
    """joint_combine_mc.simulate() with the two daily-loss limits parameterized.

    Every other line is the sealed engine's logic unchanged: shared $50k account,
    $48k opening floor trailing at bal-$2000 (clamped at $50k), blow on
    bal <= floor, pass on profit >= $3000 with best_day < 0.5*profit.
    """
    rng = np.random.default_rng(seed)
    nd = len(day_lists)
    pass_n = blow_n = 0
    dtp = []
    for _ in range(N_SIM):
        bal, floor, best_day, outcome = 50_000.0, 48_000.0, 0.0, None
        idx = rng.integers(0, nd, size=MAX_DAYS)
        for dn, di in enumerate(idx):
            day_pnl = mim_d = yank_d = 0.0
            mim_dead = yank_dead = False
            for _ts, pnl1, strat in day_lists[di]:
                if strat == "M":
                    if mim_dead:
                        continue
                    pnl = pnl1 * 1
                else:
                    if yank_dead:
                        continue
                    pnl = pnl1 * n_yank
                bal += pnl
                day_pnl += pnl
                if strat == "M":
                    mim_d += pnl
                    if mim_d <= mim_dll:
                        mim_dead = True
                else:
                    yank_d += pnl
                    if yank_d <= yank_dll:
                        yank_dead = True
                if bal <= floor:
                    outcome = "blow"
                    break
            if outcome:
                break
            best_day = max(best_day, day_pnl)
            profit = bal - 50_000.0
            if profit >= 3000.0 and best_day < 0.5 * profit:
                outcome = "pass"
                dtp.append(dn + 1)
                break
            floor = min(50_000.0, max(floor, bal - 2000.0))
        if outcome == "pass":
            pass_n += 1
        elif outcome == "blow":
            blow_n += 1
    med = int(np.median(dtp)) if dtp else None
    return pass_n / N_SIM, blow_n / N_SIM, med


def main() -> None:
    mim_days = load_mim()
    yank_days = load_yank_constrained()
    pool = primary_pool(mim_days, yank_days)
    print(f"Constrained primary pool: {len(pool)} ET days "
          f"(MIM {len(mim_days)} traded days / YANK {len(yank_days)})")

    mim_pool = [mim_days[d] for d in sorted(mim_days)
                if OVL_START <= pd.Timestamp(d) <= OVL_END]

    configs = [
        ("SEALED   (MIM -1000 / YANK -1000)", -1000.0, -1000.0),
        ("LIVE-ACC (MIM -1000 / YANK  -750)", -1000.0, -750.0),
    ]

    for label, mim_dll, yank_dll in configs:
        print(f"\n{'='*66}\n{label}\n{'='*66}")
        p, b, m = simulate_dll(mim_pool, 0, mim_dll, yank_dll)
        print(f"  MIM solo (n_yank=0) : pass={p:6.1%}  blow={b:6.1%}  med_days={m}")
        for ny in (1, 2, 3):
            p, b, m = simulate_dll(pool, ny, mim_dll, yank_dll)
            flag = "   <-- DEPLOYED" if ny == 2 else ""
            print(f"  MIM 1ct : YANK {ny}ct : pass={p:6.1%}  blow={b:6.1%}  med_days={m}{flag}")

    print("\nSealed ADOPT gate: pass% > 54% AND blow% <= 33%.")
    print("Note: no halt-trigger overlay is modelled here, and none was modelled in")
    print("the sealed run either -- the triggers were derived from these runs, not")
    print("inputs to them. See the correction doc.")


if __name__ == "__main__":
    main()
