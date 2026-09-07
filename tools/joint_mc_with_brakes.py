"""Joint combine MC with the two derived halt triggers applied as REAL halts.

The sealed engine never modelled these (they were derived FROM instrumented runs,
not inputs TO them -- see correction_option3_and_joint_mc_rerun_20260905.md). This
runs them as actual trading halts so the brake's cost/benefit is quantified rather
than argued.

Triggers (as derived, 2026-06-17 deployment prereg):
  1. distance-to-floor: halt when combined equity <= trailing floor + $500
  2. combined PF < 0.70 evaluated at the 30-trade checkpoint

Modelling decisions, stated because they matter:
  * A halt is PERMANENT for the path -> a third outcome, "frozen". We cannot model
    a human re-enabling, and pretending the account resumes would flatter the brake.
    Frozen is neither pass nor blow: alive, above the floor, unable to reach target.
  * Floor trigger is evaluated END OF DAY, before the floor ratchets -- faithful to
    the original derivation, which recorded per-DAY distance-to-floor states.
  * PF trigger is evaluated ONCE, when the path's cumulative trade count first
    reaches 30 -- the faithful reading of "at the 30-trade checkpoint".

Everything else is byte-identical to joint_combine_mc.simulate().
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
MIM_DLL, YANK_DLL = -1000.0, -1000.0     # sealed engine values
FLOOR_TRIGGER_BUFFER = 500.0             # halt when bal <= floor + this
PF_TRIGGER = 0.70                        # halt when PF < this at the checkpoint
PF_CHECKPOINT_TRADES = 30


def simulate(day_lists, n_yank, *, brakes: bool, seed: int = 42):
    """Returns (pass%, blow%, frozen%, timeout%, median_days_to_pass)."""
    rng = np.random.default_rng(seed)
    nd = len(day_lists)
    pass_n = blow_n = frozen_n = 0
    dtp = []

    for _ in range(N_SIM):
        bal, floor, best_day, outcome = 50_000.0, 48_000.0, 0.0, None
        gross_w = gross_l = 0.0
        n_trades = 0
        pf_checked = False
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
                n_trades += 1
                if pnl > 0:
                    gross_w += pnl
                else:
                    gross_l += -pnl

                if strat == "M":
                    mim_d += pnl
                    if mim_d <= MIM_DLL:
                        mim_dead = True
                else:
                    yank_d += pnl
                    if yank_d <= YANK_DLL:
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

            if brakes:
                # PF checkpoint -- evaluated once, the first day-end at/after 30 trades
                if not pf_checked and n_trades >= PF_CHECKPOINT_TRADES:
                    pf_checked = True
                    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")
                    if pf < PF_TRIGGER:
                        outcome = "frozen"
                        break
                # distance-to-floor -- evaluated end of day, before the ratchet
                if bal <= floor + FLOOR_TRIGGER_BUFFER:
                    outcome = "frozen"
                    break

            floor = min(50_000.0, max(floor, bal - 2000.0))

        if outcome == "pass":
            pass_n += 1
        elif outcome == "blow":
            blow_n += 1
        elif outcome == "frozen":
            frozen_n += 1

    med = int(np.median(dtp)) if dtp else None
    timeout_n = N_SIM - pass_n - blow_n - frozen_n
    return (pass_n / N_SIM, blow_n / N_SIM, frozen_n / N_SIM, timeout_n / N_SIM, med)


def main() -> None:
    mim_days = load_mim()
    yank_days = load_yank_constrained()
    pool = primary_pool(mim_days, yank_days)
    mim_pool = [mim_days[d] for d in sorted(mim_days)
                if OVL_START <= pd.Timestamp(d) <= OVL_END]
    print(f"Constrained primary pool: {len(pool)} ET days")
    print(f"Brake: halt if equity <= floor+${FLOOR_TRIGGER_BUFFER:.0f}, "
          f"or PF < {PF_TRIGGER} at trade {PF_CHECKPOINT_TRADES}. Halt is permanent.\n")

    header = f"{'config':<22}{'brakes':<9}{'pass':>8}{'blow':>8}{'frozen':>9}{'timeout':>9}{'med_d':>7}"
    print(header)
    print("-" * len(header))

    configs = [("MIM solo", mim_pool, 0)] + [
        (f"MIM 1 : YANK {ny}", pool, ny) for ny in (1, 2, 3)
    ]
    results = {}
    for label, pl, ny in configs:
        for brakes in (False, True):
            p, b, f, t, m = simulate(pl, ny, brakes=brakes)
            results[(label, brakes)] = (p, b, f, t)
            tag = "ON" if brakes else "off"
            star = "  <-- DEPLOYED" if label == "MIM 1 : YANK 2" else ""
            print(f"{label:<22}{tag:<9}{p:>7.1%}{b:>8.1%}{f:>9.1%}{t:>9.1%}{str(m):>7}{star}")
        print()

    print("=" * 68)
    print("WHAT THE BRAKE BUYS AND COSTS (per config)")
    print("=" * 68)
    for label, _pl, _ny in configs:
        p0, b0, f0, _ = results[(label, False)]
        p1, b1, f1, _ = results[(label, True)]
        print(f"  {label:<22} blow {b0:.1%} -> {b1:.1%}  ({b1-b0:+.1%})   "
              f"pass {p0:.1%} -> {p1:.1%}  ({p1-p0:+.1%})   frozen +{f1:.1%}")
    print("\nSealed ADOPT gate: pass% > 54% AND blow% <= 33%.")


if __name__ == "__main__":
    main()
