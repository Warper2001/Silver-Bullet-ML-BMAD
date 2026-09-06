"""How much drawdown allowance does the strategy actually need?

Holds the strategy fixed and varies the MLL, under both floor regimes:
  * TRAILING (current Topstep 50K): floor = min(start, max(floor, bal - MLL))
  * STATIC   (e.g. Topstep Labs):   floor = start - MLL, never moves

Reports blow% and pass% vs allowance, plus the realised max-drawdown
distribution of the underlying day pool. The point is to find where the
survival curve flattens -- the allowance past which more room buys little.

Profit target is held at the current $3,000 (with the 50%-consistency rule) so
the MLL is the ONLY thing varying. Real vehicles scale target with size, so a
bigger account is not purely free room -- noted, not modelled.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

JOINT_MC = "/root/Silver-Bullet-ML-BMAD/.claude/worktrees/joint-mc-prereg"
sys.path.insert(0, JOINT_MC)

from tools.joint_combine_mc import OVL_END, OVL_START, load_mim  # noqa: E402
from tools.joint_combine_mc_constrained import (  # noqa: E402
    load_yank_constrained,
    primary_pool,
)

N_SIM, MAX_DAYS = 10000, 90
START = 50_000.0
TARGET = 3_000.0
MIM_DLL, YANK_DLL = -1000.0, -1000.0
ALLOWANCES = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000]


def simulate(day_lists, n_yank, mll: float, trailing: bool, seed: int = 42):
    rng = np.random.default_rng(seed)
    nd = len(day_lists)
    pass_n = blow_n = 0
    max_dds = []

    for _ in range(N_SIM):
        bal = START
        floor = START - mll
        peak = START
        best_day = 0.0
        outcome = None
        worst_dd = 0.0
        idx = rng.integers(0, nd, size=MAX_DAYS)

        for di in idx:
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
                peak = max(peak, bal)
                worst_dd = max(worst_dd, peak - bal)
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
            profit = bal - START
            if profit >= TARGET and best_day < 0.5 * profit:
                outcome = "pass"
                break
            if trailing:
                floor = min(START, max(floor, bal - mll))

        max_dds.append(worst_dd)
        if outcome == "pass":
            pass_n += 1
        elif outcome == "blow":
            blow_n += 1

    return pass_n / N_SIM, blow_n / N_SIM, np.array(max_dds)


def main() -> None:
    mim_days = load_mim()
    yank_days = load_yank_constrained()
    joint = primary_pool(mim_days, yank_days)
    mim_pool = [mim_days[d] for d in sorted(mim_days)
                if OVL_START <= pd.Timestamp(d) <= OVL_END]

    configs = [("MIM solo (1ct)", mim_pool, 0), ("MIM 1 : YANK 2 (deployed)", joint, 2)]

    for label, pool, ny in configs:
        print(f"\n{'='*74}\n{label}\n{'='*74}")
        # realised max-drawdown distribution at a very large allowance (unconstrained)
        _, _, dds = simulate(pool, ny, mll=1e9, trailing=False)
        pct = {p: np.percentile(dds, p) for p in (50, 75, 90, 95, 99)}
        print("Max drawdown over a 90-day path, unconstrained (no floor):")
        print("  " + "   ".join(f"p{p}=${v:,.0f}" for p, v in pct.items())
              + f"   max=${dds.max():,.0f}")

        print(f"\n{'allowance':>10} | {'TRAILING blow':>14} {'pass':>7} | {'STATIC blow':>12} {'pass':>7}")
        print("-" * 66)
        for mll in ALLOWANCES:
            pt, bt, _ = simulate(pool, ny, mll=float(mll), trailing=True)
            ps, bs, _ = simulate(pool, ny, mll=float(mll), trailing=False)
            star = "  <-- current" if mll == 2000 and ny == 2 else ""
            print(f"{'$'+format(mll,','):>10} | {bt:>13.1%} {pt:>7.1%} | {bs:>11.1%} {ps:>7.1%}{star}")

    print("\nTarget held at $3,000 + 50% consistency rule so MLL is the only variable.")
    print("Real vehicles scale the target with account size -- a bigger account is")
    print("not purely free room. Noted, not modelled.")


if __name__ == "__main__":
    main()
