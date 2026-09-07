"""Can the strategy pass under stricter consistency rules?

MIM-NB's profit arrives in ~3 fat days out of 163, so a best-day cap is a
structural threat independent of drawdown. This sweeps it exhaustively.

Dimensions:
  * consistency pct: 0.30 (TradeDay/MyFundedFutures), 0.40 (Tradeify Select),
    0.50 (Topstep, current), 1.00 (no rule -- reference for what the rule costs)
  * formulation: firms word this two different ways and they are NOT equivalent
      A: best_day < pct * REALISED PROFIT   (what the sealed MC encodes)
      B: best_day < pct * PROFIT TARGET     (e.g. TradeDay's wording)
  * profit target: $1,500 (TradeDay static 50K) and $3,000 (Topstep 50K)
  * floor: trailing $2,000 vs static $2,000
  * config: MIM solo (1ct), MIM 1 : YANK 2 (deployed)

Also reports "hit target, blocked" -- paths that reached the profit target but
never satisfied consistency. That isolates the rule's bite from drawdown death.

Note: a blocked path keeps trading, and growing total profit while best_day is
fixed lowers the ratio -- so it can satisfy the rule later. The model captures
that naturally rather than failing the path at first touch.
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
MIM_DLL, YANK_DLL = -1000.0, -1000.0
MLL = 2000.0

CONSISTENCY = [0.30, 0.40, 0.50, 1.00]
TARGETS = [1500.0, 3000.0]
FORMULATIONS = ["A_vs_profit", "B_vs_target"]


def simulate(day_lists, n_yank, *, target, pct, formulation, trailing, seed=42):
    rng = np.random.default_rng(seed)
    nd = len(day_lists)
    pass_n = blow_n = hit_blocked_n = 0

    for _ in range(N_SIM):
        bal = START
        floor = START - MLL
        best_day = 0.0
        outcome = None
        ever_hit_target = False
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
            if profit >= target:
                ever_hit_target = True
                cap = pct * (profit if formulation == "A_vs_profit" else target)
                if best_day < cap:
                    outcome = "pass"
                    break
            if trailing:
                floor = min(START, max(floor, bal - MLL))

        if outcome == "pass":
            pass_n += 1
        elif outcome == "blow":
            blow_n += 1
        elif ever_hit_target:
            hit_blocked_n += 1

    return pass_n / N_SIM, blow_n / N_SIM, hit_blocked_n / N_SIM


def main() -> None:
    mim_days = load_mim()
    yank_days = load_yank_constrained()
    joint = primary_pool(mim_days, yank_days)
    mim_pool = [mim_days[d] for d in sorted(mim_days)
                if OVL_START <= pd.Timestamp(d) <= OVL_END]

    configs = [("MIM solo", mim_pool, 0), ("MIM 1 : YANK 2", joint, 2)]

    for cname, pool, ny in configs:
        for trailing in (True, False):
            regime = "TRAILING $2,000" if trailing else "STATIC $2,000"
            print(f"\n{'='*82}\n{cname}   |   {regime}\n{'='*82}")
            print(f"{'target':>8} {'formulation':>14} " + "".join(f"{f'{int(p*100)}%':>18}" for p in CONSISTENCY))
            print(f"{'':>8} {'':>14} " + "".join(f"{'pass/blocked':>18}" for _ in CONSISTENCY))
            print("-" * 82)
            for target in TARGETS:
                for form in FORMULATIONS:
                    cells = []
                    for pct in CONSISTENCY:
                        p, b, hb = simulate(pool, ny, target=target, pct=pct,
                                            formulation=form, trailing=trailing)
                        cells.append(f"{p:>10.1%}/{hb:<6.1%}")
                    print(f"{'$'+format(int(target),','):>8} {form:>14} " + "".join(cells))

    print("\nLegend: pass% / 'hit target but blocked by consistency'%")
    print("100% column = no consistency rule (reference for what the rule costs).")
    print("Formulation A = best_day < pct * realised profit; B = pct * target.")


if __name__ == "__main__":
    main()
