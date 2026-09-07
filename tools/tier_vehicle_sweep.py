"""Does more allowance still win once the profit target scales with it too?

Follow-on to vehicle_allowance_curve.py (result_vehicle_allowance_curve_20260906.md),
which flagged its own gap: it varied MLL while holding target fixed at $3,000, but
real Topstep tiers scale BOTH together. This sweep closes that gap by pairing each
tier's real target with its real MLL, current 2026 pricing:

    50K  (current) : target $3,000  MLL $2,000
    100K           : target $6,000  MLL $3,000
    150K           : target $9,000  MLL $4,500

Same engine, same trade pools, same 1:2 deployed sizing, same seed -- target and
MLL are now varied together instead of MLL alone. Also reports the "MLL-only"
comparison (bump the allowance, leave target at $3,000) from the original sweep
so the two effects are visible side by side.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from tools.joint_combine_mc import OVL_END, OVL_START, load_mim  # noqa: E402
from tools.joint_combine_mc_constrained import (  # noqa: E402
    load_yank_constrained,
    primary_pool,
)

N_SIM, MAX_DAYS = 10000, 90
# Topstep confirms (help.topstep.com, checked 2026-09-07) the Combine has NO time
# limit -- trade as long as needed, bounded only by the loss rules. MAX_DAYS=90 is
# a sealed-engine modeling horizon, not a real constraint, so it censors any path
# whose target isn't hit within 90 days as "run" (neither pass nor blow) instead of
# letting it play out. That's fine at a $3,000 target but silently corrupts a
# $6,000/$9,000-target comparison. HORIZON_DAYS below is used for the tier sweep
# specifically, long enough that "run" should collapse toward ~0 for every tier.
START = 50_000.0
MIM_DLL, YANK_DLL = -1000.0, -1000.0

# (label, target, mll) -- current 2026 Topstep Combine pricing, MLL trailing
TIERS = [
    ("50K  (current)", 3_000.0, 2_000.0),
    ("100K",           6_000.0, 3_000.0),
    ("150K",           9_000.0, 4_500.0),
]
# MLL-only comparison at the same allowances, target held at $3,000 (original sweep)
MLL_ONLY = [2_000.0, 3_000.0, 4_500.0]
HORIZON_DAYS = 900  # ~3.5 trading years -- long enough for "run" (unresolved) to collapse


def simulate(day_lists, n_yank, target: float, mll: float, trailing: bool = True,
             seed: int = 42, max_days: int = MAX_DAYS):
    rng = np.random.default_rng(seed)
    nd = len(day_lists)
    pass_n = blow_n = 0
    dtp = []

    for _ in range(N_SIM):
        bal = START
        floor = START - mll
        peak = START
        best_day = 0.0
        outcome = None
        idx = rng.integers(0, nd, size=max_days)

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
                peak = max(peak, bal)
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
            if profit >= target and best_day < 0.5 * profit:
                outcome = "pass"
                dtp.append(dn + 1)
                break
            if trailing:
                floor = min(START, max(floor, bal - mll))

        if outcome == "pass":
            pass_n += 1
        elif outcome == "blow":
            blow_n += 1

    med = int(np.median(dtp)) if dtp else None
    return pass_n / N_SIM, blow_n / N_SIM, med


def main() -> None:
    mim_days = load_mim()
    yank_days = load_yank_constrained()
    joint = primary_pool(mim_days, yank_days)
    mim_pool = [mim_days[d] for d in sorted(mim_days)
                if OVL_START <= pd.Timestamp(d) <= OVL_END]

    configs = [("MIM solo (1ct)", mim_pool, 0), ("MIM 1 : YANK 2 (deployed)", joint, 2)]

    for label, pool, ny in configs:
        print(f"\n{'='*78}\n{label}\n{'='*78}")

        print(f"{'tier':>16} | {'target':>8} {'MLL':>7} | {'pass':>7} {'blow':>7} {'run':>7} {'med_d':>6}"
              f"   [horizon={HORIZON_DAYS}d]")
        print("-" * 78)
        for tname, target, mll in TIERS:
            p, b, m = simulate(pool, ny, target=target, mll=mll, trailing=True, max_days=HORIZON_DAYS)
            print(f"{tname:>16} | ${target:>7,.0f} ${mll:>6,.0f} | {p:>6.1%} {b:>6.1%} {1-p-b:>6.1%} {str(m):>6}")

        print(f"\n  MLL-only comparison (target held at $3,000, per 09-06 sweep):")
        print(f"  {'allowance':>10} | {'pass':>7} {'blow':>7}")
        for mll in MLL_ONLY:
            p, b, _ = simulate(pool, ny, target=3_000.0, mll=mll, trailing=True)
            print(f"  {'$'+format(int(mll),','):>10} | {p:>6.1%} {b:>6.1%}")

    print("\nTiers use current (2026) Topstep Combine pricing: 50K $3,000/$2,000, "
          "100K $6,000/$3,000, 150K $9,000/$4,500 (target/MLL). Same trade pools, "
          "sizing, and DLL as the sealed 06-17/09-06 engines -- only target+MLL vary "
          "together here instead of MLL alone.")


if __name__ == "__main__":
    main()
