"""
joint_combine_mc.py — YANK + MIM-NB joint Topstep 50K combine Monte Carlo.

Implements preregistration_yank_mim_joint_combine_mc.md (sealed bfecfd1) EXACTLY:
- One shared account; trailing floor on combined equity.
- MIM-NB v2 net @ $2.24/ct; YANK re-costed from $0.80/ct -> $2.24/ct.
- Primary = common-ET-day block bootstrap over overlap window (union of traded days).
- Sensitivity = independent draw from each strategy's traded-day pool (MIM first).
- Frozen sizing grid: YANK in {1,2,3} x MIM 1ct.
- Gate: ADOPT smallest YANK size with pass%>54% AND blow%<=33%, else CLOSE.

Data is read from the main checkout (untracked artifacts) by absolute path.
"""
import numpy as np
import pandas as pd

M = "/root/Silver-Bullet-ML-BMAD"
COST_PTS, PT_VAL = 1.12, 2.0            # MIM-NB: $2.24/ct
MIM_CATSTOP_PTS = 500.0                 # deployed CAT_STOP_PTS=500 (live config)
YANK_NATIVE_COMMISSION = 4.0            # $4/roundtrip at 5ct ($0.80/ct) in the backtest
YANK_CT = 5
COMBINE_COST_CT = 2.24                  # re-cost both engines to this
N_SIM, MAX_DAYS = 5000, 90
ET = "America/New_York"
OVL_START, OVL_END = pd.Timestamp("2025-05-22"), pd.Timestamp("2026-05-04")


def load_mim():
    df = pd.concat([
        pd.read_csv(f"{M}/data/reports/mim_nb_gate0_v2_2025.csv"),
        pd.read_csv(f"{M}/data/reports/mim_nb_gate1_v2_2026oos.csv"),
    ], ignore_index=True)
    # deployed 500-pt catastrophe stop: floor each trade's point P&L before cost
    capped = df["pnl_pts"].clip(lower=-MIM_CATSTOP_PTS)
    df["net1ct"] = (capped - COST_PTS) * PT_VAL
    df["ts"] = pd.to_datetime(df["day"] + " " + df["exit_t"])  # ET-naive ordering key
    days = {}
    for d, g in df.sort_values("ts").groupby("day"):
        days[d] = [(r.ts, r.net1ct, "M") for r in g.itertuples()]
    return days


def load_yank():
    df = pd.read_csv(f"{M}/data/reports/backtest_1year_20260615_181838.csv")
    # re-cost: backtest pnl is net of $4/5ct; gross_1ct = (pnl+4)/5; net@2.24 = gross-2.24
    df["net1ct"] = (df["pnl"] + YANK_NATIVE_COMMISSION) / YANK_CT - COMBINE_COST_CT
    ex = pd.to_datetime(df["exit_time"], utc=True).dt.tz_convert(ET)
    en = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(ET)
    df["day"] = en.dt.strftime("%Y-%m-%d")          # ET date of entry (per seal)
    df["ts"] = ex.dt.tz_localize(None)              # ET exit time, naive, for interleave
    days = {}
    for d, g in df.sort_values("ts").groupby("day"):
        days[d] = [(r.ts, r.net1ct, "Y") for r in g.itertuples()]
    return days


def simulate(day_lists, n_yank, seed=42):
    """day_lists: list of per-sim-day trade lists [(ts, net1ct, strat), ...] already
    in application order. Returns (pass%, blow%, median_days_to_pass)."""
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
            for ts, pnl1, strat in day_lists[di]:
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
                    if mim_d <= -1000.0:
                        mim_dead = True
                else:
                    yank_d += pnl
                    if yank_d <= -1000.0:
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


def main():
    mim_days, yank_days = load_mim(), load_yank()
    print(f"MIM-NB traded days: {len(mim_days)} | YANK traded days: {len(yank_days)}")

    # ---- baseline validation: MIM-only through THIS engine (expect ~54%/33%) ----
    mim_pool = [mim_days[d] for d in sorted(mim_days)]
    p, b, m = simulate(mim_pool, n_yank=0)
    print(f"\n[VALIDATION] MIM-only (this engine): pass={p:.1%} blow={b:.1%} med={m}  (expect ~54%/33%)")

    # ---- PRIMARY: common-ET-day pool over overlap (union of traded days) ----
    def in_ovl(d):
        t = pd.Timestamp(d)
        return OVL_START <= t <= OVL_END
    pool_dates = sorted({d for d in mim_days if in_ovl(d)} | {d for d in yank_days if in_ovl(d)})
    primary_pool = []
    for d in pool_dates:
        trades = sorted(mim_days.get(d, []) + yank_days.get(d, []), key=lambda x: x[0])
        primary_pool.append(trades)
    n_both = sum(1 for d in pool_dates if d in mim_days and d in yank_days)
    print(f"\n[PRIMARY] overlap pool: {len(pool_dates)} union-traded days "
          f"({sum(d in mim_days for d in pool_dates)} MIM, {sum(d in yank_days for d in pool_dates)} YANK, {n_both} both)")
    print(f"{'size (M:Y)':>12} {'pass%':>7} {'blow%':>7} {'run%':>6} {'med_d':>6}")
    primary = {}
    for ny in (1, 2, 3):
        p, b, m = simulate(primary_pool, n_yank=ny)
        primary[ny] = (p, b, m)
        print(f"{'1:'+str(ny):>12} {p:>7.1%} {b:>7.1%} {1-p-b:>6.1%} {str(m):>6}")

    # ---- SENSITIVITY: independent draw, MIM-first ordering ----
    mim_keys = sorted(mim_days)
    yank_keys = sorted(yank_days)
    print(f"\n[SENSITIVITY] independent (MIM {len(mim_keys)} x YANK {len(yank_keys)} traded-day pools)")
    print(f"{'size (M:Y)':>12} {'pass%':>7} {'blow%':>7} {'run%':>6} {'med_d':>6}")
    sens = {}
    for ny in (1, 2, 3):
        rng = np.random.default_rng(42)
        nm, nyk = len(mim_keys), len(yank_keys)
        pass_n = blow_n = 0
        dtp = []
        for _ in range(N_SIM):
            bal, floor, best_day, outcome = 50_000.0, 48_000.0, 0.0, None
            im = rng.integers(0, nm, size=MAX_DAYS)
            iy = rng.integers(0, nyk, size=MAX_DAYS)
            for dn in range(MAX_DAYS):
                trades = mim_days[mim_keys[im[dn]]] + yank_days[yank_keys[iy[dn]]]  # MIM first
                day_pnl = mim_d = yank_d = 0.0
                mim_dead = yank_dead = False
                for ts, pnl1, strat in trades:
                    if strat == "M":
                        if mim_dead:
                            continue
                        pnl = pnl1
                    else:
                        if yank_dead:
                            continue
                        pnl = pnl1 * ny
                    bal += pnl
                    day_pnl += pnl
                    if strat == "M":
                        mim_d += pnl
                        if mim_d <= -1000.0:
                            mim_dead = True
                    else:
                        yank_d += pnl
                        if yank_d <= -1000.0:
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
        sens[ny] = (pass_n / N_SIM, blow_n / N_SIM, med)
        print(f"{'1:'+str(ny):>12} {pass_n/N_SIM:>7.1%} {blow_n/N_SIM:>7.1%} {1-(pass_n+blow_n)/N_SIM:>6.1%} {str(med):>6}")

    # ---- decision gate ----
    print("\n=== DECISION GATE (primary governs): ADOPT smallest YANK size with pass%>54% AND blow%<=33% ===")
    adopt = None
    for ny in (1, 2, 3):
        p, b, _ = primary[ny]
        ok = p > 0.54 and b <= 0.33
        print(f"  1:{ny}  pass={p:.1%} (>54%? {p>0.54})  blow={b:.1%} (<=33%? {b<=0.33})  -> {'QUALIFIES' if ok else 'no'}")
        if ok and adopt is None:
            adopt = ny
    print(f"\n  VERDICT: {'ADOPT MIM 1ct : YANK '+str(adopt)+'ct' if adopt else 'CLOSE one-account branch (fallback = two parallel combines)'}")


if __name__ == "__main__":
    main()
