"""study_gap_velocity_conditioned.py — GAP-V test, per sealed protocol.

Pre-registration: _bmad-output/preregistration_gap_velocity_conditioned.md
Seal: e5b40f6 | Amendment 1 (VELOCITY_SPLIT): 5f08c5d

H1: among MNQ sessions gapping DOWN >=0.5%, GAP-1's fade (LONG at open) has lower
mean net P&L on HIGH-velocity gaps than on LOW-velocity gaps. One-sided.

Constraints enforced by this script (seal §12):
  - refuses to run while §11 VELOCITY_SPLIT is unset
  - reads ONLY the two 2023/2024 test files for outcomes
  - reads mnq_1min_2025.csv ONLY to re-derive VELOCITY_SPLIT and cross-check it
  - NEVER reads data/trades.db (the 24 live trades are excluded, seal §1)
  - GAP-1's qualification/simulation logic is loaded from backtest_gap_fade.py,
    with only its repo-root line patched (asserted byte-identical otherwise)
"""
import hashlib, statistics, sys, types
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path("/root/Silver-Bullet-ML-BMAD")
SEAL = REPO / "_bmad-output/preregistration_gap_velocity_conditioned.md"
GAP1_SRC = REPO / "backtest_gap_fade.py"
D = REPO / "data/processed/dollar_bars/1_minute"
DERIV_CSV = D / "mnq_1min_2025.csv"
TEST_CSVS = [D / "mnq_1min_2023_sepnov.csv", D / "mnq_1min_2024_sepnov.csv"]

FRICTION_USD = 3.00      # seal §6
N_FLOOR = 12             # seal §7 per-subgroup floor
ALPHA = 0.05


def guard_and_read_split() -> float:
    txt = SEAL.read_text()
    if "__COMPUTE_AND_APPEND_BEFORE_TEST_RUN__" in txt:
        sys.exit("REFUSING TO RUN: seal §11 VELOCITY_SPLIT is still a placeholder.")
    for line in txt.splitlines():
        if line.startswith("| VELOCITY_SPLIT |"):
            return float(line.split("**")[1].rstrip("%"))
    sys.exit("REFUSING TO RUN: could not parse VELOCITY_SPLIT from the seal.")


def load_gap1():
    raw = GAP1_SRC.read_text()
    old = "_REPO = Path(__file__).resolve().parents[3]"
    assert raw.count(old) == 1
    patched = raw.replace(old, f'_REPO = Path("{REPO}")')
    assert patched.replace(f'_REPO = Path("{REPO}")', old) == raw, "only the _REPO line may differ"
    m = types.ModuleType("gap1"); m.__file__ = str(GAP1_SRC)
    exec(compile(patched, str(GAP1_SRC), "exec"), m.__dict__)
    m._SHA = hashlib.sha256(raw.encode()).hexdigest()
    return m


def load_bars(path, g):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    ts = df["timestamp"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df["timestamp"] = ts.dt.tz_convert(g.ET_TZ)
    return df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")


def gap_down_trades(df, g, simulate=True):
    """Run GAP-1's LONG leg on qualifying gap-down sessions. Frozen rules only."""
    sessions = g.build_session_map(df)
    rth = df[df.index.map(g.is_rth)].copy()
    rth["date_et"] = rth.index.date
    out = []
    for date_et, s in sorted(sessions.items()):
        if s["dow"] in g.EXCLUDE_DOW:
            continue
        gap = s["rth_open"] - s["prior_close"]
        gap_pct = abs(gap) / s["prior_close"]
        if gap_pct < g.GAP_MIN_PCT or gap >= 0:
            continue
        rec = {"date": str(date_et), "gap_pct": gap_pct * 100}
        if simulate:
            entry, target = s["rth_open"], s["prior_close"]
            stop = entry - g.STOP_MULT * abs(gap)
            day = rth[rth["date_et"] == date_et]
            if len(day) < 2:
                continue
            outcome, pts = g.simulate_day(day.iloc[1:], 1, entry, target, stop)
            rec["outcome"] = outcome
            rec["pnl_usd"] = pts * g.MNQ_PV * g.CONTRACTS - FRICTION_USD
        out.append(rec)
    return out


def welch_one_sided(a, b):
    """H1: mean(a) > mean(b). Returns (t, df, p) via survival of Student t."""
    from math import sqrt
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan"), float("nan")
    t = (statistics.mean(a) - statistics.mean(b)) / se
    dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy import stats
        p = stats.t.sf(t, dfree)
    except ImportError:
        p = float("nan")
    return t, dfree, p


def bootstrap_ci(a, b, n=20000, seed=20260827):
    import random
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        ds.append(statistics.mean(sa) - statistics.mean(sb))
    ds.sort()
    return ds[int(0.05 * n)], ds[int(0.95 * n)]


def pf(v):
    gp = sum(x for x in v if x > 0); gl = -sum(x for x in v if x < 0)
    return gp / gl if gl else float("inf")


def main():
    split = guard_and_read_split()
    g = load_gap1()
    L = []
    print("=" * 92)
    print("GAP-V — Velocity-Conditioned Gap-Down Response  |  sealed e5b40f6, Amendment 1 5f08c5d")
    print("=" * 92)
    print(f"VELOCITY_SPLIT      : {split:.4f}%   (< LOW / >= HIGH)")
    print(f"backtest_gap_fade   : sha256 {g._SHA[:16]}")
    print(f"Friction            : ${FRICTION_USD:.2f} RT   Contracts: {g.CONTRACTS}   PV: ${g.MNQ_PV}")
    print(f"Frozen              : gap>={g.GAP_MIN_PCT:.1%}, stop {g.STOP_MULT}x, "
          f"time-stop {g.TIME_STOP_HOUR}:00 ET, Fri excluded, min {g.MIN_RTH_BARS} prior bars")

    # cross-check the derivation (seal §5) reproduces Amendment 1
    dv = [r["gap_pct"] for r in gap_down_trades(load_bars(DERIV_CSV, g), g, simulate=False)]
    rederived = statistics.median(dv)
    print(f"\nDerivation cross-check (2025, N={len(dv)}): median = {rederived:.4f}%  "
          f"{'OK' if abs(rederived - split) < 1e-4 else 'MISMATCH'}")

    # ---- TEST WINDOW ----
    for csv in TEST_CSVS:
        rows = gap_down_trades(load_bars(csv, g), g)
        print(f"\n{csv.name}: {len(rows)} qualifying gap-down trades")
        for r in rows:
            print(f"   {r['date']}  gap {r['gap_pct']:.3f}%  {r['outcome']:>5}  {r['pnl_usd']:>9.2f}")
        L.extend((csv.name, r) for r in rows)

    low = [r["pnl_usd"] for _, r in L if r["gap_pct"] < split]
    high = [r["pnl_usd"] for _, r in L if r["gap_pct"] >= split]

    print("\n" + "=" * 92)
    print("PRIMARY METRIC — Δ = mean(LOW) − mean(HIGH)")
    print("=" * 92)
    print(f"{'subgroup':10}{'N':>5}{'net$':>11}{'mean$':>10}{'PF':>8}{'WR':>8}")
    for nm, v in (("LOW", low), ("HIGH", high)):
        if v:
            wr = 100 * sum(1 for x in v if x > 0) / len(v)
            print(f"{nm:10}{len(v):>5}{sum(v):>11.2f}{statistics.mean(v):>10.2f}{pf(v):>8.3f}{wr:>7.1f}%")
        else:
            print(f"{nm:10}{0:>5}{'—':>11}{'—':>10}{'—':>8}{'—':>8}")

    print("\n" + "=" * 92)
    print("VERDICT (seal §7 decision rule)")
    print("=" * 92)
    if len(low) < N_FLOOR or len(high) < N_FLOOR:
        print(f"  N_low={len(low)}, N_high={len(high)}; floor is {N_FLOOR} per subgroup.")
        print("\n  >>> INSUFFICIENT_SAMPLE <<<")
        print("  Pre-committed action: record, no verdict. Do NOT widen the window,")
        print("  lower the gap threshold, or pool in live trades (§7, §8.4, §8.7).")
        if low and high:
            print(f"\n  [descriptive only, NOT a verdict] Δ = "
                  f"{statistics.mean(low) - statistics.mean(high):+.2f}/trade")
    else:
        d = statistics.mean(low) - statistics.mean(high)
        t, dfree, p = welch_one_sided(low, high)
        lo, hi = bootstrap_ci(low, high)
        print(f"  Δ = {d:+.2f}/trade   Welch t={t:.3f} (df={dfree:.1f})  one-sided p={p:.4f}")
        print(f"  bootstrap 90% CI on Δ: [{lo:+.2f}, {hi:+.2f}]")
        if d <= 0:
            print("\n  >>> H0 — MECHANISM DOES NOT TRANSFER <<<  Thread CLOSED (§7).")
        elif p > ALPHA:
            print("\n  >>> DIRECTIONALLY CONSISTENT, UNPROVEN <<<  No live change (§7).")
        else:
            print("\n  >>> MECHANISM TRANSFERS <<<  Triggers §9 (new seal only).")

    # secondary: per-year, never decision-bearing
    print("\n" + "-" * 92)
    print("SECONDARY (reported, never decision-bearing) — per file")
    for csv in TEST_CSVS:
        v = [r["pnl_usd"] for n, r in L if n == csv.name]
        if v:
            print(f"  {csv.name:28} N={len(v):>3} net={sum(v):>9.2f} mean={statistics.mean(v):>8.2f}")
    print("\nGAP-1 unmodified by this study. data/trades.db never read.")


if __name__ == "__main__":
    main()
