"""study_s26_exit_atr_floor.py — S26-EXIT paired exit experiment.

Pre-registration: _bmad-output/preregistration_s26_exit_atr_floor.md (seal be099b9)

Arm A: ATR = ATR20                 (live behaviour)
Arm B: ATR = max(ATR20, ATR60)     (floor; 60 = self.max_hold, seal §4)

All 165 live entries are frozen inputs in BOTH arms. One knob differs.
Primary: two-sided Wilcoxon signed-rank on per-trade paired differences (seal §6).
"""
import sqlite3, statistics, sys
from pathlib import Path
import pandas as pd

SL_MULT, TP_MULT, MAX_HOLD, LENGTH = 2.0, 4.0, 60, 20
BARS = Path("data/s26_replay/kraken_1m_2026-05-25_2026-08-28.csv")


def load_bars():
    df = pd.read_csv(BARS)
    df["ts"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(),
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr20"] = tr.rolling(LENGTH).mean()
    df["atr60"] = tr.rolling(MAX_HOLD).mean()
    return df


def load_entries():
    c = sqlite3.connect("file:data/trades.db?mode=ro", uri=True)
    return [{"ts": pd.Timestamp(r[0]).tz_convert("UTC"), "dir": 1 if r[1] == "L" else -1,
             "entry": r[2], "live_pnl": r[3], "live_reason": r[4]}
            for r in c.execute(
                "select timestamp,direction,entry_price,pnl,exit_reason from trades "
                "where trader_id='trader-s26' and timestamp>='2026-06-01' order by timestamp")]


def simulate(df, e, atr):
    d, entry = e["dir"], e["entry"]
    sl, tp = entry - d * atr * SL_MULT, entry + d * atr * TP_MULT
    idx = df.index
    i0 = idx.get_loc(e["ts"])
    for k in range(1, MAX_HOLD + 1):
        if i0 + k >= len(idx): break
        b = df.iloc[i0 + k]
        if d == 1:
            if b["low"] <= sl:  return "SL", (sl - entry) * d
            if b["high"] >= tp: return "TP", (tp - entry) * d
        else:
            if b["high"] >= sl: return "SL", (sl - entry) * d
            if b["low"] <= tp:  return "TP", (tp - entry) * d
    j = min(i0 + MAX_HOLD, len(idx) - 1)
    return "TIME_STOP", (df.iloc[j]["close"] - entry) * d


def pf(v):
    gp = sum(x for x in v if x > 0); gl = -sum(x for x in v if x < 0)
    return gp / gl if gl else float("inf")


def main():
    df, entries = load_bars(), load_entries()
    A, B, dif, rA, rB, floor_hits = [], [], [], [], [], 0
    for e in entries:
        prev = e["ts"] - pd.Timedelta(minutes=1)
        a20, a60 = df.at[prev, "atr20"], df.at[prev, "atr60"]
        atrB = max(a20, a60)
        if atrB > a20: floor_hits += 1
        ra, pa = simulate(df, e, a20)
        rb, pb = simulate(df, e, atrB)
        A.append(pa); B.append(pb); dif.append(pb - pa); rA.append(ra); rB.append(rb)

    n = len(A)
    print("=" * 86)
    print("S26-EXIT — ATR floor, paired exit experiment   |   seal be099b9")
    print("=" * 86)
    print(f"  frozen entries: {n}   Arm A: ATR20   Arm B: max(ATR20, ATR60)")
    print(f"  floor was BINDING (ATR60 > ATR20) on {floor_hits}/{n} entries "
          f"({100*floor_hits/n:.1f}%)")

    print(f"\n{'arm':6}{'net':>12}{'mean':>10}{'PF':>8}{'WR':>8}   exits")
    from collections import Counter
    for nm, v, r in (("A", A, rA), ("B", B, rB)):
        wr = 100 * sum(1 for x in v if x > 0) / n
        cc = Counter(r)
        print(f"{nm:6}{sum(v):>12.2f}{statistics.mean(v):>10.2f}{pf(v):>8.3f}{wr:>7.1f}%   "
              f"TP={cc['TP']} SL={cc['SL']} TIME={cc['TIME_STOP']}")

    nz = [d for d in dif if d != 0]
    print("\n" + "=" * 86)
    print("PRIMARY — two-sided Wilcoxon signed-rank on paired differences (B − A)")
    print("=" * 86)
    print(f"  pairs: {n}   non-zero differences: {len(nz)}")
    print(f"  mean(d)   = {statistics.mean(dif):+.2f}")
    print(f"  median(d) = {statistics.median(dif):+.2f}")
    if len(nz) >= 1:
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(nz, alternative="two-sided")
            print(f"  Wilcoxon W = {stat:.1f}   p = {p:.5f}")
        except ImportError:
            p = float("nan"); print("  scipy unavailable")
    else:
        p = float("nan")

    print("\n" + "=" * 86)
    print("VERDICT (seal §6)")
    print("=" * 86)
    med = statistics.median(dif)
    if p != p:
        print("  cannot evaluate")
    elif p > 0.05:
        print(f"  p = {p:.5f} > 0.05")
        print("\n  >>> NO DIFFERENCE <<<  The floor does not change outcomes.")
        print("  Pre-committed: record, close the thread, no second variant (§7.1).")
    elif med < 0:
        print("\n  >>> B IS WORSE <<<  The 20-bar ATR is not the defect. Record; close.")
    else:
        print("\n  >>> FLOOR HELPS <<<  Triggers §8. NO live change.")
        print("  Fee gate (§8): s26 gross edge 2.985 bps vs 4-10 bps RT fees.")

    print("\n" + "-" * 86)
    print("SECONDARY — mechanism check (never decision-bearing, §6)")
    ca, cb = Counter(rA), Counter(rB)
    print(f"  SL exits:  A={ca['SL']}  ->  B={cb['SL']}   ({cb['SL']-ca['SL']:+d})")
    print(f"  TP exits:  A={ca['TP']}  ->  B={cb['TP']}   ({cb['TP']-ca['TP']:+d})")
    print(f"  TIME:      A={ca['TIME_STOP']}  ->  B={cb['TIME_STOP']}   "
          f"({cb['TIME_STOP']-ca['TIME_STOP']:+d})")
    for nm, v, r in (("A", A, rA), ("B", B, rB)):
        sl = [abs(x) for x, rr in zip(v, r) if rr == "SL"]
        if sl:
            print(f"  Arm {nm} SL mean/median ratio: "
                  f"{statistics.mean(sl)/statistics.median(sl):.3f}")
    print("\n  Absolute PF/net of either arm may NOT be quoted as profitability (§7.5).")
    print("  Live trader-s26 unmodified. trades.db opened read-only.")


if __name__ == "__main__":
    main()
