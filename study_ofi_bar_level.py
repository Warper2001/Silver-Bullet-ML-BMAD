"""study_ofi_bar_level.py — OFI-1. Seal fedaf84.

OFI = (upvol - downvol) / volume. Deciles. Primary horizon h=10 bars.
Verdict is ECONOMIC (both legs >= 1.53 bps), never statistical -- see seal §2/§6.7.
"""
import math
import numpy as np, pandas as pd

FRICTION_BPS, BAR_3X = 0.51, 1.53
PRIMARY_H = 10
SECONDARY_H = [1, 5, 30, 60]

df = pd.read_csv("data/mim_x/mnq_1min_by_contract.csv",
                 usecols=["contract", "timestamp", "close", "volume", "upvol", "downvol"])
for c in ("close", "volume", "upvol", "downvol"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["volume"] > 0].sort_values(["contract", "timestamp"]).reset_index(drop=True)
df["ofi"] = (df["upvol"] - df["downvol"]) / df["volume"]

for h in [PRIMARY_H] + SECONDARY_H:
    df[f"f{h}"] = (df.groupby("contract")["close"].shift(-h) / df["close"] - 1) * 10000

df["dec"] = pd.qcut(df["ofi"], 10, labels=False, duplicates="drop")

def legs(h):
    d = df.dropna(subset=[f"f{h}", "dec"])
    g = d.groupby("dec")[f"f{h}"]
    m, n = g.mean(), g.size()
    top, bot = int(d["dec"].max()), int(d["dec"].min())
    return m, n, m[top], -m[bot], top, bot

print("=" * 88)
print("OFI-1 — Bar-Level Signed Order Flow on MNQ   |   seal fedaf84")
print("=" * 88)
print(f"  bars={len(df):,}  contracts={df['contract'].nunique()}  "
      f"OFI=(upvol-downvol)/volume")
print(f"  SEALED: primary h={PRIMARY_H} bars; verdict is ECONOMIC;")
print(f"          friction {FRICTION_BPS} bps, 3x bar {BAR_3X} bps; t-stats NOT decision-bearing (§2)")

m, n, long_edge, short_edge, top, bot = legs(PRIMARY_H)
print(f"\n  DECILE MEANS of forward {PRIMARY_H}-bar return (bps), by OFI decile:")
print(f"  {'decile':>7}{'n':>10}{'mean fwd (bps)':>17}")
for d_ in sorted(m.index):
    mark = "  <- BOTTOM (short leg)" if d_ == bot else ("  <- TOP (long leg)" if d_ == top else "")
    print(f"  {int(d_):>7}{n[d_]:>10,}{m[d_]:>17.4f}{mark}")

print("\n" + "=" * 88)
print(f"VERDICT (seal §5) — ECONOMIC, primary h={PRIMARY_H}")
print("=" * 88)
print(f"  LONG  leg edge (top decile mean)      = {long_edge:+.4f} bps")
print(f"  SHORT leg edge (-1 x bottom decile)   = {short_edge:+.4f} bps")
print(f"  bars: friction {FRICTION_BPS} bps | 3x = {BAR_3X} bps")
if long_edge >= BAR_3X and short_edge >= BAR_3X:
    v = "PASS"
elif long_edge >= FRICTION_BPS and short_edge >= FRICTION_BPS:
    v = "MARGINAL"
else:
    v = "FAILS"
print(f"\n  >>> {v} <<<")
if v == "FAILS":
    print("  Bar-level signed OFI does not carry tradeable directional content at")
    print("  h=10 on MNQ. Pre-committed: record, close (§5).")
    if (long_edge >= FRICTION_BPS) != (short_edge >= FRICTION_BPS):
        print("  NOTE: exactly one leg cleared friction. §6.3 forbids the single-direction")
        print("  rescue -- the verdict is FAILS, not 'works long-only'.")
elif v == "MARGINAL":
    print("  Clears 1x friction but not 3x. No deployment, no successor seal (§5).")
else:
    print("  Triggers §7. No deployment. Must re-test with rolling deciles (§8).")

print("\n" + "-" * 88)
print("SECONDARY — reported, NEVER decision-bearing (§5, §6.2)")
print(f"  {'h':>4}{'long leg':>12}{'short leg':>12}{'spread':>10}   (bps)")
for h in [PRIMARY_H] + SECONDARY_H:
    _, _, le, se, _, _ = legs(h)
    tag = "  <- PRIMARY" if h == PRIMARY_H else ""
    print(f"  {h:>4}{le:>12.4f}{se:>12.4f}{le+se:>10.4f}{tag}")

d = df.dropna(subset=[f"f{PRIMARY_H}"])
x, y = d["ofi"].values, d[f"f{PRIMARY_H}"].values
xm, ym = x.mean(), y.mean()
b = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
res = y - (ym - b * xm + b * x)
se_b = math.sqrt((res ** 2).sum() / (len(x) - 2) / ((x - xm) ** 2).sum())
r2 = 1 - (res ** 2).sum() / ((y - ym) ** 2).sum()
print(f"\n  Regression r_fwd({PRIMARY_H}) = a + b*OFI :")
print(f"    beta = {b:+.4f} bps per unit OFI   t = {b/se_b:+.1f}   R^2 = {r2:.6f}")
print(f"    ^ t is HUGE by construction at N={len(x):,} and is NOT evidence of a")
print(f"      tradeable effect (seal §2, §6.7). Economic size is what decides.")
print(f"    A 1-sd OFI move ({df['ofi'].std():.4f}) implies {b*df['ofi'].std():+.4f} bps.")
